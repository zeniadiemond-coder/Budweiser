import torch
import torch.nn.functional as F
from torch import nn

from .container import read_tensor, write_tensor

PARAMS = '3c77d931295c40d4d75d971398fee8a1'
DILATIONS = 'b028f94998d330bc643746f7a453feed'
PRE = '932ef94a406f2ac2793fd98a2fe2e008'
LAYERS = '60934ffa2a466aa4dea617defa78f36b'
CONV = '3d8a9ac90f7ac150415ce60a7b3aea38'
SKIP = '88f599b6848cb38e980ea3467fbec4b6'
POST = 'ed68c4c9013230e898fe12529dc1a63e'
POST_WEIGHT = '96e855e8076802b01546388bf18cd3c8'
FINAL = 'a3e0d41f3cb9977373f3e0fd64b59617'
EMBEDDING = '084246b66df966440092e15db1a64d20'
CONDITION = '72e89ef9aa4547f206cecfa2269de01b'
LAYER_CONDITION = '477aa7b96a1e0944307793fdd4f4a96b'
WEIGHT = '30c16498ff30815c43d58f610a91f938'
BIAS = 'c169d8ef47cc23f097ff312d08a6f829'
EMBEDDING_ROWS = 'b6cb2eef989a98b86bc28bcbaf1cc4d1'
INPUTS = '3fecf2d78547692b4157889e04a498f0'
HIDDEN = '247e226db757c8012220030394d22b2e'
SKIPS = '731aacbf65e8845bb02f662f0b402053'
OUTPUTS = '2aff5930df33d03ad1f33c3091886f13'
POST_ACTIVATION = '69ebce9ff768f54cd4d95647e4d878cf'
TANH_ACTIVATION = '8a8d5b352266bf85b49ce7ad682a1149'
ENCODER = 'c688ac0ef614fc2004874245946c9580'
CONTROLS = 'ab464905e3eb532d4ba956c7464f5773'
SYNTHESIS = '6b46ed7b862f0e89a13164f3f8658380'
SPEAKER_NAMES = '1a314735bf3ab7b6a72db7427319eb09'


def native_exp(x):
    p = x * 1.442695
    offset = (p < 0).to(x.dtype)
    q = torch.clamp_min(p, -126.0)
    z = (q - torch.trunc(q)) + offset
    v = ((27.728024 / (4.8425255 - z) + q + 121.274055) - z * 1.4901291) * 8388608.0
    bits = v.to(torch.int64) & 0xFFFFFFFF
    return torch.where(bits >= 2**31, bits - 2**32, bits).to(torch.int32).view(torch.float32)


def native_tanh(x):
    return 2.0 / (native_exp(x * -2.0) + 1.0) - 1.0


def native_sigmoid(x):
    return 1.0 / (native_exp(-x) + 1.0)


def with_exact_gradient(native_value, exact_value):
    return exact_value + (native_value - exact_value).detach()


def tanh(x, native):
    exact = torch.tanh(x)
    return with_exact_gradient(native_tanh(x.detach()), exact) if native else exact


def sigmoid(x, native):
    exact = torch.sigmoid(x)
    return with_exact_gradient(native_sigmoid(x.detach()), exact) if native else exact


def leaky_clamp(x):
    return F.leaky_relu(x, 0.01).clamp_max(4.0)


def causal_conv(h, weight, dilation, bias=None):
    padded = F.pad(h.transpose(1, 2), (2 * dilation, 0))
    return F.conv1d(padded, weight, bias, dilation=dilation).transpose(1, 2)


class TensorSite:
    def __init__(self, holder, key):
        self.holder, self.key = holder, key

    def read(self):
        return torch.from_numpy(read_tensor(self.holder[self.key]))

    def write(self, parameter):
        self.holder[self.key] = write_tensor(parameter.detach().cpu().numpy(), like=self.holder[self.key])


def gated_sites(section):
    sites = {
        'pre_w': TensorSite(section[PRE], WEIGHT), 'pre_b': TensorSite(section[PRE], BIAS),
        'post_w': TensorSite(section[POST][0][0][POST_WEIGHT], WEIGHT),
        'post_b': TensorSite(section[POST][0][0][POST_WEIGHT], BIAS),
        'final_w': TensorSite(section[FINAL][0], WEIGHT), 'final_b': TensorSite(section[FINAL][0], BIAS),
    }
    for i, layer in enumerate(section[LAYERS]):
        sites[f'conv_w{i}'] = TensorSite(layer[CONV], WEIGHT)
        sites[f'conv_b{i}'] = TensorSite(layer[CONV], BIAS)
        sites[f'skip_w{i}'] = TensorSite(layer[SKIP], WEIGHT)
    return sites


def causal_sites(section):
    sites = {
        'pre_w': TensorSite(section[PRE], WEIGHT), 'embedding': TensorSite(section, EMBEDDING),
        'cond_w': TensorSite(section[CONDITION], WEIGHT), 'final_w': TensorSite(section[FINAL][0], WEIGHT),
    }
    for i, layer in enumerate(section[LAYERS]):
        sites[f'conv_w{i}'] = TensorSite(layer[CONV], WEIGHT)
        sites[f'layer_cond_w{i}'] = TensorSite(layer[LAYER_CONDITION], WEIGHT)
        sites[f'skip_w{i}'] = TensorSite(layer[SKIP], WEIGHT)
    for i, post in enumerate(section[POST][0]):
        sites[f'post_w{i}'] = TensorSite(post[POST_WEIGHT], WEIGHT)
    return sites


def speaker_conditioned(name):
    return name == 'embedding' or name.startswith(('cond_w', 'layer_cond_w'))


@torch.no_grad()
def randomise(stack, generator):
    for name, parameter in stack.w.items():
        if '_b' in name:
            parameter.zero_()
        elif name == 'embedding':
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * parameter.std().item())
        else:
            bound = (6 / sum(nn.init._calculate_fan_in_and_fan_out(parameter))) ** 0.5
            parameter.copy_((torch.rand(parameter.shape, generator=generator) * 2 - 1) * bound)


def shaped_parameters(sites, shapes):
    return nn.ParameterDict({name: nn.Parameter(site.read().reshape(shapes.get(name, -1)))
                             for name, site in sites.items()})


def store_parameters(sites, parameters):
    for name, site in sites.items():
        site.write(parameters[name])


class GatedStack(nn.Module):
    def __init__(self, section, native_math=False):
        super().__init__()
        self.native_math = native_math
        self.dilations = read_tensor(section[PARAMS][DILATIONS]).tolist()
        sites = gated_sites(section)
        hidden, skips = sites['pre_b'].read().numel(), sites['post_b'].read().numel()
        outputs = sites['final_b'].read().numel()
        shapes = {'pre_w': (hidden, -1), 'post_w': (skips, skips), 'final_w': (outputs, skips)}
        for i in range(len(self.dilations)):
            shapes[f'conv_w{i}'] = (2 * hidden, hidden, 3)
            shapes[f'skip_w{i}'] = (skips, hidden)
        self.w = shaped_parameters(sites, shapes)
        self.hidden, self.outputs, self.inputs = hidden, outputs, self.w['pre_w'].shape[1]

    @property
    def latency(self):
        return sum(self.dilations)

    def store(self, section):
        store_parameters(gated_sites(section), self.w)

    def run(self, x):
        w, n = self.w, self.native_math
        frames = x.shape[1]
        h = F.linear(x, w['pre_w'], w['pre_b'])
        skip = 0
        for i, d in enumerate(self.dilations):
            a = causal_conv(h, w[f'conv_w{i}'], d, w[f'conv_b{i}'])
            h = tanh(a[..., :self.hidden], n) * sigmoid(a[..., self.hidden:], n)
            s = F.linear(h, w[f'skip_w{i}'])
            delay = sum(self.dilations[i + 1:])
            if delay:
                s = F.pad(s, (0, 0, delay, 0))[:, :frames]
            skip = skip + s
        y = tanh(skip, n)
        y = tanh(F.linear(y, w['post_w'], w['post_b']), n)
        return tanh(F.linear(y, w['final_w'], w['final_b']), n)

    forward = run

    def run_aligned(self, x):
        latency, frames = self.latency, x.shape[1]
        warm = 2 * latency + 1
        padded = torch.cat([x[:, :1].expand(-1, warm, -1), x, x[:, -1:].expand(-1, latency, -1)], 1)
        return self.run(padded)[:, warm + latency:warm + latency + frames]


class CausalStack(nn.Module):
    def __init__(self, section, native_math=False):
        super().__init__()
        self.native_math = native_math
        p = section[PARAMS]
        self.inputs, self.hidden, self.skips, self.outputs = p[INPUTS], p[HIDDEN], p[SKIPS], p[OUTPUTS]
        self.tanh_post = p[POST_ACTIVATION] == TANH_ACTIVATION
        self.dilations = read_tensor(p[DILATIONS]).tolist()
        sites = causal_sites(section)
        condition_width = sites['cond_w'].read().numel() // self.skips
        shapes = {
            'pre_w': (self.hidden, self.inputs), 'embedding': (p[EMBEDDING_ROWS], -1),
            'cond_w': (self.skips, condition_width), 'final_w': (self.outputs, self.skips),
        }
        for i in range(len(self.dilations)):
            shapes[f'conv_w{i}'] = (2 * self.hidden, self.hidden, 3)
            shapes[f'layer_cond_w{i}'] = (2 * self.hidden, condition_width)
            shapes[f'skip_w{i}'] = (self.skips, self.hidden)
        for name in sites:
            if name.startswith('post_w'):
                shapes[name] = (self.skips, self.skips)
        self.w = shaped_parameters(sites, shapes)
        self.posts = len(section[POST][0])
        self.conditioning_width = condition_width - self.w['embedding'].shape[1]

    @property
    def rows(self):
        return self.w['embedding'].shape[0]

    def add_speaker(self, embedding):
        old = self.w['embedding']
        self.w['embedding'] = nn.Parameter(torch.cat([old.detach(), embedding.to(old)[None]]))

    def store(self, section):
        section[PARAMS][EMBEDDING_ROWS] = self.rows
        store_parameters(causal_sites(section), self.w)

    def activate(self, x):
        return tanh(x, self.native_math) if self.tanh_post else leaky_clamp(x)

    def head(self, skip):
        y = self.activate(skip)
        for i in range(self.posts):
            y = self.activate(F.linear(y, self.w[f'post_w{i}']))
        return F.linear(y, self.w['final_w'])

    def gate(self, a):
        return tanh(a[..., :self.hidden], self.native_math) * sigmoid(a[..., self.hidden:], self.native_math)

    def forward(self, feedback, conditioning, speaker):
        w = self.w
        embedding = w['embedding'][speaker][:, None].expand(-1, feedback.shape[1], -1)
        c = torch.cat([conditioning, embedding], -1)
        x = F.linear(feedback, w['pre_w'])
        skip = F.linear(c, w['cond_w'])
        for i, d in enumerate(self.dilations):
            a = causal_conv(x, w[f'conv_w{i}'], d) + F.linear(c, w[f'layer_cond_w{i}'])
            x = self.gate(a)
            skip = skip + F.linear(x, w[f'skip_w{i}'])
        return self.head(skip)

    def init_state(self, batch, device=None):
        return [torch.zeros(batch, 2 * d, self.hidden, device=device or self.w['pre_w'].device)
                for d in self.dilations]

    def block(self, feedback, conditioning, speaker, state):
        w = self.w
        embedding = w['embedding'][speaker][:, None].expand(-1, feedback.shape[1], -1)
        c = torch.cat([conditioning, embedding], -1)
        x = F.linear(feedback, w['pre_w'])
        skip = F.linear(c, w['cond_w'])
        next_state = []
        for i, d in enumerate(self.dilations):
            joined = torch.cat([state[i], x], 1)
            next_state.append(joined[:, -2 * d:])
            a = F.conv1d(joined.transpose(1, 2), w[f'conv_w{i}'], dilation=d).transpose(1, 2) + F.linear(c, w[f'layer_cond_w{i}'])
            x = self.gate(a)
            skip = skip + F.linear(x, w[f'skip_w{i}'])
        return self.head(skip), next_state

    def step(self, feedback, conditioning, speaker, state):
        w = self.w
        c = torch.cat([conditioning, w['embedding'][speaker]], -1)
        x = F.linear(feedback, w['pre_w'])
        skip = F.linear(c, w['cond_w'])
        next_state = []
        for i, d in enumerate(self.dilations):
            history = state[i]
            kernel = w[f'conv_w{i}']
            a = (F.linear(history[:, 0], kernel[:, :, 0]) + F.linear(history[:, d], kernel[:, :, 1])
                 + F.linear(x, kernel[:, :, 2]) + F.linear(c, w[f'layer_cond_w{i}']))
            next_state.append(torch.cat([history[:, 1:], x[:, None]], 1))
            x = self.gate(a)
            skip = skip + F.linear(x, w[f'skip_w{i}'])
        return self.head(skip), next_state
