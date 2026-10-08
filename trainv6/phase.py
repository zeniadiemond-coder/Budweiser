import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

from . import ddsp
from .frontend import SAMPLE_RATE as FS
from .targets import TIME_TAPS

FRAME = ddsp.FRAME
BINS = ddsp.BINS
KNOTS = 4
SMOOTH_HALF = 8
HEADROOM = 3.0
CURVE_HEADROOM = 3.01
NOISE_EPSILON = 1e-20
MIN_HARMONICS = 3
WEIGHT_FLOOR = 1e-4
SHIFT_GRID = 96
STARTS = 8
RANDOM_PHASE, RANDOM_DECAY = 2.0, 1.5
SEARCH_ITERATIONS, SEARCH_RATE = 100, 0.1
PROPAGATE_ITERATIONS = 60
ROUNDS = 2
NEIGHBOURS = (-1, 1, -3, 3)
REFINE_ITERATIONS, REFINE_RATE = 150, 0.02
SMOOTH_PHASE, SMOOTH_DECAY = 0.1, 0.1
ANCHOR = 1e-3
FIT_SCALE = 0.1
LOGIT_LIMIT = 4.0


class Geometry(NamedTuple):
    slope: torch.Tensor
    knot: torch.Tensor


def phase_of(logits):
    return 2 * math.pi * torch.sigmoid(logits)


def decay_of(logits):
    return math.pi / torch.clamp(400.0 * torch.exp(logits), min=1e-6)


def phase_logits(phase):
    p = (phase / (2 * math.pi)).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def read_curve(curve, f):
    padded = torch.cat([curve[:, :1], curve, curve[:, -1:], curve[:, -1:]], -1)
    f = f.expand(len(curve), -1)
    x = (ddsp.warp(f).clamp(ddsp.WARP_LOW, ddsp.WARP_HIGH) - ddsp.WARP_LOW) * ((curve.shape[-1] - 1) / (ddsp.WARP_HIGH - ddsp.WARP_LOW))
    i = x.detach().floor().long()
    idx = (i[..., None] + torch.arange(4, device=f.device)).flatten(1)
    p0, p1, p2, p3 = padded.gather(1, idx).reshape(*f.shape, 4).unbind(-1)
    return ddsp.catmull_rom(x - i, p0, p1, p2, p3), torch.maximum(p1, p2)


def knot_weights(f):
    x = (ddsp.warp(f).clamp(ddsp.WARP_LOW, ddsp.WARP_HIGH) - ddsp.WARP_LOW) * (3.0 / (ddsp.WARP_HIGH - ddsp.WARP_LOW))
    i = x.floor().long()
    t = x - i
    weights = torch.zeros(*f.shape, KNOTS, dtype=f.dtype, device=f.device)
    lower, upper = i.clamp(max=KNOTS - 1), (i + 1).clamp(max=KNOTS - 1)
    weights.scatter_add_(-1, lower[..., None], (1 - t)[..., None])
    weights.scatter_add_(-1, upper[..., None], t[..., None])
    return weights


def envelope(curve, f0):
    full_frequency = torch.arange(FRAME, device=curve.device, dtype=curve.dtype) * (FS / FRAME)
    value, neighbours = read_curve(curve, full_frequency[None])
    full = torch.minimum(value, neighbours + HEADROOM)
    harmonic_db = torch.minimum(value[:, :BINS], neighbours[:, :BINS] + CURVE_HEADROOM)
    position = FRAME * f0 / FS
    lower = position.floor().long().clamp(max=BINS - 2)
    at_f0 = harmonic_db.gather(1, lower[:, None])[:, 0] + (harmonic_db.gather(1, (lower + 1)[:, None]) - harmonic_db.gather(1, lower[:, None]))[:, 0] * (position - lower)
    taps = torch.arange(-SMOOTH_HALF, SMOOTH_HALF + 1, device=curve.device)
    weights = 0.54 + 0.46 * torch.cos(2 * math.pi * taps / (2 * SMOOTH_HALF))
    index = (torch.arange(BINS, device=curve.device)[:, None] + taps[None]).abs()
    below = (index * (FS / FRAME))[None] < f0[:, None, None]
    values = torch.where(below, at_f0[:, None, None], full[:, index.clamp(max=FRAME - 1)])
    return (values * weights).sum(-1) / weights.sum()


def bin_slope(env, f0):
    u = torch.arange(BINS, device=env.device)
    limit = (f0 * FRAME / FS).long().clamp(1, BINS - 1)
    flat = env.gather(1, (limit + 1).clamp(max=BINS - 1)[:, None]) - env.gather(1, (limit - 1)[:, None])
    inner = env[:, (u + 1).clamp(max=BINS - 1)] - env[:, (u - 1).clamp(min=0)]
    top = (env[:, 0xf6] - env[:, 0xf4])[:, None]
    return torch.where(u[None] < limit[:, None], flat, torch.where(u[None] < 0xf6, inner, top))


def geometry(logits_a, f0, harmonics):
    curve = ddsp.suppress_peaks(160.0 * torch.sigmoid(2 * logits_a) - 140.0)
    slope = bin_slope(envelope(curve, f0), f0) * (FS / FRAME)
    bins = torch.arange(BINS, device=f0.device, dtype=f0.dtype) * (FS / FRAME)
    weights = knot_weights(bins)
    position = torch.arange(1, harmonics + 1, device=f0.device, dtype=f0.dtype)[None] * f0[:, None] * (FRAME / FS)
    lower = position.floor().long().clamp(max=BINS - 2)
    frac = (position - lower)[..., None]

    def at_harmonics(rows):
        return rows[lower] * (1 - frac) + rows[lower + 1] * frac

    scaled = weights[None] * slope[:, :, None]
    spread = lower[..., None].expand(-1, -1, KNOTS)
    scaled_at = scaled.gather(1, spread) * (1 - frac) + scaled.gather(1, spread + 1) * frac
    return Geometry(scaled_at, at_harmonics(weights))


def harmonic_phases(tail, geom):
    return (geom.knot * phase_of(tail[:, None, :KNOTS])).sum(-1) + (geom.slope * decay_of(tail[:, None, KNOTS:])).sum(-1)


class Observations(NamedTuple):
    phi: torch.Tensor
    weight: torch.Tensor
    strength: torch.Tensor
    order: torch.Tensor
    geometry: Geometry


def observations(estimate, f0_hz, logits_a, device):
    z = torch.from_numpy(estimate.harmonics).to(device)
    amplitude = z.abs()
    snr = amplitude ** 2 / torch.from_numpy(estimate.noise_power).to(device).clamp_min(NOISE_EPSILON)
    reliability = snr / (1 + snr)
    weight = amplitude * reliability
    energy = (amplitude ** 2).sum(-1)
    strength = ((amplitude ** 2 * reliability).sum(-1) / energy.clamp_min(NOISE_EPSILON)) * ((snr > 1).sum(-1) >= MIN_HARMONICS)
    weight = weight / weight.sum(-1, keepdim=True).clamp_min(NOISE_EPSILON)
    active = (weight.max(0).values > WEIGHT_FLOOR).nonzero()
    used = int(active.max()) + 1 if len(active) else 1
    f0 = torch.as_tensor(f0_hz, dtype=torch.float32, device=device)
    geom = geometry(torch.as_tensor(logits_a, dtype=torch.float32, device=device), f0, used)
    return Observations(torch.angle(z[:, :used]), weight[:, :used], strength, torch.arange(1, used + 1, device=device, dtype=torch.float32), geom)


def residual(tail, shift, obs):
    return obs.phi - harmonic_phases(tail, obs.geometry) - obs.order * shift[:, None]


def data_loss(tail, shift, obs):
    return (obs.weight * (1 - torch.cos(residual(tail, shift, obs)))).sum(-1)


def best_shift(tail, obs):
    shifts = torch.arange(SHIFT_GRID, device=tail.device, dtype=tail.dtype) * (2 * math.pi / SHIFT_GRID)
    error = (obs.phi - harmonic_phases(tail, obs.geometry))[:, None] - obs.order * shifts[None, :, None]
    return shifts[(obs.weight[:, None] * (1 - torch.cos(error))).sum(-1).argmin(1)]


def descend(tail, shift, obs, iterations, rate):
    tail, shift = tail.clone().requires_grad_(), shift.clone().requires_grad_()
    optimiser = torch.optim.Adam([tail, shift], lr=rate)
    for _ in range(iterations):
        optimiser.zero_grad()
        data_loss(tail, shift, obs).sum().backward()
        optimiser.step()
    return tail.detach(), shift.detach(), data_loss(tail, shift, obs).detach()


class Fit(NamedTuple):
    tail: torch.Tensor
    shift: torch.Tensor
    loss: torch.Tensor


def improve(best, candidate):
    better = candidate.loss < best.loss
    return Fit(torch.where(better[:, None], candidate.tail, best.tail), torch.where(better, candidate.shift, best.shift),
               torch.minimum(candidate.loss, best.loss))


def repeated(obs, copies):
    geometry = Geometry(obs.geometry.slope.repeat(copies, 1, 1), obs.geometry.knot.repeat(copies, 1, 1))
    return Observations(obs.phi.repeat(copies, 1), obs.weight.repeat(copies, 1), obs.strength.repeat(copies), obs.order, geometry)


def search(obs, base, generator):
    count = len(base)
    scale = torch.tensor([RANDOM_PHASE] * KNOTS + [RANDOM_DECAY] * KNOTS, device=base.device)
    starts = torch.cat([base] + [(torch.rand(count, 2 * KNOTS, device=base.device, generator=generator) * 2 - 1) * scale for _ in range(STARTS - 1)])
    shifts = torch.cat([best_shift(start, obs) for start in starts.split(count)])
    candidates = Fit(*descend(starts, shifts, repeated(obs, STARTS), SEARCH_ITERATIONS, SEARCH_RATE))
    best, *rest = (Fit(*parts) for parts in zip(*(field.split(count) for field in candidates)))
    for candidate in rest:
        best = improve(best, candidate)
    for _ in range(ROUNDS):
        for by in NEIGHBOURS:
            index = (torch.arange(count, device=base.device) + by).clamp(0, count - 1)
            start = best.tail[index]
            best = improve(best, Fit(*descend(start, best_shift(start, obs), obs, PROPAGATE_ITERATIONS, SEARCH_RATE)))
    return best


def circular_gap(a, b):
    return 1 - torch.cos(a - b)


def refine(fit, obs, base):
    tail, shift = fit.tail.clone().requires_grad_(), fit.shift.clone().requires_grad_()
    optimiser = torch.optim.Adam([tail, shift], lr=REFINE_RATE)
    link = torch.minimum(obs.strength[1:], obs.strength[:-1])
    for _ in range(REFINE_ITERATIONS):
        optimiser.zero_grad()
        phases, decays = phase_of(tail[:, :KNOTS]), tail[:, KNOTS:]
        smooth = (link[:, None] * (SMOOTH_PHASE * circular_gap(phases[1:], phases[:-1]) + SMOOTH_DECAY * (decays[1:] - decays[:-1]) ** 2)).sum()
        loss = (obs.strength * data_loss(tail, shift, obs)).sum() + smooth + ANCHOR * ((tail - base) ** 2).sum()
        loss.backward()
        optimiser.step()
    return Fit(tail.detach(), shift.detach(), data_loss(tail, shift, obs).detach())


def confidence(fit, obs):
    return obs.strength * torch.exp(-fit.loss / FIT_SCALE)


def smooth_time(x):
    kernel = torch.tensor(TIME_TAPS, dtype=x.dtype, device=x.device)
    return F.conv1d(x.T[:, None], (kernel / kernel.sum())[None, None], padding=len(TIME_TAPS) // 2)[:, 0].T


def unit_phasor(phase):
    return torch.polar(torch.ones_like(phase), phase)


def coherent_tail(tail, confidence, anchor):
    mass = smooth_time(torch.ones_like(confidence)[:, None])
    trust = confidence[:, None]
    phasor = trust * unit_phasor(phase_of(tail[:, :KNOTS])) + (1 - trust) * unit_phasor(phase_of(anchor[:, :KNOTS]))
    smoothed = torch.complex(smooth_time(phasor.real), smooth_time(phasor.imag))
    decay = smooth_time(trust * tail[:, KNOTS:] + (1 - trust) * anchor[:, KNOTS:]) / mass
    return torch.cat([phase_logits(torch.angle(smoothed) % (2 * math.pi)), decay], -1), (smooth_time(trust) / mass)[:, 0]


@torch.enable_grad()
def phase_targets(estimate, f0_hz, logits_a, base_tail, device, seed=0):
    obs = observations(estimate, f0_hz, logits_a, device)
    base = torch.as_tensor(base_tail, dtype=torch.float32, device=device)
    generator = torch.Generator(device).manual_seed(seed)
    fit = refine(search(obs, base, generator), obs, base)
    tail, support = coherent_tail(fit.tail.clamp(-LOGIT_LIMIT, LOGIT_LIMIT), confidence(fit, obs), base)
    return tail.clamp(-LOGIT_LIMIT, LOGIT_LIMIT).cpu().numpy(), support.cpu().numpy()
