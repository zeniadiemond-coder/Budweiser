import contextlib
import functools
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import dataset, ddsp, frontend, nets
from .dataset import CONTROLS_LOOKBACK, WARMUP
from .container import file_digest
from .reference.frontend import NOISY_JITTER, QUIET_JITTER, noisy_feedback
from .export import load_checkpoint

HUBER_DB = 3.0
DB_SCALE = 10.0
LEVEL_RANGE_DB = 40.0
MIN_LEVEL_WEIGHT = 0.3
LEVEL_REGIONS_HZ = ((4000.0, 8000.0), (8000.0, 16000.0))
SPEAKER_LR_SCALE = 3.0
MAX_VALIDATION_STEPS = 2100
VALIDATION_CHUNK = 64
VALIDATION_BATCH = 12
FEEDBACK_CHUNK = 128
CONTROLS_FEATURES = 32
DEFAULT_STEPS = 6000
MAX_STEPS = 1_000_000
STAGE_SPLIT = (0.05, 0.25)
UNVOICED_A_WEIGHT = 0.1
EMA_DECAY = 0.999
WEIGHT_DECAY = 0.01
WEIGHTS = dict(curve_a=1.0, noise=1.0, phase=0.1, decay=0.1, feedback=1.0, level=1.0)
PHASE_KNOT_WEIGHTS = (1.0, 1.0, 0.25, 0.0)


def huber(error):
    return F.huber_loss(error / DB_SCALE, torch.zeros_like(error), delta=HUBER_DB / DB_SCALE, reduction='none')


def make_targets(b):
    valid = b['valid'].float()
    width = b['curve_a'].shape[-1]
    level = 1.0 + (b['curve_a'] - b['curve_a'].amax(-1, keepdim=True)) / LEVEL_RANGE_DB
    weight = (MIN_LEVEL_WEIGHT + (1 - MIN_LEVEL_WEIGHT) * level.clamp(0, 1)) * b['voiced'].float()[..., None]
    tail_weight = b['tail_weight'] * valid
    unvoiced_weight = UNVOICED_A_WEIGHT * valid * (1 - b['voiced'].float())
    return SimpleNamespace(b=b, valid=valid, weight=weight, unvoiced_weight=unvoiced_weight,
                           den_a=(weight.sum() + unvoiced_weight.sum() * width).clamp_min(1.0), den_steps=valid.sum().clamp_min(1.0),
                           tail_weight=tail_weight, den_tail=tail_weight.sum().clamp_min(1.0))


@functools.cache
def knot_weights(device):
    return torch.tensor(PHASE_KNOT_WEIGHTS, device=device)


def curve_terms(heads, t, sl):
    width = t.b['curve_a'].shape[-1]
    c = ddsp.curves(heads, width)
    valid = t.valid[:, sl, None]
    a = (huber(c.A - t.b['curve_a'][:, sl]) * (t.weight[:, sl] + t.unvoiced_weight[:, sl, None])).sum() / t.den_a
    noise = (huber(ddsp.suppress_peaks(c.A) + c.B - t.b['noise'][:, sl]) * valid).sum() / (t.den_steps * width)
    tail, target = heads[..., 2 * width:2 * width + 8], t.b['tail_target'][:, sl]
    knots = knot_weights(tail.device)
    phase = (1 - torch.cos(2 * torch.pi * (torch.sigmoid(tail[..., :4]) - torch.sigmoid(target[..., :4])))) * knots
    decay = F.huber_loss(tail[..., 4:], target[..., 4:], reduction='none')
    return dict(curve_a=a, noise=noise, phase=(phase * t.tail_weight[:, sl, None]).sum() / (t.den_tail * knots.sum()),
                decay=(decay * t.tail_weight[:, sl, None]).sum() / (t.den_tail * 4))


def feedback_term(predicted, t, sl_or_idx):
    valid = t.valid[:, sl_or_idx, None]
    return (huber(predicted - t.b['feedback'][:, sl_or_idx]) * valid).sum() / (t.den_steps * 96)


def energy_db(x, mask):
    return 10 * torch.logsumexp((x * (math.log(10) / 10)).masked_fill(~mask, -torch.inf), -1) / math.log(10)


def level_term(predicted, t, sl, band_hz):
    voiced = t.valid[:, sl] * t.b['voiced'].float()[:, sl]
    target = t.b['feedback'][:, sl]
    bias = torch.stack([((energy_db(predicted, m) - energy_db(target, m)) / DB_SCALE * voiced).sum() / voiced.sum().clamp_min(1.0)
                        for m in ((band_hz >= low) & (band_hz < high) for low, high in LEVEL_REGIONS_HZ)])
    return bias.square().mean()


def total(parts):
    return sum(WEIGHTS[k] * v for k, v in parts.items())


def source_valid(b):
    steps = torch.arange(b['valid'].shape[1], device=b['valid'].device)
    return (b['start'][:, None] + steps) >= WARMUP


def masked_states(analysis, heads, f0_hz, valid):
    states = analysis.frame_states(heads, f0_hz)
    return ddsp.FrameStates(*(x * valid.reshape(*valid.shape, *([1] * (x.dim() - 2))) for x in states))


def feedback_chunk(analysis, amps, f0, noise, w_amps, w_f0, w_noise):
    return torch.cat([analysis.feedback16(ddsp.FrameStates(amps, f0, noise)),
                      analysis.feedback80(ddsp.FrameStates(w_amps, w_f0, w_noise), torch.zeros_like(f0))], -1)


def feedback_at(analysis, states, idx):
    windows = analysis.windows(states)
    flat = [x[:, idx].flatten(0, 1) for x in (*states, *windows)]
    parts = [checkpoint(feedback_chunk, analysis, *(x[i:i + FEEDBACK_CHUNK] for x in flat), use_reentrant=False, preserve_rng_state=False)
             for i in range(0, flat[0].shape[0], FEEDBACK_CHUNK)]
    return torch.cat(parts).reshape(states.f0.shape[0], len(idx), 96)


def teacher_forced(bf8, analysis, y9, speaker, b, t, jitter, frames):
    heads = bf8(analysis.normalise(b['source'], jitter), y9, speaker)
    loss_region = slice(WARMUP, None)
    parts = curve_terms(heads[:, loss_region], t, loss_region)
    states = masked_states(analysis, heads, frontend.cents_to_hz(b['cents']), source_valid(b).float())
    idx = WARMUP + torch.randperm(heads.shape[1] - WARMUP, device=heads.device)[:frames].sort().values
    parts['feedback'] = feedback_term(feedback_at(analysis, states, idx), t, idx)
    return parts


def rollout(bf8, analysis, y9, speaker, b, free, chunk, jitter, on_chunk, train):
    batch, width = y9.shape[:2]
    real = analysis.normalise(b['source'], jitter)
    f0_hz = frontend.cents_to_hz(b['cents'])
    in_stream = (b['start'][:, None] + torch.arange(width, device=y9.device)) >= WARMUP
    emitted = in_stream.float()
    zeros = torch.zeros(batch, device=y9.device)
    state = bf8.init_state(batch)
    empty = ddsp.FrameStates(torch.zeros(batch, ddsp.HARMONICS, device=y9.device), zeros, torch.zeros(batch, ddsp.BINS, device=y9.device))
    window = [empty] * ddsp.WINDOW_FRAMES
    own = None
    heads, feedbacks = [], []
    for e in range(width):
        running = train and e >= WARMUP
        with torch.set_grad_enabled(running):
            if e >= WARMUP:
                own_ok = free[:, (e - WARMUP) // chunk] & in_stream[:, e - 1]
                x = torch.where(own_ok[:, None], own, real[:, e])
            else:
                x = real[:, e]
            head, state = bf8.step(x, y9[:, e], speaker, state)
            if e >= WARMUP - ddsp.WINDOW_FRAMES + 1:
                s = analysis.frame_states(head[:, None], f0_hz[:, e, None])
                s = ddsp.FrameStates(*(v[:, 0] * emitted[:, e].reshape(-1, *([1] * (v.dim() - 2))) for v in s))
                window = window[1:] + [s]
            if e >= WARMUP - 1:
                fb = torch.cat([analysis.feedback16(window[-1]),
                                analysis.feedback80(ddsp.FrameStates(*(torch.stack(v, 1) for v in zip(*window))), zeros)], -1)
                own = analysis.normalise(fb, jitter)
            if e >= WARMUP:
                heads.append(head)
                feedbacks.append(fb)
            done = e >= WARMUP and ((e - WARMUP + 1) % chunk == 0 or e == width - 1)
            if done:
                sl = slice(e + 1 - len(heads), e + 1)
                loss = on_chunk(torch.stack(heads, 1), torch.stack(feedbacks, 1), sl)
                if running:
                    loss.backward()
                heads, feedbacks = [], []
                state = [x.detach() for x in state]
                window = [ddsp.FrameStates(*(v.detach() for v in w)) for w in window]
                own = own.detach()


def batch_step(bf8, encoder, controls, analysis, b, stage, free_probability, chunk, jitter, feedback_frames, train_encoder, train_controls):
    t = make_targets(b)
    speakers = b['speaker']
    y9 = conditioning(encoder, controls, b, train_encoder, train_controls)
    if stage == 1:
        parts = teacher_forced(bf8, analysis, y9, speakers, b, t, jitter, feedback_frames)
        total(parts).backward()
        return parts
    parts = {}
    batch, width = y9.shape[:2]
    free = torch.rand(batch, -(-(width - WARMUP) // chunk), device=y9.device) < free_probability

    def on_chunk(heads, feedback, sl):
        chunk_parts = curve_terms(heads, t, sl)
        chunk_parts['feedback'] = feedback_term(feedback, t, sl)
        chunk_parts['level'] = level_term(feedback, t, sl, analysis.band_hz)
        for k, v in chunk_parts.items():
            parts[k] = parts.get(k, 0.0) + v.detach()
        return total(chunk_parts)

    leaf = y9.detach().requires_grad_(y9.requires_grad)
    rollout(bf8, analysis, leaf, speakers, b, free, chunk, jitter, on_chunk, True)
    if y9.requires_grad:
        y9.backward(leaf.grad)
    return parts


class CapturedStep:
    def __init__(self, parameters, step, example, warmup=3):
        self.parameters, self.step = parameters, step
        self.batch = {k: v.clone() for k, v in example.items()}
        self.probability = torch.zeros((), device=example['valid'].device)
        for p in parameters:
            p.grad = torch.zeros_like(p)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self.run()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode='thread_local'):
            self.parts = {k: v.detach() for k, v in self.run().items()}

    def run(self):
        for p in self.parameters:
            p.grad.zero_()
        return self.step(self.batch, self.probability)

    def __call__(self, batch, probability):
        for k, v in batch.items():
            self.batch[k].copy_(v, non_blocking=True)
        self.probability.fill_(probability)
        self.graph.replay()
        return self.parts


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def conditioning(encoder, controls, b, train_encoder, train_controls):
    with torch.set_grad_enabled(train_encoder):
        features = encoder.run(b['encoder_x'])[:, 2 * dataset.ENCODER_CONTEXT:, :CONTROLS_FEATURES]
    index = b['feature_index']
    gathered = features.gather(1, index.clamp_min(0)[..., None].expand(-1, -1, features.shape[-1])) * (index >= 0)[..., None]
    with torch.set_grad_enabled(train_encoder or train_controls):
        return controls.run(torch.cat([gathered, b['x9'][..., features.shape[-1]:]], -1))[:, CONTROLS_LOOKBACK:]


class Meter:
    def __init__(self):
        self.sums = dict(a=0.0, a_den=0.0, noise=0.0, fb16=0.0, fb80=0.0, steps=0.0)
        self.frame_errors = []

    def __call__(self, t, heads, feedback, sl):
        c = ddsp.curves(heads, t.b['curve_a'].shape[-1])
        valid = t.valid[:, sl]
        w = t.weight[:, sl]
        self.sums['a'] += ((c.A - t.b['curve_a'][:, sl]).abs() * w).sum().item()
        self.sums['a_den'] += w.sum().item()
        self.sums['noise'] += ((ddsp.suppress_peaks(c.A) + c.B - t.b['noise'][:, sl]).abs().mean(-1) * valid).sum().item()
        error = (feedback - t.b['feedback'][:, sl]).abs()
        self.sums['fb16'] += (error[..., :16].mean(-1) * valid).sum().item()
        self.sums['fb80'] += (error[..., 16:].mean(-1) * valid).sum().item()
        self.sums['steps'] += valid.sum().item()
        self.frame_errors.append(error.mean(-1)[valid.bool()].cpu())

    def result(self):
        s = self.sums
        errors = torch.cat(self.frame_errors)
        return dict(curve_a=s['a'] / s['a_den'], noise=s['noise'] / s['steps'], feedback16=s['fb16'] / s['steps'],
                    feedback80=s['fb80'] / s['steps'], feedback=(s['fb16'] + s['fb80']) / (2 * s['steps']),
                    feedback_p95=errors.quantile(0.95).item(),
                    steps=int(s['steps']))


@torch.no_grad()
def validate(bf8, encoder, controls, analysis, tables, index, directory, entries, speaker, jitter, seed, device):
    meter = Meter()
    cuda_devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        for i in range(0, len(entries), VALIDATION_BATCH):
            b = to_device(dataset.full_sections(directory, entries[i:i + VALIDATION_BATCH], tables, index), device)
            t = make_targets(b)
            y9 = conditioning(encoder, controls, b, False, False)
            free = torch.ones(len(b['start']), -(-(b['valid'].shape[1] - WARMUP) // VALIDATION_CHUNK), dtype=torch.bool, device=device)
            speakers = torch.full((len(b['start']),), speaker, device=device)
            rollout(bf8, analysis, y9, speakers, b, free, VALIDATION_CHUNK, jitter, lambda h, f, sl: meter(t, h, f, sl), False)
    return meter.result()


def validation_entries(index, count):
    pool = sorted((e for e in index['sections'] if e['split'] == 'validation' and e['steps'] <= MAX_VALIDATION_STEPS), key=lambda e: e['file'])
    return [pool[i] for i in np.linspace(0, len(pool) - 1, min(count, len(pool))).round().astype(int)]


def learning_rate(step, total_steps, lr, warmup, lr_end):
    ramp = min(1.0, (step + 1) / warmup)
    return ramp * (lr_end + (lr - lr_end) * 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0))))


def resolve_stage_steps(steps, stage_steps):
    if steps is not None and stage_steps is not None:
        raise ValueError('give either a total step count or the three stage lengths, not both')
    if stage_steps is None:
        steps = DEFAULT_STEPS if steps is None else steps
        first, second = (round(steps * f) for f in STAGE_SPLIT)
        stage_steps = (max(first, 1), second, steps - max(first, 1) - second)
    stage_steps = tuple(stage_steps)
    if min(stage_steps) < 0 or not 0 < sum(stage_steps) <= MAX_STEPS:
        raise ValueError(f'training needs between 1 and {MAX_STEPS} steps in total, got stages {stage_steps}')
    return stage_steps


class Ema:
    def __init__(self, parameters, decay=EMA_DECAY):
        self.parameters, self.decay, self.updates = parameters, decay, 0
        self.shadow = [p.detach().clone() for p in parameters]

    @torch.no_grad()
    def update(self):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        torch._foreach_lerp_(self.shadow, [p.detach() for p in self.parameters], 1 - decay)

    @contextlib.contextmanager
    def applied(self):
        backup = [p.detach().clone() for p in self.parameters]
        torch._foreach_copy_([p.data for p in self.parameters], self.shadow)
        try:
            yield
        finally:
            torch._foreach_copy_([p.data for p in self.parameters], backup)


def session_assignment(index, count, first_row):
    recordings = sorted({e['recording'] for e in index['sections'] if e['split'] == 'train'})
    if count > len(recordings):
        raise ValueError(f'{count} session rows need at least that many training recordings, the dataset has {len(recordings)}')
    return {str(recording): first_row + k for k, group in enumerate(np.array_split(recordings, count)) for recording in group}


def parameter_groups(trained, scales):
    groups = []
    for key, module in trained.items():
        if key == nets.SYNTHESIS:
            named = list(module.w.items())
            groups.append(dict(params=[p for n, p in named if not nets.speaker_conditioned(n)], scale=1.0))
            groups.append(dict(params=[p for n, p in named if nets.speaker_conditioned(n) and n != 'embedding'], scale=SPEAKER_LR_SCALE))
            groups.append(dict(params=[p for n, p in named if n == 'embedding'], scale=SPEAKER_LR_SCALE, weight_decay=0.0))
        else:
            groups.append(dict(params=list(module.parameters()), scale=scales[key]))
    return groups


def initial_embedding(bf8, speaker, ranking_path):
    if ranking_path is None:
        return bf8.w['embedding'][speaker].detach().clone()
    init = json.loads(Path(ranking_path).read_text())['init']
    return sum(weight * bf8.w['embedding'][row].detach() for row, weight in init)


def save_checkpoint(path, base_model, base_sha256, speaker, speaker_name, noisy, trained, config, step, metrics, collapse=None, speakers=None):
    states = {k: m.state_dict() for k, m in trained.items()}
    if collapse is not None:
        first, chosen = collapse
        embedding = states[nets.SYNTHESIS]['w.embedding']
        states[nets.SYNTHESIS]['w.embedding'] = torch.cat([embedding[:first], embedding[chosen:chosen + 1]])
    torch.save(dict(base_model=str(base_model), base_sha256=base_sha256, speaker=speaker, speaker_name=speaker_name, noisy_feedback=noisy, speakers=speakers or ({speaker_name: speaker} if speaker_name else {}), nets=states,
                    config=config, step=step, metrics=metrics), path)


def train(base_model, dataset_dir, out_dir, steps=None, stage_steps=None, batch=8, crop=544, final_crop=1088, chunk=32, lr=1e-3, lr_end=2e-5, warmup=100,
          jitter=None, val_every=250, val_sections=12, feedback_frames=64, seed=0, train_encoder=None, train_controls=True, aux_lr_scale=0.3, tf32=False,
          workers=4, log_every=10, resume=False, init=None, histories=(), speaker_name=None, adapt_speaker=False, speaker_init=None, noisy=None, cuda_graph=True, scratch=False,
          session_rows=1, extra_speakers=(), device='cuda'):
    train_encoder = scratch if train_encoder is None else train_encoder
    stage_steps = resolve_stage_steps(steps, stage_steps)
    if scratch and not (train_encoder and train_controls):
        raise ValueError('training from scratch needs the encoder and controls trained as well')
    if scratch and (init or speaker_name or speaker_init):
        raise ValueError('training from scratch starts from random weights, not a checkpoint, a named speaker or a ranked speaker row')
    if adapt_speaker and (speaker_name or speaker_init):
        raise ValueError('a speaker name or ranked speaker row needs a new speaker row, which --adapt-speaker disables')
    if extra_speakers and (adapt_speaker or scratch or session_rows > 1 or histories):
        raise ValueError('extra speakers are new rows trained from their own datasets, which --adapt-speaker, --scratch, --session-rows and --histories do not support')
    if session_rows > 1 and (adapt_speaker or scratch):
        raise ValueError('session rows are new speaker rows, which --adapt-speaker and --scratch disable')
    adapt_speaker = adapt_speaker or scratch
    config = json.loads(json.dumps(locals(), default=str))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    index = dataset.load_index(dataset_dir)
    base_sha256 = file_digest(base_model)
    if base_sha256 != index['base_sha256']:
        raise ValueError(f'dataset was not built for {base_model}')
    speaker = index['speaker']
    models = dataset.Models(base_model, speaker, index['control'], device, index.get('language', 'ja'))
    extra = []
    for item in extra_speakers:
        name, directory = item.split('=', 1)
        extra_index = dataset.load_index(directory)
        if extra_index['base_sha256'] != base_sha256 or extra_index.get('language', 'ja') != index.get('language', 'ja'):
            raise ValueError(f'{directory} was not built for the same base model and language')
        extra.append((name, directory, extra_index))
    if adapt_speaker:
        speaker_name = None
        recording_rows = {}
        rows = [speaker]
    else:
        speaker_name = speaker_name or out_dir.name
        source = initial_embedding(models.synthesis, speaker, speaker_init)
        first_row = models.synthesis.rows
        for _ in range(session_rows):
            models.synthesis.add_speaker(source)
        rows = list(range(first_row, first_row + session_rows))
        recording_rows = session_assignment(index, session_rows, first_row)
        speaker = first_row
    units = [dict(name=speaker_name, rows=rows, directory=dataset_dir, index=index)]
    sources = [dataset.Source(str(dataset_dir), recording_rows, speaker)]
    for name, directory, extra_index in extra:
        models.synthesis.add_speaker(initial_embedding(models.synthesis, extra_index['speaker'], None))
        row = models.synthesis.rows - 1
        units.append(dict(name=name, rows=[row], directory=directory, index=extra_index))
        sources.append(dataset.Source(str(directory), {}, row))
    speaker_rows = {unit['name']: unit['rows'][0] for unit in units} if not adapt_speaker else {}
    bf8, encoder, controls, analysis = models.synthesis.requires_grad_(True), models.encoder, models.controls, models.analysis
    tables = dataset.frontend_tables(models.doc)
    noisy = noisy_feedback(models.doc) if noisy is None else noisy
    analysis.noisy_floor = noisy
    if jitter is None:
        jitter = NOISY_JITTER if noisy else QUIET_JITTER
    if scratch:
        generator = torch.Generator().manual_seed(seed)
        for net in (bf8, encoder, controls):
            nets.randomise(net, generator)
    trained = {nets.SYNTHESIS: bf8}
    scales = {nets.SYNTHESIS: 1.0}
    if train_encoder:
        trained[nets.ENCODER], scales[nets.ENCODER] = encoder.requires_grad_(True), aux_lr_scale
    if train_controls:
        trained[nets.CONTROLS], scales[nets.CONTROLS] = controls.requires_grad_(True), aux_lr_scale
    parameters = [p for m in trained.values() for p in m.parameters()]
    optimizer = torch.optim.AdamW(parameter_groups(trained, scales), lr=lr, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)

    if init:
        initial = load_checkpoint(init)['nets']
        for key, module in trained.items():
            module.load_state_dict(initial[key])

    total_steps = sum(stage_steps)
    start, best, baseline, ema_state = 0, float('inf'), None, None
    state_path = out_dir / 'state.pt'
    if resume and state_path.exists():
        saved = torch.load(state_path, map_location=device)
        for key, module in trained.items():
            module.load_state_dict(saved['nets'][key])
        optimizer.load_state_dict(saved['optimizer'])
        start, best, baseline, ema_state = saved['step'], saved['best'], saved['baseline'], saved.get('ema')
    ema = Ema(parameters)
    if ema_state is not None:
        ema.shadow, ema.updates = [t.to(device) for t in ema_state['shadow']], ema_state['updates']

    for unit in units:
        unit['entries'] = validation_entries(unit['index'], val_sections)
    with (out_dir / 'log.jsonl').open('a') as log_file:

        def log(record):
            log_file.write(json.dumps(record) + '\n')
            log_file.flush()
            print(json.dumps(record), flush=True)

        def run_validation(step):
            results = []
            for unit in units:
                per_row = [validate(bf8, encoder, controls, analysis, tables, unit['index'], unit['directory'], unit['entries'], row, jitter, seed, device)
                           for row in unit['rows']]
                chosen = min(range(len(per_row)), key=lambda k: per_row[k]['feedback'])
                results.append((per_row[chosen], unit['rows'][chosen], per_row))
            first, row, per_row = results[0]
            metrics = {k: (sum(r[0][k] for r in results) / len(results) if isinstance(v, float) else sum(r[0][k] for r in results)) for k, v in first.items()}
            metrics['row'] = row
            if len(per_row) > 1:
                metrics['row_feedback'] = [m['feedback'] for m in per_row]
            if len(results) > 1:
                metrics['speaker_feedback'] = {unit['name']: r[0]['feedback'] for unit, r in zip(units, results)}
            log(dict(step=step, validation=metrics))
            return metrics

        with ema.applied():
            if baseline is None:
                baseline = run_validation(0)
        metrics = dict(baseline=baseline)

        def make_loaders(width, remaining):
            def loader_over(train_set):
                sampler = WeightedRandomSampler(train_set.durations, batch * remaining, True, torch.Generator().manual_seed(seed + total_steps - remaining))
                return iter(DataLoader(train_set, batch, sampler=sampler, num_workers=workers, drop_last=True, persistent_workers=workers > 0))

            made = loader_over(dataset.CropDataset(base_model, sources, 'train', width))
            recorded = loader_over(dataset.CropDataset(base_model, sources, 'train', width, histories)) if histories else None
            return made, recorded

        batches = recorded = loaded_crop = None
        accumulated, counts, timer = {}, {}, time.time()
        steps_captured = {}

        def step_function(kind):
            return lambda b, probability: batch_step(bf8, encoder, controls, analysis, b, kind, probability, chunk, jitter,
                                                     feedback_frames, train_encoder, train_controls)

        for step in range(start, total_steps):
            stage = 1 + (step >= stage_steps[0]) + (step >= stage_steps[0] + stage_steps[1])
            wanted_crop = final_crop if stage == 3 else crop
            if wanted_crop != loaded_crop:
                batches = recorded = None
                batches, recorded = make_loaders(wanted_crop, total_steps - step)
                loaded_crop = wanted_crop
            from_history = recorded is not None and step % 2 == 1
            host_batch = next(recorded if from_history else batches)
            free_probability = 0.0 if stage == 1 else 1.0 if stage == 3 else (step - stage_steps[0] + 1) / stage_steps[1]
            for group in optimizer.param_groups:
                group['lr'] = group['scale'] * learning_rate(step, total_steps, lr, warmup, lr_end)
            b = to_device(host_batch, device)
            kind = 1 if from_history else stage
            if cuda_graph:
                key = (kind, b['valid'].shape[1])
                if key not in steps_captured:
                    if not histories:
                        steps_captured.clear()
                        torch.cuda.empty_cache()
                    steps_captured[key] = CapturedStep(parameters, step_function(kind), b)
                parts = steps_captured[key](b, free_probability)
            else:
                for p in parameters:
                    p.grad = None
                parts = step_function(kind)(b, free_probability)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            ema.update()

            loss = sum(WEIGHTS[k] * v.item() for k, v in parts.items())
            if not np.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at step {step}')
            prefix = 'recorded_' if from_history else ''
            for k, v in dict(loss=loss, grad_norm=grad_norm.item(), **{k: v.item() for k, v in parts.items()}).items():
                accumulated[prefix + k] = accumulated.get(prefix + k, 0.0) + v
                counts[prefix + k] = counts.get(prefix + k, 0) + 1
            done = step + 1
            if done % log_every == 0:
                log(dict(step=done, stage=stage, crop=wanted_crop, free_probability=round(free_probability, 3), seconds_per_step=(time.time() - timer) / log_every,
                         peak_memory_mb=torch.cuda.max_memory_allocated() // 2 ** 20, lr=optimizer.param_groups[0]['lr'],
                         **{k: v / counts[k] for k, v in accumulated.items()}))
                accumulated, counts, timer = {}, {}, time.time()
            if done % val_every == 0 or done == total_steps:
                with ema.applied():
                    validation = run_validation(done)
                    metrics = dict(baseline=baseline, validation=validation, stage=stage)
                    chosen = validation['row']
                    collapse = (speaker, chosen) if session_rows > 1 else None
                    save_checkpoint(out_dir / 'last.pt', base_model, base_sha256, speaker, speaker_name, noisy, trained, config, done, metrics, collapse, speaker_rows)
                    if validation['feedback'] < best:
                        best = validation['feedback']
                        save_checkpoint(out_dir / 'best.pt', base_model, base_sha256, speaker, speaker_name, noisy, trained, config, done, metrics, collapse, speaker_rows)
                torch.save(dict(nets={k: m.state_dict() for k, m in trained.items()}, optimizer=optimizer.state_dict(), step=done,
                                best=best, baseline=baseline, ema=dict(shadow=ema.shadow, updates=ema.updates)), state_path)
                timer = time.time()
        return metrics
