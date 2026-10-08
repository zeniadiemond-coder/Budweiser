import copy
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from . import nets, pitch_dataset, pitch_post
from .container import Container, file_digest

CHUNK = pitch_post.CHUNK
HUBER_CENTS = 30.0
RESIDUAL_HUBER_CENTS = 15.0
LOSS_SCALE = 50.0
UNVOICED_WEIGHT = 0.1
HISTORY_NOISE_CENTS = 12.0
HISTORY_NOISE_SMOOTH = 15
EMBEDDING_LR_SCALE = 10.0
CONDITION_LR_SCALE = 3.0
STAT_BANDS = ((0.5, 2.0), (2.0, 3.5))
VIBRATO_BAND = (3.5, 9.0)
ENERGY_FLOOR = 1.0
LOG_RATIO_LIMIT = 2.0
EDGE_FRAMES = 31
SUSTAIN_MIN_POSITION = 0.1
WEIGHTS = dict(base=1.0, residual=1.0, sp=1.0, wander=0.005, vibrato_leak=0.3, vibrato_cycle=0.01)
VIBRATO_CEILING_CENTS = 4.0


def huber(error, delta):
    magnitude = error.abs()
    return torch.where(magnitude < delta, 0.5 * error ** 2 / delta, magnitude - 0.5 * delta)


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def smooth_noise(shape, device, generator=None):
    white = torch.randn(shape[0], 1, shape[1], device=device, generator=generator)
    padded = F.pad(white, (HISTORY_NOISE_SMOOTH // 2,) * 2, mode='reflect')
    return F.avg_pool1d(padded, HISTORY_NOISE_SMOOTH, 1)[:, 0] * math.sqrt(HISTORY_NOISE_SMOOTH)


def make_batch(data, config, section, start, context, length, corrupt, generator=None):
    device = data.device
    positions = start[:, None] + torch.arange(-context, length, device=device)
    frames = data.frames[section][:, None]
    origin = data.offsets[section][:, None]
    index = origin + torch.minimum(positions.clamp_min(0), frames - 1)
    arrays = data.arrays
    f0 = arrays['f0'][:, 0]
    head = arrays['head'][index]
    first_note = arrays['head'][origin, 0].expand_as(index)
    previous = torch.where(positions >= 1, f0[index - 1], torch.where(positions == 0, f0[origin.expand_as(index)], first_note))
    if corrupt is not None:
        previous = previous + corrupt[:, None] * smooth_noise(previous.shape, device, generator) * HISTORY_NOISE_CENTS
    noise = torch.randn(index.shape, device=device, generator=generator) * pitch_post.NOISE_GAIN * (positions >= 0)
    sample = lambda name: arrays[name][index][..., 0]
    return SimpleNamespace(condition=arrays['condition'][index].float(), head=head, previous=previous, noise=noise, base=sample('base'), layer=sample('layer'),
                           valid=sample('valid'), vibrato=sample('vibrato') > 0,
                           offset=(config.phase + start - context) % CHUNK, frames=pitch_post.frame_constants(head))


def crop_starts(data, section, length, generator, aligned_phase=None, config=None):
    room = (data.frames[section] - length - CHUNK).clamp_min(0).float()
    start = (torch.rand(len(section), device=data.device, generator=generator) * (room + 1)).long()
    if aligned_phase is None:
        return start
    return start + (aligned_phase - config.phase - start) % CHUNK


def sample_sections(data, count, length, generator):
    weights = data.weights * (data.frames >= length + CHUNK)
    return torch.multinomial(weights, count, replacement=True, generator=generator)


def band_signal(x, band, rate):
    spectrum = torch.fft.rfft(x * torch.hann_window(x.shape[1], device=x.device), dim=1)
    freqs = torch.fft.rfftfreq(x.shape[1], 1.0 / rate).to(x.device)
    return torch.fft.irfft(spectrum * ((freqs >= band[0]) & (freqs < band[1])), n=x.shape[1], dim=1)


def band_energy(x, mask, band, rate):
    soft = F.avg_pool1d(mask[:, None], EDGE_FRAMES, 1, EDGE_FRAMES // 2, count_include_pad=False)[:, 0]
    return (band_signal(x * soft, band, rate) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)


def log_ratio(measured, reference):
    return (torch.log(measured + ENERGY_FLOOR) - torch.log(reference + ENERGY_FLOOR)).clamp(-LOG_RATIO_LIMIT, LOG_RATIO_LIMIT)


def sustain_mask(batch, length):
    positions = batch.head[:, -length:, 2]
    category = batch.head[:, -length:, 1]
    inside = ((category == 1) | (category == 2)) & (positions > SUSTAIN_MIN_POSITION) & (positions < batch.head[:, -length:, 3] - 0.05)
    return inside.float() * batch.valid[:, -length:]


def regression_terms(smoothed, residual, reference_residual, batch, length):
    valid = batch.valid[:, -length:]
    weight = UNVOICED_WEIGHT + (1 - UNVOICED_WEIGHT) * valid
    base = weighted_mean(huber(smoothed - batch.base[:, -length:], HUBER_CENTS) / LOSS_SCALE, weight)
    target = torch.where(batch.vibrato[:, -length:], batch.layer[:, -length:], reference_residual[:, -length:])
    return dict(base=base, residual=weighted_mean(huber(residual - target, RESIDUAL_HUBER_CENTS) / LOSS_SCALE, weight))


def deviation(signal, batch, length):
    return signal - batch.head[:, -length:, 0]


def statistic_terms(smoothed, residual, batch, length, rate):
    mask = sustain_mask(batch, length)
    output, target = deviation(smoothed, batch, length), deviation(batch.base[:, -length:], batch, length)
    terms = {}
    for i, band in enumerate(STAT_BANDS):
        terms[f'wander{i}'] = log_ratio(band_energy(output, mask, band, rate), band_energy(target, mask, band, rate)) ** 2
    ceiling = torch.minimum(band_energy(target, mask, VIBRATO_BAND, rate), output.new_tensor(VIBRATO_CEILING_CENTS ** 2))
    terms['vibrato_leak'] = log_ratio(band_energy(output, mask, VIBRATO_BAND, rate), ceiling).clamp_min(0.0) ** 2
    active = batch.vibrato[:, -length:].float() * batch.valid[:, -length:]
    terms['vibrato_cycle'] = log_ratio(band_energy(residual, active, VIBRATO_BAND, rate), band_energy(batch.layer[:, -length:], active, VIBRATO_BAND, rate)) ** 2
    return terms


def total(parts):
    return sum(WEIGHTS.get(name.rstrip('01'), 1.0) * value for name, value in parts.items())


def reference_residual(reference, config, batch, speaker):
    with torch.no_grad():
        y, _, _ = pitch_post.teacher_pass(reference, config, batch.previous, batch.condition, batch.noise, speaker)
        return pitch_post.ring_sequence(y, config, batch.offset[:, None])[1]


def teacher_outputs(stack, config, batch, speaker, length):
    y, _, _ = pitch_post.teacher_pass(stack, config, batch.previous, batch.condition, batch.noise, speaker)
    base, residual = pitch_post.ring_sequence(y, config, batch.offset[:, None])
    warm = base.shape[1] - length - pitch_post.POST_WARMUP
    smoothed, _ = pitch_post.post_process(base[:, warm:], pitch_post.slice_frames(batch.frames, warm, base.shape[1]), pitch_post.settled_post_state(base[:, warm]))
    return smoothed[:, -length:], residual[:, -length:]


def free_outputs(stack, config, batch, speaker, context, length):
    steps = context + 1
    with torch.no_grad():
        rollout = pitch_post.Rollout.from_context(stack, config, speaker, batch.previous[:, :steps], batch.condition[:, :steps], batch.noise[:, :steps],
                                                  pitch_post.slice_frames(batch.frames, 0, steps), 0)
    _, residual, smoothed = rollout.emit(batch.condition[:, context:], batch.noise[:, context:], pitch_post.slice_frames(batch.frames, context, context + length), length)
    return smoothed, residual


def regularisation(stack, anchor):
    deviations = [((p - a) ** 2).mean() / (a ** 2).mean().clamp_min(1e-12) for (n, p), a in zip(stack.w.items(), anchor) if n != 'embedding']
    return torch.stack(deviations).mean()


def parameter_groups(stack):
    groups = {1.0: [], CONDITION_LR_SCALE: [], EMBEDDING_LR_SCALE: []}
    for name, parameter in stack.w.items():
        groups[EMBEDDING_LR_SCALE if name == 'embedding' else CONDITION_LR_SCALE if nets.speaker_conditioned(name) else 1.0].append(parameter)
    return [dict(params=params, scale=scale) for scale, params in groups.items()]


def learning_rate(step, total_steps, lr, warmup):
    return lr * min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0))))


def initial_row(stack, init):
    embedding = stack.w['embedding'].detach()
    return embedding.mean(0) if init == 'mean' else embedding[int(init)]


def save_checkpoint(path, base_vpit, base_sha256, speaker, stack, config, step, metrics):
    torch.save(dict(base_model=str(base_vpit), base_sha256=base_sha256, speaker=speaker, nets={pitch_post.NET: stack.state_dict()}, config=config,
                    step=step, metrics=metrics), path)


@torch.no_grad()
def validate(stack, reference, config, data, speaker_row, base_row, context, length, rate, count, seed, free):
    generator = torch.Generator(data.device).manual_seed(seed)
    section = sample_sections(data, count, length, generator)
    batch = make_batch(data, config, section, crop_starts(data, section, length, generator, 0 if free else None, config), context, length, None, generator)
    speaker = torch.full((count,), speaker_row, device=data.device)
    smoothed, residual = free_outputs(stack, config, batch, speaker, context, length) if free else teacher_outputs(stack, config, batch, speaker, length)
    anchor = reference_residual(reference, config, batch, torch.full((count,), base_row, device=data.device))
    result = {name: value.item() for name, value in regression_terms(smoothed, residual, anchor, batch, length).items()}
    mask = sustain_mask(batch, length)
    error = (smoothed - batch.base[:, -length:]).abs()[mask > 0]
    result.update(sustain_median=error.median().item(), sustain_p90=error.quantile(0.9).item())
    for i, band in enumerate(STAT_BANDS + (VIBRATO_BAND,)):
        result[f'band{i}_output'] = band_energy(deviation(smoothed, batch, length), mask, band, rate).sqrt().item()
        result[f'band{i}_target'] = band_energy(deviation(batch.base[:, -length:], batch, length), mask, band, rate).sqrt().item()
    active = batch.vibrato[:, -length:].float() * batch.valid[:, -length:]
    result.update(layer_output=band_energy(residual, active, VIBRATO_BAND, rate).sqrt().item(),
                  layer_target=band_energy(batch.layer[:, -length:], active, VIBRATO_BAND, rate).sqrt().item())
    return result


def train(base_vpit, data_dir, out_dir, base_row, stage_steps=(2500, 1500), batch=16, length=768, free_length=320, lr=2e-4, warmup=100, init=None,
          val_every=250, val_crops=48, seed=0, log_every=10, device='cuda', resume=False):
    config = json.loads(json.dumps(locals(), default=str))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    document = Container.load(base_vpit).document
    stack = nets.CausalStack(document[pitch_post.NET], native_math=True).to(device)
    reference = copy.deepcopy(stack).requires_grad_(False)
    pitch = pitch_post.pitch_config(document, stack, device)
    stack.add_speaker(initial_row(stack, str(base_row) if init is None else init))
    stack.to(device)
    speaker_row = stack.rows - 1
    anchor = [p.detach().clone() for p in stack.w.values()]
    train_data = pitch_dataset.PitchData(data_dir, device, ('train',))
    val_data = pitch_dataset.PitchData(data_dir, device, ('validation',))
    optimizer = torch.optim.Adam(parameter_groups(stack), lr=lr, betas=(0.9, 0.99))
    rate = 1.0 / float(pitch_dataset.FRAME)
    generator = torch.Generator(device).manual_seed(seed)
    total_steps = sum(stage_steps)
    speakers, reference_speakers = torch.full((batch,), speaker_row, device=device), torch.full((batch,), base_row, device=device)
    state_path = out / 'state.pt'
    start_step, best = 0, float('inf')
    if resume and state_path.exists():
        saved = torch.load(state_path, map_location=device)
        stack.load_state_dict(saved['stack'])
        optimizer.load_state_dict(saved['optimizer'])
        start_step, best = saved['step'], saved['best']
    sha = file_digest(base_vpit)
    with (out / 'log.jsonl').open('a') as log_file:

        def log(record):
            log_file.write(json.dumps(record) + '\n')
            log_file.flush()
            print(json.dumps(record), flush=True)

        def run_validation(step, stage):
            metrics = dict(teacher=validate(stack, reference, pitch, val_data, speaker_row, base_row, pitch_post.CONTEXT, length, rate, val_crops, seed, False),
                           free=validate(stack, reference, pitch, val_data, speaker_row, base_row, pitch_post.CONTEXT, free_length, rate, val_crops, seed, True))
            log(dict(step=step, stage=stage, validation=metrics))
            return metrics

        accumulated, counts, timer = {}, {}, time.time()
        for step in range(start_step, total_steps):
            stage = 1 + (step >= stage_steps[0])
            free_probability = 0.0 if stage == 1 else (step - stage_steps[0] + 1) / stage_steps[1]
            free = random.Random(seed * 1_000_003 + step).random() < free_probability
            for group in optimizer.param_groups:
                group['lr'] = group['scale'] * learning_rate(step, total_steps, lr, warmup)
            if not free:
                section = sample_sections(train_data, batch, length, generator)
                crop = make_batch(train_data, pitch, section, crop_starts(train_data, section, length, generator), pitch_post.CONTEXT, length,
                                  (torch.rand(batch, device=device, generator=generator) < 0.5).float(), generator)
                smoothed, residual = teacher_outputs(stack, pitch, crop, speakers, length)
                parts = regression_terms(smoothed, residual, reference_residual(reference, pitch, crop, reference_speakers), crop, length)
            else:
                section = sample_sections(train_data, batch, free_length, generator)
                crop = make_batch(train_data, pitch, section, crop_starts(train_data, section, free_length, generator, 0, pitch), pitch_post.CONTEXT, free_length, None, generator)
                smoothed, residual = free_outputs(stack, pitch, crop, speakers, pitch_post.CONTEXT, free_length)
                parts = regression_terms(smoothed, residual, reference_residual(reference, pitch, crop, reference_speakers), crop, free_length)
                parts.update(statistic_terms(smoothed, residual, crop, free_length, rate))
            parts['sp'] = regularisation(stack, anchor)
            loss = total(parts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(stack.parameters()), 1.0)
            optimizer.step()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at step {step}')
            for name, value in dict(loss=loss, grad_norm=grad_norm, **parts).items():
                accumulated[name] = accumulated.get(name, 0.0) + float(value)
                counts[name] = counts.get(name, 0) + 1
            done = step + 1
            if done % log_every == 0:
                log(dict(step=done, stage=stage, free_probability=round(free_probability, 3), seconds_per_step=(time.time() - timer) / log_every, lr=optimizer.param_groups[0]['lr'],
                         peak_memory_mb=torch.cuda.max_memory_allocated() // 2 ** 20, **{k: v / counts[k] for k, v in accumulated.items()}))
                accumulated, counts, timer = {}, {}, time.time()
            if done % val_every == 0 or done == total_steps:
                metrics = run_validation(done, stage)
                save_checkpoint(out / 'last.pt', base_vpit, sha, speaker_row, stack, config, done, metrics)
                score = metrics['free']['base'] + metrics['teacher']['base']
                if score < best:
                    best = score
                    save_checkpoint(out / 'best.pt', base_vpit, sha, speaker_row, stack, config, done, metrics)
                torch.save(dict(stack=stack.state_dict(), optimizer=optimizer.state_dict(), step=done, best=best), state_path)
                timer = time.time()
