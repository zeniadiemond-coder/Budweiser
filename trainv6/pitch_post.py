import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F

from . import frontend
f32 = np.float32

DERIVATIVE_WIDTH = '4736d5027136dcf78ab9c95afeb80c60'
LOW, HIGH = '205192c4ec6717232a8e5e2515841a42', 'f98f2b4f4eab060dad241d3e4fb1c4dc'
CHUNK = 16
SLOTS = 32
CONTEXT = 160
POST_WARMUP = 32
NET = '20b8dac0362664a54f087204590292d1'
NOISE_GAIN = 0.5
RESIDUAL_RANGE = 1200.0
HOLD_DECAY = 0.8999999761581421

Config = namedtuple('Config', 'low high taps receptive_field phase')
Frames = namedtuple('Frames', 'floats flags')
PostState = namedtuple('PostState', 'high_x high_y hold first second')


def derivative_taps(width):
    half = int(f32(width) / frontend.PERIOD) // 2
    if 2 * half + 1 < 2:
        return np.ones(1, f32)
    taps = np.array([f32((f32(math.cos(f32(k * 6.2831855) / f32(2 * half))) * f32(0.46) + f32(0.54)) * f32(k))
                     for k in range(-half, half + 1)][::-1], f32)
    return (taps / (np.abs(taps).sum(dtype=f32) * f32(2.0))).astype(f32)


def pitch_config(document, stack, device):
    top = document[frontend.FRONTEND]
    receptive_field = 2 * sum(stack.dilations) + 1
    return Config(top[LOW], top[HIGH], torch.from_numpy(derivative_taps(top[DERIVATIVE_WIDTH])).to(device), receptive_field, receptive_field % CHUNK)


def normalise(cents, config):
    return ((cents - config.low) * 2.0 / (config.high - config.low) - 1.0).clamp(-1.0, 1.0)


def feedback_inputs(previous, history, noise, config):
    joined = torch.cat([history, previous], 1)
    slope = (joined.unfold(1, len(config.taps), 1).flip(-1) * config.taps).sum(-1)
    inputs = torch.stack([normalise(previous, config), torch.tanh(slope) + noise], -1)
    return inputs, joined[:, joined.shape[1] - len(config.taps) + 1:]


def condition_input(condition):
    return F.pad(condition, (0, 1))


def split_outputs(y, config):
    pairs = y.unflatten(-1, (SLOTS, 2))
    return (pairs[..., 0] + 1.0) * 0.5 * (config.high - config.low) + config.low, (pairs[..., 1] + 1.0) * 0.5 * RESIDUAL_RANGE - 0.5 * RESIDUAL_RANGE


def ring_values(current, older, offsets):
    index = offsets[..., None]
    chosen = current.gather(-1, index).squeeze(-1)
    earlier = older.gather(-1, (index + CHUNK).clamp_max(SLOTS - 1)).squeeze(-1)
    weight = offsets.to(chosen.dtype) * 0.125
    return torch.where(offsets < 8, (1.0 - weight) * earlier + weight * chosen, chosen)


def ring_sequence(y, config, first_offset):
    base, residual = split_outputs(y, config)
    steps = y.shape[1]
    positions = torch.arange(steps, device=y.device)
    offsets = ((torch.as_tensor(first_offset, device=y.device).reshape(-1, 1) + positions) % CHUNK).expand(len(y), -1)
    start = (positions - offsets).clamp_min(0)
    older = (start - CHUNK).clamp_min(0)
    gather = lambda values, at: values.gather(1, at[..., None].expand(-1, -1, SLOTS))
    return (ring_values(gather(base, start), gather(base, older), offsets),
            ring_values(gather(residual, start), gather(residual, older), offsets))


def frame_constants(head):
    note, category, position, dur, next_nucleus, previous_nucleus = head.unbind(-1)
    category, pos = torch.round(category), position.clamp_min(0.0)
    active = ((category == 1) | (category == 2)) & (dur > pos)
    attack = torch.minimum(torch.clamp_min((0.6 - torch.clamp_min(0.2 - dur, 0.0) * 0.2 / 0.2) * dur, 0.1), dur * 0.6)
    release = torch.maximum(dur * 0.65, torch.minimum(dur - 0.1, dur * 0.8))
    attack_weight = (pos / attack) ** 2
    release_weight = torch.sqrt(torch.clamp_min(1.0 - (pos - release) / (dur - release), 0.0))
    floats = torch.stack([note, pos, attack_weight, torch.clamp_max(attack_weight + 0.6, 1.0),
                          torch.where(pos < release, torch.ones_like(pos), release_weight)])
    flags = torch.stack([active, active & (pos < 0.005), pos < attack, previous_nucleus <= 0.5, (dur < 0.25) & (next_nucleus > 0.5)])
    return Frames(torch.nan_to_num(floats, nan=0.0, posinf=1.0, neginf=0.0).permute(2, 0, 1).contiguous(), flags.permute(2, 0, 1).contiguous())


def slice_frames(frames, start, stop):
    return Frames(frames.floats[start:stop], frames.flags[start:stop])


def biquad_coefficients():
    k = f32(math.tan(f32(1.2566371))) * f32(2.0)
    k2 = k * k
    den = k * f32(4.0) + k2 + f32(4.0)
    return float(k2 / den), float((k2 + k2) / den), float(-((k2 + k2) - f32(8.0)) / den), float(-((k2 - k * f32(4.0)) + f32(4.0)) / den)


def highpass_coefficients():
    t = float(math.tan(f32(0.25132743)))
    g = 1.0 / (t + 1.0)
    return g, -g, (1.0 - t) * g


HIGHPASS = highpass_coefficients()
BIQUAD = biquad_coefficients()


def settled_post_state(base):
    zeros = torch.zeros_like(base)
    return PostState(base, base, zeros, (base,) * 4, (base,) * 4)


def biquad(x, state):
    b0, b1, a1, a2 = BIQUAD
    x1, x2, y1, y2 = state
    y = x1 * b1 + x * b0 + b0 * x2 + y1 * a1 + a2 * y2
    return y, (x, x1, y, y1)


def post_step(base, floats, flags, state):
    hp0, hp1, hp2 = HIGHPASS
    note, pos, attack_weight, attack_forced, release_weight = floats.unbind(0)
    active, reset, attacking, previous_silent, short = flags.unbind(0)
    below = note < base
    pulled = torch.where(below, base - (base - note) * 0.7, base)
    high_x, high_y = torch.where(reset, base, state.high_x), torch.where(reset, base, state.high_y)
    hold = torch.where(reset, torch.zeros_like(state.hold), state.hold)
    high = high_x * hp1 + hp0 * base + high_y * hp2
    weight = torch.where(attacking, torch.where(below & previous_silent, attack_forced, attack_weight), release_weight)
    weight = torch.where(short & (hold > 1), weight * HOLD_DECAY ** hold, weight)
    snapped = torch.where(active, base - weight * (base - note) + high * torch.minimum(weight, pos), pulled)
    smoothed, first = biquad(snapped, state.first)
    smoothed, second = biquad(smoothed, state.second)
    hold = hold + ((smoothed - note).abs() < 10.0).to(hold.dtype)
    return smoothed, PostState(torch.where(active, base, high_x), torch.where(active, high, high_y), hold, first, second)


def post_process(base, frames, state):
    outputs = []
    for t, (floats, flags) in enumerate(zip(frames.floats.unbind(0), frames.flags.unbind(0))):
        smoothed, state = post_step(base[:, t], floats, flags, state)
        outputs.append(smoothed)
    return torch.stack(outputs, 1), state


def teacher_pass(stack, config, previous, condition, noise, speaker):
    inputs, history = feedback_inputs(previous, previous.new_zeros(len(previous), len(config.taps) - 1), noise, config)
    y, state = stack.block(inputs, condition_input(condition), speaker, stack.init_state(len(previous), previous.device))
    return y, state, history


class Rollout:
    def __init__(self, stack, config, speaker, state, history, current, older, offset, post_state):
        self.stack, self.config, self.speaker = stack, config, speaker
        self.state, self.history, self.current, self.older = state, history, current, older
        self.offset, self.post_state = offset, post_state

    @classmethod
    def from_context(cls, stack, config, speaker, previous, condition, noise, frames, offset):
        y, state, history = teacher_pass(stack, config, previous, condition, noise, speaker)
        steps = previous.shape[1]
        base, _ = ring_sequence(y, config, (offset - (steps - 1)) % CHUNK)
        warm = steps - 1 - POST_WARMUP
        _, post_state = post_process(base[:, warm:steps - 1], slice_frames(frames, warm, steps - 1), settled_post_state(base[:, warm]))
        start = steps - 1 - offset
        return cls(stack, config, speaker, state, history, y[:, start], y[:, start - CHUNK], offset, post_state)

    def step_block(self, previous, noise, condition):
        inputs, self.history = feedback_inputs(previous, self.history, noise, self.config)
        y, self.state = self.stack.block(inputs, condition_input(condition), self.speaker, self.state)
        return y

    def emit(self, condition, noise, frames, count):
        outputs, residuals, smooth = [], [], []
        j = 0
        while j < count:
            m = min(CHUNK - self.offset, count - j)
            offsets = torch.arange(self.offset, self.offset + m, device=condition.device)[None].expand(len(condition), -1)
            spread = lambda v: v[:, None].expand(-1, m, -1)
            (base, residual), (old_base, old_residual) = split_outputs(self.current, self.config), split_outputs(self.older, self.config)
            frame_base = ring_values(spread(base), spread(old_base), offsets)
            frame_residual = ring_values(spread(residual), spread(old_residual), offsets)
            smoothed, self.post_state = post_process(frame_base, slice_frames(frames, j, j + m), self.post_state)
            out = smoothed + frame_residual
            if j + m < count:
                y = self.step_block(out, noise[:, j + 1:j + m + 1], condition[:, j + 1:j + m + 1])
                if self.offset + m == CHUNK:
                    self.older, self.current = self.current, y[:, -1]
            outputs.append(out)
            residuals.append(frame_residual)
            smooth.append(smoothed)
            self.offset = (self.offset + m) % CHUNK
            j += m
        return torch.cat(outputs, 1), torch.cat(residuals, 1), torch.cat(smooth, 1)
