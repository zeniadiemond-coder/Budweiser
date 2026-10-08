import math
from typing import NamedTuple

import numpy as np
import torch

from . import ddsp
from .frontend import FRONTEND, HOP, SAMPLE_RATE, cents_to_hz

WIDTH_KEY = 'cc02b901802a2dab4ad6a87db7381441'
ALPHA_KEY = 'b378321d3911b8fa49ea0e7fb52b043c'
LOW_KEY = '01e3713136ef99dd5282f837d41d0d27'
HIGH_KEY = 'b76820dc488f421fd0cf6ae6a280ecc1'

HEAD_DELAY = 8
HARMONIC_PERIODS = 3
UNVOICED_PITCH = 150.0
MAX_HARMONIC_HZ = 15000.0
UNVOICED_A = -110.0
NFFT = 4096
NOISE_FLOOR = 1e-14
NOISE_SMOOTHING_HZ = 150.0
MAINLOBE_HALF_WIDTH = 1 / HARMONIC_PERIODS
DETECTION_FLOOR = 0.05
TIME_TAPS = (1, 2, 3, 2, 1)
BATCH_ELEMENTS = 24_000_000


class Estimate(NamedTuple):
    curve_a: np.ndarray
    curve_b: np.ndarray
    harmonics: np.ndarray
    noise_power: np.ndarray


def warped(frequency, alpha):
    return ddsp.warp(torch.as_tensor(frequency, dtype=torch.float64), alpha).numpy()


def frequency_grid(doc):
    cfg = doc[FRONTEND]
    width, alpha, low, high = cfg[WIDTH_KEY], cfg[ALPHA_KEY], cfg[LOW_KEY], cfg[HIGH_KEY]
    goal = np.linspace(warped(low, alpha), warped(high, alpha), width)
    left, right = np.full(width, float(low)), np.full(width, float(high))
    for _ in range(60):
        middle = (left + right) / 2
        below = warped(middle, alpha) < goal
        left = np.where(below, middle, left)
        right = np.where(below, right, middle)
    return (left + right) / 2


def noise_smoothing(grid):
    spacing = np.diff(np.r_[grid[0] - (grid[1] - grid[0]), grid])
    bins = np.fft.rfftfreq(NFFT, 1 / SAMPLE_RATE)
    weights = np.exp(-0.5 * ((bins[:, None] - grid[None]) / np.maximum(spacing, NOISE_SMOOTHING_HZ)) ** 2)
    return weights / weights.sum(axis=0)


def smooth_in_runs(curves, labels):
    kernel = np.asarray(TIME_TAPS, np.float64)
    half = len(kernel) // 2
    out = curves.astype(np.float64)
    edges = np.flatnonzero(np.diff(labels)) + 1
    for start, stop in zip(np.r_[0, edges], np.r_[edges, len(labels)]):
        if labels[start] < 0:
            continue
        run = np.pad(curves[start:stop].astype(np.float64), ((half, half), (0, 0)))
        weight = np.pad(np.ones(stop - start), half)
        taps = range(len(kernel))
        out[start:stop] = (sum(kernel[i] * run[i:i + stop - start] for i in taps) /
                           sum(kernel[i] * weight[i:i + stop - start] for i in taps)[:, None])
    return out.astype(np.float32)


def curves_to_logits(curve_a, curve_b):
    a = (np.asarray(curve_a, np.float64) + 140) / 160
    b = (np.asarray(curve_b, np.float64) + 120) / 240
    return np.concatenate((0.5 * np.log(a / (1 - a)), 0.5 * np.log(b / (1 - b))), axis=-1).astype(np.float32)


def analyse_batch(audio, centers, pitch, voiced, track, grid, scale_by_pitch, smoothing, bins, device):
    half = torch.round(HARMONIC_PERIODS * SAMPLE_RATE / pitch).long()
    length = 2 * half + 1
    width = int(length.max())
    n = torch.arange(width, device=device)
    inside = n[None] < length[:, None]
    index = centers[:, None] + n[None] - half[:, None]
    in_range = inside & (index >= 0) & (index < audio.shape[0])
    samples = torch.where(in_range, audio[index.clamp(0, audio.shape[0] - 1)], torch.zeros((), device=device))
    window = torch.where(inside, 0.5 - 0.5 * torch.cos(2 * math.pi * n[None] / (length[:, None] - 1)), torch.zeros((), device=device))
    window_sum = window.sum(dim=1, keepdim=True)

    count = torch.where(voiced, (MAX_HARMONIC_HZ / pitch).floor().long(), torch.zeros_like(half))
    harmonics = int(count.max())
    residual = samples
    z = torch.zeros(len(centers), 0, dtype=torch.complex64, device=device)
    projection_noise = torch.zeros(len(centers), 0, dtype=torch.float64, device=device)
    if harmonics > 0:
        cycles = (track[index.clamp(0, audio.shape[0] - 1)] - track[centers.clamp(0, audio.shape[0] - 1)][:, None]) / (2 * math.pi)
        k = torch.arange(1, harmonics + 1, device=device)
        phase = (k[None, None] * cycles[:, :, None]).float()
        phase = 2 * math.pi * (phase - phase.floor())
        cos, sin = torch.cos(phase), torch.sin(phase)
        active = (k[None] <= count[:, None]).float()
        weighted = (window * samples)[:, :, None]
        a = 2 * (weighted * cos).sum(dim=1) / window_sum * active
        b = 2 * (weighted * sin).sum(dim=1) / window_sum * active
        residual = samples - (cos * a[:, None] + sin * b[:, None]).sum(dim=2)
        z = torch.complex(a, -b)

    spectrum = torch.fft.rfft(window * residual, n=NFFT)
    power = (spectrum.abs() ** 2 / (window ** 2).sum(dim=1, keepdim=True)).double().clamp_min(NOISE_FLOOR)
    ratio = bins[None].double() / pitch[:, None].double()
    valley = ((ratio - ratio.round()).abs() >= MAINLOBE_HALF_WIDTH) | ~voiced[:, None]
    noise_db = 10 * torch.log10((power * valley) @ smoothing / (valley.double() @ smoothing).clamp_min(1e-12))

    amplitude_db = torch.full((len(centers), len(grid)), UNVOICED_A, device=device, dtype=torch.float64)
    if harmonics > 0:
        frequencies = k[None].double() * pitch[:, None].double()
        grid_rows = grid[None].expand(len(centers), -1)
        noise_at_harmonics = 10 ** (_interp_rows(frequencies, grid_rows, noise_db, torch.full_like(half, len(grid))) / 10)
        projection_noise = 4 * (window ** 2).sum(dim=1, keepdim=True).double() / window_sum.double() ** 2 * noise_at_harmonics
        measured = a.double() ** 2 + b.double() ** 2
        power_k = torch.maximum(measured - projection_noise, DETECTION_FLOOR * measured).clamp_min(1e-18)
        level = power_k * SAMPLE_RATE / pitch.double()[:, None] if scale_by_pitch else power_k
        harmonic_db = 10 * torch.log10(level)
        amplitude_db = torch.where(voiced[:, None], _interp_rows(grid_rows, frequencies, harmonic_db, count), amplitude_db)
    curve_a = amplitude_db.clamp(-139.9, 19.9)
    curve_b = (noise_db - curve_a).clamp(-119.9, 119.9)
    return curve_a, curve_b, z, projection_noise


def _interp_rows(x, xp, values, count):
    last = (count - 1).clamp_min(0)[:, None]
    upper = torch.minimum(torch.searchsorted(xp.contiguous(), x.contiguous()).clamp_min(1), last)
    lower = (upper - 1).clamp_min(0)
    x0, x1 = xp.gather(1, lower), xp.gather(1, upper)
    y0, y1 = values.gather(1, lower), values.gather(1, upper)
    t = ((x - x0) / (x1 - x0).clamp_min(1e-9)).clamp(0, 1)
    return y0 + (y1 - y0) * t


def estimate_curves(audio, centers, f0_hz, voiced, grid, scale_by_pitch, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    smoothing = torch.as_tensor(noise_smoothing(grid), device=device)
    bins = torch.as_tensor(np.fft.rfftfreq(NFFT, 1 / SAMPLE_RATE), device=device)
    grid_t = torch.as_tensor(grid, device=device)
    audio_t = torch.as_tensor(audio, dtype=torch.float32, device=device)
    centers = np.asarray(centers)
    f0_track = np.interp(np.arange(len(audio)), centers[voiced], f0_hz[voiced]) if voiced.any() else np.zeros(len(audio))
    track = torch.as_tensor(2 * np.pi * np.cumsum(f0_track) / SAMPLE_RATE, device=device)
    pitch = np.where(voiced, f0_hz, UNVOICED_PITCH)
    order = np.argsort(pitch)
    curve_a = np.empty((len(centers), len(grid)), np.float32)
    curve_b = np.empty((len(centers), len(grid)), np.float32)
    width = int(MAX_HARMONIC_HZ / pitch[voiced].min()) + 1 if voiced.any() else 1
    harmonics = np.zeros((len(centers), width), np.complex64)
    noise_power = np.zeros((len(centers), width), np.float32)
    start = 0
    while start < len(order):
        top_pitch = pitch[order[start]]
        per_frame = (2 * round(HARMONIC_PERIODS * SAMPLE_RATE / top_pitch) + 1) * (MAX_HARMONIC_HZ / top_pitch + 1)
        stop = min(len(order), start + max(1, int(BATCH_ELEMENTS // per_frame)))
        chosen = order[start:stop]
        a, b, z, noise = analyse_batch(audio_t, torch.as_tensor(centers[chosen], device=device),
                             torch.as_tensor(pitch[chosen], dtype=torch.float32, device=device),
                             torch.as_tensor(voiced[chosen], device=device), track, grid_t, scale_by_pitch, smoothing, bins, device)
        curve_a[chosen] = a.cpu().numpy()
        curve_b[chosen] = b.cpu().numpy()
        harmonics[chosen, :z.shape[1]] = z.cpu().numpy()
        noise_power[chosen, :z.shape[1]] = noise.cpu().numpy()
        start = stop
    return Estimate(curve_a, curve_b, harmonics, noise_power)


def measure(audio, f0_cents, voiced, steps, grid, scale_by_pitch, device=None):
    centers = (np.arange(steps) - HEAD_DELAY) * HOP
    frame = np.clip(np.arange(steps) - HEAD_DELAY, 0, len(f0_cents) - 1)
    f0_hz = cents_to_hz(np.asarray(f0_cents, np.float64)[frame])
    estimate = estimate_curves(audio, centers, f0_hz, np.asarray(voiced)[frame], grid, scale_by_pitch, device)
    return curves_to_logits(estimate.curve_a, estimate.curve_b), estimate, f0_hz
