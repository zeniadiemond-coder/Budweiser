import math
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from .frontend import HOP, SAMPLE_RATE as FS
from .reference.bands import analysis_tables
from .reference.bands import hann_window
from .reference.frontend import curve_width, noisy_feedback, scales_by_pitch

FRAME = 512
BINS = FRAME // 2 + 1
ANALYSIS = 1024
ANALYSIS_BINS = ANALYSIS // 2 + 1
ANALYSIS_LAG = 512 + 4 * HOP
WINDOW_FRAMES = 9
HARMONICS = 191
ALPHA = 0.34
DITHER = 5e-5
RAYLEIGH_POWER = math.pi / 4
LOW, HIGH = 50.0, 15000.0
TINY = 1.1754944e-38
SUPPRESS_LOW_HZ, SUPPRESS_HIGH_HZ = 1500.0, 4000.0


class Curves(NamedTuple):
    A: torch.Tensor
    B: torch.Tensor
    phase: torch.Tensor
    decay: torch.Tensor


class FrameStates(NamedTuple):
    amps: torch.Tensor
    f0: torch.Tensor
    noise: torch.Tensor


def curves(heads, width):
    tail = 2 * width
    return Curves(160.0 * torch.sigmoid(2 * heads[..., :width]) - 140.0,
                  240.0 * torch.sigmoid(2 * heads[..., width:tail]) - 120.0,
                  2 * math.pi * torch.sigmoid(heads[..., tail:tail + 4]),
                  math.pi / torch.clamp(400.0 * torch.exp(heads[..., tail + 4:tail + 8]), min=1e-6))


def warp(f, alpha=ALPHA):
    w = f * (2 * math.pi / FS)
    a = torch.atan(torch.sin(w) * alpha / (1 - torch.cos(w) * alpha))
    return (a + a + w) / math.pi


WARP_LOW, WARP_HIGH = (float(warp(torch.tensor(f, dtype=torch.float64))) for f in (LOW, HIGH))


SUPPRESS_LOW, SUPPRESS_HIGH = (float(warp(torch.tensor(f, dtype=torch.float64))) for f in (SUPPRESS_LOW_HZ, SUPPRESS_HIGH_HZ))


def suppress_peaks(curve):
    width = curve.shape[-1]
    a = torch.cat([curve[..., :1], curve, curve[..., -1:], curve[..., -1:]], -1)
    span = WARP_HIGH - WARP_LOW
    low, high = SUPPRESS_LOW, SUPPRESS_HIGH
    i = torch.arange(1, width - 1, device=curve.device)
    position = i * (span / (width - 1)) + WARP_LOW

    def trigger(src, floor, left, right):
        c = src[..., i + 1]
        shoulder = torch.maximum(left, right) + 3.0
        fire = (position > low) & (c > floor) & (left + 3.0 < c) & (right + 3.0 < c)
        gain = torch.clamp((c - shoulder) / 3.0, max=1.0) * torch.clamp((c - floor) / 12.0, max=1.0)
        gain = torch.clamp((position - low) / (high - low) * 4.0 * gain, max=1.0)
        return fire, shoulder, gain

    def previous(fire, by):
        return torch.cat([torch.zeros_like(fire[..., :by]), fire[..., :fire.shape[-1] - by]], -1)

    src = a.clone()
    fire, shoulder, gain = trigger(src, -60.0, src[..., i], src[..., i + 2])
    cut = (1 - previous(fire, 1).float()) * fire.float() * (src[..., i + 1] - shoulder) * gain
    a[..., i + 1] -= cut

    src = a.clone()
    fire, shoulder, gain = trigger(src, -45.0, src[..., i - 1], src[..., i + 3])
    fired = fire.float()
    seen_back = (previous(fire, 1) | previous(fire, 2)).float()
    for offset, free in ((0, 1 - seen_back), (1, 1 - previous(fire, 1).float()), (2, torch.ones_like(fired))):
        a[..., i + offset] -= free * fired * torch.clamp((src[..., i + offset] - shoulder) * gain, min=0.0)
    return a[..., 1:width + 1]


def catmull_rom(t, p0, p1, p2, p3):
    return ((((p3 - p0) * 0.5 + (p1 - p2) * 1.5) * t + ((p0 - p1 * 2.5) + p2 + p2 - p3 * 0.5)) * t + (p2 - p0) * 0.5) * t + p1


def curve_value(curve, f):
    padded = torch.cat([curve[..., :1], curve, curve[..., -1:], curve[..., -1:]], -1)
    x = (warp(f).clamp(WARP_LOW, WARP_HIGH) - WARP_LOW) * ((curve.shape[-1] - 1) / (WARP_HIGH - WARP_LOW))
    i = x.detach().floor().long()
    shape = f.shape
    idx = (i.unsqueeze(-1) + torch.arange(4, device=f.device)).reshape(*shape[:-1], -1)
    p0, p1, p2, p3 = padded.gather(-1, idx).reshape(*shape, 4).unbind(-1)
    return torch.minimum(catmull_rom(x - i, p0, p1, p2, p3), torch.maximum(p1, p2) + 3.01)


def rectangular_response(x):
    return torch.sinc(x)


def hann_response(x):
    singular = (1 - x * x).abs() < 1e-5
    return torch.where(singular, torch.full_like(x, 0.5), torch.sinc(x) / torch.where(singular, torch.ones_like(x), 1 - x * x))


def hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def band_matrix(edges, fft):
    edges = np.asarray(edges, np.float32)
    matrix = np.zeros((len(edges) - 2, fft // 2 + 1), np.float32)
    for b in range(len(matrix)):
        e0, e1, e2 = edges[b:b + 3]
        x0, x1, x2 = fft * e0 / np.float32(FS), fft * e1 / np.float32(FS), fft * e2 / np.float32(FS)
        for k in range(math.ceil(x0), int(x2) + 1):
            matrix[b, k] = min((k - x0) / (x1 - x0), 1 - (k - x1) / (x2 - x1)) * 2 / (e2 - e0)
    return torch.from_numpy(matrix)


def window_coefficients():
    window = hann(ANALYSIS)
    window = window / (window.sum() * 0.5)
    r = np.arange(ANALYSIS) - (ANALYSIS_LAG - HOP)
    h = np.zeros((WINDOW_FRAMES, ANALYSIS))
    for lag in range(WINDOW_FRAMES):
        idx = r + lag * HOP
        inside = (idx >= 0) & (idx < 2 * HOP)
        h[lag, inside] = hann(2 * HOP)[idx[inside]]
    harmonic = (h * window).sum(1) / 2
    noise = (window ** 2 * h ** 2 / (h ** 2).sum(0)).sum(1)
    return torch.from_numpy(harmonic), torch.from_numpy(noise)


def pad_time(x, frames, dim=1):
    shape = list(x.shape)
    shape[dim] = frames
    return torch.cat([x.new_zeros(shape), x], dim)


def decoder_cents(header_cents, initial_cents):
    first = torch.full_like(header_cents[:, :1], initial_cents)
    delayed = torch.cat([first.expand(-1, 3), header_cents], 1)
    return torch.cat([torch.zeros_like(header_cents[:, :6]), delayed[:, :header_cents.shape[1] - 6]], 1)


class FeedbackAnalysis(nn.Module):
    def __init__(self, doc):
        super().__init__()
        edges16, edges80, minima, maxima = analysis_tables(doc)
        self.register_buffer('matrix16', band_matrix(edges16, FRAME))
        self.register_buffer('matrix80', band_matrix(edges80, ANALYSIS))
        self.register_buffer('band_hz', torch.from_numpy(np.concatenate([edges16[1:-1], edges80[1:-1]])))
        self.register_buffer('minima', torch.from_numpy(np.concatenate(minima)))
        self.register_buffer('maxima', torch.from_numpy(np.concatenate(maxima)))
        self.width = curve_width(doc)
        self.scale_by_pitch = scales_by_pitch(doc)
        self.frame_response = hann_response if hann_window(doc) else rectangular_response
        window = hann(2 * HOP) if hann_window(doc) else np.ones(2 * HOP)
        frame_window = window * 2 / window.sum()
        self.register_buffer('window512', torch.from_numpy(np.pad(frame_window, (96, 96)).astype(np.float32)))
        window1024 = hann(ANALYSIS)
        self.register_buffer('window1024', torch.from_numpy((window1024 / (window1024.sum() * 0.5)).astype(np.float32)))
        harmonic, noise = window_coefficients()
        self.register_buffer('harmonic_weights', harmonic.float())
        self.register_buffer('noise_weights', noise.float())
        self.register_buffer('bin_hz', torch.arange(BINS) * (FS / FRAME))
        self.register_buffer('mid_bins', ((self.bin_hz >= 4000.0) & (self.bin_hz < 10000.0)).float())
        self.register_buffer('harmonic_index', torch.arange(1, HARMONICS + 1, dtype=torch.float32))
        self.energy512 = float((frame_window ** 2).sum())
        self.noisy_floor = noisy_feedback(doc)

    def frame_states(self, heads, f0):
        A, B = curves(heads, self.width)[:2]
        A = suppress_peaks(A)
        lead = heads.shape[:-1]
        f = f0.unsqueeze(-1) * self.harmonic_index
        value = 10.0 ** (curve_value(A, f) * 0.05)
        if self.scale_by_pitch:
            value = value * torch.sqrt(f0 / FS).unsqueeze(-1)
        peak = torch.where(f < 4000.0, value, torch.zeros_like(value)).amax(-1, keepdim=True)
        amps = torch.where(f > 10000.0, torch.minimum(value, peak), value) * (f <= FS / 2)

        bins = self.bin_hz.expand(*lead, BINS)
        noise_db = curve_value(A, bins) + torch.clamp(curve_value(B, bins), max=140.0)
        mean_power = ((10.0 ** (noise_db * 0.1)) * self.mid_bins).sum(-1, keepdim=True) / self.mid_bins.sum()
        cap = torch.where(self.bin_hz <= 12000.0, torch.zeros_like(noise_db), 10 * torch.log10(mean_power.clamp_min(TINY)) + 12.0)
        noise = 10.0 ** (torch.minimum(noise_db, cap) * 0.1)
        noise = noise * (torch.arange(BINS, device=noise.device) >= 2)
        return FrameStates(amps, f0, noise)

    def feedback16(self, states):
        k = self.harmonic_index
        delta = self.bin_hz * (FRAME / FS) - (k * states.f0.unsqueeze(-1) * (FRAME / FS)).unsqueeze(-1)
        x = delta * (2 * HOP / FRAME)
        harmonic = (states.amps.unsqueeze(-1) ** 2 * self.frame_response(x) ** 2).sum(-2)
        noise = (states.noise + DITHER ** 2) * self.energy512
        mag = torch.sqrt(harmonic + RAYLEIGH_POWER * noise + 1e-14)
        return 20 * torch.log10(1e-7 + mag @ self.matrix16.T.to(mag.dtype))

    def feedback80(self, window, alpha):
        c = self.harmonic_weights.flip(0)
        d = self.noise_weights.flip(0)
        amps = (window.amps * c[:, None]).sum(-2)
        f0 = (window.f0 * c).sum(-1) / (c * (window.f0 > 0)).sum(-1).clamp_min(1e-9)
        noise = (window.noise * d[:, None]).sum(-2)
        bins = torch.arange(ANALYSIS_BINS, device=amps.device, dtype=amps.dtype)
        delta = bins - (self.harmonic_index * f0.unsqueeze(-1) * (ANALYSIS / FS)).unsqueeze(-1)
        harmonic = (amps.unsqueeze(-1) ** 2 * hann_response(delta) ** 2).sum(-2)
        between = 0.5 * (noise[..., :-1] + noise[..., 1:])
        noise = torch.cat([torch.stack([noise[..., :-1], between], -1).flatten(-2), noise[..., -1:]], -1)
        mag = torch.sqrt(harmonic + RAYLEIGH_POWER * noise)
        return self.bands80(mag, alpha)

    def bands80(self, mag, alpha):
        db = 20 * torch.log10(mag.clamp_min(TINY))
        w = torch.arange(ANALYSIS_BINS, device=mag.device, dtype=mag.dtype) * (2 * math.pi / ANALYSIS)
        alpha = torch.as_tensor(alpha, device=mag.device, dtype=mag.dtype).unsqueeze(-1)
        pos = (2 * torch.atan(torch.sin(w) * alpha / (1 - torch.cos(w) * alpha)) + w) / (2 * math.pi) * ANALYSIS
        i = pos.detach().floor().long().clamp(max=ANALYSIS_BINS - 2)
        lower = db.gather(-1, i)
        lin = 10.0 ** (((db.gather(-1, i + 1) - lower) * (pos - i) + lower) * 0.05)
        return 20 * torch.log10((lin @ self.matrix80.T.to(lin.dtype)).clamp_min(1e-7))

    def windows(self, states):
        padded = FrameStates(*(pad_time(x, WINDOW_FRAMES - 1) for x in states))
        return FrameStates(*(x.unfold(1, WINDOW_FRAMES, 1).movedim(-1, 2) if x.dim() > 2 else x.unfold(1, WINDOW_FRAMES, 1) for x in padded))

    def feedback_sequence(self, states, alpha, chunk=256):
        batch, frames = states.f0.shape
        alpha = torch.as_tensor(alpha, device=states.f0.device, dtype=states.f0.dtype).expand(batch, frames).flatten()
        flat = FrameStates(*(x.flatten(0, 1) for x in states))
        windows = FrameStates(*(x.flatten(0, 1) for x in self.windows(states)))
        out = []
        for i in range(0, batch * frames, chunk):
            part = slice(i, i + chunk)
            out.append(torch.cat([self.feedback16(FrameStates(*(x[part] for x in flat))),
                                  self.feedback80(FrameStates(*(x[part] for x in windows)), alpha[part])], -1))
        return torch.cat(out).reshape(batch, frames, -1)

    def forward(self, heads, f0, alpha):
        return self.feedback_sequence(self.frame_states(heads, f0), alpha)

    def feedback_floor(self, db):
        drawn = 10.0 ** ((torch.randn(*db.shape[:-1], 2, device=db.device).abs() * 12.0 - 140.0) * 0.1)
        return torch.cat([drawn[..., :1].expand(*db.shape[:-1], 16), drawn[..., 1:].expand(*db.shape[:-1], db.shape[-1] - 16)], -1)

    def normalise(self, db, jitter=0.0):
        floor = self.feedback_floor(db) if jitter and self.noisy_floor else 1e-14
        v = 10 * torch.log10((10.0 ** (db * 0.1) + floor).clamp_min(TINY)) - self.minima
        out = (2 * v / (self.maxima - self.minima) - 1).clamp(-1, 1)
        return out + jitter * torch.randn_like(out) if jitter else out

    def real_feedback16(self, audio):
        frames = audio.shape[1] // HOP
        padded = torch.nn.functional.pad(audio, (FRAME // 2 - HOP, FRAME))
        x = padded.unfold(1, FRAME, HOP)[:, :frames] * self.window512
        mag = torch.sqrt(torch.fft.rfft(x.double()).abs() ** 2 + 1e-14).to(audio.dtype)
        return 20 * torch.log10(1e-7 + mag @ self.matrix16.T.to(mag.dtype))

    def real_feedback80(self, audio, alpha):
        frames = audio.shape[1] // HOP
        padded = torch.nn.functional.pad(audio, (ANALYSIS_LAG - HOP, ANALYSIS))
        x = padded.unfold(1, ANALYSIS, HOP)[:, :frames] * self.window1024
        mag = torch.fft.rfft(x.double()).abs().to(audio.dtype)
        alpha = torch.as_tensor(alpha, device=audio.device, dtype=audio.dtype).expand(audio.shape[0], frames)
        return self.bands80(mag, alpha)

    def real_feedback(self, audio, alpha):
        return torch.cat([self.real_feedback16(audio), self.real_feedback80(audio, alpha)], -1)
