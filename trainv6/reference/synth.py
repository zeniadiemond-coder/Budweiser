import math
from typing import NamedTuple

import numpy as np

from .bands import analysis_tables, hann_window
from .decoder import Decoder, band_analysis, FS, f32
from .frontend import ArInput, curve_width, noisy_feedback, pitch_norm, scales_by_pitch
from .rng import Lanes32, Xorshift128

HOP = 160
WARMUP = 77
FEATURE_DELAY = 2
DECODE_DELAY = 6
POLY = (f32(0.004333095), f32(-0.07943434), f32(0.64589286), f32(-1.570791))


class HarmonicRing:
    def __init__(self, size=4096, width=192):
        self.rows_ = np.zeros((size, width), f32)
        self.mask = size - 1
        self.phase = 0.0
        self.k = np.arange(1, width + 1, dtype=np.uint32)

    def advance(self, t, f_hz):
        blocks = min(24, int((FS * f32(0.5) / f_hz) * f32(0.125) + f32(1.0)))
        n = 8 * blocks
        q = np.uint32(np.int32(np.rint(f32(self.phase) * f32(16777216.0))))
        v = (self.k[:n] * q).astype(np.uint32)
        x = (v & np.uint32(0x7fffff)).astype(np.int32).astype(f32) * f32(2.3841858e-07) + f32(-1.0)
        x2 = x * x
        y = (((x2 * POLY[0] + POLY[1]) * x2 + POLY[2]) * x2 + POLY[3]) * x
        sign = (v >> np.uint32(23)) << np.uint32(31)
        self.rows_[t & self.mask, :n] = (y.view(np.uint32) ^ sign).view(f32)
        self.phase += float(f_hz) / float(FS)
        if self.phase > 1.0:
            self.phase -= 1.0

    def rows(self, start, n):
        return self.rows_[(np.arange(start, start + n)) & self.mask]


class Ola:
    def __init__(self):
        self.slots = [None, None]

    def push(self, start, segment):
        self.slots[(start // HOP) & 1] = (start, segment)

    def pop(self, t):
        s = f32(0.0)
        for slot in self.slots:
            if slot is not None and 0 <= t - slot[0] < len(slot[1]):
                s = s + slot[1][t - slot[0]]
        return s


class Result(NamedTuple):
    harm: np.ndarray
    noise: np.ndarray
    heads: np.ndarray
    sources: np.ndarray
    events: dict


class Synth:
    def __init__(self, doc, speaker, controls, bf8_step):
        edges16, edges80, minima, maxima = analysis_tables(doc)
        self.controls = controls
        self.bf8_step = bf8_step
        self.ar = ArInput(maxima, minima, noisy_floor=noisy_feedback(doc))
        self.head_width = 2 * curve_width(doc) + 8
        self.decoder = Decoder(edges16, curve_width(doc), scales_by_pitch(doc), hann_window(doc))
        self.edges80 = edges80
        self.speaker = speaker
        self.jitter = Xorshift128()
        self.noise_rng = Lanes32()
        self.ring = HarmonicRing()
        k = np.arange(2 * HOP)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * k / (2 * HOP))).astype(f32)
        self.ola = [Ola(), Ola(), Ola()]
        self.harm_out = np.zeros(1 << 20, f32)
        self.noise_out = np.zeros(1 << 20, f32)
        self.mask = (1 << 20) - 1
        k = np.arange(1024)
        w = (0.5 - 0.5 * np.cos(2 * np.pi * k / 1024)).astype(f32)
        self.analysis_window = w / (w.sum() * f32(0.5))

    def run(self, frames, initial):
        n = len(frames)
        prev = dict(initial)
        pitch_ring = np.full(4096, f32(initial['f0']), f32)
        features = [np.zeros(40, f32)] * FEATURE_DELAY
        x9 = [np.concatenate([initial['vec'][7:39], np.zeros(4, f32)])] * WARMUP
        plan = []
        for k in range(n):
            features.append(prev['vec'][7:].astype(f32))
            feat = features[-1 - FEATURE_DELAY]
            p = pitch_ring_value(initial, frames, k)
            control = min(max(prev['control'], -1.5), 1.5)
            x9.append(np.concatenate([feat[:32], [pitch_norm(p), 1.0, 0.0, control]]).astype(f32))
            plan.append(p)
            prev = frames[k]
        y9 = self.controls(np.array(x9, f32))
        source = np.full(96, f32(-140.0))
        for w in range(WARMUP):
            self.bf8_step(self.ar(source), y9[w], self.speaker)
        harm = np.zeros(n * HOP, f32)
        noise = np.zeros(n * HOP, f32)
        prev = dict(initial)
        pitches = []
        heads = np.zeros((n, self.head_width), f32)
        sources = np.zeros((n, 96), f32)
        for k in range(n):
            cur = frames[k]
            t0 = k * HOP
            for s in range(HOP):
                t = t0 + s
                cents = f32(f32(cur['f0'] - prev['f0']) * (f32(s) / f32(HOP)) + f32(prev['f0']))
                g = f32(self.jitter.gauss())
                g = min(max(g, f32(-2.0)), f32(2.0))
                f_hz = f32(2.0) ** ((g + cents) / f32(1200.0)) * f32(440.0)
                self.ring.advance(t, f_hz)
                pitch_ring[t & 4095] = cents
                if s == 0:
                    heads[k] = self.frame(k, t, plan[k], y9[WARMUP + k], source, pitches)[:self.head_width]
                    self.analyse(t, source, cur['alpha'])
                    sources[k] = source
                harm[t] = self.harm_out[(t - HOP) & self.mask]
                noise[t] = self.noise_out[(t - HOP) & self.mask]
            fade(harm[t0:t0 + HOP], noise[t0:t0 + HOP], prev['vec'])
            prev = cur
        return Result(harm, noise, heads, sources, dict(self.decoder.events, frames=n))

    def frame(self, k, t, pitch, cond, source, pitches):
        head = self.bf8_step(self.ar(source), cond, self.speaker)
        pitches.append(pitch)
        cents = pitches[k - DECODE_DELAY] if k >= DECODE_DELAY else f32(0.0)
        shaped, noise, fb = self.decoder(head, cents, 192, t - (FEATURE_DELAY + DECODE_DELAY) * HOP, self.ring, self.noise_rng)
        source[:16] = fb
        source[16:] = -140.0
        centre = 256
        seg_h = shaped[centre - HOP:centre + HOP] * self.window
        seg_n = noise[centre - HOP:centre + HOP] * self.window
        self.ola[0].push(t, seg_h)
        self.ola[1].push(t, seg_n)
        self.ola[2].push(t, self.window * self.window)
        for i in range(HOP):
            u = t + i
            h = self.ola[0].pop(u)
            nz = self.ola[1].pop(u)
            w2 = self.ola[2].pop(u)
            self.harm_out[(u - HOP) & self.mask] = h
            self.noise_out[(u - HOP) & self.mask] = nz / max(f32(math.sqrt(w2)), f32(0.01))
        return head

    def analyse(self, t, source, alpha):
        idx = (np.arange(1024) + t - 512 - 4 * HOP) & self.mask
        x = (self.harm_out[idx] + self.noise_out[idx]) * self.analysis_window
        mag = np.abs(np.fft.rfft(x.astype(np.float64)))
        db = 20 * np.log10(np.maximum(mag, 1.1754944e-38))
        u = np.arange(513)
        w = u * 2 * np.pi / 1024
        pos = (2 * np.arctan(np.sin(w) * alpha / (1 - np.cos(w) * alpha)) + w) / (2 * np.pi) * 1024
        i = np.minimum(np.floor(pos).astype(int), 511)
        lin = 10 ** (((db[i + 1] - db[i]) * (pos - i) + db[i]) * 0.05)
        source[16:] = band_analysis(lin.astype(f32), self.edges80, 1024, add_floor=False)


def pitch_ring_value(initial, frames, k):
    t = k * HOP - 2 * HOP
    if t < 0:
        return f32(initial['f0'])
    j, s = divmod(t, HOP)
    prev = initial if j == 0 else frames[j - 1]
    return f32(f32(frames[j]['f0'] - prev['f0']) * (f32(s) / f32(HOP)) + f32(prev['f0']))


def fade(harm, noise, vec):
    if round(float(vec[0])) != 0 or not vec[5] > 0.2:
        return
    end, h6 = vec[5] - f32(0.1), vec[6]
    s = np.arange(HOP, dtype=f32) * f32(0.5) / f32(HOP)
    if 0.09 <= h6 < 0.095:
        g = f32(1.0) - s
    elif 0.095 <= h6 < 0.1:
        g = f32(0.5) - s
    elif 0.1 <= h6 < end:
        g = np.zeros(HOP, f32)
    elif end <= h6 < 0.11 and not h6 < 0.09:
        g = s + f32(0.5)
    else:
        return
    harm *= g
    noise *= g
