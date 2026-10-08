import math

import numpy as np

from ..frontend import FRONTEND
from .rng import Xorshift128

f32 = np.float32
NOISE_CONFIG = '2e24c75e83acd7307571ebfcf965cb26'
FEEDBACK_NOISE = '75ffe1f947489bf0c52a4fb6a0c27ee7'
QUIET_VALUE, NOISY_VALUE = 0.4, 1.2
NOISY_JITTER, QUIET_JITTER = 0.6, 0.1
NOISY_THRESHOLD = f32(0.59900004)
CURVE_WIDTH = 'cc02b901802a2dab4ad6a87db7381441'
SCALE_BY_PITCH = '08626dfefe919f108fb4ba6002db1b23'


def noisy_feedback(doc):
    return bool(f32(doc[NOISE_CONFIG][FEEDBACK_NOISE]) - NOISY_THRESHOLD > 0)


def curve_width(doc):
    return int(doc[FRONTEND][CURVE_WIDTH])


def scales_by_pitch(doc):
    return bool(doc[FRONTEND][SCALE_BY_PITCH])


class ArInput:
    def __init__(self, maxima, minima, groups=(16, 80), noisy_floor=False):
        self.maxima, self.minima, self.groups = maxima, minima, groups
        self.noisy_floor = noisy_floor
        self.rng = Xorshift128()
        self.countdown = 0

    def __call__(self, source):
        out = np.empty(sum(self.groups), f32)
        start = 0
        for g, n in enumerate(self.groups):
            floor_db = -140.0
            if self.noisy_floor:
                floor_db = abs(self.rng.gauss()) * 12.0 - 140.0
            floor = f32(10.0) ** f32(floor_db * 0.1)
            lo, hi = self.minima[g], self.maxima[g]
            for i in range(n):
                v = f32(10.0) ** (source[start + i] * f32(0.1)) + floor
                v = f32(math.log10(max(v, f32(1.1754944e-38)))) * f32(10.0) - lo[i]
                out[start + i] = min(max((v + v) / (hi[i] - lo[i]) - f32(1.0), f32(-1.0)), f32(1.0))
            if self.countdown < 1:
                for _ in range(12):
                    self.rng.next()
                self.countdown = self.rng.next() % 30 + 10
            else:
                self.countdown -= 1
            scale = NOISY_JITTER if self.noisy_floor else QUIET_JITTER
            for i in range(n):
                out[start + i] = f32(self.rng.gauss() * scale + float(out[start + i]))
            start += n
        return out


def pitch_norm(cents):
    p = min(max(cents, -4800.0), 3600.0)
    return f32((p - -4800.0) * 2.0 / (3000.0 - -4800.0) - 1.0)
