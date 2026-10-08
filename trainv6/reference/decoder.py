import math

import numpy as np

from .native_math import native_exp

f32 = np.float32
FS = f32(32000.0)
LOW, HIGH = f32(50.0), f32(15000.0)
ALPHA = f32(0.34)
FRAME = 512


def warp(f, alpha=ALPHA, fs=FS):
    w = f32(f) * f32(6.2831855) / fs
    s, c = f32(math.sin(w)), f32(math.cos(w))
    a = f32(math.atan(s * alpha / (f32(1.0) - c * alpha)))
    return (a + a + w) / f32(3.1415927)


def allpass_hz(f, alpha, fs=FS):
    w = f32(f) * f32(6.2831855) / fs
    s, c = f32(math.sin(w)), f32(math.cos(w))
    a = f32(math.atan(s * alpha / (f32(1.0) - c * alpha)))
    return (a + a + w) * fs / f32(6.2831855)


def catmull_rom(t, p):
    p0, p1, p2, p3 = p
    return ((((p3 - p0) * f32(0.5) + (p1 - p2) * f32(1.5)) * t
             + (((p0 - p1 * f32(2.5)) + p2 + p2) - p3 * f32(0.5))) * t + (p2 - p0) * f32(0.5)) * t + p1


WARP_LOW, WARP_HIGH = warp(LOW), warp(HIGH)


def curve_value(curve, f):
    x = (min(max(warp(f), WARP_LOW), WARP_HIGH) - WARP_LOW) * f32(len(curve) - 4) / (WARP_HIGH - WARP_LOW)
    i = int(x)
    t = f32(x - f32(i))
    v = catmull_rom(t, curve[i:i + 4])
    cap = max(curve[i + 1], curve[i + 2]) + f32(3.01)
    return min(v, cap)


class Curves:
    def __init__(self, width):
        self.width = width
        self.A = np.zeros(width + 3, f32)
        self.B = np.zeros(width + 3, f32)

    def update(self, head):
        width, tail = self.width, 2 * self.width
        a = native_exp(head[:width] * f32(-2.0))
        self.A[1:width + 1] = (f32(1.0) / (a + f32(1.0))) * f32(160.0) - f32(140.0)
        suppress_peaks(self.A)
        pad(self.A)
        b = native_exp(head[width:tail] * f32(-2.0))
        self.B[1:width + 1] = (f32(1.0) / (b + f32(1.0))) * f32(240.0) - f32(120.0)
        pad(self.B)
        p = (f32(1.0) / (native_exp(-head[tail:tail + 4]) + f32(1.0))) * f32(6.2831855)
        self.phase = np.array([p[0], p[0], p[1], p[2], p[3], p[3], p[3]], f32)
        d = f32(3.1415927) / np.maximum(np.exp(head[tail + 4:tail + 8].astype(f32)) * f32(400.0), f32(1e-6))
        self.decay = np.array([d[0], d[0], d[1], d[2], d[3], d[3], d[3]], f32)


def pad(curve):
    width = len(curve) - 3
    curve[0] = curve[1]
    curve[width + 1] = curve[width + 2] = curve[width]


def suppress_peaks(A):
    lo, hi = warp(1500.0), warp(4000.0)
    span = WARP_HIGH - WARP_LOW
    width = len(A) - 3
    last = width - 1
    one = f32(1.0)
    for floor in (f32(-60.0), f32(-45.0)):
        src = A.copy()
        mask = np.zeros(width + 3, f32)
        for i in range(1, last):
            pos = f32(i) * span / f32(last) + WARP_LOW
            if floor == -60.0:
                c, left, right = src[i + 1], src[i], src[i + 2]
            else:
                c, left, right = src[i + 1], src[i - 1], src[i + 3]
            if not (lo < pos and floor < c and left + f32(3.0) < c and right + f32(3.0) < c):
                continue
            shoulder = max(left, right) + f32(3.0)
            gain = min(one, (c - shoulder) / f32(3.0)) * min(one, (c - floor) / f32(12.0))
            gain = min(one, (pos - lo) / (hi - lo) * f32(4.0) * gain)
            if floor == -60.0:
                A[i + 1] -= (one - mask[i]) * (c - shoulder) * gain
                mask[i + 1] = one
            else:
                for j in (i, i + 1, i + 2):
                    A[j] -= (one - mask[j]) * max(f32(0.0), (src[j] - shoulder) * gain)
                mask[i:i + 3] = one


def harmonic_amplitudes(A, f0, count, scale_by_pitch, alpha_formant=f32(0.0)):
    amps = np.zeros(count, f32)
    scale = f32(math.sqrt(f0 / FS)) if scale_by_pitch else f32(1.0)
    peak = f32(-300.0)
    for k in range(1, count):
        fk = allpass_hz(f32(k) * f0, -alpha_formant)
        if FS * f32(0.5) < fk:
            break
        v = f32(10.0) ** (curve_value(A, fk) * f32(0.05)) * scale
        if fk < 4000.0:
            peak = max(peak, v)
        amps[k - 1] = v if fk <= 10000.0 else min(v, peak)
    return amps


def spectra(A, B, alpha_formant=f32(0.0)):
    bins = FRAME // 2 + 1
    harm = np.empty(bins, f32)
    noise = np.empty(bins, f32)
    rel = np.empty(bins, f32)
    for j in range(bins):
        f = allpass_hz(f32(j) * FS / f32(FRAME), -alpha_formant)
        a = curve_value(A, f)
        b = min(curve_value(B, f), f32(140.0))
        harm[j] = a
        rel[j] = b
        noise[j] = a + b
    return harm, noise, rel


def band_analysis(spectrum, edges, fft, add_floor, floor=f32(1e-7)):
    out = np.empty(len(edges) - 2, f32)
    for b in range(len(edges) - 2):
        e0, e1, e2 = edges[b], edges[b + 1], edges[b + 2]
        x0, x1, x2 = f32(fft) * e0 / FS, f32(fft) * e1 / FS, f32(fft) * e2 / FS
        norm = f32(2.0) / (e2 - e0)
        s = f32(0.0)
        for k in range(int(math.ceil(x0)), int(x2) + 1):
            w = min((f32(k) - x0) / (x1 - x0), f32(1.0) - (f32(k) - x1) / (x2 - x1))
            s = s + w * norm * spectrum[k]
        v = floor + s if add_floor else max(floor, s)
        out[b] = f32(math.log10(v)) * f32(20.0)
    return out


def rfft_native(x):
    return np.fft.rfft(x.astype(np.float64))


def irfft_native(X, n):
    return np.fft.irfft(X, n) * n


class Decoder:
    def __init__(self, edges16, width, scale_by_pitch, hann):
        self.curves = Curves(width)
        self.scale_by_pitch = scale_by_pitch
        self.edges16 = edges16
        self.hann512 = np.zeros(512, f32)
        k = np.arange(320)
        h = (0.5 - 0.5 * np.cos(2 * np.pi * k / 320)).astype(f32) if hann else np.ones(320, f32)
        self.hann512[96:416] = h * f32(2.0) / h.sum()
        self.dither_rng = None
        self.first = True
        self.events = dict(noise_guard=0, attenuated=0, zeroed=0)

    def noise_frame(self, noise_db, noise_rng):
        bins = FRAME // 2 + 1
        freqs = np.arange(bins, dtype=f32) * FS / f32(FRAME)
        guard_ref = max(max(self.harm_db[j], noise_db[j]) for j in range(int(7000.0 / FS * FRAME)))
        high = noise_db[int(15000.0 / FS * FRAME):]
        if np.any((high > -32.0) | ((high > -44.0) & (high > guard_ref + 12.0))):
            noise_db = np.full(bins, f32(-140.0))
            self.events['noise_guard'] += 1
        white = noise_rng.draw(FRAME).astype(f32)
        X = rfft_native(np.roll(white, -FRAME // 2))
        band = (freqs >= 4000.0) & (freqs < 10000.0)
        mean = np.mean(f32(10.0) ** (noise_db[band] * f32(0.1)))
        cap = np.where(freqs <= 12000.0, f32(0.0), f32(math.log10(max(mean, 1.1754944e-38))) * f32(10.0) + f32(12.0))
        X = X * (f32(10.0) ** (np.minimum(noise_db, cap) * f32(0.05)))
        X[0] = X[1] = 0
        y = irfft_native(X, FRAME)
        return (np.roll(y, FRAME // 2) / FRAME).astype(f32)

    def harmonic_frame(self, amps, f0, ring, start):
        count = len(amps)
        top = min(count - 1, int(FS * f32(0.5) / f0))
        rows = ring.rows(start, FRAME)[:, :top]
        return (rows.astype(np.float64) @ amps[:top].astype(np.float64)).astype(f32)

    def envelope(self, f0):
        A = self.curves.A
        full = np.array([min(curve_value(A, f32(u) * FS / f32(FRAME)),
                             max(A[self._bin(u) + 1], A[self._bin(u) + 2]) + f32(3.0)) for u in range(FRAME)], f32)
        at_f0 = np.interp(FRAME * f0 / FS, np.arange(257), self.harm_db)
        width = int(round(256000.0 / float(FS)))
        t = np.arange(-width, width + 1)
        w = (0.54 + 0.46 * np.cos(2 * np.pi * t / (2 * width))).astype(f32)
        env = np.empty(257, f32)
        for u in range(257):
            idx = np.abs(u + t)
            vals = np.where(idx * FS / FRAME < f0, at_f0, full[np.minimum(idx, FRAME - 1)])
            env[u] = np.sum(vals * w) / np.sum(w)
        return env

    def _bin(self, u):
        f = f32(u) * FS / f32(FRAME)
        x = (min(max(warp(f), WARP_LOW), WARP_HIGH) - WARP_LOW) * f32(self.curves.width - 1) / (WARP_HIGH - WARP_LOW)
        return int(x)

    def phase(self, env, f0):
        phase, decay = self.curves.phase, self.curves.decay
        lim = min(max(int(f0 * 512.0 / FS), 1), 256)
        theta = np.empty(257, f32)
        for u in range(257):
            f = f32(u) * FS / f32(FRAME)
            x = (min(max(warp(f), WARP_LOW), WARP_HIGH) - WARP_LOW) * f32(3.0) / (WARP_HIGH - WARP_LOW)
            i = int(x)
            t = x - i
            ph = (phase[i + 2] - phase[i + 1]) * t + phase[i + 1]
            dc = (decay[i + 2] - decay[i + 1]) * t + decay[i + 1]
            if u < lim:
                slope = env[lim + 1] - env[lim - 1]
            elif u < 0xf6:
                slope = env[min(u + 1, 256)] - env[max(u - 1, 0)]
            else:
                slope = env[0xf6] - env[0xf4]
            theta[u] = slope * dc * FS / f32(512.0) + ph
        return theta

    def __call__(self, head, cents, count, start, ring, noise_rng, gain=f32(1.0), noise_gain=f32(1.0)):
        f0 = f32(440.0) * f32(2.0) ** (f32(cents) / f32(1200.0))
        self.curves.update(head)
        A, B = self.curves.A, self.curves.B
        amps = harmonic_amplitudes(A, f0, count, self.scale_by_pitch)
        self.harm_db, noise_db, _ = spectra(A, B)
        noise = self.noise_frame(noise_db, noise_rng)
        harm = self.harmonic_frame(amps, f0, ring, start - FRAME // 2)
        harm, noise = harm * gain, noise * gain
        power = float(np.mean((harm + noise).astype(np.float64) ** 2))
        p = 10 * math.log10(max(power, 1.1754944e-38))
        if p > -6.02:
            harm[:], noise[:] = 0, 0
            self.events['zeroed'] += 1
        elif p > -12.04:
            a = f32(10.0) ** f32(-(p + 12.04) * 0.75 * 0.05)
            harm, noise = harm * a, noise * a
            self.events['attenuated'] += 1
        k = np.arange(FRAME)
        hann = (0.5 - 0.5 * np.cos(2 * np.pi * k / FRAME)).astype(f32)
        self.rms = f32(math.sqrt(np.sum((harm * hann) ** 2) / np.sum(hann ** 2)))
        env = self.envelope(f0)
        theta = self.phase(env, f0)
        X = rfft_native(np.roll(harm, -FRAME // 2)) * np.exp(1j * theta.astype(np.float64))
        shaped = (np.roll(irfft_native(X, FRAME), FRAME // 2) / FRAME).astype(f32)
        noise = noise * noise_gain
        dither = noise_rng.draw(FRAME).astype(f32)
        level = f32(1e-4) if self.first else f32(5e-5)
        self.first = False
        x = (noise + shaped + level * dither) * self.hann512
        mag = np.sqrt(np.abs(rfft_native(x)) ** 2 + 1e-14).astype(f32)
        fb = band_analysis(mag, self.edges16, FRAME, add_floor=True)
        return shaped, noise, fb
