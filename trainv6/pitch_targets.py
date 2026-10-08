import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import frontend

FRAME = float(frontend.PERIOD)
VIBRATO_BAND = (3.5, 9.0)
FILTER_ORDER = 4
WINDOW_START = 0.10
WINDOW_END = 0.02
RAMP = 0.04
MIN_WINDOW = 0.25
DETECT_START = 0.12
DETECT_END = 0.03
DETECT_MIN = 0.35
DETECT_DEPTH = 12.0
DETECT_PROMINENCE = 3.0


BAND_FILTER = butter(FILTER_ORDER, VIBRATO_BAND, 'bandpass', fs=1.0 / FRAME, output='sos')


def sung_notes(notes):
    return [n for n in notes if n['pitch'] is not None]


def note_window(note, frames):
    a, b = int(round((note['start'] + WINDOW_START) / FRAME)), min(int(round((note['stop'] - WINDOW_END) / FRAME)), frames)
    return (a, b) if (b - a) * FRAME >= MIN_WINDOW else None


def ramp_weights(length):
    ramp = max(int(round(RAMP / FRAME)), 1)
    rise = 0.5 - 0.5 * np.cos(np.pi * (np.arange(ramp) + 0.5) / ramp)
    weights = np.ones(length)
    weights[:ramp] = rise
    weights[-ramp:] = np.minimum(weights[-ramp:], rise[::-1])
    return weights


def has_vibrato(f0, start, stop):
    a, b = int(round((start + DETECT_START) / FRAME)), int(round((stop - DETECT_END) / FRAME))
    if (b - a) * FRAME < DETECT_MIN:
        return False
    window = f0[a:b]
    x = np.arange(len(window))
    deviation = window - np.polyval(np.polyfit(x, window, 2), x)
    taper = np.hanning(len(deviation))
    spectrum = 2 * np.abs(np.fft.rfft(deviation * taper, 4096)) / taper.sum()
    freqs = np.fft.rfftfreq(4096, FRAME)
    peak = spectrum[(freqs >= VIBRATO_BAND[0]) & (freqs <= VIBRATO_BAND[1])].max()
    context = np.median(spectrum[(freqs >= 1.0) & (freqs <= 15.0)])
    return bool(peak >= DETECT_DEPTH and peak >= DETECT_PROMINENCE * context)


def split_vibrato(f0, notes):
    layer, active = np.zeros(len(f0)), np.zeros(len(f0), bool)
    detected = {id(n) for n in sung_notes(notes) if has_vibrato(f0, n['start'], n['stop'])}
    for n in sung_notes(notes):
        window = note_window(n, len(f0))
        if window is None:
            continue
        a, b = window
        x = np.arange(b - a)
        deviation = f0[a:b] - np.polyval(np.polyfit(x, f0[a:b], 2), x)
        layer[a:b] = ramp_weights(b - a) * sosfiltfilt(BAND_FILTER, deviation)
        active[a:b] = id(n) in detected
    return f0 - layer, np.where(active, layer, 0.0), active
