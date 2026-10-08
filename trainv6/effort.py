from typing import NamedTuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import frontend

FFT = 1024
FRAME_SECONDS = frontend.HOP / frontend.SAMPLE_RATE
LOW_BAND_HZ = (60.0, 1000.0)
MID_BAND_HZ = (1000.0, 4000.0)
SMOOTHING_FRAMES = 15
MIN_PHONEME_FRAMES = 200
LEVEL_DB_PER_CONTROL = 5.0
TILT_DB_PER_CONTROL = 5.7
EXPRESSION_SPREAD = 0.4
WINDOW = np.hanning(FFT)
FREQUENCIES = np.fft.rfftfreq(FFT, 1 / frontend.SAMPLE_RATE)


class Measure(NamedTuple):
    level: np.ndarray
    tilt: np.ndarray
    voiced: np.ndarray
    phonemes: np.ndarray


def frame_phonemes(labels, frames):
    names = np.full(frames, '', dtype=object)
    for start, end, name in labels:
        names[round(start / FRAME_SECONDS):round(end / FRAME_SECONDS)] = name
    return names


def measure(audio, voiced, labels):
    frames = len(voiced)
    padded = np.pad(audio, (FFT // 2, FFT))
    blocks = np.lib.stride_tricks.sliding_window_view(padded, FFT)[::frontend.HOP][:frames]
    power = np.abs(np.fft.rfft(blocks * WINDOW, axis=1)) ** 2

    def band_db(low, high):
        return 10 * np.log10(power[:, (FREQUENCIES >= low) & (FREQUENCIES < high)].sum(1) + 1e-10)

    return Measure((10 * np.log10(power.sum(1) + 1e-10)).astype(np.float32),
                   (band_db(*MID_BAND_HZ) - band_db(*LOW_BAND_HZ)).astype(np.float32), voiced, frame_phonemes(labels, frames))


def phoneme_residual(values, voiced, names):
    residual = np.zeros_like(values)
    pooled = np.median(values[voiced])
    for name in np.unique(names[voiced]):
        mask = voiced & (names == name)
        residual[mask] = values[mask] - (np.median(values[mask]) if mask.sum() >= MIN_PHONEME_FRAMES else pooled)
    return residual


def smooth_through_unvoiced(values, voiced):
    if not voiced.any():
        return np.zeros_like(values)
    frames = np.arange(len(values))
    return uniform_filter1d(np.interp(frames, frames[voiced], values[voiced]), SMOOTHING_FRAMES, mode='nearest')


def inference_control(x, native):
    return x - native * x ** 2 + native


def mean_centring(x, native):
    lo, hi = -1.0, 1.0
    for _ in range(40):
        shift = (lo + hi) / 2
        if inference_control(np.clip(x + shift, -1, 1), native).mean() < native:
            lo = shift
        else:
            hi = shift
    return (lo + hi) / 2


def frame_controls(measures, native):
    names = list(measures)
    voiced = np.concatenate([measures[n].voiced for n in names])
    phonemes = np.concatenate([measures[n].phonemes for n in names])
    effort = (phoneme_residual(np.concatenate([measures[n].level for n in names]), voiced, phonemes) / LEVEL_DB_PER_CONTROL
              + phoneme_residual(np.concatenate([measures[n].tilt for n in names]), voiced, phonemes) / TILT_DB_PER_CONTROL) / 2
    bounds = np.cumsum([len(measures[n].voiced) for n in names])[:-1]
    smooth = np.concatenate([smooth_through_unvoiced(e, v) for e, v in zip(np.split(effort, bounds), np.split(voiced, bounds))])
    x = EXPRESSION_SPREAD * (smooth - smooth[voiced].mean()) / smooth[voiced].std()
    x = np.clip(x + mean_centring(x[voiced], native), -1, 1)
    return dict(zip(names, np.split(inference_control(x, native).astype(np.float32), bounds)))
