import math
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

f32 = np.float32

SAMPLE_RATE = 32000
HOP = 160
PERIOD = f32(HOP / SAMPLE_RATE)
WARMUP = 77
LATENCY = 9
OUTPUT_DELAY = LATENCY * HOP
FEATURE_WIDTH = 40

PHONEMES = 'edc2ce94290a3b18bc7f745276643253'
FRONTEND = 'a4ba655a061e6151f80574e6088492a0'
MIN_DURATION = 'f962293ecd429e2bdcc6b736d5fbe03f'
MAX_DURATION = '52deffcc05ab8eef20105c2860706a51'
NEXT = '3abe629e45940435ab13470c11104d50'
PREV = '36cdeb254eea602bbf99240c5287b151'
CLASSES = 'd809560074085c5b6f43bf86bd3bebd4'

JAPANESE_CLASS = 0
ENGLISH_CLASS = 1
CHINESE_CLASS = 2
EDGE_CLASS = -1
LANGUAGE_CLASS = {'ja': JAPANESE_CLASS, 'en': ENGLISH_CLASS, 'zh': CHINESE_CLASS}
SILENCE_NAMES = {'SP', 'sil', 'pau', 'SIL', 'sp'}

Phoneme = namedtuple('Phoneme', 'start end name cls')

CORPUS_TO_MODEL = {
    'SP': 'sil', 'sil': 'sil', 'pau': 'sil', 'br': 'br', 'cl': 'cl', 'N': 'N',
    'a': 'a', 'i': 'i', 'u': 'u', 'e': 'e', 'o': 'o',
    'k': 'k', 'ky': 'ky', 'g': 'g', 'gy': 'gy', 's': 's', 'sh': 'sh', 't': 't', 'ty': 'ty', 'd': 'd', 'dy': 'dy',
    'ch': 'ch', 'ts': 'ts', 'n': 'n', 'ny': 'ny', 'z': 'z', 'j': 'j', 'h': 'h', 'hy': 'hy',
    'f': 'f', 'fy': 'f', 'p': 'p', 'py': 'py', 'b': 'b', 'by': 'by', 'm': 'm', 'my': 'my',
    'y': 'y', 'r': 'r', 'ry': 'ry', 'w': 'w',
    'vf': None,
}

ARPABET_TO_MODEL = {
    'aa': 'aa', 'ae': 'ae', 'ah': 'ah', 'ao': 'ao', 'aw': 'aw', 'ay': 'ay', 'b': 'b', 'ch': 'ch', 'd': 'd', 'dh': 'dh', 'eh': 'eh',
    'er': 'er', 'ey': 'ey', 'f': 'f', 'g': 'g', 'hh': 'hh', 'ih': 'ih', 'iy': 'iy', 'jh': 'jh', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n',
    'ng': 'ng', 'ow': 'ow', 'oy': 'oy', 'p': 'p', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't', 'th': 'th', 'uh': 'uh', 'uw': 'uw', 'v': 'v',
    'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'zh', 'br': 'br', 'cl': 'cl',
}

PLAIN_BEFORE_I = {'ky': 'k', 'ty': 't', 'ny': 'n', 'hy': 'h', 'my': 'm', 'ry': 'r', 'gy': 'g', 'dy': 'd', 'by': 'b', 'py': 'p'}

NUCLEI = {
    'ih', 'eh', 'ae', 'aa', 'ah', 'uh', 'ax', 'iy', 'uw', 'ao', 'er', 'ey', 'ay', 'oy', 'ow', 'aw', 'a', 'i', 'u',
    'e', 'o', 'wo', 'N', 'i_n', 'uei', 'iE_n', 'yE_r', '@N', 'u@N', '@U', 'ia', 'i`', '@`', 'i\\', 'i@U', 'aI', 'uaI',
    'y_n', '7', 'iN', 'iE_r', 'ua_n', 'iUN', 'AN', 'ei', 'AU', 'uAN', 'a_n', 'iAN', 'y{_n', 'u@_n', '@_n', 'uo', 'UN',
    'iAU', 'ua', 'ia+1', 'iE_r+1', 'yE_r+1', 'ua+1', 'uo+1', 'iAU+1', 'i@U+1', 'uaI+1', 'uei+1', 'iAN+1', 'iUN+1',
    'iE_n+1', 'y{_n+1', 'u@N+1', 'u@_n+1', 'ua_n+1', 'uAN+1', 'y|CH', 'a|CH', 'i|CH', 'u|CH', 'o|CH', 'SIL', 'Sil',
    'BR', 'Br'}


def read_lab(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f'{path}:{number}: expected start end phoneme')
        rows.append((int(fields[0]) * 1e-7, int(fields[1]) * 1e-7, fields[2]))
    return rows


def model_phoneme_name(name, language):
    if name in SILENCE_NAMES:
        return 'sil'
    if language == 'en':
        return ARPABET_TO_MODEL.get(name.rstrip('012').lower(), '?' + name)
    return name


def model_phonemes(labels, language='ja'):
    if language != 'ja':
        return foreign_phonemes(labels, language)
    unknown = sorted({name for _, _, name in labels} - CORPUS_TO_MODEL.keys())
    if unknown:
        raise ValueError(f'labels without a model phoneme: {unknown}')
    names = [CORPUS_TO_MODEL[name] for _, _, name in labels]
    merged = []
    pending_start = None
    for i, (start, end, _) in enumerate(labels):
        if names[i] is None:
            pending_start = start if pending_start is None else pending_start
            continue
        if pending_start is not None:
            start, pending_start = pending_start, None
        name = names[i]
        following = next((n for n in names[i + 1:] if n is not None), None)
        if following == 'i':
            name = PLAIN_BEFORE_I.get(name, name)
        merged.append([start, end, name])
    if pending_start is not None:
        merged[-1][1] = labels[-1][1]
    result = [Phoneme(start, end, name, JAPANESE_CLASS) for start, end, name in merged]
    result[0] = result[0]._replace(cls=EDGE_CLASS)
    result[-1] = result[-1]._replace(cls=EDGE_CLASS)
    return result


def foreign_phonemes(labels, language):
    names = [model_phoneme_name(name, language) for _, _, name in labels]
    unknown = sorted({name[1:] for name in names if name.startswith('?')})
    if unknown:
        raise ValueError(f'labels without a model phoneme: {unknown}')
    result = [Phoneme(start, end, name, LANGUAGE_CLASS[language]) for (start, end, _), name in zip(labels, names)]
    result[0] = result[0]._replace(cls=EDGE_CLASS)
    result[-1] = result[-1]._replace(cls=EDGE_CLASS)
    return result


def duration_feature(duration, lo, hi):
    a, b = np.log(f32(lo)), np.log(f32(hi))
    x = np.log(np.maximum(duration.astype(f32), f32(lo)))
    return np.clip(((x - a) + (x - a)) / (b - a) - f32(1.0), f32(-1.0), f32(1.0)).astype(f32)


def frame_layout(phonemes):
    period = float(PERIOD)
    ends = np.array([p.end for p in phonemes])
    starts = np.array([p.start for p in phonemes])
    total = int(math.ceil(ends[-1] / period))
    last = np.minimum(np.ceil(ends / period).astype(np.int64), total)
    stop = np.maximum.accumulate(last)
    first = np.concatenate(([0], stop[:-1]))
    frames = np.arange(total)
    index = np.minimum(np.searchsorted(stop, frames, side='right'), len(phonemes) - 1)
    k = (frames - first[index]).astype(f32)
    position = (k * PERIOD).astype(np.float64) + (first[index] * period - starts[index])
    return index, k, position.astype(f32), starts, (ends - starts), int(ends[-1] / period)


def phoneme_id(table, name):
    if name in table:
        return table[name]
    resolved = ''.join(c for c in name if c.isascii() and c.isalpha())
    if resolved in ('SIL', 'Sil'):
        return 0
    if resolved in ('BR', 'Br'):
        return 1
    return table[resolved]


def encoder_input(doc, phonemes):
    cfg, table = doc[FRONTEND], doc[PHONEMES]
    before, after, classes = cfg[PREV], cfg[NEXT], cfg[CLASSES]
    base = classes + 4
    width = len(table) + base
    n = len(phonemes)
    blocks = np.zeros((n, width), f32)
    durations = np.array([p.end - p.start for p in phonemes])
    blocks[:, 0] = duration_feature(durations, cfg[MIN_DURATION], cfg[MAX_DURATION])
    blocks[:, 2] = 1.0
    blocks[np.arange(n), base - classes + np.array([p.cls for p in phonemes])] = 1.0
    blocks[np.arange(n), base + np.array([phoneme_id(table, p.name) for p in phonemes])] = 1.0
    ids = np.arange(n)
    context = [ids] + [np.clip(ids - before + k, 0, n - 1) for k in range(before)] + [np.clip(ids + 1 + k, 0, n - 1) for k in range(after)]
    rows = np.concatenate([blocks[c] for c in context], axis=1)

    index, k, position, starts, spans, frames = frame_layout(phonemes)
    duration = spans.astype(f32)[index]
    ratio = position / duration
    extra = np.stack([ratio, f32(1.0) - ratio, f32(1.0) - np.tanh(position * f32(10.0)),
                      np.tanh((position - duration) * f32(10.0)) + f32(1.0)], axis=1).astype(f32)
    x = np.concatenate([rows[index], extra], axis=1)[:frames]

    ids_by_phoneme = np.array([phoneme_id(table, p.name) for p in phonemes], f32)
    offsets = (starts - np.floor(starts / float(PERIOD)).astype(f32) * PERIOD).astype(f32)
    nuclear = np.array([p.name in NUCLEI for p in phonemes], f32)
    zeros = np.zeros(len(index), f32)
    header = np.stack([ids_by_phoneme[index], zeros, zeros, nuclear[index], zeros, duration, k * PERIOD + offsets[index]], axis=1)
    return x, header[:frames]


def cents_to_hz(cents):
    return 440.0 * 2.0 ** (cents / 1200.0)


def pitch_norm(cents):
    return ((np.clip(cents, -4800.0, 3600.0) + 4800.0) * 2.0 / 7800.0 - 1.0).astype(f32)


def controls_input(features, f0_cents, control):
    n = len(features)
    x = np.zeros((WARMUP + n, 36), f32)
    x[:WARMUP, :32] = features[0, :32]
    body = x[WARMUP:]
    lagged = np.maximum(np.arange(n) - 3, 0)
    body[2:, :32] = features[lagged[2:], :32]
    body[:, 32] = pitch_norm(f0_cents[lagged])
    body[:, 33] = 1.0
    body[1:, 35] = np.clip(control[:-1], -1.5, 1.5)
    return x


def pad_latency(values):
    return values[np.minimum(np.arange(len(values) + LATENCY), len(values) - 1)]


def load_audio(path):
    import soundfile
    audio, rate = soundfile.read(path, dtype='float64', always_2d=True)
    audio = audio.mean(axis=1)
    divisor = math.gcd(SAMPLE_RATE, rate)
    return resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)


def extract_f0(audio, floor=60.0, ceiling=1100.0):
    import pyworld
    period_ms = 1000.0 * HOP / SAMPLE_RATE
    raw, times = pyworld.dio(audio, SAMPLE_RATE, f0_floor=floor, f0_ceil=ceiling, frame_period=period_ms)
    raw = pyworld.stonemask(audio, raw, times, SAMPLE_RATE)
    voiced = raw > 0
    cents = np.zeros(len(raw))
    if voiced.any():
        positions = np.arange(len(raw))
        cents = np.interp(positions, positions[voiced], 1200.0 * np.log2(raw[voiced] / 440.0))
    return cents.astype(f32), voiced
