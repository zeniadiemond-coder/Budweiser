import math

import numpy as np
import torch

from . import frontend, nets
from .container import read_tensor

f32 = np.float32
PERIOD = float(frontend.PERIOD)

ENCODER = '6dbe9f75292d82b3cc6c57bd65089b20'
NOTE = '1dce9c8b2329f1cb9c73fd879bdeb54d'
NOTE_CONFIG = '5f6f2b54407c50ebf5343fcddbdb7c10'
NOTE_STACK = '9680b2b3ca5c84fcfbf088a3ead0ae22'
NOTE_INPUTS, NOTE_HIDDEN, NOTE_BOTTLENECK = ('3fecf2d78547692b4157889e04a498f0', 'b8efaa0567206ae04ab50308481882b2',
                                             'c1f0216d696655c00da1ce68cbc92033')
NOTE_W1, NOTE_B1, NOTE_W2, NOTE_B2 = ('220316df91912792cb42bc663a8df636', 'bf1fb08e8ffbab205dbb90120465e856',
                                      '7352c2e9486a73ce344f167dfad31876', 'c968ce1719cfab911f40b224b5bb28df')
BINS, LOW, HIGH = '903afc65900371c7521ac167421196a6', '205192c4ec6717232a8e5e2515841a42', 'f98f2b4f4eab060dad241d3e4fb1c4dc'

CONDITION_WIDTH = 64
POST_COLUMNS = (0, 10, 11, 12, 15, 23)

CLASS_NAMES = [['sil'], ['br'], ['a', 'i', 'u', 'e', 'o'], ['n', 'N', 'm', 'ny', 'my'], ['g', 'd', 'b', 'gy', 'dy', 'by'], ['z'], ['j'],
               ['y', 'w'], ['r', 'ry'], ['k', 't', 'p', 'ky', 'ty', 'py'], ['s', 'sh', 'h', 'f', 'hy'], ['ch', 'ts'], ['cl']]
VOICED_CLASSES = (2, 3, 7, 8)
ABSENT_CLASS = 13
CLASS_OF = {name: i for i, names in enumerate(CLASS_NAMES) for name in names}


def phoneme_class(name):
    return CLASS_OF.get(name, ABSENT_CLASS)


def breath_to_silence(phonemes):
    result = []
    for p in phonemes:
        if p.name != 'br':
            result.append(p)
        elif result and result[-1].name == 'sil':
            result[-1] = result[-1]._replace(end=p.end)
        else:
            result.append(p._replace(name='sil'))
    return result


def phoneme_onset(phonemes, i):
    p = phonemes[i]
    if i > 0 and not phonemes[i - 1].pitch < p.pitch + 50.0 and phoneme_class(p.name) in VOICED_CLASSES \
            and 2 <= phoneme_class(phonemes[i - 1].name) <= 8:
        return 1.0
    return float(p.nucleus and p.name != 'sil')


def frame_owners(phonemes, frames):
    ends = np.array([p.end for p in phonemes])
    stop = np.maximum.accumulate(np.minimum(np.ceil(ends / PERIOD).astype(np.int64), frames))
    first = np.concatenate(([0], stop[:-1]))
    owner = np.searchsorted(stop, np.arange(frames), side='right')
    return owner, first, stop


def phoneme_records(table, phonemes):
    records, category = np.zeros((len(phonemes), 11), f32), 0
    for i, p in enumerate(phonemes):
        if p.nucleus:
            category = 1
        elif p.note_start:
            category = 0
        elif category in (1, 2):
            category = 2
        offset = f32(p.start - float(f32(int(p.start / PERIOD)) * frontend.PERIOD))
        records[i] = [frontend.phoneme_id(table, p.name), p.note_start, p.nucleus, p.name in frontend.NUCLEI, False, f32(p.end - p.start),
                      offset, f32(p.pitch), category, 0.0, 0.0]
    return records


def frame_records(table, phonemes):
    records = phoneme_records(table, phonemes)
    frames = math.ceil(phonemes[-1].end / PERIOD)
    absent = np.zeros(11, f32)
    absent[5] = 1.0
    owner, first, stop = frame_owners(phonemes, frames)
    index = np.minimum(owner, len(phonemes) - 1)
    out = np.zeros((frames, 3, 11), f32)
    out[:, 0] = records[index]
    out[:, 1] = np.where((owner < len(phonemes) - 1)[:, None], records[np.minimum(owner + 1, len(phonemes) - 1)], absent)
    out[:, 2] = np.where((owner > 0)[:, None], records[np.maximum(owner - 1, 0)], absent)
    out[owner >= len(phonemes), 2] = 0.0
    k = np.arange(frames) - np.where(owner < len(phonemes), first[index], stop[-1])
    out[:, 0, 6] = k.astype(f32) * frontend.PERIOD + records[index, 6]
    out[:, 1:, 6] = 0.0
    for i in np.nonzero(records[:, 8] == 1)[0]:
        total = f32(phonemes[i].end - phonemes[i].start)
        for j in range(i + 1, len(phonemes)):
            if records[j, 8] < 2:
                break
            total += f32(phonemes[j].end - phonemes[j].start)
        start = f32(phonemes[i].start)
        if total < 1e-5:
            continue
        a, b = min(max(int(start / frontend.PERIOD), 0), frames), min(max(int((total + start) / frontend.PERIOD + f32(1.0)), 0), frames)
        out[a:b, 0, 10] = np.arange(a, b, dtype=f32) * frontend.PERIOD - start
        out[a:b, 0, 9] = total
    return out


def head_columns(table, phonemes):
    merged = breath_to_silence(phonemes)
    frames = int(merged[-1].end / PERIOD)
    records = frame_records(table, merged)[:frames]
    owner, _, _ = frame_owners(merged, math.ceil(merged[-1].end / PERIOD))
    owner = owner[:frames]
    pitch = np.array([p.pitch for p in merged], f32)
    onset = np.array([phoneme_onset(merged, i) for i in range(len(merged))], f32)
    inside = owner < len(merged)
    c0, c1, c2 = records[:, 0], records[:, 1], records[:, 2]
    return np.concatenate([np.where(inside, pitch[np.minimum(owner, len(merged) - 1)], pitch[-1])[:, None],
                           np.where(inside, onset[np.minimum(owner, len(merged) - 1)], 0.0)[:, None].astype(f32),
                           c0[:, :9], c0[:, 10:11], c0[:, 9:10], c1[:, :8], c2[:, :8]], 1)


def encoder_phonemes(phonemes):
    kept = breath_to_silence(phonemes)
    ends = [b.start for b in kept[1:]] + [phonemes[-1].end]
    return [frontend.Phoneme(p.start, end, p.name, p.cls) for p, end in zip(kept, ends)]


def gaussian_bins(pitch, bins, low, high):
    centres = np.arange(bins, dtype=f32) * f32(high - low) / f32(bins) + f32(low)
    return np.exp(-(centres[None] - pitch[:, None].astype(f32)) ** 2 / f32(20000.0)).astype(f32)


class Conditioner:
    def __init__(self, document, device):
        self.device = device
        self.encoder_document = document[ENCODER]
        self.table = self.encoder_document[frontend.PHONEMES]
        self.encoder = nets.GatedStack(self.encoder_document[nets.ENCODER], native_math=True).to(device).eval().requires_grad_(False)
        note = document[NOTE]
        config = note[NOTE_CONFIG]
        self.inputs, hidden, bottleneck = config[NOTE_INPUTS], config[NOTE_HIDDEN], config[NOTE_BOTTLENECK]
        read = lambda key, *shape: torch.from_numpy(read_tensor(note[key])).reshape(*shape).to(device)
        self.w1, self.b1 = read(NOTE_W1, hidden, self.inputs), read(NOTE_B1, hidden)
        self.w2, self.b2 = read(NOTE_W2, bottleneck, hidden), read(NOTE_B2, bottleneck)
        self.stack = nets.GatedStack(note[NOTE_STACK], native_math=True).to(device).eval().requires_grad_(False)
        top = document[frontend.FRONTEND]
        self.bins, self.low, self.high = top[BINS], top[LOW], top[HIGH]

    @torch.no_grad()
    def rows(self, phonemes):
        head = head_columns(self.table, phonemes)
        frames = len(head)
        x, _ = frontend.encoder_input(self.encoder_document, encoder_phonemes(phonemes))
        encoded = self.encoder.run_aligned(torch.from_numpy(x).to(self.device)[None])[0, :frames]
        h = torch.tanh(torch.tanh(encoded[:, :self.inputs] @ self.w1.T + self.b1) @ self.w2.T + self.b2)
        gauss = torch.from_numpy(gaussian_bins(head[:, 0], self.bins, self.low, self.high) * head[:, 1:2]).to(self.device)
        condition = self.stack.run_aligned(torch.cat([h, gauss], 1)[None])[0]
        return head, condition.cpu().numpy()
