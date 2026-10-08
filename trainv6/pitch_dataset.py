import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from . import frontend, pitch_frontend, pitch_score, pitch_targets
from .container import Container

FIELDS = dict(condition=(np.float16, pitch_frontend.CONDITION_WIDTH), head=(np.float32, len(pitch_frontend.POST_COLUMNS)), f0=(np.float32, 1),
              base=(np.float32, 1), layer=(np.float32, 1), voiced=(np.uint8, 1), vibrato=(np.uint8, 1))
OUTLIER_CENTS = 900.0
MIN_PHONEME = 0.01
FRAME = float(frontend.PERIOD)


def f0_path(directory, name):
    return Path(directory) / 'f0' / f'{name}.npz'


def cache_f0(recording, directory):
    cents, voiced = frontend.extract_f0(frontend.load_audio(recording['wav']))
    np.savez_compressed(f0_path(directory, recording['id']), cents=cents, voiced=voiced)


def cache_all_f0(recordings, directory, workers):
    (Path(directory) / 'f0').mkdir(parents=True, exist_ok=True)
    missing = [(r, directory) for r in recordings if not f0_path(directory, r['id']).exists()]
    if missing:
        with multiprocessing.get_context('spawn').Pool(workers) as pool:
            pool.starmap(cache_f0, missing)


def jittered(timed, rng, sigma):
    bounds = np.array([p.start for p in timed] + [timed[-1].end])
    moved = bounds.copy()
    for i in range(1, len(timed)):
        moved[i] = np.clip(bounds[i] + np.clip(rng.normal(0, sigma), -2.5 * sigma, 2.5 * sigma), moved[i - 1] + MIN_PHONEME, bounds[i + 1] - MIN_PHONEME)
    return [p._replace(start=moved[i], end=moved[i + 1]) for i, p in enumerate(timed)]


def section_targets(song, origin, frames):
    start = int(round(origin / FRAME))
    return [np.pad(values[start:start + frames], (0, max(start + frames - len(values), 0)), mode='edge') for values in song]


def write_chunk(files, arrays):
    for name, array in arrays.items():
        dtype = FIELDS[name][0]
        files[name].write(np.ascontiguousarray(array, dtype).tobytes())


def build(base_vpit, manifest, out_dir, scores, kana_table, variants=3, jitter=0.02, seed=0, workers=8, device='cuda'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recordings = json.loads(Path(manifest).read_text())['recordings']
    table = pitch_score.kana_table(kana_table)
    cache_all_f0(recordings, out, workers)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    conditioner = pitch_frontend.Conditioner(Container.load(base_vpit).document, device)
    rng = np.random.default_rng(seed)
    sections, skipped, total = [], {}, 0
    files = {name: (out / f'{name}.bin').open('wb') for name in FIELDS}
    for recording in recordings:
        name = recording['id']
        try:
            timed, notes = pitch_score.timed_score(name, scores, table)
        except ValueError as error:
            skipped[name] = str(error)
            continue
        series = np.load(f0_path(out, name))
        f0 = series['cents'].astype(float)
        base, layer, active = pitch_targets.split_vibrato(f0, notes)
        song = (f0, base, layer, series['voiced'], active)
        for variant in range(variants if recording['split'] == 'train' else 1):
            for origin, section in pitch_score.sections(timed if variant == 0 else jittered(timed, rng, jitter)):
                head, condition = conditioner.rows(section)
                frames = len(head)
                targets = section_targets(song, origin, frames)
                write_chunk(files, dict(condition=condition, head=head[:, pitch_frontend.POST_COLUMNS], **dict(zip(('f0', 'base', 'layer', 'voiced', 'vibrato'), targets))))
                sections.append(dict(song=name, split=recording['split'], variant=variant, origin=origin, start=total, frames=frames))
                total += frames
    for file in files.values():
        file.close()
    (out / 'index.json').write_text(json.dumps(dict(sections=sections, frames=total, skipped=skipped, variants=variants, jitter=jitter,
                                                    base_vpit=str(base_vpit))))
    return total


class PitchData:
    def __init__(self, directory, device, splits=('train', 'validation')):
        self.directory = Path(directory)
        index = json.loads((self.directory / 'index.json').read_text())
        self.sections = [s for s in index['sections'] if s['split'] in splits]
        keep = np.concatenate([np.arange(s['start'], s['start'] + s['frames']) for s in self.sections])
        self.arrays = {}
        for name, (dtype, width) in FIELDS.items():
            mapped = np.memmap(self.directory / f'{name}.bin', dtype, 'r').reshape(index['frames'], width)
            self.arrays[name] = torch.from_numpy(np.ascontiguousarray(mapped[keep])).to(device)
        frames = np.array([s['frames'] for s in self.sections])
        self.frames = torch.from_numpy(frames).to(device)
        self.offsets = torch.from_numpy(np.concatenate([[0], np.cumsum(frames)[:-1]])).to(device)
        self.device = device
        self.weights = self.frames.float()
        note = self.arrays['head'][:, 0]
        distance = (self.arrays['f0'][:, 0] - note).abs()
        self.arrays['valid'] = ((self.arrays['voiced'][:, 0] > 0) & (distance < OUTLIER_CENTS)).float()[:, None]
        self.arrays['f0'] = note[:, None] + (self.arrays['f0'] - note[:, None]).clamp(-OUTLIER_CENTS, OUTLIER_CENTS)
