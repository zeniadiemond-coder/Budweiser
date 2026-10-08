import functools
import hashlib
import json
import multiprocessing
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import ddsp, effort, frontend, nets, phase, targets
from .container import Container, file_digest
from .export import document_stacks
from .reference.frontend import scales_by_pitch

SILENCES = frontend.SILENCE_NAMES
CUT_SILENCE = 0.3
SILENCE_MARGIN = 0.25
MIN_SPLIT_SILENCE = 0.05
MAX_SECTION = 10.0
MIN_SECTION = 0.5
FEEDBACK_ALPHA = 0.0
HEAD_DELAY = targets.HEAD_DELAY
WARMUP = frontend.WARMUP
CONTROLS_LOOKBACK = 16
ENCODER_CONTEXT = 31
DB_LOW, DB_HIGH = -140.0, 20.0
FRAME_SECONDS = frontend.HOP / frontend.SAMPLE_RATE


class Recording(NamedTuple):
    id: str
    wav: str
    labels: list
    split: str


class PreparedSection(NamedTuple):
    name: str
    recording: str
    split: str
    offset: float
    audio: np.ndarray
    cents: np.ndarray
    voiced: np.ndarray
    labels: list


def quantise(db):
    return np.clip(np.round((db - DB_LOW) * (255 / (DB_HIGH - DB_LOW))), 0, 255).astype(np.uint8)


def dequantise(q):
    return q.astype(np.float32) * ((DB_HIGH - DB_LOW) / 255) + DB_LOW


def recording_split(ids, validation):
    ranked = sorted(ids, key=lambda i: hashlib.md5(i.encode()).hexdigest())
    return {i: 'validation' if i in ranked[:validation] else 'train' for i in ids}


def corpus_recordings(corpus, validation=0):
    corpus = Path(corpus)
    if corpus.suffix == '.json':
        manifest = json.loads(corpus.read_text())
        return [Recording(r['id'], r['wav'], [(l['start_100ns'] * 1e-7, l['end_100ns'] * 1e-7, l['phoneme']) for l in r['labels']], r['split'])
                for r in manifest['recordings']]
    wavs = sorted(corpus.glob('*.wav')) or sorted((corpus / 'wav').glob('*.wav'))
    lab_dir = corpus if (corpus / wavs[0].with_suffix('.lab').name).exists() else corpus / 'lab'
    splits = recording_split([w.stem for w in wavs], validation)
    return [Recording(w.stem, str(w), frontend.read_lab(lab_dir / f'{w.stem}.lab'), splits[w.stem]) for w in wavs]


def merged_silences(labels, frames, names=SILENCES):
    spans = []
    for start, end, name in labels:
        if name not in names:
            continue
        s, e = round(start / FRAME_SECONDS), min(round(end / FRAME_SECONDS), frames)
        if spans and s <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
        elif e > s:
            spans.append([s, e])
    return spans


def split_long(a, b, silences):
    if (b - a) * FRAME_SECONDS <= MAX_SECTION:
        return [(a, b)]
    inner = [(s, e) for s, e in silences if a < s and e < b and (e - s) * FRAME_SECONDS >= MIN_SPLIT_SILENCE]
    if not inner:
        return [(a, b)]
    central = [(s, e) for s, e in inner if a + 0.25 * (b - a) <= (s + e) / 2 <= a + 0.75 * (b - a)]
    s, e = max(central or inner, key=lambda span: span[1] - span[0])
    cut = (s + e) // 2
    return split_long(a, cut, silences) + split_long(cut, b, silences)


def cut_sections(labels, duration):
    frames = int(duration / FRAME_SECONDS)
    silences = merged_silences(labels, frames)
    margin = round(SILENCE_MARGIN / FRAME_SECONDS)
    bounds, begin = [], 0
    for s, e in silences:
        if (e - s) * FRAME_SECONDS >= CUT_SILENCE:
            keep = min(margin, (e - s) // 2)
            bounds.append((begin, s + keep))
            begin = e - keep
    bounds.append((begin, frames))
    breaks = merged_silences(labels, frames, SILENCES | {'cl'})
    sections = [part for a, b in bounds for part in split_long(a, b, breaks)]
    voiced_content = [(s, e) for s, e, n in labels if n not in SILENCES]
    return [(a, b) for a, b in sections
            if (b - a) * FRAME_SECONDS >= MIN_SECTION and any(e > a * FRAME_SECONDS and s < b * FRAME_SECONDS for s, e in voiced_content)]


def section_labels(labels, a, b):
    t0, t1 = a * FRAME_SECONDS, b * FRAME_SECONDS
    return [(max(s, t0) - t0, min(e, t1) - t0, n) for s, e, n in labels if min(e, t1) - max(s, t0) > 1e-9]


def prepare_recording(recording):
    audio = frontend.load_audio(recording.wav)
    cents, voiced = frontend.extract_f0(audio)
    result = []
    for i, (a, b) in enumerate(cut_sections(recording.labels, len(audio) / frontend.SAMPLE_RATE)):
        samples = slice(a * frontend.HOP, b * frontend.HOP)
        result.append(PreparedSection(f'{recording.id}_{i:03d}', recording.id, recording.split, a * FRAME_SECONDS,
                                      audio[samples].astype(np.float32), cents[a:b], voiced[a:b], section_labels(recording.labels, a, b)))
    return result


class Models:
    def __init__(self, base_model, speaker, control, device='cuda', language='ja'):
        self.device = device
        self.language = language
        self.speaker = speaker
        self.control = control
        doc = Container.load(base_model).document
        self.doc = doc
        stacks = document_stacks(doc)
        self.encoder, self.controls, self.synthesis = stacks[nets.ENCODER], stacks[nets.CONTROLS], stacks[nets.SYNTHESIS]
        for net in (self.encoder, self.controls, self.synthesis):
            net.to(device).eval().requires_grad_(False)
        self.analysis = ddsp.FeedbackAnalysis(doc).to(device)
        self.grid = targets.frequency_grid(doc)
        self.scale_by_pitch = scales_by_pitch(doc)


def head_validity(steps, samples):
    k = np.arange(steps)
    return (k >= HEAD_DELAY) & ((k - HEAD_DELAY) * frontend.HOP < samples)


def frontend_tables(doc):
    return {key: doc[key] for key in (frontend.PHONEMES, frontend.FRONTEND)}


def controls_rows(features, cents, control):
    return frontend.controls_input(features, cents, np.full(len(features), control, np.float32))


@torch.no_grad()
def label_features(doc, encoder, phonemes, cents, device='cpu'):
    x, header = frontend.encoder_input(doc, phonemes)
    frames = min(len(x), len(cents))
    features = encoder.run_aligned(torch.from_numpy(x[:frames])[None].to(device))[0].cpu().numpy()
    return frontend.pad_latency(header[:frames]), frontend.pad_latency(features), frontend.pad_latency(cents[:frames])


def step_voicing(voiced, valid):
    return voiced[np.clip(np.arange(len(valid)) - HEAD_DELAY, 0, len(valid) - 1)] & valid


def measure_targets(grid, scale_by_pitch, audio, cents, voiced, valid, base_tail, device):
    steps = len(valid)
    _, estimate, f0_hz = targets.measure(audio, cents, voiced, steps, grid, scale_by_pitch, device)
    labels = np.where(valid, step_voicing(voiced, valid), -1)
    curve_a = targets.smooth_in_runs(estimate.curve_a, labels)
    noise = targets.smooth_in_runs(estimate.curve_a + estimate.curve_b, labels)
    tail, weight = phase.phase_targets(estimate, f0_hz, targets.curves_to_logits(curve_a, noise - curve_a)[:, :len(grid)], base_tail, device)
    return dict(curve_a=quantise(curve_a * valid[:, None]), noise=quantise(noise * valid[:, None]), tail_target=tail, tail_weight=weight * valid)


@torch.no_grad()
def compute_section(m, section):
    phonemes = frontend.model_phonemes(section.labels, m.language)
    _, features, cents = label_features(m.doc, m.encoder, phonemes, section.cents, m.device)
    features = features.astype(np.float16)
    steps = len(features)
    voiced = frontend.pad_latency(section.voiced[:steps - frontend.LATENCY])
    x9 = torch.from_numpy(controls_rows(features.astype(np.float32), cents, m.control))[None].to(m.device)
    y9 = m.controls.run(x9)

    delayed = np.zeros(steps * frontend.HOP)
    delayed[frontend.OUTPUT_DELAY:] = section.audio[:len(delayed) - frontend.OUTPUT_DELAY]
    feedback = m.analysis.real_feedback(torch.from_numpy(delayed)[None].to(m.device), FEEDBACK_ALPHA).float()

    valid = head_validity(steps, len(section.audio))
    source = torch.cat([torch.full((1, WARMUP + 1, 96), -140.0, device=m.device), feedback[:, :-1]], 1)
    heads = m.synthesis(m.analysis.normalise(source), y9, torch.tensor([m.speaker], device=m.device))
    decoder_cents = ddsp.decoder_cents(torch.from_numpy(cents)[None], float(cents[0]))[0].numpy()

    tail = 2 * len(m.grid)
    base_tail = heads[0, WARMUP:, tail:tail + 8].cpu().numpy()
    return dict(
        features=features, cents=cents, decoder_cents=decoder_cents.astype(np.float32), valid=valid, voiced=step_voicing(voiced, valid),
        feedback=quantise(feedback[0].cpu().numpy()), base_tail=base_tail.astype(np.float16),
        **measure_targets(m.grid, m.scale_by_pitch, section.audio, cents, voiced, valid, base_tail, m.device),
        phoneme_start=np.array([p.start for p in phonemes], np.float64), phoneme_end=np.array([p.end for p in phonemes], np.float64),
        phoneme_name=np.array([p.name for p in phonemes]), phoneme_class=np.array([p.cls for p in phonemes], np.int8))


def section_measure(section):
    return effort.measure(section.audio.astype(np.float64), section.voiced, section.labels)


def write_controls(directory, index, measures):
    missing = {e['file'] for e in index['sections']} - measures.keys()
    if missing:
        raise ValueError(f'{len(missing)} shards have no measured recording, e.g. {sorted(missing)[0]}')
    controls = effort.frame_controls(measures, index['control'])
    for name, control in controls.items():
        arrays = load_section(directory, dict(file=name))
        frames = len(arrays['valid']) - frontend.LATENCY
        padded = frontend.pad_latency(control[np.minimum(np.arange(frames), len(control) - 1)])
        partial = directory / f'{name}.partial.npz'
        np.savez_compressed(partial, **dict(arrays, control=padded))
        partial.replace(directory / name)
    index['frame_control'] = True
    (directory / 'index.json').write_text(json.dumps(index, indent=1))


worker_models = None


def load_worker_models(base_model, speaker, control, device, language):
    global worker_models
    worker_models = Models(base_model, speaker, control, device, language)


def build_recording(recording, out_dir):
    entries, measures = [], {}
    for section in prepare_recording(recording):
        arrays = compute_section(worker_models, section)
        np.savez_compressed(out_dir / f'{section.name}.npz', **arrays)
        measures[f'{section.name}.npz'] = section_measure(section)
        torch.cuda.empty_cache()
        entries.append(dict(
            file=f'{section.name}.npz', recording=section.recording, split=section.split, offset=section.offset,
            steps=len(arrays['valid']), duration=len(section.audio) / frontend.SAMPLE_RATE,
            phonemes=dict(Counter(arrays['phoneme_name'].tolist()))))
    return entries, measures


def build(base_model, corpus, out_dir, speaker, control, validation=8, workers=6, device='cuda', max_recordings=None, language='ja'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings = corpus_recordings(corpus, validation)[:max_recordings]
    entries, measures = [], {}
    started = time.time()
    with multiprocessing.get_context('spawn').Pool(workers, load_worker_models, (base_model, speaker, control, device, language)) as pool:
        for done, (found, measured) in enumerate(pool.imap(functools.partial(build_recording, out_dir=out_dir), recordings), 1):
            entries += found
            measures.update(measured)
            print(f'{done}/{len(recordings)} recordings, {len(entries)} sections, {time.time() - started:.0f}s', flush=True)
    index = dict(base_sha256=file_digest(base_model), speaker=speaker, control=control, language=language, sections=entries)
    (out_dir / 'index.json').write_text(json.dumps(index, indent=1))
    write_controls(out_dir, index, measures)
    return index


def refresh_recording(recording, directory, shards, index):
    refreshed = 0
    for section in prepare_recording(recording):
        name = f'{section.name}.npz'
        if name not in shards:
            continue
        arrays = load_section(directory, dict(file=name))
        steps = len(arrays['valid'])
        voiced = frontend.pad_latency(section.voiced[:steps - frontend.LATENCY])
        phonemes = frontend.model_phonemes(section.labels, index.get('language', 'ja'))
        rows, _ = frontend.encoder_input(worker_models.doc, phonemes)
        assert min(len(rows), len(section.cents)) + frontend.LATENCY == steps
        assert [p.name for p in phonemes] == arrays['phoneme_name'].tolist()
        arrays.update(measure_targets(worker_models.grid, worker_models.scale_by_pitch, section.audio, arrays['cents'], voiced, arrays['valid'],
                                      arrays['base_tail'].astype(np.float32), worker_models.device))
        partial = directory / f'{name}.partial.npz'
        np.savez_compressed(partial, **arrays)
        partial.replace(directory / name)
        torch.cuda.empty_cache()
        refreshed += 1
    return refreshed


def refresh_targets(base_model, corpus, directory, workers=6, device='cuda', max_recordings=None):
    directory = Path(directory)
    index = load_index(directory)
    shards = {e['file'] for e in index['sections']}
    recordings = corpus_recordings(corpus)[:max_recordings]
    started, done = time.time(), 0
    pool_arguments = (base_model, index['speaker'], index['control'], device, index.get('language', 'ja'))
    with multiprocessing.get_context('spawn').Pool(workers, load_worker_models, pool_arguments) as pool:
        for count, refreshed in enumerate(pool.imap(functools.partial(refresh_recording, directory=directory, shards=shards, index=index), recordings), 1):
            done += refreshed
            print(f'{count}/{len(recordings)} recordings, {done} shards, {time.time() - started:.0f}s', flush=True)


def measure_recording(recording):
    return {f'{section.name}.npz': section_measure(section) for section in prepare_recording(recording)}


def add_controls(corpus, directory, workers=6, max_recordings=None):
    directory = Path(directory)
    index = load_index(directory)
    shards = {e['file'] for e in index['sections']}
    recordings = corpus_recordings(corpus)[:max_recordings]
    measures = {}
    started = time.time()
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        for count, found in enumerate(pool.imap(measure_recording, recordings), 1):
            measures.update({name: m for name, m in found.items() if name in shards})
            print(f'{count}/{len(recordings)} recordings, {time.time() - started:.0f}s', flush=True)
    write_controls(directory, index, measures)
    print(f'{len(measures)} shards rewritten, {time.time() - started:.0f}s', flush=True)


def load_index(directory):
    return json.loads((Path(directory) / 'index.json').read_text())


def load_section(directory, entry):
    with np.load(Path(directory) / entry['file']) as z:
        return {k: z[k] for k in z.files}


def section_control(index, arrays):
    return arrays['control'] if index.get('frame_control') else index['control']


def section_window(arrays, start, crop, tables, control, history=None):
    steps = len(arrays['valid'])
    width = WARMUP + crop
    extent = WARMUP + steps
    lo, hi = start, min(start + width, extent)

    def take(rows, fill=0):
        out = np.full((width, *rows.shape[1:]), fill, rows.dtype)
        out[:hi - lo] = rows[lo:hi]
        return out

    def lead(rows):
        return np.concatenate([np.zeros((WARMUP, *rows.shape[1:]), rows.dtype), rows])

    feedback = dequantise(arrays['feedback'])
    seen = feedback if history is None else dequantise(history)
    source = np.concatenate([np.full((WARMUP + 1, 96), -140.0, np.float32), seen[:-1]])
    static = controls_rows(np.zeros((steps, frontend.FEATURE_WIDTH), np.float32), arrays['cents'], control)
    lookback = np.concatenate([np.zeros((CONTROLS_LOOKBACK, 36), np.float32), static])
    x9_window = np.zeros((CONTROLS_LOOKBACK + width, 36), np.float32)
    x9_window[:CONTROLS_LOOKBACK + hi - lo] = lookback[lo:CONTROLS_LOOKBACK + hi]
    index, encoder_x = encoder_window(arrays, tables, lo, width)
    return dict(
        x9=x9_window, feature_index=index, encoder_x=encoder_x, source=take(source), feedback=take(lead(feedback)),
        curve_a=take(lead(dequantise(arrays['curve_a']))), noise=take(lead(dequantise(arrays['noise']))),
        valid=take(lead(arrays['valid'])), voiced=take(lead(arrays['voiced'])),
        cents=take(lead(arrays['decoder_cents'])), tail_target=take(lead(arrays['tail_target'].astype(np.float32))),
        tail_weight=take(lead(arrays['tail_weight'].astype(np.float32))), start=np.int64(start))


def stored_phonemes(arrays):
    n = len(arrays['phoneme_name'])
    classes = arrays['phoneme_class'].tolist() if 'phoneme_class' in arrays else [frontend.EDGE_CLASS if i in (0, n - 1) else frontend.JAPANESE_CLASS for i in range(n)]
    return [frontend.Phoneme(float(s), float(e), str(name), c)
            for s, e, name, c in zip(arrays['phoneme_start'], arrays['phoneme_end'], arrays['phoneme_name'], classes)]


def encoder_window(arrays, tables, lo, width):
    steps = len(arrays['valid'])
    frames = steps - frontend.LATENCY
    rows = np.arange(lo - CONTROLS_LOOKBACK, lo + width)
    k = rows - WARMUP
    source = np.where(k < 0, 0, np.maximum(k - 3, 0))
    absent = (rows < 0) | (rows >= WARMUP + steps) | ((k >= 0) & (k < 2))
    source = np.minimum(source, frames - 1)
    first = source[~absent].min()
    index = np.where(absent, -1, source - first)
    x415, _ = frontend.encoder_input(tables, stored_phonemes(arrays))
    if len(x415) < frames:
        raise ValueError('stored phoneme times rebuild fewer encoder rows than the shard has; rebuild the shard with float64 times')
    context = np.clip(np.arange(first - ENCODER_CONTEXT, first + len(rows) + ENCODER_CONTEXT), 0, frames - 1)
    return index, x415[context]


class Source(NamedTuple):
    directory: str
    rows: dict
    default_row: int


class CropDataset(Dataset):
    def __init__(self, base_model, sources, split, crop, histories=()):
        self.sources, self.crop = sources, crop
        self.indexes = [load_index(s.directory) for s in sources]
        self.entries = [dict(e, source=k) for k, index in enumerate(self.indexes) for e in index['sections'] if e['split'] == split]
        if histories:
            self.entries = [dict(e, history=str(h)) for h in histories for e in self.entries if (Path(h) / e['file']).exists()]
        self.tables = frontend_tables(Container.load(base_model).document)

    @property
    def durations(self):
        return [e['duration'] for e in self.entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        entry = self.entries[i]
        source, index = self.sources[entry['source']], self.indexes[entry['source']]
        arrays = load_section(source.directory, entry)
        start = int(torch.randint(0, max(len(arrays['valid']) - self.crop, 0) + 1, (1,)))
        history = load_section(entry['history'], entry)['feedback'] if 'history' in entry else None
        window = section_window(arrays, start, self.crop, self.tables, section_control(index, arrays), history)
        window['speaker'] = np.int64(source.rows.get(entry['recording'], source.default_row))
        return window


def full_sections(directory, entries, tables, index):
    sections = [load_section(directory, e) for e in entries]
    crop = max(len(s['valid']) for s in sections)
    windows = [section_window(s, 0, crop, tables, section_control(index, s)) for s in sections]
    return {k: torch.from_numpy(np.stack([w[k] for w in windows])) for k in windows[0]}
