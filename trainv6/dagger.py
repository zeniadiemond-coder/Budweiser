import multiprocessing
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import dataset, evaluate, frontend

_worker = {}


def choose_sections(index, minutes, seed):
    by_recording = {}
    for entry in index['sections']:
        if entry['split'] == 'train':
            by_recording.setdefault(entry['recording'], []).append(entry)
    rng = random.Random(seed)
    queues = [rng.sample(v, len(v)) for _, v in sorted(by_recording.items())]
    rng.shuffle(queues)
    chosen, seconds = [], 0.0
    while seconds < minutes * 60 and any(queues):
        for queue in queues:
            if queue and seconds < minutes * 60:
                chosen.append(queue.pop())
                seconds += chosen[-1]['duration']
    return chosen


def record_section(doc, stacks, speaker, index, arrays):
    steps = len(arrays['valid'])
    result = evaluate.synthesize(doc, stacks, speaker, dataset.section_control(index, arrays), dataset.stored_phonemes(arrays), arrays['cents'][:steps - frontend.LATENCY])
    return dict(feedback=dataset.quantise(result.sources), heads=result.heads.astype(np.float16))


def init_worker(checkpoint, dataset_dir, out_dir):
    torch.set_num_threads(1)
    document, stacks, speaker = evaluate.load_variants(checkpoint, None)['trained']
    _worker.update(doc=document, speaker=speaker, index=dataset.load_index(dataset_dir), stacks=stacks, dataset_dir=dataset_dir, out_dir=Path(out_dir))


def record_entry(entry):
    target = _worker['out_dir'] / entry['file']
    if not target.exists():
        arrays = dataset.load_section(_worker['dataset_dir'], entry)
        partial = target.with_name(f'{target.name}.partial.npz')
        np.savez_compressed(partial, **record_section(_worker['doc'], _worker['stacks'], _worker['speaker'], _worker['index'], arrays))
        partial.replace(target)
    return entry['duration']


def record(checkpoint, dataset_dir, out_dir, minutes=75.0, workers=12, seed=0):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    entries = sorted(choose_sections(dataset.load_index(dataset_dir), minutes, seed), key=lambda e: -e['duration'])
    started, recorded = time.time(), 0.0
    with multiprocessing.get_context('spawn').Pool(workers, init_worker, (checkpoint, dataset_dir, out_dir)) as pool:
        for done, duration in enumerate(pool.imap_unordered(record_entry, entries), 1):
            recorded += duration
            print(f'{done}/{len(entries)} sections, {recorded / 60:.1f} min audio, {time.time() - started:.0f}s', flush=True)
