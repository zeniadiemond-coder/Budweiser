import json
import math
from pathlib import Path

import numpy as np
import torch

from . import dagger, dataset, frontend, train
from .container import file_digest
from .export import speaker_names
from .reference.frontend import noisy_feedback

BATCH = 8
FRAMES = 128
BLEND_ROWS = 3
BLEND_MARGIN = 0.98


@torch.no_grad()
def score_rows(models, directory, index, entries, rows, seed):
    sums = {row: {} for row in rows}
    weight = 0.0
    cuda = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    for i in range(0, len(entries), BATCH):
        b = train.to_device(dataset.full_sections(directory, entries[i:i + BATCH], dataset.frontend_tables(models.doc), index), models.device)
        t = train.make_targets(b)
        y9 = train.conditioning(models.encoder, models.controls, b, False, False)
        steps = t.den_steps.item()
        weight += steps
        for row in rows:
            with torch.random.fork_rng(devices=cuda):
                torch.manual_seed(seed + i)
                speakers = torch.full((len(b['start']),), row, device=models.device)
                parts = train.teacher_forced(models.synthesis, models.analysis, y9, speakers, b, t, 0.0, FRAMES)
            parts['loss'] = train.total(parts)
            for key, value in parts.items():
                sums[row][key] = sums[row].get(key, 0.0) + steps * value.item()
    return {row: {key: value / weight for key, value in parts.items()} for row, parts in sums.items()}


def rank_speakers(base_model, dataset_dir, out, minutes=12.0, seed=0, device='cuda'):
    index = dataset.load_index(dataset_dir)
    models = dataset.Models(base_model, index['speaker'], index['control'], device, index.get('language', 'ja'))
    entries = sorted(dagger.choose_sections(index, minutes, seed), key=lambda e: e['steps'])
    names = {row: name for name, row in speaker_names(models.doc).items()}
    rows = list(range(models.synthesis.rows))
    scores = score_rows(models, dataset_dir, index, entries, rows, seed)
    ranking = sorted(rows, key=lambda row: scores[row]['loss'])
    top = ranking[:BLEND_ROWS]
    spread = max(scores[top[-1]]['loss'] - scores[top[0]]['loss'], 1e-9)
    raw = [math.exp(-(scores[row]['loss'] - scores[top[0]]['loss']) / spread) for row in top]
    weights = [value / sum(raw) for value in raw]
    blend = sum(weight * models.synthesis.w['embedding'][row].detach() for row, weight in zip(top, weights))
    models.synthesis.add_speaker(blend)
    blend_scores = score_rows(models, dataset_dir, index, entries, [models.synthesis.rows - 1], seed)[models.synthesis.rows - 1]
    use_blend = blend_scores['loss'] < BLEND_MARGIN * scores[top[0]]['loss']
    result = dict(
        seconds=sum(e['duration'] for e in entries), sections=len(entries),
        ranking=[dict(row=row, name=names.get(row), **scores[row]) for row in ranking],
        blend=dict(rows=top, weights=weights, **blend_scores), use_blend=use_blend,
        init=[[row, weight] for row, weight in zip(top, weights)] if use_blend else [[top[0], 1.0]])
    Path(out).write_text(json.dumps(result, indent=1))
    return result


def incompatibility(reference, candidate, index):
    if reference.grid.shape != candidate.grid.shape or not np.allclose(reference.grid, candidate.grid):
        return 'curve frequency grid differs from the dataset base'
    if reference.scale_by_pitch != candidate.scale_by_pitch:
        return 'harmonic level scaling differs from the dataset base'
    if reference.analysis.band_hz.shape != candidate.analysis.band_hz.shape or not torch.allclose(reference.analysis.band_hz, candidate.analysis.band_hz):
        return 'feedback band layout differs from the dataset base'
    table = candidate.doc[frontend.PHONEMES]
    names = {name for e in index['sections'] for name in e['phonemes']}
    missing = sorted(name for name in names if name not in table and ''.join(c for c in name if c.isascii() and c.isalpha()) not in table)
    if missing:
        return f'phonemes missing from the model: {missing}'
    probe = [frontend.Phoneme(0.0, 0.1, 'sil', frontend.EDGE_CLASS), frontend.Phoneme(0.1, 0.2, 'sil', frontend.EDGE_CLASS)]
    if frontend.encoder_input(candidate.doc, probe)[0].shape[1] != candidate.encoder.inputs:
        return 'encoder input layout differs'
    return None


def rank_bases(build_base, dataset_dir, out, candidates, minutes=12.0, seed=0, device='cuda'):
    index = dataset.load_index(dataset_dir)
    if file_digest(build_base) != index['base_sha256']:
        raise ValueError(f'dataset was not built for {build_base}')
    reference = dataset.Models(build_base, index['speaker'], index['control'], device, index.get('language', 'ja'))
    entries = sorted(dagger.choose_sections(index, minutes, seed), key=lambda e: e['steps'])
    results = []
    for path in candidates:
        models = dataset.Models(path, index['speaker'], index['control'], device, index.get('language', 'ja'))
        reason = incompatibility(reference, models, index)
        if reason:
            results.append(dict(base=str(path), skipped=reason))
        else:
            rows = list(range(models.synthesis.rows))
            scores = score_rows(models, dataset_dir, index, entries, rows, seed)
            row = min(rows, key=lambda r: scores[r]['loss'])
            names = {r: name for name, r in speaker_names(models.doc).items()}
            results.append(dict(base=str(path), sha256=file_digest(path), row=row, name=names.get(row), noisy=noisy_feedback(models.doc), **scores[row]))
        del models
        torch.cuda.empty_cache()
    ranked = sorted((r for r in results if 'skipped' not in r), key=lambda r: r['loss']) + [r for r in results if 'skipped' in r]
    Path(out).write_text(json.dumps(dict(seconds=sum(e['duration'] for e in entries), sections=len(entries), ranking=ranked), indent=1))
    return ranked
