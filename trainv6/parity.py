import json

import numpy as np
import torch

from . import dataset, ddsp, evaluate, frontend
from .container import Container
from .export import document_stacks

LEVEL_BUCKETS = 3


def section_errors(doc, stacks, analysis, speaker, index, directory, entry):
    arrays = dataset.load_section(directory, entry)
    steps = len(arrays['valid'])
    result = evaluate.synthesize(doc, stacks, speaker, dataset.section_control(index, arrays), dataset.stored_phonemes(arrays), arrays['cents'][:steps - frontend.LATENCY])
    count = len(result.heads)
    f0 = torch.from_numpy(frontend.cents_to_hz(arrays['decoder_cents'][:count].astype(np.float64)).astype(np.float32))[None]
    with torch.no_grad():
        rendered = analysis(torch.from_numpy(result.heads)[None], f0, 0.0)[0].numpy()
    keep = arrays['valid'][:count] & arrays['voiced'][:count]
    keep[:ddsp.WINDOW_FRAMES] = False
    return rendered[keep], result.sources[keep]


def renderer_parity(base_model, dataset_dir, sections=4, out=None):
    index = dataset.load_index(dataset_dir)
    doc = Container.load(base_model).document
    stacks = document_stacks(doc, native_math=True)
    analysis = ddsp.FeedbackAnalysis(doc)
    pool = [e for e in index['sections'] if e['split'] == 'validation']
    entries = [pool[i] for i in np.linspace(0, len(pool) - 1, min(sections, len(pool))).round().astype(int)]
    pairs = [section_errors(doc, stacks, analysis, index['speaker'], index, dataset_dir, e) for e in entries]
    rendered, reference = np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
    error = rendered - reference
    level = reference.mean(axis=1)
    edges = np.quantile(level, np.linspace(0, 1, LEVEL_BUCKETS + 1))
    bucket = np.clip(np.searchsorted(edges, level, side='right') - 1, 0, LEVEL_BUCKETS - 1)
    report = dict(
        frames=len(error),
        bias_db=error.mean(axis=0).tolist(), std_db=error.std(axis=0).tolist(),
        bias_16=float(error[:, :16].mean()), bias_80=float(error[:, 16:].mean()),
        std_16=float(error[:, :16].std()), std_80=float(error[:, 16:].std()),
        by_level=[dict(mean_level_db=float(level[bucket == b].mean()), bias_db=float(error[bucket == b].mean()), std_db=float(error[bucket == b].std()))
                  for b in range(LEVEL_BUCKETS)])
    if out:
        with open(out, 'w') as f:
            json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ('bias_db', 'std_db')}, indent=1))
    return report
