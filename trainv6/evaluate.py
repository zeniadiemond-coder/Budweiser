import json
import math
from pathlib import Path

import numpy as np
import scipy.fft
import soundfile
import torch

from . import dataset, effort, frontend, nets
from .container import Container
from .export import document_stacks, exported_container, load_checkpoint
from .reference.synth import Synth

RATE = frontend.SAMPLE_RATE
FFT = 1024
STEP = 320
BANDS = 80
CEPSTRA = 24
BURST_DB = 12.0
DROPOUT_DB = -20.0
METRIC_FRAME_SECONDS = STEP / RATE


def select_sections(corpus, dataset_dir, split, max_sections):
    splits = {e['recording']: e['split'] for e in dataset.load_index(dataset_dir)['sections']}
    recordings = [r for r in dataset.corpus_recordings(corpus) if splits.get(r.id) == split]
    recordings = [recordings[i] for i in np.linspace(0, len(recordings) - 1, min(max_sections, len(recordings))).round().astype(int)]
    per_recording = -(-max_sections // len(recordings))
    sections = []
    for recording in recordings:
        prepared = dataset.prepare_recording(recording)
        sections += [prepared[i] for i in np.linspace(0, len(prepared) - 1, min(per_recording, len(prepared))).round().astype(int)]
    return sections[:max_sections]


class StatefulBf8:
    def __init__(self, stack):
        self.stack = stack
        self.state = stack.init_state(1)

    def __call__(self, feedback, conditioning, speaker):
        with torch.no_grad():
            head, self.state = self.stack.step(torch.from_numpy(feedback)[None], torch.from_numpy(conditioning)[None],
                                               torch.tensor([speaker]), self.state)
        return head[0].numpy()


def synthesize(doc, stacks, speaker, control, phonemes, cents):
    header, features, f0 = dataset.label_features(doc, stacks[nets.ENCODER], phonemes, cents)
    vec = np.concatenate([header, features], axis=1)
    controls = np.broadcast_to(np.float32(control), (len(vec),))
    rows = [dict(vec=v, f0=p, control=c, alpha=0.0) for v, p, c in zip(vec, f0, controls)]

    def controls(x9):
        with torch.no_grad():
            return stacks[nets.CONTROLS].run(torch.from_numpy(x9)[None])[0].numpy()

    return Synth(doc, speaker, controls, StatefulBf8(stacks[nets.SYNTHESIS])).run(rows, rows[0])


def resynthesize(doc, stacks, speaker, control, phonemes, cents, length):
    result = synthesize(doc, stacks, speaker, control, phonemes, cents)
    mixed = result.harm + result.noise
    return mixed[frontend.OUTPUT_DELAY:frontend.OUTPUT_DELAY + length], result.events


def mel_filterbank():
    def mel(f):
        return 2595 * np.log10(1 + f / 700)

    edges = 700 * (10 ** (np.linspace(0, mel(RATE / 2), BANDS + 2) / 2595) - 1)
    bins = np.fft.rfftfreq(FFT, 1 / RATE)
    rising = (bins[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    falling = (edges[2:, None] - bins[None]) / (edges[2:] - edges[1:-1])[:, None]
    return np.maximum(0, np.minimum(rising, falling))


FILTERBANK = mel_filterbank()
WINDOW = np.hanning(FFT)


def frame_features(audio, frames):
    padded = np.pad(audio, (FFT // 2, FFT))
    blocks = np.lib.stride_tricks.sliding_window_view(padded, FFT)[::STEP][:frames]
    power = np.abs(np.fft.rfft(blocks * WINDOW, axis=1)) ** 2
    mel_db = 10 * np.log10(power @ FILTERBANK.T + 1e-10)
    level_db = 10 * np.log10((blocks ** 2).mean(axis=1) + 1e-12)
    return mel_db, level_db


def cepstra(mel_db):
    return math.sqrt(2 / BANDS) * scipy.fft.dct(mel_db * math.log(10) / 20, type=2, norm='ortho', axis=1)[:, 1:CEPSTRA + 1]


def frame_metrics(recording, synthesized, voiced):
    frames = min(len(recording), len(synthesized)) // STEP
    voiced_frames = np.minimum((np.arange(frames) * METRIC_FRAME_SECONDS / dataset.FRAME_SECONDS).astype(int), len(voiced) - 1)
    keep = voiced[voiced_frames]
    reference_mel, reference_level = frame_features(recording, frames)
    mel, level = frame_features(synthesized, frames)
    difference = (mel - reference_mel)[keep]
    return {
        'band_difference': difference,
        'lsd': np.sqrt((difference ** 2).mean(axis=1)),
        'mcd': 10 / math.log(10) * np.sqrt(2 * ((cepstra(mel) - cepstra(reference_mel))[keep] ** 2).sum(axis=1)),
        'loudness': (level - reference_level)[keep],
    }


def summarize(rows):
    band_difference = np.concatenate([r['band_difference'] for r in rows])
    loudness = np.concatenate([r['loudness'] for r in rows])
    return {
        'voiced_frames': len(loudness),
        'lsd_db': float(np.concatenate([r['lsd'] for r in rows]).mean()),
        'band_median_abs_db': np.median(np.abs(band_difference), axis=0).tolist(),
        'band_median_signed_db': np.median(band_difference, axis=0).tolist(),
        'mcd_db': float(np.concatenate([r['mcd'] for r in rows]).mean()),
        'loudness_diff_db': float(loudness.mean()),
        'loudness_abs_diff_db': float(np.abs(loudness).mean()),
        'burst_fraction': float((loudness > BURST_DB).mean()),
        'dropout_fraction': float((loudness < DROPOUT_DB).mean()),
    }


def load_variants(model, base_speaker):
    if str(model).endswith('.vtb2'):
        container = Container.load(model)
        return {'base': (container.document, document_stacks(container.document, native_math=True), base_speaker)}
    checkpoint = load_checkpoint(model)
    container = Container.load(checkpoint['base_model'])
    shipped, _ = exported_container(checkpoint)
    return {'base': (container.document, document_stacks(container.document, native_math=True), base_speaker),
            'trained': (shipped.document, document_stacks(shipped.document, native_math=True), checkpoint['speaker'])}


def control_modes(index, directory, section):
    modes = {'': index['control']}
    if index.get('frame_control'):
        modes['_frame_control'] = dataset.load_section(directory, dict(file=f'{section.name}.npz'))['control']
    return modes


def mean_loudness_difference(document, stacks, row, control, sections, language):
    differences = []
    for section in sections:
        audio, _ = resynthesize(document, stacks, row, control, frontend.model_phonemes(section.labels, language), section.cents, len(section.audio))
        differences.append(frame_metrics(section.audio.astype(np.float64), audio, section.voiced)['loudness'])
    return float(np.concatenate(differences).mean())


def calibrate_param_exp(document, stacks, row, start, sections, language, iterations=4, tolerance=0.25):
    tried = {}
    control = start
    for _ in range(iterations + 1):
        tried[control] = mean_loudness_difference(document, stacks, row, control, sections, language)
        if abs(tried[control]) <= tolerance:
            break
        (c1, f1), *earlier = reversed(list(tried.items()))
        slope = (f1 - earlier[0][1]) / (c1 - earlier[0][0]) if earlier and f1 != earlier[0][1] else effort.LEVEL_DB_PER_CONTROL
        control = min(max(c1 - f1 / slope, -1.0), 1.0)
        if control in tried:
            break
    best = min(tried, key=lambda c: abs(tried[c]))
    return dict(control=best, param_exp=round(3000 + 3000 * best), loudness_diff_db=tried[best], tried={f'{c:.4f}': v for c, v in tried.items()})


def evaluate(model, corpus, dataset_dir, out_dir, split='validation', max_sections=4, speaker=None, calibrate=False):
    index = dataset.load_index(dataset_dir)
    variants = load_variants(model, index['speaker'] if speaker is None else speaker)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_variant, per_events = {}, {}
    sections = []
    language = index.get('language', 'ja')
    selected = select_sections(corpus, dataset_dir, split, max_sections)
    for section in selected:
        phonemes = frontend.model_phonemes(section.labels, language)
        soundfile.write(out / f'{section.name}_recording.wav', section.audio, RATE, subtype='FLOAT')
        entry = {'id': section.name}
        for variant, (document, stacks, row) in variants.items():
            for suffix, control in control_modes(index, dataset_dir, section).items():
                name = variant + suffix
                audio, events = resynthesize(document, stacks, row, control, phonemes, section.cents, len(section.audio))
                soundfile.write(out / f'{section.name}_{name}.wav', audio, RATE, subtype='FLOAT')
                rows = frame_metrics(section.audio.astype(np.float64), audio, section.voiced)
                per_variant.setdefault(name, []).append(rows)
                entry[name] = summarize([rows])
                entry[name]['decoder_events'] = events
                totals = per_events.setdefault(name, dict.fromkeys(events, 0))
                for key, count in events.items():
                    totals[key] += count
        sections.append(entry)
    overall = {name: summarize(rows) for name, rows in per_variant.items()}
    for name, totals in per_events.items():
        overall[name]['decoder_events'] = totals
    metrics = {'sections': sections, 'overall': overall}
    if calibrate:
        variant = 'trained' if 'trained' in variants else 'base'
        document, stacks, row = variants[variant]
        metrics['calibration'] = dict(variant=variant, **calibrate_param_exp(document, stacks, row, index['control'], selected, language))
        print(f'ParamExp for zero loudness difference at constant control ({variant}): {metrics["calibration"]["param_exp"]} '
              f'(control {metrics["calibration"]["control"]:+.3f}, {metrics["calibration"]["loudness_diff_db"]:+.2f} dB)')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    print_table(metrics['overall'])
    return metrics


def print_table(overall):
    rows = [('voiced_frames', '{:d}'), ('lsd_db', '{:.2f}'), ('mcd_db', '{:.2f}'), ('loudness_diff_db', '{:+.2f}'),
            ('loudness_abs_diff_db', '{:.2f}'), ('burst_fraction', '{:.3f}'), ('dropout_fraction', '{:.3f}')]
    print(f'{"metric":24}' + ''.join(f'{name:>22}' for name in overall))
    for key, fmt in rows:
        print(f'{key:24}' + ''.join(f'{fmt.format(m[key]):>22}' for m in overall.values()))
    print(f'{"band_median_abs_db":24}' + ''.join(f'{np.median(m["band_median_abs_db"]):>22.2f}' for m in overall.values()))
    for key in ('noise_guard', 'attenuated', 'zeroed'):
        print(f'{key + "_fraction":24}' + ''.join(f'{m["decoder_events"][key] / m["decoder_events"]["frames"]:>22.4f}' for m in overall.values()))
