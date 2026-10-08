import argparse
import json
from pathlib import Path


def build_dataset(args):
    from .dataset import build
    from .export import package_voice
    model, speaker, control = package_voice(args.base_package)
    build(model, args.corpus, args.out, speaker, control, args.validation, args.workers, args.device, args.max_recordings, args.language)


def refresh_targets(args):
    from .dataset import refresh_targets
    refresh_targets(args.base_model, args.corpus, args.dataset, args.workers, args.device, args.max_recordings)


def add_control(args):
    from .dataset import add_controls
    add_controls(args.corpus, args.dataset, args.workers, args.max_recordings)


def train_model(args):
    from .train import train
    options = {k: v for k, v in vars(args).items() if k not in ('command', 'run', 'config', 'base_model', 'dataset', 'out')}
    train(args.base_model, args.dataset, args.out, **options)


def export_model(args):
    from .export import export
    export(args.checkpoint, args.out)


def package(args):
    from .export import export_package
    export_package(args.checkpoint, args.base_package, args.out, args.speaker)


def rank_bases(args):
    from .speakers import rank_bases
    rank_bases(args.build_base, args.dataset, args.out, args.candidates, args.minutes, args.seed, args.device)


def rank(args):
    from .speakers import rank_speakers
    rank_speakers(args.base_model, args.dataset, args.out, args.minutes, args.seed, args.device)


def evaluate_model(args):
    from .evaluate import evaluate
    evaluate(args.model, args.corpus, args.dataset, args.out, args.split, args.max_sections, args.speaker, args.calibrate)


def renderer_parity(args):
    from .parity import renderer_parity
    renderer_parity(args.base_model, args.dataset, args.sections, args.out)


def record_history(args):
    from .dagger import record
    record(args.checkpoint, args.dataset, args.out, args.minutes, args.workers, args.seed)


def pitch_build(args):
    from .pitch_dataset import build
    build(args.base_vpit, args.corpus, args.out, args.scores, args.kana_table, args.variants, args.jitter, args.seed, args.workers, args.device)


def pitch_train(args):
    from .pitch_train import train
    options = {k: v for k, v in vars(args).items() if k not in ('command', 'run', 'config', 'base_vpit', 'dataset', 'out', 'base_row')}
    train(args.base_vpit, args.dataset, args.out, args.base_row, **options)


def pitch_export(args):
    from .pitch_export import export
    export(args.checkpoint, args.out, args.name)


def pitch_package(args):
    from .pitch_export import export_package
    export_package(args.checkpoint, args.base_package, args.out, args.name)


def parser():
    root = argparse.ArgumentParser(prog='trainv6')
    commands = root.add_subparsers(dest='command', required=True)

    build = commands.add_parser('build-dataset')
    build.add_argument('--base-package', required=True, help='voice package directory; supplies the model, its native speaker row and its ParamExp control')
    build.add_argument('--corpus', required=True)
    build.add_argument('--out', required=True)
    build.add_argument('--validation', type=int, default=8)
    build.add_argument('--workers', type=int, default=4)
    build.add_argument('--device', default='cuda')
    build.add_argument('--max-recordings', type=int)
    build.add_argument('--language', choices=('ja', 'en', 'zh'), default='ja', help='label language: ja (UTAU/NNSVS), en (ARPAbet) or zh (model phoneme names, e.g. iE_n+0)')
    build.set_defaults(run=build_dataset)

    augment = commands.add_parser('refresh-targets')
    augment.add_argument('base_model')
    augment.add_argument('corpus')
    augment.add_argument('dataset')
    augment.add_argument('--workers', type=int, default=4)
    augment.add_argument('--device', default='cuda')
    augment.add_argument('--max-recordings', type=int)
    augment.set_defaults(run=refresh_targets)

    framing = commands.add_parser('add-control', help='derive the per-frame control from the recordings and store it in the shards in place')
    framing.add_argument('corpus')
    framing.add_argument('dataset')
    framing.add_argument('--workers', type=int, default=6)
    framing.add_argument('--max-recordings', type=int)
    framing.set_defaults(run=add_control)

    trainer = commands.add_parser('train')
    trainer.add_argument('base_model')
    trainer.add_argument('dataset')
    trainer.add_argument('out')
    trainer.add_argument('--steps', type=int, help='total steps, 1 to 1000000 (default 6000), split 5%%/25%%/70%% across the three stages')
    trainer.add_argument('--stage-steps', type=int, nargs=3, help='explicit stage lengths instead of --steps')
    trainer.add_argument('--final-crop', type=int, default=1088, help='crop length of the fully free-running stage')
    trainer.add_argument('--lr-end', type=float, default=2e-5)
    trainer.add_argument('--extra-speaker', dest='extra_speakers', action='append', default=[], metavar='NAME=DATASET', help='train another new speaker row from its own dataset in the same run (same base model and language); repeatable')
    trainer.add_argument('--session-rows', type=int, default=1, help='train this many new speaker rows on separate groups of recordings and keep the one that validates best')
    for name, kind, default in [('batch', int, 8), ('crop', int, 544), ('chunk', int, 32), ('lr', float, 1e-3), ('warmup', int, 100),
                                ('jitter', float, None), ('val-every', int, 250), ('val-sections', int, 12),
                                ('feedback-frames', int, 64), ('seed', int, 0), ('workers', int, 4), ('log-every', int, 10),
                                ('aux-lr-scale', float, 0.3)]:
        trainer.add_argument(f'--{name}', type=kind, default=default)
    trainer.add_argument('--init')
    trainer.add_argument('--scratch', action='store_true', help='randomly initialise the synthesis, encoder and controls nets; the base model only supplies the architecture and frontend')
    trainer.add_argument('--speaker-name', help='name of the new embedding row; defaults to the run directory name')
    trainer.add_argument('--adapt-speaker', action='store_true', help='train the dataset speaker row in place instead of adding a new embedding row')
    trainer.add_argument('--speaker-init', help='rank-speakers output that initialises the new row; defaults to a copy of the dataset speaker row')
    trainer.add_argument('--histories', nargs='*', default=())
    trainer.add_argument('--device', default='cuda')
    trainer.add_argument('--noisy', action=argparse.BooleanOptionalAction, default=None, help="feedback noise mode; default is the base model's")
    trainer.add_argument('--train-encoder', action=argparse.BooleanOptionalAction, default=None, help='the encoder is shared between voices and frozen unless --scratch')
    for flag in ('train-controls', 'cuda-graph'):
        trainer.add_argument(f'--{flag}', action=argparse.BooleanOptionalAction, default=True)
    for flag in ('tf32', 'resume'):
        trainer.add_argument(f'--{flag}', action='store_true')
    trainer.set_defaults(run=train_model)

    base_ranker = commands.add_parser('rank-bases', help="score candidate base models' speaker rows on a dataset and rank the models")
    base_ranker.add_argument('build_base', help='the base model the dataset was built for')
    base_ranker.add_argument('dataset')
    base_ranker.add_argument('out')
    base_ranker.add_argument('candidates', nargs='+')
    base_ranker.add_argument('--minutes', type=float, default=12.0)
    base_ranker.add_argument('--seed', type=int, default=0)
    base_ranker.add_argument('--device', default='cuda')
    base_ranker.set_defaults(run=rank_bases)

    ranker = commands.add_parser('rank-speakers')
    ranker.add_argument('base_model')
    ranker.add_argument('dataset')
    ranker.add_argument('out')
    ranker.add_argument('--minutes', type=float, default=12.0)
    ranker.add_argument('--seed', type=int, default=0)
    ranker.add_argument('--device', default='cuda')
    ranker.set_defaults(run=rank)

    exporter = commands.add_parser('export')
    exporter.add_argument('checkpoint')
    exporter.add_argument('out')
    exporter.set_defaults(run=export_model)

    packager = commands.add_parser('package')
    packager.add_argument('checkpoint')
    packager.add_argument('base_package')
    packager.add_argument('out')
    packager.add_argument('--speaker', help='trained speaker the package points at; defaults to the primary one')
    packager.set_defaults(run=package)

    evaluator = commands.add_parser('evaluate')
    evaluator.add_argument('model', help='checkpoint (.pt) for base vs trained, or a .vtb2 for the base alone')
    evaluator.add_argument('corpus')
    evaluator.add_argument('dataset')
    evaluator.add_argument('out')
    evaluator.add_argument('--split', default='validation')
    evaluator.add_argument('--max-sections', type=int, default=4)
    evaluator.add_argument('--speaker', type=int, help='embedding row; required for a .vtb2')
    evaluator.add_argument('--calibrate', action='store_true', help='search the constant control that zeroes the loudness difference and print the matching ParamExp')
    evaluator.set_defaults(run=evaluate_model)

    parity = commands.add_parser('renderer-parity', help='compare the training feedback renderer with the reference synthesizer on the same heads')
    parity.add_argument('base_model')
    parity.add_argument('dataset')
    parity.add_argument('--sections', type=int, default=4)
    parity.add_argument('--out', help='write the full per-band report as JSON')
    parity.set_defaults(run=renderer_parity)

    recorder = commands.add_parser('record-history')
    recorder.add_argument('checkpoint')
    recorder.add_argument('dataset')
    recorder.add_argument('out')
    recorder.add_argument('--minutes', type=float, default=75.0)
    recorder.add_argument('--workers', type=int, default=12)
    recorder.add_argument('--seed', type=int, default=0)
    recorder.set_defaults(run=record_history)

    pitch_builder = commands.add_parser('pitch-build', help='build the pitch dataset from the corpus recordings and the UTAU scores')
    pitch_builder.add_argument('base_vpit')
    pitch_builder.add_argument('corpus', help='corpus manifest.json')
    pitch_builder.add_argument('out')
    pitch_builder.add_argument('--scores', required=True, help='directory with one <song>/ folder per recording id, each holding <song>.ust and <song>.lab')
    pitch_builder.add_argument('--kana-table', required=True, help='oto2lab kana-to-phonemes table')
    pitch_builder.add_argument('--variants', type=int, default=3)
    pitch_builder.add_argument('--jitter', type=float, default=0.02)
    pitch_builder.add_argument('--seed', type=int, default=0)
    pitch_builder.add_argument('--workers', type=int, default=8)
    pitch_builder.add_argument('--device', default='cuda')
    pitch_builder.set_defaults(run=pitch_build)

    pitch_trainer = commands.add_parser('pitch-train')
    pitch_trainer.add_argument('base_vpit')
    pitch_trainer.add_argument('dataset')
    pitch_trainer.add_argument('out')
    pitch_trainer.add_argument('--stage-steps', type=int, nargs=2, default=(2500, 1500))
    for name, kind, default in [('batch', int, 16), ('length', int, 768), ('free-length', int, 320), ('lr', float, 2e-4), ('warmup', int, 100),
                                ('val-every', int, 250), ('val-crops', int, 48), ('seed', int, 0), ('log-every', int, 10)]:
        pitch_trainer.add_argument(f'--{name}', type=kind, default=default)
    pitch_trainer.add_argument('--base-row', type=int, required=True, help='existing pitch row the new row is distilled towards')
    pitch_trainer.add_argument('--init', help="pitch row that initialises the new row, or 'mean'; defaults to --base-row")
    pitch_trainer.add_argument('--device', default='cuda')
    pitch_trainer.add_argument('--resume', action='store_true')
    pitch_trainer.set_defaults(run=pitch_train)

    pitch_exporter = commands.add_parser('pitch-export')
    pitch_exporter.add_argument('checkpoint')
    pitch_exporter.add_argument('out')
    pitch_exporter.add_argument('--name', required=True, help='name of the new speaker row')
    pitch_exporter.set_defaults(run=pitch_export)

    pitch_packager = commands.add_parser('pitch-package', help='copy a voice package and replace its pitch model with the exported one')
    pitch_packager.add_argument('checkpoint')
    pitch_packager.add_argument('base_package')
    pitch_packager.add_argument('out')
    pitch_packager.add_argument('--name', required=True, help='character name, also used as the speaker row name')
    pitch_packager.set_defaults(run=pitch_package)
    for command in commands.choices.values():
        command.add_argument('--config', action='append', default=[], metavar='FILE', help='JSON file of option values; repeat for several, later files win and command-line options override them')
    return root


def load_configs(paths):
    merged = {}
    for path in paths:
        values = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(values, dict):
            raise ValueError(f'{path}: a config is a JSON object of option values')
        merged.update({key.replace('-', '_'): value for key, value in values.items()})
    return merged


def apply_configs(root, argv):
    reader = argparse.ArgumentParser(add_help=False)
    reader.add_argument('--config', action='append', default=[])
    paths = reader.parse_known_args(argv)[0].config
    if not paths:
        return
    command = next((a for a in argv if a in root._subparsers._group_actions[0].choices), None)
    sub = root._subparsers._group_actions[0].choices[command]
    values = load_configs(paths)
    known = {}
    for action in sub._actions:
        if action.option_strings and action.dest not in ('help', 'config'):
            known[action.dest] = action
            known.update({o.lstrip('-').replace('-', '_'): action for o in action.option_strings})
    unknown = sorted(set(values) - known.keys())
    if unknown:
        raise ValueError(f'{command} has no such options (positional arguments cannot come from a config): {unknown}')
    for key, value in values.items():
        known[key].required = False
        sub.set_defaults(**{known[key].dest: value})


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    root = parser()
    apply_configs(root, argv)
    args = root.parse_args(argv)
    args.run(args)
