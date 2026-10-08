import json

import pytest
import torch

from trainv6 import cli, frontend, train


def test_learning_rate_decays_from_peak_to_end():
    peak = train.learning_rate(500, 1000, 1e-3, 100, 2e-5)
    assert peak < 1e-3 and train.learning_rate(0, 1000, 1e-3, 100, 2e-5) < 2e-5 * 10
    assert train.learning_rate(1000, 1000, 1e-3, 100, 2e-5) == pytest.approx(2e-5)


def test_stage_steps_default_and_custom_total():
    assert train.resolve_stage_steps(None, None) == (300, 1500, 4200)
    assert sum(train.resolve_stage_steps(600_000, None)) == 600_000
    with pytest.raises(ValueError):
        train.resolve_stage_steps(1_000_001, None)
    with pytest.raises(ValueError):
        train.resolve_stage_steps(6000, (1, 1, 1))


def test_ema_weights_apply_and_restore():
    p = torch.nn.Parameter(torch.zeros(3))
    ema = train.Ema([p])
    for value in (1.0, 2.0, 3.0):
        p.data.fill_(value)
        ema.update()
    with ema.applied():
        assert 0 < p.mean() < 3
    assert torch.all(p == 3.0)


def test_unvoiced_frames_pull_curve_a_down():
    b = dict(valid=torch.tensor([[1, 1]]), voiced=torch.tensor([[1, 0]]), curve_a=torch.zeros(1, 2, 4), tail_weight=torch.zeros(1, 2))
    t = train.make_targets(b)
    assert t.weight[0, 1].sum() == 0 and t.weight[0, 0].sum() > 0
    assert t.unvoiced_weight[0, 1] == train.UNVOICED_A_WEIGHT and t.unvoiced_weight[0, 0] == 0


def test_session_rows_split_recordings_contiguously():
    index = dict(sections=[dict(recording=r, split='train') for r in 'abcdef'])
    assert train.session_assignment(index, 3, 10) == dict(a=10, b=10, c=11, d=11, e=12, f=12)
    with pytest.raises(ValueError):
        train.session_assignment(index, 7, 10)


def test_configs_merge_in_order_and_command_line_wins(tmp_path):
    (tmp_path / 'a.json').write_text(json.dumps({'steps': 12000, 'lr': 5e-4, 'final-crop': 700}))
    (tmp_path / 'b.json').write_text(json.dumps({'lr': 2e-4}))
    argv = ['train', 'm', 'd', 'o', '--config', str(tmp_path / 'a.json'), '--config', str(tmp_path / 'b.json'), '--final-crop', '800']
    root = cli.parser()
    cli.apply_configs(root, argv)
    args = root.parse_args(argv)
    assert (args.steps, args.lr, args.final_crop) == (12000, 2e-4, 800)


def test_config_rejects_unknown_option(tmp_path):
    (tmp_path / 'a.json').write_text(json.dumps({'nonsense': 1}))
    argv = ['train', 'm', 'd', 'o', '--config', str(tmp_path / 'a.json')]
    with pytest.raises(ValueError):
        cli.apply_configs(cli.parser(), argv)


def test_english_labels_map_to_model_phonemes_with_english_class():
    labels = [(0.0, 0.1, 'SIL'), (0.1, 0.2, 'HH'), (0.2, 0.3, 'AH0'), (0.3, 0.4, 'sil')]
    phonemes = frontend.model_phonemes(labels, 'en')
    assert [p.name for p in phonemes] == ['sil', 'hh', 'ah', 'sil']
    assert [p.cls for p in phonemes] == [frontend.EDGE_CLASS, frontend.ENGLISH_CLASS, frontend.ENGLISH_CLASS, frontend.EDGE_CLASS]
    with pytest.raises(ValueError):
        frontend.model_phonemes([(0.0, 0.1, 'QQ')], 'en')


def test_chinese_labels_keep_model_names():
    phonemes = frontend.model_phonemes([(0.0, 0.1, 'sil'), (0.1, 0.3, 'iE_n+0'), (0.3, 0.4, 'sil')], 'zh')
    assert phonemes[1].name == 'iE_n+0' and phonemes[1].cls == frontend.CHINESE_CLASS


def test_config_accepts_option_names_as_written_on_the_command_line(tmp_path):
    (tmp_path / 'a.json').write_text(json.dumps({'extra-speaker': ['power=data2'], 'final_crop': 640}))
    argv = ['train', 'm', 'd', 'o', '--config', str(tmp_path / 'a.json')]
    root = cli.parser()
    cli.apply_configs(root, argv)
    args = root.parse_args(argv)
    assert args.extra_speakers == ['power=data2'] and args.final_crop == 640


def test_checkpoint_speakers_fall_back_to_single_speaker():
    from trainv6.export import checkpoint_speakers
    assert checkpoint_speakers(dict(speaker=5, speaker_name='a')) == {'a': 5}
    assert checkpoint_speakers(dict(speaker=5, speaker_name='a', speakers={'a': 5, 'b': 6})) == {'a': 5, 'b': 6}
    assert checkpoint_speakers(dict(speaker=5, speaker_name=None)) == {}
