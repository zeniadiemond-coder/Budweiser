import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from . import nets
from .reference.frontend import FEEDBACK_NOISE, NOISE_CONFIG, NOISY_VALUE, QUIET_VALUE, noisy_feedback
from .container import Container, file_digest, read_tensor, tensor_tag, DTYPES, NATIVE

STACKS = {nets.ENCODER: nets.GatedStack, nets.CONTROLS: nets.GatedStack, nets.SYNTHESIS: nets.CausalStack}
SITES = {nets.GatedStack: nets.gated_sites, nets.CausalStack: nets.causal_sites}


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def document_stacks(document, native_math=False):
    return {key: stack(document[key], native_math) for key, stack in STACKS.items()}


def trained_stacks(container, state_dicts):
    stacks = {}
    for key, state in state_dicts.items():
        stack = STACKS[key](container.document[key])
        if key == nets.SYNTHESIS:
            for _ in range(state['w.embedding'].shape[0] - stack.rows):
                stack.add_speaker(torch.zeros(stack.w['embedding'].shape[1]))
        stack.load_state_dict(state)
        stacks[key] = stack
    return stacks


def checkpoint_speakers(state):
    return state.get('speakers') or ({state['speaker_name']: state['speaker']} if state.get('speaker_name') else {})


def speaker_names(document):
    return next(value for key, value in document.items() if key.startswith(nets.SPEAKER_NAMES))


def exported_container(state, base_model=None):
    container = Container.load(base_model or state['base_model'])
    stacks = trained_stacks(container, state['nets'])
    for key, stack in stacks.items():
        stack.store(container.document[key])
    speaker_names(container.document).update(checkpoint_speakers(state))
    if noisy_feedback(container.document) != state['noisy_feedback']:
        container.document[NOISE_CONFIG][FEEDBACK_NOISE] = NOISY_VALUE if state['noisy_feedback'] else QUIET_VALUE
    return container, stacks


def verify_export(base, exported_path, stacks, state):
    exported = Container.load(exported_path)
    if exported.header[:4] + exported.header[8:] != base.header[:4] + base.header[8:] or exported.trailer != base.trailer:
        raise ValueError('exported header or trailer differs from the base')
    for key, stack in stacks.items():
        base_sites = SITES[type(stack)](base.document[key])
        for name, site in SITES[type(stack)](exported.document[key]).items():
            tag = tensor_tag(base_sites[name].holder[base_sites[name].key])
            expected = stack.w[name].detach().cpu().numpy().reshape(-1).astype(DTYPES[tag]).astype(NATIVE[tag])
            if not np.array_equal(read_tensor(site.holder[site.key]), expected):
                raise ValueError(f'{key}/{name} differs from the checkpoint weights')
            site.holder[site.key] = base_sites[name].holder[base_sites[name].key]
    synthesis = exported.document[nets.SYNTHESIS][nets.PARAMS]
    if synthesis[nets.EMBEDDING_ROWS] != stacks[nets.SYNTHESIS].rows:
        raise ValueError('embedding row count was not updated')
    synthesis[nets.EMBEDDING_ROWS] = base.document[nets.SYNTHESIS][nets.PARAMS][nets.EMBEDDING_ROWS]
    for name in checkpoint_speakers(state):
        del speaker_names(exported.document)[name]
    if noisy_feedback(exported.document) != state['noisy_feedback']:
        raise ValueError('feedback noise mode does not match the checkpoint')
    exported.document[NOISE_CONFIG][FEEDBACK_NOISE] = base.document[NOISE_CONFIG][FEEDBACK_NOISE]
    if exported.document != base.document:
        raise ValueError('sections outside the trained tensors differ from the base')


def export(checkpoint, output_vtb2, base_model=None):
    state = load_checkpoint(checkpoint)
    container, stacks = exported_container(state, base_model)
    container.save(output_vtb2)
    verify_export(Container.load(base_model or state['base_model']), output_vtb2, stacks, state)


def bank_component(bank_text):
    return next(value for key, value in json.loads(bank_text).items() if key.startswith('Components\\') and 'PModelName' in value)


def package_voice(package_dir):
    component = bank_component((Path(package_dir) / 'bank.json').read_text(encoding='utf-8'))
    model = Path(package_dir) / f"{component.get('TModelName', component['PModelName'])}.vtb2"
    return model, component['NPIdx'] & 0xFF, (component.get('ParamExp', 3000) - 3000) / 3000


def export_package(checkpoint, base_package_dir, out_dir, speaker=None):
    state = load_checkpoint(checkpoint)
    speakers = checkpoint_speakers(state)
    if speaker is not None and speaker not in speakers:
        raise ValueError(f'checkpoint has no speaker {speaker!r}, it has {sorted(speakers)}')
    row = state['speaker'] if speaker is None else speakers[speaker]
    shutil.copytree(base_package_dir, out_dir)
    bank_path = Path(out_dir) / 'bank.json'
    bank_text = bank_path.read_text(encoding='utf-8')
    component = bank_component(bank_text)
    base_model, _, _ = package_voice(base_package_dir)
    if file_digest(base_model) != state['base_sha256']:
        raise ValueError(f'{base_model.name} in the package is not the checkpoint base model')
    export(checkpoint, Path(out_dir) / base_model.name, base_model)
    np_idx = component['NPIdx']
    if np_idx & 0xFF != row:
        bank_path.write_text(re.sub(r'("NPIdx":\s*)\d+', lambda m: f'{m.group(1)}{np_idx & ~0xFF | row}', bank_text, count=1),
                             encoding='utf-8')
