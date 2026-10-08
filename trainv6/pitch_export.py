import re
import shutil
from pathlib import Path

import numpy as np
import torch

from . import nets
from .container import DTYPES, NATIVE, Container, file_digest, read_tensor, tensor_tag
from .export import bank_component, load_checkpoint, speaker_names
from .pitch_post import NET


def trained_stack(container, state):
    stack = nets.CausalStack(container.document[NET])
    for _ in range(state['w.embedding'].shape[0] - stack.rows):
        stack.add_speaker(torch.zeros(stack.w['embedding'].shape[1]))
    stack.load_state_dict(state)
    return stack


def export(checkpoint, output_vpit, name):
    state = load_checkpoint(checkpoint)
    base = Container.load(state['base_model'])
    container = Container.load(state['base_model'])
    stack = trained_stack(container, state['nets'][NET])
    stack.store(container.document[NET])
    speaker_names(container.document)[name] = state['speaker']
    container.save(output_vpit)
    verify_export(base, output_vpit, stack, state['speaker'], name)


def verify_export(base, exported_path, stack, speaker, name):
    exported = Container.load(exported_path)
    if exported.header[:4] + exported.header[8:] != base.header[:4] + base.header[8:] or exported.trailer != base.trailer:
        raise ValueError('exported header or trailer differs from the base')
    base_sites, sites = nets.causal_sites(base.document[NET]), nets.causal_sites(exported.document[NET])
    for key, site in sites.items():
        tag = tensor_tag(base_sites[key].holder[base_sites[key].key])
        expected = stack.w[key].detach().cpu().numpy().reshape(-1).astype(DTYPES[tag]).astype(NATIVE[tag])
        if not np.array_equal(read_tensor(site.holder[site.key]), expected):
            raise ValueError(f'{key} differs from the checkpoint weights')
        site.holder[site.key] = base_sites[key].holder[base_sites[key].key]
    parameters = exported.document[NET][nets.PARAMS]
    if parameters[nets.EMBEDDING_ROWS] != stack.rows or stack.rows != speaker + 1:
        raise ValueError('embedding row count does not match the new speaker row')
    parameters[nets.EMBEDDING_ROWS] = base.document[NET][nets.PARAMS][nets.EMBEDDING_ROWS]
    if speaker_names(exported.document).pop(name) != speaker:
        raise ValueError('speaker name does not map to the new row')
    if exported.document != base.document:
        raise ValueError('sections outside the pitch network differ from the base')


def export_package(checkpoint, base_package_dir, out_dir, character_name):
    state = load_checkpoint(checkpoint)
    shutil.copytree(base_package_dir, out_dir)
    bank_path = Path(out_dir) / 'bank.json'
    bank_text = bank_path.read_text(encoding='utf-8')
    component = bank_component(bank_text)
    model = Path(out_dir) / f"{component['PModelName']}.vpit"
    if file_digest(Path(base_package_dir) / model.name) != state['base_sha256']:
        raise ValueError(f'{model.name} in the package is not the checkpoint base model')
    export(checkpoint, model, character_name)
    np_idx = component['NPIdx'] & 0xFF | state['speaker'] << 8
    bank_path.write_text(re.sub(r'("NPIdx":\s*)\d+', lambda m: f'{m.group(1)}{np_idx}', bank_text, count=1), encoding='utf-8')
    (Path(out_dir) / 'character.txt').write_text(f'name={character_name}\n', encoding='utf-8')
    yaml = Path(out_dir) / 'character.yaml'
    yaml.write_text(re.sub(r'(?m)^name: .*$', f'name: {character_name}', yaml.read_text(encoding='utf-8')), encoding='utf-8')
