import re
from collections import namedtuple
from pathlib import Path

from . import frontend

SILENCES = {'pau', 'sil', 'br'}
CANONICAL = {'A': 'a', 'I': 'i', 'U': 'u', 'E': 'e', 'O': 'o', 'GlottalStop': 'vf', 'Edge': 'vf', 'pau': 'SP', 'sil': 'SP'}
NUCLEI = {'a', 'i', 'u', 'e', 'o', 'N'}
SECTION_SILENCE = 0.12
SECTION_PAD = 0.06
LONG_REST = 1.0
BREATH = 0.6
TICKS_PER_BEAT = 480

Timed = namedtuple('Timed', 'start end name pitch note_start nucleus cls note')


def read_ust(path):
    raw = Path(path).read_bytes()
    try:
        text = raw.decode('shift_jis')
    except UnicodeDecodeError:
        text = raw.decode('utf-8', 'replace')
    parts = re.split(r'\[#(\w+)\]', text)
    sections = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        sections[name] = dict(line.split('=', 1) for line in body.splitlines() if '=' in line)
    return sections


def ust_notes(path):
    sections = read_ust(path)
    tempo = float(sections['SETTING']['Tempo'])
    now, notes = 0.0, []
    for name, fields in sections.items():
        if not name.isdigit():
            continue
        if fields.get('Tempo'):
            tempo = float(fields['Tempo'])
        duration = int(fields['Length']) * 60.0 / (tempo * TICKS_PER_BEAT)
        lyric = fields.get('Lyric', '')
        notes.append(dict(start=now, duration=duration, ticks=int(fields['Length']), lyric=lyric, midi=int(fields.get('NoteNum', 60)), rest=lyric in ('R', 'r', '')))
        now += duration
    return notes


def kana_table(path):
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    return {fields[0]: fields[1:] for fields in (line.split() for line in lines) if len(fields) >= 2}


def expand(table, lyric):
    if lyric in table:
        return table[lyric]
    if lyric.endswith('っ') and lyric[:-1] in table:
        return table[lyric[:-1]] + ['cl']
    raise ValueError(f'lyric {lyric!r} is not in the kana table')


def note_owners(table, notes, labels):
    owners, at = {}, 0
    for i, note in enumerate(notes):
        if note['rest']:
            continue
        expected = expand(table, note['lyric'])
        while at < len(labels) and labels[at][2] in SILENCES:
            at += 1
        found = list(range(at, min(at + len(expected), len(labels))))
        if [labels[j][2] for j in found] != expected:
            raise ValueError(f'note {i} {note["lyric"]!r}: labels {[labels[j][2] for j in found]} differ from {expected}')
        owners.update((j, i) for j in found)
        at = found[-1] + 1
    return owners


def sung_phonemes(labels, owners, notes):
    canonical = [(s, e, CANONICAL.get(n, n)) for s, e, n in labels]
    index_of_end = {e: i for i, (_, e, n) in enumerate(canonical) if n != 'vf'}
    model = frontend.model_phonemes(canonical)
    source = [index_of_end[p.end] for p in model]
    result, v6_notes, k = [], [], 0
    while k < len(model):
        if source[k] not in owners:
            j = k
            while j < len(model) and source[j] not in owners:
                j += 1
            result.append(Timed(model[k].start, model[j - 1].end, 'sil', None, True, True, 0, len(v6_notes)))
            v6_notes.append(dict(start=model[k].start, end=model[j - 1].end, pitch=None))
            k = j
            continue
        owner = owners[source[k]]
        j = k
        while j < len(model) and owners.get(source[j]) == owner:
            j += 1
        group = model[k:j]
        cents = (notes[owner]['midi'] - 69) * 100.0
        nucleus = max((m for m, p in enumerate(group) if p.name in NUCLEI), default=0)
        v6_notes.append(dict(start=group[nucleus].start, end=group[-1].end, pitch=cents, stop=group[nucleus].end))
        result += [Timed(p.start, p.end, p.name, cents, m == 0, m == nucleus, 0, len(v6_notes) - 1) for m, p in enumerate(group)]
        k = j
    first = next(p.pitch for p in result if p.pitch is not None)
    pitches = []
    for p in result:
        first = p.pitch if p.pitch is not None else first
        pitches.append(first)
    return [p._replace(pitch=c) for p, c in zip(result, pitches)], v6_notes


def vtmg_silences(timed):
    result = []
    for i, p in enumerate(timed):
        p = p._replace(cls=-1) if i == 0 else p
        if p.name != 'sil' or i == len(timed) - 1:
            result.append(p)
        elif p.end - p.start >= LONG_REST:
            result += [p._replace(end=p.end - BREATH), p._replace(start=p.end - BREATH, name='br')]
        else:
            result.append(p._replace(name='br'))
    result[-1] = result[-1]._replace(cls=-1)
    return result


def timed_score(song, scores, table):
    directory = Path(scores) / song
    notes = ust_notes(directory / f'{song}.ust')
    labels = frontend.read_lab(directory / f'{song}.lab')
    timed, v6_notes = sung_phonemes(labels, note_owners(table, notes, labels), notes)
    return vtmg_silences(timed), v6_notes


def sections(timed):
    result, current, origin = [], [], 0.0
    for i, p in enumerate(timed):
        if p.name != 'sil' or p.end - p.start <= SECTION_SILENCE:
            current.append(p._replace(start=max(p.start - origin, SECTION_PAD), end=p.end - origin))
            continue
        if i != 0:
            current.append(p._replace(start=p.start - origin, end=p.start - origin + SECTION_PAD))
            result.append((origin, current))
            current = []
        if i < len(timed) - 1:
            origin = p.end - SECTION_PAD
            current.append(p._replace(start=0.0, end=SECTION_PAD))
    if current:
        result.append((origin, current))
    return result
