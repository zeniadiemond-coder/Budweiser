import base64
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

HEADER_SIZE = 16
TRAILER_SIZE = 128
MASK32 = 0xFFFFFFFF
DTYPES = {'*F16': '<f2', '*F32': '<f4', '*S32': '<i4'}
NATIVE = {'*F16': np.float32, '*F32': np.float32, '*S32': np.int32}


def xor_stream(data, seed):
    x, y, z = 0x05491333, 0x159A55E5, 0x1F123BB5
    words, tail_length = divmod(len(data), 4)
    keys = np.empty(words, np.uint32)
    for i in range(words):
        mixed = ((seed << 11) & MASK32) ^ seed
        old_x = x
        x = old_x ^ mixed ^ (((old_x >> 11) ^ mixed) >> 8)
        keys[i] = x
        seed, y, z = y, z, old_x
    out = bytearray((np.frombuffer(data, '<u4', words) ^ keys).astype('<u4').tobytes())
    if tail_length:
        mixed = ((seed << 11) & MASK32) ^ seed
        key = (((x >> 11) ^ mixed) >> 8) ^ mixed ^ x ^ int.from_bytes(data[words * 4:], 'little')
        out += (key & ((1 << (tail_length * 8)) - 1)).to_bytes(tail_length, 'little')
    return bytes(out)


def strict_json(data):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON object key {key!r}')
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f'non-standard JSON constant {value}')

    return json.loads(data.decode('utf-8'), object_pairs_hook=unique_object, parse_constant=reject_constant)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_tag(value):
    return value[:4]


def read_tensor(value):
    return np.frombuffer(base64.b64decode(value[4:]), DTYPES[value[:4]]).astype(NATIVE[value[:4]])


def write_tensor(array, like):
    tag = tensor_tag(like)
    dtype = DTYPES[tag]
    array = np.asarray(array)
    if not np.isfinite(array).all():
        raise ValueError('non-finite tensor values')
    cast = array.astype(dtype)
    if not np.isfinite(cast).all():
        raise ValueError(f'tensor values overflow {tag}')
    return tag + base64.b64encode(cast.tobytes()).decode('ascii')


class Container:
    def __init__(self, header, document, trailer):
        self.header = header
        self.document = document
        self.trailer = trailer

    @classmethod
    def load(cls, path):
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def from_bytes(cls, raw):
        header, body, trailer = raw[:HEADER_SIZE], raw[HEADER_SIZE:-TRAILER_SIZE], raw[-TRAILER_SIZE:]
        seed = struct.unpack_from('<I', header, 12)[0]
        return cls(header, strict_json(xor_stream(body, seed)), trailer)

    def to_bytes(self):
        body = json.dumps(self.document).encode('utf-8')
        header = bytearray(self.header)
        struct.pack_into('<I', header, 4, HEADER_SIZE + len(body) + TRAILER_SIZE - 136)
        seed = struct.unpack_from('<I', header, 12)[0]
        return bytes(header) + xor_stream(body, seed) + self.trailer

    def save(self, path):
        Path(path).write_bytes(self.to_bytes())
