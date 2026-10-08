import numpy as np

f32 = np.float32


def native_exp(x):
    x = np.asarray(x, f32)
    p = x * f32(1.442695)
    offset = (p < 0).astype(f32)
    q = np.maximum(p, f32(-126.0))
    z = (q - np.trunc(q).astype(f32)) + offset
    v = ((f32(27.728024) / (f32(4.8425255) - z) + q + f32(121.274055)) - z * f32(1.4901291)) * f32(8388608.0)
    return (v.astype(np.int64) & 0xffffffff).astype(np.uint32).view(np.float32)
