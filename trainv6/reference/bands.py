import base64
import math

import numpy as np

f32 = np.float32
FS = f32(32000.0)
ANALYSES = 'a6d8c9aaef6372d25b26a8d1924776aa'
RANGES = '7a90c360132d8ebf6f4d4a6f7a002ba6'
BANDS = '81580c0610f9cab4a99e1254028175e6'
LOW = '597abe5c91fad4df0ac43693d3c1dffe'
HIGH = '11752ac1d5abcf1d58c585e0e02aee8c'
WARP = 'de62ed5a302c6d0248b600effc6919eb'
WINDOW = '23d0c1bc1cc7a4017306e8ec01cc5286'
HANN = 'b8421b2df71c2003122543a863983ff6'


def tensor(value):
    assert value[:4] == '*F32'
    return np.frombuffer(base64.b64decode(value[4:]), '<f4').copy()


def edges(config):
    bands, low, high, alpha = config[BANDS], f32(config[LOW]), f32(config[HIGH]), config[WARP]
    if alpha is None:
        def forward(f):
            return f / f32(66.666664) if f < 1000 else np.log(f / f32(1000.0)) / (np.log(f32(6.4)) / f32(27.0)) + f32(15.000001)

        def inverse(m):
            return m * f32(66.666664) if m < f32(15.000001) else np.exp(np.log(f32(6.4)) / f32(27.0) * (m - f32(15.000001))) * f32(1000.0)
    else:
        a = f32(alpha)

        def forward(f):
            w = f * f32(6.2831855) / FS
            t = f32(math.atan(math.sin(w) * a / (1 - math.cos(w) * a)))
            return (t + t + w) / f32(3.1415927)

        def inverse(m):
            w = m * f32(3.1415927)
            t = f32(math.atan(math.sin(w) * -a / (1 - math.cos(w) * -a)))
            return (t + t + w) * FS * f32(0.5) / f32(3.1415927)
    lo, hi = f32(forward(low)), f32(forward(high))
    return np.array([inverse(f32(i * (hi - lo) / f32(bands + 1) + lo)) for i in range(bands + 2)], f32)


def hann_window(doc):
    return doc[ANALYSES][0][WINDOW] == HANN


def analysis_tables(doc):
    edges16, edges80 = (edges(c) for c in doc[ANALYSES])
    minima, maxima = ([tensor(v) for v in group] for group in doc[RANGES])
    return edges16, edges80, minima, maxima
