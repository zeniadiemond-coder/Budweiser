import numpy as np
import torch

from trainv6 import ddsp
from trainv6.reference import decoder


def reference_suppressed(curve):
    width = len(curve)
    padded = np.zeros(width + 3, np.float32)
    padded[1:width + 1] = curve
    padded[0] = padded[1]
    padded[width + 1] = padded[width + 2] = padded[width]
    decoder.suppress_peaks(padded)
    return padded[1:width + 1]


def spiky_curve(rng, width=64):
    base = rng.normal(0, 1, width).cumsum() * 2 - 40
    spikes = rng.random(width) < 0.15
    return np.clip(base + spikes * rng.uniform(0, 14, width), -139, 19).astype(np.float32)


def test_suppress_peaks_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(200):
        curve = spiky_curve(rng)
        expected = reference_suppressed(curve)
        actual = ddsp.suppress_peaks(torch.from_numpy(curve).double()[None])[0].numpy()
        assert not np.allclose(curve, expected)
        np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_suppress_peaks_is_differentiable():
    curve = torch.from_numpy(spiky_curve(np.random.default_rng(1)))[None].requires_grad_()
    ddsp.suppress_peaks(curve).sum().backward()
    assert torch.isfinite(curve.grad).all() and curve.grad.abs().sum() > 0
