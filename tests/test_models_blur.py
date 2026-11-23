import math

import pytest
import torch

from optix.deconv import RichardsonLucy
from optix.deconv.utils import PaddedConvolution
from optix.models import gaussian_psf, simulate_blur


@pytest.mark.parametrize("shape", [(15, 15), (7, 9, 11)])
def test_gaussian_psf_shape_norm_centre(shape):
    h = gaussian_psf(shape, 1.5, dtype=torch.float64)
    assert h.shape == shape and h.dtype == torch.float64
    torch.testing.assert_close(h.sum(), torch.tensor(1.0, dtype=h.dtype))
    peak = torch.unravel_index(h.argmax(), shape)
    assert tuple(int(i) for i in peak) == tuple(k // 2 for k in shape)
    torch.testing.assert_close(h, h.flip(tuple(range(h.ndim))))


def test_gaussian_psf_sigma_per_axis():
    h = gaussian_psf((41, 41), (1.0, 3.0), dtype=torch.float64)
    for axis, s in ((0, 1.0), (1, 3.0)):
        p = h.sum(1 - axis)
        r = torch.arange(41, dtype=h.dtype) - 20
        assert math.isclose(float((p * r**2).sum()) ** 0.5, s, rel_tol=1e-3)
    with pytest.raises(ValueError):
        gaussian_psf((5, 5), (1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        gaussian_psf((5, 5), 0.0)


def test_simulate_blur_matches_forward_operator():
    x = torch.rand(2, 24, 20, dtype=torch.float64)
    h = gaussian_psf((7, 5), (1.5, 1.0), dtype=torch.float64)
    op = PaddedConvolution(h, (24, 20))
    y = simulate_blur(x, h)
    assert y.shape == x.shape
    torch.testing.assert_close(y, op.forward(op.pad(x)))
    # Interior: direct linear convolution.
    ref = torch.nn.functional.conv2d(x[:, None], h.flip(0, 1)[None, None])
    torch.testing.assert_close(y[:, 3:-3, 2:-2], ref[:, 0])


def test_deconvolve_simulated_blur():
    torch.manual_seed(0)
    h = gaussian_psf((15, 15), 2.0)
    x = torch.full((64, 64), 5.0)
    x[16:48:8, 16:48] = 200.0
    y = simulate_blur(x, h)
    x_hat = RichardsonLucy(h, n_iter=50).fit(y).apply(y)
    assert (x_hat - x).abs().mean() < (y - x).abs().mean()
