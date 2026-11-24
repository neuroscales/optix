import math

import pytest
import torch

from optix.models import phantoms, random_shading

F64 = torch.float64


def _draws(seed, **kw):
    """One draw of every random phantom, as a dict."""
    return {
        "cells": phantoms.cells((2, 24, 20), seed=seed, **kw),
        "beads": phantoms.beads((2, 24, 20), 5, seed=seed, **kw)[0],
        "filaments": phantoms.filaments((2, 24, 20), 3, seed=seed, **kw),
        "stripes": phantoms.stripes((2, 24, 20), seed=seed, **kw),
        "background": phantoms.background((2, 24, 20), seed=seed, **kw),
    }


def test_backend(device, dtype):
    draws = _draws(0, dtype=dtype, device=device)
    draws["ramp"] = phantoms.ramp((2, 24, 20), dtype=dtype, device=device)
    for name, x in draws.items():
        assert x.shape == (2, 24, 20), name
        assert x.dtype == dtype and x.device.type == device.type, name
        assert torch.isfinite(x).all(), name


def test_seed_determinism_and_samples():
    a, b, c = _draws(1), _draws(1), _draws(2)
    for name in a:
        assert torch.equal(a[name], b[name]), name
        assert not torch.equal(a[name], c[name]), name
        # every leading index is an independent draw
        assert not torch.equal(a[name][0], a[name][1]), name


def test_same_phantom_on_every_backend():
    x32 = phantoms.cells((32, 32), seed=3, dtype=torch.float32)
    x64 = phantoms.cells((32, 32), seed=3, dtype=F64)
    torch.testing.assert_close(x32, x64.float(), atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("fn", ["cells", "filaments", "background"])
def test_3d(fn):
    x = getattr(phantoms, fn)((2, 12, 16, 20), ndim=3, seed=0)
    assert x.shape == (2, 12, 16, 20)
    assert (x >= 0).all() and x.amax() > 0
    with pytest.raises(ValueError):
        getattr(phantoms, fn)((16, 20), ndim=3)


def test_cells_values():
    x = phantoms.cells((64, 64), 8, intensity=100.0, variation=0.0, seed=0)
    assert x.min() >= 0 and x.max() <= 100.0
    assert x.max() > 99.0  # flat tops at the brightness
    assert 0.05 < (x > 50).double().mean() < 0.6  # cells and black
    assert (x < 1).double().mean() > 0.3
    assert torch.equal(phantoms.cells((8, 8), 0), torch.zeros(8, 8))
    with pytest.raises(ValueError):
        phantoms.cells((8, 8), edge=0.0)


def test_cells_radius():
    # one round cell: its area at half height is π a²
    x = phantoms.cells((128, 128), 1, radius=10.0, seed=4, dtype=F64)
    area = float((x > x.max() / 2).sum())
    a = math.sqrt(area / math.pi)
    assert 10.0 * math.exp(-0.6) < a < 10.0 * math.exp(0.6)


def test_beads_positions_match_peaks():
    sigma, brightness = 1.5, 1000.0
    x, pos = phantoms.beads(
        (96, 96), 12, sigma=sigma, brightness=brightness, separation=15.0,
        seed=0, dtype=F64,
    )  # fmt: skip
    assert pos.shape == (12, 2) and pos.dtype == F64
    assert (pos >= 4.5).all() and (pos <= 96 - 1 - 4.5).all()  # 3σ margin
    dist = torch.cdist(pos, pos) + 1e3 * torch.eye(12, dtype=F64)
    assert dist.min() >= 15.0
    torch.testing.assert_close(
        x.sum(), torch.tensor(12 * brightness, dtype=F64)
    )
    peak = brightness / (2 * math.pi * sigma**2)
    assert x.max() <= peak
    r = torch.arange(-5, 6, dtype=F64)
    for c in pos:
        i, j = (int(v) for v in c.round())
        window = x[i - 5 : i + 6, j - 5 : j + 6]
        assert window[5, 5] == window.max()  # peak at the bead
        w = window / window.sum()  # centroid at the sub-pixel position
        centroid = torch.stack([(w.sum(1) * r).sum(), (w.sum(0) * r).sum()])
        torch.testing.assert_close(centroid + c.round(), c, atol=1e-2, rtol=0)


def test_beads_3d_anisotropic():
    x, pos = phantoms.beads(
        (2, 24, 32, 32), 3, sigma=(2.0, 1.0, 1.0), ndim=3, seed=1, dtype=F64
    )
    assert x.shape == (2, 24, 32, 32) and pos.shape == (2, 3, 3)
    assert (pos[..., 0] >= 6.0).all() and (pos[..., 1:] >= 3.0).all()
    assert (pos[..., 0] <= 24 - 7.0).all() and (pos[..., 1:] <= 28.0).all()
    torch.testing.assert_close(
        x.sum((1, 2, 3)),
        torch.full((2,), 3000.0, dtype=F64),
        rtol=1e-2,
        atol=0,
    )


def test_beads_errors():
    with pytest.raises(ValueError, match="apart"):
        phantoms.beads((16, 16), 50, separation=8.0, seed=0)
    with pytest.raises(ValueError, match="margin"):
        phantoms.beads((8, 8), 1, sigma=3.0)
    with pytest.raises(ValueError):
        phantoms.beads((8, 8), 1, sigma=(1.0, 1.0, 1.0))


@pytest.mark.parametrize("ndim", [2, 3])
def test_filament_profile(ndim):
    # a straight filament: peak `intensity`, cross-section σ = `width`
    shape = (32,) * ndim
    x = phantoms.filaments(
        shape, 1, width=1.5, intensity=50.0, length=400.0, curvature=0.0,
        ndim=ndim, seed=2, dtype=F64,
    )  # fmt: skip
    assert 45.0 < x.max() < 55.0
    total = 50.0 * (2 * math.pi * 1.5**2) ** ((ndim - 1) / 2)
    # line density: at most `total` per pixel of length inside the field
    assert x.sum() <= total * 32 * math.sqrt(ndim) * 1.01


def test_filaments_curvature():
    straight = phantoms.filaments((64, 64), 1, curvature=0.0, seed=5)
    bent = phantoms.filaments((64, 64), 1, curvature=0.5, seed=5)
    assert not torch.equal(straight, bent)
    with pytest.raises(ValueError):
        phantoms.filaments((16, 16), width=0.0)


@pytest.mark.parametrize("axis", [-2, -1, 0])
def test_stripes_along_axis(axis):
    s = phantoms.stripes((3, 64, 48), axis=axis, strength=2.0, seed=0)
    spatial = axis % 2 + 1  # axis of the (64, 48) frames, in s
    first = s.narrow(spatial, 0, 1)
    assert torch.equal(s, first.expand_as(s))  # constant along `axis`
    assert 1.6 < float(first.std()) < 2.4
    with pytest.raises(ValueError):
        phantoms.stripes((8, 8), axis=2)


def test_stripes_fraction_and_3d():
    s = phantoms.stripes((64, 64), fraction=0.25, seed=1)
    lines = s[0]
    assert 0.1 < float((lines != 0).double().mean()) < 0.4
    assert torch.equal(
        phantoms.stripes((8, 8), fraction=0.0), torch.zeros(8, 8)
    )
    v = phantoms.stripes((10, 12, 14), axis=0, ndim=3, seed=2)
    assert torch.equal(v, v[:1].expand_as(v))
    assert not torch.equal(v[:, 0], v[:, 1])


def test_background_matches_random_shading():
    x = phantoms.background((5, 30, 20), level=40.0, seed=0, dtype=F64)
    assert (x > 0).all()
    torch.testing.assert_close(
        x.mean((1, 2)), torch.full((5,), 40.0, dtype=F64)
    )
    torch.testing.assert_close(x.log().std((1, 2)).mean(), x.new_tensor(0.3))
    one = phantoms.background((30, 20), level=1.0, variation=0.2, seed=7)
    flat = random_shading((30, 20), strength=0.2, seed=7).flat
    torch.testing.assert_close(one, flat)


def test_ramp():
    x = phantoms.ramp((4, 16, 32), low=20.0, high=3020.0, dtype=F64)
    assert x.shape == (4, 16, 32)
    assert float(x[..., 0].max()) == 20.0 and float(x[..., -1].min()) == 3020.0
    torch.testing.assert_close(
        x.diff(dim=-1), torch.full((4, 16, 31), 3000 / 31.0, dtype=F64)
    )
    assert torch.equal(x, x[:1, :1].expand_as(x))
    y = phantoms.ramp((8, 3), axis=0)
    assert torch.equal(y[:, 0], y[:, 2]) and y[0, 0] == 0 and y[-1, 0] == 1000
