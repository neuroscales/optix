"""Tests of optix.psf bead detection, alignment and averaging."""

import math

import numpy as np
import pytest
import torch

from optix.estimates import PSFEstimate
from optix.io import labeled
from optix.psf import BeadPSF, bead_psf
from optix.psf.utils import (
    fit_gaussian,
    fourier_shift,
    fwhm,
    gaussian_filter,
    linear_shift,
    local_maxima,
    threshold_centroid,
    tukey_window,
)

F64 = torch.float64
FWHM = 2 * math.sqrt(2 * math.log(2))


def gaussian(shape, sigma, center, dtype=F64):
    out = torch.ones(shape, dtype=dtype)
    for i, (n, s, c) in enumerate(zip(shape, sigma, center, strict=True)):
        view = [1] * len(shape)
        view[i] = n
        g = torch.arange(n, dtype=dtype) - c
        out = out * torch.exp(-0.5 * (g / s) ** 2).view(view)
    return out


def airy(shape, scale, center, dtype=F64):
    """2-D Airy pattern (2 J₁(v) / v)², v = ρ / scale."""
    y = torch.arange(shape[0], dtype=dtype)[:, None] - center[0]
    x = torch.arange(shape[1], dtype=dtype)[None, :] - center[1]
    v = torch.sqrt(y**2 + x**2) / scale
    vs = torch.where(v == 0, torch.ones_like(v), v)
    a = torch.where(
        v == 0, torch.ones_like(v), 2 * torch.special.bessel_j1(vs) / vs
    )
    return a**2


def jittered_centers(shape, spacing, margin, gen):
    """One random sub-voxel centre per cell of a grid (beads isolated)."""
    axes = [
        torch.arange(margin, n - margin, s, dtype=F64)
        for n, s in zip(shape, spacing, strict=True)
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(
        -1, len(shape)
    )
    jitter = torch.rand(grid.shape, generator=gen, dtype=F64) - 0.5
    return grid + jitter * torch.tensor([s / 3 for s in spacing], dtype=F64)


def bead_stack(shape, model, centers, *, photons, background, gen):
    expected = torch.full(shape, background, dtype=F64)
    for c in centers.tolist():
        expected += photons * model(shape, c)
    return torch.poisson(expected, generator=gen)


def match(found, truth):
    d = torch.cdist(found, truth)
    return d.min(1).values


def rel_l2(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


SIGMA3 = (2.2, 1.3, 1.3)
RADIUS3 = (8, 5, 5)


@pytest.fixture(scope="module")
def beads3d():
    gen = torch.Generator().manual_seed(0)
    shape = (40, 72, 72)
    centers = jittered_centers(shape, (20, 14, 14), 10, gen)
    data = bead_stack(
        shape,
        lambda s, c: gaussian(s, SIGMA3, c),
        centers,
        photons=3000.0,
        background=20.0,
        gen=gen,
    )
    truth = gaussian(tuple(2 * r + 1 for r in RADIUS3), SIGMA3, RADIUS3)
    return data, centers, truth / truth.sum()


def test_gaussian_beads_recovered(beads3d):
    data, centers, truth = beads3d
    est = BeadPSF(radius=RADIUS3).fit(data)
    e = est.estimates["psf"]
    assert isinstance(e, PSFEstimate)
    assert e.psf.shape == truth.shape
    assert float(e.psf.sum()) == pytest.approx(1.0, abs=1e-5)
    assert int(e.count) == len(centers)
    assert rel_l2(e.psf, truth) < 0.05
    expected = torch.tensor([FWHM * s for s in SIGMA3])
    assert torch.allclose(e.fwhm.double(), expected.double(), rtol=0.05)
    # sub-voxel centres
    err = match(e.centers, centers)
    assert float(err.max()) < 0.15
    assert float(err.pow(2).mean().sqrt()) < 0.06


def test_centroid_and_linear_options(beads3d):
    data, centers, truth = beads3d
    e = BeadPSF(
        radius=RADIUS3, center="centroid", shift="linear", weights="brightness"
    ).fit(data)
    e = e.estimates["psf"]
    assert int(e.count) == len(centers)
    assert float(match(e.centers, centers).max()) < 0.3
    # linear interpolation smooths, so the tolerance is looser
    assert rel_l2(e.psf, truth) < 0.15


def test_float32_matches_float64(beads3d):
    data, _, _ = beads3d
    a = bead_psf(data, radius=RADIUS3, dtype=torch.float32)
    b = bead_psf(data, radius=RADIUS3, dtype=F64)
    assert a.dtype == torch.float32
    assert torch.allclose(a.double(), b, atol=1e-5 * float(b.max()))


def test_airy_2d_beads():
    gen = torch.Generator().manual_seed(1)
    shape = (160, 160)
    scale = 1.2
    centers = jittered_centers(shape, (20, 20), 12, gen)
    data = bead_stack(
        shape,
        lambda s, c: airy(s, scale, c),
        centers,
        photons=5000.0,
        background=10.0,
        gen=gen,
    )
    e = BeadPSF(radius=7).fit(data).estimates["psf"]
    truth = airy((15, 15), scale, (7, 7))
    truth = truth / truth.sum()
    assert int(e.count) == len(centers)
    assert rel_l2(e.psf, truth) < 0.05
    assert float(match(e.centers, centers).max()) < 0.1


def test_lazy_chunked_input(beads3d, tmp_path):
    zarr = pytest.importorskip("zarr")
    data, centers, _ = beads3d
    store = zarr.open_array(
        str(tmp_path / "beads.zarr"),
        mode="w",
        shape=tuple(data.shape),
        chunks=(16, 24, 24),
        dtype="uint16",
    )
    store[:] = data.numpy().astype(np.uint16)
    whole = BeadPSF(radius=RADIUS3, dtype=F64).fit(data).estimates["psf"]
    lazy = BeadPSF(radius=RADIUS3, block=(16, 24, 24), dtype=F64)
    lazy = lazy.fit(store).estimates["psf"]
    assert int(lazy.count) == len(centers)
    assert torch.allclose(lazy.psf, whole.psf, atol=1e-10)
    order = lambda c: c[torch.argsort(c[:, 0] * 1e6 + c[:, 1] * 1e3 + c[:, 2])]  # noqa: E731
    assert torch.allclose(order(lazy.centers), order(whole.centers), atol=1e-8)


def test_rejects_overlapping_saturated_and_edge_beads():
    shape = (64, 128)
    sigma = (1.5, 1.5)
    model = torch.full(shape, 5.0, dtype=F64)
    good = [(20.3, 20.6), (20.0, 60.2), (45.4, 30.1)]
    for c in good:
        model += 1000 * gaussian(shape, sigma, c)
    pair = [(45.0, 80.0), (47.0, 84.0)]  # overlapping
    for c in pair:
        model += 1000 * gaussian(shape, sigma, c)
    model += 1000 * gaussian(shape, sigma, (30.0, 2.0))  # at the edge
    model += 1e5 * gaussian(shape, sigma, (20.0, 105.0))  # saturates
    gen = torch.Generator().manual_seed(3)
    counts = torch.poisson(model, generator=gen).clamp(max=4095)
    data = counts.numpy().astype(np.uint16)
    e = BeadPSF(radius=6, saturation=4095).fit(data).estimates["psf"]
    assert int(e.count) == 3
    err = match(e.centers, torch.tensor(good, dtype=F64))
    assert float(err.max()) < 0.2
    # the default saturation of integer data is the type maximum
    clipped = np.minimum(counts.numpy(), 255).astype(np.uint8)
    with pytest.raises(ValueError, match="No isolated beads"):
        BeadPSF(radius=6).fit(clipped)


def test_layout_groups_and_axis_order():
    gen = torch.Generator().manual_seed(4)
    spatial = (30, 60, 50)
    sigmas = [(2.0, 1.2, 1.0), (2.6, 1.5, 1.2)]
    centers = jittered_centers(spatial, (15, 15, 15), 8, gen)
    channels = [
        bead_stack(
            spatial,
            lambda s, c, sg=sg: gaussian(s, sg, c),
            centers,
            photons=3000.0,
            background=10.0,
            gen=gen,
        )
        for sg in sigmas
    ]
    zyx = torch.stack(channels)  # (c, z, y, x)
    stored = zyx.permute(3, 0, 2, 1).contiguous()  # (x, c, y, z)
    data = labeled(stored, axes="xcyz", spatial="zyx", group="c")
    e = BeadPSF(radius=(7, 6, 5)).fit(data).estimates["psf"]
    assert e.psf.shape == (2, 15, 13, 11)
    assert e.fwhm.shape == (2, 3)
    assert e.count.tolist() == [len(centers)] * 2
    for g, sg in enumerate(sigmas):
        truth = gaussian((15, 13, 11), sg, (7, 6, 5))
        assert rel_l2(e.psf[g], truth / truth.sum()) < 0.06
    # centres are in storage order (x, c, y, z)
    first = e.centers[e.centers[:, 1] == 0]
    assert first.shape[0] == len(centers)
    zyx_found = first[:, [3, 2, 0]]
    assert float(match(zyx_found, centers).max()) < 0.2


def test_symmetrize_and_taper():
    gen = torch.Generator().manual_seed(5)
    shape = (120, 120)
    centers = jittered_centers(shape, (20, 20), 12, gen)
    data = bead_stack(
        shape,
        lambda s, c: gaussian(s, (1.5, 1.5), c),
        centers,
        photons=2000.0,
        background=5.0,
        gen=gen,
    )
    psf = bead_psf(data, radius=6, symmetrize=(0, 1), taper=0.5)
    assert torch.allclose(psf, psf.flip(0))
    assert torch.allclose(psf, psf.flip(1))
    assert float(psf[0].abs().max()) == 0.0


def test_no_beads_raises():
    with pytest.raises(ValueError, match="No isolated beads"):
        BeadPSF(radius=4).fit(torch.zeros(32, 32))


def test_apply_is_not_defined(beads3d):
    est = BeadPSF(radius=RADIUS3).fit(beads3d[0])
    with pytest.raises(NotImplementedError):
        est.apply(beads3d[0])


def test_estimate_roundtrip(beads3d, tmp_path):
    e = BeadPSF(radius=RADIUS3).fit(beads3d[0]).estimates["psf"]
    path = tmp_path / "psf.pt"
    e.save(path)
    back = PSFEstimate.load(path)
    assert back.allclose(e)
    assert back.meta["estimator"] == "BeadPSF"


# -- building blocks -----------------------------------------------------


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_fourier_shift_sub_voxel(ndim):
    shape = (21,) * ndim
    c = (10.0,) * ndim
    sigma = (1.7,) * ndim
    x = gaussian(shape, sigma, c)[None]
    d = torch.tensor([[0.37, -0.81, 0.25][:ndim]], dtype=F64)
    out = fourier_shift(x, d)[0]
    expected = gaussian(
        shape,
        sigma,
        [ci + di for ci, di in zip(c, d[0].tolist(), strict=True)],
    )
    # a sampled Gaussian is band-limited up to ~1e-7 of its peak
    assert torch.allclose(out, expected, atol=1e-6)


def test_linear_shift_integer_and_fraction():
    x = torch.arange(10, dtype=F64)[None]
    assert torch.allclose(
        linear_shift(x, torch.tensor([[2.0]], dtype=F64))[0, 2:], x[0, :-2]
    )
    out = linear_shift(x, torch.tensor([[0.25]], dtype=F64))[0]
    assert torch.allclose(out[1:], x[0, 1:] - 0.25)
    # a smooth peak moves by the shift
    g = gaussian((31, 31), (3.0, 3.0), (15.0, 15.0))[None]
    out = linear_shift(g, torch.tensor([[0.4, -0.3]], dtype=F64))
    c = threshold_centroid(out, level=0.0)[0]
    assert torch.allclose(c, torch.tensor([15.4, 14.7], dtype=F64), atol=1e-3)


def test_fit_gaussian_sub_voxel(dtype):
    truth_c = torch.tensor([[5.31, 6.77, 4.92], [6.4, 5.1, 5.55]], dtype=F64)
    sigma = torch.tensor([[1.8, 1.2, 1.4], [2.1, 1.0, 1.3]], dtype=F64)
    boxes = torch.stack(
        [
            300 * gaussian((11, 13, 11), s.tolist(), c.tolist()) + 7
            for c, s in zip(truth_c, sigma, strict=True)
        ]
    ).to(dtype)
    c, s, a, b = fit_gaussian(boxes)
    tol = 1e-6 if dtype == F64 else 1e-3
    assert torch.allclose(c.double(), truth_c, atol=tol)
    assert torch.allclose(s.double(), sigma, atol=tol)
    assert torch.allclose(
        a.double(), torch.full((2,), 300.0, dtype=F64), rtol=tol
    )
    assert torch.allclose(
        b.double(), torch.full((2,), 7.0, dtype=F64), rtol=10 * tol
    )


def test_threshold_centroid_symmetric_peak():
    c = (8.3, 7.6)
    box = gaussian((17, 17), (2.0, 2.0), c)[None]
    found = threshold_centroid(box)[0]
    assert torch.allclose(found, torch.tensor(c, dtype=F64), atol=0.02)


def test_fwhm_of_gaussian():
    s = (2.0, 3.0)
    g = gaussian((41, 41), s, (20, 20))
    assert torch.allclose(
        # linear interpolation between samples: about 1 % wide at σ = 2
        fwhm(g),
        torch.tensor([FWHM * v for v in s], dtype=F64),
        rtol=0.015,
    )
    assert math.isnan(float(fwhm(torch.ones(5))[0]))


def test_smoothing_and_maxima():
    x = torch.zeros(20, 20, dtype=F64)
    x[5, 6] = 1.0
    x[14, 12] = 2.0
    sm = gaussian_filter(x, 1.0)
    assert float(sm.sum()) == pytest.approx(3.0)
    peaks = local_maxima(sm, 2, 0.01)
    assert sorted(map(tuple, peaks.tolist())) == [(5, 6), (14, 12)]
    w = tukey_window(11, 0.5, dtype=F64)
    assert float(w[0]) == 0.0 and float(w[5]) == 1.0
    assert torch.allclose(w, w.flip(0))
    assert torch.equal(tukey_window(4, 0.0), torch.ones(4))
