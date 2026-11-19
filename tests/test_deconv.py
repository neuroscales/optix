"""Tests of optix.deconv (Richardson–Lucy)."""

import os
import warnings

import numpy as np
import pytest
import torch

from optix.deconv import RichardsonLucy, richardson_lucy
from optix.deconv.utils import (
    PaddedConvolution,
    chi2_probability,
    next_fast_len,
    rl_block,
    rl_step,
)
from optix.estimates import SensorEstimate, ShadingEstimate
from optix.io import Layout, labeled

F64 = torch.float64
FIXTURES = os.path.join(
    os.path.dirname(__file__), "data", "skimage_rl_fixtures.npz"
)


def gaussian_psf(size, sigma, ndim=2, dtype=F64):
    x = torch.arange(size, dtype=dtype) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    out = g
    for _ in range(ndim - 1):
        out = out[..., None] * g
    return out / out.sum()


def scene(shape, *, seed=0, n=None, peak=400.0, base=5.0):
    """Sparse bright points on a flat base, blurred inside the image."""
    gen = torch.Generator().manual_seed(seed)
    numel = int(np.prod(shape))
    n = numel // 80 if n is None else n
    truth = torch.full(shape, base, dtype=F64)
    idx = torch.randint(0, numel, (n,), generator=gen)
    truth.view(-1)[idx] += peak
    return truth, gen


def blur(truth, psf):
    op = PaddedConvolution(psf, truth.shape)
    return op.forward(op.pad(truth)).clamp(min=0)


def poisson_image(shape, psf, seed=0):
    truth, gen = scene(shape, seed=seed)
    return truth, torch.poisson(blur(truth, psf), generator=gen)


def loglik(data, mu):
    return float((data * torch.log(mu) - mu).sum())


# -- building blocks --------------------------------------------------------
def test_next_fast_len():
    for n in range(1, 400):
        m = next_fast_len(n)
        assert m >= n
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        assert r == 1
        assert all(not _smooth(k) for k in range(n, m)), (
            f"{n}: {m} is not the smallest"
        )
    with pytest.raises(ValueError):
        next_fast_len(0)


def _smooth(k):
    for p in (2, 3, 5):
        while k % p == 0:
            k //= p
    return k == 1


@pytest.mark.parametrize("ndim", [2, 3])
def test_padded_convolution_is_linear_convolution(ndim):
    gen = torch.Generator().manual_seed(1)
    shape = (12, 10, 9)[:ndim]
    psf = torch.rand((5, 3, 5)[:ndim], generator=gen, dtype=F64)
    op = PaddedConvolution(psf, shape)
    assert all(
        g >= n + k - 1 and _smooth(g)
        for g, n, k in zip(op.grid, shape, psf.shape, strict=True)
    )
    f = torch.rand(shape, generator=gen, dtype=F64)
    # zero-boundary "same" convolution, PSF centred at K // 2
    conv = getattr(torch.nn.functional, f"conv{ndim}d")
    ref = conv(
        f[None, None],
        psf.flip(list(range(ndim)))[None, None],
        padding=[k // 2 for k in psf.shape],
    )[0, 0]
    torch.testing.assert_close(op.forward(op.pad(f)), ref)
    # adjoint
    g = torch.rand(op.grid, generator=gen, dtype=F64)
    r = torch.rand(shape, generator=gen, dtype=F64)
    lhs = (op.forward(g) * r).sum()
    rhs = (g * op.adjoint(r)).sum()
    torch.testing.assert_close(lhs, rhs)
    # normaliser: ∑S in the interior, positive on the margin it reaches
    sizes = zip(psf.shape, shape, strict=True)
    inner = tuple(slice(k, n - k) for k, n in sizes)
    torch.testing.assert_close(
        op.norm[inner], torch.full_like(op.norm[inner], float(psf.sum()))
    )
    torch.testing.assert_close(op.norm.sum(), psf.sum() * np.prod(shape))


def test_chi2_probability():
    mu = torch.full((10000,), 50.0, dtype=F64)
    y = torch.poisson(mu, generator=torch.Generator().manual_seed(0))
    assert 0.01 < chi2_probability(y, mu) < 0.99
    assert chi2_probability(y, mu * 1.5) < 1e-6


# -- reference implementation -----------------------------------------------
@pytest.mark.parametrize("n", [10, 30])
def test_matches_scikit_image(n):
    """scikit-image 0.26.0 (BSD-3), on objects away from the border."""
    ref = np.load(FIXTURES)
    data = torch.from_numpy(ref["image"])
    psf = torch.from_numpy(ref["psf"])
    out = richardson_lucy(data, psf, n_iter=n, dtype=F64)
    expected = torch.from_numpy(ref[f"rl{n}"])
    torch.testing.assert_close(out, expected, rtol=1e-8, atol=1e-8)


# -- paper properties -------------------------------------------------------
def test_flux_nonnegativity_and_likelihood():
    """Lucy (1974): constraints conserved, likelihood increases."""
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((48, 40), psf)
    op = PaddedConvolution(psf, data.shape)
    norm = op.norm
    inv = torch.where(norm > 1e-12, 1 / norm.clamp(min=1e-300), 0)
    f = torch.where(norm > 1e-12, data.mean(), 0.0)
    total = float(data.sum())
    prev = -np.inf
    for _ in range(25):
        f = rl_step(f, data, op, inv_norm=inv)
        mu = op.forward(f)
        assert (f >= 0).all()
        # ∑ᵢ (H f)ᵢ = ∑ᵢ yᵢ
        assert float(mu.sum()) == pytest.approx(total, rel=1e-10)
        ll = loglik(data, mu.clamp(min=1e-300))
        assert ll >= prev - 1e-8 * abs(ll)
        prev = ll


def test_start_level_does_not_matter():
    """The first iterate is the same from any uniform start."""
    psf = gaussian_psf(7, 1.2)
    _, data = poisson_image((32, 32), psf)
    op = PaddedConvolution(psf, data.shape)
    a, _ = rl_block(data, op, 1.0, n_iter=3)
    b, _ = rl_block(data, op, 123.0, n_iter=3)
    torch.testing.assert_close(a, b)


def test_noiseless_convergence():
    """Richardson (1972): without noise, the restoration tends to the truth."""
    psf = gaussian_psf(7, 1.5)
    truth = torch.zeros(40, 40, dtype=F64)
    truth[8:32, 8:32] = 10.0
    truth[14:18, 14:30] = 60.0
    data = blur(truth, psf)
    errs = [
        float((richardson_lucy(data, psf, n_iter=n) - truth).norm())
        for n in (1, 10, 100, 500)
    ]
    assert errs == sorted(errs, reverse=True)
    assert errs[-1] < 0.5 * errs[0]


def test_convergence_on_poisson_data():
    """A few iterations bring the estimate closer to the truth."""
    psf = gaussian_psf(9, 2.0)
    truth = torch.full((64, 64), 20.0, dtype=F64)
    truth[12:52, 12:52] = 100.0
    truth[20:28, 16:48] = 600.0
    truth[36:40, 24:44] = 0.0
    data = torch.poisson(
        blur(truth, psf), generator=torch.Generator().manual_seed(3)
    )
    inner = (slice(9, -9),) * 2
    errs = [
        float((richardson_lucy(data, psf, n_iter=n) - truth)[inner].norm())
        for n in (1, 5, 20, 50)
    ]
    assert errs == sorted(errs, reverse=True)
    assert errs[-1] < 0.7 * errs[0]
    assert errs[-1] < 0.8 * float((data - truth)[inner].norm())
    est = RichardsonLucy(psf, dtype=F64).fit(data)
    est.apply(data)
    n = est.history.extras["iterations"][0]
    assert 1 <= n < 100
    # Lucy's rule: the fit is acceptable where it stopped, not before
    op = PaddedConvolution(psf, data.shape)
    start = float(data.mean())
    f, _ = rl_block(data, op, start, n_iter=int(n))
    assert chi2_probability(data, op.forward(f)) >= 0.05
    f, _ = rl_block(data, op, start, n_iter=int(n) - 1)
    assert chi2_probability(data, op.forward(f)) < 0.05


# -- blocks -----------------------------------------------------------------
@pytest.mark.parametrize("n_iter", [10, 30])
def test_blocks_match_whole_image_2d(n_iter):
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((96, 80), psf, seed=1)
    whole = richardson_lucy(data, psf, n_iter=n_iter, block=(96, 80))
    parts = richardson_lucy(data, psf, n_iter=n_iter, block=(32, 24))
    err = float((parts - whole).abs().max() / whole.abs().max())
    assert err < 1e-3


def test_blocks_match_whole_image_3d():
    psf = gaussian_psf(5, 1.0, ndim=3)
    _, data = poisson_image((20, 24, 22), psf, seed=2)
    whole = richardson_lucy(data, psf, n_iter=10, block=(20, 24, 22))
    parts = richardson_lucy(data, psf, n_iter=10, block=(10, 12, 11))
    err = float((parts - whole).abs().max() / whole.abs().max())
    assert err < 1e-3
    # flux: ∑ H f = ∑ y on the whole image
    op = PaddedConvolution(psf, data.shape)
    f, _ = rl_block(data, op, float(data.mean()), n_iter=5)
    assert float(op.forward(f).sum()) == pytest.approx(
        float(data.sum()), rel=1e-10
    )


def test_halo_must_cover_psf():
    psf = gaussian_psf(9, 1.5)
    data = torch.ones(32, 32, dtype=F64)
    est = RichardsonLucy(psf, n_iter=1, halo=4).fit(data)
    with pytest.raises(ValueError, match="halo"):
        est.apply(data)


def test_stack_with_layout():
    """Sample axes are iterated: each image is deconvolved on its own."""
    psf = gaussian_psf(7, 1.2)
    imgs = [poisson_image((32, 28), psf, seed=s)[1] for s in range(3)]
    stack = torch.stack(imgs, dim=-1)  # (y, x, t)
    layout = Layout("yxt", spatial="yx")
    out = richardson_lucy(stack, psf, n_iter=5, layout=layout)
    for i, img in enumerate(imgs):
        torch.testing.assert_close(
            out[..., i], richardson_lucy(img, psf, n_iter=5)
        )


def test_lazy_zarr_input_and_output(tmp_path):
    zarr = pytest.importorskip("zarr")
    psf = gaussian_psf(7, 1.2)
    data = torch.stack(
        [poisson_image((64, 48), psf, seed=s)[1] for s in (4, 5)]
    )
    src = zarr.open_array(
        str(tmp_path / "in.zarr"),
        mode="w",
        shape=data.shape,
        chunks=(1, 16, 16),
        dtype="f4",
    )
    src[:] = data.numpy().astype("f4")
    dst = zarr.open_array(
        str(tmp_path / "out.zarr"),
        mode="w",
        shape=data.shape,
        chunks=(1, 16, 16),
        dtype="f4",
    )
    kw = {"n_iter": 8, "block": (32, 16), "dtype": torch.float32}
    richardson_lucy(src, psf, out=dst, **kw)
    ref = richardson_lucy(data.float(), psf, **kw)
    torch.testing.assert_close(torch.from_numpy(dst[:]), ref)


def test_float32_matches_float64():
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((48, 48), psf, seed=5)
    a = richardson_lucy(data, psf, n_iter=20, dtype=F64)
    b = richardson_lucy(data.float(), psf.float(), n_iter=20, dtype="float32")
    assert float((a - b).abs().max() / a.max()) < 1e-4


# -- extensions -------------------------------------------------------------
def test_faithful_flags():
    psf = gaussian_psf(5, 1.0)
    assert RichardsonLucy(psf).is_faithful
    assert RichardsonLucy(psf, n_iter=3).is_faithful is False
    assert RichardsonLucy(psf, background=1.0).is_faithful is False
    assert RichardsonLucy(psf, damping=0.5).is_faithful is False
    data = torch.ones(16, 16, dtype=F64)
    sensor = SensorEstimate(offset=torch.tensor(0.0), gain=torch.tensor(1.0))
    est = RichardsonLucy(psf).fit(data, init=sensor)
    assert est.is_faithful is False
    with pytest.raises(ValueError):
        RichardsonLucy(psf, damping=0.0)


def test_extensions_reduce_to_faithful():
    psf = gaussian_psf(7, 1.2)
    _, data = poisson_image((40, 36), psf, seed=6)
    ref = richardson_lucy(data, psf, n_iter=10)
    sensor = SensorEstimate(
        offset=torch.tensor(0.0),
        gain=torch.tensor(1.0),
        variance=torch.tensor(0.0),
    )
    for kw in (
        {"background": 0.0},
        {"damping": 1.0},
        {"sensor": sensor},
        {"sensor": sensor, "poisson_gaussian": True},
    ):
        out = richardson_lucy(data, psf, n_iter=10, **kw)
        torch.testing.assert_close(out, ref, msg=str(kw))


def test_sensor_converts_to_electrons():
    psf = gaussian_psf(7, 1.2)
    _, electrons = poisson_image((40, 36), psf, seed=7)
    gain = torch.linspace(1.5, 2.5, 36, dtype=F64).expand(40, 36)
    offset = torch.tensor(100.0, dtype=F64)
    adu = offset + gain * electrons
    sensor = SensorEstimate(offset=offset, gain=gain)
    out = richardson_lucy(adu, psf, n_iter=10, sensor=sensor, block=(20, 18))
    ref = richardson_lucy(electrons, psf, n_iter=10, block=(20, 18))
    torch.testing.assert_close(out, ref)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning):
            RichardsonLucy(psf, n_iter=1).fit(
                adu, init=ShadingEstimate(flat=torch.ones(40, 36))
            )


def test_background_improves_estimate():
    psf = gaussian_psf(9, 1.5)
    truth, gen = scene((64, 64), seed=8, base=0.0)
    b = 30.0
    data = torch.poisson(blur(truth, psf) + b, generator=gen)
    plain = richardson_lucy(data, psf, n_iter=30)
    with_b = richardson_lucy(data, psf, n_iter=30, background=b)
    assert float((with_b - truth).norm()) < float((plain - truth).norm())


def test_poisson_gaussian_shift():
    """y + σ² ~ Poisson(H f + σ²): the shifted data are used."""
    psf = gaussian_psf(7, 1.2)
    truth, gen = scene((48, 48), seed=9, base=2.0)
    var = 25.0  # electrons²
    data = torch.poisson(blur(truth, psf), generator=gen)
    data = data + var**0.5 * torch.randn(data.shape, generator=gen, dtype=F64)
    sensor = SensorEstimate(
        offset=torch.tensor(0.0),
        gain=torch.tensor(1.0),
        variance=torch.tensor(var),
    )
    out = richardson_lucy(
        data,
        psf,
        n_iter=10,
        sensor=sensor,
        poisson_gaussian=True,
        dtype=F64,
    )
    op = PaddedConvolution(psf, data.shape)
    f, _ = rl_block(
        (data + var).clamp(min=0),
        op,
        float(data.mean()),
        n_iter=10,
        background=var,
    )
    torch.testing.assert_close(out, op.crop(f))
    assert (out >= 0).all()
    with pytest.raises(ValueError, match="poisson_gaussian"):
        RichardsonLucy(psf, poisson_gaussian=True).fit(data)


def test_damping_keeps_flux_and_likelihood():
    """Lucy (1974, §III): a fraction ε of the correction still ascends."""
    psf = gaussian_psf(7, 1.2)
    _, data = poisson_image((40, 40), psf, seed=10)
    op = PaddedConvolution(psf, data.shape)
    inv = torch.where(op.norm > 1e-12, 1 / op.norm.clamp(min=1e-300), 0)
    f = torch.where(op.norm > 1e-12, data.mean(), 0.0)
    prev = -np.inf
    for _ in range(10):
        f = rl_step(f, data, op, inv_norm=inv, damping=0.5)
        mu = op.forward(f)
        assert float(mu.sum()) == pytest.approx(float(data.sum()), rel=1e-10)
        ll = loglik(data, mu.clamp(min=1e-300))
        assert ll >= prev
        prev = ll
    full = richardson_lucy(data, psf, n_iter=10)
    half = richardson_lucy(data, psf, n_iter=10, damping=0.5)
    assert not torch.allclose(full, half)


def test_save_load(tmp_path):
    from optix import load

    psf = gaussian_psf(5, 1.0)
    _, data = poisson_image((24, 24), psf, seed=11)
    est = RichardsonLucy(psf, n_iter=4).fit(data)
    est.save(tmp_path / "rl.pt")
    new = load(tmp_path / "rl.pt")
    torch.testing.assert_close(new.apply(data), est.apply(data))


def test_labeled_input():
    psf = gaussian_psf(5, 1.0)
    _, data = poisson_image((24, 20), psf, seed=12)
    out = richardson_lucy(
        labeled(data, Layout("yx", spatial="yx")), psf, n_iter=3
    )
    torch.testing.assert_close(out, richardson_lucy(data, psf, n_iter=3))
