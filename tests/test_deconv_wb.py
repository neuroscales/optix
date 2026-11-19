"""Tests of the unmatched back projectors (Guo et al. 2020)."""

import math

import pytest
import torch

from optix.deconv import (
    RichardsonLucy,
    WienerButterworth,
    richardson_lucy,
    wiener_butterworth,
)
from optix.deconv.utils import (
    BackProjection,
    PaddedConvolution,
    back_projector,
    psf_fwhm,
    rl_block,
    rl_step,
)

F64 = torch.float64


def gaussian_psf(size, sigma, ndim=2, dtype=F64):
    x = torch.arange(size, dtype=dtype) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    out = g
    for _ in range(ndim - 1):
        out = out[..., None] * g
    return out / out.sum()


def beads(shape, psf, *, step, peak=20000.0, base=1.0, seed=1):
    """Isolated point sources on a grid, blurred, Poisson noise."""
    truth = torch.full(shape, base, dtype=F64)
    ranges = [range(step // 2 + 4, n - 4, step) for n in shape]
    pos = [tuple(p) for p in torch.cartesian_prod(*map(torch.tensor, ranges))]
    if len(shape) == 1:
        pos = [(int(p),) for p in pos]
    for p in pos:
        truth[tuple(int(c) for c in p)] += peak
    op = PaddedConvolution(psf, shape)
    gen = torch.Generator().manual_seed(seed)
    data = torch.poisson(op.forward(op.pad(truth)).clamp(min=0), generator=gen)
    return data, [tuple(int(c) for c in p) for p in pos]


def mean_width(img, pos, radius=10):
    """Mean FWHM along the first axis of the beads at `pos`."""
    widths = []
    for p in pos:
        crop = img[tuple(slice(max(c - radius, 0), c + radius + 1) for c in p)]
        widths.append(psf_fwhm(crop)[0])
    return sum(widths) / len(widths)


def flipped(psf):
    return psf.flip(list(range(psf.ndim)))


# -- filters ----------------------------------------------------------------
def test_psf_fwhm():
    sigma = 3.0
    w = psf_fwhm(gaussian_psf(31, sigma, ndim=3))
    expected = 2 * math.sqrt(2 * math.log(2)) * sigma
    assert w == pytest.approx((expected,) * 3, rel=5e-3)
    with pytest.raises(ValueError, match="half"):
        psf_fwhm(torch.ones(5, 5, dtype=F64))


def _spectrum(b):
    centre = [k // 2 for k in b.shape]
    return torch.fft.fftn(
        torch.roll(b, [-c for c in centre], tuple(range(b.ndim)))
    )


def _radius2(shape, fwhm):
    grids = torch.meshgrid(
        *(
            torch.fft.fftfreq(n, dtype=F64) * w
            for n, w in zip(shape, fwhm, strict=True)
        ),
        indexing="ij",
    )
    return sum(g.square() for g in grids)


@pytest.mark.parametrize("ndim", [2, 3])
def test_frequency_responses_match_supplementary_note_2(ndim):
    """Eqs. 13, 18, 21, 23, 25 and 27, sampled on the PSF grid."""
    shape = (32, 24, 20)[:ndim]
    fwhm = (4.0, 6.0, 5.0)[:ndim]  # cutoffs 1/4, 1/6, 1/5: grid frequencies
    gen = torch.Generator().manual_seed(0)
    psf = gaussian_psf(32, 1.7, ndim)[tuple(slice(0, n) for n in shape)]
    psf = psf * (1 + 0.1 * torch.rand(shape, generator=gen, dtype=F64))
    psf = psf / psf.sum()
    otf = _spectrum(psf)
    alpha, beta, n = 0.01, 0.02, 6
    eps2 = 1 / beta**2 - 1
    butter = 1 / torch.sqrt(1 + eps2 * _radius2(shape, fwhm) ** n)
    kw = {"wiener": alpha, "cutoff_gain": beta, "order": n, "fwhm": fwhm}
    # traditional: conj(OTF)
    b = back_projector(psf, "traditional")
    torch.testing.assert_close(_spectrum(b), otf.conj())
    # Butterworth, gain β at each cutoff (Eq. 23)
    b = back_projector(psf, "butterworth", **kw)
    torch.testing.assert_close(_spectrum(b), butter.to(otf.dtype))
    for axis, w in enumerate(fwhm):
        index = [0] * ndim
        index[axis] = round(shape[axis] / w)
        assert float(_spectrum(b)[tuple(index)].real) == pytest.approx(beta)
    # Wiener–Butterworth, normalised to unit DC gain (Eq. 25, Note 1)
    b = back_projector(psf, "wiener-butterworth", **kw)
    wiener = otf.conj() / (otf.abs().square() + alpha)
    wb = wiener * butter * (1 + alpha)
    torch.testing.assert_close(_spectrum(b), wb, atol=1e-10, rtol=1e-8)
    index = (round(shape[0] / fwhm[0]),) + (0,) * (ndim - 1)
    torch.testing.assert_close(  # Eq. 27
        _spectrum(b)[index], wiener[index] * beta * (1 + alpha)
    )
    assert float(b.sum()) == pytest.approx(1.0)
    # Gaussian with the FWHM of the PSF (Eq. 18), unit sum
    b = back_projector(psf, "gaussian", fwhm=fwhm)
    assert float(b.sum()) == pytest.approx(1.0)
    assert psf_fwhm(b) == pytest.approx(fwhm, rel=2e-2)


def test_wb_flattens_the_spectral_product():
    """Supplementary Note 1: |DFT(b) DFT(f)| flatter than |OTF|²."""
    psf = gaussian_psf(41, 3.0)
    otf = _spectrum(psf)
    trad = (_spectrum(back_projector(psf, "traditional")) * otf).abs()
    wb = (_spectrum(back_projector(psf, "wiener-butterworth")) * otf).abs()
    k = 2  # a low, passed frequency
    assert float(wb[k, 0]) > float(trad[k, 0]) + 0.2
    assert float(wb[k, 0]) == pytest.approx(1.0, abs=0.1)


def test_back_projector_errors():
    psf = gaussian_psf(9, 1.5)
    with pytest.raises(ValueError, match="kind"):
        back_projector(psf, "delta")
    with pytest.raises(ValueError, match="cutoff_gain"):
        back_projector(psf, "butterworth", cutoff_gain=1.0)
    with pytest.raises(ValueError, match="wiener"):
        back_projector(psf, wiener=0.0)
    with pytest.raises(ValueError, match="fwhm"):
        back_projector(psf, fwhm=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        WienerButterworth(psf, back_projector="delta")
    with pytest.raises(ValueError):
        WienerButterworth(psf, start="zeros")


# -- reduction to Richardson–Lucy ---------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
def test_transpose_back_projector_is_richardson_lucy(ndim):
    psf = gaussian_psf(7, 1.2, ndim)
    psf = psf * torch.linspace(0.8, 1.2, 7, dtype=F64)  # asymmetric
    shape = (24, 20, 18)[:ndim]
    gen = torch.Generator().manual_seed(3)
    data = torch.poisson(torch.full(shape, 30.0, dtype=F64), generator=gen)
    op = PaddedConvolution(psf, shape)
    back = BackProjection(flipped(psf), op)
    torch.testing.assert_close(back.usable, op.pad(torch.ones(shape)) > 0)
    r = torch.rand(shape, generator=gen, dtype=F64)
    torch.testing.assert_close(back(r), op.adjoint(r))
    ref, _ = rl_block(data, op, 30.0, n_iter=5)
    out, _ = rl_block(data, op, 30.0, n_iter=5, back=back)
    torch.testing.assert_close(out, ref)
    # estimator: traditional back projector, uniform start
    est = wiener_butterworth(
        data, psf, back_projector="traditional", start="uniform", n_iter=5
    )
    torch.testing.assert_close(est, richardson_lucy(data, psf, n_iter=5))


def test_first_iteration_from_uniform_start_is_a_linear_filter():
    """Supplementary Note 3, Eq. 34: e⁽¹⁾ = y * b."""
    psf = gaussian_psf(15, 2.0)
    shape = (64, 64)
    data, _ = beads(shape, psf, step=16, peak=500.0, base=50.0)
    op = PaddedConvolution(psf, shape)
    b = back_projector(psf)
    level = float(data.mean())
    f, _ = rl_block(data, op, level, n_iter=1, back=BackProjection(b, op))
    # away from the edges, H e⁽⁰⁾ = level and B1 = 1
    filtered = BackProjection(b, op)(data)
    inner = (slice(15, 49), slice(15, 49))
    expected = filtered.clamp(min=0)
    torch.testing.assert_close(
        op.crop(f)[inner], op.crop(expected)[inner], rtol=1e-8, atol=1e-8
    )


# -- paper properties -------------------------------------------------------
def test_iteration_savings():
    """Fig. 1d: WB reaches a given bead width in far fewer iterations."""
    psf = gaussian_psf(25, 3.0)
    data, pos = beads((128, 128), psf, step=24)
    op = PaddedConvolution(psf, data.shape)
    level = float(data.mean())
    start = op.pad(data.clamp(min=1e-3) - level) + level

    def width(back, n):
        f, _ = rl_block(data, op, start, n_iter=n, back=back)
        return mean_width(op.crop(f), pos)

    raw = mean_width(data, pos)
    assert raw == pytest.approx(psf_fwhm(psf)[0], rel=0.05)
    # Table S2.1, diSPIM: α = 0.05, β = 0.01, n = 10
    kw = {"wiener": 0.05, "cutoff_gain": 0.01, "order": 10}
    wb = BackProjection(back_projector(psf, **kw), op)
    assert width(wb, 1) < width(None, 1)
    assert width(wb, 5) < width(None, 15)
    assert width(wb, 10) < width(None, 50)
    # defaults (iSIM: α = β = 0.001, n = 8): sharper at equal iterations
    wb = BackProjection(back_projector(psf), op)
    for n in (1, 2, 3):
        assert width(wb, n) < width(None, n)


def test_negative_values_are_set_to_zero():
    """Main text and Supplementary Note 1: negatives truncated each step."""
    psf = gaussian_psf(15, 2.0)
    data, _ = beads((64, 64), psf, step=16, peak=5000.0, base=0.5)
    op = PaddedConvolution(psf, data.shape)
    back = BackProjection(back_projector(psf), op)
    inv_norm = torch.where(op.norm > 1e-12, 1 / back(torch.ones_like(data)), 0)
    f0 = op.pad(data.clamp(min=1e-3))
    mu = op.forward(f0)
    raw = f0 * back(data / mu) * inv_norm
    assert float(raw.min()) < 0  # the WB update itself goes negative
    f1 = rl_step(f0, data, op, inv_norm=inv_norm, back=back)
    assert float(f1.min()) == 0.0
    torch.testing.assert_close(f1, raw.clamp(min=0))
    assert bool((op.crop(f1) == 0).any())
    # zeros stay zero (multiplicative update)
    f2 = rl_step(f1, data, op, inv_norm=inv_norm, back=back)
    assert bool((f2[f1 == 0] == 0).all())
    out = wiener_butterworth(data, psf, n_iter=3)
    assert float(out.min()) >= 0
    assert bool(torch.isfinite(out).all())


@pytest.mark.parametrize("kind", ["wiener-butterworth", "butterworth"])
def test_many_iterations_stay_bounded(kind):
    """100 iterations at low counts: bounded, float32 ≈ float64."""
    psf = gaussian_psf(25, 2.0)
    data, _ = beads((64, 64), psf, step=16, peak=1000.0, base=2.0)
    kw = {"back_projector": kind, "n_iter": 100}
    a = wiener_butterworth(data, psf, dtype=F64, **kw)
    b = wiener_butterworth(data.float(), psf.float(), dtype="float32", **kw)
    assert float(a.max()) < 1000.0  # the flux of one bead
    assert float(a.sum()) == pytest.approx(float(data.sum()), rel=0.05)
    assert float((a - b).abs().max() / a.max()) < 1e-3


# -- blocks -----------------------------------------------------------------
@pytest.mark.parametrize("n_iter", [1, 5])
def test_blocks_match_whole_image_2d(n_iter):
    psf = gaussian_psf(9, 1.5)
    data, _ = beads((96, 80), psf, step=12, peak=400.0, base=5.0)
    kw = {"n_iter": n_iter}
    whole = wiener_butterworth(data, psf, block=(96, 80), **kw)
    parts = wiener_butterworth(data, psf, block=(32, 24), **kw)
    err = float((parts - whole).abs().max() / whole.abs().max())
    assert err < 1e-3


def test_blocks_match_whole_image_3d():
    psf = gaussian_psf(5, 1.0, ndim=3)
    data, _ = beads((20, 24, 22), psf, step=6, peak=400.0, base=5.0)
    whole = wiener_butterworth(data, psf, n_iter=2, block=(20, 24, 22))
    parts = wiener_butterworth(data, psf, n_iter=2, block=(10, 12, 11))
    err = float((parts - whole).abs().max() / whole.abs().max())
    assert err < 1e-3


# -- estimator ----------------------------------------------------------------
def test_faithful_flags_and_defaults():
    psf = gaussian_psf(9, 1.5)
    est = WienerButterworth(psf)
    assert est.is_faithful
    assert est.options["n_iter"] == 1
    assert est.options["back_projector"] == "wiener-butterworth"
    assert WienerButterworth(psf, chi2_stop=True).is_faithful is False
    assert WienerButterworth(psf, start="uniform").is_faithful is False
    assert (
        WienerButterworth(psf, back_projector="gaussian").is_faithful is False
    )
    # plain Richardson–Lucy is unaffected
    assert RichardsonLucy(psf).is_faithful
    assert RichardsonLucy(psf).options["n_iter"] is None


def test_start_from_data_and_chi2_stop():
    psf = gaussian_psf(9, 1.5)
    data, _ = beads((48, 48), psf, step=12, peak=400.0, base=5.0)
    est = WienerButterworth(psf, n_iter=0).fit(data)
    out = est.apply(data)  # zero iterations: the starting estimate
    torch.testing.assert_close(out, data.clamp(min=0.001).to(out.dtype))
    est = WienerButterworth(psf, chi2_stop=True, max_iter=40).fit(data)
    est.apply(data)
    assert 1 <= est.history.extras["iterations"][-1] <= 40


def test_kernel_and_float32():
    psf = gaussian_psf(9, 1.5)
    est = WienerButterworth(psf)
    torch.testing.assert_close(
        est.kernel(dtype=F64, device="cpu"), back_projector(psf)
    )
    data, _ = beads((48, 48), psf, step=12, peak=400.0, base=5.0)
    a = wiener_butterworth(data, psf, n_iter=3, dtype=F64)
    b = wiener_butterworth(
        data.float(), psf.float(), n_iter=3, dtype="float32"
    )
    assert float((a - b).abs().max() / a.max()) < 1e-4
