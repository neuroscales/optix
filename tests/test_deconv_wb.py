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
    kernel_shape,
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


def padded(psf, shape):
    """Zero-pad a PSF centred at `K // 2` to `shape`, centred."""
    out = psf.new_zeros(tuple(shape))
    index = tuple(
        slice(n // 2 - k // 2, n // 2 - k // 2 + k)
        for n, k in zip(shape, psf.shape, strict=True)
    )
    out[index] = psf
    return out


def operator(psf, shape, kernel):
    """Blur operator whose padded grid also fits `kernel`."""
    return PaddedConvolution(padded(psf, kernel.shape), shape)


def airy_psf(size=25, cutoff=0.29):
    """In-focus PSF of a circular pupil: OTF cutoff `cutoff` cycles/px."""
    n = 256
    k = torch.fft.fftfreq(n, dtype=F64)
    nu = (torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2) / cutoff).clamp(max=1)
    otf = 2 / math.pi * (torch.acos(nu) - nu * torch.sqrt(1 - nu**2))
    psf = torch.fft.fftshift(torch.fft.ifft2(otf).real).clamp(min=0)
    h = size // 2
    psf = psf[n // 2 - h : n // 2 - h + size, n // 2 - h : n // 2 - h + size]
    return psf / psf.sum()


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


def _embed(b, shape):
    """Spectrum of a centred kernel, zero-padded to `shape`."""
    return _spectrum(padded(b, shape))


@pytest.mark.parametrize("ndim", [2, 3])
def test_frequency_responses_match_supplementary_note_2(ndim):
    """Eqs. 13, 18, 21, 23, 25 and 27, on a fine frequency grid."""
    big = (128, 96, 80)[:ndim]
    shape = (17, 15, 13)[:ndim]
    fwhm = (4.0, 6.0, 5.0)[:ndim]  # cutoffs 1/4, 1/6, 1/5: grid frequencies
    gen = torch.Generator().manual_seed(0)
    psf = gaussian_psf(17, 1.2, ndim)[tuple(slice(0, n) for n in shape)]
    psf = psf * (1 + 0.1 * torch.rand(shape, generator=gen, dtype=F64))
    psf = psf / psf.sum()
    otf = _embed(psf, big)
    alpha, beta, n = 0.01, 0.02, 6
    eps2 = 1 / beta**2 - 1
    butter = 1 / torch.sqrt(1 + eps2 * _radius2(big, fwhm) ** n)
    kw = {"wiener": alpha, "cutoff_gain": beta, "order": n, "fwhm": fwhm}
    # traditional: conj(OTF), the flipped PSF
    b = back_projector(psf, "traditional")
    assert b.shape == psf.shape
    torch.testing.assert_close(_embed(b, big), otf.conj())
    # Butterworth, gain β at each cutoff (Eq. 23)
    b = back_projector(psf, "butterworth", **kw)
    assert b.shape == kernel_shape(psf, fwhm)
    assert b.shape == tuple(2 * math.ceil(6 * w) + 1 for w in fwhm)
    spec = _embed(b, big)
    assert float((spec - butter).abs().max()) < 0.01
    for axis, w in enumerate(fwhm):
        index = [0] * ndim
        index[axis] = round(big[axis] / w)
        assert float(spec[tuple(index)].real) == pytest.approx(beta, abs=0.01)
    # Wiener–Butterworth, normalised to unit DC gain (Eq. 25, Note 1)
    b = back_projector(psf, "wiener-butterworth", **kw)
    spec = _embed(b, big)
    wiener = otf.conj() / (otf.abs().square() + alpha)
    wb = wiener * butter * (1 + alpha)
    assert float((spec - wb).abs().max()) < 0.03 * float(wb.abs().max())
    index = (round(big[0] / fwhm[0]),) + (0,) * (ndim - 1)
    ref = abs(complex(wiener[index])) * beta * (1 + alpha)  # Eq. 27
    assert abs(complex(spec[index])) == pytest.approx(ref, abs=0.01)
    assert float(b.sum()) == pytest.approx(1.0)
    # Gaussian with the FWHM of the PSF (Eq. 18), unit sum
    b = back_projector(psf, "gaussian", fwhm=fwhm)
    assert float(b.sum()) == pytest.approx(1.0)
    assert psf_fwhm(b) == pytest.approx(fwhm, rel=2e-2)


@pytest.mark.parametrize("size", [15, 25, 41, 65])
def test_wb_response_does_not_depend_on_the_psf_array(size):
    """Eq. 25 is followed whatever the size of the PSF array."""
    psf = airy_psf(size)
    big = (512, 512)
    otf = _embed(psf, big)
    fwhm = psf_fwhm(psf)
    k2 = _radius2(big, fwhm)
    wb = otf.conj() / (otf.abs().square() + 1e-3)
    wb = wb / torch.sqrt(1 + (1e6 - 1) * k2**8) * (1 + 1e-3)
    b = back_projector(psf)
    assert b.shape == (max(size, 45),) * 2  # 12 FWHM of 3.54 px
    spec = _embed(b, big)
    assert float((spec - wb).abs().max()) < 0.03 * float(wb.abs().max())
    assert float(spec.abs()[k2 > 1.2**2].max()) < 0.01  # stop band


def test_wb_flattens_the_spectral_product():
    """Supplementary Note 1: |DFT(b) DFT(f)| flatter than |OTF|²."""
    psf = gaussian_psf(41, 3.0)
    big = (128, 128)
    otf = _embed(psf, big)
    trad = (_embed(back_projector(psf, "traditional"), big) * otf).abs()
    wb = (_embed(back_projector(psf), big) * otf).abs()
    k = 6  # a low, passed frequency
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


@pytest.mark.parametrize("accelerate", [False, True])
def test_hooks_compose_with_the_matched_back_projector(accelerate):
    """back = Hᵀ with the Snyder hooks at their defaults is plain RL."""
    psf = gaussian_psf(7, 1.2)
    psf = psf * torch.linspace(0.8, 1.2, 7, dtype=F64)  # asymmetric
    psf = psf / psf.sum()
    shape = (26, 22)
    gen = torch.Generator().manual_seed(5)
    data = torch.poisson(torch.full(shape, 40.0, dtype=F64), generator=gen)
    op = PaddedConvolution(psf, shape)
    back = BackProjection(flipped(psf), op)
    kw = {"n_iter": 6, "accelerate": accelerate}
    ref, n = rl_block(data, op, 40.0, **kw)
    out, m = rl_block(
        data, op, 40.0, back=back, flat=None, expect=None, variance=0.0, **kw
    )
    assert n == m == 6

    # FFT round-off between BackProjection and op.adjoint, relative to the
    # peak. With acceleration the extrapolation and its clipping at 0
    # amplify it (seen on torch 2.3), hence a looser float64 bound.
    tol = 1e-9 if accelerate else 1e-12

    def err(a, b):
        return float((a - b).abs().max() / b.abs().max())

    assert err(out, ref) < tol, err(out, ref)
    # through the estimator hooks (_init_from, _solve)
    est = WienerButterworth(
        psf,
        back_projector="traditional",
        start="uniform",
        n_iter=6,
        faithful=False,
        dtype=F64,
    )
    est._back = lambda op: BackProjection(flipped(psf).to(op.norm), op)
    out = est.fit(data).apply(data)
    ref = richardson_lucy(data, psf, n_iter=6, dtype=F64)
    assert err(out, ref) < 1e-12, err(out, ref)
    with pytest.raises(ValueError, match="back and flat"):
        rl_block(data, op, 40.0, n_iter=1, back=back, flat=torch.ones(shape))


def test_first_iteration_from_uniform_start_is_a_linear_filter():
    """Supplementary Note 3, Eq. 34: e⁽¹⁾ = y * b."""
    psf = gaussian_psf(15, 2.0)
    shape = (128, 128)
    data, _ = beads(shape, psf, step=16, peak=500.0, base=50.0)
    b = back_projector(psf)
    op = operator(psf, shape, b)
    level = float(data.mean())
    f, _ = rl_block(data, op, level, n_iter=1, back=BackProjection(b, op))
    # away from the edges, H e⁽⁰⁾ = level and B1 = 1
    filtered = BackProjection(b, op)(data)
    inner = (slice(32, 96), slice(32, 96))  # b is 59 px wide
    expected = filtered.clamp(min=0)
    torch.testing.assert_close(
        op.crop(f)[inner], op.crop(expected)[inner], rtol=1e-8, atol=1e-8
    )


# -- paper properties -------------------------------------------------------
def test_iteration_savings():
    """Fig. 1d: WB reaches a given bead width in far fewer iterations."""
    psf = gaussian_psf(25, 3.0)
    data, pos = beads((128, 128), psf, step=24)
    op = operator(psf, data.shape, back_projector(psf))
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


def test_iterations_on_beads_over_a_background():
    """Fig. 1d regime: beads on a background of 25 % of their peak.

    Iterations to bring the mean bead FWHM to FWHM_PSF / 1.4, from the
    data (measured: traditional 10, WB defaults 21, WB with
    fwhm = FWHM_PSF / 1.3 4). With the cutoff of the text (1/FWHM) the
    WB filter stops near 0.6 k_c and is no faster than Richardson–Lucy;
    the cutoff of Fig. S2.3 (about 1.3/FWHM) is.
    """
    psf = airy_psf()
    shape, step = (96, 96), 24
    truth = torch.zeros(shape, dtype=F64)
    pos = [(i, j) for i in range(12, 96, step) for j in range(12, 96, step)]
    for p in pos:
        truth[p] = 1.0
    op = PaddedConvolution(psf, shape)
    clean = op.forward(op.pad(truth))
    clean = clean / clean.max() * 3000.0 + 750.0
    gen = torch.Generator().manual_seed(2)
    data = torch.poisson(clean, generator=gen)
    fwhm = psf_fwhm(psf)[0]

    def width(img):
        out = []
        for i, j in pos:
            crop = img[i - 10 : i + 11, j - 10 : j + 11]
            out.append(psf_fwhm(crop - crop.median())[0])
        return sum(out) / len(out)

    def iterations(target, limit=30, **kw):
        for n in range(1, limit + 1):
            out = wiener_butterworth(data, psf, n_iter=n, **kw)
            if width(out) <= target:
                return n
        return limit + 1

    target = fwhm / 1.4
    trad = iterations(target, back_projector="traditional", faithful=False)
    wb = iterations(target)
    fast = iterations(target, fwhm=fwhm / 1.3, faithful=False)
    assert 8 <= trad <= 12
    assert fast <= trad // 2
    assert fast <= wb // 4


def test_negative_values_are_set_to_zero():
    """Main text and Supplementary Note 1: negatives truncated each step."""
    psf = gaussian_psf(15, 2.0)
    data, _ = beads((64, 64), psf, step=16, peak=5000.0, base=0.5)
    b = back_projector(psf)
    op = operator(psf, data.shape, b)
    back = BackProjection(b, op)
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
