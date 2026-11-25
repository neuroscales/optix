"""NCS, the sCMOS noise-correction algorithm (Liu et al. 2017)."""

import math

import pytest
import torch

from optix.denoise import NCS, ncs, ncs_halo
from optix.denoise import utils as U
from optix.estimates import SensorEstimate
from optix.io import Layout
from optix.models import phantoms
from optix.optim import ShiftedPoisson, SquaredL2, Term, admm_fft

F64 = dict(dtype=torch.float64)
OPTICS = dict(numerical_aperture=1.4, wavelength=0.7, pixel_size=0.1)
CUTOFF = 2 * 1.4 / 0.7  # OTF radius 2NA/λ, cycles per µm


def radial(shape, dtype=torch.float64):
    ky = torch.fft.fftfreq(shape[0], d=0.1, dtype=dtype)
    kx = torch.fft.fftfreq(shape[1], d=0.1, dtype=dtype)
    return torch.sqrt(ky[:, None] ** 2 + kx[None] ** 2)


def band_limited(shape=(64, 64), seed=0):
    """Filaments seen through the OTF of an ideal circular pupil, + 10."""
    x = phantoms.filaments(
        shape, 6, width=1.0, intensity=100, seed=seed, **F64
    )
    rho = (radial(shape) / CUTOFF).clamp(max=1)
    otf = (2 / math.pi) * (torch.acos(rho) - rho * torch.sqrt(1 - rho**2))
    return torch.fft.ifft2(torch.fft.fft2(x) * otf).real + 10


def camera(shape, seed=1):
    """Per-pixel offset, gain and (log-normal) read-noise variance maps."""
    g = torch.Generator().manual_seed(seed)
    return SensorEstimate(
        offset=100 + 5 * torch.randn(shape, generator=g, **F64),
        variance=20 * torch.exp(1.5 * torch.randn(shape, generator=g, **F64)),
        gain=2 + 0.2 * torch.randn(shape, generator=g, **F64),
    )


def record(truth, sensor, frames=None, seed=2):
    """A = o + g·Poisson(truth) + 𝒩(0, σ²) (Supp. Note 14)."""
    g = torch.Generator().manual_seed(seed)
    shape = truth.shape if frames is None else (frames, *truth.shape)
    electrons = torch.poisson(truth.expand(shape), generator=g)
    noise = torch.randn(shape, generator=g, dtype=truth.dtype)
    return (
        sensor.offset
        + sensor.gain * electrons
        + sensor.variance.sqrt() * noise
    )


def segment(m=16, seed=0):
    """D, γ and M of one padded segment."""
    truth = band_limited((m + 2, m + 2), seed)
    sensor = camera(truth.shape, seed)
    d, gamma = U.ncs_precorrect(
        record(truth, sensor), sensor.offset, sensor.variance, sensor.gain
    )
    mask = U.ncs_mask(truth.shape, **OPTICS, **F64)
    return d, gamma, mask


# -- mask --------------------------------------------------------------------
def test_raised_cosine():
    k = torch.linspace(0, 10, 2001, **F64)
    for beta, period in [(1.0, 0.7 / (5.6 * 1.4)), (0.2, 0.8 * 0.7 / 5.6)]:
        m = U.raised_cosine(k, beta, period)
        lo, hi = (1 - beta) / (2 * period), (1 + beta) / (2 * period)
        assert (m[k < lo] == 0).all() and (m[k > hi] == 1).all()
        assert (m.diff() >= 0).all()  # high-pass, continuous: monotone
        assert m.diff().abs().max() < 0.01
    step = U.raised_cosine(k, 0.0, 0.25)
    assert torch.equal(step, (k > 2).double())


def test_paper_masks():
    """Supp. Note 5 and Fig. SS4B: OTF-weighted ≈ 0.28 at 2NA/λ and
    ≈ 0.7 at 3.5NA/λ; noise-only 0 inside 2NA/λ, 1 beyond 3NA/λ."""
    na_l = 1.4 / 0.7
    k = torch.tensor([0, 2 * na_l, 3.5 * na_l, 5.6 * na_l], **F64)
    m = U.raised_cosine(k, 1.0, 0.7 / (5.6 * 1.4))
    expect = 0.5 - 0.5 * torch.cos(math.pi * k / (5.6 * na_l))
    torch.testing.assert_close(m, expect)
    assert m[0] == 0 and abs(m[1] - 0.283) < 1e-3 and abs(m[2] - 0.69) < 0.01
    beta = 0.2
    period = (1 - beta) * 0.7 / (4 * 1.4)
    k = torch.tensor([1.99 * na_l, 2.01 * na_l, 3.01 * na_l], **F64)
    m = U.raised_cosine(k, beta, period)
    assert m[0] == 0 and 0 < m[1] < 0.01 and m[2] == 1
    grid = U.ncs_mask((18, 18), mask="noise-only", **OPTICS, **F64)
    torch.testing.assert_close(
        grid, U.raised_cosine(radial((18, 18)), beta, period)
    )


def test_mask_grid_and_callable():
    for shape in [(18, 18), (17, 20)]:
        m = U.ncs_mask(shape, **OPTICS, **F64)
        assert m.shape == shape and m[0, 0] == 0
        # even in k: M(−k) = M(k) on the DFT grid
        flip = torch.roll(m.flip(0, 1), (1, 1), (0, 1))
        torch.testing.assert_close(m, flip, rtol=0, atol=1e-15)
    user = U.ncs_mask(
        (10, 12), mask=lambda k: (k > 5).double(), **OPTICS, **F64
    )
    assert torch.equal(user, (radial((10, 12)) > 5).double())
    with pytest.raises(ValueError):
        U.ncs_mask((8, 8), mask="gaussian", **OPTICS)


# -- cost --------------------------------------------------------------------
def test_precorrect():
    data = torch.tensor([[90.0, 100.0], [104.0, 120.0]], **F64)
    d, gamma = U.ncs_precorrect(data, 100.0, torch.tensor(8.0, **F64), 2.0)
    torch.testing.assert_close(
        d, torch.tensor([[1e-6, 1e-6], [2.0, 10.0]], **F64)
    )
    torch.testing.assert_close(gamma, torch.full((2, 2), 2.0, **F64))


def test_cost_matches_equations():
    """σ_N = ∑|μ̂ M|² with the unitary DFT (Eqs. 2–3); LL_s of Eq. 7."""
    d, gamma, mask = segment()
    x = d + torch.rand(d.shape, **F64)
    hat = torch.fft.fft2(x, norm="ortho")
    sigma_n = (hat * mask).abs().square().sum()
    lls = (x - (d + gamma) * torch.log(x + gamma)).sum()
    f = U.ncs_objective(x, d, gamma, mask, 0.2)
    torch.testing.assert_close(f, lls + 0.2 * sigma_n)


def test_gradient_matches_autograd():
    d, gamma, mask = segment()
    x = (d + torch.rand(d.shape, **F64)).requires_grad_()
    U.ncs_objective(x, d, gamma, mask, 0.2).backward()
    g = U.ncs_gradient(x.detach(), d, gamma, mask, 0.2)
    torch.testing.assert_close(g, x.grad)


# -- solver ------------------------------------------------------------------
def test_alpha_zero_keeps_data():
    """α = 0: D is the minimum (Fig. SS6)."""
    d, gamma, mask = segment()
    res = U.ncs_solve(d[None], gamma[None], mask, 0.0)
    assert res.n_iter.item() == 0
    assert torch.equal(res.x[0], d)


@pytest.mark.parametrize("m", [8, 16, 32])
def test_map_optimality(m):
    """∇f ≈ 0 at the result, and f is lower than at random neighbours."""
    d, gamma, mask = segment(m)
    res = U.ncs_solve(d[None], gamma[None], mask, 0.2)
    assert bool(res.converged.all())
    x = res.x[0]
    g = U.ncs_gradient(x, d, gamma, mask, 0.2)
    assert g.abs().max() < 1e-9
    f0 = U.ncs_objective(x, d, gamma, mask, 0.2)
    gen = torch.Generator().manual_seed(0)
    for _ in range(5):
        dx = 1e-3 * torch.randn(x.shape, generator=gen, **F64)
        assert U.ncs_objective(x + dx, d, gamma, mask, 0.2) > f0
    assert U.ncs_objective(d, d, gamma, mask, 0.2) > f0


def test_matches_admm():
    """Cross-solver: ADMM of `optix.optim` on LL_s + (2α/2)‖F⁻¹ M F μ‖²."""
    d, gamma, mask = segment(8)
    res = U.ncs_solve(d[None], gamma[None], mask, 0.2)
    half = mask[:, : mask.shape[1] // 2 + 1]
    terms = [
        Term(None, ShiftedPoisson(d, gamma)),
        Term(half, SquaredL2(weight=0.4)),
    ]
    out = admm_fft(terms, d.shape, rho=0.5, max_iter=3000, tol=1e-13, **F64)
    torch.testing.assert_close(out.x, res.x[0], rtol=1e-6, atol=1e-6)


def test_batch_independence():
    """Each image of a batch gives the result it gives alone."""
    pairs = [segment(16, seed) for seed in range(3)]
    d = torch.stack([p[0] for p in pairs])
    gamma = torch.stack([p[1] for p in pairs])
    mask = pairs[0][2]
    together = U.ncs_solve(d, gamma, mask, 0.2)
    for i in range(3):
        alone = U.ncs_solve(d[i : i + 1], gamma[i : i + 1], mask, 0.2)
        assert torch.equal(alone.x[0], together.x[i])
        assert alone.n_iter.item() == together.n_iter[i].item()


def test_float32_matches_float64():
    d, gamma, mask = segment()
    r64 = U.ncs_solve(d[None], gamma[None], mask, 0.2)
    r32 = U.ncs_solve(d[None].float(), gamma[None].float(), mask.float(), 0.2)
    assert bool(r32.converged.all())
    torch.testing.assert_close(r32.x.double(), r64.x, rtol=1e-4, atol=1e-4)


# -- segmentation ------------------------------------------------------------
def test_segment_bounds():
    assert U.segment_bounds(64, 16) == [(0, 16), (16, 32), (32, 48), (48, 64)]
    assert U.segment_bounds(40, 16) == [(0, 16), (16, 40)]
    assert U.segment_bounds(10, 16) == [(0, 10)]
    assert ncs_halo(16) == 31


def test_segments_are_independent_with_paper_padding():
    """A segment uses its pixels and the adjacent ones (Supp. Note 7)."""
    truth = band_limited((48, 48))
    sensor = camera(truth.shape)
    raw = record(truth, sensor)
    out = ncs(raw, sensor, segment=16, **OPTICS)
    d, gamma = U.ncs_precorrect(
        raw, sensor.offset, sensor.variance, sensor.gain
    )
    mask = U.ncs_mask((18, 18), **OPTICS, **F64)
    # interior segment (16:32, 16:32), padded with its neighbours
    seg = (slice(15, 33), slice(15, 33))
    res = U.ncs_solve(d[seg][None], gamma[seg][None], mask, 0.2)
    torch.testing.assert_close(out[16:32, 16:32], res.x[0, 1:-1, 1:-1])
    # corner segment (0:16, 0:16), padded with itself at the image border
    idx = torch.arange(-1, 17).clamp(0, 47)
    pick = (idx[:, None], idx[None, :])
    res = U.ncs_solve(d[pick][None], gamma[pick][None], mask, 0.2)
    torch.testing.assert_close(out[:16, :16], res.x[0, 1:-1, 1:-1])


# -- validation on synthetic sCMOS data ---------------------------------------
@pytest.fixture(scope="module")
def movie():
    truth = band_limited()
    sensor = camera(truth.shape)
    return truth, sensor, record(truth, sensor, frames=8)


def band_energy(err, band):
    return (torch.fft.fft2(err, norm="ortho").abs().square() * band).sum(
        (-2, -1)
    )


@pytest.mark.parametrize(
    "mask, ratio", [("otf-weighted", 0.2), ("noise-only", 0.45)]
)
def test_noise_outside_otf_is_reduced(movie, mask, ratio):
    truth, sensor, raw = movie
    d = (raw - sensor.offset) / sensor.gain
    out = ncs(raw, sensor, mask=mask, **OPTICS)
    outside = radial(truth.shape) > CUTOFF
    before = band_energy(d - truth, outside).mean()
    after = band_energy(out - truth, outside).mean()
    assert after < ratio * before


@pytest.mark.parametrize("mask", ["otf-weighted", "noise-only"])
def test_signal_inside_otf_is_preserved(movie, mask):
    """The in-band error does not grow, and the mean over frames keeps
    the in-band signal (Supp. Fig. 2: mean pixel values unchanged)."""
    truth, sensor, raw = movie
    d = (raw - sensor.offset) / sensor.gain
    out = ncs(raw, sensor, mask=mask, **OPTICS)
    inside = radial(truth.shape) <= 0.8 * CUTOFF
    assert (
        band_energy(out - truth, inside).mean()
        < band_energy(d - truth, inside).mean()
    )
    signal = band_energy(truth - truth.mean(), inside)
    bias = band_energy(out.mean(0) - truth, inside)
    assert bias < 0.01 * signal


def test_high_variance_pixels_are_corrected(movie):
    """Temporal fluctuation of the noisiest pixels drops (Fig. 1c–e)."""
    truth, sensor, raw = movie
    d = (raw - sensor.offset) / sensor.gain
    out = ncs(raw, sensor, **OPTICS)
    hot = sensor.variance > sensor.variance.quantile(0.9)
    assert out.std(0)[hot].mean() < 0.5 * d.std(0)[hot].mean()
    assert ((out - truth) ** 2).mean() < 0.5 * ((d - truth) ** 2).mean()


# -- estimator and streaming --------------------------------------------------
def test_estimator_matches_function(movie):
    truth, sensor, raw = movie
    est = NCS(**OPTICS, **F64).fit(init=sensor)
    assert est.is_faithful and est.halo() == 31
    torch.testing.assert_close(est.apply(raw), ncs(raw, sensor, **OPTICS))


@pytest.mark.parametrize("block", [(16, 32), (23, 41), (7, 64), (64, 9)])
def test_blocks_match_whole(block, dtype):
    """Streaming with the halo 2M − 1 gives the whole-image result."""
    truth = band_limited((53, 64)).to(dtype)
    sensor = camera(truth.shape).to(dtype=dtype)
    raw = record(truth, sensor)
    whole = ncs(raw, sensor, segment=8, **OPTICS)
    est = NCS(segment=8, block=block, dtype=dtype, **OPTICS).fit(init=sensor)
    out = est.apply(raw)
    if dtype == torch.float64:
        assert torch.equal(out, whole)
    else:
        torch.testing.assert_close(out, whole, rtol=4e-7, atol=0)


def test_group_maps_and_adu_units():
    """Maps per group (one camera per channel), output in ADU."""
    truth = band_limited((32, 32))
    sensors = [camera(truth.shape, seed) for seed in (1, 2)]
    stacked = SensorEstimate(
        offset=torch.stack([s.offset for s in sensors]),
        variance=torch.stack([s.variance for s in sensors]),
        gain=torch.stack([s.gain for s in sensors]),
    )
    raw = torch.stack([record(truth, s, frames=2) for s in sensors])
    layout = Layout("ctyx", spatial="yx", group="c")
    est = NCS(
        segment=8, units="adu", block=(13, 20), layout=layout, **OPTICS, **F64
    )
    out = est.fit(init=stacked).apply(raw)
    for c, s in enumerate(sensors):
        ref = ncs(raw[c], s, segment=8, units="adu", **OPTICS)
        torch.testing.assert_close(out[c], ref)
    electrons = ncs(raw[0], sensors[0], segment=8, **OPTICS)
    torch.testing.assert_close(
        out[0], electrons * sensors[0].gain + sensors[0].offset
    )


def test_faithful_flags_and_errors():
    assert NCS(**OPTICS, mask="noise-only", segment=8).is_faithful
    assert not NCS(**OPTICS, alpha=0.5).is_faithful
    assert not NCS(**OPTICS, segment=64).is_faithful
    assert not NCS(**OPTICS, mask=lambda k: k).is_faithful
    with pytest.raises(ValueError):
        NCS(wavelength=0.7, pixel_size=0.1)
    with pytest.raises(ValueError):
        NCS(**OPTICS, units="photons")
    with pytest.raises(ValueError):
        NCS(**OPTICS).fit()
    with pytest.raises(RuntimeError):
        NCS(**OPTICS).apply(torch.zeros(16, 16))
    with pytest.raises(ValueError):
        ncs(
            torch.zeros(8, 8),
            SensorEstimate(offset=torch.tensor(0.0)),
            **OPTICS,
        )


def test_save_load(movie):
    truth, sensor, raw = movie
    est = NCS(segment=8, **OPTICS, **F64).fit(init=sensor)
    other = NCS(segment=8, **OPTICS, **F64).load_state_dict(est.state_dict())
    torch.testing.assert_close(other.apply(raw[:1]), est.apply(raw[:1]))
