"""PURE-LET (Luisier, Blu & Unser 2011), validated on synthetic data."""

import math

import numpy as np
import pytest
import torch

import optix
from optix.denoise import PURELET, pure_let, pure_let_halo
from optix.denoise import utils as U
from optix.denoise._purelet import _combine, _wrap
from optix.estimates import SensorEstimate
from optix.io import Layout
from optix.models import phantoms

F64 = dict(dtype=torch.float64)
D = torch.float64


def scene(shape=(64, 64), peak=30.0, seed=0) -> torch.Tensor:
    """Cells and filaments on a smooth background, in [0, peak] photons."""
    ndim = len(shape)
    x = phantoms.cells(shape, 6, radius=7.0, ndim=ndim, seed=seed, **F64)
    x = x + phantoms.filaments(
        shape, 4, width=1.5, ndim=ndim, seed=seed, **F64
    )
    grid = torch.meshgrid(
        *(torch.arange(n, dtype=D) for n in shape), indexing="ij"
    )
    x = x + 30 * torch.sin(grid[-1] / 9) + 40
    x = x - x.min()
    return x * (peak / x.max())


def observe(x, sigma, gen):
    """y = z + b, z ~ 𝒫(x), b ~ 𝒩(0, σ²) (Eq. (1))."""
    z = torch.poisson(x, generator=gen)
    return z + sigma * torch.randn(x.shape, generator=gen, dtype=x.dtype)


def basis(y, s2, levels, ndim=2, derivatives=True):
    """LET functions of a whole image, periodic boundaries."""
    h = pure_let_halo(levels)
    s2 = _wrap(s2, ndim, h) if isinstance(s2, torch.Tensor) else s2
    return U.let_basis(
        _wrap(y, ndim, h), s2, levels, ndim=ndim, derivatives=derivatives
    )


def sums_of(y, s2, levels, ndim=2):
    f, low, d = basis(y, s2, levels, ndim)
    sums = U.PureSums.from_basis(f, low, d, y, s2, levels=levels, ndim=ndim)
    return f, low, sums


def psnr(x, ref, peak):
    return 10 * math.log10(peak**2 / float((x - ref).square().mean()))


# -- transform ---------------------------------------------------------------
def test_haar_uwt_filters():
    y = torch.rand(2, 9, 10, **F64)
    (b1, l1), (b2, l2) = U.haar_uwt(y, 2)
    assert b1.shape == (3, 2, 8, 9) and b2.shape == (3, 2, 6, 7)
    # level 1: 2 × 2 sums and differences, unit norm
    blk = y[:, :2, :2]
    torch.testing.assert_close(l1[:, 0, 0], blk.sum((1, 2)) / 2)
    torch.testing.assert_close(
        b1[0, :, 0, 0], (blk[:, 0] - blk[:, 1]).sum(1) / 2
    )
    torch.testing.assert_close(
        b1[1, :, 0, 0], (blk[:, :, 0] - blk[:, :, 1]).sum(1) / 2
    )
    # level 2: the scaling coefficient is 4^(-1) × the 4 × 4 block sum,
    # aligned with the wavelet coefficients (Q(z) = 1 for Haar)
    torch.testing.assert_close(l2[:, 0, 0], y[:, :4, :4].sum((1, 2)) / 4)
    # every wavelet filter has zero mean
    ones = torch.ones(9, 10, **F64)
    for bands, _ in U.haar_uwt(ones, 2):
        assert bands.abs().max() < 1e-14


@pytest.mark.parametrize("shape", [(12, 13), (10, 9, 11)])
def test_perfect_reconstruction(shape):
    """R·D = I: Σ_j f_{j,1} + f_low = y for a_{j,1} = 1, a_{j,2} = 0."""
    ndim = len(shape)
    y = torch.randn(shape, **F64)
    f, low, _ = basis(y, 1.0, 2, ndim, derivatives=False)
    a = torch.tensor([1.0, 0.0] * (f.shape[0] // 2), **F64)
    torch.testing.assert_close(_combine(f, low, a), y, rtol=0, atol=1e-12)
    # and every band alone: diag{D_j R_j} = 1/M_j (Section III-B)
    j = 1
    m = 2.0 ** (ndim * j)
    delta = torch.zeros(shape, **F64)
    delta[(3,) * ndim] = 1.0
    f, _, _ = basis(delta, 0.0, 1, ndim, derivatives=False)
    torch.testing.assert_close(
        f[0::2][(slice(None),) + (3,) * ndim],
        torch.full((2**ndim - 1,), 1 / m, **F64),
    )


# -- thresholding function ---------------------------------------------------
def test_let_threshold_derivatives():
    g = torch.Generator().manual_seed(0)
    w = 6 * torch.randn(200, generator=g, dtype=D)
    wbar = 0.05 * torch.randn(200, generator=g, dtype=D)
    wbar[100:] = 40 * torch.rand(100, generator=g, dtype=D)
    beta, s2 = 0.5, 0.7
    out = U.let_threshold(w, wbar, beta, s2)

    def theta(w, wbar):
        t = torch.sqrt(beta * torch.tanh(100 * wbar) * wbar + s2)
        return w * torch.exp(-((w / (3 * t)) ** 8))

    hess = torch.func.hessian(theta, argnums=(0, 1))
    grad = torch.func.grad(theta, argnums=(0, 1))
    for i in range(200):
        wi, bi = w[i], wbar[i]
        g1, g2 = grad(wi, bi)
        (h11, h12), (_, h22) = hess(wi, bi)
        expected = [theta(wi, bi), g1, g2, h11, h22, h12]
        for got, ref in zip(out, expected, strict=True):
            torch.testing.assert_close(got[i], ref, rtol=1e-9, atol=1e-9)
    # Fig. 5: θ ≈ w below 3·t and 0 above (t = 2 here), so that
    # a = (1, −1) is a smooth hard thresholding
    w = torch.tensor([0.5, -1.0, 9.0, -12.0], **F64)
    theta = U.let_threshold(w, torch.full_like(w, 6.0), 0.5, 1.0)[0]
    assert (theta[:2] / w[:2]).min() > 0.999
    assert theta[2:].abs().max() < 1e-9


def test_threshold_pure_poisson_zero_background():
    """σ = 0 and w̄ = 0 (no photon): finite values, θ = 0."""
    z = torch.zeros(5, **F64)
    for t in U.let_threshold(z, z, 0.5, 0.0):
        assert torch.isfinite(t).all()
    y = torch.zeros(16, 16, **F64)
    y[4, 4] = 3.0
    out = pure_let(y, 0.0, levels=2)
    assert torch.isfinite(out).all()


# -- PURE: closed forms ------------------------------------------------------
def _diagonal_derivatives(func, y, eps=1e-4):
    """∂f_{k,n}/∂y_n and ∂²f_{k,n}/∂y_n² by central differences."""
    n = y.numel()
    e = torch.eye(n, **F64).reshape(n, *y.shape)
    idx = torch.arange(n)

    def diag(t):
        return func(t).reshape(-1, n, n)[:, idx, idx]

    fp, f0, fm = diag(y + eps * e), diag(y + 0 * e), diag(y - eps * e)
    return (fp - fm) / (2 * eps), (fp - 2 * f0 + fm) / eps**2


@pytest.mark.parametrize("pixelwise", [False, True])
def test_closed_forms_match_derivatives(pixelwise):
    """δ_k = yᵀ∂f_k + Σσ_n²(∂f_{k,n} − ∂²f_{k,n}): Eqs. (11)-(13)."""
    g = torch.Generator().manual_seed(1)
    x = 2 + 20 * torch.rand(10, 11, generator=g, dtype=D)
    y = observe(x, 1.0, g)
    s2 = (
        0.5 + 2 * torch.rand(10, 11, generator=g, dtype=D)
        if pixelwise
        else 1.0
    )
    _, _, delta = basis(y, s2, 2)

    def func(t):
        return basis(t, s2, 2, derivatives=False)[0]

    d1, d2 = _diagonal_derivatives(func, y)
    sv = s2.reshape(-1) if pixelwise else torch.full((y.numel(),), s2, **F64)
    expected = d1 @ y.reshape(-1) + (d1 - d2) @ sv
    torch.testing.assert_close(delta, expected, rtol=1e-6, atol=1e-6)


def test_closed_forms_3d():
    g = torch.Generator().manual_seed(2)
    x = 2 + 10 * torch.rand(6, 6, 6, generator=g, dtype=D)
    y = observe(x, 0.5, g)
    _, _, delta = basis(y, 0.25, 1, ndim=3)

    def func(t):
        return basis(t, 0.25, 1, ndim=3, derivatives=False)[0]

    d1, d2 = _diagonal_derivatives(func, y)
    expected = d1 @ y.reshape(-1) + 0.25 * (d1 - d2).sum(1)
    torch.testing.assert_close(delta, expected, rtol=1e-6, atol=1e-6)


def test_risk_is_the_taylor_pure():
    """PureSums.risk(a) is Eq. (6) for f = Σ a_k f_k + f_low."""
    g = torch.Generator().manual_seed(3)
    x = scene((12, 12), peak=15.0)
    y = observe(x, 1.5, g)
    s2 = 2.25
    f, low, sums = sums_of(y, s2, 2)
    a = torch.randn(f.shape[0], generator=g, dtype=D)

    def func(t):
        ft, lt, _ = basis(t, s2, 2, derivatives=False)
        return _combine(ft, lt, a)[None]

    fy = func(y)[0]
    d1, d2 = _diagonal_derivatives(func, y)
    d1, d2 = d1[0], d2[0]
    n = y.numel()
    yv = y.reshape(-1)
    expected = (
        (
            float(fy.square().sum())
            - 2 * float(yv @ (fy.reshape(-1) - d1))
            + 2 * s2 * float((d1 - d2).sum())
        )
        / n
        + float(yv.square().sum() - yv.sum()) / n
        - s2
    )
    assert sums.risk(a) == pytest.approx(expected, rel=1e-8)


# -- PURE: unbiasedness ------------------------------------------------------
def test_pure_exact_is_unbiased():
    """Theorem 1, Eq. (2), with f⁻(y) by brute force, over realisations.

    E{ε} = E{MSE}: the mean of ε − MSE is 0 within its standard error.
    """
    g = torch.Generator().manual_seed(4)
    x = scene((8, 8), peak=20.0, seed=3)
    sigma, levels, reps = 1.0, 2, 150
    n = x.numel()
    # LET coefficients fixed in advance (from an independent realisation)
    _, _, s0 = sums_of(observe(x, sigma, g), sigma**2, levels)
    a = s0.solve()

    def func(t):
        ft, lt, _ = basis(t, sigma**2, levels, derivatives=False)
        return _combine(ft, lt, a)

    e = torch.eye(n, **F64).reshape(n, 8, 8)
    idx = torch.arange(n)
    eps = 1e-4
    diffs, taylor = [], []
    for _ in range(reps):
        y = observe(x, sigma, g)
        fy = func(y)
        fminus = func(y - e).reshape(n, n)[idx, idx]  # f_n(y − e_n)
        fp = func(y - (1 - eps) * e).reshape(n, n)[idx, idx]
        fm = func(y - (1 + eps) * e).reshape(n, n)[idx, idx]
        div = float((fp - fm).sum()) / (2 * eps)  # div{f⁻(y)}
        yv = y.reshape(-1)
        pure = (
            (
                float(fy.square().sum())
                - 2 * float(yv @ fminus)
                + 2 * sigma**2 * div
            )
            / n
            + float(yv.square().sum() - yv.sum()) / n
            - sigma**2
        )
        mse = float((fy - x).square().mean())
        diffs.append(pure - mse)
        taylor.append(sums_of(y, sigma**2, levels)[2].risk(a) - mse)
    d = torch.tensor(diffs, **F64)
    assert abs(float(d.mean())) < 3 * float(d.std()) / reps**0.5
    t = torch.tensor(taylor, **F64)
    assert abs(float(t.mean())) < 3 * float(t.std()) / reps**0.5


@pytest.mark.parametrize(
    ("peak", "sigma"), [(30.0, 3.0), (5.0, 0.5), (1.0, 0.1)]
)
def test_pure_taylor_is_unbiased(peak, sigma):
    """Eq. (6) (what is minimised) against the true MSE, 64 × 64 images.

    With the LET coefficients fixed, the mean of ε̂ − MSE over noise
    realisations is 0 within 3 standard errors, at the intensities of
    Table II (peak 30, 5 and 1 photons).
    """
    g = torch.Generator().manual_seed(5)
    x = scene((64, 64), peak=peak)
    levels, reps = 4, 40
    _, _, s0 = sums_of(observe(x, sigma, g), sigma**2, levels)
    a = s0.solve()
    diffs, mses = [], []
    for _ in range(reps):
        y = observe(x, sigma, g)
        f, low, sums = sums_of(y, sigma**2, levels)
        mse = float((_combine(f, low, a) - x).square().mean())
        diffs.append(sums.risk(a) - mse)
        mses.append(mse)
    d = torch.tensor(diffs, **F64)
    se = float(d.std()) / reps**0.5
    assert abs(float(d.mean())) < 3 * se
    assert se < 0.15 * sum(mses) / reps  # a sharp test


# -- the linear system -------------------------------------------------------
def test_solution_minimises_pure():
    g = torch.Generator().manual_seed(6)
    x = scene((48, 48), peak=20.0)
    y = observe(x, 2.0, g)
    _, _, sums = sums_of(y, 4.0, 3)
    a = sums.solve()
    keep = sums.active()
    assert keep.all()
    # Eq. (22): M a = ĉ
    torch.testing.assert_close(
        sums.gram @ a, sums.cross - sums.delta, rtol=1e-10, atol=1e-8
    )
    best = sums.risk(a)
    for _ in range(20):
        step = 1e-3 * torch.randn(a.shape, generator=g, dtype=D)
        assert sums.risk(a + step) > best
    # the quadratic has Hessian 2G/N
    step = torch.randn(a.shape, generator=g, dtype=D)
    gain = sums.risk(a + step) - best
    assert gain == pytest.approx(
        float(step @ sums.gram @ step) / sums.count, rel=1e-8
    )


def test_reliability_rule():
    """Section III-D: f_{j,2} is kept only when M_j·E_mean > T."""
    g = torch.Generator().manual_seed(7)
    x = scene((64, 64), peak=1.0)
    y = observe(x, 0.1, g)
    _, _, sums = sums_of(y, 0.01, 4)
    energy = sums.energy
    assert energy == pytest.approx(float(x.square().mean()), rel=0.2)
    keep = sums.active(10.0)
    j = torch.arange(1, 5).repeat_interleave(6)
    kind2 = torch.arange(24) % 2 == 1
    torch.testing.assert_close(keep[~kind2], torch.ones(12, dtype=torch.bool))
    torch.testing.assert_close(keep[kind2], (4.0 ** j[kind2]) * energy > 10)
    assert not keep.all() and keep[kind2].any()
    a = sums.solve(10.0)
    assert (a[~keep] == 0).all()
    assert sums.active(0.0).all()


def test_psnr_gain():
    """PSNR gain, and PURE-optimal LET close to the MSE-optimal LET."""
    g = torch.Generator().manual_seed(8)
    for peak, sigma, gain in [(30.0, 3.0, 8.0), (2.0, 0.2, 10.0)]:
        x = scene((96, 96), peak=peak)
        y = observe(x, sigma, g)
        out = pure_let(y, sigma)
        p_in, p_out = psnr(y, x, peak), psnr(out, x, peak)
        assert p_out > p_in + gain
        # oracle: the LET minimising the true MSE (same active set)
        f, low, sums = sums_of(y, sigma**2, 4)
        keep = sums.active()
        fk = f[keep].reshape(int(keep.sum()), -1)
        ao = torch.linalg.solve(fk @ fk.T, fk @ (x - low).reshape(-1))
        oracle = (ao @ fk).reshape(x.shape) + low
        assert p_out > psnr(oracle, x, peak) - 1.0


def test_gain_offset_units():
    """Data in ADU: offset + gain·(z + b); the result is in ADU."""
    g = torch.Generator().manual_seed(9)
    x = scene((40, 40), peak=20.0)
    y = observe(x, 1.5, g)
    alpha, delta = 3.0, 100.0
    expected = pure_let(y, 1.5)
    out = pure_let(alpha * y + delta, alpha * 1.5, gain=alpha, offset=delta)
    torch.testing.assert_close(out, alpha * expected + delta)
    # constant maps are the scalar case
    full = torch.full((40, 40), alpha, **F64)
    torch.testing.assert_close(
        pure_let(alpha * y + delta, alpha * 1.5, gain=full, offset=delta),
        out,
    )


def test_bad_arguments():
    y = torch.rand(16, 16)
    with pytest.raises(ValueError):
        pure_let(y, -1.0)
    with pytest.raises(ValueError):
        pure_let(y, 1.0, gain=0.0)
    with pytest.raises(ValueError):
        pure_let(y, 1.0, levels=0)
    with pytest.raises(ValueError):
        pure_let(torch.rand(16), 1.0)
    with pytest.raises(ValueError):
        U.let_basis(torch.rand(10, 10), 1.0, 2, halo=1)


# -- estimator and streaming -------------------------------------------------
def test_estimator_defaults():
    est = PURELET(sigma=1.0)
    assert est.is_faithful and est.options["levels"] == 4
    assert est.halo() == pure_let_halo(4) == 15
    assert PURELET(sigma=1.0, levels=3).is_faithful
    assert not PURELET(sigma=1.0, ndim=3).is_faithful
    assert not PURELET(sigma=1.0, pixelwise=True).is_faithful
    assert not PURELET(sigma=1.0, reliability=5).is_faithful
    assert "PURELET" in optix.estimators()
    with pytest.raises(ValueError):
        PURELET(sigma=-1.0)
    with pytest.raises(ValueError):
        PURELET().fit()  # no noise model
    with pytest.raises(RuntimeError):
        PURELET(sigma=1.0).apply(torch.rand(16, 16))
    with pytest.raises(ValueError):  # 3 spatial axes for 2-D data
        PURELET(sigma=1.0, ndim=3).fit().apply(torch.rand(16, 16))


def _frames(shape=(70, 83), seed=10):
    g = torch.Generator().manual_seed(seed)
    x = scene(shape, peak=30.0)
    y = observe(x, 3.0, g)
    return torch.stack([y, y.flip(-1).roll(5, 0)]) * 2 + 100


@pytest.mark.parametrize("block", [(20, 23), (9, 83), (70, 30)])
def test_blocks_match_whole(block):
    data = _frames()
    expected = pure_let(data, 6.0, gain=2.0, offset=100.0)
    est = PURELET(sigma=6.0, gain=2.0, offset=100.0, block=block, **F64)
    out = est.fit().apply(data.numpy())
    np.testing.assert_allclose(
        np.asarray(out), expected.numpy(), rtol=0, atol=1e-10
    )


def test_blocks_memory_budget():
    data = _frames()
    est = PURELET(sigma=6.0, gain=2.0, offset=100.0, **F64).fit()
    with optix.config(memory=2**21):
        out = est.apply(data)
    expected = pure_let(data, 6.0, gain=2.0, offset=100.0)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-10)


def test_each_frame_has_its_coefficients():
    data = _frames()
    out = PURELET(sigma=6.0, gain=2.0, offset=100.0, **F64).fit().apply(data)
    single = pure_let(data[1], 6.0, gain=2.0, offset=100.0)
    torch.testing.assert_close(out[1], single, rtol=0, atol=1e-10)


def test_small_image_wraps_several_times():
    """Halo (15) larger than the image: the periodic halo wraps around.

    The level-4 bands nearly coincide on so small an image (their
    filters wrap around it), so the PURE system is singular and only its
    well-determined part is solved: agreement to 1e-8, not 1e-10.
    """
    data = _frames((11, 9))
    est = PURELET(sigma=6.0, gain=2.0, offset=100.0, block=(4, 5), **F64)
    torch.testing.assert_close(
        est.fit().apply(data),
        pure_let(data, 6.0, gain=2.0, offset=100.0),
        rtol=0,
        atol=1e-8,
    )


def test_float32_matches_float64():
    data = _frames()
    ref = pure_let(data, 6.0, gain=2.0, offset=100.0)
    est = PURELET(
        sigma=6.0, gain=2.0, offset=100.0, block=(20, 23), dtype=torch.float32
    )
    out = est.fit().apply(data.float())
    assert out.dtype == torch.float32
    eps = torch.finfo(torch.float32).eps
    torch.testing.assert_close(
        out.double(), ref, rtol=0, atol=8 * eps * float(ref.abs().max())
    )


def test_sensor_estimate():
    data = _frames()
    sensor = SensorEstimate(offset=100.0, variance=36.0, gain=2.0)
    est = PURELET(block=(30, 40), **F64).fit(data, init=sensor)
    assert est.estimates["sensor"].gain == 2.0
    torch.testing.assert_close(
        est.apply(data),
        pure_let(data, 6.0, gain=2.0, offset=100.0),
        rtol=0,
        atol=1e-10,
    )


def _scmos(shape=(60, 64), seed=11):
    g = torch.Generator().manual_seed(seed)
    x = scene(shape, peak=25.0)
    offset = 100 + 10 * torch.rand(shape, generator=g, dtype=D)
    variance = 4 + 30 * torch.rand(shape, generator=g, dtype=D)
    gain = 1.5 + torch.rand(shape, generator=g, dtype=D)
    z = gain * torch.poisson(x, generator=g) + offset
    z = z + variance.sqrt() * torch.randn(shape, generator=g, dtype=D)
    sensor = SensorEstimate(offset=offset, variance=variance, gain=gain)
    return z, gain * x + offset, sensor


def test_pixelwise_maps():
    z, truth, sensor = _scmos()
    with pytest.raises(ValueError, match="pixelwise"):
        PURELET(**F64).fit(init=sensor)
    est = PURELET(pixelwise=True, block=(17, 30), **F64).fit(init=sensor)
    out = est.apply(z)
    expected = pure_let(
        z, sensor.variance.sqrt(), gain=sensor.gain, offset=sensor.offset
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-10)
    # denoising in photon units
    g = sensor.gain
    mse_in = float(((z - truth) / g).square().mean())
    mse_out = float(((out - truth) / g).square().mean())
    assert mse_out < mse_in / 8
    # constant maps are the scalar case (faithful, no flag needed)
    flat = SensorEstimate(
        offset=torch.full((60, 64), 100.0, **F64), variance=9.0, gain=2.0
    )
    a = PURELET(**F64).fit(init=flat).apply(z)
    b = pure_let(z, 3.0, gain=2.0, offset=100.0)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-10)


def test_pixelwise_pure_is_unbiased():
    """Per-pixel σ_n²: Stein's lemma per pixel keeps PURE unbiased."""
    g = torch.Generator().manual_seed(12)
    x = scene((48, 48), peak=20.0)
    s2 = 1 + 8 * torch.rand(48, 48, generator=g, dtype=D)
    levels, reps = 3, 40
    _, _, s0 = sums_of(x + s2.sqrt() * torch.randn_like(x), s2, levels)
    a = s0.solve()
    diffs = []
    for _ in range(reps):
        y = torch.poisson(x, generator=g)
        y = y + s2.sqrt() * torch.randn(x.shape, generator=g, dtype=D)
        f, low, sums = sums_of(y, s2, levels)
        diffs.append(
            sums.risk(a) - float((_combine(f, low, a) - x).square().mean())
        )
    d = torch.tensor(diffs, **F64)
    assert abs(float(d.mean())) < 3 * float(d.std()) / reps**0.5


def test_volumes_and_space_time():
    """ndim = 3: separable 3-D transform; blocks match the whole volume."""
    g = torch.Generator().manual_seed(13)
    x = scene((14, 20, 22), peak=20.0)
    v = observe(x, 1.0, g)
    expected = pure_let(v, 1.0, ndim=3, levels=2)
    assert psnr(expected, x, 20.0) > psnr(v, x, 20.0) + 5
    est = PURELET(sigma=1.0, ndim=3, levels=2, block=(5, 9, None), **F64)
    torch.testing.assert_close(
        est.fit().apply(v), expected, rtol=0, atol=1e-10
    )
    # space-time: time declared spatial in the layout
    stack = v.permute(1, 2, 0).contiguous()  # (y, x, t)
    lay = Layout("yxt", spatial="tyx")
    est = PURELET(
        sigma=1.0, ndim=3, levels=2, layout=lay, block=(7, None, 8), **F64
    )
    torch.testing.assert_close(
        est.fit().apply(stack),
        expected.permute(1, 2, 0),
        rtol=0,
        atol=1e-10,
    )


def test_lazy_zarr():
    zarr = pytest.importorskip("zarr")
    data = _frames((48, 52))[0]
    src = zarr.zeros(data.shape, chunks=(24, 26), dtype="f8")
    src[:] = data.numpy()
    dst = zarr.zeros(data.shape, chunks=(24, 26), dtype="f8")
    est = PURELET(sigma=6.0, gain=2.0, offset=100.0, block=(24, 26), **F64)
    est.fit(src).apply(src, out=dst)
    expected = pure_let(data, 6.0, gain=2.0, offset=100.0)
    np.testing.assert_allclose(dst[:], expected.numpy(), rtol=0, atol=1e-10)


def test_in_place_refused():
    data = np.random.default_rng(0).random((20, 20))
    with pytest.raises(ValueError):
        PURELET(sigma=0.1, **F64).fit().apply(data, out=data)


def test_save_load(tmp_path):
    z, _, sensor = _scmos((32, 32))
    est = PURELET(pixelwise=True, **F64).fit(init=sensor)
    path = tmp_path / "purelet.pt"
    est.save(path)
    back = optix.load(path)
    assert isinstance(back, PURELET) and back.options == est.options
    torch.testing.assert_close(
        back.apply(z).double(), est.apply(z), rtol=0, atol=1e-3
    )
