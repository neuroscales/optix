"""Tests of optix.deconv.HSPIRAL (Lefkimmiatis & Unser 2013)."""

import math
import warnings

import pytest
import torch

from optix import load
from optix.deconv import HSPIRAL, hspiral
from optix.deconv.utils import PaddedConvolution, transfer_function
from optix.estimates import SensorEstimate
from optix.io import Layout
from optix.optim import (
    Box,
    Convolution,
    Gradient,
    GroupL2,
    Hessian,
    Schatten,
    ShiftedPoisson,
    Term,
    admm_fft,
)

F64 = torch.float64
SQ2 = math.sqrt(2)


def gaussian2d(n=9, s=2.0):
    r = torch.arange(n, dtype=F64) - n // 2
    g = torch.exp(-(r[:, None] ** 2 + r[None] ** 2) / (2 * s * s))
    return g / g.sum()


def grids(shape):
    return torch.meshgrid(
        *[torch.arange(n, dtype=F64) for n in shape], indexing="ij"
    )


def scene(shape, seed=0):
    """Piecewise-smooth: blobs, rectangles with ramps, a filament; max 1."""
    gen = torch.Generator().manual_seed(seed)
    g = grids(shape)
    x = torch.full(shape, 0.1, dtype=F64)
    for _ in range(4):
        a = [float(torch.rand(1, generator=gen)) * n for n in shape]
        w = [4 + float(torch.rand(1, generator=gen)) * n / 3 for n in shape]
        inside = torch.ones(shape, dtype=torch.bool)
        for gi, ai, wi in zip(g, a, w, strict=True):
            inside &= (gi >= ai) & (gi < ai + wi)
        ramp = 0.02 * (g[0] - a[0]) * float(torch.randn(1, generator=gen))
        x = torch.where(inside, 0.4 + ramp, x)
    for _ in range(6):
        c = [float(torch.rand(1, generator=gen)) * n for n in shape]
        rad = 1.5 + 3 * float(torch.rand(1, generator=gen))
        d2 = sum((gi - ci) ** 2 for gi, ci in zip(g, c, strict=True))
        x = x + torch.exp(-d2 / (2 * rad**2))
    x = x + 0.6 * torch.exp(-((g[0] - g[1] * 0.7 - 5) ** 2) / 1.5)
    x = x.clamp(min=0)
    return x / x.max()


def simulate(shape, psf, peak, seed=0, background=0.0):
    """x = peak·(scene + background) and Poisson data of its circular
    blur."""
    x = peak * (scene(shape, seed) + background)
    otf = transfer_function(psf, shape)
    mean = torch.fft.irfftn(torch.fft.rfftn(x) * otf, s=shape).clamp(min=0)
    gen = torch.Generator().manual_seed(1000 + seed)
    return x, torch.poisson(mean, generator=gen)


def psnr(est, x, peak):
    """The paper's PSNR, 10 log10(M² / MSE)."""
    return 10 * math.log10(peak**2 / float((est - x).square().mean()))


def prox_schatten_svd(v, t, p):
    """Schatten prox by SVD of the packed 2 × 2 matrices (reference)."""
    a, d, c = v.unbind(0)
    b = c / SQ2
    m = torch.stack([torch.stack([a, b], -1), torch.stack([b, d], -1)], -2)
    u, s, vh = torch.linalg.svd(m)
    if p == 1:
        s = (s - t).clamp(min=0)
    elif p == 2:
        n = s.norm(dim=-1, keepdim=True).clamp(min=1e-300)
        s = s * (1 - t / n).clamp(min=0)
    else:  # Eq. (46) at σ/t: min(σ, t·γ)
        s1, s2 = s[..., :1] / t, s[..., 1:] / t
        gam = torch.where(
            s1 <= 1 - s2,
            torch.zeros_like(s1),
            torch.where(s1 <= 1 + s2, (s1 + s2 - 1) / 2, s1 - 1),
        )
        s = torch.minimum(s, t * gam)
    r = u @ torch.diag_embed(s) @ vh
    return torch.stack([r[..., 0, 0], r[..., 1, 1], SQ2 * r[..., 0, 1]])


def objective(x, y, otf, tau, p):
    """Eq. (12) (without the constant), +inf if x has negative entries."""
    if bool((x < 0).any()):
        return math.inf
    kx = torch.fft.irfftn(torch.fft.rfftn(x) * otf, s=x.shape)
    fid = ShiftedPoisson(y).value(kx)
    return float(fid) + float(Schatten(tau, p).value(Hessian(2).forward(x)))


# -- the paper's method -----------------------------------------------------
def test_defaults_and_paper_rules():
    psf = gaussian2d()
    x, y = simulate((48, 48), psf, 25.0, seed=1)
    est = HSPIRAL(psf, dtype=F64).fit(y)
    assert est.is_faithful
    assert est.options["p"] == 1
    assert est.options["max_iter"] == 400
    assert est.options["tol"] == 1e-5
    assert est.options["boundary"] == "periodic"
    assert float(est.peak) == float(y.max())
    q = float(torch.quantile(y.flatten(), 0.999))
    assert float(est.robust_peak) == pytest.approx(q, rel=1e-12)
    med = float(torch.quantile(y.flatten(), 0.5))
    assert float(est.median) == pytest.approx(med, rel=1e-12)
    tau = 1.6 / (q - med)
    assert float(est.tau) == pytest.approx(tau)
    assert float(est.rho) == pytest.approx(60 * tau / float(y.max()))
    out = est.apply(y)
    extras = est.history.extras
    n = int(extras["iterations"][0])
    assert n == 400 or extras["change"][0] <= 1e-5
    assert psnr(out, x, 25.0) > psnr(y, x, 25.0) + 3
    assert HSPIRAL.tags.noise == "poisson"


@pytest.mark.parametrize("p", [1, 2, math.inf])
def test_algorithm2_written_out(p):
    """Algorithm 2 line by line (Eqs. 27, 32, 36/43–47 by SVD, 42)."""
    psf = gaussian2d(7, 1.5)
    x_true, y = simulate((24, 20), psf, 30.0, seed=2)
    tau = 0.05
    alpha = 60 * tau / float(y.max())
    shape = y.shape
    otf = transfer_function(psf, shape)
    hess = Hessian(2)

    def k(v):
        return torch.fft.irfftn(torch.fft.rfftn(v) * otf, s=shape)

    def kt(v):
        return torch.fft.irfftn(torch.fft.rfftn(v) * otf.conj(), s=shape)

    den = otf.abs().square() + hess.gram(shape) + 1
    x = y.clone()  # x⁰ = y
    s1 = torch.zeros(shape, dtype=F64)
    s2 = torch.zeros(3, *shape, dtype=F64)
    s3 = torch.zeros(shape, dtype=F64)
    n_iter = 6
    for _ in range(n_iter):
        u1, u2, u3 = k(x) + s1, hess.forward(x) + s2, x + s3
        w = u1 - 1 / alpha
        z1 = 0.5 * (w + torch.sqrt(w.square() + 4 * y / alpha))
        z2 = prox_schatten_svd(u2, tau / alpha, p)
        z3 = u3.clamp(min=0)
        v = kt(z1 - s1) + hess.adjoint(z2 - s2) + (z3 - s3)
        x = torch.fft.irfftn(torch.fft.rfftn(v) / den, s=shape)
        s1 = s1 + k(x) - z1
        s2 = s2 + hess.forward(x) - z2
        s3 = s3 + x - z3
    est = HSPIRAL(psf, tau=tau, p=p, max_iter=n_iter, tol=None, dtype=F64)
    out = est.fit(y).apply(y)
    torch.testing.assert_close(out, x, rtol=0, atol=1e-9 * float(x.max()))
    assert est.history.extras["iterations"] == [float(n_iter)]
    assert x_true.shape == out.shape


def test_same_minimiser_as_x_first_admm():
    """HSPIRAL (p = 2, z-first from x⁰ = y, α = 60τ/M) and the x-first
    ADMM of Ikoma et al. from zeros with another ρ reach the same
    minimiser of Eq. (12)."""
    psf = gaussian2d(7, 1.5)
    _, y = simulate((24, 24), psf, 40.0, seed=3)
    tau = 0.1
    est = HSPIRAL(psf, tau=tau, p=2, tol=1e-11, max_iter=6000, dtype=F64)
    a = est.fit(y).apply(y)
    otf = transfer_function(psf, y.shape)
    terms = [
        Term(otf, ShiftedPoisson(y)),
        Term(None, Box(0.0)),
        Term(Hessian(2), GroupL2(tau)),
    ]
    b = admm_fft(terms, y.shape, rho=0.2, max_iter=6000, tol=1e-11).x
    scale = float(b.abs().max())
    torch.testing.assert_close(a, b, rtol=0, atol=1e-3 * scale)
    fa = objective(a.clamp(min=0), y, otf, tau, 2)
    fb = objective(b.clamp(min=0), y, otf, tau, 2)
    assert fa == pytest.approx(fb, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("p", [1, math.inf])
def test_optimality_certificate(p):
    """At a converged solve, Eq. (12) is not lowered by feasible
    perturbations (convexity makes this a global check)."""
    psf = gaussian2d(5, 1.0)
    _, y = simulate((20, 20), psf, 30.0, seed=4)
    tau = 0.1
    est = HSPIRAL(psf, tau=tau, p=p, tol=1e-12, max_iter=4000, dtype=F64)
    x = est.fit(y).apply(y).clamp(min=0)
    otf = transfer_function(psf, y.shape)
    best = objective(x, y, otf, tau, p)
    assert est.history.extras["primal_residual"][0] < 1e-5
    g = torch.Generator().manual_seed(5)
    for scale in (1e-3, 1e-2, 1e-1):
        for _ in range(30):
            dx = scale * torch.randn(x.shape, generator=g, dtype=F64)
            other = objective((x + dx).clamp(min=0), y, otf, tau, p)
            assert other >= best - 1e-7 * abs(best), (scale, other, best)


def test_paper_setting_in_miniature():
    """Gaussian blur σ_b = 4 (9 × 9), peak 25, circular blur: each HS_p
    at its best τ beats the data and at least matches TV (same solver
    and α rule) at its best τ, minus 0.2 dB, as in Table I. The order
    among the three norms (within about 0.2 dB in Table I) is not
    checked: it is not stable on images this small."""
    psf = gaussian2d(9, 4.0)
    peak = 25.0
    x, y = simulate((64, 64), psf, peak, seed=6)
    taus = [0.01, 0.03, 0.1, 0.3]
    best = {}
    for p in (1, 2, math.inf):
        best[p] = max(
            psnr(HSPIRAL(psf, tau=t, p=p, dtype=F64).fit(y).apply(y), x, peak)
            for t in taus
        )
    otf = transfer_function(psf, y.shape)
    tv = []
    for t in taus:
        terms = [
            Term(Convolution(otf, y.shape), ShiftedPoisson(y)),
            Term(Gradient(2), GroupL2(t)),
            Term(None, Box(0.0)),
        ]
        rho = 60 * t / float(y.max())
        res = admm_fft(
            terms,
            y.shape,
            rho=rho,
            x0=y,
            order="z-first",
            max_iter=400,
            tol=1e-5,
        )
        tv.append(psnr(res.x, x, peak))
    data = psnr(y, x, peak)
    for p, value in best.items():
        assert value > data + 3, (p, value, data)
        assert value >= max(tv) - 0.2, (p, value, max(tv))


def test_default_tau_across_intensities_and_backgrounds():
    """The default τ (not in the paper) is close to the best τ of a
    grid at 10, 100 and 1000 photons, with and without a constant
    background, and better than the data."""
    psf = gaussian2d(9, 2.0)
    for peak, bg in (
        (10.0, 0.0),
        (100.0, 0.0),
        (1000.0, 0.0),
        (20.0, 3.0),
        (200.0, 10.0),
    ):
        x, y = simulate((64, 64), psf, peak, seed=7, background=bg)
        top = float(x.max())
        est = HSPIRAL(psf, dtype=F64).fit(y)
        default = psnr(est.apply(y), x, top)
        tau = float(est.tau)
        grid = [
            psnr(HSPIRAL(psf, tau=tau * f, dtype=F64).fit(y).apply(y), x, top)
            for f in (1 / 9, 1 / 3, 3, 9)
        ]
        assert default >= max(grid) - 1.5, (peak, bg, default, grid)
        assert default > psnr(y, x, top) + 1, (peak, bg, default)


def test_median_from_a_sample():
    """The median of `fit` is exact up to 8192 pixels and sampled above."""
    psf = gaussian2d(5, 1.0)
    gen = torch.Generator().manual_seed(21)
    y = torch.poisson(
        torch.rand(200, 200, generator=gen, dtype=F64) * 50, generator=gen
    )
    est = HSPIRAL(psf, max_iter=1, dtype=F64).fit(y)
    q = torch.quantile(y.flatten(), torch.tensor([0.45, 0.55], dtype=F64))
    assert float(q[0]) <= float(est.median) <= float(q[1])


def test_three_d_frobenius_only():
    r = torch.arange(5, dtype=F64) - 2
    psf = torch.exp(
        -(r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None] ** 2)
        / 2
    )
    gen = torch.Generator().manual_seed(8)
    y = torch.poisson(torch.full((8, 12, 12), 20.0, dtype=F64), generator=gen)
    est = HSPIRAL(psf, p=2, max_iter=20, dtype=F64)
    out = est.fit(y).apply(y)
    assert out.shape == y.shape and out.isfinite().all()
    assert HSPIRAL(psf, p=2).is_faithful
    for p in (1, "inf"):
        with pytest.raises(NotImplementedError, match="p=2"):
            HSPIRAL(psf, p=p)
        # a 3-D PSF set after construction fails at the start of fit
        est = HSPIRAL(p=p)
        est.psf = psf
        with pytest.raises(NotImplementedError, match="p=2"):
            est.fit(y)


def test_unknown_boundary_extension():
    """With different edges, the unobserved margin (extension) beats the
    periodic default at the edges; it is not faithful."""
    psf = gaussian2d(9, 1.6)
    full = 5 + 100 * scene((60, 60), 3)
    full[:, :30] += 60.0
    op = PaddedConvolution(psf, full.shape)
    mean = op.forward(op.pad(full))
    gen = torch.Generator().manual_seed(9)
    window = (slice(10, 50), slice(15, 45))
    y = torch.poisson(mean, generator=gen)[window]
    x = full[window]
    tau = 0.02
    per = HSPIRAL(psf, tau=tau, dtype=F64).fit(y).apply(y)
    est = HSPIRAL(psf, tau=tau, boundary="unknown", dtype=F64)
    assert not est.is_faithful
    unk = est.fit(y).apply(y)
    band = torch.zeros(y.shape, dtype=torch.bool)
    band[:, :4] = band[:, -4:] = True
    err = float((unk - x)[band].square().mean())
    err_per = float((per - x)[band].square().mean())
    assert err < 0.5 * err_per, (err, err_per)


def test_blocks_and_memory():
    psf = gaussian2d(7, 1.2)
    _, y = simulate((64, 60), psf, 50.0, seed=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        whole = HSPIRAL(psf, max_iter=50, dtype=F64).fit(y).apply(y)
    diffs = []
    for halo in (7, 21):
        est = HSPIRAL(psf, block=(32, 30), halo=halo, max_iter=50, dtype=F64)
        with pytest.warns(RuntimeWarning, match="blocks"):
            out = est.fit(y).apply(y)
        assert len(est.history.extras["iterations"]) == 4
        inner = (slice(16, 48), slice(15, 45))  # around the seams
        diffs.append(float((out - whole)[inner].abs().max()))
    assert diffs[1] < diffs[0]
    assert diffs[1] < 0.05 * float(whole.max()), diffs
    assert HSPIRAL(psf, block=(32, 30), halo=14).is_faithful
    # periodic: the FFT grid is the loaded block itself
    assert est._grid_shape((40, 44), (7, 7)) == (40, 44)
    unk = HSPIRAL(psf, boundary="unknown")
    assert unk._grid_shape((40, 44), (7, 7)) == (48, 50)
    with pytest.raises(ValueError, match="at least as large"):
        HSPIRAL(psf, max_iter=2).fit(y[:5, :5]).apply(y[:5, :5])


def test_float32_matches_float64():
    psf = gaussian2d(7, 1.5)
    for peak in (1.0, 100.0, 1e4):
        _, y = simulate((40, 40), psf, peak, seed=11)
        out = {}
        for dt in (torch.float32, F64):
            est = HSPIRAL(psf, dtype=dt, tol=None, max_iter=60)
            out[dt] = est.fit(y.to(dt)).apply(y.to(dt))
        scale = float(out[F64].abs().max())
        err = float((out[torch.float32].double() - out[F64]).abs().max())
        assert err < 1e-3 * scale, (peak, err, scale)


def test_sensor_units():
    psf = gaussian2d(7, 1.2)
    _, e = simulate((32, 36), psf, 20.0, seed=12)
    ref = HSPIRAL(psf, max_iter=40, dtype=F64).fit(e).apply(e)
    s = SensorEstimate(
        offset=torch.tensor(50.0, dtype=F64),
        gain=torch.tensor(3.0, dtype=F64),
        variance=torch.tensor(9.0, dtype=F64),  # not used (Poisson model)
    )
    raw = e * 3 + 50
    out = hspiral(raw, psf, sensor=s, max_iter=40, dtype=F64)
    torch.testing.assert_close(out, ref)
    # negative values (below the offset) are set to 0
    raw[0, 0] = 20.0
    est = HSPIRAL(psf, max_iter=2, dtype=F64).fit(raw, init=s)
    assert float(est.peak) == pytest.approx(float(e.max()))


def test_layout_and_state(tmp_path):
    psf = gaussian2d(5, 1.0)
    imgs = [simulate((24, 20), psf, 30.0, seed=s)[1] for s in range(3)]
    stack = torch.stack(imgs, dim=-1)
    layout = Layout("yxt", spatial="yx")
    est = HSPIRAL(psf, max_iter=20, dtype=F64, layout=layout)
    out = est.fit(stack).apply(stack)
    assert est.tau.shape == (3,)
    for i, img in enumerate(imgs):
        ref = HSPIRAL(psf, max_iter=20, dtype=F64).fit(img)
        torch.testing.assert_close(out[..., i], ref.apply(img))
    est.save(tmp_path / "hspiral.pt")
    new = load(tmp_path / "hspiral.pt")
    new.dtype = F64
    torch.testing.assert_close(new.apply(stack), out)
    torch.testing.assert_close(new.tau, est.tau)


def test_options_and_faithfulness():
    psf = gaussian2d(5, 1.0)
    assert HSPIRAL(psf).is_faithful
    # the faithful=False preset uses the unobserved margin
    rec = HSPIRAL(psf, faithful=False)
    assert rec.options["boundary"] == "unknown" and not rec.is_faithful
    assert HSPIRAL(psf, boundary="periodic", faithful=False).is_faithful
    for opts in (
        dict(tau=0.01),
        dict(p=2),
        dict(p="inf"),
        dict(p=math.inf),
        dict(halo=12),
        dict(block=(8, None)),
    ):
        assert HSPIRAL(psf, **opts).is_faithful, opts
    for opts in (
        dict(rho=0.1),
        dict(max_iter=10),
        dict(tol=1e-3),
        dict(boundary="unknown"),
    ):
        assert not HSPIRAL(psf, **opts).is_faithful, opts
    for bad in (
        dict(p=3),
        dict(p="nuclear"),
        dict(boundary="mirror"),
        dict(tau=-1.0),
        dict(rho=0.0),
        dict(max_iter=0),
        dict(tol=-1.0),
    ):
        with pytest.raises(ValueError):
            HSPIRAL(psf, **bad)
    with pytest.raises(ValueError, match="positive maximum"):
        HSPIRAL(psf).fit(torch.zeros(16, 16))
    with pytest.raises(ValueError, match="needs a psf"):
        HSPIRAL().fit(torch.ones(16, 16))
