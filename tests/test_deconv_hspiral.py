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
    HessianSchatten,
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


def objective(x, y, otf, tau, p, boundary="periodic"):
    """Eq. (12) (without the constant), +inf if x has negative entries;
    H with the boundary of HSPIRAL2 ("periodic") or HSPIRAL1
    ("mirror")."""
    if bool((x < 0).any()):
        return math.inf
    kx = torch.fft.irfftn(torch.fft.rfftn(x) * otf, s=x.shape)
    fid = ShiftedPoisson(y).value(kx)
    h = Hessian(x.ndim, boundary=boundary).forward(x)
    return float(fid) + float(Schatten(tau, p).value(h))


def hessian_eq10(x):
    """Eq. (10) of Lefkimmiatis, Ward & Unser (2013) by indices, packed
    (D₁₁, D₂₂, √2·D₁₂)."""
    nx, ny = x.shape
    out = torch.zeros(3, nx, ny, dtype=x.dtype)
    for i in range(nx):
        for j in range(ny):
            if i <= nx - 3:
                out[0, i, j] = x[i + 2, j] - 2 * x[i + 1, j] + x[i, j]
            else:
                out[0, i, j] = x[nx - 2, j] - x[nx - 1, j]
            if j <= ny - 3:
                out[1, i, j] = x[i, j + 2] - 2 * x[i, j + 1] + x[i, j]
            else:
                out[1, i, j] = x[i, ny - 2] - x[i, ny - 1]
            if i <= nx - 2 and j <= ny - 2:
                out[2, i, j] = SQ2 * (
                    x[i + 1, j + 1] - x[i + 1, j] - x[i, j + 1] + x[i, j]
                )
    return out


def project_ball_svd(v, q):
    """Projection of packed 2 × 2 matrices onto the unit S_q ball by SVD
    (Eqs. 33–36 of Lefkimmiatis, Ward & Unser 2013)."""
    a, d, c = v.unbind(0)
    b = c / SQ2
    m = torch.stack([torch.stack([a, b], -1), torch.stack([b, d], -1)], -2)
    u, s, vh = torch.linalg.svd(m)
    if q == 2:
        s = s / s.norm(dim=-1, keepdim=True).clamp(min=1)
    elif q == math.inf:
        s = s.clamp(max=1)
    else:  # Eq. (36)
        s1, s2 = s[..., :1], s[..., 1:]
        gam = torch.where(
            s1 <= 1 - s2,
            torch.zeros_like(s1),
            torch.where(s1 <= 1 + s2, (s1 + s2 - 1) / 2, s1 - 1),
        )
        s = (s - gam).clamp(min=0)
    r = u @ torch.diag_embed(s) @ vh
    return torch.stack([r[..., 0, 0], r[..., 1, 1], SQ2 * r[..., 0, 1]])


# -- the paper's method -----------------------------------------------------
def test_defaults_and_paper_rules():
    psf = gaussian2d()
    x, y = simulate((48, 48), psf, 25.0, seed=1)
    est = HSPIRAL(psf, dtype=F64).fit(y)
    assert est.is_faithful
    assert est.options["p"] == 1
    assert est.options["variant"] == 1
    assert est.options["inner_iter"] == 5
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
    est = HSPIRAL(
        psf, tau=tau, p=p, variant=2, max_iter=n_iter, tol=None, dtype=F64
    )
    out = est.fit(y).apply(y)
    torch.testing.assert_close(out, x, rtol=0, atol=1e-9 * float(x.max()))
    assert est.history.extras["iterations"] == [float(n_iter)]
    assert x_true.shape == out.shape


@pytest.mark.parametrize("p", [1, 2, math.inf])
def test_algorithm1_written_out(p):
    """Algorithm 1 (HSPIRAL1) line by line (Eqs. 27, 28, 29, 23), with the
    prox of Eq. (28) by Algorithm 2 of Lefkimmiatis, Ward & Unser (2013)
    written out: Eq. (10) stencil by indices, its adjoint as the
    transpose of that matrix, P_{B∞,q} by SVD, 5 iterations from the
    previous Ω with the momentum restarted."""
    psf = gaussian2d(5, 1.2)
    _, y = simulate((12, 10), psf, 30.0, seed=2)
    tau = 0.05
    alpha = 60 * tau / float(y.max())
    shape = y.shape
    otf = transfer_function(psf, shape)
    q = {1: math.inf, 2: 2, math.inf: 1}[p]
    mat = torch.stack(  # the matrix of H (columns: H of the unit images)
        [
            hessian_eq10(e.reshape(shape)).flatten()
            for e in torch.eye(y.numel(), dtype=F64)
        ],
        1,
    )

    def h(v):
        return hessian_eq10(v)

    def ht(w):
        return (mat.T @ w.flatten()).reshape(shape)

    def k(v):
        return torch.fft.irfftn(torch.fft.rfftn(v) * otf, s=shape)

    def kt(v):
        return torch.fft.irfftn(torch.fft.rfftn(v) * otf.conj(), s=shape)

    def denoise(z, weight, omega):
        """Algorithm 2 of Lefkimmiatis, Ward & Unser (2013)."""
        psi, prev, t = omega.clone(), omega.clone(), 1.0
        for _ in range(5):
            omega = project_ball_svd(
                psi + h((z - weight * ht(psi)).clamp(min=0)) / (64 * weight),
                q,
            )
            t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
            psi = omega + (t - 1) / t_next * (omega - prev)
            prev, t = omega, t_next
        return (z - weight * ht(omega)).clamp(min=0), omega

    den = otf.abs().square() + 1
    x = y.clone()  # x⁰ = y
    s1 = torch.zeros(shape, dtype=F64)
    s2 = torch.zeros(shape, dtype=F64)
    omega = torch.zeros(3, *shape, dtype=F64)
    n_iter = 6
    for _ in range(n_iter):
        u1, u2 = k(x) + s1, x + s2
        w = u1 - 1 / alpha
        z1 = 0.5 * (w + torch.sqrt(w.square() + 4 * y / alpha))
        z2, omega = denoise(u2, tau / alpha, omega)
        v = kt(z1 - s1) + (z2 - s2)
        x = torch.fft.irfftn(torch.fft.rfftn(v) / den, s=shape)
        s1 = s1 + k(x) - z1
        s2 = s2 + x - z2
    est = HSPIRAL(psf, tau=tau, p=p, max_iter=n_iter, tol=None, dtype=F64)
    out = est.fit(y).apply(y)
    torch.testing.assert_close(out, x, rtol=0, atol=1e-9 * float(x.max()))
    assert est.history.extras["iterations"] == [float(n_iter)]


def test_same_minimiser_as_x_first_admm():
    """HSPIRAL (p = 2, z-first from x⁰ = y, α = 60τ/M) and the x-first
    ADMM of Ikoma et al. from zeros with another ρ reach the same
    minimiser of Eq. (12)."""
    psf = gaussian2d(7, 1.5)
    _, y = simulate((24, 24), psf, 40.0, seed=3)
    tau = 0.1
    est = HSPIRAL(
        psf, tau=tau, p=2, variant=2, tol=1e-11, max_iter=6000, dtype=F64
    )
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


@pytest.mark.parametrize(
    ("p", "variant"), [(1, 2), (math.inf, 2), (1, 1), (math.inf, 1)]
)
def test_optimality_certificate(p, variant):
    """At a converged solve, Eq. (12) (with the Hessian of the variant)
    is not lowered by feasible perturbations, including perturbations
    of the edge rows and columns, where the two Hessians differ
    (convexity makes this a global check)."""
    psf = gaussian2d(5, 1.0)
    _, y = simulate((20, 20), psf, 30.0, seed=4)
    tau = 0.1
    est = HSPIRAL(
        psf, tau=tau, p=p, variant=variant, tol=1e-12, max_iter=4000, dtype=F64
    )
    x = est.fit(y).apply(y).clamp(min=0)
    otf = transfer_function(psf, y.shape)
    boundary = "mirror" if variant == 1 else "periodic"
    best = objective(x, y, otf, tau, p, boundary)
    assert est.history.extras["primal_residual"][0] < 1e-5
    g = torch.Generator().manual_seed(5)
    edge = torch.zeros(y.shape, dtype=torch.bool)
    edge[-2:] = edge[:, -2:] = True
    for scale in (1e-3, 1e-2, 1e-1):
        for mask in (None, edge):
            for _ in range(15):
                dx = scale * torch.randn(x.shape, generator=g, dtype=F64)
                if mask is not None:
                    dx = dx * mask
                other = objective(
                    (x + dx).clamp(min=0), y, otf, tau, p, boundary
                )
                assert other >= best - 1e-7 * abs(best), (scale, other, best)
    if variant == 1:
        # the HSPIRAL2 minimiser is not optimal for the mirror Hessian
        x2 = HSPIRAL(
            psf, tau=tau, p=p, variant=2, tol=1e-12, max_iter=4000, dtype=F64
        )
        x2 = x2.fit(y).apply(y).clamp(min=0)
        assert objective(x2, y, otf, tau, p, boundary) > best


def test_hspiral1_structure_with_periodic_hessian():
    """Cross-solver: the splitting of HSPIRAL1 (z₂ = x, inexact inner
    prox, warm-started) with the periodic Hessian reaches the HSPIRAL2
    minimiser of the same problem."""
    psf = gaussian2d(5, 1.2)
    _, y = simulate((14, 12), psf, 40.0, seed=3)
    tau, p = 0.1, 1
    a = HSPIRAL(
        psf, tau=tau, p=p, variant=2, tol=1e-12, max_iter=5000, dtype=F64
    )
    a = a.fit(y).apply(y)
    otf = transfer_function(psf, y.shape)
    penalty = HessianSchatten(tau, p, ndim=2, n_iter=5, boundary="periodic")
    terms = [Term(otf, ShiftedPoisson(y)), Term(None, penalty)]
    b = admm_fft(
        terms,
        y.shape,
        rho=60 * tau / float(y.max()),
        x0=y,
        order="z-first",
        max_iter=5000,
        tol=1e-12,
    ).x
    scale = float(a.abs().max())
    torch.testing.assert_close(b, a, rtol=0, atol=2e-3 * scale)
    fa = objective(a.clamp(min=0), y, otf, tau, p)
    fb = objective(b.clamp(min=0), y, otf, tau, p)
    assert fb == pytest.approx(fa, rel=1e-6)


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


@pytest.mark.parametrize("variant", [1, 2])
def test_three_d(variant):
    """3-D, every p: the 3 × 3 Hessian (6 components), finite results
    that beat the data, and a converged solve that no feasible
    perturbation improves."""
    r = torch.arange(5, dtype=F64) - 2
    psf = torch.exp(
        -(r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None] ** 2)
        / 2
    )
    psf = psf / psf.sum()
    shape = (10, 12, 12)
    g = grids(shape)
    x = 2 + 40 * torch.exp(
        -sum((gi - n / 2) ** 2 for gi, n in zip(g, shape, strict=True)) / 8
    )
    x = x + 30 * ((g[1] - 3).abs() < 1.5) * ((g[2] - 4).abs() < 1.5)
    otf = transfer_function(psf, shape)
    mean = torch.fft.irfftn(torch.fft.rfftn(x) * otf, s=shape).clamp(min=0)
    y = torch.poisson(mean, generator=torch.Generator().manual_seed(8))
    boundary = "mirror" if variant == 1 else "periodic"
    for p in (1, 2, math.inf):
        assert HSPIRAL(psf, p=p, variant=variant).is_faithful
        est = HSPIRAL(psf, p=p, variant=variant, tau=0.01, max_iter=100)
        est.dtype = F64
        out = est.fit(y).apply(y)
        assert out.shape == y.shape and out.isfinite().all()
        mse = float((out - x).square().mean())
        assert mse < 0.6 * float((y - x).square().mean()), (p, mse)
    # optimality at a converged solve (p = 1, the closed-form 3 × 3 prox)
    y = y[2:8, 2:10, 2:10]
    otf = transfer_function(psf, y.shape)
    est = HSPIRAL(
        psf,
        p=1,
        variant=variant,
        tau=0.01,
        tol=1e-11,
        max_iter=3000,
        dtype=F64,
    )
    sol = est.fit(y).apply(y).clamp(min=0)
    best = objective(sol, y, otf, 0.01, 1, boundary)
    gen = torch.Generator().manual_seed(9)
    for scale in (1e-3, 1e-1):
        for _ in range(10):
            dx = scale * torch.randn(y.shape, generator=gen, dtype=F64)
            other = objective(
                (sol + dx).clamp(min=0), y, otf, 0.01, 1, boundary
            )
            assert other >= best - 1e-7 * abs(best), (scale, other, best)


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
        dict(variant=2),
        dict(variant=2, p=2),
        dict(halo=12),
        dict(block=(8, None)),
    ):
        assert HSPIRAL(psf, **opts).is_faithful, opts
    for opts in (
        dict(rho=0.1),
        dict(max_iter=10),
        dict(tol=1e-3),
        dict(inner_iter=10),
        dict(boundary="unknown"),
    ):
        assert not HSPIRAL(psf, **opts).is_faithful, opts
    for bad in (
        dict(p=3),
        dict(p="nuclear"),
        dict(variant=3),
        dict(variant="1"),
        dict(inner_iter=0),
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
