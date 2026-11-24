"""Tests of the exact Poisson–Gaussian likelihood and the primal–dual solver
of optix.optim (Chouzenoux et al. 2015)."""

import math

import pytest
import torch

from optix.deconv.utils import transfer_function
from optix.optim import (
    Box,
    Convolution,
    Gradient,
    GroupL2,
    Hessian,
    PoissonGaussian,
    ShiftedPoisson,
    Smooth,
    SquaredL2,
    Term,
    admm_fft,
    primal_dual,
    primal_dual_step,
)
from optix.special import poisson_normal_logpdf

F64 = torch.float64


def gaussian2d(n=7, s=1.5):
    r = torch.arange(n, dtype=F64) - n // 2
    g = torch.exp(-(r[:, None] ** 2 + r[None] ** 2) / (2 * s * s))
    return g / g.sum()


def blocky(shape, peak, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.full(shape, 0.2 * peak, dtype=F64)
    for _ in range(4):
        a = [int(torch.randint(0, n - 8, (1,), generator=gen)) for n in shape]
        x[a[0] : a[0] + 8, a[1] : a[1] + 6] += peak * 0.8
    return x


# -- the exact Poisson–Gaussian likelihood ---------------------------------
def test_poisson_gaussian_value_and_extension():
    gen = torch.Generator().manual_seed(0)
    y = torch.poisson(torch.full((50,), 6.0, dtype=F64), generator=gen)
    y = y + 2 * torch.randn(50, generator=gen, dtype=F64)
    var = 4.0
    f = PoissonGaussian(y, var)
    assert isinstance(f, Smooth)
    z = 0.1 + 10 * torch.rand(50, generator=gen, dtype=F64)
    # Eqs. (2.6)–(2.7) on z ≥ 0
    ref = -poisson_normal_logpdf(y, z, var).sum()
    assert float(f.value(z)) == pytest.approx(float(ref), rel=1e-13)
    # Eqs. (2.28)–(2.30) on z < 0, and ξ(0), η(0) at the junction
    xi0 = torch.exp((2 * y - 1) / (2 * var))
    eta0 = (1 - math.exp(-1 / var)) * torch.exp((2 * y - 1) / var)
    phi0 = -poisson_normal_logpdf(y, 0.0, var)
    u = -torch.rand(50, generator=gen, dtype=F64)
    quad = 0.5 * eta0 * u**2 + (1 - xi0) * u + phi0
    for i in range(5):
        one = PoissonGaussian(y[i : i + 1], var)
        assert float(one.value(u[i : i + 1])) == pytest.approx(
            float(quad[i]), rel=1e-12
        )
    torch.testing.assert_close(f.grad(torch.zeros(50, dtype=F64)), 1 - xi0)
    torch.testing.assert_close(f.grad(u), 1 - (xi0 - eta0 * u))
    # Lipschitz constant (Theorem 2.1)
    assert f.lipschitz == pytest.approx(float(eta0.max()), rel=1e-12)
    assert f.log_lipschitz == pytest.approx(math.log(float(eta0.max())))
    big = PoissonGaussian(torch.tensor([500.0]), 1.0)
    assert big.lipschitz == math.inf
    assert math.isfinite(big.log_lipschitz)
    with pytest.raises(ValueError, match="positive read-noise"):
        PoissonGaussian(y, 0.0)


def test_poisson_gaussian_gradient_convexity_and_lipschitz():
    """∇φ against central differences of φ, across 0; φ″ ≥ 0 and ≤ η(0)
    (Theorem 2.1); the quadratic extension is C¹ with matching second
    derivative at 0 (Eq. 2.28)."""
    var = 9.0
    for yv in (-6.0, 0.0, 3.0, 12.0, 25.0):
        y = torch.tensor([yv], dtype=F64)
        f = PoissonGaussian(y, var)
        eta0 = f.lipschitz
        z = torch.linspace(-3.0, 40.0, 431, dtype=F64)
        h = 1e-5
        g = torch.stack([f.grad(t.reshape(1)) for t in z]).squeeze(1)
        fd = torch.stack(
            [
                (f.value((t + h).reshape(1)) - f.value((t - h).reshape(1)))
                / (2 * h)
                for t in z
            ]
        )
        torch.testing.assert_close(g, fd, rtol=1e-6, atol=1e-6)
        # convexity and bounded curvature from the gradient's increments
        slope = (g[1:] - g[:-1]) / (z[1:] - z[:-1])
        assert float(slope.min()) >= -1e-9, yv
        assert float(slope.max()) <= eta0 * (1 + 1e-9), yv
        # second derivative continuous at 0: η just above 0 ≈ η(0)
        e = 1e-4
        d2 = float(
            (
                f.grad(torch.tensor([e], dtype=F64))
                - f.grad(torch.zeros(1, dtype=F64))
            )
            / e
        )
        assert d2 == pytest.approx(eta0, rel=1e-3)


def test_primal_dual_step():
    # Theorem 3.2 (iii): γ = (1 − ε)/δ, δ = μ + √∑‖V‖², ε < 1/(δ + 1)
    delta = 2.0 + math.sqrt(8 + 64)
    eps = min(1e-3, 0.5 / (delta + 1))
    assert primal_dual_step(2.0, [8.0, 64.0]) == pytest.approx(
        (1 - eps) / delta
    )
    assert primal_dual_step(0.0, [0.25], eps=0.1) == pytest.approx(0.9 / 0.5)
    assert primal_dual_step(math.inf, [8.0]) == 0.0
    with pytest.raises(ValueError):
        primal_dual_step(1.0, [8.0], eps=0.5)
    with pytest.raises(ValueError):
        primal_dual_step(0.0, [])


# -- primal_dual -----------------------------------------------------------
def test_tikhonov_closed_form():
    """Gaussian fidelity + quadratic gradient penalty: the Fourier filter."""
    n = (24, 20)
    gen = torch.Generator().manual_seed(1)
    otf = transfer_function(gaussian2d(5, 1.0), n)
    conv = Convolution(otf, n)
    y = torch.rand(n, generator=gen, dtype=F64) * 10
    lam = 0.7
    grad_op = Gradient(2)

    def grad(x):
        return conv.adjoint(conv.forward(x) - y[None])

    step = primal_dual_step(conv.norm2, [grad_op.norm2])
    res = primal_dual(
        grad,
        [(grad_op, SquaredL2(lam))],
        torch.zeros(n, dtype=F64),
        step=step,
        max_iter=20000,
        tol=1e-13,
    )
    sym = grad_op.gram(n)
    den = otf.real.square() + otf.imag.square() + lam * sym
    ref = torch.fft.irfftn(otf.conj() * torch.fft.rfftn(y) / den, s=n)
    torch.testing.assert_close(res.x, ref, rtol=0, atol=1e-8)
    assert res.residual < 1e-8


def test_cross_solver_with_admm():
    """Shifted Poisson (smooth, Lipschitz from Table 1) + TV + Hessian + box:
    primal_dual and admm_fft reach the same minimiser."""
    n = (24, 24)
    otf = transfer_function(gaussian2d(), n)
    conv = Convolution(otf, n)
    x = blocky(n, 10.0)
    mean = torch.fft.irfftn(torch.fft.rfftn(x) * otf, s=n)
    gen = torch.Generator().manual_seed(2)
    var = 9.0
    y = torch.poisson(mean, generator=gen) + 3 * torch.randn(
        n, generator=gen, dtype=F64
    )
    y = y.clamp(min=-var)
    fid = ShiftedPoisson(y, var)
    lt, lh = 0.05, 0.1
    ref = admm_fft(
        [
            Term(conv, fid),
            Term(None, Box(0.0)),
            Term(Gradient(2), GroupL2(lt)),
            Term(Hessian(2), GroupL2(lh)),
        ],
        n,
        rho=0.05,
        max_iter=20000,
        tol=1e-12,
    ).x.clamp(min=0)

    def grad(z):
        return conv.adjoint(fid.grad(conv.forward(z)))

    def value(z):
        return fid.value(conv.forward(z)[0])

    terms = [(Gradient(2), GroupL2(lt)), (Hessian(2), GroupL2(lh))]
    step = primal_dual_step(
        fid.lipschitz * conv.norm2, [op.norm2 for op, _ in terms]
    )
    trace = {}
    res = primal_dual(
        grad,
        terms,
        torch.full(n, float(y.mean()), dtype=F64),
        step=step,
        constraint=Box(0.0),
        max_iter=20000,
        tol=1e-9,
        value=value,
        trace=trace,
    )
    rel = float((res.x - ref).norm() / ref.norm())
    assert rel < 1e-5, rel
    assert float(res.x.min()) >= 0  # p₁ is in C
    assert len(trace["objective"]) == res.n_iter
    assert trace["objective"][-1] <= trace["objective"][10]
    # at the limit the dual variables are projections: ‖v_r‖ ≤ λ_r
    for v, lam in zip(res.duals, (lt, lh), strict=True):
        assert float(v.norm(dim=0).max()) <= lam * (1 + 1e-6)


def test_options():
    n = (8, 8)
    y = torch.ones(n, dtype=F64)

    def grad(x):
        return x - y

    calls = []
    res = primal_dual(
        grad,
        [],
        torch.zeros(n, dtype=F64),
        step=0.5,
        max_iter=7,
        tol=None,
        callback=lambda k, x: calls.append(k),
    )
    assert res.n_iter == 7 and calls == list(range(1, 8))
    assert res.duals == []
    res = primal_dual(
        grad, [], torch.zeros(n, dtype=F64), step=0.5, max_iter=500, tol=1e-6
    )
    torch.testing.assert_close(res.x, y, rtol=0, atol=1e-5)
    assert res.n_iter < 500 and res.change <= 1e-6
    # warm start of the duals
    op = Gradient(2)
    duals = [torch.full((2, *n), 0.01, dtype=F64)]
    res = primal_dual(
        grad,
        [(op, GroupL2(1.0))],
        y.clone(),
        step=0.1,
        duals=duals,
        max_iter=1,
        tol=None,
    )
    assert res.duals[0] is not duals[0]
    with pytest.raises(ValueError, match="one entry per term"):
        primal_dual(grad, [], y, step=0.1, duals=duals)
    for bad in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError, match="step"):
            primal_dual(grad, [], y, step=bad)
