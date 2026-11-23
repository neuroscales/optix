"""Tests of optix.optim (proximity operators, differences, ADMM)."""

import math

import pytest
import torch

from optix.deconv.utils import linear_filter, transfer_function
from optix.optim import (
    Box,
    Gradient,
    GroupL2,
    Hessian,
    ShiftedPoisson,
    SquaredL2,
    Term,
    admm_fft,
    diff,
    project_group_l2,
    prox_box,
    prox_conjugate,
    prox_group_l2,
    prox_shifted_poisson,
    prox_squared_l2,
)

F64 = torch.float64


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


# -- proximity operators ----------------------------------------------------
@pytest.mark.parametrize("var", [0.0, 4.0])
def test_prox_shifted_poisson_stationarity(var):
    """1 − c/(z + σ²) + ρ(z − v) = 0 at the prox, c = y + σ² > 0."""
    g = gen(1)
    v = 50 * torch.randn(2000, generator=g, dtype=F64)
    y = torch.poisson(20 * torch.rand(2000, generator=g, dtype=F64)) - 3
    y = y.clamp(min=-var + 0.5) if var > 0 else y.clamp(min=0.5)
    for t in (1e-3, 0.37, 50.0):
        z = prox_shifted_poisson(v, t, y, var)
        assert (z + var > 0).all()
        a, b = (y + var) / (z + var), (z - v) / t
        res = (1 - a + b) / (1 + a.abs() + b.abs())
        assert float(res.abs().max()) < 1e-10


def test_prox_shifted_poisson_brute_force():
    """Scalar oracle: minimum of f(z) + (z − v)²/(2t) on a fine grid."""
    var = 2.25
    for y, v, t in [
        (3.0, 1.0, 0.5),
        (0.0, 2.0, 1.0),
        (-2.0, -1.0, 2.0),
        (40.0, -5.0, 0.1),
        (1.0, 30.0, 3.0),
    ]:
        z = float(
            prox_shifted_poisson(
                torch.tensor(v, dtype=F64), t, torch.tensor(y, dtype=F64), var
            )
        )
        grid = torch.linspace(-var + 1e-9, 80, 2_000_001, dtype=F64)
        c = y + var
        obj = grid - c * torch.log(grid + var) + (grid - v) ** 2 / (2 * t)
        best = float(grid[obj.argmin()])
        assert abs(z - best) < 1e-4, (y, v, t, z, best)


def test_prox_shifted_poisson_matches_s17():
    """Stable form = literal Ikoma et al. (2018) Eq. (S17), benign inputs."""
    g = gen(2)
    v = 10 * torch.randn(500, generator=g, dtype=F64)
    y = 10 * torch.rand(500, generator=g, dtype=F64)
    var, rho = 2.0, 0.7
    b = (1 - rho * (v - var)) / (2 * rho)
    s17 = -b + torch.sqrt(b**2 + (y + rho * var * v) / rho)
    torch.testing.assert_close(
        prox_shifted_poisson(v, 1 / rho, y, var), s17, rtol=1e-10, atol=1e-10
    )


def test_prox_shifted_poisson_extremes_float32():
    """ρs ≫ 1, ρs ≪ 1, c = 0, σ² = 0 in float32: finite, z + σ² ≥ 0."""
    f32 = torch.float32
    v = torch.tensor([1e6, -1e6, 0.0, 1e-6, 5.0, -5.0, 1e4, 0.0], dtype=f32)
    y = torch.tensor([0.0, 0.0, 0.0, 1e4, 1e-3, 1e3, 1e4, 1.0], dtype=f32)
    for var in (0.0, 9.0):
        yy = y.clamp(min=-var)
        for t in (1e-6, 1.0, 1e6):
            z = prox_shifted_poisson(v, t, yy, var)
            assert z.isfinite().all()
            assert (z + var >= 0).all()
            ref = prox_shifted_poisson(v.double(), t, yy.double(), var)
            torch.testing.assert_close(
                z.double(), ref, rtol=1e-4, atol=1e-4 * (1 + ref.abs().max())
            )


def test_prox_group_l2_and_moreau():
    g = gen(3)
    v = torch.randn(3, 20, 20, generator=g, dtype=F64)
    t = 0.8
    z = prox_group_l2(v, t, dim=0)
    norm = v.norm(dim=0, keepdim=True)
    ref = torch.where(norm > t, v * (1 - t / norm), 0)
    torch.testing.assert_close(z, ref)
    # Moreau: the conjugate prox of t‖·‖ is the projection on the t-ball
    conj = prox_conjugate(lambda x, s: prox_group_l2(x, s * t, dim=0), v, 1.0)
    torch.testing.assert_close(conj, project_group_l2(v, t, dim=0))
    torch.testing.assert_close(z + project_group_l2(v, t, dim=0), v)
    # stationarity of GroupL2.prox where it is not zero
    f = GroupL2(t)
    z = f.prox(v, 0.5)
    nz = z.norm(dim=0, keepdim=True) > 0
    res = (z - v) / 0.5 + t * z / z.norm(dim=0, keepdim=True).clamp(min=1e-300)
    assert float((res * nz).abs().max()) < 1e-12


def test_box_and_squared_l2():
    v = torch.linspace(-3, 3, 13, dtype=F64)
    torch.testing.assert_close(prox_box(v), v.clamp(min=0))
    torch.testing.assert_close(prox_box(v, -1.0, 2.0), v.clamp(-1, 2))
    assert float(Box().value(v)) == math.inf
    assert float(Box().value(v.clamp(min=0))) == 0.0
    y = torch.ones_like(v)
    z = prox_squared_l2(v, 0.5, y, 2.0)
    torch.testing.assert_close((z - v) / 0.5 + 2.0 * (z - y), 0 * v)
    torch.testing.assert_close(
        SquaredL2(2.0, y).value(y), torch.tensor(0.0, dtype=F64)
    )


def test_shifted_poisson_function():
    y = torch.tensor([0.0, 3.0, -1.0], dtype=F64)
    f = ShiftedPoisson(y, 2.0)
    z = torch.tensor([1.0, 2.0, 0.5], dtype=F64)
    ref = (z - (y + 2) * torch.log(z + 2)).sum()
    torch.testing.assert_close(f.value(z), ref)
    eps = 1e-6
    num = (f.value(z + eps) - f.value(z - eps)) / (2 * eps)
    torch.testing.assert_close(
        float(num), float(f.grad(z).sum()), rtol=1e-6, atol=1e-8
    )
    assert f.lipschitz == pytest.approx(float(((y + 2) / 4).max()))
    assert ShiftedPoisson(y.clamp(min=0)).lipschitz == math.inf
    assert float(f.value(torch.tensor([-3.0, 0, 0], dtype=F64))) == math.inf


# -- difference operators ---------------------------------------------------
OPS = [Gradient, Hessian]


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("boundary", ["periodic", "neumann"])
@pytest.mark.parametrize("cls", OPS)
def test_dot_product(cls, boundary, ndim):
    op = cls(ndim, weights=[1.0, 0.5, 2.0][:ndim], boundary=boundary)
    g = gen(4)
    x = torch.randn((7, 6, 5)[:ndim], generator=g, dtype=F64)
    y = torch.randn((op.components, *x.shape), generator=g, dtype=F64)
    lhs = (op.forward(x) * y).sum()
    rhs = (x * op.adjoint(y)).sum()
    torch.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("cls", OPS)
def test_symbols(cls, ndim):
    op = cls(ndim, weights=[1.0, 0.7, 1.3][:ndim])
    g = gen(5)
    shape = (8, 9, 6)[:ndim]
    x = torch.randn(shape, generator=g, dtype=F64)
    spec = torch.fft.rfftn(x)
    sym = op.symbols(shape)
    got = torch.stack([torch.fft.irfftn(s * spec, s=shape) for s in sym])
    torch.testing.assert_close(got, op.forward(x))
    gram = op.gram(shape)
    ref = sum(s.real**2 + s.imag**2 for s in sym)
    torch.testing.assert_close(gram, torch.broadcast_to(ref, gram.shape))
    with pytest.raises(ValueError):
        cls(ndim, boundary="neumann").symbols(shape)


@pytest.mark.parametrize("ndim", [2, 3])
def test_hessian_frobenius(ndim):
    """‖(Hx)ⱼ‖₂ is the Frobenius norm of the full Hessian matrix."""
    g = gen(6)
    x = torch.randn((6, 7, 5)[:ndim], generator=g, dtype=F64)
    dims = list(range(ndim))
    full = torch.stack(
        [torch.stack([diff(diff(x, b), a) for b in dims]) for a in dims]
    )
    ref = full.square().sum(dim=(0, 1)).sqrt()
    got = Hessian(ndim).forward(x).norm(dim=0)
    torch.testing.assert_close(got, ref)


def test_hessian_of_quadratic():
    """Neumann, interior: the Hessian of a quadratic is its coefficients."""
    i, j, k = torch.meshgrid(
        *[torch.arange(n, dtype=F64) for n in (8, 9, 7)], indexing="ij"
    )
    x = 1.5 * i**2 - 0.5 * j**2 + 2 * k**2 + 0.7 * i * j - 1.1 * j * k + 3 * i
    h = Hessian(3, boundary="neumann").forward(x)[:, :-2, :-2, :-2]
    s2 = math.sqrt(2)
    ref = [3.0, -1.0, 4.0, s2 * 0.7, 0.0, s2 * -1.1]
    for c, r in enumerate(ref):
        torch.testing.assert_close(h[c], torch.full_like(h[c], r))


@pytest.mark.parametrize("cls", OPS)
def test_norm_bound(cls):
    op = cls(3, weights=[0.5, 1.0, 1.0])
    g = gen(7)
    x = torch.randn(8, 8, 8, generator=g, dtype=F64)
    for _ in range(300):
        x = op.adjoint(op.forward(x))
        x = x / x.norm()
    est = float(op.forward(x).norm() ** 2)
    assert est <= op.norm2 * (1 + 1e-9)
    assert est >= 0.99 * op.norm2
    exact = float(op.gram((8, 8, 8)).max())
    assert exact == pytest.approx(op.norm2)


def test_stencil_option():
    assert Hessian(2, stencil="forward").stencil == "forward"
    with pytest.raises(ValueError):
        Hessian(2, stencil="central")


# -- ADMM -------------------------------------------------------------------
def tikhonov_problem():
    g = gen(8)
    psf = torch.rand(5, 5, generator=g, dtype=F64)
    psf /= psf.sum()
    grid = (16, 18)
    y = torch.randn(grid, generator=g, dtype=F64)
    return transfer_function(psf, grid), y, grid


@pytest.mark.parametrize("order", ["x-first", "z-first"])
def test_admm_tikhonov_closed_form(order):
    """Gaussian fidelity + quadratic penalty = the Tikhonov filter."""
    otf, y, grid = tikhonov_problem()
    lam = 0.3
    terms = [Term(otf, SquaredL2(1.0, y)), Term(None, SquaredL2(lam))]
    res = admm_fft(terms, grid, rho=1.0, order=order, max_iter=2000, tol=1e-13)
    ref = torch.fft.irfftn(
        linear_filter(otf, lam) * torch.fft.rfftn(y), s=grid
    )
    torch.testing.assert_close(res.x, ref, rtol=0, atol=1e-6)
    assert res.n_iter < 2000


def test_admm_trace_and_fixed_iterations():
    otf, y, grid = tikhonov_problem()
    terms = [Term(otf, SquaredL2(1.0, y)), Term(None, SquaredL2(0.1))]
    trace: dict = {}
    seen = []
    res = admm_fft(
        terms,
        grid,
        rho=1.0,
        max_iter=7,
        tol=None,
        trace=trace,
        callback=lambda k, x: seen.append(k),
    )
    assert res.n_iter == 7 and seen == list(range(1, 8))
    assert len(trace["change"]) == 7 and len(trace["residual"]) == 7
    assert math.isnan(trace["change"][0])  # x⁰ = 0: not measured
    assert len(res.splits) == 2 and res.splits[0].shape == (1, *grid)


def test_admm_needs_invertible_system():
    with pytest.raises(ValueError, match="identity"):
        admm_fft([Term(Gradient(2), GroupL2(1.0))], (8, 8), rho=1.0, dtype=F64)
    otf, y, grid = tikhonov_problem()
    with pytest.raises(ValueError):
        admm_fft([Term(otf, SquaredL2(1.0, y))], grid, rho=-1.0)


def poisson_problem(var, seed=9):
    g = gen(seed)
    grid, block = (20, 22), (14, 15)
    psf = torch.exp(-((torch.arange(5, dtype=F64) - 2) ** 2) / 2)
    psf = psf[:, None] * psf[None]
    psf /= psf.sum()
    otf = transfer_function(psf, grid)
    x = 5 + 20 * torch.rand(grid, generator=g, dtype=F64)
    ax = torch.fft.irfftn(otf * torch.fft.rfftn(x), s=grid)
    region = tuple(slice(0, n) for n in block)
    y = torch.poisson(ax[region], generator=g)
    if var:
        y = y + math.sqrt(var) * torch.randn(y.shape, generator=g, dtype=F64)
    y = y.clamp(min=-var)
    return otf, y, grid, region


def objective(x, otf, fid, region, nu):
    grid = x.shape
    ax = torch.fft.irfftn(otf * torch.fft.rfftn(x), s=grid)
    val = fid.value(ax[region])
    if nu:
        val = val + GroupL2(nu).value(Hessian(2).forward(x))
    return float(val)


def test_admm_optimality_certificate():
    """Shifted Poisson + Hessian + box: no feasible perturbation is better."""
    otf, y, grid, region = poisson_problem(var=4.0)
    nu = 0.5
    fid = ShiftedPoisson(y, 4.0)
    terms = [
        Term(otf, fid, region),
        Term(None, Box(0.0)),
        Term(Hessian(2), GroupL2(nu)),
    ]
    res = admm_fft(terms, grid, rho=0.05, max_iter=5000, tol=1e-10)
    x = res.x.clamp(min=0)
    best = objective(x, otf, fid, region, nu)
    g = gen(10)
    for scale in (1e-1, 1e-2):
        for _ in range(50):
            d = scale * torch.randn(grid, generator=g, dtype=F64)
            other = objective((x + d).clamp(min=0), otf, fid, region, nu)
            assert best <= other + 1e-7 * abs(best)
    # primal residuals ‖Aₖx − zₖ‖ are small
    for term, z in zip(terms, res.splits, strict=True):
        if term.operator is None:
            ax = res.x[None]
        elif isinstance(term.operator, torch.Tensor):
            ax = torch.fft.irfftn(otf * torch.fft.rfftn(res.x), s=grid)[None]
        else:
            ax = term.operator.forward(res.x)
        assert float((ax - z).norm()) < 1e-6 * float(res.x.norm())


def test_admm_poisson_ml_is_rl_fixed_point():
    """σ² = 0, ν = 0: x·(1 − Aᵀ(y/Ax)/Aᵀ1) = 0 (KKT of Poisson ML, x ≥ 0).

    A mild periodic blur, so that the ML problem is well conditioned.
    """
    g = gen(11)
    grid = (16, 18)
    psf = torch.tensor([0.1, 0.8, 0.1], dtype=F64)
    psf = psf[:, None] * psf[None]
    otf = transfer_function(psf, grid)
    truth = 2 + 20 * torch.rand(grid, generator=g, dtype=F64)
    truth[4:8, 3:9] = 0.0
    ax = torch.fft.irfftn(otf * torch.fft.rfftn(truth), s=grid)
    y = torch.poisson(ax.clamp(min=0), generator=g)
    terms = [Term(otf, ShiftedPoisson(y)), Term(None, Box(0.0))]
    res = admm_fft(terms, grid, rho=0.1, max_iter=5000, tol=1e-12)
    x = res.x.clamp(min=0)

    def fwd(v):
        return torch.fft.irfftn(otf * torch.fft.rfftn(v), s=grid)

    def adj(r):
        return torch.fft.irfftn(otf.conj() * torch.fft.rfftn(r), s=grid)

    grad = 1 - adj(y / fwd(x).clamp(min=1e-300)) / adj(torch.ones_like(y))
    assert float((x * grad).abs().max()) < 1e-6 * float(x.max())
    zero = x == 0
    assert zero.any()  # the constraint is active somewhere
    assert float(grad[zero].min()) > -1e-6
