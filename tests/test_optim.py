"""Tests of optix.optim (proximity operators, differences, ADMM)."""

import math
import random

import pytest
import torch

from optix.deconv.utils import linear_filter, transfer_function
from optix.optim import (
    Box,
    Circulant,
    Convolution,
    Gradient,
    GroupL2,
    Hessian,
    Identity,
    LinearOperator,
    Schatten,
    ShiftedPoisson,
    SquaredL2,
    Term,
    admm_fft,
    as_circulant,
    diff,
    project_group_l2,
    prox_box,
    prox_conjugate,
    prox_group_l2,
    prox_schatten,
    prox_shifted_poisson,
    prox_squared_l2,
    schatten_norm,
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


# -- Schatten norms ---------------------------------------------------------
SQ2 = math.sqrt(2)


def unpack2(v):
    """Packed (a, d, √2·b) → 2 × 2 matrices `(..., 2, 2)`."""
    a, d, c = v.unbind(0)
    b = c / SQ2
    return torch.stack([torch.stack([a, b], -1), torch.stack([b, d], -1)], -2)


def pack2(x):
    return torch.stack([x[..., 0, 0], x[..., 1, 1], SQ2 * x[..., 0, 1]])


def project_l1_ball(s, radius):
    """Sort-based projection of each row of s ≥ 0 onto the ℓ1 ball."""
    u = torch.sort(s, dim=-1, descending=True).values
    cs = u.cumsum(-1)
    k = torch.arange(1, s.shape[-1] + 1, dtype=s.dtype)
    rank = (u - (cs - radius) / k > 0).sum(-1, keepdim=True).clamp(min=1)
    theta = (cs.gather(-1, rank - 1) - radius) / rank
    inside = s.sum(-1, keepdim=True) <= radius
    return torch.where(inside, s, (s - theta).clamp(min=0))


def prox_schatten_svd(v, t, p):
    """Reference: SVD, ℓ_p prox of the singular values, reconstruction.

    ℓ∞ by Moreau: prox_{t‖·‖∞}(σ) = σ − proj_{t·ℓ1 ball}(σ).
    """
    u, s, vh = torch.linalg.svd(unpack2(v))
    if p == 1:
        s = (s - t).clamp(min=0)
    elif p == 2:
        n = s.norm(dim=-1, keepdim=True)
        s = s * (1 - t / n.clamp(min=1e-300)).clamp(min=0)
    else:
        s = s - project_l1_ball(s, t)
    return pack2(u @ torch.diag_embed(s) @ vh)


def symmetric_fields(n=3000, seed=11):
    """Random packed 2 × 2 fields with the edge cases of the closed form."""
    v = 2 * torch.randn(3, n, generator=gen(seed), dtype=F64)
    v[:, :50] = 0  # zero matrices
    v[1, 50:150] = v[0, 50:150]  # equal eigenvalues: a = d, b = 0
    v[2, 50:150] = 0
    v[1, 150:250] = -v[0, 150:250]  # λ₋ = −λ₊ (trace 0)
    v[2, 150:250] = 0
    v[1, 250:350] = v[0, 250:350] + 1e-9  # nearly equal eigenvalues
    v[0, 350:450] = v[0, 350:450].abs() + 3  # positive definite
    v[1, 350:450] = v[0, 350:450]
    v[2, 350:450] *= 0.1
    return v


@pytest.mark.parametrize("p", [1, 2, math.inf])
def test_prox_schatten_matches_svd(p):
    v = symmetric_fields()
    for t in (0.0, 0.05, 1.0, 4.0):
        z = prox_schatten(v, t, p)
        torch.testing.assert_close(
            z, prox_schatten_svd(v, t, p), rtol=0, atol=1e-12
        )
    # tensor threshold broadcasting over voxels
    t = torch.rand(1, v.shape[1], generator=gen(2), dtype=F64)
    ref = torch.stack(
        [
            prox_schatten_svd(v[:, i : i + 1], float(t[0, i]), p)[:, 0]
            for i in range(0, v.shape[1], 97)
        ],
        1,
    )
    torch.testing.assert_close(
        prox_schatten(v, t, p)[:, ::97], ref, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("p", [1, 2, math.inf])
def test_schatten_norm_and_function(p):
    v = symmetric_fields(500, seed=12)
    s = torch.linalg.svdvals(unpack2(v))
    ref = {1: s.sum(-1), 2: s.norm(dim=-1), math.inf: s.amax(-1)}[p]
    torch.testing.assert_close(schatten_norm(v, p), ref)
    f = Schatten(0.7, p)
    assert float(f.value(v)) == pytest.approx(0.7 * float(ref.sum()))
    torch.testing.assert_close(f.prox(v, 2.0), prox_schatten(v, 1.4, p))
    # optimality: the prox beats random perturbations of itself
    y, t = v[:, :200], 0.9
    z = prox_schatten(y, t, p)

    def obj(x):
        return 0.5 * (x - y).square().sum(0) + t * schatten_norm(x, p)

    best = obj(z)
    g = gen(13)
    for scale in (1e-3, 1e-1):
        dz = scale * torch.randn(20, *z.shape, generator=g, dtype=F64)
        assert bool((obj_batch(obj, z + dz) >= best - 1e-12).all())


def obj_batch(obj, xs):
    return torch.stack([obj(x) for x in xs])


def test_prox_schatten_special_cases():
    g = gen(14)
    # p = 2 is the block soft threshold, in any dimension
    for c in (1, 3, 6):
        v = torch.randn(c, 30, 30, generator=g, dtype=F64)
        torch.testing.assert_close(
            prox_schatten(v, 0.8, 2), prox_group_l2(v, 0.8)
        )
    # 1 × 1 matrices: soft threshold for every p
    v = torch.randn(1, 50, generator=g, dtype=F64)
    soft = torch.sign(v) * (v.abs() - 0.3).clamp(min=0)
    for p in (1, math.inf):
        torch.testing.assert_close(prox_schatten(v, 0.3, p), soft)
        torch.testing.assert_close(schatten_norm(v, p), v.abs()[0])
    # components along another dim
    v = torch.randn(8, 3, generator=g, dtype=F64)
    torch.testing.assert_close(
        prox_schatten(v, 0.5, 1, dim=1), prox_schatten(v.T, 0.5, 1).T
    )
    # 4 × 4 with p in {1, inf}: not implemented; bad p and C rejected
    v6 = torch.randn(6, 4, generator=g, dtype=F64)
    v10 = torch.randn(10, 4, generator=g, dtype=F64)
    for p in (1, math.inf):
        with pytest.raises(NotImplementedError):
            prox_schatten(v10, 0.1, p)
        with pytest.raises(NotImplementedError):
            schatten_norm(v10, p)
        with pytest.raises(NotImplementedError):
            Schatten(1.0, p, ndim=4)
        Schatten(1.0, p, ndim=3)  # 3 × 3: closed form
    assert Schatten(1.0, "inf").p == math.inf
    with pytest.raises(ValueError, match="p must be"):
        prox_schatten(v6, 0.1, 3)
    with pytest.raises(ValueError, match="symmetric"):
        prox_schatten(torch.zeros(4, 2, dtype=F64), 0.1, 2)


@pytest.mark.parametrize("p", [1, math.inf])
def test_prox_schatten_float32(p):
    v = symmetric_fields(2000, seed=15)
    for t in (0.01, 1.0):
        z = prox_schatten(v.float(), t, p).double()
        ref = prox_schatten(v, t, p)
        err = float((z - ref).abs().max())
        assert err < 1e-5 * (1 + float(v.abs().max())), err


def test_hessian_schatten_of_quadratic():
    """On x = ½ uᵀ A u (periodic interior), ‖H x‖_{S_p} = ‖A‖_{S_p}."""
    a = torch.tensor([[2.0, -1.5], [-1.5, 0.5]], dtype=F64)
    u = torch.stack(
        torch.meshgrid(*[torch.arange(12, dtype=F64)] * 2, indexing="ij")
    )
    x = 0.5 * torch.einsum("i...,ij,j...->...", u, a, u)
    h = Hessian(2, boundary="neumann").forward(x)[:, :8, :8]
    s = torch.linalg.svdvals(a)
    for p, ref in ((1, s.sum()), (2, s.norm()), (math.inf, s.max())):
        norm = schatten_norm(h, p)
        torch.testing.assert_close(norm, torch.full_like(norm, float(ref)))


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


def _python_poisson_data(seed, grid, fwd):
    """Truth with a zero block and Poisson data, from Python's RNG.

    `random.Random` is the same on every platform and Python version, so
    the problem does not depend on torch's samplers.
    """
    rng = random.Random(seed)
    truth = torch.tensor(
        [
            [2 + 20 * rng.random() for _ in range(grid[1])]
            for _ in range(grid[0])
        ],
        dtype=F64,
    )
    truth[4:8, 3:9] = 0.0
    mean = fwd(truth).clamp(min=0)

    def poisson(lam):  # inversion
        u, k, p = rng.random(), 0, math.exp(-lam)
        c = p
        while u > c:
            k += 1
            p *= lam / k
            c += p
        return k

    return torch.tensor(
        [[poisson(float(m)) for m in row] for row in mean], dtype=F64
    )


def test_admm_poisson_ml_is_rl_fixed_point():
    """σ² = 0, ν = 0: x·(1 − Aᵀ(y/Ax)/Aᵀ1) = 0 (KKT of Poisson ML, x ≥ 0).

    A mild periodic blur, so that the ML problem is well conditioned.
    The data come from Python's RNG (platform independent). The KKT
    conditions are checked at z₂ = x₊, which is exactly 0 on the active
    set, after a solve that must have converged.
    """
    grid = (16, 18)
    psf = torch.tensor([0.1, 0.8, 0.1], dtype=F64)
    psf = psf[:, None] * psf[None]
    otf = transfer_function(psf, grid)

    def fwd(v):
        return torch.fft.irfftn(otf * torch.fft.rfftn(v), s=grid)

    def adj(r):
        return torch.fft.irfftn(otf.conj() * torch.fft.rfftn(r), s=grid)

    y = _python_poisson_data(1, grid, fwd)
    assert float(y.sum()) == 3128.0  # the same data everywhere
    terms = [Term(otf, ShiftedPoisson(y)), Term(None, Box(0.0))]
    max_iter = 10000
    res = admm_fft(terms, grid, rho=1.0, max_iter=max_iter, tol=1e-12)
    assert res.n_iter < max_iter, "ADMM did not converge"
    assert res.primal_residual < 1e-8 * float(res.x.norm())
    x = res.splits[1][0]
    assert bool((x >= 0).all())
    grad = 1 - adj(y / fwd(x).clamp(min=1e-300)) / adj(torch.ones_like(y))
    assert float((x * grad).abs().max()) < 1e-8 * float(x.max())
    zero = x == 0
    assert zero.any()  # the constraint is active somewhere
    assert float(grad[zero].min()) > -1e-8


def test_admm_residuals_and_trace():
    """Primal/dual residuals (Boyd et al. 2011, §3.3) written out, and the
    trace (filled during the run) for both orders."""
    otf, y, grid, region = poisson_problem(var=4.0)
    nu = 0.5
    terms = [
        Term(otf, ShiftedPoisson(y, 4.0), region),
        Term(None, Box(0.0)),
        Term(Hessian(2), GroupL2(nu)),
    ]
    ops = [Convolution(otf, grid), Identity(2), Hessian(2)]
    rho, k = 0.05, 12
    for order in ("x-first", "z-first"):
        seen = []
        trace = {}

        def callback(n, x, seen=seen, trace=trace):
            seen.append(len(trace["change"]))

        prev = admm_fft(
            terms, grid, rho=rho, max_iter=k - 1, tol=None, order=order
        )
        res = admm_fft(
            terms,
            grid,
            rho=rho,
            max_iter=k,
            tol=None,
            order=order,
            trace=trace,
            callback=callback,
        )
        assert seen == list(range(1, k + 1))  # lists grow during the run
        assert set(trace) == {
            "change",
            "residual",
            "dual_residual",
            "objective",
        }
        assert all(len(v) == k for v in trace.values())
        r = sum(
            float((op.forward(res.x) - z).square().sum())
            for op, z in zip(ops, res.splits, strict=True)
        )
        if order == "x-first":
            # s = ρ Aᵀ(zᵏ⁺¹ − zᵏ); r = A xᵏ⁺¹ − zᵏ⁺¹
            s = sum(
                op.adjoint(z - zp)
                for op, z, zp in zip(ops, res.splits, prev.splits, strict=True)
            )
        else:
            # s = ρ A(xᵏ⁺¹ − xᵏ); r = A xᵏ⁺¹ − zᵏ (z is updated last)
            s = torch.cat([op.forward(res.x - prev.x) for op in ops], dim=0)
            r = sum(
                float((op.forward(res.x) - z).square().sum())
                for op, z in zip(ops, prev.splits, strict=True)
            )
        s = rho * float(s.norm())
        assert res.dual_residual == pytest.approx(s, rel=1e-8)
        assert trace["dual_residual"][-1] == pytest.approx(s, rel=1e-8)
        assert res.primal_residual == pytest.approx(r**0.5, rel=1e-8)
        assert trace["residual"][-1] == res.primal_residual
        # objective at the splits: ∑ₖ fₖ(zₖ)
        z = res.splits
        obj = float(
            terms[0].function.value(z[0][(slice(None), *region)])
        ) + float(GroupL2(nu).value(z[2]))
        assert trace["objective"][-1] == pytest.approx(obj, rel=1e-10)
    # long run: both residuals go to 0, the objective settles
    long = {}
    res = admm_fft(terms, grid, rho=rho, max_iter=3000, tol=1e-10, trace=long)
    assert res.primal_residual < 1e-6 * float(res.x.norm())
    assert res.dual_residual < 1e-6 * float(res.x.norm())
    assert long["objective"][-1] == pytest.approx(
        objective(res.x.clamp(min=0), otf, terms[0].function, region, nu),
        rel=1e-8,
    )


def test_admm_prox_returning_its_argument():
    """A prox that returns v itself (here f = 0) must not alias zₖ and uₖ."""

    class Zero:
        def value(self, z):
            return torch.zeros((), dtype=F64)

        def prox(self, v, t):
            return v  # the argument itself

    class ZeroCopy(Zero):
        def prox(self, v, t):
            return v.clone()

    g = gen(12)
    grid = (12, 10)
    data = torch.randn(grid, generator=g, dtype=F64)
    base = [
        Term(None, SquaredL2(1.0, data)),
        Term(Gradient(2), SquaredL2(0.3)),
    ]
    out = {}
    for f in (Zero(), ZeroCopy()):
        terms = [*base, Term(Hessian(2), f)]
        out[type(f)] = admm_fft(terms, grid, rho=1.0, max_iter=50, tol=None)
    res, ref = out[Zero], out[ZeroCopy]
    torch.testing.assert_close(res.x, ref.x, rtol=0, atol=0)
    assert float(res.duals[2].abs().max()) == 0  # u = A x + u − z = 0
    assert float(res.splits[2].abs().max()) > 0
    assert res.splits[2].data_ptr() != res.duals[2].data_ptr()


def test_operator_protocols():
    """Every operator is a Circulant; Term accepts None and tensors."""
    g = gen(13)
    grid = (10, 12)
    psf = torch.rand(3, 3, generator=g, dtype=F64)
    otf = transfer_function(psf / psf.sum(), grid)
    conv = Convolution(otf, grid)
    ops = [Identity(2), conv, Gradient(2), Hessian(2)]
    x = torch.randn(grid, generator=g, dtype=F64)
    for op in ops:
        assert isinstance(op, LinearOperator) and isinstance(op, Circulant)
        y = op.forward(x)
        assert y.shape[1:] == grid and y.shape[0] == len(op.symbols(grid))
        r = torch.randn(y.shape, generator=g, dtype=F64)
        lhs = float((y * r).sum())
        rhs = float((x * op.adjoint(r)).sum())
        assert lhs == pytest.approx(rhs, rel=1e-10)
        # the symbols are the operator, and ‖A‖² bounds the gram
        spec = torch.fft.rfftn(x)
        for c, sym in enumerate(op.symbols(grid)):
            yc = torch.fft.irfftn(sym * spec, s=grid)
            torch.testing.assert_close(yc, y[c])
        assert float(op.gram(grid).max()) <= op.norm2 * (1 + 1e-12)
    assert conv.norm2 == pytest.approx(float(otf.abs().square().max()))
    assert isinstance(as_circulant(None, grid), Identity)
    assert isinstance(as_circulant(otf, grid), Convolution)
    with pytest.raises(TypeError):
        as_circulant("blur", grid)
    with pytest.raises(ValueError):
        Convolution(otf, (8, 8))
    # a Convolution object and the bare transfer function give one solve
    y = torch.randn(grid, generator=g, dtype=F64)
    a = admm_fft(
        [Term(otf, SquaredL2(1.0, y)), Term(None, SquaredL2(0.2))],
        grid,
        rho=1.0,
        max_iter=20,
        tol=None,
    )
    b = admm_fft(
        [Term(conv, SquaredL2(1.0, y)), Term(Identity(2), SquaredL2(0.2))],
        grid,
        rho=1.0,
        max_iter=20,
        tol=None,
    )
    torch.testing.assert_close(a.x, b.x, rtol=0, atol=0)


def test_symbols_float32_low_frequencies():
    """Phase factors are built in float64: float32 symbols are accurate
    to float32 rounding at the lowest frequencies."""
    grid = (4096, 6)
    for cls in (Gradient, Hessian):
        op = cls(2)
        s32 = op.symbols(grid, dtype=torch.float32)
        s64 = op.symbols(grid, dtype=F64)
        for a, b in zip(s32, s64, strict=True):
            a, b = torch.broadcast_tensors(a.to(torch.complex128), b)
            nz = b.abs() > 0
            rel = ((a - b).abs()[nz] / b.abs()[nz]).max()
            assert float(rel) < 1e-6, (cls.__name__, float(rel))
