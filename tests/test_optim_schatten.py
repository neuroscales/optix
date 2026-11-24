"""Tests of the Schatten-norm tools of optix.optim: closed-form 2 × 2 and
3 × 3 spectral maps, projections onto Schatten balls, the mirror Hessian
of Lefkimmiatis, Ward & Unser (2013, Eq. 10) and their dual algorithm
for the Hessian Schatten-norm prox (Algorithm 2)."""

import math

import pytest
import torch

from optix.optim import (
    Box,
    Hessian,
    HessianSchatten,
    Schatten,
    _spectral,
    primal_dual,
    primal_dual_step,
    project_group_l2,
    project_schatten,
    prox_box,
    prox_group_l2,
    prox_hessian_schatten,
    prox_schatten,
    schatten_norm,
)

F64 = torch.float64
F32 = torch.float32
SQ2 = math.sqrt(2)
INF = math.inf
DUAL = {1: INF, 2: 2, INF: 1}


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


# -- packing and references ------------------------------------------------
def unpack(v):
    """Packed `(C, n)` → matrices `(n, d, d)` (d = 1, 2, 3)."""
    c = v.shape[0]
    d = {1: 1, 3: 2, 6: 3}[c]
    out = v.new_zeros(v.shape[1], d, d)
    pairs = [(a, a) for a in range(d)] + [
        (a, b) for a in range(d) for b in range(a + 1, d)
    ]
    for k, (a, b) in enumerate(pairs):
        val = v[k] if a == b else v[k] / SQ2
        out[:, a, b] = val
        out[:, b, a] = val
    return out


def pack(x):
    d = x.shape[-1]
    pairs = [(a, a) for a in range(d)] + [
        (a, b) for a in range(d) for b in range(a + 1, d)
    ]
    return torch.stack(
        [x[:, a, b] if a == b else SQ2 * x[:, a, b] for a, b in pairs]
    )


def l1_ball(s, radius):
    """Sort-based projection of each row of s ≥ 0 onto the ℓ1 ball."""
    u = torch.sort(s, dim=-1, descending=True).values
    cs = u.cumsum(-1)
    k = torch.arange(1, s.shape[-1] + 1, dtype=s.dtype)
    rank = (u - (cs - radius) / k > 0).sum(-1, keepdim=True).clamp(min=1)
    theta = (cs.gather(-1, rank - 1) - radius) / rank
    inside = s.sum(-1, keepdim=True) <= radius
    return torch.where(inside, s, (s - theta).clamp(min=0))


def reference(v, op, r, p):
    """`op` ∈ {"prox", "proj"} by `eigh` in float64 (singular values =
    |eigenvalues|, signs and eigenvectors kept)."""
    x = unpack(v.double())
    mu, vec = torch.linalg.eigh(x)
    s, sign = mu.abs(), torch.sign(mu)
    r = torch.as_tensor(r, dtype=F64).reshape(-1, 1)
    if op == "prox":
        if p == 1:
            s = (s - r).clamp(min=0)
        elif p == 2:
            n = s.norm(dim=-1, keepdim=True).clamp(min=1e-300)
            s = s * (1 - r / n).clamp(min=0)
        else:
            s = s - l1_ball(s, r)
    else:  # projection onto the S_p ball
        if p == 1:
            s = l1_ball(s, r)
        elif p == 2:
            n = s.norm(dim=-1, keepdim=True).clamp(min=1e-300)
            s = s * (r / n).clamp(max=1)
        else:
            s = torch.minimum(s, r)
    return pack(vec @ torch.diag_embed(sign * s) @ vec.transpose(-1, -2))


def fields(c, n=3000, seed=0, dtype=F64):
    """Packed symmetric fields with the hard cases of the closed forms:
    zero, repeated (exactly and nearly), triple, rank-deficient, scaled
    by many orders of magnitude."""
    d = {3: 2, 6: 3}[c]
    g = gen(seed)
    q, _ = torch.linalg.qr(torch.randn(n, d, d, generator=g, dtype=F64))
    mu = torch.randn(n, d, generator=g, dtype=F64)
    mu = mu * torch.exp(2 * torch.randn(n, 1, generator=g, dtype=F64))
    k = n // 10
    mu[:k] = 0  # zero matrices
    mu[k : 2 * k, 1] = mu[k : 2 * k, 0]  # exactly repeated
    mu[2 * k : 3 * k, 1] = mu[2 * k : 3 * k, 0] * (1 + 1e-9)  # nearly
    mu[3 * k : 4 * k] = mu[3 * k : 4 * k, :1]  # all equal (m·I)
    mu[4 * k : 5 * k, 0] = 0  # singular
    mu[5 * k : 6 * k, 1] = -mu[5 * k : 6 * k, 0]  # ±μ
    mu[6 * k : 7 * k] = mu[6 * k : 7 * k, :1] * (
        1 + 1e-7 * torch.randn(k, d, generator=g, dtype=F64)
    )  # nearly m·I
    if d == 3:
        mu[7 * k : 8 * k, 2] = mu[7 * k : 8 * k, 1]  # pair below/above
    x = q @ torch.diag_embed(mu) @ q.transpose(-1, -2)
    x[3 * k : 4 * k] = torch.eye(d, dtype=F64) * mu[3 * k : 4 * k, :1, None]
    x[8 * k : 8 * k + 20] *= 1e-30 if dtype == F64 else 1e-15
    x[8 * k + 20 : 8 * k + 40] *= 1e30 if dtype == F64 else 1e15
    return pack(x).to(dtype)


def scale_of(v):
    return unpack(v.double()).flatten(1).abs().amax(1)


# -- closed-form eigenvalues -------------------------------------------------
@pytest.mark.parametrize("dtype", [F32, F64])
@pytest.mark.parametrize("c", [3, 6])
def test_eigenvalues_match_eigh(c, dtype):
    v = fields(c, dtype=dtype)
    mu = _spectral.eigenvalues(v).T.double().sort(-1).values
    ref = torch.linalg.eigvalsh(unpack(v.double()))
    err = (mu - ref).abs().amax(-1) / scale_of(v).clamp(min=1e-300)
    eps = torch.finfo(dtype).eps
    assert float(err.max()) < 8 * eps, float(err.max())
    # zero matrices: exactly zero; m·I: exactly m
    assert bool((_spectral.eigenvalues(v[:, :300]) == 0).all())


def test_eig3_decomposition_reconstructs():
    """X = m·I + s·(λuuᵀ + m₂P + D), with u a unit eigenvector."""
    v = fields(6, n=1000, seed=3)
    e = _spectral.eig3(v)
    u = torch.stack(e.u, -1)
    assert torch.allclose(u.norm(dim=-1), torch.ones(1000, dtype=F64))
    x = unpack(v)
    mu = e.m + e.s * e.lam
    res = x @ u[..., None] - mu[:, None, None] * u[..., None]
    sc = scale_of(v)[:, None]
    assert float((res[..., 0].abs() / sc.clamp(min=1e-300)).max()) < 1e-14
    p = torch.eye(3, dtype=F64) - u[:, :, None] * u[:, None, :]
    dev = unpack(torch.stack(list(e.dev[:3]) + [SQ2 * t for t in e.dev[3:]]))
    rec = e.m[:, None, None] * torch.eye(3, dtype=F64) + e.s[:, None, None] * (
        e.lam[:, None, None] * (u[:, :, None] * u[:, None, :])
        + e.m2[:, None, None] * p
        + dev
    )
    err = (rec - x).flatten(1).abs().amax(1) / scale_of(v).clamp(min=1e-300)
    assert float(err.max()) < 1e-14


# -- norms, proxes and projections ---------------------------------------
@pytest.mark.parametrize("c", [3, 6])
@pytest.mark.parametrize("p", [1, 2, INF])
def test_schatten_norm_matches_svd(c, p):
    v = fields(c, n=1500, seed=4)
    s = torch.linalg.svdvals(unpack(v))
    ref = {1: s.sum(-1), 2: s.norm(dim=-1), INF: s.amax(-1)}[p]
    err = (schatten_norm(v, p) - ref).abs() / scale_of(v).clamp(min=1e-300)
    assert float(err.max()) < 1e-14


@pytest.mark.parametrize("dtype", [F32, F64])
@pytest.mark.parametrize("c", [3, 6])
@pytest.mark.parametrize("p", [1, 2, INF])
def test_prox_and_projection_match_eigh(c, p, dtype):
    v = fields(c, seed=5, dtype=dtype)
    sc = scale_of(v)
    eps = torch.finfo(dtype).eps
    for frac in (0.0, 0.1, 0.7, 3.0):
        r = frac * sc  # radius relative to each matrix
        rt = r.to(dtype)[None]
        z = prox_schatten(v, rt, p).double()
        err = (z - reference(v, "prox", r, p)).abs().amax(0)
        assert float((err / sc.clamp(min=1e-300)).max()) < 16 * eps, frac
        z = project_schatten(v, rt, p).double()
        err = (z - reference(v, "proj", r, p)).abs().amax(0)
        assert float((err / sc.clamp(min=1e-300)).max()) < 16 * eps, frac
    # a scalar radius
    z = prox_schatten(v[:, :500], 0.5, p).double()
    ref = reference(v[:, :500], "prox", 0.5, p)
    assert float((z - ref).abs().max()) < 16 * eps * (1 + float(sc.max()))


@pytest.mark.parametrize("c", [3, 6])
@pytest.mark.parametrize("p", [1, 2, INF])
def test_moreau_decomposition(c, p):
    """X = prox_{r‖·‖_{S_p}}(X) + P_{r B_{S_q}}(X), 1/p + 1/q = 1, and
    the projection lies in the ball and fixes its points."""
    v = fields(c, n=2000, seed=6)
    sc = scale_of(v)
    q = DUAL[p]
    for frac in (0.0, 0.2, 1.0, 5.0):
        r = (frac * sc)[None]
        a = prox_schatten(v, r, p)
        b = project_schatten(v, r, q)
        err = (a + b - v).abs().amax(0) / sc.clamp(min=1e-300)
        assert float(err.max()) < 1e-14, frac
        norm = schatten_norm(b, q)
        assert bool((norm <= r[0] * (1 + 1e-12) + 1e-300).all())
        torch.testing.assert_close(project_schatten(b, r, q), b)
    # the S2 tools are the group-ℓ2 ones in any dimension
    w = torch.randn(10, 40, generator=gen(7), dtype=F64)
    torch.testing.assert_close(prox_schatten(w, 0.4, 2), prox_group_l2(w, 0.4))
    torch.testing.assert_close(
        project_schatten(w, 0.4, 2), project_group_l2(w, 0.4)
    )


@pytest.mark.parametrize("p", [1, INF])
def test_three_by_three_reduces_to_two_by_two(p):
    """Block-diagonal diag(X₂, e): the 3-D maps act as the 2-D ones on X₂
    and as the scalar map on e (p = 1); with e = 0 the S∞ maps agree."""
    v2 = fields(3, n=1000, seed=8)[:, :800]  # no extreme scales
    for e in (0.0, 0.7):
        v3 = torch.zeros(6, v2.shape[1], dtype=F64)
        v3[0], v3[1], v3[2], v3[3] = v2[0], v2[1], e, v2[2]
        z3 = prox_schatten(v3, 0.3, p)
        if p == 1 or e == 0:
            z2 = prox_schatten(v2, 0.3, p)
            torch.testing.assert_close(z3[[0, 1, 3]], z2, atol=1e-12, rtol=0)
            torch.testing.assert_close(
                z3[4:], torch.zeros_like(z3[4:]), atol=1e-12, rtol=0
            )
        r = torch.linalg.svdvals(unpack(v2))
        n3 = schatten_norm(v3, p)
        ref = (
            r.sum(-1) + e
            if p == 1
            else torch.maximum(r.amax(-1), r.new_tensor(e))
        )
        torch.testing.assert_close(n3, ref)
    # 1 × 1: soft threshold and clip
    s = torch.randn(1, 50, generator=gen(9), dtype=F64)
    torch.testing.assert_close(project_schatten(s, 0.3, 1), s.clamp(-0.3, 0.3))
    torch.testing.assert_close(
        project_schatten(s, 0.3, INF), s.clamp(-0.3, 0.3)
    )


@pytest.mark.parametrize("p", [1, INF])
def test_prox_three_by_three_is_optimal(p):
    """The prox beats random perturbations of itself."""
    y = fields(6, n=300, seed=10)[:, :240] / 3  # no extreme scales
    t = 0.4

    def obj(x):
        return 0.5 * (x - y).square().sum(0) + t * schatten_norm(x, p)

    z = prox_schatten(y, t, p)
    best = obj(z)
    g = gen(11)
    for scale in (1e-4, 1e-2, 1.0):
        for _ in range(10):
            dz = scale * torch.randn(z.shape, generator=g, dtype=F64)
            assert bool((obj(z + dz) >= best - 1e-11).all())


def test_derivatives_are_finite():
    """No NaN or inf in the derivatives, at zero, repeated and triple
    eigenvalues included, in float32 and float64; gradcheck on generic
    matrices."""
    for c in (3, 6):
        for dtype in (F32, F64):
            v = fields(c, n=400, seed=12, dtype=dtype)[:, :340]  # no huge
            v = v.clone().requires_grad_()
            outs = [
                schatten_norm(v, 1).sum(),
                schatten_norm(v, INF).sum(),
                (prox_schatten(v, 0.3, 1) * 1.3).sum(),
                prox_schatten(v, 0.3, INF).square().sum(),
                project_schatten(v, 0.3, 1).square().sum(),
                project_schatten(v, 0.3, INF).sum(),
            ]
            for out in outs:
                (grad,) = torch.autograd.grad(out, v)
                assert bool(grad.isfinite().all()), (c, dtype)
    for c in (3, 6):
        v = torch.randn(c, 6, generator=gen(13), dtype=F64, requires_grad=True)
        for p in (1, INF):
            assert torch.autograd.gradcheck(
                lambda x, p=p: schatten_norm(x, p), (v,)
            )
            assert torch.autograd.gradcheck(
                lambda x, p=p: prox_schatten(x, 0.2, p), (v,)
            )
            assert torch.autograd.gradcheck(
                lambda x, p=p: project_schatten(x, 0.5, p), (v,)
            )


def test_chunks_and_component_dim(monkeypatch):
    v = torch.randn(4, 6, 5, 7, generator=gen(14), dtype=F64)
    t = torch.rand(4, 1, 5, 7, generator=gen(15), dtype=F64)
    whole = prox_schatten(v, t, 1, dim=1)
    norm = schatten_norm(v, INF, dim=1)
    monkeypatch.setattr(_spectral, "_CHUNK", 16)
    torch.testing.assert_close(prox_schatten(v, t, 1, dim=1), whole)
    torch.testing.assert_close(schatten_norm(v, INF, dim=1), norm)
    ref = prox_schatten(v.movedim(1, 0), t.movedim(1, 0), 1).movedim(0, 1)
    torch.testing.assert_close(whole, ref)
    assert norm.shape == (4, 5, 7)


def test_float32_and_scale_invariance():
    v = fields(6, n=2000, seed=16)[:, :1600]  # no extreme scales
    for p in (1, INF):
        z32 = prox_schatten(v.float(), 0.3, p).double()
        z64 = prox_schatten(v, 0.3, p)
        err = (z32 - z64).abs().amax(0) / (scale_of(v) + 0.3)
        assert float(err.max()) < 2e-6
        # homogeneity: prox_{a t}(a X) = a prox_t(X)
        for a in (1e-20, 1e20):
            torch.testing.assert_close(
                prox_schatten(a * v, a * 0.3, p) / a,
                z64,
                rtol=1e-12,
                atol=1e-12,
            )


# -- mirror Hessian (Eq. 10) ---------------------------------------------
def hessian_eq10(x):
    """Eq. (10a–c) of Lefkimmiatis, Ward & Unser (2013), by indices
    (1-based in the paper), packed as (D₁₁, D₂₂, √2·D₁₂)."""
    nx, ny = x.shape
    out = torch.zeros(3, nx, ny, dtype=x.dtype)
    for i in range(nx):
        for j in range(ny):
            if i <= nx - 3:
                out[0, i, j] = x[i + 2, j] - 2 * x[i + 1, j] + x[i, j]
            elif nx >= 2:
                out[0, i, j] = x[nx - 2, j] - x[nx - 1, j]
            if j <= ny - 3:
                out[1, i, j] = x[i, j + 2] - 2 * x[i, j + 1] + x[i, j]
            elif ny >= 2:
                out[1, i, j] = x[i, ny - 2] - x[i, ny - 1]
            if i <= nx - 2 and j <= ny - 2:
                out[2, i, j] = SQ2 * (
                    x[i + 1, j + 1] - x[i + 1, j] - x[i, j + 1] + x[i, j]
                )
    return out


def test_mirror_hessian_is_eq10():
    g = gen(20)
    for shape in ((7, 6), (3, 2), (1, 4), (2, 2)):
        x = torch.randn(shape, generator=g, dtype=F64)
        torch.testing.assert_close(
            Hessian(2, boundary="mirror").forward(x), hessian_eq10(x)
        )
    # the mixed terms are those of "neumann"; the diagonal differs only
    # at the last index
    x = torch.randn(9, 8, generator=g, dtype=F64)
    hm = Hessian(2, boundary="mirror").forward(x)
    hn = Hessian(2, boundary="neumann").forward(x)
    torch.testing.assert_close(hm[2], hn[2])
    torch.testing.assert_close(hm[0, :-1], hn[0, :-1])
    torch.testing.assert_close(hm[0, -1], hm[0, -2])
    with pytest.raises(ValueError, match="periodic"):
        Hessian(2, boundary="mirror").symbols((8, 8))
    with pytest.raises(ValueError, match="boundary"):
        Hessian(2, boundary="reflect")


@pytest.mark.parametrize(
    "shape", [(6, 5), (1, 3), (2, 7), (4, 3, 5), (2, 1, 3)]
)
def test_mirror_hessian_adjoint_and_norm(shape):
    """⟨Hx, Y⟩ = ⟨x, H*Y⟩ (Eq. 13) and ‖H‖² ≤ (4∑w²)² (Eq. 70)."""
    g = gen(21)
    w = [1.0, 0.5, 2.0][: len(shape)]
    for weights in (None, w):
        h = Hessian(len(shape), boundary="mirror", weights=weights)
        x = torch.randn(shape, generator=g, dtype=F64)
        y = torch.randn(h.components, *shape, generator=g, dtype=F64)
        lhs = float((h.forward(x) * y).sum())
        rhs = float((x * h.adjoint(y)).sum())
        assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)
        # power iteration on H*H
        u = torch.randn(shape, generator=g, dtype=F64)
        for _ in range(300):
            u = h.adjoint(h.forward(u))
            lam = float(u.norm())
            u = u / lam
        assert lam <= h.norm2 * (1 + 1e-9)


def test_mirror_hessian_of_quadratic():
    """On a quadratic, every interior Hessian is its matrix A (3-D)."""
    a = torch.tensor(
        [[2.0, -1.0, 0.5], [-1.0, 0.5, 0.3], [0.5, 0.3, -1.5]], dtype=F64
    )
    u = torch.stack(
        torch.meshgrid(*[torch.arange(9, dtype=F64)] * 3, indexing="ij")
    )
    x = 0.5 * torch.einsum("i...,ij,j...->...", u, a, u)
    h = Hessian(3, boundary="mirror").forward(x)[:, :7, :7, :7]
    s = torch.linalg.svdvals(a)
    for p, ref in ((1, s.sum()), (2, s.norm()), (INF, s.max())):
        norm = schatten_norm(h, p)
        torch.testing.assert_close(norm, torch.full_like(norm, float(ref)))


# -- the dual algorithm (Algorithm 2 of Lefkimmiatis, Ward & Unser) -------
def primal(x, z, tau, p, hess):
    return 0.5 * float((x - z).square().sum()) + tau * float(
        schatten_norm(hess.forward(x), p).sum()
    )


def dual_value(omega, z, tau, hess, lower=0.0):
    """s(Ω) of Eq. (28) with C = [lower, ∞)."""
    v = z - tau * hess.adjoint(omega)
    pc = v.clamp(min=lower)
    return 0.5 * float(
        (pc - v).square().sum() + z.square().sum() - v.square().sum()
    )


@pytest.mark.parametrize("shape", [(14, 12), (6, 7, 5)])
@pytest.mark.parametrize("p", [1, 2, INF])
def test_duality_gap_closes(shape, p):
    """Weak duality s(Ω) ≤ ϱ(x̂) (Eqs. 26–28), Ω in the dual ball, and
    the gap shrinks to near 0: x̂ solves Eq. (19)."""
    z = 3 * torch.rand(shape, generator=gen(22), dtype=F64) - 0.5
    hess = Hessian(len(shape), boundary="mirror")
    tau = 0.15
    gaps = []
    for n in (5, 50, 800):
        res = prox_hessian_schatten(z, tau, p, max_iter=n)
        assert res.n_iter == n and res.dual.shape == (hess.components, *shape)
        assert bool((res.x >= 0).all())
        ball = schatten_norm(res.dual, DUAL[p])
        assert bool((ball <= 1 + 1e-12).all())
        top = primal(res.x, z, tau, p, hess)
        low = dual_value(res.dual, z, tau, hess)
        assert low <= top + 1e-9
        gaps.append((top - low) / top)
    assert gaps[2] < gaps[0]
    assert gaps[2] < 2e-3, gaps


@pytest.mark.parametrize("shape", [(10, 9), (4, 5, 4)])
@pytest.mark.parametrize("p", [2, INF])
def test_same_solution_as_primal_dual(shape, p):
    """Cross-solver: the primal–dual method of Chouzenoux et al. (FBF)
    on ½‖x − z‖² + τ‖Hx‖₁,ₚ + ι₊ gives the same x̂."""
    z = 3 * torch.rand(shape, generator=gen(23), dtype=F64) - 0.5
    hess = Hessian(len(shape), boundary="mirror")
    tau = 0.2
    a = prox_hessian_schatten(z, tau, p, max_iter=1500).x
    b = primal_dual(
        lambda x: x - z,
        [(hess, Schatten(tau, p))],
        z.clone(),
        step=primal_dual_step(1.0, [hess.norm2]),
        constraint=Box(0.0),
        max_iter=3000,
        tol=1e-13,
    ).x
    assert float((a - b).abs().max()) < 5e-4 * float(z.abs().max())
    fa = primal(a, z, tau, p, hess)
    fb = primal(b, z, tau, p, hess)
    assert fa == pytest.approx(fb, rel=1e-5)


def project_dual_ball(y, q):
    """P_{B∞,q} by SVD of the 2 × 2 matrices (Eqs. 33–36)."""
    v = y.reshape(3, -1)
    out = reference(v, "proj", 1.0, q)
    return out.reshape(y.shape)


def algorithm2(z, tau, p, n_iter, omega0=None, lower=0.0, upper=None):
    """Algorithm 2 of Lefkimmiatis, Ward & Unser (2013), line by line,
    with the Eq. (10) stencil by indices and its adjoint by Eq. (13)."""
    q = DUAL[p]
    shape = z.shape

    def h(x):
        return hessian_eq10(x)

    # adjoint by the transpose of the explicit matrix of Eq. (10)
    n = z.numel()
    mat = torch.stack(
        [h(e.reshape(shape)).flatten() for e in torch.eye(n, dtype=F64)], 1
    )

    def ht(y):
        return (mat.T @ y.flatten()).reshape(shape)

    def pc(x):
        return prox_box(x, lower, upper)

    psi = torch.zeros(3, *shape, dtype=F64) if omega0 is None else omega0
    omega_prev = psi.clone()
    t = 1.0
    omega = omega_prev
    for _ in range(n_iter):
        omega = project_dual_ball(
            psi + 1 / (64 * tau) * h(pc(z - tau * ht(psi))), q
        )
        t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
        psi = omega + ((t - 1) / t_next) * (omega - omega_prev)
        omega_prev = omega
        t = t_next
    return pc(z - tau * ht(omega)), omega


@pytest.mark.parametrize("p", [1, 2, INF])
def test_algorithm2_written_out(p):
    z = 2 * torch.rand(9, 8, generator=gen(24), dtype=F64) - 0.3
    tau = 0.3
    ref_x, ref_omega = algorithm2(z, tau, p, 7)
    res = prox_hessian_schatten(z, tau, p, max_iter=7)
    torch.testing.assert_close(res.x, ref_x, rtol=0, atol=1e-12)
    torch.testing.assert_close(res.dual, ref_omega, rtol=0, atol=1e-12)
    # a warm start restarts the momentum from the given dual
    ref2, _ = algorithm2(z, tau, p, 4, omega0=ref_omega)
    res2 = prox_hessian_schatten(z, tau, p, max_iter=4, dual=res.dual)
    torch.testing.assert_close(res2.x, ref2, rtol=0, atol=1e-12)
    # a box [lower, upper]
    ref3, _ = algorithm2(z, tau, p, 5, lower=0.1, upper=1.0)
    res3 = prox_hessian_schatten(z, tau, p, max_iter=5, lower=0.1, upper=1.0)
    torch.testing.assert_close(res3.x, ref3, rtol=0, atol=1e-12)


def test_prox_hessian_schatten_edge_cases():
    z = torch.randn(2, 6, 5, generator=gen(25), dtype=F64)  # batch of 2
    res = prox_hessian_schatten(z, 0.0, 1, ndim=2)
    torch.testing.assert_close(res.x, z.clamp(min=0))
    assert res.n_iter == 0
    both = prox_hessian_schatten(z, 0.2, 1, ndim=2, max_iter=6)
    one = prox_hessian_schatten(z[1], 0.2, 1, max_iter=6)
    torch.testing.assert_close(both.x[1], one.x)
    assert both.dual.shape == (2, 3, 6, 5)
    # an image in the null space of H (periodic: constants) is fixed
    const = torch.full((6, 5), 2.5, dtype=F64)
    for boundary in ("mirror", "periodic", "neumann"):
        out = prox_hessian_schatten(
            const, 0.5, 1, lower=None, max_iter=3, boundary=boundary
        ).x
        torch.testing.assert_close(out, const)
    with pytest.raises(ValueError, match="dual"):
        prox_hessian_schatten(z[0], 0.1, 1, dual=torch.zeros(3, 4, 4))
    with pytest.raises(ValueError, match="tau"):
        prox_hessian_schatten(z[0], -0.1, 1)
    with pytest.raises(ValueError, match="p must"):
        prox_hessian_schatten(z[0], 0.1, 3)
    with pytest.raises(ValueError, match="dims"):
        prox_hessian_schatten(z[0], 0.1, 1, hessian=Hessian(3))


def test_hessian_schatten_function():
    z = 2 * torch.rand(1, 8, 9, generator=gen(26), dtype=F64)
    f = HessianSchatten(0.5, 1, ndim=2, n_iter=4)
    hess = Hessian(2, boundary="mirror")
    assert float(f.value(z)) == pytest.approx(
        0.5 * float(schatten_norm(hess.forward(z[0]), 1).sum())
    )
    assert float(f.value(z - 10)) == math.inf
    # prox(v, t) = 4 dual iterations at weight t·w, warm-started
    a = f.prox(z, 0.4)
    first = prox_hessian_schatten(z, 0.2, 1, ndim=2, max_iter=4)
    torch.testing.assert_close(a, first.x)
    b = f.prox(z, 0.4)
    second = prox_hessian_schatten(
        z, 0.2, 1, ndim=2, max_iter=4, dual=first.dual
    )
    torch.testing.assert_close(b, second.x)
    f.reset()
    torch.testing.assert_close(f.prox(z, 0.4), first.x)
    cold = HessianSchatten(0.5, 1, ndim=2, n_iter=4, warm_start=False)
    cold.prox(z, 0.4)
    torch.testing.assert_close(cold.prox(z, 0.4), first.x)
    # a new grid shape drops the stored dual
    f.prox(z[:, :5], 0.4)
    assert f.dual.shape == (1, 3, 5, 9)
    # the prox of the 2013 inner problem, many iterations: optimal
    g = HessianSchatten(0.5, INF, ndim=2, n_iter=600)
    x = g.prox(z, 0.4)
    best = 0.5 * float((x - z).square().sum()) / 0.4 + float(g.value(x))
    gen_ = gen(27)
    for _ in range(20):
        y = (x + 1e-3 * torch.randn(x.shape, generator=gen_, dtype=F64)).clamp(
            min=0
        )
        other = 0.5 * float((y - z).square().sum()) / 0.4 + float(g.value(y))
        assert other >= best - 1e-6 * best
