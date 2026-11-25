"""Total-variation denoising (Rudin, Osher & Fatemi 1992) and the
Chambolle–Pock solver of optix.optim."""

import math
import warnings

import pytest
import torch

import optix
from optix.denoise import ROF, rof
from optix.denoise.utils import tv_denoise
from optix.estimates import SensorEstimate
from optix.io import Layout
from optix.models import phantoms
from optix.optim import (
    Gradient,
    GroupL2,
    L2Ball,
    SquaredL2,
    chambolle_pock,
    primal_dual,
    primal_dual_step,
    project_l2_ball,
)

F64 = dict(dtype=torch.float64)


def scene(shape=(40, 40), seed=0, peak=100.0):
    x = phantoms.cells(shape, 5, radius=6.0, ndim=len(shape), seed=seed, **F64)
    return x * (peak / x.max())


def noisy(clean, sigma, seed=1):
    g = torch.Generator().manual_seed(seed)
    return clean + sigma * torch.randn(
        clean.shape, generator=g, dtype=clean.dtype
    )


def mse(a, b):
    return float((a - b).square().mean())


# -- optim -------------------------------------------------------------------
def test_project_l2_ball():
    g = torch.Generator().manual_seed(0)
    v = torch.randn(30, generator=g, **F64) * 5
    c = torch.randn(30, generator=g, **F64)
    w = torch.rand(30, generator=g, **F64) + 0.1
    r = 2.0
    z = project_l2_ball(v, r, c, w)
    assert float((w * (z - c).square()).sum()) == pytest.approx(r * r)
    # KKT: z − c parallel to (v − c)/(1 + μw) for one μ ≥ 0
    mu = ((v - c) / (z - c) - 1) / w
    torch.testing.assert_close(mu, mu.mean().expand(30))
    assert float(mu.mean()) > 0
    # constant weights: the closed form
    torch.testing.assert_close(
        project_l2_ball(v, r, c, torch.full((30,), 4.0, **F64)),
        project_l2_ball(v, r, c, 4.0),
    )
    # inside: unchanged
    inside = c + 0.01 * (v - c) / (v - c).norm()
    torch.testing.assert_close(project_l2_ball(inside, r, c, w), inside)
    assert float(L2Ball(r, c, w).value(z)) == 0
    assert math.isinf(float(L2Ball(r, c, w).value(v)))


def test_chambolle_pock_matches_primal_dual():
    """Penalised ROF: Chambolle–Pock (Algorithm 2, stopped on the
    primal–dual gap) and the forward–backward–forward solver reach the
    same minimiser. P is λ-strongly convex, so a relative gap ε bounds
    ‖u − u*‖² ≤ 2εP/λ."""
    f = noisy(scene((16, 16)), 15.0)
    lam = 0.05
    op = Gradient(2, boundary="neumann")
    tv = GroupL2(1.0, dim=0)
    a1 = tv_denoise(f, lam=lam, tol=1e-8, max_iter=50000)
    p = float(tv.value(op.forward(a1.x))) + float(
        SquaredL2(lam, f).value(a1.x)
    )
    bound = math.sqrt(2 * a1.gap * p / lam)
    assert bound < 0.05
    gap = a1.gap * p

    def objective(x):
        return float(tv.value(op.forward(x))) + float(
            SquaredL2(lam, f).value(x)
        )

    # the other solvers reach P* to within their accuracy, never below
    step = primal_dual_step(lam, [op.norm2])
    a3 = primal_dual(
        lambda x: lam * (x - f), [(op, tv)], f, step=step, max_iter=1500
    )
    a2 = chambolle_pock(op, tv, SquaredL2(lam, f), f, max_iter=1500, tol=None)
    for x in (a2.x, a3.x):
        assert objective(x) >= p - gap
        assert objective(x) - p < 1e-3 * p


def test_chambolle_pock_checks():
    op = Gradient(2)
    f = torch.zeros(4, 4, **F64)
    with pytest.raises(ValueError, match="at most 1"):
        chambolle_pock(op, GroupL2(1.0), SquaredL2(1.0, f), f, tau=1, sigma=1)
    # τσ‖K‖² = 1 exactly is accepted, slightly above is not
    t = 1 / math.sqrt(op.norm2)
    args = (op, GroupL2(1.0), SquaredL2(1.0, f), f)
    chambolle_pock(*args, tau=t, sigma=t, max_iter=2)
    with pytest.raises(ValueError, match="at most 1"):
        chambolle_pock(*args, tau=t, sigma=1.001 * t, max_iter=2)


def test_chambolle_pock_algorithm2_vs_1():
    """Penalised ROF: Algorithm 2 (γ = λ) and Algorithm 1 reach the same
    minimiser; Algorithm 2 needs fewer iterations."""
    f = noisy(scene((16, 16)), 15.0)
    lam = 0.05
    op = Gradient(2, boundary="neumann")
    n = math.sqrt(op.norm2)
    args = (op, GroupL2(1.0), SquaredL2(lam, f), f)

    def gap(x, y):
        p = float(op.forward(x).norm(dim=0).sum())
        p += lam / 2 * float((x - f).square().sum())
        z = -op.adjoint(y)
        d = -float((z * f).sum() + z.square().sum() / (2 * lam))
        return (p - d) / p

    kw = dict(tau=1 / (lam * n), sigma=0.99 * lam / n, tol=1e-7, gap=gap)
    a1 = chambolle_pock(*args, max_iter=40000, **kw)
    a2 = chambolle_pock(*args, strong_convexity=lam, max_iter=40000, **kw)
    assert a2.n_iter < a1.n_iter
    torch.testing.assert_close(a1.x, a2.x, atol=0.05, rtol=0)


def test_project_l2_ball_cap_feasible():
    """Far from the ball with a cap of one Newton step: still feasible."""
    g = torch.Generator().manual_seed(2)
    v = torch.randn(1000, generator=g, **F64) * 100
    w = 10 ** (8 * torch.rand(1000, generator=g, **F64) - 4)
    r = 1e-6
    z = project_l2_ball(v, r, 0.0, w, max_iter=1)
    assert float((w * z.square()).sum()) <= r * r * (1 + 1e-9)
    exact = project_l2_ball(v, r, 0.0, w)
    assert float((w * exact.square()).sum()) == pytest.approx(r * r)


# -- the ROF model -----------------------------------------------------------
@pytest.fixture(scope="module")
def solved():
    clean = scene((24, 24))
    f = noisy(clean, 20.0)
    res = tv_denoise(f, sigma=20.0, tol=1e-6, max_iter=30000)
    return clean, f, res


def test_constraint_active_and_mean(solved):
    """Eqs. 2.3b–c: residual variance σ², mean of the data."""
    clean, f, res = solved
    assert res.gap <= 1e-6
    assert mse(res.x, f) == pytest.approx(400.0, rel=1e-6)
    assert float(res.x.mean()) == pytest.approx(float(f.mean()), abs=1e-4)
    assert res.lam > 0
    assert mse(res.x, clean) < 0.25 * mse(f, clean)


def test_constrained_equals_penalised(solved):
    """The penalised model with the multiplier λ of the constrained
    solution has the same minimiser (Lagrange duality, Eq. 2.4a)."""
    _, f, res = solved
    pen = tv_denoise(f, lam=res.lam, tol=1e-8, max_iter=30000)
    assert pen.gap <= 1e-8
    torch.testing.assert_close(pen.x, res.x, atol=2e-3 * 20, rtol=0)


def test_euler_lagrange(solved):
    """Optimality: λ(u − f) = −∇ᵀp with ‖pⱼ‖ ≤ 1, pⱼ = ∇uⱼ/‖∇uⱼ‖
    where ∇u ≠ 0 (Eq. 2.4a), checked through the penalised dual."""
    _, f, res = solved
    op = Gradient(2, boundary="neumann")
    pen = chambolle_pock(
        op,
        GroupL2(1.0),
        SquaredL2(res.lam, f),
        f,
        tau=1 / (res.lam * math.sqrt(op.norm2)),
        sigma=0.99 * res.lam / math.sqrt(op.norm2),
        strong_convexity=res.lam,
        max_iter=20000,
        tol=1e-13,
    )
    u, p = pen.x, pen.duals[0]
    assert float(p.norm(dim=0).max()) <= 1 + 1e-9
    torch.testing.assert_close(
        res.lam * (u - f), -op.adjoint(p), atol=1e-3, rtol=0
    )


def test_large_sigma_gives_constant():
    f = noisy(torch.zeros(16, 16, **F64), 1.0)
    res = tv_denoise(f, sigma=10.0, tol=1e-8, max_iter=5000)
    torch.testing.assert_close(res.x, f.mean().expand_as(f), atol=1e-4, rtol=0)
    assert (res.n_iter, res.gap, res.lam) == (0, 0.0, 0.0)


@pytest.mark.parametrize("weighted", [False, True])
def test_feasible_constant_exits_early(weighted):
    """Pure noise with its true σ: a constant is often feasible; the
    ω-weighted mean is returned at once, at the default settings."""
    f = noisy(torch.zeros(32, 32, **F64), 1.0, seed=3)
    w = None
    c = f.mean()
    if weighted:
        w = torch.rand(32, 32, generator=torch.Generator().manual_seed(0))
        w = (w + 0.5).double()
        c = (w * f).sum() / w.sum()
    ww = 1.0 if w is None else w
    assert float((ww * (f - c).square()).sum()) <= f.numel()  # feasible
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = tv_denoise(f, sigma=1.0, weights=w)
    assert (res.n_iter, res.gap, res.lam) == (0, 0.0, 0.0)
    torch.testing.assert_close(res.x, c.expand_as(f), atol=1e-12, rtol=0)


def test_constrained_weight_map():
    """A non-constant weight map in the constraint (the Newton branch of
    project_l2_ball inside the solver): the weighted constraint is
    active, and the penalised model with the returned λ agrees."""
    f = noisy(scene((24, 24)), 10.0)
    g = torch.Generator().manual_seed(7)
    w = (0.5 + torch.rand(24, 24, generator=g)).double()
    res = tv_denoise(f, sigma=10.0, weights=w, tol=1e-6, max_iter=30000)
    assert res.gap <= 1e-6
    r2 = float((w * (res.x - f).square()).sum())
    assert r2 == pytest.approx(f.numel() * 100.0, rel=1e-6)
    pen = tv_denoise(f, lam=res.lam, weights=w, tol=1e-8, max_iter=30000)
    torch.testing.assert_close(pen.x, res.x, atol=0.05, rtol=0)


def test_weights_and_axis_weights_equivalences():
    f = noisy(scene((16, 16)), 10.0)
    kw = dict(tol=1e-7, max_iter=20000)
    # identical iterates up to round-off (the steps scale with σ/√ω̄)
    a = tv_denoise(f, sigma=10.0, tol=1e-5).x
    b = tv_denoise(f, sigma=20.0, weights=torch.full_like(f, 4.0), tol=1e-5)
    torch.testing.assert_close(a, b.x, atol=1e-8, rtol=0)
    c = tv_denoise(f, lam=0.1, **kw).x
    d = tv_denoise(f, lam=0.05, weights=torch.full_like(f, 2.0), **kw).x
    torch.testing.assert_close(c, d, atol=1e-4, rtol=0)
    e = tv_denoise(f, lam=0.1, axis_weights=(1.0, 1.0), **kw).x
    torch.testing.assert_close(c, e, atol=1e-6, rtol=0)
    # an axis weighted down is smoothed less along that axis
    g = tv_denoise(f, lam=0.1, axis_weights=(0.2, 1.0), **kw).x
    assert float(g.diff(dim=0).abs().sum()) > float(c.diff(dim=0).abs().sum())


def test_kl_data_term():
    clean = scene((32, 32), peak=20.0) + 2
    g = torch.Generator().manual_seed(3)
    counts = torch.poisson(clean, generator=g)
    res = tv_denoise(counts, lam=3.0, data_term="kl", tol=1e-5, max_iter=20000)
    assert res.gap <= 1e-5
    assert float(res.x.min()) >= 0
    assert mse(res.x, clean) < 0.5 * mse(counts, clean)


def test_kl_float32_low_counts():
    """Low counts with zeros in float32: the gap stays finite (the dual
    scaling is done in float64) and reaches tol."""
    clean = scene((64, 64), peak=3.0) + 0.05
    g = torch.Generator().manual_seed(11)
    counts = torch.poisson(clean, generator=g).float()
    assert bool((counts == 0).any())
    trace: dict = {}
    res = tv_denoise(
        counts, lam=0.3, data_term="kl", max_iter=6000, trace=trace
    )
    assert res.x.dtype == torch.float32
    assert all(math.isfinite(v) for v in trace["gap"][5:])
    assert math.isfinite(res.gap) and res.gap <= 1e-4


# -- functional and estimator ------------------------------------------------
def test_rof_functional_and_estimator_agree():
    clean = scene((32, 32))
    f = torch.stack([noisy(clean, 15.0, seed=s) for s in range(3)])
    a = rof(f, 15.0)
    est = ROF(sigma=15.0, **F64)
    assert est.is_faithful
    b = est.fit().apply(f)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert all(mse(a[i], clean) < 0.5 * mse(f[i], clean) for i in range(3))


def test_float32_matches_float64():
    f = noisy(scene((24, 24)), 15.0)
    a = ROF(sigma=15.0, tol=1e-5, **F64).fit().apply(f)
    b = ROF(sigma=15.0, tol=1e-5, dtype=torch.float32)
    b = b.fit().apply(f.float())
    assert b.dtype == torch.float32
    torch.testing.assert_close(b.double(), a, atol=0.05, rtol=0)


def test_max_iter_warning_and_history():
    f = torch.stack([noisy(scene((24, 24)), 15.0, seed=s) for s in range(3)])
    with pytest.warns(RuntimeWarning, match="3 of 3 image") as rec:
        rof(f, 15.0, max_iter=20)
    assert len(rec) == 1
    with pytest.warns(RuntimeWarning, match="max_iter=20") as rec:
        tv_denoise(f[0], sigma=15.0, max_iter=20)
    assert len(rec) == 1
    est = ROF(sigma=15.0, max_iter=20, **F64).fit()
    with pytest.warns(RuntimeWarning, match="ROF: 3 of 3") as rec:
        est.apply(f)
    assert len(rec) == 1
    assert est.history.extras["iterations"] == [20.0] * 3
    assert all(g > 1e-4 for g in est.history.extras["gap"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ROF(sigma=15.0, **F64).fit().apply(f[:1])


def test_memory_budget_error():
    f = noisy(scene((32, 32)), 10.0)
    with optix.config(memory=8 * 32 * 32 * 8):
        with pytest.raises(MemoryError, match="memory budget"):
            ROF(sigma=10.0, **F64).fit().apply(f)
    with optix.config(memory=9 * 32 * 32 * 8):
        ROF(sigma=10.0, **F64).fit().apply(f)


def test_save_load(tmp_path):
    f = noisy(scene((24, 24)), 15.0)
    est = ROF(sigma=15.0).fit()
    assert est.is_faithful
    est.save(tmp_path / "rof.pt")
    new = optix.load(tmp_path / "rof.pt")
    assert new.is_faithful
    torch.testing.assert_close(new.apply(f), est.apply(f), atol=0, rtol=0)
    assert not optix.load(_saved(ROF(lam=0.1).fit(), tmp_path)).is_faithful
    z = 2.0 * torch.poisson(scene((24, 24), peak=20.0) + 2) + 100.0
    est = ROF().fit(init=sensor())
    new = optix.load(_saved(est, tmp_path))
    assert new.is_faithful
    torch.testing.assert_close(new.apply(z), est.apply(z), atol=0, rtol=0)


def _saved(est, tmp_path):
    path = tmp_path / "e.pt"
    est.save(path)
    return path


def test_extensions_flagged():
    assert not ROF(lam=0.1).is_faithful
    assert not ROF(lam=0.1, data_term="kl").is_faithful
    assert not ROF(sigma=1.0, ndim=3).is_faithful
    assert not ROF(sigma=1.0, weights=torch.ones(4, 4)).is_faithful
    assert not ROF(sigma=1.0, axis_weights=(1.0, 0.5)).is_faithful
    assert ROF(sigma=1.0, tol=1e-6).is_faithful
    with pytest.raises(ValueError, match="needs lam"):
        ROF(data_term="kl")
    with pytest.raises(ValueError, match="not both"):
        ROF(sigma=1.0, lam=1.0)
    with pytest.raises(ValueError, match="needs sigma"):
        ROF().fit()
    with pytest.raises(ValueError, match="one entry per"):
        ROF(sigma=1.0, axis_weights=(1.0,))


def test_volume_and_layout():
    clean = scene((12, 16, 16))
    f = noisy(clean, 15.0)
    layout = Layout.default(4, 3)
    data = torch.stack([f, f])
    out = ROF(sigma=15.0, ndim=3, layout=layout, **F64).fit().apply(data)
    torch.testing.assert_close(out[0], out[1], atol=0, rtol=0)
    assert mse(out[0], clean) < 0.3 * mse(f, clean)


def test_pixel_weight_map_per_group():
    f = noisy(scene((24, 24)), 10.0)
    data = torch.stack([f, f])
    w = torch.stack([torch.ones(24, 24), torch.full((24, 24), 4.0)]).double()
    layout = Layout(spatial=(1, 2), group=0)
    out = ROF(lam=0.1, weights=w, layout=layout, **F64).fit().apply(data)
    a = tv_denoise(f, lam=0.1).x
    b = tv_denoise(f, lam=0.4).x
    torch.testing.assert_close(out[0], a, atol=0, rtol=0)
    torch.testing.assert_close(out[1], b, atol=1e-6, rtol=0)


def sensor(gain=2.0, offset=100.0, variance=4.0):
    t = lambda v: torch.tensor(v, **F64)  # noqa: E731
    return SensorEstimate(offset=t(offset), variance=t(variance), gain=t(gain))


def test_poisson_gaussian_vst_and_kl():
    clean = scene((32, 32), peak=30.0) + 5
    g = torch.Generator().manual_seed(4)
    alpha, mu, s2 = 2.0, 100.0, 4.0
    z = alpha * torch.poisson(clean, generator=g) + mu
    z = z + math.sqrt(s2) * torch.randn(z.shape, generator=g, **F64)
    target = alpha * clean + mu
    vst = ROF(**F64).fit(init=sensor()).apply(z)
    assert mse(vst, target) < 0.5 * mse(z, target)
    kl = ROF(lam=5.0, data_term="kl", **F64).fit(init=sensor()).apply(z)
    assert mse(kl, target) < 0.5 * mse(z, target)


def test_kl_sensor_units():
    """With no read noise the KL model on ADU is that on counts."""
    clean = scene((24, 24), peak=20.0) + 2
    counts = torch.poisson(clean, generator=torch.Generator().manual_seed(5))
    adu = 3.0 * counts + 50.0
    est = ROF(lam=1.0, data_term="kl", max_iter=6000, **F64).fit(
        init=sensor(gain=3.0, offset=50.0, variance=0.0)
    )
    a = est.apply(adu)
    b = tv_denoise(counts, lam=1.0, data_term="kl", max_iter=6000).x
    torch.testing.assert_close(a, 3.0 * b + 50.0, atol=1e-9, rtol=0)


def test_sensor_checks():
    with pytest.raises(ValueError, match="sigma is not used"):
        ROF(sigma=2.0).fit(init=sensor())
    with pytest.raises(ValueError, match="gain must be > 0"):
        ROF().fit(init=sensor(gain=0.0))


@pytest.mark.parametrize("data_term", ["gaussian", "kl"])
def test_sensor_maps_with_group_axis(data_term):
    """Maps of a SensorEstimate, one per group: each group is denoised
    as with the scalar sensor of its values."""
    clean = scene((24, 24), peak=20.0) + 2
    g = torch.Generator().manual_seed(6)
    params = [(2.0, 100.0, 4.0), (3.0, 50.0, 1.0)]
    data = torch.stack(
        [
            a * torch.poisson(clean, generator=g)
            + o
            + math.sqrt(v) * torch.randn(clean.shape, generator=g, **F64)
            for a, o, v in params
        ]
    )

    def field(k):
        return torch.stack([torch.full((24, 24), p[k], **F64) for p in params])

    maps = SensorEstimate(gain=field(0), offset=field(1), variance=field(2))
    kw = {}
    if data_term == "kl":
        kw = dict(lam=1.0, data_term="kl", max_iter=6000)
    layout = Layout(spatial=(1, 2), group=0)
    out = ROF(layout=layout, **kw, **F64).fit(init=maps).apply(data)
    for i, (a, o, v) in enumerate(params):
        ref = ROF(**kw, **F64).fit(init=sensor(a, o, v)).apply(data[i])
        torch.testing.assert_close(out[i], ref, atol=1e-9, rtol=1e-6)
