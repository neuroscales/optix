"""Tests of optix.deconv.Chouzenoux (Chouzenoux et al. 2015)."""

import math
import warnings

import pytest
import torch

from optix import load
from optix.deconv import Chouzenoux, chouzenoux
from optix.deconv.utils import PaddedConvolution
from optix.estimates import SensorEstimate
from optix.io import Layout
from optix.optim import (
    Box,
    Convolution,
    Gradient,
    GroupL2,
    Hessian,
    PoissonGaussian,
    ShiftedPoisson,
    Term,
    admm_fft,
    primal_dual,
    primal_dual_step,
)

F64 = torch.float64
TV_H_NORM2 = [8.0, 64.0]  # ‖∇‖², ‖H‖² in 2-D


def gaussian2d(n=7, s=1.6):
    r = torch.arange(n, dtype=F64) - n // 2
    g = torch.exp(-(r[:, None] ** 2 + r[None] ** 2) / (2 * s * s))
    return g / g.sum()


def scene(shape, seed=0):
    """Blobs and a bar on a dim background; max 1."""
    gen = torch.Generator().manual_seed(seed)
    g = torch.meshgrid(
        *[torch.arange(n, dtype=F64) for n in shape], indexing="ij"
    )
    x = torch.full(shape, 0.1, dtype=F64)
    for _ in range(5):
        c = [float(torch.rand(1, generator=gen)) * n for n in shape]
        r = 1.5 + 2.5 * float(torch.rand(1, generator=gen))
        d2 = sum((gi - ci) ** 2 for gi, ci in zip(g, c, strict=True))
        x = x + torch.exp(-d2 / (2 * r * r))
    bar = (g[0] > shape[0] / 3) & (g[0] < shape[0] / 3 + 3)
    x = torch.where(bar & (g[1] > 4) & (g[1] < shape[1] - 4), x + 0.5, x)
    return x / x.max()


def simulate(shape, psf, peak, var, seed=0, margin=8):
    """x = peak·scene; y = Poisson(blur of a larger field) + N(0, σ²),
    cropped, so that the boundary is unknown."""
    big = tuple(n + 2 * margin for n in shape)
    x = peak * scene(big, seed)
    pad = PaddedConvolution(psf, big)
    mean = pad.forward(pad.pad(x)).clamp(min=0)
    gen = torch.Generator().manual_seed(100 + seed)
    y = torch.poisson(mean, generator=gen) + math.sqrt(var) * torch.randn(
        big, generator=gen, dtype=F64
    )
    crop = tuple(slice(margin, margin + n) for n in shape)
    return x[crop].contiguous(), y[crop].contiguous()


def sensor(var, offset=0.0, gain=1.0):
    return SensorEstimate(
        offset=torch.tensor(offset, dtype=F64),
        gain=torch.tensor(gain, dtype=F64),
        variance=torch.as_tensor(var, dtype=F64),
    )


def snr(est, x):
    return 10 * math.log10(float(x.square().sum() / (est - x).square().sum()))


def objective(x, y, var, op, tv, hessian):
    """The exact objective (3.1) on the object grid, +inf outside C."""
    if bool((x < 0).any()):
        return math.inf
    total = float(PoissonGaussian(y, var).value(op.forward(x)))
    total += float(GroupL2(tv).value(Gradient(2).forward(x)))
    total += float(GroupL2(hessian).value(Hessian(2).forward(x)))
    return total


# -- the paper's method -----------------------------------------------------
def test_defaults_and_paper_rules():
    psf = gaussian2d()
    var = 9.0
    _, y = simulate((24, 28), psf, 10.0, var, seed=1)
    assert bool((y < 0).any())  # read noise makes some pixels negative
    est = Chouzenoux(psf, max_iter=3, dtype=F64).fit(y, init=sensor(var))
    assert Chouzenoux(psf).is_faithful
    assert Chouzenoux(psf).options["max_iter"] == 5000
    assert Chouzenoux(psf).options["tol"] == 1e-5
    # data are not clipped: x₀ is the plain mean
    assert float(est.mean) == pytest.approx(float(y.mean()))
    assert float(est.peak) == pytest.approx(float(y.max()))
    # default weights κ / (q₉₉.₉ − q₅₀) (not in the paper)
    q = torch.quantile(y.flatten(), torch.tensor([0.5, 0.999], dtype=F64))
    spread = float(q[1] - q[0])
    from optix.deconv import _chouzenoux as mod

    assert float(est.tv) == pytest.approx(mod._KAPPA_TV / spread)
    assert float(est.hessian) == pytest.approx(mod._KAPPA_HESSIAN / spread)
    # μ of Eq. (2.13) with ‖A‖ = 1, γ of Theorem 3.2 (iii)
    ymax = float(y.max())
    mu = (1 - math.exp(-1 / var)) * math.exp((2 * ymax - 1) / var)
    assert float(est.lipschitz) == pytest.approx(mu, rel=1e-12)
    assert float(est.step) == pytest.approx(primal_dual_step(mu, TV_H_NORM2))
    # TV alone: one term in δ
    tv = Chouzenoux(psf, hessian=0.0, max_iter=3, dtype=F64)
    tv.fit(y, init=sensor(var))
    assert float(tv.step) == pytest.approx(primal_dual_step(mu, [8.0]))
    assert Chouzenoux(psf, hessian=0.0).is_faithful


def test_algorithm1_written_out():
    """The estimator is Algorithm 1 with the exact likelihood, written out
    line by line here."""
    psf = gaussian2d(5, 1.2)
    var = 4.0
    _, y = simulate((16, 20), psf, 8.0, var, seed=2)
    est = Chouzenoux(psf, max_iter=25, tol=0.0, dtype=F64)
    out = est.fit(y, init=sensor(var)).apply(y)
    assert est.history.extras["iterations"] == [25.0]
    op = PaddedConvolution(psf, y.shape)
    fid = PoissonGaussian(y, var)

    def grad_h(x):
        return op.adjoint(fid.grad(op.forward(x)))  # Eq. (2.8)

    ops = [Gradient(2), Hessian(2)]
    lams = [float(est.tv), float(est.hessian)]
    gamma = float(est.step)
    x = torch.full(op.grid, float(y.mean()), dtype=F64)
    v = [o.forward(x) * 0 for o in ops]
    for _ in range(25):
        y1 = x - gamma * (
            grad_h(x) + sum(o.adjoint(w) for o, w in zip(ops, v, strict=True))
        )
        p1 = y1.clamp(min=0)  # prox of γ ι_C
        p2 = []
        for r, (o, lam) in enumerate(zip(ops, lams, strict=True)):
            y2 = v[r] + gamma * o.forward(x)
            prox = GroupL2(lam).prox(y2 / gamma, 1 / gamma)
            p2.append(y2 - gamma * prox)
            q2 = p2[r] + gamma * o.forward(p1)
            v[r] = v[r] - y2 + q2
        back = sum(o.adjoint(w) for o, w in zip(ops, p2, strict=True))
        q1 = p1 - gamma * (grad_h(p1) + back)
        x = x - y1 + q1
    torch.testing.assert_close(out, op.crop(p1), rtol=1e-10, atol=1e-10)


def test_optimality_certificate_and_shifted_poisson():
    """At convergence: no feasible perturbation lowers the exact objective
    (a global check by convexity), and the exact objective is below its
    value at the minimiser of the shifted-Poisson approximation (Ikoma's
    model) with the same prior."""
    psf = gaussian2d(5, 1.2)
    var = 9.0
    _, y = simulate((16, 16), psf, 10.0, var, seed=3)
    tv, hessian = 0.05, 0.1
    op = PaddedConvolution(psf, y.shape)
    fid = PoissonGaussian(y, var)

    def grad(x):
        return op.adjoint(fid.grad(op.forward(x)))

    terms = [(Gradient(2), GroupL2(tv)), (Hessian(2), GroupL2(hessian))]
    res = primal_dual(
        grad,
        terms,
        torch.full(op.grid, float(y.mean()), dtype=F64),
        step=primal_dual_step(fid.lipschitz, TV_H_NORM2),
        constraint=Box(0.0),
        max_iter=30000,
        tol=1e-10,
    )
    xh = res.x
    best = objective(xh, y, var, op, tv, hessian)
    gen = torch.Generator().manual_seed(4)
    scale = float(xh.mean())
    for t in (1e-3, 1e-2, 1e-1):
        for _ in range(20):
            d = torch.randn(xh.shape, generator=gen, dtype=F64)
            z = (xh + t * scale * d).clamp(min=0)
            assert objective(z, y, var, op, tv, hessian) >= best - 1e-9
    # the shifted-Poisson minimiser of the same prior (ADMM)
    region = tuple(slice(0, n) for n in op.shape)
    sp = admm_fft(
        [
            Term(
                Convolution(op.otf, op.grid),
                ShiftedPoisson(y.clamp(min=-var), var),
                region,
            ),
            Term(None, Box(0.0)),
            Term(Gradient(2), GroupL2(tv)),
            Term(Hessian(2), GroupL2(hessian)),
        ],
        op.grid,
        rho=0.1,
        max_iter=3000,
        tol=1e-9,
    ).x.clamp(min=0)
    assert best <= objective(sp, y, var, op, tv, hessian)
    assert float((sp - xh).norm() / xh.norm()) < 0.1  # close models


def test_recovery_and_constraint():
    psf = gaussian2d()
    var = 9.0
    x, y = simulate((24, 24), psf, 12.0, var, seed=5)
    est = Chouzenoux(psf, dtype=F64).fit(y, init=sensor(var))
    out = est.apply(y)
    assert snr(out, x) > snr(y, x) + 3
    assert float(out.min()) >= 0
    assert est.history.extras["iterations"][0] < 5000  # stopped by tol
    assert est.history.extras["change"][0] <= 1e-5
    # C = [0, x₊]
    capped = Chouzenoux(psf, upper=8.0, max_iter=200, dtype=F64)
    out = capped.fit(y, init=sensor(var)).apply(y)
    assert float(out.max()) <= 8.0 and float(out.min()) >= 0
    assert Chouzenoux(psf, upper=8.0).is_faithful


def test_step_collapse_raises():
    """μ ∝ e^{(2 max y − 1)/σ²} (Eq. 2.13): high counts with low read noise
    leave no usable step; the error points to Ikoma."""
    psf = gaussian2d()
    _, y = simulate((24, 24), psf, 100.0, 1.0, seed=6)
    with pytest.raises(ValueError, match="Ikoma"):
        Chouzenoux(psf).fit(y, init=sensor(1.0))
    # also when μ overflows float64
    with pytest.raises(ValueError, match="Ikoma"):
        Chouzenoux(psf).fit(y * 10, init=sensor(1.0))
    # a given step skips the check (not the paper's method)
    est = Chouzenoux(psf, step=1e-3, max_iter=2).fit(y, init=sensor(1.0))
    assert not est.is_faithful and float(est.step) == 1e-3


def test_needs_read_noise():
    psf = gaussian2d()
    _, y = simulate((16, 16), psf, 10.0, 4.0, seed=7)
    with pytest.raises(ValueError, match="read-noise variance"):
        Chouzenoux(psf).fit(y)
    with pytest.raises(ValueError, match="read-noise variance"):
        Chouzenoux(psf).fit(y, init=sensor(0.0))


def test_sensor_units_and_variance_map():
    psf = gaussian2d(5, 1.2)
    var = 4.0
    _, e = simulate((20, 24), psf, 8.0, var, seed=8)
    ref = Chouzenoux(psf, max_iter=30, dtype=F64).fit(e, init=sensor(var))
    gain, offset = 2.5, 100.0
    raw = e * gain + offset
    out = chouzenoux(
        raw,
        psf,
        sensor=sensor(var * gain**2, offset, gain),
        max_iter=30,
        dtype=F64,
    )
    torch.testing.assert_close(out, ref.apply(e))
    # per-pixel read noise: μ is the maximum over the pixels
    vmap = torch.full(e.shape, var, dtype=F64)
    vmap[3, 4] = 16.0
    est = Chouzenoux(psf, max_iter=2, dtype=F64).fit(e, init=sensor(vmap))
    eta = -torch.expm1(-1 / vmap) * torch.exp((2 * e - 1) / vmap)
    assert float(est.lipschitz) == pytest.approx(float(eta.max()))


def test_default_weights_near_best():
    """The default weights (not in the paper) are within 1.5 dB of the
    best of a grid around them, in the paper's regime (its image x₁:
    peak 12, σ² = 9)."""
    psf = gaussian2d()
    for peak, var in ((12.0, 9.0),):
        x, y = simulate((20, 20), psf, peak, var, seed=9)
        est = Chouzenoux(psf, dtype=F64).fit(y, init=sensor(var))
        default = snr(est.apply(y), x)
        tv, hessian = float(est.tv), float(est.hessian)
        grid = [
            snr(
                Chouzenoux(psf, tv=tv * f, hessian=hessian * f, dtype=F64)
                .fit(y, init=sensor(var))
                .apply(y),
                x,
            )
            for f in (1 / 4, 4)
        ]
        assert default >= max(grid) - 1.5, (peak, default, grid)
        assert default > snr(y, x) + 3, (peak, default)


def test_blocks():
    psf = gaussian2d(5, 1.0)
    var = 4.0
    _, y = simulate((40, 36), psf, 8.0, var, seed=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        whole = (
            Chouzenoux(psf, max_iter=300, dtype=F64)
            .fit(y, init=sensor(var))
            .apply(y)
        )
    est = Chouzenoux(psf, block=(20, 18), halo=10, max_iter=300, dtype=F64)
    with pytest.warns(RuntimeWarning, match="blocks"):
        out = est.fit(y, init=sensor(var)).apply(y)
    assert len(est.history.extras["iterations"]) == 4
    assert float((out - whole).abs().max()) < 0.1 * float(whole.max())
    assert Chouzenoux(psf, block=(20, 18), halo=10).is_faithful


def test_float32_matches_float64():
    psf = gaussian2d(5, 1.2)
    var = 9.0
    _, y = simulate((20, 20), psf, 12.0, var, seed=11)
    out = {}
    for dt in (torch.float32, F64):
        est = Chouzenoux(psf, max_iter=100, tol=0.0, dtype=dt)
        out[dt] = est.fit(y.to(dt), init=sensor(var)).apply(y.to(dt))
    scale = float(out[F64].abs().max())
    err = float((out[torch.float32].double() - out[F64]).abs().max())
    assert err < 1e-3 * scale, (err, scale)


def test_layout_and_state(tmp_path):
    psf = gaussian2d(5, 1.0)
    var = 4.0
    imgs = [simulate((16, 14), psf, 8.0, var, seed=s)[1] for s in range(3)]
    stack = torch.stack(imgs, dim=-1)
    layout = Layout("yxt", spatial="yx")
    est = Chouzenoux(psf, max_iter=20, dtype=F64, layout=layout)
    out = est.fit(stack, init=sensor(var)).apply(stack)
    assert est.tv.shape == (3,) and est.step.shape == (3,)
    for i, img in enumerate(imgs):
        ref = Chouzenoux(psf, max_iter=20, dtype=F64).fit(
            img, init=sensor(var)
        )
        torch.testing.assert_close(out[..., i], ref.apply(img))
    est.save(tmp_path / "chouzenoux.pt")
    new = load(tmp_path / "chouzenoux.pt")
    new.dtype = F64
    torch.testing.assert_close(new.apply(stack), out)
    torch.testing.assert_close(new.step, est.step)


def test_options_and_faithfulness():
    psf = gaussian2d(5, 1.0)
    assert Chouzenoux(psf).is_faithful
    assert Chouzenoux(psf, faithful=False).is_faithful  # same preset
    for opts in (
        dict(tv=0.01),
        dict(hessian=0.0),
        dict(upper=50.0),
        dict(halo=12),
        dict(block=(8, None)),
    ):
        assert Chouzenoux(psf, **opts).is_faithful, opts
    for opts in (dict(step=0.1), dict(max_iter=10), dict(tol=1e-3)):
        assert not Chouzenoux(psf, **opts).is_faithful, opts
    for bad in (
        dict(tv=-1.0),
        dict(hessian=-1.0),
        dict(upper=0.0),
        dict(step=0.0),
        dict(step=math.inf),
        dict(max_iter=0),
        dict(tol=-1.0),
    ):
        with pytest.raises(ValueError):
            Chouzenoux(psf, **bad)
    with pytest.raises(ValueError, match="needs a psf"):
        Chouzenoux().fit(torch.ones(16, 16), init=sensor(1.0))
    with pytest.raises(ValueError, match="99.9th percentile"):
        Chouzenoux(psf).fit(torch.ones(16, 16), init=sensor(1.0))
