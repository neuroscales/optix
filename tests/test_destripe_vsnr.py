"""VSNR (Fehrenbach, Weiss & Lorenzo 2012)."""

import math

import numpy as np
import pytest
import torch

import optix
from optix.destripe import VSNR, utils, vsnr, wavelet_fft
from optix.destripe._vsnr import _embed, _grad, _grad_adjoint
from optix.io import Layout, labeled

F64 = torch.float64
SHAPE = (48, 48)


def content(shape=SHAPE, *, dtype=F64):
    """Piecewise-smooth image (the TV prior) without full-frame edges."""
    h, w = shape
    y, x = torch.meshgrid(
        torch.arange(h, dtype=dtype),
        torch.arange(w, dtype=dtype),
        indexing="ij",
    )
    disc = ((y - 0.3 * h).square() + (x - 0.45 * w).square()) < (0.15 * h) ** 2
    box = (x > 0.6 * w) & (y > 0.5 * h) & (x < 0.9 * w) & (y < 0.85 * h)
    return 3 * disc.to(dtype) + 2 * box.to(dtype) + 0.02 * y


def convolve(lam, psi, shape):
    """b = λ ∗ ψ (periodic), the model of Eq. (4)."""
    fh = torch.fft.rfft2(_embed([psi], shape, lam.dtype, lam.device))[0]
    return torch.fft.irfft2(torch.fft.rfft2(lam) * fh, s=shape)


def rms(a):
    return float((a - a.mean((-2, -1), keepdim=True)).square().mean().sqrt())


def gen(seed):
    return torch.Generator().manual_seed(seed)


def line_case(n=1, sd=0.03, seed=0, shape=SHAPE):
    """Table I: vertical straight lines with Gaussian white weights."""
    psi = utils.line(shape, 0, dtype=F64)
    lam = torch.randn(n, *shape, generator=gen(seed), dtype=F64) * sd
    u = content(shape).expand(n, *shape)
    return u, u + convolve(lam, psi, shape), psi


def sparse_case(shape=SHAPE, seed=0):
    """Elongated Gaussian stripes with sparse (Bernoulli–Gaussian) weights."""
    psi = utils.gaussian_stripe((15, 41), (1.0, 10.0), dtype=F64)
    g = gen(seed)
    keep = torch.rand(*shape, generator=g, dtype=F64) < 0.01
    lam = torch.randn(*shape, generator=g, dtype=F64) * 3 * keep
    u = content(shape)
    return u, u + convolve(lam, psi, shape), psi


# -- filters ------------------------------------------------------------------
def test_filters():
    d = utils.dirac((3, 5), dtype=F64)
    assert d.shape == (3, 5) and d[1, 2] == 1 and d.sum() == 1
    assert utils.dirac().shape == (1, 1)
    ln = utils.line((5, 4), 0)
    assert bool((ln[:, 2] == 1).all()) and ln.sum() == 5
    ln = utils.line((5, 4), -1)
    assert bool((ln[2] == 1).all()) and ln.sum() == 4
    g = utils.gaussian_stripe((7, 9), (1.5, 3.0), dtype=F64)
    x = torch.arange(7, dtype=F64)[:, None] - 3
    y = torch.arange(9, dtype=F64)[None, :] - 4
    torch.testing.assert_close(g, torch.exp(-((x / 1.5) ** 2) - (y / 3) ** 2))
    # θ = π/2 swaps the roles of the axes
    rot = utils.gaussian_stripe((9, 9), (1.5, 3.0), math.pi / 2, dtype=F64)
    torch.testing.assert_close(
        rot, utils.gaussian_stripe((9, 9), (3.0, 1.5), dtype=F64)
    )
    with pytest.raises(ValueError, match="sigma"):
        utils.gaussian_stripe((5, 5), (0.0, 1.0))
    with pytest.raises(ValueError, match="axis"):
        utils.line((5, 5), 2)


def test_embedding_puts_the_centre_at_the_origin():
    psi = utils.gaussian_stripe((5, 7), (1.0, 2.0), dtype=F64)
    emb = _embed([psi], (16, 16), F64, torch.device("cpu"))[0]
    assert emb[0, 0] == psi[2, 3] and emb[-2, -3] == psi[0, 0]
    # convolving a Dirac at p gives ψ centred at p
    lam = torch.zeros(16, 16, dtype=F64)
    lam[8, 8] = 1
    b = convolve(lam, psi, (16, 16))
    torch.testing.assert_close(b[6:11, 5:12], psi)
    with pytest.raises(ValueError, match="larger than the frame"):
        vsnr(torch.zeros(4, 4), [utils.dirac((5, 1))], alpha=1.0)


# -- operators ----------------------------------------------------------------
def test_gradient_adjoint():
    u = torch.randn(2, 9, 11, generator=gen(0), dtype=F64)
    q = torch.randn(2, 2, 9, 11, generator=gen(1), dtype=F64)
    lhs = (_grad(u) * q).sum()
    rhs = (u * _grad_adjoint(q)).sum()
    assert float(lhs) == pytest.approx(float(rhs), rel=1e-12)
    # Section II: forward differences, zero at the last row / column
    g = _grad(u)
    assert bool((g[:, 0, -1] == 0).all() and (g[:, 1, :, -1] == 0).all())
    torch.testing.assert_close(g[:, 0, :-1], u[:, 1:] - u[:, :-1])


def test_operator_norm():
    """Appendix B: exact for periodic differences, an upper bound here."""
    shape = (24, 20)
    filters = [
        utils.dirac(),
        utils.gaussian_stripe((9, 15), (1.0, 4.0), 0.3, dtype=F64),
    ]
    lip = utils.operator_norm(filters, shape, dtype=F64)
    fh = torch.fft.rfft2(_embed(filters, shape, F64, torch.device("cpu")))

    def a(lam, periodic):
        b = torch.fft.irfft2((torch.fft.rfft2(lam) * fh).sum(0), s=shape)
        if periodic:
            return torch.stack([b.roll(-1, 0) - b, b.roll(-1, 1) - b])
        return _grad(b)

    def a_t(q, periodic):
        if periodic:
            w = (q[0].roll(1, 0) - q[0]) + (q[1].roll(1, 1) - q[1])
        else:
            w = _grad_adjoint(q)
        return torch.fft.irfft2(torch.fft.rfft2(w) * fh.conj(), s=shape)

    for periodic in (True, False):
        lam = torch.randn(2, *shape, generator=gen(2), dtype=F64)
        for _ in range(3000):
            lam = a_t(a(lam, periodic), periodic)
            lam = lam / lam.norm()
        est = float(a(lam, periodic).norm())
        if periodic:
            assert est == pytest.approx(lip, rel=1e-3)
        else:
            assert 0.9 * lip < est <= lip * (1 + 1e-9)


# -- solver -------------------------------------------------------------------
def test_quadratic_case_matches_conjugate_gradients():
    """Tikhonov case (Section V): ℓ² prior and ε above every |∇u|.

    The problem is then min (1/2ε)‖∇(u₀ − Hλ)‖² + α/2 ‖λ‖², solved here
    by conjugate gradients on the normal equations; this also exercises
    the constant steps of Algorithm 2 for ε > 0.
    """
    shape = (20, 24)
    psi = utils.gaussian_stripe((7, 15), (1.0, 4.0), dtype=F64)
    u0 = 0.1 * content(shape) + 0.05 * torch.randn(
        *shape, generator=gen(3), dtype=F64
    )
    eps, alpha = 1.0, 0.5
    fh = torch.fft.rfft2(_embed([psi], shape, F64, torch.device("cpu")))[0]

    def h(x):
        return torch.fft.irfft2(torch.fft.rfft2(x) * fh, s=shape)

    def h_t(x):
        return torch.fft.irfft2(torch.fft.rfft2(x) * fh.conj(), s=shape)

    def normal(x):
        return h_t(_grad_adjoint(_grad(h(x)))) / eps + alpha * x

    rhs = h_t(_grad_adjoint(_grad(u0))) / eps
    x = torch.zeros_like(rhs)
    r = rhs - normal(x)
    p = r.clone()
    for _ in range(2000):
        ap = normal(p)
        a = (r * r).sum() / (p * ap).sum()
        x = x + a * p
        r_new = r - a * ap
        if r_new.norm() < 1e-13 * rhs.norm():
            break
        p = r_new + (r_new * r_new).sum() / (r * r).sum() * p
        r = r_new
    sol = utils.vsnr_solve(
        u0,
        [psi],
        alpha=alpha,
        prior="l2",
        eps=eps,
        tol=1e-12,
        max_iter=20000,
        history=True,
    )
    assert float(_grad(sol.image).norm(dim=-3).max()) < eps  # quadratic regime
    assert int(sol.iterations) < 20000
    # G is α-strongly convex: ‖λ − λ*‖² ≤ 2Δ/α
    gap = float(sol.gap[-1])
    assert 0 <= gap <= 1e-12 * float(sol.gap[0])
    assert float((sol.weights[0] - x).square().sum()) <= 2 * gap / alpha
    torch.testing.assert_close(sol.noise, h(x), rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "prior, eps", [("l1", 0.0), ("l2", 0.0), ("l2", 0.05), ("l1", 0.05)]
)
def test_objective_and_gap_decrease(prior, eps):
    u, u0, psi = sparse_case()
    alpha = 3.0 if prior == "l1" else 30.0
    kw = dict(alpha=alpha, prior=prior, eps=eps, bound=50.0)
    sol = utils.vsnr_solve(u0, [psi], max_iter=400, history=True, **kw)
    ref = utils.vsnr_solve(u0, [psi], max_iter=5000, tol=1e-6, **kw)
    best = float(
        utils.vsnr_solve(
            u0, [psi], max_iter=5000, tol=1e-6, history=True, **kw
        ).primal.min()
    )
    p, gap = sol.primal, sol.gap
    assert p.shape == gap.shape == (int(sol.iterations) + 1,)
    # the gap bounds the suboptimality (Theorem 5.1, item 4)
    assert bool((gap >= -1e-9 * gap[0]).all())
    assert bool((p - best <= gap + 1e-9 * gap[0]).all())
    # objective and gap decrease (checkpoints: Chambolle–Pock is not
    # monotone iteration by iteration)
    k = len(p) - 1
    marks = [0, k // 8, k // 2, k]
    assert all(p[b] < p[a] for a, b in zip(marks, marks[1:], strict=False))
    assert all(gap[b] < gap[a] for a, b in zip(marks, marks[1:], strict=False))
    assert float(p[-1] - best) < 0.02 * float(p[0] - best)
    assert float(ref.image.sub(sol.image).abs().max()) < 0.25


def test_stops_on_relative_gap():
    u, u0, psi = line_case()
    sol = utils.vsnr_solve(u0, [psi], alpha=100.0, prior="l2", history=True)
    k = int(sol.iterations)
    assert 0 < k < 1000 and sol.gap.shape == (k + 1, 1)
    assert float(sol.gap[-1, 0]) <= 1e-3 * float(sol.gap[0, 0])
    assert float(sol.gap[-2, 0]) > 1e-3 * float(sol.gap[0, 0])
    # Δ(λ₀, q₀) = P(0) = ‖∇u₀‖₁
    tv = float(_grad(u0).norm(dim=-3).sum())
    assert float(sol.gap[0, 0]) == pytest.approx(tv, rel=1e-12)
    capped = utils.vsnr_solve(u0, [psi], alpha=100.0, prior="l2", max_iter=7)
    assert int(capped.iterations) == 7


def test_removes_line_stripes_better_than_munch():
    """The setting of Table I (vertical lines, Gaussian weights)."""
    for sd in (0.01, 0.03, 0.1):
        u, u0, psi = line_case(sd=sd, seed=int(100 * sd))
        out = vsnr(u0, psi, alpha=100.0, prior="l2")
        assert rms(out - u) < 0.25 * rms(u0 - u)
        assert rms(out - u) < 0.25 * rms(wavelet_fft(u0) - u)


def test_removes_sparse_gaussian_stripes():
    u, u0, psi = sparse_case()
    sol = utils.vsnr_solve(u0, [psi], alpha=3.0, max_iter=1500)
    assert rms(sol.image - u) < 0.1 * rms(u0 - u)
    assert rms(wavelet_fft(u0) - u) > 3 * rms(sol.image - u)
    assert sol.weights.shape == (1, *SHAPE)
    torch.testing.assert_close(sol.image + sol.noise, u0)


def test_two_components_dirac_and_stripes():
    """Section VI: m = 2, a Dirac (white noise) and a stripe pattern."""
    u, u0, psi = line_case(sd=0.05, seed=4)
    white = 0.1 * torch.randn(*SHAPE, generator=gen(9), dtype=F64)
    u0 = u0 + white
    out = vsnr(
        u0, [utils.dirac(), psi], alpha=[10.0, 100.0], prior="l2", tol=1e-4
    )
    assert rms(out - u) < 0.4 * rms(u0 - u)
    # each component alone does worse
    stripes_only = vsnr(u0, psi, alpha=100.0, prior="l2", tol=1e-4)
    assert rms(out - u) < rms(stripes_only - u)


def test_uniform_prior():
    """φ = χ(‖λ‖∞ ≤ α): bounded weights."""
    u, u0, psi = sparse_case()
    sol = utils.vsnr_solve(u0, [psi], alpha=0.5, prior="linf", max_iter=300)
    assert float(sol.weights.abs().max()) <= 0.5 + 1e-12
    assert rms(sol.image - u) < rms(u0 - u)


def test_each_filter_type():
    shape = (32, 32)
    u = content(shape)
    cases = {
        "dirac": utils.dirac(),
        "line": utils.line(shape, 0),
        "gaussian": utils.gaussian_stripe((31, 9), (8.0, 1.0)),
    }
    for name, psi in cases.items():
        lam = 0.05 * torch.randn(*shape, generator=gen(5), dtype=F64)
        u0 = u + convolve(lam, psi.double(), shape)
        out = vsnr(
            u0, psi, alpha=50.0 if name == "dirac" else 30.0, prior="l2"
        )
        assert rms(out - u) < 0.6 * rms(u0 - u), name


def test_orientation_is_transposition():
    u, u0, psi = line_case()
    kw = dict(alpha=100.0, prior="l2")
    vertical = vsnr(u0, psi, **kw)
    horizontal = vsnr(
        u0.transpose(-1, -2), utils.line(SHAPE, 1, dtype=F64), **kw
    )
    torch.testing.assert_close(horizontal.transpose(-1, -2), vertical)


def test_box_bound():
    u, u0, psi = sparse_case()
    sol = utils.vsnr_solve(u0, [psi], alpha=3.0, bound=0.5, max_iter=100)
    assert float(sol.weights.abs().max()) <= 0.5
    assert bool((sol.bound == 0.5).all())
    # default: P(0) / α for ℓ¹
    sol = utils.vsnr_solve(u0, [psi], alpha=3.0, max_iter=0)
    tv = float(_grad(u0).norm(dim=-3).sum())
    assert float(sol.bound) == pytest.approx(tv / 3.0)
    sol = utils.vsnr_solve(u0, [psi], alpha=2.0, prior="l2", max_iter=0)
    assert float(sol.bound) == pytest.approx(math.sqrt(tv))
    assert int(sol.iterations) == 0
    torch.testing.assert_close(sol.image, u0)


def test_l1_is_scale_equivariant():
    u, u0, psi = sparse_case()
    a = vsnr(u0, psi, alpha=3.0, max_iter=60)
    b = vsnr(1000 * u0, psi, alpha=3.0, max_iter=60)
    torch.testing.assert_close(b, 1000 * a)


def test_float32_matches_float64():
    u, u0, psi = line_case()
    kw = dict(alpha=100.0, prior="l2", max_iter=150, tol=0.0)
    hi = vsnr(u0, psi, **kw)
    lo = vsnr(u0.float(), psi.float(), **kw)
    assert lo.dtype == torch.float32
    assert float((lo.double() - hi).abs().max()) < 1e-4 * float(u0.abs().max())


def test_device_and_dtype(device, dtype):
    u, u0, psi = line_case(n=2)
    data = u0.to(device=device, dtype=dtype)
    out = vsnr(data, psi, alpha=100.0, prior="l2")
    assert out.device.type == device.type and out.dtype == dtype
    assert rms(out.double().cpu() - u) < 0.25 * rms(u0 - u)


def test_errors():
    u0 = torch.zeros(8, 8)
    psi = utils.dirac()
    for kw, match in [
        (dict(alpha=0.0), "alpha"),
        (dict(alpha=1.0, prior="tv"), "prior"),
        (dict(alpha=[1.0, 2.0]), "one per filter"),
        (dict(alpha=1.0, eps=-1.0), "eps"),
        (dict(alpha=1.0, max_iter=-1), "max_iter"),
        (dict(alpha=1.0, bound=0.0), "bound"),
        (dict(alpha=1.0, tau=1.0, sigma=1.0), "tau"),
    ]:
        with pytest.raises(ValueError, match=match):
            vsnr(u0, psi, **kw)
    with pytest.raises(ValueError, match="filters"):
        vsnr(u0, torch.zeros(1, 1, 1, 1), alpha=1.0)
    with pytest.raises(ValueError, match="filters and alpha"):
        VSNR(psi)


def test_integer_and_batched_input():
    u, u0, psi = line_case(n=3)
    out = vsnr((100 * u0).round().to(torch.int32), psi, alpha=1.0, max_iter=5)
    assert out.is_floating_point()
    stack = u0.reshape(3, 1, *SHAPE)
    out = vsnr(stack, psi, alpha=100.0, prior="l2")
    assert out.shape == stack.shape
    for i in range(3):
        torch.testing.assert_close(
            out[i, 0], vsnr(u0[i], psi, alpha=100.0, prior="l2")
        )


# -- estimator ----------------------------------------------------------------
def test_estimator_is_stateless(tmp_path):
    u, u0, psi = line_case(n=2)
    kw = dict(alpha=100.0, prior="l2")
    est = VSNR(psi, **kw, dtype=F64)
    assert est.is_faithful and est.fit() is est and est.estimates == {}
    assert est.fit(u0) is est
    assert "VSNR" in optix.estimators()
    assert est.tags.streaming == "per-frame"
    expected = vsnr(u0, psi, **kw)
    torch.testing.assert_close(est.apply(u0), expected)
    est = VSNR(psi, **kw)
    est.save(tmp_path / "v.pt")
    back = optix.load(tmp_path / "v.pt")
    assert isinstance(back, VSNR)
    torch.testing.assert_close(back.apply(u0), est.apply(u0))
    assert VSNR.paper["tol"] == 1e-3


def test_stream_matches_batch():
    u, u0, psi = line_case(n=5)
    kw = dict(alpha=100.0, prior="l2")
    expected = vsnr(u0, psi, **kw)
    est = VSNR(psi, **kw, dtype=F64)
    with optix.config(batch=2):
        out = est.apply(u0.numpy())
    torch.testing.assert_close(torch.as_tensor(out), expected)
    out = np.zeros(u0.shape)
    with optix.config(batch=1):
        est.apply(u0.numpy(), out=out)
    torch.testing.assert_close(torch.as_tensor(out), expected)
    # (y, t, x) storage with the frames along t, spatial axes (y, x)
    perm = u0.permute(1, 0, 2).contiguous()
    with optix.config(batch=3):
        out = VSNR(psi, **kw, dtype=F64).apply(
            labeled(perm, Layout("ytx", spatial="yx"))
        )
    torch.testing.assert_close(out.permute(1, 0, 2), expected)
    with pytest.raises(ValueError, match="2 spatial axes"):
        VSNR(psi, **kw, layout=Layout("tyx", spatial="tyx")).apply(u0)


class _Lazy:
    """Array-like that records the size of every read (like zarr)."""

    def __init__(self, a):
        self._a = a
        self.shape = a.shape
        self.dtype = a.dtype
        self.reads = []

    def __getitem__(self, index):
        out = self._a[index]
        self.reads.append(out.size)
        return out


def test_lazy_input():
    u, u0, psi = line_case(n=4)
    kw = dict(alpha=100.0, prior="l2")
    lazy = _Lazy(u0.numpy())
    with optix.config(batch=1):
        out = VSNR(psi, **kw, dtype=F64).apply(lazy)
    assert len(lazy.reads) == 4 and max(lazy.reads) == u0[0].numel()
    torch.testing.assert_close(torch.as_tensor(out), vsnr(u0, psi, **kw))


def test_stream_zarr(tmp_path):
    zarr = pytest.importorskip("zarr")
    u, u0, psi = line_case(n=3)
    kw = dict(alpha=100.0, prior="l2")
    src = zarr.open_array(
        str(tmp_path / "in.zarr"),
        mode="w",
        shape=u0.shape,
        chunks=(1, *SHAPE),
        dtype="f8",
    )
    src[:] = u0.numpy()
    dst = zarr.open_array(
        str(tmp_path / "out.zarr"),
        mode="w",
        shape=u0.shape,
        chunks=(1, *SHAPE),
        dtype="f8",
    )
    VSNR(psi, **kw, dtype=F64).apply(src, out=dst)
    torch.testing.assert_close(torch.as_tensor(dst[:]), vsnr(u0, psi, **kw))
