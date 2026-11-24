"""Tests of optix.deconv.MultiviewRL (Preibisch et al. 2014)."""

import math

import pytest
import torch

from optix.deconv import MultiviewRL, multiview_rl, richardson_lucy
from optix.deconv._multiview import _normalise_weights
from optix.deconv.utils import (
    BackProjection,
    Multiview,
    PaddedConvolution,
    chi2_probability,
    compound_kernels,
    rl_block,
)
from optix.io import Layout
from optix.psf import lightsheet_psf

F64 = torch.float64
KERNELS = ("bayesian", "efficient", "optimization-1", "optimization-2")
UPDATES = ("sequential", "multiplicative", "additive")


def aniso(size, long, short, angle, dtype=F64):
    """2-D Gaussian PSF elongated by `long` along the direction `angle`."""
    x = torch.arange(size, dtype=dtype) - size // 2
    y, x = torch.meshgrid(x, x, indexing="ij")
    c, s = math.cos(angle), math.sin(angle)
    u, v = c * x + s * y, -s * x + c * y
    g = torch.exp(-(u**2) / (2 * long**2) - v**2 / (2 * short**2))
    return g / g.sum()


def gaussian(size, sigma, dtype=F64):
    """Separable Gaussian PSF, `sigma` per axis."""
    out = None
    for s in sigma:
        x = torch.arange(size, dtype=dtype) - size // 2
        g = torch.exp(-(x**2) / (2 * s**2))
        out = g if out is None else out[..., None] * g
    return out / out.sum()


def skewed(size, seed):
    """Random PSF without any symmetry (P ≠ P*)."""
    gen = torch.Generator().manual_seed(seed)
    p = torch.rand(size, size, generator=gen, dtype=F64) ** 3
    return p / p.sum()


def comet(size, sigma, tail, angle, dtype=F64):
    """Gaussian core with an exponential tail towards `angle`."""
    x = torch.arange(size, dtype=dtype) - size // 2
    y, x = torch.meshgrid(x, x, indexing="ij")
    c, s = math.cos(angle), math.sin(angle)
    u, v = c * x + s * y, -s * x + c * y
    core = torch.exp(-(u**2 + v**2) / (2 * sigma**2))
    trail = torch.exp(-u.clamp(min=0) / tail - v**2 / (2 * sigma**2))
    g = core + 0.5 * trail * (u > 0)
    return g / g.sum()


def blur(truth, psf):
    op = PaddedConvolution(psf, truth.shape)
    return op.forward(op.pad(truth)).clamp(min=0)


def scene(shape, *, seed=0, n=None, peak=500.0, base=2.0):
    gen = torch.Generator().manual_seed(seed)
    numel = math.prod(shape)
    n = numel // 70 if n is None else n
    truth = torch.full(shape, base, dtype=F64)
    idx = torch.randint(0, numel, (n,), generator=gen)
    truth.view(-1)[idx] += peak
    return truth, gen


def views_of(truth, psfs, gen, scale=20.0):
    return [
        torch.poisson(blur(truth, p) * scale, generator=gen) / scale
        for p in psfs
    ]


def cropped_views(psfs, shape=(64, 64), pad=12, **kw):
    """Noise-free views of a scene that extends `pad` beyond the field."""
    big = tuple(n + 2 * pad for n in shape)
    truth, _ = scene(big, **kw)
    inner = (slice(pad, -pad),) * len(shape)
    return truth[inner], [blur(truth, p)[inner] for p in psfs]


def kl(mv, f, data):
    """Poisson data misfit ∑ y log(y/μ) − y + μ of all views."""
    mu = mv.forward(f).clamp(min=1e-300)
    return float(
        (data * (data / mu).clamp(min=1e-300).log() - data + mu).sum()
    )


def rel_err(x, truth):
    return float((x - truth).norm() / truth.norm())


def four_views(n=4, shape=(64, 64), seed=0):
    truth, gen = scene(shape, seed=seed)
    psfs = [aniso(21, 4.0, 1.0, math.pi * k / n) for k in range(n)]
    return truth, psfs, views_of(truth, psfs, gen)


# -- compound kernels ---------------------------------------------------------
def test_compound_kernels_definitions():
    psfs = [skewed(9, seed) for seed in range(3)]
    mirror = [p.flip(0, 1) for p in psfs]
    assert all(
        float((p - m).abs().max()) > 1e-3
        for p, m in zip(psfs, mirror, strict=True)
    )
    ks = {k: compound_kernels(psfs, k) for k in KERNELS}
    for v in range(3):
        torch.testing.assert_close(ks["bayesian"][v], mirror[v])
        opt2 = mirror[v] ** 3
        torch.testing.assert_close(ks["optimization-2"][v], opt2 / opt2.sum())
        for k in KERNELS:
            assert float(ks[k][v].sum()) == pytest.approx(1.0)
            assert bool((ks[k][v] >= 0).all())
    # Eq. 103 by direct summation of the convolutions
    v, w, u = 0, 1, 2
    full = torch.zeros(27, 27, dtype=F64)

    def conv(a, b):
        out = torch.zeros(
            a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1, dtype=F64
        )
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                out[i : i + a.shape[0], j : j + a.shape[1]] += a * b[i, j]
        return out

    expected = mirror[v].clone()
    for x in (w, u):
        full = conv(conv(mirror[v], psfs[x]), mirror[x])  # 25 x 25
        expected = expected * full[8:17, 8:17]
    torch.testing.assert_close(ks["efficient"][v], expected / expected.sum())
    # one view: every variant is the mirrored PSF
    for k in KERNELS:
        torch.testing.assert_close(compound_kernels(psfs[:1], k)[0], mirror[0])
    # the "bayesian" kernel through BackProjection is Hᵀ
    op = PaddedConvolution(psfs[1], (20, 24))
    r = torch.rand(
        20, 24, dtype=F64, generator=torch.Generator().manual_seed(5)
    )
    back = BackProjection(ks["bayesian"][1], op)
    torch.testing.assert_close(back(r), op.adjoint(r))


@pytest.mark.parametrize("kernel", KERNELS)
def test_asymmetric_psfs_recover_point_position(kernel):
    """Comet-shaped PSFs: the mirroring P* puts the point back in place."""
    truth = torch.full((48, 48), 1.0, dtype=F64)
    truth[20, 27] += 2000.0
    psfs = [comet(15, 1.2, 3.0, a) for a in (0.0, math.pi / 2, 2.4)]
    views = [blur(truth, p) for p in psfs]
    y, x = torch.meshgrid(
        torch.arange(48, dtype=F64), torch.arange(48, dtype=F64), indexing="ij"
    )

    def centroid(img):
        w = (img - img.median()).clamp(min=0)
        w = w * ((y - 20).abs() <= 4) * ((x - 27).abs() <= 4)
        return float((w * y).sum() / w.sum()), float((w * x).sum() / w.sum())

    # the tails pull the centroids of the views off the point by ~0.5 px
    assert (
        max(
            abs(c - t)
            for c, t in zip(centroid(views[1]), (20, 27), strict=True)
        )
        > 0.4
    )
    kw = {"kernel": kernel, "dtype": F64}
    out = multiview_rl(views, psfs, n_iter=30, **kw)
    assert divmod(int(out.argmax()), 48) == (20, 27)
    # early iterates already centred (without the mirroring: off by >= 0.09)
    out = multiview_rl(views, psfs, n_iter=3, **kw)
    assert centroid(out) == pytest.approx((20.0, 27.0), abs=0.05)


def test_compound_kernels_no_underflow_in_float32_range():
    psfs = [gaussian(15, (3.0, 3.0, 3.0)) for _ in range(8)]
    for k in KERNELS:
        b = compound_kernels(psfs, k)[0]
        assert bool(torch.isfinite(b).all())
        assert float(b.float().sum()) == pytest.approx(1.0, rel=1e-6)


# -- one view -----------------------------------------------------------------
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("update", UPDATES)
def test_one_view_is_richardson_lucy(kernel, update):
    truth, gen = scene((40, 36), seed=3)
    psf = aniso(9, 2.0, 1.0, 0.3)
    data = views_of(truth, [psf], gen)[0]
    ref = richardson_lucy(data, psf, n_iter=12)
    out = multiview_rl([data], [psf], kernel=kernel, update=update, n_iter=12)
    assert torch.equal(out, ref)
    # χ² stop as well
    ref = richardson_lucy(data * 20, psf)
    out = multiview_rl([data * 20], [psf], kernel=kernel, chi2_stop=True)
    assert torch.equal(out, ref)


# -- several views ------------------------------------------------------------
def test_views_beat_any_single_view():
    truth, psfs, views = four_views(n=3)
    singles = [
        rel_err(richardson_lucy(v, p, n_iter=50), truth)
        for v, p in zip(views, psfs, strict=True)
    ]
    for kernel in KERNELS:
        out = multiview_rl(views, psfs, kernel=kernel, n_iter=50)
        assert rel_err(out, truth) < 0.6 * min(singles)


def _width(img, axis, reach=6):
    """RMS width of the profile through the peak along `axis`, ±`reach`."""
    peak = divmod(int(img.argmax()), img.shape[1])
    index = list(peak)
    index[axis] = slice(peak[axis] - reach, peak[axis] + reach + 1)
    w = (img[tuple(index)] - img.median()).clamp(min=0)
    x = torch.arange(-reach, reach + 1, dtype=img.dtype)
    m = (w * x).sum() / w.sum()
    return float(((w * (x - m) ** 2).sum() / w.sum()).sqrt())


def test_resolution_is_isotropic_and_beats_single_views():
    """A point seen through PSFs elongated along y and along x."""
    truth = torch.full((48, 48), 1.0, dtype=F64)
    truth[24, 24] += 2000.0
    psfs = [aniso(21, 4.0, 1.0, math.pi / 2), aniso(21, 4.0, 1.0, 0.0)]
    gen = torch.Generator().manual_seed(1)
    views = views_of(truth, psfs, gen, scale=50.0)
    worst_single = []
    for v, p in zip(views, psfs, strict=True):
        img = richardson_lucy(v, p, n_iter=30)
        worst_single.append(max(_width(img, 0), _width(img, 1)))
    out = multiview_rl(views, psfs, n_iter=30)
    worst = max(_width(out, 0), _width(out, 1))
    assert worst < 0.6 * min(worst_single)


def _errors(views, psfs, truth, kernel, update, n):
    """Error after each of n iterations."""
    data = torch.stack(views)
    ops = [PaddedConvolution(p, truth.shape) for p in psfs]
    ks = None if kernel == "bayesian" else compound_kernels(psfs, kernel)
    mv = Multiview(ops, ks, update=update)
    f = float(data.mean())
    errs = []
    for _ in range(n):
        f, _ = rl_block(data, mv, f, n_iter=1, step=mv.step)
        errs.append(rel_err(mv.crop(f), truth))
    return errs


def _iterations_to(errs, target):
    return next((i + 1 for i, e in enumerate(errs) if e <= target), None)


def test_efficient_variants_converge_in_fewer_iterations():
    """Supplementary Note 7.1: efficient < Bayesian, optimisations faster,
    sequential updates about N times faster than combined ones."""
    truth, psfs, views = four_views(n=4)
    n = 40
    errs = {
        (k, u): _errors(views, psfs, truth, k, u, n)
        for k in KERNELS
        for u in ("sequential", "multiplicative")
    }
    target = errs[("bayesian", "sequential")][19]  # 20 iterations
    needed = {key: _iterations_to(e, target) for key, e in errs.items()}
    assert needed[("bayesian", "sequential")] == 20
    seq = [needed[(k, "sequential")] for k in KERNELS]
    assert seq[1] < seq[0] and seq[2] <= seq[1] and seq[3] < seq[2]
    assert seq[3] <= 0.6 * seq[0]
    # combined updates need more iterations than sequential ones
    for k in KERNELS:
        comb = needed[(k, "multiplicative")]
        assert comb is None or comb > 2 * needed[(k, "sequential")]


def test_combined_updates_agree_and_keep_flux():
    """Geometric and arithmetic means of the corrections converge alike
    (Supplementary Fig. 4b); the additive update conserves the flux."""
    truth, psfs, views = four_views(n=3, shape=(48, 48))
    for n in (5, 20):
        kw = {"kernel": "bayesian", "n_iter": n}
        a = rel_err(multiview_rl(views, psfs, update="additive", **kw), truth)
        m = multiview_rl(views, psfs, update="multiplicative", **kw)
        assert abs(rel_err(m, truth) - a) < 0.02 * a
    data = torch.stack(views)
    ops = [PaddedConvolution(p, truth.shape) for p in psfs]
    mv = Multiview(ops, update="additive")
    f, _ = rl_block(data, mv, float(data.mean()), n_iter=5, step=mv.step)
    assert float(mv.forward(f).sum()) == pytest.approx(
        float(data.sum()), rel=1e-10
    )


def test_step_rejects_penalty():
    """Multiview RL-TV is not implemented: a penalty given to rl_block
    with the multiview step hook raises rather than being ignored."""
    truth, psfs, views = four_views(n=2, shape=(32, 32))
    data = torch.stack(views)
    ops = [PaddedConvolution(p, truth.shape) for p in psfs]
    mv = Multiview(ops)
    with pytest.raises(NotImplementedError, match="penalty"):
        rl_block(
            data, mv, 1.0, n_iter=1, step=mv.step, penalty=torch.ones_like
        )


def test_nonnegative_and_finite_with_zeros():
    truth, psfs, views = four_views(n=3, shape=(32, 32))
    views = [v * (torch.rand(v.shape) > 0.3) for v in views]
    for k in KERNELS:
        for u in UPDATES:
            out = multiview_rl(views, psfs, kernel=k, update=u, n_iter=5)
            assert bool(torch.isfinite(out).all()) and bool((out >= 0).all())


def test_psfs_of_different_shapes():
    truth, gen = scene((40, 40), seed=2)
    psfs = [aniso(9, 2.0, 1.0, 0.0), aniso(12, 2.0, 1.0, 1.5)]
    views = views_of(truth, psfs, gen)
    out = multiview_rl(views, psfs, n_iter=10)
    assert out.shape == truth.shape
    est = MultiviewRL(psfs)
    assert [tuple(k.shape) for k in est.kernels(dtype=F64)] == [(13, 13)] * 2


# -- weights (Online Methods, Eq. 3) ------------------------------------------
def test_unit_weights_and_partial_coverage():
    truth, psfs, views = four_views(n=2, shape=(48, 48))
    ones = [torch.ones(48, 48) for _ in psfs]
    plain = multiview_rl(views, psfs, n_iter=8)
    same = multiview_rl(views, psfs, n_iter=8, weights=ones, weight_sum=2.0)
    torch.testing.assert_close(same, plain)
    # w = 1/2 each: every view applies half of its correction
    half = multiview_rl(views, psfs, n_iter=8, weights=ones)
    assert not torch.allclose(half, plain)
    # view 1 covers the left half only: the right half is view 0 alone
    w1 = torch.ones(48, 48)
    w1[:, 24:] = 0
    out = multiview_rl(
        views, psfs, n_iter=8, weights=[torch.ones(48, 48), w1], block=(48, 48)
    )
    alone = multiview_rl(
        views,
        psfs,
        n_iter=8,
        weights=[torch.ones(48, 48), torch.zeros(48, 48)],
    )
    torch.testing.assert_close(out[:, 40:], alone[:, 40:], rtol=0.05, atol=1.0)
    with pytest.raises(ValueError, match="sequential"):
        MultiviewRL(psfs, weights=ones, update="additive")


def test_missing_data_where_weight_is_zero():
    """View 1 has no data (and weight 0) on the right: there, the result
    is view 0 alone, up to the border and on the margin beyond it."""
    psfs = [aniso(21, 4.0, 1.5, 0.0), aniso(21, 4.0, 1.5, math.pi / 2)]
    truth, clean = cropped_views(psfs, base=20.0)
    x = torch.arange(64, dtype=F64)
    fall = 0.5 * (1 + torch.cos(math.pi * (x - 24) / 16))
    taper = torch.where(x < 24, 1.0, torch.where(x < 40, fall, 0.0))
    w1 = taper.expand(64, 64).clone()
    w0 = torch.ones(64, 64, dtype=F64)
    views = [clean[0], clean[1] * (x < 40)]
    ref = richardson_lucy(clean[0], psfs[0], n_iter=30)
    out = multiview_rl(
        views,
        psfs,
        weights=[w0, w1],
        kernel="bayesian",
        n_iter=30,
        block=(64, 64),
    )
    for col in (48, 56, 63):
        diff = (out[:, col] - ref[:, col]).norm() / ref[:, col].norm()
        assert float(diff) < 0.03, col
    # the margin weight of view 1 vanishes beyond the uncovered border
    # (columns 64 to 63 + PSF half-width; the rest of the grid wraps)
    ops = [PaddedConvolution(p, (64, 64)) for p in psfs]
    w = _normalise_weights(torch.stack([w0, w1]), 1.0)
    mv = Multiview(ops, weights=w)
    assert float(mv.weights[1, :, 64:74].max()) < 1e-6
    assert bool(((mv.weights >= 0) & (mv.weights <= 1)).all())


def test_normalise_weights():
    w = torch.tensor(
        [[1.0, 0.0, 0.0, 3.0, 2.0], [1.0, 2.0, 0.0, 1.0, -1.0]], dtype=F64
    )
    out = _normalise_weights(w, 1.0)
    expected = torch.tensor(
        [[0.5, 0.0, 0.0, 0.75, 1.0], [0.5, 1.0, 0.0, 0.25, 0.0]], dtype=F64
    )
    torch.testing.assert_close(out, expected)
    # k = 2: the sum is k where two views overlap, each weight capped at 1
    out = _normalise_weights(w, 2.0)
    expected = torch.tensor(
        [[1.0, 0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.5, 0.0]], dtype=F64
    )
    torch.testing.assert_close(out, expected)
    three = torch.rand(3, 50, generator=torch.Generator().manual_seed(0))
    out = _normalise_weights(three.double(), 2.0)
    assert bool((out <= 1).all())
    capped = (2 * three / three.sum(0) >= 1).any(0)
    torch.testing.assert_close(
        out.sum(0)[~capped], torch.full_like(out[0][~capped], 2.0)
    )


# -- edge damping and the margin ----------------------------------------------
def test_margin_weights_invariants():
    psfs = [aniso(21, 4.0, 1.0, math.pi * k / 3) for k in range(3)]
    ops = [PaddedConvolution(p, (40, 36)) for p in psfs]
    assert Multiview(ops[:1]).weights is None
    for kernels in (None, compound_kernels(psfs, "efficient")):
        mv = Multiview(ops, kernels)
        inside = mv.weights[(slice(None), slice(0, 40), slice(0, 36))]
        assert bool((inside == 1).all())
        assert bool(((mv.weights >= 0) & (mv.weights <= 1)).all())
        # ρᵥ = Hᵥᵀ1 / maxᵤ Hᵤᵀ1 on the margin, for every kernel
        norms = torch.stack([o.norm for o in ops])
        rho = norms / norms.max(0).values.clamp(min=1e-300)
        margin = torch.ones(mv.grid, dtype=torch.bool)
        margin[:40, :36] = False
        support = mv.norm > mv.norm.max() * 1e-6
        sel = margin & support
        torch.testing.assert_close(mv.weights[:, sel], rho[:, sel])


@pytest.mark.parametrize("kernel", ["bayesian", "efficient"])
def test_scene_beyond_the_field(kernel):
    """Light from outside the field: the margin takes it, and the misfit
    of the noise-free views decreases at every iteration."""
    psfs = [aniso(21, 4.0, 1.0, math.pi * k / 3) for k in range(3)]
    truth, clean = cropped_views(psfs)
    data = torch.stack(clean)
    ops = [PaddedConvolution(p, truth.shape) for p in psfs]
    ks = None if kernel == "bayesian" else compound_kernels(psfs, kernel)
    for update in ("sequential", "multiplicative"):
        mv = Multiview(ops, ks, update=update)
        f = float(data.mean())
        misfit = []
        for _ in range(60):
            f, _ = rl_block(data, mv, f, n_iter=1, step=mv.step)
            misfit.append(kl(mv, f, data))
        if kernel == "bayesian":  # the convergence proof (Suppl. Note 2)
            assert all(
                b <= a * (1 + 1e-12)
                for a, b in zip(misfit, misfit[1:], strict=False)
            ), update
        assert misfit[-1] < 0.05 * misfit[0]


@pytest.mark.parametrize("update", ["multiplicative", "additive"])
def test_identical_views_are_one_view(update):
    truth, gen = scene((40, 36), seed=3)
    psf = aniso(9, 2.0, 1.0, 0.3)
    data = views_of(truth, [psf], gen)[0]
    ref = richardson_lucy(data, psf, n_iter=12, dtype=F64)
    kw = {"kernel": "bayesian", "update": update, "n_iter": 12, "dtype": F64}
    for n in (2, 3):
        out = multiview_rl([data] * n, [psf] * n, **kw)
        torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("update", UPDATES)
def test_chi2_stop_with_several_views(update):
    truth, gen = scene((48, 48), seed=1)
    psfs = [aniso(15, 3.0, 1.0, math.pi * k / 3) for k in range(3)]
    data = torch.stack(
        [torch.poisson(blur(truth, p), generator=gen) for p in psfs]
    )
    ops = [PaddedConvolution(p, truth.shape) for p in psfs]
    mv = Multiview(ops, update=update)
    start = float(data.mean())
    f, k = rl_block(data, mv, start, alpha=0.05, max_iter=100, step=mv.step)
    assert 1 < k < 100
    assert chi2_probability(data, mv.forward(f)) >= 0.05
    g, _ = rl_block(data, mv, start, n_iter=k - 1, step=mv.step)
    assert chi2_probability(data, mv.forward(g)) < 0.05
    kw = {"update": update, "kernel": "bayesian", "chi2_stop": True}
    out = multiview_rl(list(data), psfs, dtype=F64, **kw)
    est = MultiviewRL(psfs, dtype=F64, **kw).fit(list(data))
    torch.testing.assert_close(est.apply(list(data)), out)
    assert est.history.extras["iterations"] == [float(k)]


# -- blocks, lazy input, 3-D --------------------------------------------------
@pytest.mark.parametrize("kernel", ["efficient", "optimization-2"])
def test_blocks_match_whole_image_2d(kernel):
    truth, psfs, views = four_views(n=3, shape=(80, 72))
    psfs = [p[5:-5, 5:-5] for p in psfs]  # 11 x 11
    kw = {"kernel": kernel, "n_iter": 10}
    whole = multiview_rl(views, psfs, block=(80, 72), **kw)
    parts = multiview_rl(views, psfs, block=(32, 24), **kw)
    err = float((parts - whole).abs().max() / whole.abs().max())
    assert err < 2e-3


def _lightsheet_views(shape=(20, 24, 22), seed=4):
    det = gaussian(11, (2.5, 1.0, 1.0))  # elongated along the detection axis
    kw = {"waist": 1.5, "wavelength": 0.5, "normalize": True}
    p0 = lightsheet_psf(det, detection=0, **kw)
    p1 = lightsheet_psf(det.permute(2, 1, 0), detection=2, **kw)
    psfs = [p0, p1]
    truth, gen = scene(shape, seed=seed, n=40)
    return truth, psfs, views_of(truth, psfs, gen)


def test_3d_lightsheet_views():
    truth, psfs, views = _lightsheet_views()
    singles = [
        rel_err(richardson_lucy(v, p, n_iter=20), truth)
        for v, p in zip(views, psfs, strict=True)
    ]
    out = multiview_rl(views, psfs, n_iter=20)
    assert rel_err(out, truth) < 0.85 * min(singles)
    parts = multiview_rl(views, psfs, n_iter=20, block=(10, 12, 11))
    err = float((parts - out).abs().max() / out.abs().max())
    assert err < 2e-3


def test_lazy_zarr_views_and_output(tmp_path):
    zarr = pytest.importorskip("zarr")
    truth, psfs, views = four_views(n=2, shape=(64, 48))
    psfs = [p[5:-5, 5:-5] for p in psfs]
    stacks = [torch.stack([v, v.flip(0)]) for v in views]  # 2 samples
    src = []
    for i, s in enumerate(stacks):
        z = zarr.open_array(
            str(tmp_path / f"v{i}.zarr"),
            mode="w",
            shape=s.shape,
            chunks=(1, 16, 16),
            dtype="f4",
        )
        z[:] = s.numpy().astype("f4")
        src.append(z)
    w = zarr.open_array(
        str(tmp_path / "w.zarr"), mode="w", shape=(64, 48), dtype="f4"
    )
    w[:] = 1.0
    dst = zarr.open_array(
        str(tmp_path / "out.zarr"),
        mode="w",
        shape=stacks[0].shape,
        chunks=(1, 16, 16),
        dtype="f4",
    )
    kw = {"n_iter": 6, "block": (32, 16), "dtype": torch.float32}
    multiview_rl(src, psfs, out=dst, weights=[w, w], **kw)
    ref = multiview_rl(
        [s.float() for s in stacks],
        psfs,
        weights=[torch.ones(64, 48)] * 2,
        **kw,
    )
    torch.testing.assert_close(torch.from_numpy(dst[:]), ref)
    # each sample is deconvolved on its own
    one = multiview_rl(
        [s[1].float() for s in stacks],
        psfs,
        weights=[torch.ones(64, 48)] * 2,
        **kw,
    )
    torch.testing.assert_close(ref[1], one, rtol=1e-4, atol=1e-3)


def test_layout_with_time_axis():
    truth, psfs, views = four_views(n=2, shape=(32, 28))
    psfs = [p[5:-5, 5:-5] for p in psfs]
    stacks = [torch.stack([v, 2 * v], dim=-1) for v in views]  # (y, x, t)
    layout = Layout("yxt", spatial="yx")
    out = multiview_rl(stacks, psfs, n_iter=5, layout=layout)
    ref = multiview_rl(views, psfs, n_iter=5)
    torch.testing.assert_close(out[..., 0], ref)


def test_float32_matches_float64():
    truth, psfs, views = four_views(n=3, shape=(48, 48))
    a = multiview_rl(views, psfs, n_iter=15, dtype=F64)
    b = multiview_rl(
        [v.float() for v in views],
        [p.float() for p in psfs],
        n_iter=15,
        dtype="float32",
    )
    assert float((a - b).abs().max() / a.max()) < 1e-4


# -- estimator ----------------------------------------------------------------
def test_faithful_flags_and_validation():
    psfs = [aniso(9, 2.0, 1.0, 0.0), aniso(9, 2.0, 1.0, 1.0)]
    est = MultiviewRL(psfs)
    assert est.is_faithful
    assert est.options["kernel"] == "efficient"
    assert est.options["update"] == "sequential"
    assert MultiviewRL(psfs, kernel="optimization-2").is_faithful is False
    assert MultiviewRL(psfs, chi2_stop=True).is_faithful is False
    with pytest.raises(ValueError, match="kernel"):
        MultiviewRL(psfs, kernel="nope")
    with pytest.raises(ValueError, match="update"):
        MultiviewRL(psfs, update="nope")
    views = [torch.ones(16, 16)] * 2
    with pytest.raises(ValueError, match="views"):
        MultiviewRL(psfs).fit(views[:1])
    with pytest.raises(ValueError, match="same shape"):
        MultiviewRL(psfs).fit([torch.ones(16, 16), torch.ones(16, 15)])
    with pytest.raises(RuntimeError, match="fit"):
        MultiviewRL(psfs).apply(views)
    with pytest.raises(ValueError, match="halo"):
        MultiviewRL(psfs, halo=4, n_iter=1).fit(views).apply(views)


def test_history_and_save_load(tmp_path):
    truth, psfs, views = four_views(n=2, shape=(32, 32))
    est = MultiviewRL(psfs, n_iter=4, block=(16, 32)).fit(views)
    out = est.apply(views)
    assert est.history.extras["iterations"] == [4.0, 4.0]
    path = tmp_path / "mv.pt"
    est.save(path)
    from optix import load

    again = load(path)
    torch.testing.assert_close(again.apply(views), out)


def test_save_keeps_lazy_weights_out(tmp_path):
    zarr = pytest.importorskip("zarr")
    from optix import load

    truth, psfs, views = four_views(n=2, shape=(32, 32))
    lazy = zarr.open_array(
        str(tmp_path / "w.zarr"), mode="w", shape=(32, 32), dtype="f8"
    )
    lazy[:] = 0.5
    dense = torch.full((32, 32), 0.5, dtype=F64)
    est = MultiviewRL(psfs, n_iter=4, weights=[dense, lazy]).fit(views)
    out = est.apply(views)
    est.save(tmp_path / "mv.pt")
    again = load(tmp_path / "mv.pt")
    assert again.weights[1] is None
    torch.testing.assert_close(again.weights[0], dense)
    with pytest.raises(ValueError, match="Lazy weights"):
        again.apply(views)
    again.weights = [dense, lazy]
    torch.testing.assert_close(again.apply(views), out)
