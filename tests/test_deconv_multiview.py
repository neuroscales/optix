"""Tests of optix.deconv.MultiviewRL (Preibisch et al. 2014)."""

import math

import pytest
import torch

from optix.deconv import MultiviewRL, multiview_rl, richardson_lucy
from optix.deconv.utils import (
    Multiview,
    PaddedConvolution,
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


def rel_err(x, truth):
    return float((x - truth).norm() / truth.norm())


def four_views(n=4, shape=(64, 64), seed=0):
    truth, gen = scene(shape, seed=seed)
    psfs = [aniso(21, 4.0, 1.0, math.pi * k / n) for k in range(n)]
    return truth, psfs, views_of(truth, psfs, gen)


# -- compound kernels ---------------------------------------------------------
def test_compound_kernels_definitions():
    psfs = [aniso(9, 2.0, 0.8, a) for a in (0.0, 1.0, 2.0)]
    mirror = [p.flip(0, 1) for p in psfs]
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
