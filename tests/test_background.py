import itertools
import math
import os

import numpy as np
import pytest
import torch

import optix
from optix.background import (
    RollingBall,
    TopHat,
    rolling_ball,
    rolling_ball_halo,
    top_hat,
)
from optix.background.utils import (
    ball,
    footprint,
    grey_dilation,
    grey_erosion,
    grey_opening,
)
from optix.io import Layout, labeled

F64 = dict(dtype=torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def ref():
    path = os.path.join(HERE, "data", "skimage_rolling_ball_fixtures.npz")
    return dict(np.load(path))


def brute(f, b, sign):
    """Direct transcription of Sternberg's min/max formulas."""
    half = [s // 2 for s in b.shape]
    out = np.empty_like(f)
    for x in itertools.product(*(range(n) for n in f.shape)):
        vals = []
        for o in itertools.product(*(range(-k, k + 1) for k in half)):
            h = b[tuple(oi + k for oi, k in zip(o, half, strict=True))]
            if h == -np.inf:
                continue
            y = tuple(xi + sign * oi for xi, oi in zip(x, o, strict=True))
            if all(0 <= yi < n for yi, n in zip(y, f.shape, strict=True)):
                vals.append(f[y] - h if sign > 0 else f[y] + h)
        out[x] = min(vals) if sign > 0 else max(vals)
    return out


def scene(shape=(72, 80), seed=0):
    """Narrow Gaussian spots on a smooth background."""
    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float64) for n in shape),
        indexing="ij",
    )
    bg = 100 + 0.2 * x + 0.1 * y + 10 * torch.sin(x / 30) * torch.cos(y / 40)
    spots = torch.zeros(shape, dtype=torch.float64)
    centres = torch.rand(5, 2, generator=g, dtype=torch.float64)
    for cy, cx in centres * (torch.tensor(shape) - 30) + 15:
        spots += 40 * torch.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 8.0)
    return bg, spots


# -- structuring elements ---------------------------------------------------
def test_ball_heights():
    b = ball(2.5, 2)
    assert b.shape == (5, 5)
    assert b[2, 2] == 0
    assert b[2, 0] == pytest.approx(math.sqrt(2.5**2 - 4) - 2.5)
    assert b[0, 0] == -math.inf  # ‖(2, 2)‖ > 2.5
    assert ball(3, 3).shape == (7, 7, 7)
    e = ball(4, 2, height=8)
    torch.testing.assert_close(e, 2 * ball(4, 2))
    with pytest.raises(ValueError):
        ball(0)


def test_footprints():
    d = footprint(2, 2, "disk")
    assert torch.isfinite(d).sum() == 13
    assert (footprint(2, 2, "square") == 0).all()
    assert torch.isfinite(footprint(1, 3, "disk")).sum() == 7
    with pytest.raises(ValueError):
        footprint(2, 2, "hexagon")
    with pytest.raises(ValueError):
        footprint(1.5)


# -- exact morphology ---------------------------------------------------------
@pytest.mark.parametrize(
    "se",
    [
        ball(3.5, 2),
        ball(2, 2),
        footprint(2, 2, "disk"),
        footprint(2, 2, "square"),
        footprint(0, 2, "square"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_brute_force_2d(se, dtype):
    f = torch.rand(11, 13, generator=torch.Generator().manual_seed(1)) * 10
    f = f.to(dtype)
    b = se.to(dtype)
    e = brute(f.numpy(), b.numpy(), 1)
    assert np.array_equal(grey_erosion(f, b).numpy(), e)
    d = brute(e, b.numpy(), -1)
    assert np.array_equal(grey_dilation(torch.as_tensor(e), b).numpy(), d)
    assert np.array_equal(grey_opening(f, b).numpy(), d)


def test_matches_brute_force_3d():
    f = torch.rand(6, 7, 8, generator=torch.Generator().manual_seed(2))
    f = f.double() * 4
    for se in (ball(2.2, 3), footprint(1, 3, "disk")):
        e = brute(f.numpy(), se.numpy(), 1)
        d = brute(e, se.numpy(), -1)
        assert np.array_equal(grey_opening(f, se).numpy(), d)


def test_asymmetric_element():
    # erosion reads f(x + o) − B(o), dilation f(x − o) + B(o)
    f = torch.rand(9, 10, generator=torch.Generator().manual_seed(3))
    f = f.double()
    se = ball(2, 2)
    se[0, :] = -math.inf
    se[1, 3] = 0.3
    assert np.array_equal(
        grey_erosion(f, se).numpy(), brute(f.numpy(), se.numpy(), 1)
    )
    assert np.array_equal(
        grey_dilation(f, se).numpy(), brute(f.numpy(), se.numpy(), -1)
    )
    flat = footprint(2, 2, "square")
    flat[0, :3] = -math.inf
    flat[4, 4] = -math.inf
    assert np.array_equal(
        grey_erosion(f, flat).numpy(), brute(f.numpy(), flat.numpy(), 1)
    )
    assert np.array_equal(
        grey_dilation(f, flat).numpy(), brute(f.numpy(), flat.numpy(), -1)
    )


def test_opening_properties():
    f = torch.rand(4, 20, 24, generator=torch.Generator().manual_seed(4))
    f = f.double()
    se = ball(3, 2)
    op = grey_opening(f, se)
    assert (op <= f + 1e-12).all()  # anti-extensive
    torch.testing.assert_close(grey_opening(op, se), op)  # idempotent
    # batch dims are independent
    torch.testing.assert_close(op[2], grey_opening(f[2], se))


def test_bad_structure():
    f = torch.rand(5, 5)
    with pytest.raises(ValueError):
        grey_erosion(f, torch.zeros(2, 3))
    se = footprint(1)
    se[1, 1] = -math.inf
    with pytest.raises(ValueError):
        grey_erosion(f, se)


# -- scikit-image ------------------------------------------------------------
@pytest.mark.parametrize(
    "key, image, radius, ndim",
    [
        ("ball2d_5", "image2d", 5, 2),
        ("ball2d_7_5", "image2d", 7.5, 2),
        ("ball3d_3", "image3d", 3, 3),
    ],
)
def test_matches_skimage(ref, key, image, radius, ndim):
    # scikit-image returns the erosion by the ball (the apex of the ball);
    # the rolling-ball background of the paper is its dilation
    f = torch.as_tensor(ref[image])
    sk = torch.as_tensor(ref[key])
    se = ball(radius, ndim, dtype=torch.float64)
    torch.testing.assert_close(grey_erosion(f, se), sk, rtol=0, atol=1e-12)
    expected = torch.minimum(grey_dilation(sk, se), f)
    background = rolling_ball(f, radius, ndim=ndim, output="background")
    torch.testing.assert_close(background, expected, rtol=0, atol=1e-12)


# -- background removal ------------------------------------------------------
@pytest.mark.parametrize("radius", [12, 25])
def test_removes_smooth_background(radius):
    bg, spots = scene()
    corrected = rolling_ball(bg + spots, radius)
    assert (corrected >= 0).all()
    inner = (slice(10, -10), slice(10, -10))
    assert (corrected - spots)[inner].abs().max() < 2.5
    assert (corrected - spots)[inner].abs().mean() < 0.3
    # the background alone is (nearly) removed entirely
    assert rolling_ball(bg, radius)[inner].abs().max() < 0.05
    torch.testing.assert_close(
        rolling_ball(bg + spots, radius, output="background") + corrected,
        bg + spots,
    )


def test_top_hat_removes_background():
    bg, spots = scene()
    for shape in ("disk", "square"):
        corrected = top_hat(bg + spots, 8, shape=shape)
        assert (corrected >= 0).all()
        inner = (slice(10, -10), slice(10, -10))
        assert (corrected - spots)[inner].abs().mean() < 0.2


def test_top_hat_is_flat_opening():
    f = torch.rand(15, 17, generator=torch.Generator().manual_seed(5))
    f = f.double()
    for shape in ("disk", "square"):
        se = footprint(3, 2, shape)
        expected = brute(brute(f.numpy(), se.numpy(), 1), se.numpy(), -1)
        out = top_hat(f, 3, shape=shape, output="background")
        assert np.array_equal(out.numpy(), expected)


def test_3d_volume():
    z, y, x = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float64) for n in (20, 22, 24)),
        indexing="ij",
    )
    bg = 50 + 0.3 * x + 0.2 * z
    spot = 30 * torch.exp(-((z - 10) ** 2 + (y - 11) ** 2 + (x - 12) ** 2) / 4)
    corrected = rolling_ball(bg + spot, 8, ndim=3)
    assert corrected[10, 11, 12] == pytest.approx(30, rel=0.06)
    assert (corrected - spot).abs().max() < 2.0


def test_float32_matches_float64():
    bg, spots = scene()
    data = bg + spots
    expected = rolling_ball(data, 10)
    out = rolling_ball(data.float(), 10)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.double(), expected, rtol=0, atol=1e-4)
    assert (out >= 0).all()


def test_integer_input():
    data = (scene()[0] + scene()[1]).round().to(torch.int32)
    out = rolling_ball(data, 10)
    assert out.is_floating_point()
    torch.testing.assert_close(
        out, rolling_ball(data.to(out.dtype), 10), rtol=0, atol=0
    )


# -- extensions ------------------------------------------------------------
def test_shrink_approximates_opening():
    bg, spots = scene()
    data = bg + spots
    exact = rolling_ball(data, 20, output="background")
    for s in (2, 4):
        approx = rolling_ball(data, 20, shrink=s, output="background")
        assert (approx <= data).all()
        inner = (slice(10, -10), slice(10, -10))
        assert (approx - exact)[inner].abs().mean() < 0.25 * s
    # shrink=1 is the exact opening
    torch.testing.assert_close(
        rolling_ball(data, 20, shrink=1, output="background"), exact
    )


def test_smooth_reduces_noise():
    bg, spots = scene()
    g = torch.Generator().manual_seed(6)
    noisy = bg + spots + torch.randn(bg.shape, generator=g, dtype=bg.dtype)
    inner = (slice(10, -10), slice(10, -10))
    plain = rolling_ball(noisy, 20, output="background")
    smooth = rolling_ball(noisy, 20, smooth=True, output="background")
    # the opening of noisy data hugs the noise minima; smoothing lifts it
    assert (smooth - bg)[inner].abs().mean() < (plain - bg)[inner].abs().mean()


def test_bad_arguments():
    f = torch.rand(8, 8)
    with pytest.raises(ValueError):
        rolling_ball(f, -1)
    with pytest.raises(ValueError):
        rolling_ball(f, 3, output="both")
    with pytest.raises(ValueError):
        rolling_ball(f, 3, shrink=0)
    with pytest.raises(ValueError):
        rolling_ball(f, 3, ndim=3)
    with pytest.raises(ValueError):
        RollingBall()
    with pytest.raises(ValueError):
        TopHat()


def test_halo():
    assert rolling_ball_halo(10) == 20
    assert rolling_ball_halo(10.7, smooth=True) == 21
    assert rolling_ball_halo(20, shrink=4) == 48
    assert rolling_ball_halo(20, shrink=4, smooth=True) == 52


# -- estimators and streaming ------------------------------------------------
def test_estimator_defaults():
    est = RollingBall(radius=10)
    assert est.is_faithful
    assert est.options == dict(
        radius=10.0, output="corrected", shrink=1, smooth=False
    )
    assert not RollingBall(radius=10, shrink=2).is_faithful
    assert not RollingBall(radius=10, smooth=True).is_faithful
    assert RollingBall(radius=10, output="background").is_faithful
    assert "RollingBall" in optix.estimators()
    assert "TopHat" in optix.estimators()


@pytest.mark.parametrize(
    "kw",
    [
        dict(radius=6.5),
        dict(radius=7, smooth=True),
        dict(radius=7, output="background"),
    ],
)
def test_blocks_match_whole(kw):
    bg, spots = scene((70, 83))
    data = torch.stack([bg + spots, (bg + spots).flip(-1), bg])
    expected = rolling_ball(data, **kw)
    est = RollingBall(**kw, **F64)
    with optix.config(memory=8 * 40 * 40 * 4):
        out = est.apply(data.numpy())
    assert np.array_equal(np.asarray(out), expected.numpy())


@pytest.mark.parametrize(
    "kw", [dict(radius=12, shrink=4), dict(radius=9, shrink=3, smooth=True)]
)
def test_blocks_match_whole_shrink(kw):
    bg, spots = scene((70, 83))
    data = torch.stack([bg + spots, bg])
    expected = rolling_ball(data, **kw)
    with optix.config(memory=8 * 90 * 90 * 4):
        out = RollingBall(**kw, **F64).apply(data)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-10)


def test_top_hat_blocks_match_whole():
    bg, spots = scene((60, 66))
    data = bg + spots
    for shape in ("disk", "square"):
        expected = top_hat(data, 5, shape=shape)
        with optix.config(memory=8 * 30 * 30 * 4):
            out = TopHat(radius=5, shape=shape, **F64).apply(data)
        assert torch.equal(out, expected)


def test_blocks_3d_layout():
    v = torch.rand(2, 18, 20, 22, generator=torch.Generator().manual_seed(7))
    v = v.double() * 10
    expected = rolling_ball(v, 4, ndim=3)
    lay = Layout("tzyx", spatial="zyx")
    with optix.config(memory=8 * 18 * 18 * 18 * 4):
        out = RollingBall(radius=4, layout=lay, **F64).apply(v)
    assert torch.equal(out, expected)
    # spatial axes in a permuted order, from a labelled array
    perm = v.permute(3, 0, 1, 2).contiguous()  # (x, t, z, y)
    lab = labeled(perm, Layout("xtzy", spatial="zyx"))
    out = RollingBall(radius=4, **F64).apply(lab)
    torch.testing.assert_close(out, expected.permute(3, 0, 1, 2))


def test_lazy_zarr():
    zarr = pytest.importorskip("zarr")
    bg, spots = scene((48, 52))
    data = torch.stack([bg + spots, bg + 2 * spots])
    src = zarr.zeros(data.shape, chunks=(1, 24, 26), dtype="f8")
    src[:] = data.numpy()
    dst = zarr.zeros(data.shape, chunks=(1, 24, 26), dtype="f8")
    with optix.config(memory=8 * 40 * 40 * 4):
        RollingBall(radius=6, **F64).apply(src, out=dst)
    assert np.array_equal(dst[:], rolling_ball(data, 6).numpy())


def test_in_place_refused():
    data = np.random.default_rng(0).random((2, 20, 20))
    with pytest.raises(ValueError):
        RollingBall(radius=3, **F64).apply(data, out=data)


def test_save_load(tmp_path):
    est = RollingBall(radius=5, smooth=True)
    path = tmp_path / "rb.pt"
    est.save(path)
    back = optix.load(path)
    assert isinstance(back, RollingBall) and back.options == est.options
