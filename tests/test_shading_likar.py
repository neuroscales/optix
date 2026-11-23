import math

import numpy as np
import pytest
import torch

import optix
from optix.estimates import ShadingEstimate
from optix.io import Layout, labeled
from optix.models import simulate_shading
from optix.shading import Likar
from optix.shading._likar import (
    _basis,
    _blurred_entropy,
    _GreyCounts,
    _paper_coefficients,
    _powell,
)


def scene(shape, *, seed=0, noise=5.0, discs=40):
    """Background 180 with discs at 60, 100 and 140, plus Gaussian noise."""
    H, W = shape
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    u = torch.full(shape, 180.0, dtype=torch.float64)
    scale = min(H, W) / 128
    for k in range(discs):
        cy = int(torch.randint(0, H, (1,), generator=g))
        cx = int(torch.randint(0, W, (1,), generator=g))
        r = float(torch.randint(4, 12, (1,), generator=g)) * scale
        u[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = (60.0, 100.0, 140.0)[
            k % 3
        ]
    return u + noise * torch.randn(shape, generator=g, dtype=torch.float64)


def polynomial_shading(shape, *, seed=100, strength=0.06):
    """Random neutralised quadratic flat and dark (Eqs. 16-17), 8-bit."""
    g = torch.Generator().manual_seed(seed)
    ta = strength * torch.randn(5, generator=g, dtype=torch.float64)
    tm = strength * torch.randn(5, generator=g, dtype=torch.float64)
    B = _basis(shape)
    return ShadingEstimate(
        flat=1 + torch.einsum("k,khw->hw", tm, B),
        dark=255 * torch.einsum("k,khw->hw", ta, B),
    ), np.concatenate([ta.numpy(), tm.numpy()])


def shaded(u, truth):
    """8-bit acquisition of `u` with shading `truth` (Eq. 3)."""
    return simulate_shading(u, truth).clamp(0, 255).round()


@pytest.fixture(scope="module")
def fitted():
    shape = (128, 128)
    u = scene(shape)
    truth, theta = polynomial_shading(shape)
    data = shaded(u[None], truth)
    est = Likar(data_range=(0, 255)).fit(data)
    return est, data, u, truth, theta


def objective(est, data, shape):
    counts, ranges = est._counts(data, shape)
    lo, hi = ranges[()]
    return est._objective(counts[()], lo, hi, _basis(shape))


# -- building blocks ---------------------------------------------------------
def test_partial_interpolation_and_blur():
    w = torch.ones(1, dtype=torch.float64)
    # g′ = 3.25 adds 0.75 to bin 3 and 0.25 to bin 4
    h = _blurred_entropy(torch.tensor([3.25], dtype=torch.float64), w, 8, 0)
    assert float(h) == pytest.approx(
        -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    )
    # triangular window t = 2: weights (1, 2, 3, 2, 1) / 9
    h = _blurred_entropy(torch.tensor([4.0], dtype=torch.float64), w, 8, 2)
    q = np.array([1, 2, 3, 2, 1]) / 9
    assert float(h) == pytest.approx(-(q * np.log(q)).sum())
    # values outside [−G, 2G] are clamped to the end bins
    far = torch.tensor([-1e6, 1e6], dtype=torch.float64)
    h = _blurred_entropy(far, torch.ones(2, dtype=torch.float64), 8, 0)
    assert float(h) == pytest.approx(math.log(2))


def test_basis_neutralisation_and_paper_coefficients():
    H, W = 30, 40
    B = _basis((H, W))
    # Eq. 14-15: zero-mean terms (exactly, up to the pixel-centre sampling)
    assert B.mean((1, 2)).abs().max() < 1e-3
    theta = np.array([0.1, -0.2, 0.3, 0.05, -0.07])
    a = _paper_coefficients(theta, (H, W))
    y = torch.arange(H, dtype=torch.float64) - (H - 1) / 2
    x = torch.arange(W, dtype=torch.float64) - (W - 1) / 2
    Y, X = torch.meshgrid(y, x, indexing="ij")
    eq16 = (
        a[0] * X
        + a[1] * Y
        + a[2] * X * Y
        + a[3] * (X**2 - W**2 / 12)
        + a[4] * (Y**2 - H**2 / 12)
    )
    ref = torch.einsum("k,khw->hw", torch.as_tensor(theta), B)
    torch.testing.assert_close(eq16, ref)
    # sampling keeps full-image coordinates
    torch.testing.assert_close(_basis((H, W), 3), B[:, ::3, ::3])


def test_powell_brent_minimise_quadratic():
    A = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
    c = np.array([1.0, -2.0, 0.5])

    def f(p):
        d = p - c
        return float(d @ A @ d) + 1.0

    p, fret, ok = _powell(f, np.zeros(3), ftol=1e-10, itmax=100)
    assert ok and fret == pytest.approx(1.0, abs=1e-8)
    np.testing.assert_allclose(p, c, atol=1e-4)


def test_grey_counts_merge_exactly():
    g = torch.Generator().manual_seed(0)
    pix = torch.randint(0, 50, (400,), generator=g)
    grey = torch.randint(1, 9, (400,), generator=g)
    one, two = _GreyCounts(8), _GreyCounts(8)
    one.update(pix, grey)
    two.update(pix[:150], grey[:150])
    two.update(pix[150:], grey[150:])
    for a, b in zip(one.pairs(), two.pairs(), strict=True):
        assert torch.equal(a, b)
    assert int(one.counts.sum()) == 400


def test_marginal_grey_values_ignored():
    data = torch.full((1, 8, 8), 100.0)
    data[0, 0, :4] = 0.0  # grey level 1
    data[0, 1, :3] = 255.0  # grey level G
    est = Likar(data_range=(0, 255))
    counts, _ = est._counts(data, (8, 8))
    pixel, grey, count = counts[()].pairs()
    assert int(count.sum()) == 64 - 7
    assert grey.min() > 1 and grey.max() < 256
    # 8-bit data with data_range=(0, 255) use its integer values
    assert torch.equal(grey.unique(), torch.tensor([101]))


# -- fit ---------------------------------------------------------------------
def test_recovers_polynomial_flat_and_dark(fitted):
    est, data, u, truth, _ = fitted
    assert est.is_faithful and est.history.converged
    s = est.estimates["shading"]
    assert s.flat.shape == s.dark.shape == (128, 128)
    assert (s.flat - truth.flat).abs().max() < 0.02
    assert (s.dark - truth.dark).abs().max() < 3.0
    corrected = est.apply(data)[0]
    assert (corrected - u).abs().mean() < 1.0
    assert (data[0] - u).abs().mean() > 5.0
    # Eq. 14-15: zero-mean additive, unit-mean multiplicative component
    assert abs(float(s.dark.mean())) < 0.05
    assert abs(float(s.flat.mean()) - 1) < 1e-3
    assert s.meta["flat_norm"] == "mean" and s.meta["faithful"]
    coef = s.meta["coefficients"][0]
    assert len(coef["additive"]) == len(coef["multiplicative"]) == 5


def test_entropy_decreases(fitted):
    est, data, _, _, theta = fitted
    entropy = [-v for v in est.history.objective]
    steps = np.diff(entropy)
    assert (steps <= 1e-12).all()
    assert entropy[-1] < entropy[0] - 0.5
    assert entropy[-1] == pytest.approx(
        est.estimates["shading"].meta["entropy"][0]
    )
    # the optimum is at least as good as the true shading
    f = objective(est, data, (128, 128))
    assert entropy[-1] <= f(theta) + 1e-9
    assert f(np.zeros(10)) == pytest.approx(entropy[0])


def test_rejects_nonpositive_flat():
    shape = (32, 32)
    data = shaded(scene(shape, discs=10)[None], polynomial_shading(shape)[0])
    f = objective(Likar(data_range=(0, 255)), data, shape)
    theta = np.zeros(10)
    theta[5] = 3.0  # S_M = 1 + 3x̃ < 0 on the left
    assert f(theta) > math.log(3 * 256 + 1)


def test_stack_pools_histograms(fitted):
    """Repeated frames double every count: same entropy, same fit."""
    est, data, *_ = fitted
    s = est.estimates["shading"]
    pooled = Likar(data_range=(0, 255)).fit(torch.cat([data, data]))
    p = pooled.estimates["shading"]
    torch.testing.assert_close(p.flat, s.flat)
    torch.testing.assert_close(p.dark, s.dark)


def test_streaming_parity():
    zarr = pytest.importorskip("zarr")
    shape = (48, 64)
    truth, _ = polynomial_shading(shape, seed=3)
    u = torch.stack([scene(shape, seed=s, discs=15) for s in range(4)])
    data = shaded(u, truth)
    ref = Likar(max_iterations=5).fit(data).estimates["shading"]
    Z = zarr.create_array(
        store={}, shape=data.shape, chunks=(1, 48, 64), dtype="f4"
    )
    Z[:] = data.numpy().astype("f4")
    with optix.config(batch=1):
        out = Likar(max_iterations=5).fit(Z).estimates["shading"]
    torch.testing.assert_close(out.flat.double(), ref.flat.double())
    torch.testing.assert_close(out.dark.double(), ref.dark.double())
    assert out.meta["data_range"] == ref.meta["data_range"]


def test_float32_matches_float64():
    shape = (48, 48)
    truth, _ = polynomial_shading(shape, seed=4)
    data = shaded(scene(shape, seed=4, discs=20)[None], truth)
    a = Likar(max_iterations=5, dtype=torch.float64).fit(data)
    b = Likar(max_iterations=5, dtype=torch.float32).fit(data.float())
    sa, sb = a.estimates["shading"], b.estimates["shading"]
    assert sb.flat.dtype == torch.float32
    torch.testing.assert_close(sb.flat.double(), sa.flat, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(sb.dark.double(), sa.dark, rtol=1e-5, atol=1e-4)
    out = b.apply(data.float())
    torch.testing.assert_close(
        out.double(), a.apply(data), rtol=1e-5, atol=1e-3
    )


def test_groups_and_layout():
    shape = (40, 48)
    stacks, truths = [], []
    for c in range(2):
        truth, _ = polynomial_shading(shape, seed=10 + c)
        truths.append(truth)
        stacks.append(shaded(scene(shape, seed=c, discs=15)[None], truth))
    y = torch.stack(stacks, 1)  # (t, c, y, x)
    layout = Layout("tcyx", spatial="yx", group="c")
    est = Likar(data_range=(0, 255), max_iterations=3).fit(labeled(y, layout))
    s = est.estimates["shading"]
    assert s.flat.shape == s.dark.shape == (2, *shape)
    assert len(s.meta["coefficients"]) == 2
    for c in range(2):
        single = Likar(data_range=(0, 255), max_iterations=3).fit(stacks[c])
        torch.testing.assert_close(s.flat[c], single.estimates["shading"].flat)
    out = est.apply(labeled(y, layout))
    ref = (y - s.dark[None].double()) / s.flat[None].double()
    torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-3)


# -- extensions --------------------------------------------------------------
def test_additive_only():
    shape = (64, 64)
    truth, _ = polynomial_shading(shape, seed=5)
    truth = truth.replace(flat=None)
    u = scene(shape, seed=5, discs=25)
    data = shaded(u[None], truth)
    est = Likar(components="additive", data_range=(0, 255)).fit(data)
    assert not est.is_faithful
    s = est.estimates["shading"]
    assert s.flat is None
    assert (s.dark - truth.dark).abs().max() < 3.0
    assert (est.apply(data)[0] - u).abs().mean() < 1.0
    est = Likar(components="multiplicative", max_iterations=2).fit(data)
    assert est.estimates["shading"].dark is None


def test_sampling(fitted):
    est, data, _, truth, _ = fitted
    sub = Likar(data_range=(0, 255), sampling=2).fit(data)
    assert not sub.is_faithful
    s = sub.estimates["shading"]
    assert s.flat.shape == (128, 128)
    assert (s.flat - truth.flat).abs().max() < 0.03
    assert (s.dark - truth.dark).abs().max() < 4.0


def test_options_and_errors():
    est = Likar()
    assert est.is_faithful
    assert est.options["levels"] == 256 and est.options["window"] == 2
    for kw in (
        {"components": "nope"},
        {"levels": 2},
        {"window": -1},
        {"sampling": 0},
        {"data_range": (5, 5)},
    ):
        with pytest.raises(ValueError):
            Likar(**kw)
    with pytest.raises(ValueError, match="constant"):
        Likar().fit(torch.ones(1, 8, 8))
    with pytest.raises(RuntimeError, match="not fitted"):
        Likar().apply(np.zeros((1, 8, 8)))
    with pytest.raises(ValueError, match="two spatial"):
        Likar().fit(
            torch.rand(2, 8, 8, 8), layout=Layout("tzyx", spatial="zyx")
        )
