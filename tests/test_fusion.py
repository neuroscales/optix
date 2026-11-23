"""Tile intensity harmonisation (Hörl et al. 2019, Supplementary Note 17)."""

import numpy as np
import pytest
import torch

import optix
from optix.estimates import TileIntensityEstimate
from optix.fusion import IntensityAdjustment, utils
from optix.io import Layout, labeled
from optix.models import random_shading, simulate_tiles


def field(shape=(120, 150), seed=1, dtype=torch.float64):
    """Smooth, textured image of the whole field (no flat regions)."""
    flat = random_shading(shape, order=6, strength=0.5, seed=seed).flat
    return (100 * flat + 20).to(dtype)


def correspondences(tiles, positions):
    """Pixel pairs of every overlapping pair, from a canvas (independent)."""
    pos = [tuple(int(v) for v in p) for p in positions]
    pairs = {}
    for i, (ti, pi) in enumerate(zip(tiles, pos, strict=True)):
        for j, (tj, pj) in enumerate(zip(tiles, pos, strict=True)):
            if j <= i:
                continue
            lo = [max(a, b) for a, b in zip(pi, pj, strict=True)]
            hi = [
                min(a + n, b + m)
                for a, n, b, m in zip(pi, ti.shape, pj, tj.shape, strict=True)
            ]
            if any(h <= lo_ for lo_, h in zip(lo, hi, strict=True)):
                continue
            si = tuple(
                slice(a - p, b - p) for a, b, p in zip(lo, hi, pi, strict=True)
            )
            sj = tuple(
                slice(a - p, b - p) for a, b, p in zip(lo, hi, pj, strict=True)
            )
            pairs[(i, j)] = (
                ti[si].reshape(-1).double(),
                tj[sj].reshape(-1).double(),
            )
    return pairs


def iterate_reference(tiles, positions, weights, fixed=(), n_iter=5000):
    """The per-tile iterative fit of the paper, on the raw pixel pairs.

    Gauss–Seidel over tiles: each tile fits a linear and an additive model
    by least squares to (its pixels, the transformed pixels of the other
    tiles), then combines them with the identity (eq. 9).
    """
    l1, l2, l3 = weights
    pairs = correspondences(tiles, positions)
    t = len(tiles)
    a = np.ones(t)
    c = np.zeros(t)
    for _ in range(n_iter):
        old = np.concatenate([a, c])
        for i in range(t):
            if i in fixed:
                continue
            p, q = [], []
            for (u, v), (yu, yv) in pairs.items():
                if u == i:
                    p.append(yu)
                    q.append(a[v] * yv + c[v])
                elif v == i:
                    p.append(yv)
                    q.append(a[u] * yu + c[u])
            if not p:
                continue
            p = torch.cat(p).numpy()
            q = torch.cat(q).numpy()
            slope, icpt = np.polyfit(p, q, 1)
            trans = (q - p).mean()
            a[i] = l1 * slope + l2 + l3
            c[i] = l1 * icpt + l2 * trans
        if np.abs(np.concatenate([a, c]) - old).max() < 1e-13:
            break
    return a, c


@pytest.fixture(scope="module")
def mosaic():
    return simulate_tiles(field(), (3, 3), overlap=0.2, seed=3)


def test_recovers_gain_offset_with_fixed_tile(mosaic, dtype):
    tiles, pos, truth = mosaic
    tiles = [t.to(dtype) for t in tiles]
    est = IntensityAdjustment(weights=(1, 0, 0), dtype=dtype)
    est.fit(tiles, pos, fixed=[4])
    e = est.estimates["intensity"]
    assert isinstance(e, TileIntensityEstimate)
    assert e.scale.shape == (9,) and e.scale.dtype == dtype
    # relative to the fixed tile 4: aᵢ = sᵢ / s₄, cᵢ = (hᵢ − h₄) / s₄
    s, h = truth.scale.double(), truth.shift.double()
    want_a, want_c = s / s[4], (h - h[4]) / s[4]
    tol = 1e-4 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(e.scale.double(), want_a, rtol=tol, atol=tol)
    torch.testing.assert_close(e.shift.double(), want_c, rtol=0, atol=50 * tol)
    assert e.meta["mse_after"][0] < 1e-6 * e.meta["mse_before"][0]
    # harmonised tiles agree on their overlaps
    out = est.apply(tiles)
    for (i, j), (yi, yj) in correspondences(out, pos).items():
        assert (yi - yj).abs().max() < 1e3 * tol, (i, j)


def test_matches_iterative_scheme(mosaic):
    tiles, pos, _ = mosaic
    for weights in [(0.8, 0.1, 0.1), (0.3, 0.3, 0.4), (0.6, 0.0, 0.4)]:
        e = IntensityAdjustment(weights=weights, dtype=torch.float64)
        e = e.fit(tiles, pos).estimates["intensity"]
        a, c = iterate_reference(tiles, pos, weights)
        np.testing.assert_allclose(e.scale.numpy(), a, rtol=1e-8)
        np.testing.assert_allclose(e.shift.numpy(), c, atol=1e-6)
    e = IntensityAdjustment(weights=(0.9, 0.1, 0.0), dtype=torch.float64)
    e = e.fit(tiles, pos, fixed=[0, 8]).estimates["intensity"]
    a, c = iterate_reference(tiles, pos, (0.9, 0.1, 0.0), fixed=(0, 8))
    np.testing.assert_allclose(e.scale.numpy(), a, rtol=1e-8)
    np.testing.assert_allclose(e.shift.numpy(), c, atol=1e-6)
    assert e.scale[0] == 1 and e.shift[8] == 0


def test_regularised_fit_reduces_mismatch(mosaic):
    tiles, pos, _ = mosaic
    est = IntensityAdjustment().fit(tiles, pos)
    assert est.is_faithful
    m = est.estimates["intensity"].meta
    assert m["mse_after"][0] < 0.05 * m["mse_before"][0]
    # weaker regularisation: closer agreement
    strong = IntensityAdjustment(weights=(0.99, 0.005, 0.005)).fit(tiles, pos)
    weak = strong.estimates["intensity"].meta["mse_after"][0]
    assert weak < m["mse_after"][0]
    assert not strong.is_faithful


def test_noise_and_float32_vs_float64():
    tiles, pos, truth = simulate_tiles(
        field(dtype=torch.float32), (2, 3), overlap=0.25, noise=0.5, seed=7
    )
    kw = {"weights": (1, 0, 0)}
    e32 = IntensityAdjustment(**kw, dtype=torch.float32).fit(
        tiles, pos, fixed=[0]
    )
    e64 = IntensityAdjustment(**kw, dtype=torch.float64).fit(
        [t.double() for t in tiles], pos, fixed=[0]
    )
    a32, a64 = (e.estimates["intensity"].scale.double() for e in (e32, e64))
    torch.testing.assert_close(a32, a64, rtol=1e-5, atol=1e-5)
    # noise attenuates the least-squares slopes only slightly here
    want = truth.scale.double() / truth.scale.double()[0]
    torch.testing.assert_close(a64, want, rtol=2e-2, atol=0)


def test_noise_bias_and_binning():
    # signal std ≈ 50, noise σ = 5 (SNR ≈ 10): errors-in-variables bias
    tiles, pos, truth = simulate_tiles(
        field((240, 300)), (3, 3), overlap=0.2, noise=5.0, seed=3
    )
    s = truth.scale.double()
    want = s / s[4]
    kw = {"weights": (1, 0, 0), "dtype": torch.float64}
    raw = IntensityAdjustment(**kw).fit(tiles, pos, fixed=[4])
    rec = IntensityAdjustment(**kw, faithful=False)
    assert rec.options["downsample"] == 4
    rec.fit(tiles, pos, fixed=[4])
    err_raw = raw.estimates["intensity"].scale / want - 1
    err_rec = rec.estimates["intensity"].scale / want - 1
    # without binning the slopes shrink by several percent ...
    assert float(err_raw.mean()) < -0.03
    # ... binning by 4 brings the bias under 1 % (max under 3 %)
    assert abs(float(err_rec.mean())) < 0.01
    assert float(err_rec.abs().max()) < 0.03


def test_positions_with_unequal_tiles_and_rounding():
    x = field((90, 110))
    tiles = [x[0:50, 0:60], x[30:90, 40:110], x[0:45, 55:110]]
    gains = torch.tensor([1.0, 1.5, 0.7], dtype=torch.float64)
    offsets = torch.tensor([0.0, -10.0, 5.0], dtype=torch.float64)
    tiles = [t * g + o for t, g, o in zip(tiles, gains, offsets, strict=True)]
    pos = [[0.2, -0.3], [30.0, 39.6], [0, 55]]
    e = IntensityAdjustment(weights=(1, 0, 0)).fit(tiles, pos, fixed=[0])
    e = e.estimates["intensity"]
    torch.testing.assert_close(e.scale.double(), 1 / gains, rtol=1e-5, atol=0)
    torch.testing.assert_close(
        e.shift.double(), -offsets / gains, rtol=0, atol=1e-3
    )
    assert e.meta["pairs"] == [[0, 1], [0, 2], [1, 2]]


def test_isolated_tile_stays_identity(mosaic):
    tiles, pos, _ = mosaic
    tiles = [*tiles, tiles[0] * 3]
    pos = torch.cat([pos, torch.tensor([[1000, 1000]])])
    e = IntensityAdjustment().fit(tiles, pos).estimates["intensity"]
    assert e.scale[-1] == 1 and e.shift[-1] == 0


def test_trivial_solution_needs_fixed_tile(mosaic):
    tiles, pos, _ = mosaic
    for weights in [(1, 0, 0), (0.5, 0.5, 0)]:
        with pytest.raises(ValueError, match="fixed tile"):
            IntensityAdjustment(weights=weights).fit(tiles, pos)
    # a fixed tile in the connected set is enough
    IntensityAdjustment(weights=(0.5, 0.5, 0)).fit(tiles, pos, fixed=[3])


def test_constant_overlap_warns():
    x = field((60, 80))
    tiles = [x[:, :50].clone(), x[:, 30:].clone()]
    tiles[0][:, 30:] = 5.0
    with pytest.warns(RuntimeWarning, match="constant"):
        e = IntensityAdjustment().fit(tiles, [[0, 0], [0, 30]])
    assert torch.isfinite(e.estimates["intensity"].scale).all()


def test_downsample_bins_corresponding_pixels(mosaic):
    tiles, pos, truth = mosaic
    est = IntensityAdjustment(weights=(1, 0, 0), downsample=4)
    assert not est.is_faithful
    e = est.fit(tiles, pos, fixed=[0]).estimates["intensity"]
    want = truth.scale.double() / truth.scale.double()[0]
    torch.testing.assert_close(e.scale.double(), want, rtol=1e-5, atol=0)
    # binning reduces the number of pixel pairs, not the agreement
    b = utils.bin_mean(torch.arange(10.0).reshape(2, 5), (2, 2))
    torch.testing.assert_close(b, torch.tensor([[3.0, 5.0, 6.5]]))


def test_max_pixels(mosaic):
    tiles, pos, truth = mosaic
    full = IntensityAdjustment(dtype=torch.float64).fit(tiles, pos)
    big = IntensityAdjustment(max_pixels=10**6, dtype=torch.float64)
    big.fit(tiles, pos)
    assert full.estimates["intensity"].allclose(
        big.estimates["intensity"], rtol=1e-10, atol=1e-8
    )
    # a random subset still fits a noiseless mosaic exactly
    small = IntensityAdjustment(weights=(1, 0, 0), max_pixels=50, seed=0)
    e = small.fit(tiles, pos, fixed=[0]).estimates["intensity"]
    want = truth.scale.double() / truth.scale.double()[0]
    torch.testing.assert_close(e.scale.double(), want, rtol=1e-4, atol=0)
    again = IntensityAdjustment(weights=(1, 0, 0), max_pixels=50, seed=0)
    again.fit(tiles, pos, fixed=[0])
    assert again.estimates["intensity"].allclose(small.estimates["intensity"])


def test_group_and_sample_axes(mosaic):
    tiles, pos, truth = mosaic
    # (c, t, y, x) stored as (y, t, c, x); channel 1 has an extra gain kᵢ
    k = 1 + 0.1 * torch.arange(9, dtype=torch.float64)
    stacks = [
        torch.stack([torch.stack([t, t]), k[i] * torch.stack([t, t])])
        .permute(2, 1, 0, 3)
        .contiguous()
        for i, t in enumerate(tiles)
    ]
    layout = Layout("ytcx", spatial="yx", group="c")
    data = [labeled(s, layout) for s in stacks]
    est = IntensityAdjustment(weights=(1, 0, 0)).fit(data, pos, fixed=[0])
    assert est.layout == layout
    e = est.estimates["intensity"]
    assert e.scale.shape == (2, 9)
    s = truth.scale.double()
    torch.testing.assert_close(
        e.scale[0].double(), s / s[0], rtol=1e-5, atol=0
    )
    s1 = s / k
    torch.testing.assert_close(
        e.scale[1].double(), s1 / s1[0], rtol=1e-5, atol=0
    )
    out = est.apply(data)
    assert out[3].shape == stacks[3].shape
    # (c, t, y, x) view: aᵢ yᵢ + cᵢ per channel
    view = stacks[3].permute(2, 1, 0, 3)
    for c in range(2):
        want = e.scale[c, 3] * view[c].to(out[3].dtype) + e.shift[c, 3]
        torch.testing.assert_close(out[3].permute(2, 1, 0, 3)[c], want)


def test_distinct_samples_are_pooled():
    x1, x2 = field((100, 120), seed=1), field((100, 120), seed=7)
    t1, pos, truth = simulate_tiles(x1, (2, 2), overlap=0.2, seed=3)
    g = 1 / truth.scale.double()
    o = -truth.shift.double() * g
    t2, _, _ = simulate_tiles(
        x2, (2, 2), overlap=0.2, tile_gain=g, tile_offset=o
    )
    stacks = [torch.stack([a, b]) for a, b in zip(t1, t2, strict=True)]
    layout = Layout("tyx", spatial="yx", sample="t")
    w = (0.8, 0.1, 0.1)
    est = IntensityAdjustment(weights=w, dtype=torch.float64, layout=layout)
    e = est.fit(stacks, pos).estimates["intensity"]
    # reference: the pixel pairs of both samples pooled by hand
    boxes = utils.overlaps(pos.tolist(), [t.shape for t in t1])
    pairs = [(i, j) for i, j, _, _ in boxes]
    st = utils.OverlapStats.zeros((len(pairs),))
    for k, (i, j, si, sj) in enumerate(boxes):
        a = torch.cat([t1[i][si].reshape(-1), t2[i][si].reshape(-1)])
        b = torch.cat([t1[j][sj].reshape(-1), t2[j][sj].reshape(-1)])
        st.add((k,), a, b)
    a, c = utils.solve_intensity(pairs, st.n, st.mean, st.scatter, 4, w)
    torch.testing.assert_close(e.scale, a, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(e.shift, c, rtol=1e-12, atol=1e-10)
    # different from either sample alone
    one = IntensityAdjustment(weights=w, dtype=torch.float64).fit(t1, pos)
    assert (one.estimates["intensity"].scale - e.scale).abs().max() > 1e-6


def test_max_pixels_with_groups(mosaic):
    tiles, pos, truth = mosaic
    # channel 1 = 2 · channel 0 + 3: the same transforms up to the offset
    stacks = [torch.stack([t, 2 * t + 3]) for t in tiles]
    layout = Layout("cyx", spatial="yx", group="c")
    kw = {"weights": (1, 0, 0), "layout": layout, "dtype": torch.float64}
    full = IntensityAdjustment(**kw).fit(stacks, pos, fixed=[0])
    small = IntensityAdjustment(**kw, max_pixels=50, seed=0)
    small.fit(stacks, pos, fixed=[0])
    s = truth.scale.double()
    for est in (full, small):
        e = est.estimates["intensity"]
        assert e.scale.shape == (2, 9)
        for g in range(2):
            torch.testing.assert_close(e.scale[g], s / s[0], rtol=1e-6, atol=0)
    # c¹ᵢ = 2 c⁰ᵢ + 3 (1 − aᵢ) maps channel 1 like channel 0
    for est in (full, small):
        e = est.estimates["intensity"]
        want = 2 * e.shift[0] + 3 * (1 - e.scale[0])
        torch.testing.assert_close(e.shift[1], want, rtol=0, atol=1e-6)


def test_refit_infers_layout_again():
    x = field((60, 70))
    t2, p2, _ = simulate_tiles(x, (2, 2), overlap=0.3, seed=1)
    x3 = x[None].repeat(20, 1, 1) + torch.arange(20.0)[:, None, None]
    t3, p3, truth = simulate_tiles(x3.double(), (2, 2, 2), overlap=0.3, seed=1)
    est = IntensityAdjustment(weights=(1, 0, 0))
    est.fit(t2, p2, fixed=[0])
    assert est.layout is None
    est.fit(t3, p3, fixed=[0])
    assert est.layout is None
    e = est.estimates["intensity"]
    assert e.scale.shape == (8,) and e.meta["ndim"] == 3
    out = est.apply(t3)
    want = e.correct(t3[5].to(out[5].dtype), 5)
    torch.testing.assert_close(out[5], want)
    est.fit(t2, p2, fixed=[0])
    assert est.estimates["intensity"].scale.shape == (4,)


def test_partial_bins_are_weighted():
    x = torch.arange(10.0).reshape(2, 5)
    mean, count = utils.bin_mean(x, (2, 2), count=True)
    torch.testing.assert_close(count, torch.tensor([[4.0, 4.0, 2.0]]))
    # integer weights are the same as repeated pairs
    g = torch.Generator().manual_seed(0)
    a = torch.randn(50, generator=g, dtype=torch.float64)
    b = 3 * a + torch.randn(50, generator=g, dtype=torch.float64)
    w = torch.randint(1, 4, (50,), generator=g).double()
    rep = w.long()
    one, ref = utils.OverlapStats.zeros((1,)), utils.OverlapStats.zeros((1,))
    one.add((0,), a, b, w)
    ref.add((0,), a.repeat_interleave(rep), b.repeat_interleave(rep))
    torch.testing.assert_close(one.n, ref.n)
    torch.testing.assert_close(one.mean, ref.mean)
    torch.testing.assert_close(one.scatter, ref.scatter)
    # a noiseless mosaic is still fitted exactly with partial bins
    tiles, pos, truth = simulate_tiles(field(), (3, 3), overlap=0.2, seed=3)
    for cap in (None, 40):
        est = IntensityAdjustment(
            weights=(1, 0, 0), downsample=7, max_pixels=cap, seed=1
        )
        e = est.fit(tiles, pos, fixed=[0]).estimates["intensity"]
        want = truth.scale.double() / truth.scale.double()[0]
        torch.testing.assert_close(e.scale.double(), want, rtol=1e-4, atol=0)


def test_three_dimensional_tiles():
    x = 50 + 10 * torch.rand(
        30, 40, 36, generator=torch.Generator().manual_seed(0)
    )
    tiles, pos, truth = simulate_tiles(
        x.double(), (2, 2, 2), overlap=0.3, seed=1
    )
    e = IntensityAdjustment(weights=(1, 0, 0)).fit(tiles, pos, fixed=[0])
    e = e.estimates["intensity"]
    want = truth.scale.double() / truth.scale.double()[0]
    torch.testing.assert_close(e.scale.double(), want, rtol=1e-4, atol=0)


def test_small_memory_budget_streams_blocks(mosaic):
    tiles, pos, _ = mosaic
    ref = IntensityAdjustment(dtype=torch.float64).fit(tiles, pos)
    with optix.config(memory=4096):
        est = IntensityAdjustment(dtype=torch.float64, downsample=3)
        est.fit(tiles, pos)
        ds = IntensityAdjustment(dtype=torch.float64, downsample=3)
    with optix.config(memory=2**26):
        ds.fit(tiles, pos)
    assert est.estimates["intensity"].allclose(
        ds.estimates["intensity"], rtol=1e-10, atol=1e-8
    )
    assert ref.estimates["intensity"].scale.shape == (9,)


def test_lazy_zarr_tiles(tmp_path, mosaic):
    zarr = pytest.importorskip("zarr")
    tiles, pos, _ = mosaic
    src, dst = [], []
    for i, t in enumerate(tiles):
        kw = {"shape": t.shape, "chunks": (16, 16), "dtype": "f8"}
        a = zarr.open_array(str(tmp_path / f"in{i}.zarr"), mode="w", **kw)
        a[:] = t.numpy()
        src.append(a)
        dst.append(
            zarr.open_array(str(tmp_path / f"o{i}.zarr"), mode="w", **kw)
        )
    est = IntensityAdjustment(dtype=torch.float64).fit(src, pos.numpy())
    ref = IntensityAdjustment(dtype=torch.float64).fit(tiles, pos)
    assert est.estimates["intensity"].allclose(
        ref.estimates["intensity"], rtol=1e-12, atol=1e-10
    )
    est.apply(src, out=dst)
    want = ref.apply(tiles)
    for d, w in zip(dst, want, strict=True):
        np.testing.assert_allclose(d[:], w.numpy(), rtol=1e-12)


def test_save_load_and_errors(tmp_path, mosaic):
    tiles, pos, _ = mosaic
    est = IntensityAdjustment().fit(tiles, pos)
    est.save(tmp_path / "est.pt")
    back = optix.load(tmp_path / "est.pt")
    assert back.estimates["intensity"].allclose(est.estimates["intensity"])
    torch.testing.assert_close(back.apply(tiles)[2], est.apply(tiles)[2])
    with pytest.raises(ValueError, match="weights"):
        IntensityAdjustment(weights=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError, match="downsample"):
        IntensityAdjustment(downsample=0)
    with pytest.raises(TypeError, match="sequence of tiles"):
        est.fit(torch.stack(tiles), pos)
    with pytest.raises(ValueError, match="positions"):
        est.fit(tiles, pos[:3])
    with pytest.raises(ValueError, match="spatial"):
        IntensityAdjustment().fit(tiles, torch.zeros(9, 3))
    with pytest.raises(ValueError, match="fixed"):
        est.fit(tiles, pos, fixed=[9])
    with pytest.raises(ValueError, match="tiles"):
        est.apply(tiles[:2])


def test_overlap_stats_merge_is_exact():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(1000, generator=g, dtype=torch.float64) + 1e4
    b = 2 * a + torch.randn(1000, generator=g, dtype=torch.float64)
    one = utils.OverlapStats.zeros((1,))
    one.add((0,), a, b)
    parts = utils.OverlapStats.zeros((1,))
    for k in range(0, 1000, 137):
        parts.add((0,), a[k : k + 137], b[k : k + 137])
    torch.testing.assert_close(parts.n, one.n)
    torch.testing.assert_close(parts.mean, one.mean)
    torch.testing.assert_close(parts.scatter, one.scatter)
    da, db = a - a.mean(), b - b.mean()
    torch.testing.assert_close(one.scatter[0, 2], (da * db).sum())
    assert utils.overlap_box([0, 0], [5, 5], [5, 0], [5, 5]) is None
    assert utils.components(4, [(0, 1), (2, 3)]) == [0, 0, 2, 2]


def test_simulate_tiles_covers_field():
    x = field((50, 70))
    tiles, pos, truth = simulate_tiles(x, (2, 3), overlap=0.1, seed=0)
    assert len(tiles) == 6 and pos.shape == (6, 2)
    last = pos[-1] + torch.tensor(tiles[-1].shape)
    assert last.tolist() == [50, 70]
    torch.testing.assert_close(
        truth.correct(tiles[4], 4),
        x[
            int(pos[4, 0]) : int(pos[4, 0]) + tiles[4].shape[0],
            int(pos[4, 1]) : int(pos[4, 1]) + tiles[4].shape[1],
        ],
    )
