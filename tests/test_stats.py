import math

import numpy as np
import pytest
import torch

import optix
from optix._config import resolve_backend
from optix.stats import (
    Histogram,
    MinMax,
    Moments,
    Reservoir,
    median_mad,
    order_statistics,
    reduce,
    tile_reduce,
)


def _splits(x, sizes):
    out, start = [], 0
    for s in sizes:
        out.append(x[start : start + s])
        start += s
    return out


SPLITS = [(1, 59), (17, 23, 20), (60,)]


def _skew_kurt(x):
    d = x - x.mean(0)
    m2, m3, m4 = (d.pow(p).mean(0) for p in (2, 3, 4))
    return m3 / m2.pow(1.5), m4 / m2.square() - 3


# --- Moments ----------------------------------------------------------------


@pytest.mark.parametrize("sizes", SPLITS)
def test_moments_reference_and_merge_law(sizes, device):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(60, 3, 4, generator=g, dtype=torch.float64).exp()
    x = x.to(device)
    ref_skew, ref_kurt = _skew_kurt(x)
    seq = Moments(4)
    merged = Moments(4)
    for part in _splits(x, sizes):
        seq.update(part)
        merged.merge(Moments(4).update(part))
    for m in (seq, merged):
        assert m.count.item() == 60
        torch.testing.assert_close(m.mean, x.mean(0))
        torch.testing.assert_close(m.var(), x.var(0))
        torch.testing.assert_close(m.var(ddof=0), x.var(0, correction=0))
        torch.testing.assert_close(m.std(), x.std(0))
        torch.testing.assert_close(m.skewness(), ref_skew)
        torch.testing.assert_close(m.kurtosis(), ref_kurt)
        torch.testing.assert_close(m.kurtosis(excess=False), ref_kurt + 3)


def test_moments_float32_chunks_float64_state():
    x = torch.randn(10, 2)
    m = Moments().update(x)
    assert m.mean.dtype == torch.float64
    m32 = Moments(state_dtype=torch.float32).update(x)
    assert m32.mean.dtype == torch.float32


def test_moments_merge_identity():
    x = torch.randn(5, 2)
    a = Moments(3).update(x)
    ref = a.copy()
    a.merge(Moments(3))
    torch.testing.assert_close(a.m3, ref.m3)
    b = Moments(3).merge(ref)
    torch.testing.assert_close(b.mean, ref.mean)
    b.update(x)  # b's state is not shared with ref
    assert ref.count.item() == 5
    with pytest.raises(ValueError):
        Moments(2).merge(Moments(3))
    with pytest.raises(TypeError):
        Moments().merge(MinMax())


def test_moments_weighted_equals_repeated():
    x = torch.randn(6, 3, dtype=torch.float64)
    w = torch.tensor([1, 3, 0, 2, 1, 1], dtype=torch.float64)[:, None]
    weighted = Moments(4).update(x[:2], weight=w[:2])
    weighted.update(x[2:], weight=w[2:])
    rep = x.repeat_interleave(w[:, 0].long(), dim=0)
    plain = Moments(4).update(rep)
    torch.testing.assert_close(weighted.count, plain.count.expand(3))
    for attr in ("mean", "m2", "m3", "m4"):
        torch.testing.assert_close(
            getattr(weighted, attr), getattr(plain, attr)
        )


def test_moments_masked_empty_pixels():
    x = torch.randn(4, 3)
    mask = torch.ones(4, 3, dtype=torch.bool)
    mask[:, 1] = False
    mask[1:, 2] = False
    m = Moments().update(x, weight=mask)
    assert m.count.tolist() == [4, 0, 1]
    assert m.mean[1] == 0 and m.m2[1] == 0
    var = m.var()
    assert torch.isnan(var[1]) and torch.isnan(var[2])
    assert not torch.isnan(var[0])


@pytest.mark.parametrize("dt", [torch.float32, torch.float64])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_moments_masked_nan_frame(dt, bad):
    # Regression: 0 * nan = nan leaked masked frames into the sums.
    g = torch.Generator().manual_seed(5)
    x = torch.randn(6, 3, generator=g, dtype=torch.float64).to(dt)
    x[2] = bad
    x[4, 1] = bad
    mask = torch.ones(6, 3, dtype=torch.bool)
    mask[2] = False
    mask[4, 1] = False
    m = (
        Moments(4)
        .update(x[:3], weight=mask[:3])
        .update(x[3:], weight=mask[3:])
    )
    for attr in ("count", "mean", "m2", "m3", "m4"):
        assert torch.isfinite(getattr(m, attr)).all()
    assert m.count.tolist() == [5, 4, 5]
    for j in range(3):
        ref = x[:, j][mask[:, j]].double()
        dev = ref - ref.mean()
        torch.testing.assert_close(m.mean[j], ref.mean())
        torch.testing.assert_close(m.m2[j], dev.square().sum())
        torch.testing.assert_close(m.m4[j], dev.pow(4).sum())


def test_moments_large_offset_stability():
    g = torch.Generator().manual_seed(3)
    m = Moments()
    for _ in range(10):
        m.update(1e4 + torch.randn(1000, 4, generator=g))
    torch.testing.assert_close(
        m.var(), torch.ones(4, dtype=torch.float64), rtol=0.1, atol=0
    )
    # relative accuracy against a float64 computation of the same data
    g = torch.Generator().manual_seed(4)
    data = 1e4 + torch.randn(10, 1000, 4, generator=g, dtype=torch.float64)
    m = Moments()
    for chunk in data.float():
        m.update(chunk)
    exact = data.float().double().reshape(-1, 4).var(0)
    torch.testing.assert_close(m.var(), exact, rtol=1e-3, atol=0)


def test_moments_dim_none_and_tuple():
    x = torch.randn(4, 5, 6, dtype=torch.float64)
    m = Moments(dim=None).update(x)
    assert m.mean.shape == ()
    torch.testing.assert_close(m.var(), x.var())
    m = Moments(dim=(0, 2)).update(x)
    torch.testing.assert_close(m.mean, x.mean((0, 2)))


def test_moments_state_dict_and_to():
    m = Moments(3).update(torch.randn(5, 2))
    new = Moments.from_state_dict(m.state_dict())
    torch.testing.assert_close(new.m3, m.m3)
    assert new.order == 3
    m.to(dtype=torch.float32)
    assert m.mean.dtype == torch.float32


# --- MinMax -----------------------------------------------------------------


@pytest.mark.parametrize("sizes", SPLITS)
def test_minmax_merge_law(sizes):
    x = torch.randn(60, 5)
    seq, merged = MinMax(), MinMax()
    for part in _splits(x, sizes):
        seq.update(part)
        merged.merge(MinMax().update(part))
    for r in (seq, merged):
        torch.testing.assert_close(r.min, x.amin(0))
        torch.testing.assert_close(r.max, x.amax(0))


def test_minmax_mask_and_nan():
    x = torch.tensor([[1.0, 5.0], [3.0, float("nan")]])
    mask = torch.tensor([[True, False], [True, False]])
    r = MinMax().update(x, weight=mask)
    assert r.min[0] == 1 and r.max[0] == 3
    assert r.min[1] == math.inf and r.max[1] == -math.inf
    r = MinMax().update(x)
    assert torch.isnan(r.min[1])


# --- Histogram --------------------------------------------------------------


@pytest.mark.parametrize("sizes", SPLITS)
def test_histogram_exact_merge_law(sizes):
    x = torch.randn(60, 4)
    whole = Histogram(-2.0, 2.0, 16).update(x)
    seq = Histogram(-2.0, 2.0, 16)
    merged = Histogram(-2.0, 2.0, 16)
    for part in _splits(x, sizes):
        seq.update(part)
        merged.merge(Histogram(-2.0, 2.0, 16).update(part))
    assert torch.equal(seq.counts, whole.counts)
    assert torch.equal(merged.counts, whole.counts)
    assert whole.counts.sum(0).tolist() == [60] * 4


def test_histogram_bins_edges_and_flow():
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0, float("nan")])[:, None]
    h = Histogram(0.0, 1.0, 2).update(x)
    assert h.counts[:, 0].tolist() == [1, 1, 2, 2]


def test_histogram_zero_width_and_maps():
    lo = torch.tensor([0.0, 3.0])
    hi = torch.tensor([1.0, 3.0])
    x = torch.tensor([[0.5, 3.0], [0.7, 3.0], [0.1, 4.0]])
    h = Histogram(lo, hi, 4).update(x)
    assert h.counts[:, 1].tolist() == [0, 2, 0, 0, 0, 1]
    assert h.counts[:, 0].sum() == 3


def test_histogram_quantile_integer_data():
    g = torch.Generator().manual_seed(5)
    x = torch.randint(0, 10, (501, 3), generator=g).float()
    h = Histogram(-0.5, 9.5, 10).update(x)
    for q in (0.1, 0.5, 0.9):
        exact = x.sort(0).values[math.ceil(q * 501) - 1]
        assert (h.quantile(q) - exact).abs().max() <= 1.0
    qs = h.quantile(torch.tensor([0.0, 1.0]))
    assert qs.shape == (2, 3)
    assert (qs[0] >= -0.5).all() and (qs[1] <= 9.5).all()
    cdf = h.cdf()
    assert cdf.shape == (12, 3)
    torch.testing.assert_close(cdf[-1], torch.ones(3, dtype=torch.float64))


def test_histogram_quantile_continuous_within_bin_width():
    g = torch.Generator().manual_seed(6)
    x = torch.randn(20000, 2, generator=g, dtype=torch.float64)
    h = Histogram(-5.0, 5.0, 100).update(x)
    exact = torch.quantile(x, 0.3, dim=0)
    assert (h.quantile(0.3) - exact).abs().max() <= 0.1


def test_histogram_weights_and_merge_checks():
    x = torch.tensor([[0.2], [0.8]])
    h = Histogram(0.0, 1.0, 2).update(x, weight=torch.tensor([[3], [1]]))
    assert h.counts[:, 0].tolist() == [0, 3, 1, 0]
    with pytest.raises(ValueError, match="edges"):
        h.merge(Histogram(0.0, 2.0, 2).update(x))
    with pytest.raises(ValueError, match="bins"):
        h.merge(Histogram(0.0, 1.0, 3).update(x))


def test_histogram_memory_warning():
    with optix.config(memory=1000):
        with pytest.warns(RuntimeWarning, match="Histogram state"):
            Histogram(torch.zeros(20, 20), torch.ones(20, 20), 64)
        h = Histogram(0.0, 1.0, 64)
    with pytest.warns(RuntimeWarning, match="Histogram state"):
        h.update(torch.rand(2, 20, 20))


def test_histogram_default_count_dtype_is_int64():
    h = Histogram(0.0, 1.0, 4).update(torch.rand(3, 2, 2))
    assert h.count_dtype == torch.int64 and h.counts.dtype == torch.int64


# --- Reservoir --------------------------------------------------------------


def test_reservoir_size_and_seen():
    r = Reservoir(10, seed=0)
    r.update(torch.arange(5.0))
    assert r.values.numel() == 5 and r.seen == 5
    for _ in range(4):
        r.update(torch.arange(100.0))
    assert r.values.numel() == 10 and r.seen == 405


def test_reservoir_mask():
    x = torch.arange(100.0)
    r = Reservoir(1000, seed=1).update(x, weight=x % 2 == 0)
    assert r.seen == 50
    assert sorted(r.values.tolist()) == list(range(0, 100, 2))


def test_reservoir_reproducible():
    x = torch.randn(1000)
    a = Reservoir(20, seed=3).update(x[:500]).update(x[500:])
    b = Reservoir(20, seed=3).update(x[:500]).update(x[500:])
    assert torch.equal(a.values.sort().values, b.values.sort().values)


def _chi_square_uniform(counts):
    expected = counts.mean()
    return float(((counts - expected) ** 2 / expected).sum())


@pytest.mark.parametrize("mode", ["update", "merge"])
def test_reservoir_uniform(mode):
    n, size, trials = 40, 8, 2000
    hits = torch.zeros(n)
    x = torch.arange(float(n))
    for t in range(trials):
        if mode == "update":
            r = Reservoir(size, seed=t)
            for part in (x[:3], x[3:20], x[20:]):
                r.update(part)
        else:
            r = Reservoir(size, seed=2 * t)
            r.update(x[:25])
            r.merge(Reservoir(size, seed=2 * t + 1).update(x[25:]))
        assert r.values.numel() == size
        hits[r.values.long()] += 1
    # 39 degrees of freedom: the 99.9% quantile is about 72
    assert _chi_square_uniform(hits) < 72
    assert hits.sum() == size * trials


@pytest.mark.parametrize("ratio", [1.5 / 16, 0.3 / 16, 13.7 / 16])
def test_reservoir_thinning_is_exact(monkeypatch, ratio):
    # Regression: the Bernoulli thinning compared float32 draws (multiples
    # of 2^-24) with kmax / 2^62, which over-includes values once that
    # ratio is not a multiple of 2^-24. Simulate a 4-bit uniform generator
    # so that the effect is visible with few draws.
    from optix.stats import _reservoir

    rand = torch.rand

    def coarse(*args, **kwargs):
        return (rand(*args, **kwargs) * 16).floor() / 16

    monkeypatch.setattr(_reservoir, "_RAND_BITS", 4, raising=False)
    monkeypatch.setattr(torch, "rand", coarse)
    size, n = 200_000, 200_000
    kmax = int(ratio * 2**62)
    r = Reservoir(size, seed=0)
    r.values = torch.full((size,), -1.0)
    r.keys = torch.full((size,), kmax, dtype=torch.int64)
    r.count = torch.tensor(size)
    r.update(torch.zeros(n))
    new = r.keys[r.values == 0]
    expected = n * kmax / 2**62
    assert abs(new.numel() - expected) < 5 * expected**0.5
    # the new keys are uniform in [0, kmax)
    assert int(new.max()) < kmax
    mean = new.double().mean().item() / kmax
    assert abs(mean - 0.5) < 5 * (1 / 12 / new.numel()) ** 0.5


def test_reservoir_state_and_copy():
    r = Reservoir(5, seed=0).update(torch.arange(50.0))
    c = r.copy()
    c.update(torch.arange(50.0))
    assert r.seen == 50 and c.seen == 100
    s = Reservoir.from_state_dict(r.state_dict())
    assert torch.equal(s.values, r.values) and s.seen == 50


# --- driver and order statistics --------------------------------------------


@pytest.mark.parametrize("batch", [1, 7, None])
def test_reduce_chunk_independent(batch):
    y = np.random.default_rng(0).normal(size=(2, 30, 4, 5))
    m, mm = reduce(y, Moments(), MinMax(), batch=batch, dtype=torch.float64)
    flat = torch.from_numpy(y).reshape(60, 4, 5)
    torch.testing.assert_close(m.mean, flat.mean(0))
    torch.testing.assert_close(m.var(), flat.var(0))
    torch.testing.assert_close(mm.max, flat.amax(0))


def test_reduce_zarr():
    zarr = pytest.importorskip("zarr")
    y = np.random.default_rng(1).integers(0, 100, (20, 6, 6))
    z = zarr.create_array(
        store={}, shape=y.shape, chunks=(5, 6, 6), dtype="u2"
    )
    z[:] = y
    (m,) = reduce(z, Moments(), batch=5)
    np.testing.assert_allclose(m.mean.numpy(), y.mean(0))


@pytest.mark.parametrize("batch", [1, 3, None])
def test_order_statistics_vs_sort(batch):
    y = np.random.default_rng(2).normal(size=(2, 9, 7, 5)).astype("f4")
    s = torch.from_numpy(y).reshape(18, 7, 5).sort(0).values
    s = s.to(resolve_backend()["dtype"])
    torch.testing.assert_close(order_statistics(y, 4, batch=batch), s[3])
    ks = [1, 5, 18]
    torch.testing.assert_close(
        order_statistics(y, ks, batch=batch), s[[k - 1 for k in ks]]
    )
    many = list(range(1, 19, 2))
    torch.testing.assert_close(
        order_statistics(y, many, batch=batch), s[[k - 1 for k in many]]
    )
    with pytest.raises(ValueError):
        order_statistics(y, 19)


def test_median_mad():
    y = torch.randn(3, 11, 4, 6, dtype=torch.float64)
    med, mad = median_mad(y, batch=1, dtype=torch.float64)
    flat = y.reshape(33, 4, 6)
    ref = flat.median(0).values
    torch.testing.assert_close(med, ref)
    torch.testing.assert_close(mad, (flat - ref).abs().median(0).values)


def test_tile_reduce_3d_and_amplification_warning():
    y = torch.randn(5, 3, 4, 6, dtype=resolve_backend()["dtype"])
    out = tile_reduce(y, lambda t: t.mean(-1), ndim=3, batch=1)
    torch.testing.assert_close(out, y.mean(0))
    zarr = pytest.importorskip("zarr")
    z = zarr.create_array(
        store={}, shape=(8, 16, 4), chunks=(1, 16, 4), dtype="f4"
    )
    z[:] = np.ones((8, 16, 4), dtype="f4")
    with pytest.warns(RuntimeWarning, match="Row tiles"):
        median_mad(z, batch=1)


# --- layouts ----------------------------------------------------------------


def test_reduce_with_layout_and_groups():
    from optix.io import Layout, labeled
    from optix.stats import reduce_groups

    dt = resolve_backend()["dtype"]
    y = torch.randn(2, 3, 4, 5, 6, dtype=dt)  # t, c, z, y, x
    data = labeled(y, axes="tczyx", spatial="xy", group="c")
    with pytest.raises(ValueError, match="reduce_groups"):
        reduce(data, Moments(1))
    template = Moments(2)
    groups = reduce_groups(data, template, batch=3)
    assert template.empty and list(groups) == [(0,), (1,), (2,)]
    for (c,), (m,) in groups.items():
        ref = y[:, c].reshape(8, 5, 6).transpose(-1, -2)
        torch.testing.assert_close(m.mean.to(dt), ref.mean(0))
    # no group axes: samples t and z pooled, spatial in layout order
    (m,) = reduce(y[:, 0], Moments(1), layout=Layout("tzyx", spatial="xy"))
    torch.testing.assert_close(m.mean.to(dt), y[:, 0].mean((0, 1)).T)


def test_order_statistics_per_group():
    from optix.io import labeled

    dt = resolve_backend()["dtype"]
    y = torch.randn(3, 2, 5, 4, 6, dtype=dt)  # t, c, z, y, x
    data = labeled(y, axes="tczyx", spatial="yx", group="c")
    med, mad = median_mad(data, batch=1)
    assert med.shape == (2, 4, 6)
    flat = y.transpose(0, 1).reshape(2, 15, 4, 6)
    ref = flat.median(1).values
    torch.testing.assert_close(med, ref)
    torch.testing.assert_close(
        mad, (flat - ref[:, None]).abs().median(1).values
    )
    ks = order_statistics(data, [1, 15], batch=2)
    assert ks.shape == (2, 2, 4, 6)
    torch.testing.assert_close(ks[0], flat.amin(1))
    torch.testing.assert_close(ks[1], flat.amax(1))
