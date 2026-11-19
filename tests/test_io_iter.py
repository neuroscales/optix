import random
import threading
import time
import warnings

import numpy as np
import pytest
import torch

import optix
from optix.io import blocks, chunk_length, frame_batches, row_tiles, store
from optix.io._iter import _prefetch
from optix.io._plan import plan_blocks, read_amplification


def _shape_of(index, shape):
    out = []
    for i, n in zip(index, shape, strict=True):
        if isinstance(i, slice):
            out.append(len(range(*i.indices(n))))
    return tuple(out)


# --- sizing -----------------------------------------------------------------


def test_chunk_length():
    f32 = torch.float32
    assert chunk_length((10, 10), dtype=f32, memory=4000) == 10
    assert chunk_length((10, 10), dtype=f32, memory=4000, overhead=2) == 5
    assert chunk_length((10, 10), dtype=f32, memory=4000, maximum=3) == 3
    assert chunk_length((10, 10), dtype=f32, memory=4000, multiple=4) == 8
    assert chunk_length((10, 10), dtype=f32, memory=4000, multiple=16) == 10
    assert chunk_length((10, 10), dtype=f32, memory=1) == 1
    # batch wins over memory
    assert chunk_length((10, 10), dtype=f32, memory=4000, batch=3) == 3
    # rows: batch counts frames worth of elements
    assert chunk_length((5, 8), dtype=f32, batch=2, frame_shape=(4, 8)) == 1


# --- planner ----------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_plans_partition_exactly(seed):
    rng = random.Random(seed)
    ndim = rng.randint(1, 4)
    shape = [rng.randint(0, 7) for _ in range(ndim)]
    lead = rng.randint(0, ndim - 1)
    block = [rng.randint(1, 8) for _ in shape[lead:]]
    halo = [rng.randint(0, 3) for _ in shape[lead:]]
    pad = rng.random() < 0.5
    count = np.zeros(shape, dtype=int)
    for plan in plan_blocks(shape, lead, block, halo, pad):
        count[plan.index] += 1
        loaded = [
            s.stop - s.start + b + a
            for s, (b, a) in zip(plan.read[lead:], plan.pad, strict=True)
        ]
        if pad:
            expected = zip(block, halo, strict=True)
            assert loaded == [b + 2 * h for b, h in expected]
        cropped = _shape_of(plan.inner, loaded)
        assert cropped == _shape_of(plan.index[lead:], shape[lead:])
    assert (count == 1).all()


def test_read_amplification():
    regions = [slice(i, i + 1) for i in range(10)]
    assert read_amplification(regions, 5, 10) == 5
    assert read_amplification([slice(0, 10)], 5, 10) == 1
    assert read_amplification([slice(0, 5), slice(5, 10)], 5, 10) == 1


# --- frame batches ----------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 3, None])
def test_frame_batches_reassemble(batch):
    y = np.random.default_rng(0).normal(size=(2, 3, 11, 4, 5)).astype("f4")
    out = torch.zeros(y.shape)
    chunks = frame_batches(y, batch=batch)
    for chunk in chunks:
        assert chunk.data.shape[1:] == (4, 5)
        chunk.store(out, chunk.data)
    np.testing.assert_array_equal(out.numpy(), y)
    if batch is None:
        # leading dims are samples: merged into one chunk when they fit
        assert len(chunks) == 1
    else:
        assert len(chunks) == 6 * -(-11 // batch)


def test_frame_batches_select():
    data = torch.arange(10 * 3 * 4.0).reshape(10, 3, 4)
    chunks = frame_batches(data, batch=2)
    picked = chunks.select([3, 1])
    assert len(picked) == 2 and len(chunks) == 5
    got = [c.data for c in picked]
    assert torch.equal(got[0], data[6:8]) and torch.equal(got[1], data[2:4])
    assert len(list(chunks)) == 5


def test_frame_batches_memory_budget():
    y = np.zeros((20, 4, 4), dtype=np.uint16)
    chunks = frame_batches(y, memory=4 * 16 * 3, dtype=torch.float32)
    assert [d.shape[0] for _, d in chunks] == [3] * 6 + [2]
    chunks = frame_batches(y, memory=4 * 16 * 3, batch=5)
    assert [d.shape[0] for _, d in chunks] == [5] * 4


def test_frame_batches_config_captured_at_construction():
    y = np.zeros((10, 2, 2))
    with optix.config(batch=2):
        chunks = frame_batches(y)
    with optix.config(batch=5):
        assert len(chunks) == 5
        assert [i for i, _ in chunks] == [i for i, _ in chunks]


def test_frame_batches_edge_cases():
    assert len(frame_batches(np.zeros((0, 3, 3)))) == 0
    assert list(frame_batches(np.zeros((2, 0, 3, 3)))) == []
    with pytest.raises(ValueError, match="frame axis"):
        frame_batches(np.zeros((3, 3)))
    # no batch dims
    idx, data = next(iter(frame_batches(np.ones((4, 2, 2)))))
    assert idx == (slice(0, 4), slice(0, 2), slice(0, 2))
    # ndim=0: a stack of scalars
    assert next(iter(frame_batches(np.ones(5), ndim=0)))[1].shape == (5,)


def test_frame_batches_owned_chunks(device):
    y = torch.zeros(4, 2, 2, device=device)
    for _, data in frame_batches(y):
        data += 1
    assert y.eq(0).all()
    a = np.zeros((4, 2, 2), dtype=np.float32)
    for _, data in frame_batches(a, device="cpu"):
        data += 1
    assert (a == 0).all()


def test_frame_batches_follow_tensor_device():
    y = torch.zeros(3, 2, 2, device="meta")
    _, data = next(iter(frame_batches(y)))
    assert data.device.type == "meta"


def test_frame_batches_readonly_memmap(tmp_path):
    path = tmp_path / "y.dat"
    np.arange(60, dtype="<u2").tofile(path)
    mm = np.memmap(path, dtype="<u2", mode="r", shape=(5, 3, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parts = [d.clone() for _, d in frame_batches(mm, batch=2)]
        for _, d in frame_batches(mm, batch=2, dtype=torch.float32):
            d += 1
    np.testing.assert_array_equal(torch.cat(parts).numpy(), mm)


def test_frame_batches_zarr_alignment_and_warning():
    zarr = pytest.importorskip("zarr")
    y = np.random.default_rng(1).integers(0, 1000, (2, 12, 4, 4))
    z = zarr.create_array(
        store={}, shape=y.shape, chunks=(1, 4, 4, 4), dtype="u2"
    )
    z[:] = y
    chunks = frame_batches(z, batch=6)
    assert [d.shape[0] for _, d in chunks] == [4, 4, 4] * 2
    out = np.zeros(y.shape)
    for index, data in chunks:
        store(out, index, data)
    np.testing.assert_array_equal(out, y)
    with pytest.warns(RuntimeWarning, match="Frame batches"):
        frame_batches(z, batch=1, align=(1, 12, 4, 4))


# --- row tiles --------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, None])
def test_row_tiles_reassemble(batch):
    y = np.random.default_rng(2).normal(size=(2, 5, 7, 3)).astype("f4")
    out = torch.zeros(y.shape)
    chunks = row_tiles(y, batch=batch)
    for index, data in chunks:
        assert data.shape[:2] == (2, 5) and data.shape[-1] == 3
        store(out, index, data)
    np.testing.assert_array_equal(out.numpy(), y)


def test_row_tiles_budget():
    y = np.zeros((4, 10, 8), dtype=np.float32)
    # each row holds 4 * 8 float32 elements = 128 bytes
    with optix.config(dtype="float32"):
        tiles = row_tiles(y, memory=128 * 3)
    assert [d.shape[1] for _, d in tiles] == [3, 3, 3, 1]
    # batch = 1 frame of 10 * 8 elements -> 2 rows of 32 elements
    tiles = row_tiles(y, batch=1)
    assert [d.shape[1] for _, d in tiles] == [2] * 5


def test_row_tiles_huge_row_warns():
    with pytest.warns(RuntimeWarning, match="single row"):
        tiles = row_tiles(np.zeros((100, 3, 10)), memory=16)
    assert len(tiles) == 3


def test_row_tiles_other_axis():
    y = np.random.default_rng(3).normal(size=(3, 4, 6, 5))
    out = np.zeros_like(y)
    for index, data in row_tiles(y, ndim=3, axis=-3, batch=1):
        out[index] = data.numpy()
    np.testing.assert_allclose(out, y, rtol=1e-6)
    # axis is a position among the spatial axes
    with pytest.raises(ValueError, match="not a spatial axis"):
        row_tiles(y, ndim=2, axis=2)


def test_row_tiles_frame_chunked_zarr_warns():
    zarr = pytest.importorskip("zarr")
    z = zarr.create_array(
        store={}, shape=(16, 32, 8), chunks=(1, 32, 8), dtype="u2"
    )
    with pytest.warns(RuntimeWarning, match="Row tiles"):
        row_tiles(z, batch=1)


# --- blocks -----------------------------------------------------------------


def _box(x, radius):
    """Box filter with replicate boundary over every dim of x."""
    for d in range(x.ndim):
        n = x.shape[d]
        idx = torch.arange(-radius, n + radius).clamp(0, n - 1)
        xp = x.index_select(d, idx)
        x = sum(xp.narrow(d, k, n) for k in range(2 * radius + 1)) / (
            2 * radius + 1
        )
    return x


@pytest.mark.parametrize("block", [(4, 4, 4), (3, 5, 7), (9, 9, 9), None])
@pytest.mark.parametrize("pad", [None, "replicate"])
def test_blocked_equals_unblocked(block, pad):
    y = torch.randn(9, 10, 11, dtype=torch.float64)
    whole = _box(y, 2)
    out = torch.zeros_like(y)
    memory = 8 * 6**3 if block is None else None
    for index, data, inner in blocks(
        y, block, 2, pad=pad, memory=memory, dtype=torch.float64
    ):
        if pad is not None and block is not None:
            assert data.shape == tuple(b + 4 for b in block)
        store(out, index, _box(data, 2)[inner])
    torch.testing.assert_close(out, whole)


def test_blocks_pad_shapes_constant():
    y = np.arange(7 * 5).reshape(7, 5).astype("f4")
    shapes = {d.shape for _, d, _ in blocks(y, (3, 2), 1, pad="zeros")}
    assert shapes == {(5, 4)}
    _, first, inner = next(iter(blocks(y, (3, 2), (1, 1), pad="zeros")))
    assert first[0].eq(0).all() and first[:, 0].eq(0).all()
    assert first[inner].tolist() == y[:3, :2].tolist()


def test_blocks_reflect_values_and_fallback():
    y = torch.arange(6.0)
    ((_, data, inner),) = list(blocks(y, (6,), 2, pad="reflect"))
    assert data.tolist() == [2, 1, 0, 1, 2, 3, 4, 5, 4, 3]
    assert data[inner].tolist() == y.tolist()
    with pytest.warns(RuntimeWarning, match="replicate"):
        chunks = blocks(y, (4,), 3, pad="reflect")
    for _, data, _ in chunks:
        assert data.shape == (10,)


def test_blocks_leading_dims_and_extremes():
    y = np.random.default_rng(4).normal(size=(2, 3, 5, 4))
    chunks = blocks(y, (None, 2), 1)
    for index, data, _ in chunks:
        assert isinstance(index[0], int) and isinstance(index[1], int)
        assert data.shape[0] == 5
    assert len(chunks) == 6 * 2
    # halo and block larger than the array
    ((index, data, inner),) = list(blocks(y[0, 0], (50, 50), 10))
    assert data.shape == (5, 4) and inner == (slice(0, 5), slice(0, 4))


def test_blocks_auto_fits_budget():
    y = np.zeros((16, 16, 16), dtype=np.float32)
    chunks = blocks(y, halo=1, memory=4 * 10**3)
    for _, data, _ in chunks:
        assert data.numel() <= 10**3


# --- prefetch ---------------------------------------------------------------


def test_prefetch_same_items():
    assert list(_prefetch(range(50), 3)) == list(range(50))


def test_prefetch_iterator_matches_plain():
    y = np.random.default_rng(5).normal(size=(3, 7, 4, 4))
    plain = [(i, d) for i, d in frame_batches(y, batch=2)]
    fetched = [(i, d) for i, d in frame_batches(y, batch=2, prefetch=2)]
    assert [i for i, _ in plain] == [i for i, _ in fetched]
    for (_, a), (_, b) in zip(plain, fetched, strict=True):
        torch.testing.assert_close(a, b)


def test_prefetch_exception_propagates():
    def source():
        yield 1
        raise KeyError("boom")

    it = _prefetch(source(), 2)
    assert next(it) == 1
    with pytest.raises(KeyError, match="boom"):
        next(it)


def test_prefetch_early_break_stops_thread():
    before = threading.active_count()

    def slow():
        for i in range(1000):
            time.sleep(0.001)
            yield i

    for i in _prefetch(slow(), 2):
        if i == 3:
            break
    assert threading.active_count() == before
    y = np.zeros((40, 2, 2))
    for _ in frame_batches(y, batch=1, prefetch=1):
        break
    assert threading.active_count() == before
