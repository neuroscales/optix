import warnings

import numpy as np
import pytest
import torch

from optix.io import (
    Chunk,
    Labeled,
    Layout,
    blocks,
    chunks_of,
    frame_batches,
    labeled,
    row_tiles,
)

# --- Layout -----------------------------------------------------------------


def test_layout_roles_and_shapes():
    lay = Layout("tczyx", spatial="yx", group="c")
    assert lay.axes == tuple("tczyx")
    assert lay.positions(5) == ((1,), (0, 2), (3, 4))
    assert lay.shape_of((2, 3, 4, 5, 6)) == ((3,), (2, 4), (5, 6))
    assert lay.map_shape((2, 3, 4, 5, 6)) == (3, 5, 6)
    perm = lay.permutation(5)
    assert perm == (1, 0, 2, 3, 4)
    x = np.zeros((2, 3, 4, 5, 6))
    canon = x.transpose(perm)
    assert canon.transpose(lay.inverse_permutation(5)).shape == x.shape
    # spatial order is the user's, not the storage order
    sheet = Layout("zyx", spatial="yz", sample="x")
    assert sheet.positions(3) == ((), (2,), (1, 0))
    assert sheet.shape_of(np.zeros((4, 5, 6))) == ((), (6,), (5, 4))
    # positions, negative allowed
    assert Layout(spatial=(-1, 1), group=0).positions(4) == (
        (0,),
        (2,),
        (3, 1),
    )
    # multi-character names
    long = Layout(("time", "y", "x"), spatial=("y", "x"), sample="time")
    assert long.positions(3) == ((), (0,), (1, 2))


def test_layout_default():
    assert Layout.default(5, 2) == Layout(spatial=(-2, -1))
    assert Layout.default(5, 2).positions(5) == ((), (0, 1, 2), (3, 4))
    assert Layout.default(1, 0).positions(1) == ((), (0,), ())
    with pytest.raises(ValueError):
        Layout.default(2, 3)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(axes="zyx"), "at least one spatial"),
        (dict(spatial=()), "at least one spatial"),
        (dict(axes="zyx", spatial="yq"), "Unknown axis 'q'"),
        (dict(axes="zyx", spatial="yx", sample="x"), "given twice"),
        (dict(axes="zyx", spatial="yy"), "given twice"),
        (dict(axes="zyx", spatial="yx", group="y"), "given twice"),
        (dict(spatial="yx"), "no `axes`"),
        (dict(spatial=(-1, -1)), "Repeated"),
        (dict(axes="zyy", spatial="y"), "Repeated axis labels"),
        (dict(axes="zyx", spatial=(5,)), "out of range"),
        (dict(axes=("t", "yx"), spatial="y"), "Unknown axis 'y'"),
    ],
)
def test_layout_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Layout(**kwargs)


def test_layout_validation_at_bind():
    lay = Layout(spatial=(-1, 2))
    with pytest.raises(ValueError, match="given twice"):
        lay.positions(3)
    with pytest.raises(ValueError, match="out of range"):
        lay.positions(2)
    with pytest.raises(ValueError, match="3 axes"):
        Layout("zyx", spatial="yx").positions(4)
    with pytest.raises(TypeError):
        Layout(spatial=1.5)


# --- labeled ----------------------------------------------------------------


def test_labeled_passthrough():
    y = np.arange(24.0).reshape(2, 3, 4)
    Y = labeled(y, axes="zyx", spatial="yz")
    assert isinstance(Y, Labeled)
    assert Y.shape == (2, 3, 4) and Y.ndim == 3 and Y.dtype == y.dtype
    assert Y.layout == Layout("zyx", spatial="yz")
    assert (Y[0, 1:, :] == y[0, 1:, :]).all()
    assert labeled(Y, spatial=(0,)).array is y
    with pytest.raises(TypeError):
        labeled(y, Layout(spatial=(-1,)), spatial=(0,))
    with pytest.raises(TypeError):
        labeled(y)
    with pytest.raises(ValueError):
        labeled(y, axes="zyxw", spatial="yx")
    # conflicting layouts
    with pytest.raises(ValueError, match="conflicts"):
        frame_batches(Y, layout=Layout(spatial=(-1,)))
    with pytest.raises(ValueError, match="conflicts"):
        frame_batches(Y, ndim=1)


def test_labeled_zarr_chunks_passthrough():
    zarr = pytest.importorskip("zarr")
    z = zarr.create_array(
        store={}, shape=(4, 5, 6), chunks=(2, 5, 3), dtype="f4"
    )
    assert chunks_of(labeled(z, axes="zyx", spatial="yz")) == (2, 5, 3)


# --- frame batches ----------------------------------------------------------


def _sheet(zarr, shape, chunks, dtype="u2"):
    z = zarr.create_array(store={}, shape=shape, chunks=chunks, dtype=dtype)
    y = np.random.default_rng(0).integers(0, 4000, shape).astype(dtype)
    z[:] = y
    return z, y


@pytest.mark.parametrize("batch", [1, 3, 4, 8, None])
def test_light_sheet_frames_round_trip(batch):
    """(z, y, x) data, (y, z) pictures, one per x."""
    zarr = pytest.importorskip("zarr")
    z, y = _sheet(zarr, (6, 5, 20), (6, 5, 4))
    Y = labeled(z, axes="zyx", spatial="yz", sample="x")
    out = zarr.create_array(
        store={}, shape=z.shape, chunks=(3, 5, 4), dtype="u2"
    )
    chunks = frame_batches(Y, batch=batch)
    ref = torch.from_numpy(y.astype("f8")).permute(2, 1, 0)  # (x, y, z)
    seen = []
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.data.shape[1:] == (5, 6)
        assert chunk.data.is_contiguous()
        assert chunk.group == ()
        (xs,) = chunk.samples
        torch.testing.assert_close(
            chunk.data.double(), ref[xs], check_dtype=False
        )
        seen.append(chunk.n)
        chunk.store(out, chunk.data)
    assert sum(seen) == 20
    if batch in (1, 3):
        assert seen[0] == batch
    elif batch is not None:
        assert seen[0] % 4 == 0  # aligned to the storage chunks along x
    np.testing.assert_array_equal(out[:], y)


def test_light_sheet_amplification_warning():
    zarr = pytest.importorskip("zarr")
    # every storage chunk spans all of x: reading few x at a time decodes
    # each chunk many times
    z, _ = _sheet(zarr, (6, 5, 20), (1, 5, 20))
    Y = labeled(z, axes="zyx", spatial="yz", sample="x")
    with pytest.warns(RuntimeWarning, match="Frame batches"):
        frame_batches(Y, batch=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame_batches(Y)  # all of x fits the default budget
        # chunked along x: batches aligned to the chunks read each once
        z2, _ = _sheet(zarr, (6, 5, 20), (6, 5, 4))
        frame_batches(labeled(z2, Y.layout), batch=5)


def test_ome_groups():
    y = np.random.default_rng(1).normal(size=(2, 3, 4, 5, 6))
    Y = labeled(y, axes="tczyx", spatial="yx", group="c")
    chunks = frame_batches(Y, batch=3)
    assert chunks.group_shape == (3,) and chunks.sample_shape == (2, 4)
    assert chunks.spatial_shape == (5, 6)
    groups = [chunk.group for chunk in chunks]
    # groups are visited one after the other
    assert groups == sorted(groups) and set(groups) == {(0,), (1,), (2,)}
    counts = {}
    out = np.zeros_like(y)
    for chunk in chunks:
        (c,) = chunk.group
        ref = y[:, c].reshape(8, 5, 6)
        flat = np.ravel_multi_index(
            np.meshgrid(
                *(np.arange(s.start, s.stop) for s in chunk.samples),
                indexing="ij",
            ),
            (2, 4),
        ).ravel()
        np.testing.assert_allclose(chunk.data.numpy(), ref[flat], rtol=1e-6)
        counts[c] = counts.get(c, 0) + chunk.n
        chunk.store(out, chunk.data)
    assert counts == {0: 8, 1: 8, 2: 8}
    np.testing.assert_allclose(out, y, rtol=1e-6)
    # parameter maps (C, Y, X) from row tiles holding all samples
    maps = torch.zeros(3, 5, 6, dtype=torch.float64)
    for chunk in row_tiles(Y, batch=2, dtype=torch.float64):
        assert chunk.data.shape[:2] == (2, 4)
        chunk.store(maps, chunk.data.mean((0, 1)), into="map")
    np.testing.assert_allclose(maps.numpy(), y.mean((0, 2)))


@pytest.mark.parametrize("batch", [1, 2, 5, 7, 24, None])
def test_non_adjacent_samples_flattened(batch):
    # axes s, y, t, x, u: samples s, t, u interleaved with spatial axes,
    # spatial requested as (x, y)
    y = np.random.default_rng(2).normal(size=(2, 5, 3, 6, 4))
    lay = Layout("sytxu", spatial="xy")
    assert lay.positions(5) == ((), (0, 2, 4), (3, 1))
    ref = y.transpose(0, 2, 4, 3, 1).reshape(24, 6, 5)
    parts = []
    out = np.zeros_like(y)
    for chunk in frame_batches(y, layout=lay, batch=batch, dtype="float64"):
        assert chunk.data.shape[1:] == (6, 5)
        parts.append(chunk.data.numpy())
        chunk.store(out, chunk.data)
    np.testing.assert_array_equal(np.concatenate(parts), ref)
    np.testing.assert_array_equal(out, y)
    if batch is not None:
        assert max(len(p) for p in parts) <= batch


def test_store_per_sample_state_and_errors():
    y = np.random.default_rng(3).normal(size=(2, 3, 4, 5, 6))
    Y = labeled(y, axes="tczyx", spatial="yx", group="c")
    state = torch.zeros(3, 2, 4, 2, dtype=torch.float64)  # (C, T, Z, 2)
    for chunk in frame_batches(Y, batch=3, dtype=torch.float64):
        d = chunk.data
        chunk.store(
            state,
            torch.stack([d.mean((1, 2)), d.amax((1, 2))], -1),
            into="samples",
        )
        assert (
            state[chunk.group + chunk.samples].shape[:2] == chunk.sample_shape
        )
    ref = y.transpose(1, 0, 2, 3, 4).reshape(3, 2, 4, 30)
    np.testing.assert_allclose(state[..., 0].numpy(), ref.mean(-1))
    np.testing.assert_allclose(state[..., 1].numpy(), ref.max(-1))
    chunk = next(iter(frame_batches(Y)))
    with pytest.raises(ValueError, match="expected"):
        chunk.store(np.zeros(y.shape), chunk.data[:, :2])
    with pytest.raises(ValueError, match="into"):
        chunk.store(np.zeros(y.shape), chunk.data, into="nope")


def test_default_layout_matches_explicit():
    y = np.random.default_rng(4).normal(size=(2, 7, 4, 5))
    a = [(c.index, c.data) for c in frame_batches(y, batch=3)]
    b = [
        (c.index, c.data)
        for c in frame_batches(y, layout=Layout.default(4, 2), batch=3)
    ]
    assert [i for i, _ in a] == [i for i, _ in b]
    # within one leading index, the old (int, slice, ...) indices
    assert a[0][0] == (0, slice(0, 3), slice(0, 4), slice(0, 5))
    for (_, d), (_, e) in zip(a, b, strict=True):
        torch.testing.assert_close(d, e)


def test_tensor_source_permuted_chunks_are_owned():
    y = torch.zeros(3, 4, 5)
    for chunk in frame_batches(y, layout=Layout("zyx", spatial="yz")):
        assert chunk.data.shape[1:] == (4, 3)
        chunk.data += 1
    assert y.eq(0).all()


# --- row tiles and blocks ---------------------------------------------------


def test_row_tiles_axis_by_name():
    y = np.random.default_rng(5).normal(size=(4, 5, 6))
    Y = labeled(y, axes="zyx", spatial="yz")
    out = np.zeros_like(y)
    for axis in ("z", 1, -1):
        for chunk in row_tiles(Y, axis=axis, batch=1):
            assert chunk.data.shape[0] == 6  # all x
            assert chunk.data.shape[1] == 5  # y whole, z cut
            chunk.store(out, chunk.data)
        np.testing.assert_allclose(out, y, rtol=1e-6)
    with pytest.raises(ValueError, match="not a spatial"):
        row_tiles(Y, axis="x")


@pytest.mark.parametrize("pad", [None, "zeros"])
def test_blocks_with_layout(pad):
    y = np.random.default_rng(6).normal(size=(7, 5, 3))
    Y = labeled(y, axes="zyx", spatial="yz")
    out = np.zeros_like(y)
    chunks = blocks(Y, (2, 3), 1, pad=pad)
    assert len(chunks) == 3 * 3 * 3
    for chunk in chunks:
        index, data, inner = chunk
        core = data[inner]
        assert core.shape == (
            index[1].stop - index[1].start,
            index[0].stop - index[0].start,
        )
        if pad is not None:
            assert data.shape == (4, 5)
        np.testing.assert_allclose(core.numpy(), y[index].T, rtol=1e-6)
        chunk.store(out, data)  # cropped automatically
    np.testing.assert_allclose(out, y, rtol=1e-6)
    # maps (*group, *spatial) from blocks
    maps = torch.zeros(5, 7)
    for chunk in blocks(Y, (2, 3), 1):
        if chunk.index[2] == 0:
            chunk.store(maps, chunk.data, into="map")
    np.testing.assert_allclose(maps.numpy(), y[:, :, 0].T, rtol=1e-6)
    with pytest.raises(ValueError, match="per spatial axis"):
        blocks(Y, (2, 3, 4))
