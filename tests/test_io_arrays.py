import warnings

import numpy as np
import pytest
import torch

from optix.io import as_tensor, chunks_of, flush, prepare_out, store


def test_tensor_input(device):
    x = torch.randn(3, 4, device=device, requires_grad=True)
    t = as_tensor(x)
    assert not t.requires_grad
    assert t.device == x.device
    assert t.dtype == torch.get_default_dtype()
    # no-op conversion aliases unless copy=True
    assert t.data_ptr() == x.data_ptr()
    c = as_tensor(x, copy=True)
    assert c.data_ptr() != x.data_ptr()
    torch.testing.assert_close(c, x.detach())


def test_tensor_dtype_cast():
    x = torch.arange(6, dtype=torch.int32)
    t = as_tensor(x, dtype=torch.float64)
    assert t.dtype == torch.float64
    torch.testing.assert_close(t, x.double())


@pytest.mark.parametrize(
    "npdt", ["bool", "uint8", "uint16", "int32", "int64", "float16", "f8"]
)
def test_numpy_to_working_float(npdt):
    a = (np.arange(12) % 3).astype(npdt).reshape(3, 4)
    t = as_tensor(a)
    assert t.dtype == torch.get_default_dtype()
    np.testing.assert_array_equal(t.numpy(), a.astype(np.float32))


def test_complex_maps_to_matching_precision():
    a = (np.arange(4) + 1j).astype(np.complex128)
    assert as_tensor(a).dtype == torch.complex64
    assert as_tensor(a, dtype=torch.float64).dtype == torch.complex128


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(3, dtype=[("a", "f4"), ("b", "i4")]),
        np.array(["a", "b"]),
        np.array([object(), object()]),
    ],
)
def test_unsupported_dtypes(bad):
    with pytest.raises(TypeError):
        as_tensor(bad)


def test_big_endian_and_negative_strides():
    a = np.arange(20, dtype=">f4").reshape(4, 5)[::-1, ::2]
    t = as_tensor(a)
    np.testing.assert_array_equal(t.numpy(), a.astype("<f4"))
    b = np.arange(10, dtype=">u2")
    np.testing.assert_array_equal(as_tensor(b).numpy(), np.arange(10))


def test_copy_numpy():
    a = np.zeros(5, dtype=np.float32)
    t = as_tensor(a, copy=True)
    t += 1
    assert (a == 0).all()


def test_readonly_memmap_never_aliased(tmp_path):
    path = tmp_path / "x.dat"
    np.arange(10, dtype=np.float32).tofile(path)
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(10,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = as_tensor(mm)
    t += 1  # would segfault / be undefined if aliased
    np.testing.assert_array_equal(mm, np.arange(10))


def test_store_tensor_out():
    out = torch.zeros(3, 4, dtype=torch.float64)
    store(out, (slice(0, 2), slice(None)), torch.ones(2, 4))
    assert out[:2].eq(1).all() and out[2].eq(0).all()
    iout = torch.zeros(3, dtype=torch.int16)
    store(iout, (slice(None),), torch.tensor([-1e9, 2.6, float("nan")]))
    assert iout.tolist() == [-32768, 3, 0]


def test_store_integer_rounding_and_clamping():
    out = np.zeros(6, dtype=np.uint8)
    value = torch.tensor([-5.0, 3.6, 300.0, float("nan"), 254.5, float("inf")])
    store(out, (slice(None),), value)
    assert out.tolist() == [0, 4, 255, 0, 254, 255]


@pytest.mark.parametrize("npdt", ["int32", "int64", "uint64"])
def test_store_extreme_integer_no_wraparound(npdt):
    out = np.zeros(2, dtype=npdt)
    store(out, (slice(None),), torch.tensor([1e30, -1e30]))
    info = np.iinfo(npdt)
    assert out[0] > info.max // 2
    if info.min == 0:
        assert out[1] == 0
    else:
        assert out[1] < info.min // 2


def test_store_bool():
    out = np.zeros(3, dtype=bool)
    store(out, (slice(None),), torch.tensor([0.0, 0.2, -1.0]))
    assert out.tolist() == [False, True, True]


def test_store_without_dtype():
    class Sink:
        shape = (2,)

        def __getitem__(self, index):
            raise NotImplementedError

        def __setitem__(self, index, value):
            self.value = value

    sink = Sink()
    store(sink, (slice(None),), torch.tensor([1.5, 2.5]))
    assert sink.value.dtype == np.float32
    np.testing.assert_array_equal(sink.value, [1.5, 2.5])


def test_store_zarr():
    zarr = pytest.importorskip("zarr")
    z = zarr.create_array(store={}, shape=(4, 3), chunks=(2, 3), dtype="u2")
    store(z, (slice(0, 2), slice(None)), torch.full((2, 3), 7.4))
    assert z[:2].tolist() == [[7] * 3] * 2


def test_prepare_out():
    y = np.zeros((2, 3), dtype=np.uint16)
    out = prepare_out(None, (2, 3), like=y)
    assert isinstance(out, torch.Tensor) and out.device.type == "cpu"
    assert out.dtype == torch.get_default_dtype()
    t = torch.zeros(2, 3, device="meta")
    assert prepare_out(None, (2, 3), like=t).device.type == "meta"
    assert prepare_out(None, (2,), dtype="float64").dtype == torch.float64
    given = np.zeros((2, 3))
    assert prepare_out(given, (2, 3)) is given
    with pytest.raises(ValueError, match=r"\(2, 3\).*\(3, 2\)"):
        prepare_out(given, (3, 2))


def test_chunks_of():
    assert chunks_of(np.zeros(3)) is None
    assert chunks_of(torch.zeros(3)) is None

    class Dask:
        chunks = ((2, 2), (3,))
        chunksize = (2, 3)

    class H5:
        chunks = None

    assert chunks_of(Dask()) == (2, 3)
    assert chunks_of(H5()) is None


def test_chunks_of_zarr():
    zarr = pytest.importorskip("zarr")
    z = zarr.create_array(store={}, shape=(8, 4), chunks=(2, 4), dtype="f4")
    assert chunks_of(z) == (2, 4)
    s = zarr.create_array(
        store={}, shape=(8, 4), chunks=(1, 2), shards=(4, 4), dtype="f4"
    )
    assert chunks_of(s) == (4, 4)


def test_flush(tmp_path):
    mm = np.memmap(tmp_path / "f.dat", dtype=np.uint8, mode="w+", shape=(4,))
    mm[:] = 3
    flush(mm)
    assert np.fromfile(tmp_path / "f.dat", dtype=np.uint8).tolist() == [3] * 4
    flush(object())  # no-op
