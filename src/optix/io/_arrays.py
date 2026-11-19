"""Array-like protocols, tensor conversion and output storage.

optix never imports zarr, h5py or dask: array-likes are duck-typed. Only
basic indexing is used on them: full-length tuples of `int` and step-1
`slice`.
"""

import warnings
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from .._config import resolve_backend

__all__ = [
    "ArrayLike",
    "Index",
    "WritableArrayLike",
    "as_tensor",
    "chunks_of",
    "flush",
    "integer_range",
    "prepare_out",
    "store",
]

Index = tuple[int | slice, ...]
"""Basic index: a full-length tuple of ints and step-1 slices."""


@runtime_checkable
class ArrayLike(Protocol):
    """Anything with `.shape` and basic `__getitem__`.

    Tensor, ndarray, memmap, `zarr.Array`, `h5py.Dataset`, dask array.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, index: Index) -> object: ...


@runtime_checkable
class WritableArrayLike(ArrayLike, Protocol):
    """An [`ArrayLike`][optix.io.ArrayLike] that supports `__setitem__`."""

    def __setitem__(self, index: Index, value: object) -> None: ...


_COMPLEX = {
    torch.float16: torch.complex32,
    torch.bfloat16: torch.complex64,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def _target_dtype(source: torch.dtype, working: torch.dtype) -> torch.dtype:
    """Working dtype for data of dtype `source` (complex stays complex)."""
    if source.is_complex:
        return _COMPLEX[working]
    return working


def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    if dtype == torch.bfloat16:
        return np.dtype(np.float32)
    if dtype == torch.complex32:
        return np.dtype(np.complex64)
    return torch.empty(0, dtype=dtype).numpy().dtype


def _host(
    x: object, *, source: object, working: torch.dtype
) -> tuple[Tensor, bool, bool]:
    """Convert an (already sliced) array-like to a CPU tensor.

    Returns the tensor, whether it may share memory with `source` (the
    user's array), and whether it aliases a read-only buffer.
    """
    a = np.asarray(x)
    if a.dtype.kind not in "biufc":
        raise TypeError(f"Unsupported dtype for conversion: {a.dtype}")
    if not a.dtype.isnative:
        a = a.astype(a.dtype.newbyteorder("="))
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    if isinstance(source, np.ndarray):
        shared = bool(np.may_share_memory(a, source))
    else:
        # Slices of zarr / h5py / dask are normally fresh arrays; be
        # conservative when the result does not own its memory.
        shared = not a.flags.owndata and a.base is not None
    readonly = not a.flags.writeable
    with warnings.catch_warnings():
        # Read-only buffers: the aliasing tensor is temporary and is copied
        # before it is handed out (see _finish).
        warnings.filterwarnings("ignore", message=".*not writable.*")
        try:
            t = torch.from_numpy(a)
        except TypeError:
            # dtype unsupported by this torch: cast on the host first
            target = _COMPLEX[working] if a.dtype.kind == "c" else working
            t = torch.from_numpy(a.astype(_numpy_dtype(target)))
            shared = readonly = False
    return t, shared, readonly


def _finish(
    t: Tensor, *, alias: bool, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Move then cast a host tensor; copy if the result would alias."""
    out = t.to(device=device)
    out = out.to(dtype=_target_dtype(t.dtype, dtype))
    if alias and _same_storage(out, t):
        out = out.clone()
    return out


def _same_storage(a: Tensor, b: Tensor) -> bool:
    if a.device != b.device:
        return False
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def as_tensor(
    x: object,
    *,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
    copy: bool = False,
) -> Tensor:
    """Materialise an (already sliced) array-like as a working tensor.

    Parameters
    ----------
    x : array-like
        Tensor, ndarray, memmap, or anything `numpy.asarray` accepts
        (calling it is what triggers zarr / h5py / dask I/O).
    dtype, device : optional
        Working backend (see `resolve_backend`; the device follows a
        Tensor input when unset). Boolean, integer and floating data are
        cast to the working floating dtype, complex data to the complex
        dtype of matching precision.
    copy : bool
        If True, the result never shares memory with `x`.

    Returns
    -------
    Tensor

    Raises
    ------
    TypeError
        For structured, object, string or datetime dtypes.

    Notes
    -----
    Non-native byte order and non-contiguous / negatively strided arrays
    are normalised first. Data are moved to the device *before* the cast,
    so e.g. uint16 data cross the host-device link at half the size of
    float32. A tensor aliasing a read-only buffer (read-only memmap) is
    never returned: such data are copied.
    """
    backend = resolve_backend(dtype, device, like=x)
    if isinstance(x, Tensor):
        x = x.detach()
        return _finish(x, alias=copy, **backend)
    t, shared, readonly = _host(x, source=x, working=backend["dtype"])
    return _finish(t, alias=readonly or (copy and shared), **backend)


def _integer_info(dtype: object) -> tuple[int, int] | None:
    """(min, max) if `dtype` is an integer dtype, else None."""
    if isinstance(dtype, torch.dtype):
        if dtype.is_floating_point or dtype.is_complex:
            return None
        if dtype == torch.bool:
            return None
        info = torch.iinfo(dtype)
        return info.min, info.max
    dtype = np.dtype(dtype)
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        return int(info.min), int(info.max)
    return None


def _is_bool(dtype: object) -> bool:
    if isinstance(dtype, torch.dtype):
        return dtype == torch.bool
    return np.dtype(dtype).kind == "b"


def _round_clamp(value: Tensor, lo: int, hi: int) -> Tensor:
    """Round and clamp floats into [lo, hi] without wrap-around."""
    # float16 / bfloat16 cannot represent the bounds of wide integer types
    # (65504 is the float16 maximum): work in float32 at least.
    if value.dtype not in (torch.float32, torch.float64):
        value = value.to(torch.float32)
    npdt = _numpy_dtype(value.dtype)
    flo, fhi = np.array(lo, dtype=npdt), np.array(hi, dtype=npdt)
    # the float nearest to an integer bound may lie outside it
    if int(fhi) > hi:
        fhi = np.nextafter(fhi, np.array(0, dtype=npdt))
    if int(flo) < lo:
        flo = np.nextafter(flo, np.array(0, dtype=npdt))
    flo, fhi = float(flo), float(fhi)
    value = torch.nan_to_num(value, nan=0.0, posinf=fhi, neginf=flo)
    return value.round().clamp(flo, fhi)


def _clamp_integer(value: Tensor, lo: int, hi: int) -> Tensor:
    """Clamp integers into [lo, hi] so that a narrowing cast cannot wrap."""
    info = torch.iinfo(value.dtype)
    lo, hi = max(lo, info.min), min(hi, info.max)
    if lo == info.min and hi == info.max:
        return value
    return value.clamp(lo, hi)


def _cast_for(value: Tensor, dtype: object) -> Tensor:
    """Prepare `value` (on its device) for writing into `dtype`."""
    if dtype is None:
        return value
    if value.dtype.is_floating_point:
        if _is_bool(dtype):
            return value != 0
        bounds = _integer_info(dtype)
        if bounds is not None:
            return _round_clamp(value, *bounds)
    elif value.dtype != torch.bool and not value.dtype.is_complex:
        bounds = _integer_info(dtype)
        if bounds is not None:
            return _clamp_integer(value, *bounds)
    return value


def store(out: WritableArrayLike, index: Index, value: Tensor) -> None:
    """Write a tensor into a region of a tensor / numpy / zarr output.

    Parameters
    ----------
    out : writable array-like
        Destination.
    index : Index
        Region of `out` to write.
    value : Tensor
        Data, broadcastable to `out[index]`.

    Notes
    -----
    Floating values written to an integer output are rounded and clamped
    to the output's range on the value's device, before the transfer (a
    raw cast wraps around); NaN becomes 0. Floating values of less than
    single precision (float16, bfloat16) are promoted to float32 first.
    Integer values written to a narrower integer output are clamped to
    its range in the same way, instead of wrapping around. Floating values
    written to a boolean output become `value != 0`. An output without a
    `dtype` receives the values as they are.
    """
    value = value.detach()
    if isinstance(out, Tensor):
        value = _cast_for(value, out.dtype)
        out[index] = value.to(device=out.device, dtype=out.dtype)
        return
    dtype = getattr(out, "dtype", None)
    value = _cast_for(value, dtype)
    array = value.cpu().numpy()
    if dtype is not None:
        array = array.astype(np.dtype(dtype), copy=False)
    out[index] = array


def prepare_out(
    out: WritableArrayLike | None,
    shape: tuple[int, ...],
    *,
    like: object = None,
    dtype: torch.dtype | str | None = None,
) -> WritableArrayLike:
    """Validate a user `out=` or allocate an in-memory output.

    Parameters
    ----------
    out : writable array-like, optional
        User-provided output; must have exactly `shape`.
    shape : tuple of int
        Required shape.
    like : object, optional
        Input data. A new output is allocated on the device of `like` if
        it is a Tensor, else on the **CPU** (a result the size of a memmap
        input must not land on the GPU).
    dtype : optional
        Dtype of a new output (default: working dtype).

    Returns
    -------
    writable array-like

    Raises
    ------
    ValueError
        If `out` does not have the required shape.
    """
    shape = tuple(int(s) for s in shape)
    if out is not None:
        if tuple(out.shape) != shape:
            raise ValueError(
                f"Output has shape {tuple(out.shape)}, expected {shape}"
            )
        return out
    device = like.device if isinstance(like, Tensor) else torch.device("cpu")
    backend = resolve_backend(dtype, device)
    return torch.empty(shape, **backend)


def chunks_of(x: object) -> tuple[int, ...] | None:
    """Storage chunk shape of an array-like.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    tuple of int or None
        zarr `.shards` (if sharded) or `.chunks`, h5py `.chunks`,
        dask `.chunksize`; None for in-memory or contiguous arrays.
    """
    for attr in ("shards", "chunksize", "chunks"):
        value = getattr(x, attr, None)
        if value is None or callable(value):
            continue
        try:
            value = tuple(value)
        except TypeError:
            continue
        if value and all(
            isinstance(c, int | np.integer) and not isinstance(c, bool)
            for c in value
        ):
            return tuple(int(c) for c in value)
    return None


def flush(out: object) -> None:
    """Call `out.flush()` if it exists (memmaps).

    Parameters
    ----------
    out : object
    """
    fn = getattr(out, "flush", None)
    if callable(fn):
        fn()


def integer_range(dtype: object) -> tuple[float, float]:
    """`(min, max)` of an integer dtype (torch or numpy-like)."""
    if isinstance(dtype, torch.dtype):
        info = torch.iinfo(dtype)
    else:
        info = np.iinfo(np.dtype(dtype))
    return float(info.min), float(info.max)
