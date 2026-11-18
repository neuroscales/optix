"""Chunked iteration over array-likes, with optional prefetching."""

import math
import queue
import threading
import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import TypeVar

import torch
from torch import Tensor

from .._config import get_config, parse_bytes, resolve_backend
from ._arrays import ArrayLike, _finish, _host, chunks_of
from ._plan import Plan, plan_blocks, read_amplification

__all__ = ["Chunks", "blocks", "chunk_length", "frame_batches", "row_tiles"]

T = TypeVar("T")

_AMPLIFICATION_WARNING = 4.0
_PAD_MODES = ("reflect", "replicate", "zeros")


def chunk_length(
    item_shape: Sequence[int],
    *,
    dtype: torch.dtype,
    memory: int | None = None,
    batch: int | None = None,
    frame_shape: Sequence[int] | None = None,
    overhead: float = 1.0,
    multiple: int | None = None,
    maximum: int | None = None,
) -> int:
    """How many items (frames, rows, ...) of ``item_shape`` fit the budget.

    The budget is the size of **one chunk on the working device**, not the
    total memory used by the process.

    Parameters
    ----------
    item_shape : sequence of int
        Shape of one item.
    dtype : torch.dtype
        Working dtype (sets the bytes per element).
    memory : int, optional
        Budget in bytes (default: configuration).
    batch : int, optional
        If set, the budget is ``batch * prod(frame_shape)`` elements
        instead (``batch`` wins over ``memory``).
    frame_shape : sequence of int, optional
        Shape of one frame (used with ``batch``; default ``item_shape``).
    overhead : float
        Number of chunk-sized temporaries, in the working dtype, that the
        consumer holds, counting the chunk itself.
    multiple : int, optional
        Storage chunk length along the iterated axis: the result is rounded
        down to a multiple of it when it is at least that large.
    maximum : int, optional
        Upper bound (e.g. the length of the iterated axis).

    Returns
    -------
    int
        Number of items, >= 1.
    """
    if memory is None and batch is None:
        memory = get_config().memory
    item = math.prod(item_shape)
    if batch is not None:
        frame = math.prod(item_shape if frame_shape is None else frame_shape)
        budget = batch * frame
    else:
        budget = memory // dtype.itemsize
    n = max(1, int(budget // (overhead * max(item, 1))))
    if maximum is not None:
        n = max(1, min(n, maximum))
    if multiple and n >= multiple:
        n = (n // multiple) * multiple
    return n


def _prefetch(source: Iterable[T], depth: int) -> Iterator[T]:
    """Run ``source`` in a daemon thread, keeping up to ``depth`` items ready.

    Exceptions raised by ``source`` are re-raised in the consumer. When the
    consumer stops early (``break``, ``close()``), the thread is stopped and
    joined. Up to ``depth + 2`` items are alive at a time.
    """
    items: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    done = object()

    def worker() -> None:
        iterator = iter(source)
        try:
            for item in iterator:
                if stop.is_set():
                    return
                items.put((True, item))
                if stop.is_set():
                    return
            items.put((True, done))
        except BaseException as error:  # re-raised in the consumer
            if not stop.is_set():
                items.put((False, error))
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while True:
            ok, item = items.get()
            if not ok:
                raise item
            if item is done:
                return
            yield item
    finally:
        stop.set()
        # drain so that a worker blocked on put() can observe `stop`
        while thread.is_alive():
            try:
                items.get(timeout=0.01)
            except queue.Empty:
                pass
        thread.join()


def _pad(x: Tensor, pad: Sequence[tuple[int, int]], mode: str) -> Tensor:
    """Pad the trailing ``len(pad)`` dims of ``x``."""
    offset = x.ndim - len(pad)
    for d, (before, after) in enumerate(pad):
        if not (before or after):
            continue
        dim = offset + d
        n = x.shape[dim]
        if mode == "zeros":
            shape = list(x.shape)
            shape[dim] = before
            lo = x.new_zeros(shape)
            shape[dim] = after
            hi = x.new_zeros(shape)
            x = torch.cat([lo, x, hi], dim)
            continue
        idx = torch.arange(-before, n + after, device=x.device)
        if mode == "replicate" or max(before, after) > n - 1:
            idx = idx.clamp(0, n - 1)
        else:  # reflect (edge not repeated)
            idx = idx.abs()
            idx = torch.where(idx > n - 1, 2 * (n - 1) - idx, idx)
        x = x.index_select(dim, idx)
    return x


class Chunks:
    """Re-iterable, sized sequence of lazily loaded chunks.

    Built by :func:`frame_batches`, :func:`row_tiles` and :func:`blocks`.
    Sizes and the backend are resolved at construction, so iterating twice
    gives identical chunks and ``len()`` is exact.

    Every yielded data tensor is owned by the consumer: it may be modified
    in place and never aliases the source.

    Attributes
    ----------
    shape : tuple of int
        Shape of the source.
    """

    def __init__(
        self,
        source: ArrayLike,
        plans: list[Plan],
        *,
        dtype: torch.dtype,
        device: torch.device,
        prefetch: int = 0,
        pad: str | None = None,
        with_inner: bool = False,
    ) -> None:
        self.shape = tuple(int(s) for s in source.shape)
        self._source = source
        self._plans = plans
        self._dtype = dtype
        self._device = device
        self._prefetch = prefetch
        self._pad = pad
        self._with_inner = with_inner

    def __len__(self) -> int:
        return len(self._plans)

    def _load_host(self) -> Iterator[tuple[Plan, Tensor, bool]]:
        """Host side only: slice and convert to CPU tensors."""
        for plan in self._plans:
            t, shared, readonly = _host(
                self._source[plan.read],
                source=self._source,
                working=self._dtype,
            )
            yield plan, t, shared or readonly

    def _load_tensor(self) -> Iterator[tuple[Plan, Tensor, bool]]:
        for plan in self._plans:
            yield plan, self._source[plan.read].detach(), True

    def __iter__(self) -> Iterator[tuple]:
        if isinstance(self._source, Tensor):
            loaded = self._load_tensor()
        else:
            loaded = self._load_host()
            if self._prefetch > 0:
                loaded = _prefetch(loaded, self._prefetch)
        try:
            for plan, t, alias in loaded:
                data = _finish(
                    t, alias=alias, dtype=self._dtype, device=self._device
                )
                if self._pad is not None:
                    data = _pad(data, plan.pad, self._pad)
                if self._with_inner:
                    yield plan.index, data, plan.inner
                else:
                    yield plan.index, data
        finally:
            loaded.close()

    def __repr__(self) -> str:
        return f"Chunks(shape={self.shape}, len={len(self)})"


def _resolve(
    source: ArrayLike,
    memory: int | str | None,
    batch: int | None,
    prefetch: int | None,
    dtype: torch.dtype | str | None,
    device: torch.device | str | None,
    align: Sequence[int] | None,
) -> dict:
    """Capture the configuration once, at construction."""
    cfg = get_config()
    if batch is not None and (isinstance(batch, bool) or batch < 1):
        raise ValueError(f"batch must be a positive int, got {batch}")
    if prefetch is not None and prefetch < 0:
        raise ValueError(f"prefetch must be >= 0, got {prefetch}")
    shape = tuple(int(s) for s in source.shape)
    align = chunks_of(source) if align is None else tuple(align)
    if align is not None and len(align) != len(shape):
        align = None
    return dict(
        shape=shape,
        memory=cfg.memory if memory is None else parse_bytes(memory),
        batch=cfg.batch if batch is None else batch,
        prefetch=cfg.prefetch if prefetch is None else prefetch,
        backend=resolve_backend(dtype, device, like=source),
        align=align,
    )


def _check_ndim(shape: tuple[int, ...], ndim: int) -> None:
    if ndim < 0:
        raise ValueError(f"ndim must be >= 0, got {ndim}")
    if len(shape) < ndim + 1:
        raise ValueError(
            f"Data of shape {shape} have fewer than ndim + 1 = {ndim + 1} "
            f"dims. Data must be laid out as (*batch, frames, *spatial); "
            f"add a frame axis (e.g. Y[None]) if there is a single frame."
        )


def _warn_amplification(factor: float, what: str, advice: str) -> None:
    if factor > _AMPLIFICATION_WARNING:
        warnings.warn(
            f"{what} decompress each storage chunk about {factor:.1f} "
            f"times. {advice}",
            RuntimeWarning,
            stacklevel=3,
        )


def frame_batches(
    Y: ArrayLike,
    *,
    ndim: int = 2,
    overhead: float = 1.0,
    memory: int | str | None = None,
    batch: int | None = None,
    align: Sequence[int] | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
    prefetch: int | None = None,
) -> Chunks:
    """Batches of frames, one batch element at a time.

    Parameters
    ----------
    Y : (*batch, frames, *spatial) array-like
        Source data.
    ndim : int
        Number of trailing spatial dims.
    overhead : float
        Number of chunk-sized temporaries the consumer holds (see
        :func:`chunk_length`).
    memory, batch, prefetch : optional
        Chunk budget in bytes, explicit frame count, and read-ahead depth
        (default: configuration).
    align : sequence of int, optional
        Storage chunks to align to (default: ``chunks_of(Y)``). Pass
        ``chunks_of(out)`` when writing to a chunked output.
    dtype, device : optional
        Working backend.

    Returns
    -------
    Chunks
        Yields ``(index, data)``: ``data`` is ``(n, *spatial)``; ``index``
        is a full-length tuple (ints for batch dims, a slice for frames,
        full slices for spatial dims) so that ``out[index] = result``
        writes the batch back.
    """
    opt = _resolve(Y, memory, batch, prefetch, dtype, device, align)
    shape = opt["shape"]
    _check_ndim(shape, ndim)
    lead = len(shape) - ndim - 1
    nframes, spatial = shape[lead], shape[lead + 1 :]
    multiple = opt["align"][lead] if opt["align"] else None
    n = chunk_length(
        spatial,
        dtype=opt["backend"]["dtype"],
        memory=opt["memory"],
        batch=opt["batch"],
        frame_shape=spatial,
        overhead=overhead,
        multiple=multiple,
        maximum=max(nframes, 1),
    )
    plans = plan_blocks(shape, lead, (n,) + spatial, (0,) * (ndim + 1), False)
    if multiple and nframes:
        regions = [slice(s, min(s + n, nframes)) for s in range(0, nframes, n)]
        _warn_amplification(
            read_amplification(regions, multiple, nframes),
            "Frame batches",
            "Increase `memory` or rechunk along the frame axis.",
        )
    return Chunks(Y, plans, prefetch=opt["prefetch"], **opt["backend"])


def row_tiles(
    Y: ArrayLike,
    *,
    ndim: int = 2,
    axis: int = -2,
    overhead: float = 1.0,
    memory: int | str | None = None,
    batch: int | None = None,
    align: Sequence[int] | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
    prefetch: int | None = None,
) -> Chunks:
    """Strips along one spatial axis holding all frames of all batches.

    Parameters
    ----------
    Y : (*batch, frames, *spatial) array-like
        Source data.
    ndim : int
        Number of trailing spatial dims.
    axis : int
        Spatial axis along which strips are cut (negative, counted from
        the end, or non-negative).
    overhead, memory, batch, align, dtype, device, prefetch
        See :func:`frame_batches`. With ``batch``, a tile holds about as
        many elements as ``batch`` frames.

    Returns
    -------
    Chunks
        Yields ``(index, data)``: ``data`` is ``(*batch, frames, rows,
        ...)`` and ``index`` selects the strip in ``Y``.

    Notes
    -----
    On a store chunked along frames, every tile touches every chunk; the
    read amplification is computed and a warning is emitted when it is
    large.
    """
    opt = _resolve(Y, memory, batch, prefetch, dtype, device, align)
    shape = opt["shape"]
    _check_ndim(shape, ndim)
    if ndim < 1:
        raise ValueError("row_tiles needs at least one spatial dim")
    axis = axis if axis >= 0 else len(shape) + axis
    if not len(shape) - ndim <= axis < len(shape):
        raise ValueError(f"axis {axis} is not a spatial axis")
    nrows = shape[axis]
    item = shape[:axis] + shape[axis + 1 :]
    multiple = opt["align"][axis] if opt["align"] else None
    dt = opt["backend"]["dtype"]
    n = chunk_length(
        item,
        dtype=dt,
        memory=opt["memory"],
        batch=opt["batch"],
        frame_shape=shape[len(shape) - ndim :],
        overhead=overhead,
        multiple=multiple,
        maximum=max(nrows, 1),
    )
    if opt["batch"] is None:
        fits = opt["memory"] // dt.itemsize >= overhead * math.prod(item)
        if not fits and math.prod(shape):
            warnings.warn(
                "A single row exceeds the memory budget; tiles of one row "
                "are used anyway.",
                RuntimeWarning,
                stacklevel=2,
            )
    block = list(shape)
    block[axis] = n
    block = [max(b, 1) for b in block]
    plans = plan_blocks(shape, 0, block, (0,) * len(shape), False)
    if opt["align"] and math.prod(shape):
        regions = [slice(s, min(s + n, nrows)) for s in range(0, nrows, n)]
        _warn_amplification(
            read_amplification(regions, opt["align"][axis], nrows),
            "Row tiles",
            "Increase `memory`, rechunk the store along this axis, or use "
            "histogram-based quantiles (optix.stats), which read frame "
            "batches.",
        )
    return Chunks(Y, plans, prefetch=opt["prefetch"], **opt["backend"])


def _auto_block(
    extents: Sequence[int],
    halo: Sequence[int],
    budget: float,
    overhead: float,
    align: Sequence[int] | None,
) -> list[int]:
    block = [max(e, 1) for e in extents]

    def size(b: Sequence[int]) -> float:
        return math.prod(x + 2 * h for x, h in zip(b, halo, strict=True))

    while size(block) * overhead > budget and max(block) > 1:
        d = max(range(len(block)), key=lambda i: block[i])
        block[d] = -(-block[d] // 2)
    if align is not None:
        block = [
            (b // c) * c if c and b >= c else b
            for b, c in zip(block, align, strict=True)
        ]
    if size(block) * overhead > budget:
        warnings.warn(
            "The halo alone exceeds the memory budget; using blocks of "
            "size one.",
            RuntimeWarning,
            stacklevel=3,
        )
    return block


def blocks(
    Y: ArrayLike,
    block: Sequence[int | None] | None = None,
    halo: int | Sequence[int] = 0,
    *,
    pad: str | None = None,
    overhead: float = 1.0,
    memory: int | str | None = None,
    batch: int | None = None,
    align: Sequence[int] | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
    prefetch: int | None = None,
) -> Chunks:
    """nD blocks over trailing dims, with a halo.

    Parameters
    ----------
    Y : array-like
        Source data.
    block : sequence of int or None, optional
        Core block size along each of the trailing ``len(block)`` dims;
        ``None`` entries mean the whole extent. If ``block`` is None, the
        blocked dims are the trailing ``len(halo)`` dims (all dims if
        ``halo`` is a scalar) and the size is chosen automatically:
        starting from the full extents, the largest dim is halved until
        ``prod(block + 2 * halo) * overhead`` fits the budget, then sizes
        are aligned to storage chunks. Dims before the blocked dims are
        iterated one index at a time.
    halo : int or sequence of int
        Halo width along each blocked dim.
    pad : {None, "reflect", "replicate", "zeros"}
        None: halos are clipped at the array boundary, so edge blocks are
        smaller and the algorithm applies its own boundary condition, as it
        would on the whole array. Otherwise every block has shape exactly
        ``block + 2 * halo``, boundary halos and short trailing cores being
        filled by padding. Where reflection would need a pad at least as
        large as the loaded extent, replicate padding is used along that
        dim instead (with a warning at construction).
    overhead, memory, batch, align, dtype, device, prefetch
        See :func:`frame_batches`. With ``batch``, the budget is ``batch``
        frames of the trailing two dims.

    Returns
    -------
    Chunks
        Yields ``(index, data, inner)``: ``data`` is core + halo,
        ``data[inner]`` is the core, and ``index`` is where the core lives
        in ``Y`` (and in an output of the same shape).

    Notes
    -----
    Writing results back into ``Y`` itself is wrong when ``halo > 0`` (a
    block's halo would read already-processed neighbours).
    """
    opt = _resolve(Y, memory, batch, prefetch, dtype, device, align)
    shape = opt["shape"]
    if pad is not None and pad not in _PAD_MODES:
        raise ValueError(f"pad must be None or one of {_PAD_MODES}")
    if block is not None:
        nblk = len(block)
    elif not isinstance(halo, int):
        nblk = len(halo)
    else:
        nblk = len(shape)
    if nblk > len(shape):
        raise ValueError(f"Cannot block {nblk} dims of a {len(shape)}D array")
    halo = (halo,) * nblk if isinstance(halo, int) else tuple(halo)
    if len(halo) != nblk:
        raise ValueError("block and halo must have the same length")
    lead = len(shape) - nblk
    extents = shape[lead:]
    calign = opt["align"][lead:] if opt["align"] else None
    if block is None:
        dt = opt["backend"]["dtype"]
        if opt["batch"] is not None:
            budget = opt["batch"] * math.prod(shape[-2:])
        else:
            budget = opt["memory"] // dt.itemsize
        core = _auto_block(extents, halo, budget, overhead, calign)
    else:
        core = [
            max(e, 1) if b is None else min(int(b), max(e, 1))
            for b, e in zip(block, extents, strict=True)
        ]
    plans = plan_blocks(shape, lead, core, halo, pad is not None)
    if pad == "reflect" and any(
        max(p) > s.stop - s.start - 1
        for plan in plans
        for p, s in zip(plan.pad, plan.read[lead:], strict=True)
    ):
        warnings.warn(
            "Reflect padding needs a pad smaller than the loaded extent; "
            "some blocks (short or thin edge blocks) fall back to "
            "replicate padding along those dims. Use a block size that "
            "divides the array, a smaller halo, or pad='replicate'.",
            RuntimeWarning,
            stacklevel=2,
        )
    return Chunks(
        Y,
        plans,
        prefetch=opt["prefetch"],
        pad=pad,
        with_inner=True,
        **opt["backend"],
    )
