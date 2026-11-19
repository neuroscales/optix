"""Chunked iteration over array-likes, with optional prefetching."""

import math
import queue
import threading
import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import TypeVar

import numpy as np
import torch
from torch import Tensor

from .._config import get_config, parse_bytes, resolve_backend
from ._arrays import (
    ArrayLike,
    Index,
    WritableArrayLike,
    _finish,
    _host,
    _same_storage,
    as_tensor,
    chunks_of,
    store,
)
from ._layout import Labeled, Layout, bind
from ._plan import Plan, plan_grid, read_amplification

__all__ = [
    "Chunk",
    "Chunks",
    "blocks",
    "chunk_length",
    "frame_batches",
    "group_frames",
    "row_tiles",
]

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
    """How many items (frames, rows, ...) of `item_shape` fit the budget.

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
        If set, the budget is `batch * prod(frame_shape)` elements
        instead (`batch` wins over `memory`).
    frame_shape : sequence of int, optional
        Shape of one frame (used with `batch`; default `item_shape`).
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
        Number of items, ≥ 1.
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
    """Run `source` in a daemon thread, keeping up to `depth` items ready.

    Exceptions raised by `source` are re-raised in the consumer. When the
    consumer stops early (`break`, `close()`), the thread is stopped and
    joined. Up to `depth + 2` items are alive at a time.
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
    """Pad the trailing `len(pad)` dims of `x`."""
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


def _length(index: int | slice) -> int:
    return 1 if isinstance(index, int) else index.stop - index.start


_INTO = ("source", "map", "samples")


class _View:
    """How the chunks of one iteration relate to the layout."""

    def __init__(
        self, shape: tuple[int, ...], layout: Layout, flatten: bool
    ) -> None:
        self.shape = shape
        self.layout = layout
        self.group, self.sample, self.spatial = layout.positions(len(shape))
        self.flatten = flatten
        self._blocked = tuple(sorted(self.spatial))

    def arrange(self, plan: Plan) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Kept sample axes and the permutation applied after loading."""
        kept = [a for a, r in enumerate(plan.read) if isinstance(r, slice)]
        samples = tuple(a for a in kept if a in self.sample)
        perm = tuple(kept.index(a) for a in samples + self.spatial)
        if len(perm) != len(kept):  # a group axis was not iterated
            raise AssertionError("group axes must be integer-indexed")
        return samples, perm

    def copies(self, contiguous: bool) -> bool:
        """Whether turning a loaded block into a chunk may copy it.

        With `contiguous`, any permutation to layout order copies.
        Otherwise only flattening sample axes that are not adjacent in
        storage (a spatial axis lies between them) does.
        """
        sample = sorted(self.sample)
        if contiguous:
            order = list(sample) + list(self.spatial)
            return order != sorted(order)
        if self.flatten and sample:
            return any(sample[0] < a < sample[-1] for a in self.spatial)
        return False

    def inner(self, plan: Plan) -> Index:
        """`plan.inner` (blocked axes in storage order) in layout order."""
        ranks = {a: i for i, a in enumerate(self._blocked)}
        return tuple(plan.inner[ranks[a]] for a in self.spatial)


class Chunk(tuple):
    """One loaded chunk.

    Unpacks as `(index, data)`, or `(index, data, inner)` for
    [`blocks`][optix.io.blocks].

    Attributes
    ----------
    index : Index
        Region of the core in the source, in storage order (full-length:
        ints and slices). `out[index] = ...` is only valid for the default
        layout and a chunk spanning one sample axis: use
        [`store`][optix.io.Chunk.store].
    data : Tensor
        Loaded data, owned by the consumer; a strided view in layout
        order unless the iterator was built with `contiguous=True`.
        [`frame_batches`][optix.io.frame_batches]: `(n, *spatial)`;
        [`row_tiles`][optix.io.row_tiles]: `(*sample, *spatial)`;
        [`blocks`][optix.io.blocks]: `(*spatial)` core + halo. Spatial
        axes are in layout order, sample axes in storage order.
    inner : Index or None
        Crop of `data` that gives the core ([`blocks`][optix.io.blocks]
        only).
    group : tuple of int
        Position along the group axes (layout order).
    samples : tuple of slice
        Region along the sample axes (storage order); index a per-sample
        state of shape `(*group, *sample, ...)` with
        `state[chunk.group + chunk.samples]`.
    sample_shape : tuple of int
        Shape of `samples`; `n` is its product.
    spatial : tuple of slice
        Region of the core along the spatial axes (layout order).
    """

    index: Index
    data: Tensor
    inner: Index | None

    def __new__(
        cls,
        view: _View,
        plan: Plan,
        data: Tensor,
        inner: Index | None,
        samples: tuple[int, ...],
    ) -> "Chunk":
        items = (
            (plan.index, data) if inner is None else (plan.index, data, inner)
        )
        self = super().__new__(cls, items)
        self.index = plan.index
        self.data = data
        self.inner = inner
        self._view = view
        self._kept = samples
        return self

    @property
    def group(self) -> tuple[int, ...]:
        return tuple(self.index[a] for a in self._view.group)

    @property
    def samples(self) -> tuple[slice, ...]:
        out = []
        for a in self._view.sample:
            i = self.index[a]
            out.append(slice(i, i + 1) if isinstance(i, int) else i)
        return tuple(out)

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return tuple(_length(self.index[a]) for a in self._view.sample)

    @property
    def n(self) -> int:
        return math.prod(self.sample_shape)

    @property
    def spatial(self) -> tuple[slice, ...]:
        return tuple(self.index[a] for a in self._view.spatial)

    def _nlead(self) -> int:
        return 1 if self._view.flatten else len(self._kept)

    def _core(self, result: Tensor) -> Tensor:
        if (
            self.inner is not None
            and tuple(result.shape) == tuple(self.data.shape)
            and tuple(result.shape) != self._core_shape()
        ):
            return result[self.inner]
        return result

    def _core_shape(self) -> tuple[int, ...]:
        return tuple(_length(self.index[a]) for a in self._view.spatial)

    def store(
        self,
        out: WritableArrayLike,
        result: Tensor,
        *,
        into: str = "source",
    ) -> None:
        """Write a result computed from this chunk into `out`.

        Parameters
        ----------
        out : writable array-like
            Destination (a [`Labeled`][optix.io.Labeled] is unwrapped).
        result : Tensor
            Shaped like `data` along the sample and spatial axes (see
            `into`). For [`blocks`][optix.io.blocks], a result of the loaded
            shape is cropped to the core.
        into : {"source", "map", "samples"}
            `"source"`: `out` has the shape of the source, in storage
            order, and `result` is `(n, *spatial)` (or `data`'s
            shape); it is un-flattened and permuted back.
            `"map"`: `out` is a `(*group, *spatial)` map in layout
            order and `result` is `(*spatial)`, reduced over the
            samples (e.g. over a row tile, which holds all samples of a
            group).
            `"samples"`: `out` is a per-sample state
            `(*group, *sample, *rest)` and `result` is `(n, *rest)`.
        """
        if into not in _INTO:
            raise ValueError(f"into must be one of {_INTO}, got {into!r}")
        if isinstance(out, Labeled):
            out = out.array
        view = self._view
        if into == "samples":
            nlead = 0 if self.inner is not None else self._nlead()
            rest = tuple(result.shape[nlead:])
            value = result.reshape(*self.sample_shape, *rest)
            index = self.group + self.samples
            index += (slice(None),) * (len(out.shape) - len(index))
            store(out, index, value)
            return
        core = self._core_shape()
        if into == "map":
            value = self._core(result)
            if tuple(value.shape) != core:
                raise ValueError(
                    f"Map result has shape {tuple(value.shape)}, expected "
                    f"{core}"
                )
            store(out, self.group + self.spatial, value)
            return
        value = self._core(result)
        nlead = 0 if self.inner is not None else self._nlead()
        lead = tuple(_length(self.index[a]) for a in self._kept)
        if tuple(value.shape[nlead:]) != core or (
            value.numel() != math.prod(lead) * math.prod(core)
        ):
            raise ValueError(
                f"Result has shape {tuple(value.shape)}, expected "
                f"({math.prod(lead)}, *{core}) for this chunk"
            )
        value = value.reshape(*lead, *core)
        current = self._kept + view.spatial
        kept = sorted(current)
        back = [current.index(a) for a in kept]
        if back != list(range(len(back))):
            value = value.permute(back)
        store(out, self.index, value)

    def __repr__(self) -> str:
        return (
            f"Chunk(index={self.index}, data={tuple(self.data.shape)}, "
            f"group={self.group})"
        )


class Chunks:
    """Re-iterable, sized sequence of lazily loaded chunks.

    Built by [`frame_batches`][optix.io.frame_batches],
    [`row_tiles`][optix.io.row_tiles] and [`blocks`][optix.io.blocks].
    Sizes and the backend are resolved at construction, so iterating twice
    gives identical chunks and `len()` is exact. Items are
    [`Chunk`][optix.io.Chunk].

    Every yielded data tensor is owned by the consumer: it may be modified
    in place and never aliases the source. It is a permuted (strided) view
    of the loaded block, not a dense copy, unless `contiguous=True`.

    Attributes
    ----------
    shape : tuple of int
        Shape of the source.
    layout : Layout
        Layout of the source (the default one when none was given).
    group_shape, sample_shape, spatial_shape : tuple of int
        See [`Layout.shape_of`][optix.io.Layout.shape_of].
    """

    def __init__(
        self,
        source: ArrayLike,
        plans: list[Plan],
        *,
        view: _View,
        dtype: torch.dtype,
        device: torch.device,
        prefetch: int = 0,
        pad: str | None = None,
        with_inner: bool = False,
        contiguous: bool = False,
    ) -> None:
        self.shape = tuple(int(s) for s in source.shape)
        self.layout = view.layout
        shapes = view.layout.shape_of(self.shape)
        self.group_shape, self.sample_shape, self.spatial_shape = shapes
        self._source = source
        self._plans = plans
        self._view = view
        self._dtype = dtype
        self._device = device
        self._prefetch = prefetch
        self._pad = pad
        self._with_inner = with_inner
        self._contiguous = contiguous

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

    def __iter__(self) -> Iterator[Chunk]:
        if isinstance(self._source, Tensor):
            loaded = self._load_tensor()
        else:
            loaded = self._load_host()
            if self._prefetch > 0:
                loaded = _prefetch(loaded, self._prefetch)
        view = self._view
        try:
            for plan, t, alias in loaded:
                data = _finish(
                    t, alias=False, dtype=self._dtype, device=self._device
                )
                if self._pad is not None:
                    data = _pad(data, plan.pad, self._pad)
                samples, perm = view.arrange(plan)
                if perm != tuple(range(len(perm))):
                    data = data.permute(perm)
                if self._contiguous:
                    data = data.contiguous()
                if alias and _same_storage(data, t):
                    data = data.clone()
                if view.flatten:  # copies only if the sample axes don't merge
                    data = data.reshape(-1, *data.shape[len(samples) :])
                inner = view.inner(plan) if self._with_inner else None
                yield Chunk(view, plan, data, inner, samples)
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
            f"add a frame axis (e.g. data[None]) if there is a single frame, "
            f"or pass a `layout`."
        )


def _layout_for(
    data: ArrayLike, layout: Layout | None, ndim: int | None
) -> tuple[ArrayLike, Layout]:
    """Raw source and layout (default: `ndim` trailing spatial dims)."""
    source, layout = bind(data, layout, ndim)
    if layout is None:
        ndim = 2 if ndim is None else ndim
        shape = tuple(int(s) for s in source.shape)
        _check_ndim(shape, ndim)
        layout = Layout.default(len(shape), ndim)
    return source, layout


def _warn_amplification(factor: float, what: str, advice: str) -> None:
    if factor > _AMPLIFICATION_WARNING:
        warnings.warn(
            f"{what} decompress each storage chunk about {factor:.1f} "
            f"times. {advice}",
            RuntimeWarning,
            stacklevel=3,
        )


def _amplification(
    shape: tuple[int, ...],
    specs: Sequence[tuple[int, int] | None],
    align: Sequence[int],
    axes: Sequence[int],
) -> float:
    """Product of the per-axis read amplifications along `axes`."""
    factor = 1.0
    for a in axes:
        n, spec = shape[a], specs[a]
        if spec is None:
            regions = [slice(i, i + 1) for i in range(n)]
        else:
            b = spec[0]
            regions = [slice(s, min(s + b, n)) for s in range(0, n, b)]
        factor *= read_amplification(regions, align[a], n)
    return factor


def frame_batches(
    data: ArrayLike,
    *,
    ndim: int | None = None,
    layout: Layout | None = None,
    overhead: float = 1.0,
    memory: int | str | None = None,
    batch: int | None = None,
    align: Sequence[int] | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
    prefetch: int | None = None,
    contiguous: bool = False,
) -> Chunks:
    """Batches of samples (frames), one group at a time.

    Parameters
    ----------
    data : array-like or Labeled
        Source data.
    ndim : int, optional
        Without a layout: number of trailing spatial dims (default 2); the
        leading dims are samples (`(*batch, frames, *spatial)`).
    layout : Layout, optional
        Axis roles (default: that of a [`Labeled`][optix.io.Labeled]
        `data`, else `Layout.default(data.ndim, ndim)`).
    overhead : float
        Number of chunk-sized temporaries the consumer holds (see
        [`chunk_length`][optix.io.chunk_length]).
    memory, batch, prefetch : optional
        Chunk budget in bytes, explicit frame count, and read-ahead depth
        (default: configuration).
    align : sequence of int, optional
        Storage chunks to align to (default: `chunks_of(data)`). Pass
        `chunks_of(out)` when writing to a chunked output.
    dtype, device : optional
        Working backend.
    contiguous : bool
        If True, every chunk is made dense (C-contiguous in layout order),
        e.g. for FFTs or convolutions, and that copy is counted in the
        budget (`overhead + 1`) whenever the layout permutes the axes.
        By default chunks are strided views of the loaded block: the only
        copy is the unavoidable one when sample axes that are not adjacent
        in storage are flattened (also counted in the budget).

    Returns
    -------
    Chunks
        Yields [`Chunk`][optix.io.Chunk] items, unpacking as
        `(index, data)`: `data` is `(n, *spatial)`, the sample axes
        being flattened in storage order and the spatial axes in layout
        order. Write a result back with `chunk.store(out, result)`.

    Notes
    -----
    Groups are visited one at a time. Within a group, the innermost (in
    storage) sample axes are taken whole while the budget allows; the
    next one is split into blocks (rounded down to a multiple of its
    storage chunk when at least that large) and the outer ones are
    iterated one index at a time. A warning is emitted when the read
    amplification over the group and sample axes is large.
    """
    source, layout = _layout_for(data, layout, ndim)
    opt = _resolve(source, memory, batch, prefetch, dtype, device, align)
    shape = opt["shape"]
    view = _View(shape, layout, flatten=True)
    group, sample, spatial = view.group, view.sample, view.spatial
    overhead = overhead + view.copies(contiguous)
    spatial_shape = tuple(shape[a] for a in spatial)
    nsamples = math.prod(shape[a] for a in sample)
    budget = chunk_length(
        spatial_shape,
        dtype=opt["backend"]["dtype"],
        memory=opt["memory"],
        batch=opt["batch"],
        frame_shape=spatial_shape,
        overhead=overhead,
        maximum=max(nsamples, 1),
    )
    specs: list[tuple[int, int] | None] = [None] * len(shape)
    for a in spatial:
        specs[a] = (max(shape[a], 1), 0)
    inner = 1
    split = len(sample) - 1
    while split >= 0 and inner * shape[sample[split]] <= budget:
        a = sample[split]
        specs[a] = (max(shape[a], 1), 0)
        inner *= shape[a]
        split -= 1
    if split >= 0:
        a = sample[split]
        n = max(1, min(budget // max(inner, 1), shape[a]))
        multiple = opt["align"][a] if opt["align"] else None
        if multiple and n >= multiple:
            n = (n // multiple) * multiple
        specs[a] = (n, 0)
    order = group + tuple(a for a in range(len(shape)) if a not in group)
    plans = plan_grid(shape, specs, False, order)
    if opt["align"] and math.prod(shape):
        _warn_amplification(
            _amplification(shape, specs, opt["align"], group + sample),
            "Frame batches",
            "Increase `memory` or rechunk along the sample axes.",
        )
    return Chunks(
        source,
        plans,
        view=view,
        prefetch=opt["prefetch"],
        contiguous=contiguous,
        **opt["backend"],
    )


def _spatial_axis(axis: int | str, layout: Layout, spatial: tuple) -> int:
    """Storage position of a spatial axis given by name or position."""
    if isinstance(axis, str):
        if layout.axes is None or axis not in layout.axes:
            raise ValueError(f"Unknown axis {axis!r}")
        pos = layout.axes.index(axis)
        if pos not in spatial:
            raise ValueError(f"axis {axis!r} is not a spatial axis")
        return pos
    if not -len(spatial) <= axis < len(spatial):
        raise ValueError(
            f"axis {axis} is not a spatial axis (there are {len(spatial)})"
        )
    return spatial[axis]


def row_tiles(
    data: ArrayLike,
    *,
    ndim: int | None = None,
    layout: Layout | None = None,
    axis: int | str = -2,
    overhead: float = 1.0,
    memory: int | str | None = None,
    batch: int | None = None,
    align: Sequence[int] | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
    prefetch: int | None = None,
    contiguous: bool = False,
) -> Chunks:
    """Strips along one spatial axis holding all samples of one group.

    Parameters
    ----------
    data : array-like or Labeled
        Source data.
    ndim, layout
        See [`frame_batches`][optix.io.frame_batches].
    axis : int or str
        Spatial axis along which strips are cut: a position among the
        spatial axes in layout order (negative allowed), or a name.
    overhead, memory, batch, align, dtype, device, prefetch, contiguous
        See [`frame_batches`][optix.io.frame_batches]. With `batch`, a
        tile holds about as many elements as `batch` frames.

    Returns
    -------
    Chunks
        Yields [`Chunk`][optix.io.Chunk] items, unpacking as
        `(index, data)`: `data` is `(*sample, *spatial)` (sample
        axes in storage order, not flattened; spatial axes in layout order,
        cut along `axis`).
        Store a per-pixel reduction with `chunk.store(map, r,
        into="map")`.

    Notes
    -----
    On a store chunked along samples, every tile touches every chunk; the
    read amplification is computed and a warning is emitted when it is
    large.
    """
    source, layout = _layout_for(data, layout, ndim)
    opt = _resolve(source, memory, batch, prefetch, dtype, device, align)
    shape = opt["shape"]
    view = _View(shape, layout, flatten=False)
    group, sample, spatial = view.group, view.sample, view.spatial
    overhead = overhead + view.copies(contiguous)
    if not spatial:
        raise ValueError("row_tiles needs at least one spatial dim")
    cut = _spatial_axis(axis, layout, spatial)
    nrows = shape[cut]
    item = tuple(shape[a] for a in sample + spatial if a != cut)
    dt = opt["backend"]["dtype"]
    n = chunk_length(
        item,
        dtype=dt,
        memory=opt["memory"],
        batch=opt["batch"],
        frame_shape=tuple(shape[a] for a in spatial),
        overhead=overhead,
        multiple=opt["align"][cut] if opt["align"] else None,
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
    specs: list[tuple[int, int] | None] = [None] * len(shape)
    for a in sample + spatial:
        specs[a] = (max(shape[a], 1), 0)
    specs[cut] = (max(n, 1), 0)
    order = group + tuple(a for a in range(len(shape)) if a not in group)
    plans = plan_grid(shape, specs, False, order)
    if opt["align"] and math.prod(shape):
        _warn_amplification(
            _amplification(shape, specs, opt["align"], group + (cut,)),
            "Row tiles",
            "Increase `memory`, rechunk the store along this axis, or use "
            "histogram-based quantiles (optix.stats), which read frame "
            "batches.",
        )
    return Chunks(
        source,
        plans,
        view=view,
        prefetch=opt["prefetch"],
        contiguous=contiguous,
        **opt["backend"],
    )


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
    data: ArrayLike,
    block: Sequence[int | None] | None = None,
    halo: int | Sequence[int] = 0,
    *,
    layout: Layout | None = None,
    pad: str | None = None,
    overhead: float = 1.0,
    memory: int | str | None = None,
    batch: int | None = None,
    align: Sequence[int] | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
    prefetch: int | None = None,
    contiguous: bool = False,
) -> Chunks:
    """nD blocks over the spatial axes, with a halo.

    Parameters
    ----------
    data : array-like or Labeled
        Source data.
    block : sequence of int or None, optional
        Core block size along each spatial axis (layout order); `None`
        entries mean the whole extent. Without a layout, the blocked dims
        are the trailing `len(block)` dims. If `block` is None, the
        blocked dims are the spatial axes of the layout, or without one the
        trailing `len(halo)` dims (all dims if `halo` is a scalar), and
        the size is chosen automatically: starting from the full extents,
        the largest dim is halved until `prod(block + 2 * halo) *
        overhead` fits the budget, then sizes are aligned to storage
        chunks. Group and sample axes are iterated one index at a time.
    halo : int or sequence of int
        Halo width along each blocked dim.
    layout : Layout, optional
        Axis roles (default: that of a [`Labeled`][optix.io.Labeled]
        `data`, else as described above).
    pad : {None, "reflect", "replicate", "zeros"}
        None: halos are clipped at the array boundary, so edge blocks are
        smaller and the algorithm applies its own boundary condition, as it
        would on the whole array. Otherwise every block has shape exactly
        `block + 2 * halo`, boundary halos and short trailing cores being
        filled by padding. Where reflection would need a pad at least as
        large as the loaded extent, replicate padding is used along that
        dim instead (with a warning at construction).
    overhead, memory, batch, align, dtype, device, prefetch, contiguous
        See [`frame_batches`][optix.io.frame_batches]. With `batch`, the
        budget is `batch` frames of the last two spatial axes.

    Returns
    -------
    Chunks
        Yields [`Chunk`][optix.io.Chunk] items, unpacking as
        `(index, data, inner)`: `data` is core + halo with the spatial
        axes in layout order, `data[inner]` is the core, and `index` is
        where the core lives in `data`. `chunk.store(out, result)`
        crops and writes back.

    Notes
    -----
    Writing results back into `data` itself is wrong when `halo > 0` (a
    block's halo would read already-processed neighbours).
    """
    source, layout = bind(data, layout)
    shape = tuple(int(s) for s in source.shape)
    if pad is not None and pad not in _PAD_MODES:
        raise ValueError(f"pad must be None or one of {_PAD_MODES}")
    if layout is None:
        if block is not None:
            nblk = len(block)
        elif not isinstance(halo, int):
            nblk = len(halo)
        else:
            nblk = len(shape)
        if nblk > len(shape):
            raise ValueError(
                f"Cannot block {nblk} dims of a {len(shape)}D array"
            )
        layout = Layout.default(len(shape), nblk)
    opt = _resolve(source, memory, batch, prefetch, dtype, device, align)
    view = _View(shape, layout, flatten=False)
    spatial = view.spatial
    overhead = overhead + view.copies(contiguous)
    nblk = len(spatial)
    halo = (halo,) * nblk if isinstance(halo, int) else tuple(halo)
    if len(halo) != nblk or (block is not None and len(block) != nblk):
        raise ValueError("block and halo must have one entry per spatial axis")
    extents = [shape[a] for a in spatial]
    calign = [opt["align"][a] for a in spatial] if opt["align"] else None
    if block is None:
        dt = opt["backend"]["dtype"]
        if opt["batch"] is not None:
            budget = opt["batch"] * math.prod(extents[-2:])
        else:
            budget = opt["memory"] // dt.itemsize
        core = _auto_block(extents, halo, budget, overhead, calign)
    else:
        core = [
            max(e, 1) if b is None else min(int(b), max(e, 1))
            for b, e in zip(block, extents, strict=True)
        ]
    specs: list[tuple[int, int] | None] = [None] * len(shape)
    for a, b, h in zip(spatial, core, halo, strict=True):
        specs[a] = (b, h)
    group = view.group
    order = group + tuple(a for a in range(len(shape)) if a not in group)
    plans = plan_grid(shape, specs, pad is not None, order)
    blocked = sorted(spatial)
    if pad == "reflect" and any(
        max(p) > plan.read[a].stop - plan.read[a].start - 1
        for plan in plans
        for p, a in zip(plan.pad, blocked, strict=True)
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
        source,
        plans,
        view=view,
        prefetch=opt["prefetch"],
        pad=pad,
        with_inner=True,
        contiguous=contiguous,
        **opt["backend"],
    )


def group_frames(
    data: ArrayLike,
    layout: Layout | None = None,
    frame: int = 0,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Iterator[tuple[tuple[int, ...], Tensor]]:
    """Load one frame of each group.

    Parameters
    ----------
    data : array-like or Labeled
    layout : Layout, optional
        Default: that of a labelled `data`, else `(*samples, H, W)`.
    frame : int
        Index of the frame in the flattened sample axes (storage order).

    Yields
    ------
    group : tuple of int
        Position along the group axes (layout order).
    image : Tensor
        The frame, with its spatial axes in layout order. Only this frame
        is read.
    """
    raw = data.array if isinstance(data, Labeled) else data
    if layout is None:
        layout = (
            data.layout
            if isinstance(data, Labeled)
            else Layout.default(len(data.shape), 2)
        )
    shape = tuple(int(s) for s in data.shape)
    gpos, spos, xpos = layout.positions(len(shape))
    sshape = tuple(shape[p] for p in spos)
    nsamples = math.prod(sshape)
    if not -nsamples <= frame < nsamples:
        raise IndexError(f"frame {frame} out of range ({nsamples} frames)")
    sidx = np.unravel_index(frame % max(nsamples, 1), sshape)
    remaining = sorted(xpos)
    perm = [remaining.index(p) for p in xpos]
    for g in np.ndindex(*(shape[p] for p in gpos)):
        index: list[object] = [slice(None)] * len(shape)
        for p, i in zip(gpos, g, strict=True):
            index[p] = int(i)
        for p, i in zip(spos, sidx, strict=True):
            index[p] = int(i)
        image = as_tensor(raw[tuple(index)], dtype=dtype, device=device)
        yield tuple(int(i) for i in g), image.permute(perm)
