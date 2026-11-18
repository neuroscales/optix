"""Exact per-pixel order statistics over row tiles."""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from ..io import ArrayLike, Layout, row_tiles
from ..io._layout import bind, nsamples

__all__ = ["median_mad", "order_statistics", "tile_reduce"]

_MAX_KTHVALUE = 4


def tile_reduce(
    Y: ArrayLike,
    fn: Callable[[Tensor], Tensor | tuple[Tensor, ...]],
    *,
    ndim: int | None = None,
    layout: Layout | None = None,
    overhead: float = 2.0,
    **chunk_options: object,
) -> Tensor | tuple[Tensor, ...]:
    """Apply ``fn`` to row tiles holding all samples and stitch the maps.

    Parameters
    ----------
    Y : array-like or Labeled
        Data.
    fn : callable
        Takes a ``(rows, *rest, N)`` tile (spatial axes in layout order,
        ``N`` = all samples of one group, last) and returns one or several
        ``(rows, *rest)`` maps.
    ndim : int, optional
        Without a layout: number of trailing spatial dims (default 2).
    layout : Layout, optional
        Axis roles (see :func:`optix.io.row_tiles`). Tiles are cut along
        the first spatial axis.
    overhead : float
        Chunk-sized temporaries held by ``fn`` (see
        :func:`optix.io.chunk_length`).
    **chunk_options
        Forwarded to :func:`optix.io.row_tiles`.

    Returns
    -------
    Tensor or tuple of Tensor
        Full ``(*group, *spatial)`` maps, on the working device.
    """
    tiles = row_tiles(
        Y, ndim=ndim, layout=layout, axis=0, overhead=overhead, **chunk_options
    )
    shape = tiles.group_shape + tiles.spatial_shape
    nspatial = len(tiles.spatial_shape)
    outs: list[Tensor] | None = None
    single = False
    with torch.no_grad():
        for chunk in tiles:
            data = chunk.data
            rest = data.shape[data.ndim - nspatial :]
            tile = data.reshape(-1, *rest).movedim(0, -1)
            result = fn(tile)
            if isinstance(result, Tensor):
                single, result = True, (result,)
            if outs is None:
                outs = [
                    torch.empty(shape, dtype=r.dtype, device=r.device)
                    for r in result
                ]
            for out, r in zip(outs, result, strict=True):
                chunk.store(out, r, into="map")
    if outs is None:
        raise ValueError("No data to reduce")
    return outs[0] if single else tuple(outs)


def _nsamples(Y: ArrayLike, ndim: int | None, layout: Layout | None) -> int:
    source, layout = bind(Y, layout, ndim)
    if layout is None:
        layout = Layout.default(len(source.shape), 2 if ndim is None else ndim)
    return nsamples(source, layout)


def order_statistics(
    Y: ArrayLike,
    k: int | Sequence[int],
    *,
    ndim: int | None = None,
    layout: Layout | None = None,
    **kw: object,
) -> Tensor:
    """Per-pixel k-th smallest value(s) over all samples of each group.

    Parameters
    ----------
    Y : array-like or Labeled
        Data.
    k : int or sequence of int
        1-based rank(s), as in ``torch.kthvalue``.
    ndim, layout : optional
        See :func:`tile_reduce`.
    **kw
        Forwarded to :func:`tile_reduce` / :func:`optix.io.row_tiles`.

    Returns
    -------
    (*group, *spatial) or (K, *group, *spatial) Tensor

    Notes
    -----
    Up to four ranks use repeated ``torch.kthvalue``; more ranks use one
    ``sort`` (whose int64 indices make it heavier: ``overhead`` defaults to
    3 then). ``torch.quantile`` is not used: it rejects large inputs.
    NaNs give undefined results.
    """
    n = _nsamples(Y, ndim, layout)
    ks = [k] if isinstance(k, int) else [int(i) for i in k]
    for i in ks:
        if not 1 <= i <= n:
            raise ValueError(f"k={i} out of range [1, {n}]")
    if isinstance(k, int):
        return tile_reduce(
            Y,
            lambda t: t.kthvalue(k, -1).values,
            ndim=ndim,
            layout=layout,
            **kw,
        )
    if len(ks) <= _MAX_KTHVALUE:

        def fn(t: Tensor) -> tuple[Tensor, ...]:
            return tuple(t.kthvalue(i, -1).values for i in ks)

    else:
        kw.setdefault("overhead", 3.0)

        def fn(t: Tensor) -> tuple[Tensor, ...]:
            s = t.sort(-1).values
            return tuple(s[..., i - 1] for i in ks)

    return torch.stack(tile_reduce(Y, fn, ndim=ndim, layout=layout, **kw))


def median_mad(
    Y: ArrayLike,
    *,
    ndim: int | None = None,
    layout: Layout | None = None,
    **kw: object,
) -> tuple[Tensor, Tensor]:
    """Per-pixel median and median absolute deviation, in one pass.

    Parameters
    ----------
    Y : array-like or Labeled
        Data.
    ndim, layout : optional
        See :func:`tile_reduce`.
    **kw
        Forwarded to :func:`tile_reduce` / :func:`optix.io.row_tiles`.

    Returns
    -------
    median, mad : (*group, *spatial) Tensor
        Lower median (rank ``(N + 1) // 2``, as ``torch.median``) and the
        lower median of ``|Y - median|``, unscaled (multiply by 1.4826 for
        a Gaussian standard deviation).
    """
    n = _nsamples(Y, ndim, layout)
    if n < 1:
        raise ValueError("No data to reduce")
    k = (n + 1) // 2
    kw.setdefault("overhead", 3.0)

    def fn(t: Tensor) -> tuple[Tensor, Tensor]:
        med = t.kthvalue(k, -1).values
        mad = (t - med.unsqueeze(-1)).abs().kthvalue(k, -1).values
        return med, mad

    return tile_reduce(Y, fn, ndim=ndim, layout=layout, **kw)
