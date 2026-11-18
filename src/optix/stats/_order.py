"""Exact per-pixel order statistics over row tiles."""

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from ..io import ArrayLike, row_tiles

__all__ = ["median_mad", "order_statistics", "tile_reduce"]

_MAX_KTHVALUE = 4


def tile_reduce(
    Y: ArrayLike,
    fn: Callable[[Tensor], Tensor | tuple[Tensor, ...]],
    *,
    ndim: int = 2,
    overhead: float = 2.0,
    **chunk_options: object,
) -> Tensor | tuple[Tensor, ...]:
    """Apply ``fn`` to row tiles holding all frames and stitch the maps.

    Parameters
    ----------
    Y : (*batch, frames, *spatial) array-like
        Data.
    fn : callable
        Takes a ``(rows, *rest, N)`` tile (``N`` = all frames of all
        batches, last) and returns one or several ``(rows, *rest)`` maps.
    ndim : int
        Number of trailing spatial dims. Tiles are cut along the first
        spatial axis.
    overhead : float
        Chunk-sized temporaries held by ``fn`` (see
        :func:`optix.io.chunk_length`).
    **chunk_options
        Forwarded to :func:`optix.io.row_tiles`.

    Returns
    -------
    Tensor or tuple of Tensor
        Full ``(*spatial)`` maps, on the working device.
    """
    tiles = row_tiles(
        Y, ndim=ndim, axis=-ndim, overhead=overhead, **chunk_options
    )
    lead = len(tiles.shape) - ndim
    spatial = tiles.shape[lead:]
    outs: list[Tensor] | None = None
    single = False
    with torch.no_grad():
        for index, data in tiles:
            tile = data.reshape(-1, *data.shape[lead:]).movedim(0, -1)
            result = fn(tile)
            if isinstance(result, Tensor):
                single, result = True, (result,)
            if outs is None:
                outs = [
                    torch.empty(spatial, dtype=r.dtype, device=r.device)
                    for r in result
                ]
            for out, r in zip(outs, result, strict=True):
                out[index[lead:]] = r
    if outs is None:
        raise ValueError("No data to reduce")
    return outs[0] if single else tuple(outs)


def _nsamples(Y: ArrayLike, ndim: int) -> int:
    shape = tuple(Y.shape)
    return math.prod(shape[: len(shape) - ndim])


def order_statistics(
    Y: ArrayLike,
    k: int | Sequence[int],
    *,
    ndim: int = 2,
    **kw: object,
) -> Tensor:
    """Per-pixel k-th smallest value(s) over all frames of all batches.

    Parameters
    ----------
    Y : (*batch, frames, *spatial) array-like
        Data.
    k : int or sequence of int
        1-based rank(s), as in ``torch.kthvalue``.
    ndim : int
        Number of trailing spatial dims.
    **kw
        Forwarded to :func:`tile_reduce` / :func:`optix.io.row_tiles`.

    Returns
    -------
    (*spatial) or (K, *spatial) Tensor

    Notes
    -----
    Up to four ranks use repeated ``torch.kthvalue``; more ranks use one
    ``sort`` (whose int64 indices make it heavier: ``overhead`` defaults to
    3 then). ``torch.quantile`` is not used: it rejects large inputs.
    NaNs give undefined results.
    """
    n = _nsamples(Y, ndim)
    ks = [k] if isinstance(k, int) else [int(i) for i in k]
    for i in ks:
        if not 1 <= i <= n:
            raise ValueError(f"k={i} out of range [1, {n}]")
    if isinstance(k, int):
        return tile_reduce(
            Y, lambda t: t.kthvalue(k, -1).values, ndim=ndim, **kw
        )
    if len(ks) <= _MAX_KTHVALUE:

        def fn(t: Tensor) -> tuple[Tensor, ...]:
            return tuple(t.kthvalue(i, -1).values for i in ks)

    else:
        kw.setdefault("overhead", 3.0)

        def fn(t: Tensor) -> tuple[Tensor, ...]:
            s = t.sort(-1).values
            return tuple(s[..., i - 1] for i in ks)

    return torch.stack(tile_reduce(Y, fn, ndim=ndim, **kw))


def median_mad(
    Y: ArrayLike, *, ndim: int = 2, **kw: object
) -> tuple[Tensor, Tensor]:
    """Per-pixel median and median absolute deviation, in one pass.

    Parameters
    ----------
    Y : (*batch, frames, *spatial) array-like
        Data.
    ndim : int
        Number of trailing spatial dims.
    **kw
        Forwarded to :func:`tile_reduce` / :func:`optix.io.row_tiles`.

    Returns
    -------
    median, mad : (*spatial) Tensor
        Lower median (rank ``(N + 1) // 2``, as ``torch.median``) and the
        lower median of ``|Y - median|``, unscaled (multiply by 1.4826 for
        a Gaussian standard deviation).
    """
    n = _nsamples(Y, ndim)
    if n < 1:
        raise ValueError("No data to reduce")
    k = (n + 1) // 2
    kw.setdefault("overhead", 3.0)

    def fn(t: Tensor) -> tuple[Tensor, Tensor]:
        med = t.kthvalue(k, -1).values
        mad = (t - med.unsqueeze(-1)).abs().kthvalue(k, -1).values
        return med, mad

    return tile_reduce(Y, fn, ndim=ndim, **kw)
