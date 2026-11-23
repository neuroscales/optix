"""Tiled acquisition: overlapping tiles with per-tile gain and offset."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from ..estimates import TileIntensityEstimate
from ._random import generator

__all__ = ["simulate_tiles"]


def _grid_axis(n: int, count: int, overlap: float) -> tuple[int, list[int]]:
    """Tile side and first-pixel positions of `count` tiles over `n` pixels."""
    if count == 1:
        return n, [0]
    side = math.ceil(n / (count - (count - 1) * overlap))
    side = min(side, n)
    step = (n - side) / (count - 1)
    return side, [round(k * step) for k in range(count)]


def simulate_tiles(
    x: Tensor,
    grid: Sequence[int],
    *,
    overlap: float = 0.1,
    tile_gain: Tensor | Sequence[float] | None = None,
    tile_offset: Tensor | Sequence[float] | None = None,
    spread: tuple[float, float] = (0.2, 0.1),
    noise: float = 0.0,
    seed: int | None = None,
) -> tuple[list[Tensor], Tensor, TileIntensityEstimate]:
    """Cut an image into a grid of overlapping tiles with intensity changes.

    Tile i, a crop of `x`, is recorded as

        yᵢ = gᵢ x[crop i] + oᵢ + ε,  ε ~ N(0, noise²),

    so that the transform that maps every tile back onto `x` is
    aᵢ = 1/gᵢ, cᵢ = −oᵢ/gᵢ.

    Parameters
    ----------
    x : Tensor
        `(*spatial)` image of the whole field.
    grid : sequence of int
        Number of tiles along each spatial axis.
    overlap : float
        Fraction of the tile side shared by neighbouring tiles (at least;
        tiles are spread to cover `x` exactly).
    tile_gain, tile_offset : (T,) Tensor or sequence of float, optional
        Per-tile gᵢ (dimensionless, not the camera gain) and oᵢ (in the
        units of `x`), tiles in C order of the grid. Default: random,
        log gᵢ ~ N(0, spread[0]²) and oᵢ ~ N(0, (spread[1] · mean(x))²).
    spread : (float, float)
        Spreads of the random gains and offsets.
    noise : float
        Standard deviation of additive Gaussian noise.
    seed : int, optional

    Returns
    -------
    tiles : list of Tensor
        T tiles (all of the same shape).
    positions : (T, D) Tensor
        Position of the first pixel of each tile in `x` (int64).
    truth : TileIntensityEstimate
        `scale = 1/g`, `shift = −o/g`, which map each tile onto `x`.
    """
    grid = tuple(int(g) for g in grid)
    if len(grid) != x.ndim:
        raise ValueError(f"grid has {len(grid)} entries for {x.ndim}D x")
    gen = generator(seed, torch.device("cpu"))
    axes = [
        _grid_axis(n, c, overlap) for n, c in zip(x.shape, grid, strict=True)
    ]
    side = [a[0] for a in axes]
    starts = torch.cartesian_prod(
        *[torch.tensor(a[1], dtype=torch.int64) for a in axes]
    ).reshape(-1, x.ndim)
    ntiles = starts.shape[0]
    dt = torch.float64
    gain, offset = tile_gain, tile_offset
    if gain is None:
        gain = (spread[0] * torch.randn(ntiles, generator=gen, dtype=dt)).exp()
    if offset is None:
        offset = (
            spread[1]
            * float(x.double().mean())
            * torch.randn(ntiles, generator=gen, dtype=dt)
        )
    gain = torch.as_tensor(gain, dtype=dt)
    offset = torch.as_tensor(offset, dtype=dt)
    if gain.shape != (ntiles,) or offset.shape != (ntiles,):
        raise ValueError(
            f"tile_gain and tile_offset must have {ntiles} entries"
        )
    tiles = []
    for i in range(ntiles):
        crop = tuple(
            slice(int(p), int(p) + s)
            for p, s in zip(starts[i], side, strict=True)
        )
        y = x[crop] * float(gain[i]) + float(offset[i])
        if noise > 0:
            eps = torch.randn(y.shape, generator=gen, dtype=dt)
            y = y + noise * eps.to(y)
        tiles.append(y)
    truth = TileIntensityEstimate(
        scale=(1 / gain).to(x.dtype),
        shift=(-offset / gain).to(x.dtype),
        meta={"simulated": True},
    )
    return tiles, starts, truth
