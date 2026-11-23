"""Overlapping tiles: a consistency term on the flat field.

When consecutive tiles of a mosaic overlap, the same piece of the scene is
seen at two positions of the field of view. Its signal u is the same in
both, so for a voxel p of tile i and the matching voxel p − s of tile j,

    yᵢ(p) − c(p) ≈ m(p)·u,    yⱼ(p − s) − c(p − s) ≈ m(p_s)·u

and the log-ratio of the two excesses measures a difference of the log
flat field, free of u:

    d = log(yᵢ − c) − log(yⱼ − c) ≈ log m(p) − log m(p − s).

`MixtureFit` adds the Gaussian pseudo-likelihood of these log-ratios to
its bound (a composite likelihood: the mixture term treats the tiles as
independent pictures, this term restores the information carried by the
shared signal),

    −½ ∑ w·(d − log m(p) + log m(p − s))²,

over the voxels that are foreground in both tiles (both excesses more
than `threshold` standard deviations of an empty pixel), with the delta-
method weight w = 1 / (Var[yᵢ]/(yᵢ − c)² + Var[yⱼ]/(yⱼ − c)²), scaled by
`weight`. Because log m = Φ·θ, the term is a quadratic in θ that enters
the Fisher-scoring system of θ (its curvature is exact); d and w are
recomputed from the current c and v at each local step and held fixed
in the parameter step.

Conventions. Each tile has a pose: an integer position in a common
voxel frame, with one signed component per axis of the (group-free) data
other than the tile axis, in storage order (for data stored as
(tile, x, y, z), with pictures in the (y, z) plane indexed by x, a
position is (x, y, z)). Every pair i < j whose boxes intersect is used,
with s = positionⱼ − positionᵢ: voxel p of tile i shows the same scene
point as voxel p − s of tile j. A shift along a sample axis (x here)
pairs frame k of tile i with frame k − sx of tile j.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from ...basis import project
from ...io import ArrayLike, Layout

__all__ = ["TileOverlap", "overlap_slices", "tile_pairs"]

# Largest number of voxels read at once from one tile of a pair
_READ = 1 << 22


class TileOverlap:
    """Integer positions of overlapping tiles in a common voxel frame.

    One pose per tile: tile t's voxel q is the scene voxel
    `positions[t] + q`. Every pair of tiles whose boxes intersect is
    used (2D grids, snake scans); tiles that do not touch are skipped.
    Give exactly one of `positions`, `steps` or `step`.

    Parameters
    ----------
    positions : (N, D) array-like of int
        Absolute position of each tile. D is the number of data axes
        other than the group axes and the tile axis, in storage order:
        for (tile, x, y, z) data, a position is (x, y, z).
    steps : (N - 1, D) array-like of int
        Steps between consecutive tiles (positions are their cumulative
        sum, the first tile at 0).
    step : (D,) array-like of int
        One step between every two consecutive tiles, e.g. (0, 64, 0).
    axis : int or str
        Tile axis of the data (position, or label of a labelled layout).
        It must be a sample axis: the tiles share the flat field.
        Default 0.
    weight : float
        Scale of the overlap term relative to the mixture likelihood.
        Default 1.

    Notes
    -----
    Only integer translations are supported (no resampling). Internally
    each tile has one pose; a later version may accept a `(N, D, D + 1)`
    affine pose per tile in place of its translation.
    """

    def __init__(
        self,
        *,
        positions: object = None,
        steps: object = None,
        step: object = None,
        axis: int | str = 0,
        weight: float = 1.0,
    ) -> None:
        given = [x is not None for x in (positions, steps, step)]
        if sum(given) != 1:
            raise ValueError("give exactly one of positions, steps or step")
        self.positions = None if positions is None else _ints(positions, 2)
        self.steps = None if steps is None else _ints(steps, 2)
        self.step = None if step is None else _ints(step, 1)
        self.axis = axis
        self.weight = float(weight)
        if not self.weight >= 0:
            raise ValueError("overlap weight must be non-negative")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TileOverlap):
            return NotImplemented

        def same(a: object, b: object) -> bool:
            if a is None or b is None:
                return a is None and b is None
            return np.array_equal(a, b)

        return (
            same(self.positions, other.positions)
            and same(self.steps, other.steps)
            and same(self.step, other.step)
            and self.axis == other.axis
            and self.weight == other.weight
        )

    __hash__ = None

    def __repr__(self) -> str:
        for name in ("positions", "steps", "step"):
            v = getattr(self, name)
            if v is not None:
                return (
                    f"TileOverlap({name}={v.tolist()}, axis={self.axis!r}, "
                    f"weight={self.weight})"
                )
        return "TileOverlap()"

    def poses(self, ntiles: int, ndim: int) -> np.ndarray:
        """Absolute `(ntiles, ndim)` integer positions of the tiles."""
        if self.positions is not None:
            pos = self.positions
            if len(pos) != ntiles:
                raise ValueError(
                    f"{len(pos)} tile positions for {ntiles} tiles"
                )
        else:
            steps = self.steps
            if steps is None:
                n = len(self.step)
                steps = np.broadcast_to(self.step, (ntiles - 1, n))
            if len(steps) != ntiles - 1:
                raise ValueError(
                    f"{len(steps)} steps for {ntiles} tiles (need "
                    f"{ntiles - 1})"
                )
            pos = np.concatenate(
                [np.zeros((1, steps.shape[1]), np.int64), np.cumsum(steps, 0)]
            )
        if pos.shape[1] != ndim:
            raise ValueError(
                f"tile positions need {ndim} components (one per non-tile "
                f"axis), got {pos.shape[1]}"
            )
        return pos


def _ints(x: object, ndim: int) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != ndim:
        raise ValueError(f"expected a {ndim}-D array of ints, got {a.shape}")
    if a.size and not np.issubdtype(a.dtype, np.integer):
        if not (np.all(np.isfinite(a)) and np.all(a == np.round(a))):
            raise ValueError("tile positions must be integers (no resampling)")
    return a.astype(np.int64)


def tile_pairs(
    positions: np.ndarray, shape: Sequence[int]
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Every pair i < j of tiles of `shape` whose boxes intersect.

    Returns `(i, j, s)` with s = positions[j] − positions[i]: voxel p of
    tile i is voxel p − s of tile j.
    """
    out = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            s = tuple(int(v) for v in positions[j] - positions[i])
            if overlap_slices(s, shape) is not None:
                out.append((i, j, s))
    return out


def overlap_slices(
    shift: Sequence[int], shape: Sequence[int]
) -> tuple[tuple[slice, ...], tuple[slice, ...]] | None:
    """Overlap of two grids of `shape` translated by `shift`.

    Voxel p of grid i matches voxel p − s of grid j. Returns the slices
    of the overlap in grid i and in grid j (same lengths), or None when
    the grids do not overlap.
    """
    a, b = [], []
    for s, n in zip(shift, shape, strict=True):
        lo, hi = max(0, s), min(n, n + s)
        if hi <= lo:
            return None
        a.append(slice(lo, hi))
        b.append(slice(lo - s, hi - s))
    return tuple(a), tuple(b)


def _tensor(x: object, backend: dict) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(**backend)
    return torch.as_tensor(np.asarray(x)).to(**backend)


@dataclass
class OverlapStats:
    """Per spatial shift, the sums over matched voxels of w, w·d, w·d².

    Each entry maps a spatial shift (sy, sx) to the slices of the overlap
    in tile i's and tile j's pixel grids and three maps over tile i's
    slice.
    """

    terms: dict = field(default_factory=dict)
    count: int = 0

    def add(self, key: tuple, si: tuple, sj: tuple, maps: Tensor) -> None:
        if key in self.terms:
            self.terms[key][2].add_(maps)
        else:
            self.terms[key] = [si, sj, maps]

    def _delta(self, logm: Tensor, si: tuple, sj: tuple) -> Tensor:
        return logm[si] - logm[sj]

    def bound(self, logm: Tensor) -> float:
        """−½ ∑ w·(d − Δ log m)²."""
        total = 0.0
        for si, sj, (sw, swd, swdd) in self.terms.values():
            delta = self._delta(logm, si, sj).to(sw)
            q = sw * delta.square() - 2 * swd * delta + swdd
            total -= 0.5 * q.sum().item()
        return total

    def system(
        self, logm: Tensor, bases: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Curvature `(k, k)` and score `(k,)` of the term in θ."""
        by, bx = bases
        k = by.shape[1] * bx.shape[1]
        H = by.new_zeros((k, k))
        g = by.new_zeros(k)
        for si, sj, (sw, swd, _) in self.terms.values():
            sw, swd = sw.to(by), swd.to(by)
            e = swd - sw * self._delta(logm, si, sj)
            yi, xi = by[si[0]], bx[si[1]]
            yj, xj = by[sj[0]], bx[sj[1]]
            g += project(e, [yi, xi]).flatten()
            g -= project(e, [yj, xj]).flatten()
            ii = _cross(sw, yi, xi, yi, xi)
            jj = _cross(sw, yj, xj, yj, xj)
            ij = _cross(sw, yi, xi, yj, xj)
            H += ii + jj - ij - ij.T
        return H, g


def _cross(
    w: Tensor, ay: Tensor, ax: Tensor, by: Tensor, bx: Tensor
) -> Tensor:
    """∑_p w(p)·(a_y ⊗ a_x)(p)·(b_y ⊗ b_x)(p)ᵀ, flattened (ky·kx)²."""
    t = torch.einsum("yx,xj,xl->yjl", w, ax, bx)
    t = torch.einsum("yjl,yi,yk->ijkl", t, ay, by)
    ky, kx = ay.shape[1], ax.shape[1]
    return t.reshape(ky * kx, ky * kx)


@dataclass
class Overlap:
    """Tile overlap of one group, resolved against its layout."""

    axis: int  # tile axis of the group-free data
    pairs: list  # [(i, j, shift)] of intersecting tiles, from the poses
    weight: float
    threshold: float

    @classmethod
    def build(
        cls,
        spec: "TileOverlap | object",
        raw_shape: tuple[int, ...],
        layout: Layout,
        threshold: float,
    ) -> "Overlap":
        """Resolve `spec` against data of `raw_shape` (with groups)."""
        if not isinstance(spec, TileOverlap):
            spec = TileOverlap(positions=spec)
        ndim = len(raw_shape)
        group, sample, _ = layout.positions(ndim)
        axis = spec.axis
        if isinstance(axis, str):
            if layout.axes is None or axis not in layout.axes:
                raise ValueError(f"overlap axis {axis!r} is not a data axis")
            axis = layout.axes.index(axis)
        axis = int(axis) % ndim
        if axis not in sample:
            raise ValueError(
                f"overlap axis {spec.axis!r} must be a sample axis (tiles "
                f"share the flat field)"
            )
        keep = [a for a in range(ndim) if a not in group]
        sub_axis = keep.index(axis)
        tile = tuple(raw_shape[a] for a in keep if a != axis)
        pos = spec.poses(raw_shape[axis], len(tile))
        pairs = tile_pairs(pos, tile)
        return cls(sub_axis, pairs, float(spec.weight), float(threshold))

    def stats(
        self,
        source: ArrayLike,
        layout: Layout,
        C: Tensor,
        V: Tensor,
        gain: float,
        full: bool,
        backend: dict,
    ) -> OverlapStats:
        """Accumulate the matched log-ratios of every pair."""
        shape = tuple(int(n) for n in source.shape)
        ndim = len(shape)
        _, _, spatial = layout.positions(ndim)
        other = [a for a in range(ndim) if a != self.axis]
        shape_o = tuple(shape[a] for a in other)
        # positions of the spatial axes among the non-tile axes
        sp = [other.index(a) for a in spatial]
        rest = [a for a in range(len(other)) if a not in sp]
        out = OverlapStats()
        if not self.weight > 0:
            return out
        sd = V.sqrt()
        for i, j, shift in self.pairs:
            sl = overlap_slices(shift, shape_o)
            if sl is None:
                continue
            si, sj = sl
            key = tuple(shift[a] for a in sp)
            pi = tuple(si[a] for a in sp)
            pj = tuple(sj[a] for a in sp)
            Ci, Cj = C[pi], C[pj]
            Vi, Vj = V[pi], V[pj]
            ti, tj = self.threshold * sd[pi], self.threshold * sd[pj]
            acc = torch.zeros(
                (3, *Ci.shape), dtype=torch.float64, device=C.device
            )
            # read in blocks along the first non-spatial, non-tile axis
            npix = math.prod(Ci.shape)
            if rest:
                lead = rest[0]
                n = si[lead].stop - si[lead].start
                inner = math.prod(si[a].stop - si[a].start for a in rest[1:])
                step = max(1, _READ // max(npix * inner, 1))
                blocks = [(b, min(b + step, n)) for b in range(0, n, step)]
            else:
                lead, blocks = None, [(0, 1)]
            for b0, b1 in blocks:
                ii, jj = list(si), list(sj)
                if lead is not None:
                    ii[lead] = slice(si[lead].start + b0, si[lead].start + b1)
                    jj[lead] = slice(sj[lead].start + b0, sj[lead].start + b1)
                A = self._read(source, i, ii, sp, rest, backend)
                B = self._read(source, j, jj, sp, rest, backend)
                ri, rj = A - Ci, B - Cj
                ok = (ri > ti) & (rj > tj)
                ri, rj = ri.clamp_min(1e-12), rj.clamp_min(1e-12)
                vi, vj = Vi, Vj
                if full:
                    vi, vj = Vi + gain * ri, Vj + gain * rj
                var = vi / ri.square() + vj / rj.square()
                w = torch.where(ok, self.weight / var, 0)
                d = torch.where(ok, ri.log() - rj.log(), 0)
                wd = w * d
                acc[0] += w.sum(0).double()
                acc[1] += wd.sum(0).double()
                acc[2] += (wd * d).sum(0).double()
                out.count += int(ok.sum())
            out.add(key, pi, pj, acc)
        return out

    def _read(
        self,
        source: ArrayLike,
        tile: int,
        index: list,
        sp: list,
        rest: list,
        backend: dict,
    ) -> Tensor:
        full = list(index)
        full.insert(self.axis, tile)
        x = _tensor(source[tuple(full)], backend)
        x = x.permute(*rest, *sp)
        return x.reshape(-1, *x.shape[len(rest) :])
