"""Overlapping tiles: poses, pairs, and the log-ratio initialiser.

When consecutive tiles of a mosaic overlap, the same piece of the scene
is seen at several positions of the field of view. The model of
`_model.MixtureFit` treats each scene voxel as one latent signal u
(and one class) shared by every tile that sees it (see `_scene` for the
geometry); this module holds the poses and pairs of the tiles, and the
statistics used to **initialise** the flat field and the tile gains.

Initialiser. For a voxel p of tile t and the matching voxel p − s of
tile t′, both tissue, the log-ratio of the excesses

    d = log(yₜ − c) − log(y_{t′} − c) ≈ log f(p) − log f(p − s) + aₜ − a_{t′}

is free of u. `OverlapStats` holds the delta-method weighted sums of d
over every pair (w = 1 / (Var yₜ/(yₜ − c)² + Var y_{t′}/(y_{t′} − c)²), voxels
more than `threshold` background standard deviations above c in both
tiles), from which the initial θ is fitted (with the moment term,
`fit_modulation(extra=…)`) and the initial log gains a are profiled
(weighted Laplacian of the overlap graph, ∑ a = 0 per connected group).
These statistics do not enter the bound.

Conventions. Each tile has a pose: an integer position in a common
voxel frame, with one signed component per axis of the (group-free) data
other than the tile axis, in storage order (for data stored as
(tile, x, y, z), with pictures in the (y, z) plane indexed by x, a
position is (x, y, z)). Tile t's voxel q is the scene voxel
position_t + q; a shift along a sample axis (x here) pairs frame n of
tile t with frame n − sₓ of tile t′.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from ...basis import project
from ...io import ArrayLike, Layout
from ...stats._base import state_dtype
from ._scene import SceneIndex

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
    fit_gain : bool
        Also fit one gain g_t per tile: tile t sees the scene through
        b_t = g_t·f, so that a uniform per-tile fade (bleaching, lamp
        drift) is not mistaken for a ramp of the flat field. The gains
        have a geometric mean of one over the tiles that overlap another
        (the common scale of the gains is that of the class means);
        a tile that overlaps no other keeps g = 1. Fitted values:
        `GammaMixture.tile_gain_`. Default False.
    coarse : int, optional
        Start from a fit on the data averaged over `coarse` × `coarse`
        pixels (positions scaled): its flat field and gains initialise
        the full fit. Default None (no coarse fit).

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
        fit_gain: bool = False,
        coarse: int | None = None,
    ) -> None:
        given = [x is not None for x in (positions, steps, step)]
        if sum(given) != 1:
            raise ValueError("give exactly one of positions, steps or step")
        self.positions = None if positions is None else _ints(positions, 2)
        self.steps = None if steps is None else _ints(steps, 2)
        self.step = None if step is None else _ints(step, 1)
        self.axis = axis
        self.fit_gain = bool(fit_gain)
        self.coarse = None if coarse is None else int(coarse)
        if self.coarse is not None and self.coarse < 2:
            raise ValueError("coarse must be an integer factor of at least 2")

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
            and self.fit_gain == other.fit_gain
            and self.coarse == other.coarse
        )

    __hash__ = None

    def __repr__(self) -> str:
        for name in ("positions", "steps", "step"):
            v = getattr(self, name)
            if v is not None:
                return (
                    f"TileOverlap({name}={v.tolist()}, axis={self.axis!r}"
                    + (", fit_gain=True" if self.fit_gain else "")
                    + (f", coarse={self.coarse}" if self.coarse else "")
                    + ")"
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
    pos = np.asarray(positions, dtype=np.int64)
    n = len(pos)
    if n == 0:
        return []
    shape = np.asarray(shape, dtype=np.int64)
    if pos.ndim != 2 or pos.shape[1] != len(shape):
        raise ValueError(
            f"tile positions of shape {pos.shape} for tiles of shape "
            f"{tuple(shape.tolist())}"
        )
    if np.any(shape <= 0):
        return []
    # bucket the tiles on a grid of cells of the tile size: boxes that
    # intersect lie in the same or in adjacent cells
    cell = np.floor_divide(pos, shape)
    buckets: dict[tuple, list[int]] = {}
    for t, c in enumerate(map(tuple, cell.tolist())):
        buckets.setdefault(c, []).append(t)
    near = np.stack(
        np.meshgrid(*[[-1, 0, 1]] * len(shape), indexing="ij"), -1
    ).reshape(-1, len(shape))
    out = []
    for i in range(n):
        ci = cell[i]
        for off in near:
            for j in buckets.get(tuple((ci + off).tolist()), ()):
                if j <= i:
                    continue
                s = pos[j] - pos[i]
                if np.all(np.abs(s) < shape):
                    out.append((i, j, tuple(int(v) for v in s)))
    out.sort()
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

    Each entry maps a spatial shift (sy, sx) (or, with per-tile gains, a
    pair (i, j)) to the slices of the overlap in tile i's and tile j's
    pixel grids, three maps over tile i's slice, and the pair's tiles.
    """

    terms: dict = field(default_factory=dict)
    count: int = 0
    ntiles: int = 0
    fit_gain: bool = False
    dtype: torch.dtype = torch.float64
    _A: Tensor | None = field(default=None, repr=False)

    def add(
        self,
        key: tuple,
        si: tuple,
        sj: tuple,
        maps: Tensor,
        pair: tuple[int, int] = (0, 0),
    ) -> None:
        if key in self.terms:
            self.terms[key][2].add_(maps)
        else:
            self.terms[key] = [si, sj, maps, *pair]
        self._A = None

    def _delta(self, logm: Tensor, si: tuple, sj: tuple) -> Tensor:
        return logm[si] - logm[sj]

    @property
    def gains(self) -> bool:
        """Whether the term profiles per-tile gains."""
        return self.fit_gain and self.ntiles > 0 and bool(self.terms)

    def _graph(self) -> Tensor:
        """Laplacian of the overlap graph plus a pinning block per component.

        Edges are the pairs with a positive total weight. The added
        blocks make the matrix invertible and, since every right-hand
        side is orthogonal to the component indicators, the solution
        sums to zero over each component (and is 0 on isolated tiles).
        """
        if self._A is not None:
            return self._A
        n = self.ntiles
        L = torch.zeros((n, n), dtype=self.dtype)
        parent = list(range(n))

        def root(t: int) -> int:
            while parent[t] != t:
                parent[t] = parent[parent[t]]
                t = parent[t]
            return t

        for _, _, maps, i, j in self.terms.values():
            w = float(maps[0].sum())
            if not w > 0:
                continue
            L[i, i] += w
            L[j, j] += w
            L[i, j] -= w
            L[j, i] -= w
            parent[root(i)] = root(j)
        roots = torch.tensor([root(t) for t in range(n)])
        P = (roots[:, None] == roots[None]).to(self.dtype)
        # pinning blocks scaled like L, for conditioning
        scale = max(float(L.diagonal().max()), 1.0)
        self._A = L + scale * P / P.sum(1, keepdim=True)
        return self._A

    def log_gain(self, logm: Tensor) -> Tensor:
        """Profiled log gains a*(θ), `(ntiles,)` (zeros without gains)."""
        a = torch.zeros(self.ntiles, dtype=self.dtype)
        if not self.gains:
            return a
        b = torch.zeros_like(a)
        for si, sj, (sw, swd, _), i, j in self.terms.values():
            delta = self._delta(logm, si, sj).to(sw)
            r = float((swd - sw * delta).sum())
            b[i] += r
            b[j] -= r
        return torch.linalg.solve(self._graph(), b)

    def bound(self, logm: Tensor) -> float:
        """−½ ∑ w·(d − Δ log f − Δ log g)², at the profiled gains."""
        gains = self.gains
        a = self.log_gain(logm).tolist()
        total = 0.0
        for si, sj, (sw, swd, swdd), i, j in self.terms.values():
            delta = self._delta(logm, si, sj).to(sw)
            if gains:
                delta = delta + (a[i] - a[j])
            q = sw * delta.square() - 2 * swd * delta + swdd
            total -= 0.5 * q.sum().item()
        return total

    def system(
        self, logm: Tensor, bases: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Curvature `(k, k)` and score `(k,)` of the term in θ.

        With per-tile gains, of the term profiled over them: the Schur
        complement H_θθ − U·L⁺·Uᵀ of the gain block, and the score at
        a*(θ) (where the score in a vanishes).
        """
        by, bx = bases
        k = by.shape[1] * bx.shape[1]
        H = by.new_zeros((k, k))
        g = by.new_zeros(k)
        gains = self.gains
        a = self.log_gain(logm).tolist()
        U = torch.zeros((k, self.ntiles), dtype=self.dtype)
        for si, sj, (sw, swd, _), i, j in self.terms.values():
            sw, swd = sw.to(by), swd.to(by)
            delta = self._delta(logm, si, sj)
            if gains:
                delta = delta + (a[i] - a[j])
            e = swd - sw * delta
            yi, xi = by[si[0]], bx[si[1]]
            yj, xj = by[sj[0]], bx[sj[1]]
            g += project(e, [yi, xi]).flatten()
            g -= project(e, [yj, xj]).flatten()
            ii = _cross(sw, yi, xi, yi, xi)
            jj = _cross(sw, yj, xj, yj, xj)
            ij = _cross(sw, yi, xi, yj, xj)
            H += ii + jj - ij - ij.T
            if gains:
                u = project(sw, [yi, xi]) - project(sw, [yj, xj])
                u = u.flatten().to(U)
                U[:, i] += u
                U[:, j] -= u
        if gains:
            H -= (U @ torch.linalg.solve(self._graph(), U.T)).to(H)
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
    threshold: float
    ntiles: int = 0
    fit_gain: bool = False
    positions: np.ndarray | None = None  # (ntiles, D) poses
    coarse: int | None = None  # binning factor of a coarse first fit
    scene: "SceneIndex | None" = None  # cells seen by several tiles

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
        if raw_shape[axis] == 0:
            raise ValueError("the overlap term needs at least one tile")
        pos = spec.poses(raw_shape[axis], len(tile))
        pairs = tile_pairs(pos, tile)
        return cls(
            sub_axis,
            pairs,
            float(threshold),
            ntiles=int(raw_shape[axis]),
            fit_gain=spec.fit_gain,
            positions=pos,
            coarse=spec.coarse,
            scene=SceneIndex(pos, tile),
        )

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
        dtype = state_dtype(C.device)
        out = OverlapStats(
            ntiles=self.ntiles, fit_gain=self.fit_gain, dtype=dtype
        )
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
            acc = torch.zeros((3, *Ci.shape), dtype=dtype, device=C.device)
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
                w = torch.where(ok, 1 / var, 0)
                d = torch.where(ok, ri.log() - rj.log(), 0)
                wd = w * d
                acc[0] += w.sum(0, dtype=dtype)
                acc[1] += wd.sum(0, dtype=dtype)
                acc[2] += (wd * d).sum(0, dtype=dtype)
                out.count += int(ok.sum())
            if self.fit_gain:
                key = (i, j)
            out.add(key, pi, pj, acc, (i, j))
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
