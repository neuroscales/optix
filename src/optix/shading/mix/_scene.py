"""Scene geometry of overlapping tiles: which tiles see each scene voxel.

Tile t's voxel q (over the D data axes other than the tile axis, in
storage order) is the scene voxel z = pos[t] + q. The scene is cut into
boxes ("cells") inside which the set S of covering tiles is constant;
only cells with |S| ≥ 2 are kept, since a voxel seen once is the
single-picture model. Everything is slices: the observations of a cell
by tile t are `tile_t[cell.local(t)]`.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._overlap import tile_pairs

__all__ = ["Cell", "SceneIndex"]


@dataclass(frozen=True)
class Cell:
    """A box of the scene seen by the same tiles (at least two)."""

    tiles: tuple[int, ...]  # S, sorted
    lo: tuple[int, ...]  # scene coordinates of the box, D axes
    hi: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(b - a for a, b in zip(self.lo, self.hi, strict=True))

    def local(self, pos: np.ndarray) -> tuple[slice, ...]:
        """The box in the voxel grid of the tile at position `pos`."""
        return tuple(
            slice(int(a - p), int(b - p))
            for a, b, p in zip(self.lo, self.hi, pos, strict=True)
        )


class SceneIndex:
    """Cells of the scene covered by two tiles or more.

    Parameters
    ----------
    positions : (T, D) array of int
        Position of each tile in the scene.
    shape : (D,) sequence of int
        Shape of a tile over the same axes.
    """

    def __init__(self, positions: np.ndarray, shape: Sequence[int]) -> None:
        self.pos = np.asarray(positions, dtype=np.int64)
        self.shape = tuple(int(n) for n in shape)
        T = len(self.pos)
        self.neighbours: list[list[int]] = [[] for _ in range(T)]
        for i, j, _ in tile_pairs(self.pos, self.shape):
            self.neighbours[i].append(j)
            self.neighbours[j].append(i)
        cells = []
        for i in range(T):
            cells.extend(self._cells_from(i))
        cells.sort(key=lambda c: (c.tiles, c.lo))
        self.cells: list[Cell] = cells
        self._of: list[list[Cell]] = [[] for _ in range(T)]
        for c in cells:
            for t in c.tiles:
                self._of[t].append(c)

    def _box(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        return self.pos[t], self.pos[t] + np.asarray(self.shape)

    def _cells_from(self, i: int) -> list[Cell]:
        """Cells whose lowest covering tile is i."""
        nb = self.neighbours[i]
        if not nb:
            return []
        lo_i, hi_i = self._box(i)
        boxes = {j: self._box(j) for j in nb}
        cuts = []
        for a in range(len(self.shape)):
            edges = {int(lo_i[a]), int(hi_i[a])}
            for lo, hi in boxes.values():
                edges.update(int(e) for e in (lo[a], hi[a]))
            edges = sorted(e for e in edges if lo_i[a] <= e <= hi_i[a])
            cuts.append(list(zip(edges[:-1], edges[1:], strict=True)))
        # cover set of every atomic sub-box
        label: dict[tuple, tuple[int, ...]] = {}
        for idx in itertools.product(*[range(len(c)) for c in cuts]):
            lo = [cuts[a][k][0] for a, k in enumerate(idx)]
            hi = [cuts[a][k][1] for a, k in enumerate(idx)]
            cover = [i] + [
                j
                for j, (blo, bhi) in boxes.items()
                if np.all(blo <= lo) and np.all(np.asarray(hi) <= bhi)
            ]
            if len(cover) >= 2 and min(cover) == i:
                label[idx] = tuple(sorted(cover))
        return _merge(label, cuts)

    def cells_of(self, t: int) -> list[Cell]:
        """The cells that tile t sees."""
        return self._of[t]

    def covered(self, t: int) -> bool:
        """Whether tile t shares at least one voxel with another tile."""
        return bool(self._of[t])

    def single_mask(self, t: int, box: Sequence[slice]) -> np.ndarray:
        """Voxels of tile t in `box` (local slices) seen by t only."""
        start = np.array([s.start for s in box], dtype=np.int64)
        stop = np.array([s.stop for s in box], dtype=np.int64)
        mask = np.ones(tuple((stop - start).tolist()), dtype=bool)
        for j in self.neighbours[t]:
            off = self.pos[j] - self.pos[t]  # j's box in t's grid
            lo = np.maximum(off, start)
            hi = np.minimum(off + np.asarray(self.shape), stop)
            if np.all(hi > lo):
                mask[_slices(lo - start, hi - start)] = False
        return mask

    def cover_count(self, t: int) -> np.ndarray:
        """Number of tiles covering each voxel of tile t (tests)."""
        count = np.ones(self.shape, dtype=np.int64)
        for j in self.neighbours[t]:
            off = self.pos[j] - self.pos[t]
            lo = np.maximum(off, 0)
            hi = np.minimum(off + np.asarray(self.shape), self.shape)
            if np.all(hi > lo):
                count[_slices(lo, hi)] += 1
        return count


def _slices(lo: np.ndarray, hi: np.ndarray) -> tuple[slice, ...]:
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))


def _merge(label: dict, cuts: list) -> list[Cell]:
    """Merge atomic sub-boxes with the same cover set into larger boxes.

    Runs of adjacent sub-boxes are merged along each axis in turn; the
    result is a partition of the labelled sub-boxes into boxes.
    """
    # box: (lo-index tuple, hi-index tuple) in units of atomic intervals
    boxes = {(k, tuple(x + 1 for x in k)): s for k, s in label.items()}
    for a in range(len(cuts)):
        merged = {}
        keyed: dict = {}
        for (lo, hi), s in boxes.items():
            other = (lo[:a] + lo[a + 1 :], hi[:a] + hi[a + 1 :], s)
            keyed.setdefault(other, []).append((lo[a], hi[a]))
        for (olo, ohi, s), runs in keyed.items():
            runs.sort()
            start, stop = runs[0]
            for b0, b1 in runs[1:] + [(None, None)]:
                if b0 == stop:
                    stop = b1
                    continue
                lo = olo[:a] + (start,) + olo[a:]
                hi = ohi[:a] + (stop,) + ohi[a:]
                merged[(lo, hi)] = s
                start, stop = b0, b1
        boxes = merged
    out = []
    for (lo, hi), s in boxes.items():
        out.append(
            Cell(
                tiles=s,
                lo=tuple(cuts[a][k][0] for a, k in enumerate(lo)),
                hi=tuple(cuts[a][k - 1][1] for a, k in enumerate(hi)),
            )
        )
    return out
