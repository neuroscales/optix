"""Pure-Python chunk planning (no torch)."""

import itertools
from collections.abc import Sequence
from typing import NamedTuple

Index = tuple[int | slice, ...]


class Plan(NamedTuple):
    """One chunk of a blocked iteration.

    Attributes
    ----------
    read : Index
        Region of the source to load (core + halo clipped to the array).
    index : Index
        Region of the core in the source / output.
    inner : Index
        Crop of the loaded (and padded) data that gives the core. Only
        covers the blocked dims (leading dims are integer-indexed away).
    pad : tuple of (int, int)
        Padding to add before / after each blocked dim after loading.
    """

    read: Index
    index: Index
    inner: Index
    pad: tuple[tuple[int, int], ...]


def _segments(
    n: int, b: int, h: int, pad: bool
) -> list[tuple[slice, slice, slice, tuple[int, int]]]:
    """Blocks of one dim: (read, index, inner, (before, after))."""
    out = []
    for start in range(0, n, b):
        stop = min(start + b, n)
        rstart, rstop = max(start - h, 0), min(stop + h, n)
        if pad:
            before = h - (start - rstart)
            after = (b + 2 * h) - (rstop - rstart) - before
            offset = h
        else:
            before = after = 0
            offset = start - rstart
        out.append(
            (
                slice(rstart, rstop),
                slice(start, stop),
                slice(offset, offset + stop - start),
                (before, after),
            )
        )
    return out


def plan_grid(
    shape: Sequence[int],
    specs: Sequence[tuple[int, int] | None],
    pad: bool = False,
    order: Sequence[int] | None = None,
) -> list[Plan]:
    """Plan a blocked iteration with a spec per axis.

    Parameters
    ----------
    shape : sequence of int
        Shape of the source.
    specs : sequence of (int, int) or None
        Per axis: None to iterate it one integer index at a time, or
        `(block, halo)` with `block ≥ 1` and `halo ≥ 0`.
    pad : bool
        See `plan_blocks`.
    order : sequence of int, optional
        Nesting of the iteration, outermost axis first (default: C order).

    Returns
    -------
    list of Plan
        `read` and `index` are full-length, in storage order; `inner`
        and `pad` cover the blocked axes, in storage order. The `index`
        regions partition the array exactly.
    """
    shape = tuple(int(s) for s in shape)
    n = len(shape)
    if len(specs) != n:
        raise ValueError("Need one spec per axis")
    order = tuple(range(n)) if order is None else tuple(order)
    if sorted(order) != list(range(n)):
        raise ValueError(f"order {order} is not a permutation")
    segs: dict[int, list] = {}
    blocked = []
    for axis, (size, spec) in enumerate(zip(shape, specs, strict=True)):
        if spec is None:
            segs[axis] = list(range(size))
            continue
        b, h = spec
        if b < 1 or h < 0:
            raise ValueError("Need block >= 1 and halo >= 0")
        segs[axis] = _segments(size, b, h, pad)
        blocked.append(axis)
    plans = []
    for combo in itertools.product(*(segs[a] for a in order)):
        per = dict(zip(order, combo, strict=True))

        def part(axis: int, k: int, per: dict = per) -> int | slice:
            return per[axis] if specs[axis] is None else per[axis][k]

        plans.append(
            Plan(
                read=tuple(part(a, 0) for a in range(n)),
                index=tuple(part(a, 1) for a in range(n)),
                inner=tuple(per[a][2] for a in blocked),
                pad=tuple(per[a][3] for a in blocked),
            )
        )
    return plans


def plan_blocks(
    shape: Sequence[int],
    lead: int,
    block: Sequence[int],
    halo: Sequence[int],
    pad: bool,
) -> list[Plan]:
    """Plan a blocked iteration over an array.

    Parameters
    ----------
    shape : sequence of int
        Shape of the source.
    lead : int
        Number of leading dims iterated one integer index at a time.
    block : sequence of int
        Core block size along each of the remaining dims (>= 1).
    halo : sequence of int
        Halo along each of the remaining dims (>= 0).
    pad : bool
        If True, every loaded block is padded to exactly
        `block + 2 * halo` (boundary halos and short trailing cores). If
        False, halos are clipped at the array boundary.

    Returns
    -------
    list of Plan
        In C order. The `index` regions partition the array exactly.
    """
    shape = tuple(int(s) for s in shape)
    nblk = len(shape) - lead
    if lead < 0 or len(block) != nblk or len(halo) != nblk:
        raise ValueError("block and halo must cover the non-leading dims")
    if any(b < 1 for b in block) or any(h < 0 for h in halo):
        raise ValueError("Need block >= 1 and halo >= 0")
    specs = [None] * lead + list(zip(block, halo, strict=True))
    return plan_grid(shape, specs, pad)


def read_amplification(regions: Sequence[slice], chunk: int, n: int) -> float:
    """Elements decompressed per element of the array along one axis.

    Parameters
    ----------
    regions : sequence of slice
        Distinct regions read along the axis (one per pass over the other
        dims).
    chunk : int
        Storage chunk length along the axis.
    n : int
        Length of the axis.

    Returns
    -------
    float
        `sum(chunks touched * chunk) / n` (1 when perfectly aligned).
    """
    if n == 0 or chunk <= 0:
        return 1.0
    touched = 0
    for region in regions:
        first = region.start // chunk
        last = -(-region.stop // chunk)
        touched += last - first
    return touched * min(chunk, n) / n
