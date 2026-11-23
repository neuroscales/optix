"""Building blocks of tile intensity harmonisation.

Overlap geometry, streaming overlap statistics and the solver of
[`IntensityAdjustment`][optix.fusion.IntensityAdjustment].
"""

import itertools
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "OverlapStats",
    "bin_mean",
    "components",
    "intensity_mse",
    "overlap_box",
    "overlaps",
    "solve_intensity",
]


def overlap_box(
    pos_a: Sequence[int],
    shape_a: Sequence[int],
    pos_b: Sequence[int],
    shape_b: Sequence[int],
) -> tuple[tuple[slice, ...], tuple[slice, ...]] | None:
    """Overlap of two axis-aligned tiles placed on a common pixel grid.

    Parameters
    ----------
    pos_a, pos_b : sequence of int
        Position of the first pixel of each tile in the common grid.
    shape_a, shape_b : sequence of int
        Spatial shapes of the tiles.

    Returns
    -------
    (tuple of slice, tuple of slice) or None
        The overlap as local slices of tile A and of tile B (pixel
        `a[sa]` lies on `b[sb]`), or None when the tiles do not overlap.
    """
    sa, sb = [], []
    for pa, na, pb, nb in zip(pos_a, shape_a, pos_b, shape_b, strict=True):
        lo, hi = max(pa, pb), min(pa + na, pb + nb)
        if hi <= lo:
            return None
        sa.append(slice(lo - pa, hi - pa))
        sb.append(slice(lo - pb, hi - pb))
    return tuple(sa), tuple(sb)


def overlaps(
    positions: Sequence[Sequence[int]], shapes: Sequence[Sequence[int]]
) -> list[tuple[int, int, tuple[slice, ...], tuple[slice, ...]]]:
    """All overlapping pairs `(i, j, slices_i, slices_j)`, with i < j."""
    out = []
    for i, j in itertools.combinations(range(len(positions)), 2):
        box = overlap_box(positions[i], shapes[i], positions[j], shapes[j])
        if box is not None:
            out.append((i, j, *box))
    return out


def bin_mean(x: Tensor, factor: Sequence[int]) -> Tensor:
    """Mean over non-overlapping bins of `factor` pixels along each dim.

    Trailing partial bins average the pixels they hold.
    """
    factor = tuple(int(f) for f in factor)
    if all(f == 1 for f in factor):
        return x
    pad = []
    for n, f in zip(reversed(x.shape), reversed(factor), strict=True):
        pad += [0, (-n) % f]
    total = torch.nn.functional.pad(x, pad)
    count = torch.nn.functional.pad(torch.ones_like(x), pad)
    shape = []
    for n, f in zip(total.shape, factor, strict=True):
        shape += [n // f, f]
    dims = tuple(range(1, 2 * x.ndim, 2))
    total = total.reshape(shape).sum(dims)
    count = count.reshape(shape).sum(dims)
    return total / count


@dataclass
class OverlapStats:
    """Bivariate sufficient statistics of corresponding pixel pairs.

    One entry per (group, pair) along the leading dims of the state.
    Moments are kept centred (count, means and scatter / co-scatter about
    the means) and merged with the pairwise formulas of
    [Chan et al. 1983](https://doi.org/10.1080/00031305.1983.10483115),
    so that the fit does not suffer from the cancellation of raw sums.

    Attributes
    ----------
    n : Tensor
        (..., P) number of pixel pairs.
    mean : Tensor
        (..., P, 2) means of the first and second tile of each pair.
    scatter : Tensor
        (..., P, 3) sums of squared deviations `M_aa`, `M_bb` and of
        cross products `M_ab`.
    """

    n: Tensor
    mean: Tensor
    scatter: Tensor

    @classmethod
    def zeros(
        cls,
        shape: Sequence[int],
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "OverlapStats":
        """Empty statistics of shape `(*shape,)` (e.g. `(*group, P)`)."""
        shape = tuple(shape)
        kw = dict(dtype=dtype, device=device)
        return cls(
            torch.zeros(shape, **kw),
            torch.zeros((*shape, 2), **kw),
            torch.zeros((*shape, 3), **kw),
        )

    @staticmethod
    def of(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Count, means and scatters of the pairs `(a, b)` (flattened)."""
        a, b = a.reshape(-1), b.reshape(-1)
        n = torch.as_tensor(float(a.numel()), dtype=a.dtype, device=a.device)
        if a.numel() == 0:
            return n, a.new_zeros(2), a.new_zeros(3)
        ma, mb = a.mean(), b.mean()
        da, db = a - ma, b - mb
        scatter = torch.stack(
            [da.square().sum(), db.square().sum(), (da * db).sum()]
        )
        return n, torch.stack([ma, mb]), scatter

    def add(self, index: tuple[int, ...], a: Tensor, b: Tensor) -> None:
        """Merge the pixel pairs `(a, b)` into entry `index`."""
        dtype = self.n.dtype
        n2, m2, s2 = self.of(a.to(dtype), b.to(dtype))
        self.merge_at(index, n2, m2, s2)

    def merge_at(
        self, index: tuple[int, ...], n2: Tensor, m2: Tensor, s2: Tensor
    ) -> None:
        """Merge precomputed `(n, mean, scatter)` into entry `index`."""
        n1, m1, s1 = self.n[index], self.mean[index], self.scatter[index]
        n = n1 + n2
        if float(n) == 0:
            return
        delta = m2.to(m1) - m1
        w = n1 * n2 / n
        self.scatter[index] = (
            s1
            + s2.to(s1)
            + w
            * torch.stack(
                [delta[0] * delta[0], delta[1] * delta[1], delta[0] * delta[1]]
            )
        )
        self.mean[index] = m1 + delta * (n2 / n)
        self.n[index] = n


def components(nodes: int, edges: Sequence[tuple[int, int]]) -> list[int]:
    """Connected-component label of each node (union-find)."""
    parent = list(range(nodes))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return [find(i) for i in range(nodes)]


def solve_intensity(
    pairs: Sequence[tuple[int, int]],
    n: Tensor,
    mean: Tensor,
    scatter: Tensor,
    ntiles: int,
    weights: Sequence[float],
    fixed: Sequence[int] = (),
) -> tuple[Tensor, Tensor]:
    """Per-tile intensity transforms from overlap statistics (one group).

    Each tile i is mapped to aᵢ yᵢ + cᵢ, the weighted average (Hörl et al.
    2019, Supplementary Note 17, eq. 9)

        aᵢ y + cᵢ = λ₁ (a′ᵢ y + c′ᵢ) + λ₂ (y + c″ᵢ) + λ₃ y

    of a linear model (a′ᵢ, c′ᵢ) and an additive model c″ᵢ, each fitted
    by least squares to the corresponding pixels (p, q) of tile i:
    p = yᵢ(r) and q = aⱼ yⱼ(r) + cⱼ, the transformed value of every
    overlapping tile j at the same position r. The transforms are the
    fixed point of this per-tile fit (the global optimisation of the
    paper's Supplementary Note 12 applied to one-dimensional points); as
    the fits are linear in the (aⱼ, cⱼ), the fixed point is found by one
    linear solve of size 2T.

    Parameters
    ----------
    pairs : sequence of (int, int)
        Tile indices (i, j) of each overlap.
    n : (P,) Tensor
        Number of pixel pairs in each overlap.
    mean : (P, 2) Tensor
        Means of tile i and of tile j over each overlap.
    scatter : (P, 3) Tensor
        Scatters `M_ii`, `M_jj` and co-scatter `M_ij` over each overlap.
    ntiles : int
        Number of tiles T.
    weights : (float, float, float)
        λ₁, λ₂, λ₃ (summing to 1).
    fixed : sequence of int
        Tiles held at the identity.

    Returns
    -------
    scale, shift : (T,) Tensor
        aᵢ and cᵢ, in the dtype of `mean`.

    Raises
    ------
    ValueError
        When λ₃ = 0 and a connected set of tiles has no fixed tile. The
        linear and additive models are both equivariant to a common
        offset, so without the identity term any offset common to all
        tiles of the set is a solution (and with λ₂ = 0 too, any common
        affine map, zero included).
    """
    l1, l2, l3 = (float(w) for w in weights)
    dt, dev = torch.float64, mean.device
    n = n.to(dt)
    mean = mean.to(dt)
    scatter = scatter.to(dt)
    keep = [k for k in range(len(pairs)) if float(n[k]) > 0]
    pairs = [pairs[k] for k in keep]
    n, mean, scatter = n[keep], mean[keep], scatter[keep]
    fixed = {int(f) for f in fixed}
    labels = components(ntiles, pairs)
    if l3 == 0:
        for root in set(labels):
            members = {i for i in range(ntiles) if labels[i] == root}
            linked = any(labels[i] == root for i, _ in pairs)
            if linked and not members & fixed:
                raise ValueError(
                    "With λ₃ = 0 every connected set of overlapping tiles "
                    "needs a fixed tile: a common offset of all tiles (and, "
                    "with λ₂ = 0 too, a common scale) is not determined."
                )
    # Global shift for conditioning: the fit is equivariant to y -> y - μ₀.
    mu0 = (
        (n[:, None] * mean).sum() / (2 * n.sum())
        if len(pairs)
        else torch.zeros((), dtype=dt, device=dev)
    )
    mean = mean - mu0
    # Neighbour lists: (j, n, mean_i, mean_j, M_ii, M_ij) seen from tile i.
    neigh: list[list[tuple]] = [[] for _ in range(ntiles)]
    for k, (i, j) in enumerate(pairs):
        mi, mj = mean[k, 0], mean[k, 1]
        sii, sjj, sij = scatter[k, 0], scatter[k, 1], scatter[k, 2]
        neigh[i].append((j, n[k], mi, mj, sii, sij))
        neigh[j].append((i, n[k], mj, mi, sjj, sij))
    t = ntiles
    m = torch.zeros(2 * t, 2 * t, dtype=dt, device=dev)
    rhs = torch.zeros(2 * t, dtype=dt, device=dev)
    degenerate = []
    for i in range(t):
        m[i, i] = 1.0
        m[t + i, t + i] = 1.0
        if i in fixed or not neigh[i]:
            rhs[i] = 1.0
            continue
        tot = sum(e[1] for e in neigh[i])
        mp = sum(e[1] * e[2] for e in neigh[i]) / tot
        spp = sum(e[4] + e[1] * (e[2] - mp) ** 2 for e in neigh[i])
        flat = float(spp) <= 1e-12 * float(
            sum(e[1] * (e[2] ** 2 + e[3] ** 2) for e in neigh[i]) + 1e-300
        )
        if flat:
            degenerate.append(i)
        for j, nij, mi, mj, _, sij in neigh[i]:
            # m_q = Σ nᵢⱼ (aⱼ μⱼ + cⱼ) / N
            wa, wc = nij * mj / tot, nij / tot
            if flat:
                # the linear model is undefined: use the additive one
                ua = uc = torch.zeros((), dtype=dt, device=dev)
                lin_a, lin_c = 0.0, l1
            else:
                # a′ = S_pq / S_pp, linear in (aⱼ, cⱼ)
                ua = (sij + nij * (mi - mp) * mj) / spp
                uc = nij * (mi - mp) / spp
                lin_a, lin_c = l1, l1
            # aᵢ = λ₁ a′ + λ₂ + λ₃
            m[i, j] -= lin_a * ua
            m[i, t + j] -= lin_a * uc
            # cᵢ = λ₁ (m_q − a′ m_p) + λ₂ (m_q − m_p)
            m[t + i, j] -= (lin_c + l2) * wa - lin_a * mp * ua
            m[t + i, t + j] -= (lin_c + l2) * wc - lin_a * mp * uc
        if flat:
            rhs[i] = l1 + l2 + l3
            rhs[t + i] = -(l1 + l2) * mp
        else:
            rhs[i] = l2 + l3
            rhs[t + i] = -l2 * mp
    if degenerate:
        warnings.warn(
            f"Tiles {degenerate} are constant over their overlaps: their "
            f"linear model is replaced by the additive one.",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        theta = torch.linalg.solve(m, rhs)
    except RuntimeError as e:  # singular system
        raise ValueError(f"The intensity system is singular: {e}") from None
    scale, shift = theta[:t], theta[t:]
    # back to the original intensities: a (y − μ₀) + c + μ₀
    shift = shift + mu0 * (1 - scale)
    for i in fixed:
        scale[i], shift[i] = 1.0, 0.0
    return scale, shift


def intensity_mse(
    pairs: Sequence[tuple[int, int]],
    n: Tensor,
    mean: Tensor,
    scatter: Tensor,
    scale: Tensor,
    shift: Tensor,
) -> float:
    """Mean squared difference of corresponding transformed pixels.

    The objective of Hörl et al. 2019, Supplementary Note 17, eq. 8 (with
    both tiles of a pair transformed), divided by the number of pairs of
    pixels: Σ ‖(aᵢ yᵢ + cᵢ) − (aⱼ yⱼ + cⱼ)‖² / Σ nᵢⱼ.
    """
    dt = torch.float64
    total, count = 0.0, 0.0
    for k, (i, j) in enumerate(pairs):
        nk = float(n[k])
        if nk == 0:
            continue
        ai, aj = float(scale[i]), float(scale[j])
        ci, cj = float(shift[i]), float(shift[j])
        mi, mj = mean[k].to(dt).tolist()
        sii, sjj, sij = scatter[k].to(dt).tolist()
        bias = ai * mi + ci - aj * mj - cj
        total += nk * bias**2 + ai**2 * sii + aj**2 * sjj - 2 * ai * aj * sij
        count += nk
    return total / count if count else math.nan
