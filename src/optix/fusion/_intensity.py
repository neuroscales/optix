"""Brightness and contrast adjustment of tiles (Hörl et al., Nat Methods 2019).

After [Hörl et al. 2019](https://doi.org/10.1038/s41592-019-0501-0),
Supplementary Note 17 (BigStitcher).
"""

import math
from collections.abc import Callable, Sequence
from typing import Self

import numpy as np
import torch
from torch import Tensor

from .._config import get_config
from .._estimator import Estimator, Tags, _same
from ..estimates import TileIntensityEstimate
from ..io import ArrayLike, Labeled, Layout, WritableArrayLike, blocks, labeled
from ..stats import Reservoir
from .utils import (
    OverlapStats,
    bin_mean,
    intensity_mse,
    overlaps,
    solve_intensity,
)

__all__ = ["IntensityAdjustment"]

_REFERENCE = "doi:10.1038/s41592-019-0501-0"


class _Crop:
    """Lazy crop of an array-like: indexing reads only what is asked."""

    def __init__(self, array: ArrayLike, index: tuple[slice, ...]) -> None:
        self.array = array
        shape = tuple(int(s) for s in array.shape)
        self.index = tuple(
            slice(*s.indices(n)[:2]) for s, n in zip(index, shape, strict=True)
        )
        self.shape = tuple(s.stop - s.start for s in self.index)

    @property
    def dtype(self) -> object:
        return self.array.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, index: object) -> object:
        if not isinstance(index, tuple):
            index = (index,)
        index = index + (slice(None),) * (len(self.shape) - len(index))
        out = []
        for base, sub, n in zip(self.index, index, self.shape, strict=True):
            if isinstance(sub, slice):
                start, stop, step = sub.indices(n)
                if step != 1:
                    raise IndexError("strided crops are not supported")
                out.append(slice(base.start + start, base.start + stop))
            else:
                out.append(base.start + int(sub) % n)
        return self.array[tuple(out)]


def _crop(array: ArrayLike, index: tuple[slice, ...]) -> ArrayLike:
    if isinstance(array, Tensor | np.ndarray):
        return array[index]  # a view: nothing is loaded
    return _Crop(array, index)


def _raw(tile: ArrayLike) -> ArrayLike:
    return tile.array if isinstance(tile, Labeled) else tile


def _core(
    extents: Sequence[int], factor: Sequence[int], budget: int
) -> list[int]:
    """Block size of at most `budget` pixels, halving the largest side.

    Sides smaller than the extent are multiples of `factor`, so that bins
    never straddle two blocks.
    """
    core = list(extents)
    while math.prod(core) > budget:
        k = max(range(len(core)), key=lambda d: core[d] // factor[d])
        half = math.ceil(core[k] / 2 / factor[k]) * factor[k]
        if half >= core[k]:
            break
        core[k] = half
    return core


def _reservoirs(
    size: int, seed: int | None, group: int, pair: int, weights: object
) -> tuple[Reservoir, Reservoir | None]:
    """Reservoirs of the pixel pairs (and bin weights) of one overlap.

    The seed of each (group, overlap) is derived from (`seed`, group,
    overlap), so that the samples are independent across groups and
    overlaps. The weights, when given, go to a twin reservoir with the
    same seed: both see streams of the same lengths, so they draw the
    same keys and keep the same positions.
    """
    entropy = None if seed is None else (int(seed), group, pair)
    ss = np.random.SeedSequence(entropy)
    s = int(ss.generate_state(1, np.uint64)[0]) >> 1
    values = Reservoir(size, seed=s)
    return values, (None if weights is None else Reservoir(size, seed=s))


class IntensityAdjustment(Estimator):
    """Brightness and contrast adjustment of overlapping tiles (BigStitcher).

    Tiles of a mosaic, placed on a common pixel grid by known positions,
    differ in brightness and contrast (bleaching, illumination drift). Each
    tile i gets a linear intensity transform

        x̂ᵢ = aᵢ yᵢ + cᵢ,

    chosen so that the transformed intensities of corresponding pixels
    agree in the overlap Oᵢⱼ of every pair of tiles. The mismatch is the
    squared difference (Supplementary Note 17, eq. 8, read with both
    tiles of a pair transformed)

        Σᵢ Σ_{j ≠ i} Σ_{r ∈ Oᵢⱼ} ((aⱼ yⱼ(r) + cⱼ) − (aᵢ yᵢ(r) + cᵢ))².

    As printed, eq. 8 transforms only one tile of the pair; the two-sided
    reading is the one of the point-correspondence objective of the
    paper's global optimisation (Supplementary Note 12, eq. 3), which the
    paper uses to solve it. As unconstrained, this sum has the trivial
    solution a = c = 0. Each
    transform is therefore the weighted average of a linear, an additive
    and the identity transform (eq. 9),

        aᵢ y + cᵢ = λ₁ (a′ᵢ y + c′ᵢ) + λ₂ (y + c″ᵢ) + λ₃ y,
        λ₁ + λ₂ + λ₃ = 1,

    which pulls the solution towards the identity when λ₂, λ₃ > 0.
    Without the identity term (λ₃ = 0), a common offset of all tiles is
    left free (and with λ₂ = 0 a common scale too), so tiles must then be
    fixed (`fixed=`). The
    paper solves eq. 8 with the iterative optimiser of its image
    registration (Supplementary Note 12): each tile's models are fitted by
    least squares to its corresponding points, the pixel values pᵣ = yᵢ(r)
    paired with the current transformed values qᵣ = aⱼ yⱼ(r) + cⱼ of the
    other tiles, and the fits are repeated until they agree. The result
    is the fixed point of these regularised per-tile fits: it minimises
    the sum above only for λ = (1, 0, 0), not for other weights. Those fits
    only need the count, means, scatters and co-scatter of the pixel pairs
    of each overlap, which are accumulated in one streaming pass over the
    overlaps; the agreed transforms are then the solution of one linear
    system of size 2T for T tiles, whose solution is the limit of the
    paper's iteration (see
    [`solve_intensity`][optix.fusion.utils.solve_intensity]).

    Paper notation: Iₐ → yᵢ, α^{Iₐ} → aᵢ, β^{Iₐ} → cᵢ, α′ → a′ᵢ,
    β₁ → c′ᵢ, β₂ → c″ᵢ, Oₐᵦ → Oᵢⱼ.

    !!! note "Faithful mode"
        Eq. 8 (read with both tiles transformed) and eq. 9. The paper
        leaves the weights λ to the user and does not give defaults; the
        default here, (λ₁, λ₂, λ₃) = (0.8, 0.1, 0.1), is not specified in
        the paper. The paper fits on precomputed downsampled images and
        offers to cap the number of corresponding pixels per overlap,
        without stating the factor or the cap: both are off in the
        faithful preset (`downsample=1`, `max_pixels=None`). The
        `recommended` preset (`faithful=False`) bins by `downsample=4`
        (see "Noise"). Tiles held at the identity (`fixed=`) are those of
        the paper's global optimisation (Supplementary Note 12, eq. 3).

    !!! note "Positions and correspondences"
        Tiles are placed by translations on a common pixel grid:
        `positions[i]` is the grid position of the first pixel of tile i
        along each spatial axis (layout order). Non-integer positions are
        rounded to the nearest pixel: corresponding pixels are those that
        land on the same grid pixel. Rotated or affinely registered tiles
        are not supported. Sample axes (e.g. time points) share the tile
        transforms and pool their pixels; group axes (e.g. channels) are
        fitted independently.

    !!! note "Downsampling"
        With `downsample=f`, both tiles of an overlap are averaged over the
        same bins of f pixels per axis (anchored at the corner of the
        overlap), so that binned values still correspond exactly. Partial
        bins at the far edges of an overlap are weighted by the fraction
        of a full bin that they hold, i.e. by the inverse of their noise
        variance relative to a full bin.

    !!! warning "Noise"
        The per-tile fits regress the other tiles' values on the tile's
        own noisy values, so noise biases them (errors-in-variables
        attenuation): with signal variance s² and noise variance σ² in the
        overlaps, each linear fit shrinks the slope by about
        s² / (s² + σ²), and the fixed point compounds the shrinkage along
        the chain of overlaps from the fixed tiles. On a 3 × 3 mosaic with
        signal standard deviation s ≈ 50 and 20 % overlap
        (λ = (1, 0, 0), centre tile fixed), the mean error of `scale` is
        about −6 % at σ = 5 and −48 % at σ = 20 without binning, and
        −0.4 % and −6 % with `downsample=4`. Binning by f per axis divides
        σ² by f^D, which is why the paper fits on downsampled images; use
        `downsample` > 1 on noisy data. `max_pixels` does not reduce the
        bias.

    !!! references "References"
        1. Hörl D, Rojas Rusak F, Preusser F, Tillberg P, Randel N,
           Chhetri RK, Cardona A, Keller PJ, Harz H, Leonhardt H, Treier M,
           Preibisch S. *BigStitcher: reconstructing high-resolution image
           datasets of cleared and expanded samples.* Nat Methods
           **16**(9):870–874, 2019.
           [doi:10.1038/s41592-019-0501-0](https://doi.org/10.1038/s41592-019-0501-0)

        ??? quote "BibTeX"
            ```bibtex
            @article{horl2019bigstitcher,
              author  = {H{\\"o}rl, David and Rojas Rusak, Fabio and
                         Preusser, Friedrich and Tillberg, Paul and
                         Randel, Nadine and Chhetri, Raghav K. and
                         Cardona, Albert and Keller, Philipp J. and
                         Harz, Hartmann and Leonhardt, Heinrich and
                         Treier, Mathias and Preibisch, Stephan},
              title   = {{BigStitcher}: reconstructing high-resolution image
                         datasets of cleared and expanded samples},
              journal = {Nature Methods},
              volume  = {16}, number = {9}, pages = {870--874},
              year    = {2019}, doi = {10.1038/s41592-019-0501-0}
            }
            ```

    Attributes
    ----------
    estimates : dict
        `{"intensity": TileIntensityEstimate}` with `scale` and `shift` of
        shape `(*group, T)`; `meta["mse_before"]` and `meta["mse_after"]`
        hold the mean squared difference of corresponding pixels before
        and after the adjustment, per group (flattened).
    """

    tags = Tags(
        targets=("intensity",),
        streaming="pointwise",
        modality=("lightsheet", "confocal", "widefield"),
        reference=_REFERENCE,
    )
    paper = {
        "weights": (0.8, 0.1, 0.1),
        "downsample": 1,
        "max_pixels": None,
        "seed": None,
    }
    recommended = {**paper, "downsample": 4}
    _free = ("seed",)

    def __init__(
        self,
        *,
        weights: Sequence[float] | None = None,
        downsample: int | Sequence[int] | None = None,
        max_pixels: int | None = None,
        seed: int | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the adjustment.

        Parameters
        ----------
        weights : (float, float, float)
            λ₁, λ₂, λ₃ of the linear, additive and identity transforms
            (eq. 9); non-negative, summing to 1. λ₃ > 0 makes the
            solution unique; with λ₃ = 0 (a common offset is then free,
            and with λ₂ = 0 too a common scale), every connected set of
            overlapping tiles needs a fixed tile. Default (0.8, 0.1,
            0.1), not specified in the paper.
        downsample : int or sequence of int
            Bin size along each spatial axis (layout order) before the
            fit, which reduces the bias that noise puts on the fit (see
            "Noise" above). Default 1 (no binning) in faithful mode, 4
            with `faithful=False`; the paper fits on downsampled images
            without stating the factor.
        max_pixels : int, optional
            Maximum number of corresponding pixels (after binning) per
            overlap, drawn uniformly at random without replacement, so
            that large overlaps do not dominate small ones. Default None:
            all pixels.
        seed : int, optional
            Seed of the random selection of `max_pixels`. Each overlap
            and group draws from its own stream, derived from
            (`seed`, group, overlap).
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        w = tuple(float(v) for v in o["weights"])
        if len(w) != 3 or min(w) < 0 or abs(sum(w) - 1) > 1e-9:
            raise ValueError(
                f"weights must be 3 non-negative numbers summing to 1, "
                f"got {o['weights']!r}"
            )
        o["weights"] = w
        ds = o["downsample"]
        if isinstance(ds, int | np.integer):
            ds = int(ds)
            if ds < 1:
                raise ValueError(f"downsample must be >= 1, got {ds}")
        else:
            ds = tuple(int(v) for v in ds)
            if min(ds) < 1:
                raise ValueError(f"downsample must be >= 1, got {ds}")
        o["downsample"] = ds
        if o["max_pixels"] is not None and int(o["max_pixels"]) < 1:
            raise ValueError("max_pixels must be >= 1")
        self.is_faithful = all(
            _same(value, self.paper.get(name))
            for name, value in o.items()
            if name not in self._free
        )

    # -- public API ------------------------------------------------------
    def fit(
        self,
        data: Sequence[ArrayLike],
        positions: ArrayLike,
        *,
        fixed: Sequence[int] = (),
        layout: Layout | None = None,
        init: object = None,
        callback: Callable | None = None,
        verbose: bool = False,
    ) -> Self:
        """Estimate one intensity transform per tile from the overlaps.

        Parameters
        ----------
        data : sequence of array-like or Labeled
            The T tiles, all in the same layout (spatial shapes may
            differ). Without a layout, the trailing D dims are spatial,
            with D the number of columns of `positions`.
        positions : (T, D) array-like
            Grid position of the first pixel of each tile, along each
            spatial axis in layout order (rounded to whole pixels).
        fixed : sequence of int
            Tiles held at the identity (a = 1, c = 0).
        layout, init, callback, verbose
            See [`Estimator.fit`][optix.Estimator.fit]; `init` is not used
            (the solution is computed directly).

        Returns
        -------
        IntensityAdjustment
            self
        """
        if isinstance(data, Labeled | Tensor | np.ndarray) or not isinstance(
            data, Sequence
        ):
            raise TypeError("data must be a sequence of tiles")
        if layout is None:
            layouts = {y.layout for y in data if isinstance(y, Labeled)}
            if len(layouts) > 1:
                raise ValueError("Tiles are labelled with different layouts")
            if layouts:
                layout = layouts.pop()
        pos = np.rint(np.asarray(positions, dtype=np.float64)).astype(np.int64)
        if pos.ndim != 2 or pos.shape[0] != len(data):
            raise ValueError(
                f"positions must be (T, D) with T = {len(data)} tiles, got "
                f"shape {pos.shape}"
            )
        fixed = tuple(int(f) for f in fixed)
        if any(not 0 <= f < len(data) for f in fixed):
            raise ValueError(f"fixed tiles out of range: {fixed}")
        self._positions, self._fixed = pos, fixed
        try:
            return super().fit(
                list(data),
                layout=layout,
                init=init,
                callback=callback,
                verbose=verbose,
            )
        finally:
            del self._positions, self._fixed

    def apply(
        self,
        data: Sequence[ArrayLike],
        out: Sequence[WritableArrayLike] | None = None,
    ) -> list[WritableArrayLike]:
        """Harmonise every tile: x̂ᵢ = aᵢ yᵢ + cᵢ.

        Parameters
        ----------
        data : sequence of array-like or Labeled
            The T tiles, in the order and layout of the fit.
        out : sequence of writable array-like, optional
            One output per tile (in place is allowed). Default: new
            tensors.

        Returns
        -------
        list of writable array-like
            The outputs.
        """
        if isinstance(data, Labeled | Tensor | np.ndarray) or not isinstance(
            data, Sequence
        ):
            raise TypeError("data must be a sequence of tiles")
        return super().apply(list(data), out)

    # -- implementation ----------------------------------------------------
    @staticmethod
    def _labeled(data: object, layout: Layout | None) -> object:
        if isinstance(data, list):
            return [Estimator._labeled(y, layout) for y in data]
        return Estimator._labeled(data, layout)

    def _tile_layout(self, tiles: list, ndim: int) -> Layout:
        layout = self.layout
        if layout is None:
            nd = {len(_raw(t).shape) for t in tiles}
            if len(nd) != 1:
                raise ValueError("Tiles have different numbers of dims")
            layout = Layout.default(nd.pop(), ndim)
        return layout

    def _factor(self, ndim: int) -> tuple[int, ...]:
        ds = self.options["downsample"]
        if isinstance(ds, int):
            return (ds,) * ndim
        if len(ds) != ndim:
            raise ValueError(
                f"downsample has {len(ds)} entries for {ndim} spatial axes"
            )
        return ds

    def _fit(
        self,
        data: list,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        self._warn_unused(init)
        pos, fixed = self._positions, self._fixed
        ntiles, ndim = pos.shape
        # an inferred layout is not stored: the next fit infers its own
        layout = self._tile_layout(data, ndim)
        raws = [_raw(t) for t in data]
        _, _, spatial = layout.positions(len(raws[0].shape))
        if len(spatial) != ndim:
            raise ValueError(
                f"positions have {ndim} columns, the layout has "
                f"{len(spatial)} spatial axes"
            )
        shapes = [layout.shape_of(r) for r in raws]
        gshape = shapes[0][0]
        if any(s[0] != gshape for s in shapes):
            raise ValueError("Tiles have different group shapes")
        if any(s[1] != shapes[0][1] for s in shapes):
            raise ValueError("Tiles have different sample shapes")
        factor = self._factor(ndim)
        boxes = overlaps(pos.tolist(), [s[2] for s in shapes])
        pairs = [(i, j) for i, j, _, _ in boxes]
        ngroups = math.prod(gshape)
        backend = self._backend(raws[0])
        stats = OverlapStats.zeros(
            (ngroups, len(pairs)), device=backend["device"]
        )
        o = self.options
        cap = o["max_pixels"]
        binned = any(f > 1 for f in factor)
        full = math.prod(factor)
        itemsize = backend["dtype"].itemsize
        budget = max(1, get_config().memory // (8 * itemsize))
        for k, (i, j, si, sj) in enumerate(boxes):
            views = []
            for t, sl in ((i, si), (j, sj)):
                index = [slice(None)] * len(raws[t].shape)
                for ax, s in zip(spatial, sl, strict=True):
                    index[ax] = s
                views.append(labeled(_crop(raws[t], tuple(index)), layout))
            extents = [s.stop - s.start for s in si]
            core = _core(extents, factor, budget)
            ca, cb = (
                blocks(
                    v,
                    core,
                    0,
                    dtype=backend["dtype"],
                    device=backend["device"],
                )
                for v in views
            )
            # reservoirs of this overlap only, per group: (pairs, weights)
            reservoirs: dict[int, tuple[Reservoir, Reservoir | None]] = {}
            for a, b in zip(ca, cb, strict=True):
                g = int(np.ravel_multi_index(a.group, gshape)) if gshape else 0
                xa, count = bin_mean(a.data, factor, count=True)
                xb = bin_mean(b.data, factor)
                # partial bins weigh by the fraction of a full bin they hold
                w = count / full if binned else None
                if cap is None:
                    stats.add((g, k), xa, xb, w)
                    continue
                if g not in reservoirs:
                    reservoirs[g] = _reservoirs(int(cap), o["seed"], g, k, w)
                rv, rw = reservoirs[g]
                rv.update(
                    torch.complex(
                        xa.reshape(-1).to(torch.float64),
                        xb.reshape(-1).to(torch.float64),
                    )
                )
                if rw is not None:
                    rw.update(w.reshape(-1).to(torch.float64))
            # overlap k is complete: merge its samples and free them
            for g, (rv, rw) in reservoirs.items():
                if rv.values is not None and rv.values.numel():
                    v = rv.values
                    wv = None if rw is None else rw.values
                    stats.merge_at(
                        (g, k), *OverlapStats.of(v.real, v.imag, wv)
                    )
            del reservoirs
        scale = torch.ones(ngroups, ntiles, dtype=torch.float64)
        shift = torch.zeros(ngroups, ntiles, dtype=torch.float64)
        before, after = [], []
        for g in range(ngroups):
            n, mean, scatter = stats.n[g], stats.mean[g], stats.scatter[g]
            a, c = solve_intensity(
                pairs, n, mean, scatter, ntiles, o["weights"], fixed
            )
            scale[g], shift[g] = a.cpu(), c.cpu()
            before.append(
                intensity_mse(
                    pairs,
                    n,
                    mean,
                    scatter,
                    torch.ones(ntiles),
                    torch.zeros(ntiles),
                )
            )
            after.append(intensity_mse(pairs, n, mean, scatter, a, c))
        self.history.record(-sum(after) if after else 0.0)
        self.history.converged = True
        self.history.reason = "direct solve of the fixed point"
        out = dict(dtype=backend["dtype"], device=backend["device"])
        self.estimates = {
            "intensity": TileIntensityEstimate(
                scale=scale.reshape(*gshape, ntiles).to(**out),
                shift=shift.reshape(*gshape, ntiles).to(**out),
                meta={
                    "mse_before": before,
                    "mse_after": after,
                    "pairs": [list(p) for p in pairs],
                    "ndim": ndim,
                },
            )
        }

    def _apply(
        self, data: list, out: Sequence[WritableArrayLike] | None
    ) -> list[WritableArrayLike]:
        est = self.estimates["intensity"]
        ntiles = est.scale.shape[-1]
        if len(data) != ntiles:
            raise ValueError(f"Expected {ntiles} tiles, got {len(data)}")
        if out is not None and len(out) != ntiles:
            raise ValueError(f"Expected {ntiles} outputs, got {len(out)}")
        nlead = est.scale.ndim - 1
        results = []
        for i, tile in enumerate(data):
            if not isinstance(tile, Labeled):
                layout = self.layout
                if layout is None:
                    nd = len(tile.shape)
                    layout = Layout.default(nd, est.meta.get("ndim", nd))
                tile = labeled(tile, layout)
            nsp = len(tile.layout.spatial)

            def fn(chunk: object, batch: Tensor, i: int = i) -> Tensor:
                g = tuple(chunk.group[:nlead])
                a = est.scale[g + (i,)].to(batch)
                c = est.shift[g + (i,)].to(batch)
                return batch * a + c

            target = None if out is None else out[i]
            results.append(self._apply_frames(tile, target, fn, ndim=nsp))
        return results
