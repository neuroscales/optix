"""Shared machinery of the regularised (variational) deconvolutions."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .._config import get_config
from ..io import ArrayLike, Chunk, Chunks, Labeled, blocks, chunks_of
from ._block import _BlockDeconvolution
from ._fft import next_fast_len

__all__: list[str] = []

#: Quantile of the prepared data stored as `robust_peak` by `fit`.
_PEAK_QUANTILE = 0.999
#: Working buffers per loaded voxel besides the grid (data, read noise).
_LOADED_BUFFERS = 4.0
#: Smallest fraction of a loaded block kept as its core, when the
#: memory budget forces the block size.
_MIN_CORE_FRACTION = 0.25


def _hashable(index: tuple) -> tuple:
    """A dict key for an index of ints and slices."""
    return tuple(
        (s.start, s.stop, s.step) if isinstance(s, slice) else s for s in index
    )


@dataclass
class _Problem:
    """Per-block inputs: data in electrons and read-noise variance."""

    data: Tensor
    variance: Tensor | float = 0.0


class _UpperQuantile:
    """Exact upper quantile of a stream of values, from its largest ones.

    For n values in all and the quantile q (linear interpolation between
    order statistics, as `torch.quantile`), only the n − ⌊q(n − 1)⌋
    largest values are needed; they are kept in float64.
    """

    def __init__(self, n: int, q: float) -> None:
        self.pos = q * (n - 1)
        self.k = n - math.floor(self.pos)
        self.top: Tensor | None = None

    def update(self, x: Tensor) -> None:
        x = x.detach().reshape(-1)
        if x.numel() > self.k:
            x = torch.topk(x, self.k, sorted=False).values
        x = x.to(dtype=torch.float64, device="cpu")
        if self.top is not None:
            x = torch.cat([self.top, x])
            if x.numel() > self.k:
                x = torch.topk(x, self.k, sorted=False).values
        self.top = x

    def value(self) -> float:
        v = torch.sort(self.top).values  # order statistics ⌊pos⌋, …, n − 1
        frac = self.pos - math.floor(self.pos)
        hi = v[1] if v.numel() > 1 else v[0]
        return float(v[0] + frac * (hi - v[0]))


class _Variational(_BlockDeconvolution):
    """Base of the estimators that minimise a regularised objective.

    `fit` makes one pass over the prepared data and stores, per image
    `(*group, *sample)`, its maximum (`peak`), its 99.9th percentile
    (`robust_peak`) and the mean read-noise variance (`noise`);
    subclasses derive their per-image parameters from them in
    `_setup_images`. `apply` solves each block with `_solve(blk, op,
    index, trace)`, where `index` selects the image in those per-image
    tensors.

    The block size counts the solver's buffers on the padded FFT grid of
    a block (`_grid_buffers()` per grid voxel), not on the loaded block.
    When the memory budget leaves a core smaller than a quarter of the
    loaded block, `apply` raises and says how much memory is needed.
    """

    _warn_split = True

    def _setup_images(self, psf: Tensor) -> None:
        """Per-image parameters from `peak`, `robust_peak` and `noise`."""

    def _grid_buffers(self) -> float:
        """Working buffers of the solver per voxel of the FFT grid."""
        raise NotImplementedError

    # -- hooks of _BlockDeconvolution --------------------------------------
    def _fitted_images(self) -> tuple[int, ...] | None:
        peak = getattr(self, "peak", None)
        return None if peak is None else tuple(peak.shape)

    def _block_input(self, chunk: Chunk) -> tuple[int | slice, ...]:
        """Index of the chunk's image in the per-image tensors."""
        return chunk.group + chunk.samples

    def _block_bytes(
        self,
        core: Sequence[int],
        extents: Sequence[int],
        halo: Sequence[int],
        psf: Tensor,
        itemsize: int,
    ) -> float:
        """Memory of the solve of one block with this core."""
        loaded = [
            min(c + 2 * h, e)
            for c, e, h in zip(core, extents, halo, strict=True)
        ]
        grid = math.prod(
            next_fast_len(n + k - 1)
            for n, k in zip(loaded, psf.shape, strict=True)
        )
        return itemsize * (
            self._grid_buffers() * grid + _LOADED_BUFFERS * math.prod(loaded)
        )

    def _auto_block(
        self,
        data: Labeled,
        psf: Tensor,
        halo: tuple[int, ...],
        align: Sequence[int] | None,
        itemsize: int,
    ) -> list[int]:
        """Core block size from the memory budget (grid-based count)."""
        _, _, extents = data.layout.shape_of(data.array)
        extents = [max(int(e), 1) for e in extents]
        cfg = get_config()
        if cfg.batch is not None:
            budget = cfg.batch * math.prod(extents[-2:]) * itemsize
        else:
            budget = cfg.memory

        def cost(core: Sequence[int]) -> float:
            return self._block_bytes(core, extents, halo, psf, itemsize)

        def kept(core: Sequence[int]) -> float:
            loaded = [
                min(c + 2 * h, e)
                for c, e, h in zip(core, extents, halo, strict=True)
            ]
            return math.prod(
                min(c, e) for c, e in zip(core, extents, strict=True)
            ) / (math.prod(loaded))

        block = list(extents)
        enough = None  # cheapest block keeping enough of its solve
        while True:
            if kept(block) >= _MIN_CORE_FRACTION:
                enough = cost(block)
            if cost(block) <= budget or max(block) <= 1:
                break
            d = max(range(len(block)), key=lambda i: block[i])
            block[d] = -(-block[d] // 2)
        if align is not None:
            block = [
                (b // c) * c if c and b >= c else b
                for b, c in zip(block, align, strict=True)
            ]
        if kept(block) < _MIN_CORE_FRACTION:
            mib = 2.0**20
            loaded = [
                min(c + 2 * h, e)
                for c, e, h in zip(block, extents, halo, strict=True)
            ]
            need = (
                f" Blocks that keep at least a quarter of each solve need "
                f"about {enough / mib:.0f} MiB."
                if enough is not None
                else ""
            )
            raise ValueError(
                f"{type(self).__name__}: the memory budget of "
                f"{budget / mib:.0f} MiB allows blocks with a core of "
                f"{tuple(block)} in a loaded block of {tuple(loaded)} "
                f"(halo {halo}): only {100 * kept(block):.1f} % of each "
                "solve would be kept. Solving the whole image needs about "
                f"{cost(extents) / mib:.0f} MiB.{need} Raise the budget "
                "(optix.config(memory=...)), or pass a smaller halo= or an "
                "explicit block=."
            )
        return block

    def _blocks(
        self,
        data: Labeled,
        psf: Tensor,
        halo: tuple[int, ...],
        out: object,
        backend: dict,
    ) -> Chunks:
        block = self.options["block"]
        align = chunks_of(out)
        if block is None:
            _, _, spatial = data.layout.positions(len(data.array.shape))
            spatial_align = (
                None if align is None else [align[p] for p in spatial]
            )
            itemsize = torch.empty((), dtype=backend["dtype"]).element_size()
            block = self._auto_block(data, psf, halo, spatial_align, itemsize)
        return blocks(
            data, block, halo, contiguous=True, align=align, **backend
        )

    # -- estimator protocol ------------------------------------------------
    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        data = self._labeled_for(data)
        self.layout = data.layout
        backend = self._backend(data)
        shape = data.layout.map_shape(data.array)
        self._init_from(init, shape, data)
        psf = self._psf(backend)
        if psf.ndim != len(data.layout.spatial):
            raise ValueError(
                f"The PSF has {psf.ndim} dims but the layout has "
                f"{len(data.layout.spatial)} spatial axes"
            )
        group, sample, spatial = data.layout.shape_of(data.array)
        f64 = torch.float64
        count = torch.zeros(group + sample, dtype=f64)
        peak = torch.full(group + sample, -torch.inf, dtype=f64)
        noise = torch.zeros(group + sample, dtype=f64)
        # keyed by (start, stop) pairs: slices are unhashable before 3.12
        upper: dict[tuple, tuple[tuple, _UpperQuantile]] = {}
        n = math.prod(spatial)
        maps = self._maps(data, backend)
        for chunk in blocks(data, None, 0, **backend):
            blk = self._prepare(chunk, maps)
            index = chunk.group + chunk.samples
            count[index] += blk.data.numel()
            peak[index] = torch.maximum(
                peak[index], torch.tensor(float(blk.data.max()), dtype=f64)
            )
            key = _hashable(index)
            if key not in upper:
                upper[key] = (index, _UpperQuantile(n, _PEAK_QUANTILE))
            upper[key][1].update(blk.data)
            var = torch.as_tensor(blk.variance)
            noise[index] += float(var.sum(dtype=f64)) * (
                blk.data.numel() / max(var.numel(), 1)
            )
        robust = torch.full(group + sample, -torch.inf, dtype=f64)
        for index, q in upper.values():
            robust[index] = q.value()
        tiny = torch.finfo(f64).tiny
        self.peak = peak
        self.robust_peak = robust
        self.noise = noise / count.clamp(min=tiny)
        self._setup_images(psf)
        self.history.extras["iterations"] = []
        self.history.extras["change"] = []
        self.history.extras["primal_residual"] = []
        self.history.extras["dual_residual"] = []

    def _extra_state(self) -> dict[str, object]:
        state = super()._extra_state()
        state["peak"] = getattr(self, "peak", None)
        state["robust_peak"] = getattr(self, "robust_peak", None)
        state["noise"] = getattr(self, "noise", None)
        return state

    def _load_extra_state(self, state: dict[str, object]) -> None:
        super()._load_extra_state(state)
        self.peak = state.get("peak")
        self.robust_peak = state.get("robust_peak")
        self.noise = state.get("noise")
