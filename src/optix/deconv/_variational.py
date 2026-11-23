"""Shared machinery of the regularised (variational) deconvolutions."""

import warnings
from collections.abc import Callable

import torch
from torch import Tensor

from ..io import ArrayLike, Chunk, WritableArrayLike, blocks
from ._rl import RichardsonLucy

__all__: list[str] = []


class _Variational(RichardsonLucy):
    """Base of the estimators that minimise a regularised objective.

    Reuses the PSF, halo, layout and block handling of
    [`RichardsonLucy`][optix.deconv.RichardsonLucy]. `fit` makes one pass
    over the prepared data and stores, per image `(*group, *sample)`,
    its mean (`level`), its maximum (`peak`) and the mean read-noise
    variance (`noise`); subclasses derive their per-image parameters
    from them in `_setup_images`. `apply` solves each block with
    `_solve(blk, op, index, trace)`, where `index` selects the image in
    those per-image tensors. When an image is split into several blocks,
    one `RuntimeWarning` is emitted per call.
    """

    def _setup_images(self, psf: Tensor) -> None:
        """Per-image parameters from `level`, `peak` and `noise`."""

    # -- hooks of RichardsonLucy -----------------------------------------
    def _start(self, chunk: Chunk) -> object:
        index = chunk.group + chunk.samples
        if not self._split_warned and any(
            s.indices(n)[:2] != (0, n)
            for s, n in zip(
                _slices(chunk.inner, len(chunk.spatial)),
                chunk.data.shape[-len(chunk.spatial) :],
                strict=True,
            )
        ):
            self._split_warned = True
            warnings.warn(
                f"{type(self).__name__}: the image does not fit in the "
                "memory budget and is solved in blocks with a halo; the "
                "result is approximate near the block seams. Raise the "
                "memory budget (optix.config(memory=...)) or pass block=.",
                RuntimeWarning,
                stacklevel=4,
            )
        return index

    # -- estimator protocol ----------------------------------------------
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
        group, sample, _ = data.layout.shape_of(data.array)
        f64 = torch.float64
        total = torch.zeros(group + sample, dtype=f64)
        count = torch.zeros(group + sample, dtype=f64)
        peak = torch.full(group + sample, -torch.inf, dtype=f64)
        noise = torch.zeros(group + sample, dtype=f64)
        maps = self._maps(data, backend)
        for chunk in blocks(data, None, 0, **backend):
            blk = self._prepare(chunk, maps)
            index = chunk.group + chunk.samples
            total[index] += float(blk.data.sum(dtype=f64))
            count[index] += blk.data.numel()
            peak[index] = torch.maximum(
                peak[index], torch.tensor(float(blk.data.max()), dtype=f64)
            )
            var = torch.as_tensor(blk.variance)
            noise[index] += float(var.sum(dtype=f64)) * (
                blk.data.numel() / max(var.numel(), 1)
            )
        tiny = torch.finfo(f64).tiny
        self.level = total / count.clamp(min=tiny)
        self.peak = peak
        self.noise = noise / count.clamp(min=tiny)
        self._setup_images(psf)
        self.history.extras["iterations"] = []
        self.history.extras["change"] = []

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        self._split_warned = False
        return super()._apply(data, out)

    def _extra_state(self) -> dict[str, object]:
        state = super()._extra_state()
        state["peak"] = getattr(self, "peak", None)
        state["noise"] = getattr(self, "noise", None)
        return state

    def _load_extra_state(self, state: dict[str, object]) -> None:
        super()._load_extra_state(state)
        self.peak = state.get("peak")
        self.noise = state.get("noise")


def _slices(inner: object, n: int) -> tuple[slice, ...]:
    """The last `n` entries of a chunk's `inner` index."""
    inner = tuple(inner) if isinstance(inner, tuple) else (inner,)
    return inner[-n:]
