"""Shared machinery of the block-iterative deconvolutions."""

import warnings

from torch import Tensor

from .._estimator import Estimator
from ..io import (
    ArrayLike,
    Chunk,
    Chunks,
    Labeled,
    Layout,
    WritableArrayLike,
    blocks,
    chunks_of,
    flush,
    prepare_out,
)
from ._fft import PaddedConvolution
from ._shiftvariant import ShiftVariantPSF

__all__: list[str] = []


class _BlockDeconvolution(Estimator):
    """Base of the deconvolutions that solve each block of an image.

    Holds what does not depend on the algorithm: the PSF and its halo,
    the layout, the loaded region of a block, and the loop of `apply`.
    That loop cuts every image `(*group, *sample)` into blocks with a
    halo (the whole image when it fits in the memory budget), prepares
    each block with `_prepare`, solves it with `_solve` on the grid of a
    [`PaddedConvolution`][optix.deconv.utils.PaddedConvolution] and
    stores the core. Subclasses provide the hooks below.

    - `_psf(backend)`: the normalised PSF.
    - `_maps(data, backend)`, `_prepare(chunk, maps)`: the per-block
      inputs.
    - `_fitted_images()`: the per-image shape `(*group, *sample)` of the
      fitted state, or None before `fit`.
    - `_block_input(chunk)`: the per-image input of `_solve`.
    - `_solve(blk, op, value, trace)`: the estimate on `op.grid` and the
      number of updates.
    - `_blocks(data, psf, halo, out, backend)`: the block iterator
      (default: block size from the memory budget with `_overhead()`
      working buffers per loaded voxel).
    - `_begin_apply()`, `_after_block(trace)`: per-call and per-block
      bookkeeping.

    With `_warn_split`, one `RuntimeWarning` is emitted per `apply` call
    when an image is solved in several blocks.
    """

    _warn_split = False

    # -- hooks -------------------------------------------------------------
    def _overhead(self) -> float:
        """Working buffers per loaded voxel, for the block size."""
        return 1.0

    def _fitted_images(self) -> tuple[int, ...] | None:
        raise NotImplementedError

    def _block_input(self, chunk: Chunk) -> object:
        raise NotImplementedError

    def _begin_apply(self) -> None:
        """Called once per `apply`, before the first block."""

    def _after_block(self, trace: dict) -> None:
        """Called after each block with the trace of its solve."""

    def _overhead_for(self, psf: object, data: Labeled) -> float:
        """Working buffers per loaded voxel for this PSF and data."""
        return self._overhead()

    # -- helpers -----------------------------------------------------------
    def _halo(self, psf: Tensor) -> tuple[int, ...]:
        halo = self.options["halo"]
        need = tuple(psf.shape)
        if halo is None:
            return tuple(2 * k for k in need)
        halo = (
            (int(halo),) * psf.ndim
            if isinstance(halo, int)
            else tuple(int(h) for h in halo)
        )
        if len(halo) != psf.ndim:
            raise ValueError("halo must have one entry per PSF dim")
        if any(h < k for h, k in zip(halo, need, strict=True)):
            raise ValueError(
                f"halo {halo} must be at least the PSF size {need}"
            )
        return halo

    def _labeled_for(self, data: ArrayLike) -> Labeled:
        if isinstance(data, Labeled):
            return data
        psf = self.psf
        ndim = 2 if psf is None else psf.ndim
        return self._labeled(data, Layout.default(len(data.shape), ndim))

    def _block_operator(
        self,
        psf: Tensor | ShiftVariantPSF,
        chunk: Chunk,
        shape: tuple[int, ...],
        cache: dict,
    ) -> PaddedConvolution:
        """Blur operator of a loaded block (core + halo).

        A shift-variant PSF gives the operator of the block at its global
        position; a single PSF gives one operator per block shape.
        """
        if isinstance(psf, ShiftVariantPSF):
            origin = tuple(s.start for s in self._region(chunk))
            return psf.operator(origin, shape)
        if shape not in cache:
            cache[shape] = PaddedConvolution(psf, shape)
        return cache[shape]

    @staticmethod
    def _region(chunk: Chunk) -> tuple[slice, ...]:
        """Loaded region (core + halo) along the spatial axes."""
        region = []
        for core, inner, n in zip(
            chunk.spatial, chunk.inner, chunk.data.shape, strict=True
        ):
            start = core.start - inner.start
            region.append(slice(start, start + n))
        return tuple(region)

    @staticmethod
    def _is_split(chunk: Chunk) -> bool:
        """Whether the core of `chunk` is only part of its image."""
        n = len(chunk.spatial)
        inner = (
            chunk.inner if isinstance(chunk.inner, tuple) else (chunk.inner,)
        )
        return any(
            s.indices(m)[:2] != (0, m)
            for s, m in zip(inner[-n:], chunk.data.shape[-n:], strict=True)
        )

    def _blocks(
        self,
        data: Labeled,
        psf: Tensor,
        halo: tuple[int, ...],
        out: WritableArrayLike,
        backend: dict,
    ) -> Chunks:
        return blocks(
            data,
            self.options["block"],
            halo,
            overhead=self._overhead_for(psf, data),
            contiguous=True,
            align=chunks_of(out),
            **backend,
        )

    # -- estimator protocol ------------------------------------------------
    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        images = self._fitted_images()
        if images is None:
            raise RuntimeError("Call fit(data) before apply(data)")
        data = self._labeled_for(data)
        group, sample, _ = data.layout.shape_of(data.array)
        if tuple(images) != group + sample:
            raise ValueError(
                "apply(data) needs the data given to fit (images per "
                f"group/sample {tuple(images)}, got {group + sample})"
            )
        backend = self._backend(data)
        psf = self._psf(backend)
        halo = self._halo(psf)
        raw = data.array
        target = out.array if isinstance(out, Labeled) else out
        if out is not None and target is raw:
            raise ValueError("In-place deconvolution is not possible")
        out = prepare_out(out, tuple(data.shape), like=raw, dtype=self.dtype)
        maps = self._maps(data, backend)
        ops: dict[tuple[int, ...], PaddedConvolution] = {}
        iterations = self.history.extras.setdefault("iterations", [])
        self._begin_apply()
        trace: dict = {}
        chunks = self._blocks(data, psf, halo, out, backend)
        warned = not self._warn_split
        for chunk in chunks:
            if not warned and self._is_split(chunk):
                warned = True
                warnings.warn(
                    f"{type(self).__name__}: the image does not fit in the "
                    "memory budget and is solved in blocks with a halo; the "
                    "result is approximate near the block seams. Raise the "
                    "memory budget (optix.config(memory=...)) or pass "
                    "block=.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            blk = self._prepare(chunk, maps)
            op = self._block_operator(psf, chunk, tuple(blk.data.shape), ops)
            f, n = self._solve(blk, op, self._block_input(chunk), trace)
            iterations.append(float(n))
            self._after_block(trace)
            chunk.store(out, op.crop(f)[chunk.inner])
        flush(out)
        return out

    def _extra_state(self) -> dict[str, object]:
        psf = self.psf
        if isinstance(psf, ShiftVariantPSF):
            psf = psf._payload()
        return {"psf": psf}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        psf = state.get("psf")
        if isinstance(psf, dict):
            psf = ShiftVariantPSF._from_payload(psf)
        self.psf = psf
