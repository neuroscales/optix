"""Biggs–Andrews iterations over halo blocks in lockstep, with a global α.

Design
------
The iterate of one image lives on its extended object domain E, the
image plus the margin of the PSF support (K − 1 pixels per axis, of which
`K − 1 − K // 2` before the image and `K // 2` after it): exactly the
pixels that the whole-image iteration updates. Three full-size buffers
hold the last three iterates, xₙ, xₙ₋₁ and xₙ₋₂, in rotation.

Each iteration is one pass over the blocks of the image. A block reads
the data of its core + halo, and the window of xₙ (and of xₙ₋₁ when the
prediction needs it) that covers that region plus the PSF margin, then
makes one Richardson–Lucy step on its own padded grid. Because the halo
is at least the PSF size, the step is exact on the pixels the block owns
(its core, extended into the outer margin of E along the image edges):
the blurred estimate, the ratio and the correlation back all involve
pixels that the block holds. The owned pixels of xₙ₊₁ are written into
the buffer of xₙ₋₂, which no other block reads during this pass, so
every block sees its neighbours' current iterates. The owned regions
partition E, and the result is the whole-image iteration up to FFT
round-off.

During the pass, each block also accumulates over its owned pixels, in
float64, ∑ gₙ·gₙ and ∑ gₙ·gₙ₋₁ (gₙ₋₁ = xₙ − yₙ₋₁ is rebuilt from the
buffers, so no change is stored), and the χ² statistic of xₙ over its
data core. After the pass, α and the χ² stopping test are evaluated once
for the image, exactly as on a single block.
"""

import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..io import Chunk, Chunks, WritableArrayLike
from ..io._arrays import _numpy_dtype
from ._fft import PaddedConvolution
from ._rl import _expected, rl_step

__all__ = ["lockstep"]


def _alpha(num: float, den: float) -> float:
    """αₖ of Eq. 10 from its sums, clipped to [0, 1]."""
    if not den > 0 or not num > 0:
        return 0.0
    return min(num / den, 1.0)


def _chi2_tail(chi2: float, n: int) -> float:
    """P(χ² > chi2) for ν = n − 1 degrees of freedom."""
    nu = torch.tensor(max(n - 1, 1), dtype=torch.float64)
    value = torch.tensor(chi2, dtype=torch.float64)
    return float(torch.special.gammaincc(nu / 2, value / 2))


def _dot(a: Tensor, b: Tensor) -> float:
    return float((a * b).sum(dtype=torch.float64))


class _State:
    """Three iterates on the extended domain: tensors, or memmaps on disk.

    The buffers are held in memory (on the working device) when they fit
    the memory budget, else in a temporary directory removed by `close`.
    """

    def __init__(
        self, shape: tuple[int, ...], backend: dict, memory: int
    ) -> None:
        self.backend = backend
        nbytes = 3 * math.prod(shape) * backend["dtype"].itemsize
        self._tmp: tempfile.TemporaryDirectory | None = None
        if nbytes <= memory:
            self.kind = "memory"
            self.buffers: list = [torch.zeros(shape, **backend) for _ in "abc"]
        else:
            self.kind = "disk"
            self._tmp = tempfile.TemporaryDirectory(prefix="optix-ba-")
            dtype = _numpy_dtype(backend["dtype"])
            self.buffers = [
                np.lib.format.open_memmap(
                    f"{self._tmp.name}/x{i}.npy",
                    mode="w+",
                    dtype=dtype,
                    shape=shape,
                )
                for i in range(3)
            ]

    def read(self, i: int, index: tuple[slice, ...]) -> Tensor:
        value = self.buffers[i % 3][index]
        if isinstance(value, Tensor):
            return value.clone()
        return torch.from_numpy(np.array(value)).to(**self.backend)

    def write(self, i: int, index: tuple[slice, ...], value: Tensor) -> None:
        buf = self.buffers[i % 3]
        if isinstance(buf, Tensor):
            buf[index] = value
        else:
            buf[index] = value.detach().cpu().numpy()

    def close(self) -> None:
        self.buffers = []
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None


@dataclass
class _Geometry:
    """Where one block sits in the extended domain E."""

    window: tuple[slice, ...]  # region of E loaded into the block's grid
    owned: tuple[slice, ...]  # region of E this block updates
    local: tuple[slice, ...]  # `owned` inside the window
    core: tuple[slice, ...]  # data core, in E coordinates
    before: tuple[int, ...]  # margin of E before the image, per axis

    @classmethod
    def of(
        cls,
        region: tuple[slice, ...],
        core: tuple[slice, ...],
        extent: tuple[int, ...],
        psf: tuple[int, ...],
    ) -> "_Geometry":
        window, owned, local, core_e, before = [], [], [], [], []
        for r, c, n, k in zip(region, core, extent, psf, strict=True):
            m = k - 1 - k // 2
            lo = 0 if c.start == 0 else c.start + m
            hi = n + k - 1 if c.stop == n else c.stop + m
            window.append(slice(r.start, r.stop + k - 1))
            owned.append(slice(lo, hi))
            local.append(slice(lo - r.start, hi - r.start))
            core_e.append(slice(c.start + m, c.stop + m))
            before.append(m)
        return cls(
            tuple(window),
            tuple(owned),
            tuple(local),
            tuple(core_e),
            tuple(before),
        )

    def to_grid(self, w: Tensor, grid: tuple[int, ...]) -> Tensor:
        """Place a window of E on the block's periodic object grid."""
        out = w.new_zeros(grid)
        out[tuple(slice(0, s) for s in w.shape)] = w
        dims = tuple(range(out.ndim))
        return torch.roll(out, tuple(-m for m in self.before), dims)

    def from_grid(self, f: Tensor) -> Tensor:
        """The owned pixels of an object on the block's grid."""
        dims = tuple(range(f.ndim))
        return torch.roll(f, self.before, dims)[self.local]


def _images(chunks: Chunks) -> dict[tuple, list[tuple[int, Chunk, tuple]]]:
    """Blocks of each image, without reading the data.

    Returns, per image (group + sample position), the position of each
    block in `chunks`, a data-less chunk that can store results, and the
    loaded spatial region.
    """
    view = chunks._view
    images: dict[tuple, list] = {}
    for i, plan in enumerate(chunks._plans):
        samples, _ = view.arrange(plan)
        handle = Chunk(view, plan, torch.empty(0), view.inner(plan), samples)
        key = handle.group + tuple(s.start for s in handle.samples)
        region = tuple(plan.read[a] for a in view.spatial)
        images.setdefault(key, []).append((i, handle, region))
    return images


@dataclass
class _Sums:
    gg: float = 0.0  # ∑ gₙ·gₙ
    cross: float = 0.0  # ∑ gₙ·gₙ₋₁
    chi2: float = 0.0
    clipped: bool = False


@dataclass
class _Image:
    """The blocks of one image and the shared iterate buffers."""

    start: float
    loaded: Chunks
    geos: list[_Geometry]
    prepare: Callable
    operator: Callable
    state: _State
    chi2: bool
    damping: float

    def sweep(self, n: int, a: float, a_prev: float) -> _Sums:
        """Pass n over the blocks: xₙ₊₁ from yₙ, and the sums for αₙ₊₁."""
        state = self.state
        sums = _Sums()
        for chunk, geo in zip(self.loaded, self.geos, strict=True):
            blk = self.prepare(chunk)
            op, valid, inv = self.operator(tuple(blk.data.shape))
            xn = None
            if n == 0:
                f = torch.zeros_like(inv).masked_fill_(valid, self.start)
                state.write(0, geo.owned, geo.from_grid(f))
            else:
                xn = state.read(n, geo.window)
                f = geo.to_grid(xn, op.grid)
            fp = None
            if a > 0:
                fp = state.read(n - 1, geo.window)
                y = f + a * (f - geo.to_grid(fp, op.grid))
                if (y < 0).any():
                    sums.clipped = True
                    y.clamp_(min=0)
            else:
                y = f
            mu = None
            if self.chi2:
                mu = _expected(f, op, blk.background)
                d, m = blk.data[chunk.inner], mu[chunk.inner]
                chi2 = ((d - m).square() / m).sum(dtype=torch.float64)
                sums.chi2 += float(chi2)
                if a > 0:
                    mu = None
            new = rl_step(
                y,
                blk.data,
                op,
                inv_norm=inv,
                background=blk.background,
                damping=self.damping,
                mu=mu,
            )
            owned = geo.from_grid(new)
            if n >= 1:  # gₙ = ψ(yₙ) − yₙ; g₀ is not in the history
                g = owned - geo.from_grid(y)
                sums.gg += _dot(g, g)
            if n >= 2:  # gₙ₋₁ = xₙ − yₙ₋₁, rebuilt from the buffers
                if fp is not None:
                    x1 = fp[geo.local]
                else:
                    x1 = state.read(n - 1, geo.owned)
                if a_prev > 0:
                    x2 = state.read(n - 2, geo.owned)
                    x1 = (x1 + a_prev * (x1 - x2)).clamp_(min=0)
                sums.cross += _dot(g, xn[geo.local] - x1)
            # xₙ₊₁ replaces xₙ₋₂, read above and by no other block
            state.write(n + 1, geo.owned, owned)
        return sums


def lockstep(
    chunks: Chunks,
    out: WritableArrayLike,
    *,
    prepare: Callable,
    psf: Tensor,
    level: Tensor,
    n_iter: int | None,
    alpha: float,
    max_iter: int,
    damping: float,
    memory: int,
    backend: dict,
) -> list[dict]:
    """Biggs–Andrews iterations with one α per image and iteration.

    Parameters
    ----------
    chunks : Chunks
        Blocks with halos ([`blocks`][optix.io.blocks], no padding).
    out : writable array-like
        Output, shaped like the source.
    prepare : callable
        `prepare(chunk)` returns the block's counts and background.
    psf : Tensor
        Normalised PSF.
    level : Tensor
        Start level per image, `(*group, *sample)`.
    n_iter, alpha, max_iter, damping
        See [`rl_block`][optix.deconv.utils.rl_block].
    memory : int
        Budget in bytes above which the iterates are kept on disk.
    backend : dict
        Working dtype and device.

    Returns
    -------
    list of dict
        Per image: `"iterations"`, `"effective"`, `"alpha"`, `"clipped"`
        and `"state"` (`"memory"` or `"disk"`).
    """
    extent = tuple(chunks.spatial_shape)
    kernel = tuple(psf.shape)
    domain = tuple(n + k - 1 for n, k in zip(extent, kernel, strict=True))
    npix = math.prod(extent)
    limit = max_iter if n_iter is None else n_iter
    ops: dict[tuple[int, ...], tuple] = {}

    def operator(shape: tuple[int, ...]) -> tuple:
        if shape not in ops:
            op = PaddedConvolution(psf, shape)
            norm = op.norm
            valid = norm > norm.max() * torch.finfo(norm.dtype).eps * 16
            tiny = torch.finfo(norm.dtype).tiny
            inv = torch.where(valid, 1 / norm.clamp(min=tiny), 0)
            ops[shape] = (op, valid, inv)
        return ops[shape]

    reports = []
    state = _State(domain, backend, memory)
    try:
        for key, parts in _images(chunks).items():
            image = _Image(
                start=float(level[key]),
                loaded=chunks.select([i for i, _, _ in parts]),
                geos=[
                    _Geometry.of(region, handle.spatial, extent, kernel)
                    for _, handle, region in parts
                ],
                prepare=prepare,
                operator=operator,
                state=state,
                chi2=n_iter is None,
                damping=damping,
            )
            alphas: list[float] = []
            gg: list[float] = []  # ∑ gₘ·gₘ, m ≥ 1
            cross: list[float] = []  # ∑ gₘ·gₘ₋₁, m ≥ 2
            clipped = 0
            effective = 0.0
            eta = 1.0
            n = 0
            while n < limit:
                a = gamma = 0.0
                if n >= 3:
                    num, den = cross[n - 3], gg[n - 3]
                    a = _alpha(num, den)
                    if a > 0:
                        # γₙ = (αₙ βₙ)², βₙ = ∑ gₙ₋₁·gₙ₋₂ / ∑ gₙ₋₁·gₙ₋₁
                        d1 = gg[n - 2]
                        gamma = (a * num / d1) ** 2 if d1 > 0 else 0.0
                sums = image.sweep(n, a, alphas[-1] if alphas else 0.0)
                if n_iter is None and _chi2_tail(sums.chi2, npix) >= alpha:
                    break  # xₙ, still in its buffer, is the result
                if n >= 1:
                    gg.append(sums.gg)
                if n >= 2:
                    cross.append(sums.cross)
                clipped += sums.clipped
                alphas.append(a)
                eta = 1.0 + gamma * eta
                effective += eta
                n += 1
            for (_, handle, _), geo in zip(parts, image.geos, strict=True):
                if limit == 0:
                    shape = tuple(s.stop - s.start for s in geo.core)
                    value = torch.full(shape, image.start, **backend)
                else:
                    value = state.read(n, geo.core)
                handle.store(out, value)
            reports.append(
                {
                    "iterations": n,
                    "effective": effective,
                    "alpha": alphas,
                    "clipped": clipped,
                    "state": state.kind,
                }
            )
    finally:
        state.close()
    return reports
