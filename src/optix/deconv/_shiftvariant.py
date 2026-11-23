"""Shift-variant blur: sums of convolutions with spatial weights."""

import itertools
import math
from collections.abc import Callable, Iterator, Sequence

import torch
from torch import Tensor

from .._config import get_config
from ._fft import next_fast_len

__all__ = [
    "DenisPSF",
    "NagyOLearyPSF",
    "ShiftVariantConvolution",
    "ShiftVariantPSF",
    "interpolation_weights",
    "object_positions",
]

_KINDS = ("nearest", "simplex", "multilinear")


def _grid(shape: Sequence[int], kernel: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        next_fast_len(n + k - 1) for n, k in zip(shape, kernel, strict=True)
    )


def object_positions(
    shape: Sequence[int],
    kernel: Sequence[int],
    *,
    device: torch.device | str | None = None,
) -> tuple[Tensor, ...]:
    """Position of each object pixel of a padded block, per axis.

    The object grid of
    [`PaddedConvolution`][optix.deconv.utils.PaddedConvolution] and
    [`ShiftVariantConvolution`][optix.deconv.utils.ShiftVariantConvolution]
    has N ≥ L + K − 1 pixels per axis. Index q < L is the block pixel q;
    the margin is stored periodically: the K − 1 − K // 2 pixels before
    the block at the end of the grid, the K // 2 pixels after it right
    after the block. Pixels that reach no block pixel are given positions
    after the block.

    Parameters
    ----------
    shape : sequence of int
        Block shape L.
    kernel : sequence of int
        Kernel shape K.
    device : optional
        Device of the result.

    Returns
    -------
    tuple of Tensor
        Per axis, the position (float64, relative to the first block
        pixel) of each of the N object pixels.
    """
    out = []
    for k, m in zip(kernel, _grid(shape, kernel), strict=True):
        before = k - 1 - k // 2
        q = torch.arange(m, dtype=torch.float64, device=device)
        out.append(torch.where(q < m - before, q, q - m))
    return tuple(out)


def _axis_cells(centres: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    """Cell index and local coordinate t ∈ [0, 1] of positions `x`."""
    g = centres.numel()
    if g == 1:
        return torch.zeros_like(x, dtype=torch.long), torch.zeros_like(x)
    cell = torch.searchsorted(centres, x.contiguous(), right=True) - 1
    cell = cell.clamp(0, g - 2)
    lo, hi = centres[cell], centres[cell + 1]
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return cell, t


def interpolation_weights(
    centres: Sequence[Tensor],
    coords: Sequence[Tensor],
    kind: str,
) -> tuple[Tensor, Tensor]:
    """Interpolation weights of a rectilinear grid of nodes.

    The nodes sit at the outer product of `centres` (one increasing 1-D
    tensor per axis). A position x falls in the grid cell
    [cᵢ, cᵢ₊₁] along each axis, with local coordinate
    tₐ = (xₐ − cᵢ)/(cᵢ₊₁ − cᵢ); positions outside the grid are clamped
    to it (constant extrapolation). With δ ∈ {0, 1}ᵈ the corner of the
    cell, the weight of that node is

    - `"nearest"`: 1 for the corner nearest to x along every axis
      (tₐ ≥ ½ rounds up), else 0: piecewise-constant regions with
      boundaries half-way between centres;
    - `"simplex"`: barycentric weight in the Kuhn (Freudenthal)
      triangulation of the cell, which splits it along its main
      diagonal, max(0, min_{δₐ=1} tₐ − max_{δₐ=0} tₐ) (empty min = 1,
      empty max = 0): piecewise-linear interpolation on simplices;
    - `"multilinear"`: ∏ₐ (tₐ if δₐ = 1 else 1 − tₐ).

    All three are non-negative and sum to 1 at every position.

    Parameters
    ----------
    centres : sequence of Tensor
        Node positions per axis, increasing.
    coords : sequence of Tensor
        Positions per axis (an open grid: the weights are evaluated at
        their outer product), in the same units as `centres`.
    kind : {"nearest", "simplex", "multilinear"}
        Interpolation scheme.

    Returns
    -------
    nodes : Tensor
        Multi-indices `(T, d)` of the T nodes with a non-zero weight
        somewhere on the positions.
    weights : Tensor
        Their weights, `(T, *n)` with n the lengths of `coords`
        (float64).
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    d = len(centres)
    if len(coords) != d:
        raise ValueError("coords must have one entry per axis")
    cells, ts = [], []
    for c, x in zip(centres, coords, strict=True):
        cell, t = _axis_cells(c.to(x), x.double())
        cells.append(cell)
        ts.append(t)
    sizes = [c.numel() for c in centres]
    ranges = [
        range(int(cell.min()), min(int(cell.max()) + 1, g - 1) + 1)
        for cell, g in zip(cells, sizes, strict=True)
    ]
    shape = tuple(x.numel() for x in coords)

    def bcast(v: Tensor, a: int) -> Tensor:
        view = [1] * d
        view[a] = -1
        return v.reshape(view)

    nodes, maps = [], []
    for node in itertools.product(*ranges):
        delta = [cell.neg() + v for cell, v in zip(cells, node, strict=True)]
        if kind == "simplex":
            ok = torch.ones((), dtype=torch.bool, device=coords[0].device)
            lo = torch.ones((), dtype=torch.float64, device=coords[0].device)
            hi = torch.zeros_like(lo)
            for a in range(d):
                da, ta = bcast(delta[a], a), bcast(ts[a], a)
                ok = ok & ((da == 0) | (da == 1))
                lo = torch.minimum(lo, torch.where(da == 1, ta, 1.0))
                hi = torch.maximum(hi, torch.where(da == 0, ta, 0.0))
            w = torch.where(ok, (lo - hi).clamp(min=0), 0.0)
            w = torch.broadcast_to(w, shape)
        else:
            w = None
            for a in range(d):
                da, ta = delta[a], ts[a]
                if kind == "nearest":
                    # the nearest node along the axis is cell + round(t)
                    near = (ta >= 0.5).long() * int(sizes[a] > 1)
                    f = (da == near).double()
                else:
                    f = torch.where(
                        da == 1, ta, torch.where(da == 0, 1 - ta, 0.0)
                    )
                f = bcast(f, a)
                w = f if w is None else w * f
            w = torch.broadcast_to(w, shape)
        if bool((w > 0).any()):
            nodes.append(node)
            maps.append(w)
    device = coords[0].device
    if not maps:  # cannot happen: weights sum to one
        raise RuntimeError("no interpolation node covers the positions")
    return (
        torch.tensor(nodes, dtype=torch.long, device=device).reshape(-1, d),
        torch.stack(maps),
    )


class ShiftVariantConvolution:
    """Shift-variant blur operator H of a block, a sum of convolutions.

    The blur is a sum of T convolutions with kernels kₜ and spatial
    weights wₜ, applied either before the convolutions (on the object,
    `order="input"`) or after them (on the blurred image,
    `order="output"`):

        input:   H f = ∑ₜ kₜ ∗ (wₜ ⊙ f),
        output:  H f = ∑ₜ wₜ ⊙ (kₜ ∗ f).

    As in [`PaddedConvolution`][optix.deconv.utils.PaddedConvolution],
    the object lives on a padded grid of shape N ≥ L + K − 1 (rounded up
    to 2ᵃ3ᵇ5ᶜ), the convolutions are circular on that grid and the result
    is cropped to the block, so the circular and linear models agree on
    the block. The adjoint is the exact transpose,

        input:   Hᵀ r = ∑ₜ wₜ ⊙ (kₜ ⋆ r),
        output:  Hᵀ r = ∑ₜ kₜ ⋆ (wₜ ⊙ r),

    with ⋆ the correlation (multiplication by the conjugate transfer
    function). With a single term and w = 1 it is `PaddedConvolution`.
    Kernels are centred at index `K // 2`.

    Attributes
    ----------
    shape : tuple of int
        Block shape L.
    grid : tuple of int
        Padded object grid N.
    order : str
        `"input"` or `"output"`.
    weights : Tensor
        `(T, *N)` (input) or `(T, *L)` (output).
    norm : Tensor
        Hᵀ1 on the object grid.
    """

    def __init__(
        self,
        kernels: Tensor,
        weights: Tensor | Callable[[tuple[Tensor, ...]], Tensor],
        shape: Sequence[int],
        *,
        order: str = "input",
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Precompute the transfer functions and the normaliser.

        Parameters
        ----------
        kernels : Tensor
            `(T, *K)`, used as given.
        weights : Tensor or callable
            `(T, *N)` for `order="input"` (on the object grid, see
            [`object_positions`][optix.deconv.utils.object_positions]),
            `(T, *L)` for `order="output"`; or a callable mapping the
            per-axis positions of those pixels (a tuple of 1-D float64
            tensors, relative to the first block pixel) to that tensor.
        shape : sequence of int
            Block shape L.
        order : {"input", "output"}
            Where the weights are applied.
        dtype, device : optional
            Working backend (default: those of `kernels`).
        """
        if order not in ("input", "output"):
            raise ValueError(f"order must be 'input' or 'output': {order!r}")
        shape = tuple(int(s) for s in shape)
        ndim = kernels.ndim - 1
        if len(shape) != ndim:
            raise ValueError(
                f"Block has {len(shape)} dims but the kernels have {ndim}"
            )
        dtype = kernels.dtype if dtype is None else dtype
        device = kernels.device if device is None else device
        kernels = kernels.to(dtype=dtype, device=device)
        ksize = tuple(kernels.shape[1:])
        self.ndim = ndim
        self.shape = shape
        self.order = order
        self.grid = _grid(shape, ksize)
        self._dims = tuple(range(-ndim, 0))
        if callable(weights):
            if order == "input":
                coords = object_positions(shape, ksize, device=device)
            else:
                coords = tuple(
                    torch.arange(n, dtype=torch.float64, device=device)
                    for n in shape
                )
            weights = weights(coords)
        weights = torch.as_tensor(weights).to(dtype=dtype, device=device)
        expect = (kernels.shape[0],) + (
            self.grid if order == "input" else shape
        )
        if tuple(weights.shape) != expect:
            raise ValueError(
                f"weights must have shape {expect}, got {tuple(weights.shape)}"
            )
        self.weights = weights
        buf = kernels.new_zeros((kernels.shape[0],) + self.grid)
        buf[(slice(None),) + tuple(slice(0, k) for k in ksize)] = kernels
        buf = torch.roll(
            buf, shifts=tuple(-(k // 2) for k in ksize), dims=self._dims
        )
        self.otf = torch.fft.rfftn(buf, dim=self._dims)
        self.norm = self.adjoint(kernels.new_ones(shape))

    def pad(self, x: Tensor) -> Tensor:
        """Zero-pad a block `(..., *L)` to the grid `(..., *N)`."""
        pad = []
        for n, m in reversed(list(zip(self.shape, self.grid, strict=True))):
            pad += [0, m - n]
        return torch.nn.functional.pad(x, pad)

    def crop(self, x: Tensor) -> Tensor:
        """Restrict an object `(..., *N)` to the block `(..., *L)`."""
        lead = (slice(None),) * (x.ndim - self.ndim)
        return x[lead + tuple(slice(0, n) for n in self.shape)]

    def _irfft(self, spec: Tensor) -> Tensor:
        return torch.fft.irfftn(spec, s=self.grid, dim=self._dims)

    def forward(self, f: Tensor) -> Tensor:
        """H f: object `(..., *N)` to blurred block `(..., *L)`."""
        if self.order == "input":
            spec = None
            for w, otf in zip(self.weights, self.otf, strict=True):
                term = torch.fft.rfftn(w * f, dim=self._dims) * otf
                spec = term if spec is None else spec + term
            return self.crop(self._irfft(spec))
        fs = torch.fft.rfftn(f, dim=self._dims)
        out = None
        for w, otf in zip(self.weights, self.otf, strict=True):
            term = w * self.crop(self._irfft(fs * otf))
            out = term if out is None else out + term
        return out

    def adjoint(self, r: Tensor) -> Tensor:
        """Hᵀ r: block `(..., *L)` to object `(..., *N)`."""
        if self.order == "input":
            rs = torch.fft.rfftn(self.pad(r), dim=self._dims)
            out = None
            for w, otf in zip(self.weights, self.otf, strict=True):
                term = w * self._irfft(rs * otf.conj())
                out = term if out is None else out + term
            return out
        spec = None
        for w, otf in zip(self.weights, self.otf, strict=True):
            term = torch.fft.rfftn(self.pad(w * r), dim=self._dims)
            term = term * otf.conj()
            spec = term if spec is None else spec + term
        return self._irfft(spec)


def _normalise(psf: Tensor, d: int) -> Tensor:
    """Each kernel `(..., *K)` divided by its sum."""
    dims = tuple(range(-d, 0))
    total = psf.sum(dim=dims, keepdim=True)
    if not bool((total > 0).all()):
        raise ValueError("every PSF must have a positive sum")
    return psf / total


def _check_grid(
    psf: Tensor, centres: Sequence[Tensor | Sequence[float]], ndim: int | None
) -> tuple[Tensor, list[Tensor], int]:
    centres = [
        torch.as_tensor(c, dtype=torch.float64).reshape(-1) for c in centres
    ]
    d = len(centres)
    if d not in (2, 3):
        raise ValueError(f"centres must give 2 or 3 axes, got {d}")
    if ndim is not None and ndim != d:
        raise ValueError(f"expected {ndim} axes of centres, got {d}")
    psf = torch.as_tensor(psf)
    if psf.ndim != 2 * d:
        raise ValueError(
            f"psf must be (*grid, *K) with {d} grid and {d} kernel dims, "
            f"got shape {tuple(psf.shape)}"
        )
    grid = tuple(psf.shape[:d])
    for a, c in enumerate(centres):
        if c.numel() != grid[a]:
            raise ValueError(
                f"axis {a}: {c.numel()} centres for a grid of {grid[a]} PSFs"
            )
        if c.numel() > 1 and not bool((c[1:] > c[:-1]).all()):
            raise ValueError(f"axis {a}: centres must be increasing")
    if (psf < 0).any():
        raise ValueError("psf must be non-negative")
    return psf, centres, d


class ShiftVariantPSF:
    """A shift-variant PSF model: builds the blur operator of any block.

    Base class of the models
    [`NagyOLearyPSF`][optix.deconv.NagyOLearyPSF] and
    [`DenisPSF`][optix.deconv.DenisPSF]. A model can be given as the
    `psf` of [`RichardsonLucy`][optix.deconv.RichardsonLucy],
    [`BiggsAndrews`][optix.deconv.BiggsAndrews] and
    [`Snyder`][optix.deconv.Snyder]. The weights of each block are
    evaluated at the global positions of its pixels, so the operator of
    a block is the restriction of the whole-image operator, whatever the
    blocking.

    Attributes
    ----------
    ndim : int
        Number of spatial dims d.
    shape : tuple of int
        Kernel shape K (the support of every PSF of the model).
    order : str
        `"input"` (weights on the object) or `"output"` (on the image).
    kernels : Tensor
        Kernels of the terms, `(P, *K)`.
    """

    order: str = "input"
    kernels: Tensor
    centres: list[Tensor]

    @property
    def ndim(self) -> int:
        return self.kernels.ndim - 1

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.kernels.shape[1:])

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(c.numel() for c in self.centres)

    def _flat(self, nodes: Tensor) -> Tensor:
        strides = []
        s = 1
        for g in reversed(self.grid_shape):
            strides.append(s)
            s *= g
        strides = torch.tensor(strides[::-1], device=nodes.device)
        return (nodes * strides).sum(-1)

    def terms(self, coords: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        """Kernels `(T, *K)` and weights `(T, *n)` at global positions."""
        raise NotImplementedError

    def _axis_kind(self) -> str | None:
        """Per-axis weight kind bounding the terms (None: all kernels)."""
        return "multilinear"

    def n_terms(self, extent: Sequence[int]) -> int:
        """Upper bound on the number of terms T of any block of an image.

        It counts the nodes whose weights are non-zero somewhere on the
        image of shape `extent` or within one kernel size of it (the
        object grid of a block): at most the number of PSFs on the grid,
        or the number of kernels for models with full-field weights.
        Blocks, and the memory they take, are sized from it.

        Parameters
        ----------
        extent : sequence of int
            Image shape.

        Returns
        -------
        int
        """
        kind = self._axis_kind()
        if kind is None:
            return int(self.kernels.shape[0])
        total = 1
        for c, n, k in zip(self.centres, extent, self.shape, strict=True):
            x = torch.arange(-k, int(n) + k, dtype=torch.float64)
            nodes, _ = interpolation_weights([c.cpu()], [x], kind)
            total *= nodes.shape[0]
        return total

    def operator(
        self,
        origin: Sequence[int],
        shape: Sequence[int],
    ) -> ShiftVariantConvolution:
        """Blur operator of the block of shape `shape` at `origin`.

        Parameters
        ----------
        origin : sequence of int
            Global position of the first block pixel, per spatial axis.
        shape : sequence of int
            Block shape L.

        Returns
        -------
        ShiftVariantConvolution
        """
        shape = tuple(int(s) for s in shape)
        device = self.kernels.device
        if self.order == "input":
            local = object_positions(shape, self.shape, device=device)
        else:
            local = tuple(
                torch.arange(n, dtype=torch.float64, device=device)
                for n in shape
            )
        coords = tuple(
            x + float(o) for x, o in zip(local, origin, strict=True)
        )
        kernels, weights = self.terms(coords)
        return ShiftVariantConvolution(
            kernels, weights, shape, order=self.order
        )

    def to(self, **backend: object) -> "ShiftVariantPSF":
        """A copy with kernels on another dtype / device."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new._convert(**backend)
        return new

    def _convert(self, **backend: object) -> None:
        self.kernels = self.kernels.to(**backend)
        device = backend.get("device")
        if device is not None:
            self.centres = [c.to(device) for c in self.centres]

    def _payload(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def _from_payload(payload: dict) -> "ShiftVariantPSF":
        kind = payload["model"]
        if kind == "nagy":
            return NagyOLearyPSF(
                payload["psf"],
                payload["centres"],
                interpolation=payload["interpolation"],
                normalize=False,
            )
        if kind == "denis" and payload["kind"] == "optimal":
            return DenisPSF._restore_optimal(payload)
        if kind == "denis":
            return DenisPSF(
                payload["psf"],
                payload["centres"],
                model=payload["kind"],
                rank=payload["rank"],
                normalize=False,
            )
        raise ValueError(f"unknown shift-variant PSF payload {kind!r}")


class NagyOLearyPSF(ShiftVariantPSF):
    """Shift-variant blur of Nagy & O'Leary (1998): convolve, then weight.

    PSFs kᵢ are given at p points sᵢ of a rectangular grid. The blur is

        K = D₁ K₁ + D₂ K₂ + ⋯ + Dₚ Kₚ   (Eq. 2.1),

    Kᵢ the convolution with kᵢ and Dᵢ a non-negative diagonal matrix
    with ∑ Dᵢ = I, applied to the blurred images: each pixel of the
    image is a weighted sum of the images blurred by each PSF. With
    `interpolation="constant"` (§3.2.1), Dᵢ is the indicator of the
    region of PSF i: the image is partitioned into rectangles, one per
    PSF, with boundaries half-way between neighbouring centres. With
    `interpolation="linear"` (§3.2.2), the weights interpolate linearly
    between the PSF centres on a triangulation of the grid: each
    rectangle of four neighbouring centres is split into two triangles
    along the diagonal from its first corner (lowest index along every
    axis) to the opposite one; in 3-D, the Kuhn triangulation of each
    box into six tetrahedra. The paper only asks for a triangulation;
    its Fig. 3.2 draws the other diagonal, and either is valid. Beyond
    the outermost centres the weights are constant.
    The paper recommends the piecewise-constant model where linear
    interpolation brings no gain (§5).

    Denis et al. (2015) show that this model is not the blur of an
    interpolated PSF: see [`DenisPSF`][optix.deconv.DenisPSF].

    !!! references "References"

        1. Nagy JG, O'Leary DP. *Restoring images degraded by spatially
           variant blur.* SIAM J Sci Comput **19**(4):1063-1082, 1998.
           [doi:10.1137/S106482759528507X](https://doi.org/10.1137/S106482759528507X)

        ??? quote "BibTeX"

            ```bibtex
            @article{nagy1998restoring,
              author  = {Nagy, James G. and O'Leary, Dianne P.},
              title   = {Restoring Images Degraded by Spatially Variant
                         Blur},
              journal = {SIAM Journal on Scientific Computing},
              volume  = {19},
              number  = {4},
              pages   = {1063--1082},
              year    = {1998},
              doi     = {10.1137/S106482759528507X}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import NagyOLearyPSF
    >>> psf = torch.ones(2, 2, 5, 5)  # 2 × 2 grid of 5 × 5 PSFs
    >>> model = NagyOLearyPSF(psf, [[16, 48], [16, 48]])
    >>> model.operator((0, 0), (64, 64)).shape
    (64, 64)
    """

    order = "output"

    def __init__(
        self,
        psf: Tensor,
        centres: Sequence[Tensor | Sequence[float]],
        *,
        interpolation: str = "constant",
        normalize: bool = True,
    ) -> None:
        """Set up the model.

        Parameters
        ----------
        psf : Tensor
            PSFs `(*G, *K)`: a G grid of d-dimensional PSFs (d = 2 or 3,
            spatial axes in layout order), each centred at index
            `K // 2`.
        centres : sequence of 1-D tensors
            Position (in pixels, along each spatial axis) of the grid
            nodes: PSF `psf[i, j, ...]` is measured at
            `(centres[0][i], centres[1][j], ...)`.
        interpolation : {"constant", "linear"}
            Weights Dᵢ: region indicators (§3.2.1) or piecewise-linear
            interpolation on a triangulation (§3.2.2). Default:
            `"constant"`.
        normalize : bool
            Divide each PSF by its sum (unit-sum PSFs, as in §5).
        """
        if interpolation not in ("constant", "linear"):
            raise ValueError(
                "interpolation must be 'constant' or 'linear': "
                f"{interpolation!r}"
            )
        psf, centres, d = _check_grid(psf, centres, None)
        if not psf.is_floating_point():
            psf = psf.double()
        if normalize:
            psf = _normalise(psf, d)
        self.centres = centres
        self.interpolation = interpolation
        self.kernels = psf.reshape((-1,) + tuple(psf.shape[d:]))

    def _axis_kind(self) -> str | None:
        return "nearest" if self.interpolation == "constant" else "multilinear"

    def terms(self, coords: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        kind = "nearest" if self.interpolation == "constant" else "simplex"
        nodes, maps = interpolation_weights(self.centres, coords, kind)
        return self.kernels[self._flat(nodes)], maps

    def regions(self, coords: tuple[Tensor, ...]) -> list[tuple[int, tuple]]:
        """Piecewise-constant regions met by an open grid of positions.

        Returns, per region, the flat index of its PSF and the slices
        (into `coords`) of its intersection with the positions.
        """
        per_axis = []
        for c, x in zip(self.centres, coords, strict=True):
            cell, t = _axis_cells(c.to(x), x)
            near = cell + ((t >= 0.5) & (c.numel() > 1)).long()
            runs = []
            start = 0
            values = near.tolist()
            for i in range(1, len(values) + 1):
                if i == len(values) or values[i] != values[start]:
                    runs.append((values[start], slice(start, i)))
                    start = i
            per_axis.append(runs)
        out = []
        for combo in itertools.product(*per_axis):
            node = torch.tensor([[v for v, _ in combo]])
            out.append((int(self._flat(node)[0]), tuple(s for _, s in combo)))
        return out

    def _payload(self) -> dict:
        grid = self.grid_shape
        return {
            "model": "nagy",
            "psf": self.kernels.reshape(grid + self.shape),
            "centres": list(self.centres),
            "interpolation": self.interpolation,
        }


class DenisPSF(ShiftVariantPSF):
    """Shift-variant blur of Denis et al. (2015): weight, then convolve.

    The un-shifted PSF at each object location j (the column K⋅ⱼ of the
    PSF matrix K) is approximated by a few terms, K ≈ ∑ₚ cₚ wₚᵀ, which
    gives the blur as a sum of convolutions of weighted objects,

        H ≈ ∑ₚ conv(cₚ) · diag(wₚ)   (Eq. 14).

    Three approximations of the paper are available (`model`):

    - `"interpolation"` (§2.3, Eqs. 21–22): cₚ = kₚ are the PSFs sampled
      on a grid and wₚ = φₚ the weights of their bilinear (trilinear in
      3-D) interpolation, so the PSF at every location is the linear
      interpolation of the sampled PSFs. It preserves the positivity,
      the normalisation and the symmetries of the PSFs (§3.2).
    - `"modes"` (§2.4; the practical scheme of Flicker & Rigaut 2005):
      PSF modes cₚ = uₚ, the first `rank` left singular vectors of the
      matrix of the G *sampled* PSFs (L × G), with coefficients σₚ vₚ
      at the nodes, interpolated between nodes with the bilinear weights
      φ. This departs from Eqs. 26–28, which take the SVD of K, the PSFs
      of every pixel (L × M), with no interpolation. With all modes it
      equals `"interpolation"`. The truncation is the best rank-`rank`
      approximation of the samples only, with error ∑_{p > rank} σₚ²
      on them (Eckart–Young), not of the PSFs over the whole field.
    - `"optimal"` (§2.5, Eqs. 29–38): the optimal local approximation
      proposed by the paper. The weights keep the support of the bilinear
      weights, and both the kernels and the weights minimise
      ‖K − C Wᵀ‖²_F, the error on the PSFs of every pixel of the image,
      by alternating C* = K W (WᵀW)⁻¹ (Eq. 35) and, per pixel,
      ω*ⱼ = (C′ⱼᵀC′ⱼ)⁻¹C′ⱼᵀK⋅ⱼ over the 2ᵈ nodes of its grid cell
      (Eq. 37; C′ⱼ is the same for every pixel of a cell, so its
      pseudo-inverse is computed once per cell), starting from the
      interpolation model; `n_alt` = 10 alternations, as in the paper.
      It needs a `psf_model` that gives the PSF at any position, and the
      image `extent`. As in §2.5.2, the kernels and the weights ω* of
      every pixel are computed once, when the model is built, and
      stored (2ᵈ values per pixel); operators then only slice them, and
      the model can be saved without `psf_model`. The fit evaluates
      `psf_model` at every pixel n_alt + 1 times, in batches sized from
      the memory budget, at a cost of about (n_alt + 1)·M·L·2ᵈ
      operations (M pixels, L the PSF size): fine in 2-D, heavy for
      large volumes with large 3-D PSFs.

    The paper recommends PSF interpolation over the convolve-then-weight
    model of Nagy & O'Leary
    ([`NagyOLearyPSF`][optix.deconv.NagyOLearyPSF]) and proposes the
    optimal local approximation for the best accuracy at the same cost.

    !!! note "Choices not specified in the paper"
        - Outside the grid of centres the weights are constant
          (clamped).
        - With `"modes"`, the singular value decomposition is that of the
          sampled PSFs (one column per grid node) and the mode
          coefficients are interpolated bilinearly between nodes; the
          default `rank` keeps all modes.
        - With `"optimal"`, positions outside the image are clamped to it
          before evaluating `psf_model`, and least-squares problems are
          solved with the pseudo-inverse (minimum-norm solution) so that
          identical neighbouring PSFs are allowed.
        - With `"optimal"`, all 2ᵈ nodes of a pixel's (clamped) grid
          cell carry a weight, the paper's "weights of a cell". The
          support constraint of Eq. 31, supp(wₚ) ⊂ supp(φₚ), is
          therefore loosened where φₚ vanishes on a cell node: beyond the
          outermost centres, where the clamped φ is non-zero only on the
          edge nodes, and exactly on grid lines.

    !!! references "References"

        1. Denis L, Thiébaut É, Soulez F, Becker J-M, Mourya R. *Fast
           approximations of shift-variant blur.* Int J Comput Vis
           **115**(3):253-278, 2015.
           [doi:10.1007/s11263-015-0817-x](https://doi.org/10.1007/s11263-015-0817-x)

        ??? quote "BibTeX"

            ```bibtex
            @article{denis2015fast,
              author  = {Denis, Lo{\\"\\i}c and Thi{\\'e}baut, Eric and
                         Soulez, Ferr{\\'e}ol and Becker, Jean-Marie and
                         Mourya, Rahul},
              title   = {Fast Approximations of Shift-Variant Blur},
              journal = {International Journal of Computer Vision},
              volume  = {115},
              number  = {3},
              pages   = {253--278},
              year    = {2015},
              doi     = {10.1007/s11263-015-0817-x}
            }
            ```

    Attributes
    ----------
    kernels : Tensor
        cₚ, `(P, *K)`.
    singular_values : Tensor or None
        σ of the sampled PSFs (`"modes"` only), decreasing.
    omega : Tensor
        `"optimal"` only: ω* of every image pixel, `(*extent, 2ᵈ)`, over
        the nodes of its grid cell.

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import DenisPSF
    >>> psf = torch.ones(2, 2, 5, 5)
    >>> model = DenisPSF(psf, [[16, 48], [16, 48]])
    >>> model.operator((0, 0), (64, 64)).shape
    (64, 64)
    """

    order = "input"

    def __init__(
        self,
        psf: Tensor,
        centres: Sequence[Tensor | Sequence[float]],
        *,
        model: str = "interpolation",
        rank: int | None = None,
        psf_model: Callable[[Tensor], Tensor] | None = None,
        extent: Sequence[int] | None = None,
        n_alt: int = 10,
        normalize: bool = True,
    ) -> None:
        """Set up the model (and fit it, for `model="optimal"`).

        Parameters
        ----------
        psf : Tensor
            Sampled PSFs `(*G, *K)`: a G grid of d-dimensional PSFs
            (d = 2 or 3, spatial axes in layout order), centred at index
            `K // 2`.
        centres : sequence of 1-D tensors
            Position (in pixels, per spatial axis) of the grid nodes.
        model : {"interpolation", "modes", "optimal"}
            Approximation (see above). Default: `"interpolation"`.
        rank : int, optional
            Number of PSF modes (`"modes"`). Default: all (not specified
            in the paper, which uses 5 and 13).
        psf_model : callable, optional
            `"optimal"` only: maps positions `(n, d)` (float64, pixels)
            to the PSFs there, `(n, *K)`, centred at `K // 2`.
        extent : sequence of int, optional
            `"optimal"` only: image shape; the fit uses the PSF of every
            pixel.
        n_alt : int
            `"optimal"` only: number of alternations (paper: 10).
        normalize : bool
            Divide each PSF (sampled, and from `psf_model`) by its sum.
            Set to False to keep a varying total (e.g. vignetting, §3.2.2).
        """
        if model not in ("interpolation", "modes", "optimal"):
            raise ValueError(
                "model must be 'interpolation', 'modes' or 'optimal': "
                f"{model!r}"
            )
        psf, centres, d = _check_grid(psf, centres, None)
        if not psf.is_floating_point():
            psf = psf.double()
        if normalize:
            psf = _normalise(psf, d)
        self.centres = centres
        self.model = model
        self.normalize = bool(normalize)
        self.rank = None if rank is None else int(rank)
        self.singular_values: Tensor | None = None
        self.psf_model = psf_model
        ksize = tuple(psf.shape[d:])
        samples = psf.reshape((-1,) + ksize)
        self._samples = samples
        self.kernels = samples
        if model == "modes":
            mat = samples.reshape(samples.shape[0], -1).double().T  # L × G
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
            rank = s.numel() if self.rank is None else self.rank
            if not 1 <= rank <= s.numel():
                raise ValueError(
                    f"rank must be in [1, {s.numel()}], got {rank}"
                )
            self.rank = rank
            self.singular_values = s
            self.kernels = (
                u[:, :rank].T.reshape((rank,) + ksize).to(samples.dtype)
            )
            self._coeff = (s[:rank, None] * vh[:rank]).to(samples.dtype)
        elif model == "optimal":
            if psf_model is None or extent is None:
                raise ValueError("model='optimal' needs psf_model and extent")
            self.extent = tuple(int(n) for n in extent)
            if len(self.extent) != d:
                raise ValueError("extent must have one entry per axis")
            self.n_alt = int(n_alt)
            self._fit_optimal()

    # -- optimal local approximation (§2.5) --------------------------------
    def _corners(self, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Flat nodes `(n, c)`, bilinear weights `(n, c)`, cell `(n,)`."""
        sizes = self.grid_shape
        cells, ts = [], []
        for a, c in enumerate(self.centres):
            cell, t = _axis_cells(c.to(pos), pos[:, a])
            cells.append(cell)
            ts.append(t)
        cell = torch.stack(cells, -1)  # (n, d)
        var = [a for a, g in enumerate(sizes) if g > 1]
        offs = []
        for bits in itertools.product((0, 1), repeat=len(var)):
            o = [0] * len(sizes)
            for a, b in zip(var, bits, strict=True):
                o[a] = b
            offs.append(o)
        offs_t = torch.tensor(offs, device=pos.device)  # (c, d)
        nodes = cell[:, None, :] + offs_t[None]
        weight = torch.ones(
            nodes.shape[:2], dtype=pos.dtype, device=pos.device
        )
        for a in var:
            t = ts[a][:, None]
            weight = weight * torch.where(offs_t[None, :, a] == 1, t, 1 - t)
        return self._flat(nodes), weight, self._flat(cell)

    def _evaluate(self, pos: Tensor) -> Tensor:
        """PSFs of `psf_model` at `pos`, flattened `(n, L)`, float64."""
        hi = torch.tensor(self.extent, dtype=pos.dtype, device=pos.device) - 1
        pos = torch.minimum(pos.clamp(min=0), hi)
        k = torch.as_tensor(self.psf_model(pos)).double().to(pos.device)
        if tuple(k.shape[1:]) != self.shape or k.shape[0] != pos.shape[0]:
            raise ValueError(
                f"psf_model must return ({pos.shape[0]}, *{self.shape}), "
                f"got {tuple(k.shape)}"
            )
        if self.normalize:
            k = _normalise(k, len(self.shape))
        return k.reshape(k.shape[0], -1)

    def _pinvs(self, c: Tensor) -> Tensor:
        """Pseudo-inverses of C′ per cell, indexed by the cell's flat node."""
        sizes = self.grid_shape
        n_nodes = math.prod(sizes)
        lower = [
            torch.arange(max(g - 1, 1), dtype=torch.float64) for g in sizes
        ]
        pos = torch.cartesian_prod(*lower).reshape(-1, len(sizes))
        # positions at the lower corner of each cell, in centre units
        pos = torch.stack(
            [
                cen[p.long()]
                for cen, p in zip(self.centres, pos.T, strict=True)
            ],
            -1,
        )
        nodes, _, cell = self._corners(pos.to(c.device))
        out = c.new_zeros((n_nodes, nodes.shape[1], c.shape[1]))
        for i in range(nodes.shape[0]):
            out[cell[i]] = torch.linalg.pinv(c[nodes[i]].T)
        return out

    def _weights_at(
        self, kmat: Tensor, nodes: Tensor, cell: Tensor, pinvs: Tensor
    ) -> Tensor:
        """ω*ⱼ = C′ⱼ⁺ K⋅ⱼ (Eq. 37), grouped by cell."""
        out = kmat.new_zeros(nodes.shape)
        for c in torch.unique(cell):
            sel = cell == c
            out[sel] = kmat[sel] @ pinvs[c].T
        return out

    def _batch(self) -> int:
        """Pixels per batch of the fit, from the memory budget.

        A batch holds the PSFs there (float64, L values per pixel), the
        output of `psf_model` and two temporaries of the same size.
        """
        per_pixel = 4 * 8 * math.prod(self.shape)
        return max(1, int(get_config().memory) // per_pixel)

    def _pixels(self) -> Iterator[tuple[int, Tensor]]:
        """First flat index and positions `(n, d)` of the image pixels."""
        total = math.prod(self.extent)
        batch = self._batch()
        device = self.kernels.device
        for start in range(0, total, batch):
            flat = torch.arange(
                start, min(start + batch, total), device=device
            )
            idx = torch.unravel_index(flat, self.extent)
            yield start, torch.stack(idx, -1).double()

    def _fit_optimal(self) -> None:
        g = self._samples.shape[0]
        c = self._samples.reshape(g, -1).double()  # G × L
        pinvs = None
        for _ in range(self.n_alt):
            kw = c.new_zeros(c.shape[1], g)
            wtw = c.new_zeros(g * g)
            for _, pos in self._pixels():
                kmat = self._evaluate(pos)
                nodes, w, cell = self._corners(pos)
                if pinvs is not None:
                    w = self._weights_at(kmat, nodes, cell, pinvs)
                for s in range(nodes.shape[1]):
                    kw.index_add_(1, nodes[:, s], kmat.T * w[:, s])
                    for s2 in range(nodes.shape[1]):
                        wtw.index_add_(
                            0,
                            nodes[:, s] * g + nodes[:, s2],
                            w[:, s] * w[:, s2],
                        )
            c = (kw @ torch.linalg.pinv(wtw.reshape(g, g))).T  # Eq. 35
            pinvs = self._pinvs(c)
        # ω* of every pixel for the final kernels (Eq. 37), computed once
        n_corners = 2 ** sum(n > 1 for n in self.grid_shape)
        omega = c.new_zeros(math.prod(self.extent), n_corners)
        for start, pos in self._pixels():
            nodes, _, cell = self._corners(pos)
            omega[start : start + pos.shape[0]] = self._weights_at(
                self._evaluate(pos), nodes, cell, pinvs
            )
        dtype = self._samples.dtype
        self.omega = omega.reshape(self.extent + (n_corners,)).to(dtype)
        self.kernels = c.reshape((g,) + self.shape).to(dtype)

    # -- terms ---------------------------------------------------------
    def _axis_kind(self) -> str | None:
        return None if self.model == "modes" else "multilinear"

    def terms(self, coords: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        if self.model == "interpolation":
            nodes, maps = interpolation_weights(
                self.centres, coords, "multilinear"
            )
            return self.kernels[self._flat(nodes)], maps
        if self.model == "modes":
            nodes, maps = interpolation_weights(
                self.centres, coords, "multilinear"
            )
            coeff = self._coeff[:, self._flat(nodes)].to(maps)
            return self.kernels, torch.tensordot(coeff, maps, dims=1)
        # optimal: slice the stored ω* (positions clamped to the image)
        shape = tuple(x.numel() for x in coords)
        index = []
        for x, n in zip(coords, self.extent, strict=True):
            r = x.double().round()
            if not bool(((x.double() - r).abs() < 1e-6).all()):
                raise ValueError(
                    "model='optimal' is defined at integer pixel positions"
                )
            index.append(r.long().clamp(0, n - 1))
        grids = torch.meshgrid(*index, indexing="ij")
        omega = self.omega.to(index[0].device)[grids].reshape(
            -1, self.omega.shape[-1]
        )
        pos = torch.stack([gi.reshape(-1) for gi in grids], -1).double()
        nodes, _, _ = self._corners(pos)
        active, inv = torch.unique(nodes, return_inverse=True)
        maps = omega.new_zeros(active.numel(), pos.shape[0])
        pix = torch.arange(pos.shape[0], device=pos.device)
        for s in range(nodes.shape[1]):
            maps.index_put_((inv[:, s], pix), omega[:, s], accumulate=True)
        return self.kernels[active], maps.reshape((-1,) + shape)

    def _convert(self, **backend: object) -> None:
        super()._convert(**backend)
        self._samples = self._samples.to(**backend)
        if self.model == "modes":
            self._coeff = self._coeff.to(**backend)
        if self.model == "optimal":
            self.omega = self.omega.to(**backend)

    def _payload(self) -> dict:
        grid = self.grid_shape
        payload = {
            "model": "denis",
            "psf": self._samples.reshape(grid + self.shape),
            "centres": list(self.centres),
            "kind": self.model,
            "rank": self.rank,
        }
        if self.model == "optimal":
            payload["kernels"] = self.kernels
            payload["omega"] = self.omega
            payload["n_alt"] = self.n_alt
            payload["normalize"] = self.normalize
        return payload

    @classmethod
    def _restore_optimal(cls, payload: dict) -> "DenisPSF":
        """A fitted `"optimal"` model from its kernels and ω*."""
        new = cls(payload["psf"], payload["centres"], normalize=False)
        new.model = "optimal"
        new.normalize = bool(payload["normalize"])
        new.n_alt = int(payload["n_alt"])
        new.omega = payload["omega"]
        new.extent = tuple(int(n) for n in new.omega.shape[:-1])
        new.kernels = payload["kernels"]
        return new
