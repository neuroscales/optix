"""Nagy & O'Leary (1998): preconditioned CG with a shift-variant PSF."""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from .._estimator import Estimator, Tags
from ..io import (
    ArrayLike,
    Chunk,
    Labeled,
    Layout,
    WritableArrayLike,
    blocks,
    chunks_of,
    flush,
    prepare_out,
)
from ._shiftvariant import NagyOLearyPSF, ShiftVariantConvolution

__all__ = [
    "NagyOLeary",
    "RegionPreconditioner",
    "cgls",
    "nagy_oleary",
    "optimal_circulant",
]

_OVERHEAD = 16.0  # CG vectors, transfer functions and weights per voxel


def optimal_circulant(kernel: Tensor, shape: Sequence[int]) -> Tensor:
    """Eigenvalues of the optimal block-circulant approximation of a blur.

    The convolution with `kernel` on a region of shape ρ (zero boundary)
    is a block-Toeplitz matrix with Toeplitz blocks T. The block-circulant
    matrix C with circulant blocks closest to T in Frobenius norm (Chan &
    Olkin 1993) has first column

        c(j) = ∑_{d ≡ j mod ρ} ∏ₐ (1 − |dₐ|/ρₐ) k(d),

    k(d) the kernel at offset d from its centre (index `K // 2`). Its
    eigenvalues are the DFT of c.

    Parameters
    ----------
    kernel : Tensor
        Kernel `(*K)`.
    shape : sequence of int
        Region shape ρ.

    Returns
    -------
    Tensor
        Eigenvalues as an `rfftn` spectrum of shape ρ.
    """
    shape = tuple(int(s) for s in shape)
    d = kernel.ndim
    index = torch.zeros((), dtype=torch.long, device=kernel.device)
    weight = torch.ones((), dtype=kernel.dtype, device=kernel.device)
    stride = 1
    for a in reversed(range(d)):
        k, rho = kernel.shape[a], shape[a]
        off = torch.arange(k, device=kernel.device) - k // 2
        w = (1 - off.abs().to(kernel.dtype) / rho).clamp(min=0)
        view = [1] * d
        view[a] = -1
        index = index + (off % rho).reshape(view) * stride
        weight = weight * w.reshape(view)
        stride *= rho
    first = kernel.new_zeros(stride)
    first.index_add_(0, index.reshape(-1), (weight * kernel).reshape(-1))
    return torch.fft.rfftn(first.reshape(shape))


class RegionPreconditioner:
    """Block-diagonal circulant preconditioner Cτ of Nagy & O'Leary (§4).

    On each region i of the piecewise-constant partition, Cᵢ is the
    optimal circulant approximation
    ([`optimal_circulant`][optix.deconv.utils.optimal_circulant]) of the
    blur by PSF i restricted to the region; C is block diagonal with
    blocks Cᵢ, and Cτ replaces the eigenvalues of magnitude below τ by 1
    (Hanke, Nagy & Plemmons). Eigenvalues that vanish to round-off
    (below ε·max |λ|, ε the machine epsilon) are also replaced by 1, so
    that τ = 0 is well defined. Object pixels outside the regions (the
    margin of the padded grid) are left unchanged.

    Attributes
    ----------
    boxes : list of tuple of slice
        Region boxes on the object grid.
    eigenvalues : list of Tensor
        Eigenvalues of Cτ per box (`rfftn` spectra).
    """

    def __init__(
        self,
        regions: list[tuple[Tensor, tuple[slice, ...]]],
        *,
        tau: float = 0.0,
    ) -> None:
        """Precompute the eigenvalues.

        Parameters
        ----------
        regions : list of (Tensor, tuple of slice)
            Kernel and box of each region, the boxes on the object grid.
        tau : float
            Truncation level τ ≥ 0.
        """
        self.boxes = []
        self.eigenvalues = []
        for kernel, box in regions:
            shape = tuple(s.stop - s.start for s in box)
            lam = optimal_circulant(kernel, shape)
            mag = lam.abs()
            eps = torch.finfo(mag.dtype).eps
            small = (mag < tau) | (mag <= mag.max() * eps)
            lam = torch.where(small, torch.ones_like(lam), lam)
            self.boxes.append(box)
            self.eigenvalues.append(lam)

    def _solve(self, x: Tensor, conj: bool) -> Tensor:
        out = x.clone()
        for box, lam in zip(self.boxes, self.eigenvalues, strict=True):
            shape = tuple(s.stop - s.start for s in box)
            dims = tuple(range(-len(shape), 0))
            spec = torch.fft.rfftn(x[box], dim=dims)
            spec = spec / (lam.conj() if conj else lam)
            out[box] = torch.fft.irfftn(spec, s=shape, dim=dims)
        return out

    def apply(self, x: Tensor) -> Tensor:
        """Cτ⁻¹ x."""
        return self._solve(x, False)

    def transpose(self, x: Tensor) -> Tensor:
        """Cτ⁻ᵀ x."""
        return self._solve(x, True)


def _dot(a: Tensor, b: Tensor) -> float:
    return float((a * b).sum(dtype=torch.float64))


def cgls(
    data: Tensor,
    op: ShiftVariantConvolution,
    n_iter: int,
    *,
    precond: RegionPreconditioner | None = None,
    callback: Callable[[int, Tensor], object] | None = None,
) -> tuple[Tensor, int]:
    """Conjugate gradients for least squares, optionally preconditioned.

    Minimises ‖H f − y‖² by CG on the normal equations (CGLS, Björck),
    from f⁽⁰⁾ = 0, with the right preconditioner P = C⁻¹: the iteration
    runs on z with f = P z,

        s₀ = Pᵀ Hᵀ y,  p₀ = s₀,
        tₖ = P pₖ,  qₖ = H tₖ,  αₖ = ‖sₖ‖² / ‖qₖ‖²,
        fₖ₊₁ = fₖ + αₖ tₖ,  rₖ₊₁ = rₖ − αₖ qₖ,
        sₖ₊₁ = Pᵀ Hᵀ rₖ₊₁,  pₖ₊₁ = sₖ₊₁ + (‖sₖ₊₁‖²/‖sₖ‖²) pₖ.

    Stopping early regularises the solution. Inner products are
    accumulated in float64.

    Parameters
    ----------
    data : Tensor
        Observed block y, `op.shape`.
    op : ShiftVariantConvolution or PaddedConvolution
        Blur operator H.
    n_iter : int
        Number of iterations.
    precond : RegionPreconditioner, optional
        Preconditioner C (default: none, plain CGLS).
    callback : callable, optional
        `callback(k, f)` after each iteration.

    Returns
    -------
    f : Tensor
        Estimate on `op.grid`.
    n : int
        Number of iterations made (fewer than `n_iter` if the residual
        of the normal equations vanishes).
    """

    def p(x: Tensor) -> Tensor:
        return x if precond is None else precond.apply(x)

    def pt(x: Tensor) -> Tensor:
        return x if precond is None else precond.transpose(x)

    f = torch.zeros_like(op.norm)
    r = data.clone()
    s = pt(op.adjoint(r))
    d = s
    gamma = _dot(s, s)
    n = 0
    while n < n_iter and gamma > 0:
        t = p(d)
        q = op.forward(t)
        qq = _dot(q, q)
        if not qq > 0:
            break
        alpha = gamma / qq
        f = f + alpha * t
        r = r - alpha * q
        s = pt(op.adjoint(r))
        new = _dot(s, s)
        d = s + (new / gamma) * d
        gamma = new
        n += 1
        if callback is not None:
            callback(n, f)
    return f, n


class NagyOLeary(Estimator):
    """Restoration with a spatially variant PSF (Nagy & O'Leary 1998).

    Model: the image is the blur of the object by PSFs kᵢ measured at
    the nodes of a rectangular grid, combined after convolution,

        g = K f + η,   K = D₁ K₁ + ⋯ + Dₚ Kₚ   (Eq. 2.1),

    see [`NagyOLearyPSF`][optix.deconv.NagyOLearyPSF]. The object is
    recovered by conjugate gradients on the least-squares problem
    min ‖K f − g‖² (CGLS), regularised by stopping the iteration early,
    and preconditioned by the block-diagonal circulant matrix Cτ (§4,
    [`RegionPreconditioner`][optix.deconv.utils.RegionPreconditioner]):
    on each region, the optimal circulant approximation of the blur by
    that region's PSF (Chan & Olkin), with eigenvalues below τ replaced
    by 1. As in the paper's experiments (§5), τ = 0. With
    `interpolation="linear"` the preconditioner uses, on each region of
    the piecewise-constant partition, the PSF of that region (§4.1.1:
    one of the vertex kernels). `precondition=False` runs plain CGLS,
    the paper's reference method.

    **Boundaries and blocks.** As for
    [`RichardsonLucy`][optix.deconv.RichardsonLucy], the object extends
    past the observed region by the PSF support, and large inputs are
    processed in blocks with a halo, each block solved as an image of its
    own and only its core kept. The blur weights are those of the whole
    image (global positions); the preconditioner of a block uses the
    parts of the regions that fall in it, and CG inner products are per
    block, so block-wise and whole-image results differ slightly away
    from the image edges. With the default `block`, the image is
    processed whole when it fits in the memory budget.

    !!! warning "Smooth PSFs need τ > 0"
        The paper's PSFs have a sharp core, so the eigenvalues of C stay
        away from 0 and τ = 0 works (§4: "for moderately ill-posed
        problems with little noise, we can take τ ≈ 0"). For smooth PSFs
        (e.g. Gaussian), the eigenvalues of C decay to tiny values and
        Cτ⁻¹ with τ = 0 amplifies the mismatch between C and K at the
        region boundaries, and the noise: PCG then diverges from the
        first iterations. Choose τ a few per cent of the largest
        eigenvalue (≈ 1 for unit-sum PSFs), or `precondition=False`.

    !!! note "Choices not specified in the paper"
        - The paper avoids a stopping criterion and compares solutions
          after a fixed number of iterations: 14 PCG iterations in
          Fig. 5.8, the default `n_iter`.
        - CGLS starts from f⁽⁰⁾ = 0; the object outside the observed
          region is not preconditioned.
        - Region boundaries are half-way between neighbouring centres.

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
    >>> from optix.deconv import NagyOLeary
    >>> psf = torch.ones(2, 2, 5, 5)  # 2 × 2 grid of 5 × 5 PSFs
    >>> data = torch.rand(64, 64)
    >>> est = NagyOLeary(psf, [[16, 48], [16, 48]]).fit(data)
    >>> est.apply(data).shape
    torch.Size([64, 64])
    """

    tags = Tags(
        targets=("deconvolution",),
        streaming="halo",
        noise="gaussian",
        reference="doi:10.1137/S106482759528507X",
    )
    paper = {
        "interpolation": "constant",
        "precondition": True,
        "tau": 0.0,
        "n_iter": 14,
        "halo": None,
        "block": None,
    }
    recommended = dict(paper)

    def __init__(
        self,
        psf: Tensor | None = None,
        centres: Sequence[Tensor | Sequence[float]] | None = None,
        *,
        interpolation: str | None = None,
        precondition: bool | None = None,
        tau: float | None = None,
        n_iter: int | None = None,
        halo: int | Sequence[int] | None = None,
        block: Sequence[int | None] | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the restoration.

        Parameters
        ----------
        psf : Tensor
            PSFs `(*G, *K)` on a grid (2 or 3 spatial dims, in layout
            order), each centred at index `K // 2`; normalised to unit
            sum.
        centres : sequence of 1-D tensors
            Position of the grid nodes along each spatial axis, in pixels.
        interpolation : {"constant", "linear"}
            Weights of the blur model (see
            [`NagyOLearyPSF`][optix.deconv.NagyOLearyPSF]). Default
            (paper, §5): `"constant"`.
        precondition : bool
            Preconditioned CG (PCG) or plain CG. Default: True.
        tau : float
            Truncation level τ of the preconditioner. Default (§5): 0.
        n_iter : int
            Number of CG iterations. Default: 14 (see above).
        halo : int or sequence of int, optional
            Halo width per spatial axis, at least the PSF size. Default:
            twice the PSF size (not specified in the paper).
        block : sequence of int or None, optional
            Core block size per spatial axis. Default: from the memory
            budget.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        params = {
            k: v for k, v in locals().items() if k not in ("psf", "centres")
        }
        self._resolve(faithful, params)
        opts = self.options
        if opts["interpolation"] not in ("constant", "linear"):
            raise ValueError(
                "interpolation must be 'constant' or 'linear': "
                f"{opts['interpolation']!r}"
            )
        if int(opts["n_iter"]) < 0:
            raise ValueError("n_iter must be non-negative")
        if float(opts["tau"]) < 0:
            raise ValueError("tau must be non-negative")
        self.psf = psf
        self.centres = centres
        self.model: NagyOLearyPSF | None = None
        if psf is not None:
            if centres is None:
                raise ValueError("NagyOLeary needs the PSF centres")
            self.model = NagyOLearyPSF(
                psf, centres, interpolation=opts["interpolation"]
            )
        self._fitted = False

    def _labeled_for(self, data: ArrayLike) -> Labeled:
        if isinstance(data, Labeled):
            return data
        ndim = 2 if self.model is None else self.model.ndim
        return self._labeled(data, Layout.default(len(data.shape), ndim))

    def _halo(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        halo = self.options["halo"]
        if halo is None:
            return tuple(2 * k for k in shape)
        halo = (
            (int(halo),) * len(shape)
            if isinstance(halo, int)
            else tuple(int(h) for h in halo)
        )
        if len(halo) != len(shape) or any(
            h < k for h, k in zip(halo, shape, strict=True)
        ):
            raise ValueError(
                f"halo {halo} must be at least the PSF size {shape}"
            )
        return halo

    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        if self.model is None:
            raise ValueError("NagyOLeary needs a psf and its centres")
        data = self._labeled_for(data)
        self.layout = data.layout
        self._warn_unused(init)
        if self.model.ndim != len(data.layout.spatial):
            raise ValueError(
                f"The PSFs have {self.model.ndim} dims but the layout has "
                f"{len(data.layout.spatial)} spatial axes"
            )
        self.history.extras["iterations"] = []
        self._fitted = True

    @staticmethod
    def _origin(chunk: Chunk) -> tuple[int, ...]:
        return tuple(
            core.start - inner.start
            for core, inner in zip(chunk.spatial, chunk.inner, strict=True)
        )

    def _preconditioner(
        self,
        model: NagyOLearyPSF,
        origin: tuple[int, ...],
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> RegionPreconditioner:
        coords = tuple(
            torch.arange(n, dtype=torch.float64, device=model.kernels.device)
            + float(o)
            for n, o in zip(shape, origin, strict=True)
        )
        regions = [
            (model.kernels[i].to(dtype), box)
            for i, box in model.regions(coords)
        ]
        return RegionPreconditioner(regions, tau=float(self.options["tau"]))

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if not self._fitted:
            raise RuntimeError("Call fit(data) before apply(data)")
        data = self._labeled_for(data)
        backend = self._backend(data)
        model = self.model.to(**backend)
        raw = data.array
        target = out.array if isinstance(out, Labeled) else out
        if out is not None and target is raw:
            raise ValueError("In-place deconvolution is not possible")
        out = prepare_out(out, tuple(data.shape), like=raw, dtype=self.dtype)
        n_iter = int(self.options["n_iter"])
        iterations = self.history.extras.setdefault("iterations", [])
        chunks = blocks(
            data,
            self.options["block"],
            self._halo(model.shape),
            overhead=_OVERHEAD + 3.0 * 3**model.ndim,
            contiguous=True,
            align=chunks_of(out),
            **backend,
        )
        for chunk in chunks:
            y = chunk.data
            shape = tuple(y.shape)
            origin = self._origin(chunk)
            op = model.operator(origin, shape)
            precond = None
            if self.options["precondition"]:
                precond = self._preconditioner(model, origin, shape, y.dtype)
            f, n = cgls(y, op, n_iter, precond=precond)
            iterations.append(float(n))
            chunk.store(out, op.crop(f)[chunk.inner])
        flush(out)
        return out

    def _extra_state(self) -> dict[str, object]:
        centres = None
        if self.centres is not None:
            centres = [torch.as_tensor(c) for c in self.centres]
        return {"psf": self.psf, "centres": centres, "fitted": self._fitted}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.psf = state.get("psf")
        self.centres = state.get("centres")
        self._fitted = bool(state.get("fitted", False))
        if self.psf is not None:
            self.model = NagyOLearyPSF(
                self.psf,
                self.centres,
                interpolation=self.options["interpolation"],
            )


def nagy_oleary(
    data: ArrayLike,
    psf: Tensor,
    centres: Sequence[Tensor | Sequence[float]],
    *,
    out: WritableArrayLike | None = None,
    layout: Layout | None = None,
    **options: object,
) -> WritableArrayLike:
    """Restoration of `data` blurred by a spatially variant PSF.

    Shortcut for `NagyOLeary(psf, centres, **options).fit(data)
    .apply(data, out)`; see [`NagyOLeary`][optix.deconv.NagyOLeary] for
    the model and the defaults.

    Parameters
    ----------
    data : array-like or Labeled
        Images; `(*samples, *spatial)` without a layout.
    psf : Tensor
        PSFs `(*G, *K)` on a grid.
    centres : sequence of 1-D tensors
        Position of the grid nodes along each spatial axis.
    out : writable array-like, optional
        Output. Default: a new tensor.
    layout : Layout, optional
        Axis roles.
    **options
        Options of [`NagyOLeary`][optix.deconv.NagyOLeary].

    Returns
    -------
    writable array-like
        `out`, the restored images.
    """
    est = NagyOLeary(psf, centres, layout=layout, **options)
    est.fit(data)
    return est.apply(data, out)
