"""Lefkimmiatis & Unser (2013): Poisson deconvolution with a Hessian
Schatten-norm prior (HSPIRAL)."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .._estimator import Tags, _same
from ..estimates import SensorEstimate, ShadingEstimate
from ..io import ArrayLike, Chunk, Layout, WritableArrayLike
from ..optim import (
    Box,
    Convolution,
    Hessian,
    Schatten,
    ShiftedPoisson,
    Term,
    admm_fft,
)
from ._fft import PaddedConvolution, transfer_function
from ._linear import smooth_periodic_extension
from ._variational import _Problem, _Variational

__all__ = ["HSPIRAL", "hspiral"]

_BOUNDARY = ("periodic", "unknown")
#: Constant κ of the default weight τ = κ / q₉₉.₉(y) (not in the paper).
_KAPPA = 1.4
#: Working buffers of `admm_fft` per voxel of the FFT grid (measured
#: peak), per number of Hessian components C: base + per_component·C.
_GRID_BASE = 21.0
_GRID_COMPONENT = 3.5


class _CircularConvolution:
    """Blur of a block by circular convolution on the block itself.

    The interface used by the block loop of `_BlockDeconvolution`
    (`ndim`, `shape`, `grid`, `otf`, `crop`), with N = L: the periodic
    boundary of the paper.
    """

    def __init__(self, psf: Tensor, shape: Sequence[int]) -> None:
        self.ndim = psf.ndim
        self.shape = tuple(int(n) for n in shape)
        if any(n < k for n, k in zip(self.shape, psf.shape, strict=True)):
            raise ValueError(
                f"HSPIRAL with boundary='periodic' needs blocks at least as "
                f"large as the PSF {tuple(psf.shape)}, got {self.shape}"
            )
        self.grid = self.shape
        self.otf = transfer_function(psf, self.grid)

    def crop(self, x: Tensor) -> Tensor:
        """The object grid is the block: identity."""
        return x


class HSPIRAL(_Variational):
    """Poisson deconvolution with a Hessian Schatten-norm prior (HSPIRAL2).

    **Model** (Lefkimmiatis & Unser 2013, Eqs. 1–3). The data are photon
    counts y ~ Poisson(K x), with K the convolution with the PSF and
    x ≥ 0. The estimate is the penalised maximum-likelihood solution of
    Eq. (12),

        minimise_x  ∑ⱼ (K x)ⱼ − yⱼ log (K x)ⱼ + τ ∑ⱼ ‖(H x)ⱼ‖_{S_p}
                    subject to x ≥ 0,

    where (H x)ⱼ is the 2 × 2 Hessian matrix of x at pixel j and ‖·‖_{S_p}
    the Schatten norm of order p (Eq. 5), the ℓ_p norm of its singular
    values, which are the absolute eigenvalues of the symmetric Hessian:
    the nuclear norm (p = 1, sum), the Frobenius norm (p = 2) or the
    spectral norm (p = ∞, largest). The second differences compose
    forward differences, D_ab = D_a D_b (Lefkimmiatis, Bourquard & Unser
    2012, Eq. 21; [`Hessian`][optix.optim.Hessian]).

    **Solver** (Algorithm 2, HSPIRAL2). ADMM with the splits z₁ = K x,
    z₂ = H x, z₃ = x and one penalty α (see
    [`admm_fft`][optix.optim.admm_fft], `order="z-first"`): z₁ by the
    closed-form Poisson prox (Eq. 27), z₂ by the Schatten-norm prox
    (Proposition 1; closed forms of Eqs. 43–47, see
    [`prox_schatten`][optix.optim.prox_schatten]), z₃ by projection onto
    x ≥ 0 (Eq. 32), then the x-update (Eq. 42), solved in the Fourier
    domain, and the dual updates. It starts from x⁰ = y with zero duals,
    uses α = 60τ/M (Section IV-C) and stops when
    ‖xᵏ⁺¹ − xᵏ‖₂ ≤ 10⁻⁵ ‖xᵏ‖₂ or after 400 iterations (Section IV-B).
    The result is xᵏ (Algorithm 2 returns x); it is not projected onto
    x ≥ 0. Each solve records its number of iterations, last relative
    change and last primal and dual residuals in `history.extras`.

    **Boundary.** The paper assumes periodic (or mirror) boundaries for
    the FFT solve of Eq. (42) and does not say which its experiments
    use; the default here is periodic: the blur and the differences wrap
    around the block, as in a circular-blur simulation.
    `boundary="unknown"` is an extension (not in the paper): x lives on
    a larger grid (the block plus the PSF, as in
    [`Ikoma`][optix.deconv.Ikoma]), the Poisson term sums over the
    observed pixels only, and x⁰ is the data continued smoothly over the
    margin; it avoids wrap-around artefacts on real images.

    **Choices not specified in the paper.**

    - M in α = 60τ/M is the maximum intensity of the ground-truth image
      in the paper; here it is max(y) per image.
    - p: the paper compares 1, 2 and ∞ with no default; p = 1 gave the
      best pooled PSNR in its Table I and is the default. All three are
      the paper's method.
    - Dimension: the paper is 2-D. In 3-D, p = 2 works unchanged (the
      Frobenius norm of the 3 × 3 Hessian, the prox needs no
      eigenvalues); p ∈ {1, ∞} raises `NotImplementedError`.
    - The regularisation weight τ: the paper tunes τ per image for the
      best PSNR. The default here is

          τ = κ / q₉₉.₉(y),   κ = 1.4,

      per image, with q₉₉.₉(y) the 99.9th percentile of the data
      (computed exactly in the one pass of `fit`; a hot pixel barely
      changes it). The 1/scale factor is that of a MAP estimate whose
      prior on the normalised image x/s (s the intensity scale) is the
      same at every photon count: the Poisson log-likelihood is
      homogeneous, ℓ(s x̂; s ŷ) = s ℓ(x̂; ŷ) up to a constant, so
      ℓ(x; y) + κ‖H x/s‖ has the weight τ = κ/s.

      κ was calibrated for the paper's α rule and stopping rule on
      simulated 64 × 64 images with circular blur (Gaussian, σ_b = 2,
      9 × 9; uniform 7 × 7): blobs, filaments, their mixture and
      piecewise-smooth scenes, peaks of 10, 100 and 1000 photons, three
      seeds each (72 cases for each of p = 1, 2, ∞). The best κ of a
      grid (0.004 to 8, factor √2) hardly depends on the photon count,
      which supports the 1/scale rule, but depends on the content:
      about 0.2 to 0.5 for thin filaments, 1 to 6 for smooth blobs.
      With κ = 1.4 the PSNR is on average 0.5 dB below that of the best
      κ (at most 2.0, 3.0 and 3.9 dB for p = 1, 2, ∞, on blobs or
      filaments) and at least 2 dB above that of the data (5 dB on
      average). Pass `tau` to set the weight; the values used are in
      `tau` and `rho` after `fit`.

    **Camera calibration** (not in the paper, which takes photon
    counts). With a [`SensorEstimate`][optix.estimates.SensorEstimate]
    given to `fit`, y = (data − offset)/gain; with a
    [`ShadingEstimate`][optix.estimates.ShadingEstimate], the dark field
    is subtracted and the data are divided by the flat field (unit
    median), as for [`Ikoma`][optix.deconv.Ikoma]. The model is Poisson:
    read noise is ignored, and negative values are set to 0. Without an
    estimate the data are taken as photon counts.

    **Blocks.** An image that fits in the memory budget is solved whole;
    a larger one is cut into blocks with a halo (default: twice the PSF
    size, not specified in the paper), each solved on its own with the
    boundary above, keeping its core; τ and α are those of the whole
    image. A `RuntimeWarning` says when an image is split. With the
    periodic boundary each block wraps around, so seams are worse than
    with `boundary="unknown"`. `halo` and `block` are memory settings:
    they do not change `is_faithful`.

    Paper notation: K the blur (h here elsewhere), H the Hessian
    (ℋ in the paper), τ, α, p as here; the paper's s are the scaled
    duals u of `admm_fft`.

    !!! note "HSPIRAL1"
        The paper's second variant, HSPIRAL1 (Algorithm 1, used for its
        tables), splits only z₁ = K x and z₂ = x and evaluates the prox
        of τ‖H·‖₁,ₚ + ι_C iteratively with the algorithm of Lefkimmiatis,
        Ward & Unser (2013). Both variants minimise Eq. (12); only
        HSPIRAL2 is implemented.

    !!! references "References"
        1. Lefkimmiatis S, Unser M. *Poisson image reconstruction with
           Hessian Schatten-norm regularization.* IEEE Trans Image
           Process **22**(11):4314-4327, 2013.
           [doi:10.1109/TIP.2013.2271852](https://doi.org/10.1109/TIP.2013.2271852)
        2. Lefkimmiatis S, Bourquard A, Unser M. *Hessian-based norm
           regularization for image restoration with biomedical
           applications.* IEEE Trans Image Process **21**(3):983-995,
           2012.
           [doi:10.1109/TIP.2011.2168232](https://doi.org/10.1109/TIP.2011.2168232)

        ??? quote "BibTeX"
            ```bibtex
            @article{lefkimmiatis2013poisson,
              author  = {Lefkimmiatis, Stamatios and Unser, Michael},
              title   = {Poisson Image Reconstruction With {Hessian}
                         {Schatten}-Norm Regularization},
              journal = {IEEE Transactions on Image Processing},
              volume  = {22},
              number  = {11},
              pages   = {4314--4327},
              year    = {2013},
              doi     = {10.1109/TIP.2013.2271852}
            }
            @article{lefkimmiatis2012hessian,
              author  = {Lefkimmiatis, Stamatios and Bourquard, Aur{\\'e}lien
                         and Unser, Michael},
              title   = {Hessian-Based Norm Regularization for Image
                         Restoration With Biomedical Applications},
              journal = {IEEE Transactions on Image Processing},
              volume  = {21},
              number  = {3},
              pages   = {983--995},
              year    = {2012},
              doi     = {10.1109/TIP.2011.2168232}
            }
            ```

    Attributes
    ----------
    tau : Tensor
        Regularisation weight τ per image `(*group, *sample)`, after
        `fit`.
    rho : Tensor
        ADMM penalty α per image, after `fit`.
    peak : Tensor
        max y per image (M of the α rule).
    robust_peak : Tensor
        99.9th percentile q₉₉.₉(y) per image (used by the default τ).

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import HSPIRAL
    >>> psf = torch.ones(5, 5)
    >>> data = torch.poisson(torch.full((32, 32), 10.0))
    >>> HSPIRAL(psf, max_iter=5).fit(data).apply(data).shape
    torch.Size([32, 32])
    """

    tags = Tags(
        targets=("deconvolution",),
        streaming="halo",
        noise="poisson",
        reference="doi:10.1109/TIP.2013.2271852",
    )
    paper = {
        "tau": None,
        "p": 1,
        "rho": None,
        "max_iter": 400,
        "tol": 1e-5,
        "boundary": "periodic",
        "halo": None,
        "block": None,
    }
    recommended = dict(paper)
    #: Options that do not make an estimate unfaithful: τ is tuned per
    #: image and p ∈ {1, 2, ∞} are all compared in the paper; `halo` and
    #: `block` are memory settings.
    _free = ("tau", "p", "halo", "block")

    def __init__(
        self,
        psf: Tensor | None = None,
        *,
        tau: float | None = None,
        p: float | str | None = None,
        rho: float | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        boundary: str | None = None,
        halo: int | Sequence[int] | None = None,
        block: Sequence[int | None] | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the deconvolution.

        Parameters
        ----------
        psf : Tensor
            Point spread function, `(*K)` with 2 or 3 dims, in the order
            of the spatial axes of the layout, centred at index
            `K // 2`. Normalised to unit sum.
        tau : float, optional
            Regularisation weight τ ≥ 0. Default: 1.4 / q₉₉.₉(y) per
            image (see above; not in the paper).
        p : {1, 2, inf}
            Order of the Schatten norm. Default: 1 (all three are the
            paper's method; p ∈ {1, inf} needs a 2-D PSF).
        rho : float, optional
            ADMM penalty α > 0. Default (paper): 60τ / max y per image.
        max_iter : int
            Maximum number of iterations. Default (paper): 400.
        tol : float
            Relative-change tolerance. Default (paper): 10⁻⁵.
        boundary : {"periodic", "unknown"}
            Periodic blur and differences on the block (default, paper),
            or an unobserved margin around it (extension).
        halo : int or sequence of int, optional
            Halo width per spatial axis, at least the PSF size along that
            axis. Default: twice the PSF size (not specified in the
            paper). A memory setting: does not change `is_faithful`.
        block : sequence of int or None, optional
            Core block size per spatial axis (None entries: whole
            extent). Default: chosen from the memory budget, the whole
            image when it fits. A memory setting: does not change
            `is_faithful`.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        params = {k: v for k, v in locals().items() if k != "psf"}
        self._resolve(faithful, params)
        self._faithful_options = self.is_faithful
        self.psf = psf
        self.sensor: SensorEstimate | None = None
        self.shading: ShadingEstimate | None = None
        self.peak: Tensor | None = None
        self.robust_peak: Tensor | None = None
        self.noise: Tensor | None = None
        self.tau: Tensor | None = None
        self.rho: Tensor | None = None
        self._check()

    def _resolve(self, faithful: bool, params: object) -> None:
        super()._resolve(faithful, params)
        self.is_faithful = all(
            _same(value, self.paper.get(name))
            for name, value in self.options.items()
            if name not in self._free
        )

    def _check(self) -> None:
        opts = self.options
        try:
            p = float(opts["p"])
        except (TypeError, ValueError):
            p = math.nan
        if p not in (1.0, 2.0, math.inf):
            raise ValueError(f"p must be 1, 2 or inf, got {opts['p']!r}")
        opts["p"] = p
        psf = self.psf
        if psf is not None:
            ndim = torch.as_tensor(psf).ndim
            if ndim not in (2, 3):
                raise ValueError("psf must be 2-D or 3-D")
            if ndim == 3 and p != 2:
                raise NotImplementedError(
                    "HSPIRAL in 3-D is implemented for p=2 only (the "
                    "Frobenius norm of the Hessian)"
                )
        if opts["boundary"] not in _BOUNDARY:
            raise ValueError(f"boundary must be one of {_BOUNDARY}")
        if opts["tau"] is not None and not float(opts["tau"]) >= 0:
            raise ValueError("tau must be non-negative")
        if opts["rho"] is not None and not float(opts["rho"]) > 0:
            raise ValueError("rho must be positive")
        if int(opts["max_iter"]) < 1:
            raise ValueError("max_iter must be positive")
        if opts["tol"] is not None and not float(opts["tol"]) >= 0:
            raise ValueError("tol must be non-negative")

    # -- hooks -----------------------------------------------------------
    def _psf(self, backend: dict) -> Tensor:
        if self.psf is None:
            raise ValueError("HSPIRAL needs a psf")
        psf = torch.as_tensor(self.psf).to(**backend)
        if (psf < 0).any():
            raise ValueError("psf must be non-negative")
        if not psf.sum() > 0:
            raise ValueError("psf must have a positive sum")
        return psf / psf.sum()

    def _prepare(self, chunk: Chunk, maps: dict) -> _Problem:
        """Photon counts y ≥ 0 (read noise ignored: Poisson model)."""
        blk = super()._prepare(chunk, maps)
        return _Problem(blk.data.clamp(min=0), variance=0.0)

    def _setup_images(self, psf: Tensor) -> None:
        opts = self.options
        peak = self.peak
        if not bool((peak > 0).all()):
            raise ValueError(
                "HSPIRAL needs data with a positive maximum in every image"
            )
        if opts["tau"] is None:
            robust = self.robust_peak
            if not bool((robust > 0).all()):
                raise ValueError(
                    "HSPIRAL needs data with a positive 99.9th percentile "
                    "in every image for the default tau: pass tau"
                )
            tau = _KAPPA / robust
        else:
            tau = torch.full_like(peak, float(opts["tau"]))
        if opts["rho"] is not None:
            rho = torch.full_like(peak, float(opts["rho"]))
        else:
            rho = 60 * tau / peak
        if not bool((rho > 0).all()):
            raise ValueError("rho must be positive: give tau > 0 or rho")
        self.tau = tau
        self.rho = rho

    def _grid_buffers(self) -> float:
        psf = self.psf
        ndim = 2 if psf is None else torch.as_tensor(psf).ndim
        return _GRID_BASE + _GRID_COMPONENT * (ndim * (ndim + 1) // 2)

    def _grid_shape(
        self, loaded: Sequence[int], kernel: Sequence[int]
    ) -> tuple[int, ...]:
        if self.options["boundary"] == "periodic":
            return tuple(int(n) for n in loaded)
        return super()._grid_shape(loaded, kernel)

    def _block_operator(
        self,
        psf: Tensor,
        chunk: Chunk,
        shape: tuple[int, ...],
        cache: dict,
    ) -> PaddedConvolution | _CircularConvolution:
        if self.options["boundary"] == "unknown":
            return super()._block_operator(psf, chunk, shape, cache)
        if shape not in cache:
            cache[shape] = _CircularConvolution(psf, shape)
        return cache[shape]

    def _solve(
        self,
        blk: _Problem,
        op: PaddedConvolution | _CircularConvolution,
        index: tuple[int | slice, ...],
        trace: dict | None = None,
    ) -> tuple[Tensor, int]:
        tau = float(self.tau[index].squeeze())
        rho = float(self.rho[index].squeeze())
        data = blk.data
        if isinstance(op, _CircularConvolution):
            region = None
            x0 = data
        else:
            region = tuple(slice(0, n) for n in op.shape)
            mirror = [
                (g - n) // 2 for g, n in zip(op.grid, op.shape, strict=True)
            ]
            x0 = smooth_periodic_extension(data, op.grid, mirror)
        terms = [
            Term(Convolution(op.otf, op.grid), ShiftedPoisson(data), region),
            Term(
                Hessian(op.ndim),
                Schatten(tau, self.options["p"], ndim=op.ndim),
            ),
            Term(None, Box(0.0)),
        ]
        tol = self.options["tol"]
        res = admm_fft(
            terms,
            op.grid,
            rho=rho,
            x0=x0,
            order="z-first",
            max_iter=int(self.options["max_iter"]),
            tol=None if tol is None else float(tol),
            dtype=data.dtype,
            device=data.device,
        )
        extras = self.history.extras
        change = res.change if math.isfinite(res.change) else math.nan
        extras.setdefault("change", []).append(change)
        extras.setdefault("primal_residual", []).append(res.primal_residual)
        extras.setdefault("dual_residual", []).append(res.dual_residual)
        return res.x, res.n_iter

    def _extra_state(self) -> dict[str, object]:
        state = super()._extra_state()
        state["tau"] = self.tau
        state["rho"] = self.rho
        return state

    def _load_extra_state(self, state: dict[str, object]) -> None:
        super()._load_extra_state(state)
        self.tau = state.get("tau")
        self.rho = state.get("rho")


def hspiral(
    data: ArrayLike,
    psf: Tensor,
    *,
    out: WritableArrayLike | None = None,
    sensor: SensorEstimate | None = None,
    shading: ShadingEstimate | None = None,
    layout: Layout | None = None,
    **options: object,
) -> WritableArrayLike:
    """HSPIRAL deconvolution of Lefkimmiatis & Unser (2013) of `data`.

    Shortcut for `HSPIRAL(psf, **options).fit(data, init=[sensor,
    shading]).apply(data, out)`; see [`HSPIRAL`][optix.deconv.HSPIRAL]
    for the model, the defaults and the extensions.

    Parameters
    ----------
    data : array-like or Labeled
        Images; `(*samples, *spatial)` without a layout, with as many
        spatial dims as the PSF.
    psf : Tensor
        Point spread function, centred at index `K // 2`.
    out : writable array-like, optional
        Output. Default: a new tensor.
    sensor : SensorEstimate, optional
        Camera offset and gain (the read noise is not used).
    shading : ShadingEstimate, optional
        Dark and flat fields.
    layout : Layout, optional
        Axis roles.
    **options
        Options of [`HSPIRAL`][optix.deconv.HSPIRAL] (`tau`, `p`,
        `max_iter`, ...).

    Returns
    -------
    writable array-like
        `out`, the deconvolved images (in photons).
    """
    est = HSPIRAL(psf, layout=layout, **options)
    init = [e for e in (sensor, shading) if e is not None]
    est.fit(data, init=init or None)
    return est.apply(data, out)
