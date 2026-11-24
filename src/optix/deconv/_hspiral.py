"""Lefkimmiatis & Unser (2013): Poisson deconvolution with a Hessian
Schatten-norm prior (HSPIRAL)."""

import math
import numbers
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
    HessianSchatten,
    Schatten,
    ShiftedPoisson,
    Term,
    _spectral,
    admm_fft,
)
from ._fft import PaddedConvolution, transfer_function
from ._linear import smooth_periodic_extension
from ._variational import _Problem, _Variational

__all__ = ["HSPIRAL", "hspiral"]

_BOUNDARY = ("periodic", "unknown")
_VARIANTS = (1, 2)
#: Constant κ of the default weight τ = κ / (q₉₉.₉(y) − q₅₀(y)) (not in
#: the paper).
_KAPPA = 1.6
#: Working buffers of `admm_fft` per voxel of the FFT grid (measured
#: peak), per number of Hessian components C: base + per_component·C,
#: for HSPIRAL2 and HSPIRAL1 (with its inner dual algorithm).
_GRID_BASE = {2: 21.0, 1: 27.0}
_GRID_COMPONENT = {2: 3.5, 1: 3.0}
#: Temporaries of the closed-form eigenvalue solver used for p ∈ {1, ∞}, in
#: buffers of min(grid, `_spectral._CHUNK`) voxels (measured peak), per
#: spatial dimension: a fixed term of about 9 MB (2-D) and 27 MB (3-D)
#: in float32, whatever the block size.
_SPECTRAL_BUFFERS = {2: 36.0, 3: 110.0}


def _is_int(value: object) -> bool:
    """An integer, not a bool or a float."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


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
    """Poisson deconvolution with a Hessian Schatten-norm prior (HSPIRAL).

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
    spectral norm (p = ∞, largest). The second differences are forward
    stencils (Section IV: "defined as in [16]", Lefkimmiatis, Ward &
    Unser 2013, Eq. 10; [`Hessian`][optix.optim.Hessian]).

    **Solvers.** The paper gives two ADMM algorithms for Eq. (12), with
    the z-update first (Eqs. 18a–c; see
    [`admm_fft`][optix.optim.admm_fft], `order="z-first"`), one penalty
    α and scaled duals s. Both start from x⁰ = y with zero duals, use
    α = 60τ/M (Sections IV-A and IV-C) and stop when
    ‖xᵏ⁺¹ − xᵏ‖₂ ≤ 10⁻⁵ ‖xᵏ‖₂ or after 400 iterations (Section IV-B).
    The result is xᵏ (the algorithms return x); it is not projected onto
    x ≥ 0. Each solve records its number of iterations, last relative
    change and last primal and dual residuals in `history.extras`.

    - **HSPIRAL1** (Algorithm 1, `variant=1`, the default: the variant
      of the paper's restoration results, Sections IV-B to IV-D).
      Splits z₁ = K x and z₂ = x (Eqs. 24–25): z₁ by the closed-form
      Poisson prox (Eq. 27), z₂ = prox of (τ/α)‖H·‖₁,ₚ + ι_C (Eq. 28),
      x = (KᵀK + I)⁻¹(Kᵀ(z₁ − s₁) + z₂ − s₂) (Eq. 29, in the Fourier
      domain), then the dual updates (Eq. 23). The prox of Eq. (28) has
      no closed form: it is computed by 5 iterations (Section IV-B) of
      the dual algorithm of Lefkimmiatis, Ward & Unser (2013,
      Algorithm 2; see
      [`prox_hessian_schatten`][optix.optim.prox_hessian_schatten] and
      [`HessianSchatten`][optix.optim.HessianSchatten]), started from
      its dual of the previous ADMM iteration (Section III-C). H has
      the boundary of that paper (Eq. 10: half-sample mirror,
      `Hessian(boundary="mirror")`).
    - **HSPIRAL2** (Algorithm 2, `variant=2`). Splits z₁ = K x, z₂ = H x,
      z₃ = x (Eqs. 30–31): z₁ by Eq. (27), z₂ by the Schatten-norm prox
      (Proposition 1; closed forms of Eqs. 43–47, see
      [`prox_schatten`][optix.optim.prox_schatten]), z₃ by projection
      onto x ≥ 0 (Eq. 32), then the x-update (Eq. 42) in the Fourier
      domain, and the dual updates. The Fourier solve needs H periodic
      (D_ab = D_a D_b with wrap-around; Lefkimmiatis, Bourquard & Unser
      2012, Eq. 21), so the two variants differ in the Hessian at the
      last two rows and columns of the block.

    **Boundary.** The paper assumes periodic (or mirror) boundaries for
    the FFT solve of Eqs. (29) and (42) and does not say which its
    experiments use; the default here is periodic: the blur (and, for
    HSPIRAL2, the differences) wrap around the block, as in a
    circular-blur simulation. `boundary="unknown"` is an extension (not
    in the paper): x lives on a larger grid (the block plus the PSF, as
    in [`Ikoma`][optix.deconv.Ikoma]), the Poisson term sums over the
    observed pixels only, and x⁰ is the data continued smoothly over the
    margin; it avoids wrap-around artefacts on real images and is the
    default of the `faithful=False` preset.

    **Choices not specified in the paper.**

    - M in α = 60τ/M is the maximum intensity of the ground-truth image
      in the paper; here it is max(y) per image.
    - p: the paper compares 1, 2 and ∞ with no default; p = 1 gave the
      best pooled PSNR in its Table I and is the default. All three are
      the paper's method.
    - Dimension: the paper is 2-D. In 3-D both variants are unchanged,
      with the 3 × 3 Hessian (6 components, ‖H‖² ≤ 144 for the step of
      the inner algorithm of HSPIRAL1): p = 2 needs no eigenvalues, and
      p ∈ {1, ∞} use the closed-form 3 × 3 eigen-decomposition of
      [`prox_schatten`][optix.optim.prox_schatten].
    - HSPIRAL1 warm start: the inner algorithm restarts its momentum at
      every ADMM iteration (t₁ = 1, Ψ₁ = Ω from the previous iteration);
      the paper says only that its internal variables are initialised
      with those of the previous iteration.
    - The regularisation weight τ: the paper tunes τ per image for the
      best PSNR. The default here is

          τ = κ / (q₉₉.₉(y) − q₅₀(y)),   κ = 1.6,

      per image, with q₉₉.₉(y) the 99.9th percentile of the data
      (computed exactly in the one pass of `fit`; a hot pixel barely
      changes it) and q₅₀(y) their median (from a uniform sample of 8192
      pixels in the same pass, exact for smaller images). The
      difference is a robust range of the signal above the background:
      the Hessian ignores a constant, so it is the signal amplitude, not
      the intensity, that sets the scale of the prior. The 1/scale
      factor is that of a MAP estimate whose prior on the normalised
      image x/s (s the signal scale) is the same at every photon count:
      the Poisson log-likelihood is homogeneous, ℓ(s x̂; s ŷ) = s ℓ(x̂; ŷ)
      up to a constant, so ℓ(x; y) + κ‖H x/s‖ has the weight τ = κ/s.
      The rule needs q₉₉.₉(y) > q₅₀(y) (pass `tau` otherwise).

      κ is a 2-D calibration. It was found with HSPIRAL2 (in 2-D both
      variants have the same best κ: they minimise Eq. 12 with Hessians
      that differ only at the last two rows and columns), for the
      paper's α rule and stopping rule, on 2-D simulated images with
      circular blur (Gaussian, σ_b = 2, 9 × 9;
      uniform 7 × 7): blobs, filaments, their mixture and
      piecewise-smooth scenes, peaks of 10, 100 and 1000 photons on a
      constant background of 0, 1, 3 and 10 times the peak, 64 × 64
      (two seeds) and 128 × 128 (one seed) pixels, 288 cases, against
      a grid of τ (factor √2). Among the scales q₉₉.₉, q₉₉.₉ − q₁ and
      q₉₉.₉ − q₅₀, the last gives the smallest mean and worst-case
      loss; with q₉₉.₉ alone the best κ grows with the background (about
      18 times from no background to 10 times the peak). The best κ
      depends only weakly on the photon count for smooth content (blobs
      and piecewise-smooth scenes: about 1.3 to 2.9 at every count) but
      falls by up to 5 times for thin filaments between 10 and 1000
      photons (about 1.6 to 0.3). With κ = 1.6 the PSNR is on
      average 0.4 dB below that of the best τ of the grid (at most 2.4,
      2.7 and 3.2 dB for p = 1, 2, ∞; the grid was searched with p = 1)
      and at least 1.3 dB above that of the data (8 dB on average).
      In 3-D the same κ was about 2 to 4 times too large for both
      variants on the scenes tried (24 × 32 × 32, peaks 30 and 300:
      0.55 to 1.24 dB below the best τ), so a smaller τ is worth trying
      there. Pass `tau` to set the weight; the values used are in `tau`
      and `rho` after `fit`.

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
    duals u of `admm_fft`; the inner algorithm's Ω, Ψ, t are those of
    [`prox_hessian_schatten`][optix.optim.prox_hessian_schatten].

    !!! references "References"
        1. Lefkimmiatis S, Unser M. *Poisson image reconstruction with
           Hessian Schatten-norm regularization.* IEEE Trans Image
           Process **22**(11):4314-4327, 2013.
           [doi:10.1109/TIP.2013.2271852](https://doi.org/10.1109/TIP.2013.2271852)
        2. Lefkimmiatis S, Ward JP, Unser M. *Hessian Schatten-norm
           regularization for linear inverse problems.* IEEE Trans Image
           Process **22**(5):1873-1888, 2013.
           [doi:10.1109/TIP.2013.2237919](https://doi.org/10.1109/TIP.2013.2237919)
        3. Lefkimmiatis S, Bourquard A, Unser M. *Hessian-based norm
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
            @article{lefkimmiatis2013hessian,
              author  = {Lefkimmiatis, Stamatios and Ward, John Paul and
                         Unser, Michael},
              title   = {{Hessian} {Schatten}-Norm Regularization for
                         Linear Inverse Problems},
              journal = {IEEE Transactions on Image Processing},
              volume  = {22},
              number  = {5},
              pages   = {1873--1888},
              year    = {2013},
              doi     = {10.1109/TIP.2013.2237919}
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
    median : Tensor
        Median q₅₀(y) per image (used by the default τ), from a uniform
        sample of 8192 pixels (exact for smaller images).

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
        "variant": 1,
        "inner_iter": 5,
        "rho": None,
        "max_iter": 400,
        "tol": 1e-5,
        "boundary": "periodic",
        "halo": None,
        "block": None,
    }
    #: `faithful=False` preset: the unobserved margin, which avoids the
    #: wrap-around of the periodic boundary on real images.
    recommended = {**paper, "boundary": "unknown"}
    #: Options that do not make an estimate unfaithful: τ is tuned per
    #: image, p ∈ {1, 2, ∞} are all compared in the paper and both
    #: variants are its algorithms; `halo` and `block` are memory
    #: settings.
    _free = ("tau", "p", "variant", "halo", "block")

    def __init__(
        self,
        psf: Tensor | None = None,
        *,
        tau: float | None = None,
        p: float | str | None = None,
        variant: int | None = None,
        inner_iter: int | None = None,
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
            Regularisation weight τ ≥ 0. Default:
            1.6 / (q₉₉.₉(y) − q₅₀(y)) per image (see above; not in the
            paper).
        p : {1, 2, inf}
            Order of the Schatten norm. Default: 1 (all three are the
            paper's method).
        variant : {1, 2}
            HSPIRAL1 (default, the paper's experiments) or HSPIRAL2; both
            are faithful.
        inner_iter : int
            Iterations of the inner prox of HSPIRAL1 per ADMM iteration.
            Default (paper): 5. Unused by HSPIRAL2, where it does not
            change `is_faithful`.
        rho : float, optional
            ADMM penalty α > 0. Default (paper): 60τ / max y per image.
        max_iter : int
            Maximum number of iterations. Default (paper): 400.
        tol : float
            Relative-change tolerance. Default (paper): 10⁻⁵.
        boundary : {"periodic", "unknown"}
            Periodic blur and differences on the block (default, paper),
            or an unobserved margin around it (extension; the default
            with `faithful=False`).
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
        self.median: Tensor | None = None
        self.noise: Tensor | None = None
        self.tau: Tensor | None = None
        self.rho: Tensor | None = None
        self._check()

    def _resolve(self, faithful: bool, params: object) -> None:
        super()._resolve(faithful, params)
        free = self._free
        if self.options.get("variant") == 2:  # HSPIRAL2 has no inner loop
            free = (*free, "inner_iter")
        self.is_faithful = all(
            _same(value, self.paper.get(name))
            for name, value in self.options.items()
            if name not in free
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
        if not _is_int(opts["variant"]) or opts["variant"] not in _VARIANTS:
            raise ValueError(
                f"variant must be 1 or 2, got {opts['variant']!r}"
            )
        if not _is_int(opts["inner_iter"]) or opts["inner_iter"] < 1:
            raise ValueError(
                f"inner_iter must be a positive integer, got "
                f"{opts['inner_iter']!r}"
            )
        psf = self.psf
        if psf is not None and torch.as_tensor(psf).ndim not in (2, 3):
            raise ValueError("psf must be 2-D or 3-D")
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
        if psf.ndim not in (2, 3):
            raise ValueError("psf must be 2-D or 3-D")
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
            spread = self.robust_peak - self.median
            if not bool((spread > 0).all()):
                raise ValueError(
                    "HSPIRAL needs data whose 99.9th percentile is above "
                    "their median in every image for the default tau: pass "
                    "tau"
                )
            tau = _KAPPA / spread
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

    def _block_bytes(
        self,
        core: Sequence[int],
        extents: Sequence[int],
        halo: Sequence[int],
        psf: Tensor,
        itemsize: int,
    ) -> float:
        nbytes = super()._block_bytes(core, extents, halo, psf, itemsize)
        if self.options["p"] == 2:
            return nbytes
        loaded = [
            min(c + 2 * h, e)
            for c, e, h in zip(core, extents, halo, strict=True)
        ]
        grid = math.prod(self._grid_shape(loaded, tuple(psf.shape)))
        chunk = min(grid, _spectral._CHUNK)
        return nbytes + itemsize * _SPECTRAL_BUFFERS[psf.ndim] * chunk

    def _grid_buffers(self) -> float:
        psf = self.psf
        ndim = 2 if psf is None else torch.as_tensor(psf).ndim
        variant = self.options["variant"]
        components = ndim * (ndim + 1) // 2
        return _GRID_BASE[variant] + _GRID_COMPONENT[variant] * components

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
        p = self.options["p"]
        blur = Term(Convolution(op.otf, op.grid), ShiftedPoisson(data), region)
        if self.options["variant"] == 1:
            # Algorithm 1: z₂ = x, f₂ = τ‖H·‖₁,ₚ + ι_C (Eqs. 24–25, 28)
            penalty = HessianSchatten(
                tau,
                p,
                ndim=op.ndim,
                lower=0.0,
                n_iter=int(self.options["inner_iter"]),
                boundary="mirror",
            )
            terms = [blur, Term(None, penalty)]
        else:
            # Algorithm 2: z₂ = H x, z₃ = x (Eqs. 30–31)
            terms = [
                blur,
                Term(Hessian(op.ndim), Schatten(tau, p, ndim=op.ndim)),
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
