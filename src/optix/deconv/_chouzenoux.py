"""Chouzenoux et al. (2015): deconvolution with the exact Poisson–Gaussian
likelihood, solved by primal–dual splitting."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .._estimator import Tags, _same
from ..estimates import SensorEstimate, ShadingEstimate
from ..io import ArrayLike, Layout, WritableArrayLike
from ..optim import (
    Box,
    Gradient,
    GroupL2,
    Hessian,
    PoissonGaussian,
    primal_dual,
    primal_dual_step,
)
from ._fft import PaddedConvolution
from ._variational import _hashable, _Problem, _Variational

__all__ = ["Chouzenoux", "chouzenoux"]

#: Constants κ of the default weights λ = κ / (q₉₉.₉(y) − q₅₀(y)) (not in
#: the paper).
_KAPPA_TV = 0.2
_KAPPA_HESSIAN = 0.5
#: Largest admitted (max y + σ²)/γ (not in the paper): beyond it the
#: step is too small for the iterations to get anywhere.
_MAX_CONDITION = 1e4
#: Working buffers of `primal_dual` per voxel of the FFT grid (measured
#: peak, series included), per number of regulariser components C:
#: base + per_component·C.
_GRID_BASE = 16.0
_GRID_COMPONENT = 5.5


class Chouzenoux(_Variational):
    """Deconvolution with the exact Poisson–Gaussian likelihood.

    **Model** (Chouzenoux et al. 2015, Eqs. 2.1–2.7). In electrons, the
    data are y = Poisson(A x) + N(0, σ²), with A the convolution with
    the PSF and x ≥ 0. The estimate minimises (Eqs. 2.25–2.28, 3.1)

        φ(A x) + ι_C(x) + λ_TV ∑ⱼ ‖(∇x)ⱼ‖₂ + λ_H ∑ⱼ ‖(H x)ⱼ‖₂,

    where φ is the exact negative log-likelihood of the observed pixels,
    −log ∑ₙ zⁿe^{−z}/n! · e^{−(y − n)²/2σ²}/√(2πσ²) per pixel, continued
    by a quadratic for z < 0 (Eq. 2.28; see
    [`PoissonGaussian`][optix.optim.PoissonGaussian]); C = [0, x₊]
    (x₊ = `upper`, +∞ by default); ∇x is the gradient by forward
    differences (isotropic total variation) and (H x)ⱼ packs the second
    differences with √2 on the cross terms, so ‖(H x)ⱼ‖₂ is the
    Frobenius norm of the Hessian matrix (Section 3.1; the TVH prior of
    the paper's experiments). With `hessian=0` the prior is the total
    variation alone (TV, also the paper's method). The data are used as
    they are: the exact likelihood is convex for any y, so negative
    values are not clipped.

    **Solver** (Algorithm 1). The primal–dual forward–backward–forward
    iteration of [`primal_dual`][optix.optim.primal_dual], with
    h = φ(A ·) (∇h = Aᵀ(1 − ξ(A x)), Eqs. 2.8, 2.10), ψ₀ = ι_C
    (projection onto C) and one dual variable per regulariser (its prox
    is the projection onto the ℓ₂ balls of radius λ). The step is the
    largest of Theorem 3.2 (iii), γ = (1 − ε)/δ with
    δ = μ + √(∑_r ‖V_r‖²), ‖∇‖² ≤ 4d, ‖H‖² ≤ (4d)² (d dimensions) and the
    Lipschitz constant of Eq. (2.13),

        μ = ‖A‖² · maxᵢ (1 − e^{−1/σᵢ²}) e^{(2yᵢ − 1)/σᵢ²},

    with ‖A‖ = 1 for a unit-sum PSF and the maximum over the pixels of
    the whole image (so that every block of an image uses the same
    step).

    **The step collapses at high counts.** μ grows like e^{2 max y/σ²}:
    the method is usable for low counts with high read noise (the paper
    uses peaks of 12 to 60 electrons with σ² = 9 to 36). Iterations
    advance x by about γ per unit of gradient, so the number needed
    grows with (max y + σ²)/γ; when this exceeds 10⁴ (not in the paper),
    `fit` raises a `ValueError` that points to
    [`Ikoma`][optix.deconv.Ikoma], whose shifted-Poisson approximation of
    the same noise model has no such limit. Below it, with the default
    stopping rule, the result can still be stopped before convergence
    when the step is small (see `history.extras["change"]` and
    `"residual"`).

    **Series.** ξ and φ are evaluated with
    [`poisson_normal_ratio`][optix.special.poisson_normal_ratio] and
    [`poisson_normal_logpdf`][optix.special.poisson_normal_logpdf]: in
    float64, over a window of n adapted per pixel that leaves out terms
    below e⁻⁵⁰ times the largest one. This departs from the paper's
    window n* ± Δσ around the Lambert-function estimate n* with a fixed
    Δ = 3 (Section 3.4, Eqs. 3.4–3.7); the truncation errors are
    negligible, as Theorem 3.2 (iv) requires.

    **Choices not specified in the paper.**

    - x₀ is the mean of the image's data (constant), the duals start at
      0, and ε = min(10⁻³, ½/(δ + 1)) in the step.
    - Stopping: ‖xₖ₊₁ − xₖ‖₂ ≤ 10⁻⁵ ‖xₖ‖₂ or 5000 iterations (the paper
      stops "when the difference in norm between two consecutive
      iterates becomes small" and reports 2000 to 9000 iterations). The
      result is p₁ of the last iteration, which lies in C.
    - Boundary: the blur is computed on a grid larger than the block
      (the block plus the PSF, rounded up to a fast FFT size), the
      likelihood sums over the observed pixels only, and the
      differences are periodic on that grid (as for
      [`Ikoma`][optix.deconv.Ikoma]).
    - The weights: the paper tunes λ_TV and λ_H per image for the least
      mean absolute error. The defaults here are

          λ_TV = κ_TV / (q₉₉.₉(y) − q₅₀(y)),   λ_H = κ_H / (q₉₉.₉(y) − q₅₀(y)),

      per image, κ_TV = 0.2 and κ_H = 0.5, with q₉₉.₉(y) the 99.9th
      percentile of the data (exact, in the one pass of `fit`) and
      q₅₀(y) their median (from a uniform sample of 8192 pixels, exact
      for smaller images), as for [`HSPIRAL`][optix.deconv.HSPIRAL]: the
      priors ignore a constant, so the signal amplitude above the
      background sets their scale, and the 1/scale factor makes the
      prior on the normalised image the same at every count (the
      likelihood is close to homogeneous, ℓ(s x; s y) ≈ s ℓ(x; y)).
      The ratio κ_TV/κ_H = 0.4 is within the flat optimum of a pilot
      grid of both weights; with it, the scale was calibrated for the
      default stopping rule on simulated images (Gaussian PSF,
      σ_b = 1.6, 9 × 9, blurred on a larger field and cropped): cells,
      filaments and their mixture; peak 12 (σ² = 9), 30 (σ² = 16) and
      60 (σ² = 36) electrons with a smooth background of 0, 1/3 or 1/2
      of the peak; 32 × 32 (two seeds) and 64 × 64 (one seed) pixels;
      47 cases inside the usable range ((max y + σ²)/γ from 215 to
      about 10⁴), against seven scales of the weights from 1/4 to 4
      times the default. There, the default SNR is on average 0.23 dB
      below that of the best scale (at most 1.5 dB) and 2.4 to 11 dB
      (6.6 dB on average) above that of the data. In two thirds of
      these runs the 5000 iterations end before the tolerance is met;
      the calibration includes this early stop. Pass `tv` and `hessian`
      to set the weights; the values used are in `tv` and `hessian`
      after `fit`.

    **Camera calibration** (not in the paper, which takes electrons).
    With a [`SensorEstimate`][optix.estimates.SensorEstimate] given to
    `fit`, y = (data − offset)/gain and σ² = variance/gain² (maps
    allowed: the likelihood and μ use the variance of each pixel); with
    a [`ShadingEstimate`][optix.estimates.ShadingEstimate], the dark
    field is subtracted and the data are divided by the flat field (unit
    median), as for [`Ikoma`][optix.deconv.Ikoma]. A sensor estimate
    with a positive variance is required: without read noise the model
    is Poisson, whose likelihood has no Lipschitz gradient.

    **Blocks.** An image that fits in the memory budget is solved whole;
    a larger one is cut into blocks with a halo (default: twice the PSF
    size, not specified in the paper), each solved on its own, keeping
    its core; the weights and γ are those of the whole image. A
    `RuntimeWarning` says when an image is split. `halo` and `block` are
    memory settings: they do not change `is_faithful`.

    **Cost.** Each iteration evaluates the series twice over every
    pixel, and thousands of iterations are typical: the method is slow
    on large images and in 3-D.

    Paper notation: H the blur (A here); Δʰ, Δᵛ the first and Δʰʰ, Δʰᵛ,
    Δᵛᵛ the second differences (∇ and H here); μ, γ, ε, δ, ξ, η, λ_TV,
    λ_H as here; ψ₀ = ι_C.

    !!! note "Not implemented"
        The paper also compares approximations of the likelihood (its
        Table 1: Gaussian, generalised Anscombe transform, exponential,
        shifted Poisson, weighted least squares) and a non-local total
        variation prior. Only the exact likelihood with the TV and TV +
        Hessian priors is implemented; the shifted-Poisson approximation
        is [`Ikoma`][optix.deconv.Ikoma].

    !!! references "References"
        1. Chouzenoux E, Jezierska A, Pesquet J-C, Talbot H. *A convex
           approach for image restoration with exact Poisson–Gaussian
           likelihood.* SIAM J Imaging Sci **8**(4):2662-2682, 2015.
           [doi:10.1137/15M1014395](https://doi.org/10.1137/15M1014395)
        2. Combettes PL, Pesquet J-C. *Primal-dual splitting algorithm
           for solving inclusions with mixtures of composite, Lipschitzian,
           and parallel-sum type monotone operators.* Set-Valued Var Anal
           **20**(2):307-330, 2012.
           [doi:10.1007/s11228-011-0191-y](https://doi.org/10.1007/s11228-011-0191-y)

        ??? quote "BibTeX"
            ```bibtex
            @article{chouzenoux2015convex,
              author  = {Chouzenoux, Emilie and Jezierska, Anna and
                         Pesquet, Jean-Christophe and Talbot, Hugues},
              title   = {A Convex Approach for Image Restoration with
                         Exact {Poisson}--{Gaussian} Likelihood},
              journal = {SIAM Journal on Imaging Sciences},
              volume  = {8},
              number  = {4},
              pages   = {2662--2682},
              year    = {2015},
              doi     = {10.1137/15M1014395}
            }
            @article{combettes2012primal,
              author  = {Combettes, Patrick L. and Pesquet, Jean-Christophe},
              title   = {Primal-Dual Splitting Algorithm for Solving
                         Inclusions with Mixtures of Composite,
                         {Lipschitzian}, and Parallel-Sum Type Monotone
                         Operators},
              journal = {Set-Valued and Variational Analysis},
              volume  = {20},
              number  = {2},
              pages   = {307--330},
              year    = {2012},
              doi     = {10.1007/s11228-011-0191-y}
            }
            ```

    Attributes
    ----------
    tv, hessian : Tensor
        Weights λ_TV and λ_H per image `(*group, *sample)`, after `fit`.
    lipschitz : Tensor
        μ per image (Eq. 2.13); inf when it overflows.
    step : Tensor
        γ per image.
    peak : Tensor
        max y per image.
    robust_peak, median : Tensor
        q₉₉.₉(y) and q₅₀(y) per image (used by the default weights); the
        median is computed from a uniform sample of 8192 pixels (exact
        for smaller images).
    mean : Tensor
        Mean of y per image (x₀).

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import Chouzenoux
    >>> from optix.estimates import SensorEstimate
    >>> psf = torch.ones(5, 5)
    >>> noise = 3 * torch.randn(32, 32)
    >>> data = torch.poisson(torch.full((32, 32), 5.0)) + noise
    >>> sensor = SensorEstimate(
    ...     offset=torch.tensor(0.0), gain=torch.tensor(1.0),
    ...     variance=torch.tensor(9.0),
    ... )
    >>> est = Chouzenoux(psf, max_iter=5).fit(data, init=sensor)
    >>> est.apply(data).shape
    torch.Size([32, 32])
    """

    tags = Tags(
        targets=("deconvolution",),
        streaming="halo",
        noise="poisson-gaussian",
        reference="doi:10.1137/15M1014395",
    )
    paper = {
        "tv": None,
        "hessian": None,
        "upper": None,
        "step": None,
        "max_iter": 5000,
        "tol": 1e-5,
        "halo": None,
        "block": None,
    }
    recommended = dict(paper)
    #: Options that do not make an estimate unfaithful: the weights are
    #: tuned per image in the paper (TV alone and TV + Hessian are both
    #: its method), the upper bound x₊ of C is the paper's when known,
    #: and `halo` and `block` are memory settings.
    _free = ("tv", "hessian", "upper", "halo", "block")
    _clip_data = False

    def __init__(
        self,
        psf: Tensor | None = None,
        *,
        tv: float | None = None,
        hessian: float | None = None,
        upper: float | None = None,
        step: float | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
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
        tv : float, optional
            Weight λ_TV ≥ 0 of the total variation (0 removes it).
            Default: κ_TV / (q₉₉.₉(y) − q₅₀(y)) per image (see above; not
            in the paper).
        hessian : float, optional
            Weight λ_H ≥ 0 of the Frobenius norm of the Hessian (0
            removes it: the TV prior). Default: κ_H / (q₉₉.₉(y) − q₅₀(y))
            per image (not in the paper).
        upper : float, optional
            Upper bound x₊ > 0 of C = [0, x₊], in electrons. Default:
            none, C = [0, +∞) (the paper's choice when x₊ is unknown).
        step : float, optional
            Step γ > 0. Default (paper): the largest step of Theorem 3.2
            (see above). A given step skips the high-count check.
        max_iter : int
            Maximum number of iterations. Default: 5000 (not in the
            paper).
        tol : float
            Relative-change tolerance (0 runs `max_iter` iterations).
            Default: 10⁻⁵ (not in the paper).
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
        self.mean: Tensor | None = None
        self.noise: Tensor | None = None
        self.tv: Tensor | None = None
        self.hessian: Tensor | None = None
        self.lipschitz: Tensor | None = None
        self.step: Tensor | None = None
        self._eta_store: dict = {}
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
        psf = self.psf
        if psf is not None and torch.as_tensor(psf).ndim not in (2, 3):
            raise ValueError("psf must be 2-D or 3-D")
        for name in ("tv", "hessian"):
            if opts[name] is not None and not float(opts[name]) >= 0:
                raise ValueError(f"{name} must be non-negative")
        if opts["upper"] is not None and not float(opts["upper"]) > 0:
            raise ValueError("upper must be positive")
        if opts["step"] is not None and not (
            0 < float(opts["step"]) < math.inf
        ):
            raise ValueError("step must be positive")
        if int(opts["max_iter"]) < 1:
            raise ValueError("max_iter must be positive")
        if opts["tol"] is not None and not float(opts["tol"]) >= 0:
            raise ValueError("tol must be non-negative")

    # -- hooks -----------------------------------------------------------
    def _fit(self, data: ArrayLike, **kwargs: object) -> None:
        self._eta_store = {}  # filled by `_observe`, read by `_setup_images`
        try:
            super()._fit(data, **kwargs)
        finally:
            self._eta_store = {}

    def _psf(self, backend: dict) -> Tensor:
        if self.psf is None:
            raise ValueError("Chouzenoux needs a psf")
        psf = torch.as_tensor(self.psf).to(**backend)
        if (psf < 0).any():
            raise ValueError("psf must be non-negative")
        if not psf.sum() > 0:
            raise ValueError("psf must have a positive sum")
        return psf / psf.sum()

    def _observe(self, index: tuple, blk: _Problem) -> None:
        """Track maxᵢ log ηᵢ(0) per image (Eq. 2.30)."""
        var = torch.as_tensor(blk.variance).to(blk.data)
        if not bool((var > 0).all()):
            raise ValueError(
                "Chouzenoux needs the read-noise variance σ² > 0: fit with "
                "init=SensorEstimate(offset=..., gain=..., variance=...)"
            )
        y = blk.data.double()
        var = var.double()
        log_eta = torch.log(-torch.expm1(-1 / var)) + (2 * y - 1) / var
        top = float(log_eta.max())
        vmax = float(var.max())
        key = _hashable(index)
        store = self._eta_store
        if key in store:
            top = max(top, store[key][1])
            vmax = max(vmax, store[key][2])
        store[key] = (index, top, vmax)

    def _setup_images(self, psf: Tensor) -> None:
        opts = self.options
        store = self._eta_store
        log_eta = torch.full_like(self.peak, -math.inf)
        var_max = torch.zeros_like(self.peak)
        for index, value, vmax in store.values():
            log_eta[index] = value
            var_max[index] = vmax
        ndim = psf.ndim
        norms2 = []
        defaults = {"tv": _KAPPA_TV, "hessian": _KAPPA_HESSIAN}
        weights = {}
        for name, operator in (("tv", Gradient), ("hessian", Hessian)):
            if opts[name] is None:
                spread = self.robust_peak - self.median
                if not bool((spread > 0).all()):
                    raise ValueError(
                        "Chouzenoux needs data whose 99.9th percentile is "
                        "above their median in every image for the default "
                        "weights: pass tv and hessian"
                    )
                weights[name] = defaults[name] / spread
            else:
                weights[name] = torch.full_like(self.peak, float(opts[name]))
            if bool((weights[name] > 0).any()):
                norms2.append(operator(ndim).norm2)
        # ‖H‖² = max |otf|² = (∑ psf)² = 1 for a non-negative unit-sum PSF
        lipschitz = log_eta.exp()
        step = torch.empty_like(lipschitz)
        for i in range(lipschitz.numel()):
            mu = float(lipschitz.reshape(-1)[i])
            if opts["step"] is not None:
                gamma = float(opts["step"])
            else:
                gamma = primal_dual_step(mu, norms2)
                scale = float(self.peak.reshape(-1)[i]) + float(
                    var_max.reshape(-1)[i]
                )
                if not gamma > 0 or scale / gamma > _MAX_CONDITION:
                    raise ValueError(
                        "Chouzenoux: the exact Poisson–Gaussian likelihood "
                        "has a gradient of Lipschitz constant "
                        f"μ = e^{float(log_eta.reshape(-1)[i]):.1f} "
                        "(Eq. 2.13: μ grows like e^(2·max y/σ²)), so the "
                        f"admissible step γ = {gamma:.3g} is too small for "
                        "the iterations to converge. The method is usable "
                        "for low counts with high read noise only (max y/σ² "
                        "of a few units). Use Ikoma (shifted-Poisson "
                        "approximation of the same noise model) instead."
                    )
            step.reshape(-1)[i] = gamma
        self.tv = weights["tv"]
        self.hessian = weights["hessian"]
        self.lipschitz = lipschitz
        self.step = step

    def _grid_buffers(self) -> float:
        psf = self.psf
        ndim = 2 if psf is None else torch.as_tensor(psf).ndim
        return _GRID_BASE + _GRID_COMPONENT * (ndim + ndim * (ndim + 1) // 2)

    def _solve(
        self,
        blk: _Problem,
        op: PaddedConvolution,
        index: tuple[int | slice, ...],
        trace: dict | None = None,
    ) -> tuple[Tensor, int]:
        data = blk.data
        fidelity = PoissonGaussian(data, blk.variance)

        def grad(x: Tensor) -> Tensor:
            return op.adjoint(fidelity.grad(op.forward(x)))

        terms = []
        tv = float(self.tv[index].squeeze())
        hessian = float(self.hessian[index].squeeze())
        if tv > 0:
            terms.append((Gradient(op.ndim), GroupL2(tv)))
        if hessian > 0:
            terms.append((Hessian(op.ndim), GroupL2(hessian)))
        upper = self.options["upper"]
        constraint = Box(0.0, None if upper is None else float(upper))
        x0 = torch.full(
            op.grid,
            float(self.mean[index].squeeze()),
            dtype=data.dtype,
            device=data.device,
        )
        tol = self.options["tol"]
        res = primal_dual(
            grad,
            terms,
            x0,
            step=float(self.step[index].squeeze()),
            constraint=constraint,
            max_iter=int(self.options["max_iter"]),
            tol=None if tol is None else float(tol),
        )
        extras = self.history.extras
        change = res.change if math.isfinite(res.change) else math.nan
        extras.setdefault("change", []).append(change)
        extras.setdefault("residual", []).append(res.residual)
        return res.x, res.n_iter

    def _extra_state(self) -> dict[str, object]:
        state = super()._extra_state()
        for name in ("tv", "hessian", "lipschitz", "step"):
            state[name] = getattr(self, name)
        return state

    def _load_extra_state(self, state: dict[str, object]) -> None:
        super()._load_extra_state(state)
        for name in ("tv", "hessian", "lipschitz", "step"):
            setattr(self, name, state.get(name))


def chouzenoux(
    data: ArrayLike,
    psf: Tensor,
    *,
    out: WritableArrayLike | None = None,
    sensor: SensorEstimate | None = None,
    shading: ShadingEstimate | None = None,
    layout: Layout | None = None,
    **options: object,
) -> WritableArrayLike:
    """Deconvolution of Chouzenoux et al. (2015) of `data`.

    Shortcut for `Chouzenoux(psf, **options).fit(data, init=[sensor,
    shading]).apply(data, out)`; see
    [`Chouzenoux`][optix.deconv.Chouzenoux] for the model and the
    defaults.

    Parameters
    ----------
    data : array-like or Labeled
        Images; `(*samples, *spatial)` without a layout, with as many
        spatial dims as the PSF.
    psf : Tensor
        Point spread function, centred at index `K // 2`.
    out : writable array-like, optional
        Output. Default: a new tensor.
    sensor : SensorEstimate
        Camera model (offset, gain and read-noise variance, required).
    shading : ShadingEstimate, optional
        Dark and flat fields.
    layout : Layout, optional
        Axis roles.
    **options
        Options of [`Chouzenoux`][optix.deconv.Chouzenoux] (`tv`,
        `hessian`, `upper`, `max_iter`, ...).

    Returns
    -------
    writable array-like
        `out`, the deconvolved images (in electrons).
    """
    est = Chouzenoux(psf, layout=layout, **options)
    init = [e for e in (sensor, shading) if e is not None]
    est.fit(data, init=init or None)
    return est.apply(data, out)
