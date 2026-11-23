"""Ikoma et al. (2018): shifted-Poisson deconvolution with a Hessian prior."""

import math
import warnings
from collections.abc import Sequence

import torch
from torch import Tensor

from .._estimator import Tags, _same
from ..estimates import SensorEstimate, ShadingEstimate
from ..io import ArrayLike, Chunk, Labeled, Layout, WritableArrayLike
from ..optim import (
    Box,
    Convolution,
    Gradient,
    GroupL2,
    Hessian,
    ShiftedPoisson,
    Term,
    admm_fft,
)
from ._fft import PaddedConvolution
from ._variational import _Problem, _Variational

__all__ = ["Ikoma", "ikoma"]

_REGULARISER = ("hessian", "tv", "none")
_PSF_NORM = ("plane", "sum")
#: Constant κ of the default weight ν = κ·∑S / q₉₉.₉(ỹ) (not in the paper).
_KAPPA = 0.1
#: Working buffers of `admm_fft` per voxel of the FFT grid (measured
#: peak), per number of regulariser components C: base + per_component·C.
_GRID_BASE = 21.0
_GRID_COMPONENT = 2.5


class Ikoma(_Variational):
    """Shifted-Poisson deconvolution with a Hessian prior (Ikoma et al. 2018).

    **Model** (Eqs. 3–4). In electrons, after dark-field and flat-field
    correction and division by the gain g, the data are

        ỹ = Poisson(A x) + N(0, σ̃²),   σ̃² = σ²/g²,

    with A the 3-D convolution with the PSF, approximated by the shifted
    Poisson model ỹ + σ̃² ~ Poisson(A x + σ̃²), which has the same mean
    and variance. The estimate solves (Eq. 5, SI Eq. S8)

        minimise_x  ∑_{j∈C} (A x)ⱼ − (ỹⱼ + σ̃²) log((A x)ⱼ + σ̃²)
                    + ν ∑_{j∈Ω} ‖(H x)ⱼ‖₂  subject to x ≥ 0,

    where H x packs the six (three in 2-D) distinct second differences
    of x, the off-diagonal ones times √2 (SI Eq. S7), so ‖(H x)ⱼ‖₂ is
    the Frobenius norm of the Hessian at voxel j. Second differences
    compose forward differences, D_ab = D_a D_b (SI Eq. S5). The data
    are clipped at −σ̃² so that the fidelity is convex (SI §B).

    **Unknown boundary.** x lives on a grid N larger than the observed
    block (N > M), and the fidelity sums over the observed voxels C
    only, so that light from outside the field of view is modelled
    (Eq. 5). The grid is N = L + K − 1 (L the block, K the PSF), rounded
    up to a fast FFT size (not specified in the paper); the convolution
    and the differences are periodic on it, and Ω is the whole grid.

    **Solver** (SI §E). ADMM with the splits z₁ = A x, z₂ = x,
    z₃ = H x and one penalty ρ (see
    [`admm_fft`][optix.optim.admm_fft]): the x-update is solved in the
    Fourier domain (Eq. S16), z₁ by the closed-form shifted-Poisson
    prox on C and z₁ = A x + u₁ outside C (Eq. S17), z₂ by projection
    onto x ≥ 0 (Eq. S18), z₃ by block soft thresholding at ν/ρ
    (Eq. S19). All variables start at 0; ρ = 6000ν / max(ỹ)
    (ρ = 6 / max(ỹ) without a regulariser); the iterations stop when
    ‖xᵏ⁺¹ − xᵏ‖₂ < 10⁻³ ‖xᵏ‖₂ or after 150 iterations. The result is x
    restricted to the observed voxels (the paper does not say whether
    x or z₂ is returned); it is not projected onto x ≥ 0.

    **The stopping rule is part of the method.** With the paper's ρ
    rule, tolerance and iteration cap, the result is an early-stopped
    ADMM iterate, not the minimiser of the objective above. On
    simulated 2-D data (γ = 2, σ = 3, peaks of 10 to 1000
    photoelectrons) it is 26 to 34 % (relative ℓ2) away from the
    minimiser, and its SNR is 2 to 4.5 dB higher; 13 to 26 % of its
    voxels are negative, by less than 0.4 % of the peak. The stop
    therefore regularises too: the result depends on `tol`, `max_iter`
    and `rho`, not only on ν, and tightening `tol` with the default ν
    lowers the SNR. Each solve records its number of iterations, last
    relative change and last primal and dual residuals in
    `history.extras` (`"iterations"`, `"change"`, `"primal_residual"`,
    `"dual_residual"`); the residuals tend to 0 only at convergence.

    **PSF.** Normalised so that each xy-plane (the last two spatial
    axes) sums to 1 (SI §H): x is then in photoelectrons per exposure
    of one image of the focal stack. This normalisation assumes that
    every plane of the PSF carries the same energy (as for a widefield
    PSF); `psf_norm="sum"` normalises to unit sum instead (the optix
    convention). For a PSF of K_z planes of equal energy the two give
    x_plane = x_sum / K_z and ν_plane = K_z ν_sum. A PSF as large as the
    stack (as simulated in the paper) doubles the grid: crop it to its
    support.

    **Regularisation weight ν.** The paper tunes ν per data set (it
    scans 0.005–0.100 on its experimental stacks) and gives no default.
    The default here is not in the paper:

        ν = κ · ∑S / q₉₉.₉(ỹ),   κ = 0.1,

    per image, with q₉₉.₉(ỹ) the 99.9th percentile of the prepared
    data (computed exactly in the single pass of `fit`; unlike max ỹ, a
    hot pixel or a single bright bead barely changes it) and ∑S the sum
    of the normalised PSF (1 for `psf_norm="sum"`, the number of planes
    for `"plane"`). The 1/peak scaling is that of a MAP estimate with a
    prior on the normalised image x̂ = x/s (s the peak) that is the same
    at every photon count, −log p(x̂) = κ‖H x̂‖ + const: the objective
    ℓ(x; ỹ) + κ‖H x̂‖ is ℓ(x; ỹ) + (κ/s)‖H x‖, so ν = κ/s. The rule does
    not use σ̃²; with read noise the shifted-Poisson likelihood is not
    homogeneous in s (the Poisson one is, ℓ(s x̂; s ŷ) = s ℓ(x̂; ŷ) up
    to a constant), so it is an approximation there. The ∑S factor makes
    the default the same for both PSF normalisations.

    κ was calibrated for the default stopping rule (the paper's
    `tol`, `max_iter` and ρ rule) and is not valid with other values of
    them. Calibration data: blobs, filaments and their mixture, 2-D and
    3-D, Gaussian and Gaussian-beam PSFs, peaks of 10, 100 and 1000
    photoelectrons, γ = 2, σ = 3 as in the paper's simulation, three
    seeds each (54 cases). There, κ = 0.1 minimises the mean loss of SNR
    against the best ν of a fine grid: 1.2 dB on average, 4.8 dB in the
    worst case. The loss comes from the image content: smooth blobs
    want a larger ν (best κ up to 0.5), thin filaments a smaller one
    (best κ ≈ 0.006 to 0.03). In 6 of the 54 cases (2-D blobs at 1000
    and 2-D filaments at 100 photoelectrons) the default is 0.6 to
    1.1 dB below no regulariser at all. Pass `nu` to set the weight;
    the values used are in `nu` and `rho` after `fit`.

    **Camera calibration.** From the estimates given to `fit`
    (`init=sensor`, `init=shading` or both), which describe the raw
    data:

    - [`SensorEstimate`][optix.estimates.SensorEstimate]: ỹ =
      (y − offset)/gain, σ̃² = variance/gain² (maps allowed). The paper
      estimates g and σ with Foi's method (see `optix.sensor`). Without
      one, the data are taken in electrons with σ̃² = 0 (Poisson).
    - [`ShadingEstimate`][optix.estimates.ShadingEstimate]: the dark
      field d (which includes the camera offset) is subtracted in place
      of the offset, and the data are divided by the flat field f,
      rescaled to unit median as in the paper's Methods, before the
      division by the gain: ỹ = (y − d)/(f·gain). The read-noise
      variance of the corrected data is then σ̃² = variance/(f·gain)²;
      with f ≡ 1 this is the paper's model. The shot noise of the
      corrected data, of variance (A x)/f, is still modelled as A x
      (the paper's approximation). A per-frame baseline is ignored.

    **Blocks.** An image that fits in the memory budget is solved
    whole. A larger one is cut into blocks with a halo (default: twice
    the PSF size, not specified in the paper), each solved as an image
    of its own with the unknown boundary, keeping its core; ν and ρ are
    those of the whole image. The ADMM x-update is a global solve, so
    the block-wise result is not the whole-image one near the seams; a
    `RuntimeWarning` says when an image is split. The memory of a
    block is counted on its FFT grid N (the loaded block plus the PSF,
    rounded up), about 21 + 2.5·C buffers per grid voxel, with C the
    number of regulariser components (6 for the 3-D Hessian). When the
    budget only allows blocks whose core is less than a quarter of the
    loaded block, `apply` raises a `ValueError` that gives the memory a
    whole-image solve needs. `halo` and `block` are memory settings:
    they do not change `is_faithful`. Sample and group axes are iterated
    over.

    Paper notation: ỹ, σ̃, A, H, ν, ρ as here; the PSF is h.

    !!! note "Variants and extensions"
        - `regulariser="none"` (shifted Poisson with x ≥ 0 only) and,
          with σ̃² = 0, the Poisson model are the paper's comparison
          variants of Table 1 (ρ = 6/max ỹ). `regulariser="tv"`
          (isotropic total variation, also in Table 1) uses
          ρ = 6000ν/max ỹ, which is not specified in the paper; it is
          an extension.
        - `weights=(w₁, …, w_d)`: per-axis derivative weights, Dₐ
          replaced by wₐDₐ, for anisotropic sampling (the paper notes
          this is possible but uses unit weights), e.g.
          wₐ = min(spacing)/spacingₐ.
        - `psf_norm="sum"`: unit-sum PSF (see above).

    !!! references "References"
        1. Ikoma H, Broxton M, Kudo T, Wetzstein G. *A convex 3D
           deconvolution algorithm for low photon count fluorescence
           imaging.* Sci Rep **8**:11489, 2018.
           [doi:10.1038/s41598-018-29768-x](https://doi.org/10.1038/s41598-018-29768-x)

        ??? quote "BibTeX"
            ```bibtex
            @article{ikoma2018convex,
              author  = {Ikoma, Hayato and Broxton, Michael and Kudo,
                         Takamasa and Wetzstein, Gordon},
              title   = {A convex {3D} deconvolution algorithm for low
                         photon count fluorescence imaging},
              journal = {Scientific Reports},
              volume  = {8},
              pages   = {11489},
              year    = {2018},
              doi     = {10.1038/s41598-018-29768-x}
            }
            ```

    Attributes
    ----------
    nu : Tensor
        Regularisation weight ν per image `(*group, *sample)`, after
        `fit`.
    rho : Tensor
        ADMM penalty ρ per image, after `fit`.
    peak : Tensor
        max ỹ per image (used by the ρ rule).
    robust_peak : Tensor
        99.9th percentile q₉₉.₉(ỹ) per image (used by the default ν).

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import Ikoma
    >>> psf = torch.ones(5, 5)
    >>> data = torch.poisson(torch.full((32, 32), 10.0))
    >>> Ikoma(psf, max_iter=5).fit(data).apply(data).shape
    torch.Size([32, 32])
    """

    tags = Tags(
        targets=("deconvolution",),
        streaming="halo",
        noise="poisson-gaussian",
        reference="doi:10.1038/s41598-018-29768-x",
    )
    paper = {
        "nu": None,
        "regulariser": "hessian",
        "rho": None,
        "max_iter": 150,
        "tol": 1e-3,
        "psf_norm": "plane",
        "weights": None,
        "halo": None,
        "block": None,
    }
    recommended = dict(paper)
    #: Options that do not make an estimate unfaithful: ν is tuned per
    #: data set in the paper; `halo` and `block` are memory settings
    #: (the regulariser variants are handled in `_resolve`).
    _free = ("nu", "halo", "block")

    def __init__(
        self,
        psf: Tensor | None = None,
        *,
        nu: float | None = None,
        regulariser: str | None = None,
        rho: float | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        psf_norm: str | None = None,
        weights: Sequence[float] | None = None,
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
            `K // 2`. Normalised as set by `psf_norm`.
        nu : float, optional
            Regularisation weight ν ≥ 0. Default: 0.1·∑S / q₉₉.₉(ỹ) per
            image (see above; not in the paper, calibrated for the
            default `rho`, `max_iter` and `tol`).
        regulariser : {"hessian", "tv", "none"}
            Penalty: Frobenius norm of the Hessian (default, the paper's
            method), isotropic total variation (extension), or none.
        rho : float, optional
            ADMM penalty ρ > 0. Default (paper): 6000ν / max ỹ per image
            (6 / max ỹ without a regulariser).
        max_iter : int
            Maximum number of iterations. Default (paper): 150.
        tol : float
            Relative-change tolerance. Default (paper): 10⁻³.
        psf_norm : {"plane", "sum"}
            PSF normalisation: each xy-plane sums to 1 (default, paper)
            or the whole PSF does (extension).
        weights : sequence of float, optional
            Extension. Per-axis derivative weights, in layout order.
            Default: none (unit weights).
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
        self.nu: Tensor | None = None
        self.rho: Tensor | None = None
        self._check()

    def _resolve(self, faithful: bool, params: object) -> None:
        super()._resolve(faithful, params)
        self.is_faithful = all(
            _same(value, self.paper.get(name))
            for name, value in self.options.items()
            if name not in (*self._free, "regulariser")
        ) and self.options.get("regulariser", "hessian") in ("hessian", "none")

    def _check(self) -> None:
        opts = self.options
        psf = self.psf
        if psf is not None and torch.as_tensor(psf).ndim not in (2, 3):
            raise ValueError("psf must be 2-D or 3-D")
        if opts["regulariser"] not in _REGULARISER:
            raise ValueError(f"regulariser must be one of {_REGULARISER}")
        if opts["psf_norm"] not in _PSF_NORM:
            raise ValueError(f"psf_norm must be one of {_PSF_NORM}")
        if opts["nu"] is not None and not float(opts["nu"]) >= 0:
            raise ValueError("nu must be non-negative")
        if opts["rho"] is not None and not float(opts["rho"]) > 0:
            raise ValueError("rho must be positive")
        if int(opts["max_iter"]) < 1:
            raise ValueError("max_iter must be positive")
        if opts["tol"] is not None and not float(opts["tol"]) >= 0:
            raise ValueError("tol must be non-negative")
        w = opts["weights"]
        if w is not None:
            w = tuple(float(v) for v in w)
            if psf is not None and len(w) != torch.as_tensor(psf).ndim:
                raise ValueError("weights must have one entry per PSF dim")
            if any(not v > 0 for v in w):
                raise ValueError("weights must be positive")

    # -- hooks -----------------------------------------------------------
    def _psf(self, backend: dict) -> Tensor:
        if self.psf is None:
            raise ValueError("Ikoma needs a psf")
        psf = torch.as_tensor(self.psf).to(**backend)
        if (psf < 0).any():
            raise ValueError("psf must be non-negative")
        if not psf.sum() > 0:
            raise ValueError("psf must have a positive sum")
        if self.options["psf_norm"] == "sum" or psf.ndim == 2:
            return psf / psf.sum()
        plane = psf.sum(dim=(-2, -1), keepdim=True)
        return torch.where(plane > 0, psf / plane.clamp(min=1e-300), 0)

    def _init_from(self, init: object, shape: tuple, like: object) -> None:
        self.sensor = self._pick(init, SensorEstimate, shape=shape, like=like)
        self.shading = self._pick(
            init, ShadingEstimate, shape=shape, like=like
        )
        self._warn_unused(init, SensorEstimate, ShadingEstimate)
        if self.shading is not None and self.shading.baseline is not None:
            warnings.warn(
                "Ikoma ignores the per-frame baseline of the shading estimate",
                RuntimeWarning,
                stacklevel=4,
            )
        self.is_faithful = self._faithful_options

    def _maps(self, data: Labeled, backend: dict) -> dict[str, Tensor | None]:
        """Camera maps (data units), broadcast to `(*group, *spatial)`."""
        shape = data.layout.map_shape(data.array)

        def full(value: object) -> Tensor | None:
            if value is None:
                return None
            return torch.broadcast_to(
                torch.as_tensor(value).to(**backend), shape
            )

        sensor, shading = self.sensor, self.shading
        out = {
            name: full(None if sensor is None else getattr(sensor, name))
            for name in ("offset", "gain", "variance")
        }
        flat = None if shading is None else shading.flat
        if flat is not None:
            flat = flat.to(**backend)
            if (flat <= 0).any() or not flat.isfinite().all():
                raise ValueError("The flat field must be finite and positive")
            flat = flat / flat.median()
        out["flat"] = full(flat)
        out["dark"] = full(None if shading is None else shading.dark)
        return out

    def _prepare(self, chunk: Chunk, maps: dict) -> _Problem:
        """Electrons ỹ (clipped at −σ̃²) and read noise σ̃²."""
        index = chunk.group + self._region(chunk)

        def at(name: str) -> Tensor | None:
            value = maps[name]
            return None if value is None else value[index]

        y = chunk.data
        dark = at("dark")
        if dark is None:
            dark = at("offset")
        if dark is not None:
            y = y - dark
        flat = at("flat")
        if flat is not None:
            y = y / flat
        gain = at("gain")
        if gain is not None:
            y = y / gain
        var: Tensor | float = 0.0
        v = at("variance")
        if v is not None:
            var = v if gain is None else v / gain.square()
            if flat is not None:
                var = var / flat.square()
        y = torch.maximum(y, -torch.as_tensor(var).to(y))
        return _Problem(y, variance=var)

    def _setup_images(self, psf: Tensor) -> None:
        opts = self.options
        peak = self.peak
        if not bool((peak > 0).all()):
            raise ValueError(
                "Ikoma needs data with a positive maximum in every image"
            )
        if opts["nu"] is None:
            robust = self.robust_peak
            if not bool((robust > 0).all()):
                raise ValueError(
                    "Ikoma needs data with a positive 99.9th percentile in "
                    "every image for the default nu: pass nu"
                )
            scale = float(psf.sum(dtype=torch.float64))
            nu = _KAPPA * scale / robust
        else:
            nu = torch.full_like(peak, float(opts["nu"]))
        if opts["rho"] is not None:
            rho = torch.full_like(peak, float(opts["rho"]))
        elif opts["regulariser"] == "none":
            rho = 6 / peak
        else:
            rho = 6000 * nu / peak
        if not bool((rho > 0).all()):
            raise ValueError("rho must be positive: give nu > 0 or rho")
        self.nu = nu
        self.rho = rho

    def _grid_buffers(self) -> float:
        psf = self.psf
        ndim = 2 if psf is None else torch.as_tensor(psf).ndim
        comps = {
            "hessian": ndim * (ndim + 1) // 2,
            "tv": ndim,
            "none": 0,
        }[self.options["regulariser"]]
        return _GRID_BASE + _GRID_COMPONENT * comps

    def _regulariser(self, ndim: int) -> object:
        reg = self.options["regulariser"]
        weights = self.options["weights"]
        if reg == "hessian":
            return Hessian(ndim, weights=weights)
        if reg == "tv":
            return Gradient(ndim, weights=weights)
        return None

    def _solve(
        self,
        blk: _Problem,
        op: PaddedConvolution,
        index: tuple[int, ...],
        trace: dict | None = None,
    ) -> tuple[Tensor, int]:
        nu = float(self.nu[index].squeeze())
        rho = float(self.rho[index].squeeze())
        region = tuple(slice(0, n) for n in op.shape)
        terms = [
            Term(
                Convolution(op.otf, op.grid),
                ShiftedPoisson(blk.data, blk.variance),
                region,
            ),
            Term(None, Box(0.0)),
        ]
        reg = self._regulariser(op.ndim)
        if reg is not None:
            terms.append(Term(reg, GroupL2(nu)))
        tol = self.options["tol"]
        res = admm_fft(
            terms,
            op.grid,
            rho=rho,
            max_iter=int(self.options["max_iter"]),
            tol=None if tol is None else float(tol),
            dtype=blk.data.dtype,
            device=blk.data.device,
        )
        extras = self.history.extras
        change = res.change if math.isfinite(res.change) else math.nan
        extras.setdefault("change", []).append(change)
        extras.setdefault("primal_residual", []).append(res.primal_residual)
        extras.setdefault("dual_residual", []).append(res.dual_residual)
        return res.x, res.n_iter

    def _extra_state(self) -> dict[str, object]:
        state = super()._extra_state()
        sensor, shading = self.sensor, self.shading
        state["sensor"] = None if sensor is None else sensor._payload()
        state["shading"] = None if shading is None else shading._payload()
        state["nu"] = self.nu
        state["rho"] = self.rho
        return state

    def _load_extra_state(self, state: dict[str, object]) -> None:
        super()._load_extra_state(state)
        sensor = state.get("sensor")
        if sensor is not None:
            self.sensor = SensorEstimate._from_payload(sensor)
        shading = state.get("shading")
        if shading is not None:
            self.shading = ShadingEstimate._from_payload(shading)
        self.nu = state.get("nu")
        self.rho = state.get("rho")


def ikoma(
    data: ArrayLike,
    psf: Tensor,
    *,
    out: WritableArrayLike | None = None,
    sensor: SensorEstimate | None = None,
    shading: ShadingEstimate | None = None,
    layout: Layout | None = None,
    **options: object,
) -> WritableArrayLike:
    """Deconvolution of Ikoma et al. (2018) of `data` with a known PSF.

    Shortcut for `Ikoma(psf, **options).fit(data, init=[sensor,
    shading]).apply(data, out)`; see [`Ikoma`][optix.deconv.Ikoma] for
    the model, the defaults and the extensions.

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
        Camera model (offset, gain, read-noise variance).
    shading : ShadingEstimate, optional
        Dark and flat fields.
    layout : Layout, optional
        Axis roles.
    **options
        Options of [`Ikoma`][optix.deconv.Ikoma] (`nu`, `regulariser`,
        `max_iter`, ...).

    Returns
    -------
    writable array-like
        `out`, the deconvolved images (in electrons).
    """
    est = Ikoma(psf, layout=layout, **options)
    init = [e for e in (sensor, shading) if e is not None]
    est.fit(data, init=init or None)
    return est.apply(data, out)
