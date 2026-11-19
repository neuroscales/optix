"""Pixel-wise sCMOS calibration (Huang et al., Nat Methods 2013)."""

from collections.abc import Callable

import torch
from torch import Tensor

from .._estimator import Tags
from ..estimates import SensorEstimate
from ..io import ArrayLike, WritableArrayLike
from ..stats import PixelStats
from ._base import MAD_TO_SD, CalibrationEstimator

__all__ = ["SCMOSCalibration", "gain_regression"]

HUBER_C = 1.345


def _median(x: Tensor, dim: int = 0) -> Tensor:
    """Midpoint median (average of the two central values)."""
    s = x.sort(dim).values
    n = x.shape[dim]
    lo = s.select(dim, (n - 1) // 2)
    hi = s.select(dim, n // 2)
    return (lo + hi) / 2


def gain_regression(
    dmean: Tensor,
    dvar: Tensor,
    *,
    robust: str | None = None,
    iterations: int = 20,
) -> tuple[Tensor, Tensor]:
    """Per-pixel slope of the variance on the mean, through the origin.

    For every pixel, solves (Huang et al. 2013, Supplementary Note,
    Eq. 2.3)

        ĝ = argmin_g ∑ₖ ((vᵏ − var) − g·(D̄ᵏ − o))²,

    with D̄ᵏ − o the mean excess and vᵏ − var the variance excess at
    illumination level k = 1, …, K. With A = (vᵏ − var)ₖ and
    B = (D̄ᵏ − o)ₖ, the closed form is (Eq. 2.5)

        ĝ = (B·Bᵀ)⁻¹·B·Aᵀ = ∑ₖ Bₖ·Aₖ / ∑ₖ Bₖ².

    Parameters
    ----------
    dmean, dvar : (K, ...) Tensor
        Mean minus offset and variance minus read variance, per level.
    robust : {None, "huber", "theil-sen"}
        None: unweighted least squares (Eq. 2.3). `"huber"`
        (extension): iteratively reweighted least squares with Huber
        weights (c = 1.345, scale from the MAD of the residuals).
        `"theil-sen"` (extension): median of the per-level slopes Aₖ/Bₖ
        (the Theil-Sen estimator through the origin).
    iterations : int
        IRLS iterations for `"huber"`.

    Returns
    -------
    gain, stderr : (...) Tensor
        Slope and its standard error √(s²/∑ₖ wₖ·Bₖ²), with s² the residual
        variance (NaN with a single level).
    """
    a, b = dmean, dvar
    nlev = a.shape[0]

    def wls(w: Tensor) -> Tensor:
        den = (w * a * a).sum(0)
        return (w * a * b).sum(0) / den

    w = torch.ones_like(a)
    if robust is None:
        g = wls(w)
    elif robust == "huber":
        g = wls(w)
        tiny = torch.finfo(a.dtype).tiny
        for _ in range(iterations):
            r = b - g * a
            scale = MAD_TO_SD * _median((r - _median(r)).abs())
            u = r.abs() / scale.clamp_min(tiny)
            w = torch.where(u > HUBER_C, HUBER_C / u.clamp_min(tiny), 1.0)
            g = wls(w)
    elif robust == "theil-sen":
        g = _median(b / a)
    else:
        raise ValueError(
            f"robust must be None, 'huber' or 'theil-sen': {robust}"
        )
    r = b - g * a
    if nlev > 1:
        s2 = (w * r * r).sum(0) / (nlev - 1)
        se = (s2 / (w * a * a).sum(0)).sqrt()
    else:
        se = torch.full_like(g, torch.nan)
    return g, se


class SCMOSCalibration(CalibrationEstimator):
    """Pixel-wise sCMOS camera calibration (Huang et al., 2013).

    Each pixel follows

        ADU = g·Poisson(λ) + o + 𝒩(0, var),

    so that E = g·λ + o and Var = g²·λ + var = g·(E − o) + var.

    - Offset o: the per-pixel temporal mean of a dark stack of M frames,
      o = (1/M)·∑ₘ sᵐ (Supplementary Note, Eq. 2.1).
    - Read variance var: the per-pixel temporal variance of the same
      stack, var = (1/M)·∑ₘ (sᵐ)² − o² (Eq. 2.2). The paper used 60,000
      dark frames.
    - Gain g: per-pixel least-squares slope, through the origin, of the
      variance excess vᵏ − var on the mean excess D̄ᵏ − o over the
      illumination levels k = 1, …, K, D̄ᵏ and vᵏ being the per-pixel
      temporal mean and variance of the stack at level k (Eq. 2.3–2.5;
      see [`gain_regression`][optix.sensor.utils.gain_regression]). The paper
      used 15 levels of 20,000 frames, ≈ 20 to 200 photons per pixel.

    [`apply`][optix.Estimator.apply] converts data to
    photoelectrons, (ADU − o)/g.

    !!! note "Gain estimate"
        The gain of each pixel minimises the unweighted sum of squares
        ∑ₖ ((vᵏ − var) − g·(D̄ᵏ − o))² over the K illumination levels, a
        regression through the origin with equal weight per level
        (Supplementary Note, Section 2.3, Eq. 2.3), solved in closed form
        as ĝ = ∑ₖ (D̄ᵏ − o)·(vᵏ − var) / ∑ₖ (D̄ᵏ − o)² (Eq. 2.5). All
        temporal variances (dark and illuminated) divide by the number of
        frames M, as in Eq. 2.2; `ddof=1` (extension) divides by M − 1
        instead.

    !!! note
        `fit(levels, dark=darks)`: `levels` is a sequence of stacks, one
        per illumination level (each with at least two samples per group),
        `darks` a stack of dark frames. Every stack is streamed once
        (Welford/Pebay moments). Maps are `(*group, *spatial)`.

    !!! references "References"
        1. Huang F, Hartwich TMP, Rivera-Molina FE, Lin Y, Duim WC, Long JJ,
           Uchil PD, Myers JR, Baird MA, Mothes W, Davidson MW, Toomre D,
           Bewersdorf J. *Video-rate nanoscopy using sCMOS camera-specific
           single-molecule localization algorithms.* Nat Methods
           **10**(7):653–658, 2013.
           [doi:10.1038/nmeth.2488](https://doi.org/10.1038/nmeth.2488)
           Supplementary Note, Section 2 "Characterization of sCMOS
           Camera", Eq. 2.1–2.5.

        ??? quote "BibTeX"
            ```bibtex
            @article{huang2013video,
              author  = {Huang, Fang and Hartwich, Tobias M. P. and
                         Rivera-Molina, Felix E. and Lin, Yu and
                         Duim, Whitney C. and Long, Jane J. and
                         Uchil, Pradeep D. and Myers, Jordan R. and
                         Baird, Michelle A. and Mothes, Walther and
                         Davidson, Michael W. and Toomre, Derek and
                         Bewersdorf, Joerg},
              title   = {Video-rate nanoscopy using {sCMOS}
                         camera-specific single-molecule localization
                         algorithms},
              journal = {Nature Methods},
              volume  = {10}, number = {7}, pages = {653--658},
              year    = {2013}, doi = {10.1038/nmeth.2488}
            }
            ```
    """

    tags = Tags(
        targets=("sensor",),
        streaming="per-pixel-stats",
        modality=("widefield", "smlm", "any"),
        noise="poisson-gaussian",
        many_frames=True,
        reference="doi:10.1038/nmeth.2488",
    )
    paper = {"robust": "none", "confidence": None, "ddof": 0}
    recommended = {"robust": "none", "confidence": None, "ddof": 1}

    def __init__(
        self,
        *,
        robust: str | None = None,
        confidence: float | None = None,
        ddof: int | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: object = None,
    ) -> None:
        """Configure the calibration.

        Parameters
        ----------
        robust : {"none", "huber", "theil-sen"}
            Regression of the gain (see
            [`gain_regression`][optix.sensor.utils.gain_regression]). Default:
            `"none"` (ordinary least squares through the origin).
        confidence : float, optional
            Extension: flag as `defective` the pixels whose relative
            standard error of the gain exceeds `confidence` (also those with
            a non-finite or non-positive gain). Paper: None (no mask).
        ddof : {0, 1}, optional
            Delta degrees of freedom of the temporal variances: they
            divide by M − `ddof` for a stack of M frames. Paper: 0
            (Supplementary Note, Eq. 2.2). `1` (extension, the default
            with `faithful=False`): unbiased variances.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        if self.options["robust"] not in ("none", "huber", "theil-sen"):
            raise ValueError("robust must be 'none', 'huber' or 'theil-sen'")
        if self.options["ddof"] not in (0, 1):
            raise ValueError("ddof must be 0 or 1")

    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        levels = data if isinstance(data, list) else [data]
        if self._dark_input is None:
            raise ValueError("SCMOSCalibration needs a dark stack: dark=")
        ddof = self.options["ddof"]

        def variance(st: PixelStats) -> Tensor:
            # PixelStats.var divides by M − 1; rescale to M − ddof.
            if ddof == 1:
                return st.var
            return st.var * (st.count - 1) / (st.count - ddof)

        dark = self._stats(self._dark_stack())
        dark_var = variance(dark)
        means, variances = [], []
        for level in levels:
            st = self._stats(level)
            means.append(st.mean)
            variances.append(variance(st))
        a = torch.stack(means) - dark.mean
        b = torch.stack(variances) - dark_var
        robust = self.options["robust"]
        gain, se = gain_regression(
            a, b, robust=None if robust == "none" else robust
        )
        defective = None
        if self.options["confidence"] is not None:
            rel = se / gain
            defective = ~(
                torch.isfinite(gain)
                & (gain > 0)
                & (rel <= self.options["confidence"])
            )
        self.estimates = {
            "sensor": SensorEstimate(
                offset=dark.mean,
                variance=dark_var,
                gain=gain,
                defective=defective,
                meta={"units": "ADU", "levels": len(levels)},
            )
        }
        self.gain_stderr = se

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        """Convert ADU to photoelectrons: (y − o)/g."""
        s = self.estimates["sensor"]
        return self._apply_maps(
            data, out, lambda y, o, g: (y - o) / g, s.offset, s.gain
        )

    def _extra_state(self) -> dict[str, object]:
        return {"gain_stderr": getattr(self, "gain_stderr", None)}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.gain_stderr = state.get("gain_stderr")
