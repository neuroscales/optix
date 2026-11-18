"""Pixel-wise sCMOS calibration (Huang et al., Nat Methods 2013)."""

from collections.abc import Callable

import torch
from torch import Tensor

from .._estimator import Tags
from ..estimates import SensorEstimate
from ..io import ArrayLike, WritableArrayLike
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
    r"""Per-pixel slope of the variance on the mean, through the origin.

    Solves :math:`B = g A` in the least-squares sense for every pixel,
    with :math:`A_i = \bar D_i - o` and :math:`B_i = v_i - \mathrm{var}`
    over illumination levels :math:`i`:

    .. math::

        \hat g = (A^\top A)^{-1} A^\top B.

    Parameters
    ----------
    dmean, dvar : (L, ...) Tensor
        Mean minus offset and variance minus read variance, per level.
    robust : {None, "huber", "theil-sen"}
        None: ordinary least squares (Huang et al.). ``"huber"``:
        iteratively reweighted least squares with Huber weights
        (``c = 1.345``, scale from the MAD of the residuals).
        ``"theil-sen"``: median of the per-level slopes ``B_i / A_i``
        (the Theil-Sen estimator through the origin).
    iterations : int
        IRLS iterations for ``"huber"``.

    Returns
    -------
    gain, stderr : (...) Tensor
        Slope and its standard error ``sqrt(s^2 / sum w A^2)``, with ``s^2``
        the residual variance (NaN with a single level).
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
    r"""Pixel-wise sCMOS camera calibration (Huang et al., 2013).

    Each pixel follows

    .. math::

        \mathrm{ADU} = g \cdot \mathrm{Poisson}(\lambda) + o
                       + \mathcal N(0, \mathrm{var}),

    so that ``E = g λ + o`` and ``Var = g² λ + var = g (E - o) + var``.
    The offset ``o`` and the read variance ``var`` are the per-pixel
    temporal mean and variance of a dark stack. The gain is the
    per-pixel least-squares slope, through the origin, of
    ``v_i - var`` on ``D_i - o`` over illumination levels ``i``, where
    ``D_i`` and ``v_i`` are the per-pixel temporal mean and variance of
    the stack at level ``i``.

    :meth:`apply` converts data to photoelectrons, ``(ADU - o) / g``.

    Parameters
    ----------
    robust : {"none", "huber", "theil-sen"}
        Regression of the gain (extension). Paper: ``"none"`` (ordinary
        least squares).
    confidence : float, optional
        Extension: flag as ``defective`` the pixels whose relative
        standard error of the gain exceeds ``confidence`` (also those
        with a non-finite or non-positive gain). Paper: None (no mask).
    faithful, dtype, device, layout
        See :class:`optix.Estimator`.

    Notes
    -----
    ``fit(levels, dark=darks)``: ``levels`` is a sequence of stacks, one
    per illumination level (each with at least two samples per group),
    ``darks`` a stack of dark frames. Every stack is streamed once
    (Welford/Pebay moments). Maps are ``(*group, *spatial)``.

    References
    ----------
    Huang F, Hartwich TMP, Rivera-Molina FE, et al. Video-rate
    nanoscopy using sCMOS camera-specific single-molecule localization
    algorithms. Nat Methods 10, 653-658 (2013). doi:10.1038/nmeth.2488
    """

    tags = Tags(
        targets=("sensor",),
        streaming="per-pixel-stats",
        modality=("widefield", "smlm", "any"),
        noise="poisson-gaussian",
        many_frames=True,
        reference="doi:10.1038/nmeth.2488",
    )
    paper = {"robust": "none", "confidence": None}
    recommended = {"robust": "none", "confidence": None}

    def __init__(
        self,
        *,
        robust: str | None = None,
        confidence: float | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: object = None,
    ) -> None:
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        if self.options["robust"] not in ("none", "huber", "theil-sen"):
            raise ValueError("robust must be 'none', 'huber' or 'theil-sen'")

    def _fit(
        self,
        Y: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        levels = Y if isinstance(Y, list) else [Y]
        if self._dark_input is None:
            raise ValueError("SCMOSCalibration needs a dark stack: dark=")
        dark = self._stats(self._dark_stack())
        means, variances = [], []
        for level in levels:
            st = self._stats(level)
            means.append(st.mean)
            variances.append(st.var)
        a = torch.stack(means) - dark.mean
        b = torch.stack(variances) - dark.var
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
                variance=dark.var,
                gain=gain,
                defective=defective,
                meta={"units": "ADU", "levels": len(levels)},
            )
        }
        self.gain_stderr = se

    def _apply(
        self, Y: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        """Convert ADU to photoelectrons: ``(y - o) / g``."""
        s = self.estimates["sensor"]
        return self._apply_maps(
            Y, out, lambda y, o, g: (y - o) / g, s.offset, s.gain
        )

    def _extra_state(self) -> dict[str, object]:
        return {"gain_stderr": getattr(self, "gain_stderr", None)}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.gain_stderr = state.get("gain_stderr")
