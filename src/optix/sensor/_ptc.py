"""Photon-transfer curve (EMVA 1288 Release 4.0)."""

from collections.abc import Callable

import torch
from torch import Tensor

from .._estimator import Tags
from ..estimates import SensorEstimate
from ..io import ArrayLike
from ._base import CalibrationEstimator
from ._scmos import gain_regression

__all__ = ["PhotonTransferCurve"]


def _tolist(x: Tensor) -> object:
    return x.squeeze().tolist()


class PhotonTransferCurve(CalibrationEstimator):
    r"""Photon-transfer curve of a camera (EMVA 1288, Release 4.0).

    The temporal variance of the digital signal is linear in its mean:

    .. math::

        \sigma^2_y = \underbrace{K^2 \sigma^2_d + \sigma^2_q}_{
            \sigma^2_{y,\mathrm{dark}}}
            + K (\mu_y - \mu_{y,\mathrm{dark}}),
        \qquad \sigma^2_q = 1/12 \ \mathrm{DN}^2,

    with :math:`K` the overall system gain (DN per electron) and
    :math:`\sigma_d` the read noise in electrons. For every illumination
    level, :math:`\mu_y` is the spatial mean of the per-pixel temporal
    mean, and :math:`\sigma^2_y` the spatial mean of the per-pixel
    temporal variance over the frames of that level (Welford/Pebay
    moments). Differencing frames of the same pixel removes the
    fixed-pattern noise; with two frames this is the EMVA estimator
    :math:`\frac{1}{2NM}\sum (y_A - y_B)^2`.

    :math:`K` is the least-squares slope, through the origin, of
    :math:`\sigma^2_y - \sigma^2_{y,\mathrm{dark}}` on
    :math:`\mu_y - \mu_{y,\mathrm{dark}}` over the levels from 0 up to
    ``fit_range`` (70 %) of the saturation point, the level of maximal
    temporal variance.

    Spatial non-uniformities (EMVA 1288 section 8), from the frame
    averages :math:`\langle y \rangle` of ``L`` frames:
    :math:`s^2_y = \mathrm{var}_{xy}\langle y\rangle - \sigma^2_y / L`,
    ``DSNU = s_{y,dark} / K`` (electrons) and
    ``PRNU = sqrt(s²_{y,50} - s²_{y,dark}) / (μ_{y,50} - μ_{y,dark})``
    (fraction), the "50" level being the one closest to half of the
    saturation point.

    Parameters
    ----------
    fit_range : float
        Upper end of the gain fit, as a fraction of the saturation point
        (above the dark level). Paper: 0.7.
    quantization : float
        Quantisation noise variance (DN²) removed from the dark variance
        for the read noise. Paper: 1/12 (integer data). Use 0 for data
        that were never quantised.
    per_pixel : bool
        Extension: also fit a per-pixel gain map (per-pixel
        photon-transfer curve, as in :class:`SCMOSCalibration`) over the
        same levels. Paper: False.
    faithful, dtype, device, layout
        See :class:`optix.Estimator`.

    Attributes
    ----------
    curve : dict of str to Tensor
        ``"mean"`` and ``"variance"`` (``(levels, *group)``) of the
        photon-transfer curve, dark subtracted, and ``"used"`` (bool), the
        levels of the gain fit.

    Notes
    -----
    ``fit(levels, dark=darks)``: ``levels`` is a sequence of stacks (at
    least two frames per level and group), ``darks`` a dark stack.

    The fit range, the saturation criterion and the DSNU/PRNU formulas
    have not yet been checked against the text of the standard: they
    are provisional. The non-linearity check is not implemented.

    Estimates: ``"sensor"``: ``offset`` (per-pixel dark mean, DN),
    ``variance`` (read-noise variance ``K² σ_d²``, DN²) and ``gain``
    (``K``, per group, or per pixel with ``per_pixel``). ``meta`` holds
    ``read_noise_e``, ``dsnu_e``, ``prnu``, ``saturation`` (DN above
    dark) and ``quantization``.

    References
    ----------
    EMVA Standard 1288, Release 4.0 Linear, European Machine Vision
    Association (2021).
    """

    tags = Tags(
        targets=("sensor",),
        streaming="per-pixel-stats",
        noise="poisson-gaussian",
        many_frames=False,
        reference="EMVA 1288 Release 4.0",
    )
    paper = {"fit_range": 0.7, "quantization": 1 / 12, "per_pixel": False}
    recommended = dict(paper)

    def __init__(
        self,
        *,
        fit_range: float | None = None,
        quantization: float | None = None,
        per_pixel: bool | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: object = None,
    ) -> None:
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        self.curve: dict[str, Tensor] = {}

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
            raise ValueError("PhotonTransferCurve needs a dark stack: dark=")
        dark = self._stats(self._dark_stack())
        stats = [self._stats(level) for level in levels]
        for st in [dark, *stats]:
            if (st.count < 2).any():
                raise ValueError("Every level needs at least two frames")

        def summary(st: object) -> tuple[Tensor, Tensor, Tensor]:
            mu = st.spatial_mean(st.mean)
            var = st.spatial_mean(st.var)
            s2 = st.mean.var(st.spatial_dims, keepdim=True) - var / st.count
            return mu, var, s2

        mu_d, var_d, s2_d = summary(dark)
        mus, vs, s2s = zip(*(summary(st) for st in stats), strict=True)
        x = torch.stack(mus) - mu_d
        v = torch.stack(vs) - var_d
        s2 = torch.stack(s2s)
        # saturation point: level of maximal temporal variance
        sat = v.argmax(0, keepdim=True)
        x_sat = x.gather(0, sat)
        used = (x >= 0) & (x <= self.options["fit_range"] * x_sat)
        w = used.to(x.dtype)
        gain = (w * x * v).sum(0) / (w * x * x).sum(0)
        # "50 %" level for the PRNU
        half = (x - 0.5 * x_sat).abs().argmin(0, keepdim=True)
        x50 = x.gather(0, half)[0]
        s2_50 = s2.gather(0, half)[0]
        prnu = (s2_50 - s2_d).clamp_min(0).sqrt() / x50
        q = float(self.options["quantization"])
        read_var = (var_d - q).clamp_min(0)
        read_e = read_var.sqrt() / gain
        dsnu = s2_d.clamp_min(0).sqrt() / gain
        gain_out = gain
        if self.options["per_pixel"]:
            # levels used in every group
            keep = used.reshape(used.shape[0], -1).all(1)
            sel = keep.nonzero().flatten().tolist()
            a = torch.stack([stats[i].mean for i in sel]) - dark.mean
            b = torch.stack([stats[i].var for i in sel]) - dark.var
            gain_out, _ = gain_regression(a, b)
        meta = {
            "units": "ADU",
            "read_noise_e": _tolist(read_e),
            "dsnu_e": _tolist(dsnu),
            "prnu": _tolist(prnu),
            "saturation": _tolist(x_sat[0]),
            "quantization": q,
            "levels": len(levels),
        }
        self.estimates = {
            "sensor": SensorEstimate(
                offset=dark.mean,
                variance=read_var,
                gain=gain_out,
                meta=meta,
            )
        }
        self.curve = {"mean": x, "variance": v, "used": used}

    def _extra_state(self) -> dict[str, object]:
        return {"curve": self.curve}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.curve = state.get("curve", {})
