"""Defective, hot and random-telegraph-signal (RTS) pixel detection."""

import itertools
from collections.abc import Callable

import torch
from torch import Tensor

from .._estimator import Tags
from ..estimates import SensorEstimate
from ..io import ArrayLike, WritableArrayLike
from ._base import MAD_TO_SD, CalibrationEstimator

__all__ = ["DefectivePixels", "bimodality", "replace_defective", "robust_z"]


def robust_z(x: Tensor, nspatial: int) -> Tensor:
    """``(x - median) / (1.4826 MAD)`` over the spatial dims (per group).

    Non-finite values are ignored in the median and MAD.
    """
    lead = x.shape[: x.ndim - nspatial]
    flat = x.reshape(*lead, -1)
    med = flat.nanmedian(-1, keepdim=True).values
    mad = (flat - med).abs().nanmedian(-1, keepdim=True).values
    tiny = torch.finfo(x.dtype).tiny
    z = (flat - med) / (MAD_TO_SD * mad).clamp_min(tiny)
    return z.reshape(x.shape)


def bimodality(skewness: Tensor, kurtosis: Tensor) -> Tensor:
    r"""Sarle's bimodality coefficient :math:`(g_1^2 + 1) / \beta_2`.

    1/3 for a Gaussian, 5/9 for a uniform distribution and 1 for any
    two-point distribution (a noiseless random telegraph signal).
    ``kurtosis`` is the non-excess kurtosis :math:`\beta_2`.
    """
    return (skewness.square() + 1) / kurtosis


def replace_defective(y: Tensor, defective: Tensor) -> Tensor:
    """Replace defective pixels by the median of their good neighbours.

    Parameters
    ----------
    y : (n, *spatial) Tensor
        Frames.
    defective : (*spatial) bool Tensor

    Returns
    -------
    Tensor
        Copy of ``y`` where every defective pixel is the (lower) median
        of its ``3^d - 1`` neighbours that are in bounds and not
        defective; pixels without such neighbours are left unchanged.
    """
    idx = defective.nonzero()
    if idx.numel() == 0:
        return y
    shape = torch.tensor(defective.shape, device=idx.device)
    nd = defective.ndim
    vals = []
    for shift in itertools.product((-1, 0, 1), repeat=nd):
        if not any(shift):
            continue
        j = idx + torch.tensor(shift, device=idx.device)
        inside = ((j >= 0) & (j < shape)).all(1)
        j = torch.where(inside[:, None], j, 0)
        jj = tuple(j.T)
        v = y[(slice(None), *jj)]
        ok = inside & ~defective[jj]
        vals.append(torch.where(ok, v, torch.nan))
    med = torch.stack(vals, -1).nanmedian(-1).values
    out = y.clone()
    ii = tuple(idx.T)
    out[(slice(None), *ii)] = torch.where(
        med.isnan(), out[(slice(None), *ii)], med
    )
    return out


class DefectivePixels(CalibrationEstimator):
    r"""Defective, hot and RTS pixel detection from a dark stack.

    Per-pixel temporal moments of a dark stack (one streaming pass) give
    the dark mean :math:`m`, variance :math:`v`, skewness :math:`g_1` and
    kurtosis :math:`\beta_2`. A pixel is flagged when

    * ``"offset"``: :math:`|z(m)| > t` (hot or cold pixel),
    * ``"variance"``: :math:`z(v) > t` (noisy pixel),
    * ``"kurtosis"``: :math:`|z(\beta_2)| > t` (non-Gaussian noise),
    * ``"rts"``: bimodal dark histogram, i.e. Sarle's bimodality
      coefficient :math:`(g_1^2 + 1)/\beta_2 > b` (random telegraph
      signal; 1/3 for Gaussian noise, 1 for a two-level signal),

    with :math:`z(x) = (x - \mathrm{med}\,x) / (1.4826\,\mathrm{MAD}\,x)`
    the robust z-score over the pixels of a group.

    Parameters
    ----------
    threshold : float
        Robust z-score threshold ``t``. Default 5.
    bimodality : float
        Threshold ``b`` of the bimodality coefficient. Default 5/9 (the
        value of a uniform distribution).
    faithful, dtype, device, layout
        See :class:`optix.Estimator`.

    Attributes
    ----------
    masks : dict of str to Tensor
        The individual boolean ``(*group, *spatial)`` masks.

    Notes
    -----
    ``fit(darks)``. Estimates: ``"sensor"`` with ``defective`` (union of
    the masks), ``offset`` and ``variance`` (dark mean and variance).
    :meth:`apply` replaces defective pixels with the median of their
    good neighbours (:func:`replace_defective`).
    """

    tags = Tags(
        targets=("sensor",),
        streaming="per-pixel-stats",
        noise="gaussian",
        many_frames=True,
        reference="standard practice",
    )
    paper = {"threshold": 5.0, "bimodality": 5 / 9}
    recommended = dict(paper)

    def __init__(
        self,
        *,
        threshold: float | None = None,
        bimodality: float | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: object = None,
    ) -> None:
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        self.masks: dict[str, Tensor] = {}

    def _fit(
        self,
        Y: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        if isinstance(Y, list):
            raise TypeError("DefectivePixels takes a single dark stack")
        st = self._stats(Y, order=4)
        t = self.options["threshold"]
        nd = st.nspatial
        bc = bimodality(st.skewness, st.kurtosis)
        self.masks = {
            "offset": robust_z(st.mean, nd).abs() > t,
            "variance": robust_z(st.var, nd) > t,
            "kurtosis": robust_z(st.kurtosis, nd).abs() > t,
            "rts": bc > self.options["bimodality"],
        }
        defective = torch.zeros_like(self.masks["offset"])
        for m in self.masks.values():
            defective |= m
        self.estimates = {
            "sensor": SensorEstimate(
                offset=st.mean,
                variance=st.var,
                defective=defective,
                meta={"units": "ADU"},
            )
        }

    def _apply(
        self, Y: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        defective = self.estimates["sensor"].defective
        return self._apply_maps(
            Y,
            out,
            lambda y, d: replace_defective(y, d.bool()),
            defective,
            overhead=3.0,
        )

    def _extra_state(self) -> dict[str, object]:
        return {"masks": self.masks}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.masks = state.get("masks", {})
