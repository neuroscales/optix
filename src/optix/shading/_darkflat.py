"""Prospective dark/flat calibration (standard practice)."""

from collections.abc import Callable

from torch import Tensor

from .._estimator import Tags
from ..estimates import SensorEstimate, ShadingEstimate
from ..io import ArrayLike, WritableArrayLike
from ..sensor._base import CalibrationEstimator

__all__ = ["DarkFlatCalibration", "correct"]


def correct(y: Tensor, dark: Tensor, flat: Tensor) -> Tensor:
    """``(y - dark) / flat``."""
    return (y - dark) / flat


class DarkFlatCalibration(CalibrationEstimator):
    r"""Prospective flat-field correction from dark and flat references.

    Model and correction:

    .. math::

        x = \frac{I - D}{F - D} \cdot \overline{F - D}

    with :math:`D` the per-pixel dark reference (camera offset plus dark
    current), :math:`F` the per-pixel flat reference (uniform sample) and
    :math:`\overline{F - D}` the spatial mean of the flat, which keeps the
    corrected intensities in ADU.

    Parameters
    ----------
    statistic : {"mean", "median"}
        Per-pixel reduction of the reference stacks. ``"mean"``: streaming
        Welford/Pebay moments (one pass over frame batches). ``"median"``:
        exact per-pixel median over row tiles (robust to cosmic rays and
        dust moving between flat frames). Paper: ``"mean"``.
    normalize : {"mean", "none"}
        ``"mean"``: the flat ``F - D`` is divided by its spatial mean (per
        group), the usual practice. ``"none"``: ``(I - D) / (F - D)`` as
        is.
        Paper: ``"mean"``.
    faithful, dtype, device, layout
        See :class:`optix.Estimator`.

    Notes
    -----
    ``fit(flats, dark=darks)``: both are stacks in the same layout; the
    dark may also be a precomputed map (a Tensor that broadcasts to the
    ``(*group, *spatial)`` maps), a :class:`ShadingEstimate` (its
    ``dark``) or a :class:`SensorEstimate` (its ``offset``). Without a
    dark, ``D = 0``.

    Estimates: ``"shading"`` (``flat``, ``dark``) and, when a dark stack
    is given, ``"sensor"`` (``offset = D`` and the per-pixel temporal
    ``variance`` of the dark stack).
    """

    tags = Tags(
        targets=("shading", "sensor"),
        streaming="per-pixel-stats",
        noise="none",
        many_frames=True,
        reference="standard practice",
    )
    paper = {"statistic": "mean", "normalize": "mean"}
    recommended = {"statistic": "mean", "normalize": "mean"}

    def __init__(
        self,
        *,
        statistic: str | None = None,
        normalize: str | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: object = None,
    ) -> None:
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        if self.options["statistic"] not in ("mean", "median"):
            raise ValueError("statistic must be 'mean' or 'median'")
        if self.options["normalize"] not in ("mean", "none"):
            raise ValueError("normalize must be 'mean' or 'none'")

    def _fit(
        self,
        Y: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        if isinstance(Y, list):
            raise TypeError("DarkFlatCalibration takes a single flat stack")
        statistic = self.options["statistic"]
        flat = self._stats(Y, statistic=statistic)
        backend = self._backend(Y)
        raw = self._dark_input
        dark = self._dark_stack()
        sensor = None
        if dark is None:
            dmap = flat.mean.new_zeros(())
        elif isinstance(dark, SensorEstimate):
            dmap = dark.offset.to(**backend)
        elif isinstance(dark, ShadingEstimate):
            dmap = dark.dark.to(**backend)
        elif isinstance(raw, Tensor) and _is_map(raw, flat.mean):
            dmap = raw.to(**backend)
        else:
            ds = self._stats(dark, statistic=statistic)
            dmap = ds.mean
            sensor = SensorEstimate(
                offset=ds.mean,
                variance=ds.var,
                meta={"units": "ADU", "statistic": statistic},
            )
        f = flat.mean - dmap
        norm = "none"
        if self.options["normalize"] == "mean":
            f = f / flat.spatial_mean(f)
            norm = "mean"
        self.estimates = {
            "shading": ShadingEstimate(
                flat=f,
                dark=dmap.expand_as(f).clone(),
                meta={"flat_norm": norm, "statistic": statistic},
            )
        }
        if sensor is not None:
            self.estimates["sensor"] = sensor

    def _apply(
        self, Y: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        shading = self.estimates["shading"]
        return self._apply_maps(Y, out, correct, shading.dark, shading.flat)


def _is_map(x: Tensor, like: Tensor) -> bool:
    """Whether ``x`` is a map (broadcasts to ``like``) rather than a stack."""
    return x.ndim <= like.ndim and all(
        a in (1, b)
        for a, b in zip(x.shape[::-1], like.shape[::-1], strict=False)
    )
