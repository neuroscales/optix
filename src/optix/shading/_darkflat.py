"""Prospective dark/flat calibration (optix convention: standard practice)."""

from collections.abc import Callable

from torch import Tensor

from .._estimator import Tags
from ..estimates import SensorEstimate, ShadingEstimate
from ..io import ArrayLike, WritableArrayLike
from ..sensor._base import CalibrationEstimator

__all__ = ["DarkFlatCalibration", "correct"]


def correct(y: Tensor, dark: Tensor, flat: Tensor) -> Tensor:
    """(y − dark)/flat."""
    return (y - dark) / flat


class DarkFlatCalibration(CalibrationEstimator):
    """Prospective flat-field correction from dark and flat references.

    !!! note "optix convention"
        Dark/flat correction is standard practice with no single defining
        publication; the reductions and normalisation below are documented
        optix choices (`faithful` refers to these defaults).

    Model and correction:

        x = (I − D)/(F − D) · mean(F − D),

    with D the per-pixel dark reference (camera offset plus dark current),
    F the per-pixel flat reference (uniform sample) and mean(F − D) its
    spatial mean, which keeps the corrected intensities in ADU.

    !!! note
        `fit(flats, dark=darks)`: both are stacks in the same layout; the
        dark may also be a precomputed map (a Tensor that broadcasts to the
        `(*group, *spatial)` maps), a
        [`ShadingEstimate`][optix.estimates.ShadingEstimate] (its `dark`)
        or a [`SensorEstimate`][optix.estimates.SensorEstimate] (its
        `offset`). Without a dark, D = 0.

        Estimates: `"shading"` (`flat`, `dark`) and, when a dark stack is
        given, `"sensor"` (`offset` = D and the per-pixel temporal
        `variance` of the dark stack).
    """

    tags = Tags(
        targets=("shading", "sensor"),
        streaming="per-pixel-stats",
        noise="none",
        many_frames=True,
        reference="optix convention",
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
        """Configure the calibration.

        Parameters
        ----------
        statistic : {"mean", "median"}
            Per-pixel reduction of the reference stacks. `"mean"`:
            streaming Welford/Pébay moments (one pass over frame batches).
            `"median"`: exact per-pixel median over row tiles (robust to
            cosmic rays and dust moving between flat frames). Default
            (optix convention): `"mean"`.
        normalize : {"mean", "none"}
            `"mean"`: the flat F − D is divided by its spatial mean (per
            group), the usual practice. `"none"`: (I − D)/(F − D) as is.
            Default (optix convention): `"mean"`.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        if self.options["statistic"] not in ("mean", "median"):
            raise ValueError("statistic must be 'mean' or 'median'")
        if self.options["normalize"] not in ("mean", "none"):
            raise ValueError("normalize must be 'mean' or 'none'")

    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        if isinstance(data, list):
            raise TypeError("DarkFlatCalibration takes a single flat stack")
        statistic = self.options["statistic"]
        flat = self._stats(data, statistic=statistic)
        backend = self._backend(data)
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
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        shading = self.estimates["shading"]
        return self._apply_maps(data, out, correct, shading.dark, shading.flat)


def _is_map(x: Tensor, like: Tensor) -> bool:
    """Whether `x` is a map (broadcasts to `like`) rather than a stack."""
    return x.ndim <= like.ndim and all(
        a in (1, b)
        for a, b in zip(x.shape[::-1], like.shape[::-1], strict=False)
    )
