"""Model-sized results shared by estimators, simulators and corrections.

Conventions
-----------
- Every field is a Tensor (or None) that broadcasts against the spatial
  shape. `None` means "not estimated": consumers fall back to their own
  default, never silently to zero.
- `ShadingEstimate.dark` is the whole additive image of
  `x = (y - dark - baseline) / flat`: it **includes** the camera offset.
- `ShadingEstimate.baseline` is in raw units (`y = flat * x + dark +
  baseline`) unless `meta["baseline_units"] == "signal"`
  (`y = flat * (x + baseline) + dark`).
  `SensorEstimate.offset` is the camera offset alone. A model with a
  single additive term returns it in both; corrections never subtract
  both.
- Gains are in ADU per electron (EMVA 1288 `K`).
- Flat fields are stored as produced by the algorithm;
  `meta["flat_norm"]` records the normalisation (`"mean"` or
  `"geomean"`) and
  [`ShadingEstimate.normalize`][optix.estimates.ShadingEstimate.normalize]
  converts.
- Data-sized outputs (class probabilities, corrected stacks) are never
  stored in an estimate: they come from `Estimator.apply(data, out=)`.
"""

import copy
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import ClassVar, Self

import torch
from torch import Tensor

from ._version import __version__

__all__ = [
    "Classification",
    "Estimate",
    "SensorEstimate",
    "ShadingEstimate",
]

_REGISTRY: dict[str, type["Estimate"]] = {}


@dataclass(eq=False)
class Estimate:
    """Base for model-sized results. Fields are tensors or None.

    Numbers given for tensor fields are converted to 0-d tensors.

    Attributes
    ----------
    meta : dict
        Provenance and conventions (estimator class, optix version,
        resolved options, `faithful`, units, `flat_norm`...). Must be
        JSON-serialisable.
    """

    _registry: ClassVar[dict[str, type["Estimate"]]] = _REGISTRY
    # fields indexed by sample (frame), not broadcast against the maps
    _per_sample: ClassVar[tuple[str, ...]] = ()

    meta: dict[str, object] = field(default_factory=dict, kw_only=True)

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _REGISTRY[cls.__name__] = cls

    def __post_init__(self) -> None:
        for name in self._field_names():
            value = getattr(self, name)
            if value is not None and not isinstance(value, Tensor):
                setattr(self, name, torch.as_tensor(value))

    @classmethod
    def _field_names(cls) -> tuple[str, ...]:
        return tuple(
            f.name for f in dataclasses.fields(cls) if f.name != "meta"
        )

    def _fields(self) -> dict[str, Tensor | None]:
        return {n: getattr(self, n) for n in self._field_names()}

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Return a copy on `device`; `dtype` casts floating fields only.

        Boolean masks and integer fields keep their dtype.
        """
        changes = {}
        for name, value in self._fields().items():
            if value is None:
                continue
            if dtype is not None and value.is_floating_point():
                value = value.to(dtype)
            if device is not None:
                value = value.to(device)
            changes[name] = value
        return dataclasses.replace(self, meta=dict(self.meta), **changes)

    def clone(self) -> Self:
        """Deep copy (tensors cloned, meta deep-copied)."""
        changes = {
            n: v.clone() for n, v in self._fields().items() if v is not None
        }
        return dataclasses.replace(
            self, meta=copy.deepcopy(self.meta), **changes
        )

    def replace(self, **changes: object) -> Self:
        """Copy with some fields replaced (`dataclasses.replace`)."""
        return dataclasses.replace(self, **changes)

    def allclose(self, other: "Estimate", **tol: float) -> bool:
        """Same type, same estimated fields, and `torch.allclose` values.

        Parameters
        ----------
        other : Estimate
        **tol
            `rtol` / `atol` / `equal_nan` for `torch.allclose`.
            `meta` is not compared.
        """
        if type(other) is not type(self):
            return False
        for name, a in self._fields().items():
            b = getattr(other, name)
            if (a is None) != (b is None):
                return False
            if a is None:
                continue
            b = b.to(a.device)
            if a.shape != b.shape:
                return False
            if a.dtype == torch.bool or b.dtype == torch.bool:
                if not torch.equal(a, b):
                    return False
            elif not torch.allclose(a, b.to(a.dtype), **tol):
                return False
        return True

    def to_dict(self) -> dict[str, object]:
        """Fields and `meta` as a plain dict (write it where you like)."""
        return {**self._fields(), "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        """Inverse of [`to_dict`][optix.estimates.Estimate.to_dict]."""
        return cls(**d)

    def _payload(self) -> dict[str, object]:
        try:
            json.dumps(self.meta)
        except (TypeError, ValueError) as e:
            raise TypeError(f"meta must be JSON-serialisable: {e}") from e
        return {
            "type": type(self).__name__,
            "version": __version__,
            "fields": self._fields(),
            "meta": self.meta,
        }

    @classmethod
    def _from_payload(cls, payload: dict[str, object]) -> "Estimate":
        name = payload["type"]
        target = _REGISTRY.get(name)
        if target is None or not issubclass(target, cls):
            raise TypeError(f"Saved {name} is not a {cls.__name__}")
        return target(**payload["fields"], meta=payload["meta"])

    def save(self, path: str | os.PathLike) -> None:
        """Save with `torch.save` (no pickled classes).

        Raises
        ------
        TypeError
            If `meta` is not JSON-serialisable.
        """
        torch.save(self._payload(), path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> Self:
        """Load an estimate saved with [`save`][optix.estimates.Estimate.save].

        Loading uses `weights_only`.
        """
        payload = torch.load(path, weights_only=True)
        return cls._from_payload(payload)


def _need(estimate: Estimate, *names: str) -> None:
    missing = [n for n in names if getattr(estimate, n) is None]
    if missing:
        raise ValueError(
            f"{type(estimate).__name__} lacks {missing} (not estimated)"
        )


@dataclass(eq=False)
class SensorEstimate(Estimate):
    """Camera model `y = offset + gain * n + e`.

    `n ~ Poisson(electrons)` and `e ~ N(0, variance)`. The Foi
    parametrisation is derived, not stored: `a = gain`,
    `b = variance - gain * offset`.

    Attributes
    ----------
    offset : Tensor, optional
        Camera offset in ADU; scalar or (H, W).
    variance : Tensor, optional
        Read-noise variance in ADU².
    gain : Tensor, optional
        ADU per electron; scalar or (H, W).
    defective : Tensor, optional
        Boolean (H, W) map, True = unusable pixel.
    """

    offset: Tensor | None = None
    variance: Tensor | None = None
    gain: Tensor | None = None
    defective: Tensor | None = None

    def noise_variance(self, mean: Tensor) -> Tensor:
        """Variance of the data at a given mean signal (ADU).

        Parameters
        ----------
        mean : Tensor
            Expected value of `y` (ADU).

        Returns
        -------
        Tensor
            `variance + gain * (mean - offset)`.
        """
        _need(self, "offset", "variance", "gain")
        return self.variance + self.gain * (mean - self.offset)


_BASELINE_UNITS = ("raw", "signal")


@dataclass(eq=False)
class ShadingEstimate(Estimate):
    """Shading model with a per-frame baseline.

    The baseline b_i is a scalar per frame, in one of two units, recorded
    in `meta["baseline_units"]`:

    - `"raw"` (default, also when the key is absent): an offset of the
      measured frame, y = flat · x + dark + b.
    - `"signal"`: an offset of the signal before shading, so that it is
      shaded too, y = flat · (x + b) + dark (as in BaSiC).

    The two agree only for a uniform flat field: a signal baseline b is
    the raw offset flat · b, which is not a scalar.

    Attributes
    ----------
    flat : Tensor, optional
        (H, W) multiplicative flat field.
    dark : Tensor, optional
        (H, W) total additive term in ADU (includes the camera offset).
    baseline : Tensor, optional
        (*batch, frames) per-frame offset, in `baseline_units`.
    """

    _per_sample: ClassVar[tuple[str, ...]] = ("baseline",)

    flat: Tensor | None = None
    dark: Tensor | None = None
    baseline: Tensor | None = None

    @property
    def baseline_units(self) -> str:
        """`meta["baseline_units"]`: `"raw"` (the default) or `"signal"`."""
        return _check_units(self.meta.get("baseline_units", "raw"))

    def shade(
        self,
        x: Tensor,
        baseline: Tensor | None = None,
        *,
        units: str | None = None,
    ) -> Tensor:
        """Apply the model to a signal `x` of shape `(*batch, frames, H, W)`.

        Parameters
        ----------
        x : Tensor
            Unshaded signal.
        baseline : Tensor, optional
            `(*batch, frames)` baseline to use instead of `self.baseline`.
        units : {"raw", "signal"}, optional
            Units of the baseline (default: `baseline_units`).

        Returns
        -------
        Tensor
            flat · x + dark + b (raw) or flat · (x + b) + dark (signal).
            Fields that are None are absent terms (flat = 1, dark = 0,
            b = 0).
        """
        units = self.baseline_units if units is None else _check_units(units)
        b = self.baseline if baseline is None else baseline
        y = x
        if b is not None and units == "signal":
            y = y + b[..., None, None]
        if self.flat is not None:
            y = self.flat * y
        if self.dark is not None:
            y = y + self.dark
        if b is not None and units == "raw":
            y = y + b[..., None, None]
        return y

    def correct(
        self,
        y: Tensor,
        baseline: Tensor | None = None,
        *,
        units: str | None = None,
        norm: Tensor | float = 0.0,
    ) -> Tensor:
        """Invert the model for frames `y` of shape `(*batch, frames, H, W)`.

        Parameters
        ----------
        y : Tensor
            Measured frames.
        baseline : Tensor, optional
            `(*batch, frames)` baseline to use instead of `self.baseline`.
        units : {"raw", "signal"}, optional
            Units of the baseline (default: `baseline_units`).
        norm : Tensor or float
            Background added back, in the units of the baseline
            (broadcasts against `baseline`).

        Returns
        -------
        Tensor
            (y − dark − (b − norm)) / flat (raw) or
            (y − dark) / flat − (b − norm) (signal).
        """
        units = self.baseline_units if units is None else _check_units(units)
        b = self.baseline if baseline is None else baseline
        if b is not None:
            b = (b - torch.as_tensor(norm).to(b))[..., None, None]
        x = y
        if self.dark is not None:
            x = x - self.dark
        if b is not None and units == "raw":
            x = x - b
        if self.flat is not None:
            x = x / self.flat
        if b is not None and units == "signal":
            x = x - b
        return x

    def normalize(self, mode: str = "mean") -> Self:
        """Rescale the flat field so that its mean (or geometric mean) is 1.

        With c the mean (or geometric mean) of the flat, the flat becomes
        f / c and the scale c moves into the signal: x̂ = c · x.  The
        dark is unchanged.  The baseline bᵢ is unchanged when
        `baseline_units == "raw"` (y = f · x + d + bᵢ), and becomes
        c · bᵢ when it is `"signal"` (y = f ⊙ (x + bᵢ) + d), so that
        `shade` and `correct` stay consistent with x̂.  A baseline norm
        derived from the baseline (BaSiC's `baseline_norm="mean"`)
        follows automatically; a numeric `norm` passed to `correct` is
        in the units of the (rescaled) baseline.

        Parameters
        ----------
        mode : {"mean", "geomean"}

        Returns
        -------
        ShadingEstimate
            Copy with the rescaled flat (and signal-units baseline) and
            `meta["flat_norm"] = mode`.
        """
        _need(self, "flat")
        if mode == "mean":
            scale = self.flat.mean()
        elif mode == "geomean":
            scale = self.flat.log().mean().exp()
        else:
            raise ValueError(f"mode must be 'mean' or 'geomean', got {mode}")
        baseline = self.baseline
        if baseline is not None and self.baseline_units == "signal":
            baseline = baseline * scale.to(baseline)
        return dataclasses.replace(
            self,
            flat=self.flat / scale,
            baseline=baseline,
            meta={**self.meta, "flat_norm": mode},
        )


def _check_units(units: object) -> str:
    if units not in _BASELINE_UNITS:
        raise ValueError(
            f"baseline units must be one of {_BASELINE_UNITS}: {units!r}"
        )
    return str(units)


@dataclass(eq=False)
class Classification(Estimate):
    """Class model summary (provisional).

    Per-voxel probabilities are data-sized and come from
    `apply(data, out=)`; only the class model is stored here.

    Attributes
    ----------
    weights : Tensor, optional
        (K,) mixing weights.
    parameters : Tensor, optional
        (K, P) class parameters (meaning recorded in `meta`).
    concentration : Tensor, optional
        (K,) Dirichlet concentration of the mixing weights, for models
        that treat them as uncertain (`weights` is then its mean).
    """

    weights: Tensor | None = None
    parameters: Tensor | None = None
    concentration: Tensor | None = None
