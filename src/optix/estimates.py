"""Model-sized results shared by estimators, simulators and corrections.

Conventions
-----------
- Every field is a Tensor (or None) that broadcasts against the spatial
  shape. `None` means "not estimated": consumers fall back to their own
  default, never silently to zero.
- `ShadingEstimate.dark` is the whole additive image of
  `x = (y - dark - baseline) / flat`: it **includes** the camera offset.
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


@dataclass(eq=False)
class ShadingEstimate(Estimate):
    """Shading model `y = flat * x + dark + baseline`.

    Attributes
    ----------
    flat : Tensor, optional
        (H, W) multiplicative flat field.
    dark : Tensor, optional
        (H, W) total additive term in ADU (includes the camera offset).
    baseline : Tensor, optional
        (*batch, frames) per-frame additive offset.
    """

    flat: Tensor | None = None
    dark: Tensor | None = None
    baseline: Tensor | None = None

    def normalize(self, mode: str = "mean") -> Self:
        """Rescale the flat field so that its mean (or geometric mean) is 1.

        Parameters
        ----------
        mode : {"mean", "geomean"}

        Returns
        -------
        ShadingEstimate
            Copy with the rescaled flat and `meta["flat_norm"] = mode`.
            `dark` and `baseline` are unchanged (the scale moves into
            the signal `x`).
        """
        _need(self, "flat")
        if mode == "mean":
            scale = self.flat.mean()
        elif mode == "geomean":
            scale = self.flat.log().mean().exp()
        else:
            raise ValueError(f"mode must be 'mean' or 'geomean', got {mode}")
        return dataclasses.replace(
            self, flat=self.flat / scale, meta={**self.meta, "flat_norm": mode}
        )


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
    """

    weights: Tensor | None = None
    parameters: Tensor | None = None
