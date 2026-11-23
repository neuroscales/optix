"""Plumbing for the Poisson-Gaussian noise fit.

Paper-independent parts of the fit of
[Foi et al. (2008)](https://doi.org/10.1109/TIP.2008.2001399)
(the method itself is in `_foi_local`, `_foi_clip`, `_foi_fit` and
[`PoissonGaussian`][optix.sensor.PoissonGaussian]):

- per-segment sufficient statistics that merge exactly across frames
  ([`SegmentStats`][optix.sensor.utils.SegmentStats]), for pooling the
  fit over a stack;
- unit conversions between normalised and data units, and between the
  (a, b) parametrisation and
  [`SensorEstimate`][optix.estimates.SensorEstimate].

Model and units
---------------
The variance of an observation z with expectation y (both in ADU) is
σ²(y) = a·y + b. With the camera model of
[`SensorEstimate`][optix.estimates.SensorEstimate], a = gain (ADU per
electron) and b = variance − gain·offset (ADU²): b is the variance
extrapolated to y = 0 ADU and is negative when the offset is large.
Clipped data are clip(z, low, high).
"""

import torch
from torch import Tensor

from ..estimates import SensorEstimate
from ..stats import Moments

__all__ = [
    "SegmentStats",
    "from_normalized",
    "sensor_estimate",
    "to_normalized",
]


class SegmentStats:
    """Mergeable per-segment statistics of `(value, noise)` pairs.

    For each segment `i` (for example a level set), this accumulates
    the count `nᵢ`, the mean of `value` (for example approximation
    coefficients, giving the local mean) and the mean and variance of
    `noise` (for example detail coefficients, giving the local standard
    deviation).

    Attributes
    ----------
    value : Moments
        Order-1 moments of `value`, with state of shape `(segments,)`.
    noise : Moments
        Order-2 moments of `noise`, with state of shape `(segments,)`.

    Notes
    -----
    Each update computes the segment sums in float64 (temporaries the
    size of the selected entries only) and centres the noise on its
    segment mean before squaring. It then merges them with
    [Pébay's pairwise formulas](https://doi.org/10.2172/1028931)
    ([`Moments`][optix.stats.Moments]). The result does not depend on
    chunking, up to round-off, and is the same for float32 and float64
    inputs, up to their precision.

    Only sufficient statistics of the sample standard deviation are kept.
    Median-based estimators (such as the MAD) are not mergeable.
    """

    def __init__(self, segments: int) -> None:
        """Empty statistics.

        Parameters
        ----------
        segments : int
            Number of segments; labels are in `[0, segments)`.
        """
        if segments < 1:
            raise ValueError(f"segments must be >= 1, got {segments}")
        self.segments = int(segments)
        self.value = Moments(1, state_dtype=torch.float64)
        self.noise = Moments(2, state_dtype=torch.float64)

    @property
    def empty(self) -> bool:
        return self.value.empty

    def update(
        self,
        labels: Tensor,
        value: Tensor,
        noise: Tensor,
        mask: Tensor | None = None,
    ) -> "SegmentStats":
        """Add a chunk of labelled pairs.

        Parameters
        ----------
        labels : Tensor
            Integer segment of each entry. Entries outside
            `[0, segments)` are ignored.
        value, noise : Tensor
            Same shape as `labels`.
        mask : Tensor, optional
            Boolean; False entries are ignored.

        Returns
        -------
        SegmentStats
            self
        """
        keep = (labels >= 0) & (labels < self.segments)
        if mask is not None:
            keep = keep & mask.to(device=labels.device, dtype=torch.bool)
        keep = keep & value.isfinite() & noise.isfinite()
        lab = labels[keep].long()
        v = value[keep].to(torch.float64)
        e = noise[keep].to(torch.float64)
        dev = lab.device
        L = self.segments
        count = torch.zeros(L, dtype=torch.float64, device=dev)
        count.index_add_(0, lab, torch.ones_like(v))
        safe = count.clamp_min(1)
        vmean = torch.zeros_like(count).index_add_(0, lab, v) / safe
        emean = torch.zeros_like(count).index_add_(0, lab, e) / safe
        m2 = torch.zeros_like(count).index_add_(
            0, lab, (e - emean[lab]).square()
        )
        self.value.merge(_moments(1, count, vmean))
        self.noise.merge(_moments(2, count, emean, m2))
        return self

    def merge(self, other: "SegmentStats") -> "SegmentStats":
        """Merge another `SegmentStats` (same number of segments)."""
        if other.segments != self.segments:
            raise ValueError("Cannot merge SegmentStats of different sizes")
        self.value.merge(other.value)
        self.noise.merge(other.noise)
        return self

    def copy(self) -> "SegmentStats":
        new = SegmentStats(self.segments)
        new.value = self.value.copy()
        new.noise = self.noise.copy()
        return new

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SegmentStats":
        """Move the state (in place)."""
        self.value.to(device, dtype)
        self.noise.to(device, dtype)
        return self

    def result(self, min_count: int = 2) -> dict[str, Tensor]:
        """Statistics of the segments with at least `min_count` entries.

        Returns
        -------
        dict
            `segment` (labels, int64), `count`, `mean` (of
            `value`), `noise_mean` and `std` (sample standard
            deviation of `noise`, `ddof=1`), all float64 and of the
            same length.
        """
        if self.empty:
            raise ValueError("No data: call update() first")
        count = self.value.count
        ok = count >= max(min_count, 2)
        idx = torch.nonzero(ok).squeeze(-1)
        return {
            "segment": idx,
            "count": count[idx],
            "mean": self.value.mean[idx],
            "noise_mean": self.noise.mean[idx],
            "std": self.noise.std(ddof=1)[idx],
        }


def _moments(order: int, count: Tensor, mean: Tensor, *m: Tensor) -> Moments:
    state = {"count": count, "mean": mean, "m2": None, "m3": None, "m4": None}
    if m:
        state["m2"] = m[0]
    return Moments.from_state_dict(
        {
            "class": "Moments",
            "args": {"order": order, "dim": 0, "state_dtype": count.dtype},
            "state": state,
        }
    )


def to_normalized(
    a: float, b: float, low: float, scale: float
) -> tuple[float, float]:
    """(a, b) in the units of z′ = (z − low)/scale.

    σ′²(y′) = (a/scale)·y′ + (a·low + b)/scale².
    """
    return a / scale, (a * low + b) / scale**2


def from_normalized(
    a: float, b: float, low: float, scale: float
) -> tuple[float, float]:
    """Inverse of [`to_normalized`][optix.sensor.utils.to_normalized].

    Back to data units.
    """
    an = a * scale
    return an, b * scale**2 - an * low


def sensor_estimate(
    a: Tensor | float,
    b: Tensor | float,
    offset: Tensor | float | None = None,
    *,
    meta: dict | None = None,
) -> SensorEstimate:
    """[`SensorEstimate`][optix.estimates.SensorEstimate] for σ²(y) = a·y + b.

    Parameters
    ----------
    a : Tensor or float
        Slope of the variance function, i.e. the gain (ADU per electron).
    b : Tensor or float
        Variance at y = 0 ADU (ADU²).
    offset : Tensor or float, optional
        Known camera offset (ADU). (a, b) alone do not determine it.

    Returns
    -------
    SensorEstimate
        `gain = a`. With an offset, `variance = b + a·offset` (the
        read-noise variance). Without one, `offset = None` (not
        estimated) and `variance = b`, the variance extrapolated to
        0 ADU, which is negative when the offset is large;
        `meta["offset_estimated"] = False`.
    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b).to(a)
    meta = dict(meta or {})
    meta.setdefault("units", "ADU")
    if offset is None:
        meta["offset_estimated"] = False
        return SensorEstimate(offset=None, variance=b, gain=a, meta=meta)
    meta.setdefault("offset_estimated", True)
    offset = torch.as_tensor(offset).to(a)
    return SensorEstimate(
        offset=offset, variance=b + a * offset, gain=a, meta=meta
    )
