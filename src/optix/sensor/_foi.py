"""Plumbing for the Poisson-Gaussian fit of Foi et al. (2008).

Reference: A. Foi, M. Trimeche, V. Katkovnik, K. Egiazarian, "Practical
Poissonian-Gaussian noise modeling and fitting for single-image raw-data",
IEEE TIP 17(10):1737-1754, 2008, doi:10.1109/TIP.2008.2001399.

Status
------
Only the parts that do not depend on details of the paper are here:

- per-segment sufficient statistics that merge exactly across frames
  (:class:`SegmentStats`), for pooling the fit over a stack;
- the moments of a Gaussian censored at the clipping bounds
  (:func:`clipped_normal_moments`), derived from first principles;
- unit conversions between normalised and data units, and between the
  ``(a, b)`` parametrisation and :class:`~optix.estimates.SensorEstimate`;
- loading of one frame per group under an arbitrary
  :class:`~optix.io.Layout`.

The wavelet decomposition, the segmentation, the local estimators and
the likelihood are paper-specific. They will be implemented from the
paper itself once it is available.

Model and units
---------------
The variance of an observation ``z`` with expectation ``y`` (both in
ADU) is ``sigma^2(y) = a * y + b``. With the camera model of
:class:`~optix.estimates.SensorEstimate`, ``a = gain`` (ADU per
electron) and ``b = variance - gain * offset`` (ADU^2): ``b`` is the
variance extrapolated to ``y = 0`` ADU and is negative when the offset is
large. Clipped data are ``clip(z, low, high)``.
"""

import math
from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor

from ..estimates import SensorEstimate
from ..io import ArrayLike, Labeled, Layout, as_tensor
from ..stats import Moments

__all__ = [
    "SegmentStats",
    "clipped_normal_moments",
    "from_normalized",
    "group_frames",
    "integer_range",
    "sensor_estimate",
    "to_normalized",
]


class SegmentStats:
    """Mergeable per-segment statistics of ``(value, noise)`` pairs.

    For each segment ``i`` (for example a level set), this accumulates
    the count ``n_i``, the mean of ``value`` (for example approximation
    coefficients, giving the local mean) and the mean and variance of
    ``noise`` (for example detail coefficients, giving the local standard
    deviation).

    Parameters
    ----------
    segments : int
        Number of segments; labels are in ``[0, segments)``.

    Attributes
    ----------
    value : Moments
        Order-1 moments of ``value``, with state of shape ``(segments,)``.
    noise : Moments
        Order-2 moments of ``noise``, with state of shape ``(segments,)``.

    Notes
    -----
    Each update computes the segment sums in float64 (temporaries the
    size of the selected entries only) and centres the noise on its
    segment mean before squaring. It then merges them with Pebay's
    pairwise formulas (:class:`optix.stats.Moments`). The result does not
    depend on chunking, up to round-off, and is the same for float32 and
    float64 inputs, up to their precision.

    Only sufficient statistics of the sample standard deviation are kept.
    Median-based estimators (such as the MAD) are not mergeable.
    """

    def __init__(self, segments: int) -> None:
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
            ``[0, segments)`` are ignored.
        value, noise : Tensor
            Same shape as ``labels``.
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
        """Merge another :class:`SegmentStats` (same number of segments)."""
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
        """Statistics of the segments with at least ``min_count`` entries.

        Returns
        -------
        dict
            ``segment`` (labels, int64), ``count``, ``mean`` (of
            ``value``), ``noise_mean`` and ``std`` (sample standard
            deviation of ``noise``, ``ddof=1``), all float64 and of the
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


def clipped_normal_moments(
    mean: Tensor,
    std: Tensor,
    low: float | None = 0.0,
    high: float | None = 1.0,
) -> tuple[Tensor, Tensor]:
    """Mean and standard deviation of ``clip(z, low, high)``.

    The variable is ``z ~ N(mean, std^2)``. With ``t ~ N(0, 1)``,
    ``alpha = (low - mean) / std`` and ``beta = (high - mean) / std``,
    ``clip(z) = mean + std * clip(t, alpha, beta)``, and::

        m  = E[clip(t)]   = alpha Phi(alpha) + beta Phi(-beta)
                            + phi(alpha) - phi(beta)
        s2 = E[clip(t)^2] = alpha^2 Phi(alpha) + beta^2 Phi(-beta)
                            + Phi(beta) - Phi(alpha)
                            + alpha phi(alpha) - beta phi(beta)
        E[clip(z)]   = mean + std * m
        Var[clip(z)] = std^2 * (s2 - m^2)

    Parameters
    ----------
    mean, std : Tensor
        Parameters of the unclipped Gaussian (``std > 0``).
    low, high : float or None
        Clipping bounds (None: no clipping on that side).

    Returns
    -------
    mean, std : Tensor
        Moments of the clipped variable.

    Notes
    -----
    Writing the moments about ``mean`` in units of ``std`` avoids the
    cancellation of ``E[z^2] - E[z]^2`` when clipping is negligible: then
    ``m -> 0`` and ``s2 -> 1``. ``Phi(-beta)`` is used instead of
    ``1 - Phi(beta)``. Compute in float64 when nearly every sample is
    clipped, because ``s2 - m^2`` then cancels.
    """
    mean, std = torch.broadcast_tensors(
        torch.as_tensor(mean), torch.as_tensor(std).to(mean)
    )
    zero = torch.zeros_like(mean)
    ndtr = torch.special.ndtr

    def pdf(x: Tensor) -> Tensor:
        return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    m = zero.clone()
    s2 = torch.ones_like(mean)
    if low is not None:
        a = (low - mean) / std
        pa, fa = ndtr(a), pdf(a)
        m = m + a * pa + fa
        s2 = s2 + a * a * pa - pa + a * fa
    if high is not None:
        b = (high - mean) / std
        qb, fb = ndtr(-b), pdf(b)
        m = m + b * qb - fb
        s2 = s2 + b * b * qb - qb - b * fb
    var = (s2 - m * m).clamp_min(0)
    return mean + std * m, std * var.sqrt()


def to_normalized(
    a: float, b: float, low: float, scale: float
) -> tuple[float, float]:
    """``(a, b)`` in the units of ``(z - low) / scale``.

    ``sigma'^2(y') = (a / scale) y' + (a low + b) / scale^2``.
    """
    return a / scale, (a * low + b) / scale**2


def from_normalized(
    a: float, b: float, low: float, scale: float
) -> tuple[float, float]:
    """Inverse of :func:`to_normalized` (back to data units)."""
    an = a * scale
    return an, b * scale**2 - an * low


def sensor_estimate(
    a: Tensor | float,
    b: Tensor | float,
    offset: Tensor | float | None = None,
    *,
    meta: dict | None = None,
) -> SensorEstimate:
    """:class:`SensorEstimate` for ``sigma^2(y) = a y + b`` (ADU).

    Parameters
    ----------
    a : Tensor or float
        Slope of the variance function, i.e. the gain (ADU per electron).
    b : Tensor or float
        Variance at ``y = 0`` ADU (ADU^2).
    offset : Tensor or float, optional
        Known camera offset (ADU). ``(a, b)`` alone do not determine it.

    Returns
    -------
    SensorEstimate
        ``gain = a`` and ``variance = b + a * offset``. Without an
        offset, ``offset = 0``, ``variance = b`` and
        ``meta["offset_estimated"] = False``. ``noise_variance(mean)`` is
        then still exactly ``a * mean + b``, but ``variance`` is not the
        read-noise variance.
    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b).to(a)
    meta = dict(meta or {})
    meta.setdefault("units", "ADU")
    if offset is None:
        meta["offset_estimated"] = False
        offset = torch.zeros_like(a)
    else:
        meta.setdefault("offset_estimated", True)
        offset = torch.as_tensor(offset).to(a)
    return SensorEstimate(
        offset=offset, variance=b + a * offset, gain=a, meta=meta
    )


def integer_range(dtype: object) -> tuple[float, float]:
    """``(min, max)`` of an integer dtype (torch or numpy-like)."""
    if isinstance(dtype, torch.dtype):
        info = torch.iinfo(dtype)
    else:
        info = np.iinfo(np.dtype(dtype))
    return float(info.min), float(info.max)


def group_frames(
    Y: ArrayLike,
    layout: Layout | None = None,
    frame: int = 0,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Iterator[tuple[tuple[int, ...], Tensor]]:
    """Load one frame of each group.

    Parameters
    ----------
    Y : array-like or Labeled
    layout : Layout, optional
        Default: that of a labelled ``Y``, else ``(*samples, H, W)``.
    frame : int
        Index of the frame in the flattened sample axes (storage order).

    Yields
    ------
    group : tuple of int
        Position along the group axes (layout order).
    image : Tensor
        The frame, with its spatial axes in layout order. Only this frame
        is read.
    """
    raw = Y.array if isinstance(Y, Labeled) else Y
    if layout is None:
        layout = (
            Y.layout
            if isinstance(Y, Labeled)
            else Layout.default(len(Y.shape), 2)
        )
    shape = tuple(int(s) for s in Y.shape)
    gpos, spos, xpos = layout.positions(len(shape))
    sshape = tuple(shape[p] for p in spos)
    nsamples = math.prod(sshape)
    if not -nsamples <= frame < nsamples:
        raise IndexError(f"frame {frame} out of range ({nsamples} frames)")
    sidx = np.unravel_index(frame % max(nsamples, 1), sshape)
    remaining = sorted(xpos)
    perm = [remaining.index(p) for p in xpos]
    for g in np.ndindex(*(shape[p] for p in gpos)):
        index: list[object] = [slice(None)] * len(shape)
        for p, i in zip(gpos, g, strict=True):
            index[p] = int(i)
        for p, i in zip(spos, sidx, strict=True):
            index[p] = int(i)
        image = as_tensor(raw[tuple(index)], dtype=dtype, device=device)
        yield tuple(int(i) for i in g), image.permute(perm)
