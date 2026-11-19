"""Per-pixel temporal statistics of a stack, in one streaming pass."""

import torch
from torch import Tensor

from .._config import resolve_backend
from ..io import ArrayLike, Labeled, Layout, frame_batches
from ._moments import Moments
from ._order import median_mad

__all__ = ["PixelStats", "pixel_stats"]

MAD_TO_SD = 1.482602218505602  # 1 / Phi^{-1}(3/4)


class PixelStats:
    """Per-pixel temporal statistics of one stack.

    Attributes
    ----------
    count : (*group, 1, ..., 1) Tensor
        Number of samples per group.
    mean, var : (*group, *spatial) Tensor
        Per-pixel mean and unbiased variance (or median and squared
        scaled MAD for `statistic="median"`).
    skewness, kurtosis : (*group, *spatial) Tensor or None
        Biased sample skewness and (non-excess) kurtosis (`order=4`).
    nspatial : int
        Number of spatial dims.
    """

    def __init__(
        self,
        count: Tensor,
        mean: Tensor,
        var: Tensor,
        nspatial: int,
        skewness: Tensor | None = None,
        kurtosis: Tensor | None = None,
    ) -> None:
        self.count = count
        self.mean = mean
        self.var = var
        self.nspatial = nspatial
        self.skewness = skewness
        self.kurtosis = kurtosis

    @property
    def spatial_dims(self) -> tuple[int, ...]:
        return tuple(range(-self.nspatial, 0))

    def spatial_mean(self, x: Tensor) -> Tensor:
        """Mean over the spatial dims, kept as size-1 dims."""
        return x.mean(self.spatial_dims, keepdim=True)


def _layout(data: ArrayLike) -> Layout:
    if isinstance(data, Labeled):
        return data.layout
    return Layout.default(len(data.shape), 2)


def pixel_stats(
    data: ArrayLike,
    *,
    order: int = 2,
    statistic: str = "mean",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    out_dtype: torch.dtype | None = None,
) -> PixelStats:
    """Per-pixel statistics of a stack, in one streaming pass.

    Parameters
    ----------
    data : array-like or Labeled
        `(*samples, H, W)` without a layout.
    order : int
        Highest central moment (2 or 4), for `statistic="mean"`.
    statistic : {"mean", "median"}
        `"mean"`: Welford/Pebay moments over frame batches.
        `"median"`: exact per-pixel median and MAD over row tiles; the
        variance is then (1.4826·MAD)².
    dtype, device : optional
        Working backend of the chunks.
    out_dtype : torch.dtype, optional
        Dtype of the returned maps (default: the working dtype).

    Returns
    -------
    PixelStats
    """
    layout = _layout(data)
    group_shape, sample_shape, spatial_shape = layout.shape_of(data.shape)
    nspatial = len(spatial_shape)
    nsamp = 1
    for s in sample_shape:
        nsamp *= s
    if statistic == "median":
        med, mad = median_mad(data, dtype=dtype, device=device)
        out_dtype = out_dtype or med.dtype
        count = torch.full(
            group_shape + (1,) * nspatial,
            float(nsamp),
            dtype=out_dtype,
            device=med.device,
        )
        var = (MAD_TO_SD * mad.to(out_dtype)).square()
        return PixelStats(count, med.to(out_dtype), var, nspatial)
    if statistic != "mean":
        raise ValueError(f"statistic must be 'mean' or 'median': {statistic}")
    chunks = frame_batches(data, dtype=dtype, device=device)
    reducers: dict[tuple[int, ...], Moments] = {}
    for chunk in chunks:
        key = chunk.group
        if key not in reducers:
            reducers[key] = Moments(order)
        reducers[key].update(chunk.data)
    if not reducers:
        raise ValueError("No data to reduce")
    first = next(iter(reducers.values()))
    if out_dtype is None:
        out_dtype = resolve_backend(dtype, device)["dtype"]
    dev = first.mean.device
    shape = group_shape + spatial_shape
    count = torch.zeros(group_shape, dtype=out_dtype, device=dev)
    mean = torch.full(shape, torch.nan, dtype=out_dtype, device=dev)
    var = torch.full_like(mean, torch.nan)
    skew = kurt = None
    if order >= 4:
        skew = torch.full_like(mean, torch.nan)
        kurt = torch.full_like(mean, torch.nan)
    for key, m in reducers.items():
        count[key] = m.count.to(out_dtype)
        mean[key] = m.mean.to(out_dtype)
        var[key] = m.var().to(out_dtype)
        if order >= 4:
            skew[key] = m.skewness().to(out_dtype)
            kurt[key] = m.kurtosis(excess=False).to(out_dtype)
    count = count.reshape(group_shape + (1,) * nspatial)
    return PixelStats(count, mean, var, nspatial, skew, kurt)
