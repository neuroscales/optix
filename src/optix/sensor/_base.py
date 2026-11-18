"""Shared helpers of the calibration estimators.

Calibration estimators take several stacks (a dark stack, flat stacks at
several illumination levels). They are passed as ``fit(Y, dark=...)``,
where ``Y`` is one stack or a sequence of stacks (one per level), all in
the same layout. Every stack is streamed once.
"""

from collections.abc import Callable, Sequence
from typing import Self

import torch
from torch import Tensor

from .._config import resolve_backend
from .._estimator import Estimator
from ..estimates import Estimate
from ..io import ArrayLike, Labeled, Layout, WritableArrayLike, frame_batches
from ..stats import Moments, median_mad

__all__ = ["CalibrationEstimator", "PixelStats", "pixel_stats"]

MAD_TO_SD = 1.482602218505602  # 1 / Phi^{-1}(3/4)


class PixelStats:
    """Per-pixel temporal statistics of one stack.

    Attributes
    ----------
    count : (*group, 1, ..., 1) Tensor
        Number of samples per group.
    mean, var : (*group, *spatial) Tensor
        Per-pixel mean and unbiased variance (or median and squared
        scaled MAD for ``statistic="median"``).
    skewness, kurtosis : (*group, *spatial) Tensor or None
        Biased sample skewness and (non-excess) kurtosis (``order=4``).
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


def _layout(Y: ArrayLike) -> Layout:
    if isinstance(Y, Labeled):
        return Y.layout
    return Layout.default(len(Y.shape), 2)


def pixel_stats(
    Y: ArrayLike,
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
    Y : array-like or Labeled
        ``(*samples, H, W)`` without a layout.
    order : int
        Highest central moment (2 or 4), for ``statistic="mean"``.
    statistic : {"mean", "median"}
        ``"mean"``: Welford/Pebay moments over frame batches.
        ``"median"``: exact per-pixel median and MAD over row tiles; the
        variance is then ``(1.4826 MAD)^2``.
    dtype, device : optional
        Working backend of the chunks.
    out_dtype : torch.dtype, optional
        Dtype of the returned maps (default: the working dtype).

    Returns
    -------
    PixelStats
    """
    layout = _layout(Y)
    group_shape, sample_shape, spatial_shape = layout.shape_of(Y.shape)
    nspatial = len(spatial_shape)
    nsamp = 1
    for s in sample_shape:
        nsamp *= s
    if statistic == "median":
        med, mad = median_mad(Y, dtype=dtype, device=device)
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
    chunks = frame_batches(Y, dtype=dtype, device=device)
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


def as_map(
    value: object, attr: str, backend: dict, *, name: str = "dark"
) -> Tensor:
    """A parameter map from an estimate (its ``attr`` field) or a tensor."""
    if isinstance(value, Estimate):
        field = getattr(value, attr, None)
        if field is None:
            raise ValueError(f"{name}: {type(value).__name__} lacks {attr}")
        value = field
    return torch.as_tensor(value).to(**backend)


def per_group(fn: Callable[[Tensor, Tensor], Tensor], *maps: Tensor) -> object:
    """``fn(chunk, data)`` closure indexing ``(*group, *spatial)`` maps."""

    def apply(chunk: object, data: Tensor) -> Tensor:
        picked = []
        for m in maps:
            m = m.to(data)
            nlead = m.ndim - (data.ndim - 1)
            if nlead > 0:
                m = m[chunk.group[:nlead]] if chunk.group else m
            picked.append(m)
        return fn(data, *picked)

    return apply


class CalibrationEstimator(Estimator):
    """Estimator fitted from one or several stacks plus a dark stack.

    ``fit(Y, dark=...)``: ``Y`` is one array-like, or a sequence of
    array-likes (one stack per illumination level). Every stack, and the
    dark stack, uses the same layout.
    """

    _dark_input: object = None

    def fit(
        self,
        Y: ArrayLike | Sequence[ArrayLike] | None,
        *,
        dark: object = None,
        layout: Layout | None = None,
        init: object = None,
        callback: Callable | None = None,
        verbose: bool = False,
    ) -> Self:
        """Fit the model.

        Parameters
        ----------
        Y : array-like, Labeled or sequence of them
            Data stack(s), ``(*samples, H, W)`` without a layout.
        dark : array-like, Labeled, Tensor map or Estimate, optional
            Dark reference: a stack of dark frames (same layout as ``Y``),
            or (where documented) a precomputed map or estimate.
        layout, init, callback, verbose
            See :meth:`optix.Estimator.fit`.

        Returns
        -------
        Estimator
            self
        """
        if isinstance(Y, list | tuple) and layout is None:
            layouts = {y.layout for y in Y if isinstance(y, Labeled)}
            if len(layouts) > 1:
                raise ValueError("Stacks are labelled with different layouts")
            if layouts:
                layout = layouts.pop()
        self._dark_input = dark
        try:
            return super().fit(
                Y, layout=layout, init=init, callback=callback, verbose=verbose
            )
        finally:
            self._dark_input = None

    @staticmethod
    def _labeled(Y: object, layout: Layout | None) -> object:
        if Y is None:
            return None
        if isinstance(Y, list | tuple):
            return [Estimator._labeled(y, layout) for y in Y]
        return Estimator._labeled(Y, layout)

    def _dark_stack(self) -> object:
        """The dark input, labelled with the fit layout if it is a stack."""
        dark = self._dark_input
        if dark is None or isinstance(dark, Estimate):
            return dark
        return self._labeled(dark, self.layout)

    def _stats(
        self, Y: ArrayLike, *, order: int = 2, statistic: str = "mean"
    ) -> PixelStats:
        backend = self._backend(Y)
        return pixel_stats(
            Y,
            order=order,
            statistic=statistic,
            dtype=self.dtype,
            device=self.device,
            out_dtype=backend["dtype"],
        )

    def _apply_maps(
        self,
        Y: ArrayLike,
        out: WritableArrayLike | None,
        fn: Callable[..., Tensor],
        *maps: Tensor,
        overhead: float = 2.0,
    ) -> WritableArrayLike:
        """``_apply_frames`` with per-group parameter maps."""
        return self._apply_frames(
            Y, out, per_group(fn, *maps), overhead=overhead
        )
