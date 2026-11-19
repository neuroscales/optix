"""Shared helpers of the calibration estimators.

Calibration estimators take several stacks (a dark stack, flat stacks at
several illumination levels). They are passed as `fit(data, dark=...)`,
where `data` is one stack or a sequence of stacks (one per level), all in
the same layout. Every stack is streamed once.
"""

from collections.abc import Callable, Sequence
from typing import Self

import torch
from torch import Tensor

from .._estimator import Estimator
from ..estimates import Estimate
from ..io import ArrayLike, Labeled, Layout, WritableArrayLike
from ..stats import PixelStats, pixel_stats
from ..stats._pixel import MAD_TO_SD

__all__ = ["MAD_TO_SD", "CalibrationEstimator", "as_map", "per_group"]


def as_map(
    value: object, attr: str, backend: dict, *, name: str = "dark"
) -> Tensor:
    """A parameter map from an estimate (its `attr` field) or a tensor."""
    if isinstance(value, Estimate):
        field = getattr(value, attr, None)
        if field is None:
            raise ValueError(f"{name}: {type(value).__name__} lacks {attr}")
        value = field
    return torch.as_tensor(value).to(**backend)


def per_group(fn: Callable[[Tensor, Tensor], Tensor], *maps: Tensor) -> object:
    """`fn(chunk, data)` closure indexing `(*group, *spatial)` maps."""

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

    `fit(data, dark=...)`: `data` is one array-like, or a sequence of
    array-likes (one stack per illumination level). Every stack, and the
    dark stack, uses the same layout.
    """

    _dark_input: object = None

    def fit(
        self,
        data: ArrayLike | Sequence[ArrayLike] | None,
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
        data : array-like, Labeled or sequence of them
            Data stack(s), `(*samples, H, W)` without a layout.
        dark : array-like, Labeled, Tensor map or Estimate, optional
            Dark reference: a stack of dark frames (same layout as `data`),
            or (where documented) a precomputed map or estimate.
        layout, init, callback, verbose
            See [`Estimator.fit`][optix.Estimator.fit].

        Returns
        -------
        Estimator
            self
        """
        if isinstance(data, list | tuple) and layout is None:
            layouts = {y.layout for y in data if isinstance(y, Labeled)}
            if len(layouts) > 1:
                raise ValueError("Stacks are labelled with different layouts")
            if layouts:
                layout = layouts.pop()
        self._dark_input = dark
        try:
            return super().fit(
                data,
                layout=layout,
                init=init,
                callback=callback,
                verbose=verbose,
            )
        finally:
            self._dark_input = None

    @staticmethod
    def _labeled(data: object, layout: Layout | None) -> object:
        if data is None:
            return None
        if isinstance(data, list | tuple):
            return [Estimator._labeled(y, layout) for y in data]
        return Estimator._labeled(data, layout)

    def _dark_stack(self) -> object:
        """The dark input, labelled with the fit layout if it is a stack."""
        dark = self._dark_input
        if dark is None or isinstance(dark, Estimate):
            return dark
        return self._labeled(dark, self.layout)

    def _stats(
        self, data: ArrayLike, *, order: int = 2, statistic: str = "mean"
    ) -> PixelStats:
        backend = self._backend(data)
        return pixel_stats(
            data,
            order=order,
            statistic=statistic,
            dtype=self.dtype,
            device=self.device,
            out_dtype=backend["dtype"],
        )

    def _apply_maps(
        self,
        data: ArrayLike,
        out: WritableArrayLike | None,
        fn: Callable[..., Tensor],
        *maps: Tensor,
        overhead: float = 2.0,
    ) -> WritableArrayLike:
        """`_apply_frames` with per-group parameter maps."""
        return self._apply_frames(
            data, out, per_group(fn, *maps), overhead=overhead
        )
