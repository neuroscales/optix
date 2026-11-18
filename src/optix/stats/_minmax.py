"""Streaming per-pixel minimum and maximum."""

import math
from collections.abc import Sequence
from typing import Self

import torch
from torch import Tensor

from ._base import Reducer

__all__ = ["MinMax"]


class MinMax(Reducer):
    """Per-pixel minimum and maximum.

    Attributes
    ----------
    min, max : Tensor
        Running extrema (``+inf`` / ``-inf`` where nothing was seen).

    Notes
    -----
    NaNs propagate: mask them out with ``weight`` instead. ``weight`` is a
    boolean mask (nonzero = keep).
    """

    _state = ("min", "max")

    def __init__(self, *, dim: int | Sequence[int] | None = 0) -> None:
        """Create an empty reducer.

        Parameters
        ----------
        dim : int or tuple of int or None
            Reduced dims of each chunk (see
            [`Reducer`][optix.stats.Reducer]).
        """
        super().__init__(dim=dim)

    def update(self, x: Tensor, *, weight: Tensor | None = None) -> Self:
        """Add a chunk of data; ``weight`` is an optional boolean mask."""
        x = x.detach()
        if not x.is_floating_point():
            x = x.to(torch.get_default_dtype())
        dims = self._dims(x)
        if math.prod(x.shape[d] for d in dims) == 0:
            return self
        if weight is None:
            lo, hi = x.amin(dims), x.amax(dims)
        else:
            mask = weight.detach().to(x.device).bool().expand_as(x)
            lo = torch.where(mask, x, torch.inf).amin(dims)
            hi = torch.where(mask, x, -torch.inf).amax(dims)
        if self.min is None:
            self.min, self.max = lo, hi
        else:
            self.min = torch.minimum(self.min, lo.to(self.min.dtype))
            self.max = torch.maximum(self.max, hi.to(self.max.dtype))
        return self

    def merge(self, other: "MinMax") -> Self:
        """Merge another [`MinMax`][optix.stats.MinMax] into this one."""
        state = self._other_state(other)
        if state["min"] is None:
            return self
        if self.min is None:
            self.min, self.max = state["min"].clone(), state["max"].clone()
        else:
            self.min = torch.minimum(self.min, state["min"].to(self.min))
            self.max = torch.maximum(self.max, state["max"].to(self.max))
        return self
