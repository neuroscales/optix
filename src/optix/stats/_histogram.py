"""Streaming per-pixel fixed-range histograms."""

import math
import warnings
from collections.abc import Sequence
from typing import Self

import torch
from torch import Tensor

from .._config import get_config
from ._base import Reducer

__all__ = ["Histogram"]


class Histogram(Reducer):
    """Per-pixel fixed-range histogram with underflow and overflow bins.

    !!! note
        Integer counts make the result exactly independent of chunking and
        order. A zero-width range (`hi == lo`, dead pixels) puts every value
        equal to `lo` into the first regular bin.

        The state holds (bins + 2) · ∏(item) counts: 64 bins on a
        2048 × 2048 image is about 2.2 GB in int64 (1.1 GB in int32). A
        warning is emitted when it exceeds the configured chunk `memory`.

    Attributes
    ----------
    counts : (bins + 2, *item) Tensor
        `counts[0]` counts values below `lo`, `counts[-1]` values
        above `hi` (and NaNs), `counts[1:-1]` the regular bins. A value
        equal to `hi` goes into the last regular bin.
    """

    _state = ("counts", "lo", "hi")

    def __init__(
        self,
        lo: Tensor | float,
        hi: Tensor | float,
        bins: int = 64,
        *,
        dim: int | Sequence[int] | None = 0,
        count_dtype: torch.dtype = torch.int64,
    ) -> None:
        """Create an empty histogram.

        Parameters
        ----------
        lo, hi : Tensor or float
            Range, as scalars or per-pixel maps (typically from a
            [`MinMax`][optix.stats.MinMax] pass). Fixed at construction.
        bins : int
            Number of regular bins between `lo` and `hi`.
        dim : int or tuple of int or None
            Reduced dims of each chunk (see
            [`Reducer`][optix.stats.Reducer]).
        count_dtype : torch.dtype
            Integer dtype of the counts (default int64, which cannot
            overflow in practice; int32 halves the memory but overflows
            past 2³¹ − 1 counts in one bin).
        """
        if bins < 1:
            raise ValueError(f"bins must be >= 1, got {bins}")
        super().__init__(dim=dim)
        self.bins = bins
        self.count_dtype = count_dtype
        self.lo = torch.as_tensor(lo).detach()
        self.hi = torch.as_tensor(hi).detach()
        if not self.lo.is_floating_point():
            self.lo = self.lo.to(torch.get_default_dtype())
        self.hi = self.hi.to(self.lo)
        self._memory = get_config().memory
        self._warned = False
        shape = torch.broadcast_shapes(self.lo.shape, self.hi.shape)
        if shape:
            self._check_size(shape)

    def _args(self) -> dict[str, object]:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "bins": self.bins,
            "dim": self.dim,
            "count_dtype": self.count_dtype,
        }

    def _check_size(self, item: Sequence[int]) -> None:
        nbytes = (self.bins + 2) * math.prod(item) * self.count_dtype.itemsize
        if nbytes > self._memory and not self._warned:
            self._warned = True
            warnings.warn(
                f"Histogram state ({nbytes / 2**20:.0f} MiB) is larger than "
                f"the chunk memory budget; consider fewer bins.",
                RuntimeWarning,
                stacklevel=3,
            )

    def update(self, x: Tensor, *, weight: Tensor | None = None) -> Self:
        """Add a chunk of data.

        Parameters
        ----------
        x : Tensor
            Chunk; the dims in `self.dim` are reduced.
        weight : Tensor, optional
            Boolean mask or integer counts, broadcastable to `x`.

        Returns
        -------
        Histogram
            self
        """
        x = x.detach()
        if not x.is_floating_point():
            x = x.to(self.lo.dtype)
        dims = self._dims(x)
        rest = [d for d in range(x.ndim) if d not in dims]
        item = [x.shape[d] for d in rest]
        if self.counts is None:
            self._check_size(item)
        if weight is not None:
            weight = weight.detach().to(device=x.device).expand_as(x)
            weight = weight.permute(*dims, *rest).reshape(-1, *item)
        x = x.permute(*dims, *rest).reshape(-1, *item)
        if self.counts is None:
            self.counts = torch.zeros(
                (self.bins + 2, *item), dtype=self.count_dtype, device=x.device
            )
            self.lo = self.lo.to(x.device)
            self.hi = self.hi.to(x.device)
        lo = self.lo.to(x.dtype)
        hi = self.hi.to(x.dtype)
        width = hi - lo
        scale = torch.where(
            width > 0, self.bins / torch.where(width > 0, width, 1), 0
        )
        idx = ((x - lo) * scale).floor_().clamp_(0, self.bins - 1).long() + 1
        idx = torch.where(x < lo, 0, idx)
        idx = torch.where((x > hi) | x.isnan(), self.bins + 1, idx)
        if weight is None:
            src = torch.ones((), dtype=self.count_dtype, device=x.device)
            src = src.expand(idx.shape)
        else:
            src = weight.to(self.count_dtype)
        self.counts.scatter_add_(0, idx, src)
        return self

    def merge(self, other: "Histogram") -> Self:
        """Merge another [`Histogram`][optix.stats.Histogram] (same edges)."""
        state = self._other_state(other)
        if other.bins != self.bins:
            raise ValueError("Cannot merge histograms with different bins")
        if state["counts"] is None:
            return self
        lo, hi = self.lo.to(state["lo"]), self.hi.to(state["hi"])
        if not (torch.equal(lo, state["lo"]) and torch.equal(hi, state["hi"])):
            raise ValueError("Cannot merge histograms with different edges")
        if self.counts is None:
            self.counts = state["counts"].clone()
        else:
            self.counts += state["counts"].to(self.counts.dtype)
        return self

    def _edges_cumulative(self) -> Tensor:
        if self.counts is None:
            raise ValueError("No data: call update() first")
        return self.counts.to(torch.float64).cumsum(0)

    def cdf(self) -> Tensor:
        """Cumulative fraction of values up to each bin.

        Returns
        -------
        (bins + 2, *item) Tensor
            `cdf[j]` = fraction of values in bins `0..j` (float64; NaN
            where nothing was counted).
        """
        cum = self._edges_cumulative()
        return cum / cum[-1]

    def quantile(self, q: float | Tensor) -> Tensor:
        """Per-pixel quantile(s), interpolated linearly inside bins.

        Parameters
        ----------
        q : float or (Q,) Tensor
            Quantile level(s) in [0, 1].

        Returns
        -------
        (*item) or (Q, *item) Tensor
            Quantiles (dtype of `lo`). The error is bounded by one bin
            width inside `[lo, hi]`; ranks falling in the underflow or
            overflow bins are clamped to `lo` / `hi`. NaN where nothing
            was counted.
        """
        cum = self._edges_cumulative()  # cum[j] = count below edge j
        edges = cum[: self.bins + 1].movedim(0, -1).contiguous()
        total = cum[-1]
        scalar = not isinstance(q, Tensor) or q.ndim == 0
        q = torch.as_tensor(q, dtype=torch.float64, device=cum.device)
        q = q.reshape(-1)
        rank = total.unsqueeze(-1) * q  # (*item, Q)
        idx = torch.searchsorted(edges, rank, right=True)
        j = (idx - 1).clamp(0, self.bins - 1)
        below = edges.gather(-1, j)
        above = edges.gather(-1, j + 1)
        span = above - below
        frac = torch.where(
            span > 0, (rank - below) / torch.where(span > 0, span, 1), 0
        )
        pos = (j + frac).clamp(0, self.bins)
        lo = self.lo.to(torch.float64).unsqueeze(-1)
        hi = self.hi.to(torch.float64).unsqueeze(-1)
        value = lo + pos * (hi - lo) / self.bins
        value = torch.where(total.unsqueeze(-1) > 0, value, torch.nan)
        value = value.movedim(-1, 0).to(self.lo.dtype)
        return value[0] if scalar else value
