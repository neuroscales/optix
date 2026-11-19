"""Streaming central moments (Pebay's pairwise formulas)."""

import math
from collections.abc import Sequence
from typing import Self

import torch
from torch import Tensor

from ._base import Reducer

__all__ = ["Moments"]


def _combine(
    a: tuple[Tensor, ...], b: tuple[Tensor, ...], order: int
) -> tuple[Tensor, ...]:
    """Pairwise merge of `(n, mean, M2, ..., Mp)` (Pebay, 2008)."""
    na, ma, *mma = a
    nb, mb, *mmb = b
    n = na + nb
    safe = torch.where(n > 0, n, torch.ones_like(n))
    d = mb - ma
    mean = ma + d * nb / safe
    out = [n, mean]
    if order >= 2:
        nab = na * nb
        out.append(mma[0] + mmb[0] + d.square() * nab / safe)
    if order >= 3:
        m3 = (
            mma[1]
            + mmb[1]
            + d.pow(3) * nab * (na - nb) / safe.square()
            + 3 * d * (na * mmb[0] - nb * mma[0]) / safe
        )
        out.append(m3)
    if order >= 4:
        m4 = (
            mma[2]
            + mmb[2]
            + d.pow(4) * nab * (na.square() - nab + nb.square()) / safe.pow(3)
            + 6
            * d.square()
            * (na.square() * mmb[0] + nb.square() * mma[0])
            / safe.square()
            + 4 * d * (na * mmb[1] - nb * mma[1]) / safe
        )
        out.append(m4)
    return tuple(out)


class Moments(Reducer):
    """Per-pixel count, mean and central moments up to `order` (1..4).

    Attributes
    ----------
    count : Tensor
        Sum of weights (0-d if every update was unweighted).
    mean : Tensor
        Weighted mean.

    Notes
    -----
    Each update computes the chunk's own `(n, mean, M2..Mp)` by direct
    sums about the chunk mean, **in the chunk's dtype** (no chunk-sized
    float64 temporaries), then merges it into the running state with the
    same pairwise formulas as [`merge`][optix.stats.Moments.merge]. Raw
    power sums are never accumulated.

    Weights are frequency weights (`count` is their sum), as needed for
    masks and responsibilities; `var(ddof=1)` is only meaningful for
    integer-like weights. Pixels with zero total weight keep zeros in the
    state, and their accessors return NaN.

    !!! references "References"
        1. Pebay P. Formulas for robust, one-pass parallel computation of
           covariances and arbitrary-order statistical moments. 2008.

        ??? quote "BibTeX"
            ```bibtex
            @misc{pebay2008,
              author = {Pebay, P.},
              title = {Formulas for robust, one-pass parallel
                       computation of covariances and arbitrary-order
                       statistical moments},
              year = {2008}
            }
            ```
    """

    _state = ("count", "mean", "m2", "m3", "m4")

    def __init__(
        self,
        order: int = 2,
        *,
        dim: int | Sequence[int] | None = 0,
        state_dtype: torch.dtype | None = None,
    ) -> None:
        """Create an empty accumulator.

        Parameters
        ----------
        order : int
            Highest central moment kept (1 = mean only, 2 = variance, 3 =
            skewness, 4 = kurtosis).
        dim : int or tuple of int or None
            Reduced dims of each chunk (see
            [`Reducer`][optix.stats.Reducer]).
        state_dtype : torch.dtype, optional
            Dtype of the running state. Default: float64 (float32 on MPS).
        """
        if order not in (1, 2, 3, 4):
            raise ValueError(f"order must be in 1..4, got {order}")
        super().__init__(dim=dim)
        self.order = order
        self.state_dtype = state_dtype

    def _args(self) -> dict[str, object]:
        return {
            "order": self.order,
            "dim": self.dim,
            "state_dtype": self.state_dtype,
        }

    def _values(self) -> tuple[Tensor, ...]:
        names = self._state[: self.order + 1]
        return tuple(getattr(self, n) for n in names)

    def _set(self, values: tuple[Tensor, ...]) -> None:
        for name, value in zip(self._state, values, strict=False):
            setattr(self, name, value)

    def update(self, x: Tensor, *, weight: Tensor | None = None) -> Self:
        """Add a chunk of data.

        Parameters
        ----------
        x : Tensor
            Chunk; the dims in `self.dim` are reduced.
        weight : Tensor, optional
            Frequency weights (or boolean mask) broadcastable to `x`.

        Returns
        -------
        Moments
            self
        """
        x = x.detach()
        if not x.is_floating_point():
            x = x.to(torch.get_default_dtype())
        dims = self._dims(x)
        if math.prod(x.shape[d] for d in dims) == 0:
            return self
        sdt = self.state_dtype
        if sdt is None:
            sdt = torch.float32 if x.device.type == "mps" else torch.float64
        if weight is None:
            n = math.prod(x.shape[d] for d in dims)
            count = torch.tensor(n, dtype=sdt, device=x.device)
            mean = x.mean(dims, keepdim=True)
            dev = x - mean
            sums = [dev.pow(p).sum(dims) for p in range(2, self.order + 1)]
        else:
            weight = weight.detach().to(device=x.device).expand_as(x)
            w = weight.to(x.dtype)
            # zero-weight entries (e.g. masked NaN / inf) must not leak
            # into the sums: 0 * nan is nan
            x = torch.where(w > 0, x, 0)
            # count in the state dtype: float32 is not exact beyond 2^24
            wsum = weight.to(sdt).sum(dims, keepdim=True)
            safe = torch.where(wsum > 0, wsum, torch.ones_like(wsum))
            mean = (w * x).sum(dims, keepdim=True) / safe.to(x.dtype)
            dev = x - mean
            sums = [
                (w * dev.pow(p)).sum(dims) for p in range(2, self.order + 1)
            ]
            count = wsum.squeeze(dims)
        mean = mean.squeeze(dims)
        chunk = (count, mean.to(sdt), *(s.to(sdt) for s in sums))
        if self.count is None:
            self._set(chunk)
        else:
            self._set(_combine(self._values(), chunk, self.order))
        return self

    def merge(self, other: "Moments") -> Self:
        """Merge another [`Moments`][optix.stats.Moments] (same order)."""
        state = self._other_state(other)
        if other.order != self.order:
            raise ValueError("Cannot merge Moments of different orders")
        if state["count"] is None:
            return self
        names = self._state[: self.order + 1]
        theirs = tuple(state[n] for n in names)
        if self.count is None:
            self._set(tuple(t.clone() for t in theirs))
        else:
            theirs = tuple(t.to(self.mean.dtype) for t in theirs)
            self._set(_combine(self._values(), theirs, self.order))
        return self

    def _check(self, order: int) -> None:
        if self.order < order:
            raise ValueError(f"Needs order >= {order}, have {self.order}")
        if self.count is None:
            raise ValueError("No data: call update() first")

    def var(self, ddof: int = 1) -> Tensor:
        """Variance M₂ / (count − ddof); NaN where count ≤ ddof."""
        self._check(2)
        denom = self.count - ddof
        ok = denom > 0
        safe = torch.where(ok, denom, torch.ones_like(denom))
        return torch.where(ok, self.m2 / safe, torch.nan)

    def std(self, ddof: int = 1) -> Tensor:
        """Standard deviation: the square root of `var(ddof)`."""
        return self.var(ddof).sqrt()

    def skewness(self) -> Tensor:
        """Sample skewness √n M₃ / M₂^(3/2) (biased, g₁)."""
        self._check(3)
        ok = (self.m2 > 0) & (self.count > 0)
        m2 = torch.where(ok, self.m2, torch.ones_like(self.m2))
        value = self.count.sqrt() * self.m3 / m2.pow(1.5)
        return torch.where(ok, value, torch.nan)

    def kurtosis(self, excess: bool = True) -> Tensor:
        """Sample kurtosis n M₄ / M₂² (minus 3 if `excess`)."""
        self._check(4)
        ok = (self.m2 > 0) & (self.count > 0)
        m2 = torch.where(ok, self.m2, torch.ones_like(self.m2))
        value = self.count * self.m4 / m2.square() - (3 if excess else 0)
        return torch.where(ok, value, torch.nan)
