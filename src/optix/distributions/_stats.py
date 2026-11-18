"""Mergeable, weighted sufficient statistics of samples."""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Self

import torch
from torch import Tensor

from .._config import resolve_backend
from ._base import _fmt, _normalise

if TYPE_CHECKING:
    from ..stats import Moments

__all__ = [
    "CategoricalStats",
    "DirichletStats",
    "GammaStats",
    "NormalStats",
    "SuffStats",
]


def _batch_dims(
    name: str, dim: int | Sequence[int] | None, ndim: int
) -> tuple[int, ...]:
    """Normalise reduced batch dims (negative dims count from the last)."""
    if dim is None:
        return tuple(range(ndim))
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    out = []
    for d in dims:
        if not -ndim <= d < ndim:
            raise ValueError(
                f"{name}: dim {d} out of range for {ndim} batch dims"
            )
        out.append(d % ndim)
    if len(set(out)) != len(out):
        raise ValueError(f"{name}: repeated dims in {dim}")
    return tuple(sorted(out))


def _sum(t: Tensor, dims: tuple[int, ...]) -> Tensor:
    """``t.sum(dims)``; nothing is reduced when ``dims`` is empty."""
    return t.sum(dims) if dims else t


def _cast(t: Tensor, dtype: torch.dtype | None) -> Tensor:
    return t if dtype is None else t.to(dtype)


def _prepare(
    name: str,
    x: Tensor,
    event_ndim: int,
    weight: Tensor | None,
    dim: int | Sequence[int] | None,
) -> tuple[Tensor, Tensor | None, tuple[int, ...], torch.Size]:
    """Float ``x``, weights broadcast to its batch part, reduced dims."""
    x = torch.as_tensor(x)
    if not x.is_floating_point():
        x = x.to(resolve_backend()["dtype"])
    batch = x.shape[: x.ndim - event_ndim]
    dims = _batch_dims(name, dim, len(batch))
    if weight is not None:
        weight = torch.as_tensor(weight, device=x.device).to(x.dtype)
        weight = weight.expand(batch)
    return x, weight, dims, batch


def _count(
    weight: Tensor | None,
    dims: tuple[int, ...],
    batch: torch.Size,
    like: Tensor,
) -> Tensor:
    """Total weight over ``dims`` (exact integer count without weights)."""
    if weight is None:
        n = math.prod(batch[d] for d in dims)
        return torch.tensor(float(n), dtype=like.dtype, device=like.device)
    return _sum(weight, dims)


def _wsum(
    t: Tensor, weight: Tensor | None, dims: tuple[int, ...], e: int
) -> Tensor:
    """``sum(w * t)`` over the batch ``dims`` of ``t`` (``e`` event dims)."""
    if weight is not None:
        t = weight.reshape(weight.shape + (1,) * e) * t
    return _sum(t, dims)


def _mask(t: Tensor, weight: Tensor | None, e: int, fill: float) -> Tensor:
    """Replace entries of zero weight (0 * nan would be nan)."""
    if weight is None:
        return t
    w = weight.reshape(weight.shape + (1,) * e)
    return torch.where(w > 0, t, fill)


def _safe_log(x: Tensor) -> Tensor:
    return x.clamp_min(torch.finfo(x.dtype).tiny).log()


class SuffStats:
    """Weighted sufficient statistics of samples from one family.

    Holds ``sum_n w_n T(x_n)`` and the total weight ``n = sum_n w_n`` over
    a set of samples (observed values, or expectations of latent ones).
    This is what models accumulate over batches and what
    :func:`~optix.distributions.posterior` and the ``fit`` classmethods
    consume.

    Fields have shape ``(*batch, *field_event)`` and are broadcast against
    each other. **dtype**: the statistics of a batch are reduced in the
    data's own dtype (no data-sized upcast); ``dtype=`` in the
    constructors casts only the reduced result, which is how a float64
    running state is obtained from float32 data.

    Statistics merge with ``+`` (out of place), :meth:`add_` (in place,
    also available as :meth:`merge`) and :meth:`sum` (over batch dims);
    ``zeros()`` is the identity of ``+`` and ``n == 0`` entries are legal
    everywhere. Weights must be non-negative (not checked).
    """

    __slots__ = ("_values", "_batch_shape", "_event_shape")

    #: Field names and their event ndims.
    _fields: ClassVar[tuple[tuple[str, int], ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for i, (name, _) in enumerate(cls._fields):
            if name in cls.__dict__:
                continue

            def get(self: SuffStats, i: int = i) -> Tensor:
                return self._values[i]

            get.__doc__ = f"Field ``{name}``."
            setattr(cls, name, property(get))

    def _setup(self, *values: object) -> None:
        name = type(self).__name__
        names = [n for n, _ in self._fields]
        ndims = [e for _, e in self._fields]
        raw, batch, events = _normalise(name, values, names, ndims)
        shaped = [(n, e) for n, e in zip(names, events, strict=True) if e]
        if any(e != shaped[0][1] for _, e in shaped):
            parts = ", ".join(f"{n} {_fmt(e)}" for n, e in shaped)
            raise ValueError(f"{name}: event shapes differ: {parts}")
        self._values = raw
        self._batch_shape = batch
        self._event_shape = shaped[0][1] if shaped else torch.Size(())

    @classmethod
    def _new(cls, *values: Tensor) -> Self:
        self = object.__new__(cls)
        self._setup(*values)
        return self

    # -- accessors -------------------------------------------------------
    @property
    def n(self) -> Tensor:
        """Total weight ``sum_n w_n``, shape broadcastable to the batch."""
        return self._values[0]

    @property
    def batch_shape(self) -> torch.Size:
        """Broadcast batch shape of the fields."""
        return self._batch_shape

    @property
    def event_shape(self) -> torch.Size:
        """Event shape of the statistics (that of the family)."""
        return self._event_shape

    @property
    def dtype(self) -> torch.dtype:
        """dtype of the fields."""
        return self._values[0].dtype

    @property
    def device(self) -> torch.device:
        """Device of the fields."""
        return self._values[0].device

    def astuple(self) -> tuple[Tensor, ...]:
        """Fields, in declaration order (unexpanded)."""
        return self._values

    def _full(self) -> tuple[Tensor, ...]:
        """Fields expanded to ``batch_shape + field_event``."""
        out = []
        for t, (_, e) in zip(self._values, self._fields, strict=True):
            out.append(t.expand(self._batch_shape + t.shape[t.ndim - e :]))
        return tuple(out)

    def __repr__(self) -> str:
        dtype = str(self.dtype).removeprefix("torch.")
        return (
            f"{type(self).__name__}(batch_shape={_fmt(self._batch_shape)}, "
            f"event_shape={_fmt(self._event_shape)}, dtype={dtype}, "
            f"device={self.device})"
        )

    # -- merging (overridden by central-form statistics) ------------------
    @classmethod
    def _combine(
        cls, a: tuple[Tensor, ...], b: tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]:
        return tuple(x + y for x, y in zip(a, b, strict=True))

    def _reduce(self, dims: tuple[int, ...]) -> tuple[Tensor, ...]:
        return tuple(_sum(t, dims) for t in self._full())

    def _scaled(self, factor: Tensor) -> tuple[Tensor, ...]:
        return tuple(t * factor for t in self._values)

    def _check_type(self, other: object, op: str) -> None:
        if type(other) is not type(self):
            raise TypeError(
                f"{type(self).__name__}.{op}: cannot merge "
                f"{type(other).__name__}"
            )

    def __add__(self, other: Self) -> Self:
        """Merge, out of place, on broadcast batch shapes."""
        if isinstance(other, int) and other == 0:
            return self.copy()
        self._check_type(other, "__add__")
        other = other.to(device=self.device)
        return self._new(*self._combine(self._values, other._values))

    def __radd__(self, other: object) -> Self:
        """``0 + s``, so that ``sum([s1, s2, ...])`` works."""
        if isinstance(other, int) and other == 0:
            return self.copy()
        return NotImplemented

    def add_(self, other: Self) -> Self:
        """Merge ``other`` into ``self``, in place; return ``self``.

        ``self``'s fields must have the full batch shape and own their
        memory (build it with :meth:`zeros` or :meth:`copy`). ``other`` is
        cast to ``self``'s dtype and device first, so a float64
        accumulator accepts float32 batch statistics.
        """
        self._check_type(other, "add_")
        name = type(self).__name__
        try:
            grown = torch.broadcast_shapes(
                self._batch_shape, other.batch_shape
            )
        except RuntimeError:
            grown = None
        if grown != self._batch_shape:
            raise ValueError(
                f"{name}.add_: cannot accumulate batch shape "
                f"{_fmt(other.batch_shape)} into {_fmt(self._batch_shape)}"
            )
        for t, (field, e) in zip(self._values, self._fields, strict=True):
            full = self._batch_shape + t.shape[t.ndim - e :]
            shared = any(
                s == 0 and d > 1
                for s, d in zip(t.stride(), t.shape, strict=True)
            )
            if t.shape != full or shared:
                raise ValueError(
                    f"{name}.add_: field {field} of shape {_fmt(t.shape)} "
                    f"does not own a full {_fmt(full)} buffer; start from "
                    "zeros() or copy()"
                )
        other = other.to(device=self.device, dtype=self.dtype)
        new = self._combine(self._values, other._values)
        for t, v in zip(self._values, new, strict=True):
            t.copy_(v)
        return self

    def merge(self, other: Self) -> Self:
        """Alias of :meth:`add_` (the ``optix.stats.Reducer`` name)."""
        return self.add_(other)

    def scale(self, factor: Tensor | float) -> Self:
        """Statistics with the total weight multiplied by ``factor``.

        Used for tempering, subsampling corrections and stochastic VB.
        """
        factor = torch.as_tensor(factor, dtype=self.dtype, device=self.device)
        return self._new(*self._scaled(factor))

    def sum(self, dim: int | Sequence[int] | None = None) -> Self:
        """Merge along batch dims (``None``: all); event dims are kept.

        Negative dims count from the last batch dim.
        """
        name = f"{type(self).__name__}.sum"
        dims = _batch_dims(name, dim, len(self._batch_shape))
        return self._new(*self._reduce(dims))

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Copy on another device and/or floating dtype.

        Unlike ``optix.stats.Reducer.to``, this returns a new object.
        """
        out = []
        for t in self._values:
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            if device is not None:
                t = t.to(device)
            out.append(t)
        return self._new(*out)

    def copy(self) -> Self:
        """Independent copy with full-shape, owned fields (an accumulator)."""
        return self._new(*(t.clone() for t in self._full()))

    def state_dict(self) -> dict[str, object]:
        """``{"class", "args", "state"}``, as ``Reducer.state_dict``."""
        names = [n for n, _ in self._fields]
        return {
            "class": type(self).__name__,
            "args": {},
            "state": dict(zip(names, self._values, strict=True)),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> Self:
        """Rebuild statistics saved with :meth:`state_dict`."""
        if state.get("class", cls.__name__) != cls.__name__:
            raise TypeError(f"State of {state['class']}, not {cls.__name__}")
        values = state["state"]
        return cls._new(*(values[n] for n, _ in cls._fields))

    @classmethod
    def zeros(
        cls,
        batch_shape: Sequence[int] = (),
        *,
        event_shape: Sequence[int] = (),
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        """Empty accumulator (identity of ``+``) with owned fields.

        Parameters
        ----------
        batch_shape : sequence of int
            Batch shape.
        event_shape : sequence of int
            Event shape of the family (``(K,)`` for Dirichlet and
            Categorical statistics).
        dtype, device
            Default: the configuration (pass ``torch.float64`` for a
            precise running state).
        """
        backend = resolve_backend(dtype=dtype, device=device)
        batch, event = tuple(batch_shape), tuple(event_shape)
        values = []
        for _, e in cls._fields:
            shape = batch + (event[len(event) - e :] if e else ())
            values.append(torch.zeros(shape, **backend))
        return cls._new(*values)


class GammaStats(SuffStats):
    """Statistics of a Gamma likelihood: ``(n, sum_x, sum_log_x)``.

    Parameters
    ----------
    n : Tensor
        Total weight ``sum w``.
    sum_x : Tensor
        ``sum w x``.
    sum_log_x : Tensor
        ``sum w log(x)``.
    """

    __slots__ = ()
    _fields = (("n", 0), ("sum_x", 0), ("sum_log_x", 0))
    sum_x: Tensor
    sum_log_x: Tensor

    def __init__(
        self,
        n: Tensor | float,
        sum_x: Tensor | float,
        sum_log_x: Tensor | float,
    ) -> None:
        self._setup(n, sum_x, sum_log_x)

    @classmethod
    def from_samples(
        cls,
        x: Tensor,
        weight: Tensor | None = None,
        *,
        dim: int | Sequence[int] | None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Statistics of observed positive samples.

        Parameters
        ----------
        x : Tensor
            Samples ``> 0``.
        weight : Tensor, optional
            Non-negative frequency weights (or a boolean mask),
            broadcastable to ``x``.
        dim : int or sequence of int or None
            Reduced dims (``None``: all, ``()``: none). Required.
        dtype : torch.dtype, optional
            dtype of the result (the reduction runs in ``x``'s dtype).

        Returns
        -------
        GammaStats
            ``log(x)`` uses ``max(x, tiny)``, so zero samples of zero
            weight contribute nothing.
        """
        x, w, dims, batch = _prepare(cls.__name__, x, 0, weight, dim)
        x = _mask(x, w, 0, 1.0)
        return cls._new(
            _cast(_count(w, dims, batch, x), dtype),
            _cast(_wsum(x, w, dims, 0), dtype),
            _cast(_wsum(_safe_log(x), w, dims, 0), dtype),
        )

    @classmethod
    def from_expectations(
        cls,
        log_x: Tensor,
        x: Tensor,
        weight: Tensor | None = None,
        *,
        dim: int | Sequence[int] | None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Statistics of latent samples given ``E[log x]`` and ``E[x]``.

        The arguments are the fields of ``Gamma.expected_stats()``;
        ``weight``, ``dim`` and ``dtype`` as in :meth:`from_samples`.
        """
        log_x, x = torch.broadcast_tensors(
            torch.as_tensor(log_x), torch.as_tensor(x)
        )
        x, w, dims, batch = _prepare(cls.__name__, x, 0, weight, dim)
        log_x = log_x.to(x.dtype)
        return cls._new(
            _cast(_count(w, dims, batch, x), dtype),
            _cast(_wsum(_mask(x, w, 0, 0.0), w, dims, 0), dtype),
            _cast(_wsum(_mask(log_x, w, 0, 0.0), w, dims, 0), dtype),
        )


class DirichletStats(SuffStats):
    """Statistics of a Dirichlet likelihood: ``(n, sum_log_x)``.

    Parameters
    ----------
    n : Tensor
        Total weight ``sum w``, shape ``(*batch)``.
    sum_log_x : (*batch, K) Tensor
        ``sum w log(x)``.
    """

    __slots__ = ()
    _fields = (("n", 0), ("sum_log_x", 1))
    sum_log_x: Tensor

    def __init__(self, n: Tensor | float, sum_log_x: Tensor) -> None:
        self._setup(n, sum_log_x)

    @classmethod
    def from_samples(
        cls,
        x: Tensor,
        weight: Tensor | None = None,
        *,
        dim: int | Sequence[int] | None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Statistics of observed points ``x`` of the simplex ``(..., K)``.

        ``weight`` broadcasts against ``x.shape[:-1]``; ``dim`` are batch
        dims (the last dim of ``x`` is the event); see
        :meth:`GammaStats.from_samples`.
        """
        x, w, dims, batch = _prepare(cls.__name__, x, 1, weight, dim)
        x = _mask(x, w, 1, 1.0)
        return cls._new(
            _cast(_count(w, dims, batch, x), dtype),
            _cast(_wsum(_safe_log(x), w, dims, 1), dtype),
        )

    @classmethod
    def from_expectations(
        cls,
        log_x: Tensor,
        weight: Tensor | None = None,
        *,
        dim: int | Sequence[int] | None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Statistics of latent probability vectors given ``E[log x]``.

        ``log_x`` is ``Dirichlet.expected_stats().log_x`` (or
        ``Dirichlet.mean_log``), shape ``(..., K)``.
        """
        log_x, w, dims, batch = _prepare(cls.__name__, log_x, 1, weight, dim)
        return cls._new(
            _cast(_count(w, dims, batch, log_x), dtype),
            _cast(_wsum(_mask(log_x, w, 1, 0.0), w, dims, 1), dtype),
        )


class CategoricalStats(SuffStats):
    """Statistics of a categorical likelihood: ``counts`` ``(*batch, K)``.

    Parameters
    ----------
    counts : (*batch, K) Tensor
        Weighted counts ``sum w onehot(z)`` (or ``sum w p`` for soft
        assignments). ``n = counts.sum(-1)``.
    """

    __slots__ = ()
    _fields = (("counts", 1),)
    counts: Tensor

    def __init__(self, counts: Tensor) -> None:
        self._setup(counts)

    @property
    def n(self) -> Tensor:
        """Total weight ``counts.sum(-1)``."""
        return self._values[0].sum(-1)

    @classmethod
    def from_samples(
        cls,
        x: Tensor,
        weight: Tensor | None = None,
        *,
        dim: int | Sequence[int] | None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Statistics of one-hot or soft assignments ``x`` ``(..., K)``.

        See :meth:`GammaStats.from_samples`; for integer labels use
        :meth:`from_labels`.
        """
        x, w, dims, _ = _prepare(cls.__name__, x, 1, weight, dim)
        x = _mask(x, w, 1, 0.0)
        return cls._new(_cast(_wsum(x, w, dims, 1), dtype))

    @classmethod
    def from_expectations(
        cls,
        x: Tensor,
        weight: Tensor | None = None,
        *,
        dim: int | Sequence[int] | None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Statistics of latent labels given their probabilities ``E[x]``.

        ``x`` is ``Categorical.expected_stats().x`` (the responsibilities).
        """
        return cls.from_samples(x, weight, dim=dim, dtype=dtype)

    @classmethod
    def from_labels(
        cls,
        index: Tensor,
        num_classes: int,
        weight: Tensor | None = None,
        *,
        dim: int | Sequence[int] | None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Statistics of integer labels in ``[0, num_classes)``.

        Parameters
        ----------
        index : Tensor
            Integer labels; all its dims are batch dims.
        num_classes : int
            Number of categories ``K``.
        weight, dim, dtype
            As in :meth:`GammaStats.from_samples`. The counts are computed
            in ``dtype`` (default: the weights' floating dtype, else the
            configuration).
        """
        index = torch.as_tensor(index)
        if dtype is None:
            if isinstance(weight, Tensor) and weight.is_floating_point():
                dtype = weight.dtype
            else:
                dtype = resolve_backend()["dtype"]
        onehot = torch.nn.functional.one_hot(index.long(), num_classes)
        return cls.from_samples(onehot.to(dtype), weight, dim=dim)


class NormalStats(SuffStats):
    """Statistics of a Normal likelihood, central form ``(n, mean, scatter)``.

    ``mean = sum w x / n`` and ``scatter = sum w (x - mean)^2``, the same
    state as ``optix.stats.Moments(order=2)``. Raw power sums are never
    formed, so large offsets do not destroy the scatter in float32.

    Parameters
    ----------
    n : Tensor
        Total weight.
    mean : Tensor
        Weighted mean (any value where ``n == 0``, conventionally 0).
    scatter : Tensor
        Weighted sum of squared deviations from ``mean``.
    """

    __slots__ = ()
    _fields = (("n", 0), ("mean", 0), ("scatter", 0))
    mean: Tensor
    scatter: Tensor

    def __init__(
        self,
        n: Tensor | float,
        mean: Tensor | float,
        scatter: Tensor | float,
    ) -> None:
        self._setup(n, mean, scatter)

    @property
    def sum_x(self) -> Tensor:
        """``sum w x = n * mean``."""
        return self.n * self.mean

    @property
    def sum_x_sq(self) -> Tensor:
        """``sum w x^2 = scatter + n * mean^2``."""
        return self.scatter + self.n * self.mean.square()

    @classmethod
    def _combine(
        cls, a: tuple[Tensor, ...], b: tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]:
        # Pebay's pairwise update
        na, ma, sa = a
        nb, mb, sb = b
        n = na + nb
        safe = torch.where(n > 0, n, torch.ones_like(n))
        d = mb - ma
        mean = ma + d * nb / safe
        scatter = sa + sb + d.square() * na * nb / safe
        return n, mean, scatter

    def _reduce(self, dims: tuple[int, ...]) -> tuple[Tensor, ...]:
        n, mean, scatter = self._full()
        if not dims:
            return n, mean, scatter
        total = n.sum(dims, keepdim=True)
        safe = torch.where(total > 0, total, torch.ones_like(total))
        m = (n * mean).sum(dims, keepdim=True) / safe
        s = scatter.sum(dims) + (n * (mean - m).square()).sum(dims)
        return total.squeeze(dims), m.squeeze(dims), s

    def _scaled(self, factor: Tensor) -> tuple[Tensor, ...]:
        n, mean, scatter = self._values
        return n * factor, mean, scatter * factor

    @classmethod
    def from_samples(
        cls,
        x: Tensor,
        weight: Tensor | None = None,
        *,
        dim: int | Sequence[int] | None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Statistics of observed samples (two passes, central form).

        See :meth:`GammaStats.from_samples`.
        """
        x, w, dims, batch = _prepare(cls.__name__, x, 0, weight, dim)
        n = _count(w, dims, batch, x)
        if not dims:
            mean = x if w is None else torch.where(w > 0, x, 0)
            scatter = torch.zeros_like(mean)
        elif w is None:
            # an empty batch (static shape) gets mean 0, not 0 / 0
            mean = x.sum(dims, keepdim=True) / max(
                math.prod(batch[d] for d in dims), 1
            )
            scatter = (x - mean).square().sum(dims)
            mean = mean.squeeze(dims)
        else:
            x = _mask(x, w, 0, 0.0)
            safe = torch.where(n > 0, n, torch.ones_like(n))
            mean = (w * x).sum(dims, keepdim=True)
            mean = mean / safe.reshape(mean.shape)
            scatter = (w * (x - mean).square()).sum(dims)
            mean = mean.squeeze(dims)
        return cls._new(
            _cast(n, dtype), _cast(mean, dtype), _cast(scatter, dtype)
        )

    @classmethod
    def from_expectations(
        cls,
        x: Tensor,
        x_sq: Tensor,
        weight: Tensor | None = None,
        *,
        dim: int | Sequence[int] | None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Statistics of latent samples given ``E[x]`` and ``E[x^2]``.

        ``scatter = sum w (E[x^2] - E[x]^2) + sum w (E[x] - mean)^2``.
        """
        x, x_sq = torch.broadcast_tensors(
            torch.as_tensor(x), torch.as_tensor(x_sq)
        )
        x, w, dims, batch = _prepare(cls.__name__, x, 0, weight, dim)
        var = _mask(x_sq.to(x.dtype) - x.square(), w, 0, 0.0)
        within = _wsum(var, w, dims, 0)
        base = cls.from_samples(x, weight=w, dim=dims)
        n, mean, scatter = base.astuple()
        return cls._new(
            _cast(n, dtype), _cast(mean, dtype), _cast(scatter + within, dtype)
        )

    @classmethod
    def from_moments(cls, moments: "Moments") -> Self:
        """``(count, mean, m2)`` of an ``optix.stats.Moments`` (no copy).

        Requires ``order >= 2`` and at least one update.
        """
        if moments.order < 2 or moments.count is None:
            raise ValueError(
                "NormalStats.from_moments needs an updated Moments with "
                "order >= 2"
            )
        return cls._new(moments.count, moments.mean, moments.m2)
