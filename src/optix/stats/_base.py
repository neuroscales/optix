"""Streaming, mergeable reducers and the single-pass driver."""

import copy as _copy
from collections.abc import Sequence
from typing import ClassVar, Self

import torch
from torch import Tensor

from ..io import ArrayLike, Layout, frame_batches

__all__ = ["Reducer", "reduce", "reduce_groups", "state_dtype"]


def state_dtype(device: torch.device | str | None) -> torch.dtype:
    """Default dtype of accumulators on `device`.

    float64, except on devices without it (MPS), where it is float32.
    """
    device = torch.device("cpu" if device is None else device)
    return torch.float32 if device.type == "mps" else torch.float64


def _reduced_dims(dim: int | Sequence[int] | None, ndim: int) -> tuple:
    """Normalise `dim` against a tensor of `ndim` dims."""
    if dim is None:
        return tuple(range(ndim))
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    out = []
    for d in dims:
        if not -ndim <= d < ndim:
            raise ValueError(f"dim {d} out of range for {ndim}D data")
        out.append(d % ndim)
    if len(set(out)) != len(out):
        raise ValueError(f"Repeated dims in {dim}")
    return tuple(sorted(out))


class Reducer:
    """Streaming, mergeable summary.

    State tensors (named in `_state`) are created on the first
    `update`; a never-updated reducer is the identity of `merge`.

    !!! note
        Every reducer satisfies, up to floating-point round-off:

        ```python
        R().update(cat(a, b)) == R().update(a).update(b)
                              == R().update(a).merge(R().update(b))
        ```
    """

    _state: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, dim: int | Sequence[int] | None = 0) -> None:
        """Create an empty reducer.

        Parameters
        ----------
        dim : int or tuple of int or None
            Dims of each chunk that are reduced. Default 0: per-pixel over
            the frame axis of a `(n, *spatial)` chunk. None: everything
            (global summary).
        """
        self.dim = dim if dim is None or isinstance(dim, int) else tuple(dim)
        for name in self._state:
            setattr(self, name, None)

    # -- to implement ----------------------------------------------------
    def update(self, x: Tensor, *, weight: Tensor | None = None) -> Self:
        """Add a chunk of data (optionally weighted). Returns self."""
        raise NotImplementedError

    def merge(self, other: Self) -> Self:
        """Merge another reducer into this one, in place. Returns self."""
        raise NotImplementedError

    def _args(self) -> dict[str, object]:
        """Constructor arguments (for `state_dict`)."""
        return {"dim": self.dim}

    # -- shared ----------------------------------------------------------
    @property
    def empty(self) -> bool:
        """True until the first update or merge."""
        return all(getattr(self, n) is None for n in self._state)

    def _dims(self, x: Tensor) -> tuple[int, ...]:
        return _reduced_dims(self.dim, x.ndim)

    def _device(self) -> torch.device | None:
        for name in self._state:
            value = getattr(self, name)
            if isinstance(value, Tensor):
                return value.device
        return None

    def _other_state(self, other: "Reducer") -> dict[str, Tensor | None]:
        """`other`'s state moved to this reducer's device (not in place)."""
        if type(other) is not type(self):
            raise TypeError(
                f"Cannot merge {type(other).__name__} into "
                f"{type(self).__name__}"
            )
        device = self._device()
        out = {}
        for name in self._state:
            value = getattr(other, name)
            if isinstance(value, Tensor) and device is not None:
                value = value.to(device)
            out[name] = value
        return out

    def copy(self) -> Self:
        """Independent copy (state tensors are cloned)."""
        new = _copy.copy(self)
        for name in self._state:
            value = getattr(self, name)
            if isinstance(value, Tensor):
                setattr(new, name, value.clone())
        return new

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Move the state (in place); `dtype` casts floating state only."""
        for name in self._state:
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                continue
            if dtype is not None and value.is_floating_point():
                value = value.to(dtype)
            if device is not None:
                value = value.to(device)
            setattr(self, name, value)
        return self

    def state_dict(self) -> dict[str, object]:
        """Constructor arguments and state tensors."""
        return {
            "class": type(self).__name__,
            "args": self._args(),
            "state": {n: getattr(self, n) for n in self._state},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> Self:
        """Rebuild a reducer saved with
        [`state_dict`][optix.stats.Reducer.state_dict]."""
        if state.get("class", cls.__name__) != cls.__name__:
            raise TypeError(f"State of {state['class']}, not {cls.__name__}")
        new = cls(**state["args"])
        for name, value in state["state"].items():
            setattr(new, name, value)
        return new

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._args().items())
        return f"{type(self).__name__}({args})"


def reduce(
    data: ArrayLike,
    *reducers: Reducer,
    ndim: int | None = None,
    layout: Layout | None = None,
    **chunk_options: object,
) -> tuple[Reducer, ...]:
    """One pass over frame batches feeding every reducer.

    Parameters
    ----------
    data : array-like or Labeled
        Data.
    *reducers : Reducer
        Reducers to update with each `(n, *spatial)` frame batch.
    ndim : int, optional
        Without a layout: number of trailing spatial dims (default 2).
    layout : Layout, optional
        Axis roles (see [`frame_batches`][optix.io.frame_batches]). It must
        not have group axes: use [`reduce_groups`][optix.stats.reduce_groups].
    **chunk_options
        Forwarded to [`frame_batches`][optix.io.frame_batches] (`overhead`,
        `memory`, `batch`, `dtype`, `device`, `prefetch`,
        `contiguous`, ...).

    Returns
    -------
    tuple of Reducer
        The same reducers, updated with every sample.
    """
    chunks = frame_batches(data, ndim=ndim, layout=layout, **chunk_options)
    if chunks.group_shape:
        raise ValueError(
            "reduce() pools every sample; the layout has group axes "
            f"{chunks.group_shape}: use reduce_groups()"
        )
    with torch.no_grad():
        for _, batch in chunks:
            for reducer in reducers:
                reducer.update(batch)
    return reducers


def reduce_groups(
    data: ArrayLike,
    *reducers: Reducer,
    ndim: int | None = None,
    layout: Layout | None = None,
    **chunk_options: object,
) -> dict[tuple[int, ...], tuple[Reducer, ...]]:
    """One pass over frame batches, with one copy of the reducers per group.

    Parameters
    ----------
    data, ndim, layout, **chunk_options
        See [`reduce`][optix.stats.reduce]; the layout may have group axes.
    *reducers : Reducer
        Templates: each group gets its own
        [`copy`][optix.stats.Reducer.copy]. The templates themselves are
        not updated.

    Returns
    -------
    dict of tuple of int to tuple of Reducer
        Keyed by group position (`()` without group axes), in iteration
        order. Groups without samples are absent.
    """
    chunks = frame_batches(data, ndim=ndim, layout=layout, **chunk_options)
    out: dict[tuple[int, ...], tuple[Reducer, ...]] = {}
    with torch.no_grad():
        for chunk in chunks:
            key = chunk.group
            if key not in out:
                out[key] = tuple(r.copy() for r in reducers)
            for reducer in out[key]:
                reducer.update(chunk.data)
    return out
