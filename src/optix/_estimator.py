"""Estimator base class, tags, fit history and (de)serialisation."""

import json
import math
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import ClassVar, Literal, Self

import numpy as np
import torch
from torch import Tensor

from ._config import _parse_device, _parse_dtype, resolve_backend
from ._version import __version__
from .estimates import Estimate
from .io import (
    ArrayLike,
    Chunk,
    Labeled,
    Layout,
    WritableArrayLike,
    blocks,
    chunks_of,
    flush,
    frame_batches,
    labeled,
    prepare_out,
)
from .stats import Reducer

__all__ = ["Estimator", "History", "Tags", "estimators", "load"]

Streaming = Literal[
    "pointwise", "per-frame", "halo", "per-pixel-stats", "whole-stack"
]


@dataclass(frozen=True)
class Tags:
    """Static description of an estimator.

    Attributes
    ----------
    targets : tuple of str
        What it estimates, e.g. `("shading", "sensor")`.
    streaming : str
        Data access pattern: `"pointwise"`, `"per-frame"`, `"halo"`,
        `"per-pixel-stats"` or `"whole-stack"`.
    modality : tuple of str
        Imaging modalities it is meant for.
    noise : str
        Noise model: `"none"`, `"gaussian"`, `"poisson"`,
        `"poisson-gaussian"`, ...
    many_frames : bool
        Whether it needs many frames.
    reference : str
        Citation or DOI.
    """

    targets: tuple[str, ...]
    streaming: Streaming
    modality: tuple[str, ...] = ("any",)
    noise: str = "none"
    many_frames: bool = False
    reference: str = ""


@dataclass
class History:
    """Objective trace of an iterative fit (objective is maximised).

    Attributes
    ----------
    objective : list of float
    extras : dict of str to list of float
        Other per-iteration diagnostics.
    converged : bool
    reason : str
        Why the fit stopped.
    """

    objective: list[float] = field(default_factory=list)
    extras: dict[str, list[float]] = field(default_factory=dict)
    converged: bool = False
    reason: str = ""

    def record(self, objective: float, **extras: float) -> None:
        """Append one iteration.

        Takes Python floats: the caller does the single `.item()` per
        iteration.
        """
        self.objective.append(float(objective))
        for key, value in extras.items():
            self.extras.setdefault(key, []).append(float(value))

    def gain(self) -> float:
        """Relative change of the last step, `(Lₜ − Lₜ₋₁) / |Lₜ|`.

        Returns `inf` with fewer than two records.
        """
        if len(self.objective) < 2:
            return math.inf
        last, prev = self.objective[-1], self.objective[-2]
        return (last - prev) / (abs(last) + 1e-30)

    def check(self, tol: float, *, monotone: bool = False) -> bool:
        """Whether `|Lₜ − Lₜ₋₁| / |Lₜ| < tol`; sets `converged`.

        Parameters
        ----------
        tol : float
            Relative tolerance.
        monotone : bool
            Warn (without stopping) when the objective decreases beyond
            round-off, which flags a bug in a VB / EM bound.
        """
        gain = self.gain()
        if monotone and gain < -64 * np.finfo(np.float64).eps:
            warnings.warn(
                f"Objective decreased at iteration {len(self.objective)} "
                f"(relative change {gain:.3g})",
                RuntimeWarning,
                stacklevel=2,
            )
        if abs(gain) < tol:
            self.converged = True
            self.reason = f"relative change {abs(gain):.3g} < tol {tol:g}"
        return self.converged


_REGISTRY: dict[str, type["Estimator"]] = {}
_RESERVED = {
    "self",
    "faithful",
    "dtype",
    "device",
    "layout",  # routed to `self.layout` by `_resolve`, not an option
    "__class__",
    "kwargs",
}


def _same(a: object, b: object) -> bool:
    try:
        return bool(a == b)
    except (RuntimeError, TypeError, ValueError):
        return a is b


def _jsonable(value: object) -> object:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _move(
    value: object, device: torch.device | None, dtype: torch.dtype | None
) -> object:
    """Recursively move tensors / estimates / reducers in containers."""
    if isinstance(value, Tensor):
        if dtype is not None and value.is_floating_point():
            value = value.to(dtype)
        return value if device is None else value.to(device)
    if isinstance(value, Estimate):
        return value.to(device, dtype)
    if isinstance(value, Reducer):
        return value.to(device, dtype)
    if isinstance(value, dict):
        return {k: _move(v, device, dtype) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_move(v, device, dtype) for v in value)
    return value


def _shares_memory(a: object, b: object) -> bool:
    if a is b:
        return True
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        if a.device != b.device:
            return False
        pa, pb = a.untyped_storage(), b.untyped_storage()
        return pa.data_ptr() == pb.data_ptr()
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return bool(np.may_share_memory(a, b))
    return False


class Estimator:
    """Base class of optix estimators.

    Subclasses define `tags` (which registers them), the `paper` and
    `recommended` presets, spell out their options as keyword-only
    parameters defaulting to None, and call
    `self._resolve(faithful, locals())`, which also routes a `layout`
    parameter to `self.layout`. They implement `_fit` and
    `_apply`; the public [`fit`][optix.Estimator.fit] /
    [`apply`][optix.Estimator.apply] run them under `torch.no_grad()`.

    Attributes
    ----------
    options : dict
        Resolved hyper-parameters.
    is_faithful : bool
        Whether the resolved options equal the paper preset.
    estimates : dict of str to Estimate
        Results of [`fit`][optix.Estimator.fit], e.g.
        `{"sensor": ..., "shading": ...}`. Parameter maps are shaped
        `(*group, *spatial)`, spatial axes in layout order.
    layout : Layout or None
        Layout used by the last [`fit`][optix.Estimator.fit] (or given at
        construction); [`apply`][optix.Estimator.apply] uses it for plain
        arrays.
    history : History

    Notes
    -----
    Subclasses receive a [`Labeled`][optix.io.Labeled] array in `_fit` /
    `_apply` whenever a layout is set, so that the io and stats helpers
    pick it up; `_layout_of` and `_map_shape` give the layout (default
    included) and the shape of the parameter maps.
    """

    tags: ClassVar[Tags]
    paper: ClassVar[dict[str, object]] = {}
    recommended: ClassVar[dict[str, object]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "tags" in cls.__dict__:
            _REGISTRY[cls.__name__] = cls

    def __init__(
        self,
        *,
        faithful: bool = True,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        layout: Layout | None = None,
        **options: object,
    ) -> None:
        """Set up the working backend and resolve the options.

        Parameters
        ----------
        faithful : bool
            Selects the preset of defaults: `paper` when True,
            `recommended` when False. Explicit option values always win.
        dtype, device : optional
            Working backend (default: configuration; the device follows a
            Tensor input when unset).
        layout : Layout, optional
            Axis roles of the data (see [`Layout`][optix.io.Layout]); can
            also be given to [`fit`][optix.Estimator.fit], or carried by a
            labelled array. Default: `(*samples, H, W)`.
        **options
            Hyper-parameters (the base's view of the subclass's options).
        """
        self.dtype = _parse_dtype(dtype)
        self.device = _parse_device(device)
        self.layout = None
        self._set_layout(layout)
        self.estimates: dict[str, Estimate] = {}
        self.history = History()
        self._resolve(faithful, options)

    def _set_layout(self, layout: Layout | None) -> None:
        """Validate `layout` and store it in `self.layout` (None keeps it)."""
        if layout is None:
            return
        if not isinstance(layout, Layout):
            raise TypeError(f"layout must be a Layout, got {layout!r}")
        self.layout = layout

    def _resolve(self, faithful: bool, params: Mapping[str, object]) -> None:
        """Resolve options against the preset selected by `faithful`.

        `layout` is not an option: when `params` holds one (the
        subclass's `locals()`), it goes to `self.layout`, so it is kept
        even if the subclass did not pass it on to `Estimator.__init__`.
        The other reserved names (`faithful`, `dtype`, `device`, ...)
        and private names are skipped.
        """
        self._set_layout(params.get("layout"))
        preset = self.paper if faithful else self.recommended
        options = {}
        for name, value in params.items():
            if name in _RESERVED or name.startswith("_"):
                continue
            options[name] = preset.get(name) if value is None else value
        self.faithful = bool(faithful)
        self.options = options
        self.is_faithful = all(
            _same(value, self.paper.get(name))
            for name, value in options.items()
        )

    def _backend(self, like: object = None) -> dict:
        if isinstance(like, Labeled):
            like = like.array
        return resolve_backend(self.dtype, self.device, like=like)

    # -- public API ------------------------------------------------------
    def fit(
        self,
        data: ArrayLike,
        *,
        layout: Layout | None = None,
        init: object = None,
        callback: Callable[["Estimator"], object] | None = None,
        verbose: bool = False,
    ) -> Self:
        """Fit the model to data.

        Parameters
        ----------
        data : array-like or Labeled
            Data; `(*samples, H, W)` without a layout.
        layout : Layout, optional
            Axis roles (default: that of a labelled `data`, else the one
            given at construction). Stored in `self.layout`.
        init : Estimate, sequence of Estimate, or Estimator, optional
            Initial values (cloned; the caller's objects are not modified).
        callback : callable, optional
            Called by iterative estimators after each iteration.
        verbose : bool

        Returns
        -------
        Estimator
            self, with `estimates` and `history` filled. `history`
            is reset at the start of each call.
        """
        self.history = History()
        if isinstance(data, Labeled):
            if layout is not None and layout != data.layout:
                raise ValueError(
                    f"layout={layout!r} conflicts with the layout of the "
                    f"labelled array, {data.layout!r}"
                )
            layout = data.layout
        layout = self.layout if layout is None else layout
        data = self._labeled(data, layout)
        self.layout = layout
        with torch.no_grad():
            self._fit(data, init=init, callback=callback, verbose=verbose)
        meta = {
            "estimator": type(self).__name__,
            "version": __version__,
            "options": {k: _jsonable(v) for k, v in self.options.items()},
            "faithful": self.is_faithful,
        }
        for estimate in self.estimates.values():
            estimate.meta.update(meta)
        return self

    def apply(
        self, data: ArrayLike, out: WritableArrayLike | None = None
    ) -> WritableArrayLike:
        """Apply the fitted model (e.g. correct the data).

        Parameters
        ----------
        data : array-like or Labeled
            Data, in the layout of the fit (unless labelled otherwise).
        out : writable array-like, optional
            Output (tensor, numpy, zarr, memmap...). Default: a new tensor
            on the device of `data` if it is a Tensor, else on the CPU.

        Returns
        -------
        writable array-like
            `out`.
        """
        if not isinstance(data, Labeled):
            data = self._labeled(data, self.layout)
        with torch.no_grad():
            return self._apply(data, out)

    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        raise NotImplementedError

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        raise NotImplementedError

    # -- helpers for subclasses ------------------------------------------
    @staticmethod
    def _labeled(data: ArrayLike, layout: Layout | None) -> ArrayLike:
        """`data` labelled with `layout` (unchanged when None)."""
        if layout is None:
            return data
        if isinstance(data, Labeled):
            return data if data.layout == layout else labeled(data, layout)
        return labeled(data, layout)

    @staticmethod
    def _layout_of(data: ArrayLike, ndim: int = 2) -> Layout:
        """Layout of `data` (the default one with `ndim` spatial dims)."""
        if isinstance(data, Labeled):
            return data.layout
        return Layout.default(len(data.shape), ndim)

    def _map_shape(self, data: ArrayLike, ndim: int = 2) -> tuple[int, ...]:
        """Shape `(*group, *spatial)` of the parameter maps for `data`."""
        return self._layout_of(data, ndim).map_shape(data)

    @staticmethod
    def _init_estimates(init: object) -> list[Estimate]:
        if init is None:
            return []
        if isinstance(init, Estimator):
            return list(init.estimates.values())
        if isinstance(init, Estimate):
            return [init]
        if isinstance(init, Sequence):
            return [e for e in init if isinstance(e, Estimate)]
        raise TypeError(f"Cannot initialise from {type(init).__name__}")

    def _pick(
        self,
        init: object,
        cls: type[Estimate],
        *,
        shape: Sequence[int] | None = None,
        like: object = None,
    ) -> Estimate | None:
        """First estimate of type `cls` in `init`, cloned and moved.

        The spatial maps must broadcast to `shape` (scalars do). The
        per-sample fields (`cls._per_sample`, e.g. the per-frame
        `ShadingEstimate.baseline`) are not checked against `shape`:
        the caller matches them to its samples. `None` fields are left
        for the data-driven initialisation; `meta` (e.g.
        `baseline_units`) is kept.
        """
        for estimate in self._init_estimates(init):
            if isinstance(estimate, cls):
                backend = self._backend(like)
                picked = estimate.clone().to(**backend)
                if shape is not None:
                    for name, value in picked._fields().items():
                        if value is None or name in picked._per_sample:
                            continue
                        try:
                            torch.broadcast_shapes(value.shape, tuple(shape))
                        except RuntimeError:
                            raise ValueError(
                                f"init {cls.__name__}.{name} has shape "
                                f"{tuple(value.shape)}, which does not "
                                f"broadcast to {tuple(shape)}"
                            ) from None
                return picked
        return None

    def _warn_unused(self, init: object, *usable: type[Estimate]) -> None:
        """Warn about init estimates of a type this estimator cannot use."""
        for estimate in self._init_estimates(init):
            if not isinstance(estimate, usable):
                warnings.warn(
                    f"{type(self).__name__} ignores init of type "
                    f"{type(estimate).__name__}",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def _apply_frames(
        self,
        data: ArrayLike,
        out: WritableArrayLike | None,
        fn: Callable[[Chunk, Tensor], Tensor],
        *,
        ndim: int = 2,
        overhead: float = 2.0,
        contiguous: bool = False,
    ) -> WritableArrayLike:
        """Apply `fn(chunk, batch)` to frame batches and store the results.

        `batch` is `(n, *spatial)`; `chunk.group` selects the
        parameters of a `(*group, *spatial)` map. `fn` returns a tensor
        of the shape of `batch`, written back with `chunk.store`.
        In-place use (`out is data`) is safe: chunks are owned. `batch`
        is a strided view unless `contiguous` (see
        [`frame_batches`][optix.io.frame_batches]).
        """
        raw = data.array if isinstance(data, Labeled) else data
        out = prepare_out(out, tuple(data.shape), like=raw, dtype=self.dtype)
        chunks = frame_batches(
            data,
            ndim=None if isinstance(data, Labeled) else ndim,
            overhead=overhead,
            contiguous=contiguous,
            align=chunks_of(out),
            dtype=self.dtype,
            device=self.device,
        )
        for chunk in chunks:
            chunk.store(out, fn(chunk, chunk.data))
        flush(out)
        return out

    def _apply_blocks(
        self,
        data: ArrayLike,
        out: WritableArrayLike | None,
        fn: Callable[[Tensor], Tensor],
        *,
        halo: int | Sequence[int],
        block: Sequence[int | None] | None = None,
        pad: str | None = None,
        contiguous: bool = False,
    ) -> WritableArrayLike:
        """Apply `fn` to blocks with a halo and store the cropped cores.

        `contiguous=True` hands `fn` dense blocks (FFTs, convolutions;
        see [`blocks`][optix.io.blocks]).

        Raises
        ------
        ValueError
            If `out` is (or shares memory with) `data` while the halo is
            nonzero: halos would read already-processed neighbours.
        """
        raw = data.array if isinstance(data, Labeled) else data
        halos = [halo] if isinstance(halo, int) else list(halo)
        target = out.array if isinstance(out, Labeled) else out
        if any(halos) and out is not None and _shares_memory(target, raw):
            raise ValueError(
                "In-place processing with a halo is not possible: use a "
                "separate output."
            )
        out = prepare_out(out, tuple(data.shape), like=raw, dtype=self.dtype)
        chunks = blocks(
            data,
            block,
            halo,
            pad=pad,
            contiguous=contiguous,
            align=chunks_of(out),
            dtype=self.dtype,
            device=self.device,
        )
        for chunk in chunks:
            chunk.store(out, fn(chunk.data)[chunk.inner])
        flush(out)
        return out

    # -- moving and (de)serialisation ------------------------------------
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> Self:
        """Move estimates and any tensor state (in place)."""
        device, dtype = _parse_device(device), _parse_dtype(dtype)
        skip = {"options", "history", "dtype", "device", "layout"}
        for name, value in vars(self).items():
            if name not in skip:
                setattr(self, name, _move(value, device, dtype))
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def _extra_state(self) -> dict[str, object]:
        """Additional tensors to save (override in subclasses)."""
        return {}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        """Restore what `_extra_state` returned."""

    def state_dict(self) -> dict[str, object]:
        """Everything needed to rebuild the fitted estimator."""
        return {
            "class": type(self).__name__,
            "version": __version__,
            "faithful": self.faithful,
            "options": dict(self.options),
            "estimates": {k: e._payload() for k, e in self.estimates.items()},
            "history": asdict(self.history),
            "layout": None if self.layout is None else self.layout.to_dict(),
            "extra": self._extra_state(),
        }

    def load_state_dict(self, state: dict[str, object]) -> Self:
        """Restore a [`state_dict`][optix.Estimator.state_dict] (in place)."""
        self.faithful = state["faithful"]
        self._resolve(self.faithful, state["options"])
        self.estimates = {
            k: Estimate._from_payload(p) for k, p in state["estimates"].items()
        }
        self.history = History(**state["history"])
        layout = state.get("layout")
        self.layout = None if layout is None else Layout(**layout)
        self._load_extra_state(state["extra"])
        return self

    def save(self, path: str | os.PathLike) -> None:
        """Save with `torch.save` (no pickled classes)."""
        torch.save(self.state_dict(), path)

    def __repr__(self) -> str:
        opts = ", ".join(f"{k}={v!r}" for k, v in self.options.items())
        return f"{type(self).__name__}({opts})"


def load(path: str | os.PathLike) -> Estimator | Estimate:
    """Load an estimator or an estimate saved by optix.

    Parameters
    ----------
    path : str or PathLike

    Returns
    -------
    Estimator or Estimate
        Dispatched on the saved class name (`weights_only` loading).
    """
    state = torch.load(path, weights_only=True)
    if "type" in state:
        return Estimate._from_payload(state)
    name = state["class"]
    cls = _REGISTRY.get(name)
    if cls is None:
        raise TypeError(f"Unknown estimator class {name!r}")
    new = cls(faithful=state["faithful"], **state["options"])
    return new.load_state_dict(state)


def estimators() -> dict[str, type[Estimator]]:
    """Registered estimator classes (those defining `tags`), by name."""
    return dict(_REGISTRY)
