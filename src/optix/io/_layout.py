"""Axis roles of an array: which axes are spatial, samples or groups."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ._arrays import ArrayLike, Index

__all__ = ["Labeled", "Layout", "labeled"]

Axis = int | str
Role = Axis | Sequence[Axis]


def _is_int(x: object) -> bool:
    return isinstance(x, int | np.integer) and not isinstance(x, bool)


@dataclass(frozen=True)
class Layout:
    """Role of each axis of an array: spatial, sample or group.

    * **spatial** axes span one picture (a frame). Chunks and fitted
      parameter maps hold them **in the order given here**, which need not
      be the storage order.
    * **sample** axes index pictures that share the same parameters
      (frames, time points, z planes, ...). Iterators flatten them, in
      storage order, into one leading axis of length `n`.
    * **group** axes index independent fits (e.g. one flat field per
      channel). Parameter maps are shaped `(*group, *spatial)`, with the
      group axes in the order given here.

    Every axis has exactly one role; axes that are not named default to
    `sample`.

    Parameters
    ----------
    axes : str or sequence of str, optional
        Axis labels, in storage order: a string of single-character names
        (`"tczyx"`) or a sequence of names. Without labels, roles are
        given by (possibly negative) positions.
    spatial : str, int or sequence of str / int
        Spatial axes, in the order wanted for chunks and maps. A string is
        split into characters when every label is a single character
        (`"yx"`), else it names one axis.
    sample : str, int or sequence of str / int, optional
        Sample axes (default: every axis without a role).
    group : str, int or sequence of str / int, optional
        Group axes (default: none).

    Raises
    ------
    ValueError
        For an empty `spatial`, unknown or repeated names, positions out
        of range, or an axis given two roles.

    Examples
    --------
    Light-sheet stack stored as `(z, y, x)`, where each `x` is one
    picture of the `(y, z)` plane: chunks are `(n, Y, Z)`.

    >>> layout = Layout("zyx", spatial="yz", sample="x")
    >>> layout.shape_of((40, 30, 500))
    ((), (500,), (30, 40))

    OME-NGFF `(t, c, z, y, x)` with one flat field per channel, shared
    over `t` and `z`: parameter maps are `(C, Y, X)`.

    >>> Layout("tczyx", spatial="yx", group="c").shape_of((5, 3, 7, 64, 64))
    ((3,), (5, 7), (64, 64))

    The default, `(*samples, *spatial)` with two trailing spatial axes:

    >>> Layout.default(4, 2)
    Layout(axes=None, spatial=(-2, -1), sample=None, group=())

    Notes
    -----
    A per-axis `spacing` (voxel size) may be added later.
    """

    axes: tuple[str, ...] | None = None
    spatial: tuple[Axis, ...] = ()
    sample: tuple[Axis, ...] | None = None
    group: tuple[Axis, ...] = ()
    _allow_empty: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        axes = self.axes
        if axes is not None:
            axes = tuple(axes)
            if not all(isinstance(a, str) and a for a in axes):
                raise ValueError(f"Axis labels must be strings: {axes}")
            if len(set(axes)) != len(axes):
                raise ValueError(f"Repeated axis labels in {axes}")
        split = axes is None or all(len(a) == 1 for a in axes)

        def norm(role: object, name: str) -> tuple[Axis, ...]:
            if _is_int(role):
                return (int(role),)
            if isinstance(role, str):
                items = tuple(role) if split else (role,)
            else:
                try:
                    items = tuple(role)
                except TypeError:
                    raise TypeError(
                        f"{name} must be a str, an int or a sequence of "
                        f"them, got {role!r}"
                    ) from None
            out = []
            for item in items:
                if _is_int(item):
                    out.append(int(item))
                elif isinstance(item, str):
                    if axes is None:
                        raise ValueError(
                            f"{name} names axis {item!r}, but the layout "
                            f"has no `axes` labels"
                        )
                    if item not in axes:
                        raise ValueError(
                            f"Unknown axis {item!r} in {name}; axes are {axes}"
                        )
                    out.append(item)
                else:
                    raise TypeError(f"Bad axis {item!r} in {name}")
            return tuple(out)

        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "spatial", norm(self.spatial, "spatial"))
        if self.sample is not None:
            object.__setattr__(self, "sample", norm(self.sample, "sample"))
        object.__setattr__(self, "group", norm(self.group, "group"))
        if not self.spatial and not self._allow_empty:
            raise ValueError("A layout needs at least one spatial axis")
        if axes is not None:
            self.positions(len(axes))
        else:
            # what can be checked without knowing the number of dims
            for name in ("spatial", "sample", "group"):
                items = getattr(self, name) or ()
                if len(set(items)) != len(items):
                    raise ValueError(f"Repeated axis in {name}: {items}")

    @classmethod
    def default(cls, ndim_total: int, ndim_spatial: int) -> "Layout":
        """The `(*samples, *spatial)` convention.

        Parameters
        ----------
        ndim_total : int
            Number of dims of the data.
        ndim_spatial : int
            Number of trailing spatial dims; the others are samples.

        Returns
        -------
        Layout
            Trailing spatial axes given by negative positions, so that it
            applies to data of any number of dims.
        """
        if not 0 <= ndim_spatial <= ndim_total:
            raise ValueError(
                f"Cannot have {ndim_spatial} spatial dims in {ndim_total}D "
                f"data"
            )
        return cls(
            spatial=tuple(range(-ndim_spatial, 0)),
            _allow_empty=True,
        )

    # -- resolution --------------------------------------------------------
    def _position(self, axis: Axis, ndim: int, name: str) -> int:
        if isinstance(axis, str):
            return self.axes.index(axis)
        if not -ndim <= axis < ndim:
            raise ValueError(
                f"{name} axis {axis} is out of range for {ndim}D data"
            )
        return axis % ndim

    def positions(
        self, ndim: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Storage positions of the group, sample and spatial axes.

        Parameters
        ----------
        ndim : int
            Number of dims of the data.

        Returns
        -------
        group, sample, spatial : tuple of int
            Group and spatial axes in layout order, sample axes in storage
            order.

        Raises
        ------
        ValueError
            If the labels do not match `ndim`, a position is out of range
            or an axis has two roles.
        """
        if self.axes is not None and len(self.axes) != ndim:
            raise ValueError(
                f"Layout has {len(self.axes)} axes {self.axes}, data have "
                f"{ndim} dims"
            )
        roles = {}
        out = {}
        for name in ("group", "sample", "spatial"):
            items = getattr(self, name) or ()
            pos = []
            for axis in items:
                p = self._position(axis, ndim, name)
                if p in roles:
                    label = self.axes[p] if self.axes else p
                    raise ValueError(
                        f"Axis {label!r} is given twice (as {roles[p]} and "
                        f"{name})"
                    )
                roles[p] = name
                pos.append(p)
            out[name] = tuple(pos)
        rest = [p for p in range(ndim) if p not in roles]
        sample = tuple(sorted(out["sample"] + tuple(rest)))
        return out["group"], sample, out["spatial"]

    def shape_of(
        self, array: ArrayLike | Sequence[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Group, sample and spatial shapes of an array (or a shape).

        Parameters
        ----------
        array : array-like or sequence of int

        Returns
        -------
        group_shape, sample_shape, spatial_shape : tuple of int
            In the orders of [`positions`][optix.io.Layout.positions].
        """
        shape = tuple(
            int(s)
            for s in getattr(array, "shape", array)  # type: ignore[union-attr]
        )
        group, sample, spatial = self.positions(len(shape))
        return (
            tuple(shape[p] for p in group),
            tuple(shape[p] for p in sample),
            tuple(shape[p] for p in spatial),
        )

    def permutation(self, ndim: int) -> tuple[int, ...]:
        """Storage axes in `(*group, *sample, *spatial)` order.

        `x.permute(layout.permutation(x.ndim))` is the canonical view.
        """
        group, sample, spatial = self.positions(ndim)
        return group + sample + spatial

    def inverse_permutation(self, ndim: int) -> tuple[int, ...]:
        """Inverse of [`permutation`][optix.io.Layout.permutation].

        Maps the canonical order back to storage.
        """
        perm = self.permutation(ndim)
        inverse = [0] * ndim
        for i, p in enumerate(perm):
            inverse[p] = i
        return tuple(inverse)

    def map_shape(self, array: ArrayLike | Sequence[int]) -> tuple[int, ...]:
        """Shape `(*group, *spatial)` of a fitted parameter map."""
        group, _, spatial = self.shape_of(array)
        return group + spatial

    def to_dict(self) -> dict[str, object]:
        """Plain dict (for `state_dict`); `Layout(**d)` rebuilds it."""
        return {
            "axes": self.axes,
            "spatial": self.spatial,
            "sample": self.sample,
            "group": self.group,
        }


class Labeled:
    """An array-like bundled with its [`Layout`][optix.io.Layout].

    The wrapped array is never loaded. Built by
    [`labeled`][optix.io.labeled]. Indexing, `shape` and every other
    attribute (`dtype`, `chunks`, ...) pass through to the wrapped
    array.

    Attributes
    ----------
    array : array-like
        The wrapped array.
    layout : Layout
    """

    def __init__(self, array: ArrayLike, layout: Layout) -> None:
        """Wrap `array` and check that `layout` fits its dims.

        Parameters
        ----------
        array : array-like
            The array to label.
        layout : Layout
            Axis roles of `array`.
        """
        layout.positions(len(array.shape))
        self.array = array
        self.layout = layout

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.array.shape)

    @property
    def ndim(self) -> int:
        return len(self.array.shape)

    def __getitem__(self, index: Index) -> object:
        return self.array[index]

    def __setitem__(self, index: Index, value: object) -> None:
        self.array[index] = value

    def __getattr__(self, name: str) -> object:
        if name in ("array", "layout"):  # not yet set (unpickling)
            raise AttributeError(name)
        return getattr(self.array, name)

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return (
            f"Labeled({type(self.array).__name__}, shape={self.shape}, "
            f"{self.layout!r})"
        )


def labeled(
    data: ArrayLike,
    layout: Layout | None = None,
    *,
    axes: str | Sequence[str] | None = None,
    spatial: Role | None = None,
    sample: Role | None = None,
    group: Role | None = None,
) -> Labeled:
    """Attach a [`Layout`][optix.io.Layout] to an array-like.

    The array is not loaded.

    Parameters
    ----------
    data : array-like
        Tensor, ndarray, memmap, zarr, ... (or a [`Labeled`][optix.io.Labeled],
        which is re-labelled).
    layout : Layout, optional
        The layout. Mutually exclusive with the role keywords.
    axes, spatial, sample, group : optional
        Build the layout in place (see [`Layout`][optix.io.Layout]).

    Returns
    -------
    Labeled

    Examples
    --------
    >>> import numpy as np
    >>> data = labeled(np.zeros((40, 30, 500)), axes="zyx", spatial="yz")
    >>> data.shape, data.layout.shape_of(data)[2]
    ((40, 30, 500), (30, 40))
    """
    roles = {
        k: v
        for k, v in dict(
            axes=axes, spatial=spatial, sample=sample, group=group
        ).items()
        if v is not None
    }
    if layout is not None and roles:
        raise TypeError("Give either `layout` or role keywords, not both")
    if layout is None:
        if "spatial" not in roles:
            raise TypeError("labeled() needs a `layout` or `spatial` axes")
        roles.setdefault("group", ())
        layout = Layout(**roles)
    if isinstance(data, Labeled):
        data = data.array
    return Labeled(data, layout)


def bind(
    data: ArrayLike, layout: Layout | None = None, ndim: int | None = None
) -> tuple[ArrayLike, Layout | None]:
    """Unwrap a `Labeled` and reconcile it with `layout` / `ndim`.

    Returns the raw array and the explicit layout (None when neither `data`
    nor `layout` carries one: the caller then uses the default).
    """
    if isinstance(data, Labeled):
        if layout is not None and layout != data.layout:
            raise ValueError(
                f"layout={layout!r} conflicts with the layout of the "
                f"labelled array, {data.layout!r}"
            )
        layout, data = data.layout, data.array
    if layout is not None:
        if not isinstance(layout, Layout):
            raise TypeError(f"layout must be a Layout, got {layout!r}")
        if ndim is not None and ndim != len(layout.spatial):
            raise ValueError(
                f"ndim={ndim} conflicts with the {len(layout.spatial)} "
                f"spatial axes of the layout"
            )
        layout.positions(len(data.shape))
    return data, layout


def nsamples(data: ArrayLike, layout: Layout) -> int:
    """Number of samples (product of the sample shape)."""
    return math.prod(layout.shape_of(data)[1])
