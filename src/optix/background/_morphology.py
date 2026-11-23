"""Grey-scale morphology with flat and non-flat structuring elements.

A structuring element is given as a tensor `structure` of odd shape
`(2k₁+1, …, 2kₙ+1)` with its origin at the centre: `structure[k + o]`
is the height B(o) of offset o, and −∞ marks offsets outside the
support (the umbra convention of Sternberg 1983). A flat element has
B = 0 on its support.
"""

import itertools
import math
from collections.abc import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "ball",
    "footprint",
    "grey_dilation",
    "grey_erosion",
    "grey_opening",
]


def ball(
    radius: float,
    ndim: int = 2,
    *,
    height: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Umbra of a ball (or ellipsoid), as a non-flat structuring element.

        B(o) = h · √(1 − ‖o‖²/r²) − h   for ‖o‖ ≤ r,   −∞ elsewhere,

    on the integer offsets o ∈ ℤⁿ, with r the radius and h the height
    (h = r for a ball: B(o) = √(r² − ‖o‖²) − r). The heights are shifted
    so that B(0) = 0; an opening does not depend on this shift.

    Parameters
    ----------
    radius : float
        Radius r > 0 along the spatial axes, in pixels.
    ndim : int
        Number of spatial dims n.
    height : float, optional
        Semi-axis h along the intensity axis (default: `radius`, a ball).
    dtype, device : optional
        Of the result (default dtype: float64).

    Returns
    -------
    Tensor
        `(2k+1,)*ndim` with k = ⌊r⌋.
    """
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"radius must be finite and > 0, got {radius!r}")
    height = radius if height is None else float(height)
    k = int(math.floor(radius))
    o = torch.arange(-k, k + 1, dtype=torch.float64, device=device)
    grids = torch.meshgrid(*([o] * ndim), indexing="ij")
    q = sum(g.square() for g in grids) / (radius * radius)
    inside = q <= 1
    b = height * torch.sqrt((1 - q).clamp_min(0)) - height
    b = torch.where(inside, b, torch.full_like(b, -math.inf))
    return b.to(dtype or torch.float64)


def footprint(
    radius: int,
    ndim: int = 2,
    shape: str = "disk",
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Flat structuring element: 0 on its support, −∞ elsewhere.

    Parameters
    ----------
    radius : int
        Half-width k ≥ 0, in pixels.
    ndim : int
        Number of spatial dims n.
    shape : {"disk", "square"}
        `"disk"`: the offsets with ‖o‖ ≤ k (a ball in 3-D);
        `"square"`: the offsets with ‖o‖∞ ≤ k (a cube in 3-D).
    dtype, device : optional
        Of the result (default dtype: float64).

    Returns
    -------
    Tensor
        `(2k+1,)*ndim`.
    """
    if isinstance(radius, bool) or int(radius) != radius or radius < 0:
        raise ValueError(f"radius must be an integer >= 0, got {radius!r}")
    k = int(radius)
    o = torch.arange(-k, k + 1, dtype=torch.float64, device=device)
    grids = torch.meshgrid(*([o] * ndim), indexing="ij")
    if shape == "disk":
        inside = sum(g.square() for g in grids) <= k * k
    elif shape == "square":
        inside = torch.ones_like(grids[0], dtype=torch.bool)
    else:
        raise ValueError(f"shape must be 'disk' or 'square', got {shape!r}")
    b = torch.where(inside, 0.0, -math.inf)
    return b.to(dtype or torch.float64)


def _check_structure(structure: Tensor, data: Tensor) -> tuple[int, ...]:
    ndim = structure.ndim
    if ndim < 1 or data.ndim < ndim:
        raise ValueError(f"structure has {ndim} dims, data only {data.ndim}")
    if any(s % 2 == 0 for s in structure.shape):
        raise ValueError(
            f"structure must have odd sizes, got {tuple(structure.shape)}"
        )
    if not torch.isfinite(structure[tuple(s // 2 for s in structure.shape)]):
        raise ValueError("The origin must belong to the structuring element")
    return tuple(s // 2 for s in structure.shape)


def _pad(x: Tensor, half: Sequence[int], value: float) -> Tensor:
    pad = []
    for k in reversed(half):
        pad += [k, k]
    return F.pad(x, pad, value=value)


def _offsets(structure: Tensor) -> Iterator[tuple[tuple[int, ...], float]]:
    """Offsets of the support and their heights (float64)."""
    half = tuple(s // 2 for s in structure.shape)
    values = structure.detach().to("cpu", torch.float64)
    for idx in itertools.product(*(range(s) for s in structure.shape)):
        h = float(values[idx])
        if h != -math.inf:
            yield tuple(i - k for i, k in zip(idx, half, strict=True)), h


def _is_flat(structure: Tensor) -> bool:
    finite = torch.isfinite(structure)
    return bool((structure[finite] == 0).all()) and bool(
        (structure[~finite] == -math.inf).all()
    )


def _is_box(structure: Tensor) -> bool:
    return bool(torch.isfinite(structure).all())


def _min_filter_1d(x: Tensor, size: int, start: int, n: int) -> Tensor:
    """Running min along the last dim: out[j] = min x[start+j : +size]."""
    shape = x.shape
    y = x.reshape(-1, 1, shape[-1])
    if size > 1:
        y = -F.max_pool1d(-y, size, stride=1)
    y = y[..., start : start + n]
    return y.reshape(*shape[:-1], n)


def _flat_erosion(data: Tensor, structure: Tensor, sign: float) -> Tensor:
    """Erosion (sign 1) or dilation (sign -1) by a flat element.

    Dilation by B is −erosion(−f) by the reflected element; the reflection
    is applied to the offsets here.
    """
    ndim = structure.ndim
    half = _check_structure(structure, data)
    f = data if sign > 0 else -data
    s = structure if sign > 0 else structure.flip(list(range(ndim)))
    if _is_box(s):
        # min over a box is the nested 1-D mins (exact)
        out = f
        for d in range(ndim):
            dim = data.ndim - ndim + d
            k = half[d]
            if k == 0:
                continue
            x = out.movedim(dim, -1)
            n = x.shape[-1]
            x = F.pad(x, [k, k], value=math.inf)
            x = _min_filter_1d(x, 2 * k + 1, 0, n)
            out = x.movedim(-1, dim)
        return out if sign > 0 else -out
    # lines along the last axis: one running min per distinct run, shared
    # by every row of the element that holds it
    inside = torch.isfinite(s).cpu()
    runs: dict[tuple[int, int], list[tuple[int, ...]]] = {}
    for lead in itertools.product(*(range(n) for n in s.shape[:-1])):
        row = inside[lead].tolist()
        j = 0
        while j < len(row):
            if row[j]:
                a = j
                while j < len(row) and row[j]:
                    j += 1
                runs.setdefault((a, j - a), []).append(lead)
            else:
                j += 1
    padded = _pad(f, half, math.inf)
    shape = data.shape[-ndim:]
    batch = [slice(None)] * (data.ndim - ndim)
    out = None
    for (a, size), leads in runs.items():
        pooled = _min_filter_1d(padded, size, a, shape[-1])
        for lead in leads:
            sl = batch + [
                slice(i, i + n) for i, n in zip(lead, shape[:-1], strict=True)
            ]
            x = pooled[tuple(sl)]
            out = x.clone() if out is None else torch.minimum(out, x, out=out)
        del pooled
    return out if sign > 0 else -out


def _non_flat(data: Tensor, structure: Tensor, sign: float) -> Tensor:
    half = _check_structure(structure, data)
    ndim = structure.ndim
    lead = data.ndim - ndim
    shape = data.shape[lead:]
    fill = math.inf if sign > 0 else -math.inf
    padded = _pad(data, half, fill)
    out = torch.full_like(data, fill)
    tmp = torch.empty_like(data)
    op = torch.minimum if sign > 0 else torch.maximum
    for o, h in _offsets(structure):
        # erosion reads f(x + o) − B(o); dilation reads e(x − o) + B(o)
        sl = [slice(None)] * lead
        for oi, k, n in zip(o, half, shape, strict=True):
            start = k + oi if sign > 0 else k - oi
            sl.append(slice(start, start + n))
        x = padded[tuple(sl)]
        if h == 0:
            op(out, x, out=out)
        else:
            torch.add(x, -h if sign > 0 else h, out=tmp)
            op(out, tmp, out=out)
    return out


def grey_erosion(data: Tensor, structure: Tensor) -> Tensor:
    """Grey-scale erosion, Sternberg (1983):

        (f ⊖ B)(x) = min_o [f(x + o) − B(o)],

    over the offsets o of the support of B, on the trailing
    `structure.ndim` dims of `data`. Offsets that fall outside the image
    are left out of the minimum (the image is +∞ outside its domain).

    Parameters
    ----------
    data : Tensor
        `(..., *spatial)`, floating point.
    structure : Tensor
        Heights B, odd shape, origin at the centre, −∞ outside the support
        (see [`ball`][optix.background.utils.ball] and
        [`footprint`][optix.background.utils.footprint]).

    Returns
    -------
    Tensor
        Of the shape of `data`.
    """
    structure = structure.to(data.dtype)
    if _is_flat(structure):
        return _flat_erosion(data, structure, 1.0)
    return _non_flat(data, structure, 1.0)


def grey_dilation(data: Tensor, structure: Tensor) -> Tensor:
    """Grey-scale dilation, Sternberg (1983):

        (f ⊕ B)(x) = max_o [f(x − o) + B(o)],

    over the offsets o of the support of B, on the trailing
    `structure.ndim` dims of `data`. Offsets that fall outside the image
    are left out of the maximum (the image is −∞ outside its domain).

    Parameters
    ----------
    data : Tensor
        `(..., *spatial)`, floating point.
    structure : Tensor
        Heights B, as in [`grey_erosion`][optix.background.utils.grey_erosion].

    Returns
    -------
    Tensor
        Of the shape of `data`.
    """
    structure = structure.to(data.dtype)
    if _is_flat(structure):
        return _flat_erosion(data, structure, -1.0)
    return _non_flat(data, structure, -1.0)


def grey_opening(data: Tensor, structure: Tensor) -> Tensor:
    """Grey-scale opening, f_B = (f ⊖ B) ⊕ B (Sternberg 1983).

    The opening is the upper surface of the union of all translates of the
    umbra of B that fit under the umbra of f: it is the surface swept by
    B "rolled" under the image. Its influence radius is twice the
    half-size of B.

    Parameters
    ----------
    data : Tensor
        `(..., *spatial)`, floating point.
    structure : Tensor
        Heights B, as in [`grey_erosion`][optix.background.utils.grey_erosion].

    Returns
    -------
    Tensor
        Of the shape of `data`.
    """
    return grey_dilation(grey_erosion(data, structure), structure)
