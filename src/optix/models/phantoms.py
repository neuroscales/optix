"""Numeric phantoms: ground-truth specimens x for the simulators.

Each function draws a latent image x ≥ 0, in expected photoelectrons
per pixel (e⁻): the `signal` of
[`simulate_sensor`][optix.models.simulate_sensor], the unshaded x of
[`simulate_shading`][optix.models.simulate_shading], the object of
[`simulate_blur`][optix.models.simulate_blur]. A scene is a short
composition, for example

    x = phantoms.cells((40, 64, 64), seed=0) + 10
    y = simulate_sensor(simulate_shading(x, shading), sensor, seed=1)

`shape` is `(*sample, *spatial)`: the trailing `ndim` dims (2 by
default, 3 for volumes) are the grid Ω, and every leading index (a
frame n = 1…N, say) is an independent draw. Positions are in pixels,
pixel i at the integer coordinates of its index, as the centre `K // 2`
of [`gaussian_psf`][optix.models.gaussian_psf]. Random parameters are
drawn on the CPU in float64 from `seed`, so a seed gives the same
phantom on every device.
"""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .._config import Backend, resolve_backend
from ._random import generator
from .shading import _smooth_field

__all__ = ["background", "beads", "cells", "filaments", "ramp", "stripes"]

_F64 = torch.float64


def _split(shape: Sequence[int], ndim: int) -> tuple[tuple, tuple]:
    """`(sample, spatial)` parts of `shape`."""
    shape = tuple(int(s) for s in shape)
    if not 1 <= ndim <= len(shape):
        raise ValueError(f"Cannot have {ndim} spatial dims in {shape}")
    return shape[: len(shape) - ndim], shape[len(shape) - ndim :]


def _per_axis(value: float | Sequence[float], ndim: int) -> Tensor:
    """One positive value per spatial axis."""
    value = torch.as_tensor(value, dtype=_F64).flatten()
    if value.numel() not in (1, ndim) or (value <= 0).any():
        raise ValueError("expected positive values, one or one per axis")
    return value.expand(ndim)


def _gaussians(
    centres: Tensor,
    weights: Tensor,
    sigma: Tensor,
    spatial: tuple,
    backend: Backend,
    chunk: int = 256,
) -> Tensor:
    """∑ₖ wₖ 𝒩(r; cₖ, diag σ²) on the grid: (B, P, D) → (B, *spatial)."""
    out = torch.zeros(centres.shape[0], *spatial, **backend)
    axes = "abcdefgh"[: len(spatial)]
    expr = ",".join(f"np{a}" for a in axes) + "->n" + axes
    norm = (2 * math.pi) ** 0.5 * sigma
    for start in range(0, centres.shape[1], chunk):
        c = centres[:, start : start + chunk].to(**backend)
        factors = []
        for a, n in enumerate(spatial):
            r = torch.arange(n, **backend) - c[..., a, None]
            s = float(sigma[a])
            factors.append(torch.exp(-0.5 * (r / s) ** 2) / float(norm[a]))
        w = weights[:, start : start + chunk].to(**backend)
        factors[0] = factors[0] * w[..., None]
        out += torch.einsum(expr, *factors)
    return out


def cells(
    shape: Sequence[int],
    count: int = 10,
    *,
    radius: float = 6.0,
    intensity: float = 100.0,
    variation: float = 0.2,
    edge: float = 0.5,
    ndim: int = 2,
    seed: int | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Fluorescent cells: bright ellipses (ellipsoids in 3D) on black.

    Each cell k has a uniform centre cₖ in the field, a random
    orientation Rₖ, semi-axes aₖ = radius · exp(0.2 ξ) and a brightness
    βₖ = intensity · exp(variation · ξ), ξ ~ 𝒩(0, 1). With
    ρₖ(r) = ‖diag(aₖ)⁻¹ Rₖᵀ (r − cₖ)‖ and āₖ the mean semi-axis,

        xᵢ = maxₖ βₖ · sigmoid((1 − ρₖ(rᵢ)) āₖ / edge),

    flat-topped objects with soft boundaries; overlapping cells keep the
    brighter one. Add a constant for autofluorescence.

    Parameters
    ----------
    shape : sequence of int
        `(*sample, *spatial)` shape of x.
    count : int
        Number of cells per sample.
    radius : float
        Typical semi-axis, in pixels.
    intensity, variation : float
        Median brightness (e⁻) and log-normal spread of the cells.
    edge : float
        Width of the soft boundary, in pixels.
    ndim : int
        Number of trailing spatial dims.
    seed : int, optional
    dtype, device : optional
        Working backend.

    Returns
    -------
    Tensor
        x, of `shape`, in [0, max βₖ].
    """
    backend = resolve_backend(dtype, device)
    sample, spatial = _split(shape, ndim)
    if radius <= 0 or edge <= 0:
        raise ValueError("radius and edge must be positive")
    gen = generator(seed, torch.device("cpu"))
    b = math.prod(sample)
    size = torch.tensor(spatial, dtype=_F64)
    centre = torch.rand(b, count, ndim, generator=gen, dtype=_F64) * size
    rot, _ = torch.linalg.qr(
        torch.randn(b, count, ndim, ndim, generator=gen, dtype=_F64)
    )
    axes = torch.randn(b, count, ndim, generator=gen, dtype=_F64)
    axes = radius * (0.2 * axes).exp()
    bright = torch.randn(b, count, generator=gen, dtype=_F64)
    bright = intensity * (variation * bright).exp()
    # rᵢ ↦ diag(a)⁻¹ Rᵀ rᵢ, as a row-vector product with R diag(a)⁻¹
    warp = (rot / axes[..., None, :]).to(**backend)
    sharp = (axes.mean(-1) / edge).to(**backend)
    grid = torch.stack(
        torch.meshgrid(
            *[torch.arange(n, **backend) for n in spatial], indexing="ij"
        ),
        -1,
    ).reshape(-1, ndim)
    x = torch.zeros(b, grid.shape[0], **backend)
    centre, bright = centre.to(**backend), bright.to(**backend)
    for k in range(count):
        rho = ((grid - centre[:, k, None]) @ warp[:, k]).norm(dim=-1)
        cell = torch.sigmoid((1 - rho) * sharp[:, k, None])
        x = torch.maximum(x, bright[:, k, None] * cell)
    return x.reshape(*sample, *spatial)


def beads(
    shape: Sequence[int],
    count: int = 40,
    *,
    sigma: float | Sequence[float] = 1.0,
    brightness: float = 1000.0,
    separation: float = 0.0,
    margin: float | None = None,
    ndim: int = 2,
    seed: int | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Point-like beads at sub-pixel positions.

    Bead k at cₖ is a sampled Gaussian of total brightness β:

        xᵢ = ∑ₖ β 𝒩(rᵢ; cₖ, diag σ²),

    the image of a sub-resolution bead through a Gaussian PSF h of
    standard deviation σ (FWHM ≈ 2.355 σ). For a point source to blur
    with another PSF, keep σ small (about 0.5) and use
    [`simulate_blur`][optix.models.simulate_blur]. Large σ give blobs.

    Parameters
    ----------
    shape : sequence of int
        `(*sample, *spatial)` shape of x.
    count : int
        Number of beads per sample.
    sigma : float or sequence of float
        Standard deviation σ in pixels, one value or one per axis.
    brightness : float
        Expected electrons β of one bead, summed over its pixels (sums
        below β when σ < 0.5 or near the border).
    separation : float
        Minimum distance between bead centres, in pixels.
    margin : float, optional
        Least distance of the centres to the border, in pixels.
        Default: 3σ along each axis.
    ndim : int
        Number of trailing spatial dims.
    seed : int, optional
    dtype, device : optional
        Working backend.

    Returns
    -------
    x : Tensor
        Image of `shape`.
    positions : (*sample, count, ndim) Tensor
        Bead centres cₖ, in pixels.

    Raises
    ------
    ValueError
        If `count` beads do not fit `separation` apart.
    """
    backend = resolve_backend(dtype, device)
    sample, spatial = _split(shape, ndim)
    gen = generator(seed, torch.device("cpu"))
    sigma = _per_axis(sigma, ndim)
    low = 3 * sigma if margin is None else torch.full_like(sigma, margin)
    high = torch.tensor(spatial, dtype=_F64) - 1 - low
    if (high < low).any():
        raise ValueError(f"the margin leaves no room in {spatial}")
    pos = torch.stack(
        [
            _place(gen, count, low, high, separation)
            for _ in range(math.prod(sample))
        ]
    )
    weight = torch.full(pos.shape[:2], float(brightness), dtype=_F64)
    x = _gaussians(pos, weight, sigma, spatial, backend)
    return (
        x.reshape(*sample, *spatial),
        pos.reshape(*sample, count, ndim).to(**backend),
    )


def _place(
    gen: torch.Generator,
    count: int,
    low: Tensor,
    high: Tensor,
    separation: float,
    tries: int = 100,
) -> Tensor:
    """`count` uniform points in [low, high], `separation` apart."""

    def draw() -> Tensor:
        u = torch.rand(count, len(low), generator=gen, dtype=_F64)
        return low + (high - low) * u

    if separation <= 0:
        return draw()
    kept = torch.empty(0, len(low), dtype=_F64)
    for _ in range(tries):
        for c in draw():
            if len(kept) == count:
                return kept
            if not len(kept) or (kept - c).norm(dim=1).min() >= separation:
                kept = torch.cat([kept, c[None]])
    if len(kept) == count:
        return kept
    raise ValueError(f"cannot place {count} beads {separation} pixels apart")


def filaments(
    shape: Sequence[int],
    count: int = 10,
    *,
    width: float = 1.0,
    intensity: float = 100.0,
    length: float | None = None,
    curvature: float = 0.1,
    ndim: int = 2,
    seed: int | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Filaments: thin curved fibres, fine detail for deconvolution.

    Each filament is a smooth random curve, a walk from a uniform start
    whose unit direction d turns at every step δ:

        d ← (d + curvature √δ ξ) / ‖·‖,  ξ ~ 𝒩(0, I),

    so that the persistence length is of the order of 1 / curvature²
    pixels. The curve is drawn as a tube with a Gaussian profile of
    standard deviation `width` and a peak of about `intensity` (a
    straight filament); crossings add up.

    Parameters
    ----------
    shape : sequence of int
        `(*sample, *spatial)` shape of x.
    count : int
        Number of filaments per sample.
    width : float
        Standard deviation of the cross-section, in pixels.
    intensity : float
        Peak value (e⁻) of an isolated filament.
    length : float, optional
        Length of each filament, in pixels. Default: the largest
        spatial size.
    curvature : float
        Change of direction over one pixel, in radians (rms).
    ndim : int
        Number of trailing spatial dims.
    seed : int, optional
    dtype, device : optional
        Working backend.

    Returns
    -------
    Tensor
        x, of `shape`.
    """
    backend = resolve_backend(dtype, device)
    sample, spatial = _split(shape, ndim)
    gen = generator(seed, torch.device("cpu"))
    sigma = _per_axis(width, 1).expand(ndim)
    b = math.prod(sample)
    length = max(spatial) if length is None else float(length)
    step = min(0.5, float(sigma[0]) / 2)
    steps = max(1, math.ceil(length / step))
    size = torch.tensor(spatial, dtype=_F64)
    point = torch.rand(b, count, ndim, generator=gen, dtype=_F64) * size
    d = torch.randn(b, count, ndim, generator=gen, dtype=_F64)
    d = d / d.norm(dim=-1, keepdim=True)
    path = [point]
    for _ in range(steps):
        xi = torch.randn(b, count, ndim, generator=gen, dtype=_F64)
        d = d + curvature * step**0.5 * xi
        d = d / d.norm(dim=-1, keepdim=True)
        point = point + step * d
        path.append(point)
    path = torch.stack(path, 2).reshape(b, -1, ndim)
    # line density giving a peak `intensity` across a straight tube
    density = intensity * (2 * math.pi * float(sigma[0]) ** 2) ** (
        (ndim - 1) / 2
    )
    weight = torch.full(path.shape[:2], step * density, dtype=_F64)
    x = _gaussians(path, weight, sigma, spatial, backend)
    return x.reshape(*sample, *spatial)


def stripes(
    shape: Sequence[int],
    *,
    axis: int = -2,
    strength: float = 5.0,
    fraction: float = 1.0,
    ndim: int = 2,
    seed: int | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Stripe artefact s, constant along one axis.

    Every line along `axis` (a column of a frame, for the default
    `axis=-2`) carries one offset, 𝒩(0, strength²) on a random
    `fraction` of the lines and 0 on the others, as left by uneven
    detector rows or by the shadows of a light sheet. The destriping
    estimators model the recording as y = x + s; for multiplicative
    shadows, use x ⊙ exp(s) with a small `strength`.

    Parameters
    ----------
    shape : sequence of int
        `(*sample, *spatial)` shape of s.
    axis : int
        Spatial axis the stripes run along, as `axis` of the
        destriping estimators (negative values count from the last).
    strength : float
        Standard deviation of the offsets, in the units of x.
    fraction : float
        Fraction of the lines that carry a stripe, in [0, 1].
    ndim : int
        Number of trailing spatial dims.
    seed : int, optional
    dtype, device : optional
        Working backend.

    Returns
    -------
    Tensor
        s, of `shape`, constant along `axis`.
    """
    backend = resolve_backend(dtype, device)
    sample, spatial = _split(shape, ndim)
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is not one of {ndim} spatial dims")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    gen = generator(seed, torch.device("cpu"))
    lines = list(spatial)
    lines[axis] = 1
    lines = (*sample, *lines)
    offset = strength * torch.randn(lines, generator=gen, dtype=_F64)
    keep = torch.rand(lines, generator=gen, dtype=_F64) < fraction
    s = torch.where(keep, offset, 0.0)
    return s.expand(*sample, *spatial).to(**backend).contiguous()


def background(
    shape: Sequence[int],
    *,
    level: float = 20.0,
    variation: float = 0.3,
    order: int = 3,
    ndim: int = 2,
    seed: int | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Smooth background: haze, autofluorescence, out-of-focus light.

    x = level · exp(f) / mean(exp(f)), with f a random combination of
    the lowest `order` DCT-II functions per spatial dim (no constant
    term) scaled to std(f) = `variation`: the same smooth field as the
    flat of [`random_shading`][optix.models.random_shading], but part of
    the specimen.

    Parameters
    ----------
    shape : sequence of int
        `(*sample, *spatial)` shape of x.
    level : float
        Mean of x over Ω (e⁻), for every sample.
    variation : float
        Standard deviation of log x.
    order : int
        Number of DCT functions per dim: larger is less smooth.
    ndim : int
        Number of trailing spatial dims.
    seed : int, optional
    dtype, device : optional
        Working backend.

    Returns
    -------
    Tensor
        x > 0, of `shape`.
    """
    backend = resolve_backend(dtype, device)
    sample, spatial = _split(shape, ndim)
    gen = generator(seed, torch.device("cpu"))
    field = _smooth_field(sample, spatial, order, variation, gen)
    return (level * field).to(**backend)


def ramp(
    shape: Sequence[int],
    *,
    low: float = 0.0,
    high: float = 1000.0,
    axis: int = -1,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Linear ramp from `low` to `high` along one axis.

    A target that covers a range of intensities in a single picture,
    for noise calibration (the photon-transfer and variance-stabilising
    pages) or a known illumination gradient.

    Parameters
    ----------
    shape : sequence of int
        Shape of x, sample dims included.
    low, high : float
        Values (e⁻) at the first and last index of `axis`.
    axis : int
        Axis of the ramp.
    dtype, device : optional
        Working backend.

    Returns
    -------
    Tensor
        x, of `shape`, constant across `axis`.
    """
    backend = resolve_backend(dtype, device)
    shape = tuple(int(s) for s in shape)
    view = [1] * len(shape)
    view[axis] = shape[axis]
    r = torch.linspace(low, high, shape[axis], **backend).reshape(view)
    return r.expand(shape).contiguous()
