"""Shading model: smooth flat field, dark field and per-frame baseline."""

from collections.abc import Sequence

import torch
from torch import Tensor

from .._config import resolve_backend
from ..basis import dct_basis, reconstruct
from ..estimates import ShadingEstimate
from ._random import generator

__all__ = ["random_shading", "simulate_shading"]


def random_shading(
    shape: Sequence[int],
    *,
    order: int = 3,
    strength: float = 0.3,
    dark: float = 0.0,
    seed: int | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> ShadingEstimate:
    """Draw a smooth random flat field.

    flat = exp(f) with f a random combination of the lowest `order`
    DCT-II functions along each dim (no DC term), scaled so that
    std(f) = `strength`, then normalised to mean 1.

    Parameters
    ----------
    shape : sequence of int
        Spatial shape.
    order : int
        Number of DCT functions per dim.
    strength : float
        Standard deviation of the log flat field.
    dark : float
        Value of the (constant) dark field, ADU.
    seed : int, optional
    dtype, device : optional
        Working backend.

    Returns
    -------
    ShadingEstimate
        `flat` and `dark` maps, `meta["flat_norm"] == "mean"`.
    """
    backend = resolve_backend(dtype, device)
    gen = generator(seed, torch.device("cpu"))
    shape = tuple(shape)
    flat = _smooth_field((), shape, order, strength, gen)
    return ShadingEstimate(
        flat=flat.to(**backend),
        dark=torch.full(shape, dark, **backend),
        meta={"flat_norm": "mean", "simulated": True},
    )


def _smooth_field(
    sample: tuple[int, ...],
    spatial: tuple[int, ...],
    order: int,
    strength: float,
    gen: torch.Generator,
) -> Tensor:
    """Smooth positive fields of mean 1, one per sample (CPU, float64).

    exp(f) / mean(exp(f)), f a random combination of the lowest `order`
    DCT-II functions per spatial dim (no DC term), with std(f) =
    `strength`.
    """
    k = [min(order, n) for n in spatial]
    coef = torch.randn((*sample, *k), generator=gen, dtype=torch.float64)
    coef.view(*sample, -1)[..., 0] = 0
    bases = [
        dct_basis(n, kk, dtype=torch.float64, device="cpu")
        for n, kk in zip(spatial, k, strict=True)
    ]
    log_flat = reconstruct(coef, bases)
    dims = tuple(range(len(sample), log_flat.ndim))
    sd = log_flat.std(dims, keepdim=True)
    log_flat = torch.where(sd > 0, log_flat * (strength / sd), log_flat)
    flat = log_flat.exp()
    return flat / flat.mean(dims, keepdim=True)


def simulate_shading(x: Tensor, shading: ShadingEstimate) -> Tensor:
    """Apply a shading model: y = f ⊙ (x + bₙ) + d.

    The baseline bₙ is in the units of `shading.baseline_units` (see
    [`ShadingEstimate`][optix.estimates.ShadingEstimate]): with
    `"raw"`, the model is y = f · x + d + bₙ.

    Parameters
    ----------
    x : (*batch, frames, H, W) Tensor
        Unshaded signal.
    shading : ShadingEstimate
        Fields that are None are treated as absent terms (`flat = 1`,
        `dark = 0`, `baseline = 0`). `baseline` has shape
        `(*batch, frames)`.

    Returns
    -------
    Tensor
    """
    return shading.to(x.device).shade(x)
