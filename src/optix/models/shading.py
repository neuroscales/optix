"""Shading model: smooth flat field, dark image and per-frame baseline."""

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

    flat = exp(f) with f a random combination of the lowest ``order``
    DCT-II functions along each dim (no DC term), scaled so that
    std(f) = ``strength``, then normalised to mean 1.

    Parameters
    ----------
    shape : sequence of int
        Spatial shape.
    order : int
        Number of DCT functions per dim.
    strength : float
        Standard deviation of the log flat field.
    dark : float
        Value of the (constant) dark image, ADU.
    seed : int, optional
    dtype, device : optional
        Working backend.

    Returns
    -------
    ShadingEstimate
        ``flat`` and ``dark`` maps, ``meta["flat_norm"] == "mean"``.
    """
    backend = resolve_backend(dtype, device)
    gen = generator(seed, torch.device("cpu"))
    shape = tuple(shape)
    k = [min(order, n) for n in shape]
    coef = torch.randn(k, generator=gen, dtype=torch.float64)
    coef.view(-1)[0] = 0
    bases = [
        dct_basis(n, kk, dtype=torch.float64, device="cpu")
        for n, kk in zip(shape, k, strict=True)
    ]
    log_flat = reconstruct(coef, bases)
    sd = log_flat.std()
    if sd > 0:
        log_flat = log_flat * (strength / sd)
    flat = log_flat.exp()
    flat = flat / flat.mean()
    return ShadingEstimate(
        flat=flat.to(**backend),
        dark=torch.full(shape, dark, **backend),
        meta={"flat_norm": "mean", "simulated": True},
    )


def simulate_shading(x: Tensor, shading: ShadingEstimate) -> Tensor:
    """Apply a shading model: flat · x + dark + baseline.

    Parameters
    ----------
    x : (*batch, frames, H, W) Tensor
        Unshaded signal.
    shading : ShadingEstimate
        Fields that are None are treated as absent terms (``flat = 1``,
        ``dark = 0``, ``baseline = 0``). ``baseline`` has shape
        ``(*batch, frames)``.

    Returns
    -------
    Tensor
    """
    shading = shading.to(x.device)
    y = x
    if shading.flat is not None:
        y = shading.flat * y
    if shading.dark is not None:
        y = y + shading.dark
    if shading.baseline is not None:
        y = y + shading.baseline[..., None, None]
    return y
