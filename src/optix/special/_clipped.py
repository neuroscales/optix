"""Moments of a censored (clipped) Gaussian."""

import math

import torch
from torch import Tensor

__all__ = ["clipped_normal_moments"]


def clipped_normal_moments(
    mean: Tensor,
    std: Tensor,
    low: float | None = 0.0,
    high: float | None = 1.0,
) -> tuple[Tensor, Tensor]:
    """Mean and standard deviation of `clip(z, low, high)`.

    The variable is z ~ 𝒩(μ, σ²), with μ = `mean` and σ = `std`. With
    t ~ 𝒩(0, 1), α = (low − μ)/σ and β = (high − μ)/σ,
    clip(z) = μ + σ·clip(t, α, β), and

        m  = E[clip(t)]  = α Φ(α) + β Φ(−β) + φ(α) − φ(β)
        s² = E[clip(t)²] = α² Φ(α) + β² Φ(−β) + Φ(β) − Φ(α)
                           + α φ(α) − β φ(β)
        E[clip(z)]   = μ + σ m
        Var[clip(z)] = σ² (s² − m²)

    !!! note
        Writing the moments about μ in units of σ avoids the cancellation
        of E[z²] − E[z]² when clipping is negligible: then m → 0 and
        s² → 1. Φ(−β) is used instead of 1 − Φ(β). Compute in float64 when
        nearly every sample is clipped, because s² − m² then cancels.

    Parameters
    ----------
    mean, std : Tensor
        Parameters of the unclipped Gaussian (`std > 0`).
    low, high : float or None
        Clipping bounds (None: no clipping on that side).

    Returns
    -------
    mean, std : Tensor
        Moments of the clipped variable.
    """
    mean, std = torch.broadcast_tensors(
        torch.as_tensor(mean), torch.as_tensor(std).to(mean)
    )
    zero = torch.zeros_like(mean)
    ndtr = torch.special.ndtr

    def pdf(x: Tensor) -> Tensor:
        return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    m = zero.clone()
    s2 = torch.ones_like(mean)
    if low is not None:
        a = (low - mean) / std
        pa, fa = ndtr(a), pdf(a)
        m = m + a * pa + fa
        s2 = s2 + a * a * pa - pa + a * fa
    if high is not None:
        b = (high - mean) / std
        qb, fb = ndtr(-b), pdf(b)
        m = m + b * qb - fb
        s2 = s2 + b * b * qb - qb - b * fb
    var = (s2 - m * m).clamp_min(0)
    return mean + std * m, std * var.sqrt()
