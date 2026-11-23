"""Bessel functions of the first kind of low order."""

import math

import torch
from torch import Tensor

__all__ = ["bessel_j012"]

# Trapezoidal rule on Bessel's integral below _BESSEL_SWITCH
# (64 points on the period: error < 1e-15 there), torch's own above it.
_BESSEL_SWITCH = 25.0
_BESSEL_POINTS = 64


def bessel_j012(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Bessel functions of the first kind J₀, J₁ and J₂.

    For |x| < 25 they are computed from Bessel's integral

        Jₙ(x) = (1 / 2π) ∫₀^{2π} cos(n τ − x sin τ) dτ

    with the trapezoidal rule on 64 points. The integrand is periodic
    and entire, so the rule converges exponentially; the aliasing error
    is of order J₆₄(x), below 10⁻¹⁵ for |x| < 25. The bound is on the
    absolute error: near 0 the sum for J₂ ≈ x²/8 cancels, so its relative
    error grows there (J₂(10⁻⁸) has an absolute error of about 10⁻¹⁷).
    For |x| ≥ 25,
    `torch.special.bessel_j0` / `bessel_j1` are used, with
    J₂(x) = 2 J₁(x) / x − J₀(x). Below 25, the torch functions
    themselves have absolute errors up to about 5·10⁻⁷ in float64 (near
    x = 5), which this evaluation avoids.

    Parameters
    ----------
    x : Tensor
        Real argument.

    Returns
    -------
    j0, j1, j2 : Tensor
        Same shape, dtype and device as `x`.
    """
    j0, j1, j2 = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
    small = x.abs() < _BESSEL_SWITCH
    if bool(small.any()):
        xs = x[small]
        m = _BESSEL_POINTS
        # The quarter period j = 0 .. m/4 carries every distinct sin τ;
        # j = 0 and j = m/4 occur twice in the full period, the others four
        # times (by the symmetries τ → π − τ and τ → τ + π).
        j = torch.arange(m // 4 + 1, dtype=torch.float64)
        tau = 2 * math.pi * j / m
        weight = torch.full_like(tau, 4.0 / m)
        weight[0] = weight[-1] = 2.0 / m
        backend = {"dtype": x.dtype, "device": x.device}
        sin_tau = torch.sin(tau).to(**backend)
        cos_2tau = torch.cos(2 * tau).to(**backend)
        weight = weight.to(**backend)
        arg = xs[..., None] * sin_tau
        c, s = torch.cos(arg), torch.sin(arg)
        j0[small] = (c * weight).sum(-1)
        j1[small] = (s * (sin_tau * weight)).sum(-1)
        j2[small] = (c * (cos_2tau * weight)).sum(-1)
    large = ~small
    if bool(large.any()):
        xl = x[large]
        a0 = torch.special.bessel_j0(xl)
        a1 = torch.special.bessel_j1(xl)
        j0[large] = a0
        j1[large] = a1
        j2[large] = 2 * a1 / xl - a0
    return j0, j1, j2
