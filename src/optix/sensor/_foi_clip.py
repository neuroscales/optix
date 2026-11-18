"""Clipped-observation functions of Foi et al. (2008), Section IV-VI.

nu ~ N(μ, 1) and ν̃ = max(0, nu) (Eq. 31):

- E_m(μ) = E{ν̃} (Eq. 32) and S_m(μ) = std{ν̃}
  (Eq. 33) have closed forms, evaluated here exactly;
- E_r(ρ) = μ / E{ν̃} and S_r(ρ) = std{ν̃} as
  functions of ρ = E{ν̃} / std{ν̃} (Section IV-B.1b,
  Fig. 9) have none: the paper interpolates them with Eq. (62) and the
  coefficients of Table IV, which are used here;
- the MAD counterparts S_mᵐᵃᵈ, E_rᵐᵃᵈ, S_rᵐᵃᵈ (Section
  VI-C) are only defined numerically (Table III); Eq. (62) with the
  coefficients of Table V is used.

All functions work in float64.
"""

import math

import torch
from torch import Tensor

from ._foi import clipped_normal_moments

__all__ = [
    "E_m",
    "E_r",
    "S_m",
    "S_r",
    "clipped_curve",
    "inverse_init",
]

#: Range of ρ on which the interpolants are accurate (Section VIII-D).
RHO_RANGE = (0.65, 11.0)

# Table IV / V: coefficients p_k of p(t) = ∑ₖ pₖ tᵏ, highest first.
_TABLE_IV = {
    "S_m": (
        1.4308530e-6, 3.2172868e-7, -2.6295693e-5, -8.5123452e-5,
        -1.7851033e-5, 2.0282884e-3, 2.4377832e-2, 3.7234715e-2,
        7.0309281e-1, 1.6923658e-1,
    ),
    "E_r": (
        -5.5320701e-3, -5.5542026e-2, -2.0363415e-1, -3.4651219e-1,
        -4.1222715e-1, -9.1504182e-1, -4.3779025e0, -1.5498697e0,
    ),
    "S_r": (
        2.3348876e-7, -8.7619300e-6, 1.2534052e-4, -7.0907114e-4,
        -1.4943642e-3, 4.5981260e-2, -2.8638716e-1, 8.5412513e-1,
        -1.5725702e0, -5.2653050e-1, -1.2319839e-10,
    ),
}  # fmt: skip
_TABLE_V = {
    "S_m": (
        6.8722511e-4, -3.3132811e-3, 4.6401970e-4, 1.4193996e-2,
        -3.3370736e-3, -4.0537889e-2, 7.8410754e-2, 1.6003810e-2,
        8.3418294e-1, 7.0493620e-2,
    ),
    "E_r": (
        -2.4852997e1, 6.2356790e1, -3.4588152e1, -2.3467375e1,
        2.2066307e1, 1.7531850e0, -5.0041080e0, 1.3236318e-1,
        -4.7818191e0, -1.4931770e0,
    ),
    "S_r": (
        -6.8695634e-1, 1.0659232e1, -7.1938586e1, 2.7680210e2,
        -6.6844401e2, 1.0501179e3, -1.0745079e3, 6.9444029e2,
        -2.6249852e2, 4.5940232e1, -1.3097339e0,
    ),
}  # fmt: skip


def _poly(coef: tuple[float, ...], t: Tensor) -> Tensor:
    out = torch.zeros_like(t)
    for c in coef:
        out = out * t + c
    return out


def E_m(mu: Tensor) -> Tensor:
    """`E{max(0, nu)}`, nu ~ N(μ, 1): Φ(μ) μ + φ(μ).

    Eq. (32).
    """
    mu = torch.as_tensor(mu, dtype=torch.float64)
    m, _ = clipped_normal_moments(mu, torch.ones_like(mu), 0.0, None)
    return m


def S_m(mu: Tensor, robust: bool = False) -> Tensor:
    """`std{max(0, nu)}` (Eq. 33), or S_mᵐᵃᵈ (Section VI-C).

    `S_m` is the exact closed form of Eq. (33) (through the
    cancellation-free [`clipped_normal_moments`][]). S_mᵐᵃᵈ is
    (1 + tanh(p(μ))) / 2 with the coefficients of Table V (Eq. 62).
    """
    mu = torch.as_tensor(mu, dtype=torch.float64)
    if robust:
        return (1 + torch.tanh(_poly(_TABLE_V["S_m"], mu))) / 2
    _, s = clipped_normal_moments(mu, torch.ones_like(mu), 0.0, None)
    return s


def E_r(rho: Tensor, robust: bool = False) -> Tensor:
    """μ / E{ν̃} as a function of ρ, Eq. (62).

    1 − exp(p(log ρ)) with Table IV (or V for the MAD). ρ is
    constrained to `RHO_RANGE`, as in Section VIII-D.
    """
    table = _TABLE_V if robust else _TABLE_IV
    rho = torch.as_tensor(rho, dtype=torch.float64).clamp(*RHO_RANGE)
    return -torch.expm1(_poly(table["E_r"], rho.log()))


def S_r(rho: Tensor, robust: bool = False) -> Tensor:
    """std{ν̃} as a function of ρ, Eq. (62).

    1 − exp(p(ρ)) with Table IV (or V for the MAD); ρ is
    constrained to `RHO_RANGE` (Section VIII-D).
    """
    table = _TABLE_V if robust else _TABLE_IV
    rho = torch.as_tensor(rho, dtype=torch.float64).clamp(*RHO_RANGE)
    return -torch.expm1(_poly(table["S_r"], rho))


def _censored(mu: Tensor, robust: bool, spread: bool) -> tuple:
    """(E_m(μ), S_m(μ)) with one evaluation of Φ and phi."""
    if robust:
        e = E_m(mu)
        return e, (S_m(mu, True) if spread else None)
    # moments about μ (as in clipped_normal_moments): no cancellation
    a = -mu
    pa = torch.special.ndtr(a)
    fa = torch.exp(-0.5 * mu * mu) / math.sqrt(2 * math.pi)
    m = a * pa + fa
    e = mu + m  # Eq. (32)
    if not spread:
        return e, None
    s2 = 1 + a * a * pa - pa + a * fa
    return e, (s2 - m * m).clamp_min(0).sqrt()  # Eq. (33)


def clipped_curve(
    y: Tensor, sigma: Tensor, robust: bool = False, spread: bool = True
) -> tuple[Tensor, Tensor | None]:
    """Expectation and std of the clipped data, Eqs. (44)-(45), (50)-(51).

    ỹ = s E_m(y/s) − y + 1 − s E_m((1-y)/s) and
    σ̃ = s S_m(y/s) S_m((1-y)/s) for data clipped to
    `[0, 1]` (double censoring, under the assumption of Section IV-B
    that a value is not clipped from both sides). With `robust`,
    S_mᵐᵃᵈ replaces `S_m` in σ̃ (Section VI-C.1).

    Parameters
    ----------
    y : Tensor
        Expectation of the unclipped data.
    σ : Tensor
        Its (regularised, positive) standard deviation.
    spread : bool
        Whether to compute σ̃ (else None).
    """
    lo, hi = y / sigma, (1 - y) / sigma
    e0, s0 = _censored(lo, robust, spread)
    e1, s1 = _censored(hi, robust, spread)
    ytil = sigma * e0 - y + 1 - sigma * e1
    stil = sigma * s0 * s1 if spread else None
    return ytil, stil


def inverse_init(
    yhat: Tensor, shat: Tensor, eps: float, robust: bool = False
) -> tuple[Tensor, Tensor]:
    """Inverse transformation of the pairs for the LS initialisation.

    Eqs. (54)-(56) (and (47)-(48)): with
    ρ₀ = √((y/s)² + ε²), `ρ₁ = √(((1-y)/s)² +
    ε²)`, the unclipped expectation is
    y E_r(ρ₀) − y + 1 − (1 − y) E_r(ρ₁) and the unclipped
    standard deviation is s / (S_r(ρ₀) S_r(ρ₁)).

    Returns
    -------
    y, scale : Tensor
        Unclipped expectation and the factor `1 / (S_r(ρ₀)
        S_r(ρ₁))` that multiplies the standard deviation.
    """
    rho0 = torch.sqrt((yhat / shat) ** 2 + eps**2)
    rho1 = torch.sqrt(((1 - yhat) / shat) ** 2 + eps**2)
    y = yhat * E_r(rho0, robust) - yhat + 1
    y = y - (1 - yhat) * E_r(rho1, robust)
    scale = 1 / (S_r(rho0, robust) * S_r(rho1, robust))
    return y, scale


_SQRT_2PI = math.sqrt(2 * math.pi)
