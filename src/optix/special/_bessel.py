"""Bessel functions of the first kind of low order, accurate in float64."""

import math

import torch
from torch import Tensor

__all__ = ["bessel_j0", "bessel_j012", "bessel_j1"]

# Below this argument, torch's float64 J₀/J₁ have absolute errors up to
# ~5e-7; above it they agree with double-precision references to ~1e-14.
_SMALL = 25.0
# Points of the trapezoidal rule on the full period: the aliasing error is
# 2 |J₆₄(x)| ≈ 1e-19 for |x| < 25. By symmetry, the quarter period
# carries the 17 distinct values of sin τ.
_POINTS = 64
# Near 0 the rule has a poor relative accuracy for J₁ and J₂ (its absolute
# error is ~1e-17): below these arguments they are taken from their series,
# with these numbers of terms (truncation error < 1e-16 relative).
_SERIES = {1: (1e-8, 1), 2: (1e-2, 3)}
# Elements per block of the rule (bounded memory: 17 × _BLOCK per array).
_BLOCK = 2**15


def _quarter_period(
    dtype: torch.dtype, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """sin τⱼ and the weights of J₀, J₁, J₂ on the quarter period.

    τⱼ = 2πj/N, j = 0 … N/4. The full-period rule, by the symmetries
    τ → π − τ and τ → τ + π, counts j = 0 and j = N/4 twice and the
    others four times, so their weights are 2/N and 4/N.
    """
    j = torch.arange(_POINTS // 4 + 1, dtype=torch.float64)
    tau = 2 * math.pi * j / _POINTS
    w = torch.full_like(tau, 4.0 / _POINTS)
    w[0] = w[-1] = 2.0 / _POINTS
    sin_tau = torch.sin(tau)
    out = (sin_tau, w, w * sin_tau, w * torch.cos(2 * tau))
    return tuple(t.to(dtype=dtype, device=device) for t in out)  # type: ignore[return-value]


def _bessel_small(x: Tensor, orders: tuple[int, ...]) -> list[Tensor]:
    """Jₙ(x) for n in `orders` (subset of 0, 1, 2), 1-D `x`, |x| < 25.

    J₀ = ∑ w cos(x sin τ), J₁ = ∑ w sin τ sin(x sin τ),
    J₂ = ∑ w cos 2τ cos(x sin τ), on the quarter period, in blocks.
    """
    sin_tau, w0, w1, w2 = _quarter_period(x.dtype, x.device)
    weights = {0: w0, 1: w1, 2: w2}
    out = [torch.empty_like(x) for _ in orders]
    for start in range(0, len(x), _BLOCK):
        stop = start + _BLOCK
        arg = x[start:stop, None] * sin_tau
        cos = torch.cos(arg) if 0 in orders or 2 in orders else None
        sin = torch.sin(arg) if 1 in orders else None
        for o, n in zip(out, orders, strict=True):
            o[start:stop] = (sin if n == 1 else cos) @ weights[n]
    for o, n in zip(out, orders, strict=True):
        if n in _SERIES:
            limit, terms = _SERIES[n]
            o.copy_(torch.where(x.abs() < limit, _series(n, x, terms), o))
    return out


def _series(n: int, x: Tensor, terms: int) -> Tensor:
    """∑ₘ (−1)ᵐ (x/2)^(2m+n) / (m! (m+n)!), m < `terms`."""
    h2 = (x / 2) ** 2
    term = (x / 2) ** n / math.factorial(n)
    total = term
    for m in range(1, terms):
        term = -term * h2 / (m * (m + n))
        total = total + term
    return total


def _bessel(x: Tensor, orders: tuple[int, ...]) -> list[Tensor]:
    """Jₙ(x) for n in `orders` ⊆ (0, 1, 2), any shape.

    The rule is used for |x| < 25 in float64, and in every dtype when
    J₂ is requested (its recurrence cancels near 0); otherwise torch.
    """
    x = torch.as_tensor(x)
    if not x.dtype.is_floating_point:
        x = x.to(torch.get_default_dtype())
    shape, flat = x.shape, x.reshape(-1)
    need0 = 0 in orders or 2 in orders
    need1 = 1 in orders or 2 in orders
    j0 = torch.special.bessel_j0(flat) if need0 else None
    j1 = torch.special.bessel_j1(flat) if need1 else None
    out = []
    for n in orders:
        if n == 0:
            out.append(j0)
        elif n == 1:
            out.append(j1)
        else:
            # used only for |x| ≥ 25 below; the rule replaces the rest
            safe = torch.where(
                flat.abs() < _SMALL, torch.ones_like(flat), flat
            )
            out.append(2 * j1 / safe - j0)
    if flat.dtype == torch.float64 or 2 in orders:
        small = flat.abs() < _SMALL
        if bool(small.any()):
            idx = small.nonzero().squeeze(-1)
            values = _bessel_small(flat[idx], orders)
            out = [
                o.index_copy(0, idx, v)
                for o, v in zip(out, values, strict=True)
            ]
    return [o.reshape(shape) for o in out]


def bessel_j0(x: Tensor) -> Tensor:
    """Bessel function of the first kind of order 0, J₀(x).

    In float64 for |x| < 25, J₀ (and J₁, J₂: see
    [`bessel_j012`][optix.special.bessel_j012]) is computed from
    Bessel's integral

        Jₙ(x) = (1/2π) ∫₀^{2π} cos(nτ − x sin τ) dτ

    by the trapezoidal rule on 64 points. The integrand is periodic and
    entire, so the rule converges geometrically: its error is the
    aliased term 2 |J₆₄(x)|, about 10⁻¹⁹ for |x| < 25, and the result is
    accurate to rounding (~10⁻¹⁶), where torch's float64 J₀ and J₁ have
    absolute errors up to ~5·10⁻⁷. By the symmetries τ → π − τ and
    τ → τ + π only the 17 points of a quarter period are distinct, and
    one cos(x sin τ) and one sin(x sin τ) per point give the three
    orders:

        J₀(x) = ∑ⱼ wⱼ cos(x sin τⱼ),
        J₁(x) = ∑ⱼ wⱼ sin τⱼ sin(x sin τⱼ),
        J₂(x) = ∑ⱼ wⱼ cos 2τⱼ cos(x sin τⱼ),

    with τⱼ = 2πj/64, j = 0 … 16, and wⱼ = 1/16 (1/32 at both ends). The
    absolute error of the rule is ~10⁻¹⁷ near 0, a poor relative one for
    J₁ and J₂ there: J₁ = x/2 for |x| < 10⁻⁸, and J₂ is its three-term
    series x²/8 − x⁴/96 + x⁶/3072 for |x| < 10⁻², both exact to
    rounding. The input is processed in blocks, so the memory
    is bounded. Above 25, `torch.special.bessel_j0` and `bessel_j1` are
    used, with J₂ = 2J₁/x − J₀. In float32 (and other floating dtypes),
    `bessel_j0` and `bessel_j1` return the torch functions unchanged
    (their error, ~4·10⁻⁷, is at the level of the dtype), whereas
    `bessel_j012` uses the rule below 25, since the recurrence for J₂
    cancels near 0. Integer input is cast to the default dtype.

    !!! references "References"

        1. Watson GN. *A Treatise on the Theory of Bessel Functions*,
           2nd ed., section 2.2 (Bessel's integral). Cambridge University
           Press, 1944.
        2. Trefethen LN, Weideman JAC. *The exponentially convergent
           trapezoidal rule.* SIAM Rev **56**(3):385–458, 2014.
           [doi:10.1137/130932132](https://doi.org/10.1137/130932132)

        ??? quote "BibTeX"

            ```bibtex
            @book{watson1944treatise,
              author    = {Watson, G. N.},
              title     = {A Treatise on the Theory of {Bessel} Functions},
              edition   = {2},
              publisher = {Cambridge University Press},
              year      = {1944}
            }
            @article{trefethen2014exponentially,
              author  = {Trefethen, Lloyd N. and Weideman, J. A. C.},
              title   = {The exponentially convergent trapezoidal rule},
              journal = {SIAM Review},
              volume  = {56},
              number  = {3},
              pages   = {385--458},
              year    = {2014},
              doi     = {10.1137/130932132}
            }
            ```

    Parameters
    ----------
    x : Tensor
        Real argument.

    Returns
    -------
    Tensor
        J₀(x), same shape, dtype and device.
    """
    return _bessel(x, (0,))[0]


def bessel_j1(x: Tensor) -> Tensor:
    """Bessel function of the first kind of order 1, J₁(x).

    Computed as described in [`bessel_j0`][optix.special.bessel_j0].

    Parameters
    ----------
    x : Tensor
        Real argument.

    Returns
    -------
    Tensor
        J₁(x), same shape, dtype and device.
    """
    return _bessel(x, (1,))[0]


def bessel_j012(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Bessel functions of the first kind J₀(x), J₁(x) and J₂(x).

    Computed together as described in
    [`bessel_j0`][optix.special.bessel_j0]: below |x| = 25, one cosine
    and one sine per node give the three orders, in every floating dtype.

    Parameters
    ----------
    x : Tensor
        Real argument.

    Returns
    -------
    j0, j1, j2 : Tensor
        Same shape, dtype and device as `x`.
    """
    j0, j1, j2 = _bessel(x, (0, 1, 2))
    return j0, j1, j2
