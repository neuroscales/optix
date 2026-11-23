"""Bessel functions of the first kind, accurate in float64."""

import math

import torch
from torch import Tensor

# Below this argument, torch's float64 J₀/J₁ have absolute errors up to
# ~5e-7; above it they agree with double-precision references to ~1e-14.
_SMALL = 25.0
# Points of the periodic trapezoidal rule: the aliasing error is
# 2 |J_{N−n}(x)| < 1e-40 for x < 25.
_POINTS = 96
_BLOCK = 2**15


def _bessel_trapezoid(order: int, x: Tensor) -> Tensor:
    """J_n(x) = (1/2π) ∫₀^{2π} cos(nθ − x sin θ) dθ, trapezoidal rule."""
    theta = torch.arange(_POINTS, dtype=x.dtype, device=x.device)
    theta = theta * (2 * math.pi / _POINTS)
    sin, n_theta = torch.sin(theta), order * theta
    out = torch.empty_like(x)
    for start in range(0, len(x), _BLOCK):
        xs = x[start : start + _BLOCK].unsqueeze(-1)
        out[start : start + _BLOCK] = torch.cos(n_theta - xs * sin).mean(-1)
    return out


def _bessel(order: int, x: Tensor) -> Tensor:
    x = torch.as_tensor(x)
    if not x.dtype.is_floating_point:
        x = x.to(torch.get_default_dtype())
    fn = torch.special.bessel_j0 if order == 0 else torch.special.bessel_j1
    out = fn(x)
    if x.dtype != torch.float64:
        return out
    small = x.abs() < _SMALL
    if bool(small.any()):
        out = out.clone()
        out[small] = _bessel_trapezoid(order, x[small])
    return out


def bessel_j0(x: Tensor) -> Tensor:
    """Bessel function of the first kind of order 0, J₀(x).

    Same as `torch.special.bessel_j0`, except in float64 for |x| < 25,
    where it is computed from Bessel's integral

        Jₙ(x) = (1/2π) ∫₀^{2π} cos(nθ − x sin θ) dθ

    by the trapezoidal rule with 96 points. The integrand is periodic and
    analytic, so the rule converges geometrically: its error is the
    aliased term 2 |J₉₆(x)|, below 10⁻⁴⁰ there, and the result is
    accurate to rounding (~10⁻¹⁶), where torch's float64 J₀ has errors up
    to ~5·10⁻⁷. Other dtypes go to torch unchanged.

    Parameters
    ----------
    x : Tensor
        Argument.

    Returns
    -------
    Tensor
        J₀(x), same shape and dtype.

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
    """
    return _bessel(0, x)


def bessel_j1(x: Tensor) -> Tensor:
    """Bessel function of the first kind of order 1, J₁(x).

    Same as `torch.special.bessel_j1`, except in float64 for |x| < 25,
    where it is computed accurately from Bessel's integral (see
    [`bessel_j0`][optix.special.bessel_j0]).

    Parameters
    ----------
    x : Tensor
        Argument.

    Returns
    -------
    Tensor
        J₁(x), same shape and dtype.

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
    """
    return _bessel(1, x)
