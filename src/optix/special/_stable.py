"""Building blocks of cancellation-free Gamma and Dirichlet formulas.

``lgamma(a) = (a - 1/2) log(a) - a + log(2 pi) / 2 + stirling_residual(a)``
and ``digamma(a) = log(a) - log_minus_digamma(a)``: substituting these
into an expression that combines ``lgamma`` / ``digamma`` of large,
nearly equal arguments makes the large terms cancel algebraically, so
only small, accurately computed residuals are left. The remaining
``r - 1 - log(r)`` of a ratio ``r`` close to 1 is :func:`ratio_minus_log`.
"""

import math

import torch
from torch import Tensor

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
#: Above this argument the asymptotic series are used.
_SERIES_FROM = 10.0


def stirling_residual(a: Tensor) -> Tensor:
    """``lgamma(a) - ((a - 1/2) log(a) - a + log(2 pi) / 2)``, elementwise.

    Parameters
    ----------
    a : Tensor
        Positive argument.

    Returns
    -------
    Tensor
        The remainder of Stirling's formula. For ``a > 10`` the series
        ``1/(12a) - 1/(360a^3) + 1/(1260a^5) - 1/(1680a^7)`` is used
        (absolute error below ``1e-12``), which keeps full relative
        precision where the direct difference of ``lgamma`` and its
        approximation would cancel.
    """
    a = torch.as_tensor(a)
    big = a > _SERIES_FROM
    # each branch sees arguments it is finite on (finite gradients)
    s = torch.where(big, a, _SERIES_FROM)
    inv = 1 / s
    inv2 = inv * inv
    series = inv * (
        1 / 12 - inv2 * (1 / 360 - inv2 * (1 / 1260 - inv2 / 1680))
    )
    d = torch.where(big, 1, a)
    direct = torch.lgamma(d) - ((d - 0.5) * d.log() - d + _HALF_LOG_2PI)
    return torch.where(big, series, direct)


def ratio_minus_log(
    num: Tensor, den: Tensor | float = 1.0, diff: Tensor | None = None
) -> Tensor:
    """``y - 1 - log(y)`` with ``y = num / den > 0``, without cancellation.

    Parameters
    ----------
    num, den : Tensor
        Positive numerator and denominator of the ratio ``y``.
    diff : Tensor, optional
        ``num - den``, when the caller has a more accurate value than the
        rounded difference (e.g. a sum of elementwise differences).

    Returns
    -------
    Tensor
        Non-negative, ``~ x^2 / 2`` with ``x = (num - den) / den``. Near
        ``y = 1`` (``|x| < 0.2``), ``x - log1p(x)`` is computed as
        ``2 u^2 / (1 - u) - 2 (atanh(u) - u)`` with ``u = x / (2 + x)``
        and a series for ``atanh(u) - u``; ``x`` is formed from the
        difference, so it keeps its relative precision. Elsewhere the
        direct ``y - 1 - log(y)`` is accurate, and finite for ratios too
        small for ``1 + x`` to be represented.
    """
    num = torch.as_tensor(num)
    den = torch.as_tensor(den, dtype=num.dtype, device=num.device)
    x = ((num - den) if diff is None else diff) / den
    small = x.abs() < 0.2  # |u| < 0.112
    # each branch sees arguments it is finite on (finite gradients)
    xs = torch.where(small, x, 0)
    u = xs / (2 + xs)
    u2 = u * u
    tail = 1 / 15
    for k in (13, 11, 9, 7, 5, 3):
        tail = 1 / k + u2 * tail
    series = 2 * u2 / (1 - u) - 2 * u * u2 * tail
    y = torch.where(small, 2, num / den)
    direct = y - 1 - y.log()
    return torch.where(small, series, direct)
