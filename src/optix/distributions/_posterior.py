"""Conjugate updates registered with :func:`posterior`."""

import torch
from torch import Tensor

from ._base import _register_posterior
from ._dirichlet import Dirichlet
from ._gamma import Gamma
from ._normal import Normal
from ._stats import CategoricalStats, GammaStats, NormalStats


def _floor(x: Tensor) -> Tensor:
    """Keep positive parameters positive (improper prior, no data)."""
    return x.clamp_min(torch.finfo(x.dtype).tiny)


def _known(value: object, like: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


@_register_posterior(Dirichlet, CategoricalStats)
def _dirichlet_categorical(
    prior: Dirichlet, stats: CategoricalStats
) -> Dirichlet:
    """``alpha_N = alpha_0 + counts``."""
    (alpha,) = prior.params()
    (counts,) = stats.astuple()
    return Dirichlet._new(_floor(alpha + counts.to(device=alpha.device)))


@_register_posterior(Gamma, GammaStats)
def _gamma_rate(prior: Gamma, stats: GammaStats, *, shape: Tensor) -> Gamma:
    """Rate of a Gamma likelihood with known ``shape``.

    ``a_N = a_0 + shape * n`` and ``b_N = b_0 + sum_x``.
    """
    a, b = prior.params()
    n, sx, _ = (t.to(device=a.device) for t in stats.astuple())
    shape = _known(shape, a)
    return Gamma._new(_floor(a + shape * n), _floor(b + sx))


@_register_posterior(Gamma, NormalStats)
def _gamma_precision(
    prior: Gamma, stats: NormalStats, *, mean: Tensor
) -> Gamma:
    """Precision of a Normal likelihood with known ``mean``.

    ``a_N = a_0 + n / 2`` and
    ``b_N = b_0 + (scatter + n (xbar - mean)^2) / 2``.
    """
    a, b = prior.params()
    n, xbar, scatter = (t.to(device=a.device) for t in stats.astuple())
    mean = _known(mean, a)
    quad = scatter + n * (xbar - mean).square()
    return Gamma._new(_floor(a + 0.5 * n), _floor(b + 0.5 * quad))


@_register_posterior(Normal, NormalStats)
def _normal_mean(
    prior: Normal, stats: NormalStats, *, precision: Tensor
) -> Normal:
    """Mean of a Normal likelihood with known ``precision`` ``P``.

    ``lam_N = lam_0 + n P`` and
    ``m_N = (lam_0 m_0 + n P xbar) / lam_N`` (``m_0`` where ``lam_N = 0``).
    """
    m0, lam0 = prior.params()
    n, xbar, _ = (t.to(device=m0.device) for t in stats.astuple())
    precision = _known(precision, m0)
    lam = lam0 + n * precision
    ok = lam > 0
    safe = torch.where(ok, lam, torch.ones_like(lam))
    mean = torch.where(ok, (lam0 * m0 + n * precision * xbar) / safe, m0)
    return Normal._new(mean, lam)
