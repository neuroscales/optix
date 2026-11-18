"""Probability distributions for variational Bayes and conjugate updates.

Two layers:

- [`functional`][optix.distributions.functional]: pure tensor functions (means,
  expectations, entropies, log-densities, KL divergences), one canonical
  signature per family, safe to inline into compiled kernels;
- classes ([`Gamma`][optix.distributions.Gamma],
  [`Dirichlet`][optix.distributions.Dirichlet],
  [`Categorical`][optix.distributions.Categorical],
  [`Normal`][optix.distributions.Normal]) that accept several
  parameterisations, store the one they were given, derive the others on
  demand, and dispatch [`kl_divergence`][optix.distributions.kl_divergence]
  and [`posterior`][optix.distributions.posterior].

Accumulated sufficient statistics
([`SuffStats`][optix.distributions.SuffStats] subclasses) are mergeable
containers consumed by [`posterior`][optix.distributions.posterior] and
the ``fit`` classmethods.
"""

from . import (
    _posterior,  # noqa: F401  (registers the updates)
    functional,
)
from ._base import Distribution, kl_divergence, posterior
from ._categorical import Categorical, CategoricalExpectations
from ._dirichlet import Dirichlet, DirichletExpectations
from ._gamma import Gamma, GammaExpectations
from ._normal import Normal, NormalExpectations
from ._stats import (
    CategoricalStats,
    DirichletStats,
    GammaStats,
    NormalStats,
    SuffStats,
)

__all__ = [
    "Categorical",
    "CategoricalExpectations",
    "CategoricalStats",
    "Dirichlet",
    "DirichletExpectations",
    "DirichletStats",
    "Distribution",
    "Gamma",
    "GammaExpectations",
    "GammaStats",
    "Normal",
    "NormalExpectations",
    "NormalStats",
    "SuffStats",
    "functional",
    "kl_divergence",
    "posterior",
]
