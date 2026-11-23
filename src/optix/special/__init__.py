"""Special functions: quadrature, Newton solvers, KL divergences and
censored-Gaussian moments."""

from ._clipped import clipped_normal_moments
from ._kl import kl_dirichlet, kl_gamma, kl_normal
from ._poisson_normal import poisson_normal_logpdf, poisson_normal_mean
from ._quadrature import gauss_hermite
from ._solvers import (
    dirichlet_elog,
    fit_dirichlet,
    fit_gamma_shape,
    inv_digamma,
    log_minus_digamma,
)
from ._stable import ratio_minus_log, stirling_residual

__all__ = [
    "clipped_normal_moments",
    "dirichlet_elog",
    "fit_dirichlet",
    "fit_gamma_shape",
    "gauss_hermite",
    "inv_digamma",
    "kl_dirichlet",
    "kl_gamma",
    "kl_normal",
    "log_minus_digamma",
    "poisson_normal_logpdf",
    "poisson_normal_mean",
    "ratio_minus_log",
    "stirling_residual",
]
