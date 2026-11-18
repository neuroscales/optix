"""Special functions: quadrature, Newton solvers and KL divergences."""

from ._kl import kl_dirichlet, kl_gamma, kl_normal
from ._quadrature import gauss_hermite
from ._solvers import (
    dirichlet_elog,
    fit_dirichlet,
    fit_gamma_shape,
    inv_digamma,
    log_minus_digamma,
)

__all__ = [
    "dirichlet_elog",
    "fit_dirichlet",
    "fit_gamma_shape",
    "gauss_hermite",
    "inv_digamma",
    "kl_dirichlet",
    "kl_gamma",
    "kl_normal",
    "log_minus_digamma",
]
