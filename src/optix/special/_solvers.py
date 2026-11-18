"""Batched Newton solvers for Gamma and Dirichlet maximum likelihood.

All solvers run a fixed number of iterations with no data-dependent
control flow, so they are batched, compile-friendly and never synchronise
with the host.
"""

import torch
from torch import Tensor

_EULER = 0.5772156649015329


def _log_minus_digamma(a: Tensor) -> Tensor:
    """``log(a) - digamma(a)``, using the asymptotic series for large a."""
    inv = 1 / a
    inv2 = inv * inv
    series = inv * (0.5 + inv * (1 / 12 - inv2 * (1 / 120 - inv2 / 252)))
    direct = a.log() - torch.digamma(a)
    return torch.where(a > 10, series, direct)


def _inv_minus_trigamma(a: Tensor) -> Tensor:
    """``1/a - trigamma(a)``, using the asymptotic series for large a."""
    inv = 1 / a
    inv2 = inv * inv
    series = -inv2 * (0.5 + inv * (1 / 6 - inv2 * (1 / 30 - inv2 / 42)))
    direct = inv - torch.polygamma(1, a)
    return torch.where(a > 10, series, direct)


def fit_gamma_shape(c: Tensor, niter: int = 16) -> Tensor:
    """Solve ``log(nu) - digamma(nu) = c`` for ``nu > 0``, elementwise.

    This is the maximum-likelihood equation of the Gamma shape, with
    ``c = log(mean(x)) - mean(log(x)) > 0`` (the Jensen gap).

    Parameters
    ----------
    c : Tensor
        Right-hand side. Values ``<= 0`` (which round-off can produce) are
        clamped to ``eps / 2``, which caps ``nu`` at about ``1 / eps`` of
        the dtype.
    niter : int
        Number of Newton iterations.

    Returns
    -------
    Tensor
        Shape parameter ``nu``, same shape and dtype as ``c``.

    Notes
    -----
    Uses Minka's initialisation
    ``nu = (3 - c + sqrt((c - 3)^2 + 24 c)) / (12 c)`` and Newton's method
    on ``1 / nu`` (Minka, "Estimating a Gamma distribution", 2002). For
    ``nu > 10`` the residual and its derivative use asymptotic series, which
    avoids the cancellation in ``log(nu) - digamma(nu)``.
    """
    c = torch.as_tensor(c)
    if not c.is_floating_point():
        c = c.to(torch.get_default_dtype())
    eps = torch.finfo(c.dtype).eps
    c = c.clamp_min(eps / 2)
    nu = (3 - c + ((c - 3).square() + 24 * c).sqrt()) / (12 * c)
    cap = 1 / eps
    for _ in range(niter):
        nu = nu.clamp(min=torch.finfo(c.dtype).tiny, max=cap)
        num = _log_minus_digamma(nu) - c
        den = nu.square() * _inv_minus_trigamma(nu)
        nu = 1 / (1 / nu + num / den)
    return nu.clamp(min=torch.finfo(c.dtype).tiny, max=cap)


def inv_digamma(y: Tensor, niter: int = 5) -> Tensor:
    """Inverse of the digamma function, elementwise.

    Parameters
    ----------
    y : Tensor
        Values of ``digamma(x)``.
    niter : int
        Number of Newton iterations.

    Returns
    -------
    Tensor
        ``x > 0`` such that ``digamma(x) == y``.

    Notes
    -----
    Minka's initialisation: ``exp(y) + 1/2`` for ``y >= -2.22``, else
    ``-1 / (y + euler_gamma)``; then Newton's method.
    """
    y = torch.as_tensor(y)
    if not y.is_floating_point():
        y = y.to(torch.get_default_dtype())
    tiny = torch.finfo(y.dtype).tiny
    x = torch.where(y >= -2.22, y.exp() + 0.5, -1 / (y + _EULER))
    for _ in range(niter):
        x = x - (torch.digamma(x) - y) / torch.polygamma(1, x)
        x = x.clamp_min(tiny)
    return x


def dirichlet_elog(alpha: Tensor) -> Tensor:
    """Expected log-probabilities under a Dirichlet.

    Parameters
    ----------
    alpha : (..., K) Tensor
        Concentration parameters (last dim = categories).

    Returns
    -------
    (..., K) Tensor
        ``E[log pi] = digamma(alpha) - digamma(sum(alpha))``.
    """
    return torch.digamma(alpha) - torch.digamma(alpha.sum(-1, keepdim=True))


def fit_dirichlet(elogpi: Tensor, alpha: Tensor, niter: int = 16) -> Tensor:
    """Dirichlet maximum likelihood given expected log-probabilities.

    Maximises ``log Gamma(sum a) - sum log Gamma(a) + sum (a - 1) elogpi``
    over ``a > 0`` with Newton's method, using the diagonal-plus-rank-one
    structure of the Hessian (Minka, "Estimating a Dirichlet
    distribution", 2000).

    Parameters
    ----------
    elogpi : (..., K) Tensor
        Mean log-probabilities (sufficient statistics), last dim =
        categories.
    alpha : (..., K) Tensor
        Initial concentration parameters (broadcast against ``elogpi``).
    niter : int
        Number of iterations.

    Returns
    -------
    (..., K) Tensor
        Fitted concentration parameters (strictly positive).

    Notes
    -----
    Whenever a Newton step would make any component of a vector
    non-positive, that vector takes the fixed-point update
    ``a = inv_digamma(digamma(sum a) + elogpi)`` instead, which always
    stays positive.
    """
    elogpi, alpha = torch.broadcast_tensors(elogpi, alpha)
    alpha = alpha.to(elogpi.dtype)
    tiny = torch.finfo(alpha.dtype).tiny
    for _ in range(niter):
        alpha0 = alpha.sum(-1, keepdim=True)
        psi0 = torch.digamma(alpha0)
        grad = psi0 - torch.digamma(alpha) + elogpi
        q = -torch.polygamma(1, alpha)
        z = torch.polygamma(1, alpha0)
        b = (grad / q).sum(-1, keepdim=True) / (
            1 / z + (1 / q).sum(-1, keepdim=True)
        )
        newton = alpha - (grad - b) / q
        ok = (newton > 0).all(-1, keepdim=True)
        fixed = inv_digamma(psi0 + elogpi)
        alpha = torch.where(ok, newton, fixed).clamp_min(tiny)
    return alpha
