"""Closed-form Kullback-Leibler divergences ``KL(q || p)``."""

import torch
from torch import Tensor


def kl_dirichlet(a: Tensor, b: Tensor) -> Tensor:
    """``KL(Dir(a) || Dir(b))``.

    Parameters
    ----------
    a, b : (..., K) Tensor
        Concentration parameters of ``q`` and ``p`` (last dim =
        categories).

    Returns
    -------
    (...) Tensor
        Divergence, summed over the last dim.
    """
    a0 = a.sum(-1)
    b0 = b.sum(-1)
    elog = torch.digamma(a) - torch.digamma(a0).unsqueeze(-1)
    return (
        torch.lgamma(a0)
        - torch.lgamma(b0)
        - (torch.lgamma(a) - torch.lgamma(b)).sum(-1)
        + ((a - b) * elog).sum(-1)
    )


def kl_gamma(
    shape_q: Tensor, rate_q: Tensor, shape_p: Tensor, rate_p: Tensor
) -> Tensor:
    """``KL(Gamma(shape_q, rate_q) || Gamma(shape_p, rate_p))``, elementwise.

    Parameters
    ----------
    shape_q, rate_q : Tensor
        Shape and rate of ``q``.
    shape_p, rate_p : Tensor
        Shape and rate of ``p``.

    Returns
    -------
    Tensor
        Divergence (broadcast shape of the inputs).
    """
    shape_q, rate_q, shape_p, rate_p = map(
        torch.as_tensor, (shape_q, rate_q, shape_p, rate_p)
    )
    return (
        (shape_q - shape_p) * torch.digamma(shape_q)
        - torch.lgamma(shape_q)
        + torch.lgamma(shape_p)
        + shape_p * (rate_q.log() - rate_p.log())
        + shape_q * (rate_p - rate_q) / rate_q
    )


def kl_normal(
    mean_q: Tensor, var_q: Tensor, mean_p: Tensor, var_p: Tensor
) -> Tensor:
    """``KL(N(mean_q, var_q) || N(mean_p, var_p))``, elementwise.

    Parameters
    ----------
    mean_q, var_q : Tensor
        Mean and variance of ``q``.
    mean_p, var_p : Tensor
        Mean and variance of ``p``.

    Returns
    -------
    Tensor
        Divergence (broadcast shape of the inputs).
    """
    mean_q, var_q, mean_p, var_p = map(
        torch.as_tensor, (mean_q, var_q, mean_p, var_p)
    )
    return 0.5 * (
        var_p.log()
        - var_q.log()
        + (var_q + (mean_q - mean_p).square()) / var_p
        - 1
    )
