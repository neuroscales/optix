"""Gauss-Hermite quadrature."""

import math

import torch
from torch import Tensor

from .._config import resolve_backend


def gauss_hermite(
    n: int,
    *,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Gauss-Hermite nodes and log-weights for the weight exp(−x²).

    Computed with the Golub-Welsch algorithm: the nodes are the eigenvalues
    of the symmetric Jacobi matrix of the (physicists') Hermite
    polynomials, and the weights are √π · v₀² where v₀ holds the first
    components of the normalised eigenvectors. Here the (equivalent)
    Christoffel formula wᵢ = 1 / (n pₙ₋₁(xᵢ)²) is used, with pₖ the
    orthonormal Hermite polynomials, because tiny
    eigenvector components are not accurate in floating point. Everything
    runs in float64 on the CPU; the results are then cast and moved.

    !!! note
        `sum(exp(logw) * f(nodes))` ≈ ∫ f(x) exp(−x²) dx, exactly for
        polynomials of degree ≤ 2n − 1. For a Gaussian expectation use the
        change of variables x = μ + √2 σ t::

            E[f(x)] ≈ sum(exp(logw) * f(mu + sqrt(2) * sigma * nodes))
                      / sqrt(pi)

        for x ~ N(mu, sigma²).

    Parameters
    ----------
    n : int
        Number of nodes (≥ 1).
    dtype, device : optional
        Output dtype and device (default: from the configuration).

    Returns
    -------
    nodes : (n,) Tensor
        Quadrature nodes, increasing.
    logw : (n,) Tensor
        Natural logarithm of the weights. Log-weights are returned so that
        large `n` does not underflow.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    backend = resolve_backend(dtype, device)
    off = torch.arange(1, n, dtype=torch.float64).div(2).sqrt()
    jacobi = torch.diag(off, 1) + torch.diag(off, -1)
    nodes = torch.linalg.eigvalsh(jacobi)
    # Christoffel weights wᵢ = 1 / (n pₙ₋₁(xᵢ)²), with pₖ the
    # orthonormal Hermite polynomials, evaluated with a rescaled three-term
    # recurrence (log-scale tracked separately so nothing over/underflows).
    prev = torch.zeros_like(nodes)
    cur = torch.full_like(nodes, math.pi**-0.25)
    logscale = torch.zeros_like(nodes)
    for k in range(n - 1):
        nxt = math.sqrt(2 / (k + 1)) * nodes * cur
        nxt = nxt - math.sqrt(k / (k + 1)) * prev
        prev, cur = cur, nxt
        scale = torch.maximum(cur.abs(), prev.abs())
        prev, cur = prev / scale, cur / scale
        logscale = logscale + scale.log()
    logw = -math.log(n) - 2 * (cur.abs().log() + logscale)
    return nodes.to(**backend), logw.to(**backend)
