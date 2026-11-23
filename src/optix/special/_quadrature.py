"""Gauss-Hermite and Gauss-Legendre quadrature."""

import functools
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

    Notes
    -----
    `sum(exp(logw) * f(nodes))` ≈ ∫ f(x) exp(−x²) dx, exactly for
    polynomials of degree ≤ 2n − 1. For a Gaussian expectation use the
    change of variables x = μ + √2 σ t::

        E[f(x)] ≈ sum(exp(logw) * f(mu + sqrt(2) * sigma * nodes)) / sqrt(pi)

    for x ~ N(mu, sigma²).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    backend = resolve_backend(dtype, device)
    off = torch.arange(1, n, dtype=torch.float64).div(2).sqrt()
    jacobi = torch.diag(off, 1) + torch.diag(off, -1)
    nodes = torch.linalg.eigvalsh(jacobi)
    # Christoffel weights w_i = 1 / (n p_{n-1}(x_i)^2), with p_k the
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


def _legendre_pair(n: int, x: Tensor) -> tuple[Tensor, Tensor]:
    """Pₙ₋₁(x) and Pₙ(x) by the three-term recurrence (n ≥ 1)."""
    p0, p1 = torch.ones_like(x), x.clone()
    for j in range(2, n + 1):
        p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
    return p0, p1


@functools.lru_cache(maxsize=32)
def _gauss_legendre(n: int) -> tuple[Tensor, Tensor]:
    k = torch.arange(1, n + 1, dtype=torch.float64)
    x = torch.cos(math.pi * (k - 0.25) / (n + 0.5))
    for _ in range(100):
        p0, p1 = _legendre_pair(n, x)
        dx = p1 * (x * x - 1) / (n * (x * p1 - p0))  # Pₙ / Pₙ′
        x = x - dx
        if float(dx.abs().max()) < 1e-15:
            break
    p0, p1 = _legendre_pair(n, x)
    dp = n * (x * p1 - p0) / (x * x - 1)
    w = 2 / ((1 - x * x) * dp * dp)
    return x.flip(0), w.flip(0)


def gauss_legendre(
    n: int,
    *,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Gauss-Legendre nodes and weights on [−1, 1].

    The nodes are the roots of the Legendre polynomial Pₙ, found by
    Newton's method from xᵢ ≈ cos(π (i − ¼) / (n + ½)), with Pₙ and Pₙ₋₁
    evaluated by the three-term recurrence; the weights are
    wᵢ = 2 / ((1 − xᵢ²) Pₙ′(xᵢ)²). This costs O(n²) operations, against
    O(n³) for the eigenvalues of the Jacobi matrix. Everything runs in
    float64 on the CPU (results are cached by `n`); they are then cast
    and moved.

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
    weights : (n,) Tensor
        Quadrature weights.

    Notes
    -----
    `sum(weights * f(nodes))` ≈ ∫₋₁¹ f(x) dx, exactly for polynomials of
    degree ≤ 2n − 1.

    !!! references "References"

        1. Abramowitz M, Stegun IA. *Handbook of Mathematical Functions*,
           formula 25.4.29. National Bureau of Standards, 1964.

        ??? quote "BibTeX"

            ```bibtex
            @book{abramowitz1964handbook,
              author    = {Abramowitz, Milton and Stegun, Irene A.},
              title     = {Handbook of Mathematical Functions},
              publisher = {National Bureau of Standards},
              year      = {1964}
            }
            ```
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    backend = resolve_backend(dtype, device)
    x, w = _gauss_legendre(int(n))
    return x.to(**backend), w.to(**backend)
