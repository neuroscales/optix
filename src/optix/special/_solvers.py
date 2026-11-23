"""Batched Newton solvers for Gamma and Dirichlet maximum likelihood.

All solvers run a fixed number of iterations with no data-dependent
control flow, so they are batched, compile-friendly and never synchronise
with the host.
"""

import torch
from torch import Tensor

_EULER = 0.5772156649015329


def log_minus_digamma(a: Tensor) -> Tensor:
    """log a − ψ(a), elementwise, without cancellation.

    Parameters
    ----------
    a : Tensor
        Positive argument.

    Returns
    -------
    Tensor
        log a − ψ(a). For a > 10 the asymptotic series
        1/(2a) + 1/(12a²) − 1/(120a⁴) + 1/(252a⁶) is used, which keeps
        full relative precision where the direct difference loses its
        digits.
    """
    a = torch.as_tensor(a)
    big = a > 10
    # each branch sees arguments it is finite on (finite gradients)
    inv = 1 / torch.where(big, a, 10)
    inv2 = inv * inv
    series = inv * (0.5 + inv * (1 / 12 - inv2 * (1 / 120 - inv2 / 252)))
    d = torch.where(big, 1, a)
    direct = d.log() - torch.digamma(d)
    return torch.where(big, series, direct)


def _inv_minus_trigamma(a: Tensor) -> Tensor:
    """1/a − ψ′(a), using the asymptotic series for large a."""
    inv = 1 / a
    inv2 = inv * inv
    series = -inv2 * (0.5 + inv * (1 / 6 - inv2 * (1 / 30 - inv2 / 42)))
    direct = inv - torch.polygamma(1, a)
    return torch.where(a > 10, series, direct)


def fit_gamma_shape(c: Tensor, niter: int = 16) -> Tensor:
    """Solve log ν − ψ(ν) = c for ν > 0, elementwise.

    This is the maximum-likelihood equation of the Gamma shape, with
    c = log(mean(x)) − mean(log x) > 0 (the Jensen gap).

    !!! note
        Uses Minka's initialisation
        ν = (3 − c + √((c − 3)² + 24c)) / (12c) and Newton's method on 1/ν.
        For ν > 10 the residual and its derivative use asymptotic series,
        which avoids the cancellation in log ν − ψ(ν).

    !!! references "References"
        1. Minka T. *Estimating a Gamma distribution.* 2002.

        ??? quote "BibTeX"
            ```bibtex
            @misc{minka2002gamma,
              author = {Minka, T.},
              title = {Estimating a Gamma distribution},
              year = {2002}
            }
            ```

    Parameters
    ----------
    c : Tensor
        Right-hand side. Values ≤ 0 (which round-off can produce) are
        clamped to ε / 2, which caps ν at about 1 / ε, where ε is the
        machine epsilon of the dtype. Values above 1 / ε (including
        `inf`) are clamped to 1 / ε, which keeps ν ≳ ε and the result
        finite.
    niter : int
        Number of Newton iterations.

    Returns
    -------
    Tensor
        Shape parameter ν, same shape and dtype as `c`.
    """
    c = torch.as_tensor(c)
    if not c.is_floating_point():
        c = c.to(torch.get_default_dtype())
    eps = torch.finfo(c.dtype).eps
    c = c.clamp(min=eps / 2, max=1 / eps)
    nu = (3 - c + ((c - 3).square() + 24 * c).sqrt()) / (12 * c)
    cap = 1 / eps
    for _ in range(niter):
        nu = nu.clamp(min=torch.finfo(c.dtype).tiny, max=cap)
        num = log_minus_digamma(nu) - c
        den = nu.square() * _inv_minus_trigamma(nu)
        nu = 1 / (1 / nu + num / den)
    return nu.clamp(min=torch.finfo(c.dtype).tiny, max=cap)


def inv_digamma(y: Tensor, niter: int = 5) -> Tensor:
    """Inverse of the digamma function, elementwise.

    !!! note
        Minka's initialisation: exp(y) + ½ for y ≥ −2.22, else
        −1 / (y + γ) with γ the Euler-Mascheroni constant; then Newton's
        method.

    Parameters
    ----------
    y : Tensor
        Values of ψ(x).
    niter : int
        Number of Newton iterations.

    Returns
    -------
    Tensor
        x > 0 such that ψ(x) = `y`.
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
        E[log π] = ψ(α) − ψ(α₀) with α₀ = ∑ₖ αₖ, computed as
        log(α / α₀) − lmd(α) + lmd(α₀) with lmd =
        [`log_minus_digamma`][optix.special.log_minus_digamma], which does
        not cancel for large concentrations.
    """
    a0 = alpha.sum(-1, keepdim=True)
    return (
        (alpha / a0).log() - log_minus_digamma(alpha) + log_minus_digamma(a0)
    )


def fit_dirichlet(
    elogpi: Tensor,
    alpha: Tensor,
    niter: int = 16,
    *,
    bounds: tuple[float, float] | None = None,
) -> Tensor:
    """Dirichlet maximum likelihood given expected log-probabilities.

    Maximises log Γ(∑ₖ aₖ) − ∑ₖ log Γ(aₖ) + ∑ₖ (aₖ − 1) `elogpi`ₖ over a > 0
    with Newton's method, using the diagonal-plus-rank-one structure of
    the Hessian.

    !!! note
        Whenever a Newton step would make any component of a vector
        non-positive, that vector takes the fixed-point update
        a = ψ⁻¹(ψ(∑ₖ aₖ) + `elogpi`) instead, which always stays positive
        (and is also taken when the Newton step is not finite).

        Convergence is only local: Newton's method needs a warm start (for
        example the previous estimate) to converge in a few iterations, and
        from a poor initial guess it may need many iterations or settle on
        the slower fixed-point steps.

    !!! references "References"
        1. Minka T. *Estimating a Dirichlet distribution.* 2000.

        ??? quote "BibTeX"
            ```bibtex
            @misc{minka2000dirichlet,
              author = {Minka, T.},
              title = {Estimating a Dirichlet distribution},
              year = {2000}
            }
            ```

    Parameters
    ----------
    elogpi : (..., K) Tensor
        Mean log-probabilities (sufficient statistics), last dim =
        categories.
    alpha : (..., K) Tensor
        Initial concentration parameters (broadcast against `elogpi`).
    niter : int
        Number of iterations.
    bounds : (float, float), optional
        Box constraint (lo, hi) on every concentration: the start and
        each iterate are clamped to it (projected Newton). It keeps the
        concentrations finite when the maximum lies at infinity (for
        example, when every observation is the same point of the
        simplex).

    Returns
    -------
    (..., K) Tensor
        Fitted concentration parameters (strictly positive).
    """
    elogpi, alpha = torch.broadcast_tensors(elogpi, alpha)
    alpha = alpha.to(elogpi.dtype)
    tiny = torch.finfo(alpha.dtype).tiny
    lo, hi = (tiny, None) if bounds is None else bounds
    alpha = alpha.clamp(lo, hi)
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
        # A float32 cancellation can make the Newton step non-finite.
        ok = (torch.isfinite(newton) & (newton > 0)).all(-1, keepdim=True)
        fixed = inv_digamma(psi0 + elogpi)
        alpha = torch.where(ok, newton, fixed).clamp(lo, hi)
    return alpha
