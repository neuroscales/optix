"""First-order primal–dual algorithm of Chambolle & Pock (2011)."""

import math
from collections.abc import Callable

import torch
from torch import Tensor

from ._functions import Function
from ._operators import LinearOperator
from ._primal_dual import PrimalDualResult
from ._prox import prox_conjugate

__all__ = ["chambolle_pock"]


def _norm(x: Tensor) -> float:
    return float(torch.linalg.vector_norm(x, dtype=torch.float64))


def chambolle_pock(
    op: LinearOperator,
    dual: Function,
    primal: Function,
    x0: Tensor,
    *,
    tau: float | None = None,
    sigma: float | None = None,
    strong_convexity: float = 0.0,
    y0: Tensor | None = None,
    max_iter: int = 1000,
    tol: float | None = 1e-5,
    gap: Callable[[Tensor, Tensor], float] | None = None,
    check_every: int = 1,
    trace: dict | None = None,
    callback: Callable[[int, Tensor], object] | None = None,
) -> PrimalDualResult:
    """Minimise G(x) + F(Kx) by the primal–dual method of Chambolle & Pock.

    F and G are convex with easy proximity operators, K is linear. The
    iteration is Algorithm 1 of Chambolle & Pock (2011) with θ = 1:

        yₖ₊₁ = prox_{σF*}(yₖ + σ K x̄ₖ)
        xₖ₊₁ = prox_{τG}(xₖ − τ Kᵀ yₖ₊₁)
        x̄ₖ₊₁ = xₖ₊₁ + θ(xₖ₊₁ − xₖ)

    with prox_{σF*} from the prox of F by Moreau's identity
    ([`prox_conjugate`][optix.optim.prox_conjugate]). It converges for
    τσ‖K‖² < 1 (Theorem 1), at the ergodic rate O(1/N). When G is
    γ-strongly convex (`strong_convexity` = γ > 0), the steps follow
    Algorithm 2 (θₖ = 1/√(1 + 2γτₖ), τₖ₊₁ = θₖτₖ, σₖ₊₁ = σₖ/θₖ), which
    converges at the rate O(1/N²) (Theorem 2).

    The default steps are τ = σ = 0.99/‖K‖, with ‖K‖² the bound
    `op.norm2` (the paper leaves the split between τ and σ open).
    Iterations stop after `max_iter` iterations, or when the measure of
    convergence is at most `tol`: `gap(x, y)` when given (for example a
    relative primal–dual gap, Section 5), else the relative change
    ‖xₖ₊₁ − xₖ‖₂ / ‖xₖ₊₁‖₂ (not specified in the paper). It is
    evaluated every `check_every` iterations, and at the last one.

    !!! references "References"
        1. Chambolle A, Pock T. *A first-order primal-dual algorithm for
           convex problems with applications to imaging.* J Math Imaging
           Vis **40**(1):120–145, 2011.
           [doi:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)

        ??? quote "BibTeX"
            ```bibtex
            @article{chambolle2011first,
              author  = {Chambolle, Antonin and Pock, Thomas},
              title   = {A first-order primal-dual algorithm for convex
                         problems with applications to imaging},
              journal = {Journal of Mathematical Imaging and Vision},
              volume  = {40}, number = {1}, pages = {120--145},
              year    = {2011}, doi = {10.1007/s10851-010-0251-1}
            }
            ```

    Parameters
    ----------
    op : LinearOperator
        K, mapping `(*N)` to the dual space.
    dual : Function
        F, acting on Kx (only its prox is used).
    primal : Function
        G (only its prox is used).
    x0 : Tensor
        Starting point x₀, `(*N)`; it sets the dtype and device.
    tau, sigma : float, optional
        Primal and dual steps τ, σ > 0 with τσ‖K‖² < 1. Default: both
        0.99/‖K‖.
    strong_convexity : float
        γ ≥ 0, the modulus of strong convexity of G (0: Algorithm 1).
    y0 : Tensor, optional
        Starting dual variable. Default: zeros.
    max_iter : int
        Maximum number of iterations.
    tol : float, optional
        Tolerance on the measure of convergence; None runs `max_iter`
        iterations.
    gap : callable, optional
        `gap(x, y)`, the measure of convergence (default: the relative
        change of x).
    check_every : int
        Iterations between two evaluations of `gap`.
    trace : dict, optional
        Filled with the per-iteration list `"change"` (relative change
        of x) and the list `"gap"` of the evaluated gaps.
    callback : callable, optional
        `callback(k, x)` after each iteration.

    Returns
    -------
    PrimalDualResult
        `x` the last primal iterate, `duals` the dual variable `[y]`,
        `n_iter`, `change` and `residual` (the last evaluated `gap`; inf
        without `gap`).
    """
    norm = math.sqrt(float(op.norm2))
    if not norm > 0:
        raise ValueError("‖K‖ must be positive")
    tau = 0.99 / norm if tau is None else float(tau)
    sigma = 0.99 / norm if sigma is None else float(sigma)
    if not (tau > 0 and sigma > 0):
        raise ValueError("tau and sigma must be positive")
    if tau * sigma * norm * norm >= 1 + 1e-12:
        raise ValueError("tau·sigma·‖K‖² must be below 1")
    gamma = float(strong_convexity)
    if not gamma >= 0:
        raise ValueError("strong_convexity must be >= 0")
    x = x0.clone()
    xbar = x.clone()
    y = op.forward(x).zero_() if y0 is None else y0.to(x).clone()
    check_every = max(1, int(check_every))
    if trace is not None:
        trace["change"] = []
        trace["gap"] = []
    change = math.inf
    measure = math.inf
    n = 0
    while n < max_iter:
        y = prox_conjugate(dual.prox, y + sigma * op.forward(xbar), sigma)
        x_new = primal.prox(x - tau * op.adjoint(y), tau)
        theta = 1.0
        if gamma > 0:
            theta = 1 / math.sqrt(1 + 2 * gamma * tau)
            tau, sigma = theta * tau, sigma / theta
        diff = x_new - x
        xbar = x_new + theta * diff
        n += 1
        size = _norm(x_new)
        change = _norm(diff) / size if size > 0 else _norm(diff)
        x = x_new
        del diff
        if trace is not None:
            trace["change"].append(change)
        if callback is not None:
            callback(n, x)
        if gap is not None and (n % check_every == 0 or n == max_iter):
            measure = float(gap(x, y))
            if trace is not None:
                trace["gap"].append(measure)
            if tol is not None and measure <= tol:
                break
        elif gap is None and tol is not None and change <= tol:
            break
    return PrimalDualResult(x, [y], n, change, measure)
