"""Primal–dual splitting with a smooth term (Combettes & Pesquet 2012)."""

import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from ._functions import Function
from ._operators import LinearOperator
from ._prox import prox_conjugate

__all__ = ["PrimalDualResult", "primal_dual", "primal_dual_step"]


class PrimalDualResult(NamedTuple):
    """Result of [`primal_dual`][optix.optim.primal_dual].

    Attributes
    ----------
    x : Tensor
        The primal estimate p₁ of the last iteration, `(*N)`: it lies in
        the domain of ψ₀ (for an indicator function, in the constraint
        set) and has the same limit as the iterates xₖ.
    duals : list of Tensor
        Dual variables v_r, `(C_r, *N)`.
    n_iter : int
        Number of iterations.
    change : float
        Last relative change ‖xₖ₊₁ − xₖ‖₂ / ‖xₖ‖₂ (inf if not measured).
    residual : float
        Last ‖xₖ − p₁,ₖ‖₂ / γ, the norm of the forward–backward residual;
        it tends to 0 at a solution.
    """

    x: Tensor
    duals: list[Tensor]
    n_iter: int
    change: float
    residual: float = math.inf


def _norm(x: Tensor) -> float:
    return float(torch.linalg.vector_norm(x, dtype=torch.float64))


def primal_dual_step(
    lipschitz: float,
    norms2: Sequence[float],
    *,
    eps: float | None = None,
) -> float:
    """Largest step γ of the primal–dual iteration (Combettes & Pesquet 2012).

    With δ = μ + √(∑_r ‖V_r‖²), μ the Lipschitz constant of ∇h,
    convergence holds for γ ∈ [ε, (1 − ε)/δ] with ε ∈ (0, 1/(δ + 1))
    (Chouzenoux et al. 2015, Theorem 3.2 iii). The step returned is
    γ = (1 − ε)/δ. The default ε = min(10⁻³, ½/(δ + 1)) is not
    specified in the paper, which takes "the maximum possible value".

    Parameters
    ----------
    lipschitz : float
        μ ≥ 0.
    norms2 : sequence of float
        Upper bounds on ‖V_r‖².
    eps : float, optional
        ε ∈ (0, 1/(δ + 1)).

    Returns
    -------
    float
        γ; 0 when μ is infinite.
    """
    delta = float(lipschitz) + math.sqrt(sum(float(n) for n in norms2))
    if not delta > 0:
        raise ValueError("lipschitz + ‖V‖ must be positive")
    if not math.isfinite(delta):
        return 0.0
    bound = 1 / (delta + 1)
    if eps is None:
        eps = min(1e-3, bound / 2)
    if not 0 < eps < bound:
        raise ValueError(f"eps must be in (0, {bound:g}), got {eps}")
    return (1 - eps) / delta


def primal_dual(
    grad: Callable[[Tensor], Tensor],
    terms: Sequence[tuple[LinearOperator, Function]],
    x0: Tensor,
    *,
    step: float,
    constraint: Function | None = None,
    duals: Sequence[Tensor] | None = None,
    max_iter: int = 5000,
    tol: float | None = 1e-5,
    value: Callable[[Tensor], Tensor | float] | None = None,
    trace: dict | None = None,
    callback: Callable[[int, Tensor], object] | None = None,
) -> PrimalDualResult:
    """Minimise h(x) + ψ₀(x) + ∑_r ψ_r(V_r x) by primal–dual splitting.

    h is convex and differentiable with a μ-Lipschitz gradient; ψ₀ and
    the ψ_r are convex with easy proximity operators; the V_r are
    linear. The iteration is Algorithm 1 of Chouzenoux et al. (2015),
    an instance of the forward–backward–forward primal–dual method of
    Combettes & Pesquet (2012), without the error terms cₖ, eₖ:

        y₁ = xₖ − γ(∇h(xₖ) + ∑_r V_rᵀ v_r)
        p₁ = prox_{γψ₀}(y₁)
        for each r:
            y₂ = v_r + γ V_r xₖ
            p₂ = y₂ − γ prox_{ψ_r/γ}(y₂/γ)
            q₂ = p₂ + γ V_r p₁
            v_r ← v_r − y₂ + q₂
        q₁ = p₁ − γ(∇h(p₁) + ∑_r V_rᵀ p₂)
        xₖ₊₁ = xₖ − y₁ + q₁

    p₂ is the prox of γψ_r* (Moreau's identity; for ψ_r = λ‖·‖₂,₁ it is
    the projection of y₂ onto the ℓ₂ balls of radius λ). It needs no
    matrix inversion, and converges for any step γ in the interval of
    [`primal_dual_step`][optix.optim.primal_dual_step] (Theorem 3.2).

    Iterations stop when ‖xₖ₊₁ − xₖ‖₂ ≤ `tol`·‖xₖ‖₂ or after `max_iter`
    iterations (the paper stops "when the difference in norm between two
    consecutive iterates becomes small"; the normaliser is not
    specified). The p₁ of the last iteration is returned: xₖ and p₁,ₖ
    have the same limit (Theorem 3.2), and p₁ lies in the domain of ψ₀.

    !!! references "References"
        1. Chouzenoux E, Jezierska A, Pesquet J-C, Talbot H. *A convex
           approach for image restoration with exact Poisson–Gaussian
           likelihood.* SIAM J Imaging Sci **8**(4):2662-2682, 2015.
           [doi:10.1137/15M1014395](https://doi.org/10.1137/15M1014395)
        2. Combettes PL, Pesquet J-C. *Primal-dual splitting algorithm
           for solving inclusions with mixtures of composite, Lipschitzian,
           and parallel-sum type monotone operators.* Set-Valued Var Anal
           **20**(2):307-330, 2012.
           [doi:10.1007/s11228-011-0191-y](https://doi.org/10.1007/s11228-011-0191-y)

        ??? quote "BibTeX"
            ```bibtex
            @article{chouzenoux2015convex,
              author  = {Chouzenoux, Emilie and Jezierska, Anna and
                         Pesquet, Jean-Christophe and Talbot, Hugues},
              title   = {A Convex Approach for Image Restoration with
                         Exact {Poisson}--{Gaussian} Likelihood},
              journal = {SIAM Journal on Imaging Sciences},
              volume  = {8},
              number  = {4},
              pages   = {2662--2682},
              year    = {2015},
              doi     = {10.1137/15M1014395}
            }
            @article{combettes2012primal,
              author  = {Combettes, Patrick L. and Pesquet, Jean-Christophe},
              title   = {Primal-Dual Splitting Algorithm for Solving
                         Inclusions with Mixtures of Composite,
                         {Lipschitzian}, and Parallel-Sum Type Monotone
                         Operators},
              journal = {Set-Valued and Variational Analysis},
              volume  = {20},
              number  = {2},
              pages   = {307--330},
              year    = {2012},
              doi     = {10.1007/s11228-011-0191-y}
            }
            ```

    Parameters
    ----------
    grad : callable
        ∇h, `(*N)` → `(*N)`.
    terms : sequence of (LinearOperator, Function)
        The pairs (V_r, ψ_r); V_r maps `(*N)` to `(C_r, *N)` and ψ_r
        acts on its output.
    x0 : Tensor
        Starting point x₀, `(*N)`; it sets the dtype and device.
    step : float
        γ > 0, for example from
        [`primal_dual_step`][optix.optim.primal_dual_step].
    constraint : Function, optional
        ψ₀ (for example [`Box`][optix.optim.Box]). Default: 0.
    duals : sequence of Tensor, optional
        Starting dual variables v_{r,0}. Default: zeros.
    max_iter : int
        Maximum number of iterations.
    tol : float, optional
        Relative-change tolerance; None runs `max_iter` iterations.
    value : callable, optional
        h, used for the `"objective"` of `trace` only.
    trace : dict, optional
        Filled with per-iteration lists: `"change"` (relative change of
        x), `"residual"` (‖xₖ − p₁,ₖ‖/γ) and, with `value`, `"objective"`
        (h + ψ₀ + ∑ψ_r(V_r ·) at p₁, in float64).
    callback : callable, optional
        `callback(k, p1)` after each iteration.

    Returns
    -------
    PrimalDualResult
    """
    gamma = float(step)
    if not (gamma > 0 and math.isfinite(gamma)):
        raise ValueError(f"step must be positive and finite, got {step}")
    terms = [(op, fn) for op, fn in terms]
    x = x0.clone()
    if duals is None:
        v = [op.forward(x).zero_() for op, _ in terms]
    else:
        v = [d.to(x).clone() for d in duals]
        if len(v) != len(terms):
            raise ValueError("duals must have one entry per term")

    def prox0(y: Tensor) -> Tensor:
        return y if constraint is None else constraint.prox(y, gamma)

    def adjoint_sum(fields: list[Tensor]) -> Tensor | None:
        acc = None
        for (op, _), w in zip(terms, fields, strict=True):
            t = op.adjoint(w)
            acc = t if acc is None else acc.add_(t)
        return acc

    def objective(p: Tensor) -> float:
        total = float(value(p))
        if constraint is not None:
            total += float(constraint.value(p))
        for op, fn in terms:
            total += float(fn.value(op.forward(p)))
        return total

    if trace is not None:
        for key in ("change", "residual", "objective"):
            trace[key] = []
    change = math.inf
    residual = math.inf
    p1 = x
    n = 0
    while n < max_iter:
        g = grad(x)
        adj = adjoint_sum(v)
        if adj is not None:
            g = g + adj
        y1 = x - gamma * g
        del g, adj
        p1 = prox0(y1)
        p2s = []
        for r, (op, fn) in enumerate(terms):
            y2 = v[r] + gamma * op.forward(x)
            p2 = prox_conjugate(fn.prox, y2, gamma)
            q2 = p2 + gamma * op.forward(p1)
            v[r] = v[r] - y2 + q2
            p2s.append(p2)
            del y2, q2
        g = grad(p1)
        adj = adjoint_sum(p2s)
        if adj is not None:
            g = g + adj
        del p2s, adj
        q1 = p1 - gamma * g
        del g
        x_new = x - y1 + q1
        del y1, q1
        n += 1
        prev = _norm(x)
        change = _norm(x_new - x) / prev if prev > 0 else math.nan
        residual = _norm(x - p1) / gamma
        x = x_new
        if trace is not None:
            trace["change"].append(change)
            trace["residual"].append(residual)
            if value is not None:
                trace["objective"].append(objective(p1))
        if callback is not None:
            callback(n, p1)
        if tol is not None and prev > 0 and change <= tol:
            break
    if change != change:  # nan: never measured
        change = math.inf
    return PrimalDualResult(p1, v, n, change, residual)
