"""Global fit of Foi et al. (2008): LS initialisation and ML (Nelder-Mead).

Section III-D (non-clipped), V-B (clipped) and VI-B/C (MAD). The model
is σ²(y) = a y + b in the normalised units of the paper (data
in [0, 1]). Everything here is float64 on the CPU: the problem has
two unknowns and one term per level set.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ._foi_clip import clipped_curve, inverse_init

__all__ = [
    "Pairs",
    "least_squares_init",
    "log_likelihood",
    "nelder_mead",
    "observed_information",
]

#: Half-width of the integration window, in standard deviations of
#: `ŷ` (not specified in the paper).
WINDOW = 8.0
#: Number of points of the finite sum over `y` (not specified).
NODES = 128
#: Safeguarded Newton steps used to invert the clipped expectation.
NEWTON = 30


@dataclass
class Pairs:
    """Local estimates and their variance factors (Eqs. 19, 49, 58).

    Attributes
    ----------
    yhat : Tensor
        ŷᵢ (Eq. 15).
    shat : Tensor
        σ̂ᵢ (Eq. 16), or σ̂ᵐᵃᵈᵢ (Eq. 57).
    vhat : Tensor
        Variance estimate used by the LS initialisation, Eq. (28):
        (κ̌ σ̂)².
    c, d : Tensor
        var{ŷ} = σ² c and var{σ̂} = σ² d.
    count : Tensor
        nᵢ.
    """

    yhat: Tensor
    shat: Tensor
    vhat: Tensor
    c: Tensor
    d: Tensor
    count: Tensor


def least_squares_init(
    pairs: Pairs, *, clipped: bool, robust: bool, eps: float
) -> tuple[float, float]:
    """Initial (a₀, b₀): Eqs. (26)-(28), or (54)-(56) when clipped.

    [a₀, b₀] = v Φ (ΦᵀΦ)⁻¹ with rows [yᵢ, 1] of
    Φ and vᵢ = (κ̌ σ̂ᵢ)². For clipped data,
    the pairs are first moved to their unclipped position with the
    inverse transformation (47)-(48).
    """
    y, v = pairs.yhat, pairs.vhat
    if clipped:
        shat = pairs.shat.clamp_min(torch.finfo(torch.float64).tiny)
        y, scale = inverse_init(pairs.yhat, shat, eps, robust)
        v = v * scale**2
    phi = torch.stack([y, torch.ones_like(y)], -1)
    sol = torch.linalg.solve(phi.T @ phi, phi.T @ v)
    return float(sol[0]), float(sol[1])


def _sigma_reg(a: float, b: float, y: Tensor, eps: float) -> Tensor:
    """Eq. (21): √(max(ε², a y + b))."""
    return (a * y + b).clamp_min(eps * eps).sqrt()


def _curve(
    a: float,
    b: float,
    y: Tensor,
    eps: float,
    clipped: bool,
    robust: bool,
    spread: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """(ỹ, σ̃_reg) (Eqs. 50-51), or (y, σ_reg)."""
    s = _sigma_reg(a, b, y, eps)
    if not clipped:
        return y, s
    return clipped_curve(y, s, robust, spread)


def _invert(
    a: float,
    b: float,
    target: Tensor,
    upper: float,
    eps: float,
    robust: bool,
) -> Tensor:
    """`y` in `[0, upper]` with ỹ(y) = target.

    Safeguarded Newton iterations (bisection whenever a step leaves the
    bracket). ỹ is increasing in `y` for all realistic
    `(a, b)`. Targets out of reach give the nearest bound.
    """
    ends = torch.tensor([0.0, upper], dtype=target.dtype)
    (t0, t1), _ = _curve(a, b, ends, eps, True, robust, spread=False)
    lo = torch.zeros_like(target)
    hi = torch.full_like(target, upper)
    done = (target <= t0) | (target >= t1)
    fixed = torch.where(target <= t0, 0.0, upper)
    y = target.clamp(0, upper)
    for _ in range(NEWTON):
        ytil, _ = _curve(a, b, y, eps, True, robust, spread=False)
        err = ytil - target
        ok = err.abs() <= 1e-13
        if bool((done | ok).all()):
            break
        lo = torch.where(err < 0, y, lo)
        hi = torch.where(err < 0, hi, y)
        # slope of ỹ, neglecting the variation of σ with y
        s = _sigma_reg(a, b, y, eps)
        ndtr = torch.special.ndtr
        slope = ndtr(y / s) + ndtr((1 - y) / s) - 1
        step = y - err / slope.clamp_min(1e-3)
        inside = (step >= lo) & (step <= hi)
        y = torch.where(ok, y, torch.where(inside, step, (lo + hi) / 2))
    return torch.where(done, fixed, y)


def log_likelihood(
    a: float,
    b: float,
    pairs: Pairs,
    *,
    clipped: bool,
    robust: bool,
    eps: float,
) -> float:
    """log L(a, b): Eq. (23) (non-clipped) or Eq. (53) (clipped).

    L = ∏ᵢ ∫ p(ŷᵢ | y) p(σ̂ᵢ | y) p₀(y) dy with the Gaussian densities of
    Eq. (22) (or (52)). p₀ is uniform on [0, 1] without clipping and on
    [0, ∞) with clipping (Section V-B, footnote 9).

    The integrals are evaluated as finite sums (Section III-D.1).

    !!! note "Not specified in the paper"
        Each sum is a midpoint rule with `NODES` nodes on the interval
        of y where |ŷᵢ − ỹ(y)| is within `WINDOW` standard deviations
        of ŷᵢ (outside, the integrand is below e⁻³² of its peak). Near
        the clipping bounds, where σ̃ varies fast, the window uses the
        largest σ instead. With clipping, y is restricted to [0, Y]
        where Y = 1 + 10 σ(Y), beyond which the clipped expectation
        equals 1 to within 10⁻²³.
    """
    yhat, shat, c, d = pairs.yhat, pairs.shat, pairs.c, pairs.d
    if clipped:
        upper = 1.0
        for _ in range(8):  # fixed point of Y = 1 + 10 σ(Y)
            top = torch.tensor(upper, dtype=torch.float64)
            upper = 1 + 10 * float(_sigma_reg(a, b, top, eps))
        ystar = _invert(a, b, yhat, upper, eps, robust)
        _, sstar = _curve(a, b, ystar, eps, True, robust)
        half = WINDOW * sstar * c.sqrt()
        # Near the clipping bounds σ̃ changes fast and ŷ may be out of
        # the reach of ỹ: widen the window with the largest σ, an upper
        # bound of σ̃ on [0, Y], so that it covers the integrand.
        bounds = torch.tensor([0.0, upper], dtype=torch.float64)
        (t0, t1), _ = _curve(a, b, bounds, eps, True, robust, spread=False)
        smax = float(_sigma_reg(a, b, bounds[1:], eps)) * c.sqrt()
        edge = (yhat - half <= t0) | (yhat + half >= t1)
        half = torch.where(edge, WINDOW * smax, half)
        ends = _invert(
            a, b, torch.cat([yhat - half, yhat + half]), upper, eps, robust
        )
        y0, y1 = ends.chunk(2)
    else:
        upper = 1.0
        half = WINDOW * _sigma_reg(a, b, yhat, eps) * c.sqrt()
        # keep the full width inside the support [0, 1] of p₀
        y0 = (yhat - half).clamp(0, upper - 2 * half).clamp_min(0)
        y1 = (y0 + 2 * half).clamp_max(upper)
    width = (y1 - y0).clamp_min(1e-300)
    t = (torch.arange(NODES, dtype=torch.float64) + 0.5) / NODES
    y = y0[:, None] + width[:, None] * t
    ytil, stil = _curve(a, b, y, eps, clipped, robust)
    var = (stil * stil).clamp_min(torch.finfo(torch.float64).tiny)
    logp = (
        -math.log(2 * math.pi)
        - 0.5 * torch.log(c * d)[:, None]
        - torch.log(var)
        - (yhat[:, None] - ytil) ** 2 / (2 * var * c[:, None])
        - (shat[:, None] - stil) ** 2 / (2 * var * d[:, None])
    )
    per = torch.logsumexp(logp, -1) + torch.log(width / NODES)
    return float(per.sum())


def nelder_mead(
    f: Callable[[Tensor], float],
    x0: Tensor,
    step: Tensor,
    *,
    max_iter: int,
    tol: float,
    callback: Callable[[float], object] | None = None,
) -> tuple[Tensor, float, bool, str]:
    """Minimise `f` with the downhill simplex of Nelder and Mead (1965).

    Standard coefficients: reflection 1, expansion 2, contraction 1/2,
    shrinkage 1/2. The initial simplex is `x0` and `x0 + step_k e_k`.

    Not specified in the paper: the initial simplex and the stopping
    rule. The search stops when the spread of `f` over the simplex is
    below `tol (1 + |f_best|)` and every vertex is within
    `tol (|x_best| + step)` of the best one, or after `max_iter`
    iterations.

    Returns
    -------
    x, fx, converged, reason
    """
    n = x0.numel()
    simplex = [x0.clone()] + [x0 + step[k] * torch.eye(n)[k] for k in range(n)]
    values = [f(x) for x in simplex]
    for it in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        best = simplex[0]
        if callback is not None:
            callback(values[0])
        fspread = values[-1] - values[0]
        xspread = max(
            float(((x - best).abs() / (best.abs() + step)).max())
            for x in simplex[1:]
        )
        if math.isfinite(values[0]) and (
            fspread <= tol * (1 + abs(values[0])) and xspread <= tol
        ):
            return best, values[0], True, f"simplex converged ({it} its)"
        centroid = torch.stack(simplex[:-1]).mean(0)
        worst = simplex[-1]
        xr = centroid + (centroid - worst)
        fr = f(xr)
        if fr < values[0]:
            xe = centroid + 2 * (centroid - worst)
            fe = f(xe)
            if fe < fr:
                simplex[-1], values[-1] = xe, fe
            else:
                simplex[-1], values[-1] = xr, fr
            continue
        if fr < values[-2]:
            simplex[-1], values[-1] = xr, fr
            continue
        if fr < values[-1]:  # outside contraction
            xc = centroid + 0.5 * (xr - centroid)
            fc = f(xc)
            if fc <= fr:
                simplex[-1], values[-1] = xc, fc
                continue
        else:  # inside contraction
            xc = centroid + 0.5 * (worst - centroid)
            fc = f(xc)
            if fc < values[-1]:
                simplex[-1], values[-1] = xc, fc
                continue
        for i in range(1, n + 1):  # shrink towards the best vertex
            simplex[i] = best + 0.5 * (simplex[i] - best)
            values[i] = f(simplex[i])
    i = min(range(n + 1), key=lambda i: values[i])
    return simplex[i], values[i], False, f"max_iter={max_iter} reached"


def observed_information(
    f: Callable[[Tensor], float], x: Tensor, step: Tensor
) -> Tensor:
    """Hessian of `f` at `x` by central differences.

    With `f = -log L` at the ML estimate, its inverse is the usual
    large-sample covariance of the estimate (not part of the paper).
    """
    n = x.numel()
    eye = torch.eye(n, dtype=torch.float64)
    h = torch.zeros(n, n, dtype=torch.float64)
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei, ej = eye[i] * step[i], eye[j] * step[j]
            if i == j:
                v = (f(x + ei) - 2 * f0 + f(x - ei)) / step[i] ** 2
            else:
                v = (
                    f(x + ei + ej)
                    - f(x + ei - ej)
                    - f(x - ei + ej)
                    + f(x - ei - ej)
                ) / (4 * step[i] * step[j])
            h[i, j] = h[j, i] = v
    return h
