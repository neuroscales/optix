"""Global fit of the noise model: LS initialisation and ML (Nelder-Mead).

Follows [Foi et al. (2008)](https://doi.org/10.1109/TIP.2008.2001399):
Section III-D (non-clipped), V-B (clipped) and VI-B/C (MAD). The model
is σ²(y) = a y + b in the normalised units of the paper (data
in [0, 1]). Everything here is float64 on the CPU: the problem has
two unknowns and one term per level set.
"""

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ._foi_clip import clipped_curve, inverse_init

__all__ = [
    "Pairs",
    "least_squares_init",
    "covariance",
    "log_likelihood",
    "nelder_mead",
    "observed_information",
]

#: Initial half-width of the integration window on each side of the
#: mode, in widths of the Gaussian approximation of the integrand (not
#: specified in the paper).
WINDOW = 9.0
#: The window is widened until the integrand is below e^(-TAIL) of its
#: peak at its ends, by doubling at most DOUBLINGS times.
TAIL = 36.0
DOUBLINGS = 12
#: Gauss–Legendre nodes of the finite sum over y on each side of the
#: mode (not specified in the paper).
NODES = 96
#: Points of the coarse grid that starts the search for the mode.
GRID = 32
#: Maximum Gauss–Newton iterations of the search for the mode.
GAUSS_NEWTON = 12
#: Finite-difference step of the observed information, in standard
#: errors of the estimate.
FD_STEP_SE = 0.5
#: Hessian evaluations tried by `covariance`.
FD_TRIES = 6


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


def _upper(a: float, b: float, eps: float) -> float:
    """Y = 1 + 10 σ(Y): ỹ(y) = 1 to within 10⁻²³ for y ≥ Y."""
    upper = 1.0
    for _ in range(8):  # fixed point
        top = torch.tensor(upper, dtype=torch.float64)
        upper = 1 + 10 * float(_sigma_reg(a, b, top, eps))
    return upper


def _residuals(
    a: float,
    b: float,
    y: Tensor,
    pairs: tuple[Tensor, ...],
    eps: float,
    clipped: bool,
    robust: bool,
) -> tuple[Tensor, Tensor]:
    """Standardised residuals (ŷ − ỹ)/(σ̃ √c) and log(σ̂/σ̃)/√d at `y`."""
    yhat, shat, c, d = pairs
    ytil, stil = _curve(a, b, y, eps, clipped, robust)
    stil = stil.clamp_min(torch.finfo(torch.float64).tiny)
    return (yhat - ytil) / (stil * c.sqrt()), torch.log(shat / stil) / d.sqrt()


def _peak(
    a: float,
    b: float,
    pairs: tuple[Tensor, ...],
    y: Tensor,
    upper: float,
    eps: float,
    clipped: bool,
    robust: bool,
) -> tuple[Tensor, Tensor]:
    """Mode of the integrand of each set and its precision in y.

    Gauss–Newton on q(y) = r₁² + r₂² (the residuals of `_residuals`)
    from `y`, with forward-difference derivatives and a step shortened
    fourfold when it does not decrease q, inside [0, upper]. It stops
    when every step is below 10⁻³ of the width of the mode.

    Returns
    -------
    y, precision : Tensor
        The mode and J₁² + J₂², the inverse squared width in y of the
        Gaussian approximation of the integrand.
    """
    tiny = torch.finfo(torch.float64).tiny
    r1, r2 = _residuals(a, b, y, pairs, eps, clipped, robust)
    for _ in range(GAUSS_NEWTON):
        h = 1e-6 * _sigma_reg(a, b, y, eps)
        h = torch.where(y + h <= upper, h, -h)
        u1, u2 = _residuals(a, b, y + h, pairs, eps, clipped, robust)
        j1, j2 = (u1 - r1) / h, (u2 - r2) / h
        precision = (j1 * j1 + j2 * j2).clamp_min(tiny)
        step = torch.nan_to_num(-(j1 * r1 + j2 * r2) / precision, 0, 0, 0)
        q = r1 * r1 + r2 * r2
        for shrink in (1, 4, 16):
            new = (y + step / shrink).clamp(0, upper)
            n1, n2 = _residuals(a, b, new, pairs, eps, clipped, robust)
            better = n1 * n1 + n2 * n2 <= q
            y = torch.where(better, new, y)
            r1 = torch.where(better, n1, r1)
            r2 = torch.where(better, n2, r2)
            if bool(better.all()):
                break
            step = torch.where(better, 0, step)
        if bool((step.abs() * precision.sqrt() <= 1e-3).all()):
            break
    return y, precision


def log_likelihood(
    a: float,
    b: float,
    pairs: Pairs,
    *,
    clipped: bool,
    robust: bool,
    eps: float,
    per_set: bool = False,
) -> float | Tensor:
    """log L(a, b): Eq. (23) (non-clipped) or Eq. (53) (clipped).

    L = ∏ᵢ ∫ p(ŷᵢ | y) p(σ̂ᵢ | y) p₀(y) dy with the Gaussian densities of
    Eq. (22) (or (52)). p₀ is uniform on [0, 1] without clipping and on
    [0, ∞) with clipping (Section V-B, footnote 9).

    The integrals are evaluated as finite sums (Section III-D.1).

    !!! note "Not specified in the paper: the quadrature"
        Each integrand is a narrow bump in y: ŷᵢ pins ỹ(y) to within
        σ̃ √cᵢ and σ̂ᵢ pins σ̃(y) to within σ̃ √dᵢ. Where the data are
        heavily clipped, ỹ is nearly flat in y while σ̃ still decays
        fast, so the σ̂ factor is far narrower than the ŷ factor. The
        sum therefore follows the joint mode: the mode is found by
        Gauss–Newton on the two standardised residuals (started from
        the best of ŷᵢ and a coarse grid of the support). On each side
        of the mode, `NODES` Gauss–Legendre nodes cover `WINDOW` widths
        of its Gaussian approximation, a width doubled until the
        integrand at the end is below e⁻³⁶ of its peak (skewed bumps of
        small sets), within the support. With clipping the support is
        [0, Y] with Y = 1 + 10 σ(Y), beyond which ỹ = 1 to within
        10⁻²³. Against a dense reference, the error is below 10⁻⁶ per
        level set.

    Parameters
    ----------
    per_set : bool
        Return the log-integral of each level set instead of the sum.
    """
    yhat, shat, c, d = pairs.yhat, pairs.shat, pairs.c, pairs.d
    args = (yhat, shat, c, d)
    col = tuple(v[:, None] for v in args)
    kw = {"eps": eps, "clipped": clipped, "robust": robust}
    upper = _upper(a, b, eps) if clipped else 1.0
    # start from the best of ŷ and a coarse grid of the support
    start = yhat.clamp(0, upper)
    grid = torch.linspace(0, upper, GRID, dtype=torch.float64)
    candidates = torch.cat([start[:, None], grid.expand(len(yhat), -1)], 1)
    r1, r2 = _residuals(a, b, candidates, col, **kw)
    q = torch.nan_to_num(r1 * r1 + r2 * r2, nan=math.inf)
    y = candidates.gather(1, q.argmin(1, keepdim=True)).squeeze(1)
    y, precision = _peak(a, b, args, y, upper, **kw)
    # each side: WINDOW widths, doubled while the integrand at the end
    # is above e^(-TAIL) of its peak (skewed bumps, small n)
    half0 = WINDOW / precision.sqrt()
    half0 = torch.where(half0.isfinite() & (half0 > 0), half0, upper)
    half = torch.stack([half0, half0], -1)  # (sets, side)
    side = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    bound = torch.tensor([0.0, upper], dtype=torch.float64)
    nodes, weights = _gauss_legendre()
    tiny = torch.finfo(torch.float64).tiny
    for _ in range(DOUBLINGS + 1):
        end = (y[:, None] + side * half).clamp(0, upper)
        width = (end - y[:, None]).abs()
        # nodes from the mode outwards: the last one is nearest the end
        yy = y[:, None, None] + (end - y[:, None])[..., None] * nodes
        logp = _log_integrand(a, b, yy, tuple(v[..., None] for v in col), **kw)
        peak = logp.amax((-1, -2), keepdim=True).squeeze(-1)
        grow = (logp[..., -1] > peak - TAIL) & (end != bound)
        if not bool(grow.any()):
            break
        half = torch.where(grow, 2 * half, half)
    sums = torch.logsumexp(logp + weights.log(), -1)
    sums = sums + torch.log(width.clamp_min(tiny))
    per = torch.logsumexp(sums, -1)
    return per if per_set else float(per.sum())


def _log_integrand(
    a: float,
    b: float,
    y: Tensor,
    pairs: tuple[Tensor, ...],
    eps: float,
    clipped: bool,
    robust: bool,
) -> Tensor:
    """log p((ŷ, σ̂) | y) of Eq. (22), or (52) with clipping."""
    yhat, shat, c, d = pairs
    ytil, stil = _curve(a, b, y, eps, clipped, robust)
    var = (stil * stil).clamp_min(torch.finfo(torch.float64).tiny)
    return (
        -math.log(2 * math.pi)
        - 0.5 * torch.log(c * d)
        - torch.log(var)
        - (yhat - ytil) ** 2 / (2 * var * c)
        - (shat - stil) ** 2 / (2 * var * d)
    )


@functools.cache
def _gauss_legendre() -> tuple[Tensor, Tensor]:
    """`NODES` Gauss–Legendre nodes and weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(NODES)
    f64 = torch.float64
    return torch.tensor((x + 1) / 2, dtype=f64), torch.tensor(w / 2, dtype=f64)


def nelder_mead(
    f: Callable[[Tensor], float],
    x0: Tensor,
    step: Tensor,
    *,
    max_iter: int,
    tol: float,
    callback: Callable[[float], object] | None = None,
) -> tuple[Tensor, float, bool, str]:
    """Minimise `f` with the downhill simplex of
    [Nelder and Mead (1965)](https://doi.org/10.1093/comjnl/7.4.308).

    Standard coefficients: reflection 1, expansion 2, contraction 1/2,
    shrinkage 1/2. The initial simplex is `x0` and `x0 + stepₖ eₖ`.

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


def covariance(
    f: Callable[[Tensor], float], x: Tensor, step: Tensor
) -> tuple[Tensor, bool]:
    """Inverse observed information of `f = −log L` at the estimate `x`.

    Not part of the paper. The Hessian H is computed by central
    differences (`observed_information`), symmetrised, and accepted
    only if positive definite (its smallest eigenvalue > 0) with a step
    close to `FD_STEP_SE` standard errors of its own inverse. Steps
    much smaller than the standard error
    amplify the round-off of `f`; much larger ones measure curvature
    away from the optimum. Starting from `step`, the step is set to
    `FD_STEP_SE` standard errors of the last positive-definite H, or
    quadrupled when H is not positive definite, for at most `FD_TRIES`
    evaluations.

    Returns
    -------
    cov : (2, 2) Tensor
        H⁻¹, or all NaN if no positive-definite H was found.
    ok : bool
        Whether `cov` is a valid covariance.
    """
    h = step.to(torch.float64).abs()
    fallback = None
    for _ in range(FD_TRIES):
        info = observed_information(f, x, h)
        info = (info + info.T) / 2
        if (
            bool(info.isfinite().all())
            and float(torch.linalg.eigvalsh(info).min()) > 0
        ):
            cov = torch.linalg.inv(info)
            target = FD_STEP_SE * cov.diagonal().sqrt()
            fallback = cov
            ratio = h / target
            if bool(((ratio > 0.5) & (ratio < 2)).all()):
                return cov, True
            h = target
        else:
            h = 4 * h
    if fallback is not None:
        return fallback, True
    return torch.full((2, 2), torch.nan, dtype=torch.float64), False
