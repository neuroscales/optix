"""Closed-form proximity operators."""

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = [
    "project_group_l2",
    "prox_box",
    "prox_conjugate",
    "prox_group_l2",
    "prox_shifted_poisson",
    "prox_squared_l2",
]


def prox_box(
    v: Tensor,
    lower: float | Tensor | None = 0.0,
    upper: float | Tensor | None = None,
) -> Tensor:
    """Projection onto the box [lower, upper].

        prox(v) = min(max(v, lower), upper),

    the proximity operator of the indicator function of the box, for any
    step size.

    Parameters
    ----------
    v : Tensor
        Point to project.
    lower, upper : float or Tensor, optional
        Bounds (None: unbounded on that side). Default: [0, +∞), the
        non-negative orthant.

    Returns
    -------
    Tensor
        Projected point, a new tensor.
    """
    return torch.clamp(v, min=lower, max=upper)


def _group_norm(v: Tensor, dim: int) -> Tensor:
    """‖vⱼ‖₂ over `dim`, kept as a size-1 dim, floored at the tiny value."""
    norm = torch.linalg.vector_norm(v, dim=dim, keepdim=True)
    return norm.clamp(min=torch.finfo(norm.dtype).tiny)


def prox_group_l2(v: Tensor, t: float | Tensor, *, dim: int = 0) -> Tensor:
    """Block soft threshold, the prox of t·∑ⱼ ‖vⱼ‖₂.

        prox(v)ⱼ = max(1 − t / ‖vⱼ‖₂, 0) · vⱼ,

    where vⱼ collects the components of `v` along `dim` at voxel j
    (the gradient for isotropic total variation, the packed Hessian for
    its Frobenius norm).

    Parameters
    ----------
    v : Tensor
        Point, with the components along `dim`.
    t : float or Tensor
        Threshold t ≥ 0 (step size times weight); a tensor broadcasts
        against the voxels.
    dim : int
        Component dim.

    Returns
    -------
    Tensor
        Thresholded point.
    """
    scale = (1 - t / _group_norm(v, dim)).clamp(min=0)
    return v * scale


def project_group_l2(
    v: Tensor, radius: float | Tensor, *, dim: int = 0
) -> Tensor:
    """Projection of each vⱼ onto the ℓ2 ball of the given radius.

        proj(v)ⱼ = vⱼ · min(1, r / ‖vⱼ‖₂).

    This is the prox of the convex conjugate of r·∑ⱼ ‖vⱼ‖₂ (Moreau:
    v = prox_group_l2(v, r) + project_group_l2(v, r)).

    Parameters
    ----------
    v : Tensor
        Point, with the components along `dim`.
    radius : float or Tensor
        Ball radius r ≥ 0.
    dim : int
        Component dim.

    Returns
    -------
    Tensor
        Projected point.
    """
    return v * (radius / _group_norm(v, dim)).clamp(max=1)


def prox_squared_l2(
    v: Tensor,
    t: float | Tensor,
    data: Tensor | float = 0.0,
    weight: float | Tensor = 1.0,
) -> Tensor:
    """Prox of the quadratic (w/2)·‖z − y‖².

        prox(v) = (v + t w y) / (1 + t w).

    Parameters
    ----------
    v : Tensor
        Point.
    t : float or Tensor
        Step size t > 0.
    data : Tensor or float
        Centre y.
    weight : float or Tensor
        Weight w ≥ 0.

    Returns
    -------
    Tensor
    """
    tw = t * weight
    return (v + tw * data) / (1 + tw)


def prox_shifted_poisson(
    v: Tensor,
    t: float | Tensor,
    data: Tensor,
    variance: Tensor | float = 0.0,
) -> Tensor:
    """Prox of the shifted-Poisson negative log-likelihood.

    With c = y + σ² ≥ 0, the function is (Ikoma et al. 2018, Eq. S3)

        f(z) = ∑ⱼ zⱼ − cⱼ log(zⱼ + σⱼ²),

    and prox_{t f}(v) = argmin_z f(z) + ‖z − v‖² / (2t). Setting the
    derivative to 0 gives, with w = z + σ², s = v + σ² and ρ = 1/t, the
    quadratic ρw² + (1 − ρs)w − c = 0, whose non-negative root is
    computed without cancellation: with q = ρs − 1 and
    Δ = √(q² + 4ρc),

        w = (q + Δ) / (2ρ)   if q ≥ 0,
        w = 2c / (Δ − q)     otherwise,

    and z = w − σ². This is algebraically the update (S17) of Ikoma et
    al.; with σ² = 0 it is the Poisson prox. The result satisfies
    z ≥ −σ² (z > −σ² when c > 0).

    Parameters
    ----------
    v : Tensor
        Point.
    t : float or Tensor
        Step size t = 1/ρ > 0.
    data : Tensor
        Data y, with y + σ² ≥ 0 (clip beforehand); broadcasts with `v`.
    variance : Tensor or float
        Read-noise variance σ² ≥ 0 (σ² = 0: Poisson), broadcasts with `v`.

    Returns
    -------
    Tensor
        prox_{t f}(v).
    """
    rho = 1 / t
    c = data + variance
    q = rho * (v + variance) - 1
    delta = torch.hypot(q, 2 * torch.sqrt(rho * c))
    pos = q >= 0
    den = torch.where(pos, torch.ones_like(q), delta - q)
    w = torch.where(pos, (q + delta) * (t / 2), 2 * c / den)
    return w - variance


def prox_conjugate(
    prox: Callable[[Tensor, float], Tensor], v: Tensor, t: float
) -> Tensor:
    """Prox of the convex conjugate f*, by the Moreau identity.

        prox_{t f*}(v) = v − t · prox_{f/t}(v / t).

    Parameters
    ----------
    prox : callable
        `prox(x, s)` = prox_{s f}(x).
    v : Tensor
        Point.
    t : float
        Step size t > 0.

    Returns
    -------
    Tensor
    """
    return v - t * prox(v / t, 1 / t)
