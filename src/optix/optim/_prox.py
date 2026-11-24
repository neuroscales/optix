"""Closed-form proximity operators."""

import math
from collections.abc import Callable

import torch
from torch import Tensor

__all__ = [
    "project_group_l2",
    "prox_box",
    "prox_conjugate",
    "prox_group_l2",
    "prox_schatten",
    "prox_shifted_poisson",
    "prox_squared_l2",
    "schatten_norm",
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


def _schatten_p(p: float | str) -> float:
    """1, 2 or inf, as a float."""
    p = float(p)
    if p not in (1.0, 2.0, math.inf):
        raise ValueError(f"p must be 1, 2 or inf, got {p}")
    return p


def _packed_ndim(components: int) -> int:
    """d such that d(d + 1)/2 = C."""
    d = round((math.sqrt(8 * components + 1) - 1) / 2)
    if d < 1 or d * (d + 1) // 2 != components:
        raise ValueError(
            f"{components} components do not pack a symmetric matrix"
        )
    return d


def _eig2(v: Tensor, dim: int) -> tuple[Tensor, Tensor, Tensor]:
    """m, r and the off-centre part of packed symmetric 2 × 2 matrices.

    v holds (a, d, √2·b) along `dim`, for X = [[a, b], [b, d]]. The
    eigenvalues are μ± = m ± r with m = (a + d)/2 and
    r = √(((a − d)/2)² + b²) ≥ 0. Returns m, r (size-1 dims along `dim`)
    and X − m·I in packed form.
    """
    a, d, c = v.unbind(dim)
    m = (a + d) / 2
    h = (a - d) / 2
    r = torch.hypot(h, c / math.sqrt(2))
    dev = torch.stack([h, -h, c], dim=dim)
    return m.unsqueeze(dim), r.unsqueeze(dim), dev


def schatten_norm(v: Tensor, p: float | str, *, dim: int = 0) -> Tensor:
    """Schatten p-norm of packed symmetric matrices, voxel by voxel.

    `v` packs a symmetric d × d matrix X at every voxel along `dim`, as
    the output of [`Hessian`][optix.optim.Hessian]: the d diagonal
    entries, then √2 times the entries above the diagonal, C = d(d + 1)/2
    components. The singular values of X are the absolute values of its
    eigenvalues, so

        ‖X‖_{S₁} = ∑ |μᵢ|,   ‖X‖_{S₂} = ‖X‖_F = ‖v‖₂,
        ‖X‖_{S∞} = maxᵢ |μᵢ|.

    For d = 2 the eigenvalues are computed in closed form (Lefkimmiatis,
    Bourquard & Unser 2012, Eq. 15); p = 2 is available for any d;
    p ∈ {1, ∞} needs d ≤ 2.

    Parameters
    ----------
    v : Tensor
        Packed matrices, components along `dim`.
    p : {1, 2, inf}
        Order of the norm.
    dim : int
        Component dim.

    Returns
    -------
    Tensor
        ‖Xⱼ‖_{S_p} per voxel (`dim` removed).
    """
    p = _schatten_p(p)
    d = _packed_ndim(v.shape[dim])
    if p == 2:
        return torch.linalg.vector_norm(v, dim=dim)
    if d == 1:
        return v.abs().squeeze(dim)
    if d != 2:
        raise NotImplementedError(
            "Schatten norms with p in {1, inf} are implemented for 2 × 2 "
            "matrices only"
        )
    m, r, _ = _eig2(v, dim)
    m, r = m.squeeze(dim), r.squeeze(dim)
    if p == 1:
        # |m + r| + |m − r| = 2·max(|m|, r)
        return 2 * torch.maximum(m.abs(), r)
    return m.abs() + r


def prox_schatten(
    v: Tensor, t: float | Tensor, p: float | str, *, dim: int = 0
) -> Tensor:
    """Prox of t·∑ⱼ ‖Xⱼ‖_{S_p}, for packed symmetric matrices.

    With Y = U diag(σ) Vᴴ (singular value decomposition), the prox of a
    Schatten norm acts on the singular values only (Lefkimmiatis & Unser
    2013, Proposition 1, Eq. 36):

        prox_{t‖·‖_{S_p}}(Y) = U diag(prox_{t‖·‖_p}(σ)) Vᴴ,

    with the ℓ_p proxes (Eqs. 43–47)

    - p = 1: soft threshold, σ̂ᵢ = max(σᵢ − t, 0);
    - p = 2: block soft threshold of the whole matrix (Eq. 44), the same
      as [`prox_group_l2`][optix.optim.prox_group_l2] on the packed
      vector;
    - p = ∞: σ̂ᵢ = min(σᵢ, c), with c = 0 if σ₁ + σ₂ ≤ t,
      c = (σ₁ + σ₂ − t)/2 if |σ₁ − σ₂| < t < σ₁ + σ₂, and c = σ₁ − t
      if σ₁ − σ₂ ≥ t (σ₁ ≥ σ₂). This is Eq. (46) with the threshold
      written out: the paper states it for t = 1, and c = t·γ(σ/t).

    For symmetric 2 × 2 matrices no decomposition is computed: the
    eigenvalues μ± = m ± r (m the mean of the diagonal, r ≥ 0 half
    their gap), whose absolute values are the singular values, are
    mapped to μ̂± by the ℓ_p prox with their signs kept, and

        X̂ = m̂·I + (r̂/r)·(X − m·I),   m̂ = (μ̂₊ + μ̂₋)/2,
        r̂ = (μ̂₊ − μ̂₋)/2,

    which keeps the eigenvectors of X (X̂ = m̂·I where r = 0). The
    proxes are monotone in μ, so μ̂₊ ≥ μ̂₋ and 0 ≤ r̂ ≤ r.

    The packing is that of [`Hessian`][optix.optim.Hessian] (diagonal
    entries, then √2 times the off-diagonal ones), an isometry for the
    Frobenius inner product, so this is exactly the matrix prox.

    !!! note "Dimensions"
        p = 2 is available for matrices of any size; p ∈ {1, ∞} for
        2 × 2 matrices (C = 3) and 1 × 1 ones (C = 1), the case of the
        2-D images of the paper.

    Parameters
    ----------
    v : Tensor
        Packed symmetric matrices, components along `dim`.
    t : float or Tensor
        Threshold t ≥ 0 (step size times weight); a tensor broadcasts
        against the voxels (with a size-1 `dim`).
    p : {1, 2, inf}
        Order of the Schatten norm.
    dim : int
        Component dim.

    Returns
    -------
    Tensor
        The prox, packed like `v`.
    """
    p = _schatten_p(p)
    d = _packed_ndim(v.shape[dim])
    if p == 2:
        return prox_group_l2(v, t, dim=dim)
    if d == 1:  # |x|: soft threshold for every p
        return torch.sign(v) * (v.abs() - t).clamp(min=0)
    if d != 2:
        raise NotImplementedError(
            "prox_schatten with p in {1, inf} is implemented for 2 × 2 "
            "matrices (2-D images) only; use p=2"
        )
    m, r, dev = _eig2(v, dim)
    hi, lo = m + r, m - r  # μ₊ ≥ μ₋
    s1, s2 = hi.abs(), lo.abs()
    if p == 1:
        n1 = (s1 - t).clamp(min=0)
        n2 = (s2 - t).clamp(min=0)
    else:
        big = torch.maximum(s1, s2)
        small = torch.minimum(s1, s2)
        cap = torch.where(
            big - small >= t,
            big - t,
            ((big + small - t) / 2).clamp(min=0),
        )
        n1 = torch.minimum(s1, cap)
        n2 = torch.minimum(s2, cap)
    hi = torch.sign(hi) * n1
    lo = torch.sign(lo) * n2
    m_new = (hi + lo) / 2
    r_new = (hi - lo) / 2
    tiny = torch.finfo(r.dtype).tiny
    ratio = torch.where(r > tiny, r_new / r.clamp(min=tiny), 0)
    eye = torch.zeros_like(v)
    eye.narrow(dim, 0, 2).copy_(m_new.expand_as(eye.narrow(dim, 0, 2)))
    return eye + ratio * dev
