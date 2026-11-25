"""Closed-form proximity operators."""

import math
from collections.abc import Callable

import torch
from torch import Tensor

from ._spectral import (
    clip,
    eigenvalues,
    l1_level,
    packed_ndim,
    soft,
    spectral_map,
    voxelwise,
)

__all__ = [
    "project_group_l2",
    "project_l2_ball",
    "project_schatten",
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


def project_l2_ball(
    v: Tensor,
    radius: float,
    center: Tensor | float = 0.0,
    weight: Tensor | float = 1.0,
    *,
    max_iter: int = 50,
) -> Tensor:
    """Projection onto the weighted ℓ2 ball ∑ⱼ wⱼ (zⱼ − cⱼ)² ≤ r².

    The proximity operator of the indicator function of the ball, for
    any step size. With e = v − c, the projection is (KKT conditions)

        z = c + e / (1 + μw),

    with μ = 0 if v is inside the ball, otherwise the root μ > 0 of the
    decreasing convex function φ(μ) = ∑ⱼ wⱼ eⱼ² / (1 + μwⱼ)² − r².
    With a scalar w this is the closed form z = c + e·r / (√w ‖e‖₂);
    otherwise Newton's method on φ from μ = 0, which increases
    monotonically to the root, stopped when φ(μ) ≤ 10⁻¹²·r² or after
    `max_iter` steps (then the point is scaled radially onto the ball,
    so the result is always feasible). The scalar reductions are in
    float64.

    Parameters
    ----------
    v : Tensor
        Point.
    radius : float
        r ≥ 0.
    center : Tensor or float
        Centre c; broadcasts with `v`.
    weight : Tensor or float
        Weights w ≥ 0; broadcasts with `v`.
    max_iter : int
        Maximum number of Newton steps (non-scalar weights).

    Returns
    -------
    Tensor
        Projected point.
    """
    r2 = float(radius) ** 2
    e = v - center
    if not isinstance(weight, Tensor) or weight.numel() == 1:
        w = float(weight)
        norm2 = w * float(e.double().square().sum())
        if norm2 <= r2:
            return v.clone()
        return center + e * math.sqrt(r2 / norm2)
    we2 = (weight * e.square()).double()
    w = weight.double()
    mu = 0.0
    phi = float(we2.sum()) - r2
    if phi <= 0:
        return v.clone()
    for _ in range(max_iter):
        den = 1 + mu * w
        phi = float((we2 / den.square()).sum()) - r2
        if phi <= 1e-12 * r2:
            break
        slope = float((-2 * w * we2 / den.pow(3)).sum())
        mu -= phi / slope
    z = e / (1 + mu * weight)
    norm2 = float((weight * z.square()).double().sum())
    if norm2 > r2:
        z = z * math.sqrt(r2 / norm2)
    return center + z


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
    return packed_ndim(components)


def _check_spectral(d: int, what: str) -> None:
    if d > 3:
        raise NotImplementedError(
            f"{what} with p in {{1, inf}} is implemented for matrices up to "
            "3 × 3 (2-D and 3-D images); use p=2"
        )


def schatten_norm(v: Tensor, p: float | str, *, dim: int = 0) -> Tensor:
    """Schatten p-norm of packed symmetric matrices, voxel by voxel.

    `v` packs a symmetric d × d matrix X at every voxel along `dim`, as
    the output of [`Hessian`][optix.optim.Hessian]: the d diagonal
    entries, then √2 times the entries above the diagonal, C = d(d + 1)/2
    components. The singular values of X are the absolute values of its
    eigenvalues, so (Lefkimmiatis, Ward & Unser 2013, Eq. 4)

        ‖X‖_{S₁} = ∑ |μᵢ|,   ‖X‖_{S₂} = ‖X‖_F = ‖v‖₂,
        ‖X‖_{S∞} = maxᵢ |μᵢ|.

    p = 2 is available for any d. For p ∈ {1, ∞} the eigenvalues are
    computed in closed form for d ≤ 3: Lefkimmiatis, Bourquard & Unser
    (2012, Eq. 15) for d = 2, and the method described in
    [`prox_schatten`][optix.optim.prox_schatten] for d = 3. The result
    has finite derivatives everywhere.

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

    Raises
    ------
    NotImplementedError
        For p ∈ {1, ∞} with d > 3.
    """
    p = _schatten_p(p)
    d = packed_ndim(v.shape[dim])
    if p == 2:
        return torch.linalg.vector_norm(v, dim=dim)
    _check_spectral(d, "schatten_norm")

    def kernel(x: Tensor, t: object) -> Tensor:
        a = eigenvalues(x).abs()
        return a.sum(0) if p == 1 else a.amax(0)

    return voxelwise(kernel, v, 0.0, dim, components=False)


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
    - p = ∞: σ̂ᵢ = min(σᵢ, c), with c = max(0, maxₖ (Sₖ − t)/k) and Sₖ
      the sum of the k largest σᵢ. For 2 × 2 matrices (σ₁ ≥ σ₂) this is
      c = 0 if σ₁ + σ₂ ≤ t, c = (σ₁ + σ₂ − t)/2 if
      |σ₁ − σ₂| < t < σ₁ + σ₂, and c = σ₁ − t if σ₁ − σ₂ ≥ t: Eq. (46)
      with the threshold written out (the paper states it for t = 1,
      and c = t·γ(σ/t)). It is σ minus the projection of σ onto the ℓ1
      ball of radius t (Moreau).

    For a symmetric matrix the singular values are the absolute
    eigenvalues, and the map keeps the signs of the eigenvalues and the
    eigenvectors: X̂ = U diag(ĝ(μ)) Uᵀ with ĝ(μ) = sign(μ)·σ̂. No
    decomposition is formed for 2 × 2 matrices (eigenvalues m ± r,
    X̂ = m̂·I + (r̂/r)·(X − m·I)); 3 × 3 matrices use the closed form
    below. The packing is that of [`Hessian`][optix.optim.Hessian]
    (diagonal entries, then √2 times the off-diagonal ones), an isometry
    for the Frobenius inner product, so this is exactly the matrix prox.

    !!! note "Dimensions"
        p = 2 is available for matrices of any size; p ∈ {1, ∞} for
        1 × 1, 2 × 2 and 3 × 3 matrices (C = 1, 3, 6): 1-D, 2-D and 3-D
        images.

    ??? note "3 × 3 closed form (not in the papers, which are 2-D)"
        X = m·I + s·B with m = tr(X)/3 and s the largest absolute entry
        of X − m·I, so B is traceless with q = tr(B²) ≥ 1.

        1. The eigenvalue of B farthest from the other two is
           λ = sign(ρ)·2√(q/6)·cos(arccos(|ρ|)/3), with
           ρ = det(B)/(2(q/6)^{3/2}) (the trigonometric solution of the
           characteristic cubic, Smith 1961), refined by one Newton step
           on the cubic. Its gap to the others is at least √(q/2).
        2. Its unit eigenvector u is the largest cross product of two
           rows of B − λI, normalised; the gap keeps it well
           conditioned.
        3. The other two eigenvalues are m₂ ± r₂, those of B on the
           plane P = I − uuᵀ: m₂ = (tr B − uᵀBu)/2, r₂ = ‖D‖_F/√2 with
           D = P B P − m₂P, solved like a 2 × 2 matrix without a basis
           of the plane.

        The map is X̂ = ĝ_u·uuᵀ + m̂·P + (r̂/r)·s·D. A repeated
        eigenvalue is always the pair, which needs no eigenvectors, so
        the result is accurate to a few ulps of s with repeated, nearly
        repeated or zero eigenvalues; X = m·I maps to g(m)·I. Compared
        with batched `torch.linalg.eigh` and the reconstruction
        U diag(ĝ) Uᵀ (2·10⁵ matrices, one CPU thread), it is about 3
        times faster in float32 (1.6 times in float64), at least as
        accurate (within 5 ulps of a float64 reference in float32 and
        about 10 in float64; float32 `eigh`: 16), element-wise (processed in
        chunks of voxels to bound memory), and differentiable without
        NaN, where the derivatives of `eigh` are NaN at repeated
        eigenvalues. Smith OK, *Eigenvalues of a symmetric 3 × 3
        matrix*, Commun ACM 4(4):168, 1961,
        [doi:10.1145/355578.366316](https://doi.org/10.1145/355578.366316).

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

    Raises
    ------
    NotImplementedError
        For p ∈ {1, ∞} with d > 3.
    """
    p = _schatten_p(p)
    d = packed_ndim(v.shape[dim])
    if p == 2:
        return prox_group_l2(v, t, dim=dim)
    _check_spectral(d, "prox_schatten")
    if p == 1:
        return voxelwise(
            lambda x, s: spectral_map(x, soft, s), v, t, dim, components=True
        )

    def cap(mu: Tensor, s: float | Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return clip(mu, l1_level(mu.abs(), s))

    return voxelwise(
        lambda x, s: spectral_map(x, cap, s), v, t, dim, components=True
    )


def project_schatten(
    v: Tensor, radius: float | Tensor, q: float | str, *, dim: int = 0
) -> Tensor:
    """Projection of packed symmetric matrices onto the S_q ball of radius r.

    By Proposition 1 of Lefkimmiatis, Ward & Unser (2013), the
    projection onto {X : ‖X‖_{S_q} ≤ r} projects the singular values
    onto the ℓ_q ball of radius r and keeps the singular vectors
    (Eqs. 33–36):

    - q = 2: X · min(1, r/‖X‖_F) (Eq. 33; the same as
      [`project_group_l2`][optix.optim.project_group_l2] on the packed
      vector);
    - q = ∞: σ̂ᵢ = min(σᵢ, r) (Eq. 34);
    - q = 1: σ̂ᵢ = max(σᵢ − γ, 0), with γ = max(0, maxₖ (Sₖ − r)/k) and
      Sₖ the sum of the k largest σᵢ (Eq. 35; for 2 × 2 matrices this is
      Eq. 36 scaled to radius r).

    With 1/p + 1/q = 1 it is the prox of the convex conjugate of
    r·‖·‖_{S_p} (Moreau decomposition):

        X = prox_schatten(X, r, p) + project_schatten(X, r, q).

    The decomposition is that of [`prox_schatten`][optix.optim.prox_schatten]
    (closed form, 3 × 3 included).

    Parameters
    ----------
    v : Tensor
        Packed symmetric matrices, components along `dim`.
    radius : float or Tensor
        Radius r ≥ 0; a tensor broadcasts against the voxels (with a
        size-1 `dim`).
    q : {1, 2, inf}
        Order of the Schatten ball.
    dim : int
        Component dim.

    Returns
    -------
    Tensor
        The projection, packed like `v`.

    Raises
    ------
    NotImplementedError
        For q ∈ {1, ∞} with d > 3.
    """
    q = _schatten_p(q)
    d = packed_ndim(v.shape[dim])
    if q == 2:
        return project_group_l2(v, radius, dim=dim)
    _check_spectral(d, "project_schatten")
    if q == math.inf:
        return voxelwise(
            lambda x, s: spectral_map(x, clip, s),
            v,
            radius,
            dim,
            components=True,
        )

    def level(mu: Tensor, s: float | Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return soft(mu, l1_level(mu.abs(), s))

    return voxelwise(
        lambda x, s: spectral_map(x, level, s), v, radius, dim, components=True
    )
