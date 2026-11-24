"""Variational stripe removal of Rottmayer, Redenbach and Fahrbach (2025)."""

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Self

import torch
from torch import Tensor

from .._config import resolve_backend
from .._estimator import Estimator, Tags, _same
from ..io import ArrayLike, Layout, WritableArrayLike

__all__ = [
    "GeneralStripeRemover",
    "general_stripe_objective",
    "general_stripe_remover",
    "lattice_diff",
    "lattice_diff_adjoint",
    "stripe_offset",
]


# lattice directions of the stripe difference operator, (along the stripe
# axis, across it), and their angles in degrees
_LATTICE = ((1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1))
_LATTICE_ANGLES = tuple(math.degrees(math.atan2(q, p)) for p, q in _LATTICE)

# step sizes τ = σ of the authors' code, for 2-D and 3-D problems, and the
# bound on ‖K‖² they are matched to (τσ‖K‖² < 1)
_STEP_2D, _BOUND_2D = 0.35, 8.0
_STEP_3D, _BOUND_3D = 0.28, 12.0
_CHECK_EVERY = 10
# μ₁ at or below which several non-orthogonal stripe directions typically
# remove little or nothing (measured on synthetic frames for 0° + 45°,
# 0° + 26.6° and ±45°, where removal starts above 1/2)
_MULTI_LIMIT = 0.5


# -- operators ----------------------------------------------------------------
def _slices(offset: Sequence[int], shape: Sequence[int]) -> tuple:
    dst = tuple(
        slice(max(0, -o), n - max(0, o))
        for o, n in zip(offset, shape, strict=True)
    )
    src = tuple(
        slice(max(0, o), n - max(0, -o))
        for o, n in zip(offset, shape, strict=True)
    )
    return (Ellipsis, *dst), (Ellipsis, *src)


def _diff_into(out: Tensor, x: Tensor, offset: tuple[int, ...]) -> None:
    """Write D_v x into the valid region of D_v in `out` (rest untouched)."""
    dst, src = _slices(offset, x.shape[x.ndim - len(offset) :])
    torch.sub(x[src], x[dst], out=out[dst])


def _diff_adjoint_add(
    out: Tensor, p: Tensor, offset: tuple[int, ...], alpha: float = 1.0
) -> None:
    """out += α D_vᵀ p, in place."""
    dst, src = _slices(offset, p.shape[p.ndim - len(offset) :])
    valid = p[dst]
    out[dst].sub_(valid, alpha=alpha)
    out[src].add_(valid, alpha=alpha)


def lattice_diff(x: Tensor, offset: Sequence[int]) -> Tensor:
    """Forward difference along an integer lattice offset.

        (D_v x)[i] = x[i + v] − x[i]  if i + v lies in the grid, else 0,

    on the trailing `len(offset)` dims of `x`. The offset `(1, 0)` is the
    forward difference along the rows axis with the zero (Neumann)
    boundary of a total-variation gradient; `(2, 1)` and `(1, 1)` are the
    oblique differences used for stripes at about 26.6° and 45°.

    Parameters
    ----------
    x : Tensor
        `(..., *grid)`.
    offset : sequence of int
        Lattice offset v, one integer per trailing dim.

    Returns
    -------
    Tensor
        D_v x, of the shape of `x`.
    """
    out = torch.zeros_like(x)
    _diff_into(out, x, tuple(int(o) for o in offset))
    return out


def lattice_diff_adjoint(p: Tensor, offset: Sequence[int]) -> Tensor:
    """Adjoint D_vᵀ of [`lattice_diff`][optix.destripe.utils.lattice_diff].

    It satisfies ⟨D_v x, p⟩ = ⟨x, D_vᵀ p⟩ for all x and p (values of `p`
    outside the valid region of D_v are ignored).

    Parameters
    ----------
    p : Tensor
        `(..., *grid)`.
    offset : sequence of int
        Lattice offset v.

    Returns
    -------
    Tensor
        D_vᵀ p, of the shape of `p`.
    """
    out = torch.zeros_like(p)
    _diff_adjoint_add(out, p, tuple(int(o) for o in offset))
    return out


def stripe_offset(angle: float) -> tuple[int, int]:
    """Lattice offset of the stripe difference ∇_θ for a stripe angle θ.

    θ is in degrees, from the stripe axis towards the other in-plane axis
    (0: stripes along the stripe axis). It is rounded to the nearest of the
    directions 0°, ±26.6°, ±45°, ±63.4° and 90° (modulo 180°), whose
    offsets are (1, 0), (2, ±1), (1, ±1), (1, ±2) and (0, 1), counted
    along the stripe axis and across it.

    !!! note
        The paper replaces ∇_y by a directional difference ∇_θ for oblique
        stripes without specifying its discretisation; the rounding to
        these lattice directions follows the authors' code.

    Parameters
    ----------
    angle : float
        Stripe angle θ in degrees.

    Returns
    -------
    tuple of int
        Offset `(along, across)`.
    """
    angle = float(angle)
    if not math.isfinite(angle):
        raise ValueError(f"angle must be finite, got {angle!r}")
    theta = angle % 180.0

    def dist(a: float) -> float:
        d = abs(theta - a) % 180.0
        return min(d, 180.0 - d)

    best = min(range(len(_LATTICE)), key=lambda i: dist(_LATTICE_ANGLES[i]))
    return _LATTICE[best]


def _grad(u: Tensor, rho_z: float, volume: bool) -> list[Tensor]:
    """∇u: forward differences along every axis, ρ_z ∇_z first in 3-D."""
    g = []
    if volume and rho_z > 0:
        g.append(rho_z * lattice_diff(u, (1, 0, 0)))
    g.append(lattice_diff(u, (1, 0)))
    g.append(lattice_diff(u, (0, 1)))
    return g


def general_stripe_objective(
    u: Tensor,
    data: Tensor,
    *,
    mu1: float,
    mu2: float,
    offsets: Sequence[Sequence[int]] = ((1, 0),),
    rho_z: float = 0.0,
    volume: bool = False,
) -> Tensor:
    """Objective of Eq. (1) of Rottmayer et al. (2025), per frame.

        E(u) = μ₁‖∇u‖₂,₁ + ∑ᵢ ‖∇_θᵢ s‖₁ + μ₂‖s‖₁,   s = data − u,

    without the indicator ι_[0,1]ᴺ(u), on frames whose stripes run along
    dim -2 (offsets as in
    [`stripe_offset`][optix.destripe.utils.stripe_offset]).

    Parameters
    ----------
    u : Tensor
        Clean frames `(n, H, W)`, or volumes `(n, Z, H, W)` if `volume`.
    data : Tensor
        Observed frames u₀, of the shape of `u`.
    mu1, mu2 : float
        Weights μ₁ and μ₂.
    offsets : sequence of (int, int)
        Lattice offsets of the stripe differences ∇_θᵢ.
    rho_z : float
        Weight ρ_z of ∇_z in the total variation (volumes only).
    volume : bool
        Whether the frames are 3-D, z first.

    Returns
    -------
    Tensor
        `(n,)`, E for each frame.
    """
    s = data - u
    dims = tuple(range(1, u.ndim))
    sq = sum(g.square() for g in _grad(u, rho_z, volume))
    tv = sq.sqrt().sum(dims)
    stripes = sum(lattice_diff(s, v).abs().sum(dims) for v in offsets)
    return mu1 * tv + stripes + mu2 * s.abs().sum(dims)


# -- solver -------------------------------------------------------------------
def _grad_offsets(
    rho_z: float, volume: bool
) -> list[tuple[tuple[int, ...], float]]:
    """Offsets and weights of the components of ∇, as in `_grad`."""
    out: list[tuple[tuple[int, ...], float]] = []
    if volume and rho_z > 0:
        out.append(((1, 0, 0), rho_z))
    out += [((1, 0), 1.0), ((0, 1), 1.0)]
    return out


def _step(volume: bool, rho_z: float, offsets: Sequence) -> float:
    """τ = σ, with τσ‖K‖² < 1 for K = (∇u, ∇_θᵢ s, s).

    K is block-diagonal in (u, s), so ‖K‖² = max(‖∇‖², ‖I + ∑ᵢ D_vᵢᵀD_vᵢ‖)
    ≤ max(4(2 + ρ_z²), 1 + 4m) for m stripe directions.
    """
    if volume and rho_z > 0:
        step, bound = _STEP_3D, _BOUND_3D
        norm_u = 4.0 * (2.0 + rho_z * rho_z)
    else:
        step, bound = _STEP_2D, _BOUND_2D
        norm_u = 8.0
    # ‖D_v‖² ≤ 4 for every lattice difference
    norm_s = 1.0 + 4.0 * len(offsets)
    norm = max(norm_u, norm_s)
    return step * min(1.0, math.sqrt(bound / norm))


def _gap(
    u0: Tensor,
    u: Tensor,
    b1: list[Tensor],
    b2: list[Tensor],
    b3: Tensor,
    *,
    mu1: float,
    mu2: float,
    offsets: Sequence[tuple[int, int]],
    grads: Sequence[tuple[tuple[int, ...], float]],
    rho_z: float,
    volume: bool,
    step: float,
) -> Tensor:
    """Duality gap E(u) − D(p) per frame, for the duals p = σ b.

    With G the indicator of C = {u + s = u₀, u ∈ [0, 1]ᴺ} and Kᵀp = (a, c),
    a = ∇ᵀp₁, c = ∑ᵢ D_vᵢᵀp₂ᵢ + p₃, the dual is
    D(p) = min_{(u, s) ∈ C} ⟨Kᵀp, (u, s)⟩ = ⟨c, u₀⟩ + ∑ min(0, a − c).
    """
    dims = tuple(range(1, u0.ndim))
    primal = general_stripe_objective(
        u, u0, mu1=mu1, mu2=mu2, offsets=offsets, rho_z=rho_z, volume=volume
    )
    a = torch.zeros_like(u0)
    for (v, w), p in zip(grads, b1, strict=True):
        _diff_adjoint_add(a, p, v, alpha=step * w)
    c = b3 * step
    for v, p in zip(offsets, b2, strict=True):
        _diff_adjoint_add(c, p, v, alpha=step)
    dual = (c * u0).sum(dims) + (a - c).clamp_(max=0.0).sum(dims)
    return primal - dual


def _pdhgmp(
    u0: Tensor,
    *,
    mu1: float,
    mu2: float,
    offsets: Sequence[tuple[int, int]],
    rho_z: float,
    volume: bool,
    box: bool,
    max_iter: int,
    tol: float | None,
) -> Tensor:
    """PDHGMp iterations for Eq. (1) on normalised frames `(n, ...)`.

    The dual variables are scaled by 1/σ (b = p/σ), and all updates are
    done in place on preallocated buffers. The components of b₁, b₂ (and
    their extrapolations) stay zero outside the valid region of their
    difference operator, so only that region is ever written.
    """
    step = _step(volume, rho_z, offsets)
    ts = step * step
    r1, r2, r3 = mu1 / step, 1.0 / step, mu2 / step
    tiny = torch.finfo(u0.dtype).tiny
    grads = _grad_offsets(rho_z, volume)
    kw = dict(mu1=mu1, mu2=mu2, offsets=offsets, rho_z=rho_z, volume=volume)

    u = u0.clone()
    s = torch.zeros_like(u0)
    b1 = [torch.zeros_like(u0) for _ in grads]
    b1bar = [torch.zeros_like(u0) for _ in grads]
    b2 = [torch.zeros_like(u0) for _ in offsets]
    b2bar = [torch.zeros_like(u0) for _ in offsets]
    b3, b3bar = torch.zeros_like(u0), torch.zeros_like(u0)
    work, norm = torch.empty_like(u0), torch.empty_like(u0)
    if tol is not None:
        result = torch.empty_like(u0)
        index = torch.arange(u0.shape[0], device=u0.device)
        # Δ(u₀, 0) = E(u₀) − D(0) = μ₁‖∇u₀‖₂,₁
        gap0 = general_stripe_objective(u0, u0, **kw)

    for it in range(max_iter):
        # primal step: u ← u − τσ∇ᵀb̄₁, s ← s − τσ(∑ᵢ D_vᵢᵀb̄₂ᵢ + b̄₃)
        for (v, w), p in zip(grads, b1bar, strict=True):
            _diff_adjoint_add(u, p, v, alpha=-ts * w)
        s.sub_(b3bar, alpha=ts)
        for v, p in zip(offsets, b2bar, strict=True):
            _diff_adjoint_add(s, p, v, alpha=-ts)
        # projection onto {u + s = u₀, u ∈ [0, 1]ᴺ}: average onto the
        # constraint, then clamp u and move the excess to s; this is the
        # exact projection, as each pixel is then a 1-D problem in u
        torch.sub(u0, u, out=work).sub_(s).mul_(0.5)
        u.add_(work)
        s.add_(work)
        if box:
            torch.clamp(u, 0.0, 1.0, out=work)
            s.add_(u.sub_(work))
            u, work = work, u
        # dual steps: projections onto the balls of the conjugates (radii
        # μ₁/σ, 1/σ, μ₂/σ), into the buffers of the extrapolations, then
        # extrapolation 2bₖ₊₁ − bₖ = lerp(bₖ, bₖ₊₁, 2) into the old ones
        for (v, w), b, bb in zip(grads, b1, b1bar, strict=True):
            _diff_into(bb, u, v)
            if w != 1.0:
                bb.mul_(w)
            bb.add_(b)
        torch.mul(b1bar[0], b1bar[0], out=norm)
        for bb in b1bar[1:]:
            norm.addcmul_(bb, bb)
        norm.sqrt_().clamp_min_(tiny).reciprocal_().mul_(r1).clamp_(max=1.0)
        for b, bb in zip(b1, b1bar, strict=True):
            bb.mul_(norm)
            b.lerp_(bb, 2.0)
        b1, b1bar = b1bar, b1
        for v, b, bb in zip(offsets, b2, b2bar, strict=True):
            _diff_into(bb, s, v)
            bb.add_(b).clamp_(-r2, r2)
            b.lerp_(bb, 2.0)
        b2, b2bar = b2bar, b2
        torch.add(b3, s, out=b3bar).clamp_(-r3, r3)
        b3.lerp_(b3bar, 2.0)
        b3, b3bar = b3bar, b3

        if tol is not None and (it + 1) % _CHECK_EVERY == 0:
            gap = _gap(u0, u, b1, b2, b3, grads=grads, step=step, **kw)
            done = gap <= tol * gap0
            if bool(done.any()):
                result[index[done]] = u[done]
                keep = ~done
                if not bool(keep.any()):
                    return result
                index, gap0 = index[keep], gap0[keep]
                u0, u, s, b3, b3bar, work, norm = (
                    t[keep] for t in (u0, u, s, b3, b3bar, work, norm)
                )
                b1, b1bar, b2, b2bar = (
                    [t[keep] for t in group]
                    for group in (b1, b1bar, b2, b2bar)
                )
    if tol is None:
        return u
    result[index] = u
    return result


def _warn_no_removal(mu1: float, offsets: Sequence[tuple[int, int]]) -> None:
    """Warn when several directions and μ₁ leave the input (nearly) as is."""
    distinct = set(offsets)
    if len(distinct) < 2:
        return
    orthogonal = {(1, 0), (0, 1)} <= distinct
    limit = 1.0 if orthogonal else _MULTI_LIMIT
    if mu1 <= limit:
        what = (
            "the minimiser of Eq. (1) is the input itself"
            if orthogonal
            else "Eq. (1) typically removes little or no stripes"
        )
        warnings.warn(
            f"with several stripe directions and mu1 = {mu1:.3g} <= "
            f"{limit:g}, {what}: each direction's term also penalises the "
            f"other directions' stripes, so removal needs a larger mu1 "
            f"(see general_stripe_remover)",
            UserWarning,
            stacklevel=3,
        )


def _check(
    mu1: object,
    mu2: object,
    angle: object,
    rho_z: object,
    max_iter: object,
    tol: object,
    box: object = True,
) -> list[float]:
    for name, value in (("mu1", mu1), ("mu2", mu2)):
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value!r}")
    angles = (
        [float(angle)]
        if isinstance(angle, int | float)
        else [float(a) for a in angle]
    )
    if not angles:
        raise ValueError("angle must hold at least one direction")
    for a in angles:
        stripe_offset(a)
    if not isinstance(rho_z, int | float) or not 0 <= rho_z <= 1:
        raise ValueError(f"rho_z must lie in [0, 1], got {rho_z!r}")
    if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter < 0:
        raise ValueError(f"max_iter must be an integer >= 0, got {max_iter!r}")
    if tol is not None and (
        not isinstance(tol, int | float) or not math.isfinite(tol) or tol <= 0
    ):
        raise ValueError(f"tol must be None or > 0, got {tol!r}")
    if tol is not None and not box:
        raise ValueError(
            "tol needs box=True: the duality gap is unbounded without the "
            "indicator of [0, 1]"
        )
    return angles


def general_stripe_remover(
    data: Tensor,
    *,
    mu1: float = 1 / 3,
    mu2: float = 1 / 300,
    axis: int = -2,
    angle: float | Sequence[float] = 0.0,
    rho_z: float = 0.0,
    ndim: int = 2,
    normalize: bool = True,
    box: bool = True,
    max_iter: int = 25000,
    tol: float | None = None,
) -> Tensor:
    """Remove stripes with the variational model of Rottmayer et al. (2025).

    Each frame u₀ (or volume, if `ndim` is 3) is split into a clean image
    u and stripes s by solving Eq. (1) of the paper,

        argmin_{u + s = u₀}  μ₁‖∇u‖₂,₁ + ι_[0,1]ᴺ(u) + ‖∇_θ s‖₁ + μ₂‖s‖₁,

        ‖∇u‖₂,₁ = ∑ √((∇ₓu)² + (∇ᵧu)² + (ρ_z ∇_z u)²)          (Eq. (2)).

    The total variation (Eq. (2)) favours clean images with few strong
    edges, ‖∇_θ s‖₁ stripes that vary little along their direction θ,
    μ₂‖s‖₁ sparse stripes, and the indicator ι_[0,1]ᴺ keeps u in the value
    range of the input. μ₁ sets the strength of the removal and μ₂ the
    precaution towards image structures (Section 2.1.1). Several stripe
    directions θᵢ add one term ‖∇_θᵢ s‖₁ each, all on the same s, as
    written in Section 2.1.1. Notation: the paper's u₀ is `data`, rescaled
    to [0, 1] when `normalize`.

    The problem is solved with the primal–dual hybrid gradient method with
    extrapolation of the dual variable (PDHGMp, Burger et al., 2016) for
    `max_iter` iterations, from u = u₀, s = 0. Differences are forward
    differences with a zero (Neumann) boundary; the oblique ∇_θ uses the
    lattice offsets of [`stripe_offset`][optix.destripe.utils.stripe_offset].
    Frames are processed independently, all frames of a batch at once.

    !!! note "Settings not specified in the paper"
        The paper names the algorithm (PDHGMp) and the problem, but not the
        following, which follow the authors' code
        ([General-Stripe-Removal](https://github.com/NiklasRottmayer/General-Stripe-Removal),
        MIT licence): the initialisation, the forward differences with zero
        boundary, the rounding of θ to lattice directions, and the step
        sizes τ = σ = 0.35 in 2-D and 0.28 with ρ_z > 0 (reduced so that
        τσ‖K‖² < 1 when several directions are given).

        - **Step sizes and iterations.** The balanced steps τ = σ are
          those of the authors' current code. The code used for the
          results of the paper weighted the stripe term by 30 (it solved
          30 × Eq. (1)) with the same τ = σ = 0.35, which is PDHGMp on
          Eq. (1) with τ = 10.5 and σ = 0.35/30 ≈ 0.0117; the 25000
          iterations of the paper (Section 3) refer to that balance. Both
          converge to a minimiser of Eq. (1), along different paths. With
          the balanced steps, the authors now suggest about 1000
          iterations, the `recommended` value of
          [`GeneralStripeRemover`][optix.destripe.GeneralStripeRemover];
          `max_iter` = 25000 is kept as the faithful default.
        - **Normalisation.** `normalize` rescales each frame (or volume)
          to [0, 1] by its own minimum and maximum, the range Eq. (1)
          assumes, and maps the result back to the input range. This is a
          choice of optix: the authors rescale the whole image or stack
          once.
        - **ρ_z.** ρ_z weights ∇_z in both the forward operator ∇ and its
          adjoint ∇ᵀ, as Eq. (2) implies. The authors' code applies it in
          the adjoint only, so 3-D results with 0 < ρ_z < 1 differ from
          theirs.
        - **Negative angles.** The authors' code mirrors the frame across
          the stripe axis and solves at the positive angle, which turns
          the total variation into a backward difference across the
          stripes. optix keeps forward differences for every angle, which
          gives a slightly different result with a lower objective.
          Mirroring the input (and the output) and passing the positive
          angle reproduces the authors' result.

    !!! note "Defaults"
        μ₁ = 1/3 and μ₂ = 1/300. The Discussion of the paper prints
        μ₁ = 1/30 and μ₂ = 1/300. The authors' code at the time of
        publication (before the commit "Improved GSR" of 29 July 2025)
        weights ‖∇_y s‖₁ by λ = 30 with defaults μ = (10, 0.1), and its
        processing script states that "compared to the paper mu is
        rescaled by a factor of 30": μ₁ = 10/30 = 1/3 and μ₂ = 0.1/30 =
        1/300 in the units of Eq. (1). μ₂ agrees with the Discussion and
        μ₁ does not, so 1/30 is read as a typographical error for 1/3; the
        authors' current code defaults to the same values (0.33, 0.003).
        Increase μ₁ for stronger removal, μ₂ for more caution. Each
        iteration costs O(|Ω|·m) for |Ω| pixels and m stripe directions: on
        one CPU thread, about 1 ms at 256 × 256 and 25 ms at 1024 × 1024
        per iteration in float32 for one direction, i.e. about 10 min for
        a 1024 × 1024 frame at 25000 iterations and 25 s at 1000. On
        synthetic 96 × 96 frames, the relative duality gap (see `tol`) is
        of order 10⁻¹ to 10⁻² after 1000 iterations and 10⁻³ to 10⁻⁴
        after 25000.

    !!! warning "Several directions"
        With a shared s, each term ‖∇_θᵢ s‖₁ also penalises the stripes of
        the other directions, with weight 1, so stripes are removed only
        for μ₁ of order 1, larger than for a single direction.

        - For the orthogonal pair 0° and 90°, D_y and D_x are the
          components of ∇, so ‖D_y s‖₁ + ‖D_x s‖₁ ≥ ‖∇s‖₂,₁ and, for
          μ₁ ≤ 1, E(u) ≥ μ₁‖∇u₀‖₂,₁ + μ₂‖s‖₁: u = u₀ is the unique
          minimiser and nothing is removed.
        - For other pairs the threshold is lower and depends on the data.
          On synthetic frames, removal starts between μ₁ = 1/2 and 0.6
          for 0° + 45° and for ±45°; at μ₁ = 0.7, 0° + 45° removes both
          families of stripes better than either direction alone.

        A `UserWarning` is emitted with several directions when μ₁ ≤ 1 for
        a 0°/90° pair, or μ₁ ≤ 1/2 otherwise.

    !!! note "Extensions"
        `tol` stops each frame when the duality gap Δ(uₖ, pₖ) = E(uₖ) −
        D(pₖ) of Eq. (1) falls to `tol` · Δ(u₀, p₀), its initial value
        μ₁‖∇u₀‖₂,₁ (checked every 10 iterations), the same relative
        criterion as `tol` of `optix.destripe.VSNR`. It needs the
        indicator ι_[0,1]ᴺ (`box=True`), without which the dual is
        unbounded. `box=False` drops ι_[0,1]ᴺ. Both depart from the paper.

    !!! references "References"
        1. Rottmayer N, Redenbach C, Fahrbach FO. *A universal and
           effective variational method for destriping: application to
           light-sheet microscopy, FIB-SEM, and remote sensing images.*
           Opt Express **33**(3):5800–5809, 2025.
           [doi:10.1364/OE.542868](https://doi.org/10.1364/OE.542868)
        2. Burger M, Sawatzky A, Steidl G. *First order algorithms in
           variational image processing.* In: Glowinski R, Osher SJ, Yin W
           (eds), Splitting Methods in Communication, Imaging, Science, and
           Engineering, pp. 345–407. Springer, 2016.
           [doi:10.1007/978-3-319-41589-5_10](https://doi.org/10.1007/978-3-319-41589-5_10)

        ??? quote "BibTeX"
            ```bibtex
            @article{rottmayer2025universal,
              author  = {Rottmayer, Niklas and Redenbach, Claudia and
                         Fahrbach, Florian O.},
              title   = {A universal and effective variational method for
                         destriping: application to light-sheet microscopy,
                         {FIB-SEM}, and remote sensing images},
              journal = {Optics Express},
              volume  = {33}, number = {3}, pages = {5800--5809},
              year    = {2025}, doi = {10.1364/OE.542868}
            }
            @incollection{burger2016first,
              author    = {Burger, Martin and Sawatzky, Alex and
                           Steidl, Gabriele},
              title     = {First order algorithms in variational image
                           processing},
              booktitle = {Splitting Methods in Communication, Imaging,
                           Science, and Engineering},
              editor    = {Glowinski, Roland and Osher, Stanley J. and
                           Yin, Wotao},
              pages     = {345--407}, publisher = {Springer},
              year      = {2016}, doi = {10.1007/978-3-319-41589-5_10}
            }
            ```

    Parameters
    ----------
    data : Tensor
        Frames `(..., H, W)`, or volumes `(..., Z, H, W)` if `ndim` is 3
        (floating point; other dtypes are converted to the configured
        dtype).
    mu1 : float, default 1/3
        Weight μ₁ > 0 of the total variation: strength of the removal.
    mu2 : float, default 1/300
        Weight μ₂ > 0 of ‖s‖₁: precaution towards image structures.
    axis : {-2, -1}
        In-plane axis along which the stripes run (∇_y of the paper):
        -2 for vertical stripes in an `(H, W)` frame, the case of the
        paper, -1 for horizontal ones.
    angle : float or sequence of float, default 0
        Stripe direction θ in degrees, from `axis` towards the other
        in-plane axis (see
        [`stripe_offset`][optix.destripe.utils.stripe_offset]); several
        values for multi-directional destriping (see the warning above).
    rho_z : float, default 0
        Weight ρ_z ∈ [0, 1] of ∇_z (∇_z* = ρ_z ∇_z) in the total
        variation of volumes: 0 decouples the slices (2-D processing, as in
        all results of the paper), values below 1 trust the z axis less
        (e.g. a lower axial resolution).
    ndim : {2, 3}
        Number of trailing spatial dims of a frame; with 3, the first one
        is z.
    normalize : bool, default True
        Rescale each frame to [0, 1] by its minimum and maximum before
        solving, and back afterwards (a choice of optix, see above). With
        False, `data` must already lie in [0, 1].
    box : bool, default True
        Keep the indicator ι_[0,1]ᴺ(u) of Eq. (1). False is an extension.
    max_iter : int, default 25000
        Number of PDHGMp iterations (an upper bound when `tol` is set);
        the authors suggest about 1000 with the step sizes used here (see
        above).
    tol : float, optional
        Relative duality gap Δ(uₖ, pₖ)/Δ(u₀, p₀) at which a frame stops
        (extension, needs `box=True`); default: run all `max_iter`
        iterations, as in the paper.

    Returns
    -------
    Tensor
        Destriped frames u, of the shape of `data`.
    """
    angles = _check(mu1, mu2, angle, rho_z, max_iter, tol, box)
    _warn_no_removal(float(mu1), [stripe_offset(a) for a in angles])
    return _remove(
        data,
        mu1=mu1,
        mu2=mu2,
        axis=axis,
        angles=angles,
        rho_z=rho_z,
        ndim=ndim,
        normalize=normalize,
        box=box,
        max_iter=max_iter,
        tol=tol,
    )


def _remove(
    data: Tensor,
    *,
    mu1: float,
    mu2: float,
    axis: int,
    angles: list[float],
    rho_z: float,
    ndim: int,
    normalize: bool,
    box: bool,
    max_iter: int,
    tol: float | None,
) -> Tensor:
    """`general_stripe_remover` on checked parameters, without warnings."""
    if ndim not in (2, 3) or data.ndim < ndim:
        raise ValueError(f"ndim must be 2 or 3 and <= data.ndim, got {ndim}")
    if axis not in (-2, -1, data.ndim - 2, data.ndim - 1):
        raise ValueError(f"axis must be -2 or -1, got {axis}")
    axis = axis - data.ndim if axis >= 0 else axis
    if ndim == 2 and rho_z != 0:
        raise ValueError("rho_z must be 0 for 2-D frames (ndim=2)")
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    shape = data.shape
    x = data.reshape(-1, *shape[data.ndim - ndim :])
    if axis == -1:
        x = x.transpose(-2, -1)
    offsets = [stripe_offset(a) for a in angles]
    dims = tuple(range(1, x.ndim))
    if normalize:
        lo = x.amin(dims, keepdim=True)
        scale = x.amax(dims, keepdim=True) - lo
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        u0 = (x - lo) / scale
    else:
        u0 = x
    u = _pdhgmp(
        u0.contiguous(),
        mu1=float(mu1),
        mu2=float(mu2),
        offsets=offsets,
        rho_z=float(rho_z),
        volume=ndim == 3,
        box=bool(box),
        max_iter=int(max_iter),
        tol=None if tol is None else float(tol),
    )
    if normalize:
        u = u * scale + lo
    if axis == -1:
        u = u.transpose(-2, -1)
    return u.reshape(shape)


class GeneralStripeRemover(Estimator):
    """Variational stripe removal with intuitive parameters.

    The method of Rottmayer et al. (2025), applied to every frame (or
    volume) independently: see
    [`general_stripe_remover`][optix.destripe.general_stripe_remover] for
    the model, Eq. (1), and the solver. It has no parameters to fit:
    [`fit`][optix.destripe.GeneralStripeRemover.fit] only records the
    layout, and [`apply`][optix.Estimator.apply] streams the frames in
    batches, so arrays larger than memory (zarr, memmap) can be processed.

    With two spatial axes in the layout, frames are 2-D, as in all the
    results of the paper. With three, the first spatial axis (in layout
    order) is z and each volume is solved as a whole, ∇_z being weighted
    by ρ_z (Section 2.1.1).

    For a negative angle, the authors' code mirrors the frame and solves
    at the positive angle; optix keeps forward differences for every angle,
    which gives a slightly different result with a lower objective. Mirror
    the input (and the output) to reproduce the authors' result (see
    [`general_stripe_remover`][optix.destripe.general_stripe_remover]).

    !!! note
        μ₁ and μ₂ are to be tuned to the data (Section 2.1.1), and any
        value is faithful to the paper. The defaults μ₁ = 1/3, μ₂ = 1/300
        are those of the authors' code at the time of publication; the
        Discussion of the paper prints μ₁ = 1/30, read as a typographical
        error (see
        [`general_stripe_remover`][optix.destripe.general_stripe_remover]).
        The paper runs 25000 PDHGMp iterations (Section 3), the faithful
        default; `faithful=False` selects 1000, which the authors suggest
        for the step sizes used here. Several stripe directions need a
        larger μ₁ (of order 1) to remove anything; a `UserWarning` flags
        settings that remove (nearly) nothing.

    !!! references "References"
        1. Rottmayer N, Redenbach C, Fahrbach FO. *A universal and
           effective variational method for destriping: application to
           light-sheet microscopy, FIB-SEM, and remote sensing images.*
           Opt Express **33**(3):5800–5809, 2025.
           [doi:10.1364/OE.542868](https://doi.org/10.1364/OE.542868)
        2. Burger M, Sawatzky A, Steidl G. *First order algorithms in
           variational image processing.* In: Glowinski R, Osher SJ, Yin W
           (eds), Splitting Methods in Communication, Imaging, Science, and
           Engineering, pp. 345–407. Springer, 2016.
           [doi:10.1007/978-3-319-41589-5_10](https://doi.org/10.1007/978-3-319-41589-5_10)

        ??? quote "BibTeX"
            ```bibtex
            @article{rottmayer2025universal,
              author  = {Rottmayer, Niklas and Redenbach, Claudia and
                         Fahrbach, Florian O.},
              title   = {A universal and effective variational method for
                         destriping: application to light-sheet microscopy,
                         {FIB-SEM}, and remote sensing images},
              journal = {Optics Express},
              volume  = {33}, number = {3}, pages = {5800--5809},
              year    = {2025}, doi = {10.1364/OE.542868}
            }
            @incollection{burger2016first,
              author    = {Burger, Martin and Sawatzky, Alex and
                           Steidl, Gabriele},
              title     = {First order algorithms in variational image
                           processing},
              booktitle = {Splitting Methods in Communication, Imaging,
                           Science, and Engineering},
              editor    = {Glowinski, Roland and Osher, Stanley J. and
                           Yin, Wotao},
              pages     = {345--407}, publisher = {Springer},
              year      = {2016}, doi = {10.1007/978-3-319-41589-5_10}
            }
            ```
    """

    tags = Tags(
        targets=("stripes",),
        streaming="per-frame",
        modality=("light-sheet", "fib-sem", "remote-sensing"),
        reference="doi:10.1364/OE.542868",
    )
    paper = {
        "mu1": 1 / 3,
        "mu2": 1 / 300,
        "axis": -2,
        "angle": 0.0,
        "rho_z": 0.0,
        "normalize": True,
        "box": True,
        "max_iter": 25000,
        "tol": None,
    }
    # the authors' suggestion for the balanced step sizes τ = σ
    recommended = dict(paper, max_iter=1000)
    # free parameters of the published method: any value is faithful
    _free = ("mu1", "mu2", "axis", "angle", "rho_z", "normalize", "max_iter")

    def __init__(
        self,
        *,
        mu1: float | None = None,
        mu2: float | None = None,
        axis: int | str | None = None,
        angle: float | Sequence[float] | None = None,
        rho_z: float | None = None,
        normalize: bool | None = None,
        box: bool | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the method.

        Parameters
        ----------
        mu1 : float, default 1/3
            Weight μ₁ > 0 of the total variation: strength of the removal.
        mu2 : float, default 1/300
            Weight μ₂ > 0 of ‖s‖₁: precaution towards image structures.
        axis : int or str
            Spatial axis along which the stripes run: a position among the
            spatial axes of the layout (0 or -2: the first of two, 1 or
            -1: the second), or an axis label of the layout (e.g. `"y"`).
            With three spatial axes it must be one of the last two (the
            first is z). Paper: the first of two spatial axes (vertical
            stripes in `(y, x)` frames).
        angle : float or sequence of float, default 0
            Stripe direction θ in degrees from `axis` towards the other
            in-plane axis; several values for multi-directional
            destriping (they need a larger `mu1`).
        rho_z : float, default 0
            Weight ρ_z ∈ [0, 1] of ∇_z, for three spatial axes.
        normalize : bool, default True
            Rescale each frame to [0, 1] by its own range (and back), a
            choice of optix.
        box : bool, default True
            Keep ι_[0,1]ᴺ(u) (False is an extension).
        max_iter : int, default 25000
            Number of PDHGMp iterations (recommended: 1000).
        tol : float, optional
            Per-frame stopping tolerance on the relative duality gap, as
            `tol` of `optix.destripe.VSNR` (extension, needs
            `box=True`; default: none).
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        if not isinstance(o["angle"], int | float):
            o["angle"] = [float(a) for a in o["angle"]]
        self.is_faithful = all(
            _same(value, self.paper.get(name))
            for name, value in o.items()
            if name not in self._free
        )
        angles = _check(
            o["mu1"],
            o["mu2"],
            o["angle"],
            o["rho_z"],
            o["max_iter"],
            o["tol"],
            o["box"],
        )
        _warn_no_removal(float(o["mu1"]), [stripe_offset(a) for a in angles])

    def fit(
        self,
        data: ArrayLike | None = None,
        *,
        layout: Layout | None = None,
        **kwargs: object,
    ) -> Self:
        """No-op: the method has no parameters to estimate.

        Parameters
        ----------
        data : array-like, optional
            Unused (its layout, if labelled, is recorded).
        layout : Layout, optional
            Recorded for [`apply`][optix.Estimator.apply].
        **kwargs
            Ignored (`init`, `callback`, `verbose`).

        Returns
        -------
        GeneralStripeRemover
            self
        """
        if data is None:
            self._set_layout(layout)
            return self
        return super().fit(data, layout=layout)

    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        pass

    def _frame_axis(self, layout: Layout, ndim: int) -> tuple[int, int]:
        """Number of spatial dims and the stripe axis as a frame dim."""
        _, _, spatial = layout.positions(ndim)
        n = len(spatial)
        if n not in (2, 3):
            raise ValueError(
                f"GeneralStripeRemover needs 2 or 3 spatial axes, the "
                f"layout has {n}"
            )
        axis = self.options["axis"]
        if isinstance(axis, str):
            if layout.axes is None or axis not in layout.axes:
                raise ValueError(
                    f"axis {axis!r} is not a label of the layout {layout!r}"
                )
            pos = layout.axes.index(axis)
            if pos not in spatial:
                raise ValueError(f"axis {axis!r} is not a spatial axis")
            frame = spatial.index(pos) - n
        else:
            if isinstance(axis, bool) or axis not in range(-n, n):
                raise ValueError(f"axis {axis!r} is out of range")
            frame = axis - n if axis >= 0 else axis
        if frame not in (-2, -1):
            raise ValueError(
                "the stripe axis must be one of the last two spatial axes "
                "(the first of three is z)"
            )
        return n, frame

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        o = self.options
        angles = _check(
            o["mu1"],
            o["mu2"],
            o["angle"],
            o["rho_z"],
            o["max_iter"],
            o["tol"],
            o["box"],
        )
        layout = self._layout_of(data)
        ndim, axis = self._frame_axis(layout, len(data.shape))

        def fn(chunk: object, batch: Tensor) -> Tensor:
            return _remove(
                batch,
                mu1=o["mu1"],
                mu2=o["mu2"],
                axis=axis,
                angles=angles,
                rho_z=o["rho_z"] if ndim == 3 else 0.0,
                ndim=ndim,
                normalize=o["normalize"],
                box=o["box"],
                max_iter=o["max_iter"],
                tol=o["tol"],
            )

        if ndim == 2 and o["rho_z"] != 0:
            raise ValueError("rho_z must be 0 with two spatial axes")
        # peak frame-sized buffers: input, normalised copy and output, u₀,
        # u, s, the dual variables and their extrapolations, scratch and
        # the temporaries of the duality gap, rounded up
        n_grad = 3 if ndim == 3 and o["rho_z"] > 0 else 2
        overhead = 11.0 + 5.0 * n_grad + 3.0 * len(angles)
        return self._apply_frames(
            data, out, fn, ndim=ndim, overhead=overhead, contiguous=True
        )
