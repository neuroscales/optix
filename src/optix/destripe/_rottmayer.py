"""Variational stripe removal of Rottmayer, Redenbach and Fahrbach (2025)."""

import math
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
    offset = tuple(int(o) for o in offset)
    dst, src = _slices(offset, x.shape[x.ndim - len(offset) :])
    out = torch.zeros_like(x)
    out[dst] = x[src] - x[dst]
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
    offset = tuple(int(o) for o in offset)
    dst, src = _slices(offset, p.shape[p.ndim - len(offset) :])
    out = torch.zeros_like(p)
    out[dst] = -p[dst]
    out[src] += p[dst]
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


def _grad_adjoint(p: list[Tensor], rho_z: float, volume: bool) -> Tensor:
    out = lattice_diff_adjoint(p[-2], (1, 0))
    out += lattice_diff_adjoint(p[-1], (0, 1))
    if volume and rho_z > 0:
        out += rho_z * lattice_diff_adjoint(p[0], (1, 0, 0))
    return out


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
def _step(volume: bool, rho_z: float, offsets: Sequence) -> float:
    """τ = σ, with τσ‖K‖² < 1 for K = (∇u, ∇_θᵢ s, s)."""
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
    """PDHGMp iterations for Eq. (1) on normalised frames `(n, ...)`."""
    step = _step(volume, rho_z, offsets)
    ts = step * step
    r1, r2, r3 = mu1 / step, 1.0 / step, mu2 / step
    tiny = torch.finfo(u0.dtype).tiny
    dims = tuple(range(1, u0.ndim))

    u = u0.clone()
    s = torch.zeros_like(u0)
    n_grad = 3 if volume and rho_z > 0 else 2
    b1 = [torch.zeros_like(u0) for _ in range(n_grad)]
    b2 = [torch.zeros_like(u0) for _ in offsets]
    b3 = torch.zeros_like(u0)
    b1bar, b2bar, b3bar = list(b1), list(b2), b3
    result = torch.empty_like(u0)
    index = torch.arange(u0.shape[0], device=u0.device)

    for it in range(max_iter):
        check = tol is not None and (it + 1) % _CHECK_EVERY == 0
        if check:
            previous = u
        # primal step, then projection onto {u + s = u₀, u ∈ [0, 1]ᴺ}
        u = u - ts * _grad_adjoint(b1bar, rho_z, volume)
        ks = b3bar.clone()
        for v, p in zip(offsets, b2bar, strict=True):
            ks += lattice_diff_adjoint(p, v)
        s = s - ts * ks
        half = 0.5 * (u0 - u - s)
        u = u + half
        s = s + half
        if box:
            clamped = u.clamp(0.0, 1.0)
            s = s + (u - clamped)
            u = clamped
        # dual steps (dual variables scaled by 1/σ): projections onto the
        # balls of the conjugates, then extrapolation
        g = _grad(u, rho_z, volume)
        v1 = [b + gi for b, gi in zip(b1, g, strict=True)]
        norm = sum(v.square() for v in v1).sqrt().clamp_min(tiny)
        shrink = (r1 / norm).clamp(max=1.0)
        new1 = [v * shrink for v in v1]
        new2 = [
            (b + lattice_diff(s, v)).clamp(-r2, r2)
            for b, v in zip(b2, offsets, strict=True)
        ]
        new3 = (b3 + s).clamp(-r3, r3)
        b1bar = [2 * n - o for n, o in zip(new1, b1, strict=True)]
        b2bar = [2 * n - o for n, o in zip(new2, b2, strict=True)]
        b3bar = 2 * new3 - b3
        b1, b2, b3 = new1, new2, new3

        if check:
            change = (u - previous).square().sum(dims).sqrt()
            size = u.square().sum(dims).sqrt().clamp_min(tiny)
            done = change <= tol * size
            if bool(done.any()):
                result[index[done]] = u[done]
                keep = ~done
                if not bool(keep.any()):
                    return result
                index = index[keep]
                u0, u, s, b3, b3bar = (t[keep] for t in (u0, u, s, b3, b3bar))
                b1, b1bar, b2, b2bar = (
                    [t[keep] for t in group]
                    for group in (b1, b1bar, b2, b2bar)
                )
    result[index] = u
    return result


def _check(
    mu1: object,
    mu2: object,
    angle: object,
    rho_z: object,
    max_iter: object,
    tol: object,
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
    return angles


def general_stripe_remover(
    data: Tensor,
    *,
    mu1: float = 1 / 30,
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
    written in Section 2.1.1; each term then also penalises the stripes of
    the other directions across them. Notation: the paper's u₀ is `data`,
    rescaled to [0, 1] when `normalize`.

    The problem is solved with the primal–dual hybrid gradient method with
    extrapolation of the dual variable (PDHGMp, Burger et al., 2016) for
    `max_iter` iterations, from u = u₀, s = 0. Differences are forward
    differences with a zero (Neumann) boundary; the oblique ∇_θ uses the
    lattice offsets of [`stripe_offset`][optix.destripe.utils.stripe_offset].
    Frames are processed independently, all frames of a batch at once.

    !!! note "Settings not specified in the paper"
        The paper names the algorithm (PDHGMp) and the problem, but not the
        following, which follow the authors' code (MIT licence):
        the step sizes τ = σ = 0.35 in 2-D and 0.28 with ρ_z > 0 (reduced
        so that τσ‖K‖² < 1 when several directions are given), the
        initialisation, the forward differences with zero boundary, and the
        rounding of θ to lattice directions. The rescaling of each frame
        to [0, 1] by its minimum and maximum (`normalize`) is how the
        authors' scripts bring data into the range [0, 1] that Eq. (1)
        assumes; the result is mapped back to the input range.

    !!! note "Defaults"
        μ₁ = 1/30 and μ₂ = 1/300 are the default values stated in the
        Discussion of the paper. The figures use μ₁ between 1/10 and 2/5
        (μ₁ = 1/3, μ₂ = 1/300 for the synthetic data of Fig. 2), and the
        authors' code defaults to μ₁ = 0.33, μ₂ = 0.003: increase μ₁ for
        stronger removal. The paper runs 25000 iterations "to obtain a
        sufficient approximation of the optimum" (Section 3), the default
        here. Each iteration costs O(N) for N pixels; fewer iterations, or
        `tol`, trade accuracy of the minimiser for speed.

    !!! note "Extensions"
        `tol` stops each frame when the relative change of u,
        ‖uₖ − uₖ₋₁‖ / ‖uₖ‖, falls below it (checked every 10 iterations);
        `box=False` drops ι_[0,1]ᴺ. Both depart from the paper.

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
    mu1 : float, default 1/30
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
        values for multi-directional destriping.
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
        solving, and back afterwards. With False, `data` must already lie
        in [0, 1].
    box : bool, default True
        Keep the indicator ι_[0,1]ᴺ(u) of Eq. (1). False is an extension.
    max_iter : int, default 25000
        Number of PDHGMp iterations (an upper bound when `tol` is set).
    tol : float, optional
        Relative change of u below which a frame stops (extension);
        default: run all `max_iter` iterations, as in the paper.

    Returns
    -------
    Tensor
        Destriped frames u, of the shape of `data`.
    """
    angles = _check(mu1, mu2, angle, rho_z, max_iter, tol)
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

    !!! note
        μ₁ and μ₂ are to be tuned to the data (Section 2.1.1); the defaults
        μ₁ = 1/30, μ₂ = 1/300 are those of the Discussion of the paper, and
        any value is faithful to the paper. The figures use μ₁ between 1/10
        and 2/5. The paper runs 25000 PDHGMp iterations (Section 3).

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
        "mu1": 1 / 30,
        "mu2": 1 / 300,
        "axis": -2,
        "angle": 0.0,
        "rho_z": 0.0,
        "normalize": True,
        "box": True,
        "max_iter": 25000,
        "tol": None,
    }
    recommended = dict(paper)
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
        mu1 : float, default 1/30
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
            destriping.
        rho_z : float, default 0
            Weight ρ_z ∈ [0, 1] of ∇_z, for three spatial axes.
        normalize : bool, default True
            Rescale each frame to [0, 1] (and back).
        box : bool, default True
            Keep ι_[0,1]ᴺ(u) (False is an extension).
        max_iter : int, default 25000
            Number of PDHGMp iterations.
        tol : float, optional
            Per-frame stopping tolerance on the relative change of u
            (extension; default: none).
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
        _check(
            o["mu1"], o["mu2"], o["angle"], o["rho_z"], o["max_iter"], o["tol"]
        )

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
        _check(
            o["mu1"], o["mu2"], o["angle"], o["rho_z"], o["max_iter"], o["tol"]
        )
        layout = self._layout_of(data)
        ndim, axis = self._frame_axis(layout, len(data.shape))

        def fn(chunk: object, batch: Tensor) -> Tensor:
            return general_stripe_remover(
                batch,
                mu1=o["mu1"],
                mu2=o["mu2"],
                axis=axis,
                angle=o["angle"],
                rho_z=o["rho_z"] if ndim == 3 else 0.0,
                ndim=ndim,
                normalize=o["normalize"],
                box=o["box"],
                max_iter=o["max_iter"],
                tol=o["tol"],
            )

        if ndim == 2 and o["rho_z"] != 0:
            raise ValueError("rho_z must be 0 with two spatial axes")
        # u₀, u, s, the dual variables and their extrapolations
        overhead = 10.0 + 4.0 * ndim + 2.0 * len(_check_angles(o["angle"]))
        return self._apply_frames(
            data, out, fn, ndim=ndim, overhead=overhead, contiguous=True
        )


def _check_angles(angle: object) -> list[float]:
    return [float(angle)] if isinstance(angle, int | float) else list(angle)
