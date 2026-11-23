"""Variational stationary noise remover (VSNR) of Fehrenbach et al. (2012)."""

import math
import warnings
from collections.abc import Callable, Sequence
from typing import NamedTuple, Self

import torch
from torch import Tensor

from .._config import resolve_backend
from .._estimator import Estimator, Tags
from ..io import ArrayLike, Layout, WritableArrayLike

__all__ = [
    "VSNR",
    "VSNRSolution",
    "dirac",
    "gaussian_stripe",
    "line",
    "operator_norm",
    "vsnr",
    "vsnr_solve",
]

PRIORS = ("l1", "l2", "linf")


# -- filters ------------------------------------------------------------------
def _centred_grid(
    shape: Sequence[int], dtype: torch.dtype | None, device: object
) -> tuple[Tensor, Tensor]:
    h, w = (int(s) for s in shape)
    if h < 1 or w < 1:
        raise ValueError(f"shape must be positive, got {tuple(shape)}")
    x = torch.arange(h, dtype=dtype, device=device) - h // 2
    y = torch.arange(w, dtype=dtype, device=device) - w // 2
    return torch.meshgrid(x, y, indexing="ij")


def dirac(
    shape: Sequence[int] = (1, 1),
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Dirac filter ψ = δ: white noise in the VSNR model.

    With ψ = δ, the component λ ∗ ψ = λ is white noise (Section III of
    Fehrenbach et al. 2012), e.g. the Gaussian noise of the camera.

    Parameters
    ----------
    shape : (int, int), default `(1, 1)`
        Kernel shape; the impulse is at the centre `(h // 2, w // 2)`.
    dtype, device : optional

    Returns
    -------
    Tensor
        `shape`.
    """
    x, _ = _centred_grid(shape, dtype, device)
    out = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    out[shape[0] // 2, shape[1] // 2] = 1
    return out


def _frame_axis(axis: object, layout: Layout | None) -> int:
    """Frame axis (0 or 1) of `axis`: a frame position or a layout label."""
    if isinstance(axis, str):
        if layout is None or layout.axes is None:
            raise ValueError(
                f"axis {axis!r} is a label: pass a labelled `layout`"
            )
        if axis not in layout.axes:
            raise ValueError(
                f"axis {axis!r} is not a label of the layout {layout!r}"
            )
        _, _, spatial = layout.positions(len(layout.axes))
        pos = layout.axes.index(axis)
        if len(spatial) != 2 or pos not in spatial:
            raise ValueError(
                f"axis {axis!r} is not one of the two spatial axes of "
                f"{layout!r}"
            )
        return spatial.index(pos)
    if isinstance(axis, bool) or axis not in (0, 1, -2, -1):
        raise ValueError(f"axis must be 0, 1, -2, -1 or a label, got {axis!r}")
    return int(axis) % 2


def line(
    shape: Sequence[int],
    axis: int | str = -2,
    *,
    layout: Layout | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Straight-line filter: stripes of one pixel width.

    ψ = 1 on the line through the kernel centre along `axis`, 0
    elsewhere. With a kernel as long as the frame, λ ∗ ψ is made of
    stripes that run over the whole frame (Section VI-A, Table I of
    Fehrenbach et al. 2012).

    Parameters
    ----------
    shape : (int, int)
        Kernel shape; use the frame shape for stripes over the whole
        frame.
    axis : int or str, default -2
        Frame axis along which the line runs: a position among the two
        spatial axes (0 or -2: the first, 1 or -1: the second), or an
        axis label of `layout` (e.g. `"y"`).
    layout : Layout, optional
        Labelled layout in which a label `axis` is looked up; the
        filter is over its two spatial axes, in layout order.
    dtype, device : optional

    Returns
    -------
    Tensor
        `shape`.
    """
    frame = _frame_axis(axis, layout)
    x, y = _centred_grid(shape, dtype, device)
    on = y == 0 if frame == 0 else x == 0
    return on.to(x.dtype)


def gaussian_stripe(
    shape: Sequence[int],
    sigma: Sequence[float],
    angle: float = 0.0,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Anisotropic Gaussian stripe filter of Section III-B.

        ψ(x, y) = exp(−x′²/σₓ² − y′²/σᵧ²),
        x′ = x cos θ + y sin θ,  y′ = −x sin θ + y cos θ,

    with x the first and y the second frame axis, both centred on the
    kernel centre `(h // 2, w // 2)`. θ = 0 and σᵧ > σₓ give stripes
    along the second axis, σₓ > σᵧ stripes along the first. The peak
    value is 1, as in the formula of the paper.

    Parameters
    ----------
    shape : (int, int)
        Kernel shape (at most the frame shape).
    sigma : (float, float)
        Widths (σₓ, σᵧ) > 0, in pixels.
    angle : float, default 0
        Orientation θ, in degrees.
    dtype, device : optional

    Returns
    -------
    Tensor
        `shape`.
    """
    sx, sy = (float(s) for s in sigma)
    if not (sx > 0 and sy > 0 and math.isfinite(sx) and math.isfinite(sy)):
        raise ValueError(f"sigma must be finite and > 0, got {sigma!r}")
    x, y = _centred_grid(shape, dtype, device)
    angle = float(angle)
    if not math.isfinite(angle):
        raise ValueError(f"angle must be finite, got {angle!r}")
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    xr = x * c + y * s
    yr = -x * s + y * c
    return torch.exp(-((xr / sx).square() + (yr / sy).square()))


def _as_filters(filters: object) -> list[Tensor]:
    if isinstance(filters, Tensor):
        if filters.ndim == 2:
            return [filters]
        if filters.ndim == 3:
            return list(filters.unbind(0))
        raise ValueError(
            f"filters must be (h, w) or (m, h, w), got {tuple(filters.shape)}"
        )
    out = [torch.as_tensor(f) for f in filters]
    if not out or any(f.ndim != 2 for f in out):
        raise ValueError("filters must be a non-empty sequence of 2-D kernels")
    return out


def _embed(
    filters: Sequence[Tensor],
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Filters `(m, H, W)` of the frame size, origin at index `(0, 0)`."""
    h, w = shape
    out = torch.zeros(len(filters), h, w, dtype=dtype, device=device)
    for i, f in enumerate(filters):
        kh, kw = f.shape
        if kh > h or kw > w:
            raise ValueError(
                f"filter {i} {tuple(f.shape)} is larger than the frame "
                f"{tuple(shape)}"
            )
        out[i, :kh, :kw] = f.to(dtype=dtype, device=device)
        out[i] = out[i].roll((-(kh // 2), -(kw // 2)), dims=(0, 1))
    return out


def _filters_hat(
    filters: object,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    return torch.fft.rfft2(_embed(_as_filters(filters), shape, dtype, device))


def _norm_from_hat(filters_hat: Tensor, shape: Sequence[int]) -> float:
    h, w = shape
    real = filters_hat.real.dtype
    kx = torch.arange(h, dtype=real, device=filters_hat.device)
    ky = torch.arange(w // 2 + 1, dtype=real, device=filters_hat.device)
    d2 = (
        4 * torch.sin(math.pi * kx / h).square()[:, None]
        + 4 * torch.sin(math.pi * ky / w).square()[None, :]
    )
    power = filters_hat.abs().square().sum(0)
    return math.sqrt(float((power * d2).max()))


def operator_norm(
    filters: Tensor | Sequence[Tensor],
    shape: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> float:
    """Norm L = |||A||| of A: λ ↦ ∇(∑ᵢ λᵢ ∗ ψᵢ), Appendix B.

    With periodic differences, AAᵀ is block-diagonalised by the DFT, and

        L² = max over k of Σ(k) (|d̂₁(k)|² + |d̂₂(k)|²),  Σ(k) = ∑ᵢ |ψ̂ᵢ(k)|²,

    with |d̂ⱼ(k)|² = 4 sin²(π kⱼ / nⱼ) the transfer function of a forward
    difference. The value bounds the norm of A with the differences of
    Section II (zero at the last row and column) from above.

    Parameters
    ----------
    filters : Tensor or sequence of Tensor
        Kernels ψᵢ, `(h, w)` or `(m, h, w)`, centred on `(h // 2, w // 2)`.
    shape : (int, int)
        Frame shape.
    dtype, device : optional

    Returns
    -------
    float
        L.
    """
    backend = resolve_backend(dtype, device)
    shape = tuple(int(s) for s in shape)
    return _norm_from_hat(
        _filters_hat(filters, shape, backend["dtype"], backend["device"]),
        shape,
    )


# -- operators ----------------------------------------------------------------
def _grad(u: Tensor) -> Tensor:
    """Forward differences of Section II, `(..., H, W)` → `(..., 2, H, W)`."""
    g = u.new_zeros(*u.shape[:-2], 2, *u.shape[-2:])
    g[..., 0, :-1, :] = u[..., 1:, :] - u[..., :-1, :]
    g[..., 1, :, :-1] = u[..., :, 1:] - u[..., :, :-1]
    return g


def _grad_adjoint(q: Tensor) -> Tensor:
    """∇ᵀ q = −div q, the exact adjoint of `_grad`."""
    p, r = q[..., 0, :, :], q[..., 1, :, :]
    out = torch.zeros_like(p)
    out[..., :-1, :] -= p[..., :-1, :]
    out[..., 1:, :] += p[..., :-1, :]
    out[..., :, :-1] -= r[..., :, :-1]
    out[..., :, 1:] += r[..., :, :-1]
    return out


def _magnitude(g: Tensor) -> Tensor:
    return torch.hypot(g[..., 0, :, :], g[..., 1, :, :])


def _huber(t: Tensor, eps: float) -> Tensor:
    """φ_ε(t) of Eq. (8), for t ≥ 0."""
    if eps == 0:
        return t
    return torch.where(t <= eps, t.square() / (2 * eps), t - eps / 2)


class _Prior:
    """φᵢ with its resolvent and the conjugate of φᵢ + χ_[−C, C] (App. A)."""

    def __init__(self, kind: str, alpha: float) -> None:
        if kind not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}, got {kind!r}")
        if not (math.isfinite(alpha) and alpha > 0):
            raise ValueError(f"alpha must be finite and > 0, got {alpha!r}")
        self.kind, self.alpha = kind, float(alpha)

    def value(self, lam: Tensor) -> Tensor:
        """φ(λ) summed over the frame (λ is in the box)."""
        if self.kind == "l1":
            return self.alpha * lam.abs().sum((-2, -1))
        if self.kind == "l2":
            return 0.5 * self.alpha * lam.square().sum((-2, -1))
        return lam.new_zeros(lam.shape[:-2])

    def bound(self, p0: Tensor) -> Tensor:
        """A bound on ‖λᵢ*‖∞ from φᵢ(λᵢ*) ≤ P(λ*) ≤ P(0) = `p0`."""
        if self.kind == "l1":
            return p0 / self.alpha
        if self.kind == "l2":
            return (2 * p0 / self.alpha).sqrt()
        return torch.full_like(p0, self.alpha)

    def prox(self, lam: Tensor, tau: Tensor, c: Tensor) -> Tensor:
        """(Id + τ∂(φ + χ_[−C, C]))⁻¹, Appendix A, item 5."""
        if self.kind == "l1":
            shrunk = (lam.abs() - tau * self.alpha).clamp_min(0)
            return lam.sign() * torch.minimum(shrunk, c)
        if self.kind == "l2":
            x = lam / (tau * self.alpha + 1)
            return torch.maximum(torch.minimum(x, c), -c)
        d = c.clamp_max(self.alpha)
        return torch.maximum(torch.minimum(lam, d), -d)

    def conjugate(self, eta: Tensor, c: Tensor) -> Tensor:
        """(φ + χ_[−C, C])*(η) summed over the frame, Appendix A.

        For ℓ¹, the conjugate of φ alone (the indicator of |η| ≤ α) is
        used instead: the dual variable is first rescaled to satisfy it
        (see `vsnr_solve`), so this term is zero and the gap does not
        depend on C.
        """
        a = eta.abs()
        if self.kind == "l1":
            return eta.new_zeros(eta.shape[:-2])
        if self.kind == "l2":
            t = torch.minimum(a / self.alpha, c)
            out = t * a - 0.5 * self.alpha * t.square()
        else:
            out = c.clamp_max(self.alpha) * a
        return out.sum((-2, -1))


def _priors(prior: object, alpha: object, m: int) -> list[_Prior]:
    kinds = [prior] * m if isinstance(prior, str) else list(prior)
    if isinstance(alpha, int | float):
        alphas = [float(alpha)] * m
    else:
        alphas = [float(a) for a in alpha]
    if len(kinds) != m or len(alphas) != m:
        raise ValueError(
            f"prior and alpha need one value or one per filter ({m}), got "
            f"{len(kinds)} and {len(alphas)}"
        )
    return [_Prior(k, a) for k, a in zip(kinds, alphas, strict=True)]


# -- solver -------------------------------------------------------------------
class VSNRSolution(NamedTuple):
    """Output of [`vsnr_solve`][optix.destripe.utils.vsnr_solve].

    Attributes
    ----------
    image : Tensor
        Denoised frames u = u₀ − b, `(..., H, W)`.
    noise : Tensor
        Stationary noise b = ∑ᵢ λᵢ ∗ ψᵢ, `(..., H, W)`.
    weights : Tensor
        Primal variables λ, `(..., m, H, W)`.
    dual : Tensor
        Dual variable q (the TV dual field), `(..., 2, H, W)`.
    primal : Tensor
        P(λₖ) per iteration, `(K + 1, ...)` (empty unless `history`).
    gap : Tensor
        Duality gap Δ(λₖ, qₖ) per iteration, `(K + 1, ...)` (empty
        unless `history`), with qₖ rescaled to be dual feasible for ℓ¹
        components (see `vsnr_solve`). Converged frames keep their last
        value.
    iterations : Tensor
        Iterations run per frame, `(...)`.
    bound : Tensor
        Box bound C per frame, `(...)`.
    """

    image: Tensor
    noise: Tensor
    weights: Tensor
    dual: Tensor
    primal: Tensor
    gap: Tensor
    iterations: Tensor
    bound: Tensor


def _check_scalars(eps: object, tol: object, max_iter: object) -> None:
    if not (isinstance(eps, int | float) and math.isfinite(eps) and eps >= 0):
        raise ValueError(f"eps must be finite and >= 0, got {eps!r}")
    if not (isinstance(tol, int | float) and tol >= 0):
        raise ValueError(f"tol must be >= 0, got {tol!r}")
    if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter < 0:
        raise ValueError(f"max_iter must be an integer >= 0, got {max_iter!r}")


def vsnr_solve(
    data: Tensor,
    filters: Tensor | Sequence[Tensor],
    *,
    alpha: float | Sequence[float],
    prior: str | Sequence[str] = "l1",
    eps: float = 0.0,
    bound: float | None = None,
    tol: float = 1e-3,
    max_iter: int = 1000,
    tau: float | None = None,
    sigma: float | None = None,
    history: bool = False,
) -> VSNRSolution:
    """Primal–dual VSNR solver (Algorithms 1 and 2), with diagnostics.

    Solves, for each frame u₀ independently, problem (11) of Fehrenbach
    et al. (2012),

        min_{λ ∈ Ξ} P(λ) = ‖∇(u₀ − ∑ᵢ λᵢ ∗ ψᵢ)‖₁,ε + ∑ᵢ φᵢ(λᵢ),
        Ξ = {λ : ‖λ‖∞ ≤ C},

    through the saddle-point problem (13), min_λ max_q ⟨Aλ, q⟩ − F*(q) +
    G(λ), with Aλ = ∇(∑ᵢ λᵢ ∗ ψᵢ), F(g) = ‖∇u₀ − g‖₁,ε and
    G = ∑ᵢ φᵢ + χ_Ξ. Each iteration of Algorithm 1 (Chambolle–Pock) is

        qₖ₊₁ = (Id + σₖ∂F*)⁻¹(qₖ + σₖ A λ̄ₖ),
        λₖ₊₁ = (Id + τₖ∂G)⁻¹(λₖ − τₖ A* qₖ₊₁),
        λ̄ₖ₊₁ = λₖ₊₁ + θₖ (λₖ₊₁ − λₖ),

    with the resolvents of Appendix A, from (λ₀, q₀) = (0, 0), until the
    duality gap (15), Δ(λₖ, qₖ) = P(λₖ) − D(qₖ) with
    D(q) = −F*(q) − G*(−A*q), falls to `tol` · Δ(λ₀, q₀).

    The convolutions are periodic and computed by FFT; ∇ is the forward
    difference of Section II (zero at the last row and column), and
    ∇ᵀ = −div its exact adjoint. The step sizes follow Algorithm 2:

    - all φᵢ = αᵢ/2 ‖·‖² and ε > 0: constant steps with
      μ = 2√(γε)/L, τ = μ/(2γ), σ = μ/(2ε), θ = 1/(1 + μ), γ = min αᵢ;
    - all φᵢ = αᵢ/2 ‖·‖² and ε = 0: θₖ = 1/√(1 + 2γτₖ),
      τₖ₊₁ = θₖ τₖ, σₖ₊₁ = σₖ/θₖ;
    - otherwise constant steps τ, σ and θ = 1;

    with L the operator norm of Appendix B
    ([`operator_norm`][optix.destripe.utils.operator_norm]).

    !!! references "References"
        1. Fehrenbach J, Weiss P, Lorenzo C. *Variational algorithms to
           remove stationary noise: applications to microscopy imaging.*
           IEEE Trans Image Process **21**(10):4420–4430, 2012.
           [doi:10.1109/TIP.2012.2206037](https://doi.org/10.1109/TIP.2012.2206037)
        2. Chambolle A, Pock T. *A first-order primal-dual algorithm for
           convex problems with applications to imaging.* J Math Imaging
           Vis **40**(1):120–145, 2011.
           [doi:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)

        ??? quote "BibTeX"
            ```bibtex
            @article{fehrenbach2012variational,
              author  = {Fehrenbach, Jérôme and Weiss, Pierre
                         and Lorenzo, Corinne},
              title   = {Variational algorithms to remove stationary
                         noise: applications to microscopy imaging},
              journal = {IEEE Transactions on Image Processing},
              volume  = {21}, number = {10}, pages = {4420--4430},
              year    = {2012}, doi = {10.1109/TIP.2012.2206037}
            }
            @article{chambolle2011first,
              author  = {Chambolle, Antonin and Pock, Thomas},
              title   = {A first-order primal-dual algorithm for convex
                         problems with applications to imaging},
              journal = {Journal of Mathematical Imaging and Vision},
              volume  = {40}, number = {1}, pages = {120--145},
              year    = {2011}, doi = {10.1007/s10851-010-0251-1}
            }
            ```

    !!! note "Choices not specified in the paper"
        - The paper asks for σ₀τ₀L² < 1 and leaves the split between σ₀
          and τ₀ to experimental tuning (Section VI-C). Default:
          τ₀ = 0.99 s/L and σ₀ = 0.99/(s L), with s the mean gradient
          magnitude of the frame, ‖∇u₀‖₁,₀ / n (n pixels). Since λ
          scales with the data and q does not, this makes the iterates
          equivariant to a rescaling of the intensities.
        - The box bound C must exceed ‖λ*‖∞ (Section V-A). Default: the
          bound that follows from φᵢ(λᵢ*) ≤ P(λ*) ≤ P(0), i.e.
          the maximum over i of P(0)/αᵢ (ℓ¹), √(2P(0)/αᵢ) (ℓ²) or αᵢ (ℓ∞
          ball), per frame. The box is then inactive at the minimiser.
          With ℓ¹ priors this bound is loose; it does not enter the
          stopping rule (see the next item).
        - For ℓ¹ components, D is evaluated at the dual-feasible point
          q/κ, κ = max(1, maxᵢ ‖(A*q)ᵢ‖∞/αᵢ) over those components, with
          the conjugate of φᵢ alone (the indicator of |ηᵢ| ≤ αᵢ), which
          is then zero. The gap is still an upper bound on P(λₖ) − P*,
          and it does not depend on C.
        - A `RuntimeWarning` is emitted when frames reach `max_iter`
          before `tol` (with `tol` > 0 and `max_iter` > 0). With ℓ¹
          priors and ε = 0 the rate is only O(1/k) (Proposition 5.3),
          so this is common.
        - Appendix A writes F*(q) = ε/2 ‖q‖² − ⟨∇u₀, q⟩, which goes with
          u = u₀ + ∑ᵢ λᵢ ∗ ψᵢ. The sign used here,
          F*(q) = ε/2 ‖q‖² + ⟨∇u₀, q⟩ (for ‖q‖∞ ≤ 1), goes with model (6),
          u = u₀ − b; the two differ only by the sign of λ, as all the
          priors are even.

    Parameters
    ----------
    data : Tensor
        Frames u₀, `(..., H, W)`.
    filters : Tensor or sequence of Tensor
        The m kernels ψᵢ, `(m, h, w)`, `(h, w)` or a sequence of 2-D
        kernels of any size up to `(H, W)`, each centred on
        `(h // 2, w // 2)` (see [`dirac`][optix.destripe.utils.dirac],
        [`line`][optix.destripe.utils.line],
        [`gaussian_stripe`][optix.destripe.utils.gaussian_stripe]).
    alpha : float or sequence of float
        Weights αᵢ > 0 of the priors (one value, or one per filter).
    prior : str or sequence of str, default `"l1"`
        φᵢ, one value or one per filter (Section III-C):
        `"l1"`: αᵢ‖λᵢ‖₁ (Laplace noise); `"l2"`: αᵢ/2 ‖λᵢ‖₂² (Gaussian
        noise); `"linf"`: the indicator of ‖λᵢ‖∞ ≤ αᵢ (uniform noise).
        The default is not specified in the paper.
    eps : float, default 0
        Huber parameter ε ≥ 0 of ‖·‖₁,ε (Eq. (8)); 0 is total variation.
    bound : float, optional
        Box bound C > 0; default: see above.
    tol : float, default 1e-3
        Relative duality gap at which a frame stops (the paper's
        ε = 10⁻³, Section VI-C).
    max_iter : int, default 1000
        Iteration cap per frame (not specified in the paper).
    tau, sigma : float, optional
        Initial steps τ₀, σ₀ (σ₀τ₀L² < 1). Ignored in the constant-step
        case of ℓ² priors with ε > 0, where Algorithm 2 sets them.
    history : bool, default False
        Record P(λₖ) and Δ(λₖ, qₖ) at every iteration.

    Returns
    -------
    VSNRSolution
    """
    _check_scalars(eps, tol, max_iter)
    if data.ndim < 2:
        raise ValueError("data must have at least 2 dims (..., H, W)")
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    batch, shape = tuple(data.shape[:-2]), tuple(data.shape[-2:])
    u0 = data.reshape(-1, *shape)
    n = u0.shape[0]
    fh = _filters_hat(filters, shape, data.dtype, data.device)
    m = fh.shape[0]
    priors = _priors(prior, alpha, m)
    lip = _norm_from_hat(fh, shape)
    if lip == 0:
        raise ValueError("the filters have no energy at non-zero frequencies")
    eps = float(eps)

    def noise(lam: Tensor) -> Tensor:
        return torch.fft.irfft2((torch.fft.rfft2(lam) * fh).sum(-3), s=shape)

    def adjoint(q: Tensor) -> Tensor:
        w = torch.fft.rfft2(_grad_adjoint(q))
        return torch.fft.irfft2(w[:, None] * fh.conj(), s=shape)

    def expand(x: Tensor) -> Tensor:
        return x.view(-1, 1, 1, 1)

    g0 = _grad(u0)
    p0 = _huber(_magnitude(g0), eps).sum((-2, -1))
    if bound is None:
        c = torch.stack([p.bound(p0) for p in priors]).amax(0)
    else:
        if not (bound > 0):
            raise ValueError(f"bound must be > 0, got {bound!r}")
        c = torch.full_like(p0, float(bound))
    c4 = expand(c)

    def primal(lam: Tensor, b: Tensor) -> Tensor:
        r = _huber(_magnitude(g0 - _grad(b)), eps).sum((-2, -1))
        for i, p in enumerate(priors):
            r = r + p.value(lam[:, i])
        return r

    l1 = [i for i, p in enumerate(priors) if p.kind == "l1"]

    def dual(q: Tensor, ast: Tensor) -> Tensor:
        # D at the dual-feasible q/κ, κ = max(1, maxᵢ ‖(A*q)ᵢ‖∞/αᵢ) over
        # the ℓ¹ components, so that their conjugate (the indicator of
        # |η| ≤ αᵢ) is zero and the gap bounds P(λ) − P* without C.
        kappa = torch.ones_like(p0)
        for i in l1:
            peak = ast[:, i].abs().amax((-2, -1)) / priors[i].alpha
            kappa = torch.maximum(kappa, peak)
        f = (g0 * q).sum((-3, -2, -1)) / kappa
        if eps:
            f = f + 0.5 * eps * q.square().sum((-3, -2, -1)) / kappa**2
        k4 = expand(kappa)[:, 0]
        for i, p in enumerate(priors):
            if p.kind != "l1":
                f = f + p.conjugate(ast[:, i] / k4, c4[:, 0])
        return -f

    # step sizes (Algorithm 2)
    quadratic = all(p.kind == "l2" for p in priors)
    gamma = min(p.alpha for p in priors) if quadratic else 0.0
    accelerate = quadratic and eps == 0
    if quadratic and eps > 0:
        mu = 2 * math.sqrt(gamma * eps) / lip
        tau0 = torch.full_like(p0, mu / (2 * gamma))
        sigma0 = torch.full_like(p0, mu / (2 * eps))
        theta_const = 1 / (1 + mu)
    else:
        scale = _magnitude(g0).mean((-2, -1))
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        tau0 = 0.99 * scale / lip if tau is None else torch.full_like(p0, tau)
        sigma0 = (
            0.99 / (scale * lip)
            if sigma is None
            else torch.full_like(p0, sigma)
        )
        if tau is not None and sigma is not None and tau * sigma * lip**2 >= 1:
            raise ValueError("tau * sigma * L² must be < 1")
        if (tau is None) != (sigma is None):
            # one given: the other keeps the product at 0.99² / L²
            other = 0.99**2 / lip**2
            if tau is None:
                tau0 = torch.full_like(p0, other / sigma)
            else:
                sigma0 = torch.full_like(p0, other / tau)
        theta_const = 1.0
    tau_k, sigma_k = tau0.clone(), sigma0.clone()

    lam = u0.new_zeros(n, m, *shape)
    q = u0.new_zeros(n, 2, *shape)
    b = torch.zeros_like(u0)
    b_bar = b
    ast = torch.zeros_like(lam)
    active = torch.ones(n, dtype=torch.bool, device=u0.device)
    count = torch.zeros(n, dtype=torch.long, device=u0.device)
    gap0 = None
    primals, gaps = [], []
    for k in range(int(max_iter) + 1):
        pk = primal(lam, b)
        gk = pk - dual(q, ast)
        if gap0 is None:
            gap0 = gk
        if history:
            primals.append(pk)
            gaps.append(gk)
        active = active & (gk > tol * gap0)
        if k == max_iter or not bool(active.any()):
            break
        a4 = expand(active)
        # dual step: resolvent of F* (Appendix A, item 4)
        s4 = expand(sigma_k)
        t = q + s4 * _grad(b_bar) - s4 * g0
        norm = _magnitude(t).unsqueeze(-3)
        q_new = t / torch.maximum(norm, 1 + s4 * eps)
        # primal step: resolvent of G (Appendix A, item 5)
        ast_new = adjoint(q_new)
        t4 = expand(tau_k)
        step = lam - t4 * ast_new
        lam_new = torch.stack(
            [
                p.prox(step[:, i], t4[:, 0], c4[:, 0])
                for i, p in enumerate(priors)
            ],
            1,
        )
        b_new = noise(lam_new)
        if accelerate:
            theta = 1 / torch.sqrt(1 + 2 * gamma * tau_k)
            tau_k = torch.where(active, theta * tau_k, tau_k)
            sigma_k = torch.where(active, sigma_k / theta, sigma_k)
            th = theta.view(-1, 1, 1)
        else:
            th = theta_const
        b_bar = b_new + th * (b_new - b)
        q = torch.where(a4, q_new, q)
        ast = torch.where(a4, ast_new, ast)
        lam = torch.where(a4, lam_new, lam)
        b = torch.where(a4[:, 0], b_new, b)
        count += active
    if max_iter > 0 and tol > 0 and bool(active.any()):
        warnings.warn(
            f"vsnr: {int(active.sum())} of {n} frame(s) reached max_iter "
            f"= {int(max_iter)} before the relative duality gap fell to "
            f"tol = {tol:g}",
            RuntimeWarning,
            stacklevel=2,
        )
    image = u0 - b
    empty = u0.new_zeros(0, *batch)
    return VSNRSolution(
        image=image.reshape(*batch, *shape),
        noise=b.reshape(*batch, *shape),
        weights=lam.reshape(*batch, m, *shape),
        dual=q.reshape(*batch, 2, *shape),
        primal=torch.stack(primals).reshape(-1, *batch) if history else empty,
        gap=torch.stack(gaps).reshape(-1, *batch) if history else empty,
        iterations=count.reshape(batch),
        bound=c.reshape(batch),
    )


def vsnr(
    data: Tensor,
    filters: Tensor | Sequence[Tensor],
    *,
    alpha: float | Sequence[float],
    prior: str | Sequence[str] = "l1",
    eps: float = 0.0,
    bound: float | None = None,
    tol: float = 1e-3,
    max_iter: int = 1000,
    tau: float | None = None,
    sigma: float | None = None,
) -> Tensor:
    """Remove stationary noise (stripes) with VSNR.

    The variational stationary noise remover of Fehrenbach et al. (2012)
    models each frame as u₀ = u + b (Eq. (6)), with a stationary noise

        b = ∑ᵢ λᵢ ∗ ψᵢ  (Eq. (4)),

    made of known patterns ψᵢ (e.g. stripes) convolved with white noises
    λᵢ of densities p(λᵢ) ∝ exp(−φᵢ(λᵢ)) (Eq. (5)). The MAP estimate
    under the image prior p(u) ∝ exp(−‖∇u‖₁,ε) (Eq. (8)) solves (9),

        min_λ ‖∇(u₀ − ∑ᵢ λᵢ ∗ ψᵢ)‖₁,ε + ∑ᵢ φᵢ(λᵢ),

    where ‖q‖₁,ε = ∑ₓ φ_ε(|q(x)|₂) is the Huber-smoothed total variation
    (φ_ε(t) = t²/(2ε) for |t| ≤ ε, |t| − ε/2 otherwise; ε = 0 is the
    total variation). The denoised frame is u = u₀ − ∑ᵢ λᵢ ∗ ψᵢ.
    The problem is solved by the primal–dual algorithm of Chambolle and
    Pock (Algorithms 1 and 2), stopped on the duality gap; see
    [`vsnr_solve`][optix.destripe.utils.vsnr_solve] for the iteration,
    the step sizes and the diagnostics.

    !!! references "References"
        1. Fehrenbach J, Weiss P, Lorenzo C. *Variational algorithms to
           remove stationary noise: applications to microscopy imaging.*
           IEEE Trans Image Process **21**(10):4420–4430, 2012.
           [doi:10.1109/TIP.2012.2206037](https://doi.org/10.1109/TIP.2012.2206037)
        2. Chambolle A, Pock T. *A first-order primal-dual algorithm for
           convex problems with applications to imaging.* J Math Imaging
           Vis **40**(1):120–145, 2011.
           [doi:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)

        ??? quote "BibTeX"
            ```bibtex
            @article{fehrenbach2012variational,
              author  = {Fehrenbach, Jérôme and Weiss, Pierre
                         and Lorenzo, Corinne},
              title   = {Variational algorithms to remove stationary
                         noise: applications to microscopy imaging},
              journal = {IEEE Transactions on Image Processing},
              volume  = {21}, number = {10}, pages = {4420--4430},
              year    = {2012}, doi = {10.1109/TIP.2012.2206037}
            }
            @article{chambolle2011first,
              author  = {Chambolle, Antonin and Pock, Thomas},
              title   = {A first-order primal-dual algorithm for convex
                         problems with applications to imaging},
              journal = {Journal of Mathematical Imaging and Vision},
              volume  = {40}, number = {1}, pages = {120--145},
              year    = {2011}, doi = {10.1007/s10851-010-0251-1}
            }
            ```

    !!! note
        Each frame is processed independently (the convolutions are
        periodic over the frame). The cost per iteration is 2m + 2 FFTs
        of the frame size; the paper reports fewer than 50 iterations to
        reduce the duality gap by 10³ with well-tuned steps
        (Section VI-C). αᵢ sets the strength of each noise component and
        depends on the amplitude of ψᵢ: the paper leaves it to the user.
        With ℓ¹ priors and ε = 0, the solution is equivariant to a
        rescaling of the intensities (u₀ → s u₀ gives λ → s λ), so αᵢ
        does not depend on the intensity unit. This does not hold for
        ε > 0 (the Huber function is not 1-homogeneous), nor for
        `"linf"` priors, whose bound αᵢ is in intensity units.

    Parameters
    ----------
    data : Tensor
        Frames `(..., H, W)` (floating point; other dtypes are converted
        to the configured dtype).
    filters : Tensor or sequence of Tensor
        Noise patterns ψᵢ: `(h, w)`, `(m, h, w)` or a sequence of 2-D
        kernels, each at most `(H, W)` and centred on `(h // 2, w // 2)`;
        e.g. [`gaussian_stripe`][optix.destripe.utils.gaussian_stripe],
        [`line`][optix.destripe.utils.line] or
        [`dirac`][optix.destripe.utils.dirac] (white noise).
    alpha : float or sequence of float
        Prior weights αᵢ > 0, one value or one per filter.
    prior : str or sequence of str, default `"l1"`
        φᵢ: `"l1"` (αᵢ‖λᵢ‖₁, Laplace), `"l2"` (αᵢ/2 ‖λᵢ‖₂², Gaussian)
        or `"linf"` (‖λᵢ‖∞ ≤ αᵢ, uniform); one value or one per filter.
        The default is not specified in the paper.
    eps : float, default 0
        Huber parameter ε of the image prior; 0 is total variation.
    bound : float, optional
        Box bound C of problem (11); default: a bound on ‖λ*‖∞ derived
        from the data (see [`vsnr_solve`][optix.destripe.utils.vsnr_solve]).
    tol : float, default 1e-3
        Relative duality gap at which to stop (the paper's 10⁻³).
    max_iter : int, default 1000
        Iteration cap per frame (not specified in the paper).
    tau, sigma : float, optional
        Initial primal and dual steps (σ₀τ₀L² < 1); default: see
        [`vsnr_solve`][optix.destripe.utils.vsnr_solve].

    Returns
    -------
    Tensor
        Denoised frames u, of the shape of `data`.
    """
    return vsnr_solve(
        data,
        filters,
        alpha=alpha,
        prior=prior,
        eps=eps,
        bound=bound,
        tol=tol,
        max_iter=max_iter,
        tau=tau,
        sigma=sigma,
    ).image


class VSNR(Estimator):
    """Variational stationary noise remover (VSNR).

    The method of Fehrenbach et al. (2012), applied to every frame
    independently: see [`vsnr`][optix.destripe.vsnr] for the model and
    [`vsnr_solve`][optix.destripe.utils.vsnr_solve] for the solver. It
    has no parameters to fit: [`fit`][optix.destripe.VSNR.fit] only
    records the layout, and [`apply`][optix.Estimator.apply] streams the
    frames in batches, so arrays larger than memory (zarr, memmap) can be
    processed.

    !!! note
        The noise patterns ψᵢ, the priors φᵢ and their weights αᵢ describe
        the data and are chosen by the user (Section III); the paper gives
        no defaults for them. The filters are 2-D kernels over the two
        spatial axes of the layout, in layout order.

    !!! references "References"
        1. Fehrenbach J, Weiss P, Lorenzo C. *Variational algorithms to
           remove stationary noise: applications to microscopy imaging.*
           IEEE Trans Image Process **21**(10):4420–4430, 2012.
           [doi:10.1109/TIP.2012.2206037](https://doi.org/10.1109/TIP.2012.2206037)

        ??? quote "BibTeX"
            ```bibtex
            @article{fehrenbach2012variational,
              author  = {Fehrenbach, Jérôme and Weiss, Pierre
                         and Lorenzo, Corinne},
              title   = {Variational algorithms to remove stationary
                         noise: applications to microscopy imaging},
              journal = {IEEE Transactions on Image Processing},
              volume  = {21}, number = {10}, pages = {4420--4430},
              year    = {2012}, doi = {10.1109/TIP.2012.2206037}
            }
            ```
    """

    tags = Tags(
        targets=("stripes", "stationary-noise"),
        streaming="per-frame",
        modality=("light-sheet", "sem", "fib-sem", "afm"),
        reference="doi:10.1109/TIP.2012.2206037",
    )
    paper = {
        "filters": None,
        "alpha": None,
        "prior": "l1",
        "eps": 0.0,
        "bound": None,
        "tol": 1e-3,
        "max_iter": 1000,
        "tau": None,
        "sigma": None,
    }
    recommended = dict(paper)
    # free parameters of the published method: any value is faithful
    _free = tuple(paper)

    def __init__(
        self,
        filters: Tensor | Sequence[Tensor] | None = None,
        *,
        alpha: float | Sequence[float] | None = None,
        prior: str | Sequence[str] | None = None,
        eps: float | None = None,
        bound: float | None = None,
        tol: float | None = None,
        max_iter: int | None = None,
        tau: float | None = None,
        sigma: float | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the solver.

        Parameters
        ----------
        filters : Tensor or sequence of Tensor
            Noise patterns ψᵢ (required): `(h, w)`, `(m, h, w)` or a
            sequence of 2-D kernels over the two spatial axes, each
            centred on `(h // 2, w // 2)`; see
            [`vsnr`][optix.destripe.vsnr].
        alpha : float or sequence of float
            Prior weights αᵢ > 0 (required), one value or one per filter.
        prior : str or sequence of str, default `"l1"`
            φᵢ: `"l1"`, `"l2"` or `"linf"`, one value or one per filter.
        eps : float, default 0
            Huber parameter ε of the image prior (0: total variation).
        bound : float, optional
            Box bound C; default: derived from each frame.
        tol : float, default 1e-3
            Relative duality gap at which to stop (the paper's value).
        max_iter : int, default 1000
            Iteration cap per frame.
        tau, sigma : float, optional
            Initial primal and dual steps.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        if o["filters"] is None or o["alpha"] is None:
            raise ValueError("filters and alpha must be given")
        o["filters"] = [f.detach().clone() for f in _as_filters(o["filters"])]
        if not isinstance(o["alpha"], int | float):
            o["alpha"] = [float(a) for a in o["alpha"]]
        if not isinstance(o["prior"], str):
            o["prior"] = [str(p) for p in o["prior"]]
        _priors(o["prior"], o["alpha"], len(o["filters"]))
        _check_scalars(o["eps"], o["tol"], o["max_iter"])
        self.is_faithful = True

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
        VSNR
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

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        o = dict(self.options)
        layout = self._layout_of(data)
        _, _, spatial = layout.positions(len(data.shape))
        if len(spatial) != 2:
            raise ValueError(
                f"VSNR needs 2 spatial axes, the layout has {len(spatial)}"
            )
        filters = o.pop("filters")

        def fn(chunk: object, batch: Tensor) -> Tensor:
            return vsnr(batch, filters, **o)

        m = len(filters)
        return self._apply_frames(
            data, out, fn, overhead=16.0 + 9.0 * m, contiguous=True
        )
