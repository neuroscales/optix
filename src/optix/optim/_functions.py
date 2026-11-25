"""Functions with a value and a proximity operator."""

import math
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from ..special import poisson_normal_logpdf, poisson_normal_ratio
from ._prox import (
    _packed_ndim,
    _schatten_p,
    project_group_l2,
    project_l2_ball,
    prox_box,
    prox_group_l2,
    prox_schatten,
    prox_shifted_poisson,
    prox_squared_l2,
    schatten_norm,
)

__all__ = [
    "Box",
    "Function",
    "GroupL2",
    "L2Ball",
    "PoissonGaussian",
    "Schatten",
    "ShiftedPoisson",
    "Smooth",
    "SquaredL2",
]

#: Series terms held in memory at once by `PoissonGaussian` (about
#: 25 MiB of float64 temporaries).
_SERIES_TERMS = 2**18


@runtime_checkable
class Function(Protocol):
    """A proper, closed, convex function f with an easy prox."""

    def value(self, z: Tensor) -> Tensor:
        """f(z), a float64 scalar (+inf outside the domain)."""
        ...

    def prox(self, v: Tensor, t: float) -> Tensor:
        """prox_{t f}(v) = argmin_z f(z) + ‖z − v‖² / (2t)."""
        ...


@runtime_checkable
class Smooth(Protocol):
    """A convex, differentiable function h with a Lipschitz gradient."""

    @property
    def lipschitz(self) -> float:
        """Lipschitz constant of ∇h (may be +inf)."""
        ...

    def value(self, z: Tensor) -> Tensor:
        """h(z), a float64 scalar."""
        ...

    def grad(self, z: Tensor) -> Tensor:
        """∇h(z)."""
        ...


class Box:
    """Indicator function of the box [lower, upper] (0 inside, +∞ outside)."""

    def __init__(
        self,
        lower: float | Tensor | None = 0.0,
        upper: float | Tensor | None = None,
    ) -> None:
        """Set the bounds.

        Parameters
        ----------
        lower, upper : float or Tensor, optional
            Bounds; None is unbounded. Default: [0, +∞).
        """
        self.lower = lower
        self.upper = upper

    def value(self, z: Tensor) -> Tensor:
        """0 inside the box, +inf outside."""
        inside = True
        if self.lower is not None:
            inside = bool((z >= self.lower).all())
        if inside and self.upper is not None:
            inside = bool((z <= self.upper).all())
        out = 0.0 if inside else math.inf
        return torch.tensor(out, dtype=torch.float64)

    def prox(self, v: Tensor, t: float) -> Tensor:
        """Projection onto the box (any t)."""
        return prox_box(v, self.lower, self.upper)


class L2Ball:
    """Indicator function of the weighted ℓ2 ball ∑ⱼ wⱼ (zⱼ − cⱼ)² ≤ r².

    0 inside, +∞ outside. With z an image and c the data, this is the
    discrepancy constraint of Rudin, Osher & Fatemi (1992, Eq. 2.3c).
    """

    def __init__(
        self,
        radius: float,
        center: Tensor | float = 0.0,
        weight: Tensor | float = 1.0,
    ) -> None:
        """Set the ball.

        Parameters
        ----------
        radius : float
            r ≥ 0.
        center : Tensor or float
            Centre c.
        weight : Tensor or float
            Weights w ≥ 0, broadcasting with c.
        """
        self.radius = float(radius)
        if not self.radius >= 0:
            raise ValueError(f"radius must be >= 0, got {radius}")
        self.center = center
        self.weight = weight

    def value(self, z: Tensor) -> Tensor:
        """0 inside the ball (to a relative 10⁻⁶ of r²), +inf outside."""
        w = self.weight
        if isinstance(w, Tensor):
            w = w.double()
        r = (z - self.center).double().square()
        inside = float((w * r).sum()) <= self.radius**2 * (1 + 1e-6)
        return torch.tensor(0.0 if inside else math.inf, dtype=torch.float64)

    def prox(self, v: Tensor, t: float) -> Tensor:
        """Projection onto the ball (any t; see `project_l2_ball`)."""
        return project_l2_ball(v, self.radius, self.center, self.weight)


class GroupL2:
    """Weighted mixed ℓ2,1 norm, w·∑ⱼ ‖zⱼ‖₂ (components along `dim`).

    With z the gradient of an image this is its isotropic total
    variation; with z the packed Hessian of
    [`Hessian`][optix.optim.Hessian], ‖zⱼ‖₂ is the Frobenius norm of the
    Hessian matrix at voxel j.
    """

    def __init__(self, weight: float | Tensor, dim: int = 0) -> None:
        """Set the weight.

        Parameters
        ----------
        weight : float or Tensor
            w ≥ 0; a tensor broadcasts against the voxels.
        dim : int
            Component dim.
        """
        self.weight = weight
        self.dim = dim

    def value(self, z: Tensor) -> Tensor:
        """w·∑ⱼ ‖zⱼ‖₂ in float64."""
        norm = torch.linalg.vector_norm(z.double(), dim=self.dim)
        w = self.weight
        if isinstance(w, Tensor):
            w = w.double()
        return (w * norm).sum()

    def prox(self, v: Tensor, t: float) -> Tensor:
        """Block soft threshold at t·w."""
        return prox_group_l2(v, t * self.weight, dim=self.dim)

    def prox_conjugate(self, v: Tensor, t: float) -> Tensor:
        """Prox of t·F*: the projection of each vⱼ onto the ball of radius w.

        The same as `prox_conjugate(self.prox, v, t)` (Moreau), for any
        t, with fewer temporaries.
        """
        return project_group_l2(v, self.weight, dim=self.dim)


class Schatten:
    """Mixed Schatten norm w·∑ⱼ ‖Zⱼ‖_{S_p} of a field of symmetric matrices.

    zⱼ packs a symmetric d × d matrix Zⱼ along `dim`, as the output of
    [`Hessian`][optix.optim.Hessian]. With z = H x this is the Hessian
    Schatten-norm regulariser ‖H x‖₁,ₚ of Lefkimmiatis & Unser (2013,
    Eq. 11): the nuclear norm (p = 1, the sum of the absolute
    eigenvalues), the Frobenius norm (p = 2, equal to
    [`GroupL2`][optix.optim.GroupL2]) or the spectral norm (p = ∞, the
    largest absolute eigenvalue). See
    [`prox_schatten`][optix.optim.prox_schatten] for the prox; p ∈ {1, ∞}
    is available for matrices up to 3 × 3 (2-D and 3-D images).
    """

    def __init__(
        self,
        weight: float | Tensor,
        p: float | str = 1,
        *,
        ndim: int | None = None,
        dim: int = 0,
    ) -> None:
        """Set the weight and the order.

        Parameters
        ----------
        weight : float or Tensor
            w ≥ 0; a tensor broadcasts against the voxels.
        p : {1, 2, inf}
            Order of the Schatten norm.
        ndim : int, optional
            Matrix size d, checked against p (p ∈ {1, ∞} needs d ≤ 3).
            Default: taken from the number of components.
        dim : int
            Component dim.

        Raises
        ------
        NotImplementedError
            For p ∈ {1, ∞} with d > 3.
        """
        self.weight = weight
        self.p = _schatten_p(p)
        self.ndim = ndim
        self.dim = dim
        if ndim is not None:
            self._check(int(ndim))

    def _check(self, d: int) -> None:
        if self.p != 2 and d > 3:
            raise NotImplementedError(
                "Schatten norms with p in {1, inf} are implemented for "
                "matrices up to 3 × 3 (2-D and 3-D images); use p=2"
            )

    def value(self, z: Tensor) -> Tensor:
        """w·∑ⱼ ‖Zⱼ‖_{S_p} in float64."""
        self._check(_packed_ndim(z.shape[self.dim]))
        norm = schatten_norm(z.double(), self.p, dim=self.dim)
        w = self.weight
        if isinstance(w, Tensor):
            w = w.double()
        return (w * norm).sum()

    def prox(self, v: Tensor, t: float) -> Tensor:
        """Prox of t·w·∑ⱼ ‖·‖_{S_p} (see `prox_schatten`)."""
        return prox_schatten(v, t * self.weight, self.p, dim=self.dim)


class SquaredL2:
    """Quadratic (w/2)·‖z − y‖²."""

    def __init__(
        self, weight: float | Tensor = 1.0, data: Tensor | float = 0.0
    ) -> None:
        """Set the weight and the centre.

        Parameters
        ----------
        weight : float or Tensor
            w ≥ 0.
        data : Tensor or float
            Centre y.
        """
        self.weight = weight
        self.data = data

    def value(self, z: Tensor) -> Tensor:
        """(w/2)·‖z − y‖² in float64."""
        r = (z - self.data).double().square()
        w = self.weight
        if isinstance(w, Tensor):
            w = w.double()
        return (w * r).sum() / 2

    def prox(self, v: Tensor, t: float) -> Tensor:
        """(v + t w y) / (1 + t w)."""
        return prox_squared_l2(v, t, self.data, self.weight)


class ShiftedPoisson:
    """Shifted-Poisson negative log-likelihood (Ikoma et al. 2018, Eq. S3).

        f(z) = ∑ⱼ zⱼ − (yⱼ + σⱼ²) log(zⱼ + σⱼ²),

    up to a constant. The data y + σ² must be non-negative (clip y at
    −σ² beforehand). With σ² = 0 this is the Poisson likelihood.
    """

    def __init__(self, data: Tensor, variance: Tensor | float = 0.0) -> None:
        """Set the data and the read-noise variance.

        Parameters
        ----------
        data : Tensor
            Data y (in electrons), y + σ² ≥ 0.
        variance : Tensor or float
            Read-noise variance σ² ≥ 0 (in electrons²); broadcasts with
            `data`.
        """
        self.data = data
        self.variance = variance

    @property
    def lipschitz(self) -> float:
        """Lipschitz constant of ∇f on z ≥ 0: max (y + σ²)/σ⁴.

        +inf when some σ² = 0 (Poisson).
        """
        c = (self.data + self.variance).double()
        var = torch.as_tensor(self.variance).double()
        if not bool((var > 0).all()):
            return math.inf
        return float((c / var.square()).max())

    def value(self, z: Tensor) -> Tensor:
        """f(z) in float64; +inf where z + σ² ≤ 0 with y + σ² > 0."""
        s = (z + self.variance).double()
        c = (self.data + self.variance).double()
        if bool(((s <= 0) & (c > 0)).any()) or bool((s < 0).any()):
            return torch.tensor(math.inf, dtype=torch.float64)
        log = torch.where(c > 0, torch.log(s.clamp(min=1e-300)), 0)
        return (z.double() - c * log).sum()

    def grad(self, z: Tensor) -> Tensor:
        """∇f(z) = 1 − (y + σ²)/(z + σ²)."""
        return 1 - (self.data + self.variance) / (z + self.variance)

    def prox(self, v: Tensor, t: float) -> Tensor:
        """Closed form (see `prox_shifted_poisson`)."""
        return prox_shifted_poisson(v, t, self.data, self.variance)


class PoissonGaussian:
    """Exact Poisson–Gaussian negative log-likelihood (Chouzenoux et al. 2015).

    With yᵢ = Poisson(zᵢ) + N(0, σᵢ²) (Eqs. 2.1–2.4), the function of
    z ∈ [0, +∞)^Q is (Eqs. 2.6–2.7)

        φ(z) = ∑ᵢ φᵢ(zᵢ),
        φᵢ(u) = −log ∑ₙ uⁿe^{−u}/n! · e^{−(yᵢ − n)²/2σᵢ²}/√(2πσᵢ²),

    computed by
    [`poisson_normal_logpdf`][optix.special.poisson_normal_logpdf].
    Its derivative is φᵢ′(u) = 1 − ξᵢ(u), ξᵢ(u) = s(u, yᵢ − 1)/s(u, yᵢ)
    (Eqs. 2.8, 2.10, 2.12; see
    [`poisson_normal_ratio`][optix.special.poisson_normal_ratio]). It is
    convex, with φᵢ″(u) = ηᵢ(u) ≤ ηᵢ(0) (Eq. 2.11, Theorem 2.1). For
    u < 0 it is continued by the quadratic of Eq. (2.28),

        φᵢ(u) = ½ηᵢ(0)u² + (1 − ξᵢ(0))u + φᵢ(0),
        ξᵢ(0) = e^{(2yᵢ − 1)/2σᵢ²},
        ηᵢ(0) = (1 − e^{−1/σᵢ²}) e^{(2yᵢ − 1)/σᵢ²}

    (Eqs. 2.29–2.30), so that φ is convex and twice differentiable on
    ℝ^Q with a gradient of Lipschitz constant maxᵢ ηᵢ(0). The data need
    no clipping. The Lipschitz constant grows like e^{2 max y/σ²}: at
    high counts with low read noise it is too large to be useful.
    """

    def __init__(
        self,
        data: Tensor,
        variance: Tensor | float,
        *,
        max_terms: int = _SERIES_TERMS,
    ) -> None:
        """Set the data and the read-noise variance.

        Parameters
        ----------
        data : Tensor
            Data y (in electrons).
        variance : Tensor or float
            Read-noise variance σ² > 0 (in electrons²); broadcasts with
            `data`.
        max_terms : int
            Series terms evaluated at once (memory bound).
        """
        var = torch.as_tensor(variance)
        if not bool((var > 0).all()):
            raise ValueError(
                "PoissonGaussian needs a positive read-noise variance; "
                "use ShiftedPoisson for Poisson data"
            )
        self.data = data
        self.variance = variance
        self.max_terms = int(max_terms)

    def _at_zero(self, like: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """ξ(0), η(0) and φ(0) per element, in float64."""
        y = torch.as_tensor(self.data).to(device=like.device).double()
        var = torch.as_tensor(self.variance).to(device=like.device).double()
        a = (2 * y - 1) / var
        xi = torch.exp(a / 2)
        eta = -torch.expm1(-1 / var) * torch.exp(a)
        phi = y.square() / (2 * var) + 0.5 * torch.log(2 * math.pi * var)
        return xi, eta, phi

    @property
    def log_lipschitz(self) -> float:
        """log maxᵢ ηᵢ(0), finite even when the constant overflows."""
        y = torch.as_tensor(self.data).double()
        var = torch.as_tensor(self.variance).double().to(y.device)
        log_eta = torch.log(-torch.expm1(-1 / var)) + (2 * y - 1) / var
        return float(log_eta.max())

    @property
    def lipschitz(self) -> float:
        """Lipschitz constant maxᵢ ηᵢ(0) of ∇φ (Theorem 2.1); may be inf."""
        log = self.log_lipschitz
        return math.exp(log) if log < 709 else math.inf

    def value(self, z: Tensor) -> Tensor:
        """φ(z) in float64."""
        zd = z.double()
        y = torch.as_tensor(self.data).to(zd)
        var = torch.as_tensor(self.variance).to(zd)
        inside = -poisson_normal_logpdf(
            y, zd.clamp(min=0), var, max_terms=self.max_terms
        )
        xi, eta, phi = self._at_zero(zd)
        outside = 0.5 * eta * zd.square() + (1 - xi) * zd + phi
        return torch.where(zd >= 0, inside, outside).sum()

    def grad(self, z: Tensor) -> Tensor:
        """∇φ(z) = 1 − ξ(z), with ξ(z) = ξ(0) − η(0)·z for z < 0.

        1 − ξ is formed in float64 (float32 on MPS) and then cast to the
        dtype of `z`. Raises a `ValueError` when the gradient is not
        finite in that dtype (ξ overflows at large y/σ²).
        """
        wd = torch.float32 if z.device.type == "mps" else torch.float64
        zw = z.to(wd)
        data = torch.as_tensor(self.data).to(device=z.device, dtype=wd)
        var = self.variance
        if isinstance(var, Tensor):
            var = var.to(device=z.device, dtype=wd)
        xi = poisson_normal_ratio(
            data, zw.clamp(min=0), var, max_terms=self.max_terms
        )
        neg = zw < 0
        if bool(neg.any()):
            xi0, eta0, _ = self._at_zero(zw)
            xi = torch.where(neg, (xi0 - eta0 * zw).to(wd), xi)
        out = (1 - xi).to(z.dtype)
        if not bool(torch.isfinite(out).all()):
            raise ValueError(
                "PoissonGaussian: the gradient 1 − ξ is not finite in "
                f"{z.dtype} (ξ overflows: max y/σ² is too large for the "
                "exact likelihood); use ShiftedPoisson (Ikoma) instead"
            )
        return out
