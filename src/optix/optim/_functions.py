"""Functions with a value and a proximity operator."""

import math
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from ._prox import (
    prox_box,
    prox_group_l2,
    prox_shifted_poisson,
    prox_squared_l2,
)

__all__ = [
    "Box",
    "Function",
    "GroupL2",
    "ShiftedPoisson",
    "SquaredL2",
]


@runtime_checkable
class Function(Protocol):
    """A proper, closed, convex function f with an easy prox."""

    def value(self, z: Tensor) -> Tensor:
        """f(z), a float64 scalar (+inf outside the domain)."""
        ...

    def prox(self, v: Tensor, t: float) -> Tensor:
        """prox_{t f}(v) = argmin_z f(z) + ‖z − v‖² / (2t)."""
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
