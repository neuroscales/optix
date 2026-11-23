"""Dirichlet distribution."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, Self

import torch
from torch import Tensor

from ..special import fit_dirichlet
from . import functional as F
from ._base import Distribution, _register_kl
from ._stats import REQUIRED, DirichletStats, _as_stats

if TYPE_CHECKING:
    from ..io import ArrayLike

__all__ = ["Dirichlet", "DirichletExpectations"]


class DirichletExpectations(NamedTuple):
    """E[T(x)] of a Dirichlet, ordered like its natural parameters."""

    log_x: Tensor


class Dirichlet(Distribution):
    """Dirichlet distribution over the simplex (event shape `(K,)`).

    p(π) = Γ(α₀) ∏ₖ πₖ^(αₖ − 1) / ∏ₖ Γ(αₖ) with α₀ = ∑ₖ αₖ.

    !!! note
        Exponential family: T(π) = log π, η = α − 1,
        A = ∑ₖ log Γ(αₖ) − log Γ(α₀), h = 1.
    """

    _forms = {("concentration",): (1,)}
    _canonical = ("concentration",)
    _constraints = {"concentration": "positive"}
    _event_ndim = 1

    def __init__(
        self,
        concentration: Tensor,
        *,
        validate: bool = False,
        cache: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Create a Dirichlet distribution.

        Parameters
        ----------
        concentration : (*batch, K) Tensor
            Concentration α > 0.
        validate, cache, dtype, device
            See [`Gamma`][optix.distributions.Gamma].
        """
        self._init(
            ("concentration",),
            [concentration],
            validate=validate,
            cache=cache,
            dtype=dtype,
            device=device,
        )

    @property
    def concentration(self) -> Tensor:
        """Concentration α, `(*batch, K)`."""
        return self._view("concentration")

    @property
    def mean(self) -> Tensor:
        """E[π] = α / α₀, `(*batch, K)`."""
        return self._out(F.dirichlet_mean(*self.params()), 1)

    @property
    def variance(self) -> Tensor:
        """Var[πₖ] = αₖ (α₀ − αₖ) / (α₀² (α₀ + 1))."""
        return self._out(F.dirichlet_variance(*self.params()), 1)

    @property
    def mean_log(self) -> Tensor:
        """E[log π] = ψ(α) − ψ(α₀), `(*batch, K)`."""
        return self._out(F.dirichlet_mean_log(*self.params()), 1)

    def entropy(self) -> Tensor:
        """∑ₖ log Γ(αₖ) − log Γ(α₀) + (α₀ − K) ψ(α₀) − ∑ₖ (αₖ − 1) ψ(αₖ)."""
        return self._out(F.dirichlet_entropy(*self.params()))

    def log_prob(self, value: Tensor) -> Tensor:
        """log Γ(α₀) − ∑ₖ log Γ(αₖ) + ∑ₖ (αₖ − 1) log πₖ."""
        value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        shape = self._check_value_shape(value)
        return F.dirichlet_log_prob(value, *self.params()).expand(shape)

    def natural_params(self) -> tuple[Tensor]:
        """(α − 1,)."""
        (alpha,) = self.params()
        return (self._out(alpha - 1, 1),)

    def log_normalizer(self) -> Tensor:
        """A = ∑ₖ log Γ(αₖ) − log Γ(α₀)."""
        return self._out(F.dirichlet_log_normalizer(*self.params()))

    def expected_stats(self) -> DirichletExpectations:
        """(E[log π],)."""
        return DirichletExpectations(self.mean_log)

    @classmethod
    def from_natural(cls, eta_log_x: Tensor) -> Self:
        """Dirichlet with natural parameters (α − 1,)."""
        return cls._new(eta_log_x + 1)

    def to_torch(self) -> torch.distributions.Dirichlet:
        """`torch.distributions.Dirichlet(concentration)`."""
        return torch.distributions.Dirichlet(
            self.concentration, validate_args=False
        )

    @classmethod
    def fit(
        cls,
        data: "DirichletStats | Tensor | ArrayLike",
        init: "Dirichlet | Tensor",
        weight: Tensor | None = None,
        *,
        dim: int | Sequence[int] | None = REQUIRED,
        niter: int = 16,
    ) -> Self:
        """Maximum-likelihood Dirichlet from samples or accumulated statistics.

        With n = ∑ₙ wₙ and sₖ = ∑ₙ wₙ log xₙₖ, maximises
        n (log Γ(α₀) − ∑ₖ log Γ(αₖ)) + ∑ₖ (αₖ − 1) sₖ with
        [`fit_dirichlet`][optix.special.fit_dirichlet], started at
        `init`.

        Parameters
        ----------
        data : DirichletStats, Tensor or array-like
            Either accumulated statistics, or points xₙ of the simplex
            with shape `(..., K)`. Points are reduced with
            [`DirichletStats.from_samples`][optix.distributions.DirichletStats.from_samples].
            A Tensor is reduced in one go. Other array-likes (numpy
            arrays, memory maps, zarr or h5py arrays, ...) are read in
            chunks along the first reduced dim, sized by the memory
            budget of [`config`][optix.config], and their statistics
            are merged in float64.
        init : Dirichlet or (..., K) Tensor
            Starting point (required: Newton's method only converges
            locally; pass the current estimate). Returned where
            n = 0.
        weight : Tensor, optional
            Non-negative frequency weights (or a boolean mask) of the
            points, broadcastable to `data.shape[:-1]`. Points only.
        dim : int or sequence of int or None
            Reduced batch dims of the points (`None`: all, `()`: none;
            the last dim is the event). Required with points; not
            accepted with statistics.
        niter : int
            Iterations.

        Returns
        -------
        Dirichlet
            Batch shape: that of the statistics, broadcast with `init`.

        Raises
        ------
        TypeError
            If `data` is neither a
            [`DirichletStats`][optix.distributions.DirichletStats] nor
            array-like, or if `dim` is missing with points.
        """
        stats = _as_stats(
            DirichletStats, "Dirichlet.fit", data, weight, dim, 1
        )
        if isinstance(init, Dirichlet):
            alpha0 = init._get("concentration")
        else:
            alpha0 = torch.as_tensor(init, dtype=stats.dtype)
        n, slx = stats.astuple()
        n = n.to(slx.dtype)  # int64 / float64 count
        alpha0 = alpha0.to(device=slx.device)
        ok = (n > 0)[..., None]
        safe = torch.where(ok, n[..., None], 1)
        # where n == 0, the start is already stationary
        elog = torch.where(ok, slx / safe, F.dirichlet_mean_log(alpha0))
        alpha = fit_dirichlet(elog, alpha0, niter)
        return cls._new(torch.where(ok, alpha, alpha0))


@_register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet(q: Dirichlet, p: Dirichlet) -> Tensor:
    return F.dirichlet_kl(*q.params(), *p.params())
