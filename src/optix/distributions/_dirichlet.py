"""Dirichlet distribution."""

from typing import NamedTuple, Self

import torch
from torch import Tensor

from ..special import fit_dirichlet
from . import functional as F
from ._base import Distribution, _register_kl
from ._stats import DirichletStats

__all__ = ["Dirichlet", "DirichletExpectations"]


class DirichletExpectations(NamedTuple):
    """``E[T(x)]`` of a Dirichlet, ordered like its natural parameters."""

    log_x: Tensor


class Dirichlet(Distribution):
    """Dirichlet distribution over the simplex (event shape ``(K,)``).

    ``p(pi) = Gamma(a0) / prod(Gamma(alpha)) prod(pi^(alpha - 1))`` with
    ``a0 = sum(alpha)``.

    Parameters
    ----------
    concentration : (*batch, K) Tensor
        Concentration ``alpha > 0``.
    validate, cache, dtype, device
        See :class:`Gamma`.

    Notes
    -----
    Exponential family: ``T(pi) = log(pi)``, ``eta = alpha - 1``,
    ``A = sum(lgamma(alpha)) - lgamma(a0)``, ``h = 1``.
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
        """Concentration ``alpha``, ``(*batch, K)``."""
        return self._view("concentration")

    @property
    def mean(self) -> Tensor:
        """``E[pi] = alpha / a0``, ``(*batch, K)``."""
        return self._out(F.dirichlet_mean(*self.params()), 1)

    @property
    def variance(self) -> Tensor:
        """``Var[pi_k] = alpha_k (a0 - alpha_k) / (a0^2 (a0 + 1))``."""
        return self._out(F.dirichlet_variance(*self.params()), 1)

    @property
    def mean_log(self) -> Tensor:
        """``E[log pi] = digamma(alpha) - digamma(a0)``, ``(*batch, K)``."""
        return self._out(F.dirichlet_mean_log(*self.params()), 1)

    def entropy(self) -> Tensor:
        """``sum(lgamma(alpha)) - lgamma(a0) + (a0 - K) digamma(a0)
        - sum((alpha - 1) digamma(alpha))``.
        """
        return self._out(F.dirichlet_entropy(*self.params()))

    def log_prob(self, value: Tensor) -> Tensor:
        """``lgamma(a0) - sum(lgamma(alpha)) + sum((alpha - 1) log pi)``."""
        value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        shape = self._check_value_shape(value)
        return F.dirichlet_log_prob(value, *self.params()).expand(shape)

    def natural_params(self) -> tuple[Tensor]:
        """``(alpha - 1,)``."""
        (alpha,) = self.params()
        return (self._out(alpha - 1, 1),)

    def log_normalizer(self) -> Tensor:
        """``A = sum(lgamma(alpha)) - lgamma(a0)``."""
        return self._out(F.dirichlet_log_normalizer(*self.params()))

    def expected_stats(self) -> DirichletExpectations:
        """``(E[log pi],)``."""
        return DirichletExpectations(self.mean_log)

    @classmethod
    def from_natural(cls, eta_log_x: Tensor) -> Self:
        """Dirichlet with natural parameters ``(alpha - 1,)``."""
        return cls._new(eta_log_x + 1)

    def to_torch(self) -> torch.distributions.Dirichlet:
        """``torch.distributions.Dirichlet(concentration)``."""
        return torch.distributions.Dirichlet(
            self.concentration, validate_args=False
        )

    @classmethod
    def fit(
        cls,
        stats: DirichletStats,
        init: "Dirichlet | Tensor",
        *,
        niter: int = 16,
    ) -> Self:
        """Maximum-likelihood (type-II) Dirichlet from mean log-probabilities.

        Maximises ``n (lgamma(a0) - sum(lgamma(alpha)))
        + sum((alpha - 1) sum_log_x)`` with
        :func:`optix.special.fit_dirichlet`, started at ``init``.

        Parameters
        ----------
        stats : DirichletStats
            Accumulated statistics.
        init : Dirichlet or (..., K) Tensor
            Starting point (required: Newton's method only converges
            locally; pass the current estimate). Returned where
            ``n == 0``.
        niter : int
            Iterations.

        Returns
        -------
        Dirichlet
            Batch shape ``broadcast(stats.batch_shape, init)``.
        """
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
