"""Normal distribution (scalar, or diagonal through a trailing batch dim)."""

from typing import NamedTuple, Self

import torch
from torch import Tensor

from . import functional as F
from ._base import Distribution, _register_kl

__all__ = ["Normal", "NormalExpectations"]


class NormalExpectations(NamedTuple):
    """``E[T(x)]`` of a Normal, ordered like its natural parameters."""

    x: Tensor
    x_sq: Tensor


class Normal(Distribution):
    """Normal distribution, elementwise (event shape ``()``).

    Parameters
    ----------
    mean : Tensor or float
        Mean ``m``.
    var : Tensor or float, optional
        Variance ``v`` (also the unnamed second argument:
        ``Normal(m, v)``).
    std : Tensor or float, optional
        Standard deviation ``sqrt(v)``.
    precision : Tensor or float, optional
        Precision ``lam = 1 / v``. Zero is a flat (improper) direction:
        entropy and KL are infinite, conjugate updates stay valid.
    validate, cache, dtype, device
        See :class:`Gamma`.

    Notes
    -----
    Give exactly one of ``var``, ``std`` or ``precision``. It is stored as
    given; the others are derived on access. The canonical parameters of
    :mod:`~optix.distributions.functional` are ``(mean, precision)``.

    A "diagonal" multivariate Normal is this class with a trailing batch
    dim of size K; reduce it yourself, e.g. ``kl_divergence(q, p).sum(-1)``.

    Exponential family: ``T(x) = (x, x^2)``, ``eta = (lam m, -lam / 2)``,
    ``A = lam m^2 / 2 - log(lam) / 2``, ``log h = -log(2 pi) / 2``.
    """

    _forms = {
        ("mean", "var"): (0, 0),
        ("mean", "std"): (0, 0),
        ("mean", "precision"): (0, 0),
    }
    _canonical = ("mean", "precision")
    _constraints = {
        "var": "positive",
        "std": "positive",
        "precision": "positive",
    }
    _event_ndim = 0

    def __init__(
        self,
        mean: Tensor | float,
        var: Tensor | float | None = None,
        *,
        std: Tensor | float | None = None,
        precision: Tensor | float | None = None,
        validate: bool = False,
        cache: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        given = {"var": var, "std": std, "precision": precision}
        if sum(v is not None for v in given.values()) != 1:
            raise ValueError(
                "Normal: give exactly one of var, std or precision"
            )
        (key,) = (k for k, v in given.items() if v is not None)
        self._init(
            ("mean", key),
            [mean, given[key]],
            validate=validate,
            cache=cache,
            dtype=dtype,
            device=device,
        )

    def _derive(self, name: str) -> Tensor:
        kind = self._form[1]
        value = self._raw[1]
        if name == "precision":
            return 1 / value if kind == "var" else value.square().reciprocal()
        if name == "var":
            return 1 / value if kind == "precision" else value.square()
        if name == "std":
            return value.sqrt() if kind == "var" else value.rsqrt()
        return super()._derive(name)

    # -- parameters ------------------------------------------------------
    @property
    def mean(self) -> Tensor:
        """Mean ``m = E[x]``, ``(*batch)``."""
        return self._view("mean")

    @property
    def precision(self) -> Tensor:
        """Precision ``lam = 1 / v``."""
        return self._view("precision")

    @property
    def var(self) -> Tensor:
        """Variance ``v``."""
        return self._view("var")

    @property
    def variance(self) -> Tensor:
        """Variance ``Var[x] = v`` (alias of :attr:`var`)."""
        return self._view("var")

    @property
    def std(self) -> Tensor:
        """Standard deviation ``sqrt(v)``."""
        return self._view("std")

    # -- expectations ----------------------------------------------------
    @property
    def mean_sq(self) -> Tensor:
        """``E[x^2] = m^2 + v``."""
        return self._out(self._get("mean").square() + self._get("var"))

    def entropy(self) -> Tensor:
        """``H = (1 + log(2 pi) - log(lam)) / 2``."""
        return self._out(F.normal_entropy(self._get("precision")))

    def log_prob(self, value: Tensor) -> Tensor:
        """``(log(lam) - log(2 pi) - lam (x - m)^2) / 2``."""
        value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        shape = self._check_value_shape(value)
        if self._form[1] == "var":
            out = F.normal_log_prob_var(value, *self._raw)
        else:
            out = F.normal_log_prob(value, *self.params())
        return out.expand(shape)

    def natural_params(self) -> tuple[Tensor, Tensor]:
        """``(lam m, -lam / 2)``."""
        m, lam = self.params()
        return self._out(lam * m), self._out(-0.5 * lam)

    def log_normalizer(self) -> Tensor:
        """``A = lam m^2 / 2 - log(lam) / 2``."""
        return self._out(F.normal_log_normalizer(*self.params()))

    def expected_stats(self) -> NormalExpectations:
        """``(E[x], E[x^2]) = (m, m^2 + v)``."""
        return NormalExpectations(self.mean, self.mean_sq)

    @classmethod
    def from_natural(cls, eta_x: Tensor, eta_x_sq: Tensor) -> Self:
        """Normal with natural parameters ``(lam m, -lam / 2)``."""
        lam = -2 * eta_x_sq
        return cls._new(eta_x / lam, lam)

    def to_torch(self) -> torch.distributions.Normal:
        """``torch.distributions.Normal(mean, std)``."""
        return torch.distributions.Normal(
            self.mean, self.std, validate_args=False
        )


@_register_kl(Normal, Normal)
def _kl_normal(q: Normal, p: Normal) -> Tensor:
    return F.normal_kl(*q.params(), *p.params())
