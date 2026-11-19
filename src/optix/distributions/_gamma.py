"""Gamma distribution."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, Self

import torch
from torch import Tensor

from ..special import fit_gamma_shape
from . import functional as F
from ._base import Distribution, _register_kl
from ._stats import REQUIRED, GammaStats, _as_stats

if TYPE_CHECKING:
    from ..io import ArrayLike

__all__ = ["Gamma", "GammaExpectations"]


class GammaExpectations(NamedTuple):
    """`E[T(x)]` of a Gamma, ordered like its natural parameters."""

    log_x: Tensor
    x: Tensor


class Gamma(Distribution):
    """Gamma distribution over x > 0, elementwise (event shape `()`).

    p(x) = b^a x^(a − 1) exp(−b x) / Γ(a) with shape a and rate b.

    Notes
    -----
    Give `shape` with exactly one of `rate`, `scale` or `mean`;
    the given pair is stored, everything else is derived on access. The
    canonical parameters of
    [`functional`][optix.distributions.functional] are `(shape, rate)`.

    Exponential family: T(x) = (log x, x), η = (a − 1, −b),
    A = log Γ(a) − a log b, h(x) = 1.
    """

    _forms = {
        ("shape", "rate"): (0, 0),
        ("shape", "scale"): (0, 0),
        ("mean", "shape"): (0, 0),
    }
    _canonical = ("shape", "rate")
    _constraints = {
        "shape": "positive",
        "rate": "positive",
        "scale": "positive",
        "mean": "positive",
    }
    _event_ndim = 0

    def __init__(
        self,
        shape: Tensor | float,
        rate: Tensor | float | None = None,
        *,
        scale: Tensor | float | None = None,
        mean: Tensor | float | None = None,
        validate: bool = False,
        cache: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Create a Gamma distribution from one of its parameterisations.

        Parameters
        ----------
        shape : Tensor or float
            Shape a > 0.
        rate : Tensor or float, optional
            Rate b > 0.
        scale : Tensor or float, optional
            Scale 1 / b.
        mean : Tensor or float, optional
            Mean a / b (the "mean / shape" parameterisation (μ, ν)).
        validate : bool
            Check the parameter values (synchronises). Default False.
        cache : bool
            Cache derived parameters (see
            [`Distribution`][optix.distributions.Distribution]).
        dtype, device
            Default: those of the tensors (see
            [`Distribution`][optix.distributions.Distribution]).
        """
        given = {"shape": shape, "rate": rate, "scale": scale, "mean": mean}
        form = self._pick("Gamma", given, tuple(self._forms))
        self._init(
            form,
            [given[n] for n in form],
            validate=validate,
            cache=cache,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_mean_shape(
        cls,
        mean: Tensor | float,
        shape: Tensor | float,
        **kwargs: object,
    ) -> Self:
        """Gamma with mean μ and shape ν (rate ν / μ)."""
        return cls(shape, mean=mean, **kwargs)

    def _derive(self, name: str) -> Tensor:
        form = self._form
        if name == "rate":
            if "scale" in form:
                return 1 / self._get("scale")
            return self._get("shape") / self._get("mean")
        if name == "scale":
            if "mean" in form:
                return self._get("mean") / self._get("shape")
            return 1 / self._get("rate")
        if name == "mean":
            if "scale" in form:
                return self._get("shape") * self._get("scale")
            return F.gamma_mean(self._get("shape"), self._get("rate"))
        return super()._derive(name)

    # -- parameters ------------------------------------------------------
    @property
    def shape(self) -> Tensor:
        """Shape a, `(*batch)`."""
        return self._view("shape")

    @property
    def rate(self) -> Tensor:
        """Rate b, `(*batch)`."""
        return self._view("rate")

    @property
    def scale(self) -> Tensor:
        """Scale 1 / b, `(*batch)`."""
        return self._view("scale")

    # -- expectations ----------------------------------------------------
    @property
    def mean(self) -> Tensor:
        """E[x] = a / b."""
        return self._view("mean")

    @property
    def variance(self) -> Tensor:
        """Var[x] = a / b²."""
        if "scale" in self._form:
            a, s = self._get("shape"), self._get("scale")
            return self._out(a * s.square())
        return self._out(F.gamma_variance(*self.params()))

    @property
    def mean_log(self) -> Tensor:
        """E[log x] = ψ(a) − log b.

        In the mean / shape parameterisation it is computed as
        log μ − (log ν − ψ(ν)), without cancellation.
        """
        if "mean" in self._form:
            mu, nu = self._raw
            return self._out(F.gamma_mean_log_ms(mu, nu))
        return self._out(F.gamma_mean_log(*self.params()))

    @property
    def mean_inv(self) -> Tensor:
        """E[1 / x] = b / (a − 1); `inf` where a ≤ 1."""
        return self._out(F.gamma_mean_inv(*self.params()))

    def entropy(self) -> Tensor:
        """H = a − log b + log Γ(a) + (1 − a) ψ(a)."""
        return self._out(F.gamma_entropy(*self.params()))

    def log_prob(self, value: Tensor) -> Tensor:
        """a log b − log Γ(a) + (a − 1) log x − b x."""
        value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        shape = self._check_value_shape(value)
        return F.gamma_log_prob(value, *self.params()).expand(shape)

    def natural_params(self) -> tuple[Tensor, Tensor]:
        """(a − 1, −b)."""
        a, b = self.params()
        return self._out(a - 1), self._out(-b)

    def log_normalizer(self) -> Tensor:
        """A = log Γ(a) − a log b."""
        return self._out(F.gamma_log_normalizer(*self.params()))

    def expected_stats(self) -> GammaExpectations:
        """(E[log x], E[x])."""
        return GammaExpectations(self.mean_log, self.mean)

    @classmethod
    def from_natural(cls, eta_log_x: Tensor, eta_x: Tensor) -> Self:
        """Gamma with natural parameters (a − 1, −b)."""
        return cls._new(eta_log_x + 1, -eta_x)

    def to_torch(self) -> torch.distributions.Gamma:
        """`torch.distributions.Gamma(shape, rate)`."""
        return torch.distributions.Gamma(
            self.shape, self.rate, validate_args=False
        )

    # -- fit -------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        data: "GammaStats | Tensor | ArrayLike",
        weight: Tensor | None = None,
        *,
        dim: int | Sequence[int] | None = REQUIRED,
        mean: Tensor | float | None = None,
        init: "Gamma | None" = None,
        niter: int = 16,
    ) -> Self:
        """Maximum-likelihood Gamma from samples or accumulated statistics.

        With n = ∑ₙ wₙ, x̄ = ∑ₙ wₙ xₙ / n and l̄ = ∑ₙ wₙ log xₙ / n, the
        shape ν solves log ν − ψ(ν) = c (see
        [`fit_gamma_shape`][optix.special.fit_gamma_shape]) with

        - `mean` free: μ = x̄ and c = log x̄ − l̄;
        - `mean` fixed: μ = `mean`, ρ = x̄ / μ and
          c = (ρ − 1 − log ρ) + (log x̄ − l̄).

        Parameters
        ----------
        data : GammaStats, Tensor or array-like
            Either accumulated statistics, or positive samples xₙ > 0.
            Samples are reduced with
            [`GammaStats.from_samples`][optix.distributions.GammaStats.from_samples].
            A Tensor is reduced in one go. Other array-likes (numpy
            arrays, memory maps, zarr or h5py arrays, ...) are read in
            chunks along the first reduced dim, sized by the memory
            budget of [`config`][optix.config], and their statistics
            are merged in float64. Statistics are best accumulated in
            float64: in float32, the Jensen gap c ≈ 1 / (2ν) loses its
            digits for large ν.
        weight : Tensor, optional
            Non-negative frequency weights (or a boolean mask) of the
            samples, broadcastable to `data`. Samples only.
        dim : int or sequence of int or None
            Reduced dims of the samples (`None`: all, `()`: none).
            Required with samples; not accepted with statistics.
        mean : Tensor or float, optional
            Fixed mean; only the shape is fitted.
        init : Gamma, optional
            Returned where n = 0; otherwise those entries are NaN.
        niter : int
            Newton iterations.

        Returns
        -------
        Gamma
            Stored in the mean / shape parameterisation (μ, ν), with the
            batch shape of the statistics (broadcast with `mean` and
            `init`). Gradients stay finite where n = 0.

        Raises
        ------
        TypeError
            If `data` is neither a
            [`GammaStats`][optix.distributions.GammaStats] nor
            array-like, or if `dim` is missing with samples.
        """
        stats = _as_stats(GammaStats, "Gamma.fit", data, weight, dim, 0)
        n, sx, slx = stats.astuple()
        n = n.to(sx.dtype)  # int64 / float64 count
        ok = n > 0
        safe = torch.where(ok, n, torch.ones_like(n))
        xbar = torch.where(ok, sx / safe, 1)
        lbar = torch.where(ok, slx / safe, -1)  # c = 1 where n == 0
        if mean is None:
            mu = xbar
            c = xbar.log() - lbar
        else:
            mu = torch.as_tensor(mean, dtype=xbar.dtype, device=xbar.device)
            r = xbar / mu
            c = (r - 1 - r.log()) + (xbar.log() - lbar)
        nu = fit_gamma_shape(c, niter)
        if init is not None:
            mu0, nu0 = init._get("mean"), init._get("shape")
        else:
            mu0 = mu if mean is not None else torch.full_like(mu, torch.nan)
            nu0 = torch.full_like(nu, torch.nan)
        mu = torch.where(ok, mu, mu0)
        nu = torch.where(ok, nu, nu0)
        return cls._new(mu, nu, form=("mean", "shape"))


@_register_kl(Gamma, Gamma)
def _kl_gamma(q: Gamma, p: Gamma) -> Tensor:
    return F.gamma_kl(*q.params(), *p.params())
