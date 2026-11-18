"""Categorical distribution (labels or one-hot / soft assignments)."""

from collections.abc import Sequence
from typing import NamedTuple, Self

import torch
from torch import Tensor

from . import functional as F
from ._base import Distribution, _is_compiling, _register_kl

__all__ = ["Categorical", "CategoricalExpectations"]


class CategoricalExpectations(NamedTuple):
    """E[onehot(z)] (the probabilities), like the natural parameters."""

    x: Tensor


class Categorical(Distribution):
    """Categorical distribution over ``K`` categories (event shape ``()``).

    Notes
    -----
    Give exactly one of ``probs`` or ``logits``; it is stored as given
    (unnormalised, categories in the last dim) and the normalised forms
    are derived on access. ``Categorical(logits=elog_pi + elog_lik)`` is
    the variational update of responsibilities.

    **Values.** The representation of a value is read from its dtype:

    - integer dtype: labels in ``[0, K)``, shape ``(*batch,)``;
    - floating or boolean dtype: one-hot or soft assignments ``r``, shape
      ``(*batch, K)``; ``log_prob(r)`` = ∑ₖ rₖ ln pₖ is the expected
      log-likelihood used by variational Bayes.

    Integers always mean labels, including the int64 output of
    ``torch.nn.functional.one_hot``: cast it with ``.float()`` or
    ``.bool()`` first.
    [`sample`][optix.distributions.Categorical.sample] returns int64
    labels, or float one-hot vectors with ``one_hot=True``. Expectations
    ([`mean`][optix.distributions.Categorical.mean] = ``probs``), entropy
    and KL do not depend on the representation. There is no
    reparameterised sampling (as in torch,
    [`rsample`][optix.distributions.Distribution.rsample] raises).

    Exponential family: T(z) = onehot(z), η = ln p (up to an additive
    constant), A = logsumexp(η), E[T] = p.
    """

    _forms = {("logits",): (1,), ("probs",): (1,)}
    _canonical = ("logits",)
    _constraints = {"logits": "logits", "probs": "nonnegative"}
    _ndims = {"norm_logits": 1, "norm_probs": 1, "eta": 1}
    _event_ndim = 0

    def __init__(
        self,
        probs: Tensor | None = None,
        *,
        logits: Tensor | None = None,
        validate: bool = False,
        cache: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Create a Categorical distribution.

        Parameters
        ----------
        probs : (*batch, K) Tensor, optional
            Non-negative weights, normalised by their sum.
        logits : (*batch, K) Tensor, optional
            Log-weights, normalised by ``log_softmax``.
        validate, cache, dtype, device
            See [`Gamma`][optix.distributions.Gamma]. ``validate=True``
            also makes [`log_prob`][optix.distributions.Categorical.log_prob]
            check that integer labels are in range (kept by ``expand``,
            indexing, ``to`` and ``detach``).
        """
        if (probs is None) == (logits is None):
            raise ValueError(
                "Categorical: give exactly one of probs or logits"
            )
        form = ("probs",) if logits is None else ("logits",)
        self._init(
            form,
            [logits if probs is None else probs],
            validate=validate,
            cache=cache,
            dtype=dtype,
            device=device,
        )

    def _kl_event_shape(self) -> torch.Size:
        return self._raw[0].shape[-1:]

    def _kl_event_desc(self) -> str:
        return f"K = {self.num_categories} categories"

    def _derive(self, name: str) -> Tensor:
        (raw,) = self._raw
        given_probs = self._form == ("probs",)
        if name == "norm_logits":
            if given_probs:
                return F.categorical_logits_from_probs(raw)
            return F.categorical_normalize(raw)
        if name == "norm_probs":
            if given_probs:
                return raw / raw.sum(-1, keepdim=True)
            return torch.softmax(raw, -1)
        if name == "eta":  # unnormalised log-weights
            if not given_probs:
                return raw
            pos = raw > 0
            safe = torch.where(pos, raw, torch.ones_like(raw))
            return torch.where(pos, safe.log(), -torch.inf)
        if name == "logits":
            return self._get("norm_logits")
        if name == "probs":
            return self._get("norm_probs")
        return super()._derive(name)

    def params(self) -> tuple[Tensor]:
        """Normalised logits (ln p,), unexpanded."""
        return (self._get("norm_logits"),)

    @property
    def num_categories(self) -> int:
        """Number of categories K."""
        return self._raw[0].shape[-1]

    @property
    def logits(self) -> Tensor:
        """Normalised logits ln p, ``(*batch, K)``."""
        return self._view("norm_logits")

    @property
    def probs(self) -> Tensor:
        """Normalised probabilities p, ``(*batch, K)``."""
        return self._view("norm_probs")

    @property
    def mean(self) -> Tensor:
        """E[onehot(z)] = p, ``(*batch, K)``."""
        return self.probs

    @property
    def variance(self) -> Tensor:
        """Var[onehot(z)ₖ] = pₖ (1 − pₖ), ``(*batch, K)``."""
        p = self._get("norm_probs")
        return self._out(p * (1 - p), 1)

    def entropy(self) -> Tensor:
        """H = −∑ pₖ ln pₖ (0 ln 0 = 0, finite gradients)."""
        return self._out(F.categorical_entropy(*self.params()))

    def log_prob(self, value: Tensor) -> Tensor:
        """Log-probability of labels, or expected log-likelihood.

        Parameters
        ----------
        value : Tensor
            Integer labels ``(*sample, *batch)``: ln p[value]. Or
            floating / boolean one-hot or soft assignments
            ``(*sample, *batch, K)``: ∑ₖ rₖ ln pₖ (with 0 ln 0 = 0).

        Returns
        -------
        Tensor
            Broadcast of the value's batch shape and
            [`batch_shape`][optix.distributions.Distribution.batch_shape].

        Raises
        ------
        ValueError
            If the last dim of an assignment is not ``K``, or if a label
            is out of range and the object was built with
            ``validate=True`` (that check synchronises and is skipped
            while compiling).

        Notes
        -----
        Without ``validate=True``, labels are not range-checked: an
        out-of-range label raises a ``RuntimeError`` on the CPU and a
        device-side assert on CUDA (which leaves the CUDA context
        unusable), so validate labels when their source is not trusted.
        """
        value = torch.as_tensor(value, device=self.device)
        name = type(self).__name__
        (logits,) = self.params()
        k = self.num_categories
        labels = not (value.is_floating_point() or value.dtype == torch.bool)
        if not labels and (value.ndim < 1 or value.shape[-1] != k):
            raise ValueError(
                f"{name}.log_prob: floating or boolean values are one-hot "
                f"/ soft assignments and must end with K = {k} categories, "
                f"got shape {tuple(value.shape)}"
            )
        batch = value.shape if labels else value.shape[:-1]
        try:
            shape = torch.broadcast_shapes(batch, self._batch_shape)
        except RuntimeError:
            raise ValueError(
                f"{name}.log_prob: value batch shape {tuple(batch)} is not "
                f"broadcastable to {tuple(self._batch_shape)}"
            ) from None
        if not labels:
            value = value.to(self.dtype)
            # 0 * log(0) = 0, with a zero (not NaN) gradient w.r.t. both
            safe = torch.where(value != 0, logits, 0)
            out = (value * safe).sum(-1)
        else:
            check = self._validate and not _is_compiling()
            if check and bool(((value < 0) | (value >= k)).any()):
                raise ValueError(
                    f"{name}.log_prob: labels must be in [0, {k}); integer "
                    "values are labels (cast one-hot vectors to float or "
                    "bool)"
                )
            out = F.categorical_log_prob(value, logits)
        return out.expand(shape)

    def sample(
        self, sample_shape: Sequence[int] = (), *, one_hot: bool = False
    ) -> Tensor:
        """Samples: int64 labels, or float one-hot vectors (``one_hot``).

        Labels have shape ``(*sample_shape, *batch)``; one-hot vectors
        ``(*sample_shape, *batch, K)`` in the distribution's dtype.
        """
        labels = self.to_torch().sample(torch.Size(sample_shape))
        if not one_hot:
            return labels
        out = torch.nn.functional.one_hot(labels, self.num_categories)
        return out.to(self.dtype)

    def natural_params(self) -> tuple[Tensor]:
        """(η,): the stored log-weights, up to an additive constant.

        For ``logits`` these are the given logits, for ``probs`` their
        logarithm (unnormalised), so that ``log_normalizer`` is
        logsumexp(η).
        """
        return (self._view("eta"),)

    def log_normalizer(self) -> Tensor:
        """A = logsumexp(η) (zero when normalised)."""
        return self._out(F.categorical_log_normalizer(self._get("eta")))

    def expected_stats(self) -> CategoricalExpectations:
        """(E[onehot(z)],) = (p,), whatever the view."""
        return CategoricalExpectations(self.probs)

    @classmethod
    def from_natural(cls, eta: Tensor) -> Self:
        """Distribution with logits ``eta`` (η), stored as given."""
        return cls._new(eta, form=("logits",))

    def to_torch(self) -> torch.distributions.Categorical:
        """``torch.distributions.Categorical(logits=logits)``."""
        return torch.distributions.Categorical(
            logits=self.logits, validate_args=False
        )


@_register_kl(Categorical, Categorical)
def _kl_categorical(q: Categorical, p: Categorical) -> Tensor:
    return F.categorical_kl(*q.params(), *p.params())
