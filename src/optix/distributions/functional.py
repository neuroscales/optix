"""Pure tensor functions of the distribution families (the maths).

Every function takes and returns tensors, broadcasts its inputs, keeps
event dims last, never reads the configuration, never synchronises with
the host and has no data-dependent Python control flow, so it can be
inlined into a `torch.compile` kernel without graph breaks. Nothing is
validated: invalid parameters give NaN or infinities.

Each family has one canonical signature:

- Gamma: shape `a` and rate `b`;
- Dirichlet: concentration `alpha` of shape `(..., K)`;
- Categorical: normalised logits `l` = log p of shape `(..., K)`;
- Normal: mean `m` and precision `lam` (plus
  [`normal_log_prob_var`][optix.distributions.functional.normal_log_prob_var]
  in variance form, for likelihood kernels).

KL divergences take all parameters of `q`, then all of `p`, and are
not clamped at zero (in float32 nearly equal distributions can give a
slightly negative value).
"""

import math

import torch
from torch import Tensor

from ..special import (
    dirichlet_elog,
    kl_dirichlet,
    kl_gamma,
    log_minus_digamma,
    ratio_minus_log,
    stirling_residual,
)

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)

__all__ = [
    "categorical_entropy",
    "categorical_kl",
    "categorical_log_normalizer",
    "categorical_log_prob",
    "categorical_logits_from_probs",
    "categorical_normalize",
    "dirichlet_entropy",
    "dirichlet_kl",
    "dirichlet_log_normalizer",
    "dirichlet_log_prob",
    "dirichlet_mean",
    "dirichlet_mean_log",
    "dirichlet_variance",
    "gamma_entropy",
    "gamma_kl",
    "gamma_log_normalizer",
    "gamma_log_prob",
    "gamma_mean",
    "gamma_mean_inv",
    "gamma_mean_log",
    "gamma_mean_log_ms",
    "gamma_variance",
    "normal_entropy",
    "normal_kl",
    "normal_log_normalizer",
    "normal_log_prob",
    "normal_log_prob_var",
    "normal_mean_sq",
    "normal_variance",
]

_LOG_2PI = math.log(2 * math.pi)

# -- Gamma -------------------------------------------------------------


def gamma_mean(a: Tensor, b: Tensor) -> Tensor:
    """Mean of Gamma(a, b): E[x] = a / b.

    Parameters
    ----------
    a, b : Tensor
        Shape and rate.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs.
    """
    return a / b


def gamma_variance(a: Tensor, b: Tensor) -> Tensor:
    """Variance of Gamma(a, b): Var[x] = a / b².

    Parameters
    ----------
    a, b : Tensor
        Shape and rate.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs.
    """
    return a / b.square()


def gamma_mean_log(a: Tensor, b: Tensor) -> Tensor:
    """E[log x] = ψ(a) − log b under Gamma(a, b).

    Parameters
    ----------
    a, b : Tensor
        Shape and rate.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs. Computed as
        log(a / b) − lmd(a), with lmd =
        [`log_minus_digamma`][optix.special.log_minus_digamma] (no
        cancellation for large a and b).
    """
    return (a / b).log() - log_minus_digamma(a)


def gamma_mean_log_ms(mean: Tensor, shape: Tensor) -> Tensor:
    """E[log x] of a Gamma given its mean μ and shape ν.

    E[log x] = log μ − (log ν − ψ(ν)), which avoids the cancellation between
    ψ(ν) and log(rate) for large ν (see
    [`log_minus_digamma`][optix.special.log_minus_digamma]).

    Parameters
    ----------
    mean, shape : Tensor
        Mean μ = a / b and shape ν = a.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs.
    """
    return mean.log() - log_minus_digamma(shape)


def gamma_mean_inv(a: Tensor, b: Tensor) -> Tensor:
    """E[1 / x] = b / (a − 1) under Gamma(a, b); `inf` if a ≤ 1.

    Parameters
    ----------
    a, b : Tensor
        Shape and rate.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs. The gradient is finite (zero)
        where the value is infinite.
    """
    ok = a > 1
    safe = torch.where(ok, a - 1, torch.ones_like(a))
    return torch.where(ok, b / safe, torch.inf)


def gamma_entropy(a: Tensor, b: Tensor) -> Tensor:
    """Entropy of Gamma(a, b).

    H = a − log b + log Γ(a) + (1 − a) ψ(a), computed as
    ½ log(2π a) + r(a) + (a − 1) lmd(a) − log b with
    [`stirling_residual`][optix.special.stirling_residual] r and
    [`log_minus_digamma`][optix.special.log_minus_digamma] lmd, so that
    the large log Γ and ψ terms do not cancel at large a.

    Parameters
    ----------
    a, b : Tensor
        Shape and rate.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs.
    """
    return (
        0.5 * a.log()
        + _HALF_LOG_2PI
        + stirling_residual(a)
        + (a - 1) * log_minus_digamma(a)
        - b.log()
    )


def gamma_log_prob(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Log-density of Gamma(a, b) at x > 0.

    log p(x) = a log b − log Γ(a) + (a − 1) log x − b x, computed for x > 0 as
    −a g(b x / a) + ½ log a − log x − ½ log(2π) − r(a) with
    [`ratio_minus_log`][optix.special.ratio_minus_log] g and
    [`stirling_residual`][optix.special.stirling_residual] r (no
    cancellation between a log(b x), log Γ(a) and b x at large a); the
    direct formula is used at x = 0.

    Parameters
    ----------
    x : Tensor
        Values.
    a, b : Tensor
        Shape and rate.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs.
    """
    pos = x > 0
    xs = torch.where(pos, x, 1)
    stable = (
        -a * ratio_minus_log(b * xs, a)
        + 0.5 * a.log()
        - xs.log()
        - _HALF_LOG_2PI
        - stirling_residual(a)
    )
    direct = a * b.log() - torch.lgamma(a) + torch.xlogy(a - 1, x) - b * x
    return torch.where(pos, stable, direct)


def gamma_log_normalizer(a: Tensor, b: Tensor) -> Tensor:
    """Log-normalizer A = log Γ(a) − a log b of Gamma(a, b).

    With sufficient statistics T(x) = (log x, x) and natural parameters
    η = (a − 1, −b), ∂A/∂η = E[T].

    Parameters
    ----------
    a, b : Tensor
        Shape and rate.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs.
    """
    return torch.lgamma(a) - a * b.log()


gamma_kl = kl_gamma
"""KL(Gamma(aq, bq) ‖ Gamma(ap, bp)), elementwise.

```text
KL = (aq − ap) ψ(aq) − log Γ(aq) + log Γ(ap)
     + ap (log bq − log bp) + aq (bp − bq) / bq
```

The same function object as [`kl_gamma`][optix.special.kl_gamma].
"""

# -- Dirichlet ---------------------------------------------------------


def dirichlet_mean(alpha: Tensor) -> Tensor:
    """Mean E[π] = α / α₀ of Dir(α), α₀ = ∑ₖ αₖ.

    Parameters
    ----------
    alpha : (..., K) Tensor
        Concentration.

    Returns
    -------
    (..., K) Tensor
    """
    return alpha / alpha.sum(-1, keepdim=True)


def dirichlet_variance(alpha: Tensor) -> Tensor:
    """Marginal variances of Dir(α).

    Var[πₖ] = αₖ (α₀ − αₖ) / (α₀² (α₀ + 1)).

    Parameters
    ----------
    alpha : (..., K) Tensor
        Concentration.

    Returns
    -------
    (..., K) Tensor
    """
    a0 = alpha.sum(-1, keepdim=True)
    return alpha * (a0 - alpha) / (a0.square() * (a0 + 1))


dirichlet_mean_log = dirichlet_elog
"""E[log πₖ] = ψ(αₖ) − ψ(α₀) under Dir(α).

The same function object as
[`dirichlet_elog`][optix.special.dirichlet_elog].
"""


def dirichlet_entropy(alpha: Tensor) -> Tensor:
    """Entropy of Dir(α).

    H = ∑ₖ log Γ(αₖ) − log Γ(α₀) + (α₀ − K) ψ(α₀) − ∑ₖ (αₖ − 1) ψ(αₖ),
    computed with [`stirling_residual`][optix.special.stirling_residual] r
    and [`log_minus_digamma`][optix.special.log_minus_digamma] lmd as

    ```text
    ½ ∑ₖ log(αₖ/α₀) − ½ (K − 1) log(α₀/(2π)) + ∑ₖ r(αₖ) − r(α₀)
    + ∑ₖ (αₖ − 1) lmd(αₖ) − (α₀ − K) lmd(α₀)
    ```

    (no cancellation at large concentrations).

    Parameters
    ----------
    alpha : (..., K) Tensor
        Concentration.

    Returns
    -------
    (...) Tensor
    """
    a0 = alpha.sum(-1)
    k = alpha.shape[-1]
    return (
        0.5 * (alpha / a0[..., None]).log().sum(-1)
        - 0.5 * (k - 1) * (a0.log() - _LOG_2PI)
        + stirling_residual(alpha).sum(-1)
        - stirling_residual(a0)
        + ((alpha - 1) * log_minus_digamma(alpha)).sum(-1)
        - (a0 - k) * log_minus_digamma(a0)
    )


def _dirichlet_log_norm(alpha: Tensor) -> Tensor:
    """log Γ(α₀) − ∑ₖ log Γ(αₖ), without the Stirling terms.

    Equal to −∑ₖ (αₖ − ½) log(αₖ/α₀) + ½ (K − 1) log(α₀/(2π)) + r(α₀)
    − ∑ₖ r(αₖ), whose terms grow like α log(α/α₀) instead of α log α.
    """
    a0 = alpha.sum(-1)
    k = alpha.shape[-1]
    return (
        -((alpha - 0.5) * (alpha / a0[..., None]).log()).sum(-1)
        + 0.5 * (k - 1) * (a0.log() - _LOG_2PI)
        + stirling_residual(a0)
        - stirling_residual(alpha).sum(-1)
    )


def dirichlet_log_prob(x: Tensor, alpha: Tensor) -> Tensor:
    """Log-density of Dir(α) at a point `x` of the simplex.

    log p(x) = log Γ(α₀) − ∑ₖ log Γ(αₖ) + ∑ₖ (αₖ − 1) log xₖ.

    Parameters
    ----------
    x : (..., K) Tensor
        Values.
    alpha : (..., K) Tensor
        Concentration.

    Returns
    -------
    (...) Tensor
        Computed with the mean m = α / α₀ and
        [`stirling_residual`][optix.special.stirling_residual] r as
        ∑ₖ (αₖ − 1) log(xₖ/mₖ) − ½ ∑ₖ log mₖ + ½ (K − 1) log(α₀/(2π))
        + r(α₀) − ∑ₖ r(αₖ): the log Γ terms and α log x do not cancel at large
        concentrations.
    """
    a0 = alpha.sum(-1, keepdim=True)
    m = alpha / a0
    k = alpha.shape[-1]
    return (
        torch.xlogy(alpha - 1, x / m).sum(-1)
        - 0.5 * m.log().sum(-1)
        + 0.5 * (k - 1) * (a0[..., 0].log() - _LOG_2PI)
        + stirling_residual(a0[..., 0])
        - stirling_residual(alpha).sum(-1)
    )


def dirichlet_log_normalizer(alpha: Tensor) -> Tensor:
    """Log-normalizer A = ∑ₖ log Γ(αₖ) − log Γ(α₀).

    With T(x) = log x and η = α − 1, ∂A/∂η = E[T].

    Parameters
    ----------
    alpha : (..., K) Tensor
        Concentration.

    Returns
    -------
    (...) Tensor
        Computed without the cancellation of the log Γ terms at large
        concentrations (Stirling residuals, as in
        [`dirichlet_log_prob`][optix.distributions.functional.dirichlet_log_prob]).
    """
    return -_dirichlet_log_norm(alpha)


dirichlet_kl = kl_dirichlet
"""KL(Dir(a) ‖ Dir(b)), summed over the last dim.

```text
KL = log Γ(a₀) − log Γ(b₀) − ∑ₖ (log Γ(aₖ) − log Γ(bₖ))
     + ∑ₖ (aₖ − bₖ) (ψ(aₖ) − ψ(a₀))
```

The same function object as [`kl_dirichlet`][optix.special.kl_dirichlet].
"""

# -- Categorical -------------------------------------------------------


def categorical_normalize(logits: Tensor) -> Tensor:
    """Normalised logits `log_softmax(logits)` (last dim = categories).

    Parameters
    ----------
    logits : (..., K) Tensor
        Unnormalised log-probabilities.

    Returns
    -------
    (..., K) Tensor
        `l` with `logsumexp(l, -1) == 0`.
    """
    return torch.log_softmax(logits, -1)


def categorical_logits_from_probs(probs: Tensor) -> Tensor:
    """Normalised logits log(p / ∑ₖ pₖ) of non-negative weights.

    Zero weights give `-inf` with a zero (finite) gradient, thanks to a
    double `where`.

    Parameters
    ----------
    probs : (..., K) Tensor
        Non-negative, unnormalised probabilities.

    Returns
    -------
    (..., K) Tensor
    """
    pos = probs > 0
    safe = torch.where(pos, probs, torch.ones_like(probs))
    log = torch.where(pos, safe.log(), -torch.inf)
    return log - probs.sum(-1, keepdim=True).log()


def categorical_entropy(logits: Tensor) -> Tensor:
    """Entropy H = −∑ₖ pₖ log pₖ, with p = exp(l), 0 log 0 = 0.

    Parameters
    ----------
    logits : (..., K) Tensor
        Normalised logits `l` = log p.

    Returns
    -------
    (...) Tensor
        Entropy; the gradient stays finite where p = 0 (double
        `where`).
    """
    p = logits.exp()
    pos = p > 0
    safe = torch.where(pos, logits, torch.zeros_like(logits))
    return -torch.where(pos, p * safe, torch.zeros_like(p)).sum(-1)


def categorical_log_prob(index: Tensor, logits: Tensor) -> Tensor:
    """Log-probability `l[index]` of integer labels.

    Parameters
    ----------
    index : Tensor
        Integer labels, broadcastable against `logits.shape[:-1]`.
    logits : (..., K) Tensor
        Normalised logits.

    Returns
    -------
    Tensor
        Broadcast shape of `index` and `logits.shape[:-1]`.
    """
    shape = torch.broadcast_shapes(index.shape, logits.shape[:-1])
    k = logits.shape[-1]
    index = index.long().expand(shape)[..., None]
    return logits.expand(*shape, k).gather(-1, index)[..., 0]


def categorical_log_normalizer(eta: Tensor) -> Tensor:
    """Log-normalizer A = logsumexp(η) (zero for normalised logits).

    With T(z) = onehot(z), ∂A/∂η = E[T] = p.

    Parameters
    ----------
    eta : (..., K) Tensor
        Natural parameters (logits up to an additive constant).

    Returns
    -------
    (...) Tensor
    """
    return torch.logsumexp(eta, -1)


def categorical_kl(lq: Tensor, lp: Tensor) -> Tensor:
    """KL(Cat(q) ‖ Cat(p)) = ∑ₖ qₖ (log qₖ − log pₖ), 0 log 0 = 0.

    Parameters
    ----------
    lq, lp : (..., K) Tensor
        Normalised logits of `q` and `p`.

    Returns
    -------
    (...) Tensor
        Divergence; `+inf` where q > 0 and p = 0. Gradients are finite
        where q = 0 (double `where`).
    """
    q = lq.exp()
    pos = q > 0
    zero = torch.zeros_like(q)
    d = torch.where(pos, lq, zero) - torch.where(pos, lp, zero)
    return torch.where(pos, q * d, zero).sum(-1)


# -- Normal ------------------------------------------------------------


def normal_variance(lam: Tensor) -> Tensor:
    """Variance 1 / λ of a Normal with precision `lam` (λ).

    Parameters
    ----------
    lam : Tensor
        Precision.

    Returns
    -------
    Tensor
    """
    return 1 / lam


def normal_mean_sq(m: Tensor, lam: Tensor) -> Tensor:
    """Second moment E[x²] = μ² + 1 / λ.

    Parameters
    ----------
    m, lam : Tensor
        Mean and precision.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs.
    """
    return m.square() + 1 / lam


def normal_entropy(lam: Tensor) -> Tensor:
    """Entropy H = (1 + log(2π) − log λ) / 2.

    Parameters
    ----------
    lam : Tensor
        Precision.

    Returns
    -------
    Tensor
    """
    return 0.5 * (1 + _LOG_2PI - lam.log())


def normal_log_prob(x: Tensor, m: Tensor, lam: Tensor) -> Tensor:
    """Log-density (log λ − log(2π) − λ (x − μ)²) / 2.

    Parameters
    ----------
    x : Tensor
        Values.
    m, lam : Tensor
        Mean and precision.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs.
    """
    return 0.5 * (lam.log() - _LOG_2PI - lam * (x - m).square())


def normal_log_prob_var(x: Tensor, m: Tensor, var: Tensor) -> Tensor:
    """Log-density in variance form.

    log p(x) = −(log σ² + log(2π) + (x − μ)² / σ²) / 2.

    The form to call inside likelihood kernels whose variance is computed
    per sample (it saves a reciprocal).

    Parameters
    ----------
    x : Tensor
        Values.
    m, var : Tensor
        Mean and variance.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs.
    """
    return -0.5 * (var.log() + _LOG_2PI + (x - m).square() / var)


def normal_log_normalizer(m: Tensor, lam: Tensor) -> Tensor:
    """Log-normalizer A = λμ² / 2 − ½ log λ.

    With T(x) = (x, x²), η = (λμ, −λ/2) and base measure
    log h = −½ log(2π), ∂A/∂η = E[T] = (μ, μ² + 1/λ).

    Parameters
    ----------
    m, lam : Tensor
        Mean and precision.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs.
    """
    return 0.5 * (lam * m.square() - lam.log())


def normal_kl(mq: Tensor, lq: Tensor, mp: Tensor, lp: Tensor) -> Tensor:
    """KL(N(mq, 1/lq) ‖ N(mp, 1/lp)), elementwise, in precision form.

    KL = (log lq − log lp + lp / lq + lp (mq − mp)² − 1) / 2.

    !!! note
        [`kl_normal`][optix.special.kl_normal] takes variances.

    Parameters
    ----------
    mq, lq : Tensor
        Mean and precision of `q`.
    mp, lp : Tensor
        Mean and precision of `p`.

    Returns
    -------
    Tensor
        Broadcast shape of the inputs.
    """
    # r - 1 - log(r) with r = lp / lq, via ratio_minus_log (no cancellation)
    return 0.5 * (ratio_minus_log(lp, lq) + lp * (mq - mp).square())
