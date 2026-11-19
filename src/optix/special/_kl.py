"""Closed-form Kullback-Leibler divergences KL(q ‖ p)."""

import torch
from torch import Tensor

from ._solvers import log_minus_digamma as _lmd
from ._stable import ratio_minus_log as _g
from ._stable import stirling_residual as _res


def kl_dirichlet(a: Tensor, b: Tensor) -> Tensor:
    """KL(Dir(a) ‖ Dir(b)).

    Parameters
    ----------
    a, b : (..., K) Tensor
        Concentration parameters of `q` and `p` (last dim =
        categories).

    Returns
    -------
    (...) Tensor
        Divergence, summed over the last dim.

    Notes
    -----
    With a₀ = ∑ₖ aₖ and b₀ = ∑ₖ bₖ,
    KL = log Γ(a₀) − log Γ(b₀) − ∑ₖ (log Γ(aₖ) − log Γ(bₖ)) +
    ∑ₖ (aₖ − bₖ) (ψ(aₖ) − ψ(a₀)), rewritten with
    [`stirling_residual`][optix.special.stirling_residual] r,
    [`log_minus_digamma`][optix.special.log_minus_digamma] lmd and
    [`ratio_minus_log`][optix.special.ratio_minus_log] g as

    ```text
    ∑ₖ (bₖ g(aₖ/bₖ) + ½ log(aₖ/bₖ)) − (b₀ g(a₀/b₀) + ½ log(a₀/b₀))
    + r(a₀) − r(b₀) − ∑ₖ (r(aₖ) − r(bₖ))
    − ∑ₖ (aₖ − bₖ) lmd(aₖ) + (a₀ − b₀) lmd(a₀)
    ```

    so that the log Γ terms of large concentrations cancel exactly.
    """
    a0 = a.sum(-1)
    b0 = b.sum(-1)
    diff = a - b
    diff0 = diff.sum(-1)
    return (
        (b * _g(a, b, diff) + 0.5 * (a / b).log()).sum(-1)
        - (b0 * _g(a0, b0, diff0) + 0.5 * (a0 / b0).log())
        + _res(a0)
        - _res(b0)
        - (_res(a) - _res(b)).sum(-1)
        - (diff * _lmd(a)).sum(-1)
        + diff0 * _lmd(a0)
    )


def kl_gamma(
    shape_q: Tensor, rate_q: Tensor, shape_p: Tensor, rate_p: Tensor
) -> Tensor:
    """KL(Gamma(shape_q, rate_q) ‖ Gamma(shape_p, rate_p)), elementwise.

    Parameters
    ----------
    shape_q, rate_q : Tensor
        Shape and rate of `q`.
    shape_p, rate_p : Tensor
        Shape and rate of `p`.

    Returns
    -------
    Tensor
        Divergence (broadcast shape of the inputs).

    Notes
    -----
    With shapes a and rates b,

    ```text
    KL = (a_q − a_p) ψ(a_q) − log Γ(a_q) + log Γ(a_p)
         + a_p log(b_q/b_p) + a_q (b_p − b_q) / b_q
    ```

    rewritten with [`stirling_residual`][optix.special.stirling_residual] r,
    [`log_minus_digamma`][optix.special.log_minus_digamma] lmd and
    [`ratio_minus_log`][optix.special.ratio_minus_log] g as

    ```text
    a_p (g(a_q/a_p) + g(b_p/b_q)) + ½ log(a_q/a_p)
    + (a_q − a_p) ((b_p − b_q) / b_q − lmd(a_q)) + r(a_p) − r(a_q)
    ```

    which has no cancellation between large log Γ / ψ values (float32
    stays accurate at shapes ≈ 1e6).
    """
    aq, bq, ap, bp = map(torch.as_tensor, (shape_q, rate_q, shape_p, rate_p))
    return (
        ap * (_g(aq, ap) + _g(bp, bq))
        + 0.5 * (aq / ap).log()
        + (aq - ap) * ((bp - bq) / bq - _lmd(aq))
        + _res(ap)
        - _res(aq)
    )


def kl_normal(
    mean_q: Tensor, var_q: Tensor, mean_p: Tensor, var_p: Tensor
) -> Tensor:
    """KL(N(mean_q, var_q) ‖ N(mean_p, var_p)), elementwise.

    Parameters
    ----------
    mean_q, var_q : Tensor
        Mean and variance of `q`.
    mean_p, var_p : Tensor
        Mean and variance of `p`.

    Returns
    -------
    Tensor
        Divergence (broadcast shape of the inputs).
    """
    mean_q, var_q, mean_p, var_p = map(
        torch.as_tensor, (mean_q, var_q, mean_p, var_p)
    )
    # 0.5 (r - 1 - log(r) + (mq - mp)^2 / vp) with r = vq / vp, via
    # ratio_minus_log (no cancellation when vq ~ vp)
    return 0.5 * (_g(var_q, var_p) + (mean_q - mean_p).square() / var_p)
