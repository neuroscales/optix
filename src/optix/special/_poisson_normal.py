"""Poisson counts observed through additive Gaussian noise.

The observation is r = n + g with n ~ Poisson(μ) and g ~ N(m, σ²)
independent (a CCD pixel: photo-electrons plus read noise). Its density
and the conditional mean of n given r are series over n,

    p(r) = ∑ₙ e^{−μ} μⁿ/n! · N(r − n; m, σ²),
    E[n | r] = ∑ₙ n wₙ / ∑ₙ wₙ,   log wₙ = n log μ − log n! − (r − n − m)²/2σ²,

summed here in log space over a window of n around the posterior mode.
"""

import math

import torch
from torch import Tensor

from ._stable import stirling_residual

__all__ = [
    "poisson_normal_logpdf",
    "poisson_normal_mean",
    "poisson_normal_ratio",
]

#: Mass left out of the window, as a log-ratio to the largest term.
_TAIL = 50.0
#: Default number of series terms held in memory at once.
_MAX_TERMS = 2**22


def _work_dtype(t: Tensor) -> torch.dtype:
    return torch.float32 if t.device.type == "mps" else torch.float64


def _poisson_logpmf(n: Tensor, mu: Tensor, log_mu: Tensor) -> Tensor:
    """log(e^{−μ} μⁿ/n!) for integers n ≥ 0, without cancellation.

    For n ≥ 1, with Stirling's formula and its remainder r(n),
    log Γ(n + 1) = (n + ½) log n − n + ½ log(2π) + r(n), so

        log p(n) = −μ·b(n/μ) − ½ log(2πn) − r(n),
        b(y) = y log y − y + 1 = (1 + t) log(1 + t) − t,   t = y − 1,

    where μ·b is evaluated through log1p near y = 1, and as
    n log(n/μ) − n + μ elsewhere. The large terms n log μ, log n! and μ,
    which nearly cancel at n ≈ μ, never appear.
    """
    t = n / mu - 1
    near = t.abs() < 0.5
    tn = torch.where(near, t, 0)
    dev = torch.where(
        near,
        mu * ((1 + tn) * torch.log1p(tn) - tn),
        torch.xlogy(n, n) - n * log_mu - n + mu,
    )
    pos = n > 0
    n1 = torch.where(pos, n, 1)
    out = -dev - 0.5 * torch.log(2 * math.pi * n1) - stirling_residual(n1)
    return torch.where(pos, out, -mu)


def _mode(x: Tensor, log_mu: Tensor, inv_var: Tensor) -> Tensor:
    """Mode n̂ ≥ 0 of h(n) = n log μ − log Γ(n + 1) − (x − n)²/2σ².

    h is concave, h′(n) = log μ − ψ(n + 1) + (x − n)/σ² is convex and
    decreasing, so Newton's method converges monotonically (after at
    most one step past the root).
    """
    mu = log_mu.exp()
    var = 1 / inv_var
    # Gaussian approximation N(μ, μ) · N(x, σ²) as the starting point
    n = (mu * (var + x) / (var + mu)).clamp(min=0)
    for _ in range(50):
        g = log_mu - torch.digamma(n + 1) + (x - n) * inv_var
        dg = -torch.polygamma(1, n + 1) - inv_var
        new = (n - g / dg).clamp(min=0)
        step = float((new - n).abs().max()) if new.numel() else 0.0
        n = new
        if step < 1e-3:
            break
    return n


def _window(
    x: Tensor, log_mu: Tensor, inv_var: Tensor, var: Tensor
) -> tuple[Tensor, Tensor]:
    """First term and number of terms of the summation window.

    Outside [n̂ − d_L, n̂ + d_R] every term is below e^{−T} times the term
    at the mode, T = 50. The curvature −h″(n) = ψ′(n + 1) + 1/σ² decreases
    with n, so on the left it is at least its value at n̂ and
    d_L = √(2T/(−h″(n̂))). On the right it is at least 1/σ² and at least
    1/(n + 1) (ψ′(z) > 1/z), so d_R is the smaller of σ√(2T) and the root
    of (a + d) log(1 + d/a) − d = T, a = n̂ + 1, found by Newton's method
    from above (every iterate is an upper bound).
    """
    n = _mode(x, log_mu, inv_var)
    curv = torch.polygamma(1, n + 1) + inv_var
    d_left = (2 * _TAIL / curv).sqrt()
    a = n + 1
    d = _TAIL + (_TAIL**2 + 2 * _TAIL * a).sqrt()
    for _ in range(6):
        phi = (a + d) * torch.log1p(d / a) - d - _TAIL
        d = d - phi / torch.log1p(d / a)
    d_right = torch.minimum(var.sqrt() * math.sqrt(2 * _TAIL), d)
    lo = (n - d_left).floor().sub(1).clamp(min=0)
    hi = (n + d_right).ceil().add(1)
    return lo, (hi - lo + 1)


def _series(
    data: Tensor,
    rate: Tensor,
    variance: Tensor,
    offset: Tensor | float,
    max_terms: int,
) -> tuple[Tensor, Tensor]:
    """log ∑ₙ e^{−μ} wₙ and ∑ₙ n wₙ / ∑ₙ wₙ, elementwise, for σ² > 0."""
    wd = _work_dtype(data)
    x = (data.to(wd) - torch.as_tensor(offset).to(data.device, wd)).reshape(-1)
    tiny = torch.finfo(wd).tiny
    mu = rate.to(wd).clamp(min=tiny).reshape(-1)
    log_mu = mu.log()
    var = variance.to(wd).reshape(-1)
    inv_var = 1 / var
    lo, size = _window(x, log_mu, inv_var, var)
    lse = torch.empty_like(x)
    mean = torch.empty_like(x)
    # buckets of window sizes in (2ᵏ⁻¹, 2ᵏ]: at most twice the terms
    bucket = size.log2().ceil().clamp(min=0)
    for k in torch.unique(bucket).tolist():
        idx = (bucket == k).nonzero().squeeze(1)
        width = int(2**k)
        terms = torch.arange(width, device=x.device, dtype=wd)
        step = max(1, max_terms // width)
        for start in range(0, idx.numel(), step):
            i = idx[start : start + step]
            n = lo[i, None] + terms
            log_w = _poisson_logpmf(n, mu[i, None], log_mu[i, None]) - (
                x[i, None] - n
            ).square() * (0.5 * inv_var[i, None])
            top = log_w.amax(dim=1, keepdim=True)
            w = (log_w - top).exp()
            total = w.sum(dim=1)
            lse[i] = top.squeeze(1) + total.log()
            mean[i] = (w * n).sum(dim=1) / total
    return lse.reshape(data.shape), mean.reshape(data.shape)


def _log_ratio(
    data: Tensor,
    rate: Tensor,
    variance: Tensor,
    offset: Tensor | float,
    max_terms: int,
) -> Tensor:
    """log s(μ, x − 1) − log s(μ, x), x = r − m, elementwise, for σ² > 0.

    Both series are summed over one window, that of s(μ, x) widened by
    one term on each side, n = l + k (k = 0, 1, …). Only ratios of
    terms enter, so the terms are taken relative to the first one,

        log(wₙ/w_l) = k log μ − ∑ⱼ₌₁ᵏ log(l + j) − k(k − 2(x − l))/2σ²,

    and the terms of s(μ, x − 1), μⁿ⁻¹/(n − 1)! · e^{−(x − n)²/2σ²},
    are wₙ·n/μ.
    """
    wd = _work_dtype(data)
    x = (data.to(wd) - torch.as_tensor(offset).to(data.device, wd)).reshape(-1)
    tiny = torch.finfo(wd).tiny
    mu = rate.to(wd).clamp(min=tiny).reshape(-1)
    log_mu = mu.log()
    var = variance.to(wd).reshape(-1)
    inv_var = 1 / var
    lo, size = _window(x, log_mu, inv_var, var)
    lo = (lo - 1).clamp(min=0)
    size = size + 2
    out = torch.empty_like(x)
    bucket = size.log2().ceil().clamp(min=0)
    for b in torch.unique(bucket).tolist():
        idx = (bucket == b).nonzero().squeeze(1)
        width = int(2**b)
        k = torch.arange(width, device=x.device, dtype=wd)
        step = max(1, max_terms // width)
        for start in range(0, idx.numel(), step):
            i = idx[start : start + step]
            n = lo[i, None] + k
            log_n = torch.log(n)  # -inf at n = 0
            # ∑ⱼ₌₁ᵏ log(l + j): the first term is left out
            fact = torch.cumsum(log_n[:, 1:], dim=1)
            fact = torch.cat([torch.zeros_like(fact[:, :1]), fact], dim=1)
            quad = (
                k
                * (k - 2 * (x[i, None] - lo[i, None]))
                * (0.5 * inv_var[i, None])
            )
            log_w = k * log_mu[i, None] - fact - quad
            log_v = log_w + (log_n - log_mu[i, None])
            out[i] = torch.logsumexp(log_v, dim=1) - torch.logsumexp(
                log_w, dim=1
            )
    return out.reshape(data.shape)


def _broadcast(
    data: Tensor, rate: Tensor | float, variance: Tensor | float
) -> tuple[Tensor, Tensor, Tensor]:
    data = torch.as_tensor(data)
    # Python scalars in the working dtype of the series (not rounded to
    # the default float32 before the float64 sums)
    wd = _work_dtype(data)

    def scalar_or_tensor(v: Tensor | float) -> Tensor:
        if isinstance(v, Tensor):
            return v.to(device=data.device)
        return torch.as_tensor(v, dtype=wd, device=data.device)

    rate = scalar_or_tensor(rate)
    variance = scalar_or_tensor(variance)
    data, rate, variance = torch.broadcast_tensors(data, rate, variance)
    if (variance < 0).any():
        raise ValueError("variance must be non-negative")
    if (rate < 0).any():
        raise ValueError("rate must be non-negative")
    return data, rate, variance


def poisson_normal_mean(
    data: Tensor,
    rate: Tensor | float,
    variance: Tensor | float,
    *,
    offset: Tensor | float = 0.0,
    max_terms: int = _MAX_TERMS,
) -> Tensor:
    """Conditional mean E[n | r] of Poisson counts under Gaussian noise.

    With r = n + g, n ~ Poisson(μ) and g ~ N(m, σ²) independent,

        E[n | r] = ∑ₙ n μⁿ/n! · exp(−(r − n − m)²/2σ²)
                 / ∑ₙ μⁿ/n! · exp(−(r − n − m)²/2σ²),

    the function F[r, μ, σ] of Snyder et al. (1993, Eq. 21).

    The sums are evaluated in log space (largest term factored out) and
    in float64, over the integers n ≥ 0 within a window around the mode
    of the summand, outside of which every term is below e⁻⁵⁰ times the
    largest one. The window adapts per element to the Poisson and
    Gaussian spreads; the series are evaluated in batches of at most
    `max_terms` terms. As σ → 0 the result tends to r − m, which is
    returned where σ² = 0.

    !!! references "References"

        1. Snyder DL, Hammoud AM, White RL. *Image recovery from data
           acquired with a charge-coupled-device camera.* J Opt Soc Am A
           **10**(5):1014-1023, 1993.
           [doi:10.1364/JOSAA.10.001014](https://doi.org/10.1364/JOSAA.10.001014)

        ??? quote "BibTeX"

            ```bibtex
            @article{snyder1993image,
              author  = {Snyder, Donald L. and Hammoud, Abed M. and
                         White, Richard L.},
              title   = {Image recovery from data acquired with a
                         charge-coupled-device camera},
              journal = {Journal of the Optical Society of America A},
              volume  = {10},
              number  = {5},
              pages   = {1014--1023},
              year    = {1993},
              doi     = {10.1364/JOSAA.10.001014}
            }
            ```

    Parameters
    ----------
    data : Tensor
        Observations r.
    rate : Tensor or float
        Poisson mean μ ≥ 0.
    variance : Tensor or float
        Gaussian variance σ² ≥ 0.
    offset : Tensor or float
        Gaussian mean m.
    max_terms : int
        Number of series terms evaluated at once (memory bound).

    Returns
    -------
    Tensor
        E[n | r], in the dtype of `data` (broadcast shape).
    """
    data, rate, variance = _broadcast(data, rate, variance)
    dtype = data.dtype if data.is_floating_point() else torch.float64
    limit = data - torch.as_tensor(offset).to(data)
    noisy = variance > 0
    if not bool(noisy.any()):
        return limit.to(dtype)
    var = torch.where(noisy, variance, 1)
    _, mean = _series(data, rate, var, offset, max_terms)
    return torch.where(noisy, mean.to(dtype), limit.to(dtype))


def poisson_normal_logpdf(
    data: Tensor,
    rate: Tensor | float,
    variance: Tensor | float,
    *,
    offset: Tensor | float = 0.0,
    max_terms: int = _MAX_TERMS,
) -> Tensor:
    """Log-density of r = n + g, n ~ Poisson(μ), g ~ N(m, σ²).

        log p(r) = −μ − ½ log(2πσ²)
                   + log ∑ₙ μⁿ/n! · exp(−(r − n − m)²/2σ²),

    the per-pixel term of the log-likelihood of Snyder et al. (1993,
    Eq. 10). The series is evaluated as in
    [`poisson_normal_mean`][optix.special.poisson_normal_mean], with the
    Poisson terms written through Stirling's remainder so that the large,
    nearly equal n log μ, log n! and μ never have to cancel. Where
    σ² = 0 the Poisson log-probability of r − m is returned,
    (r − m) log μ − μ − log Γ(r − m + 1).

    Parameters
    ----------
    data : Tensor
        Observations r.
    rate : Tensor or float
        Poisson mean μ ≥ 0.
    variance : Tensor or float
        Gaussian variance σ² ≥ 0.
    offset : Tensor or float
        Gaussian mean m.
    max_terms : int
        Number of series terms evaluated at once (memory bound).

    Returns
    -------
    Tensor
        log p(r), in the dtype of `data` (broadcast shape).
    """
    data, rate, variance = _broadcast(data, rate, variance)
    dtype = data.dtype if data.is_floating_point() else torch.float64
    wd = _work_dtype(data)
    noisy = variance > 0
    mu = rate.to(wd)
    k = data.to(wd) - torch.as_tensor(offset).to(data.device, wd)
    tiny = torch.finfo(wd).tiny
    poisson = torch.xlogy(k, mu.clamp(min=tiny)) - mu - torch.lgamma(k + 1)
    if not bool(noisy.any()):
        return poisson.to(dtype)
    var = torch.where(noisy, variance, 1).to(wd)
    lse, _ = _series(data, rate, var, offset, max_terms)
    lse = lse.to(wd)
    out = lse - 0.5 * torch.log(2 * math.pi * var)
    return torch.where(noisy, out, poisson).to(dtype)


def poisson_normal_ratio(
    data: Tensor,
    rate: Tensor | float,
    variance: Tensor | float,
    *,
    offset: Tensor | float = 0.0,
    max_terms: int = _MAX_TERMS,
) -> Tensor:
    """Ratio ξ(μ) = s(μ, x − 1)/s(μ, x) of Poisson–Gaussian series.

    With x = r − m and

        s(μ, b) = ∑ₙ μⁿ/n! · exp(−(b − n)²/2σ²),

    ξ is the derivative term of the Poisson–Gaussian negative
    log-likelihood: −∂/∂μ log p(r) = 1 − ξ(μ), with p(r) the density of
    [`poisson_normal_logpdf`][optix.special.poisson_normal_logpdf]
    (Chouzenoux et al. 2015, Eqs. 2.8, 2.10, 2.12). It equals
    E[n | r]/μ (see
    [`poisson_normal_mean`][optix.special.poisson_normal_mean]).

    Both series are summed in log space and in float64 over one window
    of n around the mode of the summand of s(μ, x), widened by one term
    on each side (every term left out is below e⁻⁵⁰ times the largest
    one); the terms of s(μ, x − 1) are indexed from n − 1 rather than
    formed as n·μⁿ⁻¹/n!, so that the ratio stays accurate for μ → 0.
    At μ = 0 the closed form ξ(0) = exp((2x − 1)/2σ²) is returned
    (Chouzenoux et al. 2015, Eq. 2.29); where σ² = 0, ξ = x/μ (Poisson).

    !!! references "References"

        1. Chouzenoux E, Jezierska A, Pesquet J-C, Talbot H. *A convex
           approach for image restoration with exact Poisson–Gaussian
           likelihood.* SIAM J Imaging Sci **8**(4):2662-2682, 2015.
           [doi:10.1137/15M1014395](https://doi.org/10.1137/15M1014395)

        ??? quote "BibTeX"

            ```bibtex
            @article{chouzenoux2015convex,
              author  = {Chouzenoux, Emilie and Jezierska, Anna and
                         Pesquet, Jean-Christophe and Talbot, Hugues},
              title   = {A Convex Approach for Image Restoration with
                         Exact {Poisson}--{Gaussian} Likelihood},
              journal = {SIAM Journal on Imaging Sciences},
              volume  = {8},
              number  = {4},
              pages   = {2662--2682},
              year    = {2015},
              doi     = {10.1137/15M1014395}
            }
            ```

    Parameters
    ----------
    data : Tensor
        Observations r.
    rate : Tensor or float
        Poisson mean μ ≥ 0.
    variance : Tensor or float
        Gaussian variance σ² ≥ 0.
    offset : Tensor or float
        Gaussian mean m.
    max_terms : int
        Number of series terms evaluated at once (memory bound).

    Returns
    -------
    Tensor
        ξ(μ), in the dtype of `data` (broadcast shape).
    """
    data, rate, variance = _broadcast(data, rate, variance)
    dtype = data.dtype if data.is_floating_point() else torch.float64
    wd = _work_dtype(data)
    x = data.to(wd) - torch.as_tensor(offset).to(data.device, wd)
    mu = rate.to(wd)
    noisy = variance > 0
    poisson = x / mu
    if not bool(noisy.any()):
        return poisson.to(dtype)
    var = torch.where(noisy, variance, 1).to(wd)
    zero = mu == 0
    log_ratio = _log_ratio(data, rate, var, offset, max_terms).to(wd)
    log_ratio = torch.where(zero, (2 * x - 1) / (2 * var), log_ratio)
    return torch.where(noisy, log_ratio.exp(), poisson).to(dtype)
