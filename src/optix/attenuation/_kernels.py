"""Depth-resolved attenuation kernels (Vermeer 2014, Smith 2015)."""

import torch
from torch import Tensor

__all__ = [
    "compensation_gain",
    "smith_attenuation",
    "tail_sums",
    "vermeer_attenuation",
    "vermeer_gain",
    "weighted_tv",
]


def _check(pixel_size: float, passes: int) -> None:
    if not pixel_size > 0:
        raise ValueError(f"pixel_size must be > 0, got {pixel_size!r}")
    if isinstance(passes, bool) or passes not in (1, 2):
        raise ValueError(f"passes must be 1 or 2, got {passes!r}")


def tail_sums(
    data: Tensor, dim: int = -1, *, inclusive: bool = True
) -> Tensor:
    """Sums of the deeper intensities along `dim`, in float64.

        T[i] = ∑_{k ≥ i} I[k]   (inclusive),   ∑_{k > i} I[k]   (exclusive).

    Accumulated in float64 (float32 on MPS) whatever the input dtype, so
    that the long sums of the deep, small intensities keep their precision.

    Parameters
    ----------
    data : Tensor
        Intensities I.
    dim : int
        Propagation axis.
    inclusive : bool
        Whether T[i] includes I[i].

    Returns
    -------
    Tensor
        Shape of `data`, float64 (float32 on MPS).
    """
    acc = torch.float32 if data.device.type == "mps" else torch.float64
    x = data.to(acc)
    t = x.flip(dim).cumsum(dim).flip(dim)
    return t if inclusive else t - x


def vermeer_attenuation(
    data: Tensor, dim: int = -1, *, pixel_size: float = 1.0, passes: int = 2
) -> Tensor:
    """Per-pixel attenuation coefficient of Vermeer et al. (2014), Eq. (17).

        μ[i] = 1/(p·Δ) · log(1 + I[i] / ∑_{k > i} I[k]),

    with p = 2 passes in the paper (Eq. (8)) and Δ the pixel size. The sum
    runs to the last pixel of the profile (the limited depth range of
    Eq. (13)). For the last pixel the sum is empty and μ = ∞; where both
    I[i] and the sum vanish, μ is undefined and returned as NaN.

    Parameters
    ----------
    data : Tensor
        Intensities I ≥ 0 along `dim`.
    dim : int
        Propagation axis.
    pixel_size : float
        Δ, in the length unit of μ⁻¹.
    passes : {1, 2}
        p.

    Returns
    -------
    Tensor
        μ, of the shape and dtype of `data`.
    """
    _check(pixel_size, passes)
    deeper = tail_sums(data, dim, inclusive=False)
    x = data.to(deeper.dtype)
    mu = torch.log1p(x / deeper) / (passes * pixel_size)
    mu = torch.where((deeper == 0) & (x > 0), torch.inf, mu)
    return mu.to(data.dtype)


def vermeer_gain(data: Tensor, dim: int = -1) -> Tensor:
    """Compensation gain exp(p·Δ·∑_{j < i} μ[j]) for Vermeer's μ.

    Eq. (17) gives exp(p·Δ·μ[j]) = T[j] / T[j + 1], with the inclusive
    tail sums T[i] = ∑_{k ≥ i} I[k], so the product telescopes to

        G[i] = T[0] / T[i],

    which is computed directly (no exponential, no cumulated logarithm).
    G does not depend on p nor on Δ. Where T[i] ≤ 0 (no signal from i
    on), G = 0.

    Parameters
    ----------
    data : Tensor
        Intensities I along `dim`.
    dim : int
        Propagation axis.

    Returns
    -------
    Tensor
        G, in the accumulation dtype of
        [`tail_sums`][optix.attenuation.utils.tail_sums].
    """
    t = tail_sums(data, dim, inclusive=True)
    first = t.narrow(dim, 0, 1)
    gain = first / t
    return torch.where(t > 0, gain, torch.zeros_like(gain))


def compensation_gain(
    mu: Tensor, dim: int = -1, *, pixel_size: float = 1.0, passes: int = 2
) -> Tensor:
    """Compensation gain G[i] = exp(p·Δ·∑_{j < i} μ[j]) from a μ profile.

    The attenuation of the model, I[i] ∝ μ[i]·exp(−p·∫₀^{z_i} μ), up to
    the front of pixel i. The cumulative sum is taken in float64 (float32
    on MPS).

    Parameters
    ----------
    mu : Tensor
        Finite attenuation coefficients along `dim`.
    dim : int
    pixel_size : float
    passes : {1, 2}

    Returns
    -------
    Tensor
        G, in the accumulation dtype.
    """
    _check(pixel_size, passes)
    acc = torch.float32 if mu.device.type == "mps" else torch.float64
    m = mu.to(acc)
    before = m.cumsum(dim) - m
    return torch.exp(passes * pixel_size * before)


def smith_attenuation(
    data: Tensor,
    dim: int = -1,
    *,
    pixel_size: float = 1.0,
    passes: int = 2,
    noise_power: float = 0.0,
) -> Tensor:
    """Attenuation estimate of Smith et al. (2015), Eqs. (8)–(10).

        Î[m] = I[m] · I[m]² / (I[m]² + |c|²),                 (8)–(9)
        μ̂[m] = Î[m] / (p·Δ·∑_{r ≥ m} Î[r]),                   (10)

    with the confocal and fall-off functions of Eq. (9) set to 1, and
    |c|² the noise power. The sum includes pixel m, so the last pixel
    gets μ̂ = 1/(p·Δ). Where the sum is 0, μ̂ = 0.

    Parameters
    ----------
    data : Tensor
        Intensities along `dim`.
    dim : int
    pixel_size : float
        Δz.
    passes : {1, 2}
        p (2 in the paper).
    noise_power : float
        |c|² ≥ 0 of Eq. (9); 0 disables the restoration filter.

    Returns
    -------
    Tensor
        μ̂, of the shape and dtype of `data`.
    """
    _check(pixel_size, passes)
    if not noise_power >= 0:
        raise ValueError(f"noise_power must be >= 0, got {noise_power!r}")
    acc = torch.float32 if data.device.type == "mps" else torch.float64
    x = data.to(acc)
    if noise_power > 0:
        sq = x.square()
        x = x * sq / (sq + noise_power)
    t = tail_sums(x, dim, inclusive=True)
    mu = x / (passes * pixel_size * t)
    mu = torch.where(t != 0, mu, torch.zeros_like(mu))
    return mu.to(data.dtype)


def _d(x: Tensor, dim: int) -> Tensor:
    """Forward differences x[n + 1] − x[n] along `dim` (N − 1 values)."""
    return torch.diff(x, dim=dim)


def _dt(p: Tensor, dim: int) -> Tensor:
    """Adjoint of `_d`: (Dᵀp)[n] = p[n − 1] − p[n], p[−1] = p[N − 1] = 0."""
    zero = torch.zeros_like(p.narrow(dim, 0, 1))
    padded = torch.cat([zero, p, zero], dim)
    return -torch.diff(padded, dim=dim)


def weighted_tv(
    data: Tensor,
    weights: Tensor,
    lam: float,
    dim: int = -1,
    *,
    nonneg: bool = True,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> tuple[Tensor, int, float]:
    """Weighted 1-D total-variation denoising along `dim`.

    Solves, independently for every line along `dim`,

        minimise ½‖x − y‖² + λ ∑_n w[n]·|x[n + 1] − x[n]|   s.t. x ≥ 0,

    the problem (11) of Smith et al. (2015) with the weighted TV of
    Eqs. (13)–(15). The weights `w` hold N − 1 values along `dim` (or N,
    the last one being ignored). The dual problem, a box-constrained
    quadratic in the N − 1 dual variables, is solved by accelerated
    projected gradient (FISTA) with step 1/4 (‖D‖² ≤ 4); the primal
    solution is x = max(y − Dᵀp, 0). The iterations stop when the
    duality gap, summed over all lines, falls below `tol` times the
    primal objective.

    Parameters
    ----------
    data : Tensor
        y.
    weights : Tensor
        w ≥ 0, broadcastable to `data` (N or N − 1 along `dim`).
    lam : float
        λ ≥ 0.
    dim : int
    nonneg : bool
        Impose x ≥ 0.
    max_iter : int
    tol : float

    Returns
    -------
    x : Tensor
        Shape of `data`.
    iterations : int
    gap : float
        Relative duality gap reached.
    """
    if not lam >= 0:
        raise ValueError(f"lam must be >= 0, got {lam!r}")
    dim = dim % data.ndim
    n = data.shape[dim]
    y = data
    if n < 2 or lam == 0:
        x = y.clamp_min(0) if nonneg else y.clone()
        return x, 0, 0.0
    shape = _d(y, dim).shape
    w = torch.as_tensor(weights, dtype=y.dtype, device=y.device)
    try:
        w = torch.broadcast_to(w, y.shape).narrow(dim, 0, n - 1)
    except RuntimeError:
        w = torch.broadcast_to(w, shape)
    if bool((w < 0).any()):
        raise ValueError("weights must be >= 0")
    bound = lam * w

    def primal(p: Tensor) -> Tensor:
        x = y - _dt(p, dim)
        return x.clamp_min(0) if nonneg else x

    p = torch.zeros_like(bound)
    q = p
    t = 1.0
    gap = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        x = primal(q)
        p_new = torch.clamp(q + _d(x, dim) / 4, -bound, bound)
        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
        q = p_new + ((t - 1) / t_new) * (p_new - p)
        p, t = p_new, t_new
        if it % 10 == 0 or it == max_iter:
            x = primal(p)
            dx = _d(x, dim)
            fit = 0.5 * (x - y).square().sum()
            obj = fit + (bound * dx.abs()).sum()
            dual = fit + (p * dx).sum()
            gap = float((obj - dual) / obj.abs().clamp_min(1e-300))
            if gap <= tol:
                break
    return primal(p), it, gap
