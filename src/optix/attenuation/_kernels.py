"""Depth-resolved attenuation kernels (Vermeer 2014, Smith 2015)."""

import math

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
    on) or T[0] ≤ 0 (no signal in the profile, e.g. after a background
    subtraction), G = 0.

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
    return torch.where((t > 0) & (first > 0), gain, torch.zeros_like(gain))


def compensation_gain(
    mu: Tensor, dim: int = -1, *, pixel_size: float = 1.0, passes: int = 2
) -> Tensor:
    """Compensation gain G[i] = exp(p·Δ·∑_{j < i} μ[j]) from a μ profile.

    The attenuation of the model, I[i] ∝ μ[i]·exp(−p·∫₀^{z_i} μ), up to
    the front of pixel i. The cumulative sum is taken in float64 (float32
    on MPS), and the exponent is capped at log of the largest finite value
    of that dtype, so that G stays finite.

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
    top = math.log(torch.finfo(acc).max)
    return torch.exp((passes * pixel_size * before).clamp_max(top))


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
    gets μ̂ = 1/(p·Δ). Where the sum is ≤ 0 (no signal from m on, or
    noise of negative sum after a background subtraction), μ̂ = 0.

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
    x = _restored(data, noise_power)
    t = tail_sums(x, dim, inclusive=True)
    mu = x / (passes * pixel_size * t)
    mu = torch.where(t > 0, mu, torch.zeros_like(mu))
    return mu.to(data.dtype)


def _restored(data: Tensor, noise_power: float = 0.0) -> Tensor:
    """Restoration filter of Smith et al. (2015), Eqs. (8)–(9).

        Î[m] = I[m] · I[m]² / (I[m]² + |c|²),

    with the confocal and fall-off functions set to 1; `noise_power=0` is
    the identity.

    Parameters
    ----------
    data : Tensor
        I.
    noise_power : float
        |c|² ≥ 0.

    Returns
    -------
    Tensor
        Î, in float64 (float32 on MPS).
    """
    acc = torch.float32 if data.device.type == "mps" else torch.float64
    x = data.to(acc)
    if noise_power > 0:
        sq = x.square()
        x = x * sq / (sq + noise_power)
    return x


def _scan(
    knots: Tensor,
    end: Tensor,
    stop: Tensor,
    a: Tensor,
    b: Tensor,
    level: Tensor,
    step: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Remove the knots of one end that lie beyond the crossing of `level`.

    `knots` holds, for every line (last dim), the knots of a continuous,
    increasing, piecewise-linear derivative: position, and the increments
    of its slope and offset across the knot. `end` indexes the end knot
    of each line (the first one with `step=1`, the last one with
    `step=-1`), `stop` the other end, and a·x + b is the derivative on the
    outer segment. Knots are removed while the derivative at the end knot
    is ≤ `level` (`step=1`) or ≥ `level` (`step=-1`). Returns the point
    where the derivative equals `level`, the coefficients of the segment
    holding it, and the new end.
    """
    last = knots.shape[1] - 1
    while True:
        index = end.clamp(0, last).view(1, 1, -1).expand(3, 1, -1)
        pos, da, db = torch.gather(knots, 1, index)[:, 0]
        d = a * pos + b
        if step > 0:
            pop = (end <= stop) & (d <= level)
        else:
            pop = (end >= stop) & (d >= level)
        if not bool(pop.any()):
            break
        a = torch.where(pop, a + step * da, a)
        b = torch.where(pop, b + step * db, b)
        end = end + step * pop.long()
    return (level - b) / a, a, b, end


def _tv_lines(y: Tensor, lam: Tensor) -> Tensor:
    """Exact weighted 1-D TV denoising of the columns of `y` (N, L).

    argmin_x ½‖x − y‖² + ∑_n lam[n]·|x[n + 1] − x[n]|, for every column,
    by dynamic programming (Johnson 2013): the derivative of the partial
    objective F_n(x) = min over x[0..n−1] (x[n] = x) is increasing and
    piecewise linear. Minimising out x[n] in F_n(x[n]) + lam[n]·|x − x[n]|
    clamps that derivative to [−lam[n], lam[n]]; the points b⁻[n] ≤ b⁺[n]
    where it reaches ∓lam[n] are stored, and x[n] = clamp(x[n + 1], b⁻[n],
    b⁺[n]) in the backward pass. Each step adds two knots, so the forward
    pass removes at most 2N knots per column: O(N) work per column.
    """
    n, lines = y.shape
    kw = {"dtype": y.dtype, "device": y.device}
    knots = torch.zeros(3, 2 * n + 2, lines, **kw)
    head = torch.full((lines,), n + 1, dtype=torch.long, device=y.device)
    tail = head - 1
    lower = torch.empty(n - 1, lines, **kw)
    upper = torch.empty_like(lower)
    one = torch.ones(lines, **kw)
    left = right = -y[0]
    for k in range(n - 1):
        w = lam[k]
        xl, a1, b1, head = _scan(knots, head, tail, one, left, -w, 1)
        xr, a2, b2, tail = _scan(knots, tail, head, one, right, w, -1)
        lower[k], upper[k] = xl, xr
        head, tail = head - 1, tail + 1
        new = torch.stack([xl, a1, b1 + w])[:, None]
        knots.scatter_(1, head.view(1, 1, -1).expand(3, 1, -1), new)
        new = torch.stack([xr, -a2, w - b2])[:, None]
        knots.scatter_(1, tail.view(1, 1, -1).expand(3, 1, -1), new)
        left, right = -w - y[k + 1], w - y[k + 1]
    x = torch.empty_like(y)
    x[-1] = _scan(knots, head, tail, one, left, torch.zeros_like(one), 1)[0]
    for k in range(n - 2, -1, -1):
        x[k] = torch.minimum(torch.maximum(x[k + 1], lower[k]), upper[k])
    return x


def weighted_tv(
    data: Tensor,
    weights: Tensor,
    lam: float,
    dim: int = -1,
    *,
    nonneg: bool = True,
) -> Tensor:
    """Weighted 1-D total-variation denoising along `dim`.

    Solves exactly, independently for every line along `dim`,

        minimise ½‖x − y‖² + λ ∑_n w[n]·|x[n + 1] − x[n]|   s.t. x ≥ 0,

    the problem (11) of Smith et al. (2015) with the weighted TV of
    Eqs. (13)–(15). The weights `w` hold N − 1 values along `dim` (or N,
    the last one being ignored). The unconstrained problem is solved by
    the direct dynamic-programming algorithm of Johnson (2013), which
    handles a weight per difference, in O(N) per line and vectorised
    across lines; the constraint is then imposed by clamping, since in 1-D
    prox(TV + ι_{x ≥ 0})(y) = max(prox_TV(y), 0). Smith et al. use CVX.
    Computed in float64 (float32 on MPS).

    !!! references "References"

        1. Johnson NA. *A dynamic programming algorithm for the fused
           lasso and L0-segmentation.* J Comput Graph Stat
           **22**(2):246–260, 2013.
           [doi:10.1080/10618600.2012.681238](https://doi.org/10.1080/10618600.2012.681238)

        ??? quote "BibTeX"

            ```bibtex
            @article{johnson2013dynamic,
              author  = {Johnson, Nicholas A.},
              title   = {A dynamic programming algorithm for the fused
                         lasso and {L0}-segmentation},
              journal = {Journal of Computational and Graphical Statistics},
              volume  = {22}, number = {2}, pages = {246--260},
              year    = {2013}, doi = {10.1080/10618600.2012.681238}
            }
            ```

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

    Returns
    -------
    Tensor
        x, of the shape and dtype of `data`.
    """
    if not lam >= 0:
        raise ValueError(f"lam must be >= 0, got {lam!r}")
    dim = dim % data.ndim
    n = data.shape[dim]
    if n < 2 or lam == 0:
        return data.clamp_min(0) if nonneg else data.clone()
    acc = torch.float32 if data.device.type == "mps" else torch.float64
    y = data.to(acc)
    shape = list(y.shape)
    shape[dim] = n - 1
    w = torch.as_tensor(weights, dtype=acc, device=y.device)
    try:
        w = torch.broadcast_to(w, y.shape).narrow(dim, 0, n - 1)
    except RuntimeError:
        w = torch.broadcast_to(w, shape)
    if bool((w < 0).any()):
        raise ValueError("weights must be >= 0")
    # lines as columns: (N, lines)
    lines = y.movedim(dim, 0).reshape(n, -1)
    bound = (lam * w).movedim(dim, 0).reshape(n - 1, -1)
    x = _tv_lines(lines, bound)
    x = x.reshape(y.movedim(dim, 0).shape).movedim(0, dim)
    if nonneg:
        x = x.clamp_min(0)
    return x.to(data.dtype)
