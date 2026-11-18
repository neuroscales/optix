"""Local estimation of Foi et al. (2008): wavelets, segmentation, pairs.

Implements Section III-A to III-C (non-clipped) and V-A / VI-A (clipped
and robust variants: the estimators are the same, only their model
changes) of Foi et al. (2008), see [`PoissonGaussian`][].

Every image here is in the normalised units of the paper (data range
[0, 1], Section IV-A) and batched as `(n, H, W)`.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..special import stirling_residual

__all__ = [
    "daubechies3",
    "edge_kernels",
    "kappa",
    "kappa_mad",
    "kappa_mad_variance",
    "d_factor",
    "d_mad_factor",
    "level_labels",
    "local_pairs",
    "smoothness_mask",
    "wavelet_coefficients",
]

#: Φ⁻¹(3/4), the large-sample MAD factor (Section VI-A).
MAD_LIMIT = 0.6744897501960817

#: Table II: κᵐᵃᵈₙ for n = 1..20 (κ₂ₙ = κ₂ₙ₋₁).
TABLE_II = (
    0.798, 0.798, 0.732, 0.732, 0.712, 0.712, 0.702, 0.702, 0.696, 0.696,
    0.693, 0.693, 0.690, 0.690, 0.688, 0.688, 0.686, 0.686, 0.685, 0.685,
)  # fmt: skip


def daubechies3(
    dtype: torch.dtype = torch.float64, device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    """1-D Daubechies (3 vanishing moments) wavelet and scaling filters.

    Section III-A, Eq. (10): ψ₁ (zero sum, unit l2-norm) and
    φ₁ (unit sum). The paper prints them to three decimals; these
    are the exact values, from the closed form of the length-6
    Daubechies filter, and they round to Eq. (10).

    Returns
    -------
    psi1, phi1 : Tensor
        Length-6 filters, in the order of Eq. (10).
    """
    r10 = math.sqrt(10)
    s = math.sqrt(5 + 2 * r10)
    # orthonormal low-pass filter h (sum sqrt(2), unit norm)
    h = [
        1 + r10 + s,
        5 + r10 + 3 * s,
        10 - 2 * r10 + 2 * s,
        10 - 2 * r10 - 2 * s,
        5 + r10 - 3 * s,
        1 + r10 - s,
    ]
    h = [v / (16 * math.sqrt(2)) for v in h]
    phi1 = [v / math.sqrt(2) for v in reversed(h)]  # sum 1
    psi1 = [h[5], -h[4], h[3], -h[2], h[1], -h[0]]  # quadrature mirror
    kw = {"dtype": dtype, "device": device}
    return torch.tensor(psi1, **kw), torch.tensor(phi1, **kw)


def _separable(x: Tensor, k1: Tensor) -> Tensor:
    """Valid 2-D convolution of `(n, H, W)` with k1·k1ᵀ."""
    k = k1.flip(0).to(x)  # convolution, not correlation
    x = x.unsqueeze(1)
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return x.squeeze(1)


def wavelet_coefficients(z: Tensor) -> tuple[Tensor, Tensor]:
    """Approximation and detail coefficients (Section III-A).

    zʷᵃᵖᵖ = ↓₂(z ⊛ φ) and zʷᵈᵉᵗ = ↓₂(z ⊛ ψ) with the
    separable kernels φ = φ₁φ₁ᵀ and ψ = ψ₁ψ₁ᵀ
    of Eq. (10). The convolution is restricted to the inside of the
    image ("valid"); the boundary handling is not specified in the paper.

    Parameters
    ----------
    z : (n, H, W) Tensor

    Returns
    -------
    wapp, wdet : (n, h, w) Tensor
        h = ⌈(H − 5)/2⌉, w = ⌈(W − 5)/2⌉.
    """
    psi1, phi1 = daubechies3(device=z.device)
    wapp = _separable(z, phi1)[:, ::2, ::2]
    wdet = _separable(z, psi1)[:, ::2, ::2]
    return wapp, wdet


def edge_kernels(
    dtype: torch.dtype = torch.float64, device: torch.device | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """9x9 Laplacian and gradient kernels of Eq. (12).

    The paper specifies the size (9x9) but not the kernels. Not specified
    in the paper: these are the derivatives of a Gaussian of standard
    deviation 4/3 (the 9x9 support covers ±3 standard deviations),
    sampled on the grid; the Laplacian is shifted to have zero sum.

    Returns
    -------
    laplacian, d_row, d_col : (9, 9) Tensor
    """
    s = 4 / 3
    t = torch.arange(-4, 5, dtype=torch.float64)
    r, c = torch.meshgrid(t, t, indexing="ij")
    g = torch.exp(-(r * r + c * c) / (2 * s * s)) / (2 * math.pi * s * s)
    lap = g * ((r * r + c * c) / s**4 - 2 / s**2)
    lap = lap - lap.mean()
    d_row, d_col = -r / s**2 * g, -c / s**2 * g
    kw = {"dtype": dtype, "device": device}
    return lap.to(**kw), d_row.to(**kw), d_col.to(**kw)


def _conv_same(x: Tensor, k: Tensor) -> Tensor:
    """Same-size 2-D convolution of `(n, h, w)` (replicated borders)."""
    p = k.shape[-1] // 2
    x = F.pad(x.unsqueeze(1), (p, p, p, p), mode="replicate")
    return F.conv2d(x, k.flip(-1, -2).to(x)[None, None]).squeeze(1)


def _median3(x: Tensor) -> Tensor:
    """3x3 median filter of `(n, h, w)` (replicated borders)."""
    p = F.pad(x.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    patches = p.unfold(1, 3, 1).unfold(2, 3, 1)
    return patches.reshape(*patches.shape[:3], 9).median(-1).values


def smoothness_mask(
    wapp: Tensor, wdet: Tensor, tau: float
) -> tuple[Tensor, Tensor]:
    """Smoothed image and set of smoothness (Section III-B.1-2).

    zˢᵐᵒ = zʷᵃᵖᵖ ⊛ ϖ (Eq. 11) with ϖ the uniform 7x7
    kernel; s = √(π/2)·|zʷᵈᵉᵗ| ⊛ ϖ; and
    Xˢᵐᵒ = {|∇Λ| + |Λ| < τ·s} with
    Λ = ∇²(medfilt_3x3(zʷᵃᵖᵖ)) (Eq. 12). Borders are
    replicated (not specified in the paper).

    Parameters
    ----------
    wapp, wdet : (n, h, w) Tensor
    tau : float
        Threshold of Eq. (12).

    Returns
    -------
    zsmo : (n, h, w) Tensor
    mask : (n, h, w) bool Tensor
        True on Xˢᵐᵒ.
    """
    box = torch.full((7, 7), 1 / 49, dtype=wapp.dtype, device=wapp.device)
    zsmo = _conv_same(wapp, box)
    s = math.sqrt(math.pi / 2) * _conv_same(wdet.abs(), box)
    lap, d_row, d_col = edge_kernels(wapp.dtype, wapp.device)
    lam = _conv_same(_median3(wapp), lap)
    grad = torch.hypot(_conv_same(lam, d_row), _conv_same(lam, d_col))
    mask = grad + lam.abs() < tau * s
    return zsmo, mask


def level_labels(zsmo: Tensor, mask: Tensor, delta: float) -> Tensor:
    """Level-set index of each coefficient (Section III-B.3, Eq. 13).

    Sⱼ = {x ∈ Xˢᵐᵒ : zˢᵐᵒ(x) ∈ [uⱼ − Δ/2, uⱼ + Δ/2)}
    with uⱼ = Δ·j, j = 1..⌈1/Δ⌉.

    Returns
    -------
    Tensor
        int64, `j − 1` in [0, ⌈1/Δ⌉); -1 outside every set
        (and off Xˢᵐᵒ).
    """
    nbar = math.ceil(1 / delta - 1e-9)
    j = torch.floor(zsmo / delta + 0.5).long()
    ok = mask & (j >= 1) & (j <= nbar) & zsmo.isfinite()
    return torch.where(ok, j - 1, -1)


def nlevels(delta: float) -> int:
    """Number of level sets ⌈1/Δ⌉."""
    return math.ceil(1 / delta - 1e-9)


def kappa(n: Tensor) -> Tensor:
    """Unbiasing factor of the sample standard deviation, Eq. (17).

    κₙ = √(2/(n-1)) Γ(n/2) / Γ((n-1)/2), computed as
    log κ = x log1p(1/(2x)) − 1/2 + r(x + 1/2) − r(x) with
    `x = (n-1)/2` and `r` the Stirling residual, which avoids the
    cancellation of the two `lgamma` for large `n`.
    """
    return torch.exp(_log_kappa(n))


def _log_kappa(n: Tensor) -> Tensor:
    n = torch.as_tensor(n, dtype=torch.float64)
    x = (n - 1) / 2
    h = 1 / (n - 1)
    # x log1p(h) - 1/2 = (log1p(h) - h) / (2h); series for small h
    small = h < 1e-2
    hs = torch.where(small, h, 1e-2)
    series = -hs / 2 + hs**2 / 3 - hs**3 / 4 + hs**4 / 5 - hs**5 / 6
    hb = torch.where(small, 1.0, h)
    direct = (torch.log1p(hb) - hb) / (2 * hb)
    lead = torch.where(small, series / 2, direct)
    return lead + stirling_residual(x + 0.5) - stirling_residual(x)


def d_factor(n: Tensor) -> Tensor:
    """dₙ = (1 − κₙ²) / κₙ², Eq. (19)."""
    return torch.expm1(-2 * _log_kappa(n))


def kappa_mad(n: Tensor) -> Tensor:
    """Bias factor of the MAD (Section VI-A, Table II).

    Table II for `n ≤ 20`, else 1/(5n) + Φ⁻¹(3/4).
    """
    n = torch.as_tensor(n, dtype=torch.float64)
    table = torch.tensor(TABLE_II, dtype=torch.float64, device=n.device)
    idx = (n.long() - 1).clamp(0, len(TABLE_II) - 1)
    return torch.where(n <= 20, table[idx], 1 / (5 * n) + MAD_LIMIT)


def kappa_mad_variance(n: Tensor) -> Tensor:
    """(κ̌ᵐᵃᵈₙ)² ~ 1 + 1/(5n) (Section VI-A.2)."""
    n = torch.as_tensor(n, dtype=torch.float64)
    return 1 + 1 / (5 * n)


def d_mad_factor(n: Tensor) -> Tensor:
    """dᵐᵃᵈₙ ~ 1.35 / (n + 1.5), Eq. (58)."""
    n = torch.as_tensor(n, dtype=torch.float64)
    return 1.35 / (n + 1.5)


def local_pairs(
    labels: Tensor, wapp: Tensor, wdet: Tensor, segments: int, robust: bool
) -> dict[str, Tensor]:
    """Expectation / standard-deviation pairs of the level sets.

    Section III-C: ŷᵢ is the mean of zʷᵃᵖᵖ on `Sᵢ`
    (Eq. 15) and σ̂ᵢ the sample standard deviation of
    zʷᵈᵉᵗ divided by κₙᵢ (Eq. 16). With `robust`,
    σ̂ᵢ = median |zʷᵈᵉᵗ| / κᵐᵃᵈₙᵢ (Eq. 57). Only
    level sets with at least two samples are kept (Section III-B.3).

    Parameters
    ----------
    labels : Tensor
        Level-set index (-1: none), same shape as `wapp` and `wdet`.
    segments : int
        Number of level sets.

    Returns
    -------
    dict
        float64 `count`, `mean` (ŷ), `std` (sample standard
        deviation, `ddof=1`, before division by κ) and, with
        `robust`, `mad` (median |zʷᵈᵉᵗ|, before division by
        κᵐᵃᵈ).
    """
    keep = labels >= 0
    lab = labels[keep]
    v = wapp[keep].to(torch.float64)
    e = wdet[keep].to(torch.float64)
    count = torch.zeros(segments, dtype=torch.float64, device=lab.device)
    count.index_add_(0, lab, torch.ones_like(v))
    safe = count.clamp_min(1)
    mean = torch.zeros_like(count).index_add_(0, lab, v) / safe
    emean = torch.zeros_like(count).index_add_(0, lab, e) / safe
    ss = torch.zeros_like(count).index_add_(0, lab, (e - emean[lab]) ** 2)
    std = (ss / (count - 1).clamp_min(1)).sqrt()
    ok = count >= 2
    out = {"count": count[ok], "mean": mean[ok], "std": std[ok]}
    if robust:
        out["mad"] = _segment_median(lab, e.abs(), count, segments)[ok]
    return out


def _segment_median(
    lab: Tensor, x: Tensor, count: Tensor, segments: int
) -> Tensor:
    """Median of `x` within each label (mean of the two middle values)."""
    order = torch.argsort(x, stable=True)
    order = order[torch.argsort(lab[order], stable=True)]
    xs = x[order]
    n = count.long()
    start = torch.cumsum(n, 0) - n
    lo = start + (n - 1).clamp_min(0) // 2
    hi = start + n.clamp_min(1) // 2
    if xs.numel() == 0:
        return torch.full_like(count, torch.nan)
    size = xs.numel() - 1
    med = (xs[lo.clamp(0, size)] + xs[hi.clamp(0, size)]) / 2
    return torch.where(n > 0, med, torch.nan)
