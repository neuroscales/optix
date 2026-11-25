"""NCS, the sCMOS noise-correction algorithm (Liu et al. 2017).

Written from the paper and its Supplementary Notes; the authors' code
(GPL-3.0) was not used.
"""

import math
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple, Self

import torch
from torch import Tensor

from .._config import resolve_backend
from .._estimator import Estimator, Tags, _shares_memory
from ..estimates import SensorEstimate
from ..io import (
    ArrayLike,
    Labeled,
    Layout,
    WritableArrayLike,
    blocks,
    chunks_of,
    flush,
    prepare_out,
)
from ..optim import ShiftedPoisson
from ._bm3d import _select

__all__ = [
    "NCS",
    "NcsResult",
    "ncs",
    "ncs_gradient",
    "ncs_halo",
    "ncs_mask",
    "ncs_objective",
    "ncs_precorrect",
    "ncs_solve",
    "raised_cosine",
    "segment_bounds",
]

MaskKind = Literal["otf-weighted", "noise-only"]
Units = Literal["electrons", "adu"]

#: Value given to non-positive pre-corrected pixels (Supplementary
#: Notes 2 and 6).
_FLOOR = 1e-6
#: Armijo constant of the line search (not in the paper).
_ARMIJO = 1e-4


# -- checks ------------------------------------------------------------------
def _positive(name: str, value: object) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a number > 0, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return v


def _nonnegative(name: str, value: object) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a number ≥ 0, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be finite and ≥ 0, got {value!r}")
    return v


def _count(name: str, value: object) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be an integer ≥ 1, got {value!r}")
    return int(value)


def _check_mask(mask: object) -> None:
    if not (mask in ("otf-weighted", "noise-only") or callable(mask)):
        raise ValueError(
            "mask must be 'otf-weighted', 'noise-only' or a callable of "
            f"the radial frequency, got {mask!r}"
        )


def _check_units(units: object) -> None:
    if units not in ("electrons", "adu"):
        raise ValueError(f"units must be 'electrons' or 'adu', got {units!r}")


# -- OTF mask (Supplementary Notes 4, 5 and 9) -------------------------------
def raised_cosine(k: Tensor, beta: float, period: float) -> Tensor:
    """High-pass raised-cosine filter of the OTF mask (Supp. Note 5).

    With k_lo = (1 − β)/(2T) and k_hi = (1 + β)/(2T),

        M(k) = 0                                  for k < k_lo,
        M(k) = ½ − ½ cos[(πT/β)(k − k_lo)]        for k_lo ≤ k ≤ k_hi,
        M(k) = 1                                  for k > k_hi.

    !!! note "Sign of the cosine"
        Eq. (4) of the Supplementary Notes prints ½ + ½ cos[…] in the
        transition band, which jumps from 1 to 0 at k_lo and from 0 to 1
        at k_hi. The high-pass filter the text describes, continuous at
        both ends, is ½ − ½ cos[…]; it also gives the curves of Fig. SS4B
        (the default mask is about 0.28 at the OTF radius 2NA/λ and
        about 0.7 at 3.5NA/λ) and the mask height of about 0.2 at a
        sampling rate of 1.4 (Supplementary Note 11).

    Parameters
    ----------
    k : Tensor
        Radial frequency k_r ≥ 0 (cycles per unit length).
    beta : float
        Roll-off β ∈ [0, 1] (β = 0: a step at 1/(2T)).
    period : float
        T > 0, in the unit of length of `k`.

    Returns
    -------
    Tensor
        M(k), of the shape and dtype of `k`.
    """
    beta = float(beta)
    if not 0 <= beta <= 1:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    period = _positive("period", period)
    lo = (1 - beta) / (2 * period)
    hi = (1 + beta) / (2 * period)
    if beta == 0:
        return (k > lo).to(k.dtype)
    ramp = 0.5 - 0.5 * torch.cos((math.pi * period / beta) * (k - lo))
    return torch.where(k < lo, 0.0, torch.where(k > hi, 1.0, ramp)).to(k.dtype)


def _mask_parameters(
    kind: str, numerical_aperture: float, wavelength: float
) -> tuple[float, float]:
    """(β, T) of the two masks of Supplementary Note 5."""
    na, lam = numerical_aperture, wavelength
    if kind == "otf-weighted":
        return 1.0, lam / (5.6 * na)
    if kind == "noise-only":
        beta = 0.2
        return beta, (1 - beta) * lam / (4 * na)
    raise ValueError(f"Unknown mask {kind!r}")


def ncs_mask(
    shape: Sequence[int],
    *,
    numerical_aperture: float,
    wavelength: float,
    pixel_size: float,
    mask: MaskKind | Callable[[Tensor], Tensor] = "otf-weighted",
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """OTF mask M(k_x, k_y) of an image of `shape` (Supp. Notes 4–5).

    The frequencies of an L₁ × L₂ image of pixel size Δ are
    k = p/(LΔ), p ∈ [−L/2, L/2 − 1] (Eq. 2), here in the order of
    `torch.fft.fftn` (zero frequency first), and k_r = √(k_x² + k_y²).
    The two masks of the paper are raised-cosine high-pass filters
    ([`raised_cosine`][optix.denoise.utils.raised_cosine]) around the
    OTF radius 2NA/λ of an ideal system:

    | mask | β | T | M = 0 below | M = 1 above |
    |---|---|---|---|---|
    | `"otf-weighted"` (default) | 1 | λ/(5.6 NA) | 0 | 5.6 NA/λ |
    | `"noise-only"` | 0.2 | (1 − β)λ/(4 NA) | 2 NA/λ | 3 NA/λ |

    The OTF-weighted mask grows from the centre of the spectrum and
    also weighs the signal band a little; the noise-only mask leaves the
    band inside the OTF untouched (Supplementary Notes 5 and 9,
    Supplementary Fig. 5). A callable gives a user-defined radial mask
    (Supplementary Note 9).

    Parameters
    ----------
    shape : sequence of int
        (L₁, L₂).
    numerical_aperture : float
        NA of the objective.
    wavelength : float
        Detection (emission) wavelength λ.
    pixel_size : float
        Pixel size Δ in the object plane, in the unit of `wavelength`.
    mask : {"otf-weighted", "noise-only"} or callable
        One of the paper's masks, or a function of k_r (cycles per unit
        of `wavelength`) returning M.
    dtype, device : optional
        Backend.

    Returns
    -------
    Tensor
        `shape`, values in [0, 1] for the paper's masks.
    """
    backend = resolve_backend(dtype, device)
    _check_mask(mask)
    na = _positive("numerical_aperture", numerical_aperture)
    lam = _positive("wavelength", wavelength)
    px = _positive("pixel_size", pixel_size)
    ny, nx = (int(n) for n in shape)
    ky = torch.fft.fftfreq(ny, d=px, **backend)
    kx = torch.fft.fftfreq(nx, d=px, **backend)
    kr = torch.sqrt(ky[:, None].square() + kx[None, :].square())
    if callable(mask):
        out = torch.as_tensor(mask(kr)).to(**backend)
        return out.expand(ny, nx).clone()
    beta, period = _mask_parameters(mask, na, lam)
    return raised_cosine(kr, beta, period)


# -- likelihood and cost (Supplementary Notes 3, 4 and 6) --------------------
def ncs_precorrect(
    data: Tensor,
    offset: Tensor | float,
    variance: Tensor | float,
    gain: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """Pre-corrected image D and calibration γ (Supp. Notes 2, 3, 6).

        Dᵢ = (Aᵢ − oᵢ)/gᵢ  (Eq. 1),  Dᵢ ← 10⁻⁶ where Dᵢ ≤ 0,
        γᵢ = σᵢ²/gᵢ².

    Parameters
    ----------
    data : Tensor
        Raw image A, in ADU.
    offset, variance, gain : Tensor or float
        Offset o (ADU), read-noise variance σ² (ADU²) and gain g (ADU
        per electron); maps broadcast with `data`.

    Returns
    -------
    D, gamma : Tensor
        In electrons and electrons², of the broadcast shape.
    """
    offset, variance, gain = (
        torch.as_tensor(t, dtype=data.dtype, device=data.device)
        for t in (offset, variance, gain)
    )
    d = (data - offset) / gain
    d = torch.where(d > 0, d, torch.full_like(d, _FLOOR))
    gamma = variance / gain.square()
    d, gamma = torch.broadcast_tensors(d, gamma)
    return d, gamma


def _half(mask: Tensor) -> Tensor:
    """Squared mask on the half spectrum of `torch.fft.rfft2`.

    The masks are even in k (a function of |k|, and the frequency −L/2
    is its own opposite), so the half spectrum is the first L₂//2 + 1
    columns of the `fftn`-ordered mask.
    """
    return mask[..., : mask.shape[-1] // 2 + 1].square()


def _noise_operator(m2: Tensor, shape: tuple[int, int]) -> Callable:
    """v ↦ Q v = F⁻¹(M² · F v), so that σ_N(μ) = μᵀQμ (Eq. 3)."""

    def apply(v: Tensor) -> Tensor:
        if v.ndim == 3 and v.shape[0] == 1:
            # a batch of one takes another FFT code path on the CPU, which
            # rounds differently: keep results independent of the batch
            return apply(v.expand(2, *v.shape[1:]))[:1]
        return torch.fft.irfft2(torch.fft.rfft2(v) * m2, s=shape)

    return apply


def ncs_objective(
    x: Tensor, data: Tensor, gamma: Tensor, mask: Tensor, alpha: float
) -> Tensor:
    """Cost f = LL_s + α σ_N of each image (Supp. Note 2, step 3).

        LL_s(μ) = ∑ᵢ μᵢ − (Dᵢ + γᵢ) ln(μᵢ + γᵢ)   (Eq. 7),
        σ_N(μ)  = ∑_k |μ̂(k) M(k)|²                  (Eq. 3),

    with μ̂ the unitary DFT of Eq. (2) (factor 1/L for an L × L image).

    Parameters
    ----------
    x : Tensor
        Estimate μ, `(..., L₁, L₂)`, with μ + γ > 0.
    data, gamma : Tensor
        D and γ of [`ncs_precorrect`][optix.denoise.utils.ncs_precorrect],
        broadcasting with `x`.
    mask : Tensor
        M, `(L₁, L₂)` in `fftn` order (see
        [`ncs_mask`][optix.denoise.utils.ncs_mask]).
    alpha : float
        Weight α.

    Returns
    -------
    Tensor
        f per image, `(...)`, in float64.
    """
    xd = x.double()
    c = (data + gamma).double()
    q = _noise_operator(_half(mask.double()), tuple(x.shape[-2:]))
    lls = (xd - c * torch.log(xd + gamma)).sum((-2, -1))
    return lls + alpha * (xd * q(xd)).sum((-2, -1))


def ncs_gradient(
    x: Tensor, data: Tensor, gamma: Tensor, mask: Tensor, alpha: float
) -> Tensor:
    """Gradient of [`ncs_objective`][optix.denoise.utils.ncs_objective].

        ∇f(μ) = 1 − (D + γ)/(μ + γ) + 2α F⁻¹(M² F μ).

    The first two terms are the gradient of the shifted-Poisson
    likelihood ([`ShiftedPoisson`][optix.optim.ShiftedPoisson] with data
    D and variance γ).

    Parameters
    ----------
    x, data, gamma, mask, alpha
        As in [`ncs_objective`][optix.denoise.utils.ncs_objective].

    Returns
    -------
    Tensor
        Of the shape of `x`.
    """
    q = _noise_operator(_half(mask.to(x)), tuple(x.shape[-2:]))
    return ShiftedPoisson(data, gamma).grad(x) + 2 * alpha * q(x)


class NcsResult(NamedTuple):
    """Result of [`ncs_solve`][optix.denoise.utils.ncs_solve].

    Attributes
    ----------
    x : Tensor
        Estimates μ, `(B, L₁, L₂)`.
    gradient : Tensor
        max |∇f| of each image at `x`, `(B,)`.
    n_iter : Tensor
        Newton iterations taken by each image, `(B,)`.
    converged : Tensor
        Whether max |∇f| ≤ `tol`, `(B,)` (bool).
    """

    x: Tensor
    gradient: Tensor
    n_iter: Tensor
    converged: Tensor


def _default_tol(dtype: torch.dtype) -> float:
    """max(100 ε, 10⁻¹⁰): 10⁻¹⁰ in float64, 1.2·10⁻⁵ in float32."""
    return max(100 * torch.finfo(dtype).eps, 1e-10)


def _dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum((-2, -1))


def ncs_solve(
    data: Tensor,
    gamma: Tensor,
    mask: Tensor,
    alpha: float,
    *,
    x0: Tensor | None = None,
    max_iter: int = 50,
    tol: float | None = None,
    cg_iter: int = 100,
) -> NcsResult:
    """Minimise f = LL_s + α σ_N for a batch of images.

    f is strictly convex on μ + γ > 0 (LL_s is a sum of strictly convex
    terms, σ_N a positive semi-definite quadratic), so the MAP estimate
    is unique. It is found by a damped Newton method, the paper leaving
    the optimiser open (Supp. Note 2: "a generic minimization routine
    such as fminunc"). The Hessian is

        H = diag((D + γ)/(μ + γ)²) + 2α Q,  Q = F⁻¹ diag(M²) F,

    and each Newton system H p = −∇f is solved by conjugate gradients
    preconditioned by the diagonal of H (the diagonal of Q is the mean
    of M²), to a residual of min(½, ‖∇f‖^½)·‖∇f‖. The step is the
    largest of 1, ½, ¼, … that keeps 1 % of the distance to the domain
    boundary μ = −γ and passes the Armijo test
    f(μ + tp) ≤ f(μ) + 10⁻⁴ t ∇fᵀp; the decrease is computed in closed
    form (with log1p), without forming f. Each image stops once
    max |∇f| ≤ `tol`, when the line search fails (round-off) or after
    `max_iter` iterations; the images of the batch are independent, so
    the result of an image does not depend on the others.

    !!! note "Not in the paper"
        The optimiser, its stopping rule and `max_iter`; the authors'
        software runs 15 iterations of a generic routine (Supp. Note 13)
        and stops before convergence. The start is the paper's, μ⁰ = D
        (Supp. Note 2, step 3).

    Parameters
    ----------
    data, gamma : Tensor
        D and γ, `(B, L₁, L₂)` (γ broadcasts).
    mask : Tensor
        M, `(L₁, L₂)`, in `fftn` order.
    alpha : float
        Weight α ≥ 0.
    x0 : Tensor, optional
        Start (default: `data`). Must satisfy x0 + γ > 0.
    max_iter : int
        Maximum number of Newton iterations.
    tol : float, optional
        Tolerance on max |∇f| (∇f is dimensionless). Default:
        max(100 ε, 10⁻¹⁰) for the machine epsilon ε of the dtype (10⁻¹⁰
        in float64, 1.2·10⁻⁵ in float32).
    cg_iter : int
        Maximum number of conjugate-gradient iterations per Newton step.

    Returns
    -------
    NcsResult
    """
    alpha = _nonnegative("alpha", alpha)
    max_iter = _count("max_iter", max_iter)
    cg_iter = _count("cg_iter", cg_iter)
    if data.ndim != 3:
        raise ValueError(f"data must be (B, L1, L2), got {tuple(data.shape)}")
    shape = tuple(data.shape[-2:])
    if tuple(mask.shape) != shape:
        raise ValueError(
            f"mask has shape {tuple(mask.shape)}, expected {shape}"
        )
    if tol is None:
        tol = _default_tol(data.dtype)
    tol = _nonnegative("tol", tol)
    gamma = gamma.to(data).expand_as(data)
    c = data + gamma
    mask = mask.to(data)
    m2 = _half(mask)
    q = _noise_operator(m2, shape)
    two_alpha = 2 * alpha
    qdiag = two_alpha * float(mask.double().square().mean())
    x = data.clone() if x0 is None else x0.to(data).clone()
    if not bool((x + gamma > 0).all()):
        raise ValueError("x0 + gamma must be positive")
    batch = data.shape[0]
    active = torch.ones(batch, dtype=torch.bool, device=data.device)
    n_iter = torch.zeros(batch, dtype=torch.long, device=data.device)

    def gradient(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        s = x + gamma
        qx = q(x)
        return 1 - c / s + two_alpha * qx, s, qx

    g, s, qx = gradient(x)
    for _ in range(max_iter):
        active &= g.abs().amax((-2, -1)) > tol
        if not bool(active.any()):
            break
        h = c / s.square()
        precond = h + qdiag
        # Newton direction by preconditioned CG, from p = 0
        p = torch.zeros_like(x)
        r = -g
        z = r / precond
        d = z
        rz = _dot(r, z)
        gnorm = _dot(g, g).sqrt()
        eta = torch.clamp(gnorm.sqrt(), max=0.5) * gnorm
        run = active.clone()
        for _ in range(cg_iter):
            hd = h * d + two_alpha * q(d)
            step = torch.where(run, rz / _dot(d, hd), 0)
            p = p + step[:, None, None] * d
            r = r - step[:, None, None] * hd
            run &= _dot(r, r).sqrt() > eta
            if not bool(run.any()):
                break
            z = r / precond
            rz_new = _dot(r, z)
            beta = torch.where(run, rz_new / rz, 0)
            d = z + beta[:, None, None] * d
            rz = torch.where(run, rz_new, rz)
        slope = _dot(g, p)
        bad = active & ~(slope < 0)
        if bool(bad.any()):  # round-off: fall back to scaled steepest descent
            p = torch.where(bad[:, None, None], -g / precond, p)
            slope = _dot(g, p)
        # largest step inside the domain, then backtracking (Armijo)
        reach = torch.where(p < 0, s / -p, math.inf).amin((-2, -1))
        t = torch.clamp(0.99 * reach, max=1.0)
        qp = q(p)
        pqx, pqp = _dot(p, qx), _dot(p, qp)
        ratio = p / s
        done = ~active
        accept = torch.zeros_like(active)
        for _ in range(60):
            tt = t[:, None, None]
            lls = (tt * p - c * torch.log1p(tt * ratio)).sum((-2, -1))
            change = lls + alpha * (2 * t * pqx + t.square() * pqp)
            ok = ~done & (change <= _ARMIJO * t * slope)
            accept |= ok
            done |= ok
            if bool(done.all()):
                break
            t = torch.where(done, t, t / 2)
        move = active & accept
        x = torch.where(move[:, None, None], x + t[:, None, None] * p, x)
        n_iter += move.long()
        active &= accept  # a failed line search ends the image
        g, s, qx = gradient(x)
    gmax = g.abs().amax((-2, -1))
    return NcsResult(x, gmax, n_iter, gmax <= tol)


# -- segmentation (Supplementary Note 7) -------------------------------------
def segment_bounds(n: int, segment: int) -> list[tuple[int, int]]:
    """Segments [start, stop) of an axis of `n` pixels (Supp. Note 7).

    Segments of `segment` = M pixels from the first pixel; when M does
    not divide n, the last segment also takes the remaining n mod M
    pixels (not in the paper), so that every segment has between M and
    2M − 1 pixels (n pixels when n < M).

    Parameters
    ----------
    n : int
        Length of the axis.
    segment : int
        M.

    Returns
    -------
    list of (int, int)
    """
    n, m = int(n), _count("segment", segment)
    if n <= m:
        return [(0, n)] if n > 0 else []
    k = n // m
    out = [(i * m, (i + 1) * m) for i in range(k)]
    out[-1] = (out[-1][0], n)
    return out


def _runs(bounds: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Consecutive segments of equal size."""
    runs: list[list[tuple[int, int]]] = []
    for b in bounds:
        if runs and runs[-1][0][1] - runs[-1][0][0] == b[1] - b[0]:
            runs[-1].append(b)
        else:
            runs.append([b])
    return runs


def _padded_index(
    run: list[tuple[int, int]], n: int, start: int, device: torch.device
) -> Tensor:
    """Indices (in the tile) of the segments of a run with their 1-pixel
    pad: the adjacent pixel inside the image, the edge pixel itself at
    its border (Supp. Note 7)."""
    size = run[0][1] - run[0][0]
    first = torch.tensor([b[0] for b in run], device=device)
    index = first[:, None] + torch.arange(-1, size + 1, device=device)
    return index.clamp(0, n - 1) - start


def _ncs_tile(
    data: Tensor,
    gamma: Tensor,
    start: tuple[int, int],
    full: tuple[int, int],
    core: tuple[slice, slice],
    mask_of: Callable[[tuple[int, int]], Tensor],
    alpha: float,
    segment: int,
    solver: dict,
) -> Tensor:
    """NCS of the segments of an image that meet `core`.

    `data` and `gamma` are D and γ on a tile of the image (shape `full`)
    starting at `start`, which must hold the segments meeting `core`
    (global slices) and their pads. Returns the estimate on `core`.
    """
    sel, runs, index = [], [], []
    for a in range(2):
        bounds = [
            b
            for b in segment_bounds(full[a], segment)
            if b[0] < core[a].stop and b[1] > core[a].start
        ]
        sel.append((bounds[0][0], bounds[-1][1]))
        runs.append(_runs(bounds))
        index.append(
            [
                _padded_index(r, full[a], start[a], data.device)
                for r in runs[-1]
            ]
        )
        lo = min(int(i.min()) for i in index[-1])
        hi = max(int(i.max()) for i in index[-1])
        if lo < 0 or hi >= data.shape[a]:
            raise ValueError("the tile does not hold the segments of core")
    out = data.new_empty(sel[0][1] - sel[0][0], sel[1][1] - sel[1][0])
    for ry, iy in zip(runs[0], index[0], strict=True):
        for rx, ix in zip(runs[1], index[1], strict=True):
            ny, nx = len(ry), len(rx)
            sy, sx = iy.shape[1], ix.shape[1]
            pick = (iy[:, None, :, None], ix[None, :, None, :])
            d = data[pick].reshape(ny * nx, sy, sx)
            g = gamma[pick].reshape(ny * nx, sy, sx)
            res = ncs_solve(d, g, mask_of((sy, sx)), alpha, **solver)
            x = res.x[:, 1:-1, 1:-1].reshape(ny, nx, sy - 2, sx - 2)
            x = x.permute(0, 2, 1, 3).reshape(ny * (sy - 2), nx * (sx - 2))
            y0, x0 = ry[0][0] - sel[0][0], rx[0][0] - sel[1][0]
            out[y0 : y0 + x.shape[0], x0 : x0 + x.shape[1]] = x
    return out[
        core[0].start - sel[0][0] : core[0].stop - sel[0][0],
        core[1].start - sel[1][0] : core[1].stop - sel[1][0],
    ]


class _Settings(NamedTuple):
    numerical_aperture: float
    wavelength: float
    pixel_size: float
    alpha: float
    mask: object
    segment: int
    units: str
    solver: dict


def _settings(
    numerical_aperture: object,
    wavelength: object,
    pixel_size: object,
    alpha: object,
    mask: object,
    segment: object,
    units: object,
    max_iter: object,
    tol: object,
) -> _Settings:
    _check_mask(mask)
    _check_units(units)
    return _Settings(
        _positive("numerical_aperture", numerical_aperture),
        _positive("wavelength", wavelength),
        _positive("pixel_size", pixel_size),
        _nonnegative("alpha", alpha),
        mask,
        _count("segment", segment),
        units,
        {
            "max_iter": _count("max_iter", max_iter),
            "tol": None if tol is None else _nonnegative("tol", tol),
        },
    )


def _mask_cache(st: _Settings, like: Tensor) -> Callable:
    cache: dict[tuple[int, int], Tensor] = {}

    def mask_of(shape: tuple[int, int]) -> Tensor:
        if shape not in cache:
            cache[shape] = ncs_mask(
                shape,
                numerical_aperture=st.numerical_aperture,
                wavelength=st.wavelength,
                pixel_size=st.pixel_size,
                mask=st.mask,
                dtype=like.dtype,
                device=like.device,
            )
        return cache[shape]

    return mask_of


def _output(x: Tensor, units: str, offset: Tensor, gain: Tensor) -> Tensor:
    return x if units == "electrons" else x * gain + offset


# -- functional API ----------------------------------------------------------
def ncs(
    data: Tensor,
    sensor: SensorEstimate,
    *,
    numerical_aperture: float,
    wavelength: float,
    pixel_size: float,
    alpha: float = 0.2,
    mask: MaskKind | Callable[[Tensor], Tensor] = "otf-weighted",
    segment: int = 16,
    units: Units = "electrons",
    max_iter: int = 50,
    tol: float | None = None,
) -> Tensor:
    """NCS: noise correction of sCMOS images (Liu et al. 2017).

    Each pixel i of an sCMOS camera has its own offset oᵢ, gain gᵢ (ADU
    per electron) and read-noise variance σᵢ² (ADU²). The algorithm
    (Supplementary Note 2, Supplementary Fig. 1) is:

    1. **Pre-correction** (Supp. Note 3, Eq. 1): Dᵢ = (Aᵢ − oᵢ)/gᵢ for
       the raw image A, with Dᵢ ← 10⁻⁶ where Dᵢ ≤ 0 (Supp. Notes 2 and
       6), and γᵢ = σᵢ²/gᵢ² (Supp. Note 6).
    2. **Segmentation** (Supp. Note 7): the image is cut into segments
       of M × M pixels, each padded by one pixel on every side to
       (M + 2) × (M + 2): with the adjacent pixel of the image, or with
       the pixel itself at the border of the image. The pad is dropped
       from the result.
    3. **Estimation**: in each padded segment, μ minimises

           f(μ) = LL_s(μ) + α σ_N(μ),

       from μ⁰ = D. LL_s is the simplified negative log-likelihood
       (Supp. Note 6, Eqs. 5–7),

           LL_s(μ) = ∑ᵢ μᵢ − (Dᵢ + γᵢ) ln(μᵢ + γᵢ),

       of the paper's approximation of the Poisson–Gaussian pixel
       model: Dᵢ + γᵢ is taken as Poisson with mean μᵢ + γᵢ (Eq. 5),
       with Stirling's formula for Γ(Dᵢ + γᵢ + 1) (Eq. 6) and the
       constants dropped (Eq. 7). σ_N is the noise contribution near
       and outside the OTF support (Supp. Note 4, Eqs. 2–3),

           σ_N(μ) = ∑_k |μ̂(k) M(k)|²,

       with μ̂ the DFT of the segment normalised by 1/L (unitary) and M
       the OTF mask ([`ncs_mask`][optix.denoise.utils.ncs_mask]), a
       raised-cosine high-pass filter around the cut-off 2NA/λ (Supp.
       Note 5, Eq. 4). α = 0.2 (Supp. Note 2, Fig. SS6).

    The output is μ, in electrons (the paper's "effective photon
    count"), or with `units="adu"` μ·g + o.

    !!! note "Choices not specified in the paper"
        * The optimiser: damped Newton with conjugate gradients, run to
          max |∇f| ≤ `tol` ([`ncs_solve`][optix.denoise.utils.ncs_solve]).
          The paper uses "a generic minimization routine such as
          fminunc" for 15 iterations by default (Supp. Notes 2 and 13),
          which stops before the minimum; this implementation returns
          the minimiser of the paper's cost.
        * `segment` M = 16, in the recommended range 8–32 (Supp. Notes 2
          and 13). When M does not divide the image, the last segment
          along an axis takes the remaining pixels
          ([`segment_bounds`][optix.denoise.utils.segment_bounds]).
        * Eq. (4) prints the transition band of the mask with ½ + ½ cos;
          the continuous high-pass ½ − ½ cos is used (see
          [`raised_cosine`][optix.denoise.utils.raised_cosine]).
        * Segments need not be square or of even size; the frequencies
          of each axis are those of Eq. (2) for its own length.

    !!! note "Streaming"
        Segments are independent, and a segment depends only on the
        data of the segment and of its pad. The result at a pixel
        therefore depends on the data within
        [`ncs_halo`][optix.denoise.ncs_halo]`(segment)` = 2M − 1 pixels,
        and [`NCS`][optix.denoise.NCS] streams large arrays in blocks
        with this halo, segmenting on the grid of the whole image.

    !!! references "References"
        1. Liu S, Mlodzianoski MJ, Hu Z, Ren Y, McElmurry K, Suter DM,
           Huang F. *sCMOS noise-correction algorithm for microscopy
           images.* Nat Methods **14**(8):760–761, 2017.
           [doi:10.1038/nmeth.4379](https://doi.org/10.1038/nmeth.4379)

        ??? quote "BibTeX"
            ```bibtex
            @article{liu2017scmos,
              author  = {Liu, Sheng and Mlodzianoski, Michael J. and
                         Hu, Zhenhua and Ren, Yuan and McElmurry, Kristi
                         and Suter, Daniel M. and Huang, Fang},
              title   = {{sCMOS} noise-correction algorithm for
                         microscopy images},
              journal = {Nature Methods},
              volume  = {14}, number = {8}, pages = {760--761},
              year    = {2017}, doi = {10.1038/nmeth.4379}
            }
            ```

    Parameters
    ----------
    data : Tensor
        Raw images A in ADU, `(..., H, W)`; each image is corrected
        separately (other dtypes are converted to the configured
        floating dtype).
    sensor : SensorEstimate
        `offset` (ADU), `variance` (ADU²) and `gain` (ADU per electron):
        scalars or maps broadcasting to `data`.
    numerical_aperture : float
        NA of the objective.
    wavelength : float
        Detection wavelength λ.
    pixel_size : float
        Pixel size in the object plane, in the unit of `wavelength`.
    alpha : float
        Weight α of the noise contribution (paper: 0.2).
    mask : {"otf-weighted", "noise-only"} or callable
        OTF mask (paper default: `"otf-weighted"`); a callable of k_r is
        a user-defined radial mask.
    segment : int
        Segment size M (not in the paper: 16).
    units : {"electrons", "adu"}
        Units of the result (paper: electrons).
    max_iter : int
        Maximum number of Newton iterations per segment.
    tol : float, optional
        Tolerance on max |∇f| (default: 10⁻¹⁰ in float64, 1.2·10⁻⁵ in
        float32; see [`ncs_solve`][optix.denoise.utils.ncs_solve]).

    Returns
    -------
    Tensor
        The corrected images, of the shape of `data`.
    """
    st = _settings(
        numerical_aperture,
        wavelength,
        pixel_size,
        alpha,
        mask,
        segment,
        units,
        max_iter,
        tol,
    )
    _check_sensor(sensor)
    if data.ndim < 2:
        raise ValueError(f"data needs at least 2 dims, got {data.ndim}")
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    offset, variance, gain = (
        torch.as_tensor(t, dtype=data.dtype, device=data.device)
        for t in (sensor.offset, sensor.variance, sensor.gain)
    )
    shape = torch.broadcast_shapes(
        data.shape, offset.shape, variance.shape, gain.shape
    )
    if shape != data.shape:
        raise ValueError(
            f"sensor maps do not broadcast to data of shape "
            f"{tuple(data.shape)}"
        )
    d, gamma = ncs_precorrect(data, offset, variance, gain)
    full = tuple(int(n) for n in data.shape[-2:])
    core = (slice(0, full[0]), slice(0, full[1]))
    mask_of = _mask_cache(st, d)
    x = torch.stack(
        [
            _ncs_tile(
                di,
                gi,
                (0, 0),
                full,
                core,
                mask_of,
                st.alpha,
                st.segment,
                st.solver,
            )
            for di, gi in zip(
                d.reshape(-1, *full), gamma.reshape(-1, *full), strict=True
            )
        ]
    ).reshape(data.shape)
    return _output(x, st.units, offset, gain)


def ncs_halo(segment: int = 16) -> int:
    """Halo, in pixels, that makes block-wise NCS exact: 2M − 1.

    A segment meeting a block has at most 2M − 1 pixels (the last one
    of an axis takes the remainder; see
    [`segment_bounds`][optix.denoise.utils.segment_bounds]) and a pad of
    one pixel, so it reaches 2M − 1 pixels beyond the block.

    Parameters
    ----------
    segment : int
        Segment size M.

    Returns
    -------
    int
    """
    return 2 * _count("segment", segment) - 1


def _check_sensor(sensor: object) -> None:
    if not isinstance(sensor, SensorEstimate):
        raise TypeError(
            f"sensor must be a SensorEstimate, got {type(sensor).__name__}"
        )
    for name in ("offset", "variance", "gain"):
        if getattr(sensor, name) is None:
            raise ValueError(f"the SensorEstimate lacks {name}")


# -- estimator ---------------------------------------------------------------
class NCS(Estimator):
    """NCS noise correction of sCMOS images (Liu et al. 2017).

    Applies [`ncs`][optix.denoise.ncs] to each image (the two spatial
    axes of the layout; other axes are samples or groups) with the
    pixel maps of a [`SensorEstimate`][optix.estimates.SensorEstimate]
    given to [`fit`][optix.denoise.NCS.fit] (offset in ADU, read-noise
    variance in ADU², gain in ADU per electron; maps of shape
    `(*group, H, W)`, `(H, W)` or scalars, cropped to each streamed
    block).

    [`apply`][optix.Estimator.apply] streams blocks with the halo of
    [`ncs_halo`][optix.denoise.ncs_halo]; the segments are on the grid
    of the whole image, so arrays larger than memory (zarr, memmap) give
    the whole-image result.

    !!! note
        Faithful mode is the paper's cost: α = 0.2, an OTF mask of the
        paper (`"otf-weighted"`, the default, or `"noise-only"`) and a
        segment size in the recommended range 8–32. Another α, a
        user-defined mask or another segment size are extensions.
        `units`, `max_iter`, `tol` and `block` do not change the method.

    !!! references "References"
        1. Liu S, Mlodzianoski MJ, Hu Z, Ren Y, McElmurry K, Suter DM,
           Huang F. *sCMOS noise-correction algorithm for microscopy
           images.* Nat Methods **14**(8):760–761, 2017.
           [doi:10.1038/nmeth.4379](https://doi.org/10.1038/nmeth.4379)

        ??? quote "BibTeX"
            ```bibtex
            @article{liu2017scmos,
              author  = {Liu, Sheng and Mlodzianoski, Michael J. and
                         Hu, Zhenhua and Ren, Yuan and McElmurry, Kristi
                         and Suter, Daniel M. and Huang, Fang},
              title   = {{sCMOS} noise-correction algorithm for
                         microscopy images},
              journal = {Nature Methods},
              volume  = {14}, number = {8}, pages = {760--761},
              year    = {2017}, doi = {10.1038/nmeth.4379}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.denoise import NCS
    >>> from optix.estimates import SensorEstimate
    >>> g = torch.Generator().manual_seed(0)
    >>> variance = 1 + 400 * torch.rand(48, 48, generator=g)
    >>> sensor = SensorEstimate(
    ...     offset=torch.tensor(100.0), variance=variance,
    ...     gain=torch.tensor(2.0))
    >>> clean = torch.full((48, 48), 20.0)
    >>> noise = variance.sqrt() * torch.randn(48, 48, generator=g)
    >>> raw = 100 + 2 * torch.poisson(clean, generator=g) + noise
    >>> est = NCS(numerical_aperture=1.4, wavelength=0.7, pixel_size=0.1)
    >>> out = est.fit(init=sensor).apply(raw)
    >>> before = ((raw - 100) / 2 - clean).square().mean()
    >>> bool((out - clean).square().mean() < before)
    True
    """

    tags = Tags(
        targets=("denoised",),
        streaming="halo",
        noise="poisson-gaussian",
        reference="doi:10.1038/nmeth.4379",
    )
    paper = {
        "numerical_aperture": None,
        "wavelength": None,
        "pixel_size": None,
        "alpha": 0.2,
        "mask": "otf-weighted",
        "segment": 16,
        "units": "electrons",
        "max_iter": 50,
        "tol": None,
        "block": None,
    }
    recommended = dict(paper)

    def __init__(
        self,
        *,
        numerical_aperture: float | None = None,
        wavelength: float | None = None,
        pixel_size: float | None = None,
        alpha: float | None = None,
        mask: MaskKind | Callable[[Tensor], Tensor] | None = None,
        segment: int | None = None,
        units: Units | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        block: Sequence[int | None] | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the noise correction.

        Parameters
        ----------
        numerical_aperture : float
            NA of the objective (required).
        wavelength : float
            Detection wavelength λ (required).
        pixel_size : float
            Pixel size in the object plane, in the unit of `wavelength`
            (required).
        alpha : float, optional
            Weight α of the noise contribution. Default: 0.2 (the paper).
        mask : {"otf-weighted", "noise-only"} or callable, optional
            OTF mask (see [`ncs_mask`][optix.denoise.utils.ncs_mask]).
            Default: `"otf-weighted"` (the paper's recommendation).
        segment : int, optional
            Segment size M. Default: 16 (not in the paper, which
            recommends 8–32).
        units : {"electrons", "adu"}, optional
            Units of the result. Default: `"electrons"` (the paper).
        max_iter : int, optional
            Maximum number of Newton iterations per segment. Default: 50.
        tol : float, optional
            Tolerance on max |∇f|. Default: 10⁻¹⁰ in float64,
            1.2·10⁻⁵ in float32.
        block : sequence of int or None, optional
            Core block size per spatial axis (default: from the memory
            budget). The result does not depend on it.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        self._st = _settings(
            o["numerical_aperture"],
            o["wavelength"],
            o["pixel_size"],
            o["alpha"],
            o["mask"],
            o["segment"],
            o["units"],
            o["max_iter"],
            o["tol"],
        )
        self.is_faithful = (
            self._st.alpha == 0.2
            and o["mask"] in ("otf-weighted", "noise-only")
            and 8 <= self._st.segment <= 32
        )
        self._maps: tuple[Tensor, Tensor, Tensor] | None = None

    def fit(
        self,
        data: ArrayLike | None = None,
        *,
        layout: Layout | None = None,
        init: object = None,
        **kwargs: object,
    ) -> Self:
        """Set the camera maps.

        Parameters
        ----------
        data : array-like, optional
            Not read; only its layout is recorded.
        layout : Layout, optional
        init : SensorEstimate or Estimator
            Offset, read-noise variance and gain (scalars or maps), or
            an estimator holding such an estimate (e.g. a fitted
            [`SCMOSCalibration`][optix.sensor.SCMOSCalibration]).
        **kwargs
            Ignored (`callback`, `verbose`).

        Returns
        -------
        Estimator
            self
        """
        if data is None:
            self._set_layout(layout)
            self._fit(None, init=init, callback=None, verbose=False)
            return self
        return super().fit(data, layout=layout, init=init)

    def _fit(
        self,
        data: ArrayLike | None,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        sensor = self._pick(init, SensorEstimate) if init is not None else None
        if init is not None:
            self._warn_unused(init, SensorEstimate)
        if sensor is None:
            raise ValueError("NCS needs fit(init=SensorEstimate)")
        _check_sensor(sensor)
        self.estimates = {"sensor": sensor}
        self._set_sensor(sensor)

    def _set_sensor(self, sensor: SensorEstimate | None) -> None:
        self._maps = (
            None
            if sensor is None
            else (sensor.offset, sensor.variance, sensor.gain)
        )

    def halo(self) -> int:
        """Halo of the block-wise processing, 2M − 1.

        Returns
        -------
        int
        """
        return ncs_halo(self._st.segment)

    def _block_maps(self, chunk: object, x: Tensor) -> tuple[Tensor, ...]:
        """Offset, variance and gain cropped to a streamed block."""
        out = []
        for t in self._maps:
            t = _select(torch.as_tensor(t), chunk.group, 2)
            index = []
            for i, (core, inner) in enumerate(
                zip(chunk.spatial, chunk.inner, strict=True)
            ):
                if t.shape[i] == 1:
                    index.append(slice(None))
                else:
                    start = core.start - inner.start
                    index.append(slice(start, start + x.shape[i]))
            out.append(t[tuple(index)].to(x))
        return tuple(out)

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if self._maps is None:
            raise RuntimeError("Call fit before apply")
        if not isinstance(data, Labeled):
            data = self._labeled(data, self._layout_of(data, 2))
        _, _, spatial = data.layout.positions(len(data.shape))
        if len(spatial) != 2:
            raise ValueError(f"NCS needs 2 spatial axes, got {len(spatial)}")
        full = tuple(int(data.shape[a]) for a in spatial)
        raw = data.array
        target = out.array if isinstance(out, Labeled) else out
        if out is not None and _shares_memory(target, raw):
            raise ValueError(
                "In-place processing with a halo is not possible: use a "
                "separate output."
            )
        out = prepare_out(out, tuple(data.shape), like=raw, dtype=self.dtype)
        st = self._st
        chunks = blocks(
            data,
            self.options["block"],
            self.halo(),
            overhead=24.0,
            contiguous=True,
            align=chunks_of(out),
            dtype=self.dtype,
            device=self.device,
        )
        mask_of = None
        for chunk in chunks:
            x = chunk.data
            if not x.is_floating_point():
                x = x.to(self._backend(x)["dtype"])
            if mask_of is None:
                mask_of = _mask_cache(st, x)
            offset, variance, gain = self._block_maps(chunk, x)
            d, gamma = ncs_precorrect(x, offset, variance, gain)
            start = tuple(
                core.start - inner.start
                for core, inner in zip(chunk.spatial, chunk.inner, strict=True)
            )
            core = tuple(chunk.spatial)
            result = _ncs_tile(
                d,
                gamma,
                start,
                full,
                core,
                mask_of,
                st.alpha,
                st.segment,
                st.solver,
            )
            if st.units == "adu":
                result = _output(
                    result,
                    "adu",
                    offset.expand_as(x)[chunk.inner],
                    gain.expand_as(x)[chunk.inner],
                )
            chunk.store(out, result)
        flush(out)
        return out

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self._set_sensor(self.estimates.get("sensor"))
