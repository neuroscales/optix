"""Numerical core of BaSiC: DCT, resizing, the LADM solver and baselines.

Everything here works on a working stack already in memory, shaped
`(n, h, w)` (`n` frames at the working size). Frame selection, sorting,
resizing and batching live in `optix.shading._basic`.

The solver follows Supplementary Note 3 of [Peng et al.
(2017)](https://doi.org/10.1038/ncomms14836): Table 1 (flat field), Table 2
(dark field) and Eq. (9) (weights), with the automatic λ of Supplementary Note
4. The `variant="basicpy"` code paths reproduce
[BaSiCPy](https://github.com/peng-lab/BaSiCPy) 2.0.0 (MIT licence, Copyright
2022 BaSiCPy developing collaboration group; `basicpy/_torch_routines.py`,
`ApproximateFit`), and `baseline_basicpy` its `fit_only_baseline`.

Notation ([BaSiCPy](https://github.com/peng-lab/BaSiCPy) names in brackets):

* `D` (n, h, w): working stack, the matrix I of the paper [Im].
* `S` (h, w): unnormalised flat field, with orthonormal 2-D DCT
  coefficients `S_hat` [S, S_hat].
* `B` (n,): per-frame coefficients, mean 1 [B].
* `D_Z` (scalar, in units of B) and `D_R` (h, w): constant part and
  residual of the dark field. The working variable `Z` holds
  `D_Z (mean(S) − S) + D_R`, so that the low-rank part is
  `I_B = B S + Z` (Eq. 12).
* `E` (n, h, w): sparse residual I_R [I_R]; `Y`: Lagrange multipliers.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from ..basis import dct_basis

__all__ = [
    "VARIANTS",
    "BaSiCResult",
    "Dct2",
    "baseline_basicpy",
    "baseline_l1",
    "basic_ladm",
    "default_lambdas",
    "resize",
    "resize_matrix",
    "shrink",
]

VARIANTS = ("paper", "basicpy")


# -- elementary operators -------------------------------------------------
def shrink(x: Tensor, t: Tensor | float) -> Tensor:
    """Soft threshold 𝒯ₜ(x) = sign(x) max(|x| − t, 0), the prox of t‖·‖₁."""
    return torch.sign(x) * torch.clamp(x.abs() - t, min=0)


class Dct2:
    """Orthonormal 2-D DCT-II of `(..., h, w)` images, by matrices.

    `forward(x) = C_h x C_wᵀ`, with C the orthonormal DCT-II matrix;
    `inverse` is its transpose.
    """

    def __init__(
        self, h: int, w: int, *, dtype: torch.dtype, device: torch.device
    ) -> None:
        self.ch = dct_basis(h, dtype=dtype, device=device).T.contiguous()
        self.cw = dct_basis(w, dtype=dtype, device=device).T.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        return self.ch @ x @ self.cw.T

    def inverse(self, x: Tensor) -> Tensor:
        return self.ch.T @ x @ self.cw


def resize(x: Tensor, size: tuple[int, int] | None) -> Tensor:
    """Bilinear resize of `(n, h, w)` images.

    Bilinear interpolation with aligned corners and anti-aliasing (which
    only acts when shrinking). `size=None` or the same size returns `x`
    unchanged.
    """
    if size is None or tuple(x.shape[-2:]) == tuple(size):
        return x
    y = F.interpolate(
        x[:, None],
        size=tuple(size),
        mode="bilinear",
        align_corners=True,
        antialias=True,
    )
    return y[:, 0]


def resize_matrix(
    n_in: int,
    n_out: int,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
    block: int = 256,
) -> Tensor:
    """Matrix of `resize` along one axis.

    The resize is separable and linear: for an `(H, W)` image x,
    `resize(x) = R_h x R_wᵀ` with `R_h = resize_matrix(H, h)`.

    Returns
    -------
    (n_out, n_in) Tensor
    """
    if n_in == n_out:
        return torch.eye(n_in, dtype=dtype, device=device)
    cols = []
    for start in range(0, n_in, block):
        stop = min(start + block, n_in)
        eye = torch.zeros(stop - start, n_in, 4, dtype=dtype, device=device)
        idx = torch.arange(stop - start, device=device)
        eye[idx, start + idx] = 1
        # a constant second axis of length 4: the 1-D weights are those of
        # the first axis (a length-1 axis takes another code path)
        cols.append(resize(eye, (n_out, 4))[:, :, 0])
    return torch.cat(cols).T.contiguous()


def default_lambdas(D: Tensor, dct: Dct2) -> tuple[float, float]:
    """Automatic λ_S and λ_D (Supplementary Note 4, Eq. 15).

    λ_S = ‖𝓕(Ī_norm)‖₁ / 800 and λ_D = ‖𝓕(Ī_norm)‖₁ / 2000, with Ī_norm
    the per-pixel mean of the frames divided by its spatial mean and 𝓕
    the orthonormal 2-D DCT (the "Fourier transform" of the SI is the
    DCT of the authors' implementation).

    Parameters
    ----------
    D : (n, h, w) Tensor
        Working stack (sorted or not: the mean is the same).
    dct : Dct2

    Returns
    -------
    lambda_flat, lambda_dark : float
    """
    mean = D.mean(0)
    mean = mean / mean.mean()
    s = float(dct.forward(mean).abs().sum())
    return s / 800, s / 2000


def _spectral_norm(D: Tensor) -> Tensor:
    """Largest singular value of `D` seen as `(n, h·w)`."""
    return torch.linalg.matrix_norm(D.reshape(D.shape[0], -1), ord=2)


@dataclass
class BaSiCResult:
    """Working-size output of a BaSiC fit.

    Attributes
    ----------
    flat : (h, w) Tensor
        Flat field S / mean(S).
    dark : (h, w) Tensor
        Dark field D = D_Z + D_R, in intensity units (zeros when not
        estimated).
    baseline : (n,) Tensor
        Per-frame coefficients of the fitted frames in units of the
        flat-corrected signal, Bᵢ · mean(S) (the unsorted stack only:
        after sorting they belong to the sorted frames).
    iterations : list of int
        Inner iterations of each reweighting step.
    residuals : list of float
        Final relative residual ‖I − I_B − I_R‖_F / ‖I‖_F of each step.
    changes : list of float
        Change of the estimate after each step (`inf` for the first).
    converged : bool
        Whether reweighting stopped on its tolerance.
    """

    flat: Tensor
    dark: Tensor
    baseline: Tensor
    iterations: list[int] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    changes: list[float] = field(default_factory=list)
    converged: bool = False

    def update(
        self,
        flat: Tensor,
        dark: Tensor,
        baseline: Tensor,
        it: int,
        res: float,
        change: float,
    ) -> None:
        self.flat, self.dark, self.baseline = flat, dark, baseline
        self.iterations.append(it)
        self.residuals.append(res)
        self.changes.append(change)


def _relative_change(new: Tensor, last: Tensor | None) -> float:
    """‖new − last‖₁ / ‖last‖₁ (`inf` without `last`, 0 if both vanish)."""
    if last is None:
        return math.inf
    diff = float((new - last).abs().sum())
    if diff == 0:
        return 0.0
    return diff / max(float(last.abs().sum()), torch.finfo(new.dtype).tiny)


# -- dark field (Supplementary Note 3, Table 2) -----------------------------
def _dark_update(
    R: Tensor,
    B: Tensor,
    S: Tensor,
    dct: Dct2,
    lambda_dark: float,
    mu: Tensor,
    D_Z_max: Tensor,
    variant: str,
) -> tuple[Tensor, Tensor] | None:
    """One dark-field update; returns `(Z, D_Z)`, or None to keep them.

    `R = I − I_R`. D_Z is the root of the least-squares line through the
    points (Bᵢ, Aᵢ), with Aᵢ the contrast between the pixels where S is
    above and below its mean, divided by mean(R) (Table 2, lines 1–6).
    Then D_R = Z − mean(Z) − D_Z (mean(S) − S) with
    Z = meanᵢ(Rᵢ) − mean(B) S, shrunk in the image and DCT domains
    (lines 7–11).
    """
    n = R.shape[0]
    R2 = R.reshape(n, -1)
    Sf = S.reshape(-1)
    Smean = Sf.mean()
    tiny = torch.finfo(R.dtype).tiny
    if variant == "basicpy":
        # frames with Bᵢ < 1 only, and the residual shrunk by λ_D / (10 μ),
        # in the DCT domain first
        valid = B < 1
        if not bool(valid.any()):
            return None
        denom = R2.mean() + 1e-6
        t = lambda_dark / (10.0 * mu)
    else:
        valid = torch.ones_like(B, dtype=torch.bool)
        denom = R2.mean().clamp_min(tiny)
        t = lambda_dark / mu
    k = int(valid.sum())
    Rv, Bv = R2[valid], B[valid]
    inmask = Sf >= Smean
    outmask = ~inmask
    if inmask.any() and outmask.any():
        A = (Rv[:, inmask].mean(1) - Rv[:, outmask].mean(1)) / denom
    else:
        A = torch.zeros_like(Bv)
    t1, t2 = (Bv**2).sum(), Bv.sum()
    t3, t4 = A.sum(), (Bv * A).sum()
    t5 = t2 * t3 - k * t4
    if float(t5) == 0:
        D_Z = torch.zeros_like(t5)
    else:
        D_Z = (t1 * t3 - t2 * t4) / t5
    # D_Z ∈ [0, min(I)] (Eq. 13), D_Z here in units of B
    D_Z = torch.minimum(torch.clamp(D_Z, min=0), D_Z_max / Smean)
    ramp = D_Z * (Smean - Sf)
    off = Rv.mean(0) - Bv.mean() * Sf
    D_R = (off - off.mean() - ramp).reshape(S.shape)
    if variant == "basicpy":
        D_R = shrink(dct.inverse(shrink(dct.forward(D_R), t)), t)
    else:
        D_R = dct.inverse(shrink(dct.forward(shrink(D_R, t)), t))
    return D_R + ramp.reshape(S.shape), D_Z


# -- flat field (Supplementary Note 3, Table 1) -----------------------------
def _ladm_solve(
    D: Tensor,
    W: Tensor,
    dct: Dct2,
    *,
    lambda_flat: float,
    lambda_dark: float,
    darkfield: bool,
    tol: float,
    max_iterations: int,
    mu: Tensor,
    rho: float,
    max_mu: Tensor,
    variant: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float]:
    """One LADM solve for fixed weights.

    Returns `S, Z, D_Z, I_B, E, B, iterations, residual`.
    """
    n, h, w = D.shape
    D_Z_max = D.min()
    image_norm = torch.linalg.vector_norm(D)
    zeros = torch.zeros(h, w, dtype=D.dtype, device=D.device)
    if variant == "basicpy":
        S_hat = dct.forward(torch.ones_like(zeros))
        B = D.mean((1, 2)) / D.mean()
    else:
        # Table 1, line 1: Y, S, B and I_R start at zero
        S_hat = zeros.clone()
        B = torch.zeros(n, dtype=D.dtype, device=D.device)
    S = dct.inverse(S_hat)
    Z = zeros.clone()
    D_Z = torch.zeros((), dtype=D.dtype, device=D.device)
    E = torch.zeros_like(D)
    Y = torch.zeros_like(D)
    I_B = S * B[:, None, None] + Z
    res = math.inf
    it = 0
    while res > tol and it < max_iterations:
        it += 1
        # lines 3–5: S_F ← 𝒯_{λ_S/μ}(S_F + 𝓕[meanᵢ(I − I_B − I_R + Y/μ)])
        I_B = S * B[:, None, None] + Z
        S_hat = shrink(
            S_hat + dct.forward((D - I_B - E + Y / mu).mean(0)),
            lambda_flat / mu,
        )
        S = dct.inverse(S_hat)
        # line 6: I_R ← 𝒯_{W/μ}(I − I_B + Y/μ)
        I_B = S * B[:, None, None] + Z
        E = shrink(D - I_B + Y / mu, W / mu)
        # line 7: Bᵢ = mean_x(Iᵢ − I_{R,i}) / mean(I − I_R)
        R = D - E
        B = R.mean((1, 2)) / R.mean()
        if variant == "basicpy":
            B = torch.clamp(B, min=0)
        I_B = S * B[:, None, None] + Z
        if darkfield:
            dark = _dark_update(
                R, B, S, dct, lambda_dark, mu, D_Z_max, variant
            )
            if dark is not None:
                Z, D_Z = dark
        # line 8
        resid = D - I_B - E
        Y = Y + mu * resid
        # μ is capped so that it stays finite over long runs
        mu = torch.minimum(mu * rho, max_mu)
        res = float(torch.linalg.vector_norm(resid) / image_norm)
    return S, Z, D_Z, I_B, E, B, it, res


def _weights(
    I_B: Tensor,
    E: Tensor,
    epsilon: float,
    variant: str,
    normalize: bool = True,
) -> Tensor:
    """Reweighting of the sparse term.

    Eq. (9): W = 1 / (|I_R / I_B| + ε), elementwise, then scaled to mean 1 over
    the stack when `normalize`. The
    [BaSiCPy](https://github.com/peng-lab/BaSiCPy) variant divides by the mean
    of I_B over each frame instead, and always normalises.
    """
    if variant == "basicpy":
        W = 1 / ((E / (I_B.mean((1, 2), keepdim=True) + 1e-6)).abs() + epsilon)
        return W * (W.numel() / W.sum())
    tiny = torch.finfo(I_B.dtype).tiny
    W = 1 / ((E / I_B.abs().clamp_min(tiny)).abs() + epsilon)
    if normalize:
        # the λ of Supplementary Note 4 are read as tuned for weights of
        # mean 1 (Eq. 9 itself has no normalisation)
        W = W * (W.numel() / W.sum())
    return W


def basic_ladm(
    D: Tensor,
    *,
    lambda_flat: float,
    lambda_dark: float,
    darkfield: bool = False,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_iterations: int = 500,
    max_reweight: int = 10,
    reweight_tol: float = 1e-3,
    mu_coef: float = 12.5,
    rho: float = 1.5,
    max_mu_coef: float = 1e7,
    variant: str = "paper",
    normalize_weights: bool = True,
) -> BaSiCResult:
    """BaSiC on a working stack: reweighted L1 solved by LADM.

    Model (Supplementary Note 3, Eqs. 12–13), for the frames Iᵢ of the
    working stack:

        Iᵢ = Bᵢ S + Z + I_{R,i},    Z = D_Z (1 − S) + D_R

    minimising ‖W ∘ I_R‖₁ + λ_S ‖𝓕(S)‖₁ (with the dark field, also
    λ_D ‖𝓕(D_R)‖₁ + λ_D ‖D_R‖₁, and D_Z ∈ [0, min I]). Each reweighting
    step runs the LADM of Table 1 (and Table 2) from zero with fixed W,
    then updates W by Eq. (9); reweighting stops when the flat field (and
    the dark field) change by less than `reweight_tol` in relative L1
    norm, or after `max_reweight` steps.

    Parameters
    ----------
    D : (n, h, w) Tensor
        Working stack.
    lambda_flat, lambda_dark : float
        λ_S and λ_D (see `default_lambdas`).
    darkfield : bool
        Estimate the dark field.
    epsilon : float
        ε of Eq. (9).
    tol, max_iterations : float, int
        Inner stop: ‖I − I_B − I_R‖_F / ‖I‖_F < tol.
    max_reweight, reweight_tol : int, float
        Reweighting stop.
    mu_coef, rho, max_mu_coef : float
        Penalty schedule: μ₀ = mu_coef / ‖I‖₂, μ ← min(ρ μ, max_mu_coef μ₀).
    variant : {"paper", "basicpy"}
        "paper": Supplementary Note 3 as written. "basicpy":
        [BaSiCPy](https://github.com/peng-lab/BaSiCPy) 2.0.0's
        `ApproximateFit` (S starts at one, B clipped at 0, weights
        divided by the frame mean of I_B and normalised, dark field from
        the frames with Bᵢ < 1 shrunk with λ_D / (10 μ), and a
        reweighting score on S only).
    normalize_weights : bool
        Scale the weights of Eq. (9) to mean 1 ("paper" variant; the
        "basicpy" variant always does).

    Returns
    -------
    BaSiCResult
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}: {variant!r}")
    dct = Dct2(*D.shape[1:], dtype=D.dtype, device=D.device)
    mu0 = mu_coef / _spectral_norm(D)
    W = torch.ones_like(D)
    last_flat = last_dark = None
    result = None
    for _ in range(max_reweight):
        S, Z, D_Z, I_B, E, B, it, res = _ladm_solve(
            D,
            W,
            dct,
            lambda_flat=lambda_flat,
            lambda_dark=lambda_dark,
            darkfield=darkfield,
            tol=tol,
            max_iterations=max_iterations,
            mu=mu0,
            rho=rho,
            max_mu=mu0 * max_mu_coef,
            variant=variant,
        )
        dark = Z + D_Z * S
        if variant == "basicpy":
            shading = I_B.mean(0) - dark
            mean_S = shading.mean()
            flat = shading / mean_S
        else:
            mean_S = S.mean()
            flat = S / mean_S
        change = _relative_change(flat, last_flat)
        if darkfield and variant == "paper":
            change = max(change, _relative_change(dark, last_dark))
        baseline = (B - D_Z) * S.mean() if variant == "paper" else B * mean_S
        if result is None:
            result = BaSiCResult(flat, dark, baseline)
        result.update(flat, dark, baseline, it, res, change)
        W = _weights(I_B, E, epsilon, variant, normalize_weights)
        last_flat, last_dark = flat, dark
        if change <= reweight_tol:
            result.converged = True
            break
    return result


# -- per-frame baseline -----------------------------------------------------
def baseline_l1(
    D: Tensor,
    flat: Tensor,
    dark: Tensor,
    *,
    epsilon: float = 0.1,
    max_reweight: int = 10,
    reweight_tol: float = 1e-3,
) -> Tensor:
    """Per-frame baseline with fixed profiles (Methods Eq. 8, SI Eq. 14).

    Solves, for each frame, min_b ‖W ∘ (Iᵢ − b S − D)‖₁ with W
    reweighted by Eq. (9), W = 1 / (|I_R / I_B| + ε) with I_B = b S + D,
    starting from W = 1. For fixed W the minimiser is exact: the median
    of (Iᵢ − D) / S weighted by W S. A frame stops reweighting when its
    b changes by less than `reweight_tol` (relative), or after
    `max_reweight` steps, so frames are independent of the batching.

    Parameters
    ----------
    D : (n, h, w) Tensor
        Unsorted frames at the working size.
    flat, dark : (h, w) Tensor
        Flat field (mean 1) and dark field at the working size.
    epsilon, max_reweight, reweight_tol
        See `basic_ladm`.

    Returns
    -------
    (n,) Tensor
        bᵢ, in units of the flat-corrected signal: the corrected frame is
        (Iᵢ − D) / S − bᵢ.
    """
    n = D.shape[0]
    tiny = torch.finfo(D.dtype).tiny
    S = flat.reshape(-1).clamp_min(tiny)
    d = dark.reshape(-1).to(D)
    ratio = (D.reshape(n, -1) - d) / S
    ratio, order = ratio.sort(dim=1)
    Ss, ds = S[order], d[order]
    W = torch.ones_like(ratio)
    b = None
    active = torch.ones(n, dtype=torch.bool, device=D.device)
    for _ in range(max_reweight):
        cw = (W * Ss).cumsum(1)
        k = (cw < cw[:, -1:] / 2).sum(1, keepdim=True)
        k = k.clamp_max(ratio.shape[1] - 1)
        new = ratio.gather(1, k)[:, 0]
        if b is None:
            b = new
        else:
            change = (new - b).abs() / b.abs().clamp_min(tiny)
            b = torch.where(active, new, b)
            active = active & (change > reweight_tol)
            if not bool(active.any()):
                break
        I_B = b[:, None] * Ss + ds
        I_R = (ratio - b[:, None]) * Ss
        W = 1 / ((I_R / I_B.abs().clamp_min(tiny)).abs() + epsilon)
    return b


def baseline_basicpy(
    D: Tensor,
    flat: Tensor,
    dark: Tensor,
    *,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_iterations: int = 500,
    reweight: int = 5,
    mu_coef: float = 12.5,
    rho: float = 1.5,
    max_mu_coef: float = 1e7,
) -> Tensor:
    """Per-frame baseline as [BaSiCPy](https://github.com/peng-lab/BaSiCPy)
    2.0.0's `fit_only_baseline`.

    Fits Iᵢ = bᵢ S + D + I_{R,i} with a reweighted-L1 sparse I_R by
    inexact ALM, with bᵢ = mean(Iᵢ − I_{R,i}) − mean(D) clipped at 0. Only
    the pixels below each frame's 10 % quantile are trusted as
    background: the others start with weight ε and get weight 0 after the
    first reweighting. The ALM penalty, stopping rule and weight
    normalisation are shared by the frames of `D`.

    Parameters
    ----------
    D : (n, h, w) Tensor
        Frames at the working size.
    flat, dark : (h, w) Tensor
        Flat field (mean about 1) and dark field at the working size.
    epsilon, tol, max_iterations, mu_coef, rho, max_mu_coef
        See `basic_ladm`.
    reweight : int
        Number of reweighting steps.

    Returns
    -------
    (n,) Tensor
        bᵢ, in units of the flat-corrected signal.
    """
    n = D.shape[0]
    q = torch.quantile(D.reshape(n, -1), 0.1, dim=-1)
    Ws = (D < q[:, None, None]).to(D.dtype)
    W = torch.where(Ws > 0, 1.0, epsilon).to(D.dtype)
    mu0 = mu_coef / _spectral_norm(D)
    max_mu = mu0 * max_mu_coef
    image_norm = torch.linalg.vector_norm(D)
    dark_mean = dark.mean()
    B = torch.zeros(n, dtype=D.dtype, device=D.device)
    for _ in range(reweight):
        masked = torch.where(Ws > 0, D, torch.nan)
        B = torch.nan_to_num(torch.nanmean(masked, dim=(-2, -1)))
        E = torch.zeros_like(D)
        Y = torch.zeros_like(D)
        mu = mu0
        res = math.inf
        it = 0
        while res > tol and it < max_iterations:
            it += 1
            I_B = flat * B[:, None, None] + dark
            E = shrink(E + (D - I_B - E + Y / mu), W / mu)
            B = torch.clamp((D - E).mean((1, 2)) - dark_mean, min=0)
            Z = D - I_B - E
            Y = Y + mu * Z
            mu = torch.minimum(mu * rho, max_mu)
            res = float(torch.linalg.vector_norm(Z) / image_norm)
        # b = 0 is guarded (a division by zero in BaSiCPy)
        tiny = torch.finfo(D.dtype).tiny
        W = 1.0 / ((E / B.clamp_min(tiny)[:, None, None]).abs() + 0.1)
        W = torch.where(Ws > 0, W, W * epsilon)
        W = W / W.mean() * Ws
    return B
