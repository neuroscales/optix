"""Numerical core of BaSiC: DCT helpers, resizing and the ALM solvers.

Everything here works on a *working stack* already in memory, shaped
``(n, h, w)`` (``n`` frames at the working size). The streaming logic
(frame selection, resizing, batching) lives in :mod:`._basic`.

The solvers port BaSiCPy 2.0.0 (MIT licence, Copyright 2022 BaSiCPy
developing collaboration group; ``basicpy/_torch_routines.py`` and
``basicpy/basicpy.py``): :func:`basic_approximate` is its
``fitting_mode="approximate"``, the inexact-ALM algorithm of Peng et al.
(2017), which BaSiCPy checked against the original implementation;
:func:`baseline_basicpy` is its ``fit_only_baseline``. Deliberate departures
from that code are marked "Deviation".

Notation (BaSiCPy names in brackets):

* ``D`` (n, h, w): working stack [Im].
* ``S`` (h, w): flat-field component, with orthonormal 2-D DCT
  coefficients ``S_hat`` [S, S_hat].
* ``B`` (n,): per-frame coefficients [B].
* ``D_R`` (h, w), ``D_Z``: dark field and its constant part [D_R, D_Z].
* ``E`` (n, h, w): sparse residual [I_R]; ``Y``: Lagrange multipliers.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from ..basis import dct_basis

__all__ = [
    "BaSiCResult",
    "Dct2",
    "baseline_basicpy",
    "baseline_median",
    "basic_approximate",
    "default_lambdas",
    "resize",
    "shrink",
]


# -- elementary operators -------------------------------------------------
def shrink(x: Tensor, t: Tensor | float) -> Tensor:
    """Soft threshold ``sign(x) max(|x| - t, 0)`` (prox of ``t |.|_1``)."""
    return torch.sign(x) * torch.clamp(x.abs() - t, min=0)


class Dct2:
    """Orthonormal 2-D DCT-II of ``(..., h, w)`` images, by matrices.

    ``forward(X) = C_h X C_w^T`` with ``C`` the orthonormal DCT-II matrix
    (``torch_dct.dct_2d(norm="ortho")``); ``inverse`` is its transpose.
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
    """Bilinear resize of ``(n, h, w)`` images, as BaSiCPy does.

    ``F.interpolate(mode="bilinear", align_corners=True, antialias=True)``
    (BaSiCPy's ``_resize``; anti-aliasing only acts when shrinking).
    ``size=None`` or the same size returns ``x`` unchanged.
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


def default_lambdas(D: Tensor, dct: Dct2) -> tuple[float, float]:
    """Data-derived sparsity weights (BaSiCPy 2.0.0 defaults).

    ``meanD = mean_i D_i / mean(meanD)``, ``s = sum |dct2(meanD)|``, then
    ``lambda_S = s / 400 * 0.5`` and ``lambda_D = 0.1 lambda_S``.

    Parameters
    ----------
    D : (n, h, w) Tensor
        Working stack.
    dct : Dct2

    Returns
    -------
    lambda_flat, lambda_dark : float
    """
    mean = D.mean(0)
    mean = mean / mean.mean()
    lam = float(dct.forward(mean).abs().sum()) / 400 * 0.5
    return lam, 0.1 * lam


def _spectral_norm(D: Tensor) -> Tensor:
    """Largest singular value of ``D`` seen as ``(n, h * w)``."""
    return torch.linalg.matrix_norm(D.reshape(D.shape[0], -1), ord=2)


@dataclass
class BaSiCResult:
    """Working-size output of a BaSiC fit.

    Attributes
    ----------
    flat : (h, w) Tensor
        Flat field ``mean_i(XA_i) - dark``, normalised to mean 1.
    dark : (h, w) Tensor
        Dark field (zeros when not estimated).
    baseline : (n,) Tensor
        Per-frame baseline of the fitted frames, ``B_i mean(shading)``
        (BaSiCPy's ``baseline``).
    iterations : list of int
        ALM iterations of each reweighting step.
    residuals : list of float
        Final relative residual of each reweighting step.
    changes : list of float
        Relative L1 change of the flat field after each step (``inf``
        for the first).
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
    """BaSiCPy's reweighting score (its "dark" score also compares S)."""
    if last is None:
        return math.inf
    return float((new - last).abs().sum() / last.abs().sum())


# -- main solver: inexact ALM ("approximate") -------------------------------
def _approximate_dark(
    R: Tensor,
    B: Tensor,
    S: Tensor,
    dct: Dct2,
    t: Tensor,
    D_Z_max: Tensor,
) -> tuple[Tensor, Tensor] | None:
    """Dark-field update of ``ApproximateFit._step``.

    Splits the dark field into a constant ``D_Z``, fitted from the
    contrast between the bright (``S >= mean S``) and dark halves of the
    frames with ``B_i < 1``, and a residual ``D_R`` shrunk in the DCT and
    pixel domains with threshold ``t``. Returns ``D_R`` (including the
    ``D_Z (mean S - S)`` term) and ``D_Z``, or None when no frame has
    ``B_i < 1`` (Deviation: BaSiCPy then produces NaNs).
    """
    valid = B < 1
    k = int(valid.sum())
    if k == 0:
        return None
    n = R.shape[0]
    R2 = R.reshape(n, -1)
    Sf = S.reshape(-1)
    Smean = Sf.mean()
    Rv, Bv = R2[valid], B[valid]
    inmask, outmask = Sf >= Smean, Sf < Smean
    if inmask.any() and outmask.any():
        B1 = (Rv[:, inmask].mean(1) - Rv[:, outmask].mean(1)) / (
            R2.mean() + 1e-6
        )
    else:
        B1 = torch.zeros_like(Bv)
    t1, t2 = (Bv**2).sum(), Bv.sum()
    t3, t4 = B1.sum(), (Bv * B1).sum()
    t5 = t2 * t3 - k * t4
    if float(t5) == 0:
        D_Z = torch.zeros_like(t5)
    else:
        D_Z = (t1 * t3 - t2 * t4) / t5
    D_Z = torch.minimum(torch.clamp(D_Z, min=0), D_Z_max / Smean)
    Z = (D_Z * Smean - D_Z * Sf).reshape(S.shape)
    off = Rv.mean(0) - Bv.mean() * Sf
    off = off - off.mean()
    D_R = (off - off.mean()).reshape(S.shape) - Z
    D_R = shrink(dct.inverse(shrink(dct.forward(D_R), t)), t)
    return D_R + Z, D_Z


def _approximate_solve(
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
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float]:
    """One inexact-ALM solve for fixed weights (``ApproximateFit``).

    Returns ``S, D_R, D_Z, I_B, E, B, iterations, residual``.
    """
    n, h, w = D.shape
    ent1, ent2 = 1.0, 10.0
    D_Z_max = D.min()
    image_norm = torch.linalg.vector_norm(D)
    S_hat = dct.forward(torch.ones(h, w, dtype=D.dtype, device=D.device))
    S = dct.inverse(S_hat)
    D_R = torch.zeros(h, w, dtype=D.dtype, device=D.device)
    D_Z = torch.zeros((), dtype=D.dtype, device=D.device)
    B = D.mean((1, 2)) / D.mean()
    E = torch.zeros_like(D)
    Y = torch.zeros_like(D)
    I_B = S * B[:, None, None] + D_R
    res = math.inf
    it = 0
    while res > tol and it < max_iterations:
        it += 1
        S = dct.inverse(S_hat)
        I_B = S * B[:, None, None] + D_R
        temp = ((D - E - I_B + Y / mu) / ent1).mean(0)
        S_hat = shrink(S_hat + dct.forward(temp), lambda_flat / (ent1 * mu))
        S = dct.inverse(S_hat)
        I_B = S * B[:, None, None] + D_R
        E = shrink(E + (D - I_B - E + Y / mu) / ent1, W / (ent1 * mu))
        R = D - E
        B = torch.clamp(R.mean((1, 2)) / R.mean(), min=0)
        I_B = S * B[:, None, None] + D_R
        if darkfield:
            dark = _approximate_dark(
                R, B, S, dct, lambda_dark / (ent2 * mu), D_Z_max
            )
            if dark is not None:
                D_R, D_Z = dark
        Z = D - I_B - E
        Y = Y + mu * Z
        mu = torch.minimum(mu * rho, max_mu)
        res = float(torch.linalg.vector_norm(Z) / image_norm)
    return S, D_R, D_Z, I_B, E, B, it, res


def basic_approximate(
    D: Tensor,
    *,
    lambda_flat: float,
    lambda_dark: float,
    darkfield: bool = False,
    epsilon: float = 0.1,
    tol: float = 1e-3,
    max_iterations: int = 500,
    max_reweight: int = 10,
    reweight_tol: float = 1e-2,
    mu_coef: float = 12.5,
    rho: float = 1.5,
    max_mu_coef: float = 1e7,
) -> BaSiCResult:
    """BaSiC on a working stack: reweighted L1 + inexact ALM.

    Model, with ``D_i`` the frames at the working size::

        D_i = B_i S + D_dark + E_i,    S = idct2(S_hat)

    minimising ``lambda_S |S_hat|_1 + |W . E|_1`` (with the dark field,
    also ``lambda_D (|dct2(D_R)|_1 + |D_R|_1)``) by inexact ALM, with the
    weights reweighted as ``W = 1 / (|E_i / mean(XA_i)| + epsilon)``
    (normalised to mean 1) until the flat field changes by less than
    ``reweight_tol`` (relative L1).

    Parameters
    ----------
    D : (n, h, w) Tensor
        Working stack.
    lambda_flat, lambda_dark : float
        Sparsity weights of the DCT coefficients (see
        :func:`default_lambdas`).
    darkfield : bool
        Estimate the dark field.
    epsilon : float
        Reweighting constant.
    tol, max_iterations : float, int
        ALM stop: ``|D - XA - E|_F / |D|_F < tol``.
    max_reweight, reweight_tol : int, float
        Reweighting stop.
    mu_coef, rho, max_mu_coef : float
        Penalty schedule ``mu_0 = mu_coef / |D|_2``,
        ``mu <- min(rho mu, max_mu_coef mu_0)``.

    Returns
    -------
    BaSiCResult

    Notes
    -----
    Deviation: with the dark field, BaSiCPy 2.0.0 overwrites its
    iteration counter with the number of frames with ``B_i < 1``, so that
    ``max_iterations`` never stops its loop; here it does.
    """
    dct = Dct2(*D.shape[1:], dtype=D.dtype, device=D.device)
    mu0 = mu_coef / _spectral_norm(D)
    W = torch.ones_like(D)
    last = None
    result = None
    for _ in range(max_reweight):
        S, D_R, D_Z, I_B, E, B, it, res = _approximate_solve(
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
        )
        dark = D_R + D_Z * S
        shading = I_B.mean(0) - dark
        mean_S = shading.mean()
        flat = shading / mean_S
        change = _relative_change(flat, last)
        if result is None:
            result = BaSiCResult(flat, dark, B * mean_S)
        result.update(flat, dark, B * mean_S, it, res, change)
        # ApproximateFit.calc_weights
        E_norm = E / (I_B.mean((1, 2), keepdim=True) + 1e-6)
        W = 1 / (E_norm.abs() + epsilon)
        W = W * (W.numel() / W.sum())
        last = flat
        if change <= reweight_tol:
            result.converged = True
            break
    return result


# -- per-frame baseline -----------------------------------------------------
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
    """Per-frame baseline with fixed profiles (BaSiCPy ``fit_only_baseline``).

    Fits ``D_i = b_i S + D_dark + E_i`` with a reweighted-L1 sparse ``E``
    by inexact ALM, with ``b_i = mean(D_i - E_i) - mean(D_dark)`` clipped
    at 0. As in BaSiCPy, only the pixels below each frame's 10 % quantile
    are trusted as background: the others start with weight ``epsilon``
    and get weight 0 after the first reweighting.

    Parameters
    ----------
    D : (n, h, w) Tensor
        Frames at the working size.
    flat, dark : (h, w) Tensor
        Flat (mean about 1) and dark field at the working size.
    others
        See :func:`basic_approximate`; ``reweight`` is the number of
        reweighting steps.

    Returns
    -------
    (n,) Tensor
        ``b_i``, in units of the flat-corrected signal: the corrected
        frame is ``(y - dark) / flat - b_i``.
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
        # ApproximateFit.calc_weights_baseline, times the mask
        # Deviation: b = 0 is guarded (BaSiCPy divides by zero)
        tiny = torch.finfo(D.dtype).tiny
        W = 1.0 / ((E / B.clamp_min(tiny)[:, None, None]).abs() + 0.1)
        W = torch.where(Ws > 0, W, W * epsilon)
        W = W / W.mean() * Ws
    return B


def baseline_median(D: Tensor, flat: Tensor, dark: Tensor) -> Tensor:
    """Cheap per-frame baseline: ``argmin_b |D_i - b S - dark|_1``.

    The exact minimiser is the median of ``(D_i - dark) / S`` weighted by
    ``S``: the L1 baseline fit with uniform weights (no reweighting, no
    ALM). Clipped at 0.

    Parameters
    ----------
    D : (n, h, w) Tensor
    flat, dark : (h, w) Tensor

    Returns
    -------
    (n,) Tensor
    """
    n = D.shape[0]
    S = flat.reshape(-1).clamp_min(torch.finfo(flat.dtype).tiny)
    ratio = (D.reshape(n, -1) - dark.reshape(-1)) / S
    ratio, order = ratio.sort(dim=1)
    cw = S[order].cumsum(1)
    k = (cw < cw[:, -1:] / 2).sum(1, keepdim=True)
    k = k.clamp_max(ratio.shape[1] - 1)
    return ratio.gather(1, k)[:, 0].clamp_min(0)
