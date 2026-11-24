"""Separable transforms of the block-matching filters (BM3D, BM4D)."""

import math

import torch
from torch import Tensor

from ..basis import dct_basis

__all__ = [
    "bior15_matrix",
    "dct_matrix",
    "haar_matrix",
    "kaiser_window",
    "transform_matrix",
]

# analysis low-pass filter of the bior1.5 (Cohen–Daubechies–Feauveau)
# spline wavelet, taps j = −4, …, 5 around sample 2k
_BIOR15_LO = (3.0, -3.0, -22.0, 22.0, 128.0, 128.0, 22.0, -22.0, -3.0, 3.0)


def _check_dyadic(n: int, name: str) -> None:
    if n < 1 or n & (n - 1):
        raise ValueError(f"{name} needs a power-of-two size, got {n}")


def _bior15_level(n: int) -> Tensor:
    """One periodised analysis level on `n` samples (float64)."""
    h = [c * math.sqrt(2) / 256 for c in _BIOR15_LO]
    t = torch.zeros(n, n, dtype=torch.float64)
    half = n // 2
    r = 1 / math.sqrt(2)
    for k in range(half):
        for j, c in zip(range(-4, 6), h, strict=True):
            t[k, (2 * k + j) % n] += c
        t[half + k, 2 * k + 1] += r
        t[half + k, 2 * k] -= r
    return t


def bior15_matrix(
    n: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Normalised bior1.5 full dyadic wavelet transform of `n` samples.

    Analysis filters of the biorthogonal spline wavelet bior1.5: the
    low-pass filter (√2/256)·[3, −3, −22, 22, 128, 128, 22, −22, −3, 3]
    (taps −4, …, 5 around sample 2k) and the Haar high-pass filter
    (x₂ₖ₊₁ − x₂ₖ)/√2, with periodic extension, iterated on the
    approximation down to one coefficient. Rows are the analysis
    functions, coarsest approximation (a constant, the DC) first, then
    the details from coarse to fine. Each row is scaled to unit ℓ²
    norm, so that white noise of variance σ² gives coefficients of
    variance σ².

    !!! note
        The filters are those of the bior1.5 wavelet named in Dabov et
        al. (2007, Table I); the periodic extension, the alignment of
        the filters and the row normalisation are not specified in the
        paper. With n = 4 the transform is the orthonormal Haar
        transform, as stated by Maggioni et al. (2013, Section III-A).

    Parameters
    ----------
    n : int
        Size, a power of two.
    dtype, device : optional

    Returns
    -------
    (n, n) Tensor
        Forward transform T (coefficients = T·x); its inverse is
        `torch.linalg.inv(T)`.
    """
    _check_dyadic(n, "bior1.5")
    t = torch.eye(n, dtype=torch.float64)
    m = n
    while m > 1:
        step = torch.eye(n, dtype=torch.float64)
        step[:m, :m] = _bior15_level(m)
        t = step @ t
        m //= 2
    t = t / t.norm(dim=1, keepdim=True)
    return t.to(dtype=dtype or torch.get_default_dtype(), device=device)


def haar_matrix(
    n: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Orthonormal full dyadic Haar transform of `n` samples.

    Hₙ = [H_{n/2} ⊗ (1, 1)/√2; I_{n/2} ⊗ (1, −1)/√2], H₁ = 1: row 0 is
    the DC 1/√n, the others are Haar wavelets from coarse to fine.

    Parameters
    ----------
    n : int
        Size, a power of two.
    dtype, device : optional

    Returns
    -------
    (n, n) Tensor
        Hᵀ = H⁻¹.
    """
    _check_dyadic(n, "Haar")
    h = torch.ones(1, 1, dtype=torch.float64)
    r = 1 / math.sqrt(2)
    while h.shape[0] < n:
        m = h.shape[0]
        lo = torch.kron(h, torch.tensor([[r, r]], dtype=torch.float64))
        hi = torch.kron(
            torch.eye(m, dtype=torch.float64),
            torch.tensor([[r, -r]], dtype=torch.float64),
        )
        h = torch.cat([lo, hi])
    return h.to(dtype=dtype or torch.get_default_dtype(), device=device)


def dct_matrix(
    n: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Orthonormal DCT-II of `n` samples, as rows (DC first).

    The transpose of [`dct_basis`][optix.basis.dct_basis].

    Parameters
    ----------
    n : int
    dtype, device : optional

    Returns
    -------
    (n, n) Tensor
    """
    basis = dct_basis(n, dtype=torch.float64).T.contiguous()
    return basis.to(dtype=dtype or torch.get_default_dtype(), device=device)


def transform_matrix(
    name: str,
    n: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """1-D transform `"bior1.5"`, `"dct"` or `"haar"` of `n` samples.

    Parameters
    ----------
    name : {"bior1.5", "dct", "haar"}
    n : int
    dtype, device : optional

    Returns
    -------
    (n, n) Tensor
        Rows are the analysis functions, DC first.
    """
    makers = {"bior1.5": bior15_matrix, "dct": dct_matrix, "haar": haar_matrix}
    if name not in makers:
        raise ValueError(f"Unknown transform {name!r}; one of {list(makers)}")
    return makers[name](n, dtype=dtype, device=device)


def kaiser_window(
    n: int,
    beta: float,
    ndim: int = 2,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Separable Kaiser window of `n` samples per axis, flattened.

    w(i₁, …, i_d) = ∏ₖ κ_β(iₖ), with κ_β the symmetric 1-D Kaiser
    window I₀(β·√(1 − (2i/(n − 1) − 1)²)) / I₀(β).

    Parameters
    ----------
    n : int
    beta : float
    ndim : int
    dtype, device : optional

    Returns
    -------
    (n**ndim,) Tensor
        In row-major order of the block.
    """
    w1 = torch.kaiser_window(
        n, periodic=False, beta=float(beta), dtype=torch.float64
    )
    w = w1
    for _ in range(ndim - 1):
        w = torch.outer(w, w1).reshape(-1)
    return w.reshape(-1).to(
        dtype=dtype or torch.get_default_dtype(), device=device
    )
