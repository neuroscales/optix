"""Orthonormal DCT-II bases."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .._config import resolve_backend


def dct_basis(
    n: int,
    k: int | None = None,
    *,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Orthonormal DCT-II basis.

    B[i, j] = cⱼ cos(π (i + ½) j / n) with c₀ = √(1/n) and cⱼ = √(2/n)
    for j ≥ 1. Built in float64, then cast.

    Parameters
    ----------
    n : int
        Number of samples.
    k : int, optional
        Number of (lowest-frequency) basis functions, ≤ n.
        Default: `n`.
    dtype, device : optional
        Output dtype and device (default: from the configuration).

    Returns
    -------
    (n, k) Tensor
        Basis functions in columns, Bᵀ B = I.
    """
    k = n if k is None else k
    if not 0 < k <= n:
        raise ValueError(f"Need 0 < k <= n, got n={n}, k={k}")
    backend = resolve_backend(dtype, device)
    i = torch.arange(n, dtype=torch.float64).add_(0.5).unsqueeze(-1)
    j = torch.arange(k, dtype=torch.float64)
    basis = torch.cos(i * j * (math.pi / n))
    basis *= math.sqrt(2 / n)
    basis[:, 0] = math.sqrt(1 / n)
    return basis.to(**backend)


def dct_bending(
    shape: Sequence[int],
    k: Sequence[int],
    *,
    order: int = 2,
    spacing: float | Sequence[float] = 1.0,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Diagonal regulariser of a separable DCT-II basis.

    Energy of the `order`-th derivatives of each separable basis
    function `prod_d B_d[:, j_d]`. The DCT-II functions diagonalise the
    discrete Laplacian with reflecting (Neumann) boundaries, with
    eigenvalues λⱼ = (2 − 2 cos(π j / n)) / h² (h = spacing). The energy
    of the order-m derivatives (sum over all ordered derivative
    multi-indices) is then `(sum_d lambda_{j_d})^m`: m = 1 is the membrane
    energy and m = 2 the bending energy.

    The discrete eigenvalues are used (rather than the continuous
    (π j / (n h))²) so that, for `order=1`, the penalty equals the
    finite-difference energy exactly; both agree at low frequencies.

    Parameters
    ----------
    shape : sequence of int
        Number of samples along each dim.
    k : sequence of int
        Number of basis functions along each dim.
    order : int
        Derivative order (1 = membrane, 2 = bending).
    spacing : float or sequence of float
        Sample spacing along each dim.
    dtype, device : optional
        Output dtype and device (default: from the configuration).

    Returns
    -------
    (k1, ..., kd) Tensor
        Penalty weight of each coefficient; zero for the DC term and
        increasing with frequency.
    """
    shape, k = list(shape), list(k)
    if len(shape) != len(k):
        raise ValueError("shape and k must have the same length")
    if isinstance(spacing, int | float):
        spacing = [float(spacing)] * len(shape)
    spacing = list(spacing)
    if len(spacing) != len(shape):
        raise ValueError("spacing must have one value per dim")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    backend = resolve_backend(dtype, device)
    total = torch.zeros(k, dtype=torch.float64)
    for d, (n, kd, h) in enumerate(zip(shape, k, spacing, strict=True)):
        j = torch.arange(kd, dtype=torch.float64)
        lam = (2 - 2 * torch.cos(math.pi * j / n)) / (h * h)
        view = [1] * len(shape)
        view[d] = kd
        total = total + lam.view(view)
    return total.pow(order).to(**backend)


def _apply(x: Tensor, bases: Sequence[Tensor], transpose: bool) -> Tensor:
    nb = len(bases)
    if x.ndim < nb:
        raise ValueError(f"Need at least {nb} dims, got {x.ndim}")
    for i, basis in enumerate(bases):
        axis = x.ndim - nb + i
        mat = basis if transpose else basis.T
        x = torch.movedim(torch.movedim(x, axis, -1) @ mat, -1, axis)
    return x


def project(x: Tensor, bases: Sequence[Tensor]) -> Tensor:
    """Separable analysis over the trailing dims.

    Parameters
    ----------
    x : (..., n1, ..., nd) Tensor
        Signal.
    bases : sequence of (n_i, k_i) Tensor
        One basis per trailing dim (e.g. from
        [`dct_basis`][optix.basis.dct_basis]).

    Returns
    -------
    (..., k1, ..., kd) Tensor
        Coefficients `x x_1 B_1^T ... x_d B_d^T`.
    """
    return _apply(x, bases, transpose=True)


def gram(weight: Tensor, bases: Sequence[Tensor]) -> Tensor:
    """Weighted Gram matrix of a separable basis over the trailing dims.

    G = Bᵀ diag(w) B with B = B₁ ⊗ … ⊗ B_d, i.e.
    G[j, j′] = ∑ᵢ wᵢ ∏_d B_d[i_d, j_d] B_d[i_d, j′_d]. It is the normal
    matrix of a weighted least-squares fit of coefficients `c` to a
    signal `x`, ‖x − reconstruct(c)‖²_w, whose right-hand side is
    `project(w * x, bases)`.

    Parameters
    ----------
    weight : (..., n1, ..., nd) Tensor
        Weights wᵢ.
    bases : sequence of (n_i, k_i) Tensor
        One basis per trailing dim.

    Returns
    -------
    (..., K, K) Tensor
        With K = k1 ⋯ kd, coefficients flattened in row-major order (as
        `project(x, bases).flatten(-d)`).
    """
    nb = len(bases)
    if weight.ndim < nb:
        raise ValueError(f"Need at least {nb} dims, got {weight.ndim}")
    lead = weight.ndim - nb
    x = weight
    for basis in bases:
        n, k = basis.shape
        pairs = (basis[:, :, None] * basis[:, None, :]).reshape(n, k * k)
        x = torch.movedim(x, lead, -1) @ pairs
    ks = [b.shape[1] for b in bases]
    x = x.reshape(*x.shape[:lead], *[s for k in ks for s in (k, k)])
    perm = list(range(lead))
    perm += [lead + 2 * i for i in range(nb)]
    perm += [lead + 2 * i + 1 for i in range(nb)]
    size = math.prod(ks)
    return x.permute(perm).reshape(*x.shape[:lead], size, size)


def reconstruct(coef: Tensor, bases: Sequence[Tensor]) -> Tensor:
    """Separable synthesis over the trailing dims.

    Parameters
    ----------
    coef : (..., k1, ..., kd) Tensor
        Coefficients.
    bases : sequence of (n_i, k_i) Tensor
        One basis per trailing dim.

    Returns
    -------
    (..., n1, ..., nd) Tensor
        Signal `coef x_1 B_1 ... x_d B_d`.
    """
    return _apply(coef, bases, transpose=False)
