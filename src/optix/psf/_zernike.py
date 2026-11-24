"""Zernike circle polynomials (Noll ordering) and phase unwrapping."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = ["noll_index", "unwrap_phase", "zernike", "zernike_fit"]


def noll_index(j: int) -> tuple[int, int]:
    """Radial order and azimuthal frequency of the Noll index `j`.

    Noll (1976) numbers the Zernike polynomials row by row (radial order
    n = 0, 1, 2, ...), by increasing |m| within a row; for m ≠ 0, the
    even index of each pair is the cosine term (m > 0) and the odd index
    the sine term (m < 0).

    Parameters
    ----------
    j : int
        Noll index, j ≥ 1.

    Returns
    -------
    n, m : int
        Radial order and signed azimuthal frequency (|m| ≤ n, n − |m|
        even).

    Examples
    --------
    >>> [noll_index(j) for j in (1, 2, 3, 4, 11)]
    [(0, 0), (1, 1), (1, -1), (2, 0), (4, 0)]
    """
    j = int(j)
    if j < 1:
        raise ValueError(f"Noll indices start at 1, got {j}")
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    p = j - n * (n + 1) // 2 - 1
    m = 2 * ((p + 1) // 2) if n % 2 == 0 else 2 * (p // 2) + 1
    if m == 0:
        return n, 0
    return n, m if j % 2 == 0 else -m


def _radial(rho: Tensor, nmax: int) -> dict[tuple[int, int], Tensor]:
    """Radial polynomials R_n^m(ρ), n ≤ nmax, by a three-term recurrence.

    R_n^m = ρ (R_{n−1}^{|m−1|} + R_{n−1}^{m+1}) − R_{n−2}^m, R_0^0 = 1,
    which avoids the cancellation of the explicit factorial sum.
    """
    table: dict[tuple[int, int], Tensor] = {(0, 0): torch.ones_like(rho)}
    zero = torch.zeros_like(rho)
    for n in range(1, nmax + 1):
        for m in range(n % 2, n + 1, 2):
            a = table[(n - 1, abs(m - 1))]
            b = table.get((n - 1, m + 1), zero)
            c = table.get((n - 2, m), zero)
            table[(n, m)] = rho * (a + b) - c
    return table


def zernike(j: int | Sequence[int], rho: Tensor, theta: Tensor) -> Tensor:
    """Zernike circle polynomials with Noll's indexing and normalisation.

        Z_j(ρ, θ) = √(n + 1) R_n^0(ρ)                 (m = 0),
        Z_j(ρ, θ) = √(2(n + 1)) R_n^|m|(ρ) cos(m θ)    (m > 0, j even),
        Z_j(ρ, θ) = √(2(n + 1)) R_n^|m|(ρ) sin(|m| θ)  (m < 0, j odd),

    with (n, m) = [`noll_index`][optix.psf.utils.noll_index](j). They are
    orthonormal on the unit disk: (1/π) ∫∫ Z_i Z_j ρ dρ dθ = δ_ij, so a
    coefficient is the root-mean-square contribution of its term. The
    radial polynomials are computed by a three-term recurrence, stable in
    single precision.

    !!! references "References"
        1. Noll RJ. *Zernike polynomials and atmospheric turbulence.*
           J Opt Soc Am **66**(3):207–211, 1976.
           [doi:10.1364/JOSA.66.000207](https://doi.org/10.1364/JOSA.66.000207)

        ??? quote "BibTeX"
            ```bibtex
            @article{noll1976zernike,
              author  = {Noll, R. J.},
              title   = {Zernike polynomials and atmospheric turbulence},
              journal = {Journal of the Optical Society of America},
              volume  = {66}, number = {3}, pages = {207--211},
              year    = {1976}, doi = {10.1364/JOSA.66.000207}
            }
            ```

    Parameters
    ----------
    j : int or sequence of int
        Noll index or indices (≥ 1).
    rho, theta : Tensor
        Normalised radius (1 on the pupil edge) and azimuth, broadcast
        together. Values at ρ > 1 are the polynomials' continuation.

    Returns
    -------
    Tensor
        `(*shape)` for an int `j`, else `(*shape, len(j))`.
    """
    single = isinstance(j, int)
    js = [j] if single else [int(i) for i in j]
    rho, theta = torch.broadcast_tensors(rho, theta)
    nm = [noll_index(i) for i in js]
    table = _radial(rho, max(n for n, _ in nm))
    out = []
    for n, m in nm:
        r = table[(n, abs(m))]
        if m == 0:
            out.append(math.sqrt(n + 1) * r)
        elif m > 0:
            out.append(math.sqrt(2 * (n + 1)) * r * torch.cos(m * theta))
        else:
            out.append(math.sqrt(2 * (n + 1)) * r * torch.sin(-m * theta))
    return out[0] if single else torch.stack(out, -1)


def zernike_fit(
    values: Tensor,
    rho: Tensor,
    theta: Tensor,
    *,
    terms: int = 79,
    support: Tensor | None = None,
) -> Tensor:
    """Linear least-squares Zernike coefficients of maps on the unit disk.

        c = argmin ∑_{ρ ≤ 1} (v(ρ, θ) − ∑_{j=1}^{J} c_j Z_j(ρ, θ))²,

    unweighted over the pixels of the disk (Hanser et al. 2004 fit the
    magnitude and the phase of pupil functions this way, with J = 79:
    all polynomials up to radial order 11 plus the radial one of order
    12, i.e. Noll indices 1 … 79).

    Parameters
    ----------
    values : Tensor
        `(*batch, H, W)` maps.
    rho, theta : Tensor
        `(H, W)` normalised radius and azimuth of the pixels.
    terms : int
        Number J of polynomials, Noll indices 1 … J. Default 79.
    support : Tensor, optional
        `(H, W)` boolean mask of the pixels to fit (default ρ ≤ 1).

    Returns
    -------
    Tensor
        `(*batch, J)` coefficients.

    Raises
    ------
    ValueError
        If the support has fewer pixels than `terms`.
    """
    if support is None:
        support = rho <= 1
    count = int(support.sum())
    if count < terms:
        raise ValueError(
            f"{count} pupil pixels cannot determine {terms} Zernike terms: "
            f"use a larger lateral field of view"
        )
    basis = zernike(list(range(1, terms + 1)), rho[support], theta[support])
    batch = values.shape[:-2]
    rhs = values[..., support].reshape(-1, count).T
    sol = torch.linalg.lstsq(basis.to(rhs.dtype), rhs).solution
    return sol.T.reshape(*batch, terms)


def unwrap_phase(phase: Tensor, support: Tensor) -> Tensor:
    """Unwrap a phase map on a disk by linear extrapolation along paths.

    Each pixel is moved by a multiple of 2π to the value closest to the
    linear extrapolation 2φ₁ − φ₂ of the two previous pixels of its
    path (the previous pixel alone for the first step). The paths start
    at the centre pixel `(H // 2, W // 2)`, run along the centre column,
    then along each row outwards from that column. The phase is set to
    0 outside the support.

    !!! note "Interpretation"
        Hanser et al. (2004) unwrap the phase "using a simple linear
        extrapolation method" before the Zernike fit; the path is not
        specified in the paper.

    Parameters
    ----------
    phase : Tensor
        `(*batch, H, W)` wrapped phase, in radians.
    support : Tensor
        `(H, W)` boolean mask: convex, holding the centre pixel, and
        with every row meeting the centre column (a disk centred at
        `(H // 2, W // 2)`).

    Returns
    -------
    Tensor
        Unwrapped phase, same shape.
    """
    h, w = phase.shape[-2:]
    cy, cx = h // 2, w // 2
    if not bool(support[cy, cx]):
        raise ValueError("the support must hold the centre pixel")
    two_pi = 2 * math.pi
    out = phase.clone()

    def nearest(value: Tensor, guess: Tensor) -> Tensor:
        return value + two_pi * torch.round((guess - value) / two_pi)

    for step in (1, -1):
        i, k = cy + step, 1
        while 0 <= i < h and bool(support[i, cx]):
            prev = out[..., i - step, cx]
            guess = prev if k == 1 else 2 * prev - out[..., i - 2 * step, cx]
            out[..., i, cx] = nearest(phase[..., i, cx], guess)
            i, k = i + step, k + 1
    for step in (1, -1):
        c, k = cx + step, 1
        while 0 <= c < w:
            inside = support[:, c]
            if not bool(inside.any()):
                break
            prev = out[..., c - step]
            guess = prev if k == 1 else 2 * prev - out[..., c - 2 * step]
            new = nearest(phase[..., c], guess)
            out[..., c] = torch.where(inside, new, out[..., c])
            c, k = c + step, k + 1
    return torch.where(support, out, torch.zeros_like(out))
