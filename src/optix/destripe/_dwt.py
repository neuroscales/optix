"""Decimated discrete wavelet transform in torch (symmetric extension)."""

import functools
import math
import re

import torch
import torch.nn.functional as F
from torch import Tensor

from ._daubechies import DAUBECHIES

__all__ = [
    "Coefficients",
    "Wavelet",
    "dwt",
    "dwt2",
    "idwt",
    "idwt2",
    "wavedec2",
    "get_wavelet",
    "max_level",
    "waverec2",
]

Details = tuple[Tensor, Tensor, Tensor]
Coefficients = list[Tensor | Details]


class Wavelet:
    """Orthogonal wavelet filter bank.

    Attributes
    ----------
    name : str
        Name, e.g. `"db4"`.
    rec_lo, rec_hi, dec_lo, dec_hi : tuple of float
        Reconstruction and decomposition low- and high-pass filters,
        related as quadrature mirror filters:

            dec_lo[k] = rec_lo[F−1−k],
            rec_hi[k] = (−1)ᵏ·dec_lo[k],
            dec_hi[k] = rec_hi[F−1−k],

        with F the filter length.
    """

    def __init__(self, name: str, rec_lo: tuple[float, ...]) -> None:
        """Build the filter bank from its reconstruction low-pass filter.

        Parameters
        ----------
        name : str
            Name of the wavelet.
        rec_lo : tuple of float
            Reconstruction low-pass (scaling) filter, ∑ h = √2.
        """
        self.name = name
        self.rec_lo = tuple(float(v) for v in rec_lo)
        self.dec_lo = self.rec_lo[::-1]
        self.rec_hi = tuple((-1) ** k * v for k, v in enumerate(self.dec_lo))
        self.dec_hi = self.rec_hi[::-1]

    def __len__(self) -> int:
        return len(self.rec_lo)

    def __repr__(self) -> str:
        return f"Wavelet({self.name!r})"


@functools.lru_cache
def get_wavelet(name: "str | Wavelet") -> Wavelet:
    """Daubechies filter bank by name.

    Parameters
    ----------
    name : str or Wavelet
        `"dbN"` with N = 1, ..., 45 (N vanishing moments, filter length
        2N), or `"haar"` (= `"db1"`).

    Returns
    -------
    Wavelet
    """
    if isinstance(name, Wavelet):
        return name
    key = str(name).lower()
    if key == "haar":
        key = "db1"
    match = re.fullmatch(r"db(\d+)", key)
    if match is None or not 1 <= int(match[1]) <= len(DAUBECHIES):
        raise ValueError(
            f"Unknown wavelet {name!r}: use 'haar' or 'dbN' with "
            f"1 <= N <= {len(DAUBECHIES)}"
        )
    return Wavelet(key, DAUBECHIES[int(match[1]) - 1])


def _filter(values: tuple[float, ...], like: Tensor) -> Tensor:
    return torch.tensor(values, dtype=like.dtype, device=like.device).view(
        1, 1, -1
    )


def _symmetric_index(n: int, left: int, right: int, device: object) -> Tensor:
    """Indices of a half-sample symmetric extension (… x₁ x₀ | x₀ x₁ …)."""
    i = torch.arange(-left, n + right, device=device)
    i = i % (2 * n)
    return torch.where(i >= n, 2 * n - 1 - i, i)


def _to_last(x: Tensor, dim: int) -> tuple[Tensor, tuple[int, ...]]:
    x = x.movedim(dim, -1)
    lead = tuple(x.shape[:-1])
    return x.reshape(-1, 1, x.shape[-1]), lead


def _from_last(x: Tensor, lead: tuple[int, ...], dim: int) -> Tensor:
    return x.reshape(*lead, x.shape[-1]).movedim(-1, dim)


def dwt(
    x: Tensor, wavelet: "str | Wavelet", dim: int = -1
) -> tuple[Tensor, Tensor]:
    """Single-level 1-D DWT along `dim`.

    With symmetric (half-sample mirror) extension of the signal, and the
    filters h (low pass) and g (high pass) of length F:

        a[k] = ∑ⱼ dec_lo[j]·x[2k + 1 − j],
        d[k] = ∑ⱼ dec_hi[j]·x[2k + 1 − j],

    for k = 0, …, ⌊(N + F − 1)/2⌋ − 1. The convention (and the coefficients)
    are those of Matlab's `dwt` in `"sym"` mode and PyWavelets' `dwt` in
    `"symmetric"` mode.

    Parameters
    ----------
    x : Tensor
        Signal(s).
    wavelet : str or Wavelet
        See [`get_wavelet`][optix.destripe.utils.get_wavelet].
    dim : int
        Dimension to transform.

    Returns
    -------
    a, d : Tensor
        Approximation and detail coefficients.
    """
    w = get_wavelet(wavelet)
    flen = len(w)
    flat, lead = _to_last(x, dim)
    n = flat.shape[-1]
    index = _symmetric_index(n, flen - 2, flen - 1, flat.device)
    padded = flat[..., index]
    lo = torch.flip(_filter(w.dec_lo, flat), (-1,))
    hi = torch.flip(_filter(w.dec_hi, flat), (-1,))
    a = F.conv1d(padded, lo, stride=2)
    d = F.conv1d(padded, hi, stride=2)
    return _from_last(a, lead, dim), _from_last(d, lead, dim)


def idwt(
    a: Tensor,
    d: Tensor,
    wavelet: "str | Wavelet",
    dim: int = -1,
) -> Tensor:
    """Single-level 1-D inverse DWT along `dim`.

    Inverse of [`dwt`][optix.destripe.utils.dwt]:

        x[n] = ∑ₖ rec_lo[n + F − 2 − 2k]·a[k] + rec_hi[n + F − 2 − 2k]·d[k],

    for n = 0, …, 2K − F + 1 with K coefficients. The output has the input
    length of [`dwt`][optix.destripe.utils.dwt] when it was even, one more
    sample when it was odd (as Matlab's `idwt` and PyWavelets' `idwt`).

    Parameters
    ----------
    a, d : Tensor
        Approximation and detail coefficients, of identical shapes.
    wavelet : str or Wavelet
    dim : int

    Returns
    -------
    Tensor
    """
    w = get_wavelet(wavelet)
    flen = len(w)
    fa, lead = _to_last(a, dim)
    fd, _ = _to_last(d, dim)
    lo, hi = _filter(w.rec_lo, fa), _filter(w.rec_hi, fa)
    full = F.conv_transpose1d(fa, lo, stride=2) + F.conv_transpose1d(
        fd, hi, stride=2
    )
    n = 2 * fa.shape[-1] - flen + 2
    out = full[..., flen - 2 : flen - 2 + n]
    return _from_last(out, lead, dim)


def dwt2(
    x: Tensor, wavelet: "str | Wavelet", dims: tuple[int, int] = (-2, -1)
) -> tuple[Tensor, Details]:
    """Single-level 2-D DWT over `dims`.

    Parameters
    ----------
    x : Tensor
    wavelet : str or Wavelet
    dims : (int, int)
        Dimensions (first, second).

    Returns
    -------
    a : Tensor
        Approximation (low pass along both dims).
    (h, v, d) : tuple of Tensor
        Detail bands: high pass along the first dim and low pass along the
        second (`h`, horizontal details), low pass along the first and
        high pass along the second (`v`, vertical details), high pass along
        both (`d`, diagonal details). For an image `(y, x)`, the vertical
        details hold vertical structures such as stripes along `y`.
        The ordering is that of Matlab's `dwt2` and PyWavelets' `dwt2`.
    """
    d0, d1 = dims
    lo1, hi1 = dwt(x, wavelet, d1)
    a, h = dwt(lo1, wavelet, d0)
    v, d = dwt(hi1, wavelet, d0)
    return a, (h, v, d)


def idwt2(
    a: Tensor,
    details: Details,
    wavelet: "str | Wavelet",
    dims: tuple[int, int] = (-2, -1),
) -> Tensor:
    """Single-level 2-D inverse DWT; inverse of
    [`dwt2`][optix.destripe.utils.dwt2].

    Parameters
    ----------
    a : Tensor
    details : (Tensor, Tensor, Tensor)
        Horizontal, vertical and diagonal details.
    wavelet : str or Wavelet
    dims : (int, int)

    Returns
    -------
    Tensor
    """
    d0, d1 = dims
    h, v, d = details
    lo1 = idwt(a, h, wavelet, d0)
    hi1 = idwt(v, d, wavelet, d0)
    return idwt(lo1, hi1, wavelet, d1)


def wavedec2(
    x: Tensor,
    wavelet: "str | Wavelet",
    level: int,
    dims: tuple[int, int] = (-2, -1),
) -> Coefficients:
    """Multilevel 2-D DWT (Mallat's pyramid).

    Parameters
    ----------
    x : Tensor
    wavelet : str or Wavelet
    level : int
        Number of decompositions L ≥ 0.
    dims : (int, int)

    Returns
    -------
    list
        `[a_L, (h_L, v_L, d_L), …, (h_1, v_1, d_1)]`, coarsest first
        (PyWavelets' `wavedec2` ordering); level 1 is the finest.
    """
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")
    w = get_wavelet(wavelet)
    details = []
    a = x
    for _ in range(level):
        a, det = dwt2(a, w, dims)
        details.append(det)
    return [a, *details[::-1]]


def waverec2(
    coeffs: Coefficients,
    wavelet: "str | Wavelet",
    dims: tuple[int, int] = (-2, -1),
    shape: tuple[int, int] | None = None,
) -> Tensor:
    """Multilevel 2-D inverse DWT; inverse of
    [`wavedec2`][optix.destripe.utils.wavedec2].

    Before each level, the approximation is cropped to the shape of the
    details of that level (it is one sample longer along a dimension whose
    length was odd at decomposition).

    Parameters
    ----------
    coeffs : list
        `[a_L, (h_L, v_L, d_L), …, (h_1, v_1, d_1)]`.
    wavelet : str or Wavelet
    dims : (int, int)
    shape : (int, int), optional
        Shape of the original signal along `dims`, to crop the result to.

    Returns
    -------
    Tensor
    """
    w = get_wavelet(wavelet)
    d0, d1 = dims
    a = coeffs[0]
    for det in coeffs[1:]:
        n0, n1 = det[0].shape[d0], det[0].shape[d1]
        a = a.narrow(d0, 0, n0).narrow(d1, 0, n1)
        a = idwt2(a, det, w, dims)
    if shape is not None:
        a = a.narrow(d0, 0, shape[0]).narrow(d1, 0, shape[1])
    return a


def max_level(n: int, wavelet: "str | Wavelet") -> int:
    """Largest level at which the coefficients are at least one filter long.

    ⌊log₂(n/(F − 1))⌋ for a signal of length n and a filter of length F
    (the bound used by PyWavelets' `dwt_max_level` and Matlab's
    `wmaxlev`). Deeper levels are valid, but dominated by the boundary
    extension.
    """
    w = get_wavelet(wavelet)
    if len(w) < 2:
        return 0
    return max(0, int(math.floor(math.log2(n / (len(w) - 1)))))
