"""Small tensor operations on PSFs and bead crops."""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "fourier_shift",
    "fwhm",
    "gaussian_filter",
    "linear_shift",
    "local_maxima",
    "tukey_window",
]


def _per_axis(value: float | Sequence[float], ndim: int, name: str) -> list:
    if isinstance(value, int | float):
        return [value] * ndim
    value = list(value)
    if len(value) != ndim:
        raise ValueError(f"{name} needs {ndim} entries, got {len(value)}")
    return value


def gaussian_filter(
    x: Tensor, sigma: float | Sequence[float], *, truncate: float = 3.0
) -> Tensor:
    """Separable Gaussian smoothing of every dim of `x`.

    Parameters
    ----------
    x : Tensor
        Array to smooth (all dims are filtered).
    sigma : float or sequence of float
        Standard deviation in voxels, per dim; 0 leaves a dim unfiltered.
    truncate : float
        The kernel is cut at ⌈truncate · σ⌉ voxels from its centre.

    Returns
    -------
    Tensor
        Smoothed array, same shape; values past the edges are taken
        equal to the edge value (replicate boundary).
    """
    sigma = _per_axis(sigma, x.ndim, "sigma")
    out = x
    for dim, s in enumerate(sigma):
        if s <= 0:
            continue
        radius = max(1, math.ceil(truncate * s))
        t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
        kernel = torch.exp(-0.5 * (t / s) ** 2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        moved = out.movedim(dim, -1)
        shape = moved.shape
        flat = moved.reshape(-1, 1, shape[-1])
        flat = F.pad(flat, (radius, radius), mode="replicate")
        flat = F.conv1d(flat, kernel)
        out = flat.reshape(shape).movedim(-1, dim)
    return out


def local_maxima(x: Tensor, half_width: int, threshold: float) -> Tensor:
    """Voxels that are the maximum of their window and above a threshold.

    Parameters
    ----------
    x : Tensor
        1-D, 2-D or 3-D array.
    half_width : int
        The window spans `2 · half_width + 1` voxels along each dim.
    threshold : float
        Maxima must be strictly above it.

    Returns
    -------
    (M, ndim) Tensor
        Integer positions of the maxima.
    """
    if not 1 <= x.ndim <= 3:
        raise ValueError(f"Local maxima need 1 to 3 dims, got {x.ndim}")
    pool = (F.max_pool1d, F.max_pool2d, F.max_pool3d)[x.ndim - 1]
    size = 2 * half_width + 1
    pooled = pool(x[None, None], size, stride=1, padding=half_width)[0, 0]
    return torch.nonzero((x == pooled) & (x > threshold))


def fourier_shift(x: Tensor, shift: Tensor) -> Tensor:
    """Shift a batch of arrays by sub-voxel amounts, with the DFT.

        out(r) = in(r − d),   F[out](k) = F[in](k) · exp(−2πi ∑ᵢ kᵢ dᵢ / Kᵢ),

    with F the discrete Fourier transform.

    The shift is circular: content leaving one side re-enters on the
    other, so the arrays should fall to zero at their edges.

    Parameters
    ----------
    x : (M, *K) Tensor
        Arrays to shift.
    shift : (M, D) Tensor
        Shift d of each array, in voxels, along its D trailing dims.

    Returns
    -------
    (M, *K) Tensor
        Real part of the shifted arrays.
    """
    ndim = shift.shape[-1]
    dims = tuple(range(-ndim, 0))
    spectrum = torch.fft.fftn(x, dim=dims)
    phase = x.new_zeros(x.shape)
    for i, dim in enumerate(dims):
        n = x.shape[dim]
        freq = torch.fft.fftfreq(n, dtype=x.dtype, device=x.device)
        freq_shape = [1] * x.ndim
        freq_shape[dim] = n
        shift_shape = [1] * x.ndim
        shift_shape[0] = -1
        phase = phase + freq.view(freq_shape) * shift[:, i].view(shift_shape)
    spectrum = spectrum * torch.polar(
        torch.ones_like(phase), -2 * math.pi * phase
    )
    return torch.fft.ifftn(spectrum, dim=dims).real


def linear_shift(x: Tensor, shift: Tensor) -> Tensor:
    """Shift a batch of arrays by sub-voxel amounts, by linear interpolation.

        out(r) = in(r − d),

    interpolated separably along each dim; values from outside the
    array are zero.

    Parameters
    ----------
    x : (M, *K) Tensor
        Arrays to shift.
    shift : (M, D) Tensor
        Shift d of each array, in voxels, along its D trailing dims.

    Returns
    -------
    (M, *K) Tensor
    """
    ndim = shift.shape[-1]
    out = x
    for i in range(ndim):
        dim = x.ndim - ndim + i
        moved = out.movedim(dim, -1)
        shape = moved.shape
        n = shape[-1]
        flat = moved.reshape(shape[0], -1, n)
        d = shift[:, i]
        whole = torch.floor(d)
        frac = (d - whole).view(-1, 1, 1)
        grid = torch.arange(n, device=x.device).view(1, 1, n)
        lo = grid - whole.long().view(-1, 1, 1)  # in[r - floor(d)]
        hi = lo - 1  # in[r - floor(d) - 1]
        padded = F.pad(flat, (1, 1))
        value = (1 - frac) * _take(padded, lo, n) + frac * _take(padded, hi, n)
        out = value.reshape(shape).movedim(-1, dim)
    return out


def _take(padded: Tensor, index: Tensor, n: int) -> Tensor:
    """`padded[..., index + 1]`, zero (the pad) outside [0, n)."""
    m, k = padded.shape[0], padded.shape[1]
    index = (index.clamp(-1, n) + 1).expand(m, k, n)
    return torch.gather(padded, -1, index)


def tukey_window(n: int, fraction: float, **backend: object) -> Tensor:
    """Tukey (tapered cosine) window of length `n`.

        w(k) = ½ (1 + cos(π (k / a − 1)))   for k < a,
        w(k) = 1                             for a ≤ k ≤ n − 1 − a,

    mirrored at the other end, with a = `fraction` · (n − 1) / 2.

    Parameters
    ----------
    n : int
        Length.
    fraction : float
        Fraction of the window inside the two cosine tapers, in [0, 1]
        (0: rectangular, 1: Hann).

    Returns
    -------
    (n,) Tensor
    """
    if not 0 <= fraction <= 1:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    k = torch.arange(n, **backend)
    k = torch.minimum(k, n - 1 - k)
    a = fraction * (n - 1) / 2
    if a <= 0:
        return torch.ones(n, **backend)
    w = 0.5 * (1 + torch.cos(math.pi * (k / a - 1)))
    return torch.where(k < a, w, torch.ones_like(w))


def fwhm(psf: Tensor, center: Sequence[int] | None = None) -> Tensor:
    """Full width at half maximum along each axis.

    The width is measured on the line through `center`, parallel to the
    axis, between the first half-maximum crossings on either side of the
    centre, located by linear interpolation between voxels.

    Parameters
    ----------
    psf : Tensor
        Point spread function (any number of dims).
    center : sequence of int, optional
        Voxel through which the profiles pass (default: the maximum).

    Returns
    -------
    (ndim,) Tensor
        Width in voxels along each dim; NaN where the profile does not
        fall below half its central value on both sides.
    """
    if center is None:
        flat = int(torch.argmax(psf))
        center = []
        for n in reversed(psf.shape):
            center.append(flat % n)
            flat //= n
        center = center[::-1]
    center = [int(c) for c in center]
    out = []
    for dim in range(psf.ndim):
        index = list(center)
        index[dim] = slice(None)
        profile = psf[tuple(index)].double().cpu()
        c = center[dim]
        half = 0.5 * float(profile[c])
        out.append(
            _half_width(profile, c, half, -1)
            + _half_width(profile, c, half, 1)
        )
    return torch.tensor(out, dtype=psf.dtype, device=psf.device)


def _half_width(profile: Tensor, c: int, half: float, step: int) -> float:
    k = c
    while 0 <= k + step < len(profile):
        nxt = k + step
        a, b = float(profile[k]), float(profile[nxt])
        if b <= half:
            return abs(k - c) + (a - half) / (a - b)
        k = nxt
    return math.nan
