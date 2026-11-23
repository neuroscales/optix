"""Optical blur: a Gaussian PSF h and the blurred image h ∗ x."""

from collections.abc import Sequence

import torch
from torch import Tensor

from .._config import resolve_backend
from ..deconv._fft import PaddedConvolution

__all__ = ["gaussian_psf", "simulate_blur"]


def gaussian_psf(
    shape: Sequence[int],
    sigma: float | Sequence[float],
    *,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Sampled Gaussian point spread function.

    h(r) ∝ exp(−½ ∑ₐ (rₐ / σₐ)²), with r measured from the centre index
    `K // 2` along each axis (the convention of the deconvolvers), and
    normalised to ∑h = 1.

    Parameters
    ----------
    shape : sequence of int
        PSF shape K, e.g. (15, 15) or (Z, Y, X). Odd sizes centre the
        peak on a pixel.
    sigma : float or sequence of float
        Standard deviation σ in pixels, one value or one per axis.
        FWHM = 2√(2 ln 2) σ ≈ 2.355 σ.
    dtype, device : optional
        Working backend.

    Returns
    -------
    (*K) Tensor
        The PSF h, summing to 1.
    """
    backend = resolve_backend(dtype, device)
    shape = tuple(int(k) for k in shape)
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma)] * len(shape)
    sigma = [float(s) for s in sigma]
    if len(sigma) != len(shape) or min(sigma) <= 0:
        raise ValueError("sigma must be positive, one value or one per axis")
    r2 = torch.zeros(shape, **backend)
    for a, (k, s) in enumerate(zip(shape, sigma, strict=True)):
        r = (torch.arange(k, **backend) - k // 2) / s
        view = [1] * len(shape)
        view[a] = k
        r2 = r2 + r.reshape(view) ** 2
    h = torch.exp(-0.5 * r2)
    return h / h.sum()


def simulate_blur(x: Tensor, psf: Tensor) -> Tensor:
    """Blur a latent image with a PSF: y = h ∗ x.

    Linear convolution with zero boundary (no light from outside x),
    the same as the forward operator of the deconvolvers, so that
    deconvolving the result is consistent with its simulation.

    Parameters
    ----------
    x : (..., *spatial) Tensor
        Latent image; the trailing `psf.ndim` dims are spatial.
    psf : (*K) Tensor
        Point spread function h, centred at `K // 2`, used as given.

    Returns
    -------
    Tensor
        h ∗ x, the shape of `x`.
    """
    op = PaddedConvolution(
        psf, x.shape[-psf.ndim :], dtype=x.dtype, device=x.device
    )
    return op.forward(op.pad(x))
