"""Signal model: per-pixel classes with Gamma-distributed intensities."""

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from .._config import resolve_backend

__all__ = ["GammaMixtureSample", "simulate_gamma_mixture"]


class GammaMixtureSample(NamedTuple):
    """Draw of [`simulate_gamma_mixture`][optix.models.simulate_gamma_mixture].

    Attributes
    ----------
    signal : Tensor
        `shape`, signal u ≥ 0 (0 in the background).
    labels : Tensor
        `shape`, int64 class index z; the background is `K`, the last.
    proportions : Tensor
        `(*sample, K + 1)` class proportions π of each picture.
    """

    signal: Tensor
    labels: Tensor
    proportions: Tensor


def simulate_gamma_mixture(
    shape: Sequence[int],
    means: Sequence[float],
    shapes: Sequence[float],
    *,
    concentration: float | Sequence[float] = 1.0,
    ndim: int = 2,
    seed: int | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> GammaMixtureSample:
    """Draw a signal from a mixture of Gamma classes and a background.

    For every picture k (the leading, non-spatial axes of `shape`) and
    pixel i:

        πₖ ~ Dirichlet(α),  zᵢₖ ~ Categorical(πₖ)
        uᵢₖ | zᵢₖ = background ~ δ(0)
        uᵢₖ | zᵢₖ = c ~ Gamma(νc, rate νc / μc)

    Parameters
    ----------
    shape : sequence of int
        `(*sample, *spatial)` shape of the signal.
    means, shapes : sequence of float
        Mean μ and shape ν of each of the K tissue classes.
    concentration : float or sequence of float
        Dirichlet concentration α of the K + 1 proportions (the
        background last).
    ndim : int
        Number of trailing spatial dims.
    seed : int, optional
    dtype, device : optional
        Working backend.

    Returns
    -------
    GammaMixtureSample
        Signal, labels and per-picture proportions. Sampling runs on
        the CPU in float64.
    """
    backend = resolve_backend(dtype, device)
    shape = tuple(int(s) for s in shape)
    means = torch.as_tensor(means, dtype=torch.float64).flatten()
    shapes = torch.as_tensor(shapes, dtype=torch.float64).flatten()
    nclass = len(means)
    if len(shapes) != nclass:
        raise ValueError("means and shapes must have the same length")
    if not 0 <= ndim <= len(shape):
        raise ValueError(f"Cannot have {ndim} spatial dims in {shape}")
    alpha = torch.as_tensor(concentration, dtype=torch.float64)
    alpha = alpha.expand(nclass + 1)
    sample = shape[: len(shape) - ndim]
    npix = math.prod(shape[len(shape) - ndim :])
    nsample = math.prod(sample)
    with torch.random.fork_rng(devices=[]):
        if seed is None:
            torch.seed()
        else:
            torch.manual_seed(seed)
        g = torch._standard_gamma(alpha.expand(nsample, nclass + 1))
        pi = g / g.sum(-1, keepdim=True)
        z = torch.multinomial(pi, npix, replacement=True)
        a = shapes[z.clamp_max(nclass - 1)]
        u = torch._standard_gamma(a) * means[z.clamp_max(nclass - 1)] / a
    u = torch.where(z == nclass, 0.0, u)
    return GammaMixtureSample(
        signal=u.reshape(shape).to(**backend),
        labels=z.reshape(shape).to(backend["device"]),
        proportions=pi.reshape(*sample, nclass + 1).to(**backend),
    )
