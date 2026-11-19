"""Camera sensor model: offset, gain, Poisson shot noise, read noise."""

from collections.abc import Sequence

import torch
from torch import Tensor

from .._config import resolve_backend
from ..estimates import SensorEstimate
from ..io._arrays import _round_clamp
from ._random import generator

__all__ = ["random_sensor", "simulate_sensor"]


def random_sensor(
    shape: Sequence[int],
    *,
    offset: float = 100.0,
    offset_sd: float = 2.0,
    read_sd: float = 1.5,
    gain: float = 0.5,
    gain_sd: float = 0.0,
    defective: float = 0.0,
    seed: int | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> SensorEstimate:
    """Draw a random sensor.

    Parameters
    ----------
    shape : sequence of int
        Spatial shape (H, W).
    offset, offset_sd : float
        Mean and per-pixel standard deviation of the offset (ADU).
    read_sd : float
        Read-noise standard deviation (ADU), the same for every pixel.
    gain, gain_sd : float
        Mean and per-pixel standard deviation of the gain (ADU per
        electron). Gains are kept positive.
    defective : float
        Fraction of defective pixels.
    seed : int, optional
    dtype, device : optional
        Working backend.

    Returns
    -------
    SensorEstimate
        `offset`, `variance` and `gain` maps of `shape` and a
        boolean `defective` map.
    """
    backend = resolve_backend(dtype, device)
    gen = generator(seed, backend["device"])
    shape = tuple(shape)

    def normal() -> Tensor:
        return torch.randn(shape, generator=gen, **backend)

    tiny = torch.finfo(backend["dtype"]).tiny
    return SensorEstimate(
        offset=offset + offset_sd * normal(),
        variance=torch.full(shape, read_sd**2, **backend),
        gain=(gain + gain_sd * normal()).clamp_min(tiny),
        defective=torch.rand(shape, generator=gen, **backend) < defective,
        meta={"units": "ADU", "simulated": True},
    )


def simulate_sensor(
    signal: Tensor,
    sensor: SensorEstimate,
    *,
    seed: int | None = None,
    quantize: torch.dtype | None = None,
) -> Tensor:
    """Simulate camera data from a signal in electrons.

    Parameters
    ----------
    signal : (..., H, W) Tensor
        Expected number of electrons (≥ 0).
    sensor : SensorEstimate
        Needs `offset`, `variance` and `gain`. `defective` is a
        label only: defective pixels are simulated like the others.
    seed : int, optional
    quantize : torch.dtype, optional
        Integer dtype: round and saturate to its range.

    Returns
    -------
    Tensor
        offset + gain · Poisson(signal) + N(0, variance), in ADU.
    """
    for name in ("offset", "variance", "gain"):
        if getattr(sensor, name) is None:
            raise ValueError(f"sensor.{name} is required")
    gen = generator(seed, signal.device)
    sensor = sensor.to(signal.device)
    rate = signal.clamp_min(0)
    electrons = torch.poisson(rate, generator=gen)
    noise = torch.randn(
        signal.shape, generator=gen, dtype=signal.dtype, device=signal.device
    )
    y = (
        sensor.offset
        + sensor.gain * electrons
        + sensor.variance.sqrt() * noise
    )
    if quantize is not None:
        if quantize.is_floating_point or quantize == torch.bool:
            raise ValueError("quantize must be an integer dtype")
        info = torch.iinfo(quantize)
        y = _round_clamp(y, info.min, info.max).to(quantize)
    return y
