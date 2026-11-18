"""Seeded generators."""

import torch


def generator(seed: int | None, device: torch.device) -> torch.Generator:
    """A generator on ``device``, seeded from ``seed`` (random if None)."""
    gen = torch.Generator(device)
    if seed is None:
        gen.seed()
    else:
        gen.manual_seed(seed)
    return gen
