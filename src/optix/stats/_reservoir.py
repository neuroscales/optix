"""Uniform random sampling of a stream (bottom-k sampling)."""

from typing import Self

import torch
from torch import Tensor

from ._base import Reducer

__all__ = ["Reservoir"]

_KEY_BITS = 62
_KEYS = 2**_KEY_BITS
# torch.rand(float32) returns multiples of 2⁻²⁴: `rand < 2⁻ᵉ` holds
# with probability exactly 2⁻ᵉ for 0 <= e <= 24.
_RAND_BITS = 24


class Reservoir(Reducer):
    """Uniform random sample (without replacement) of the values seen.

    Every value gets a random int64 key and the `size` values with the
    smallest keys are kept. Merging concatenates and keeps the `size`
    smallest keys again, so merging is exact and the sample is exactly
    uniform.

    Attributes
    ----------
    values : (≤ size,) Tensor
        Sampled values (flattened, in no particular order).
    seen : int
        Number of values seen (after masking).

    Notes
    -----
    Once the reservoir is full, with `kmax` its largest key, a value is
    only kept if its key is below `kmax`. A float32 Bernoulli mask with
    probability `q`, the smallest power of two ≥ `kmax` / 2⁶² (and
    ≥ 2⁻²⁴, so that it is exact in float32), selects candidates, and
    int64 keys are drawn uniformly in [0, `q` 2⁶²) for those only; keys
    ≥ `kmax` are then discarded. Each value thus gets a key uniform in
    [0, `kmax`) with probability exactly `kmax` / 2⁶², as if a key had
    been drawn for every value, without a chunk-sized int64 temporary.

    Each update synchronises with the host (boolean indexing and reading
    `kmax`). The random generator is not part of `state_dict`: a
    reservoir restored with `from_state_dict` starts again from `seed`,
    and reservoirs (e.g. parallel workers) built with the same `seed`
    draw the same key stream. Give each worker its own seed (or none) for
    independent samples.
    """

    _state = ("values", "keys", "count")

    def __init__(self, size: int, *, seed: int | None = None) -> None:
        """Create an empty reservoir.

        Parameters
        ----------
        size : int
            Maximum number of values kept.
        seed : int, optional
            Seed of the random keys. Samples are reproducible for a fixed
            seed **and a fixed chunking** only.
        """
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        super().__init__(dim=None)
        self.size = size
        self.seed = seed
        self._generator: torch.Generator | None = None

    def _args(self) -> dict[str, object]:
        return {"size": self.size, "seed": self.seed}

    @property
    def seen(self) -> int:
        """Number of values seen."""
        return 0 if self.count is None else int(self.count)

    def _gen(self, device: torch.device) -> torch.Generator:
        old = self._generator
        if old is not None and old.device == device:
            return old
        gen = torch.Generator(device)
        if old is not None:
            # moved to another device: continue with a fresh stream
            seed = torch.randint(_KEYS, (1,), generator=old, device=old.device)
            gen.manual_seed(int(seed))
        elif self.seed is not None:
            gen.manual_seed(self.seed)
        else:
            gen.seed()
        self._generator = gen
        return gen

    def _keep(self, values: Tensor, keys: Tensor) -> None:
        if keys.numel() > self.size:
            idx = keys.topk(self.size, largest=False, sorted=False).indices
            values, keys = values[idx], keys[idx]
        self.values, self.keys = values, keys

    def update(self, x: Tensor, *, weight: Tensor | None = None) -> Self:
        """Add a chunk of values; `weight` is an optional boolean mask."""
        x = x.detach()
        if weight is None:
            v = x.reshape(-1)
        else:
            v = x[weight.detach().to(x.device).bool().expand_as(x)]
        device = v.device
        if self.values is None:
            self.values = v.new_empty(0)
            self.keys = torch.empty(0, dtype=torch.int64, device=device)
            self.count = torch.zeros((), dtype=torch.int64, device=device)
        self.count = self.count + v.numel()
        if v.numel() == 0:
            return self
        gen = self._gen(device)
        if self.keys.numel() < self.size:
            keys = torch.randint(_KEYS, v.shape, generator=gen, device=device)
        else:
            kmax = max(int(self.keys.max()), 1)
            # q = 2⁻ᵉ >= kmax / 2⁶², exact in float32 (e <= 24)
            e = min(_RAND_BITS, _KEY_BITS - (kmax - 1).bit_length())
            u = torch.rand(
                v.shape, generator=gen, device=device, dtype=torch.float32
            )
            v = v[u < 2.0**-e]
            keys = torch.randint(
                2 ** (_KEY_BITS - e), v.shape, generator=gen, device=device
            )
            below = keys < kmax
            v, keys = v[below], keys[below]
        self._keep(
            torch.cat([self.values, v.to(self.values.dtype)]),
            torch.cat([self.keys, keys]),
        )
        return self

    def merge(self, other: "Reservoir") -> Self:
        """Merge another [`Reservoir`][optix.stats.Reservoir] (exact)."""
        state = self._other_state(other)
        if state["values"] is None:
            return self
        if self.values is None:
            self.values = state["values"].clone()
            self.keys = state["keys"].clone()
            self.count = state["count"].clone()
            self._keep(self.values, self.keys)
            return self
        self.count = self.count + state["count"]
        self._keep(
            torch.cat([self.values, state["values"].to(self.values.dtype)]),
            torch.cat([self.keys, state["keys"]]),
        )
        return self

    def copy(self) -> Self:
        """Independent copy, including the generator state."""
        new = super().copy()
        if self._generator is not None:
            gen = torch.Generator(self._generator.device)
            gen.set_state(self._generator.get_state())
            new._generator = gen
        return new
