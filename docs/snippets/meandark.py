from collections.abc import Callable

import torch
from torch import Tensor

import optix
from optix.estimates import ShadingEstimate
from optix.io import ArrayLike, Chunk, Layout, WritableArrayLike
from optix.stats import Moments, reduce_groups


class MeanDark(optix.Estimator):
    """Dark image as the per-pixel mean of frames taken without light."""

    tags = optix.Tags(
        targets=("shading",),
        streaming="per-pixel-stats",
        noise="poisson-gaussian",
        many_frames=True,
    )

    def __init__(
        self,
        *,
        faithful: bool = True,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        layout: Layout | None = None,
    ) -> None:
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())

    def _fit(
        self,
        Y: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        dark = torch.zeros(self._map_shape(Y), **self._backend(Y))
        groups = reduce_groups(
            Y, Moments(1), dtype=self.dtype, device=self.device
        )
        for key, (moments,) in groups.items():
            dark[key] = moments.mean.to(dark)
        self.estimates = {"shading": ShadingEstimate(dark=dark)}

    def _apply(
        self, Y: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        dark = self.estimates["shading"].dark

        def subtract(chunk: Chunk, batch: Tensor) -> Tensor:
            return batch - dark[chunk.group].to(batch)

        return self._apply_frames(Y, out, subtract)
