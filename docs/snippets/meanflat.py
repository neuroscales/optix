from collections.abc import Callable

import torch
from torch import Tensor

import optix
from optix.basis import dct_basis, project, reconstruct
from optix.estimates import ShadingEstimate
from optix.io import ArrayLike, Chunk, Layout, WritableArrayLike
from optix.stats import Moments, reduce_groups


class MeanFlat(optix.Estimator):
    """Flat field as the per-pixel mean of many frames."""

    tags = optix.Tags(
        targets=("shading",),
        streaming="per-pixel-stats",
        noise="poisson-gaussian",
        many_frames=True,
    )
    paper = {"smooth": 0}
    recommended = {"smooth": 4}

    def __init__(
        self,
        *,
        smooth: int | None = None,
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
        self._warn_unused(init, ShadingEstimate)
        shape = self._map_shape(Y)  # (*group, *spatial)
        nspatial = len(self._layout_of(Y).spatial)
        backend = self._backend(Y)
        # The dark field is not estimated here: take it from `init`.
        start = self._pick(init, ShadingEstimate, shape=shape, like=Y)
        dark = torch.zeros(shape, **backend)
        if start is not None and start.dark is not None:
            dark = dark + start.dark
        # One pass over the data, one running mean per group.
        mean = torch.zeros(shape, **backend)
        groups = reduce_groups(
            Y, Moments(1), dtype=self.dtype, device=self.device
        )
        for key, (moments,) in groups.items():
            mean[key] = moments.mean.to(**backend)
        flat = mean - dark
        k = self.options["smooth"]
        if k:  # extension: keep only the lowest spatial frequencies
            sizes = shape[len(shape) - nspatial :]
            bases = [dct_basis(n, min(k, n), **backend) for n in sizes]
            flat = reconstruct(project(flat, bases), bases)
        dims = tuple(range(-nspatial, 0))
        flat = flat / flat.mean(dims, keepdim=True)
        self.estimates = {
            "shading": ShadingEstimate(
                flat=flat, dark=dark, meta={"flat_norm": "mean"}
            )
        }

    def _apply(
        self, Y: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        shading = self.estimates["shading"]

        def correct(chunk: Chunk, batch: Tensor) -> Tensor:
            flat = shading.flat[chunk.group].to(batch)
            dark = shading.dark[chunk.group].to(batch)
            return (batch - dark) / flat

        return self._apply_frames(Y, out, correct)
