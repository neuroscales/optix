"""Rolling-ball and top-hat background subtraction (Sternberg 1983)."""

import math
from collections.abc import Callable
from typing import Literal, Self

import torch
import torch.nn.functional as F
from torch import Tensor

from .._config import get_config, resolve_backend
from .._estimator import Estimator, Tags, _shares_memory
from ..io import (
    ArrayLike,
    Labeled,
    Layout,
    WritableArrayLike,
    blocks,
    chunks_of,
    flush,
    prepare_out,
)
from ..io._iter import _auto_block
from ._morphology import ball, footprint, grey_opening

__all__ = [
    "RollingBall",
    "TopHat",
    "rolling_ball",
    "rolling_ball_halo",
    "top_hat",
]

Output = Literal["corrected", "background"]


def _float(data: Tensor) -> Tensor:
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    return data


def _check_output(output: object) -> None:
    if output not in ("corrected", "background"):
        raise ValueError(
            f"output must be 'corrected' or 'background', got {output!r}"
        )


def _check_radius(radius: object) -> float:
    if radius is None or isinstance(radius, bool):
        raise ValueError(f"radius must be a number > 0, got {radius!r}")
    r = float(radius)
    if not math.isfinite(r) or r <= 0:
        raise ValueError(f"radius must be finite and > 0, got {radius!r}")
    return r


def _check_shrink(shrink: object) -> int:
    if isinstance(shrink, bool) or int(shrink) != shrink or shrink < 1:
        raise ValueError(f"shrink must be an integer >= 1, got {shrink!r}")
    return int(shrink)


def _smooth(data: Tensor, ndim: int) -> Tensor:
    """3ⁿ mean filter on the trailing `ndim` dims, replicated edges."""
    shape = data.shape
    x = data.reshape(-1, 1, *shape[-ndim:])
    pad = [1] * (2 * ndim)
    pool = (F.avg_pool1d, F.avg_pool2d, F.avg_pool3d)[ndim - 1]
    y = pool(F.pad(x, pad, mode="replicate"), 3, stride=1)
    return y.reshape(shape)


def _shrink_min(data: Tensor, s: int, ndim: int) -> Tensor:
    """Minimum over s×…×s cells of the trailing dims (partial last cells)."""
    shape = data.shape
    spatial = shape[-ndim:]
    pad = []
    for n in reversed(spatial):
        pad += [0, -n % s]
    x = data.reshape(-1, 1, *spatial)
    x = F.pad(x, pad, value=math.inf)
    pool = (F.max_pool1d, F.max_pool2d, F.max_pool3d)[ndim - 1]
    y = -pool(-x, s, stride=s)
    return y.reshape(*shape[:-ndim], *y.shape[2:])


def _expand(coarse: Tensor, s: int, spatial: tuple[int, ...]) -> Tensor:
    """Linear interpolation from cell centres back to the pixel grid."""
    ndim = len(spatial)
    lead = coarse.shape[:-ndim]
    x = coarse.reshape(-1, 1, *coarse.shape[-ndim:])
    mode = ("linear", "bilinear", "trilinear")[ndim - 1]
    size = tuple(n * s for n in coarse.shape[-ndim:])
    y = F.interpolate(x, size=size, mode=mode, align_corners=False)
    y = y[(Ellipsis, *(slice(0, n) for n in spatial))]
    return y.reshape(*lead, *spatial)


def _background(
    data: Tensor,
    radius: float,
    ndim: int,
    shrink: int,
    smooth: bool,
) -> Tensor:
    f = _smooth(data, ndim) if smooth else data
    if shrink == 1:
        b = grey_opening(f, ball(radius, ndim, dtype=f.dtype, device=f.device))
    else:
        coarse = _shrink_min(f, shrink, ndim)
        se = ball(
            radius / shrink,
            ndim,
            height=radius,
            dtype=f.dtype,
            device=f.device,
        )
        b = _expand(grey_opening(coarse, se), shrink, data.shape[-ndim:])
    # the opening is anti-extensive; this also removes the rounding of
    # (y − B) + B
    return torch.minimum(b, data)


def rolling_ball(
    data: Tensor,
    radius: float,
    *,
    ndim: int = 2,
    output: Output = "corrected",
    shrink: int = 1,
    smooth: bool = False,
) -> Tensor:
    """Subtract the background estimated by a rolling ball.

    Sternberg (1983) views a grey-scale image y as the umbra of its
    surface (the set of points under it) and estimates the background as
    the grey-scale opening of y by a ball of radius ρ:

        B(u) = √(ρ² − ‖u‖²) − ρ,   ‖u‖ ≤ ρ          (the ball's umbra)
        (y ⊖ B)(r) = min_u [y(r + u) − B(u)]         (erosion)
        (z ⊕ B)(r) = max_u [z(r − u) + B(u)]         (dilation)
        β = (y ⊖ B) ⊕ B,   result = y − β.

    The opening β is the top surface swept by the ball rolled under the
    intensity landscape: the ball follows the slowly varying background
    but cannot enter peaks narrower than itself, which the subtraction
    keeps. Pixel and intensity units are the same (the "boxels" of the
    paper are cubes), so the radius must exceed the half-width of the
    largest feature to keep, and intensities may need rescaling.

    The opening is exact (no decomposition or approximation of the ball).
    Offsets outside the image are left out of the minima and maxima (the
    paper does not specify the boundary), so the ball can overhang the
    edges. The opening never exceeds y, so the result is ≥ 0; the
    background is capped at `data`, which also removes the rounding of
    (y − B) + B.

    !!! note "Extensions"
        `shrink` and `smooth` are not from Sternberg (1983). They follow
        the practical options of ImageJ's background subtraction, written
        here from their published descriptions (they do not reproduce
        ImageJ's output bit for bit):

        * `smooth=True` opens the 3ⁿ mean of `data` (edges replicated)
          instead of `data`, which reduces the effect of noise on the
          background; the result is still `data` − background.
        * `shrink=s` takes the minimum of `data` over cells of sⁿ pixels, opens
          that coarse image with the ellipsoid of lateral semi-axis ρ/s
          and height ρ (the same ball, in coarse pixels), and interpolates
          the opening linearly from the cell centres back to the pixel
          grid. The cost drops by about s²ⁿ.

        Both make the background an approximation of the opening; it is
        then capped at `data`.

    !!! note "Cost and halo"
        The cost is O(N·rⁿ) per pass for N pixels (one operation per
        offset of the ball). The result at a pixel depends on the data
        within 2⌊ρ⌋ pixels (erosion then dilation), the halo needed to
        process blocks independently; see
        [`RollingBall`][optix.background.RollingBall].

    !!! note "scikit-image"
        `skimage.restoration.rolling_ball` returns the erosion y ⊖ B (the
        height of the ball's apex), not the opening; the two differ by
        the final dilation.

    !!! references "References"
        1. Sternberg SR. *Biomedical image processing.* Computer
           **16**(1):22–34, 1983.
           [doi:10.1109/MC.1983.1654163](https://doi.org/10.1109/MC.1983.1654163)

        ??? quote "BibTeX"
            ```bibtex
            @article{sternberg1983biomedical,
              author  = {Sternberg, Stanley R.},
              title   = {Biomedical image processing},
              journal = {Computer},
              volume  = {16}, number = {1}, pages = {22--34},
              year    = {1983}, doi = {10.1109/MC.1983.1654163}
            }
            ```

    Parameters
    ----------
    data : Tensor
        `(..., *spatial)`; the last `ndim` dims are spatial (floating
        point; other dtypes are converted to the configured dtype).
    radius : float
        Radius ρ > 0 of the ball, in pixels (and intensity units). The
        paper leaves it to the user: larger than the features to keep,
        smaller than the scale of the background variations.
    ndim : {2, 3, 1}
        Number of spatial dims: 2 for images (a 3-D ball under a 2-D
        surface, as in the paper), 3 for volumes (a 4-D ball).
    output : {"corrected", "background"}
        Return y − β (default) or the background β.
    shrink : int, default 1
        Extension: shrink factor s ≥ 1 (1 is the paper's opening).
    smooth : bool, default False
        Extension: open the 3ⁿ mean of the data.

    Returns
    -------
    Tensor
        Of the shape of `data`.
    """
    _check_output(output)
    r = _check_radius(radius)
    s = _check_shrink(shrink)
    if ndim < 1 or ndim > 3 or data.ndim < ndim:
        raise ValueError(
            f"ndim must be 1, 2 or 3 and at most data.ndim, got {ndim}"
        )
    data = _float(data)
    b = _background(data, r, ndim, s, bool(smooth))
    return b if output == "background" else data - b


def top_hat(
    data: Tensor,
    radius: int,
    *,
    shape: Literal["disk", "square"] = "disk",
    ndim: int = 2,
    output: Output = "corrected",
) -> Tensor:
    """White top-hat: subtract the opening by a flat structuring element.

    With a flat element S (B = 0 on S), the erosion and dilation of
    Sternberg (1983) are a minimum and a maximum filter:

        (y ⊖ S)(r) = min_{u ∈ S} y(r + u),   (z ⊕ S)(r) = max_{u ∈ S} z(r − u),
        β = (y ⊖ S) ⊕ S,   result = y − β,

    the top-hat of mathematical morphology. The paper defines the grey-scale
    opening for any element B and uses a ball for the background; a flat
    disk or square is the same equations with B = 0, not the paper's choice
    of element. It removes the background
    under features that S cannot fit into, whatever their height (the
    ball of [`rolling_ball`][optix.background.rolling_ball] also limits
    their slope). Offsets outside the image are left out of the minima
    and maxima. The square (cube) is computed as successive 1-D filters,
    the disk (ball) as 1-D filters along the last axis combined over the
    other axes; both are exact.

    !!! references "References"
        1. Sternberg SR. *Biomedical image processing.* Computer
           **16**(1):22–34, 1983.
           [doi:10.1109/MC.1983.1654163](https://doi.org/10.1109/MC.1983.1654163)

        ??? quote "BibTeX"
            ```bibtex
            @article{sternberg1983biomedical,
              author  = {Sternberg, Stanley R.},
              title   = {Biomedical image processing},
              journal = {Computer},
              volume  = {16}, number = {1}, pages = {22--34},
              year    = {1983}, doi = {10.1109/MC.1983.1654163}
            }
            ```

    Parameters
    ----------
    data : Tensor
        `(..., *spatial)`; the last `ndim` dims are spatial (floating
        point; other dtypes are converted to the configured dtype).
    radius : int
        Half-width k ≥ 0 of the element, in pixels.
    shape : {"disk", "square"}
        `"disk"`: ‖u‖ ≤ k (a ball in 3-D); `"square"`: ‖u‖∞ ≤ k
        (side 2k + 1; a cube in 3-D).
    ndim : {2, 3, 1}
        Number of spatial dims.
    output : {"corrected", "background"}
        Return y − β (default) or the background β.

    Returns
    -------
    Tensor
        Of the shape of `data`.
    """
    _check_output(output)
    if ndim < 1 or ndim > 3 or data.ndim < ndim:
        raise ValueError(
            f"ndim must be 1, 2 or 3 and at most data.ndim, got {ndim}"
        )
    data = _float(data)
    se = footprint(radius, ndim, shape, dtype=data.dtype, device=data.device)
    b = torch.minimum(grey_opening(data, se), data)
    return b if output == "background" else data - b


def rolling_ball_halo(
    radius: float, *, shrink: int = 1, smooth: bool = False
) -> int:
    """Halo, in pixels, that makes block-wise rolling ball exact.

    2⌊ρ⌋ for the opening (erosion then dilation), plus 1 with `smooth`.
    With `shrink=s`, a multiple of s that covers the coarse opening, the
    shrinking cells and the interpolation: s·(2⌊ρ/s⌋ + 2 + [smooth]).
    Blocks must then start on multiples of s.

    Parameters
    ----------
    radius : float
    shrink : int
    smooth : bool

    Returns
    -------
    int
    """
    r = _check_radius(radius)
    s = _check_shrink(shrink)
    if s == 1:
        return 2 * int(math.floor(r)) + int(bool(smooth))
    return s * (2 * int(math.floor(r / s)) + 2 + int(bool(smooth)))


class _Background(Estimator):
    """Shared streaming of the background estimators."""

    def fit(
        self,
        data: ArrayLike | None = None,
        *,
        layout: Layout | None = None,
        **kwargs: object,
    ) -> Self:
        """No-op: the filter has no parameters to estimate.

        Parameters
        ----------
        data : array-like, optional
            Unused (its layout, if labelled, is recorded).
        layout : Layout, optional
            Recorded for [`apply`][optix.Estimator.apply].
        **kwargs
            Ignored (`init`, `callback`, `verbose`).

        Returns
        -------
        Estimator
            self
        """
        if data is None:
            self._set_layout(layout)
            return self
        return super().fit(data, layout=layout)

    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        pass

    def _halo(self) -> tuple[int, int]:
        """Halo and the multiple that block starts must respect."""
        raise NotImplementedError

    def _filter(self, block: Tensor, ndim: int) -> Tensor:
        raise NotImplementedError

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if not isinstance(data, Labeled):
            data = self._labeled(data, self._layout_of(data))
        raw = data.array
        target = out.array if isinstance(out, Labeled) else out
        halo, step = self._halo()
        if halo and out is not None and _shares_memory(target, raw):
            raise ValueError(
                "In-place processing with a halo is not possible: use a "
                "separate output."
            )
        out = prepare_out(out, tuple(data.shape), like=raw, dtype=self.dtype)
        _, _, spatial = data.layout.positions(len(data.shape))
        ndim = len(spatial)
        if ndim > 3:
            raise ValueError(f"At most 3 spatial axes, got {ndim}")
        block = None
        if step > 1:
            # cores on multiples of `step`, so that the shrinking cells of
            # every block lie on the global grid
            shape = tuple(int(n) for n in data.shape)
            extents = [shape[a] for a in spatial]
            itemsize = self._backend(raw)["dtype"].itemsize
            budget = get_config().memory // itemsize
            block = _auto_block(extents, [halo] * ndim, budget, 4.0, None)
            block = [
                n if b >= n else max(b // step, 1) * step
                for b, n in zip(block, extents, strict=True)
            ]
        chunks = blocks(
            data,
            block,
            halo,
            overhead=4.0,
            align=chunks_of(out),
            dtype=self.dtype,
            device=self.device,
        )
        for chunk in chunks:
            chunk.store(out, self._filter(chunk.data, ndim)[chunk.inner])
        flush(out)
        return out


class RollingBall(_Background):
    """Rolling-ball background subtraction (Sternberg 1983).

    The background is the grey-scale opening of each image (or volume) by
    a ball: see [`rolling_ball`][optix.background.rolling_ball] for the
    method. There are no parameters to fit:
    [`fit`][optix.Estimator.fit] only records the layout, and
    [`apply`][optix.Estimator.apply] streams blocks with a halo of 2⌊ρ⌋
    pixels (see [`rolling_ball_halo`][optix.background.rolling_ball_halo]),
    so arrays larger than memory (zarr, memmap) give the same result as a
    whole-array computation. The spatial axes of the layout (2 or 3) set
    the dimension of the ball.

    !!! note
        Faithful mode is the paper's opening by a ball. `shrink` and
        `smooth` are extensions after ImageJ, off by default.

    !!! references "References"
        1. Sternberg SR. *Biomedical image processing.* Computer
           **16**(1):22–34, 1983.
           [doi:10.1109/MC.1983.1654163](https://doi.org/10.1109/MC.1983.1654163)

        ??? quote "BibTeX"
            ```bibtex
            @article{sternberg1983biomedical,
              author  = {Sternberg, Stanley R.},
              title   = {Biomedical image processing},
              journal = {Computer},
              volume  = {16}, number = {1}, pages = {22--34},
              year    = {1983}, doi = {10.1109/MC.1983.1654163}
            }
            ```

    """

    tags = Tags(
        targets=("background",),
        streaming="halo",
        reference="doi:10.1109/MC.1983.1654163",
    )
    paper = {
        "radius": None,
        "output": "corrected",
        "shrink": 1,
        "smooth": False,
    }
    recommended = dict(paper)
    # free parameters of the published method: any value is faithful
    _free = ("radius", "output")

    def __init__(
        self,
        *,
        radius: float | None = None,
        output: Output | None = None,
        shrink: int | None = None,
        smooth: bool | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the filter.

        Parameters
        ----------
        radius : float
            Radius ρ > 0 of the ball, in pixels and intensity units
            (required; the paper leaves it to the user).
        output : {"corrected", "background"}
            What [`apply`][optix.Estimator.apply] returns: data −
            background (default) or the background.
        shrink : int, default 1
            Extension (not in the paper): shrink factor; see
            [`rolling_ball`][optix.background.rolling_ball].
        smooth : bool, default False
            Extension (not in the paper): open the 3ⁿ mean of the data.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        o["radius"] = _check_radius(o["radius"])
        _check_output(o["output"])
        o["shrink"] = _check_shrink(o["shrink"])
        o["smooth"] = bool(o["smooth"])
        self.is_faithful = o["shrink"] == 1 and not o["smooth"]

    def _halo(self) -> tuple[int, int]:
        o = self.options
        halo = rolling_ball_halo(
            o["radius"], shrink=o["shrink"], smooth=o["smooth"]
        )
        return halo, o["shrink"]

    def _filter(self, block: Tensor, ndim: int) -> Tensor:
        o = self.options
        return rolling_ball(
            block,
            o["radius"],
            ndim=ndim,
            output=o["output"],
            shrink=o["shrink"],
            smooth=o["smooth"],
        )


class TopHat(_Background):
    """Top-hat background subtraction with a flat structuring element.

    The background is the grey-scale opening of each image (or volume) by
    a flat disk or square (Sternberg 1983): see
    [`top_hat`][optix.background.top_hat]. There are no parameters to
    fit; [`apply`][optix.Estimator.apply] streams blocks with a halo of
    2k pixels for a half-width k, which gives the whole-array result.

    !!! references "References"
        1. Sternberg SR. *Biomedical image processing.* Computer
           **16**(1):22–34, 1983.
           [doi:10.1109/MC.1983.1654163](https://doi.org/10.1109/MC.1983.1654163)

        ??? quote "BibTeX"
            ```bibtex
            @article{sternberg1983biomedical,
              author  = {Sternberg, Stanley R.},
              title   = {Biomedical image processing},
              journal = {Computer},
              volume  = {16}, number = {1}, pages = {22--34},
              year    = {1983}, doi = {10.1109/MC.1983.1654163}
            }
            ```

    """

    tags = Tags(
        targets=("background",),
        streaming="halo",
        reference="doi:10.1109/MC.1983.1654163",
    )
    paper = {"radius": None, "shape": "disk", "output": "corrected"}
    recommended = dict(paper)

    def __init__(
        self,
        *,
        radius: int | None = None,
        shape: Literal["disk", "square"] | None = None,
        output: Output | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the filter.

        Parameters
        ----------
        radius : int
            Half-width k ≥ 0 of the element, in pixels (required).
        shape : {"disk", "square"}, default `"disk"`
            Flat element: ‖u‖ ≤ k or ‖u‖∞ ≤ k.
        output : {"corrected", "background"}
            What [`apply`][optix.Estimator.apply] returns: data −
            background (default) or the background.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        if o["radius"] is None:
            raise ValueError("radius is required")
        footprint(o["radius"], 1, o["shape"])
        o["radius"] = int(o["radius"])
        _check_output(o["output"])
        self.is_faithful = True

    def _halo(self) -> tuple[int, int]:
        return 2 * self.options["radius"], 1

    def _filter(self, block: Tensor, ndim: int) -> Tensor:
        o = self.options
        return top_hat(
            block,
            o["radius"],
            shape=o["shape"],
            ndim=ndim,
            output=o["output"],
        )
