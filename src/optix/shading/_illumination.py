"""Retrospective illumination function (Singh et al., J Microsc 2014).

After [Singh et al. 2014](https://doi.org/10.1111/jmi.12178). Faithful mode
follows the paper. The extensions (disk median, Gaussian and polynomial
smoothing, automatic size, robust-minimum rescaling) follow CellProfiler's
CorrectIlluminationCalculate module (BSD-3-Clause).
"""

import itertools
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .._estimator import Tags
from ..estimates import SensorEstimate, ShadingEstimate
from ..filters import median_disk, median_filter
from ..io import ArrayLike, WritableArrayLike
from ..sensor._base import CalibrationEstimator, as_map
from ._darkflat import _is_map, correct

__all__ = [
    "IlluminationFunction",
    "fit_polynomial",
    "gaussian_smooth",
    "median_smooth",
    "median_window",
    "robust_minimum",
]

ROBUST_FACTOR = 0.02  # CellProfiler: 2nd percentile


def filter_size(shape: tuple[int, ...]) -> float:
    """CellProfiler's automatic filter size: min(30, max(shape)/40)."""
    return min(30.0, max(shape) / 40.0)


def gaussian_smooth(x: Tensor, sigma: float) -> Tensor:
    """Normalised Gaussian convolution over all dims of `x`.

    conv(x)/conv(1) with zero padding (CellProfiler's
    `smooth_with_function_and_mask` without a mask), so that borders are
    not darkened. The kernel is truncated at 4 sigma.
    """
    r = max(1, int(math.ceil(4 * sigma)))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma).square())
    k = k / k.sum()
    num, den = x, torch.ones_like(x)
    for d in range(x.ndim):
        num = _conv1d(num, k, d)
        den = _conv1d(den, k, d)
    return num / den


def _conv1d(x: Tensor, k: Tensor, dim: int) -> Tensor:
    x = x.movedim(dim, -1)
    shape = x.shape
    y = F.conv1d(x.reshape(-1, 1, shape[-1]), k.view(1, 1, -1), padding="same")
    return y.reshape(shape).movedim(-1, dim)


def median_smooth(x: Tensor, radius: int, *, rows: int = 64) -> Tensor:
    """Median filter with a disk of `radius` (2D; CellProfiler extension).

    Pixels outside the image are ignored (as a rank filter). Computed in
    strips of `rows` rows. Lower median (`torch.nanmedian`). See
    [`median_disk`][optix.filters.median_disk].
    """
    _check_2d(x)
    return median_disk(x, radius, rows=rows)


def median_window(x: Tensor, size: int, *, budget: int = 1 << 24) -> Tensor:
    """Exact median filter with a square window of side `size` (2D).

    The window of pixel (i, j) spans rows i − ⌊size/2⌋ … i + ⌈size/2⌉ − 1
    (likewise for columns); pixels outside the image are ignored. Lower
    median (`torch.nanmedian`). A side larger than 2·n − 1 along a dim of
    length n is equivalent to 2·n − 1 there (every window then spans the
    whole dim). The cost per pixel does not grow with `size`; `budget`
    bounds the memory of small windows, which are sorted. See
    [`median_filter`][optix.filters.median_filter].
    """
    _check_2d(x)
    return median_filter(x, size, budget=budget)


def _check_2d(x: Tensor) -> None:
    if x.ndim != 2:
        raise ValueError("median smoothing needs 2 spatial dims")


def fit_polynomial(x: Tensor, degree: int = 2) -> Tensor:
    """Least-squares polynomial of the coordinates (total degree <= 2).

    In 2D: A·x² + B·y² + C·xy + D·x + E·y + F (CellProfiler's "Fit
    Polynomial"). Coordinates are scaled to [-1, 1].
    """
    grids = torch.meshgrid(
        *[
            torch.linspace(-1, 1, n, dtype=x.dtype, device=x.device)
            for n in x.shape
        ],
        indexing="ij",
    )
    cols = [torch.ones_like(x)]
    for deg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(grids, deg):
            term = combo[0]
            for c in combo[1:]:
                term = term * c
            cols.append(term)
    a = torch.stack([c.reshape(-1) for c in cols], 1)
    coef = torch.linalg.lstsq(
        a.double(), x.reshape(-1, 1).double()
    ).solution.to(x.dtype)
    return (a @ coef).reshape(x.shape)


def robust_minimum(x: Tensor) -> Tensor:
    """CellProfiler rescaling: divide by the 2nd percentile of `x > 0`.

    Values below that robust minimum are clipped to it first, so the
    result is >= 1.
    """
    pos = x[x > 0].sort().values
    if pos.numel() == 0:
        return x
    rmin = pos[int(pos.numel() * ROBUST_FACTOR)]
    return x.clamp_min(rmin) / rmin


class IlluminationFunction(CalibrationEstimator):
    """Retrospective multi-image illumination function (Singh et al., 2014).

    The illumination correction function (ICF) is

        f = S(meanₙ (yₙ − d)),   x = (y − d)/f,

    the per-pixel average of all images of a batch (e.g. a plate and
    channel) smoothed by S; each image is corrected by dividing it by f.
    ([Singh et al. 2014](https://doi.org/10.1111/jmi.12178): F = f, I = y.)

    Faithful mode follows the paper: per-pixel **mean** over all images, then a
    **median filter with a window of 500 pixels** ([Singh et al. 2014,
    "Computing illumination correction functions", Fig.
    1](https://doi.org/10.1111/jmi.12178)), then division. The paper does not
    rescale the ICF and does not subtract a dark field.

    !!! note "Interpretation"
        The paper's "median filter with window size = 500 pixels" is
        implemented as a square window of side `size`
        ([`median_window`][optix.shading.utils.median_window]),
        pixels outside the image being ignored. The paper implements the
        method in CellProfiler; CellProfiler's own conventions (a disk
        median of radius ≈ size/2.35, Gaussian and polynomial smoothing,
        the automatic filter size, the robust-minimum rescaling) are
        extensions here, ported from CellProfiler's
        CorrectIlluminationCalculate (BSD-3-Clause) where the paper defers
        to it, and are off by default.

    !!! note "Cost"
        The exact square median
        ([`median_filter`][optix.filters.median_filter]) costs the same per
        pixel whatever the window size: the paper's
        500-pixel window on a 1280×1024 image takes a few seconds on one
        CPU thread.

    !!! note
        `fit(images, dark=None)`. `dark` (extension) is subtracted from the
        average before smoothing: a stack, a map, or an estimate
        (`ShadingEstimate.dark` / `SensorEstimate.offset`). Estimates:
        `"shading"` with `flat` (`meta["flat_norm"]`: `"none"` or
        `"robust-min"`) and `dark` when given.

    !!! references "References"
        1. Singh S, Bray M-A, Jones TR, Carpenter AE. *Pipeline for
           illumination correction of images for high-throughput
           microscopy.* J Microsc **256**(3):231–236, 2014.
           [doi:10.1111/jmi.12178](https://doi.org/10.1111/jmi.12178)

        ??? quote "BibTeX"
            ```bibtex
            @article{singh2014pipeline,
              author  = {Singh, S. and Bray, M.-A. and Jones, T. R. and
                         Carpenter, A. E.},
              title   = {Pipeline for illumination correction of images for
                         high-throughput microscopy},
              journal = {Journal of Microscopy},
              volume  = {256}, number = {3}, pages = {231--236},
              year    = {2014}, doi = {10.1111/jmi.12178}
            }
            ```
    """

    tags = Tags(
        targets=("shading",),
        streaming="per-pixel-stats",
        modality=("widefield", "high-content"),
        many_frames=True,
        reference="doi:10.1111/jmi.12178",
    )
    paper = {
        "statistic": "mean",
        "smoothing": "median",
        "size": 500.0,
        "rescale": "none",
    }
    recommended = dict(paper)

    def __init__(
        self,
        *,
        statistic: str | None = None,
        smoothing: str | None = None,
        size: float | str | None = None,
        rescale: str | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: object = None,
    ) -> None:
        """Configure the illumination function.

        Parameters
        ----------
        statistic : {"mean", "median"}
            Per-pixel average. `"mean"`: streaming moments (paper).
            `"median"` (extension): exact per-pixel median over row tiles.
        smoothing : {"median", "median-disk", "gaussian", "polynomial", "none"}
            `"median"` (paper): square median window of side `size`.
            Extensions (CellProfiler): `"median-disk"`, a disk median of
            radius max(1, round(σ)); `"gaussian"`, a normalised Gaussian
            filter of standard deviation σ; `"polynomial"`, a
            least-squares quadratic surface; with σ = size/2.35.
        size : float
            Window side in pixels (`"median"`), or CellProfiler's filter
            size (FWHM) for the extensions; `"auto"` gives CellProfiler's
            min(30, max(shape)/40). Paper: 500.
        rescale : {"none", "robust-min"}
            Extension: `"robust-min"` divides by the 2nd percentile after
            clipping below it (CellProfiler), so F ≥ 1. Paper: `"none"`.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        if o["smoothing"] not in (
            "none",
            "median",
            "median-disk",
            "gaussian",
            "polynomial",
        ):
            raise ValueError(f"Unknown smoothing {o['smoothing']!r}")
        if o["rescale"] not in ("none", "robust-min"):
            raise ValueError(f"Unknown rescale {o['rescale']!r}")
        if o["statistic"] not in ("mean", "median"):
            raise ValueError("statistic must be 'mean' or 'median'")

    def _smooth(self, x: Tensor) -> Tensor:
        o = self.options
        method = o["smoothing"]
        if method == "none":
            return x
        if method == "polynomial":
            return fit_polynomial(x)
        size = (
            filter_size(x.shape) if o["size"] == "auto" else float(o["size"])
        )
        if method == "median":
            return median_window(x, int(round(size)))
        sigma = size / 2.35
        if method == "gaussian":
            return gaussian_smooth(x, sigma)
        return median_smooth(x, max(1, int(sigma + 0.5)))

    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        if isinstance(data, list):
            raise TypeError("IlluminationFunction takes a single stack")
        st = self._stats(data, statistic=self.options["statistic"])
        avg = st.mean
        backend = self._backend(data)
        raw = self._dark_input
        dmap = None
        if isinstance(raw, ShadingEstimate):
            dmap = as_map(raw, "dark", backend)
        elif isinstance(raw, SensorEstimate):
            dmap = as_map(raw, "offset", backend)
        elif isinstance(raw, Tensor) and _is_map(raw, avg):
            dmap = raw.to(**backend)
        elif raw is not None:
            dmap = self._stats(
                self._dark_stack(), statistic=self.options["statistic"]
            ).mean
        if dmap is not None:
            avg = avg - dmap
        nsp = st.nspatial
        lead = avg.shape[: avg.ndim - nsp]
        flat = avg.reshape(-1, *avg.shape[avg.ndim - nsp :]).clone()
        for i in range(flat.shape[0]):
            f = self._smooth(flat[i])
            if self.options["rescale"] == "robust-min":
                f = robust_minimum(f)
            flat[i] = f
        flat = flat.reshape(*lead, *flat.shape[1:])
        self.estimates = {
            "shading": ShadingEstimate(
                flat=flat,
                dark=None if dmap is None else dmap.expand_as(flat).clone(),
                meta={"flat_norm": self.options["rescale"]},
            )
        }

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        s = self.estimates["shading"]
        dark = s.dark if s.dark is not None else torch.zeros(())
        return self._apply_maps(data, out, correct, dark, s.flat)
