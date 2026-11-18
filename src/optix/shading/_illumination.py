"""Mean/median illumination function (Singh et al., J Microsc 2014).

Smoothing and rescaling follow CellProfiler's CorrectIlluminationCalculate
module ("Regular" intensities, "All" images; BSD-3-Clause), which
implements the pipeline of the paper.
"""

import itertools
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .._estimator import Tags
from ..estimates import SensorEstimate, ShadingEstimate
from ..io import ArrayLike, WritableArrayLike
from ..sensor._base import CalibrationEstimator, as_map
from ._darkflat import _is_map, correct

__all__ = [
    "IlluminationFunction",
    "fit_polynomial",
    "gaussian_smooth",
    "median_smooth",
    "robust_minimum",
]

ROBUST_FACTOR = 0.02  # CellProfiler: 2nd percentile


def filter_size(shape: tuple[int, ...]) -> float:
    """CellProfiler's automatic filter size: ``min(30, max(shape) / 40)``."""
    return min(30.0, max(shape) / 40.0)


def gaussian_smooth(x: Tensor, sigma: float) -> Tensor:
    """Normalised Gaussian convolution over all dims of ``x``.

    ``conv(x) / conv(1)`` with zero padding (CellProfiler's
    ``smooth_with_function_and_mask`` without a mask), so that borders are
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
    """Median filter with a disk of ``radius`` (2D).

    Pixels outside the image are ignored (as a rank filter). Computed in
    strips of ``rows`` rows. Lower median (``torch.nanmedian``).
    """
    if x.ndim != 2:
        raise ValueError("median smoothing needs 2 spatial dims")
    r = int(radius)
    offs = [
        (i, j)
        for i, j in itertools.product(range(-r, r + 1), repeat=2)
        if i * i + j * j <= r * r
    ]
    h, w = x.shape
    pad = F.pad(x[None, None], (r, r, r, r), value=math.nan)[0, 0]
    out = torch.empty_like(x)
    for s in range(0, h, rows):
        e = min(h, s + rows)
        stack = torch.stack(
            [pad[s + r + i : e + r + i, r + j : r + j + w] for i, j in offs]
        )
        out[s:e] = stack.nanmedian(0).values
    return out


def fit_polynomial(x: Tensor, degree: int = 2) -> Tensor:
    """Least-squares polynomial of the coordinates (total degree <= 2).

    In 2D: ``A x² + B y² + C xy + D x + E y + F`` (CellProfiler's "Fit
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
    """CellProfiler rescaling: divide by the 2nd percentile of ``x > 0``.

    Values below that robust minimum are clipped to it first, so the
    result is >= 1.
    """
    pos = x[x > 0].sort().values
    if pos.numel() == 0:
        return x
    rmin = pos[int(pos.numel() * ROBUST_FACTOR)]
    return x.clamp_min(rmin) / rmin


class IlluminationFunction(CalibrationEstimator):
    r"""Retrospective multi-image illumination function (Singh et al.).

    .. math::

        F = \mathcal S\left(\operatorname{avg}_t\,(I_t - D)\right),
        \qquad x = (I - D) / F,

    where the average is the per-pixel mean (or median) over all images
    and :math:`\mathcal S` a smoothing operator.

    Parameters
    ----------
    statistic : {"mean", "median"}
        Per-pixel average. ``"mean"``: streaming moments (CellProfiler's
        accumulated sum). ``"median"``: exact per-pixel median over row
        tiles. Paper preset: ``"mean"``.
    smoothing : {"none", "median", "gaussian", "polynomial"}
        ``"median"``: disk median filter of radius ``max(1, round(σ))``;
        ``"gaussian"``: normalised Gaussian filter of standard deviation
        ``σ``; ``"polynomial"``: least-squares quadratic surface. With
        ``σ = size / 2.35`` (``size`` is a FWHM). Paper preset:
        ``"median"`` (provisional, see Notes).
    size : float, optional
        Filter size in pixels (FWHM); None: CellProfiler's automatic
        ``min(30, max(shape) / 40)``.
    rescale : {"none", "robust-min"}
        Extension: ``"robust-min"`` divides by the 2nd percentile after
        clipping below it (CellProfiler "Yes"), so ``F >= 1``.
        Paper preset: ``"none"``.
    faithful, dtype, device, layout
        See :class:`optix.Estimator`.

    Notes
    -----
    ``fit(images, dark=None)``. ``dark`` (extension) is subtracted from
    the average before smoothing: a stack, a map, or an estimate
    (``ShadingEstimate.dark`` / ``SensorEstimate.offset``).

    The paper's recommended smoothing method and filter size could not be
    checked against the publication: the preset is provisional.

    Estimates: ``"shading"`` with ``flat`` (``meta["flat_norm"]``:
    ``"none"`` or ``"robust-min"``) and ``dark`` when given.

    References
    ----------
    Singh S, Bray M-A, Jones TR, Carpenter AE. Pipeline for illumination
    correction of images for high-throughput microscopy. J Microsc 256,
    231-236 (2014). doi:10.1111/jmi.12178
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
        "size": None,
        "rescale": "none",
    }
    recommended = dict(paper)

    def __init__(
        self,
        *,
        statistic: str | None = None,
        smoothing: str | None = None,
        size: float | None = None,
        rescale: str | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: object = None,
    ) -> None:
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        if o["smoothing"] not in ("none", "median", "gaussian", "polynomial"):
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
        size = o["size"] if o["size"] is not None else filter_size(x.shape)
        sigma = size / 2.35
        if method == "gaussian":
            return gaussian_smooth(x, sigma)
        return median_smooth(x, max(1, int(sigma + 0.5)))

    def _fit(
        self,
        Y: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        if isinstance(Y, list):
            raise TypeError("IlluminationFunction takes a single stack")
        st = self._stats(Y, statistic=self.options["statistic"])
        avg = st.mean
        backend = self._backend(Y)
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
        self, Y: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        s = self.estimates["shading"]
        dark = s.dark if s.dark is not None else torch.zeros(())
        return self._apply_maps(Y, out, correct, dark, s.flat)
