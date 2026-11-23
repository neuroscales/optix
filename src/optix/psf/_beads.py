"""Empirical PSF from images of sub-resolution beads."""

import itertools
import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from .._estimator import Estimator, Tags
from ..estimates import PSFEstimate
from ..io import (
    ArrayLike,
    Labeled,
    Layout,
    WritableArrayLike,
    blocks,
    integer_range,
)
from ..stats import Reservoir
from ._fit import fit_gaussian, threshold_centroid
from ._ops import (
    fourier_shift,
    fwhm,
    gaussian_filter,
    linear_shift,
    local_maxima,
    tukey_window,
)

__all__ = ["BeadPSF", "bead_psf"]

_DEFAULTS: dict[str, object] = {
    "radius": 8,
    "sigma": 1.0,
    "threshold": 5.0,
    "min_distance": 2,
    "saturation": None,
    "background": "border",
    "center": "gaussian",
    "shift": "fourier",
    "weights": "equal",
    "symmetrize": (),
    "taper": 0.0,
    "max_beads": None,
    "block": None,
}
_SAMPLE = 100_000  # values kept to estimate the noise level
_MAX_DRIFT = 2.0  # voxels between the detected peak and the fitted centre
_MAD = 1.482602218505602  # 1 / Φ⁻¹(3/4)


def _axes(value: object, ndim: int, name: str) -> list:
    if isinstance(value, int | float):
        return [value] * ndim
    value = list(value)  # type: ignore[call-overload]
    if len(value) != ndim:
        raise ValueError(
            f"{name} needs one entry per spatial axis ({ndim}), got {value}"
        )
    return value


def _saturation(raw: object, option: float | None) -> float | None:
    if option is not None:
        return float(option)
    dtype = getattr(raw, "dtype", None)
    if dtype is None:
        return None
    try:
        return integer_range(dtype)[1]
    except (TypeError, ValueError):
        return None


class BeadPSF(Estimator):
    """Point spread function averaged over images of isolated beads.

    Sub-resolution fluorescent beads image the PSF directly. Each
    isolated bead is cropped, its background is removed, it is shifted so
    that its centre falls on the central voxel, and the crops are
    averaged:

        h = N(∑ₙ wₙ · Tₙ(yₙ − bₙ)),

    with yₙ the crop of bead n, bₙ its background, Tₙ the sub-voxel
    shift that moves its centre cₙ to index `K // 2`, wₙ its weight and
    N the final post-processing and normalisation to unit sum.

    There is no single published method for this procedure; every step
    below is a standard choice, and each is an option. `faithful` has no
    effect: both presets are the defaults listed here.

    **Detection.** The data are smoothed with a Gaussian of standard
    deviation `sigma` voxels. A first pass estimates the noise level of
    the smoothed data from a uniform random sample of 100 000 of its
    values (per group): the median μ and σ = 1.4826 · MAD. Candidates
    are the voxels that are the maximum of their window of half-width
    `min_distance` and exceed μ + `threshold` · σ.

    **Rejection.** A candidate is discarded when its crop of half-size
    `radius` does not fit inside the data, when another candidate lies
    inside that crop (overlapping beads), when the crop reaches the
    `saturation` level, when its fitted centre lies more than 2 voxels
    from the detected peak, or when its background-subtracted sum is not
    positive.

    **Background.** `"border"`: the median of the voxels on the faces of
    the crop, per bead; a number: a constant.

    **Centre.** `"gaussian"`: least-squares fit of an axis-aligned
    Gaussian plus a constant (see
    [`fit_gaussian`][optix.psf.utils.fit_gaussian]); `"centroid"`:
    centroid of the part of the crop above a quarter of its peak (see
    [`threshold_centroid`][optix.psf.utils.threshold_centroid]).

    **Alignment.** `"fourier"`: multiplication of the spectrum by a phase
    ramp (exact for band-limited data, circular on the crop); `"linear"`:
    separable linear interpolation (smooths slightly).

    **Averaging.** `"equal"`: each aligned crop is normalised to unit sum
    first, so all beads weigh the same; `"brightness"`: crops are summed
    as they are, so brighter beads weigh more.

    **Post-processing.** Optional mirror symmetrisation along chosen
    axes (`symmetrize`), an optional separable Tukey taper (`taper`),
    clipping of negative values to zero, and normalisation to unit sum.
    The full width at half maximum along each axis is reported as a
    quality check.

    The data are read block by block, twice (noise level, then beads),
    with a halo wide enough to see every crop and its neighbours; a bead
    belongs to the block whose core holds its peak. Group axes give one
    PSF each; sample axes (e.g. several bead acquisitions) are pooled.

    `apply` is not defined: the result is
    `estimates["psf"]`, a [`PSFEstimate`][optix.estimates.PSFEstimate],
    whose `psf` can be given to the deconvolution estimators.
    """

    tags = Tags(
        targets=("psf",),
        streaming="halo",
        modality=("widefield", "confocal", "light-sheet"),
        noise="poisson-gaussian",
    )
    paper = dict(_DEFAULTS)
    recommended = dict(_DEFAULTS)

    def __init__(
        self,
        *,
        radius: int | Sequence[int] | None = None,
        sigma: float | Sequence[float] | None = None,
        threshold: float | None = None,
        min_distance: int | None = None,
        saturation: float | None = None,
        background: str | float | None = None,
        center: str | None = None,
        shift: str | None = None,
        weights: str | None = None,
        symmetrize: Sequence[int] | None = None,
        taper: float | None = None,
        max_beads: int | None = None,
        block: Sequence[int | None] | None = None,
        faithful: bool = True,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        layout: Layout | None = None,
    ) -> None:
        """Set the detection, alignment and averaging options.

        Parameters
        ----------
        radius : int or sequence of int
            Half-size of the crop along each spatial axis (layout
            order); the PSF has shape `2 · radius + 1`. Default 8.
        sigma : float or sequence of float
            Standard deviation, in voxels, of the Gaussian smoothing used
            for detection. Default 1.
        threshold : float
            Detection threshold, in noise standard deviations of the
            smoothed data above its median. Default 5.
        min_distance : int
            Half-width of the window in which a candidate must be the
            maximum. Default 2.
        saturation : float, optional
            Beads whose crop reaches this value are rejected. Default:
            the largest value of the data type for integer data, no test
            for floating-point data.
        background : {"border"} or float
            Background estimate (default `"border"`).
        center : {"gaussian", "centroid"}
            Sub-voxel centre estimate (default `"gaussian"`).
        shift : {"fourier", "linear"}
            Sub-voxel alignment (default `"fourier"`).
        weights : {"equal", "brightness"}
            Bead weights in the average (default `"equal"`).
        symmetrize : sequence of int
            Spatial axes (layout order) along which the PSF is averaged
            with its mirror image. Default: none.
        taper : float
            Fraction of each axis inside the cosine tapers of a Tukey
            window applied to the result; 0 (default) for none.
        max_beads : int, optional
            Stop collecting beads (per group) once this many are kept.
        block : sequence of int or None, optional
            Core block size along the spatial axes (default: from the
            memory budget). Does not change the result.
        faithful : bool
            No effect (the procedure has no reference publication).
        dtype, device, layout : optional
            See [`Estimator`][optix.Estimator]. Without a layout, the
            trailing `min(ndim, 3)` dims are spatial.
        """
        super().__init__(faithful=faithful, dtype=dtype, device=device)
        self._resolve(faithful, locals())
        opt = self.options
        if opt["center"] not in ("gaussian", "centroid"):
            raise ValueError(
                f"center must be 'gaussian' or 'centroid': {center!r}"
            )
        if opt["shift"] not in ("fourier", "linear"):
            raise ValueError(f"shift must be 'fourier' or 'linear': {shift!r}")
        if opt["weights"] not in ("equal", "brightness"):
            raise ValueError(
                f"weights must be 'equal' or 'brightness': {weights!r}"
            )
        bg = opt["background"]
        if isinstance(bg, str) and bg != "border":
            raise ValueError(
                f"background must be 'border' or a number: {bg!r}"
            )

    def _labeled_for(self, data: ArrayLike) -> Labeled:
        if isinstance(data, Labeled):
            return data
        ndim = len(data.shape)
        return self._labeled(data, Layout.default(ndim, min(ndim, 3)))

    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        self._warn_unused(init)
        data = self._labeled_for(data)
        self.layout = data.layout
        raw = data.array
        layout = data.layout
        backend = self._backend(data)
        group_shape, _, spatial_shape = layout.shape_of(raw)
        ndim = len(spatial_shape)
        if ndim > 3:
            raise ValueError(f"BeadPSF needs 1 to 3 spatial axes, got {ndim}")
        opt = self.options
        radius = [int(r) for r in _axes(opt["radius"], ndim, "radius")]
        sigma = [float(s) for s in _axes(opt["sigma"], ndim, "sigma")]
        mind = int(opt["min_distance"])
        smooth_halo = [max(1, math.ceil(3 * s)) for s in sigma]
        halo = [
            r + mind + h + 1 for r, h in zip(radius, smooth_halo, strict=True)
        ]
        block = opt["block"]
        saturation = _saturation(raw, opt["saturation"])

        # pass 1: noise level of the smoothed data
        reservoirs: dict[tuple[int, ...], Reservoir] = {}
        for chunk in blocks(data, block, smooth_halo, **backend):
            smoothed = gaussian_filter(chunk.data, sigma)[chunk.inner]
            res = reservoirs.setdefault(
                chunk.group, Reservoir(_SAMPLE, seed=0)
            )
            res.update(smoothed.reshape(-1))
        level = {}
        for key, res in reservoirs.items():
            values = res.values
            med = values.median()
            mad = (values - med).abs().median() * _MAD
            level[key] = float(med + opt["threshold"] * mad)

        # pass 2: detect, crop, centre, align and accumulate
        storage_spatial = layout.positions(len(raw.shape))[2]
        dev = backend["device"]
        r_t = torch.tensor(radius, device=dev)
        offsets = torch.stack(
            torch.meshgrid(
                *[torch.arange(-r, r + 1, device=dev) for r in radius],
                indexing="ij",
            ),
            -1,
        )
        size = tuple(2 * r + 1 for r in radius)
        border = torch.zeros(size, dtype=torch.bool, device=dev)
        for i, n in enumerate(size):
            index = [slice(None)] * ndim
            index[i] = [0, n - 1]
            border[tuple(index)] = True
        sums: dict[tuple[int, ...], Tensor] = {}
        counts: dict[tuple[int, ...], int] = {}
        centers: list[Tensor] = []
        max_beads = opt["max_beads"]
        for chunk in blocks(data, block, halo, **backend):
            key = chunk.group
            have = counts.get(key, 0)
            if max_beads is not None and have >= max_beads:
                continue
            x = chunk.data
            peaks = local_maxima(gaussian_filter(x, sigma), mind, level[key])
            if not len(peaks):
                continue
            start = torch.tensor([s.start for s in chunk.inner], device=dev)
            stop = torch.tensor([s.stop for s in chunk.inner], device=dev)
            extent = torch.tensor(x.shape, device=dev)
            in_core = ((peaks >= start) & (peaks < stop)).all(-1)
            fits = ((peaks - r_t >= 0) & (peaks + r_t < extent)).all(-1)
            near = ((peaks[:, None] - peaks[None]).abs() <= r_t).all(-1)
            near.fill_diagonal_(False)
            keep = in_core & fits & ~near.any(1)
            peaks = peaks[keep]
            if not len(peaks):
                continue
            index = peaks.view([-1] + [1] * ndim + [ndim]) + offsets
            crops = x[tuple(index.unbind(-1))]
            ok = torch.ones(len(peaks), dtype=torch.bool, device=dev)
            if saturation is not None:
                ok &= crops.flatten(1).amax(1) < saturation
            if opt["background"] == "border":
                bg = crops[:, border].median(1).values
            else:
                bg = torch.full(
                    (len(peaks),), float(opt["background"]), **backend
                )
            beads = crops - bg.view([-1] + [1] * ndim)
            if opt["center"] == "gaussian":
                c = fit_gaussian(beads)[0]
            else:
                c = threshold_centroid(beads)
            ok &= (c - r_t).abs().amax(-1) <= _MAX_DRIFT
            move = r_t.to(c.dtype) - c
            if opt["shift"] == "fourier":
                aligned = fourier_shift(beads, move)
            else:
                aligned = linear_shift(beads, move)
            total = aligned.flatten(1).sum(1)
            ok &= total > 0
            if max_beads is not None:
                ok &= torch.cumsum(ok.long(), 0) <= max_beads - have
            if not bool(ok.any()):
                continue
            aligned, total, c, peaks = aligned[ok], total[ok], c[ok], peaks[ok]
            if opt["weights"] == "equal":
                aligned = aligned / total.view([-1] + [1] * ndim)
            acc = sums.get(key)
            part = aligned.sum(0)
            sums[key] = part if acc is None else acc + part
            counts[key] = have + len(aligned)
            pos = torch.zeros(
                len(aligned), len(raw.shape), dtype=torch.float64
            )
            for a, idx in enumerate(chunk.index):
                if isinstance(idx, int):
                    pos[:, a] = idx
            for i, a in enumerate(storage_spatial):
                origin = chunk.spatial[i].start - chunk.inner[i].start
                local = peaks[:, i].to(c.dtype) - radius[i] + c[:, i]
                pos[:, a] = (local + origin).double().cpu()
            centers.append(pos)

        groups = list(itertools.product(*[range(g) for g in group_shape]))
        missing = [g for g in groups if g not in sums]
        if missing:
            raise ValueError(
                f"No isolated beads found (groups {missing}): lower "
                f"`threshold`, or check `radius` and `saturation`"
            )
        maps, widths = [], []
        window = None
        if opt["taper"]:
            window = torch.ones(size, **backend)
            for i, n in enumerate(size):
                view = [1] * ndim
                view[i] = n
                w = tukey_window(n, float(opt["taper"]), **backend)
                window = window * w.view(view)
        for g in groups:
            psf = sums[g]
            for axis in opt["symmetrize"]:
                psf = 0.5 * (psf + psf.flip(int(axis)))
            if window is not None:
                psf = psf * window
            psf = psf.clamp(min=0)
            psf = psf / psf.sum()
            maps.append(psf)
            widths.append(fwhm(psf, [r for r in radius]))
        self.estimates = {
            "psf": PSFEstimate(
                psf=torch.stack(maps).reshape(group_shape + size),
                fwhm=torch.stack(widths).reshape(group_shape + (ndim,)),
                count=torch.tensor([counts[g] for g in groups]).reshape(
                    group_shape
                ),
                centers=torch.cat(centers),
                meta={"units": "voxels", "psf_center": "K // 2"},
            )
        }

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        raise NotImplementedError(
            "BeadPSF has no apply(): use estimates['psf'].psf"
        )


def bead_psf(
    data: ArrayLike,
    *,
    layout: Layout | None = None,
    **options: object,
) -> Tensor:
    """Point spread function averaged over images of isolated beads.

    Shortcut for `BeadPSF(**options).fit(data, layout=layout)`; see
    [`BeadPSF`][optix.psf.BeadPSF] for the procedure and the options.

    Parameters
    ----------
    data : array-like or Labeled
        Bead images; without a layout, the trailing `min(ndim, 3)` dims
        are spatial.
    layout : Layout, optional
        Axis roles.
    **options
        Options of [`BeadPSF`][optix.psf.BeadPSF] (`radius`, `sigma`,
        `threshold`, ..., `dtype`, `device`).

    Returns
    -------
    (*group, *K) Tensor
        PSF centred at `K // 2`, normalised to unit sum.
    """
    est = BeadPSF(**options).fit(data, layout=layout)  # type: ignore[arg-type]
    return est.estimates["psf"].psf
