"""Photon-transfer curve (EMVA 1288 Release 4.0 Linear)."""

import math
from collections.abc import Callable, Sequence
from typing import Self

import torch
from torch import Tensor

from .._config import resolve_backend
from .._estimator import Tags
from ..estimates import SensorEstimate
from ..io import ArrayLike, Layout, frame_batches
from ..stats import Moments
from ._base import CalibrationEstimator, PixelStats, _layout
from ._scmos import gain_regression

__all__ = [
    "PairStats",
    "PhotonTransferCurve",
    "bspline_slopes",
    "emva_highpass",
    "linearity_error",
    "pair_stats",
]

#: EMVA 1288 R4.0 Linear, Section 6.6 / C.4: below this dark temporal
#: variance (DN^2) only an upper limit of the read noise is given.
MIN_DARK_VARIANCE = 0.24


class PairStats:
    """EMVA statistics of one stack: two-image and per-pixel moments.

    Attributes
    ----------
    mu0, mu1 : (*group, 1, ..., 1) Tensor
        Spatial means of the first two images, Eq. (16).
    temporal : (*group, 1, ..., 1) Tensor
        Temporal variance from the two images, Eq. (18).
    spatial : (*group, 1, ..., 1) Tensor
        Spatial variance from the two images, Eq. (32).
    vmax, nmax : (*group, 1, ..., 1) Tensor
        Largest value in the two images and its number of occurrences.
    npix : int
        Number of pixels of one image.
    stack : PixelStats
        Per-pixel mean and unbiased variance over all images, Eq. (33),
        (65).
    """

    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)

    @property
    def mean(self) -> Tensor:
        """Mean gray value, Eq. (16): (μ[0] + μ[1])/2."""
        return 0.5 * (self.mu0 + self.mu1)


def pair_stats(
    data: ArrayLike,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    out_dtype: torch.dtype | None = None,
) -> PairStats:
    """EMVA 1288 statistics of a stack, in one streaming pass.

    With y[0], y[1] the first two images (in storage order) of every group
    and NM pixels (EMVA 1288 R4.0 Linear, Eqs. (16), (18), (32)):

        μ[k] = (1/NM)·∑ y[k],
        σ²_y = (1/2NM)·∑ (y[0] − y[1])² − ½·(μ[0] − μ[1])²,
        s²_y = (1/NM)·∑ (y[0] − μ[0])·(y[1] − μ[1]),

    plus the per-pixel moments of all images (Welford/Pebay).

    Parameters
    ----------
    data : array-like or Labeled
        Stack with at least two images per group.
    dtype, device, out_dtype : optional
        Working backend and dtype of the results.

    Returns
    -------
    PairStats
    """
    layout = _layout(data)
    group_shape, _, spatial_shape = layout.shape_of(data.shape)
    nsp = len(spatial_shape)
    npix = math.prod(spatial_shape)
    moments: dict[tuple[int, ...], Moments] = {}
    held: dict[tuple[int, ...], list[Tensor]] = {}
    pair: dict[tuple[int, ...], tuple[Tensor, ...]] = {}
    for chunk in frame_batches(data, dtype=dtype, device=device):
        key = chunk.group
        if key not in moments:
            moments[key] = Moments(2)
            held[key] = []
        moments[key].update(chunk.data)
        frames = held[key]
        if len(frames) < 2:
            frames.extend(chunk.data[: 2 - len(frames)].clone().unbind(0))
            if len(frames) == 2:
                a, b = frames
                m0, m1 = a.mean(), b.mean()
                temporal = 0.5 * (a - b).square().mean() - 0.5 * (m0 - m1) ** 2
                spatial = ((a - m0) * (b - m1)).mean()
                both = torch.stack([a, b])
                vmax = both.max()
                pair[key] = (
                    m0,
                    m1,
                    temporal,
                    spatial,
                    vmax,
                    (both == vmax).sum(),
                )
                held[key] = []
    if not moments:
        raise ValueError("No data to reduce")
    if len(pair) < len(moments):
        raise ValueError("Every level needs at least two frames per group")
    if out_dtype is None:
        out_dtype = resolve_backend(dtype, device)["dtype"]
    dev = next(iter(moments.values())).mean.device
    shape = group_shape + spatial_shape
    count = torch.zeros(group_shape, dtype=out_dtype, device=dev)
    mean = torch.full(shape, torch.nan, dtype=out_dtype, device=dev)
    var = torch.full_like(mean, torch.nan)
    scal = torch.full(
        (6, *group_shape), torch.nan, dtype=out_dtype, device=dev
    )
    for key, m in moments.items():
        count[key] = m.count.to(out_dtype)
        mean[key] = m.mean.to(out_dtype)
        var[key] = m.var().to(out_dtype)
        for i, v in enumerate(pair[key]):
            scal[(i, *key)] = v.to(out_dtype)
    one = group_shape + (1,) * nsp
    scal = scal.reshape(6, *one)
    stack = PixelStats(count.reshape(one), mean, var, nsp)
    return PairStats(
        mu0=scal[0],
        mu1=scal[1],
        temporal=scal[2],
        spatial=scal[3],
        vmax=scal[4],
        nmax=scal[5],
        npix=npix,
        stack=stack,
    )


def _box(x: Tensor, size: int, dim: int) -> Tensor:
    """Moving average of `size` samples along `dim` (valid part)."""
    return x.unfold(dim, size, 1).mean(-1)


def emva_highpass(x: Tensor, nspatial: int = 2) -> Tensor:
    """EMVA 1288 R4.0 highpass filter of a nonuniformity image.

    Section 8.1: the image is smoothed by a 7×7 box filter, an 11×11 box
    filter and a 3×3 binomial filter, one after the other, and the
    smoothed image is subtracted from the original. Every spatial dim is
    filtered.

    !!! note "optix convention"
        Only the part where the filters are exact is returned: 9 pixels
        are removed at each border, by analogy with the border rule of
        Release 3.1 (Appendix C.5); Release 4.0 does not state it.

    Parameters
    ----------
    x : (..., *spatial) Tensor
    nspatial : int
        Number of trailing spatial dims.

    Returns
    -------
    (..., *spatial − 18) Tensor
    """
    smooth = x
    binom = torch.tensor([0.25, 0.5, 0.25], dtype=x.dtype, device=x.device)
    for d in range(-nspatial, 0):
        smooth = _box(smooth, 7, d)
        smooth = _box(smooth, 11, d)
        smooth = (smooth.unfold(d, 3, 1) * binom).sum(-1)
    crop = tuple(
        slice(None) if i < x.ndim - nspatial else slice(9, x.shape[i] - 9)
        for i in range(x.ndim)
    )
    return x[crop] - smooth


def _beta3(t: Tensor) -> Tensor:
    a = t.abs()
    return torch.where(
        a < 1,
        2 / 3 - a * a + 0.5 * a**3,
        torch.where(a <= 2, (2 - a).clamp_min(0) ** 3 / 6, 0.0),
    )


def _dbeta3(t: Tensor) -> Tensor:
    # Eq. (49) of R4.0 General prints the outer branch without its minus
    # sign; this is the exact derivative of Eq. (45).
    a = t.abs()
    return torch.where(
        a < 1,
        -2 * t * (1 - 0.75 * a),
        torch.where(
            a <= 2, -torch.sign(t) * 0.5 * (2 - a).clamp_min(0) ** 2, 0.0
        ),
    )


def bspline_slopes(
    x: Tensor, y: Tensor, xmax: float, intervals: int = 6
) -> Tensor:
    """Slopes of a cubic B-spline regression (EMVA 1288 R4.0 General).

    Section 6.6 of Release 4.0 General, Eqs. (44)-(49): over [0, xmax]
    split into P intervals of width Δ,

        y(x) = ∑_{p=0}^{P+2} a_p·β₃(x/Δ − (p − 1)),

    fitted by least squares; the slope is the analytic derivative.

    Parameters
    ----------
    x, y : (Q,) Tensor
        Data points (Q > P + 3).
    xmax : float
        Right end of the regression range.
    intervals : int
        P (the standard recommends 6 to 8).

    Returns
    -------
    (Q,) Tensor
        dy/dx at `x`.
    """
    p = torch.arange(intervals + 3, dtype=x.dtype, device=x.device)
    delta = xmax / intervals
    t = x[:, None] / delta - (p - 1)
    a = torch.linalg.lstsq(_beta3(t), y[:, None]).solution
    return (_dbeta3(t) @ a)[:, 0] / delta


def linearity_error(
    y: Tensor, h: Tensor, ysat: float, lo: float = 0.05, hi: float = 0.95
) -> tuple[float, Tensor, tuple[float, float]]:
    """Linearity error LE (EMVA 1288 R4.0 Linear, Section 6.9).

    Weighted least squares of y = a₀ + a₁·H with weights 1/y²
    (Eqs. (59)-(61)) over the points with y ∈ [0.05, 0.95]·y_sat, relative
    deviations δ_y = 100·(y − (a₀ + a₁·H))/(a₀ + a₁·H) (Eq. (62)) and
    LE = mean |δ_y| (Eq. (63)).

    Parameters
    ----------
    y : (Q,) Tensor
        Mean photo-induced gray values μ_y − μ_y.dark.
    h : (Q,) Tensor
        Exposures (any unit, e.g. photons per pixel).
    ysat : float
        μ_y.sat − μ_y.dark.
    lo, hi : float
        Range, as fractions of `ysat`.

    Returns
    -------
    le : float
        Linearity error in percent.
    delta : (Q,) Tensor
        δ_y in percent (NaN outside the range).
    coef : (float, float)
        (a₀, a₁).
    """
    m = (y >= lo * ysat) & (y <= hi * ysat)
    yy, hh = y[m], h[m]
    s_h_y = (hh / yy).sum()
    s_h_y2 = (hh / yy**2).sum()
    s_h2_y2 = (hh**2 / yy**2).sum()
    s_1_y = (1 / yy).sum()
    s_1_y2 = (1 / yy**2).sum()
    delta_ = s_h_y2**2 - s_h2_y2 * s_1_y2
    a0 = (s_h_y * s_h_y2 - s_h2_y2 * s_1_y) / delta_
    a1 = (s_h_y2 * s_1_y - s_h_y * s_1_y2) / delta_
    fit = a0 + a1 * h
    d = torch.where(m, 100 * (y - fit) / fit, torch.nan)
    return float(d[m].abs().mean()), d, (float(a0), float(a1))


def _ols(x: Tensor, y: Tensor) -> tuple[float, float, float]:
    """Least-squares line y = b + k·x: (k, b, standard error of k)."""
    xm, ym = x.mean(), y.mean()
    sxx = (x - xm).square().sum()
    k = ((x - xm) * (y - ym)).sum() / sxx
    b = ym - k * xm
    n = x.numel()
    if n > 2:
        s2 = (y - b - k * x).square().sum() / (n - 2)
        se = float((s2 / sxx).sqrt())
    else:
        se = math.nan
    return float(k), float(b), se


def _tolist(x: Tensor) -> object:
    return x.squeeze().tolist()


class PhotonTransferCurve(CalibrationEstimator):
    """Photon-transfer curve of a camera (EMVA 1288 Release 4.0 Linear).

    The temporal variance of the digital signal is linear in its mean
    (Eq. (15)):

        σ²_y = K²·σ²_d + σ²_q + K·(μ_y − μ_y.dark),   σ²_q = 1/12 DN²,

    with K the overall system gain (DN per electron) and σ_d the read
    noise in electrons; σ²_y.dark = K²·σ²_d + σ²_q.

    Procedure (faithful mode, Sections 2.5, 4.2, 6.6, 6.9 and 8):

    - every level and the dark stack: μ_y and σ²_y from the **first two
      images** (Eqs. (16), (18), with the mean-difference correction and
      1/(2NM) normalisation), see [`pair_stats`][];
    - saturation μ_y.sat: the largest mean gray value whose two images
      have at most `saturation_fraction` (0.2 %) of their pixels at the
      maximum value (Section 6.6); the maximum is `max_value`, or the
      largest value observed in the levels;
    - K: least-squares line, **with unknown offset and slope**, of
      σ²_y − σ²_y.dark on μ_y − μ_y.dark over the levels from the smallest
      up to `fit_range` (70 %) of μ_y.sat − μ_y.dark (Eq. (50));
    - linearity of the photon-transfer curve: local slopes K_i of a cubic
      B-spline regression ([`bspline_slopes`][]) over
      [0, μ_y.sat − μ_y.dark] at the points of the gain fit; the linear
      model holds if mean |K_i − K|/K < 3 % (Eq. (51));
    - read noise σ_d = √(σ²_y.dark − σ²_q)/K (Eq. (53)); if
      σ²_y.dark < 0.24 DN², only the upper limit σ_d < 0.40/K (Eq. (54));
    - DSNU and PRNU (Section 8, Eqs. (33)-(36), (65)-(67)) from the
      highpass-filtered ([`emva_highpass`][]) mean images ⟨y⟩ of the dark
      stack and of the level closest to 50 % saturation:
      s²_y = s²_measured − σ²_y.stack/L, DSNU = s_y.dark/K (electrons),
      PRNU = √(s²_y.50 − s²_y.dark)/(μ_y.50 − μ_y.dark);
    - with `exposure` (photons per pixel): responsivity R (Eq. (49), slope
      through the origin over the gain-fit range), quantum efficiency
      η = R/K (Eq. (52)) and the linearity error LE
      ([`linearity_error`][], Eqs. (58)-(63)).

    Statistics are per group (Section 6.7: each channel separately). Every
    stack is streamed once.

    !!! note "Interpretations (optix choices, not stated by the standard)"
        - The saturation is chosen among the measured levels (the standard
          sets the exposure to reach 0.1-0.2 %).
        - The highpass border crop (see [`emva_highpass`][]).
        - The temporal-noise correction σ²_y.stack/L of Eq. (36) is applied
          to the highpass-filtered image unchanged.
        - The B-spline check uses the mean of |K_i − K|/K (the text) while
          Eq. (51) prints a sum; Eq. (49) of R4.0 General misses a minus
          sign in β₃′, the exact derivative is used.
        - The dark values are taken at a single exposure time.

    Attributes
    ----------
    curve : dict of str to Tensor
        `"mean"` and `"variance"` (`(levels, *group)`) of the
        photon-transfer curve, dark subtracted, and `"used"` (bool), the
        levels of the gain fit.

    Estimates: `"sensor"`: `offset` (per-pixel dark mean, DN), `variance`
    (read-noise variance σ²_y.dark − σ²_q, DN²) and `gain` (K, per group,
    or per pixel with `per_pixel`). `meta` holds `gain_stderr`,
    `gain_offset`, `read_noise_e`, `read_noise_upper_limit`, `dsnu_e`,
    `prnu`, `saturation` (DN above dark), `linearity_slope_dev` and
    `linear` (Eq. (51)), and with `exposure`: `responsivity`,
    `quantum_efficiency`, `linearity_error` (percent).

    !!! references "References"
        1. European Machine Vision Association. *EMVA Standard 1288:
           Standard for Characterization of Image Sensors and Cameras,
           Release 4.0 Linear.* 16 June 2021.
           [emva.org](https://www.emva.org/standards-technology/emva-1288/)
        2. European Machine Vision Association. *EMVA Standard 1288:
           Standard for Characterization of Image Sensors and Cameras,
           Release 4.0 General.* 16 June 2021.
           [emva.org](https://www.emva.org/standards-technology/emva-1288/)

        ??? quote "BibTeX"
            ```bibtex
            @techreport{emva1288linear40,
              author      = {{European Machine Vision Association}},
              title       = {{EMVA} Standard 1288: Standard for
                             Characterization of Image Sensors and Cameras,
                             Release 4.0 Linear},
              institution = {EMVA}, year = {2021}, month = jun,
              url = {https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf}
            }
            @techreport{emva1288general40,
              author      = {{European Machine Vision Association}},
              title       = {{EMVA} Standard 1288: Standard for
                             Characterization of Image Sensors and Cameras,
                             Release 4.0 General},
              institution = {EMVA}, year = {2021}, month = jun,
              url = {https://www.emva.org/wp-content/uploads/EMVA1288General_4.0Release.pdf}
            }
            ```
    """

    tags = Tags(
        targets=("sensor",),
        streaming="per-pixel-stats",
        noise="poisson-gaussian",
        many_frames=False,
        reference="EMVA 1288 Release 4.0",
    )
    paper = {
        "fit_range": 0.7,
        "quantization": 1 / 12,
        "max_value": None,
        "saturation_fraction": 0.002,
        "saturation": "histogram",
        "temporal": "pair",
        "highpass": True,
        "spline_intervals": 6,
        "per_pixel": False,
    }
    recommended = dict(paper)

    def __init__(
        self,
        *,
        fit_range: float | None = None,
        quantization: float | None = None,
        max_value: float | None = None,
        saturation_fraction: float | None = None,
        saturation: str | None = None,
        temporal: str | None = None,
        highpass: bool | None = None,
        spline_intervals: int | None = None,
        per_pixel: bool | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: object = None,
    ) -> None:
        """Configure the photon-transfer analysis.

        Parameters
        ----------
        fit_range : float
            Upper end of the gain fit, as a fraction of the saturation.
            Paper: 0.7.
        quantization : float
            Quantisation noise variance σ²_q (DN²). Paper: 1/12 (integer
            data). Extension: 0 for data that were never quantised.
        max_value : float, optional
            Maximum possible gray value (e.g. 2ᵏ − 1). Paper: None (found
            in the data, Section 6.6).
        saturation_fraction : float
            Largest fraction of pixels at the maximum below saturation.
            Paper: 0.002 (0.1-0.2 %).
        saturation : {"histogram", "variance"}
            `"histogram"`: Release 4.0 rule (see the class). `"variance"`
            (extension; the Release 3.0 rule): the level of maximal
            temporal variance.
        temporal : {"pair", "stack"}
            `"pair"`: two-image estimator (Eq. (18)). `"stack"`
            (extension): spatial mean of the per-pixel unbiased temporal
            variance over all images (Eq. (65)) and the mean over all
            images.
        highpass : bool
            Highpass-filter the nonuniformity images (Section 8.1).
            Paper: True.
        spline_intervals : int
            Number P of B-spline intervals of the linearity check.
            Paper: 6 (6 to 8 recommended).
        per_pixel : bool
            Extension: also fit a per-pixel gain map (per-pixel
            photon-transfer curve, as in
            [`SCMOSCalibration`][optix.sensor.SCMOSCalibration]) over the
            same levels. Paper: False.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        if o["saturation"] not in ("histogram", "variance"):
            raise ValueError("saturation must be 'histogram' or 'variance'")
        if o["temporal"] not in ("pair", "stack"):
            raise ValueError("temporal must be 'pair' or 'stack'")
        self.curve: dict[str, Tensor] = {}
        self._exposure: Sequence[float] | None = None

    def fit(
        self,
        data: ArrayLike | Sequence[ArrayLike] | None,
        *,
        dark: object = None,
        exposure: Sequence[float] | Tensor | None = None,
        layout: Layout | None = None,
        init: object = None,
        callback: Callable | None = None,
        verbose: bool = False,
    ) -> Self:
        """Fit the photon-transfer curve.

        Parameters
        ----------
        data : sequence of array-like
            One stack per illumination level (at least two images per
            level and group).
        dark : array-like
            Dark stack (at least two images; more images give better DSNU
            estimates, the standard uses L = 100 to 500).
        exposure : sequence of float, optional
            Mean number of photons per pixel of every level (for the
            responsivity, quantum efficiency and linearity error).
        layout, init, callback, verbose
            See [`Estimator.fit`][optix.Estimator.fit].

        Returns
        -------
        PhotonTransferCurve
            self
        """
        self._exposure = exposure
        try:
            return super().fit(
                data,
                dark=dark,
                layout=layout,
                init=init,
                callback=callback,
                verbose=verbose,
            )
        finally:
            self._exposure = None

    def _pair(self, data: ArrayLike) -> PairStats:
        backend = self._backend(data)
        return pair_stats(
            data,
            dtype=self.dtype,
            device=self.device,
            out_dtype=backend["dtype"],
        )

    def _mean_var(self, st: PairStats) -> tuple[Tensor, Tensor]:
        if self.options["temporal"] == "pair":
            return st.mean, st.temporal
        s = st.stack
        return s.spatial_mean(s.mean), s.spatial_mean(s.var)

    def _nonuniformity(self, st: PairStats) -> tuple[Tensor, Tensor]:
        """Spatial variance of the mean image, Eqs. (33)-(36), and its mean."""
        s = st.stack
        img = s.mean
        mu = s.spatial_mean(img)
        if self.options["highpass"]:
            img = emva_highpass(img, s.nspatial)
        dims = s.spatial_dims
        meas = (
            (img - img.mean(dims, keepdim=True))
            .square()
            .mean(dims, keepdim=True)
        )
        return meas - s.spatial_mean(s.var) / s.count, mu

    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        o = self.options
        levels = data if isinstance(data, list) else [data]
        if self._dark_input is None:
            raise ValueError("PhotonTransferCurve needs a dark stack: dark=")
        dark = self._pair(self._dark_stack())
        stats = [self._pair(level) for level in levels]
        mu_d, var_d = self._mean_var(dark)
        mv = [self._mean_var(st) for st in stats]
        x = torch.stack([m for m, _ in mv]) - mu_d  # (L, *group, 1..)
        v = torch.stack([s for _, s in mv]) - var_d
        nl = x.shape[0]
        gshape = x.shape[1:]
        # -- saturation (Section 6.6)
        if o["saturation"] == "variance":
            sat = v.argmax(0, keepdim=True)
        else:
            vmax = torch.stack([st.vmax for st in stats])
            nmax = torch.stack([st.nmax for st in stats])
            top = (
                vmax.amax(0, keepdim=True)
                if o["max_value"] is None
                else torch.full_like(vmax[:1], float(o["max_value"]))
            )
            frac = torch.where(vmax >= top, nmax, 0.0) / (2 * stats[0].npix)
            ok = frac <= o["saturation_fraction"]
            if not ok.any(0).all():
                raise ValueError(
                    "Every level is saturated: no level has at most "
                    f"{o['saturation_fraction']:.2%} of pixels at the maximum"
                )
            sat = torch.where(ok, x, -torch.inf).argmax(0, keepdim=True)
        x_sat = x.gather(0, sat)
        used = x <= o["fit_range"] * x_sat
        # -- per-group regressions
        xf = x.reshape(nl, -1)
        vf = v.reshape(nl, -1)
        uf = used.reshape(nl, -1)
        xs = x_sat.reshape(-1)
        ng = xf.shape[1]
        gain = torch.empty(ng, dtype=x.dtype, device=x.device)
        offs, ses, devs = [], [], []
        resp, qe, le = [], [], []
        h = None
        if self._exposure is not None:
            h = torch.as_tensor(self._exposure).to(x).reshape(-1)
            if h.numel() != nl:
                raise ValueError("exposure needs one value per level")
        for g in range(ng):
            m = uf[:, g]
            k, b, se = _ols(xf[m, g], vf[m, g])
            gain[g] = k
            offs.append(b)
            ses.append(se)
            inside = (xf[:, g] >= 0) & (xf[:, g] <= xs[g])
            q = int(inside.sum())
            if q > o["spline_intervals"] + 3:
                slopes = bspline_slopes(
                    xf[inside, g],
                    vf[inside, g],
                    float(xs[g]),
                    o["spline_intervals"],
                )
                ki = slopes[m[inside]]
                devs.append(float((ki - k).abs().mean() / abs(k)))
            else:
                devs.append(math.nan)
            if h is not None:
                r = float((xf[m, g] * h[m]).sum() / (h[m] * h[m]).sum())
                resp.append(r)
                qe.append(r / k)
                le.append(linearity_error(xf[:, g], h, float(xs[g]))[0])
        gain = gain.reshape(gshape)
        # -- temporal dark noise (Eqs. (53)-(54))
        qv = float(o["quantization"])
        limited = (var_d < MIN_DARK_VARIANCE) & (qv > 0)
        read_var = torch.where(
            limited, 0.49**2 - qv, (var_d - qv).clamp_min(0)
        )
        read_e = torch.where(limited, 0.40 / gain, read_var.sqrt() / gain)
        # -- DSNU / PRNU (Section 8)
        s2_d, _ = self._nonuniformity(dark)
        half = (x - 0.5 * x_sat).abs().argmin(0, keepdim=True)
        s2_50 = torch.empty_like(s2_d)
        mu_50 = torch.empty_like(s2_d)
        hf = half.reshape(-1)
        s2f, muf = s2_50.reshape(-1), mu_50.reshape(-1)
        cache: dict[int, tuple[Tensor, Tensor]] = {}
        for g in range(ng):
            i = int(hf[g])
            if i not in cache:
                cache[i] = self._nonuniformity(stats[i])
            s2, mu = cache[i]
            s2f[g] = s2.reshape(-1)[g]
            muf[g] = mu.reshape(-1)[g]
        mu_dark_img = dark.stack.spatial_mean(dark.stack.mean)
        prnu = (s2_50 - s2_d).clamp_min(0).sqrt() / (mu_50 - mu_dark_img)
        dsnu = s2_d.clamp_min(0).sqrt() / gain
        gain_out = gain
        if o["per_pixel"]:
            keep = used.reshape(nl, -1).all(1)
            sel = keep.nonzero().flatten().tolist()
            a = (
                torch.stack([stats[i].stack.mean for i in sel])
                - dark.stack.mean
            )
            bb = (
                torch.stack([stats[i].stack.var for i in sel]) - dark.stack.var
            )
            gain_out, _ = gain_regression(a, bb)
        meta = {
            "units": "ADU",
            "gain_stderr": _tolist(torch.tensor(ses).reshape(gshape)),
            "gain_offset": _tolist(torch.tensor(offs).reshape(gshape)),
            "read_noise_e": _tolist(read_e),
            "read_noise_upper_limit": _tolist(limited),
            "dsnu_e": _tolist(dsnu),
            "prnu": _tolist(prnu),
            "saturation": _tolist(x_sat[0]),
            "linearity_slope_dev": _tolist(torch.tensor(devs).reshape(gshape)),
            "linear": _tolist(torch.tensor(devs).reshape(gshape) < 0.03),
            "quantization": qv,
            "levels": nl,
        }
        if h is not None:
            meta["responsivity"] = _tolist(torch.tensor(resp).reshape(gshape))
            meta["quantum_efficiency"] = _tolist(
                torch.tensor(qe).reshape(gshape)
            )
            meta["linearity_error"] = _tolist(torch.tensor(le).reshape(gshape))
        self.estimates = {
            "sensor": SensorEstimate(
                offset=dark.stack.mean,
                variance=read_var,
                gain=gain_out,
                meta=meta,
            )
        }
        self.curve = {"mean": x, "variance": v, "used": used}

    def _extra_state(self) -> dict[str, object]:
        return {"curve": self.curve}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.curve = state.get("curve", {})
