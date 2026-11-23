"""`PoissonGaussian`: the Poisson-Gaussian noise-model fit.

After [Foi et al. (2008)](https://doi.org/10.1109/TIP.2008.2001399).
"""

import math
import warnings
from collections.abc import Callable, Iterator

import numpy as np
import torch
from torch import Tensor

from .._estimator import Estimator, Tags
from ..estimates import SensorEstimate
from ..io import (
    ArrayLike,
    Labeled,
    Layout,
    frame_batches,
    group_frames,
    integer_range,
)
from ._foi import SegmentStats, from_normalized, sensor_estimate
from ._foi_fit import (
    Pairs,
    covariance,
    least_squares_init,
    log_likelihood,
    nelder_mead,
)
from ._foi_local import (
    d_factor,
    d_mad_factor,
    kappa,
    kappa_mad,
    kappa_mad_variance,
    level_labels,
    local_pairs,
    nlevels,
    smoothness_mask,
    wavelet_coefficients,
)

__all__ = ["PoissonGaussian"]

#: ‖φ‖₂² of the 2-D scaling kernel φ = φ₁φ₁ᵀ of Eq. (10): (½)².
PHI_NORM2 = 0.25
#: Times τ is doubled when fewer than two level sets are found.
_TAU_RETRIES = 8


def _is_integer(dtype: object) -> bool:
    """Whether `dtype` (torch or numpy-like) is an integer type."""
    if dtype is None:
        return False
    if isinstance(dtype, torch.dtype):
        return not (
            dtype.is_floating_point or dtype.is_complex or dtype == torch.bool
        )
    try:
        return np.issubdtype(np.dtype(dtype), np.integer)
    except TypeError:
        return False


def bit_depth_range(maximum: float | Tensor) -> int:
    """2ᵏ − 1 for the smallest k ≥ 1 with 2ᵏ − 1 ≥ `maximum`.

    The largest value of the smallest unsigned bit depth that holds the
    data, for example 4095 for 12-bit data with maximum 4000.
    """
    m = max(float(maximum), 1.0)
    k = max(1, math.ceil(math.log2(m + 1)))
    while 2**k - 1 < m:  # guard against round-off of log2
        k += 1
    return 2**k - 1


class PoissonGaussian(Estimator):
    """Poisson–Gaussian noise model σ²(y) = a·y + b fitted from data.

    The variance of an observation z with expectation y (both in ADU) is σ²(y)
    = a·y + b, possibly clipped to the data range. The fit follows [Foi et al.
    (2008)](https://doi.org/10.1109/TIP.2008.2001399): wavelet-domain local
    estimates of (expectation, standard deviation) on level sets of a smoothed
    image, then a maximum-likelihood fit of (a, b) to these pairs that accounts
    for clipping.

    The result is `estimates["sensor"]`, a
    [`SensorEstimate`][optix.estimates.SensorEstimate] with `gain = a`
    and `variance = b` (the variance extrapolated to 0 ADU) and
    `offset = None`: (a, b) alone do not determine the camera offset.
    Fields are shaped `(*group, 1, 1)`. Its `meta` holds, per group,
    the `data_range`, the final edge threshold `tau` and
    `covariance_ok`.

    Attributes
    ----------
    estimates : dict
        `{"sensor": SensorEstimate}` after [`fit`][optix.Estimator.fit].
    covariance : dict of tuple to Tensor
        Per group, the 2×2 large-sample covariance of (a, b) in ADU: the
        inverse Hessian of −log L at the estimate, by central
        differences with steps of half a standard error (not part of
        the paper). All NaN, with a `RuntimeWarning` and
        `meta["covariance_ok"]` False, when no positive-definite
        Hessian is found.
    pairs : dict of tuple to Pairs
        Per group, the level-set estimates (ŷᵢ, σ̂ᵢ) in normalised units
        and their variance factors cᵢ, dᵢ.
    history : History
        log L of the best simplex vertex at each Nelder–Mead iteration
        (all groups in sequence).

    !!! note "Faithful mode"
        One frame per group, sample standard deviations (Sections III
        and V of the paper), clipped model. The opt-in extensions are
        pooling over all frames (`pooled=True`), temporal differencing
        (`differencing=True`) and the MAD estimator of Section VI
        (`robust=True`).

    !!! references "References"
        1. Foi A, Trimeche M, Katkovnik V, Egiazarian K. *Practical
           Poissonian-Gaussian noise modeling and fitting for
           single-image raw-data.* IEEE Trans Image Process
           **17**(10):1737–1754, 2008.
           [doi:10.1109/TIP.2008.2001399](https://doi.org/10.1109/TIP.2008.2001399)

           ??? quote "BibTeX"
               ```bibtex
               @article{foi2008practical,
                 author  = {Foi, Alessandro and Trimeche, Mejdi and
                            Katkovnik, Vladimir and Egiazarian, Karen},
                 title   = {Practical {Poissonian-Gaussian} noise modeling
                            and fitting for single-image raw-data},
                 journal = {IEEE Transactions on Image Processing},
                 volume  = {17},
                 number  = {10},
                 pages   = {1737--1754},
                 year    = {2008},
                 doi     = {10.1109/TIP.2008.2001399}
               }
               ```
        2. Nelder JA, Mead R. *A simplex method for function
           minimization.* Comput J **7**(4):308–313, 1965.
           [doi:10.1093/comjnl/7.4.308](https://doi.org/10.1093/comjnl/7.4.308)

           ??? quote "BibTeX"
               ```bibtex
               @article{nelder1965simplex,
                 author  = {Nelder, J. A. and Mead, R.},
                 title   = {A simplex method for function minimization},
                 journal = {The Computer Journal},
                 volume  = {7},
                 number  = {4},
                 pages   = {308--313},
                 year    = {1965},
                 doi     = {10.1093/comjnl/7.4.308}
               }
               ```
    """

    tags = Tags(
        targets=("sensor",),
        streaming="per-frame",
        noise="poisson-gaussian",
        reference="doi:10.1109/TIP.2008.2001399",
    )
    paper = {
        "data_range": None,
        "clipped": True,
        "robust": False,
        "mad_correction": "paper",
        "delta": 1 / 300,
        "tau": 0.25,
        "eps_reg": 1e-6,
        "frame": 0,
        "pooled": False,
        "differencing": False,
        "max_iter": 500,
        "tol": 1e-10,
    }
    recommended = {**paper, "pooled": True}

    def __init__(
        self,
        *,
        data_range: tuple[float, float] | str | None = None,
        clipped: bool | None = None,
        robust: bool | None = None,
        mad_correction: str | None = None,
        delta: float | None = None,
        tau: float | None = None,
        eps_reg: float | None = None,
        frame: int | None = None,
        pooled: bool | None = None,
        differencing: bool | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        faithful: bool = True,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        layout: Layout | None = None,
    ) -> None:
        """Set up the fit.

        Parameters
        ----------
        data_range : (float, float) or "auto", optional
            Data values (ADU) mapped to 0 and 1: the normalised range of
            Section IV-A, whose extremes are "the maximum and minimum
            pixel values for the considered noisy image format"; with
            `clipped`, also the clipping bounds.

            - Integer data: required, because the dtype rarely is the
              format (12-bit data are stored in `uint16`, and sensors
              saturate below the format maximum). Pass the bounds, for
              example `(0, 4095)`, or `"auto"`.
            - `"auto"` (integer data, opt-in): `(0, 2ᵏ − 1)` for the
              smallest k with 2ᵏ − 1 ≥ max of the first loaded frames of
              each group; a warning reports the inferred range.
            - Floating-point data, by default: the minimum and maximum
              of the first loaded frames of each group (not specified in
              the paper). Pass the range when the data are clipped.
        clipped : bool
            Use the clipped observation model z̃ = max(0, min(z, 1))
            (Sections IV–V). It reduces to the basic model away from the
            bounds. Default True.
        robust : bool
            MAD estimates σ̂ᵐᵃᵈ (Section VI) instead of sample standard
            deviations. Default False. Incompatible with `pooled` (the
            median does not merge across frames).
        mad_correction : {"paper", "exact"}
            Factor (κ̌ᵐᵃᵈ)² of the unbiased variance estimate (κ̌ᵐᵃᵈ σ̂ᵐᵃᵈ)²
            that the robust least-squares initialisation uses (Section
            VI-A.2). `"paper"` (default): 1 + 1/(5n), as printed.
            `"exact"` (extension): 1/(1 + dᵐᵃᵈₙ), which follows from
            E{σ̂ᵐᵃᵈ} = σ and Eq. (58): E{(σ̂ᵐᵃᵈ)²} = σ²(1 + dᵐᵃᵈ). Only
            the initial point changes, not the ML estimate. A Monte Carlo
            check of both factors is in the docstring of
            `optix.sensor._foi_local.kappa_mad_variance`.
        delta : float
            Width Δ of the level sets, Eq. (13). Default 1/300, the value
            of the paper's example (Fig. 4).
        tau : float
            Edge threshold τ of Eq. (12). Not specified in the paper;
            default 0.25, which with the 9×9 kernels used excludes about
            0.05 % of the coefficients of pure white noise. It is
            doubled while fewer than two level sets are found
            (footnote 2).
        eps_reg : float
            ε_reg of Eq. (21), in normalised units. Not specified in the
            paper; default 10⁻⁶.
        frame : int
            Frame (in the flattened sample axes) used by the single-image
            fit. Default 0.
        pooled : bool
            Extension (not in the paper): merge the level-set statistics
            of every frame of a group (streamed with
            [`frame_batches`][optix.io.frame_batches] and
            [`SegmentStats`][optix.sensor.utils.SegmentStats]) before the fit.
            Default False.
        differencing : bool
            Extension (not in the paper): use consecutive frame pairs;
            the level sets and ŷ come from their mean, σ̂ from the
            detail coefficients of their difference divided by √2, which
            removes the scene texture from σ̂. Default False.
        max_iter, tol : int, float
            Nelder–Mead budget and tolerance (not specified in the
            paper). Defaults 500 and 10⁻¹⁰.
        faithful : bool
            Preset: `paper` (True) or `recommended` (False, `pooled`).
        dtype, device : optional
            Working backend of the wavelet-domain processing. The ML fit
            itself is in float64.
        layout : Layout, optional
            Axis roles of the data. Exactly two spatial axes.
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        if self.options["robust"] and self.options["pooled"]:
            raise ValueError("robust=True is incompatible with pooled=True")
        if self.options["mad_correction"] not in ("paper", "exact"):
            raise ValueError(
                "mad_correction must be 'paper' or 'exact', got "
                f"{self.options['mad_correction']!r}"
            )
        rng = self.options["data_range"]
        if isinstance(rng, str) and rng != "auto":
            raise ValueError(
                f"data_range must be (low, high) or 'auto', got {rng!r}"
            )
        self.pairs: dict[tuple[int, ...], Pairs] = {}
        self.covariance: dict[tuple[int, ...], Tensor] = {}

    # -- fit -------------------------------------------------------------
    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        self._warn_unused(init, SensorEstimate)
        layout = self._layout_of(data)
        if len(layout.spatial) != 2:
            raise ValueError("PoissonGaussian needs two spatial axes")
        opt = self.options
        if opt["pooled"]:
            groups = self._pooled_pairs(data)
        else:
            groups = self._single_pairs(data, layout)
        group_shape = self._map_shape(data)[:-2]
        backend = self._backend(data)
        a_map = torch.zeros(group_shape, dtype=torch.float64)
        b_map = torch.zeros(group_shape, dtype=torch.float64)
        ranges, taus, cov_ok = {}, {}, {}
        self.pairs = {}
        self.covariance = {}
        for g, (pairs, low, high, tau) in groups.items():
            an, bn, cov, ok = self._ml(pairs, callback)
            if not ok:
                warnings.warn(
                    f"group {g}: the observed information is not positive "
                    "definite; covariance set to NaN",
                    RuntimeWarning,
                    stacklevel=3,
                )
            scale = high - low
            a_map[g], b_map[g] = from_normalized(an, bn, low, scale)
            jac = torch.tensor(
                [[scale, 0.0], [-scale * low, scale**2]], dtype=torch.float64
            )
            self.covariance[g] = jac @ cov @ jac.T
            self.pairs[g] = pairs
            key = str(list(g))
            ranges[key] = [low, high]
            taus[key] = tau
            cov_ok[key] = ok
            if verbose:
                print(
                    f"group {g}: a={float(a_map[g]):.6g} "
                    f"b={float(b_map[g]):.6g} ({len(pairs.yhat)} sets)"
                )
        shape = (*group_shape, 1, 1)
        meta = {
            "units": "ADU",
            "data_range": ranges,
            "tau": taus,
            "covariance_ok": cov_ok,
        }
        self.estimates = {
            "sensor": sensor_estimate(
                a_map.reshape(shape).to(**backend),
                b_map.reshape(shape).to(**backend),
                meta=meta,
            )
        }

    def _ml(self, pairs: Pairs, callback: Callable | None) -> tuple:
        """LS initialisation, then ML by
        [Nelder–Mead](https://doi.org/10.1093/comjnl/7.4.308) (Section
        III-D.1).

        Returns (a, b, covariance, covariance_ok) in normalised units.
        """
        opt = self.options
        kw = {
            "clipped": opt["clipped"],
            "robust": opt["robust"],
            "eps": opt["eps_reg"],
        }
        a0, b0 = least_squares_init(pairs, **kw)
        scale = float(pairs.vhat.mean())
        step = 0.05 * torch.tensor([max(abs(a0), scale), max(abs(b0), scale)])

        def objective(x: Tensor) -> float:
            value = -log_likelihood(float(x[0]), float(x[1]), pairs, **kw)
            return value if math.isfinite(value) else math.inf

        def record(value: float) -> None:
            self.history.record(-value)
            if callback is not None:
                callback(self)

        x, _, converged, reason = nelder_mead(
            objective,
            torch.tensor([a0, b0], dtype=torch.float64),
            step,
            max_iter=opt["max_iter"],
            tol=opt["tol"],
            callback=record,
        )
        # several groups share one history: converged only if all did
        first = not self.history.reason
        self.history.converged = converged and (
            first or self.history.converged
        )
        if first or not converged:
            self.history.reason = reason
        cov, ok = covariance(objective, x, 1e-3 * step / 0.05)
        return float(x[0]), float(x[1]), cov, ok

    # -- local estimates -------------------------------------------------
    def _range(self, data: ArrayLike, first: Tensor) -> tuple[float, float]:
        """Data range of a group, from its first loaded frames `first`."""
        rng = self.options["data_range"]
        if rng is not None and not isinstance(rng, str):
            low, high = rng
            return float(low), float(high)
        raw = data.array if isinstance(data, Labeled) else data
        integer = _is_integer(getattr(raw, "dtype", None))
        if rng == "auto":
            if not integer:
                raise ValueError("data_range='auto' needs integer data")
            low, high = 0.0, float(bit_depth_range(first.max()))
            if float(first.min()) < 0:
                raise ValueError(
                    "data_range='auto' infers an unsigned bit depth, but "
                    "the data are negative: pass data_range=(low, high)"
                )
            warnings.warn(
                f"data_range='auto': using (0, {high:.0f}) from the "
                "maximum of the data; pass data_range to set it",
                stacklevel=5,
            )
            return low, high
        if integer:
            fmt = integer_range(raw.dtype)
            raise ValueError(
                "integer data need an explicit data_range: the clipping "
                "bounds of the format, for example data_range=(0, 4095) "
                f"for 12-bit data (the dtype range is {fmt}), or "
                "data_range='auto' to infer the bit depth from the data"
            )
        return float(first.min()), float(first.max())

    def _frames(
        self, data: ArrayLike, layout: Layout
    ) -> Iterator[tuple[tuple[int, ...], Tensor]]:
        """(group, (J, H, W) frames) for the single-image fit."""
        frame = self.options["frame"]
        backend = {"dtype": self.dtype, "device": self.device}
        first = group_frames(data, layout, frame, **backend)
        if not self.options["differencing"]:
            for g, image in first:
                yield g, image[None]
            return
        second = group_frames(data, layout, frame + 1, **backend)
        for (g, x1), (_, x2) in zip(first, second, strict=True):
            yield g, torch.stack([x1, x2])

    def _single_pairs(self, data: ArrayLike, layout: Layout) -> dict:
        opt = self.options
        out = {}
        for g, frames in self._frames(data, layout):
            low, high = self._range(data, frames)
            z = self._normalise(frames, low, high)
            tau = opt["tau"]
            for _ in range(_TAU_RETRIES):
                labels, wapp, wdet = self._coefficients(z, tau)
                stats = local_pairs(
                    labels, wapp, wdet, nlevels(opt["delta"]), opt["robust"]
                )
                pairs = self._pairs(stats, z.shape[0], z.dtype)
                if len(pairs.yhat) >= 2:
                    break
                tau *= 2  # footnote 2
            else:
                raise ValueError(f"group {g}: fewer than two level sets")
            out[g] = (pairs, low, high, tau)
        return out

    def _pooled_pairs(self, data: ArrayLike) -> dict:
        """Pooled pairs, doubling τ for groups with < 2 sets (footnote 2)."""
        tau = self.options["tau"]
        out: dict = {}
        todo = None  # all groups
        for _ in range(_TAU_RETRIES):
            stats, ranges, working = self._pooled_stats(data, tau, todo)
            failed = set()
            for g, st in stats.items():
                pairs = self._pairs(st.result(), self._jframes, working)
                if len(pairs.yhat) >= 2:
                    out[g] = (pairs, *ranges[g], tau)
                else:
                    failed.add(g)
            if not failed:
                return out
            todo = failed
            tau *= 2
        raise ValueError(f"groups {sorted(failed)}: fewer than two level sets")

    @property
    def _jframes(self) -> int:
        return 2 if self.options["differencing"] else 1

    def _pooled_stats(
        self, data: ArrayLike, tau: float, todo: set | None
    ) -> tuple[dict, dict, torch.dtype | None]:
        """One streamed pass: merged level-set statistics per group.

        Only the groups in `todo` (all if None) are processed.
        """
        opt = self.options
        segments = nlevels(opt["delta"])
        stats: dict[tuple[int, ...], SegmentStats] = {}
        ranges: dict[tuple[int, ...], tuple[float, float]] = {}
        carry: dict[tuple[int, ...], Tensor] = {}
        chunks = frame_batches(
            data,
            overhead=16.0,
            dtype=self.dtype,
            device=self.device,
        )
        jframes = self._jframes
        working = None
        for chunk in chunks:
            g = chunk.group
            if todo is not None and g not in todo:
                continue
            frames = chunk.data
            if g not in ranges:
                ranges[g] = self._range(data, frames)
                stats[g] = SegmentStats(segments)
            if g in carry:
                frames = torch.cat([carry.pop(g), frames])
            usable = frames.shape[0] - frames.shape[0] % jframes
            if usable < frames.shape[0]:
                carry[g] = frames[usable:].clone()
            if usable == 0:
                continue
            z = self._normalise(frames[:usable], *ranges[g])
            working = z.dtype
            z = z.reshape(-1, jframes, *z.shape[1:]).transpose(0, 1)
            labels, wapp, wdet = self._coefficients(z, tau)
            stats[g].update(labels, wapp, wdet)
        for g, st in stats.items():
            if st.empty:
                raise ValueError(f"group {g}: no complete frame pair")
        return stats, ranges, working

    @staticmethod
    def _normalise(x: Tensor, low: float, high: float) -> Tensor:
        """(x − low) / (high − low): the [0, 1] range of Section IV-A."""
        if not high > low:
            raise ValueError(f"empty data range ({low}, {high})")
        return (x - low) / (high - low)

    def _coefficients(self, z: Tensor, tau: float) -> tuple:
        """Labels, approximation and detail coefficients of the frames.

        `z` is `(J, n, H, W)` (or `(J, H, W)` for one image): J = 1 for
        single frames, J = 2 for differencing, where the level sets and
        zʷᵃᵖᵖ come from the mean of the pair and zʷᵈᵉᵗ from its
        difference divided by √2.
        """
        if z.ndim == 3:
            z = z[:, None]
        mean = z.mean(0)
        wapp, wdet = wavelet_coefficients(mean)
        zsmo, mask = smoothness_mask(wapp, wdet, tau)
        labels = level_labels(zsmo, mask, self.options["delta"])
        if z.shape[0] == 2:
            _, wdet = wavelet_coefficients((z[0] - z[1]) / math.sqrt(2))
        return labels, wapp, wdet

    def _pairs(self, stats: dict, jframes: int, working: torch.dtype) -> Pairs:
        """Pairs and variance factors from level-set statistics.

        Level sets whose σ̂ is zero to working precision are dropped (not
        specified in the paper). They are fully clipped (all detail
        coefficients equal), where the normal model (49) of σ̂ fails
        (Section V-A: the distribution of z̃ʷᵈᵉᵗ then is a Dirac impulse)
        and the likelihood (53) would be unbounded as y → ∞.
        """
        n = stats["count"].cpu()
        yhat = stats["mean"].cpu()
        # Eq. (19), ŷ being the mean of J frames for differencing
        c = PHI_NORM2 / (jframes * n)
        if self.options["robust"]:
            shat = stats["mad"].cpu() / kappa_mad(n)  # Eq. (57)
            exact = self.options["mad_correction"] == "exact"
            vhat = kappa_mad_variance(n, exact) * shat**2  # Section VI-A.2
            d = d_mad_factor(n)  # Eq. (58)
        else:
            s = stats["std"].cpu()
            shat = s / kappa(n)  # Eq. (16)
            vhat = s**2  # κ̌² σ̂², Eq. (28)
            d = d_factor(n)  # Eq. (19)
        keep = shat > 64 * torch.finfo(working).eps
        return Pairs(
            yhat=yhat[keep],
            shat=shat[keep],
            vhat=vhat[keep],
            c=c[keep],
            d=d[keep],
            count=n[keep],
        )
