"""BM3D (Dabov et al. 2007) and BM4D (Maggioni et al. 2013) denoising.

Written from the papers; the authors' code and the `bm3d`/`bm4d`
packages (non-commercial licence) were not used.
"""

import math
from collections.abc import Callable, Sequence
from typing import Literal, Self

import torch
from torch import Tensor

from .._config import resolve_backend
from .._estimator import Estimator, Tags, _shares_memory
from ..estimates import SensorEstimate
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
from ..sensor._anscombe import anscombe, build_table, inverse_anscombe
from ..stats import MinMax, reduce_groups
from ._engine import (
    Stage,
    bm3d_profile,
    bm4d_profile,
    override,
    run_stage,
    stage_halo,
)

__all__ = ["BM3D", "BM4D", "bm3d", "bm3d_halo", "bm4d", "bm4d_halo"]

Output = Literal["final", "basic"]


# -- shared helpers ----------------------------------------------------------
def _check_positive(name: str, value: object) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a number > 0, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return v


def _check_output(output: object) -> None:
    if output not in ("final", "basic"):
        raise ValueError(f"output must be 'final' or 'basic', got {output!r}")


def _check_int(name: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return int(value)


def _stages(
    ndim: int,
    sigma: float,
    profile: str,
    search: int | None,
    step: int | None,
) -> tuple[Stage, Stage]:
    """Stages for σ in paper units, with the `search`/`step` overrides."""
    if ndim == 2:
        ht, wie = bm3d_profile(sigma, profile)
    else:
        ht, wie = bm4d_profile(profile)
    return (
        override(ht, search=search, step=step),
        override(wie, search=search, step=step),
    )


def _unit(ndim: int) -> float:
    """Intensity range of the paper: [0, 255] (BM3D), [0, 1] (BM4D)."""
    return 255.0 if ndim == 2 else 1.0


def _denoise(
    x: Tensor,
    sigma: float,
    stages: tuple[Stage, Stage],
    output: str,
    origin: tuple[int, ...] | None = None,
) -> Tensor:
    """Both stages on one image or volume, paper units."""
    ht, wie = stages
    basic = run_stage(x, sigma, ht, origin=origin)
    if output == "basic":
        return basic
    return run_stage(x, sigma, wie, pilot=basic, origin=origin)


def _halo(stages: tuple[Stage, Stage], output: str) -> int:
    ht, wie = stages
    return stage_halo(ht) + (stage_halo(wie) if output == "final" else 0)


def _functional(
    data: Tensor,
    sigma: float,
    ndim: int,
    data_range: float | None,
    profile: str,
    output: str,
    search: int | None,
    step: int | None,
) -> Tensor:
    sigma = _check_positive("sigma", sigma)
    _check_output(output)
    search = _check_int("search", search)
    step = _check_int("step", step)
    if data.ndim < ndim:
        raise ValueError(f"data needs at least {ndim} dims, got {data.ndim}")
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    if data_range is None:
        data_range = float(data.max() - data.min())
    data_range = _check_positive("data_range", data_range)
    scale = _unit(ndim) / data_range
    stages = _stages(ndim, sigma * scale, profile, search, step)
    shape = data.shape
    flat = data.reshape(-1, *shape[-ndim:])
    out = torch.stack(
        [_denoise(x * scale, sigma * scale, stages, output) for x in flat]
    )
    return (out / scale).reshape(shape)


# -- functional API ----------------------------------------------------------
def bm3d(
    data: Tensor,
    sigma: float,
    *,
    data_range: float | None = None,
    profile: Literal["normal"] = "normal",
    output: Output = "final",
    search: int | None = None,
    step: int | None = None,
) -> Tensor:
    """BM3D denoising of images with white Gaussian noise.

    Model (Dabov et al. 2007): z(x) = y(x) + η(x), η ~ 𝒩(0, σ²) i.i.d.
    Each image is denoised in two steps, each made of grouping by block
    matching, collaborative filtering and aggregation.

    1. **Basic estimate.** For each reference block Z_{x_R} (N₁ × N₁,
       every N_step pixels), block matching in an N_S × N_S window
       collects the blocks with

           d(Z_{x_R}, Z_x) = ‖Υ′(T₂D(Z_{x_R})) − Υ′(T₂D(Z_x))‖₂²/N₁² ≤ τ_match

       (Eqs. (4)-(5); Υ′ hard-thresholds at λ₂D·σ), at most N₂ of them,
       the closest first. The group is transformed by T₃D = T₂D followed
       by a 1-D Haar transform along the group, hard-thresholded at
       λ₃D·σ, and transformed back (Eq. (6)). Its blocks are aggregated
       with the weight w = 1/(σ²·N_har) (Eq. (10); N_har retained
       coefficients) times a Kaiser window (Eq. (12)).
    2. **Final estimate.** Block matching in the basic estimate ŷ
       (d = ‖Ŷ_{x_R} − Ŷ_x‖₂²/N₁² < τ_match, Eq. (7)) gives two groups,
       noisy and basic; the noisy one is shrunk by the empirical Wiener
       coefficients W = |T₃D(Ŷ)|²/(|T₃D(Ŷ)|² + σ²) (Eqs. (8)-(9)) and
       aggregated with w = σ⁻²‖W‖₂⁻² (Eq. (11)).

    The parameters are those of the "Normal Profile" of Table I, in
    8-bit units:

    | | T₂D | N₁ | N₂ | N_step | N_S | β | λ₂D | λ₃D | τ_match |
    |---|---|---|---|---|---|---|---|---|---|
    | step 1, σ ≤ 40 | bior1.5 | 8 | 16 | 3 | 39 | 2 | 0 | 2.7 | 2500 |
    | step 2, σ ≤ 40 | DCT | 8 | 32 | 3 | 39 | 2 | | | 400 |
    | step 1, σ > 40 | DCT | 12 | 16 | 4 | 39 | 2 | 2 | 2.8 | 5000 |
    | step 2, σ > 40 | DCT | 11 | 32 | 6 | 39 | 2 | | | 3500 |

    The search is exhaustive (N_FS = 1, no predictive search). The
    group size is rounded down to a power of two (Section VI), and the
    blocks are ordered by distance, the reference first.

    !!! note "Choices not specified in the paper"
        * Intensities are mapped to the paper's 8-bit range: the data
          and σ are multiplied by 255/`data_range` (which sets τ_match
          and the σ > 40 switch), and the result is mapped back.
          `data_range` defaults to max − min of `data`, which noise
          widens: pass the range of the clean signal when it is known.
        * bior1.5 is the periodised full dyadic transform with each
          basis function scaled to unit norm; see
          [`bior15_matrix`][optix.denoise.utils.bior15_matrix]. The
          distance of step 1 is taken between its coefficients (as in
          the paper's footnote 3).
        * The reference blocks are on a grid of step N_step from the
          first pixel, plus the last block along each axis so that
          every pixel is covered; search windows are clipped at the
          image border; distance ties keep the raster order of the
          window.
        * The 2-D Kaiser window is the outer product of two 1-D
          windows of parameter β.
        * Wiener weights use ‖W‖₂² ≥ the smallest positive float.

    !!! note "Extensions"
        `search` (N_S) and `step` (N_step), when given, replace the
        values of both steps. They trade quality for speed.

    !!! note "Cost and memory"
        The cost is about N_S²·N₁²/N_step² operations per pixel and
        step. The spectra of every block are kept (N₁² values per
        pixel, twice in step 2). The result at a pixel depends on the
        data within [`bm3d_halo`][optix.denoise.bm3d_halo] pixels, which
        [`BM3D`][optix.denoise.BM3D] uses to stream large arrays in
        blocks.

        The computation is vectorised over batches of reference blocks
        (gathers, sorts and scatter-adds) and runs on any torch device.
        On the CPU it is deterministic; on CUDA the aggregation uses
        atomic additions, so results may differ in the last bits.

    !!! references "References"
        1. Dabov K, Foi A, Katkovnik V, Egiazarian K. *Image denoising by
           sparse 3-D transform-domain collaborative filtering.* IEEE Trans
           Image Process **16**(8):2080–2095, 2007.
           [doi:10.1109/TIP.2007.901238](https://doi.org/10.1109/TIP.2007.901238)
        2. Maggioni M, Katkovnik V, Egiazarian K, Foi A. *A nonlocal
           transform-domain filter for volumetric data denoising and
           reconstruction.* IEEE Trans Image Process **22**(1):119–133, 2013.
           [doi:10.1109/TIP.2012.2210725](https://doi.org/10.1109/TIP.2012.2210725)

        ??? quote "BibTeX"
            ```bibtex
            @article{dabov2007image,
              author  = {Dabov, Kostadin and Foi, Alessandro and
                         Katkovnik, Vladimir and Egiazarian, Karen},
              title   = {Image denoising by sparse {3-D} transform-domain
                         collaborative filtering},
              journal = {IEEE Transactions on Image Processing},
              volume  = {16}, number = {8}, pages = {2080--2095},
              year    = {2007}, doi = {10.1109/TIP.2007.901238}
            }
            @article{maggioni2013nonlocal,
              author  = {Maggioni, Matteo and Katkovnik, Vladimir and
                         Egiazarian, Karen and Foi, Alessandro},
              title   = {A nonlocal transform-domain filter for volumetric
                         data denoising and reconstruction},
              journal = {IEEE Transactions on Image Processing},
              volume  = {22}, number = {1}, pages = {119--133},
              year    = {2013}, doi = {10.1109/TIP.2012.2210725}
            }
            ```

    Parameters
    ----------
    data : Tensor
        `(..., H, W)`; each image is denoised separately (floating point;
        other dtypes are converted to the configured dtype).
    sigma : float
        Noise standard deviation σ > 0, in the units of `data`.
    data_range : float, optional
        Intensity range mapped to [0, 255]. Default: max − min of `data`.
    profile : {"normal"}
        Parameter profile of Table I.
    output : {"final", "basic"}
        The final estimate (step 2) or the basic estimate (step 1).
    search : int, optional
        Extension: side N_S of the search window.
    step : int, optional
        Extension: step N_step between reference blocks.

    Returns
    -------
    Tensor
        Of the shape of `data`.
    """
    return _functional(
        data, sigma, 2, data_range, profile, output, search, step
    )


def bm4d(
    data: Tensor,
    sigma: float,
    *,
    data_range: float | None = None,
    profile: Literal["normal", "modified"] = "normal",
    output: Output = "final",
    search: int | None = None,
    step: int | None = None,
) -> Tensor:
    """BM4D denoising of volumes with white Gaussian noise.

    The volumetric extension of [`bm3d`][optix.denoise.bm3d] (Maggioni
    et al. 2013): cubes C of L × L × L voxels are grouped by cube
    matching,

        d(C_{x_i}, C_{x_j}) = ‖C_{x_i} − C_{x_j}‖₂² / L³ ≤ τ_match

    (Eqs. (2)-(3); no pre-filtering), stacked into 4-D groups and
    filtered with a separable 4-D transform: hard thresholding at
    λ₄D·σ, the DC never being thresholded (Eqs. (5)-(6)), with
    aggregation weight w = 1/(σ²·N_har) (Eq. (8)); then empirical Wiener
    filtering guided by the hard-thresholding estimate (Eqs. (9)-(11))
    with w = σ⁻²‖W‖₂⁻² (Eq. (12)). There is no Kaiser window. The 4-D
    transforms are a 3-D bior1.5 (stage 1) or 3-D DCT (stage 2) of each
    cube, then a 1-D Haar transform along the group, whose size is
    rounded down to a power of two.

    Parameters of Table I (data in [0, 1]):

    | | L | M | N_step | N_S | λ₄D | τ_match |
    |---|---|---|---|---|---|---|
    | hard thresholding, normal | 4 | 16 | 3 | 11 | 2.7 | 2.9 |
    | Wiener, normal | 4 | 32 | 3 | 11 | | 0.4 |
    | hard thresholding, modified | 4 | 32 | 3 | 11 | 2.8 | 24.6 |
    | Wiener, modified | 5 | 32 | 3 | 11 | | 6.7 |

    !!! note "Choices not specified in the paper"
        * The values of τ_match in Table I exceed the largest distance
          of Eq. (2) between cubes in [0, 1]: they are applied to the
          sum of squared differences ‖C_{x_i} − C_{x_j}‖₂², that is
          d ≤ τ_match/L³. Per voxel this is 0.045 (normal) and 0.0063
          (Wiener), the values of BM3D's 2500/255² and 400/255².
        * Intensities are mapped to [0, 1] by dividing the data and σ
          by `data_range` (default: max − min of `data`).
        * Reference cubes, search windows, ties and weights as in
          [`bm3d`][optix.denoise.bm3d]. Voxels covered by no cube (only
          with `step` > L) keep the value of the stage's input, as in
          Section III-D of the paper.

    !!! note "Extensions"
        `search` (N_S) and `step` (N_step) replace the values of both
        stages (Table IV of the paper studies them).

    !!! references "References"
        1. Dabov K, Foi A, Katkovnik V, Egiazarian K. *Image denoising by
           sparse 3-D transform-domain collaborative filtering.* IEEE Trans
           Image Process **16**(8):2080–2095, 2007.
           [doi:10.1109/TIP.2007.901238](https://doi.org/10.1109/TIP.2007.901238)
        2. Maggioni M, Katkovnik V, Egiazarian K, Foi A. *A nonlocal
           transform-domain filter for volumetric data denoising and
           reconstruction.* IEEE Trans Image Process **22**(1):119–133, 2013.
           [doi:10.1109/TIP.2012.2210725](https://doi.org/10.1109/TIP.2012.2210725)

        ??? quote "BibTeX"
            ```bibtex
            @article{dabov2007image,
              author  = {Dabov, Kostadin and Foi, Alessandro and
                         Katkovnik, Vladimir and Egiazarian, Karen},
              title   = {Image denoising by sparse {3-D} transform-domain
                         collaborative filtering},
              journal = {IEEE Transactions on Image Processing},
              volume  = {16}, number = {8}, pages = {2080--2095},
              year    = {2007}, doi = {10.1109/TIP.2007.901238}
            }
            @article{maggioni2013nonlocal,
              author  = {Maggioni, Matteo and Katkovnik, Vladimir and
                         Egiazarian, Karen and Foi, Alessandro},
              title   = {A nonlocal transform-domain filter for volumetric
                         data denoising and reconstruction},
              journal = {IEEE Transactions on Image Processing},
              volume  = {22}, number = {1}, pages = {119--133},
              year    = {2013}, doi = {10.1109/TIP.2012.2210725}
            }
            ```

    Parameters
    ----------
    data : Tensor
        `(..., D, H, W)`; each volume is denoised separately.
    sigma : float
        Noise standard deviation σ > 0, in the units of `data`.
    data_range : float, optional
        Intensity range mapped to [0, 1]. Default: max − min of `data`.
    profile : {"normal", "modified"}
        Parameter profile of Table I.
    output : {"final", "basic"}
        The final (Wiener) or the hard-thresholding estimate.
    search : int, optional
        Extension: side N_S of the search cube.
    step : int, optional
        Extension: step N_step between reference cubes.

    Returns
    -------
    Tensor
        Of the shape of `data`.
    """
    return _functional(
        data, sigma, 3, data_range, profile, output, search, step
    )


def bm3d_halo(
    sigma: float,
    data_range: float = 255.0,
    *,
    profile: str = "normal",
    output: Output = "final",
    search: int | None = None,
    step: int | None = None,
) -> int:
    """Halo, in pixels, that makes block-wise BM3D exact.

    Each step reaches 2·⌊N_S/2⌋ + N₁ − 1 pixels: a pixel is covered by
    blocks within N₁ − 1 of it, grouped by references within ⌊N_S/2⌋,
    whose search windows extend ⌊N_S/2⌋ further. The final estimate
    adds the reach of both steps (90 pixels for σ ≤ 40).

    Parameters
    ----------
    sigma, data_range : float
        As in [`bm3d`][optix.denoise.bm3d] (they select the profile).
    profile, output, search, step
        As in [`bm3d`][optix.denoise.bm3d].

    Returns
    -------
    int
    """
    _check_output(output)
    scale = 255.0 / _check_positive("data_range", data_range)
    stages = _stages(
        2,
        _check_positive("sigma", sigma) * scale,
        profile,
        _check_int("search", search),
        _check_int("step", step),
    )
    return _halo(stages, output)


def bm4d_halo(
    *,
    profile: str = "normal",
    output: Output = "final",
    search: int | None = None,
    step: int | None = None,
) -> int:
    """Halo, in voxels, that makes block-wise BM4D exact.

    2·⌊N_S/2⌋ + L − 1 per stage, as in
    [`bm3d_halo`][optix.denoise.bm3d_halo] (26 voxels for the normal
    profile).

    Parameters
    ----------
    profile, output, search, step
        As in [`bm4d`][optix.denoise.bm4d].

    Returns
    -------
    int
    """
    _check_output(output)
    stages = _stages(
        3, 1.0, profile, _check_int("search", search), _check_int("step", step)
    )
    return _halo(stages, output)


# -- estimators --------------------------------------------------------------
class _BlockMatching(Estimator):
    """Shared fit/apply of BM3D and BM4D."""

    _ndim = 2

    def _setup(self) -> None:
        o = self.options
        if o["sigma"] is not None:
            o["sigma"] = _check_positive("sigma", o["sigma"])
        if o["data_range"] is not None:
            o["data_range"] = _check_positive("data_range", o["data_range"])
        _check_output(o["output"])
        o["search"] = _check_int("search", o["search"])
        o["step"] = _check_int("step", o["step"])
        _stages(self._ndim, 1.0, o["profile"], None, None)  # validates
        if o["inverse"] not in (
            "exact",
            "closed-form",
            "asymptotic",
            "algebraic",
        ):
            raise ValueError(f"Unknown inverse {o['inverse']!r}")
        self.is_faithful = o["search"] is None and o["step"] is None
        self._range: float | None = o["data_range"]
        self._vst: tuple[float, float, float] | None = None
        self._table: tuple[Tensor, Tensor, Tensor] | None = None

    # -- fit -----------------------------------------------------------------
    def fit(
        self,
        data: ArrayLike | None = None,
        *,
        layout: Layout | None = None,
        init: object = None,
        **kwargs: object,
    ) -> Self:
        """Set the noise model and the intensity range.

        Gaussian noise: `sigma` given at construction. Poisson-Gaussian
        noise: `init` is a [`SensorEstimate`][optix.estimates.SensorEstimate]
        (or an estimator holding one, such as a fitted
        [`GeneralizedAnscombe`][optix.sensor.GeneralizedAnscombe]): the
        data are stabilised by the generalised Anscombe transform, whose
        noise has unit variance, denoised with σ = 1 and transformed back
        by the inverse selected by `inverse` (default: the exact unbiased
        inverse).

        Parameters
        ----------
        data : array-like, optional
            Read once for its range (max − min, after the transform in
            the Poisson-Gaussian case) when `data_range` was not given;
            otherwise only its layout is recorded.
        layout : Layout, optional
        init : SensorEstimate or Estimator, optional
            Noise parameters (gain α, offset μ, read variance σ²; scalars,
            the means of maps).
        **kwargs
            Ignored (`callback`, `verbose`).

        Returns
        -------
        Estimator
            self
        """
        if data is None:
            self._set_layout(layout)
            self._fit(None, init=init, callback=None, verbose=False)
            return self
        return super().fit(data, layout=layout, init=init)

    def _fit(
        self,
        data: ArrayLike | None,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        self.estimates = {}
        self._vst = None
        self._table = None
        sensor = self._pick(init, SensorEstimate) if init is not None else None
        if init is not None:
            self._warn_unused(init, SensorEstimate)
        if sensor is not None:
            for name in ("offset", "variance", "gain"):
                if getattr(sensor, name) is None:
                    raise ValueError(f"init SensorEstimate lacks {name}")
            alpha = float(sensor.gain.double().mean())
            s = float(sensor.variance.double().mean().clamp(min=0).sqrt())
            mu = float(sensor.offset.double().mean())
            self._vst = (alpha, s, mu)
            self.estimates = {"sensor": sensor}
        elif self.options["sigma"] is None:
            raise ValueError(
                f"{type(self).__name__} needs sigma= (Gaussian noise) or "
                "fit(init=SensorEstimate) (Poisson-Gaussian noise)"
            )
        self._range = self.options["data_range"]
        if self._range is None:
            if data is None:
                raise ValueError(
                    "data_range is not set: give data_range= or fit(data)"
                )
            self._range = self._data_range(data)

    def _data_range(self, data: ArrayLike) -> float:
        layout = self._layout_of(data, self._ndim)
        groups = reduce_groups(data, MinMax(dim=None), layout=layout)
        lo = min(float(r.min) for (r,) in groups.values())
        hi = max(float(r.max) for (r,) in groups.values())
        if self._vst is not None:
            a, s, m = self._vst
            lo, hi = (
                float(anscombe(torch.tensor(v, dtype=torch.float64), a, s, m))
                for v in (lo, hi)
            )
        return _check_positive("the data range", hi - lo)

    # -- apply ---------------------------------------------------------------
    def _sigma(self) -> float:
        return 1.0 if self._vst is not None else self.options["sigma"]

    def _stages_fitted(self) -> tuple[Stage, Stage]:
        o = self.options
        scale = _unit(self._ndim) / self._range
        return _stages(
            self._ndim,
            self._sigma() * scale,
            o["profile"],
            o["search"],
            o["step"],
        )

    def _filter(self, x: Tensor, origin: tuple[int, ...]) -> Tensor:
        scale = _unit(self._ndim) / self._range
        if self._vst is not None:
            a, s, m = self._vst
            x = anscombe(x, a, s, m)
        out = (
            _denoise(
                x * scale,
                self._sigma() * scale,
                self._stages_fitted(),
                self.options["output"],
                origin,
            )
            / scale
        )
        if self._vst is not None:
            a, s, m = self._vst
            method = self.options["inverse"]
            if method == "exact" and self._table is None:
                s2 = torch.tensor([(s / a) ** 2], dtype=torch.float64)
                self._table = build_table(s2)
            out = inverse_anscombe(
                out, a, s, m, method=method, table=self._table
            )
        return out

    def halo(self) -> int:
        """Halo of the block-wise processing (needs a fitted range).

        Returns
        -------
        int
        """
        if self._range is None:
            raise RuntimeError("Call fit before halo()")
        return _halo(self._stages_fitted(), self.options["output"])

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if self._range is None or (
            self._vst is None and self.options["sigma"] is None
        ):
            raise RuntimeError("Call fit before apply")
        if not isinstance(data, Labeled):
            data = self._labeled(data, self._layout_of(data, self._ndim))
        _, _, spatial = data.layout.positions(len(data.shape))
        if len(spatial) != self._ndim:
            raise ValueError(
                f"{type(self).__name__} needs {self._ndim} spatial axes, "
                f"got {len(spatial)}"
            )
        raw = data.array
        target = out.array if isinstance(out, Labeled) else out
        if out is not None and _shares_memory(target, raw):
            raise ValueError(
                "In-place processing with a halo is not possible: use a "
                "separate output."
            )
        out = prepare_out(out, tuple(data.shape), like=raw, dtype=self.dtype)
        k = max(s.size for s in self._stages_fitted()) ** self._ndim
        chunks = blocks(
            data,
            self.options["block"],
            self.halo(),
            overhead=3.0 * k + 8.0,
            contiguous=True,
            align=chunks_of(out),
            dtype=self.dtype,
            device=self.device,
        )
        for chunk in chunks:
            origin = tuple(
                core.start - inner.start
                for core, inner in zip(chunk.spatial, chunk.inner, strict=True)
            )
            x = chunk.data
            if not x.is_floating_point():
                x = x.to(self._backend(x)["dtype"])
            chunk.store(out, self._filter(x, origin)[chunk.inner])
        flush(out)
        return out

    def _extra_state(self) -> dict[str, object]:
        return {
            "range": None
            if self._range is None
            else torch.tensor(self._range, dtype=torch.float64),
            "vst": None
            if self._vst is None
            else torch.tensor(self._vst, dtype=torch.float64),
        }

    def _load_extra_state(self, state: dict[str, object]) -> None:
        r, v = state.get("range"), state.get("vst")
        self._range = None if r is None else float(r)
        self._vst = None if v is None else tuple(float(t) for t in v)
        self._table = None


class BM3D(_BlockMatching):
    """BM3D denoising of images (Dabov et al. 2007).

    Applies [`bm3d`][optix.denoise.bm3d] to each image (the two spatial
    axes of the layout; other axes are samples or groups). The noise is
    white Gaussian with a known σ, or Poisson-Gaussian with the
    parameters of a [`SensorEstimate`][optix.estimates.SensorEstimate],
    in which case the image is denoised after the generalised Anscombe
    transform (Mäkitalo & Foi 2013,
    [doi:10.1109/TIP.2012.2202675](https://doi.org/10.1109/TIP.2012.2202675))
    with σ = 1, and transformed back by its exact unbiased inverse.

    [`fit`][optix.Estimator.fit] sets the noise model and, unless
    `data_range` is given, reads the data once for its range.
    [`apply`][optix.Estimator.apply] streams blocks with the halo of
    [`bm3d_halo`][optix.denoise.bm3d_halo] (the reference grid is that
    of the whole image), so arrays larger than memory (zarr, memmap)
    give the whole-image result.

    !!! note
        Faithful mode is the Normal Profile of the paper. `search` and
        `step` are extensions; `data_range`, `output`, `inverse` and
        `block` do not change the method.

    !!! references "References"
        1. Dabov K, Foi A, Katkovnik V, Egiazarian K. *Image denoising by
           sparse 3-D transform-domain collaborative filtering.* IEEE Trans
           Image Process **16**(8):2080–2095, 2007.
           [doi:10.1109/TIP.2007.901238](https://doi.org/10.1109/TIP.2007.901238)
        2. Maggioni M, Katkovnik V, Egiazarian K, Foi A. *A nonlocal
           transform-domain filter for volumetric data denoising and
           reconstruction.* IEEE Trans Image Process **22**(1):119–133, 2013.
           [doi:10.1109/TIP.2012.2210725](https://doi.org/10.1109/TIP.2012.2210725)

        ??? quote "BibTeX"
            ```bibtex
            @article{dabov2007image,
              author  = {Dabov, Kostadin and Foi, Alessandro and
                         Katkovnik, Vladimir and Egiazarian, Karen},
              title   = {Image denoising by sparse {3-D} transform-domain
                         collaborative filtering},
              journal = {IEEE Transactions on Image Processing},
              volume  = {16}, number = {8}, pages = {2080--2095},
              year    = {2007}, doi = {10.1109/TIP.2007.901238}
            }
            @article{maggioni2013nonlocal,
              author  = {Maggioni, Matteo and Katkovnik, Vladimir and
                         Egiazarian, Karen and Foi, Alessandro},
              title   = {A nonlocal transform-domain filter for volumetric
                         data denoising and reconstruction},
              journal = {IEEE Transactions on Image Processing},
              volume  = {22}, number = {1}, pages = {119--133},
              year    = {2013}, doi = {10.1109/TIP.2012.2210725}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.denoise import BM3D
    >>> clean = torch.linspace(0, 255, 48).expand(48, 48)
    >>> noisy = clean + 20 * torch.randn(48, 48)
    >>> den = BM3D(sigma=20, data_range=255).fit().apply(noisy)
    >>> bool((den - clean).square().mean() < (noisy - clean).square().mean())
    True
    """

    tags = Tags(
        targets=("denoised",),
        streaming="halo",
        noise="gaussian",
        reference="doi:10.1109/TIP.2007.901238",
    )
    paper = {
        "sigma": None,
        "data_range": None,
        "profile": "normal",
        "output": "final",
        "search": None,
        "step": None,
        "inverse": "exact",
        "block": None,
    }
    recommended = dict(paper)
    _ndim = 2

    def __init__(
        self,
        *,
        sigma: float | None = None,
        data_range: float | None = None,
        profile: Literal["normal"] | None = None,
        output: Output | None = None,
        search: int | None = None,
        step: int | None = None,
        inverse: str | None = None,
        block: Sequence[int | None] | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the denoiser.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation σ of white Gaussian noise, in data units.
            Omit it to give a `SensorEstimate` to `fit` instead.
        data_range : float, optional
            Intensity range mapped to the paper's [0, 255] (not specified
            in the paper, whose images are 8-bit). Default: max − min of
            the data given to `fit` (after the variance-stabilising
            transform with a `SensorEstimate`).
        profile : {"normal"}
            Profile of Table I. Default: `"normal"`.
        output : {"final", "basic"}
            The final estimate (default) or the basic estimate of step 1.
        search : int, optional
            Extension: side N_S of the search window (paper: 39).
        step : int, optional
            Extension: step N_step between reference blocks (paper: 3,
            or 4 and 6 for σ > 40).
        inverse : {"exact", "closed-form", "asymptotic", "algebraic"}
            Inverse of the Anscombe transform (see
            [`inverse_anscombe`][optix.sensor.inverse_anscombe]).
            Default: `"exact"`.
        block : sequence of int or None, optional
            Core block size per spatial axis (default: from the memory
            budget). The result does not depend on it.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        self._setup()


class BM4D(_BlockMatching):
    """BM4D denoising of volumes (Maggioni et al. 2013).

    Applies [`bm4d`][optix.denoise.bm4d] to each volume (the three
    spatial axes of the layout). Noise models, fitting and streaming are
    those of [`BM3D`][optix.denoise.BM3D]: blocks are z-slabs (or
    smaller) with the halo of [`bm4d_halo`][optix.denoise.bm4d_halo].

    !!! note
        Faithful mode is the normal or modified profile of Table I.
        `search` and `step` are extensions.

    !!! references "References"
        1. Dabov K, Foi A, Katkovnik V, Egiazarian K. *Image denoising by
           sparse 3-D transform-domain collaborative filtering.* IEEE Trans
           Image Process **16**(8):2080–2095, 2007.
           [doi:10.1109/TIP.2007.901238](https://doi.org/10.1109/TIP.2007.901238)
        2. Maggioni M, Katkovnik V, Egiazarian K, Foi A. *A nonlocal
           transform-domain filter for volumetric data denoising and
           reconstruction.* IEEE Trans Image Process **22**(1):119–133, 2013.
           [doi:10.1109/TIP.2012.2210725](https://doi.org/10.1109/TIP.2012.2210725)

        ??? quote "BibTeX"
            ```bibtex
            @article{dabov2007image,
              author  = {Dabov, Kostadin and Foi, Alessandro and
                         Katkovnik, Vladimir and Egiazarian, Karen},
              title   = {Image denoising by sparse {3-D} transform-domain
                         collaborative filtering},
              journal = {IEEE Transactions on Image Processing},
              volume  = {16}, number = {8}, pages = {2080--2095},
              year    = {2007}, doi = {10.1109/TIP.2007.901238}
            }
            @article{maggioni2013nonlocal,
              author  = {Maggioni, Matteo and Katkovnik, Vladimir and
                         Egiazarian, Karen and Foi, Alessandro},
              title   = {A nonlocal transform-domain filter for volumetric
                         data denoising and reconstruction},
              journal = {IEEE Transactions on Image Processing},
              volume  = {22}, number = {1}, pages = {119--133},
              year    = {2013}, doi = {10.1109/TIP.2012.2210725}
            }
            ```

    """

    tags = Tags(
        targets=("denoised",),
        streaming="halo",
        noise="gaussian",
        reference="doi:10.1109/TIP.2012.2210725",
    )
    paper = dict(BM3D.paper)
    recommended = dict(paper)
    _ndim = 3

    def __init__(
        self,
        *,
        sigma: float | None = None,
        data_range: float | None = None,
        profile: Literal["normal", "modified"] | None = None,
        output: Output | None = None,
        search: int | None = None,
        step: int | None = None,
        inverse: str | None = None,
        block: Sequence[int | None] | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the denoiser.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation σ of white Gaussian noise, in data units.
            Omit it to give a `SensorEstimate` to `fit` instead.
        data_range : float, optional
            Intensity range mapped to the paper's [0, 1]. Default: max −
            min of the data given to `fit` (after the variance-stabilising
            transform with a `SensorEstimate`).
        profile : {"normal", "modified"}
            Profile of Table I. Default: `"normal"`.
        output : {"final", "basic"}
            The final (Wiener) estimate (default) or the
            hard-thresholding estimate.
        search : int, optional
            Extension: side N_S of the search cube (paper: 11).
        step : int, optional
            Extension: step N_step between reference cubes (paper: 3).
        inverse : {"exact", "closed-form", "asymptotic", "algebraic"}
            Inverse of the Anscombe transform. Default: `"exact"`.
        block : sequence of int or None, optional
            Core block size per spatial axis (default: from the memory
            budget). The result does not depend on it.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        self._setup()
