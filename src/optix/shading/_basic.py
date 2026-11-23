"""BaSiC background and shading correction (Peng et al., 2017).

After [Peng et al. 2017](https://doi.org/10.1038/ncomms14836).
"""

import itertools
import math
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from .._estimator import Estimator, Tags
from ..estimates import ShadingEstimate
from ..io import (
    ArrayLike,
    Labeled,
    Layout,
    WritableArrayLike,
    as_tensor,
    chunk_length,
    frame_batches,
)
from ..stats import Reservoir
from ._basic_solvers import (
    VARIANTS,
    BaSiCResult,
    Dct2,
    baseline_basicpy,
    baseline_l1,
    basic_ladm,
    default_lambdas,
    resize,
    resize_matrix,
)

__all__ = ["BaSiC"]

_BASELINES = ("paper", "basicpy")
_SAMPLINGS = ("stride", "reservoir")

_PRESET: dict[str, object] = {
    "working_size": 128,
    "darkfield": False,
    "timelapse": False,
    "lambda_flat": None,
    "lambda_dark": None,
    "epsilon": 0.1,
    "tol": 1e-3,
    "max_iterations": 500,
    "max_reweight": 10,
    "reweight_tol": 1e-3,
    "mu_coef": 12.5,
    "rho": 1.5,
    "max_mu_coef": 1e7,
    "max_frames": 512,
    "sampling": "stride",
    "seed": 0,
    "variant": "paper",
    "normalize_weights": True,
    "baseline_method": "paper",
    "sort_intensity": True,
    "baseline_norm": 0.0,
}


class BaSiC(Estimator):
    """BaSiC: shading and baseline by low-rank + sparse decomposition.

    Image formation for frame i of a group (one flat and dark field per
    group, shared by its samples):

        yᵢ = f ⊙ (xᵢ + bᵢ) + d

    with a smooth flat field f (mean 1), a smooth dark field d, a
    spatially constant baseline bᵢ (time-lapse data) and the latent
    image xᵢ. The corrected frame is

        x̂ᵢ = (yᵢ − d) / f − bᵢ + b_{norm}

    where b_{norm} is an arbitrary background: 0 removes the background
    (fluorescence), meanᵢ bᵢ keeps the raw intensity range (bright field).
    ([Peng et al. 2017](https://doi.org/10.1038/ncomms14836): I = y, S = f, D =
    d, Bᵢ = bᵢ, F = x, I_B = L, I_R = E.)

    **Flat and dark field.** The fitting frames, sorted at each pixel
    over the frames and resized to the working size (128 × 128), form the
    measurement matrix Y (one column per sorted frame). BaSiC decomposes
    it into a low-rank part L = f bᵀ + z, with
    z = d_Z (1 − f) + d_R, and a sparse residual E:

        min  ‖W ⊙ E‖₁ + λ_f ‖𝓕(f)‖₁ + λ_d ‖𝓕(d_R)‖₁ + λ_d ‖d_R‖₁
        s.t. Y = L + E,  d_Z ∈ [0, min Y]

    with 𝓕 the orthonormal 2-D DCT. The problem is solved by the
    linearised augmented Lagrangian method (LADM), with reweighted-L1
    weights W = 1 / (|E / L| + ε), scaled to mean 1. The default
    regularisation is
    λ_f = ‖𝓕(ȳ)‖₁ / 800 and λ_d = ‖𝓕(ȳ)‖₁ / 2000, with ȳ the per-pixel
    mean frame divided by its mean. The fields are resized back
    (bilinear) to the image size; d = d_Z + d_R.

    **Baseline** (`timelapse=True`). After sorting, the fitted bᵢ belong
    to the sorted frames, so bᵢ is estimated in a second step on every
    unsorted frame (at the working size) with f and d fixed:
    min_b ‖W ⊙ (yᵢ − bᵢ f − d)‖₁, reweighted as above.

    !!! note
        The fit reads the `max_frames` fitting frames of each group (a
        representative subset, as suggested by the authors for long
        sequences), plus one pass over the stack with `timelapse=True`.
        [`apply`][optix.Estimator.apply] works on frame batches. Sorting needs
        all the fitting frames of a strip of rows at once: the strips are
        sized to the memory budget, and each fitting frame is read once per
        strip.

    !!! note "Implementation choices"
        The inner LADM loop stops when ‖Y − L − E‖_F / ‖Y‖_F < `tol`
        or after `max_iterations`. The penalty starts at μ₀ = 12.5 / ‖Y‖₂
        and grows by ρ = 1.5, capped at 10⁷ μ₀ to stay finite. Images are
        resized by bilinear interpolation with anti-aliasing, after
        sorting. The baseline step solves each reweighted problem exactly
        (weighted median).

    !!! note "Interpretations of the paper"
        - 𝓕 is the orthonormal 2-D DCT: the "Fourier transform" of the
          Supplementary Information refers to the DCT of the authors'
          implementation.
        - The weights of Eq. (9) are scaled to mean 1, which the λ of
          Supplementary Note 4 are taken to assume
          (`normalize_weights=False` uses Eq. (9) as printed).
        - The regions "S ≥ 1" and "S < 1" of the dark-field update are
          read as f ≥ mean(f) and f < mean(f), f being unnormalised
          there.
        - The frames are sorted at full resolution and then resized to
          the working size.

    !!! references "References"
        1. Peng T, Thorn K, Schroeder T, Wang L, Theis FJ, Marr C, Navab N.
           *A BaSiC tool for background and shading correction of optical
           microscopy images.* Nat Commun **8**:14836, 2017.
           [doi:10.1038/ncomms14836](https://doi.org/10.1038/ncomms14836)

        ??? quote "BibTeX"
            ```bibtex
            @article{peng2017basic,
              author  = {Peng, Tingying and Thorn, Kurt and Schroeder, Timm
                         and Wang, Lichao and Theis, Fabian J. and Marr,
                         Carsten and Navab, Nassir},
              title   = {A {BaSiC} tool for background and shading
                         correction of optical microscopy images},
              journal = {Nature Communications},
              volume  = {8},
              pages   = {14836},
              year    = {2017},
              doi     = {10.1038/ncomms14836},
            }
            ```

    !!! note "Extensions (off by default)"
        - `sort_intensity=False`: fit the unsorted frames; `fit_baseline_`
          then holds the fitted baselines.
        - `variant="basicpy"`: the solver of
          [BaSiCPy](https://github.com/peng-lab/BaSiCPy) 2.0.0
          (`fitting_mode="approximate"`): weights divided by the frame mean
          of L and normalised to mean 1, bᵢ clipped at 0, f started at 1,
          the dark field fitted on the frames with bᵢ < 1 with a 10× smaller
          threshold, and λ_d = λ_f / 10 by default. With
          `sort_intensity=False` and `reweight_tol=1e-2` it reproduces
          [BaSiCPy](https://github.com/peng-lab/BaSiCPy).
        - `baseline_method="basicpy"`:
          [BaSiCPy](https://github.com/peng-lab/BaSiCPy)'s `fit_only_baseline`,
          a reweighted-L1 ALM trusting only the pixels below each frame's 10 %
          quantile, run per batch. It is biased low where the flat field
          varies (about −11 % with a flat of ±25 %).
        - `sampling="reservoir"`: random fitting frames.
        - `normalize_weights=False`: the weights of Eq. (9) without scaling.

    Attributes
    ----------
    estimates : dict
        `{"shading": ShadingEstimate}`: `flat` and `dark` (None without
        `darkfield`) shaped `(*group, *spatial)`; `baseline` shaped
        `(*group, *sample)` (None without `timelapse`), in units of the
        flat-corrected signal. `meta["flat_norm"] == "mean"` (normalised
        at the working size, before resizing) and
        `meta["baseline_units"] == "signal"`.
        [`apply`][optix.Estimator.apply] also handles a baseline in
        raw units (see
        [`ShadingEstimate`][optix.estimates.ShadingEstimate]).
    fit_frames_ : (k,) Tensor
        Flat sample indices (storage order) of the fitting frames.
    fit_baseline_ : (*group, k) Tensor or None
        Without `sort_intensity`: the baselines of the fitting frames from
        the decomposition itself. None after sorting.

    Examples
    --------
    >>> import torch
    >>> from optix.models import random_shading, simulate_shading
    >>> truth = random_shading((64, 64), seed=0)
    >>> x = 100 + 50 * torch.rand(40, 64, 64)
    >>> data = simulate_shading(x, truth)
    >>> basic = BaSiC(working_size=64).fit(data)
    >>> flat = basic.estimates["shading"].flat
    >>> corrected = basic.apply(data)
    """

    tags = Tags(
        targets=("shading",),
        streaming="per-frame",
        modality=("fluorescence", "brightfield"),
        noise="none",
        many_frames=True,
        reference="doi:10.1038/ncomms14836",
    )
    paper = dict(_PRESET)
    recommended = {**_PRESET, "sort_intensity": False}

    def __init__(
        self,
        *,
        working_size: int | tuple[int, int] | None = None,
        darkfield: bool | None = None,
        timelapse: bool | None = None,
        lambda_flat: float | None = None,
        lambda_dark: float | None = None,
        epsilon: float | None = None,
        tol: float | None = None,
        max_iterations: int | None = None,
        max_reweight: int | None = None,
        reweight_tol: float | None = None,
        mu_coef: float | None = None,
        rho: float | None = None,
        max_mu_coef: float | None = None,
        max_frames: int | None = None,
        sampling: str | None = None,
        seed: int | None = None,
        variant: str | None = None,
        normalize_weights: bool | None = None,
        baseline_method: str | None = None,
        sort_intensity: bool | None = None,
        baseline_norm: float | str | None = None,
        faithful: bool = True,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        layout: Layout | None = None,
    ) -> None:
        """Set the options; unset ones come from the preset.

        Parameters
        ----------
        working_size : int or (int, int) or None
            Size the frames are resized to for fitting (None: no
            resizing). Default 128.
        darkfield : bool
            Also estimate the dark field d.
        timelapse : bool
            Estimate (in [`fit`][optix.Estimator.fit]) and remove (in
            [`apply`][optix.Estimator.apply]) the per-frame baseline bᵢ.
        lambda_flat, lambda_dark : float, optional
            λ_f and λ_d, the weights of the DCT sparsity of f and of d_R.
            Default: derived from the data, per group.
        epsilon : float
            ε of the reweighting. Default 0.1.
        tol, max_iterations : float, int
            Inner stop: ‖Y − L − E‖_F / ‖Y‖_F < `tol`. Defaults 10⁻³
            and 500.
        max_reweight, reweight_tol : int, float
            Reweighting stop: relative L1 change of f (and of d) below
            `reweight_tol`, or `max_reweight` steps. Defaults 10 and 10⁻³.
        mu_coef, rho, max_mu_coef : float
            Penalty schedule: μ₀ = `mu_coef` / ‖Y‖₂ and
            μ ← min(ρ μ, `max_mu_coef` μ₀).
        max_frames : int or None
            Number of fitting frames per group (None: all). Default 512.
        sampling : {"stride", "reservoir"}
            How the fitting frames are chosen when there are more than
            `max_frames`: evenly spaced, or a uniform random sample drawn
            with [`Reservoir`][optix.stats.Reservoir] (seeded by `seed`).
        seed : int
            Seed of the reservoir.
        variant : {"paper", "basicpy"}
            Solver details: the paper's, or
            [BaSiCPy](https://github.com/peng-lab/BaSiCPy) 2.0.0's.
        normalize_weights : bool
            Scale the reweighting weights to mean 1 (default True; the
            "basicpy" variant always does).
        baseline_method : {"paper", "basicpy"}
            Baseline estimator used with `timelapse=True`.
        sort_intensity : bool
            Sort each pixel over the fitting frames before the fit.
        baseline_norm : float or "mean"
            Background b_{norm} added back by
            [`apply`][optix.Estimator.apply] with `timelapse` ("mean":
            the mean fitted baseline of the group), in the units of the
            baseline. Default 0.
        faithful : bool
            True: the `paper` preset. False: the `recommended` preset,
            which fits the unsorted frames (`sort_intensity=False`), more
            accurate when the background drifts between frames.
        dtype, device, layout : optional
            See [`Estimator`][optix.Estimator]. The layout must have two
            spatial axes; each group (e.g. channel) gets its own fields.
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        opt = self.options
        for name, allowed in (
            ("variant", VARIANTS),
            ("baseline_method", _BASELINES),
            ("sampling", _SAMPLINGS),
        ):
            if opt[name] not in allowed:
                raise ValueError(
                    f"{name} must be one of {allowed}, got {opt[name]!r}"
                )
        norm = opt["baseline_norm"]
        if isinstance(norm, str) and norm != "mean":
            raise ValueError(
                f"baseline_norm must be a number or 'mean': {norm}"
            )
        if opt["max_frames"] is not None and opt["max_frames"] < 1:
            raise ValueError(f"max_frames must be >= 1: {opt['max_frames']}")
        self.fit_frames_: Tensor | None = None
        self.fit_baseline_: Tensor | None = None

    # -- helpers -----------------------------------------------------------
    def _working_shape(
        self, spatial: tuple[int, ...]
    ) -> tuple[int, int] | None:
        size = self.options["working_size"]
        if size is None:
            return None
        if isinstance(size, int | np.integer):
            return (int(size), int(size))
        size = tuple(int(s) for s in size)
        if len(size) != 2:
            raise ValueError(f"working_size must be an int or a pair: {size}")
        return size

    def _select(self, n: int) -> Tensor:
        """Flat sample indices of the fitting frames (sorted)."""
        k = self.options["max_frames"]
        if k is None or n <= k:
            return torch.arange(n)
        if self.options["sampling"] == "stride":
            return torch.div(torch.arange(k) * n, k, rounding_mode="floor")
        reservoir = Reservoir(k, seed=self.options["seed"])
        step = 1 << 20
        for start in range(0, n, step):
            stop = min(start + step, n)
            reservoir.update(torch.arange(start, stop, dtype=torch.float64))
        return reservoir.values.long().sort().values

    @staticmethod
    def _frame(
        raw: ArrayLike,
        layout: Layout,
        group: tuple[int, ...],
        k: int,
        backend: dict,
        rows: slice | None = None,
    ) -> Tensor:
        """Frame `k` (flat sample index) of `group`, spatial in layout order.

        `rows` restricts the first spatial axis (in layout order).
        """
        shape = tuple(int(s) for s in raw.shape)
        gpos, spos, xpos = layout.positions(len(shape))
        index: list[object] = [slice(None)] * len(shape)
        for a, i in zip(gpos, group, strict=True):
            index[a] = int(i)
        sample_shape = [shape[a] for a in spos]
        if spos:
            for a, i in zip(
                spos, np.unravel_index(k, sample_shape), strict=True
            ):
                index[a] = int(i)
        if rows is not None:
            index[xpos[0]] = rows
        frame = as_tensor(raw[tuple(index)], **backend)
        stored = sorted(xpos)
        perm = [stored.index(a) for a in xpos]
        if perm != list(range(len(perm))):
            frame = frame.permute(perm)
        return frame

    def _stack(
        self,
        raw: ArrayLike,
        layout: Layout,
        group: tuple[int, ...],
        frames: Tensor,
        backend: dict,
        spatial: tuple[int, int],
        size: tuple[int, int] | None,
    ) -> Tensor:
        """Working stack `(k, h, w)` of one group, sorted then resized."""
        if not self.options["sort_intensity"]:
            return torch.stack(
                [
                    resize(
                        self._frame(raw, layout, group, k, backend)[None], size
                    )[0]
                    for k in frames.tolist()
                ]
            )
        H, W = spatial
        h, w = (H, W) if size is None else size
        dtype, device = backend["dtype"], backend["device"]
        k = len(frames)
        # strips of rows holding every fitting frame: read, sort, resize
        rows = chunk_length((k, W), dtype=dtype, overhead=3.0, maximum=H)
        rh = resize_matrix(H, h, dtype=dtype, device=device)
        rw = resize_matrix(W, w, dtype=dtype, device=device)
        out = torch.zeros(k, h, w, dtype=dtype, device=device)
        for r0 in range(0, H, rows):
            r1 = min(r0 + rows, H)
            strip = torch.stack(
                [
                    self._frame(raw, layout, group, f, backend, slice(r0, r1))
                    for f in frames.tolist()
                ]
            )
            strip = strip.sort(dim=0).values
            if size is None:
                out[:, r0:r1] = strip
            else:
                out += rh[:, r0:r1] @ strip @ rw.T
        return out

    def _solve(self, D: Tensor) -> tuple[BaSiCResult, float, float]:
        opt = self.options
        lam_f, lam_d = opt["lambda_flat"], opt["lambda_dark"]
        if lam_f is None or lam_d is None:
            dct = Dct2(*D.shape[1:], dtype=D.dtype, device=D.device)
            auto_f, auto_d = default_lambdas(D, dct)
            lam_f = auto_f if lam_f is None else lam_f
            if lam_d is None:
                lam_d = auto_d if opt["variant"] == "paper" else 0.1 * lam_f
        result = basic_ladm(
            D,
            lambda_flat=lam_f,
            lambda_dark=lam_d,
            darkfield=opt["darkfield"],
            epsilon=opt["epsilon"],
            tol=opt["tol"],
            max_iterations=opt["max_iterations"],
            max_reweight=opt["max_reweight"],
            reweight_tol=opt["reweight_tol"],
            mu_coef=opt["mu_coef"],
            rho=opt["rho"],
            max_mu_coef=opt["max_mu_coef"],
            variant=opt["variant"],
            normalize_weights=opt["normalize_weights"],
        )
        return result, lam_f, lam_d

    def _baseline(self, D: Tensor, flat: Tensor, dark: Tensor) -> Tensor:
        opt = self.options
        if opt["baseline_method"] == "paper":
            return baseline_l1(
                D,
                flat,
                dark,
                epsilon=opt["epsilon"],
                max_reweight=opt["max_reweight"],
                reweight_tol=opt["reweight_tol"],
            )
        return baseline_basicpy(
            D,
            flat,
            dark,
            epsilon=opt["epsilon"],
            tol=opt["tol"],
            max_iterations=opt["max_iterations"],
            mu_coef=opt["mu_coef"],
            rho=opt["rho"],
            max_mu_coef=opt["max_mu_coef"],
        )

    def _small(self, maps: Tensor, size: tuple[int, int] | None) -> Tensor:
        """`(*group, H, W)` maps resized to the working size."""
        lead = maps.shape[:-2]
        flat = maps.reshape(-1, *maps.shape[-2:])
        return resize(flat, size).reshape(*lead, *_resized_shape(maps, size))

    # -- fit / apply -------------------------------------------------------
    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        self._warn_unused(init)
        layout = self._layout_of(data)
        raw = data.array if isinstance(data, Labeled) else data
        group_shape, sample_shape, spatial = layout.shape_of(raw)
        if len(spatial) != 2:
            raise ValueError(
                f"BaSiC needs two spatial axes, the layout has {len(spatial)}"
            )
        backend = self._backend(data)
        size = self._working_shape(spatial)
        n = math.prod(sample_shape)
        if n < 1:
            raise ValueError("BaSiC needs at least one frame per group")
        frames = self._select(n)
        opt = self.options
        flats, darks, fit_base, converged = [], [], [], []
        for group in itertools.product(*(range(g) for g in group_shape)):
            D = self._stack(
                raw, layout, group, frames, backend, tuple(spatial), size
            )
            result, lam_f, lam_d = self._solve(D)
            for it, res, change in zip(
                result.iterations,
                result.residuals,
                result.changes,
                strict=True,
            ):
                self.history.record(
                    -res,
                    change=change,
                    iterations=it,
                    lambda_flat=lam_f,
                    lambda_dark=lam_d,
                )
            converged.append(result.converged)
            flats.append(resize(result.flat[None], spatial)[0])
            darks.append(resize(result.dark[None], spatial)[0])
            fit_base.append(result.baseline)
            if verbose:
                print(
                    f"BaSiC group {group}: {len(result.iterations)} "
                    f"reweightings, iterations {result.iterations}, "
                    f"converged={result.converged}"
                )
            if callback is not None:
                callback(self)
        self.history.converged = all(converged)
        self.history.reason = (
            "reweighting converged"
            if all(converged)
            else f"max_reweight={opt['max_reweight']} reached (some group)"
        )
        flat = torch.stack(flats).reshape(*group_shape, *spatial)
        dark = torch.stack(darks).reshape(*group_shape, *spatial)
        self.fit_frames_ = frames
        self.fit_baseline_ = (
            None
            if opt["sort_intensity"]
            else torch.stack(fit_base).reshape(*group_shape, len(frames))
        )
        baseline = None
        if opt["timelapse"]:
            baseline = self._fit_baseline(data, flat, dark, size)
        self.estimates = {
            "shading": ShadingEstimate(
                flat=flat,
                dark=dark if opt["darkfield"] else None,
                baseline=baseline,
                meta={
                    "flat_norm": "mean",
                    "flat_norm_at": "working_size",
                    "baseline_units": "signal",
                    "model": "y = flat * (x + baseline) + dark",
                },
            )
        }

    def _fit_baseline(
        self,
        data: ArrayLike,
        flat: Tensor,
        dark: Tensor,
        size: tuple[int, int] | None,
    ) -> Tensor:
        """Per-frame baseline in one pass, `(*group, *sample)`."""
        layout = self._layout_of(data)
        group_shape, sample_shape, _ = layout.shape_of(data)
        small_flat = self._small(flat, size)
        small_dark = self._small(dark, size)
        out = torch.zeros(
            *group_shape, *sample_shape, dtype=flat.dtype, device=flat.device
        )
        chunks = frame_batches(
            data,
            ndim=None if isinstance(data, Labeled) else 2,
            dtype=flat.dtype,
            device=flat.device,
            overhead=8.0,
        )
        for chunk in chunks:
            D = resize(chunk.data, size)
            b = self._baseline(
                D, small_flat[chunk.group], small_dark[chunk.group]
            )
            chunk.store(out, b.to(out), into="samples")
        return out

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if "shading" not in self.estimates:
            raise RuntimeError("BaSiC is not fitted: call fit first")
        est = self.estimates["shading"]
        layout = self._layout_of(data)
        group_shape, sample_shape, spatial = layout.shape_of(data)
        flat = est.flat
        dark = torch.zeros_like(flat) if est.dark is None else est.dark
        if tuple(flat.shape) != tuple(group_shape) + tuple(spatial):
            raise ValueError(
                f"Data have maps of shape {group_shape + spatial}, the fit "
                f"has {tuple(flat.shape)}"
            )
        timelapse = self.options["timelapse"]
        stored = est.baseline
        if stored is not None and tuple(stored.shape) != tuple(
            group_shape
        ) + tuple(sample_shape):
            stored = None
        size = self._working_shape(spatial)
        small_flat = small_dark = None
        if timelapse and stored is None:
            small_flat = self._small(flat, size)
            small_dark = self._small(dark, size)
        norm = self.options["baseline_norm"]
        if timelapse and norm == "mean":
            if est.baseline is None:
                raise ValueError(
                    "baseline_norm='mean' needs the fitted baseline"
                )
            ng = len(group_shape)
            dims = tuple(range(ng, est.baseline.ndim))
            norm = est.baseline.mean(dims) if dims else est.baseline
        elif timelapse:
            norm = torch.full(group_shape, float(norm), dtype=flat.dtype)

        units = est.baseline_units

        def fn(chunk: object, batch: Tensor) -> Tensor:
            model = ShadingEstimate(
                flat=flat[chunk.group].to(batch),
                dark=dark[chunk.group].to(batch),
            )
            if not timelapse:
                return model.correct(batch)
            if stored is not None:
                b = stored[chunk.group + chunk.samples].reshape(-1)
                b_units = units
            else:
                # computed on the fly, in units of the signal
                b = self._baseline(
                    resize(batch, size),
                    small_flat[chunk.group].to(batch),
                    small_dark[chunk.group].to(batch),
                )
                b_units = "signal"
            return model.correct(
                batch,
                b.to(batch),
                units=b_units,
                norm=norm[chunk.group].to(batch),
            )

        return self._apply_frames(data, out, fn, overhead=3.0)

    # -- state -------------------------------------------------------------
    def _extra_state(self) -> dict[str, object]:
        return {
            "fit_frames": self.fit_frames_,
            "fit_baseline": self.fit_baseline_,
        }

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.fit_frames_ = state.get("fit_frames")
        self.fit_baseline_ = state.get("fit_baseline")


def _resized_shape(
    maps: Tensor, size: tuple[int, int] | None
) -> tuple[int, int]:
    """Spatial shape of `maps` after `resize` to `size`."""
    return tuple(maps.shape[-2:]) if size is None else tuple(size)
