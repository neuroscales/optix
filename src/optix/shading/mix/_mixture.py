"""Mixture models of sensor, shading and tissue classes."""

import itertools
import math
import warnings
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from ..._estimator import Estimator, Tags
from ...estimates import Classification, SensorEstimate, ShadingEstimate
from ...io import ArrayLike, Labeled, Layout, WritableArrayLike, prepare_out
from ...io._arrays import chunks_of, flush
from ._gamma import GammaComponent
from ._model import MixtureFit, detect_quantization, is_integer_valued
from ._overlap import Overlap, TileOverlap

__all__ = ["GammaMixture"]

_NOISE = ("full", "constant")
_OUTPUTS = ("corrected", "foreground", "classes", "mean")

_PRESET: dict[str, object] = {
    "nclass": 1,
    "noise": "full",
    "order": (8, 8),
    "gain": None,
    "nodes": 24,
    "nu_bounds": (1.0, 1e3),
    "quantization": 1.0,
    "max_iterations": 100,
    "tol": 1e-6,
    "learn_modulation": True,
    "learn_alpha": True,
    "lr": 1.0,
    "max_halvings": 4,
    "max_log_step": 0.5,
    "bg_fraction": 0.5,
    "threshold": 3.0,
    "spread_floor": False,
    "concentration": 1.0,
    "v_prior": None,
    "v_dof": 5.0,
    "lam": 1.0,
    "overlap": None,
}


class _AxisView:
    """Lazy view of an array-like with some axes fixed to an index."""

    def __init__(self, array: ArrayLike, fixed: dict[int, int]) -> None:
        self.array = array
        self._fixed = dict(fixed)
        shape = tuple(int(s) for s in array.shape)
        self._ndim = len(shape)
        self.shape = tuple(
            s for a, s in enumerate(shape) if a not in self._fixed
        )
        chunks = chunks_of(array)
        self.chunks = (
            None
            if chunks is None
            else tuple(c for a, c in enumerate(chunks) if a not in fixed)
        )

    def _full(self, index: tuple) -> tuple:
        it = iter(index)
        return tuple(
            self._fixed[a] if a in self._fixed else next(it)
            for a in range(self._ndim)
        )

    def __getitem__(self, index: tuple) -> object:
        return self.array[self._full(index)]

    def __setitem__(self, index: tuple, value: object) -> None:
        self.array[self._full(index)] = value

    @property
    def dtype(self) -> object:
        return getattr(self.array, "dtype", None)

    def flush(self) -> None:
        flush(self.array)


def _fix(array: ArrayLike, fixed: dict[int, int]) -> ArrayLike:
    """`array` with the axes in `fixed` set to an index (lazily)."""
    if not fixed:
        return array
    if isinstance(array, Tensor | np.ndarray):
        index = tuple(
            fixed.get(a, slice(None)) for a in range(len(array.shape))
        )
        return array[index]
    return _AxisView(array, fixed)


def _drop_groups(layout: Layout, ndim: int) -> Layout:
    """Layout of `_fix(array, groups)`: the same roles, without groups."""
    group, sample, spatial = layout.positions(ndim)
    if not group:
        return layout
    keep = [a for a in range(ndim) if a not in group]
    new = {a: i for i, a in enumerate(keep)}
    return Layout(
        spatial=tuple(new[a] for a in spatial),
        sample=tuple(new[a] for a in sample),
    )


class GammaMixture(Estimator):
    """Joint model of the sensor, the shading and the tissue classes.

    Image formation. A pixel i of picture k records

        yᵢₖ ~ N(cᵢ + mᵢ·uᵢₖ, vᵢ + g·mᵢ·uᵢₖ)

    with the offset c (camera offset plus background, in ADU), the
    variance v of an empty pixel, the gain g (ADU per electron), the
    smooth modulation m (flat field) and the signal u ≥ 0 in ADU. It is
    the Gaussian limit of yᵢₖ = oᵢ + g·Poisson(mᵢ·(sᵢₖ + bᵢ) + dᵢ) + read
    noise, with u = g·s, c = o + g·(m·b + d) and v = σ² + g²·(m·b + d).

    Probabilistic model. The signal is a mixture of K tissue classes
    and a background class, with proportions drawn per picture:

        πₖ ~ Dirichlet(α),  zᵢₖ ~ Categorical(πₖ)
        uᵢₖ | zᵢₖ = background ~ δ(0)
        uᵢₖ | zᵢₖ = c ~ Gamma(ν_c, rate ν_c / μ_c)
        m = exp(Φ·θ),  θ ~ N(0, (λ·diag(bending energy))⁻¹)

    with Φ the separable DCT-II basis without its DC term (geometric mean
    of m = 1) and an inverse-gamma-like prior on v,
    log p(v) = −½·v_dof·(log v + v₀/v).

    Inference. The variational posterior is

        q = ∏ₖ q(πₖ) · ∏ᵢₖ q(zᵢₖ) q(uᵢₖ | zᵢₖ) · δ(c, v, θ, μ, ν, α)

    - q(z, u) is exact given everything else: the integral over u is a
      Gauss-Hermite quadrature in t = log u centred on the mode of the
      integrand (the roots of a cubic for the "full" noise model).
    - q(πₖ) is Dirichlet (conjugate update).
    - μ, ν are refitted to the aggregated posterior of u in each class,
      and α by Minka's fixed-point Newton method.
    - c, v, θ take one joint Fisher-scoring step per iteration: the
      per-pixel unknowns (c, v) are eliminated exactly, which leaves a
      small dense system in θ. A step that lowers the bound is halved.

    Data are quantised: each observation is replaced by a uniform
    distribution over its quantisation bin of width Δ, so that every
    squared residual becomes r² + Δ²/12 and the bound stays finite when a
    pixel takes the same value in many pictures. Δ = 1 (integer data in
    ADU) by default; `quantization="auto"` detects it from a sample of
    the data.

    Overlapping tiles. When the pictures are tiles of a mosaic whose
    neighbours overlap (`overlap`, a
    [`TileOverlap`][optix.shading.mix.TileOverlap] of integer tile
    positions), a scene voxel seen by several tiles has one class and one
    signal u shared by all of them: tile t sees y = c + b_t·u + noise at
    the matching pixel, with b_t = m (or g_t·m with per-tile gains). Its
    term of the bound integrates the product of the tile likelihoods over
    u in one quadrature, with the class prior of a uniformly chosen owner
    among its pictures; a voxel seen once is the single-picture term. The
    flat field is thereby constrained by the same scene structure seen at
    different places of the field of view, even where the mean signal
    depends on the position in the tiles. The log-ratios of matched
    excesses only initialise the flat field (and the gains).

    With `TileOverlap(fit_gain=True)`, each tile t also has a gain g_t
    (flat field g_t·m), so that a uniform per-tile fade (bleaching, lamp
    drift) is not mistaken for a ramp of m. log g_t is a per-tile constant
    column of the θ system; its common scale is that of the class means
    (geometric mean of the gains one over the tiles that overlap another;
    a tile that overlaps none keeps g = 1). A linear log m along the tile
    steps is then identified only through the class prior (it is nearly
    equivalent to gains ∝ exp of the tile position). Such fits stop on the
    Newton decrement of the parameter step.

    The gain g is held at its nominal value (`gain`, or the gain of an
    initial [`SensorEstimate`][optix.estimates.SensorEstimate]): in this
    model it is identified only through the shape of the signal
    distribution.

    !!! note
        Memory and streaming: the fit reads the data in batches of frames,
        once per iteration (plus one pass per halved step), and twice more
        for the initialisation (once by strips of rows holding every frame,
        for per-pixel order statistics, and once by frame batches).

        The data-driven initialisation takes the per-pixel order statistic
        of rank ½·`bg_fraction`·N as c and its distance to the rank
        0.1587·`bg_fraction`·N as the standard deviation of the background
        (clamped to [¼, 4] times its median); values more than `threshold`
        standard deviations above c are labelled as tissue. The flat field
        is initialised from their mean excess over the threshold, and the
        classes by a k-means on the log of the tissue values.

        The fit stops when the bound improves by less than `tol` per data
        point, or after `max_iterations` iterations.

    Attributes
    ----------
    estimates : dict
        - `"sensor"`: [`SensorEstimate`][optix.estimates.SensorEstimate]
          with `offset` = c and `variance` = v, `(*group, *spatial)`
          (both **include** the background), and the nominal `gain`.
        - `"shading"`: [`ShadingEstimate`][optix.estimates.ShadingEstimate]
          with `flat` = m (`meta["flat_norm"] == "geomean"`) and
          `dark` = c.
        - `"classification"`:
          [`Classification`][optix.estimates.Classification]:
          `concentration` α and `weights` α / ∑α of the K + 1 classes
          (background last), and `parameters` `(K, 2)` holding the mean μ
          and the shape ν of each tissue class.
    concentration_ : Tensor
        `(*group, *sample, K + 1)` concentrations of q(π) of the
        training pictures.
    theta_ : Tensor
        `(*group, ky, kx)` DCT coefficients of log m (of its projection
        on the basis when a given flat field is held fixed).
    quantization_ : Tensor
        `(*group,)` width Δ of the quantisation bins used by the fit.
    tile_gain_ : Tensor or None
        `(*group, ntiles)` per-tile gains g_t (tile t has the flat field
        g_t·m), with `TileOverlap(fit_gain=True)`; None otherwise.
        Geometric mean one over the tiles that overlap another, 1 for a
        tile that overlaps none. `predict` and the default `apply` use m
        alone; `apply(..., tile_gain=True)` divides tile t by g_t.

    Examples
    --------
    >>> import torch
    >>> from optix.models import random_sensor, simulate_gamma_mixture
    >>> from optix.models import simulate_sensor
    >>> sensor = random_sensor((32, 32), gain=2.0, offset_sd=0.0, seed=0)
    >>> draw = simulate_gamma_mixture((20, 32, 32), [100.0], [2.0], seed=0)
    >>> data = simulate_sensor(draw.signal / 2.0, sensor, seed=1).round()
    >>> mixture = GammaMixture(gain=2.0, max_iterations=5).fit(data)
    >>> offset = mixture.estimates["sensor"].offset
    >>> corrected = mixture.apply(data)
    >>> foreground = mixture.apply(data, output="foreground")
    """

    tags = Tags(
        targets=("shading", "sensor", "classification"),
        streaming="per-frame",
        modality=("fluorescence",),
        noise="poisson-gaussian",
        many_frames=True,
        reference="",
    )
    paper = dict(_PRESET)
    recommended = dict(_PRESET)

    def __init__(
        self,
        *,
        nclass: int | None = None,
        noise: str | None = None,
        order: int | tuple[int, int] | None = None,
        gain: float | None = None,
        nodes: int | None = None,
        nu_bounds: tuple[float, float] | None = None,
        quantization: float | str | None = None,
        max_iterations: int | None = None,
        tol: float | None = None,
        learn_modulation: bool | None = None,
        learn_alpha: bool | None = None,
        lr: float | None = None,
        max_halvings: int | None = None,
        max_log_step: float | None = None,
        bg_fraction: float | None = None,
        threshold: float | None = None,
        spread_floor: bool | None = None,
        concentration: float | tuple[float, ...] | None = None,
        v_prior: float | None = None,
        v_dof: float | None = None,
        lam: float | None = None,
        overlap: TileOverlap | object | None = None,
        faithful: bool = True,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        layout: Layout | None = None,
    ) -> None:
        """Set the options; unset ones come from the preset.

        Parameters
        ----------
        nclass : int
            Number K of tissue classes (besides the background).
            Default 1.
        noise : {"full", "constant"}
            Variance of the data given the signal: v + g·m·u (`"full"`,
            the default) or v (`"constant"`, no shot noise; g is unused).
        order : int or (int, int)
            Number of DCT basis functions of log m along each spatial
            axis. Default (8, 8).
        gain : float, optional
            Nominal gain g in ADU per electron. Default: the (mean) gain
            of an initial `SensorEstimate`, else 1.
        nodes : int
            Number of Gauss-Hermite nodes of the integral over u (per
            class, pixel and picture). The cost of the fit grows in
            proportion. The integral is least accurate when a shape ν is
            close to 1 (about 5·10⁻³ on log p(y | z) with 24 nodes,
            10⁻³ with 48); more nodes make it more accurate. Default 24.
        nu_bounds : (float, float)
            Bounds on the Gamma shapes. The lower bound of 1 (default)
            prevents a tissue class from developing a spike at zero and
            competing with the background. Default (1, 10³).
        quantization : float or "auto"
            Width Δ of one quantisation bin, in ADU (1 for integer data,
            the default; 0 treats the data as continuous). `"auto"`
            detects it on a sample of about 10⁵ values from a few frame
            batches of each group: 1 for an integer dtype, else the
            smallest gap between distinct values if nearly all of them are
            multiples of it, else 0. With Δ = 1, a warning is emitted when
            that sample is not integer-valued.
        max_iterations : int
            Maximum number of iterations. Default 100.
        tol : float
            Stop when the bound changes by less than `tol` per data
            point. Default 10⁻⁶.
        learn_modulation : bool
            Estimate m (default True); otherwise m = 1, or the flat field
            of an initial `ShadingEstimate`.
        learn_alpha : bool
            Estimate the Dirichlet concentrations α (default True).
        lr : float
            Damping of the Fisher-scoring step. Default 1.
        max_halvings : int
            When the bound decreases, the last parameter step is halved
            up to this many times. Default 4.
        max_log_step : float
            Largest change of log m in one step. Default 0.5.
        bg_fraction : float
            Lower bound on the fraction of pictures in which a pixel is
            empty, used by the initialisation. Default 0.5.
        threshold : float
            Number of background standard deviations above c beyond
            which a value is initially labelled as tissue. Default 3.
        spread_floor : bool
            Floor the initial standard deviation of an empty pixel at
            the quantisation scale Δ/√12 (so that v ≥ Δ²/12), for data
            whose read noise is well below one bin. Default False (a
            warning is emitted when the initial v is below Δ²/12).
        concentration : float or sequence of float
            Initial Dirichlet concentrations α of the K + 1 proportions
            (background last). Default 1.
        v_prior : float, optional
            Prior variance v₀ of an empty pixel. Default: the median of
            the initial v.
        v_dof : float
            Strength of the prior on v, as a number of pseudo-observations
            (0 disables it). Default 5.
        lam : float
            Precision λ of the bending-energy prior on log m. Default 1.
        overlap : TileOverlap or array-like, optional
            Integer positions of overlapping tiles: a
            [`TileOverlap`][optix.shading.mix.TileOverlap] (from
            `positions=`, `steps=` or `step=`, e.g.
            `TileOverlap(step=(0, 64, 0))` for (tile, x, y, z) data whose
            tiles step by 64 rows along y), or an `(N, D)` array of
            absolute positions (tile axis 0). Every pair of tiles whose
            boxes intersect contributes. The tile axis must be a sample
            axis; the same translations apply to every group. Default
            None: the pictures are independent.
            `TileOverlap(fit_gain=True)` also fits a gain per tile
            (see `tile_gain_`).
        faithful : bool
            True: the reference preset. False: the `recommended` preset
            (currently the same).
        dtype, device, layout : optional
            See [`Estimator`][optix.Estimator]. The layout must have two
            spatial axes (one picture); every other non-group axis indexes
            pictures. Each group gets its own fit.
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        opt = self.options
        if opt["noise"] not in _NOISE:
            raise ValueError(f"noise must be one of {_NOISE}: {opt['noise']}")
        if int(opt["nclass"]) < 1:
            raise ValueError(f"nclass must be >= 1: {opt['nclass']}")
        q = opt["quantization"]
        if isinstance(q, str):
            if q != "auto":
                raise ValueError(f"quantization must be 'auto' or >= 0: {q}")
        elif q < 0:
            raise ValueError("quantization must be non-negative")
        self.concentration_: Tensor | None = None
        self.theta_: Tensor | None = None
        self.tile_gain_: Tensor | None = None
        self.quantization_: Tensor | None = None

    # -- helpers -----------------------------------------------------------
    def _order(self) -> tuple[int, int]:
        order = self.options["order"]
        if isinstance(order, int | np.integer):
            return (int(order), int(order))
        order = tuple(int(k) for k in order)
        if len(order) != 2:
            raise ValueError(f"order must be an int or a pair: {order}")
        return order

    def _split(self, data: ArrayLike) -> tuple[ArrayLike, Layout]:
        """Raw array and layout, with two spatial axes."""
        layout = self._layout_of(data)
        raw = data.array if isinstance(data, Labeled) else data
        _, _, spatial = layout.shape_of(raw)
        if len(spatial) != 2:
            raise ValueError(
                f"GammaMixture needs two spatial axes, the layout has "
                f"{len(spatial)}"
            )
        return raw, layout

    def _groups(
        self, raw: ArrayLike, layout: Layout
    ) -> tuple[tuple[int, ...], list[tuple[int, ...]], tuple[int, ...]]:
        gpos, _, _ = layout.positions(len(raw.shape))
        group_shape, _, _ = layout.shape_of(raw)
        groups = list(itertools.product(*(range(g) for g in group_shape)))
        return gpos, groups, tuple(group_shape)

    def _model(
        self,
        source: ArrayLike,
        layout: Layout,
        backend: dict,
        *,
        alpha: Tensor,
        gain: float,
        quantization: float | None = None,
        overlap: Overlap | None = None,
    ) -> MixtureFit:
        opt = self.options
        if quantization is None:
            quantization = opt["quantization"]
        nclass = int(opt["nclass"])
        component = GammaComponent(int(opt["nodes"]), tuple(opt["nu_bounds"]))
        return MixtureFit(
            source,
            layout,
            component=component,
            nclass=nclass,
            noise=opt["noise"],
            order=self._order(),
            gain=gain,
            quantization=float(quantization),
            alpha=alpha,
            params=torch.ones(nclass, len(component.names)),
            v_prior=opt["v_prior"],
            v_dof=float(opt["v_dof"]),
            lam=float(opt["lam"]),
            gain_sd=0.1,
            backend=backend,
            overlap=overlap,
        )

    def _coarse_start(
        self,
        model: MixtureFit,
        overlap: Overlap,
        backend: dict,
        quantization: float,
    ) -> tuple[Tensor, Tensor | None]:
        """Flat field and tile gains of a fit of the binned mosaic.

        The data are averaged over `coarse` × `coarse` pixels (positions
        scaled), the scene model is fitted there with the exact gain step
        (per-tile maps are small at that size), and its log flat field is
        interpolated back to the full grid. The quantisation of a mean of
        f² values is Δ/f².
        """
        opt = self.options
        f = overlap.coarse
        data, layout, pos = model.binned(f)
        spec = TileOverlap(
            positions=pos, axis=model._tile_index, fit_gain=overlap.fit_gain
        )
        ov = Overlap.build(
            spec, tuple(data.shape), layout, threshold=overlap.threshold
        )
        coarse = self._model(
            data,
            layout,
            backend,
            alpha=self._prior_alpha(backend),
            gain=float(model.param.gain),
            quantization=float(quantization) / f**2,
            overlap=ov,
        )
        coarse.exact_gain = True
        coarse.init(
            bg_fraction=float(opt["bg_fraction"]),
            threshold=float(opt["threshold"]),
            modulation=True,
            spread_floor=bool(opt["spread_floor"]),
        )
        coarse.run(
            niter=int(opt["max_iterations"]),
            tol=float(opt["tol"]),
            learn_modulation=True,
            learn_alpha=bool(opt["learn_alpha"]),
            lr=float(opt["lr"]),
            max_halvings=int(opt["max_halvings"]),
            max_log_step=float(opt["max_log_step"]),
        )
        logm = coarse.param.M.log()[None, None]
        logm = torch.nn.functional.interpolate(
            logm,
            size=model.spatial_shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return logm.exp(), coarse.tile_gain()

    def _prior_alpha(self, backend: dict) -> Tensor:
        nclass = int(self.options["nclass"])
        alpha = torch.as_tensor(self.options["concentration"], **backend)
        alpha = alpha.flatten()
        if alpha.numel() == 1:
            alpha = alpha.expand(nclass + 1)
        if alpha.numel() != nclass + 1:
            raise ValueError(
                f"concentration needs 1 or nclass + 1 = {nclass + 1} values"
            )
        return alpha.clone()

    # -- fit -----------------------------------------------------------------
    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        raw, layout = self._split(data)
        backend = self._backend(data)
        opt = self.options
        nclass = int(opt["nclass"])
        ndim = len(raw.shape)
        map_shape = layout.map_shape(raw)
        gpos, groups, group_shape = self._groups(raw, layout)
        if math.prod(layout.shape_of(raw)[1]) < 1:
            raise ValueError("GammaMixture needs at least one picture")

        usable = (ShadingEstimate, SensorEstimate, Classification)
        self._warn_unused(init, *usable)
        shading = self._pick(init, ShadingEstimate, shape=map_shape, like=data)
        sensor = self._pick(init, SensorEstimate, shape=map_shape, like=data)
        classes = self._pick(init, Classification, like=data)

        gain = opt["gain"]
        if gain is None:
            gain = 1.0
            if sensor is not None and sensor.gain is not None:
                gain = float(sensor.gain.mean())
        gain = float(gain)

        def field(est: object, name: str, g: tuple) -> Tensor | None:
            value = None if est is None else getattr(est, name)
            if value is None:
                return None
            return torch.broadcast_to(value, map_shape)[g]

        sub = _drop_groups(layout, ndim)
        overlap = None
        if opt["overlap"] is not None:
            overlap = Overlap.build(
                opt["overlap"],
                tuple(int(n) for n in raw.shape),
                layout,
                threshold=float(opt["threshold"]),
            )
        offsets, variances, flats, thetas, alphas = [], [], [], [], []
        params, concentrations, quantizations = [], [], []
        tile_gains = []
        converged = []
        warned = False
        self.history.reason = ""
        for g in groups:
            source = _fix(raw, dict(zip(gpos, g, strict=True)))
            quantization = opt["quantization"]
            if quantization == "auto":
                quantization = detect_quantization(source, sub)
            elif quantization == 1 and not warned:
                if not is_integer_valued(source, sub):
                    warned = True
                    warnings.warn(
                        "quantization=1 assumes integer-valued data (in "
                        "ADU), but these data are not integer-valued: "
                        "pass quantization='auto', or the width of their "
                        "quantisation bins (0 for continuous data).",
                        UserWarning,
                        stacklevel=3,
                    )
            quantizations.append(float(quantization))
            model = self._model(
                source,
                sub,
                backend,
                alpha=self._prior_alpha(backend),
                gain=gain,
                quantization=quantization,
                overlap=overlap,
            )
            offset = field(shading, "dark", g)
            if offset is None:
                offset = field(sensor, "offset", g)
            flat0, gains0 = field(shading, "flat", g), None
            if overlap is not None and overlap.coarse and flat0 is None:
                flat0, gains0 = self._coarse_start(
                    model, overlap, backend, quantization
                )
            model.init(
                bg_fraction=float(opt["bg_fraction"]),
                threshold=float(opt["threshold"]),
                modulation=bool(opt["learn_modulation"]),
                spread_floor=bool(opt["spread_floor"]),
                offset=offset,
                variance=field(sensor, "variance", g),
                flat=flat0,
                classes=_class_init(classes, nclass, g, group_shape),
                gains=gains0,
            )

            def report(
                n: int,
                bound: float,
                g: tuple = g,
                model: MixtureFit = model,
            ) -> None:
                if verbose:
                    prm, prior = model.param, model.prior
                    mu = ", ".join(f"{x:.4g}" for x in prior.params[:, 0])
                    nu = ", ".join(f"{x:.4g}" for x in prior.params[:, 1])
                    halved = model.halvings[-1]
                    print(
                        f"GammaMixture {g} {n:3d} | bound/N = "
                        f"{bound / model.numel:12.6e} | mu = [{mu}] | "
                        f"nu = [{nu}] | c = {prm.C.mean().item():.4g} | "
                        f"v = {prm.V.mean().item():.4g}"
                        + (f" | halved x{halved}" if halved else "")
                    )
                if callback is not None:
                    callback(self)

            ok, reason = model.run(
                niter=int(opt["max_iterations"]),
                tol=float(opt["tol"]),
                learn_modulation=bool(opt["learn_modulation"]),
                learn_alpha=bool(opt["learn_alpha"]),
                lr=float(opt["lr"]),
                max_halvings=int(opt["max_halvings"]),
                max_log_step=float(opt["max_log_step"]),
                callback=report,
            )
            converged.append(ok)
            if not self.history.reason or not ok:
                self.history.reason = reason
            halvings = model.halvings + [0] * (
                len(model.history) - len(model.halvings)
            )
            for bound, h in zip(model.history, halvings, strict=True):
                extras = {"halvings": h}
                if groups != [()]:
                    extras["group"] = int(np.ravel_multi_index(g, group_shape))
                self.history.record(bound, **extras)
            prm, prior = model.param, model.prior
            offsets.append(prm.C)
            variances.append(prm.V)
            flats.append(prm.M)
            thetas.append(prm.theta)
            alphas.append(model.post.alpha)
            params.append(prior.params)
            concentrations.append(prior.alpha)
            tile_gains.append(model.tile_gain())
        self.history.converged = all(converged)

        def stack(items: list[Tensor]) -> Tensor:
            return torch.stack(items).reshape(*group_shape, *items[0].shape)

        offset = stack(offsets)
        parameters = stack(params)  # tissue classes only
        alpha = stack(concentrations)
        gain_map = torch.as_tensor(gain, **backend)
        if group_shape:
            gain_map = gain_map.expand(*group_shape, 1, 1).clone()
        self.concentration_ = stack(alphas)
        self.theta_ = stack(thetas)
        self.tile_gain_ = None
        if tile_gains and tile_gains[0] is not None:
            self.tile_gain_ = stack(tile_gains)
        self.quantization_ = torch.tensor(
            quantizations, dtype=torch.float64
        ).reshape(group_shape)
        model_meta = {
            "model": "y ~ N(offset + flat * u, variance + gain * flat * u)",
            "noise": opt["noise"],
        }
        self.estimates = {
            "sensor": SensorEstimate(
                offset=offset,
                variance=stack(variances),
                gain=gain_map,
                meta={
                    **model_meta,
                    "units": "ADU",
                    "offset_includes": "background",
                    "variance_includes": "background",
                    "gain": "nominal",
                },
            ),
            "shading": ShadingEstimate(
                flat=stack(flats),
                dark=offset.clone(),
                meta={**model_meta, "flat_norm": "geomean"},
            ),
            "classification": Classification(
                weights=alpha / alpha.sum(-1, keepdim=True),
                parameters=parameters,
                concentration=alpha,
                meta={
                    **model_meta,
                    "distribution": "gamma",
                    "parameters": ["mean", "shape"],
                    "background": "last",
                    "units": "ADU",
                },
            ),
        }

    # -- responsibilities ------------------------------------------------
    def _fitted(self) -> None:
        if "classification" not in self.estimates:
            raise RuntimeError("GammaMixture is not fitted: call fit first")

    def _group_model(
        self,
        g: tuple,
        source: ArrayLike,
        layout: Layout,
        backend: dict,
        overlap: Overlap | None = None,
    ) -> MixtureFit:
        """Model of group `g` with the fitted parameters."""
        sensor = self.estimates["sensor"].to(**backend)
        shading = self.estimates["shading"].to(**backend)
        classes = self.estimates["classification"].to(**backend)
        gain = torch.as_tensor(sensor.gain)
        gain = float(gain.reshape(*gain.shape[: len(g)], -1)[g].mean())
        quantization = None
        if self.quantization_ is not None:
            quantization = float(self.quantization_[g])
        model = self._model(
            source,
            layout,
            backend,
            alpha=classes.concentration[g],
            gain=gain,
            quantization=quantization,
            overlap=overlap,
        )
        if model.param.A is not None and self.tile_gain_ is not None:
            model.param.A = self.tile_gain_[g].to(**backend).log()
        _, _, spatial = layout.shape_of(source)
        maps = (*classes.concentration.shape[: len(g)], *spatial)
        prm, prior = model.param, model.prior
        prm.C = torch.broadcast_to(sensor.offset, maps)[g].clone()
        prm.V = torch.broadcast_to(sensor.variance, maps)[g].clone()
        prm.M = torch.broadcast_to(shading.flat, maps)[g].clone()
        prior.params = classes.parameters[g].clone()
        prior.alpha = classes.concentration[g].clone()
        return model

    def _responsibilities(
        self,
        data: ArrayLike,
        *,
        alpha: Tensor | None,
        per_class: bool,
        mean: bool,
        out: WritableArrayLike | None,
        out_mean: WritableArrayLike | None,
        niter: int,
        tol: float,
        probabilities: bool = True,
    ) -> tuple[WritableArrayLike, WritableArrayLike | None, Tensor]:
        self._fitted()
        if not isinstance(data, Labeled):
            data = self._labeled(data, self.layout)
        raw, layout = self._split(data)
        backend = self._backend(data)
        nclass = int(self.options["nclass"])
        shape = tuple(int(s) for s in raw.shape)
        group_shape, sample_shape, spatial = layout.shape_of(raw)
        fitted = tuple(self.estimates["shading"].flat.shape)
        if tuple(group_shape) + tuple(spatial) != fitted:
            raise ValueError(
                f"Data have maps of shape {group_shape + spatial}, the fit "
                f"has {fitted}"
            )
        if probabilities:
            out = prepare_out(
                out,
                (nclass, *shape) if per_class else shape,
                like=raw,
                dtype=self.dtype,
            )
        if mean:
            out_mean = prepare_out(out_mean, shape, like=raw, dtype=self.dtype)
        if alpha is None:
            prior = self.estimates["classification"].concentration
            alpha = prior.to(**backend)[..., *([None] * len(sample_shape)), :]
            alpha = alpha.expand(*group_shape, *sample_shape, nclass + 1)
            alpha = alpha.clone()
        else:
            alpha = alpha.to(**backend).clone()
        gpos, groups, _ = self._groups(raw, layout)
        sub = _drop_groups(layout, len(shape))
        for g in groups:
            fixed = dict(zip(gpos, g, strict=True))
            source = _fix(raw, fixed)
            model = self._group_model(g, source, sub, backend)
            if not probabilities:
                target = None
            elif per_class:
                shifted = {a + 1: i for a, i in fixed.items()}
                target = [_fix(out, {0: k, **shifted}) for k in range(nclass)]
            else:
                target = _fix(out, fixed)
            model.posterior(
                source,
                sub,
                alpha[g],
                out=target,
                out_mean=None if not mean else _fix(out_mean, fixed),
                per_class=per_class,
                niter=niter,
                tol=tol,
            )
        if probabilities:
            flush(out)
        if mean:
            flush(out_mean)
        return (
            out if probabilities else None,
            out_mean if mean else None,
            alpha,
        )

    def classify(
        self,
        data: ArrayLike,
        *,
        per_class: bool = False,
        mean: bool = False,
        out: WritableArrayLike | None = None,
        out_mean: WritableArrayLike | None = None,
        joint: bool = False,
    ) -> tuple[WritableArrayLike, WritableArrayLike | None]:
        """Posterior class probabilities of the training data.

        Uses the fitted parameters and the q(π) of each training picture
        (`concentration_`). See
        [`predict`][optix.shading.GammaMixture.predict] for new
        pictures.

        Parameters
        ----------
        data : array-like or Labeled
            The data given to [`fit`][optix.shading.GammaMixture.fit].
        per_class : bool
            False: the foreground probability p(z ≠ background | y), of
            the shape of `data`. True: p(z = c | y) for each tissue class,
            `(K, *data.shape)`; the background is 1 minus their sum.
        mean : bool
            Also compute the posterior mean E[u | y] (0 in the
            background), of the shape of `data`.
        out, out_mean : writable array-like, optional
            Outputs (tensor, numpy, zarr, ...). Default: new tensors.
        joint : bool
            With overlapping tiles (`overlap`), use the fitted scene model:
            every observation of a voxel seen by several tiles gets the
            joint posterior of that voxel (one class and one signal for
            all its tiles), and tile t has the flat field gₜ·m. Default
            False: each picture on its own, with m.

        Returns
        -------
        probabilities : writable array-like
        mean : writable array-like or None
        """
        self._fitted()
        alpha = self.concentration_
        if joint and self.options["overlap"] is not None:
            return self._classify_joint(data, per_class, mean, out, out_mean)
        raw = data.array if isinstance(data, Labeled) else data
        layout = data.layout if isinstance(data, Labeled) else self.layout
        layout = layout or Layout.default(len(raw.shape), 2)
        group_shape, sample_shape, _ = layout.shape_of(raw)
        expected = (*group_shape, *sample_shape)
        if tuple(alpha.shape[:-1]) != expected:
            raise ValueError(
                f"classify needs the training data: they have "
                f"{tuple(alpha.shape[:-1])} pictures, these data {expected}"
            )
        z, u, _ = self._responsibilities(
            data,
            alpha=alpha,
            per_class=per_class,
            mean=mean,
            out=out,
            out_mean=out_mean,
            niter=0,
            tol=0.0,
        )
        return z, u

    def _classify_joint(
        self,
        data: ArrayLike,
        per_class: bool,
        mean: bool,
        out: WritableArrayLike | None,
        out_mean: WritableArrayLike | None,
    ) -> tuple[WritableArrayLike, WritableArrayLike | None]:
        if not isinstance(data, Labeled):
            data = self._labeled(data, self.layout)
        raw, layout = self._split(data)
        backend = self._backend(data)
        nclass = int(self.options["nclass"])
        shape = tuple(int(n) for n in raw.shape)
        out = prepare_out(
            out,
            (nclass, *shape) if per_class else shape,
            like=raw,
            dtype=self.dtype,
        )
        if mean:
            out_mean = prepare_out(out_mean, shape, like=raw, dtype=self.dtype)
        overlap = Overlap.build(
            self.options["overlap"],
            shape,
            layout,
            threshold=float(self.options["threshold"]),
        )
        gpos, groups, _ = self._groups(raw, layout)
        sub = _drop_groups(layout, len(shape))
        alpha = self.concentration_.to(**backend)
        for g in groups:
            fixed = dict(zip(gpos, g, strict=True))
            source = _fix(raw, fixed)
            model = self._group_model(g, source, sub, backend, overlap)
            if per_class:
                shifted = {a + 1: i for a, i in fixed.items()}
                target = [_fix(out, {0: k, **shifted}) for k in range(nclass)]
            else:
                target = _fix(out, fixed)
            model.scene_posterior(
                alpha[g],
                out=target,
                out_mean=_fix(out_mean, fixed) if mean else None,
                per_class=per_class,
            )
        flush(out)
        if mean:
            flush(out_mean)
        return out, out_mean if mean else None

    def predict(
        self,
        data: ArrayLike,
        *,
        per_class: bool = False,
        mean: bool = False,
        out: WritableArrayLike | None = None,
        out_mean: WritableArrayLike | None = None,
        niter: int = 20,
        tol: float = 1e-3,
    ) -> tuple[WritableArrayLike, WritableArrayLike | None, Tensor]:
        """Classify pictures with the fitted model.

        c, v, m, g, μ, ν and α are kept fixed; only the proportions q(π)
        of each picture are inferred, by alternating q(z) and q(π) for
        each batch of pictures. The class likelihoods do not depend on π,
        so they are computed once per batch: the cost is that of one
        pass of the fit.

        Parameters
        ----------
        data : array-like or Labeled
            Pictures with the spatial (and group) shape of the fit.
        per_class, mean, out, out_mean
            See [`classify`][optix.shading.GammaMixture.classify].
        niter : int
            Maximum number of q(z) / q(π) iterations per batch.
        tol : float
            Stop when the class counts of every picture change by less
            than `tol` (as a fraction of the number of pixels).

        Returns
        -------
        probabilities : writable array-like
        mean : writable array-like or None
        concentration : Tensor
            `(*group, *sample, K + 1)` concentrations of q(π).
        """
        return self._responsibilities(
            data,
            alpha=None,
            per_class=per_class,
            mean=mean,
            out=out,
            out_mean=out_mean,
            niter=niter,
            tol=tol,
        )

    def apply(
        self,
        data: ArrayLike,
        out: WritableArrayLike | None = None,
        *,
        output: str = "corrected",
        tile_gain: bool = False,
    ) -> WritableArrayLike:
        """Correct the data, or classify them with the fitted model.

        Parameters
        ----------
        data : array-like or Labeled
            Pictures with the spatial (and group) shape of the fit.
        out : writable array-like, optional
            Output (tensor, numpy, zarr, ...). Default: a new tensor.
        output : {"corrected", "foreground", "classes", "mean"}
            - `"corrected"` (default): the data without the background and
              the modulation, x = (y − c) / m, with c the fitted offset
              (camera offset plus background, `estimates["sensor"].offset`)
              and m the flat field (geometric mean 1,
              `estimates["shading"].flat`). x is the signal u plus noise,
              in ADU, of the shape of `data`.
            - `"foreground"`: p(z ≠ background | y), of the shape of
              `data`.
            - `"classes"`: p(z = c | y) of each tissue class,
              `(K, *data.shape)`.
            - `"mean"`: E[u | y] (0 in the background), of the shape of
              `data`.

            The probabilities and the mean are those of
            [`predict`][optix.shading.GammaMixture.predict], with its
            defaults.
        tile_gain : bool
            `"corrected"` only: also divide tile t by its fitted gain gₜ
            (`tile_gain_`, with `TileOverlap(fit_gain=True)`), which
            equalises the tiles of the training mosaic: x = (y − c)/(gₜ·m).
            `data` must then have the tile axis of the fit. Default False.

        Returns
        -------
        writable array-like
            `out`.
        """
        if output not in _OUTPUTS:
            raise ValueError(f"output must be one of {_OUTPUTS}: {output!r}")
        if not isinstance(data, Labeled):
            data = self._labeled(data, self.layout)
        with torch.no_grad():
            if output == "corrected":
                return self._apply(data, out, tile_gain=tile_gain)
            if output == "mean":
                _, mean, _ = self._responsibilities(
                    data,
                    alpha=None,
                    per_class=False,
                    mean=True,
                    out=None,
                    out_mean=out,
                    niter=20,
                    tol=1e-3,
                    probabilities=False,
                )
                return mean
            z, _, _ = self.predict(
                data, per_class=output == "classes", out=out
            )
            return z

    def _apply(
        self,
        data: ArrayLike,
        out: WritableArrayLike | None,
        tile_gain: bool = False,
    ) -> WritableArrayLike:
        """Corrected data (y − c) / m, or (y − c) / (gₜ·m)."""
        self._fitted()
        shading = self.estimates["shading"]
        layout = self._layout_of(data)
        group_shape, _, spatial = layout.shape_of(data)
        fitted = tuple(shading.flat.shape)
        if tuple(group_shape) + tuple(spatial) != fitted:
            raise ValueError(
                f"Data have maps of shape {group_shape + spatial}, the fit "
                f"has {fitted}"
            )
        flat, dark = shading.flat, shading.dark
        gains = None
        if tile_gain:
            if self.tile_gain_ is None:
                raise ValueError(
                    "tile_gain=True needs fitted tile gains "
                    "(TileOverlap(fit_gain=True))"
                )
            gains, ti = self.tile_gain_, self._tile_sample_index(data)

        def fn(chunk: object, batch: Tensor) -> Tensor:
            batch = batch.sub_(dark[chunk.group].to(batch)).div_(
                flat[chunk.group].to(batch)
            )
            if gains is not None:
                shape = chunk.sample_shape
                start = chunk.samples[ti].start or 0
                t = torch.arange(start, start + shape[ti])
                view = [1] * len(shape)
                view[ti] = shape[ti]
                t = t.reshape(view).expand(shape).reshape(-1)
                g = gains[chunk.group].to(batch)[t.to(batch.device)]
                batch = batch.div_(g[:, None, None])
            return batch

        return self._apply_frames(data, out, fn, overhead=1.0)

    def _tile_sample_index(self, data: ArrayLike) -> int:
        """Position of the tile axis among the sample axes of `data`."""
        layout = self._layout_of(data)
        ndim = len(data.shape)
        _, sample, _ = layout.positions(ndim)
        axis = self.options["overlap"]
        axis = axis.axis if isinstance(axis, TileOverlap) else 0
        if isinstance(axis, str):
            axis = layout.axes.index(axis)
        axis = int(axis) % ndim
        if axis not in sample:
            raise ValueError("the tile axis must be a sample axis of data")
        return list(sample).index(axis)

    # -- state ---------------------------------------------------------------
    def _extra_state(self) -> dict[str, object]:
        state = {
            "concentration": self.concentration_,
            "theta": self.theta_,
            "quantization": self.quantization_,
        }
        if self.tile_gain_ is not None:
            state["tile_gain"] = self.tile_gain_
        return state

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.concentration_ = state.get("concentration")
        self.theta_ = state.get("theta")
        self.quantization_ = state.get("quantization")
        self.tile_gain_ = state.get("tile_gain")


def _class_init(
    classes: Classification | None,
    nclass: int,
    g: tuple[int, ...],
    group_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor | None] | None:
    """(params, frac, alpha) of group `g` from an initial Classification."""
    if classes is None or classes.parameters is None:
        return None
    params = classes.parameters
    if params.shape[-2:] != (nclass, 2):
        raise ValueError(
            f"init Classification has parameters of shape "
            f"{tuple(params.shape)}, expected (*group, {nclass}, 2)"
        )
    params = torch.broadcast_to(params, (*group_shape, nclass, 2))[g]
    weights = classes.weights
    if weights is None:
        weights = torch.ones(nclass + 1, dtype=params.dtype)
    weights = torch.broadcast_to(weights, (*group_shape, nclass + 1))[g]
    frac = weights[:nclass] / weights[:nclass].sum()
    alpha = classes.concentration
    if alpha is not None:
        alpha = torch.broadcast_to(alpha, (*group_shape, nclass + 1))[g]
    return params, frac, alpha
