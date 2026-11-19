"""BaSiC background and shading correction (Peng et al., 2017)."""

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
    frame_batches,
)
from ..stats import Reservoir
from ._basic_solvers import (
    BaSiCResult,
    Dct2,
    baseline_basicpy,
    baseline_median,
    basic_approximate,
    default_lambdas,
    resize,
)

__all__ = ["BaSiC"]

_BASELINES = ("median", "basicpy")
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
    "reweight_tol": 1e-2,
    "mu_coef": 12.5,
    "rho": 1.5,
    "max_mu_coef": 1e7,
    "max_frames": 512,
    "sampling": "stride",
    "seed": 0,
    "baseline_method": "median",
    "sort_intensity": True,
    "baseline_norm": 0.0,
}


class BaSiC(Estimator):
    r"""BaSiC: low-rank + sparse shading correction (Peng et al., 2017).

    Image formation, for frame :math:`i` of a group (one flat field per
    group, shared by all its samples):

    .. math::

        I_i(x) = S(x) \, \big(I^0_i(x) + b_i\big) + D(x)

    with a smooth flat field :math:`S` (mean 1), a smooth dark field
    :math:`D` (optional), and, for time-lapse data, a per-frame baseline
    :math:`b_i` (in units of the flat-corrected signal, i.e. scaled by
    :math:`S`, as in the paper's equation (2)). The correction is

    .. math::

        \hat I^0_i = (I_i - D) / S - b_i + b_\mathrm{norm}

    with :math:`b_\mathrm{norm}` an arbitrary background: 0 removes the
    background (fluorescence), :math:`\mathrm{mean}_i\, b_i` keeps the raw
    intensity range (bright field).

    **Fit.** A subsample of frames, resized to the working size, is
    sorted at each pixel over the frames (the paper; ``sort_intensity``)
    and forms the matrix :math:`\mathbf D` (one frame per row). BaSiC
    decomposes it
    as :math:`\mathbf D = \mathbf B \mathbf S^T + \mathbf 1 \mathbf D^T +
    \mathbf E` with the low-rank (rank one, plus dark field) part smooth,
    i.e. sparse in the DCT domain, and the residual :math:`\mathbf E`
    sparse:

    .. math::

        \min \; \lambda_S \|\mathrm{dct}(S)\|_1
               + \lambda_D \big(\|\mathrm{dct}(D)\|_1 + \|D\|_1\big)
               + \|W \odot E\|_1
        \quad \text{s.t.} \quad
        \mathbf D = \mathbf B \mathbf S^T + \mathbf 1 \mathbf D^T
                    + \mathbf E,

    solved by inexact augmented Lagrangian (ALM / ADMM) with penalty
    :math:`\mu_{k+1} = \min(\rho\mu_k, \bar\mu)`. The weights are
    reweighted (reweighted L1) as
    :math:`W = 1 / (|E_i / \overline{XA_i}| + \epsilon)`, normalised to
    mean 1, until :math:`S` stops changing. The default regularisation is
    derived from the data: :math:`\lambda_S = 0.5 \,
    \|\mathrm{dct}(\bar{\mathbf D} / \mathrm{mean}\,\bar{\mathbf D})\|_1
    / 400` and :math:`\lambda_D = 0.1 \lambda_S`. The profiles are then
    resized back (bilinear) to the image size.

    :math:`S` starts at zero, as in the paper.

    **Baseline** (``timelapse=True``): after sorting, the fitted
    :math:`B_i` are not the baselines of the frames, so (the paper's
    two-step strategy) :math:`b_i` is fitted on the *unsorted* frames
    with :math:`S` and :math:`D` fixed: here on every frame (at the
    working size), batch by batch over the whole stack, by the exact L1
    fit
    :math:`\arg\min_b \|I_i - bS - D\|_1` (a median of
    :math:`(I_i - D)/S` weighted by :math:`S`). :meth:`apply` subtracts the
    fitted baseline when the data have the fitted sample shape, and
    otherwise computes the same cheap per-frame reduction.

    Parameters
    ----------
    working_size : int or (int, int) or None
        Size the frames are resized to for fitting (None: no resizing).
    darkfield : bool
        Also estimate the dark field.
    timelapse : bool
        Estimate (in :meth:`fit`) and remove (in :meth:`apply`) the
        per-frame baseline.
    lambda_flat, lambda_dark : float, optional
        Sparsity weights of the DCT coefficients of the flat and dark
        fields (default: derived from the data, per group).
    epsilon : float
        Reweighting constant.
    tol, max_iterations : float, int
        ALM stop on the relative residual ``|D - XA - E|_F / |D|_F``.
    max_reweight, reweight_tol : int, float
        Reweighting stop on the relative L1 change of the flat field.
    mu_coef, rho, max_mu_coef : float
        Penalty schedule: ``mu_0 = mu_coef / |D|_2``,
        ``mu <- min(rho mu, max_mu_coef mu_0)``.
    max_frames : int or None
        Number of frames per group used for fitting (None: all).
    sampling : {"stride", "reservoir"}
        How the fitting frames are chosen when there are more than
        ``max_frames``: evenly spaced, or a uniform random sample drawn
        with :class:`optix.stats.Reservoir` (seeded by ``seed``).
    seed : int
        Seed of the reservoir.
    baseline_method : {"median", "basicpy"}
        Baseline estimator used by :meth:`fit` (``timelapse=True``).
    sort_intensity : bool
        Sort each pixel over the fitting frames before the fit (the
        paper). False is BaSiCPy's default (``recommended`` preset).
    baseline_norm : float or "mean"
        Background :math:`b_\mathrm{norm}` added back by :meth:`apply`
        with ``timelapse`` ("mean": the mean fitted baseline of the
        group).
    faithful : bool
        ``True``: the ``paper`` preset. ``False``: the ``recommended``
        preset (``sort_intensity=False``, more accurate on sparse
        foregrounds in our tests).
    dtype, device, layout : optional
        See :class:`optix.Estimator`. The layout must have two spatial
        axes; a group axis (e.g. channels) gets its own profiles.

    Attributes
    ----------
    estimates : dict
        ``{"shading": ShadingEstimate}``: ``flat`` and ``dark`` (None
        without ``darkfield``) shaped ``(*group, *spatial)``, ``baseline``
        shaped ``(*group, *sample)`` (None without ``timelapse``).
        ``meta["flat_norm"] == "mean"`` (normalised at the working size,
        before resizing), ``meta["baseline_units"] == "signal"``.
    fit_frames_ : (k,) Tensor
        Flat sample indices (storage order) of the fitting frames.
    fit_baseline_ : (*group, k) Tensor or None
        Without ``sort_intensity``: baseline of the fitting frames from
        the shading fit itself, ``B_i mean(shading)`` (BaSiCPy's
        ``baseline``). None after sorting.

    Notes
    -----
    **Faithful mode.** Follows the Methods of the paper (Peng et al., Nat
    Commun 8:14836, 2017): the model, the per-pixel sorting, the zero
    initialisation, the reweighted-L1 low-rank + sparse decomposition
    with DCT-sparse :math:`S` and :math:`D = D_Z + D_R`, the two-step
    baseline and the correction with :math:`b_\mathrm{norm}`. The
    update equations, the weights :math:`W`, the automatic
    :math:`\lambda_S, \lambda_D` and the stopping rules are those of the
    MIT-licensed BaSiCPy 2.0.0, ``fitting_mode="approximate"`` (BaSiCPy's
    port of the original BaSiC), with ``working_size=128``, ``tol=1e-3``
    and ``reweight_tol=1e-2``. The paper solves the problem with the
    linearised ALM of Lin et al. (2011); BaSiCPy's "approximate" scheme
    is its inexact-ALM variant. The per-frame baseline solver
    (weighted median) is optix's: BaSiCPy's own is the ``"basicpy"``
    extension.

    **Streaming.** :meth:`fit` reads only the ``max_frames`` fitting
    frames per group, one at a time (frames beyond ``max_frames`` are not
    used, a departure from the reference, which fits all frames), plus
    one pass over the stack when ``timelapse=True``. :meth:`apply` works
    on frame batches.

    **Extensions** (off by default):

    - ``sort_intensity=False``: fit the unsorted frames (BaSiCPy's
      default); ``fit_baseline_`` then holds the fitted baselines.
    - ``baseline_method="basicpy"``: BaSiCPy's ``fit_only_baseline``, a
      reweighted-L1 ALM trusting only the pixels below each frame's 10 %
      quantile, run per batch (so its penalty, stopping and weight
      normalisation are per batch). It is biased low where the flat field
      varies (about -11 % with a flat of +/-25 %, even without noise),
      hence not the default.
    - ``sampling="reservoir"``: random fitting frames.

    Not implemented: BaSiCPy's ``autotune`` (search of ``lambda_flat`` on
    an entropy and Fourier cost; pass ``lambda_flat`` explicitly) and its
    ``fitting_mode="ladmap"`` solver (which fails on 2-D stacks in
    BaSiCPy 2.0.0).

    References
    ----------
    Peng T, Thorn K, Schroeder T, Wang L, Theis FJ, Marr C, Navab N.
    A BaSiC tool for background and shading correction of optical
    microscopy images. Nat Commun 8, 14836 (2017).
    doi:10.1038/ncomms14836

    Examples
    --------
    >>> import torch
    >>> from optix.models import random_shading, simulate_shading
    >>> truth = random_shading((64, 64), seed=0)
    >>> x = 100 + 50 * torch.rand(40, 64, 64)
    >>> Y = simulate_shading(x, truth)
    >>> basic = BaSiC(working_size=64).fit(Y)
    >>> flat = basic.estimates["shading"].flat
    >>> corrected = basic.apply(Y)
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
        baseline_method: str | None = None,
        sort_intensity: bool | None = None,
        baseline_norm: float | str | None = None,
        faithful: bool = True,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        layout: Layout | None = None,
    ) -> None:
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        opt = self.options
        for name, allowed in (
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
    ) -> Tensor:
        """Frame ``k`` (flat sample index) of ``group``, spatial in order."""
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
        frame = as_tensor(raw[tuple(index)], **backend)
        stored = sorted(xpos)
        perm = [stored.index(a) for a in xpos]
        if perm != list(range(len(perm))):
            frame = frame.permute(perm)
        return frame

    def _solve(self, D: Tensor) -> tuple[BaSiCResult, float, float]:
        opt = self.options
        lam_f, lam_d = opt["lambda_flat"], opt["lambda_dark"]
        if lam_f is None:
            dct = Dct2(*D.shape[1:], dtype=D.dtype, device=D.device)
            lam_f, _ = default_lambdas(D, dct)
        if lam_d is None:
            lam_d = 0.1 * lam_f
        common = dict(
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
        )
        return basic_approximate(D, **common), lam_f, lam_d

    def _baseline(self, D: Tensor, flat: Tensor, dark: Tensor) -> Tensor:
        if self.options["baseline_method"] == "median":
            return baseline_median(D, flat, dark)
        opt = self.options
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
        """``(*group, H, W)`` maps resized to the working size."""
        lead = maps.shape[:-2]
        flat = maps.reshape(-1, *maps.shape[-2:])
        return resize(flat, size).reshape(*lead, *_resized_shape(maps, size))

    # -- fit / apply -------------------------------------------------------
    def _fit(
        self,
        Y: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        self._warn_unused(init)
        layout = self._layout_of(Y)
        raw = Y.array if isinstance(Y, Labeled) else Y
        group_shape, sample_shape, spatial = layout.shape_of(raw)
        if len(spatial) != 2:
            raise ValueError(
                f"BaSiC needs two spatial axes, the layout has {len(spatial)}"
            )
        backend = self._backend(Y)
        size = self._working_shape(spatial)
        n = math.prod(sample_shape)
        if n < 1:
            raise ValueError("BaSiC needs at least one frame per group")
        frames = self._select(n)
        opt = self.options
        flats, darks, fit_base, converged = [], [], [], []
        for group in itertools.product(*(range(g) for g in group_shape)):
            D = torch.stack(
                [
                    resize(
                        self._frame(raw, layout, group, k, backend)[None], size
                    )[0]
                    for k in frames.tolist()
                ]
            )
            if opt["sort_intensity"]:
                D = D.sort(dim=0).values
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
                    f"reweightings, ALM iterations {result.iterations}, "
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
            baseline = self._fit_baseline(Y, flat, dark, size)
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
        Y: ArrayLike,
        flat: Tensor,
        dark: Tensor,
        size: tuple[int, int] | None,
    ) -> Tensor:
        """Per-frame baseline in one pass, ``(*group, *sample)``."""
        layout = self._layout_of(Y)
        group_shape, sample_shape, _ = layout.shape_of(Y)
        small_flat = self._small(flat, size)
        small_dark = self._small(dark, size)
        out = torch.zeros(
            *group_shape, *sample_shape, dtype=flat.dtype, device=flat.device
        )
        chunks = frame_batches(
            Y,
            ndim=None if isinstance(Y, Labeled) else 2,
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
        self, Y: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if "shading" not in self.estimates:
            raise RuntimeError("BaSiC is not fitted: call fit first")
        est = self.estimates["shading"]
        layout = self._layout_of(Y)
        group_shape, sample_shape, spatial = layout.shape_of(Y)
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

        def fn(chunk: object, batch: Tensor) -> Tensor:
            f = flat[chunk.group].to(batch)
            d = dark[chunk.group].to(batch)
            x = (batch - d) / f
            if timelapse:
                if stored is not None:
                    b = stored[chunk.group + chunk.samples].reshape(-1)
                else:
                    b = baseline_median(
                        resize(batch, size),
                        small_flat[chunk.group].to(batch),
                        small_dark[chunk.group].to(batch),
                    )
                x = x - b.to(batch)[:, None, None]
                x = x + norm[chunk.group].to(batch)
            return x

        return self._apply_frames(Y, out, fn, overhead=3.0)

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
    """Spatial shape of ``maps`` after :func:`resize` to ``size``."""
    return tuple(maps.shape[-2:]) if size is None else tuple(size)
