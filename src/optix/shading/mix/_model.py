"""Variational fit of a mixture model on one group of pictures.

The estimator ([`GammaMixture`][optix.shading.GammaMixture]) runs one
`MixtureFit` per group. This module holds the inference loop, the
background class and the sensor / shading updates; everything that
depends on the distribution of the tissue signal (its prior, the
integral over u and the update of its parameters) is delegated to a
`Component` (see `_gamma.GammaComponent`).

Per-pixel statistics accumulated by the local step (all posterior
expectations, summed over frames), with

    r  = y − c − m·u                 (residual)
    σ² = v + g·m·ũ                   (variance; ũ = u, or 0)
    q  = (r² + δ − σ²) / (2σ⁴)       (score with respect to the variance,
                                      δ = quantisation² / 12)

| index | quantity  |
|-------|-----------|
| HCC   | 1/σ²      |
| GC    | r/σ²      |
| XU    | u/σ²      |
| G1    | u·r/σ²    |
| AU    | u²/σ²     |
| HVV   | 1/(2σ⁴)   |
| GV    | q         |
| WU    | ũ/(2σ⁴)   |
| G2    | ũ·q       |
| BU    | ũ²/(2σ⁴)  |
"""

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from ...basis import dct_basis, dct_bending, gram, project, reconstruct
from ...distributions import Dirichlet, kl_divergence
from ...io import (
    ArrayLike,
    Layout,
    WritableArrayLike,
    frame_batches,
    row_tiles,
)
from ...special import fit_dirichlet
from ...stats import tile_reduce
from ...stats._base import state_dtype
from ._overlap import Overlap, OverlapStats

__all__ = [
    "Component",
    "MixtureFit",
    "detect_quantization",
    "fit_modulation",
    "is_integer_valued",
]

HCC, GC, XU, G1, AU, HVV, GV, WU, G2, BU = range(10)
NSTATS = 10

_DIRICHLET_BOUNDS = (1e-3, 1e3)
_DIRICHLET_ITER = 32
_MAX_SAMPLES = 1_000_000
# Chunk-sized temporaries of the local step: the running (10, n, H, W)
# statistics, those of the class being integrated and the quadrature
# temporaries (`_OVERHEAD`), plus log p(y, z = k) and E[T(u) | y, z = k]
# of every tissue class, kept until the responsibilities are normalised
# (`_OVERHEAD_CLASS` per class).
_OVERHEAD = 40.0
_OVERHEAD_CLASS = 3.5
# Elements of the temporaries of one block of the cell pass (overlaps)
_CELL_READ = float(1 << 24)


class Component(Protocol):
    """Distribution of the signal u in the K tissue classes.

    Class parameters are a `(K, P)` tensor, one row per class, with the
    columns named by `names`.
    """

    names: tuple[str, ...]
    nstats: int  # number of per-class sufficient statistics of u
    mean_index: int  # which statistic is E[u]

    def loglik(
        self,
        Y: Tensor,
        C: Tensor,
        V: Tensor,
        M: Tensor,
        gain: Tensor,
        params: Tensor,
        *,
        dq: float,
        full: bool,
    ) -> tuple[Tensor, tuple[Tensor, ...], Tensor]:
        """log p(y | z = class), E[T(u) | y, z] and E[stats | y, z].

        For one class (`params` is a `(P,)` row); the statistics are
        those of the module docstring, `(NSTATS, *Y.shape)`.
        """
        ...

    def fit(self, weight: Tensor, stats: Tensor, previous: Tensor) -> Tensor:
        """New `(K, P)` parameters from the weighted sums of T(u).

        `weight` is `(K,)`, `stats` `(K, nstats)` (in the accumulator
        dtype, float64 where available); classes without weight keep
        `previous`.
        """
        ...

    def init(self, values: Tensor, nclass: int) -> tuple[Tensor, Tensor]:
        """Parameters and fractions of the classes from tissue values."""
        ...

    def default(
        self, nclass: int, scale: Tensor, backend: dict
    ) -> tuple[Tensor, Tensor]:
        """Parameters and fractions without enough tissue values."""
        ...

    def clamp(self, params: Tensor) -> Tensor:
        """Parameters projected onto their admissible set."""
        ...

    def mean(self, params: Tensor) -> Tensor:
        """Mean of u in each class, `(K,)`."""
        ...


@dataclass
class Prior:
    """Prior over the class model, and the regularisers of c, v, θ, g."""

    alpha: Tensor  # (K+1,) Dirichlet concentrations, background last
    params: Tensor  # (K, P) parameters of the tissue classes
    v: Tensor | None  # prior variance of an empty pixel
    v_dof: float
    lam: float
    gain_sd: float


@dataclass
class Posterior:
    """q(π) per picture, and the class fit of the aggregated q(u | z)."""

    alpha: Tensor  # (*sample, K+1)
    params: Tensor  # (K, P)


@dataclass
class Params:
    """Point estimates: offset c, variance v, log-modulation θ, gain g."""

    C: Tensor  # (H, W)
    V: Tensor  # (H, W)
    M: Tensor  # (H, W)
    theta: Tensor  # (ky, kx), theta[0, 0] = 0
    gain: Tensor  # ()
    A: Tensor | None = None  # (T,) log tile gains (overlapping tiles)


def _dev(t: Tensor) -> dict:
    return dict(dtype=t.dtype, device=t.device)


def _halve_(current: Tensor, previous: Tensor) -> Tensor:
    return current.mul_(0.5).add_(previous, alpha=0.5)


def _loghalve_(current: Tensor, previous: Tensor) -> Tensor:
    return current.mul_(previous).sqrt_()


class MixtureFit:
    """State and updates of the variational fit on one group.

    Parameters
    ----------
    source : array-like
        Data of the group (no group axes).
    layout : Layout
        Its layout: two spatial axes, every other axis a sample.
    """

    def __init__(
        self,
        source: ArrayLike,
        layout: Layout,
        *,
        component: Component,
        nclass: int,
        noise: str,
        order: tuple[int, int],
        gain: float,
        quantization: float,
        alpha: Tensor,
        params: Tensor,
        v_prior: float | None,
        v_dof: float,
        lam: float,
        gain_sd: float,
        backend: dict,
        overlap: Overlap | None = None,
    ) -> None:
        self.source = source
        self.layout = layout
        _, sample, spatial = layout.shape_of(source)
        self.sample_shape = tuple(sample)
        self.spatial_shape = tuple(spatial)
        self.backend = backend
        # float64 accumulators, except on devices without float64 (MPS)
        self.state_dtype = state_dtype(backend.get("device"))
        self.component = component
        self.nclass = nclass
        self.full = noise == "full"
        self.quantization = float(quantization)
        self.dq = self.quantization**2 / 12
        self.numel = math.prod(sample) * math.prod(spatial)
        height, width = spatial
        order = (min(order[0], height), min(order[1], width))
        self.by = dct_basis(height, order[0], **backend)
        self.bx = dct_basis(width, order[1], **backend)
        self.bending = dct_bending(spatial, order, order=2, **backend)
        v = None if v_prior is None else torch.as_tensor(v_prior, **backend)
        self.prior = Prior(
            alpha=alpha.to(**backend),
            params=params.to(**backend),
            v=v,
            v_dof=float(v_dof),
            lam=float(lam),
            gain_sd=float(gain_sd),
        )
        self.post = Posterior(
            alpha=torch.ones((*sample, nclass + 1), **backend),
            params=params.to(**backend).clone(),
        )
        self.param = Params(
            C=torch.zeros(spatial, **backend),
            V=torch.ones(spatial, **backend),
            M=torch.ones(spatial, **backend),
            theta=torch.zeros(order, **backend),
            gain=torch.as_tensor(float(gain), **backend),
        )
        self.gain0 = self.param.gain.clone()
        self.history: list[float] = []
        self.halvings: list[int] = []
        self._stats: Tensor | None = None
        self._counts: Tensor | None = None
        self._class_stats: tuple[Tensor, Tensor] | None = None
        self.overlap = overlap
        self._overlap: OverlapStats | None = None
        # tile gains: per-tile terms of the joint (θ, a) step
        self._gains: dict | None = None
        self.exact_gain = False  # test oracle: per-tile maps (small data)
        self._decrement = math.inf
        if overlap is not None:
            self._init_scene(source)
            if overlap.fit_gain:
                self.param.A = torch.zeros(overlap.ntiles, **backend)
                fitted = [
                    overlap.scene.covered(t) for t in range(overlap.ntiles)
                ]
                self._fitted_tiles = torch.tensor(
                    fitted, device=backend.get("device")
                )

    # -- helpers -----------------------------------------------------------
    def modulation(self) -> Tensor:
        """m = exp(Φ·θ) on the pixel grid."""
        return reconstruct(self.param.theta, [self.by, self.bx]).exp()

    def frames(
        self, source: ArrayLike | None = None, layout: Layout | None = None
    ) -> object:
        """Frame batches `(n, H, W)` of the training data (or `source`)."""
        if source is None:
            source, layout = self.source, self.layout
        return frame_batches(
            source,
            layout=layout,
            overhead=_OVERHEAD + _OVERHEAD_CLASS * self.nclass,
            contiguous=True,
            **self.backend,
        )

    def overlap_stats(self) -> OverlapStats | None:
        """Matched log-ratios of the overlapping tiles (current c, v)."""
        if self.overlap is None:
            return None
        prm = self.param
        return self.overlap.stats(
            self.source,
            self.layout,
            prm.C,
            prm.V,
            float(prm.gain),
            self.full,
            self.backend,
        )

    def tile_gain(self) -> Tensor | None:
        """Per-tile gains gₜ (b = gₜ·m), or None unless they are fitted."""
        if self.param.A is None:
            return None
        return self.param.A.exp()

    def _elogpi(self, alpha: Tensor) -> Tensor:
        return Dirichlet(alpha).mean_log

    # -- scene geometry (overlapping tiles) --------------------------------
    def _init_scene(self, source: ArrayLike) -> None:
        """Axes of the tiles: tile axis, the D other axes, their roles."""
        ndim = len(source.shape)
        _, sample, spatial = self.layout.positions(ndim)
        axis = self.overlap.axis
        other = [a for a in range(ndim) if a != axis]
        self._sample_axes = tuple(sample)
        self._tile_index = sample.index(axis)  # among the sample axes
        self._other_axis_of_tile = axis
        self._other = other
        self._sp = [other.index(a) for a in spatial]  # layout order
        self._rest = [a for a in range(len(other)) if a not in self._sp]

    def _single_mask(self, chunk: object) -> Tensor | None:
        """`(n, H, W)` mask of the voxels of a chunk seen by one tile only.

        None when every voxel of the chunk is seen once.
        """
        scene = self.overlap.scene
        samples = chunk.samples
        tiles = range(*samples[self._tile_index].indices(self.overlap.ntiles))
        if not any(scene.covered(t) for t in tiles):
            return None
        box = []
        for a, n in zip(self._other, self._tile_shape(), strict=True):
            if a in self._sample_axes:
                box.append(samples[self._sample_axes.index(a)])
            else:
                box.append(slice(0, n))
        perm = self._rest + self._sp
        masks = [scene.single_mask(t, box).transpose(perm) for t in tiles]
        mask = np.stack(masks, axis=self._tile_index)
        if mask.all():
            return None
        mask = torch.from_numpy(mask).to(self.param.C.device)
        return mask.reshape(-1, *mask.shape[-2:])

    def _tile_shape(self) -> tuple[int, ...]:
        shape = self.source.shape
        return tuple(int(shape[a]) for a in self._other)

    def _cell_pass(
        self,
        elogpi: Tensor,
        stats: Tensor,
        counts: Tensor,
        class_stats: Tensor,
    ) -> Tensor:
        """Local step on the voxels seen by several tiles; the bound.

        Each cell (a box of the scene seen by the same S tiles) is read
        from the S tiles in blocks along its first sample axis, and its
        voxels are integrated with one shared signal. Per-observation
        statistics go to the pixel of each tile (u-weighted ones times
        gₜ, gₜ²), the class counts of a voxel are split over its pictures
        by the posterior of its (uniform) owner, and the class statistics
        and the bound are counted once per voxel.
        """
        ov, prm = self.overlap, self.param
        sdt = self.state_dtype
        bound = torch.zeros((), dtype=sdt, device=prm.C.device)
        sp, rest = self._sp, self._rest
        sample_axes = self._sample_axes
        loga = self._log_tile_gain()
        for cell in ov.scene.cells:
            S = len(cell.tiles)
            local = [list(cell.local(ov.positions[t])) for t in cell.tiles]
            pix = [tuple(loc[a] for a in sp) for loc in local]
            npix = math.prod(cell.shape[a] for a in sp)
            if rest:
                lead = rest[0]
                n = cell.shape[lead]
                inner = math.prod(cell.shape[a] for a in rest[1:])
                per = S * (NSTATS + 12) + _OVERHEAD_CLASS * self.nclass
                step = max(1, int(_CELL_READ / (per * npix * max(inner, 1))))
                blocks = [(b, min(b + step, n)) for b in range(0, n, step)]
            else:
                lead, blocks = None, [(0, 1)]
            gains = loga[list(cell.tiles)].exp()
            C = torch.stack([prm.C[p] for p in pix])[:, None]
            V = torch.stack([prm.V[p] for p in pix])[:, None]
            B = torch.stack([prm.M[p] for p in pix])[:, None]
            B = B * gains.to(B)[:, None, None, None]
            for b0, b1 in blocks:
                idx = []
                for loc in local:
                    loc = list(loc)
                    if lead is not None:
                        s0 = loc[lead].start
                        loc[lead] = slice(s0 + b0, s0 + b1)
                    idx.append(loc)
                Y = torch.stack(
                    [
                        ov._read(self.source, t, loc, sp, rest, self.backend)
                        for t, loc in zip(cell.tiles, idx, strict=True)
                    ]
                )
                where = []  # per tile, its pictures in the per-sample state
                for t, loc in zip(cell.tiles, idx, strict=True):
                    w = []
                    for a in sample_axes:
                        if a == self._other_axis_of_tile:
                            w.append(t)
                        else:
                            w.append(loc[self._other.index(a)])
                    where.append(tuple(w))
                ep = torch.stack(
                    [elogpi[w].reshape(-1, self.nclass + 1) for w in where]
                )
                st, bnd, cnt, cs = self._cell_chunk(Y, C, V, B, ep)
                # statistics in terms of m: u-weighted ones times g, g²
                g = gains.to(st)[:, None, None]
                st[[XU, G1, WU, G2]] *= g
                st[[AU, BU]] *= g.square()
                for k, p in enumerate(pix):
                    stats[(slice(None), *p)] += st[:, k]
                    if prm.A is not None:
                        self._gain_add(cell.tiles[k], st[:, k], p)
                for k, w in enumerate(where):
                    view = counts[w]
                    counts[w] = view + cnt[k].reshape(view.shape)
                class_stats += cs
                bound += bnd
        return bound

    def _cell_chunk(
        self, Y: Tensor, C: Tensor, V: Tensor, B: Tensor, elogpi: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Local step on `(S, n, h, w)` observations of shared voxels.

        `elogpi` is `(S, n, K+1)`: E[log π] of the picture of each
        observation. The class prior of a voxel is the owner mixture
        w̄ₖ = (1/S) ∑ₜ exp(E[log πₜₖ]).
        """
        nk, dq, prm = self.nclass, self.dq, self.param
        S = Y.shape[0]
        sdt = self.state_dtype
        logw = torch.logsumexp(elogpi, 0) - math.log(S)  # (n, K+1)
        R0 = Y - C
        logZ = (
            R0.square()
            .add_(dq)
            .div_(V)
            .add_(V.log())
            .sum(0)
            .mul_(-0.5)
            .add_(logw[:, -1, None, None] - 0.5 * S * math.log(2 * math.pi))
        )
        acc = Y.new_zeros((NSTATS, *Y.shape))
        acc[HCC].copy_(V.reciprocal().expand_as(Y))
        acc[GC].copy_(R0).div_(V)
        acc[HVV].copy_(V.square().reciprocal().mul_(0.5).expand_as(Y))
        acc[GV].copy_(R0).square_().add_(dq).sub_(V).div_(V.square())
        acc[GV].mul_(0.5)
        del R0
        logjoint = [None] * (nk + 1)
        logjoint[nk] = logZ
        suffs = []
        for k in range(nk):
            logZ0 = logZ
            logpY, suff, st = self.component.loglik_shared(
                Y,
                C,
                V,
                B,
                prm.gain,
                self.prior.params[k],
                dq=dq,
                full=self.full,
            )
            logZY = logw[:, k, None, None] + logpY
            logZ = torch.logaddexp(logZ, logZY)
            pz_prv = (logZ0 - logZ).exp_()
            pz_new = (logZY - logZ).exp_()
            acc.mul_(pz_prv).add_(st.mul_(pz_new))
            logjoint[k] = logZY
            suffs.append(suff)
            del logZ0, logpY, suff, st, logZY, pz_prv, pz_new

        # owner of the voxel: p(o = t | k) ∝ exp(E[log π_tk])
        owner = torch.softmax(elogpi, 0)  # (S, n, K+1)
        counts = Y.new_zeros((S, Y.shape[1], nk + 1))
        class_stats = torch.zeros(
            (nk, self.component.nstats), dtype=sdt, device=Y.device
        )
        for k in range(nk + 1):
            Zk = logjoint[k].sub_(logZ).exp_()
            logjoint[k] = None
            counts[:, :, k] = owner[:, :, k] * Zk.sum((-2, -1))
            if k < nk:
                for j, t in enumerate(suffs[k]):
                    class_stats[k, j] = t.mul_(Zk).sum().to(sdt)
                suffs[k] = None
            del Zk
        stats = acc.sum(2).to(sdt)  # (NSTATS, S, h, w)
        bound = logZ.sum().to(sdt)
        return stats, bound, counts, class_stats

    def _log_tile_gain(self) -> Tensor:
        """log gₜ of every tile (zeros unless the gains are fitted)."""
        if self.param.A is not None:
            return self.param.A
        return torch.zeros(self.overlap.ntiles, **self.backend)

    def _frame_tiles(self, chunk: object) -> Tensor:
        """`(n,)` tile of every frame of a chunk."""
        shape = chunk.sample_shape
        ti = self._tile_index
        start = chunk.samples[ti].start or 0
        idx = torch.arange(
            start, start + shape[ti], device=self.param.C.device
        )
        view = [1] * len(shape)
        view[ti] = shape[ti]
        return idx.reshape(view).expand(shape).reshape(-1)

    # -- tile gains: terms of the joint (θ, a) step --------------------------
    def _gain_add(self, t: int, st: Tensor, pix: tuple | None = None) -> None:
        """Add the observations of tile t to the terms of its log gain.

        `st` holds their statistics in terms of m, summed over frames, on
        the pixels `pix`. aₜ is the coefficient of a per-tile constant
        column of the θ system (log b = aₜ + Φθ): its score sₜ is exact,
        its curvature dₜ and cross term Uₜ = Φᵀ(…) with θ come from the
        same per-pixel Fisher information as θ, with the coupling of aₜ to
        the per-pixel c and v left out (the score, hence the fixed point,
        is exact; only the curvature of the step is approximate).
        """
        prm, acc = self.param, self._gains
        pix = pix or (slice(None), slice(None))
        M = prm.M[pix].to(st)
        gM = prm.gain.to(st) * M
        F = M.square() * st[AU] + gM.square() * st[BU]  # Fisher of log gₜ
        Sm = M * st[G1] + gM * st[G2]  # score of log gₜ
        if self.exact_gain:
            Fcm, Fvm = M * st[XU], gM * st[WU]
            maps = acc["maps"][t]
            maps[(slice(None), *pix)] += torch.stack([F, Sm, Fcm, Fvm])
            return
        acc["s"][t] += Sm.sum()
        acc["d"][t] += F.sum()
        bases = [self.by[pix[0]], self.bx[pix[1]]]
        acc["U"][:, t] += project(F, [b.to(F) for b in bases]).flatten()

    def _gain_system(
        self,
        Fcm: Tensor,
        Fvm: Tensor,
        Gc: Tensor,
        Gv: Tensor,
        iHcc: Tensor,
        iHvv: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Score s `(T,)`, curvature D `(T, T)` and cross term U `(k, T)`.

        Default: the accumulated terms, D diagonal (no coupling to c, v).
        Exact (test oracle, per-tile maps): with c and v eliminated.
        """
        acc = self._gains
        if not self.exact_gain:
            return acc["s"], acc["d"].diag(), acc["U"]
        maps = acc["maps"].to(Fcm)
        F, Sm, Fc, Fv = maps.unbind(1)  # (T, H, W) each
        s = (Sm - Fc * (Gc * iHcc) - Fv * (Gv * iHvv)).sum((-2, -1))
        Fx = F - Fc * (Fcm * iHcc) - Fv * (Fvm * iHvv)
        bases = [self.by.to(Fx), self.bx.to(Fx)]
        U = torch.stack([project(f, bases).flatten() for f in Fx], 1)
        D = torch.diag(F.sum((-2, -1)))
        D -= torch.einsum("thw,shw->ts", Fc * iHcc, Fc)
        D -= torch.einsum("thw,shw->ts", Fv * iHvv, Fv)
        return s, D, U

    # -- initialisation ----------------------------------------------------
    def init(
        self,
        bg_fraction: float = 0.5,
        threshold: float = 3.0,
        modulation: bool = True,
        *,
        spread_floor: bool = False,
        offset: Tensor | None = None,
        variance: Tensor | None = None,
        flat: Tensor | None = None,
        classes: tuple[Tensor, Tensor, Tensor | None] | None = None,
    ) -> None:
        """Data-driven starting point; given fields replace their estimate.

        `classes` is `(params, frac, alpha)`: parameters and fractions of
        the tissue classes, and (optionally) the prior concentrations.
        With `spread_floor`, the initial standard deviation of an empty
        pixel is at least Δ/√12 (v ≥ Δ²/12). A given flat field is used
        as is (normalised to a geometric mean of one) when `modulation`
        is False, and projected onto the DCT basis otherwise.
        """
        prm, prior, post = self.param, self.prior, self.post
        backend = self.backend
        height, width = self.spatial_shape
        nframes, nclass = math.prod(self.sample_shape), self.nclass
        floor = self.quantization / math.sqrt(12) if spread_floor else 0.0

        # Pass 1, over row tiles holding every frame (order statistics):
        # baseline and spread from the lower order statistics of each pixel.
        if offset is None or variance is None:
            k_med = min(max(round(0.5 * bg_fraction * nframes), 1), nframes)
            k_low = min(max(round(0.1587 * bg_fraction * nframes), 1), k_med)

            def lower(t: Tensor) -> tuple[Tensor, Tensor]:
                med = t.kthvalue(k_med, -1).values
                return med, med - t.kthvalue(k_low, -1).values

            C, sd = tile_reduce(
                self.source, lower, layout=self.layout, overhead=3.0, **backend
            )
            sd_typical = sd.median()
            if not sd_typical > 0:
                sd_typical = self._typical_deviation(C)
            sd_typical = sd_typical.clamp_min(floor)
            sd = sd.clamp(0.25 * sd_typical, 4 * sd_typical).clamp_min(floor)
            V = sd.square()
        if offset is not None:
            C = offset.to(**backend).expand(height, width).clone()
            if not C.isfinite().all():
                raise ValueError("The initial offset must be finite")
        if variance is not None:
            V = _valid_variance(
                variance.to(**backend).expand(height, width), floor**2
            )
            sd = V.sqrt()
            sd_typical = sd.median()
        if self.dq > 0 and V.median() < 0.999 * self.dq:
            warnings.warn(
                f"The initial variance of an empty pixel (median "
                f"{V.median().item():.3g}) is below the quantisation "
                f"variance Δ²/12 = {self.dq:.3g}: the background class may "
                f"not fit. `spread_floor=True` floors the initial spread at "
                f"Δ/√12.",
                RuntimeWarning,
                stacklevel=3,
            )

        theta = torch.zeros_like(prm.theta)
        M = None
        if flat is not None:
            flat = flat.to(**backend).expand(height, width)
            bad = int((~(flat.isfinite() & (flat > 0))).sum())
            if bad:
                raise ValueError(
                    f"The initial flat field must be positive and finite "
                    f"({bad} pixels are not)"
                )
            logm = flat.log()
            theta = project(logm, [self.by, self.bx])
            theta[0, 0] = 0
            prm.theta = theta
            if modulation:
                M = self.modulation()
            else:
                # m is held fixed: the given flat field itself, with a
                # geometric mean of one (θ is only its projection)
                M = logm.sub(logm.mean()).exp_()

        # Pass 2, over frame batches: initial labels and signal. Every
        # batch contributes about the same share of the (at most
        # `_MAX_SAMPLES`) tissue values used to initialise the classes.
        # For M: per pixel, the number of tissue values and the sum of
        # their excess over the threshold. For an exponential signal,
        # E[r − t | r > t] = M·μ whatever the threshold t
        # (memorylessness), so the excess is not biased by the truncation
        # where M is small.
        chunks = self.frames()
        nbatches = len(chunks)
        n_tissue = torch.empty(self.sample_shape, **backend)
        cut = threshold * sd
        fg_count = torch.zeros((height, width), **backend)
        fg_excess = torch.zeros((height, width), **backend)
        moments = torch.zeros(3, dtype=self.state_dtype, device=C.device)
        values = []
        for chunk in chunks:
            y = chunk.data
            resid = y - C
            tissue = resid > cut
            n_tissue[chunk.samples] = (
                tissue.sum((-2, -1)).to(y.dtype).reshape(chunk.sample_shape)
            )
            vals = (resid if M is None else resid / M)[tissue]
            step = max(1, vals.numel() * nbatches // _MAX_SAMPLES)
            values.append(vals[::step])
            excess = resid.sub_(cut).mul_(tissue)  # resid is overwritten
            fg_count += tissue.sum(-3)
            fg_excess += excess.sum(-3)
            moments += torch.stack(
                [tissue.sum(), excess.sum(), excess.square().sum()]
            ).to(moments)
        values = torch.cat(values)

        if flat is None:
            if modulation and moments[0] > 0:
                n, s1, s2 = moments.tolist()
                mean = s1 / n
                # Information about log M per tissue value: mean² /
                # variance (= ν for a Gamma signal without noise)
                info = mean**2 / max(s2 / n - mean**2, 1e-12 * mean**2)
                extra = None
                if self.overlap is not None:
                    prm.C, prm.V = C, V
                    ov = self._overlap = self.overlap_stats()
                    if ov.terms:
                        zero = torch.zeros_like(C)
                        extra = ov.system(zero, [self.by, self.bx])
                theta = fit_modulation(
                    fg_excess,
                    fg_count,
                    self.by,
                    self.bx,
                    prior.lam * self.bending,
                    info=info,
                    extra=extra,
                )
            prm.theta = theta
            M = self.modulation()
        prm.M = M
        if prm.A is not None:
            # initial tile gains from the log-ratios of the overlaps (an
            # initialiser only: they do not enter the bound)
            prm.C, prm.V = C, V
            if self._overlap is None:
                self._overlap = self.overlap_stats()
            ov = self._overlap
            ov.fit_gain = True
            a = ov.log_gain(M.log()).to(prm.A)
            a = torch.where(self._fitted_tiles, a, 0)
            fitted = self._fitted_tiles
            a[fitted] -= a[fitted].mean()
            prm.A = a

        n_tissue = n_tissue.reshape(-1)
        alpha = None
        component = self.component
        if classes is not None:
            params, frac, alpha = (
                None if t is None else t.to(**backend) for t in classes
            )
            params = component.clamp(params)
        elif values.numel() >= 2 * nclass:
            # The tissue values are raw when M is estimated here (it is
            # close to 1, and the k-means is sensitive to small
            # perturbations), and divided by M when the flat was given.
            params, frac = component.init(values, nclass)
        else:
            params, frac = component.default(
                nclass, threshold * sd_typical, backend
            )

        if alpha is not None:
            prior.alpha = alpha.clone()
        prior.params = params.clone()
        post.params = params.clone()
        if prior.v is None:
            prior.v = V.median()

        counts = torch.cat(
            [n_tissue[:, None] * frac, (height * width - n_tissue)[:, None]],
            dim=1,
        )
        post.alpha = (prior.alpha + counts).reshape(post.alpha.shape)
        prm.C, prm.V = C, V

    def _typical_deviation(self, C: Tensor) -> Tensor:
        """Median over pixels of the per-pixel median of |y − C|.

        (The prototype takes the median of per-tile medians, which
        depends on the tiling; both agree when every pixel has the same
        distribution.)
        """
        k = (math.prod(self.sample_shape) + 1) // 2
        dev = torch.empty_like(C)
        tiles = row_tiles(
            self.source,
            layout=self.layout,
            axis=0,
            overhead=3.0,
            **self.backend,
        )
        for chunk in tiles:
            block = chunk.data
            rows = block.reshape(-1, *block.shape[-2:])
            r = rows.sub_(C[chunk.spatial]).abs_().kthvalue(k, 0).values
            chunk.store(dev, r, into="map")
        return dev.median().clamp_min(1e-3)

    # -- local step: q(z, u) -----------------------------------------------
    def update_latent(self) -> float:
        """Update q(z, u), accumulate the statistics; return the bound.

        The bound is

            ∑ᵢₖ log ∑_z exp(E[log π_{kz}]) · p(yᵢₖ | z)
            − ∑ₖ KL(q(πₖ) ‖ p(πₖ)) + log p(θ) + log p(v) + log p(g)
        """
        prm, prior, post = self.param, self.prior, self.post
        height, width = self.spatial_shape
        nk = self.nclass
        if prior.v is None:  # `init` was skipped
            prior.v = prm.V.median()
        elogpi = self._elogpi(post.alpha)

        self._stats, self._counts, self._class_stats = None, None, None
        acc = dict(dtype=self.state_dtype, device=prm.C.device)
        stats = torch.zeros((NSTATS, height, width), **acc)
        counts = prm.C.new_zeros((*self.sample_shape, nk + 1))
        class_stats = torch.zeros((nk, self.component.nstats), **acc)
        bound = torch.zeros((), **acc)
        scene = self.overlap is not None
        gains = prm.A is not None
        if gains:
            T = self.overlap.ntiles
            k = self.by.shape[1] * self.bx.shape[1]
            self._gains = dict(
                s=torch.zeros(T, **acc),
                d=torch.zeros(T, **acc),
                U=torch.zeros((k, T), **acc),
            )
            if self.exact_gain:
                self._gains["maps"] = torch.zeros((T, 4, height, width), **acc)
        for chunk in self.frames():
            ep = elogpi[chunk.samples].reshape(-1, nk + 1)
            mask = self._single_mask(chunk) if scene else None
            tiles = self._frame_tiles(chunk) if gains else None
            st, bnd, cnt, cs = self._latent_chunk(chunk.data, ep, mask, tiles)
            stats += st
            counts[chunk.samples] = cnt.reshape(*chunk.sample_shape, nk + 1)
            class_stats += cs
            bound += bnd
        if scene:
            bound += self._cell_pass(elogpi, stats, counts, class_stats)
        self._stats = stats.to(prm.C.dtype)
        self._counts = counts
        self._class_stats = class_stats

        # Terms of the bound that do not depend on the data
        bound = bound.item()
        if self.quantization > 0:
            bound += self.numel * math.log(self.quantization)  # H[q(y*)]
        q_pi, p_pi = Dirichlet(post.alpha), Dirichlet(prior.alpha)
        bound -= kl_divergence(q_pi, p_pi).sum().item()
        bound -= (
            0.5 * prior.lam * (self.bending * prm.theta.square()).sum().item()
        )
        V = prm.V
        if prior.v_dof > 0:
            bound -= (
                0.5
                * prior.v_dof
                * V.log().add_(V.reciprocal(), alpha=prior.v).sum().item()
            )
        bound -= (
            0.5 * ((prm.gain / self.gain0).log().item() / prior.gain_sd) ** 2
        )
        self.history.append(bound)
        return bound

    def _latent_chunk(
        self,
        Y: Tensor,
        elogpi: Tensor,
        mask: Tensor | None = None,
        tiles: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Local step on `(n, H, W)` frames with `(n, K+1)` E[log π].

        With a boolean `mask` `(n, H, W)`, only the voxels where it is
        True contribute (the others are seen by several tiles and belong
        to the cell pass). With fitted tile gains, `tiles` `(n,)` gives the
        tile of every frame: its flat field is b = gₜ·m.
        """
        C, V = self.param.C, self.param.V
        M = None
        if tiles is not None:
            g = self.param.A[tiles].exp()[:, None, None]
            M = self.param.M * g
        nk = self.nclass
        dq = self.dq

        # Background class: u = 0, variance V
        R0 = Y - C
        logZ = (
            R0.square()
            .add_(dq)
            .div_(V)
            .add_(V.log())
            .mul_(-0.5)
            .add_(elogpi[:, -1, None, None] - 0.5 * math.log(2 * math.pi))
        )
        acc = Y.new_zeros((NSTATS, *Y.shape))
        acc[HCC].copy_(V).reciprocal_()
        acc[GC].copy_(R0).div_(V)
        acc[HVV].copy_(V).square_().reciprocal_().mul_(0.5)
        acc[GV].copy_(R0).square_().add_(dq).sub_(V).div_(V.square())
        acc[GV].mul_(0.5)
        del R0
        logjoint = [None] * (nk + 1)
        logjoint[nk] = logZ

        # Tissue classes, folded in with a running log-sum-exp so that the
        # per-class statistics never need to be stored.
        suffs = []
        for k in range(nk):
            logZ0 = logZ
            logpY, suff, st = self._loglik(Y, k, M)
            logZY = elogpi[:, k, None, None] + logpY  # log p(y, z=k)
            logZ = torch.logaddexp(logZ, logZY)  # log ∑ p(y, z)
            pz_prv = (logZ0 - logZ).exp_()
            pz_new = (logZY - logZ).exp_()
            acc.mul_(pz_prv).add_(st.mul_(pz_new))
            logjoint[k] = logZY
            suffs.append(suff)
            del logZ0, logpY, suff, st, logZY, pz_prv, pz_new

        # Each class is freed once its responsibilities are used
        sdt = self.state_dtype
        nstats = self.component.nstats
        sum_Z = Y.new_zeros((len(Y), nk + 1))
        class_stats = torch.zeros((nk, nstats), dtype=sdt, device=Y.device)
        for k in range(nk + 1):
            Zk = logjoint[k].sub_(logZ).exp_()
            logjoint[k] = None
            if mask is not None:
                Zk = torch.where(mask, Zk, 0)
            sum_Z[:, k] = Zk.sum((-2, -1))
            if k < nk:
                for j, t in enumerate(suffs[k]):
                    class_stats[k, j] = t.mul_(Zk).sum().to(sdt)
                suffs[k] = None
            del Zk

        # Sums over the chunk in the working dtype (a float64 sum would
        # first cast the whole chunk), accumulated across chunks in `sdt`
        if mask is not None:
            acc = torch.where(mask, acc, 0)
            logZ = torch.where(mask, logZ, 0)
        bound = logZ.sum().to(sdt)
        if tiles is not None:
            # in terms of m: u-weighted statistics times g, g²
            acc[[XU, G1, WU, G2]] *= g
            acc[[AU, BU]] *= g.square()
            stats = torch.zeros_like(acc[:, 0], dtype=sdt)
            for t in tiles.unique().tolist():
                st = acc[:, tiles == t].sum(1).to(sdt)
                self._gain_add(t, st)
                stats += st
            return stats, bound, sum_Z, class_stats
        stats = acc.sum(-3).to(sdt)  # sum over frames
        return stats, bound, sum_Z, class_stats

    def _loglik(
        self, Y: Tensor, k: int, M: Tensor | None = None
    ) -> tuple[Tensor, tuple[Tensor, ...], Tensor]:
        prm = self.param
        return self.component.loglik(
            Y,
            prm.C,
            prm.V,
            prm.M if M is None else M,
            prm.gain,
            self.prior.params[k],
            dq=self.dq,
            full=self.full,
        )

    # -- parameter step: c, v, θ, g ----------------------------------------
    def update_parameters(
        self,
        c: bool = True,
        v: bool = True,
        theta: bool = True,
        gain: bool = False,
        lr: float = 1.0,
        max_log_step: float = 0.5,
    ) -> None:
        """One joint Fisher-scoring step, from the last local step.

        The complete-data Fisher information is block-diagonal across
        pixels in (c, v), and couples them to (θ, log g) only through
        per-pixel terms. c and v are eliminated (Schur complement), which
        leaves a dense system of the size of the basis for θ and a scalar
        equation for log g; they are then back-substituted.
        """
        if self._stats is None:
            raise RuntimeError("Run update_latent() first")
        prm, prior = self.param, self.prior
        Hcc, Gc, Hcm, Gm, Hmm, Hvv, Gv, Hvm, Gmg, Hmg = self._stats.unbind(0)
        M, V, gam = prm.M, prm.V, prm.gain
        gain = gain and self.full

        if prior.v_dof > 0:
            hvp = V.square().reciprocal_().mul_(prior.v_dof * 0.5)
            gvp = (prior.v - V).mul_(hvp)
            Gv = Gv + gvp
            Hvv = Hvv + hvp

        gM = gam * M
        Fcm = M * Hcm  # Fisher(c, log m)
        Fvm = gM * Hvm  # Fisher(v, log m) = Fisher(v, log g)
        Fmm = M.square().mul_(Hmm)  # Fisher(log m, log m), mean part
        Vmm = gM.square().mul_(Hmg)  # Fisher(log m, log m), variance part
        Sm = M.mul(Gm).addcmul(gM, Gmg)  # score of log m
        Sg = gM * Gmg  # score of log g (per pixel)
        tiny = torch.finfo(M.dtype).tiny
        iHcc = Hcc.clamp_min(tiny).reciprocal_()
        iHvv = Hvv.clamp_min(tiny).reciprocal_()

        # Eliminate c and v
        Fmm += Vmm
        if c:
            Fmm.sub_(Fcm.square().mul_(iHcc))
            Sm.sub_(Fcm.mul(Gc).mul_(iHcc))
        if v:
            tmp = Fvm.square().mul_(iHvv)
            Fmm.sub_(tmp)
            Vmm.sub_(tmp)
            tmp = Fvm.mul(Gv).mul_(iHvv)
            Sm.sub_(tmp)
            Sg.sub_(tmp)
        Fmm.clamp_min_(0)
        Vmm.clamp_min_(0)

        # Reduced system in θ
        dlogm = torch.zeros_like(M)
        dtheta = torch.zeros_like(prm.theta)
        if theta:
            bases = [self.by, self.bx]
            weights = prior.lam * self.bending.flatten()
            H = gram(Fmm, bases)
            H.diagonal().add_(weights)
            H.diagonal().add_(H.diagonal().mean(), alpha=1e-8)
            g = project(Sm, bases).flatten()
            g.addcmul_(weights, prm.theta.flatten(), value=-1)
            # No DC term: the geometric mean of m stays at one
            H[0, :] = H[:, 0] = 0
            H[0, 0] = 1
            g[0] = 0
        da = None
        self._decrement = 0.0
        if prm.A is not None and self._gains is not None:
            dtheta, da = self._joint_step(
                H if theta else None,
                g if theta else None,
                *self._gain_system(Fcm, Fvm, Gc, Gv, iHcc, iHvv),
            )
            dtheta = dtheta.reshape(prm.theta.shape)
            # c, v part of the decrement (their per-pixel blocks)
            if c:
                self._decrement += (Gc.square() * iHcc).sum().item()
            if v:
                self._decrement += (Gv.square() * iHvv).sum().item()
        elif theta:
            dtheta = torch.linalg.solve(H, g).reshape(prm.theta.shape)
        if theta or da is not None:
            dlogm = reconstruct(dtheta, [self.by, self.bx])
            largest = max(dlogm.abs().max().item(), 1e-12)
            if da is not None:
                largest = max(largest, da.abs().max().item())
            shrink = min(1.0, max_log_step / largest)
            dtheta *= shrink
            dlogm *= shrink
            if da is not None:
                da *= shrink

        # Gain: scalar step on log g, with v eliminated
        dloggain = 0.0
        if gain:
            deviation = (gam / self.gain0).log().item()
            score = Sg.sum().item() - deviation / prior.gain_sd**2
            curvature = Vmm.sum().item() + 1 / prior.gain_sd**2
            dloggain = max(-max_log_step, min(max_log_step, score / curvature))

        # Back-substitution (without the c, v part of the gain step)
        cross = None
        if da is not None and self.exact_gain:
            maps = self._gains["maps"].to(da)
            full_da = torch.zeros(len(maps), dtype=da.dtype, device=da.device)
            full_da[self._fitted_tiles] = da
            cross = torch.einsum(
                "t,cthw->chw", full_da, maps[:, 2:].transpose(0, 1)
            )
        if c:
            dC = Gc.addcmul(Fcm, dlogm, value=-1)
            if cross is not None:
                dC.sub_(cross[0])
            prm.C.add_(dC.mul_(iHcc), alpha=lr)
        if v:
            dV = (dlogm + dloggain).mul_(Fvm)
            if cross is not None:
                dV.add_(cross[1])
            dV = dV.sub_(Gv).neg_().mul_(iHvv)
            prm.V = torch.maximum(prm.V.add(dV, alpha=lr), prm.V.mul(0.1))
        if theta:
            prm.theta.add_(dtheta, alpha=lr)
            prm.M = self.modulation()
        if da is not None:
            A = prm.A.clone()
            A[self._fitted_tiles] += lr * da.to(A)
            prm.A = A
        if gain:
            prm.gain = gam * math.exp(lr * dloggain)

    def _joint_step(
        self,
        H: Tensor | None,
        g: Tensor | None,
        s: Tensor,
        D: Tensor,
        U: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Joint Fisher-scoring step of θ and the fitted log gains a.

        a enters as extra columns of the θ system. When every tile is
        fitted, the common level of a is free in the step (μ is held
        fixed) and is moved into μ afterwards (`_recentre`, exact gauge).
        Otherwise the tiles without overlap stay at a = 0 and ∑ a = 0 over
        the fitted tiles is a constraint of the step (a = Q·b). If the
        Schur complement of θ is not positive definite, θ and a take
        separate (block) steps. Sets the Newton decrement δ = stepᵀ·score.
        """
        F = self._fitted_tiles
        nF = int(F.sum())
        dtype = (H if H is not None else D).dtype
        s, D, U = s[F].to(dtype), D[F][:, F].to(dtype), U[:, F].to(dtype)
        if bool(F.all()):
            Q = torch.eye(nF, dtype=dtype, device=D.device)
        else:
            eye = torch.eye(nF, dtype=dtype, device=D.device)
            Q = torch.linalg.qr((eye - 1 / nF)[:, : nF - 1])[0]
        Dq = Q.T @ D @ Q
        Dq.diagonal().add_(Dq.diagonal().abs().mean(), alpha=1e-8)
        sq = Q.T @ s
        if H is None:
            b = torch.linalg.solve(Dq, sq)
            self._decrement = float(b @ sq)
            k = self.param.theta.numel()
            return torch.zeros(k, dtype=dtype, device=D.device), Q @ b
        k = len(H)
        Uq = U.to(H) @ Q
        Uq[0] = 0  # θ₀₀ is pinned
        schur = Dq - Uq.T @ torch.linalg.solve(H, Uq)
        if torch.linalg.eigvalsh(schur).min() > 0:
            A = torch.cat([torch.cat([H, Uq], 1), torch.cat([Uq.T, Dq], 1)], 0)
            rhs = torch.cat([g, sq])
            x = torch.linalg.solve(A, rhs)
            dtheta, b = x[:k], x[k:]
            self._decrement = float(x @ rhs)
        else:
            dtheta = torch.linalg.solve(H, g)
            b = torch.linalg.solve(Dq, sq)
            self._decrement = float(dtheta @ g + b @ sq)
        return dtheta, Q @ b

    def _recentre(self) -> float:
        """Move the mean log gain into the class means (exact gauge).

        g → g·e⁻ˢ for every tile with μ → μ·eˢ leaves the likelihood and
        the bound unchanged. Only when every tile is fitted (otherwise the
        step keeps ∑ a = 0 itself). Returns s.
        """
        prm = self.param
        if prm.A is None or not bool(self._fitted_tiles.all()):
            return 0.0
        shift = prm.A.mean()
        prm.A = prm.A - shift
        mi = self.component.mean_index
        for params in (self.prior.params, self.post.params):
            params[:, mi] *= shift.exp().to(params)
        return float(shift)

    # -- mixture step: q(π), then μ, ν, α ----------------------------------
    def update_post(self) -> None:
        """q(πₖ) = Dirichlet(α + ∑ᵢ zᵢₖ); class fit of the aggregated q(u)."""
        if self._counts is None:
            raise RuntimeError("Run update_latent() first")
        prior, post = self.prior, self.post
        post.alpha = (prior.alpha + self._counts).reshape(post.alpha.shape)
        weight = self._counts[..., :-1].reshape(-1, self.nclass)
        weight = weight.sum(0, dtype=self.state_dtype)
        post.params = self.component.fit(
            weight, self._class_stats, post.params
        )

    def update_prior(self, alpha: bool = True) -> None:
        """M-step of the class parameters, and of α (optional)."""
        prior, post = self.prior, self.post
        prior.params = post.params.clone()
        if alpha:
            a = post.alpha.reshape(-1, post.alpha.shape[-1])
            if len(a) > 1:
                elogpi = self._elogpi(a).mean(0)
                prior.alpha = fit_dirichlet(
                    elogpi,
                    prior.alpha,
                    _DIRICHLET_ITER,
                    bounds=_DIRICHLET_BOUNDS,
                )

    # -- driver ------------------------------------------------------------
    def run(
        self,
        niter: int,
        tol: float,
        learn_modulation: bool,
        learn_alpha: bool,
        lr: float,
        max_halvings: int,
        max_log_step: float,
        callback: Callable[[int, float], object] | None = None,
    ) -> tuple[bool, str]:
        """Alternate the local, mixture and parameter steps.

        Returns whether the bound converged, and why the loop stopped.
        """
        prm = self.param
        slack = tol * self.numel
        previous = None
        last = -math.inf
        converged, reason = False, f"max_iterations={niter} reached"
        for n in range(niter):
            bound = self.update_latent()

            halvings = 0
            while (
                previous is not None
                and not bound >= last - slack  # also when not finite
                and halvings < max_halvings
            ):
                halvings += 1
                _halve_(prm.C, previous[0])
                _halve_(prm.V, previous[1])
                _halve_(prm.theta, previous[2])
                _loghalve_(prm.gain, previous[3])
                if prm.A is not None:
                    _halve_(prm.A, previous[4])
                if learn_modulation:  # otherwise m stays as given
                    prm.M = self.modulation()
                self.history.pop()
                bound = self.update_latent()
            self.halvings.append(halvings)
            if not math.isfinite(bound):
                warnings.warn(
                    f"The bound is not finite at iteration {n} (non-finite "
                    f"data or initial values?): the fit stopped.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return False, f"the bound is not finite at iteration {n}"
            if callback is not None:
                callback(n, bound)
            if prm.A is None and abs(bound - last) < slack:
                # q(z, u) is already consistent with the parameters
                converged = True
                reason = f"bound change below tol={tol:g} per data point"
                break
            last = bound

            previous = [t.clone() for t in (prm.C, prm.V, prm.theta, prm.gain)]
            self.update_post()
            self.update_prior(alpha=learn_alpha)
            if prm.A is not None:
                # after the class step (its statistics are in the old gauge)
                self._recentre()
                previous.append(prm.A.clone())
            self.update_parameters(
                theta=learn_modulation, lr=lr, max_log_step=max_log_step
            )
            if prm.A is not None and 0.5 * self._decrement < slack:
                # with tile gains the bound changes slowly along the
                # gain/ramp direction, and the class step is not halved
                # (the bound wanders by a few units): stop on the Newton
                # decrement of (c, v, θ, a), the predicted gain of the step
                converged = True
                reason = f"Newton decrement below tol={tol:g} per data point"
                self._recentre()
                self.update_latent()
                break
        else:
            # Make the statistics and the bound consistent with the final
            # parameters (the mixture was updated after the last local
            # step).
            if prm.A is not None:
                self._recentre()
            self.update_latent()
        return converged, reason

    # -- responsibilities --------------------------------------------------
    def class_loglik(
        self, Y: Tensor, mean: bool
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """log p(y | background), log p(y | z = k), and E[u | y, z = k]."""
        prm = self.param
        V = prm.V
        logbg = (
            (Y - prm.C)
            .square_()
            .add_(self.dq)
            .div_(V)
            .add_(V.log())
            .add_(math.log(2 * math.pi))
            .mul_(-0.5)
        )
        logfg, EU = [], []
        for k in range(self.nclass):
            logpY, suff, _ = self._loglik(Y, k)
            logfg.append(logpY)
            if mean:
                EU.append(suff[self.component.mean_index])
        return logbg, torch.stack(logfg), (torch.stack(EU) if mean else None)

    def posterior(
        self,
        source: ArrayLike,
        layout: Layout,
        alpha: Tensor,
        *,
        out: WritableArrayLike | Sequence[WritableArrayLike] | None,
        out_mean: WritableArrayLike | None,
        per_class: bool,
        niter: int,
        tol: float = 0.0,
    ) -> Tensor:
        """Class probabilities (and E[u | y]) of every pixel and frame.

        `alpha` is `(*sample, K+1)`: q(π) of each frame (updated in place
        by `niter` q(z) / q(π) iterations with fixed likelihoods). `out` is
        one output (foreground probability), one per tissue class, or
        None (only the mean).
        """
        mean = out_mean is not None
        _, _, spatial = layout.shape_of(source)
        npix = math.prod(spatial)
        prior_alpha = self.prior.alpha
        for chunk in self.frames(source, layout):
            y = chunk.data
            logbg, logfg, EU = self.class_loglik(y, mean)
            a = alpha[chunk.samples].reshape(-1, self.nclass + 1)
            for _ in range(niter):  # q(π) <-> q(z), fixed likelihoods
                elogpi = self._elogpi(a)
                lbg = logbg + elogpi[:, -1, None, None]
                lfg = logfg + elogpi[:, :-1].T[..., None, None]
                logZ = torch.logaddexp(torch.logsumexp(lfg, 0), lbg)
                counts = torch.cat(
                    [
                        (lfg - logZ).exp_().sum((-2, -1)).T,
                        (lbg - logZ).exp_().sum((-2, -1))[:, None],
                    ],
                    -1,
                )
                a_new = prior_alpha + counts
                done = (a_new - a).abs().max() < tol * npix
                a = a_new
                if done:
                    break
            alpha[chunk.samples] = a.reshape(*chunk.sample_shape, -1)

            # Final responsibilities
            elogpi = self._elogpi(a)
            logbg.add_(elogpi[:, -1, None, None])
            logfg.add_(elogpi[:, :-1].T[..., None, None])
            logtissue = torch.logsumexp(logfg, 0)
            logZ = torch.logaddexp(logtissue, logbg)
            # From the tissue terms directly (not as 1 − p(bg)), so that
            # small foreground probabilities keep their precision.
            if per_class or out is None:
                z = logfg.sub_(logZ).exp_()
                for k, target in enumerate(out or ()):
                    chunk.store(target, z[k])
                if mean:
                    chunk.store(out_mean, z.mul_(EU).sum(0))
            else:
                chunk.store(out, logtissue.sub_(logZ).exp_())
                if mean:
                    z = logfg.sub_(logZ).exp_()
                    chunk.store(out_mean, z.mul_(EU).sum(0))
        return alpha


def fit_modulation(
    total: Tensor,
    count: Tensor,
    By: Tensor,
    Bx: Tensor,
    penalty: Tensor,
    info: float = 1.0,
    min_count: float = 3.0,
    robust_iter: int = 2,
    outlier: float = 4.0,
    extra: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Flat-field moment estimate of the log-modulation field.

    If the tissue signal has the same distribution in every pixel (on
    average over frames, given that the pixel contains tissue), the mean
    tissue signal of pixel i is mᵢ·E[u]. Its log is fitted by weighted,
    penalised least squares in the DCT basis,

        min ½ ∑ᵢ wᵢ (log m̂ᵢ − (Φ·θ)ᵢ)² + ½ θᵀ·diag(penalty)·θ,

    where m̂ᵢ = totalᵢ / countᵢ and wᵢ = info · countᵢ is the information
    about log mᵢ (info ≈ mean² / variance of one tissue value). Pixels
    with fewer than `min_count` tissue values are ignored. The DC
    coefficient is then dropped (geometric mean of m = 1). Pixels whose
    residual exceeds `outlier` robust standard deviations (hot pixels)
    are discarded and the fit repeated `robust_iter` times.

    `extra` = (H, g) adds a quadratic ½ θᵀHθ − gᵀθ to the objective (the
    overlap term of tiled data, whose score at θ = 0 is g); the moment
    term may then be empty.

    Parameters
    ----------
    total, count : (H, W) Tensor
        Sum and number of the tissue values of each pixel.
    By, Bx : (H, ky), (W, kx) Tensor
        DCT bases.
    penalty : (ky, kx) Tensor
        Prior precision of θ.

    Returns
    -------
    (ky, kx) Tensor
        θ.
    """
    ky, kx = By.shape[-1], Bx.shape[-1]
    ok = (count >= min_count) & (total > 0)
    if not ok.any() and extra is None:
        return torch.zeros((ky, kx), dtype=total.dtype, device=total.device)
    tiny = torch.finfo(total.dtype).tiny
    logm = torch.where(
        ok, total.clamp_min(tiny).log() - count.clamp_min(1).log(), 0
    )
    w = torch.where(ok, count * info, 0)
    bases = [By, Bx]
    for it in range(robust_iter + 1):
        H = gram(w, bases)
        H.diagonal().add_(penalty.flatten())
        g = project(w * logm, bases).flatten()
        if extra is not None:
            H += extra[0]
            g += extra[1]
        H.diagonal().add_(H.diagonal().mean(), alpha=1e-8)
        theta = torch.linalg.solve(H, g).reshape(ky, kx)
        if it == robust_iter:
            break
        if not ok.any():
            break
        res = (logm - reconstruct(theta, bases)).abs_()
        scale = 1.4826 * res[w > 0].median()
        w = torch.where(res > outlier * scale, 0, w)
    theta[0, 0] = 0
    return theta


def _valid_variance(variance: Tensor, floor: float = 0.0) -> Tensor:
    """A given variance map, made positive and finite.

    Values that are not positive and finite are replaced by the median of
    the others, and every value is clamped to at least 1/16 of it (the
    clamp of the data-driven initialisation) and to `floor`.
    """
    ok = variance.isfinite() & (variance > 0)
    if not ok.any():
        raise ValueError("The initial variance has no positive, finite value")
    typical = variance[ok].median()
    low = max(typical.item() / 16, floor)
    valid = torch.where(ok, variance, typical).clamp_min_(low)
    changed = int((valid != variance).sum())
    if changed:
        warnings.warn(
            f"{changed} values of the initial variance were not positive "
            f"and finite, or below 1/16 of its median: they were replaced "
            f"or clamped",
            RuntimeWarning,
            stacklevel=4,
        )
    return valid


# -- quantisation of the data ----------------------------------------------
def _storage(source: ArrayLike) -> tuple[bool, float]:
    """Whether `source` stores integers, and the epsilon of its floats."""
    dtype = getattr(source, "dtype", None)
    if isinstance(dtype, torch.dtype):
        if dtype.is_floating_point or dtype.is_complex:
            return False, torch.finfo(dtype).eps
        return True, 0.0
    try:
        dtype = np.dtype(dtype) if dtype is not None else None
    except TypeError:
        dtype = None
    if dtype is not None and dtype.kind in "iub":
        return True, 0.0
    if dtype is not None and dtype.kind == "f":
        return False, float(np.finfo(dtype).eps)
    return False, float(np.finfo(np.float32).eps)


def _sample(
    source: ArrayLike,
    layout: Layout,
    samples: int,
    batches: int,
    seed: int,
) -> Tensor:
    """Finite values drawn at random from a few random frame batches."""
    chunks = frame_batches(
        source,
        layout=layout,
        overhead=_OVERHEAD,  # batches no larger than those of the fit
        dtype=torch.float64,
        device="cpu",
    )
    gen = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(chunks), generator=gen)[:batches]
    per = -(-samples // max(len(pick), 1))
    parts = [torch.zeros(0, dtype=torch.float64)]
    for chunk in chunks.select(pick.sort().values.tolist()):
        x = chunk.data.reshape(-1)
        if x.numel() > per:
            x = x[torch.randint(x.numel(), (per,), generator=gen)]
        parts.append(x)
    x = torch.cat(parts)
    return x[x.isfinite()]


def detect_quantization(
    source: ArrayLike,
    layout: Layout,
    *,
    samples: int = 100_000,
    batches: int = 4,
    seed: int = 0,
    fraction: float = 0.99,
) -> float:
    """Width Δ of the quantisation bins of the data (0: continuous).

    Integer data give Δ = 1. Otherwise about `samples` values are drawn
    at random (seeded) from `batches` frame batches, and Δ is the
    smallest positive gap between their sorted distinct values, refined
    by least squares. Δ is accepted only if at least `fraction` of the
    values are integer multiples of Δ from the smallest one, to within
    1 % of Δ plus the rounding error of the stored floats; otherwise
    (and when Δ is below 10⁻⁶ of the largest value, or not resolved by
    the float precision) the data are treated as continuous.
    """
    integer, eps = _storage(source)
    if integer:
        return 1.0
    x = _sample(source, layout, samples, batches, seed)
    if x.numel() < 2:
        return 0.0
    y = x - x.min()
    levels = torch.unique(y)  # sorted
    if levels.numel() < 2:
        return 0.0
    delta = levels.diff().min().item()
    scale = x.abs().max().item()
    err = 4 * eps * scale  # rounding error of one value
    # A Δ that the floats do not resolve, or finer than 10⁻⁶ of the
    # values (whose δ = Δ²/12 is negligible), is not a quantisation
    if delta <= max(16 * err, 1e-6 * scale):
        return 0.0
    # Least squares on the levels whose multiple of Δ is unambiguous,
    # over a range that grows as Δ gets more accurate
    reach = delta / (8 * err) if err > 0 else math.inf
    for _ in range(4):
        k = (levels / delta).round()
        near = ((levels / delta - k).abs() <= 0.25) & (k > 0) & (k <= reach)
        if not near.any():
            break
        kk = k[near]
        delta = (kk * levels[near]).sum().item() / kk.square().sum().item()
        reach *= kk.max().item()
    r = y / delta
    tol = 0.01 + err / delta
    ok = ((r - r.round()).abs() <= tol).double().mean().item()
    return delta if ok >= fraction else 0.0


def is_integer_valued(
    source: ArrayLike,
    layout: Layout,
    *,
    samples: int = 100_000,
    batches: int = 4,
    seed: int = 0,
    fraction: float = 0.99,
) -> bool:
    """Whether (nearly) all the values of the data are integers.

    Integer data are; otherwise the values sampled as in
    `detect_quantization` are tested, to within 0.01 plus the rounding
    error of the stored floats.
    """
    integer, eps = _storage(source)
    if integer:
        return True
    x = _sample(source, layout, samples, batches, seed)
    if not x.numel():
        return True
    tol = x.abs().mul_(4 * eps).add_(0.01)
    ok = ((x - x.round()).abs() <= tol).double().mean().item()
    return ok >= fraction
