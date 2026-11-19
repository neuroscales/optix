"""Variational fit of a mixture model on one group of pictures.

The estimator ([`GammaMixture`][optix.shading.mix.GammaMixture]) runs one
`MixtureFit` per group. This module holds the inference loop, the
background class and the sensor / shading updates; everything that
depends on the distribution of the tissue signal (its prior, the
integral over u and the update of its parameters) is delegated to a
`Component` (see `_gamma.GammaComponent`).

Per-pixel statistics accumulated by the local step (all posterior
expectations, summed over frames), with

    r  = y − c − m·u                 (residual)
    σ² = v + γ·m·ũ                   (variance; ũ = u, or 0)
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

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

__all__ = ["Component", "MixtureFit", "fit_modulation"]

HCC, GC, XU, G1, AU, HVV, GV, WU, G2, BU = range(10)
NSTATS = 10

_DIRICHLET_BOUNDS = (1e-3, 1e3)
_DIRICHLET_ITER = 32
_MAX_SAMPLES = 1_000_000
# Chunk-sized temporaries of the local step (two (10, n, H, W) stacks of
# statistics plus the quadrature temporaries).
_OVERHEAD = 40.0


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

        `weight` is `(K,)`, `stats` `(K, nstats)` (float64); classes
        without weight keep `previous`.
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
    """Prior over the class model, and the regularisers of c, v, θ, γ."""

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
    """Point estimates: offset c, variance v, log-modulation θ, gain γ."""

    C: Tensor  # (H, W)
    V: Tensor  # (H, W)
    M: Tensor  # (H, W)
    theta: Tensor  # (ky, kx), theta[0, 0] = 0
    gain: Tensor  # ()


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
    ) -> None:
        self.source = source
        self.layout = layout
        _, sample, spatial = layout.shape_of(source)
        self.sample_shape = tuple(sample)
        self.spatial_shape = tuple(spatial)
        self.backend = backend
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

    # -- helpers -----------------------------------------------------------
    def modulation(self) -> Tensor:
        """M = exp(B·θ) on the pixel grid."""
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
            overhead=_OVERHEAD,
            contiguous=True,
            **self.backend,
        )

    def _elogpi(self, alpha: Tensor) -> Tensor:
        return Dirichlet(alpha).mean_log

    # -- initialisation ----------------------------------------------------
    def init(
        self,
        bg_fraction: float = 0.5,
        threshold: float = 3.0,
        modulation: bool = True,
        *,
        offset: Tensor | None = None,
        variance: Tensor | None = None,
        flat: Tensor | None = None,
        classes: tuple[Tensor, Tensor, Tensor | None] | None = None,
    ) -> None:
        """Data-driven starting point; given fields replace their estimate.

        `classes` is `(params, frac, alpha)`: parameters and fractions of
        the tissue classes, and (optionally) the prior concentrations.
        """
        prm, prior, post = self.param, self.prior, self.post
        backend = self.backend
        height, width = self.spatial_shape
        nframes, nclass = math.prod(self.sample_shape), self.nclass

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
            sd = sd.clamp(0.25 * sd_typical, 4 * sd_typical)
            V = sd.square()
        if offset is not None:
            C = offset.to(**backend).expand(height, width).clone()
        if variance is not None:
            V = variance.to(**backend).expand(height, width).clone()
            sd = V.sqrt()
            sd_typical = sd.median()

        theta = torch.zeros_like(prm.theta)
        M = None
        if flat is not None:
            logm = flat.to(**backend).expand(height, width).log()
            theta = project(logm, [self.by, self.bx])
            theta[0, 0] = 0
            prm.theta = theta
            M = self.modulation()

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
        moments = torch.zeros(3, dtype=torch.float64, device=C.device)
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
                theta = fit_modulation(
                    fg_excess,
                    fg_count,
                    self.by,
                    self.bx,
                    prior.lam * self.bending,
                    info=info,
                )
            prm.theta = theta
        prm.M = self.modulation()

        n_tissue = n_tissue.reshape(-1)
        alpha = None
        component = self.component
        if classes is not None:
            params, frac, alpha = (
                None if t is None else t.to(**backend) for t in classes
            )
            params = component.clamp(params)
        elif values.numel() >= 2 * nclass:
            # The classes use the raw tissue values (M is close to 1, and
            # the k-means is sensitive to small perturbations).
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
        """Median over pixels of the per-pixel median of |y − C|."""
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

            ∑ᵢₖ log ∑_z exp(E[log πₖ_z]) · p(yᵢₖ | z)
            − ∑ₖ KL(q(πₖ) ‖ p(πₖ)) + log p(θ) + log p(v) + log p(γ)
        """
        prm, prior, post = self.param, self.prior, self.post
        height, width = self.spatial_shape
        nk = self.nclass
        if prior.v is None:  # `init` was skipped
            prior.v = prm.V.median()
        elogpi = self._elogpi(post.alpha)

        self._stats, self._counts, self._class_stats = None, None, None
        f64 = dict(dtype=torch.float64, device=prm.C.device)
        stats = torch.zeros((NSTATS, height, width), **f64)
        counts = prm.C.new_zeros((*self.sample_shape, nk + 1))
        class_stats = torch.zeros((nk, self.component.nstats), **f64)
        bound = torch.zeros((), **f64)
        for chunk in self.frames():
            ep = elogpi[chunk.samples].reshape(-1, nk + 1)
            st, bnd, cnt, cs = self._latent_chunk(chunk.data, ep)
            stats += st
            counts[chunk.samples] = cnt.reshape(*chunk.sample_shape, nk + 1)
            class_stats += cs
            bound += bnd
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
        self, Y: Tensor, elogpi: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Local step on `(n, H, W)` frames with `(n, K+1)` E[log π]."""
        C, V = self.param.C, self.param.V
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
        logjoint = [None] * (nk + 1)
        logjoint[nk] = logZ

        # Tissue classes, folded in with a running log-sum-exp so that the
        # per-class statistics never need to be stored.
        suffs = []
        for k in range(nk):
            logZ0 = logZ
            logpY, suff, st = self._loglik(Y, k)
            logZY = elogpi[:, k, None, None] + logpY  # log p(y, z=k)
            logZ = torch.logaddexp(logZ, logZY)  # log ∑ p(y, z)
            pz_prv = (logZ0 - logZ).exp_()
            pz_new = (logZY - logZ).exp_()
            acc.mul_(pz_prv).add_(st.mul_(pz_new))
            logjoint[k] = logZY
            suffs.append(suff)

        f64 = torch.float64
        nstats = self.component.nstats
        sum_Z = Y.new_zeros((len(Y), nk + 1))
        class_stats = torch.zeros((nk, nstats), dtype=f64, device=Y.device)
        for k in range(nk + 1):
            Zk = (logjoint[k] - logZ).exp_()
            sum_Z[:, k] = Zk.sum((-2, -1))
            if k < nk:
                for j, t in enumerate(suffs[k]):
                    class_stats[k, j] = (Zk * t).sum(dtype=f64)

        stats = acc.sum(-3, dtype=f64)  # sum over frames
        bound = logZ.sum(dtype=f64)
        return stats, bound, sum_Z, class_stats

    def _loglik(
        self, Y: Tensor, k: int
    ) -> tuple[Tensor, tuple[Tensor, ...], Tensor]:
        prm = self.param
        return self.component.loglik(
            Y,
            prm.C,
            prm.V,
            prm.M,
            prm.gain,
            self.prior.params[k],
            dq=self.dq,
            full=self.full,
        )

    # -- parameter step: c, v, θ, γ ----------------------------------------
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
        pixels in (c, v), and couples them to (θ, log γ) only through
        per-pixel terms. c and v are eliminated (Schur complement), which
        leaves a dense system of the size of the basis for θ and a scalar
        equation for log γ; they are then back-substituted.
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
        Fvm = gM * Hvm  # Fisher(v, log m) = Fisher(v, log γ)
        Fmm = M.square().mul_(Hmm)  # Fisher(log m, log m), mean part
        Vmm = gM.square().mul_(Hmg)  # Fisher(log m, log m), variance part
        Sm = M.mul(Gm).addcmul(gM, Gmg)  # score of log m
        Sg = gM * Gmg  # score of log γ (per pixel)
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
            dtheta = torch.linalg.solve(H, g).reshape(prm.theta.shape)
            dlogm = reconstruct(dtheta, bases)
            largest = max(dlogm.abs().max().item(), 1e-12)
            shrink = min(1.0, max_log_step / largest)
            dtheta *= shrink
            dlogm *= shrink

        # Gain: scalar step on log γ, with v eliminated
        dloggain = 0.0
        if gain:
            deviation = (gam / self.gain0).log().item()
            score = Sg.sum().item() - deviation / prior.gain_sd**2
            curvature = Vmm.sum().item() + 1 / prior.gain_sd**2
            dloggain = max(-max_log_step, min(max_log_step, score / curvature))

        # Back-substitution
        if c:
            dC = Gc.addcmul(Fcm, dlogm, value=-1).mul_(iHcc)
            prm.C.add_(dC, alpha=lr)
        if v:
            dV = (dlogm + dloggain).mul_(Fvm).sub_(Gv).neg_().mul_(iHvv)
            prm.V = torch.maximum(prm.V.add(dV, alpha=lr), prm.V.mul(0.1))
        if theta:
            prm.theta.add_(dtheta, alpha=lr)
            prm.M = self.modulation()
        if gain:
            prm.gain = gam * math.exp(lr * dloggain)

    # -- mixture step: q(π), then μ, ν, α ----------------------------------
    def update_post(self) -> None:
        """q(πₖ) = Dirichlet(α + ∑ᵢ zᵢₖ); class fit of the aggregated q(u)."""
        if self._counts is None:
            raise RuntimeError("Run update_latent() first")
        prior, post = self.prior, self.post
        post.alpha = (prior.alpha + self._counts).reshape(post.alpha.shape)
        weight = self._counts[..., :-1].reshape(-1, self.nclass).sum(0)
        post.params = self.component.fit(
            weight.to(torch.float64), self._class_stats, post.params
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
                and bound < last - slack
                and halvings < max_halvings
            ):
                halvings += 1
                _halve_(prm.C, previous[0])
                _halve_(prm.V, previous[1])
                _halve_(prm.theta, previous[2])
                _loghalve_(prm.gain, previous[3])
                prm.M = self.modulation()
                self.history.pop()
                bound = self.update_latent()
            self.halvings.append(halvings)
            if callback is not None:
                callback(n, bound)
            if abs(bound - last) < slack:
                # q(z, u) is already consistent with the parameters
                converged = True
                reason = f"bound change below tol={tol:g} per data point"
                break
            last = bound

            previous = [t.clone() for t in (prm.C, prm.V, prm.theta, prm.gain)]
            self.update_post()
            self.update_prior(alpha=learn_alpha)
            self.update_parameters(
                theta=learn_modulation, lr=lr, max_log_step=max_log_step
            )
        else:
            # Make the statistics and the bound consistent with the final
            # parameters (the mixture was updated after the last local
            # step).
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
        out: WritableArrayLike | Sequence[WritableArrayLike],
        out_mean: WritableArrayLike | None,
        per_class: bool,
        niter: int,
        tol: float = 0.0,
    ) -> Tensor:
        """Class probabilities (and E[u | y]) of every pixel and frame.

        `alpha` is `(*sample, K+1)`: q(π) of each frame (updated in place
        by `niter` q(z) / q(π) iterations with fixed likelihoods). `out` is
        one output (foreground probability) or one per tissue class.
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
            if per_class:
                z = logfg.sub_(logZ).exp_()
                for k, target in enumerate(out):
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
) -> Tensor:
    """Flat-field moment estimate of the log-modulation field.

    If the tissue signal has the same distribution in every pixel (on
    average over frames, given that the pixel contains tissue), the mean
    tissue signal of pixel i is mᵢ·E[u]. Its log is fitted by weighted,
    penalised least squares in the DCT basis,

        min ½ ∑ᵢ wᵢ (log m̂ᵢ − (B·θ)ᵢ)² + ½ θᵀ·diag(penalty)·θ,

    where m̂ᵢ = totalᵢ / countᵢ and wᵢ = info · countᵢ is the information
    about log mᵢ (info ≈ mean² / variance of one tissue value). Pixels
    with fewer than `min_count` tissue values are ignored. The DC
    coefficient is then dropped (geometric mean of m = 1). Pixels whose
    residual exceeds `outlier` robust standard deviations (hot pixels)
    are discarded and the fit repeated `robust_iter` times.

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
    if not ok.any():
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
        H.diagonal().add_(H.diagonal().mean(), alpha=1e-8)
        g = project(w * logm, bases).flatten()
        theta = torch.linalg.solve(H, g).reshape(ky, kx)
        if it == robust_iter:
            break
        res = (logm - reconstruct(theta, bases)).abs_()
        scale = 1.4826 * res[w > 0].median()
        w = torch.where(res > outlier * scale, 0, w)
    theta[0, 0] = 0
    return theta
