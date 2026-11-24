"""Multiview Richardson–Lucy deconvolution (Preibisch et al. 2014)."""

import copy
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import Tensor

from .._estimator import Estimator, History, Tags
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
from ._fft import PaddedConvolution, next_fast_len
from ._rl import RichardsonLucy, _expected, rl_block
from ._wb import BackProjection, _pad_centred

__all__ = [
    "Multiview",
    "MultiviewRL",
    "compound_kernels",
    "multiview_rl",
]

KERNELS = ("bayesian", "efficient", "optimization-1", "optimization-2")
UPDATES = ("sequential", "multiplicative", "additive")
# working buffers per loaded voxel: per view the data, weight, blurred
# estimate and ratio; shared FFT grids, estimate and correction
_OVERHEAD_BASE = 8.0
_OVERHEAD_VIEW = 5.0


def _mirror(kernel: Tensor) -> Tensor:
    """P* (ξ) = P(−ξ) of a kernel with odd sizes, centred."""
    return kernel.flip(tuple(range(kernel.ndim)))


def _convolve(*kernels: Tensor) -> Tensor:
    """Linear convolution of centred kernels, cropped to the first's shape.

    The kernels have odd sizes and the same shape K; the full
    convolution (of shape m(K − 1) + 1) is computed by FFT and its
    central K block is returned, clamped at 0 (FFT round-off).
    """
    shape = kernels[0].shape
    full = tuple(len(kernels) * (k - 1) + 1 for k in shape)
    grid = tuple(next_fast_len(n) for n in full)
    dims = tuple(range(len(shape)))
    spec = None
    for k in kernels:
        s = torch.fft.rfftn(k, s=grid, dim=dims)
        spec = s if spec is None else spec * s
    out = torch.fft.irfftn(spec, s=grid, dim=dims)
    index = tuple(
        slice((n - k) // 2, (n - k) // 2 + k)
        for n, k in zip(full, shape, strict=True)
    )
    return out[index].clamp(min=0)


def _odd_shape(psfs: Sequence[Tensor]) -> tuple[int, ...]:
    """Smallest odd shape holding every PSF (centred at `K // 2`)."""
    ndim = psfs[0].ndim
    if any(p.ndim != ndim for p in psfs):
        raise ValueError("All PSFs must have the same number of dims")
    shape = []
    for d in range(ndim):
        k = max(int(p.shape[d]) for p in psfs)
        shape.append(k if k % 2 else k + 1)
    return tuple(shape)


def compound_kernels(
    psfs: Sequence[Tensor], kernel: str = "efficient"
) -> list[Tensor]:
    """Back-projection kernels of the multiview updates (compound kernels).

    For views v ∈ V with PSFs Pᵥ (N = |V|) and Wᵥ = V ∖ {v} the other
    views, the kernel bᵥ of the back projection Bᵥ r = r ∗ bᵥ is

    - `"bayesian"` (Eq. 37): bᵥ = Pᵥ*, the mirrored PSF (Bᵥ = Hᵥᵀ);
    - `"efficient"` (Eqs. 102–103): bᵥ = Pᵥ* · ∏_{w ∈ Wᵥ} Pᵥ* ∗ P_w ∗ P_w*;
    - `"optimization-1"` (Eq. 114): bᵥ = Pᵥ* · ∏_{w ∈ Wᵥ} Pᵥ* ∗ P_w;
    - `"optimization-2"` (Eq. 115): bᵥ = ∏_{w ∈ V} Pᵥ* = (Pᵥ*)ᴺ,

    with ∗ the convolution, · and ∏ pixel-wise products, and P*(ξ) =
    P(−ξ) (equation numbers of the Supplementary Notes of
    [Preibisch et al. 2014](https://doi.org/10.1038/nmeth.2929)).
    Each kernel is normalised to unit sum. With one view all four are
    Pᵥ*.

    !!! note
        The pixel-wise product with Pᵥ* restricts every compound kernel
        to the support of Pᵥ, so the kernels have the shape of the PSFs.
        The convolutions are evaluated in float64 by FFT, clamped at 0
        (round-off), and every factor of a product is scaled to a unit
        maximum before multiplying, so that the product of N small
        values does not underflow; the final normalisation makes the
        scaling irrelevant.

        In [`MultiviewRL`][optix.deconv.MultiviewRL] the compound kernels
        are used inside the field of view only: on the margin of the
        object domain (pixels outside the views), the back projection is
        Hᵥᵀ for every variant (see
        [`Multiview`][optix.deconv.utils.Multiview]).

    Parameters
    ----------
    psfs : sequence of Tensor
        PSF Pᵥ of each view, `(*K)` with odd sizes, all of the same
        shape, in the coordinates of the common grid, non-negative.
    kernel : {"bayesian", "efficient", "optimization-1", "optimization-2"}
        Variant.

    Returns
    -------
    list of Tensor
        Kernel bᵥ of each view, `(*K)`, float64, unit sum.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    shape = tuple(psfs[0].shape)
    if any(tuple(p.shape) != shape for p in psfs):
        raise ValueError("All PSFs must have the same shape")
    if any(k % 2 == 0 for k in shape):
        raise ValueError(f"PSF sizes must be odd, got {shape}")
    psfs = [p.to(torch.float64) for p in psfs]
    psfs = [p / p.sum() for p in psfs]
    n = len(psfs)
    out = []
    for v, p in enumerate(psfs):
        pv = _mirror(p)
        if kernel == "bayesian" or n == 1:
            b = pv
        elif kernel == "optimization-2":
            b = (pv / pv.max()).pow(n)
        else:
            b = pv / pv.max()
            for w, q in enumerate(psfs):
                if w == v:
                    continue
                parts = (
                    (pv, q)
                    if kernel == "optimization-1"
                    else (
                        pv,
                        q,
                        _mirror(q),
                    )
                )
                virtual = _convolve(*parts)
                b = b * (virtual / virtual.max())
        out.append(b / b.sum())
    return out


class Multiview:
    """Blur and back projection of several registered views of one block.

    The views y₁, …, y_N of the block (shape L each) are images of one
    object f on the padded grid N of their blur operators Hᵥ (one
    [`PaddedConvolution`][optix.deconv.utils.PaddedConvolution] per view,
    all on the same grid). As an operator on f this is the stacked blur
    H = [H₁; …; H_N], f ↦ (Hᵥ f)ᵥ, of shape `(N, *L)`, with Hᵀ r =
    ∑ᵥ Hᵥᵀ rᵥ: it gives [`rl_block`][optix.deconv.utils.rl_block] the
    blurred estimates for its χ² test and the support of the object. The
    multiview update itself is [`step`][optix.deconv.utils.Multiview.step],
    passed to `rl_block` as its `step` hook.

    The correction of view v, from its back projector Bᵥ (Hᵥᵀ, or the
    convolution with a compound kernel, see
    [`compound_kernels`][optix.deconv.utils.compound_kernels]), is

        cᵥ = Bᵥ[yᵥ / (Hᵥ f + b)] / Bᵥ1,

    set to 1 on object pixels that view v does not reach (no update).

    **Margin.** The object domain is the block padded by the PSF support;
    its margin (object pixels outside the block) is observed by no view
    directly, only through the tails of the PSFs. There the back
    projector of every view is Hᵥᵀ, whatever the `kernel`
    ([`BackProjection`][optix.deconv.utils.BackProjection] uses the
    compound kernel inside the block only; the compound kernels are
    narrower and would leave part of the margin without an update), so
    Bᵥ1 = Hᵥᵀ1 there. Not specified in the paper.

    **Edge damping.** On the margin the sequential and multiplicative
    updates apply view v's correction with the weight

        ρᵥ = Hᵥᵀ1 / maxᵤ Hᵤᵀ1 ∈ [0, 1],

    its reach relative to the view that sees the pixel best, as
    f ← f · (1 − ρᵥ + ρᵥ cᵥ). Ordered-subset updates assume subsets of
    equal sensitivity Hᵥᵀ1; away from the image border every view has
    sensitivity 1, but on the margin the reaches of the views differ by
    orders of magnitude, and a view that reaches a pixel only through
    the far tail of its PSF would otherwise apply a full correction
    drawn from almost no data (in the extreme, a ratio of two
    round-off-sized numbers). Scaling each view's step by ρᵥ ≤ 1
    restores the balance between the views. It changes neither the
    fixed points (1 − ρᵥ + ρᵥ cᵥ = 1 exactly where cᵥ = 1) nor
    positivity. Without it the noise-free data misfit of the sequential
    update is not monotone and stalls an order of magnitude higher, in
    double precision too, and float32 and float64 results differ
    visibly near the image border. Not specified in the paper.

    Inside the block ρᵥ = 1, so the update rule there is unchanged; the
    result next to the image border still differs from the undamped
    one, since the margin feeds the border pixels through the blur.
    With one view there is no damping (ρ = 1 everywhere) and the
    iteration is exactly single-view Richardson–Lucy.

    **Weights** (sequential update). Given weights wᵥ on the block are
    used inside it; on the margin they are carried over as

        wᵥ = ρᵥ · clamp(Bᵥ[wᵥ] / Bᵥ1, 0, 1),

    the weights of the block pixels that view v back-projects onto each
    margin pixel, so that a view with weight 0 at an image border does
    not update the margin beyond it (zero is absorbing for a
    multiplicative update, so a correction of 0 there could not be
    undone by the other views). Not specified in the paper. The weights
    only gate the update: within a PSF half-width of the edge of a
    view's coverage, its pixels outside the coverage still enter cᵥ as
    observed values (zeros if the view has no data there), so the
    fall-off of the weights should be at least as wide as the PSF.

    Attributes
    ----------
    ops : list of PaddedConvolution
        Blur operator Hᵥ of each view.
    backs : list of BackProjection or None
        Back projector Bᵥ of each view (None: Hᵥᵀ).
    update : str
        `"sequential"`, `"multiplicative"` or `"additive"`.
    weights : Tensor or None
        Weights wᵥ ∈ [0, 1] of the views on the object grid, `(N,
        *grid)`: the given weights inside the block, and outside it ρᵥ,
        or ρᵥ · clamp(Bᵥ[wᵥ] / Bᵥ1, 0, 1) with given weights (None with
        one view and no weights).
    shape : tuple of int
        `(N, *L)`.
    grid : tuple of int
        Object grid N.
    norm : Tensor
        Hᵀ1 = ∑ᵥ Hᵥᵀ1 on the object grid.
    """

    def __init__(
        self,
        ops: Sequence[PaddedConvolution],
        kernels: Sequence[Tensor | None] | None = None,
        *,
        update: str = "sequential",
        weights: Tensor | None = None,
    ) -> None:
        """Set up the per-view operators of a block.

        Parameters
        ----------
        ops : sequence of PaddedConvolution
            Blur operator Hᵥ of each view, all on the same block shape
            and grid (PSFs of the same shape).
        kernels : sequence of Tensor or None, optional
            Back-projection kernel bᵥ of each view (no larger than the
            PSFs of `ops`), or None for Hᵥᵀ. Default: Hᵥᵀ for every
            view.
        update : {"sequential", "multiplicative", "additive"}
            How the corrections of the views are combined (see
            [`step`][optix.deconv.utils.Multiview.step]).
        weights : Tensor, optional
            Weights wᵥ ∈ [0, 1] of each view on the block, `(N, *L)`
            (sequential update only). Object pixels outside the block
            get wᵥ = ρᵥ · clamp(Bᵥ[wᵥ] / Bᵥ1, 0, 1). Default: wᵥ = 1
            inside the block, ρᵥ outside it.
        """
        ops = list(ops)
        if not ops:
            raise ValueError("Multiview needs at least one view")
        if update not in UPDATES:
            raise ValueError(f"update must be one of {UPDATES}")
        first = ops[0]
        if any(o.shape != first.shape or o.grid != first.grid for o in ops):
            raise ValueError("All views must share the block shape and grid")
        if kernels is None:
            kernels = [None] * len(ops)
        if len(kernels) != len(ops):
            raise ValueError("One kernel per view is needed")
        self.ops = ops
        self.update = update
        self.ndim = first.ndim
        self.grid = first.grid
        self.shape = (len(ops), *first.shape)
        self.backs = [
            None if k is None else BackProjection(k, o)
            for k, o in zip(kernels, ops, strict=True)
        ]
        self.norm = sum(o.norm for o in ops)
        self._norms: list[Tensor] = []  # Bᵥ1, 0 where view v does not reach
        self._inv: list[Tensor] = []
        self._valid: list[Tensor] = []
        for o, b in zip(ops, self.backs, strict=True):
            bn = o.norm if b is None else b(o.norm.new_ones(o.shape))
            valid = bn > bn.max() * torch.finfo(bn.dtype).eps * 16
            tiny = torch.finfo(bn.dtype).tiny
            self._valid.append(valid)
            self._norms.append(torch.where(valid, bn, 0))
            self._inv.append(torch.where(valid, 1 / bn.clamp(min=tiny), 0))
        self._inside = (slice(None),) + tuple(slice(0, n) for n in first.shape)
        self._reach: Tensor | None = None
        if len(ops) > 1:
            norms = torch.stack(self._norms)
            top = norms.max(0).values
            tiny = torch.finfo(top.dtype).tiny
            reach = torch.where(top > 0, norms / top.clamp(min=tiny), 0)
            reach[self._inside] = 1
            self._reach = reach
        self.weights: Tensor | None = self._reach
        if weights is not None:
            self.weights = self._grid_weights(weights)

    def _grid_weights(self, weights: Tensor) -> Tensor:
        if self.update != "sequential":
            raise ValueError("weights apply to the sequential update only")
        if tuple(weights.shape) != self.shape:
            raise ValueError(
                f"weights must have shape {self.shape}, got "
                f"{tuple(weights.shape)}"
            )
        weights = weights.to(self.norm).clamp(0, 1)
        out = []
        for v in range(self.shape[0]):
            # margin: ρᵥ · clamp(Bᵥ[wᵥ] / Bᵥ1, 0, 1), the weights of the
            # block pixels that view v back-projects onto each margin pixel
            w = (self._back(v, weights[v]) * self._inv[v]).clamp_(0, 1)
            if self._reach is not None:
                w = w * self._reach[v]
            out.append(w)
        out = torch.stack(out)
        out[self._inside] = weights
        return out

    def weighted(self, weights: Tensor | None) -> "Multiview":
        """A copy with other per-view weights (operators shared)."""
        out = copy.copy(self)
        out.weights = (
            self._reach if weights is None else self._grid_weights(weights)
        )
        return out

    # -- stacked operator ------------------------------------------------
    def forward(self, f: Tensor) -> Tensor:
        """(Hᵥ f)ᵥ: object `(*N)` to views `(N_views, *L)`."""
        return torch.stack([o.forward(f) for o in self.ops])

    def adjoint(self, r: Tensor) -> Tensor:
        """∑ᵥ Hᵥᵀ rᵥ: views `(N_views, *L)` to object `(*N)`."""
        return sum(o.adjoint(r[v]) for v, o in enumerate(self.ops))

    def pad(self, x: Tensor) -> Tensor:
        """Zero-pad a block `(..., *L)` to the grid `(..., *N)`."""
        return self.ops[0].pad(x)

    def crop(self, x: Tensor) -> Tensor:
        """Restrict an object `(..., *N)` to the block `(..., *L)`."""
        return self.ops[0].crop(x)

    # -- updates ---------------------------------------------------------
    def _back(self, v: int, r: Tensor) -> Tensor:
        b = self.backs[v]
        return self.ops[v].adjoint(r) if b is None else b(r)

    def correction(
        self,
        v: int,
        f: Tensor,
        data: Tensor,
        *,
        background: Tensor | float = 0.0,
        mu: Tensor | None = None,
    ) -> Tensor:
        """Correction cᵥ = Bᵥ[yᵥ / (Hᵥ f + b)] / Bᵥ1 of view v.

        Parameters
        ----------
        v : int
            View index.
        f : Tensor
            Current estimate on the object grid.
        data : Tensor
            View yᵥ on the block, `(*L)`.
        background : Tensor or float
            Background b of the view.
        mu : Tensor, optional
            Hᵥ f + b if already computed.

        Returns
        -------
        Tensor
            cᵥ on the object grid, 1 where view v does not reach.
        """
        if mu is None:
            mu = _expected(f, self.ops[v], background)
        c = self._back(v, data / mu) * self._inv[v]
        return torch.where(self._valid[v], c, 1)

    def step(
        self,
        f: Tensor,
        data: Tensor,
        op: object = None,
        *,
        inv_norm: Tensor | None = None,
        background: Tensor | float = 0.0,
        damping: float = 1.0,
        mu: Tensor | None = None,
        flat: Tensor | None = None,
        back: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        """One multiview iteration: every view contributes once.

        With cᵥ the correction of view v and ε = `damping`,

        - `"sequential"` (Eqs. 104–107 and 109–112, Eq. 3 of the
          Online Methods with weights): for v = 1, …, N in turn,
          f ← f · (1 − εwᵥ + εwᵥ cᵥ), cᵥ evaluated at the current f;
        - `"multiplicative"` (Eqs. 36–37, with the |V|-th root of
          Eqs. 46–47): f ← f · (1 − ε + ε (∏ᵥ c̃ᵥ)^{1/N}), with
          c̃ᵥ = 1 − wᵥ + wᵥ cᵥ (wᵥ = 1 inside the block);
        - `"additive"` (maximum-likelihood EM of the stacked views,
          [Shepp & Vardi 1982](https://doi.org/10.1109/TMI.1982.4307558)):
          f ← f · (1 − ε + ε ∑ᵥ Bᵥ[yᵥ / Hᵥ f] / ∑ᵥ Bᵥ1),

        and negative values (from FFT round-off) are set to 0. With one
        view, all three are [`rl_step`][optix.deconv.utils.rl_step].

        Parameters
        ----------
        f : Tensor
            Current estimate on the object grid.
        data : Tensor
            Views on the block, `(N, *L)`.
        op : object, optional
            Ignored (the hook passes this `Multiview`).
        inv_norm : Tensor, optional
            Ignored: each view has its own normaliser Bᵥ1.
        background : Tensor or float
            Background b, broadcast to the views `(N, *L)`.
        damping : float
            Fraction ε ∈ (0, 1] of the correction applied.
        mu : Tensor, optional
            (Hᵥ f + b)ᵥ at `f` if already computed.
        flat, back
            Not supported (must be None).

        Returns
        -------
        Tensor
            Updated estimate, non-negative.
        """
        if flat is not None or back is not None:
            raise ValueError("Multiview.step supports neither flat nor back")
        n = len(self.ops)
        bg = torch.as_tensor(background, dtype=f.dtype, device=f.device)
        bg = bg.expand(self.shape) if bg.ndim else None
        bgs = [background if bg is None else bg[v] for v in range(n)]
        if self.update == "sequential":
            for v in range(n):
                c = self.correction(
                    v,
                    f,
                    data[v],
                    background=bgs[v],
                    # mu holds every view at the starting f; only view 0
                    # is evaluated there, the later views see the updated f
                    mu=None if mu is None or v > 0 else mu[0],
                )
                eps = damping
                if self.weights is not None:
                    eps = damping * self.weights[v]
                if not isinstance(eps, float) or eps != 1.0:
                    c = 1.0 - eps + eps * c
                f = (f * c).clamp_(min=0)
            return f
        if self.update == "multiplicative":
            c = None
            for v in range(n):
                cv = self.correction(
                    v,
                    f,
                    data[v],
                    background=bgs[v],
                    mu=None if mu is None else mu[v],
                )
                if self.weights is not None:
                    w = self.weights[v]
                    cv = 1.0 - w + w * cv
                if n > 1:
                    cv = cv.clamp(min=0).pow(1.0 / n)
                c = cv if c is None else c * cv
        else:
            num = den = None
            for v in range(n):
                m = mu[v] if mu is not None else None
                if m is None:
                    m = _expected(f, self.ops[v], bgs[v])
                bv = self._back(v, data[v] / m)
                nv = self._norms[v]
                num = bv if num is None else num + bv
                den = nv if den is None else den + nv
            valid = den > den.max() * torch.finfo(den.dtype).eps * 16
            tiny = torch.finfo(den.dtype).tiny
            c = num * torch.where(valid, 1 / den.clamp(min=tiny), 0)
        if damping != 1.0:
            c = 1.0 - damping + damping * c
        return (f * c).clamp_(min=0)


class MultiviewRL(Estimator):
    """Multiview Richardson–Lucy deconvolution of registered views.

    The efficient Bayesian-based multiview deconvolution of
    [Preibisch et al. 2014](https://doi.org/10.1038/nmeth.2929),
    and its variants.

    **Model** (Supplementary Note 1.2, Eqs. 12–15). N views y₁, …, y_N of
    one object f, already registered and resampled onto a common grid,
    each blurred by its own PSF Sᵥ (normalised to unit sum, in the
    coordinates of that grid):

        yᵥ ~ Poisson(Hᵥ f),   (Hᵥ f)ᵢ = ∑ⱼ Sᵥ(i − j) fⱼ.

    **Bayesian-based multiview iteration** (Eqs. 36–37). From Bayes'
    theorem, without assuming that the views are independent
    (Supplementary Note 3), the update of the object is the pixel-wise
    product of the single-view Richardson–Lucy corrections,

        f⁽ʳ⁺¹⁾ = f⁽ʳ⁾ ∏ᵥ cᵥ,   cᵥ = Hᵥᵀ[yᵥ / Hᵥ f⁽ʳ⁾] / Hᵥᵀ1,

    normalised by the |V|-th root (geometric mean, Eqs. 46–47) for the
    convergence proof to the maximum-likelihood solution (Supplementary
    Note 2). The product suggests applying the corrections one view after
    the other, each at the current estimate (sequential updates,
    Supplementary Note 5, Eqs. 104–107 and 109–112): ordered-subset EM
    ([Hudson & Larkin 1994](https://doi.org/10.1109/42.363108)) with one
    view per subset, which converges about N times faster.

    **Efficient Bayesian-based iteration** (Supplementary Note 4). The
    conditional probability of what a view w sees, given an observation
    in view v, defines "virtual" views φᵥ^{V_w} ≈ yᵥ ∗ Pᵥ* ∗ P_w (Eq. 77).
    Adding their updates (Eq. 86) and approximating
    (f ∗ g)·(f ∗ h) ≈ f ∗ (g·h) (Eq. 91) folds them into the back
    projection of view v: the mirrored PSF Pᵥ* (Pᵥ = Sᵥ, P*(ξ) = P(−ξ))
    is replaced by the compound kernel (Eqs. 102–103)

        Pᵥ^compound = Pᵥ* · ∏_{w ≠ v} Pᵥ* ∗ P_w ∗ P_w*,

    at no extra cost per iteration. The ad hoc simplifications of
    Supplementary Note 6 sharpen the virtual views further:
    optimisation I drops the convolution with P_w* (Eq. 114), and
    optimisation II assumes P_w constant, which leaves (Pᵥ*)ᴺ (Eq. 115).
    See [`compound_kernels`][optix.deconv.utils.compound_kernels].

    With one view every variant is the Richardson–Lucy iteration of
    [`RichardsonLucy`][optix.deconv.RichardsonLucy].

    Notation: f, yᵥ, Sᵥ and Hᵥᵀ are ψ, φᵥ, Pᵥ and the convolution with
    Pᵥ* of the paper.

    **Faithful default.** The efficient Bayesian-based kernels (Eq. 102)
    with sequential updates (Eqs. 104–107), the method the paper derives
    and names; optimisations I and II, the plain Bayesian-based kernels
    and the combined updates are the variants it benchmarks
    (Supplementary Note 7.1). One iteration is one pass over all the
    views (Supplementary Note 7.1); the number of iterations is the only
    parameter of the method, chosen by the user. The starting estimate
    is the constant image of the mean intensity of the views (Eq. 11).

    **Weights** (Online Methods, Eq. 3). Views that cover only part of the
    sample are blended with weights wᵥ(ξ) ∈ [0, 1] (e.g. a cosine fall-off
    near the view borders): the sequential update of view v applies the
    fraction wᵥ of its correction, f ← f · (1 − wᵥ + wᵥ cᵥ). Weights are
    normalised per pixel to sum to `weight_sum` over the views, each one
    capped at 1, wᵥ ← min(1, k wᵥ / ∑ᵤ wᵤ) (0 where all vanish). k = 1
    (the default) updates every pixel by one full correction per
    iteration; k > 1 trades uniformity for the ordered-subset speed-up
    (Supplementary Fig. 10). Beyond the field of view the weights are
    carried over by back projection (see
    [`Multiview`][optix.deconv.utils.Multiview]; not specified in the
    paper). The weights gate the update only, so a view's missing pixels
    count as observed zeros within a PSF half-width of its coverage: the
    fall-off should be at least as wide as the PSF.

    **Blocks.** As [`RichardsonLucy`][optix.deconv.RichardsonLucy]: the
    object is estimated on a domain padded by the PSF support, and large
    inputs are processed in halo blocks, the same region of every view at
    once. The PSFs are zero-padded to a common shape with odd sizes.
    Sample and group axes are iterated over; the views share one layout.

    !!! note "Extensions (off by default)"
        - **χ² stop:** `chi2_stop=True` replaces the fixed number of
          iterations by the test of
          [Lucy (1974)](https://doi.org/10.1086/111605) on all views:
          stop once P(χ² > χ²_obs) ≥ `alpha`, with χ²_obs =
          ∑ᵥ ∑ᵢ (yᵥᵢ − μᵥᵢ)²/μᵥᵢ, μᵥ = Hᵥ f, and ν = N |Ω| − 1 degrees
          of freedom (|Ω| pixels per view), capped at `max_iter`. The test
          assumes counts.

    !!! warning "Approximations"
        The virtual views rest on Q(ξ|xᵥ) ≈ P(xᵥ|ξ) (Eq. 76) and on
        Eq. 91, which hold for Gaussian-like PSFs; the convergence proof
        (Supplementary Note 2) covers the Bayesian-based kernels only.
        Views must be registered beforehand, and the PSFs given in the
        orientation of the common grid; neither is estimated here.

    !!! references "References"

        1. Preibisch S, Amat F, Stamataki E, Sarov M, Singer RH, Myers E,
           Tomancak P. *Efficient Bayesian-based multiview
           deconvolution.* Nat Methods **11**(6):645-648, 2014.
           [doi:10.1038/nmeth.2929](https://doi.org/10.1038/nmeth.2929)
        2. Hudson HM, Larkin RS. *Accelerated image reconstruction using
           ordered subsets of projection data.* IEEE Trans Med Imaging
           **13**(4):601-609, 1994.
           [doi:10.1109/42.363108](https://doi.org/10.1109/42.363108)
        3. Shepp LA, Vardi Y. *Maximum likelihood reconstruction for
           emission tomography.* IEEE Trans Med Imaging **1**(2):113-122,
           1982. [doi:10.1109/TMI.1982.4307558](https://doi.org/10.1109/TMI.1982.4307558)

        ??? quote "BibTeX"

            ```bibtex
            @article{preibisch2014efficient,
              author  = {Preibisch, Stephan and Amat, Fernando and
                         Stamataki, Evangelia and Sarov, Mihail and
                         Singer, Robert H. and Myers, Eugene and
                         Tomancak, Pavel},
              title   = {Efficient {Bayesian}-based multiview
                         deconvolution},
              journal = {Nature Methods},
              volume  = {11},
              number  = {6},
              pages   = {645--648},
              year    = {2014},
              doi     = {10.1038/nmeth.2929}
            }
            @article{hudson1994accelerated,
              author  = {Hudson, H. Malcolm and Larkin, Richard S.},
              title   = {Accelerated image reconstruction using ordered
                         subsets of projection data},
              journal = {IEEE Transactions on Medical Imaging},
              volume  = {13},
              number  = {4},
              pages   = {601--609},
              year    = {1994},
              doi     = {10.1109/42.363108}
            }
            @article{shepp1982maximum,
              author  = {Shepp, Lawrence A. and Vardi, Yehuda},
              title   = {Maximum likelihood reconstruction for emission
                         tomography},
              journal = {IEEE Transactions on Medical Imaging},
              volume  = {1},
              number  = {2},
              pages   = {113--122},
              year    = {1982},
              doi     = {10.1109/TMI.1982.4307558}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import MultiviewRL
    >>> psfs = [torch.ones(1, 5), torch.ones(5, 1)]
    >>> views = [torch.poisson(torch.full((32, 32), 10.0)) for _ in psfs]
    >>> mv = MultiviewRL(psfs, n_iter=5).fit(views)
    >>> mv.apply(views).shape
    torch.Size([32, 32])
    """

    tags = Tags(
        targets=("deconvolution",),
        streaming="halo",
        noise="poisson",
        reference="doi:10.1038/nmeth.2929",
    )
    paper = {
        "kernel": "efficient",
        "update": "sequential",
        "n_iter": 10,
        "chi2_stop": False,
        "alpha": 0.05,
        "max_iter": 100,
        "weight_sum": 1.0,
        "halo": None,
        "block": None,
    }
    recommended = dict(paper)

    def __init__(
        self,
        psfs: Sequence[Tensor] | None = None,
        *,
        kernel: str | None = None,
        update: str | None = None,
        n_iter: int | None = None,
        weights: Sequence[ArrayLike] | None = None,
        weight_sum: float | None = None,
        chi2_stop: bool | None = None,
        alpha: float | None = None,
        max_iter: int | None = None,
        halo: int | Sequence[int] | None = None,
        block: Sequence[int | None] | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the deconvolution.

        Parameters
        ----------
        psfs : sequence of Tensor
            PSF of each view, `(*K)` with 2 or 3 dims, in the order of the
            spatial axes of the layout and in the orientation of the
            common grid, centred at index `K // 2`. Each is normalised to
            unit sum; their shapes may differ (they are zero-padded to a
            common shape with odd sizes). A light-sheet PSF can come from
            [`lightsheet_psf`][optix.psf.lightsheet_psf].
        kernel : {"efficient", "bayesian", "optimization-1", "optimization-2"}
            Back-projection kernels (see
            [`compound_kernels`][optix.deconv.utils.compound_kernels]).
            Default: `"efficient"` (Eq. 102).
        update : {"sequential", "multiplicative", "additive"}
            Combination of the views (see
            [`Multiview.step`][optix.deconv.utils.Multiview.step]):
            sequential updates (Eqs. 104–107), the geometric mean of
            the corrections (Eqs. 36, 46), or their arithmetic mean
            (maximum-likelihood EM, the additive combination the paper
            compares against). Default: `"sequential"`.
        n_iter : int
            Number of iterations (passes over all views). Default: 10
            (not specified in the paper, where the user chooses it).
        weights : sequence of array-like, optional
            Weight wᵥ ≥ 0 of each view (Eq. 3), `(*spatial)` in layout
            order, the size of the views' spatial extent (tensors or lazy
            arrays); sequential update only. Default: no weights (every
            view updates every pixel fully). [`save`][optix.Estimator.save]
            stores tensor and numpy weights; lazy weights are not read
            into memory, so after [`load`][optix.load] they must be set
            again (`est.weights = [...]`) before `apply`.
        weight_sum : float
            Sum k of the weights over the views after normalisation.
            Default: 1 (Online Methods).
        chi2_stop : bool
            Extension: stop by the χ² test instead of after `n_iter`
            iterations. Default: False.
        alpha : float
            Level of the χ² test (with `chi2_stop`). Default (Lucy 1974):
            0.05.
        max_iter : int
            Iteration cap of the χ² test. Default: 100 (not specified in
            the papers).
        halo : int or sequence of int, optional
            Halo width per spatial axis, at least the (common) PSF size.
            Default: twice the PSF size (not specified in the paper).
        block : sequence of int or None, optional
            Core block size per spatial axis (None entries: whole
            extent). Default: chosen from the memory budget.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        params = {
            k: v for k, v in locals().items() if k not in ("psfs", "weights")
        }
        self._resolve(faithful, params)
        self.psfs = None if psfs is None else list(psfs)
        self.weights = None if weights is None else list(weights)
        self.level: Tensor | None = None
        self._check_options()

    def _check_options(self) -> None:
        opts = self.options
        if opts["kernel"] not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        if opts["update"] not in UPDATES:
            raise ValueError(f"update must be one of {UPDATES}")
        if int(opts["n_iter"]) < 0:
            raise ValueError("n_iter must be non-negative")
        if not 0.0 < float(opts["alpha"]) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not float(opts["weight_sum"]) > 0:
            raise ValueError("weight_sum must be positive")
        psfs = self.psfs
        if psfs is not None:
            if not psfs:
                raise ValueError("psfs must hold one PSF per view")
            if any(torch.as_tensor(p).ndim not in (2, 3) for p in psfs):
                raise ValueError("PSFs must be 2-D or 3-D")
            if self.weights is not None and len(self.weights) != len(psfs):
                raise ValueError("weights must hold one array per view")
        if self.weights is not None and opts["update"] != "sequential":
            raise ValueError("weights apply to the sequential update only")

    # -- helpers ---------------------------------------------------------
    def _psfs(self, backend: dict) -> list[Tensor]:
        """PSFs normalised to unit sum, padded to a common odd shape."""
        if self.psfs is None:
            raise ValueError("MultiviewRL needs psfs")
        out = []
        for p in self.psfs:
            p = torch.as_tensor(p).to(**backend)
            if (p < 0).any():
                raise ValueError("PSFs must be non-negative")
            total = p.sum()
            if not total > 0:
                raise ValueError("PSFs must have a positive sum")
            out.append(p / total)
        shape = _odd_shape(out)
        return [_pad_centred(p, shape) for p in out]

    def kernels(self, **backend: object) -> list[Tensor]:
        """Back-projection kernel of each view, `(*K)`, unit sum.

        See [`compound_kernels`][optix.deconv.utils.compound_kernels].

        Parameters
        ----------
        **backend
            `dtype` and `device` of the result (default: the estimator's
            backend).

        Returns
        -------
        list of Tensor
        """
        if not backend:
            backend = self._backend(None)
        psfs = self._psfs(backend)
        ks = compound_kernels(psfs, self.options["kernel"])
        return [k.to(**backend) for k in ks]

    def _halo(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        halo = self.options["halo"]
        if halo is None:
            return tuple(2 * k for k in shape)
        halo = (
            (int(halo),) * len(shape)
            if isinstance(halo, int)
            else tuple(int(h) for h in halo)
        )
        if len(halo) != len(shape):
            raise ValueError("halo must have one entry per PSF dim")
        if any(h < k for h, k in zip(halo, shape, strict=True)):
            raise ValueError(
                f"halo {halo} must be at least the PSF size {shape}"
            )
        return halo

    def _views(
        self, data: Sequence[ArrayLike], layout: Layout | None
    ) -> list[Labeled]:
        """Each view labelled with one layout; shapes checked."""
        if isinstance(data, Labeled) or not len(data):
            raise TypeError("data must be a sequence of views")
        views = list(data)
        if self.psfs is not None and len(views) != len(self.psfs):
            raise ValueError(
                f"{len(views)} views but {len(self.psfs)} PSFs were given"
            )
        if layout is None:
            for v in views:
                if isinstance(v, Labeled):
                    layout = v.layout
                    break
        layout = self.layout if layout is None else layout
        if layout is None:
            ndim = torch.as_tensor(self.psfs[0]).ndim if self.psfs else 2
            layout = Layout.default(len(views[0].shape), ndim)
        views = [self._labeled(v, layout) for v in views]
        shape = tuple(views[0].shape)
        if any(tuple(v.shape) != shape for v in views):
            raise ValueError("All views must have the same shape")
        self.layout = layout
        return views

    # -- estimator protocol ----------------------------------------------
    def fit(
        self,
        data: Sequence[ArrayLike],
        *,
        layout: Layout | None = None,
        init: object = None,
        callback: Callable | None = None,
        verbose: bool = False,
    ) -> "MultiviewRL":
        """Prepare the deconvolution of registered views.

        One pass over the views computes the starting level of each
        image (sample/group index): the mean intensity of all views.

        Parameters
        ----------
        data : sequence of array-like or Labeled
            Views, one per PSF, all of the same shape (`(*samples,
            *spatial)` without a layout), on the common grid.
        layout : Layout, optional
            Axis roles, shared by the views.
        init, callback, verbose
            Unused.

        Returns
        -------
        MultiviewRL
            self.
        """
        self.history = History()
        views = self._views(data, layout)
        with torch.no_grad():
            backend = self._backend(views[0])
            psfs = self._psfs(backend)
            if psfs[0].ndim != len(views[0].layout.spatial):
                raise ValueError(
                    f"The PSFs have {psfs[0].ndim} dims but the layout has "
                    f"{len(views[0].layout.spatial)} spatial axes"
                )
            group, sample, spatial = views[0].layout.shape_of(views[0].array)
            total = torch.zeros(group + sample, dtype=torch.float64)
            count = int(np.prod(spatial)) * len(views)
            for view in views:
                for chunk in blocks(view, None, 0, **backend):
                    value = chunk.data.sum().double().cpu()
                    total[chunk.group + chunk.samples] += value
            level = total / count
            self.level = level.clamp(min=torch.finfo(backend["dtype"]).tiny)
        self.history.extras["iterations"] = []
        return self

    def apply(
        self,
        data: Sequence[ArrayLike],
        out: WritableArrayLike | None = None,
    ) -> WritableArrayLike:
        """Deconvolve the views into one image.

        Parameters
        ----------
        data : sequence of array-like or Labeled
            The views given to [`fit`][optix.deconv.MultiviewRL.fit].
        out : writable array-like, optional
            Output, the shape of one view (tensor, numpy, zarr, ...).
            Default: a new tensor.

        Returns
        -------
        writable array-like
            `out`, the deconvolved image(s) on the common grid.
        """
        if self.level is None:
            raise RuntimeError("Call fit(data) before apply(data)")
        views = self._views(data, None)
        with torch.no_grad():
            return self._run(views, out)

    def _run(
        self, views: list[Labeled], out: WritableArrayLike | None
    ) -> WritableArrayLike:
        group, sample, spatial = views[0].layout.shape_of(views[0].array)
        if tuple(self.level.shape) != group + sample:
            raise ValueError(
                "apply(data) needs the views given to fit (images per "
                f"group/sample {tuple(self.level.shape)}, got "
                f"{group + sample})"
            )
        backend = self._backend(views[0])
        psfs = self._psfs(backend)
        kernels: list[Tensor | None] = [None] * len(psfs)
        if self.options["kernel"] != "bayesian" and len(psfs) > 1:
            kernels = self.kernels(**backend)
        halo = self._halo(tuple(psfs[0].shape))
        raw = [v.array for v in views]
        target = out.array if isinstance(out, Labeled) else out
        if out is not None and any(target is r for r in raw):
            raise ValueError("In-place deconvolution is not possible")
        out = prepare_out(
            out, tuple(views[0].shape), like=raw[0], dtype=self.dtype
        )
        weights = self._weight_arrays(spatial)
        n = len(views)
        overhead = _OVERHEAD_BASE + _OVERHEAD_VIEW * n
        align = chunks_of(out) or (1,) * len(views[0].shape)
        iters = [
            blocks(
                v,
                self.options["block"],
                halo,
                overhead=overhead,
                contiguous=True,
                align=align,
                **backend,
            )
            for v in views
        ]
        opts = self.options
        chi2 = bool(opts["chi2_stop"])
        cache: dict[tuple[int, ...], Multiview] = {}
        iterations = self.history.extras.setdefault("iterations", [])
        for chunks in zip(*iters, strict=True):
            chunk = chunks[0]
            if any(c.index != chunk.index for c in chunks[1:]):
                raise RuntimeError("The views were not blocked alike")
            data = torch.stack([c.data for c in chunks])
            key = tuple(chunk.data.shape)
            if key not in cache:
                ops = [PaddedConvolution(p, key) for p in psfs]
                cache[key] = Multiview(ops, kernels, update=opts["update"])
            mv = cache[key]
            if weights is not None:
                region = RichardsonLucy._region(chunk)
                w = torch.stack([_read(a, region, backend) for a in weights])
                mv = mv.weighted(_normalise_weights(w, opts["weight_sum"]))
            start = float(self.level[chunk.group + chunk.samples].squeeze())
            f, k = rl_block(
                data,
                mv,
                start,
                n_iter=None if chi2 else int(opts["n_iter"]),
                alpha=float(opts["alpha"]),
                max_iter=int(opts["max_iter"]),
                step=mv.step,
            )
            iterations.append(float(k))
            chunk.store(out, mv.crop(f)[chunk.inner])
        flush(out)
        return out

    def _weight_arrays(self, spatial: tuple[int, ...]) -> list | None:
        if self.weights is None:
            return None
        if any(w is None for w in self.weights):
            raise ValueError(
                "Lazy weights are not saved: set `weights` (one array per "
                "view) again after loading"
            )
        for w in self.weights:
            if tuple(w.shape) != tuple(spatial):
                raise ValueError(
                    f"weights must have the spatial shape {tuple(spatial)}, "
                    f"got {tuple(w.shape)}"
                )
        return self.weights

    def _extra_state(self) -> dict[str, object]:
        # in-memory weights are saved; lazy ones (zarr, ...) are not read
        # whole: they are stored as None and must be set again after load
        weights = None
        if self.weights is not None:
            weights = [
                torch.as_tensor(w)
                if isinstance(w, (Tensor, np.ndarray))
                else None
                for w in self.weights
            ]
        return {"psfs": self.psfs, "level": self.level, "weights": weights}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.psfs = state.get("psfs")
        self.level = state.get("level")
        self.weights = state.get("weights")


def _read(
    array: ArrayLike, region: tuple[slice, ...], backend: dict
) -> Tensor:
    value = array[region]
    if not isinstance(value, Tensor):
        value = torch.from_numpy(np.asarray(value))
    return value.to(**backend)


def _normalise_weights(w: Tensor, total: float) -> Tensor:
    """wᵥ ← min(1, k wᵥ / ∑ᵤ wᵤ), 0 where all weights vanish."""
    w = w.clamp(min=0)
    s = w.sum(0)
    tiny = torch.finfo(w.dtype).tiny
    scaled = (float(total) * w / s.clamp(min=tiny)).clamp(max=1)
    return torch.where(s > 0, scaled, 0)


def multiview_rl(
    data: Sequence[ArrayLike],
    psfs: Sequence[Tensor],
    *,
    out: WritableArrayLike | None = None,
    weights: Sequence[ArrayLike] | None = None,
    layout: Layout | None = None,
    **options: object,
) -> WritableArrayLike:
    """Multiview deconvolution of registered views.

    The method of
    [Preibisch et al. 2014](https://doi.org/10.1038/nmeth.2929).

    Shortcut for `MultiviewRL(psfs, weights=weights, **options).fit(data)
    .apply(data, out)`; see [`MultiviewRL`][optix.deconv.MultiviewRL] for
    the model, the variants and the defaults.

    Parameters
    ----------
    data : sequence of array-like or Labeled
        Views on the common grid, one per PSF, all of the same shape.
    psfs : sequence of Tensor
        PSF of each view, centred at index `K // 2`.
    out : writable array-like, optional
        Output (tensor, numpy, zarr, ...). Default: a new tensor.
    weights : sequence of array-like, optional
        Per-view weights of the sequential update.
    layout : Layout, optional
        Axis roles, shared by the views.
    **options
        Options of [`MultiviewRL`][optix.deconv.MultiviewRL] (`kernel`,
        `update`, `n_iter`, `halo`, `faithful`, ...).

    Returns
    -------
    writable array-like
        `out`, the deconvolved image(s).
    """
    est = MultiviewRL(psfs, weights=weights, layout=layout, **options)
    est.fit(data)
    return est.apply(data, out)
