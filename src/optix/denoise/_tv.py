"""Total-variation denoising (Rudin, Osher & Fatemi 1992).

Written from the paper. The model is solved with the primal–dual
algorithm of Chambolle & Pock (2011) and the building blocks of
`optix.optim`.
"""

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple, Self

import torch
from torch import Tensor

from .._config import get_config, resolve_backend
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
from ..optim import (
    Gradient,
    GroupL2,
    L2Ball,
    chambolle_pock,
    prox_shifted_poisson,
    prox_squared_l2,
)
from ..optim._prox import _sum64
from ..sensor._anscombe import anscombe, build_table, inverse_anscombe
from ._bm3d import _constant, _select

__all__ = ["ROF", "TVResult", "rof", "tv_denoise"]

DataTerm = Literal["gaussian", "kl"]
_DATA_TERMS = ("gaussian", "kl")
_INVERSES = ("exact", "closed-form", "asymptotic", "algebraic")

#: Default solver settings (not in the paper, which marches in time to
#: a steady state): at most `MAX_ITER` iterations, stopped when the
#: relative primal–dual gap is at most `TOL`, checked every `CHECK`.
MAX_ITER = 2000
TOL = 1e-4
CHECK = 10
#: Split of the primal and dual steps of the constrained model:
#: κ = `SPLIT`·σ/√ω̄ in τ_p = κ/‖∇‖, τ_d = 0.99/(κ‖∇‖) (not in the paper).
SPLIT = 1 / 8


def _peak_memory(ndim: int, data_term: DataTerm = "gaussian") -> int:
    """Peak working memory of `tv_denoise`, in multiples of the image.

    About 2d + 5 image sizes with the image itself (d spatial dims;
    7.3 measured in 2-D and 10.3 in 3-D, float32), three times that for
    the KL term, whose gap is evaluated in float64.
    """
    base = 2 * int(ndim) + 5
    return 3 * base if data_term == "kl" else base


class TVResult(NamedTuple):
    """Result of [`tv_denoise`][optix.denoise.utils.tv_denoise].

    Attributes
    ----------
    x : Tensor
        Denoised image u.
    lam : float
        Weight λ of the data term: in the constrained model, the
        Lagrange multiplier of the noise constraint, the λ of Eq. 2.5a
        (twice the value of the printed Eq. 2.6, and with pixel spacing
        h = 1 where the paper has h = 1/N); 0 when a constant is
        feasible. Else the given λ.
    n_iter : int
        Number of iterations (0 when a constant is feasible).
    gap : float
        Last relative primal–dual gap (P − D)/P, with P the primal
        objective less the minimum of the data term (0 when a constant
        is feasible).
    """

    x: Tensor
    lam: float
    n_iter: int
    gap: float


# -- data terms --------------------------------------------------------------
class _Quadratic:
    """G(u) = (λ/2)∑ wⱼ(uⱼ − fⱼ)²."""

    def __init__(self, f: Tensor, lam: float, w: Tensor | float) -> None:
        self.f, self.lam, self.w = f, lam, w

    def prox(self, v: Tensor, t: float) -> Tensor:
        return prox_squared_l2(v, t, self.f, self.lam * self.w)

    def value(self, u: Tensor) -> float:
        return self.lam / 2 * _sum64(self.w * (u - self.f).square())

    def conjugate(self, z: Tensor) -> float:
        """G*(z) = ∑ zⱼfⱼ + zⱼ²/(2λwⱼ)."""
        lw = self.lam * self.w
        return _sum64(z * self.f + z.square() / (2 * lw))

    def minimum(self) -> float:
        return 0.0


class _Poisson:
    """G(u) = λ∑ wⱼ(uⱼ − cⱼ log(uⱼ + s²)), c = f + s² (shifted Poisson)."""

    def __init__(
        self, f: Tensor, lam: float, w: Tensor | float, s2: Tensor | float
    ) -> None:
        self.s2 = s2
        self.c = (f + s2).clamp(min=0)
        self.f = self.c - s2
        self.lam, self.w = lam, w

    def prox(self, v: Tensor, t: float) -> Tensor:
        return prox_shifted_poisson(v, t * self.lam * self.w, self.f, self.s2)

    def _value(self, u: Tensor) -> Tensor:
        c = self.c.double()
        s = (u + self.s2).double()
        log = torch.where(c > 0, torch.log(s.clamp(min=1e-300)), 0)
        return self.lam * _double(self.w) * (u.double() - c * log)

    def value(self, u: Tensor) -> float:
        return float(self._value(u).sum())

    def conjugate(self, z: Tensor) -> float:
        """G*(z), with a = λw − z > 0 (else +inf where c > 0):

        ∑ a s² − λwc + λwc log(λwc / a).
        """
        z = z.double()
        lw = self.lam * _double(self.w) + torch.zeros_like(z)
        c = self.c.double() + torch.zeros_like(z)
        s2 = _double(self.s2)
        a = lw - z
        if bool((a[c > 0] <= 0).any()) or bool((a < 0).any()):
            return math.inf
        lc = lw * c
        term = torch.where(
            c > 0, lc * torch.log(lc / a.clamp(min=1e-300)) - lc, 0
        )
        return float((a * s2 + term).sum())

    def minimum(self) -> float:
        return self.value(self.f)


def _double(t: Tensor | float) -> Tensor | float:
    return t.double() if isinstance(t, Tensor) else t


def _tv(op: Gradient, u: Tensor) -> float:
    return _sum64(torch.linalg.vector_norm(op.forward(u), dim=0))


def _not_converged(res: TVResult, max_iter: int, tol: float | None) -> bool:
    return tol is not None and res.n_iter >= max_iter and not res.gap <= tol


def _warn_max_iter(
    name: str,
    gaps: Sequence[float],
    total: int,
    max_iter: int,
    tol: float,
    stacklevel: int = 3,
) -> None:
    warnings.warn(
        f"{name}: {len(gaps)} of {total} image(s) stopped at "
        f"max_iter={max_iter} with a relative primal–dual gap above "
        f"tol={tol:g} (worst {max(gaps):.3g}); raise max_iter or tol",
        RuntimeWarning,
        stacklevel=stacklevel,
    )


# -- solver ------------------------------------------------------------------
def tv_denoise(
    data: Tensor,
    *,
    sigma: float | None = None,
    lam: float | None = None,
    data_term: DataTerm = "gaussian",
    weights: Tensor | None = None,
    axis_weights: Sequence[float] | None = None,
    variance: Tensor | float = 0.0,
    max_iter: int = MAX_ITER,
    tol: float | None = TOL,
    check_every: int = CHECK,
    trace: dict | None = None,
    warn: bool = True,
) -> TVResult:
    """Total-variation denoising of one image or volume.

    The total variation is that of the discrete forward-difference
    gradient with Neumann boundaries (Eqs. 2.4b and 2.9a),

        TV(u) = ∑ⱼ (∑ₐ (wₐ D⁺ₐ u)ⱼ²)^½,

    over all dims of `data` (axis weights wₐ = 1 by default). Models:

    - Gaussian, constrained (`sigma`; Rudin, Osher & Fatemi 1992,
      Eqs. 2.3a–c): minimise TV(u) subject to ∑ⱼ ωⱼ(uⱼ − fⱼ)² ≤ Mσ²,
      M the number of samples, pixel weights ω = 1 by default.
    - Gaussian, penalised (`lam`): minimise TV(u) + (λ/2)∑ⱼ ωⱼ(uⱼ − fⱼ)².
    - Kullback–Leibler (Poisson; `lam`, `data_term="kl"`): minimise
      TV(u) + λ∑ⱼ ωⱼ(uⱼ − (fⱼ + s²) log(uⱼ + s²)), the shifted-Poisson
      likelihood of [`ShiftedPoisson`][optix.optim.ShiftedPoisson] with
      read variance s² (`variance`, 0: Poisson); u ≥ −s².

    All three are solved with
    [`chambolle_pock`][optix.optim.chambolle_pock] (K = ∇, F = ‖·‖₂,₁,
    G = the data term), with the accelerated steps of its Algorithm 2
    for the penalised Gaussian model (G is λ·min ω-strongly convex). The
    primal and dual steps are τ_p = κ/‖∇‖ and τ_d = 0.99/(κ‖∇‖), with κ
    a scale of the data term in data units: `SPLIT`·σ/√ω̄ with
    `SPLIT` = 1/8 (constrained), 1/(λω̄) (penalised) and c̄/(λω̄) (KL,
    c̄ the mean of f + s²). Iterations stop when the relative
    primal–dual gap (P − D)/(P − min G) is at most `tol` (checked every
    `check_every` iterations), with P = TV(u) + G(u) and D = −G*(−∇ᵀy).
    A run that stops at `max_iter` above `tol` emits a `RuntimeWarning`
    (with `warn`). Neither the solver nor the step sizes nor the
    stopping rule are in the paper.

    In the constrained model, when a constant image is feasible
    (∑ⱼ ωⱼ(fⱼ − c)² ≤ Mσ² with c the ω-weighted mean of f), every
    feasible constant is a minimiser (TV = 0), so the solution is not
    unique; c is returned at once, with λ = 0, no iteration and gap 0.

    The working memory is about 2d + 5 image sizes (d the number of
    dims; 9 for an image) for the Gaussian data terms, and about three
    times that for the KL term, whose gap is evaluated in float64.

    Parameters
    ----------
    data : Tensor
        Noisy image f, all dims spatial.
    sigma : float, optional
        Noise standard deviation σ (constrained Gaussian model).
    lam : float, optional
        Weight λ > 0 of the data term (penalised models).
    data_term : {"gaussian", "kl"}
        Data term.
    weights : Tensor, optional
        Pixel weights ω > 0, broadcasting with `data`. Default: 1.
    axis_weights : sequence of float, optional
        Weights wₐ > 0 of the differences along each axis. Default: 1.
    variance : Tensor or float
        Read variance s² ≥ 0 of the KL term.
    max_iter : int
        Maximum number of iterations.
    tol : float, optional
        Tolerance on the relative primal–dual gap.
    check_every : int
        Iterations between two evaluations of the gap.
    trace : dict, optional
        See [`chambolle_pock`][optix.optim.chambolle_pock].
    warn : bool
        Warn when the run stops at `max_iter` with a gap above `tol`.

    Returns
    -------
    TVResult
    """
    if data_term not in _DATA_TERMS:
        raise ValueError(f"data_term must be one of {_DATA_TERMS}")
    if (sigma is None) == (lam is None):
        raise ValueError("give exactly one of sigma (constrained) or lam")
    if data_term == "kl" and lam is None:
        raise ValueError("the KL data term needs lam")
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    f = data
    w: Tensor | float = 1.0
    if weights is not None:
        w = torch.as_tensor(weights).to(f)
        if not bool((w > 0).all()):
            raise ValueError("weights must be positive")
        if _constant(w):
            w = float(w.reshape(-1)[0])
    wmean = w if isinstance(w, float) else _sum64(w) / w.numel()
    wmin = w if isinstance(w, float) else float(w.min())
    op = Gradient(f.ndim, weights=axis_weights, boundary="neumann")
    if any(not a > 0 for a in op.weights):
        raise ValueError("axis_weights must be positive")
    norm = math.sqrt(op.norm2)
    tv_fn = GroupL2(1.0, dim=0)

    if sigma is not None:
        sigma = float(sigma)
        if not (math.isfinite(sigma) and sigma > 0):
            raise ValueError(f"sigma must be finite and > 0, got {sigma}")
        radius = math.sqrt(f.numel()) * sigma
        # a feasible constant (the ω-weighted mean) is a minimiser
        if isinstance(w, float):
            c = _sum64(f) / f.numel()
            res2 = w * _sum64((f - c).square())
        else:
            c = _sum64(w * f) / _sum64(w)
            res2 = _sum64(w * (f - c).square())
        if res2 <= radius * radius:
            return TVResult(torch.full_like(f, c), 0.0, 0, 0.0)
        primal = L2Ball(radius, f, w)
        scale = SPLIT * sigma / math.sqrt(wmean)
        gamma = 0.0
        inv_w = 1 / w

        def gap(u: Tensor, y: Tensor) -> float:
            p = _tv(op, u)
            z = op.adjoint(y)
            d = _sum64(z * f) - radius * math.sqrt(_sum64(z.square() * inv_w))
            return (p - d) / p if p > 0 else abs(d)
    else:
        lam = float(lam)
        if not (math.isfinite(lam) and lam > 0):
            raise ValueError(f"lam must be finite and > 0, got {lam}")
        if data_term == "gaussian":
            primal = _Quadratic(f, lam, w)
            scale = 1 / (lam * wmean)
            gamma = lam * wmin
        else:
            primal = _Poisson(f, lam, w, variance)
            cbar = float(primal.c.double().mean())
            scale = max(cbar, 1e-12) / (lam * wmean)
            gamma = 0.0
        low = primal.minimum()

        def gap(u: Tensor, y: Tensor) -> float:
            p = _tv(op, u) + primal.value(u) - low
            z = -op.adjoint(y)
            if data_term == "kl":
                # scale y into the domain of G* (z < λw), y = 0 is in it;
                # in float64, where the margin 1 − 10⁻⁹ is not lost
                z = z.double()
                lw = lam * _double(w) + torch.zeros_like(z)
                pos = z > 0
                if bool(pos.any()):
                    t = float((lw[pos] / z[pos]).min()) * (1 - 1e-9)
                    z = z * min(1.0, t)
            d = -primal.conjugate(z) - low
            return (p - d) / p if p > 0 else abs(d)

    res = chambolle_pock(
        op,
        tv_fn,
        primal,
        f,
        tau=scale / norm,
        sigma=0.99 / (scale * norm),
        strong_convexity=gamma,
        max_iter=max_iter,
        tol=tol,
        gap=gap,
        check_every=check_every,
        trace=trace,
    )
    u, y = res.x, res.duals[0]
    if sigma is not None:
        e = w * (u - f)
        den = _sum64(e.square())
        lam = -_sum64(op.adjoint(y) * e) / den if den > 0 else 0.0
    out = TVResult(u, float(lam), res.n_iter, res.residual)
    if warn and _not_converged(out, max_iter, tol):
        _warn_max_iter("tv_denoise", [out.gap], 1, max_iter, tol)
    return out


# -- references --------------------------------------------------------------


def _check_ndim(ndim: object) -> int:
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3, got {ndim!r}")
    return int(ndim)


def rof(
    data: Tensor,
    sigma: float | None = None,
    *,
    lam: float | None = None,
    data_term: DataTerm = "gaussian",
    weights: Tensor | None = None,
    axis_weights: Sequence[float] | None = None,
    ndim: int = 2,
    max_iter: int = MAX_ITER,
    tol: float | None = TOL,
) -> Tensor:
    """Total-variation denoising (Rudin, Osher & Fatemi 1992).

    Each image (the last `ndim` dims) is denoised separately with
    [`tv_denoise`][optix.denoise.utils.tv_denoise]. With `sigma` this
    is the model of the paper: minimise TV(u) subject to
    ∑ⱼ (uⱼ − fⱼ)² ≤ Mσ² (Eqs. 2.3a–c); `lam`, `data_term`, `weights`,
    `axis_weights` and `ndim=3` are extensions (see
    [`ROF`][optix.denoise.ROF]).

    !!! references "References"
        1. Rudin LI, Osher S, Fatemi E. *Nonlinear total variation based
           noise removal algorithms.* Physica D **60**(1–4):259–268, 1992.
           [doi:10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F)
        2. Chambolle A, Pock T. *A first-order primal-dual algorithm for
           convex problems with applications to imaging.* J Math Imaging
           Vis **40**(1):120–145, 2011.
           [doi:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)

        ??? quote "BibTeX"
            ```bibtex
            @article{rudin1992nonlinear,
              author  = {Rudin, Leonid I. and Osher, Stanley and
                         Fatemi, Emad},
              title   = {Nonlinear total variation based noise removal
                         algorithms},
              journal = {Physica D: Nonlinear Phenomena},
              volume  = {60}, number = {1--4}, pages = {259--268},
              year    = {1992}, doi = {10.1016/0167-2789(92)90242-F}
            }
            @article{chambolle2011first,
              author  = {Chambolle, Antonin and Pock, Thomas},
              title   = {A first-order primal-dual algorithm for convex
                         problems with applications to imaging},
              journal = {Journal of Mathematical Imaging and Vision},
              volume  = {40}, number = {1}, pages = {120--145},
              year    = {2011}, doi = {10.1007/s10851-010-0251-1}
            }
            ```

    Parameters
    ----------
    data : Tensor
        `(..., *spatial)`.
    sigma : float, optional
        Noise standard deviation σ, in the units of `data`.
    lam : float, optional
        Extension: weight λ of a penalised data term, instead of σ.
    data_term : {"gaussian", "kl"}
        Extension `"kl"`: Poisson (Kullback–Leibler) data term on counts
        (needs `lam`).
    weights : Tensor, optional
        Extension: pixel weights ω > 0, broadcasting with an image.
    axis_weights : sequence of float, optional
        Extension: weights of the differences along each spatial axis.
    ndim : {2, 3}
        Number of spatial dims (3: extension).
    max_iter, tol
        Solver settings (see [`tv_denoise`][optix.denoise.utils.tv_denoise]).
        One `RuntimeWarning` per call, with the worst gap, when images
        stop at `max_iter` above `tol`.

    Returns
    -------
    Tensor
        Of the shape of `data`.
    """
    ndim = _check_ndim(ndim)
    if data.ndim < ndim:
        raise ValueError(f"data needs at least {ndim} dims, got {data.ndim}")
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    shape = data.shape
    flat = data.reshape(-1, *shape[-ndim:])
    results = [
        tv_denoise(
            x,
            sigma=sigma,
            lam=lam,
            data_term=data_term,
            weights=weights,
            axis_weights=axis_weights,
            max_iter=max_iter,
            tol=tol,
            warn=False,
        )
        for x in flat
    ]
    gaps = [r.gap for r in results if _not_converged(r, max_iter, tol)]
    if gaps:
        _warn_max_iter("rof", gaps, len(results), max_iter, tol)
    return torch.stack([r.x for r in results]).reshape(shape)


class ROF(Estimator):
    """Total-variation denoising of Rudin, Osher & Fatemi (1992).

    The image u is the solution of (Eqs. 2.3a–c, discretised as in
    Eqs. 2.9a and 2.4b: forward differences, Neumann boundaries)

        minimise TV(u) = ∑ⱼ ‖(∇u)ⱼ‖₂  subject to  ∑ⱼ (uⱼ − fⱼ)² ≤ Mσ²,

    f the noisy image of M pixels and σ the standard deviation of its
    white Gaussian noise. The mean constraint (2.3b) holds at the
    solution (TV is invariant to an offset and the mean of f minimises
    the residual). The equality of Eq. 2.3c is relaxed to an inequality,
    which is the same problem unless a constant image is feasible
    (σ² at least the variance of f). Then every feasible constant is a
    minimiser, so the solution is not unique; the mean of f is returned.

    The paper reaches the minimiser by marching the Euler–Lagrange
    equation (2.5a) in time with the gradient-projection multiplier λ(t)
    of Eq. 2.6 (scheme 2.8–2.9). Here the same model is solved to a
    prescribed primal–dual gap with the convergent primal–dual method
    of Chambolle & Pock (2011)
    ([`tv_denoise`][optix.denoise.utils.tv_denoise]); the multiplier λ
    of the solution is available from that function. Each image (the
    `ndim` spatial axes of the layout) is solved whole; the other axes
    are streamed one image at a time. The solver needs about 2d + 5
    image sizes of working memory (d = `ndim`; three times that with
    the KL data term), which must fit in the memory budget of
    [`config`][optix.config], else `apply` raises a `MemoryError`. The
    iteration count and the final gap of each image are appended to
    `history.extras["iterations"]` and `history.extras["gap"]`; one
    `RuntimeWarning` per `apply` reports images stopped at `max_iter`
    above `tol`, with the worst gap.

    Poisson-Gaussian noise: give a
    [`SensorEstimate`][optix.estimates.SensorEstimate] to `fit`. With the
    Gaussian data term the data are stabilised by the generalised
    Anscombe transform (noise of unit variance, σ = 1) and transformed
    back by the inverse selected by `inverse`, as in
    [`BM3D`][optix.denoise.BM3D]; with the KL data term they are
    converted to electrons, (f − offset)/gain, with the read variance
    variance/gain² of the shifted-Poisson likelihood, and converted back.
    Maps of the estimate apply pixel by pixel.

    !!! note "Faithful mode and extensions"
        Faithful mode is the constrained Gaussian model in 2-D with the
        noise level `sigma` (or σ = 1 after the variance-stabilising
        transform, which is not in the paper). Extensions (off by
        default): `lam` (penalised data term λ/2·∑(u − f)² instead of
        the constraint), `data_term="kl"` (Poisson likelihood
        λ∑(u − f log u), with `lam`), `weights` (pixel weights ω of the
        data term), `axis_weights` (weights of the differences per axis,
        for example a smaller weight along an anisotropic z axis) and
        `ndim=3` (3-D total variation). The noise constraint is
        ∑(u − f)² ≤ Mσ², the residual variance of Eqs. 2.2c and 2.7d;
        Eqs. 2.3c and 2.6 carry an extra factor ½. There is no
        recommended preset, so `faithful=False` changes nothing.

    !!! references "References"
        1. Rudin LI, Osher S, Fatemi E. *Nonlinear total variation based
           noise removal algorithms.* Physica D **60**(1–4):259–268, 1992.
           [doi:10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F)
        2. Chambolle A, Pock T. *A first-order primal-dual algorithm for
           convex problems with applications to imaging.* J Math Imaging
           Vis **40**(1):120–145, 2011.
           [doi:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)

        ??? quote "BibTeX"
            ```bibtex
            @article{rudin1992nonlinear,
              author  = {Rudin, Leonid I. and Osher, Stanley and
                         Fatemi, Emad},
              title   = {Nonlinear total variation based noise removal
                         algorithms},
              journal = {Physica D: Nonlinear Phenomena},
              volume  = {60}, number = {1--4}, pages = {259--268},
              year    = {1992}, doi = {10.1016/0167-2789(92)90242-F}
            }
            @article{chambolle2011first,
              author  = {Chambolle, Antonin and Pock, Thomas},
              title   = {A first-order primal-dual algorithm for convex
                         problems with applications to imaging},
              journal = {Journal of Mathematical Imaging and Vision},
              volume  = {40}, number = {1}, pages = {120--145},
              year    = {2011}, doi = {10.1007/s10851-010-0251-1}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.denoise import ROF
    >>> clean = torch.zeros(48, 48)
    >>> clean[12:36, 12:36] = 100
    >>> noisy = clean + 20 * torch.randn(48, 48)
    >>> den = ROF(sigma=20).fit().apply(noisy)
    >>> bool((den - clean).square().mean() < (noisy - clean).square().mean())
    True
    """

    tags = Tags(
        targets=("denoised",),
        streaming="per-frame",
        noise="gaussian",
        reference="doi:10.1016/0167-2789(92)90242-F",
    )
    paper = {
        "sigma": None,
        "lam": None,
        "data_term": "gaussian",
        "weights": None,
        "axis_weights": None,
        "ndim": 2,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "inverse": "exact",
    }
    recommended = dict(paper)

    def __init__(
        self,
        *,
        sigma: float | None = None,
        lam: float | None = None,
        data_term: DataTerm | None = None,
        weights: Tensor | None = None,
        axis_weights: Sequence[float] | None = None,
        ndim: int | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        inverse: str | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the denoiser.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation σ of white Gaussian noise, in data units
            (constrained model). Omit it to give a `SensorEstimate` to
            `fit` instead (the stabilised data have σ = 1, and `fit`
            raises if both are given), or with `lam`.
        lam : float, optional
            Extension: weight λ > 0 of a penalised data term instead of
            the noise constraint (required by `data_term="kl"`).
        data_term : {"gaussian", "kl"}
            Data term. Default: `"gaussian"`; `"kl"` (extension) is the
            Poisson (Kullback–Leibler) likelihood, on counts or, with a
            `SensorEstimate`, on electrons.
        weights : Tensor, optional
            Extension: pixel weights ω > 0 of the data term, of shape
            `(*group, *spatial)` or broadcastable to it.
        axis_weights : sequence of float, optional
            Extension: weight wₐ > 0 of the differences along each
            spatial axis (layout order). Default: 1.
        ndim : {2, 3}
            Number of spatial axes. Default: 2; 3 is an extension.
        max_iter : int, optional
            Maximum number of solver iterations (not in the paper).
            Default: 2000.
        tol : float, optional
            Tolerance on the relative primal–dual gap (not in the paper).
            Default: 10⁻⁴.
        inverse : {"exact", "closed-form", "asymptotic", "algebraic"}
            Inverse of the Anscombe transform. Default: `"exact"`.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        o["ndim"] = _check_ndim(o["ndim"])
        if o["data_term"] not in _DATA_TERMS:
            raise ValueError(f"data_term must be one of {_DATA_TERMS}")
        if o["inverse"] not in _INVERSES:
            raise ValueError(f"Unknown inverse {o['inverse']!r}")
        if o["sigma"] is not None and o["lam"] is not None:
            raise ValueError("give sigma (constraint) or lam, not both")
        if o["data_term"] == "kl" and o["lam"] is None:
            raise ValueError("data_term='kl' needs lam")
        for name in ("sigma", "lam"):
            v = o[name]
            if v is not None and not (math.isfinite(float(v)) and v > 0):
                raise ValueError(f"{name} must be finite and > 0, got {v}")
        if (
            o["axis_weights"] is not None
            and len(o["axis_weights"]) != (o["ndim"])
        ):
            raise ValueError("axis_weights needs one entry per spatial axis")
        self._set_faithful()
        self._sensor: SensorEstimate | None = None
        self._table: tuple[Tensor, Tensor, Tensor] | None = None
        self._fitted = False

    def _set_faithful(self) -> None:
        """Faithful: the constrained Gaussian 2-D model, any σ or sensor."""
        o = self.options
        self.is_faithful = (
            o["lam"] is None
            and o["data_term"] == "gaussian"
            and o["weights"] is None
            and o["axis_weights"] is None
            and o["ndim"] == 2
        )

    # -- fit -----------------------------------------------------------------
    def fit(
        self,
        data: ArrayLike | None = None,
        *,
        layout: Layout | None = None,
        init: object = None,
        **kwargs: object,
    ) -> Self:
        """Set the noise model; the data are not read.

        Parameters
        ----------
        data : array-like, optional
            Only its layout is recorded.
        layout : Layout, optional
        init : SensorEstimate or Estimator, optional
            Noise parameters (gain α, offset μ, read variance σ²; scalars
            or maps) for Poisson-Gaussian data.
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
        sensor = self._pick(init, SensorEstimate) if init is not None else None
        if init is not None:
            self._warn_unused(init, SensorEstimate)
        o = self.options
        if sensor is not None:
            for name in ("offset", "variance", "gain"):
                if getattr(sensor, name) is None:
                    raise ValueError(f"init SensorEstimate lacks {name}")
            if o["sigma"] is not None:
                raise ValueError(
                    "sigma is not used with a SensorEstimate: the "
                    "stabilised data have σ = 1; drop sigma= or init="
                )
            if not bool((torch.as_tensor(sensor.gain) > 0).all()):
                raise ValueError("the SensorEstimate gain must be > 0")
            self.estimates = {"sensor": sensor}
        elif (
            o["data_term"] == "gaussian"
            and o["sigma"] is None
            and o["lam"] is None
        ):
            raise ValueError(
                "ROF needs sigma= (Gaussian noise), lam=, or "
                "fit(init=SensorEstimate) (Poisson-Gaussian noise)"
            )
        self._sensor = sensor
        self._table = None
        self._fitted = True

    # -- apply ---------------------------------------------------------------
    def _maps(self, group: tuple[int, ...], like: Tensor) -> tuple:
        """Gain, read standard deviation and offset at one group."""
        s = self._sensor
        nd = self.options["ndim"]
        return tuple(
            _select(t.double(), group, nd).to(like)
            for t in (s.gain, s.variance.clamp(min=0).sqrt(), s.offset)
        )

    def _denoise(self, x: Tensor, group: tuple[int, ...]) -> tuple:
        """Denoised image and the `TVResult` of the solve."""
        o = self.options
        w = o["weights"]
        if w is not None:
            w = _select(torch.as_tensor(w), group, o["ndim"]).to(x)
        kwargs = dict(
            lam=o["lam"],
            data_term=o["data_term"],
            weights=w,
            axis_weights=o["axis_weights"],
            max_iter=o["max_iter"],
            tol=o["tol"],
            warn=False,
        )
        if self._sensor is None:
            res = tv_denoise(x, sigma=o["sigma"], **kwargs)
            return res.x, res
        alpha, s, mu = self._maps(group, x)
        if o["data_term"] == "kl":
            e = (x - mu) / alpha
            res = tv_denoise(e, variance=(s / alpha).square(), **kwargs)
            return res.x * alpha + mu, res
        z = anscombe(x, alpha, s, mu)
        sigma = 1.0 if o["lam"] is None else None
        res = tv_denoise(z, sigma=sigma, **kwargs)
        d = res.x
        method = o["inverse"]
        if method == "exact" and self._table is None:
            sn = self._sensor
            s2 = sn.variance.double().clamp(min=0) / sn.gain.double().square()
            self._table = build_table(s2.reshape(-1).cpu())
        return inverse_anscombe(
            d, alpha, s, mu, method=method, table=self._table
        ), res

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if not self._fitted:
            raise RuntimeError("Call fit before apply")
        nd = self.options["ndim"]
        if not isinstance(data, Labeled):
            data = self._labeled(data, self._layout_of(data, nd))
        _, _, spatial = data.layout.positions(len(data.shape))
        if len(spatial) != nd:
            raise ValueError(
                f"ROF needs {nd} spatial axes, got {len(spatial)}"
            )
        shape = tuple(data.shape)
        dtype = self._backend(data)["dtype"]
        itemsize = torch.empty((), dtype=dtype).element_size()
        peak = _peak_memory(nd, self.options["data_term"])
        need = peak * itemsize * math.prod(shape[i] for i in spatial)
        budget = get_config().memory
        if need > budget:
            raise MemoryError(
                f"ROF solves each image whole and needs about {peak} image "
                f"sizes ({need / 2**20:.0f} MiB) of working memory, above "
                f"the memory budget ({budget / 2**20:.0f} MiB); raise it "
                "with optix.config(memory=...)"
            )
        raw = data.array
        target = out.array if isinstance(out, Labeled) else out
        if out is not None and _shares_memory(target, raw):
            raise ValueError("ROF needs a separate output (out is data)")
        out = prepare_out(out, tuple(data.shape), like=raw, dtype=self.dtype)
        chunks = blocks(
            data,
            [None] * nd,
            0,
            contiguous=True,
            align=chunks_of(out),
            dtype=self.dtype,
            device=self.device,
        )
        o = self.options
        extras = self.history.extras
        iterations = extras.setdefault("iterations", [])
        gaps = extras.setdefault("gap", [])
        stalled: list[float] = []
        start = len(iterations)
        for chunk in chunks:
            x = chunk.data
            if not x.is_floating_point():
                x = x.to(self._backend(x)["dtype"])
            u, res = self._denoise(x, chunk.group)
            iterations.append(float(res.n_iter))
            gaps.append(float(res.gap))
            if _not_converged(res, o["max_iter"], o["tol"]):
                stalled.append(res.gap)
            chunk.store(out, u[chunk.inner])
        flush(out)
        if stalled:
            n = len(iterations) - start
            _warn_max_iter(
                "ROF", stalled, n, o["max_iter"], o["tol"], stacklevel=4
            )
        return out

    def _extra_state(self) -> dict[str, object]:
        return {"fitted": torch.tensor(self._fitted)}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self._fitted = bool(state.get("fitted", False))
        self._sensor = self.estimates.get("sensor")
        self._table = None
        self._set_faithful()
