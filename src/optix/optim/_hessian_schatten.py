"""Prox of the Hessian Schatten-norm penalty, by the dual algorithm of
Lefkimmiatis, Ward & Unser (2013)."""

import math
from typing import NamedTuple

import torch
from torch import Tensor

from ._diff import Hessian
from ._prox import _schatten_p, project_schatten, prox_box, schatten_norm

__all__ = ["HessianSchatten", "HessianSchattenResult", "prox_hessian_schatten"]

#: Conjugate exponents q of p (1/p + 1/q = 1).
_DUAL = {1.0: math.inf, 2.0: 2.0, math.inf: 1.0}


class HessianSchattenResult(NamedTuple):
    """Result of [`prox_hessian_schatten`][optix.optim.prox_hessian_schatten].

    Attributes
    ----------
    x : Tensor
        The estimate x̂ = P_C(z − τ H*Ω), shaped like `z`.
    dual : Tensor
        The last dual iterate Ω, `(..., C, *N)`, in the unit
        S_q-ball at every voxel; pass it back as `dual` to warm-start.
    n_iter : int
        Number of iterations.
    """

    x: Tensor
    dual: Tensor
    n_iter: int


def _box(x: Tensor, lower: object, upper: object) -> Tensor:
    """P_C, the projection onto the box (the identity when unbounded)."""
    if lower is None and upper is None:
        return x
    return prox_box(x, lower, upper)


def _hessian(ndim: int, boundary: str, hessian: Hessian | None) -> Hessian:
    if hessian is not None:
        if hessian.ndim != ndim:
            raise ValueError(f"hessian must act on {ndim} dims")
        return hessian
    return Hessian(ndim, boundary=boundary)


def prox_hessian_schatten(
    z: Tensor,
    tau: float,
    p: float | str,
    *,
    ndim: int | None = None,
    lower: float | Tensor | None = 0.0,
    upper: float | Tensor | None = None,
    max_iter: int = 10,
    dual: Tensor | None = None,
    boundary: str = "mirror",
    hessian: Hessian | None = None,
) -> HessianSchattenResult:
    """Hessian Schatten-norm denoising: prox of τ‖H·‖₁,ₚ + ι_C.

    Solves (Lefkimmiatis, Ward & Unser 2013, Eq. 19)

        x̂ = argmin_{x ∈ C}  ½‖x − z‖₂² + τ ∑ₙ ‖[H x]ₙ‖_{S_p},

    with H the discrete Hessian ([`Hessian`][optix.optim.Hessian]; by
    default with the stencil and boundary of their Eq. 10,
    `boundary="mirror"`) and C the box [lower, upper], by their
    Algorithm 2 (`denoise`): FISTA (Beck & Teboulle 2009) on the dual
    problem (Eqs. 21–32). The mixed norm is a maximum over the dual ball
    (Lemma 1, Eqs. 21–22),

        ‖Ψ‖₁,ₚ = max_{Ω ∈ B∞,q} ⟨Ω, Ψ⟩,   1/p + 1/q = 1,

    with B∞,q the set of fields whose matrices all lie in the unit S_q
    ball. The minimiser is x̂ = P_C(z − τH*Ω̂) (Eq. 29), where Ω̂
    maximises the smooth dual objective s(Ω) (Eq. 28), whose gradient
    ∇s(Ω) = τ H P_C(z − τH*Ω) (Eq. 30) is Lipschitz with L ≤ τ²‖H‖²
    (Proposition 2: ‖H‖² ≤ 64 in 2-D). The iteration, from
    Ψ₁ = Ω₀ = `dual` (0 by default) and t₁ = 1, is

        Ωₙ = P_{B∞,q}(Ψₙ + 1/(τ‖H‖²) · H P_C(z − τH*Ψₙ))
        tₙ₊₁ = (1 + √(1 + 4tₙ²))/2
        Ψₙ₊₁ = Ωₙ + ((tₙ − 1)/tₙ₊₁)(Ωₙ − Ωₙ₋₁),

    and returns P_C(z − τH*Ω) at the last Ω. The step 1/(τ‖H‖²) uses
    the bound [`Hessian.norm2`][optix.optim.Hessian.norm2], 1/(64τ) in
    2-D as in the paper (144 in 3-D). P_{B∞,q} projects every matrix of
    the field onto the unit S_q ball
    ([`project_schatten`][optix.optim.project_schatten], Proposition 1
    and Eqs. 33–36, in closed form for 2 × 2 and 3 × 3 matrices).

    The paper runs a fixed number of iterations (10 in its deblurring
    experiments, Section IV-A; 5 inside HSPIRAL1); `max_iter` sets it.
    A dual iterate from a neighbouring problem (`dual`) warm-starts the
    iteration; the momentum restarts at t₁ = 1 (not specified in the
    paper).

    !!! note "Dimensions"
        The paper is 2-D. The algorithm is the same in 3-D, with the
        3 × 3 Hessian and ‖H‖² ≤ 144; p ∈ {1, ∞} use the closed-form
        3 × 3 eigen-decomposition of
        [`prox_schatten`][optix.optim.prox_schatten].

    !!! references "References"
        1. Lefkimmiatis S, Ward JP, Unser M. *Hessian Schatten-norm
           regularization for linear inverse problems.* IEEE Trans Image
           Process **22**(5):1873-1888, 2013.
           [doi:10.1109/TIP.2013.2237919](https://doi.org/10.1109/TIP.2013.2237919)
        2. Beck A, Teboulle M. *Fast gradient-based algorithms for
           constrained total variation image denoising and deblurring
           problems.* IEEE Trans Image Process **18**(11):2419-2434,
           2009.
           [doi:10.1109/TIP.2009.2028250](https://doi.org/10.1109/TIP.2009.2028250)

        ??? quote "BibTeX"
            ```bibtex
            @article{lefkimmiatis2013hessian,
              author  = {Lefkimmiatis, Stamatios and Ward, John Paul and
                         Unser, Michael},
              title   = {{Hessian} {Schatten}-Norm Regularization for
                         Linear Inverse Problems},
              journal = {IEEE Transactions on Image Processing},
              volume  = {22},
              number  = {5},
              pages   = {1873--1888},
              year    = {2013},
              doi     = {10.1109/TIP.2013.2237919}
            }
            @article{beck2009fast,
              author  = {Beck, Amir and Teboulle, Marc},
              title   = {Fast Gradient-Based Algorithms for Constrained
                         Total Variation Image Denoising and Deblurring
                         Problems},
              journal = {IEEE Transactions on Image Processing},
              volume  = {18},
              number  = {11},
              pages   = {2419--2434},
              year    = {2009},
              doi     = {10.1109/TIP.2009.2028250}
            }
            ```

    Parameters
    ----------
    z : Tensor
        Noisy image `(..., *N)`; the last `ndim` dims are spatial.
    tau : float
        Weight τ ≥ 0 (τ = 0 returns P_C(z)).
    p : {1, 2, inf}
        Order of the Schatten norm.
    ndim : int, optional
        Number of spatial dims. Default: `z.ndim`.
    lower, upper : float or Tensor, optional
        Bounds of the box C (None: unbounded). Default: x ≥ 0.
    max_iter : int
        Number of iterations. Default: 10 (the paper's deblurring
        experiments).
    dual : Tensor, optional
        Starting dual Ω₀ `(..., C, *N)`, e.g. the `dual` of a previous
        call. Default: 0 (the paper).
    boundary : {"mirror", "neumann", "periodic"}
        Boundary of the default Hessian. Default: `"mirror"` (Eq. 10).
    hessian : Hessian, optional
        The Hessian operator (overrides `boundary`; e.g. with axis
        weights).

    Returns
    -------
    HessianSchattenResult
    """
    if ndim is None:
        ndim = z.ndim
    ndim = int(ndim)
    p = _schatten_p(p)
    q = _DUAL[p]
    if not tau >= 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    if int(max_iter) < 0:
        raise ValueError("max_iter must be non-negative")
    hess = _hessian(ndim, boundary, hessian)
    comp = -ndim - 1
    shape = (*z.shape[:-ndim], hess.components, *z.shape[-ndim:])
    if dual is None:
        omega = z.new_zeros(shape)
    else:
        omega = dual.to(z).clone()
        if tuple(omega.shape) != shape:
            raise ValueError(
                f"dual has shape {tuple(omega.shape)}, not {shape}"
            )
    if tau == 0 or max_iter == 0:
        x = _box(z - tau * hess.adjoint(omega), lower, upper)
        return HessianSchattenResult(x, omega, 0)
    step = 1 / (tau * hess.norm2)
    t = 1.0
    psi = omega.clone()  # Ψ₁ = Ω₀
    for _ in range(int(max_iter)):
        grad = hess.forward(_box(z - tau * hess.adjoint(psi), lower, upper))
        psi.add_(grad, alpha=step)
        del grad
        new = project_schatten(psi, 1.0, q, dim=comp)  # Ωₙ
        t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
        # Ψₙ₊₁ = Ωₙ + c (Ωₙ − Ωₙ₋₁), written into the buffer of Ωₙ₋₁
        psi = omega.sub_(new).mul_(-(t - 1) / t_next).add_(new)
        omega = new
        t = t_next
    x = _box(z - tau * hess.adjoint(omega), lower, upper)
    return HessianSchattenResult(x, omega, int(max_iter))


class HessianSchatten:
    """Hessian Schatten-norm penalty with a box constraint.

        f(x) = w‖H x‖₁,ₚ + ι_C(x).

    The function f₂ of Lefkimmiatis & Unser (2013, Eq. 25), split as a
    whole in HSPIRAL1: its prox has no closed form and is computed by
    [`prox_hessian_schatten`][optix.optim.prox_hessian_schatten]
    (Algorithm 2 of Lefkimmiatis, Ward & Unser 2013) with a fixed number
    of iterations. Each call starts from the dual of the previous call
    (Lefkimmiatis & Unser 2013, Section III-C: "we initialize the
    internal variables ... with those obtained in the previous ADMM
    iteration"); here the dual Ω is kept and the momentum restarts
    (t₁ = 1, Ψ₁ = Ω; not specified in the paper). The prox is therefore
    inexact and depends on the call history: use one instance per
    problem, or `reset()`.
    """

    def __init__(
        self,
        weight: float,
        p: float | str = 1,
        *,
        ndim: int,
        lower: float | Tensor | None = 0.0,
        upper: float | Tensor | None = None,
        n_iter: int = 5,
        warm_start: bool = True,
        boundary: str = "mirror",
        hessian: Hessian | None = None,
    ) -> None:
        """Set the penalty.

        Parameters
        ----------
        weight : float
            w ≥ 0 (τ in the paper).
        p : {1, 2, inf}
            Order of the Schatten norm.
        ndim : int
            Number of spatial (trailing) dims of x.
        lower, upper : float or Tensor, optional
            Bounds of the box C. Default: x ≥ 0 (the paper).
        n_iter : int
            Iterations per prox. Default: 5 (Lefkimmiatis & Unser 2013,
            Section IV).
        warm_start : bool
            Start each prox from the previous dual. Default: True (the
            paper).
        boundary : {"mirror", "neumann", "periodic"}
            Boundary of the Hessian. Default: `"mirror"` (Lefkimmiatis,
            Ward & Unser 2013, Eq. 10).
        hessian : Hessian, optional
            The Hessian operator (overrides `boundary`).
        """
        self.weight = float(weight)
        self.p = _schatten_p(p)
        self.ndim = int(ndim)
        self.lower = lower
        self.upper = upper
        self.n_iter = int(n_iter)
        self.warm_start = bool(warm_start)
        self.hessian = _hessian(self.ndim, boundary, hessian)
        self.dual: Tensor | None = None

    def reset(self) -> None:
        """Forget the stored dual."""
        self.dual = None

    def value(self, z: Tensor) -> Tensor:
        """w‖H z‖₁,ₚ in float64, +inf outside the box."""
        zd = z.double()
        inside = True
        if self.lower is not None:
            inside = bool((zd >= self.lower).all())
        if inside and self.upper is not None:
            inside = bool((zd <= self.upper).all())
        if not inside:
            return torch.tensor(math.inf, dtype=torch.float64)
        h = self.hessian.forward(zd)
        return self.weight * schatten_norm(h, self.p, dim=-self.ndim - 1).sum()

    def prox(self, v: Tensor, t: float) -> Tensor:
        """Approximate prox of t·f by `n_iter` dual iterations."""
        dual = self.dual if self.warm_start else None
        if (
            dual is not None
            and dual.shape[-self.ndim :] != v.shape[-self.ndim :]
        ):
            dual = None
        res = prox_hessian_schatten(
            v,
            float(t) * self.weight,
            self.p,
            ndim=self.ndim,
            lower=self.lower,
            upper=self.upper,
            max_iter=self.n_iter,
            dual=dual,
            hessian=self.hessian,
        )
        self.dual = res.dual
        return res.x
