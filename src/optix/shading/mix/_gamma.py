"""Gamma tissue classes: prior, integral over the signal, class updates.

The kernels are pure tensor functions, branch-free and free of host
synchronisation, so that they can go through `torch.compile`. The
statistics they return are listed in `_model`.
"""

import math

import torch
from torch import Tensor

from ..._compile import maybe_compile
from ...distributions import Gamma, GammaStats
from ...special import gauss_hermite
from ._model import AU, BU, G1, G2, GC, GV, HCC, HVV, NSTATS, WU, XU

__all__ = [
    "GammaComponent",
    "init_gamma_classes",
    "quadrature",
    "quadrature_shared",
]


def cubic_mode(
    R0: Tensor,
    V: Tensor,
    M: Tensor,
    gain: Tensor,
    nu: Tensor,
    beta: Tensor,
    dq: float,
    polish: int = 3,
) -> Tensor:
    """Mode û of the integrand over u (in t = log u), "full" noise model.

    With c = gf, r = r₀ − f·u and σ² = v + c·u, multiplying

        l′(t) = ν − β·u + u·[f·r/σ² + ½c·(r² + δ − σ²)/σ⁴]

    by σ⁴ > 0 gives a cubic P(u) = c₃u³ + c₂u² + c₁u + c₀ with

        c₃ = −βc² − ½f²c                                  < 0
        c₂ = νc² − 2βvc − f²v − ½c²
        c₁ = 2νvc − βv² + f·r₀·v + ½c·(r₀² + δ − v)
        c₀ = νv²                                          > 0

    Since P(0) > 0 > P(+∞), it has one positive root or three. With
    three (bimodal integrand), the smallest and largest are maxima of l
    and the middle one is a minimum: the maximum with the larger l is
    kept. The roots are computed in closed form (Cardano / Viète) and
    polished by `polish` Newton steps on P.
    """
    # Every in-place operation below acts on a temporary of this function.
    # In float32, the mode loses accuracy when shot noise is negligible
    # (g·f·û / v ≲ 5·10⁻³, e.g. g ≈ 0.02 with v ≈ 10⁵): the cubic then
    # degenerates and x − a/3 cancels. Such settings are not realistic for
    # this model, and the prototype's mode is kept as is.
    tiny = torch.finfo(R0.dtype).tiny
    c = M * gain  # gm
    M2 = M.square()
    Vc = V * c
    c3 = (beta * c).add_(M2, alpha=0.5).mul_(c).neg_()  # −(βc + ½f²)·c
    c2 = (  # (ν − ½)c² − 2βvc − f²v
        c.square().mul_(nu - 0.5).sub_(Vc.mul(2 * beta)).sub_(M2.mul_(V))
    )
    c1 = (  # ½c(r₀² + δ − v) + f·r₀·v + 2νvc − βv²
        R0.square()
        .add_(dq)
        .sub_(V)
        .mul_(c)
        .mul_(0.5)
        .addcmul_(R0, M * V)
        .add_(Vc.mul_(2 * nu).sub_(V.square().mul_(beta)))
    )
    c0 = V.square().mul_(nu)  # νv²

    # Monic form u³ + a·u² + b·u + e, depressed with u = x − a/3
    a3 = c2.div(c3).div_(3)  # a/3
    b = c1.div(c3)
    e = c0.div(c3)
    p = b.addcmul(a3, a3, value=-3)  # b − a²/3
    q = (
        b.neg_().addcmul_(a3, a3, value=2).mul_(a3).add_(e)
    )  # 2a³/27 − ab/3 + e
    D = q.square().mul_(0.25).add_(p.div(3).pow_(3))  # (q/2)² + (p/3)³
    three = D <= 0

    # One real root (D > 0): Cardano
    sD = D.clamp_min_(0).sqrt_()
    s1 = q.mul(-0.5)
    s2 = s1 - sD
    s1.add_(sD)
    x1 = s1.sign().mul_(s1.abs_().pow_(1 / 3))
    x1.add_(s2.sign().mul_(s2.abs_().pow_(1 / 3)))
    # Three real roots (D ≤ 0, hence p ≤ 0): Viète
    pn = p.clamp_max_(-tiny)
    rho = pn.div(-3).sqrt_().mul_(2)
    phi = (
        q.mul_(1.5)
        .div_(pn)
        .mul_(pn.reciprocal().mul_(-3).sqrt_())
        .clamp_(-1, 1)
        .acos_()
        .div_(3)
    )
    x_hi = phi.cos().mul_(rho)
    x_lo = phi.add_(2 * math.pi / 3).cos_().mul_(rho)

    def polish_(u: Tensor) -> Tensor:
        for _ in range(polish):
            P = (c3 * u).add_(c2).mul_(u).add_(c1).mul_(u).add_(c0)
            dP = (c3 * u).mul_(3).add_(c2, alpha=2).mul_(u).add_(c1)
            u.sub_(P.div_(dP).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0))
        return u

    def log_integrand(u: Tensor) -> Tensor:
        r2 = (M * u).sub_(R0).square_()  # r²
        s2 = (c * u).add_(V)  # σ²
        return (
            u.log()
            .mul_(nu)
            .sub_(u * beta)
            .sub_(r2.add_(dq).div_(s2).mul_(0.5))
            .sub_(s2.log_().mul_(0.5))
        )

    u_hi = polish_(torch.where(three, x_hi, x1).sub_(a3)).clamp_min_(tiny)
    u_lo = polish_(torch.where(three, x_lo, x1).sub_(a3))
    use_lo = three.logical_and_(u_lo > 0)
    u_lo.clamp_min_(tiny)
    use_lo.logical_and_(log_integrand(u_lo) > log_integrand(u_hi))
    U = torch.where(use_lo, u_lo, u_hi)
    # Safety net: fall back to the prior mean
    return torch.where(U.isfinite() & (U > 0), U, nu / beta)


def quadratic_mode(
    R0: Tensor, V: Tensor, M: Tensor, nu: Tensor, beta: Tensor
) -> Tensor:
    """Mode û of the integrand over u (in t = log u), "constant" noise.

    With a fixed variance v, l′(t) = ν − β·u + u·f·(r₀ − f·u)/v = 0 is a
    quadratic in u with a single positive root, computed without
    cancellation.
    """
    A = M.square().div(V)
    B = (M * R0).div_(V).sub_(beta).neg_()
    delta = B.hypot((A * nu).mul_(4).sqrt_())
    B0 = (delta - B).div_(A).mul_(0.5)
    B1 = (delta + B).reciprocal_().mul_(nu).mul_(2)
    return torch.where(B < 0, B0, B1)


@maybe_compile
def quadrature(
    Y: Tensor,
    C: Tensor,
    V: Tensor,
    M: Tensor,
    gain: Tensor,
    mu: Tensor,
    nu: Tensor,
    *,
    dq: float,
    nodes: tuple[float, ...],
    logws: tuple[float, ...],
    full: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Integrate Gamma(u | ν, ν/μ) · N(y | c + f·u, σ²(u)) over u > 0.

    For one tissue class and one batch of frames. The integral is taken
    in t = log u by Gauss-Hermite quadrature centred on the mode t̂ of
    the integrand, with the scale given by its curvature there:
    tⱼ = t̂ + √2·s·xⱼ and

        I(h) ≈ √2·s ∑ⱼ wⱼ e^{xⱼ²} · uⱼ · p(uⱼ, y) · h(uⱼ)

    Parameters
    ----------
    Y : (n, H, W) Tensor
        Data.
    C, V, M : (H, W) Tensor
        Offset c, variance v and modulation f.
    gain : () Tensor
        Gain g.
    mu, nu : () Tensor
        Mean μ and shape ν of the Gamma class.
    dq : float
        Quantisation variance δ (every r² is r² + δ).
    nodes, logws : tuple of float
        Gauss-Hermite nodes xⱼ and log(wⱼ) + xⱼ².
    full : bool
        Noise model: σ² = v + g·f·u if True, σ² = v otherwise.

    Returns
    -------
    log_lik : (n, H, W) Tensor
        log p(y | z = class).
    EU, ElogU : (n, H, W) Tensor
        E[u | y, z] and E[log u | y, z].
    Emom : (10, n, H, W) Tensor
        Conditional expectations of the statistics listed in the module
        docstring.
    """
    beta = nu / mu  # prior rate of the Gamma distribution
    R0 = Y - C  # residuals at u = 0
    M2 = M.square()

    if full:

        def make_S2(U: Tensor) -> Tensor:
            return V.addcmul(M, U, value=gain)

    else:
        S2 = V.expand_as(Y)

        def make_S2(U: Tensor) -> Tensor:
            return S2

    def make_R(U: Tensor) -> Tensor:
        return R0.addcmul(M, U, value=-1)

    # --- mode of the integrand in t = log u
    if full:
        U = cubic_mode(R0, V, M, gain, nu, beta, dq)
    else:
        U = quadratic_mode(R0, V, M, nu, beta)
    T = U.log()

    def terms_lognormal(U: Tensor) -> tuple[Tensor, ...]:
        """Gradient G, Fisher information F and Hessian H in u."""
        R, S2 = make_R(U), make_S2(U)
        iS2 = S2.reciprocal()
        G = (M * R).mul_(iS2)  # ∂/∂u log N, mean part
        F = M2 * iS2  # Fisher information, mean part
        if not full:
            return G, F, F
        # Terms that come from the dependence of σ² on u
        tmp1 = (M * iS2).mul_(gain)  # gm/σ²
        G.addcmul_(R.square().add_(dq).sub_(S2), tmp1 * iS2, value=0.5)
        F.addcmul_(tmp1, tmp1, value=0.5)  # + ½g²f²/σ⁴
        tmp2 = R * tmp1  # gmr/σ²
        H = (  # −∂²/∂u² log N
            tmp1.square()
            .mul_(-0.5)
            .addcmul_(M2, iS2)
            .addcmul_(M * tmp2, iS2, value=2)
            .addcmul_(tmp2.square_().addcmul_(tmp1, tmp1, value=dq), iS2)
        )
        return G, F, H

    def log_integrand(T: Tensor, U: Tensor) -> Tensor:
        """l(t) = ν·t − β·u − ½ (r² + δ)/σ² − ½ log σ²   (+ const)."""
        R, S2 = make_R(U), make_S2(U)
        return (
            (nu * T)
            .addcmul_(beta, U, value=-1)
            .addcmul_(R.square_().add_(dq), S2.reciprocal(), value=-0.5)
            .sub_(S2.log(), alpha=0.5)
        )

    # Curvature of l at the mode, in t: exact where positive and not
    # smaller than the Fisher information allows, else Fisher scoring.
    G, F, H = terms_lognormal(U)
    H0 = beta.sub(G).addcmul_(U, H).mul_(U)
    F0 = U.square().mul_(F).add_(nu)
    msk = (F0 > 1.25 * nu) & (H0 > 0)
    H0 = torch.where(msk, H0, F0)
    scale = H0.reciprocal_().mul_(2).sqrt_()  # √2 · standard deviation
    L0 = log_integrand(T, U)  # l at the mode

    # --- Gauss-Hermite quadrature centred on the mode. The nodes are
    # centred on the global maximum of l, so lⱼ − l₀ ≤ 0 and the sum
    # Z = ∑ⱼ wⱼ·exp(lⱼ − l₀) cannot overflow.
    Z = torch.zeros_like(Y)
    EU = torch.zeros_like(Y)
    ElogU = torch.zeros_like(Y)
    Emom = Y.new_zeros((NSTATS, *Y.shape))
    for Xj, logWj in zip(nodes, logws, strict=True):
        Tj = T.add(scale, alpha=Xj)
        Uj = Tj.exp()
        Rj = make_R(Uj)
        Sj = make_S2(Uj)
        iSj = Sj.reciprocal()
        Lj = (
            (nu * Tj)
            .addcmul_(beta, Uj, value=-1)
            .addcmul_(Rj.square().add_(dq), iSj, value=-0.5)
            .add_(iSj.log(), alpha=0.5)
        )
        Wj = Lj.sub_(L0).add_(logWj).exp_()
        Z += Wj
        EU.addcmul_(Wj, Uj)
        ElogU.addcmul_(Wj, Tj)
        Wj *= iSj  # w / σ²
        Emom[HCC] += Wj
        Emom[GC].addcmul_(Wj, Rj)
        Emom[XU].addcmul_(Wj, Uj)
        Emom[G1].addcmul_(Wj, Uj * Rj)
        Emom[AU].addcmul_(Wj, Uj.square() if full else Uj.square_())
        Wjj = Wj.mul_(iSj).mul_(0.5)  # w / 2σ⁴
        Wq = Rj.square_().add_(dq).sub_(Sj).mul_(Wjj)  # w · q
        Emom[HVV] += Wjj
        Emom[GV] += Wq
        if full:
            Emom[WU].addcmul_(Wjj, Uj)
            Emom[G2].addcmul_(Wq, Uj)
            Emom[BU].addcmul_(Wjj, Uj.square_())

    Z = Z.clamp_min_(torch.finfo(Y.dtype).tiny)
    log_norm_gamma = beta.log() * nu - nu.lgamma()  # ν log β − log Γ(ν)
    log_norm_normal = -0.5 * math.log(2 * math.pi)
    L1 = (
        L0.add_(Z.log())
        .add_(scale.log_())
        .add_(log_norm_gamma + log_norm_normal)
    )
    EU /= Z
    ElogU /= Z
    Emom /= Z
    return L1, EU, ElogU, Emom


# -- one scene voxel seen by several tiles -----------------------------------
def gaussian_terms(
    R: Tensor, S2: Tensor, dq: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Per-observation kernel of the Gaussian likelihood.

    For an observation with residual r = y − mean and variance σ², returns

    - log p = −½ (r² + δ)/σ² − ½ log σ² (without −½ log 2π),
    - ψ₁ = ∂ log p/∂mean = r/σ², ψ₂ = ∂ log p/∂σ² = q,
    - the exact negative second derivatives h₁₁ = −∂ψ₁/∂mean,
      h₁₂ = −∂ψ₁/∂σ² (= −∂ψ₂/∂mean), h₂₂ = −∂ψ₂/∂σ²,
    - the Fisher weights f₁ = 1/σ², f₂ = 1/(2σ⁴).

    Everything else (the integral over u, its statistics and the parameter
    step) only sees these, so another observation model (e.g. a censored
    one for saturated pixels) is a `torch.where` on its outputs.
    """
    iS2 = S2.reciprocal()
    R2 = R.square().add_(dq)
    logp = (R2 * iS2).add_(S2.log()).mul_(-0.5)
    psi1 = R * iS2
    f2 = iS2.square().mul_(0.5)
    psi2 = (R2 * f2).sub_(iS2, alpha=0.5)
    h11 = iS2
    h12 = psi1 * iS2  # r/σ⁴
    h22 = (R2 * iS2).mul_(iS2).mul_(iS2).sub_(f2)  # (r² + δ)/σ⁶ − ½/σ⁴
    return logp, psi1, psi2, h11, h12, h22, f2


def _shared_terms(
    U: Tensor,
    R0: Tensor,
    V: Tensor,
    B: Tensor,
    gain: Tensor,
    nu: Tensor,
    beta: Tensor,
    dq: float,
    full: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """l(τ), and G, F, H in u summed over the tiles, at u = U `(n, h, w)`.

    `R0`, `V`, `B` are `(S, ...)`; l = ν log u − β u + ∑ₜ log pₜ.
    """
    R = R0 - B * U
    S2 = (V + gain * B * U) if full else V.expand_as(R)
    logp, psi1, psi2, h11, h12, h22, f2 = gaussian_terms(R, S2, dq)
    gb = gain * B
    G = B * psi1
    F = B.square() * h11
    if full:
        G = G + gb * psi2
        F = F + gb.square() * f2
        H = B.square() * h11 + 2 * B * gb * h12 + gb.square() * h22
    else:
        H = F
    L = (nu * U.log()).sub_(beta * U).add_(logp.sum(0))
    return L, G.sum(0), F.sum(0), H.sum(0)


def shared_mode(
    R0: Tensor,
    V: Tensor,
    B: Tensor,
    gain: Tensor,
    nu: Tensor,
    beta: Tensor,
    dq: float,
    full: bool,
    newton: int = 4,
) -> Tensor:
    """Mode û of the summed integrand over u (in τ = log u).

    `R0`, `V`, `B` are `(S, n, h, w)` (or broadcastable): residuals at
    u = 0, variances and b = g·f of the S tiles seeing each voxel.

    Constant noise: l′(τ) = ν − β·u + u·∑ₜ bₜ(r₀ₜ − bₜu)/vₜ is a quadratic
    in u, solved exactly. Full noise: the stationarity polynomial has
    degree 2S + 1, so the mode is found iteratively, with a fixed number of
    branch-free steps:

    1. candidates: the S single-tile modes (each tile with the prior,
       `cubic_mode`) and the mode of the constant-variance approximation
       with σₜ² frozen at the mean ū of the single-tile modes;
    2. keep the candidate with the largest summed l;
    3. `newton` safeguarded Newton steps in τ, of length
       l′/max(−l″, ν + u²F) clamped to ±2, each kept only if l increases.
    """
    tiny = torch.finfo(R0.dtype).tiny

    def joint_quadratic(S2: Tensor) -> Tensor:
        A = (B.square() / S2).sum(0)
        Bq = beta - (B * R0 / S2).sum(0)
        delta = Bq.hypot((A * nu).mul_(4).sqrt_())
        U0 = (delta - Bq).div_(A).mul_(0.5)
        U1 = (delta + Bq).reciprocal_().mul_(nu).mul_(2)
        U = torch.where(Bq < 0, U0, U1)
        return torch.where(U.isfinite() & (U > 0), U, nu / beta)

    if not full:
        return joint_quadratic(V).clamp_min_(tiny)

    single = cubic_mode(R0, V, B, gain, nu, beta, dq)  # (S, n, h, w)
    ubar = single.mean(0)
    cands = torch.cat([single, joint_quadratic(V + gain * B * ubar)[None]])
    cands = cands.clamp_min_(tiny)
    L = torch.stack(
        [
            _shared_terms(c, R0, V, B, gain, nu, beta, dq, full)[0]
            for c in cands.unbind(0)
        ]
    )
    L = torch.where(L.isnan(), -math.inf, L)
    U = cands.gather(0, L.argmax(0, keepdim=True))[0]
    Lu, G, F, H = _shared_terms(U, R0, V, B, gain, nu, beta, dq, full)
    for _ in range(newton):
        d1 = (U * G).add_(nu).sub_(beta * U)  # l′(τ)
        d2 = beta.sub(G).addcmul_(U, H).mul_(U)  # −l″(τ)
        F0 = U.square().mul_(F).add_(nu)
        step = (d1 / torch.maximum(d2, F0)).clamp_(-2, 2)
        step = step.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        Un = U * step.exp()
        Ln, Gn, Fn, Hn = _shared_terms(Un, R0, V, B, gain, nu, beta, dq, full)
        ok = Ln > Lu
        U = torch.where(ok, Un, U)
        Lu = torch.where(ok, Ln, Lu)
        G = torch.where(ok, Gn, G)
        F = torch.where(ok, Fn, F)
        H = torch.where(ok, Hn, H)
    return U


@maybe_compile
def quadrature_shared(
    Y: Tensor,
    C: Tensor,
    V: Tensor,
    B: Tensor,
    gain: Tensor,
    mu: Tensor,
    nu: Tensor,
    *,
    dq: float,
    nodes: tuple[float, ...],
    logws: tuple[float, ...],
    full: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Integrate Gamma(u | ν, ν/μ) · ∏ₜ N(yₜ | cₜ + bₜu, σₜ²(u)) over u.

    One scene voxel seen by S tiles shares its signal u: the integrand is
    the prior times the product of the S tile likelihoods, integrated by
    one Gauss-Hermite rule in τ = log u centred on the mode of the summed
    log integrand (`shared_mode`), with the scale from its curvature
    there (exact where positive and not below the Fisher information
    allows, else Fisher scoring, as in `quadrature`).

    Parameters
    ----------
    Y : (S, n, h, w) Tensor
        Observations of the voxels by each of their S tiles.
    C, V, B : (S, 1, h, w) or (S, n, h, w) Tensor
        Offset, variance and b = gₜ·f of each observation.
    gain, mu, nu : () Tensor
        Sensor gain γ, mean and shape of the Gamma class.
    dq, nodes, logws, full
        See `quadrature`.

    Returns
    -------
    log_lik : (n, h, w) Tensor
        log p(y₁ … y_S | z = class).
    EU, ElogU : (n, h, w) Tensor
        E[u | y, z] and E[log u | y, z].
    Emom : (10, S, n, h, w) Tensor
        Per observation, the conditional expectations of the statistics
        of `_model`, with u in place of the per-tile signal (multiply the
        u-weighted ones by gₜ, or gₜ², to express them in terms of f).
    """
    beta = nu / mu
    R0 = Y - C
    U = shared_mode(R0, V, B, gain, nu, beta, dq, full)
    T = U.log()
    L0, G, F, H = _shared_terms(U, R0, V, B, gain, nu, beta, dq, full)
    H0 = beta.sub(G).addcmul_(U, H).mul_(U)
    F0 = U.square().mul_(F).add_(nu)
    msk = (F0 > 1.25 * nu) & (H0 > 0)
    H0 = torch.where(msk, H0, F0)
    scale = H0.reciprocal_().mul_(2).sqrt_()

    # The mode is approximate, so lⱼ − l₀ may be slightly positive; it
    # stays bounded (the nodes lie within a few scales of a near-maximum).
    Z = torch.zeros_like(U)
    EU = torch.zeros_like(U)
    ElogU = torch.zeros_like(U)
    Emom = Y.new_zeros((NSTATS, *Y.shape))
    for Xj, logWj in zip(nodes, logws, strict=True):
        Tj = T.add(scale, alpha=Xj)
        Uj = Tj.exp()
        Rj = R0 - B * Uj
        Sj = (V + gain * B * Uj) if full else V.expand_as(Rj)
        logp, psi1, psi2, _, _, _, f2 = gaussian_terms(Rj, Sj, dq)
        f1 = Sj.reciprocal()
        Lj = (nu * Tj).sub_(beta * Uj).add_(logp.sum(0))
        Wj = Lj.sub_(L0).add_(logWj).exp_()
        Z += Wj
        EU.addcmul_(Wj, Uj)
        ElogU.addcmul_(Wj, Tj)
        W1 = Wj * f1
        Emom[HCC] += W1
        Emom[GC] += Wj * psi1
        Emom[XU] += W1 * Uj
        Emom[G1] += (Wj * Uj) * psi1
        Emom[AU] += W1 * Uj.square()
        W2 = Wj * f2
        Emom[HVV] += W2
        Emom[GV] += Wj * psi2
        if full:
            Emom[WU] += W2 * Uj
            Emom[G2] += (Wj * Uj) * psi2
            Emom[BU] += W2 * Uj.square()

    Z = Z.clamp_min_(torch.finfo(Y.dtype).tiny)
    nobs = Y.shape[0]
    log_norm = (
        beta.log() * nu - nu.lgamma() - 0.5 * nobs * math.log(2 * math.pi)
    )
    L1 = L0.add(Z.log()).add_(scale.log_()).add_(log_norm)
    EU /= Z
    ElogU /= Z
    Emom /= Z
    return L1, EU, ElogU, Emom


class GammaComponent:
    """Gamma tissue classes, u | z = c ~ Gamma(ν_c, rate ν_c / μ_c).

    Parameters are `(K, 2)`: the mean μ and the shape ν of each class.
    The sufficient statistics of u are (u, log u).

    Parameters
    ----------
    nodes : int
        Number of Gauss-Hermite nodes of the integral over u.
    shape_bounds : (float, float)
        Bounds on the shapes ν.
    """

    names = ("mean", "shape")
    nstats = 2
    mean_index = 0

    def __init__(self, nodes: int, shape_bounds: tuple[float, float]) -> None:
        x, logw = gauss_hermite(nodes, dtype=torch.float64, device="cpu")
        # Python floats: constants of the compiled kernel
        self.nodes = tuple(x.tolist())
        self.logws = tuple((logw + x.square()).tolist())
        self.shape_bounds = (float(shape_bounds[0]), float(shape_bounds[1]))

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
        logp, EU, ElogU, Emom = quadrature(
            Y,
            C,
            V,
            M,
            gain,
            params[0],
            params[1],
            dq=dq,
            nodes=self.nodes,
            logws=self.logws,
            full=full,
        )
        return logp, (EU, ElogU), Emom

    def loglik_shared(
        self,
        Y: Tensor,
        C: Tensor,
        V: Tensor,
        B: Tensor,
        gain: Tensor,
        params: Tensor,
        *,
        dq: float,
        full: bool,
    ) -> tuple[Tensor, tuple[Tensor, ...], Tensor]:
        """`loglik` of voxels seen by S tiles (`Y` is `(S, n, h, w)`)."""
        logp, EU, ElogU, Emom = quadrature_shared(
            Y,
            C,
            V,
            B,
            gain,
            params[0],
            params[1],
            dq=dq,
            nodes=self.nodes,
            logws=self.logws,
            full=full,
        )
        return logp, (EU, ElogU), Emom

    def fit(self, weight: Tensor, stats: Tensor, previous: Tensor) -> Tensor:
        """Gamma fit of the aggregated posterior of u in each class."""
        ok = weight > 1e-3
        safe = weight.clamp_min(1e-3)
        fit = Gamma.fit(GammaStats(safe, stats[:, 0], stats[:, 1]))
        dtype = previous.dtype
        mu = fit.mean.to(dtype).clamp_min(torch.finfo(dtype).tiny)
        nu = fit.shape.to(dtype).clamp(*self.shape_bounds)
        params = torch.stack([mu, nu], -1)
        return torch.where(ok[:, None].to(params.device), params, previous)

    def init(self, values: Tensor, nclass: int) -> tuple[Tensor, Tensor]:
        mu, nu, frac = init_gamma_classes(values, nclass)
        nu = nu.clamp(self.shape_bounds[0], min(self.shape_bounds[1], 20.0))
        return torch.stack([mu, nu], -1), frac

    def default(
        self, nclass: int, scale: Tensor, backend: dict
    ) -> tuple[Tensor, Tensor]:
        mu = scale * torch.arange(2, nclass + 2, **backend)
        nu = torch.ones(nclass, **backend)
        nu = nu.clamp(self.shape_bounds[0], min(self.shape_bounds[1], 20.0))
        frac = torch.full((nclass,), 1 / nclass, **backend)
        return torch.stack([mu, nu], -1), frac

    def clamp(self, params: Tensor) -> Tensor:
        mu = params[:, 0].clamp_min(torch.finfo(params.dtype).tiny)
        return torch.stack([mu, params[:, 1].clamp(*self.shape_bounds)], -1)

    def mean(self, params: Tensor) -> Tensor:
        return params[:, 0]


def init_gamma_classes(
    values: Tensor, nclass: int, niter: int = 50
) -> tuple[Tensor, Tensor, Tensor]:
    """Initial Gamma classes from positive samples of the signal.

    The signal is heavy-tailed (most tissue is dim, a small fraction is
    very bright), so quantiles of the values all fall in the dim bulk.
    Instead, a 1-D k-means runs on log(values), with centres initially
    spread evenly in log between the 1st and 99.9th percentiles, so that
    a bright minority can claim its own class.

    Returns
    -------
    mean, shape, fraction : (nclass,) Tensor
        Moment-matched Gamma of each class, sorted by increasing mean.
    """
    dtype, device = values.dtype, values.device
    eps = torch.finfo(dtype).eps
    tiny = torch.finfo(dtype).tiny
    logv = values.clamp_min(tiny).log_()
    if nclass == 1:
        label = torch.zeros_like(logv, dtype=torch.long)
    else:
        lims = torch.quantile(logv, logv.new_tensor([0.01, 0.999]))
        centres = torch.linspace(
            lims[0].item(), lims[1].item(), nclass, dtype=dtype, device=device
        )
        for _ in range(niter):
            label = (logv[:, None] - centres).abs_().argmin(1)
            count = torch.bincount(label, minlength=nclass).to(dtype)
            total = torch.zeros_like(centres).index_add_(0, label, logv)
            new = torch.where(count > 0, total / count.clamp_min(1), centres)
            if torch.allclose(new, centres):
                break
            centres = new
        label = (logv[:, None] - centres).abs_().argmin(1)
    count = torch.bincount(label, minlength=nclass).to(dtype)
    zeros = torch.zeros(nclass, dtype=dtype, device=device)
    total = zeros.index_add(0, label, values)
    total2 = zeros.index_add(0, label, values.square())
    n = count.clamp_min(1)
    mu = total / n
    mu2 = mu.square()
    var = (total2 / n).sub_(mu2).clamp_min_(eps * mu2)
    nu = torch.where(count > 1, mu2 / var, torch.ones_like(mu))
    # Empty classes: put them above the others
    above = values.max() * (
        1 + torch.arange(nclass, dtype=dtype, device=device)
    )
    mu = torch.where(count > 0, mu, above)
    frac = (count + 1e-3) / (count.sum() + 1e-3 * nclass)
    order = mu.argsort()
    return mu[order], nu[order], frac[order]
