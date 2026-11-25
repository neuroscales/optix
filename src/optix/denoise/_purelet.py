"""PURE-LET denoising of mixed Poisson-Gaussian noise (Luisier et al. 2011).

Written from the paper; the authors' code was not used.
"""

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Self

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
    as_tensor,
    blocks,
    chunks_of,
    flush,
    prepare_out,
)

__all__ = [
    "PURELET",
    "PureSums",
    "haar_uwt",
    "let_basis",
    "let_threshold",
    "pure_let",
    "pure_let_halo",
]

_R = 2.0**-0.5  # unit-norm Haar filters (1, ±1)/√2


# -- checks ------------------------------------------------------------------
def _check_levels(levels: object) -> int:
    if isinstance(levels, bool) or int(levels) != levels or levels < 1:
        raise ValueError(f"levels must be an integer >= 1, got {levels!r}")
    return int(levels)


def _check_ndim(ndim: object) -> int:
    if isinstance(ndim, bool) or int(ndim) != ndim or ndim < 1:
        raise ValueError(f"ndim must be an integer >= 1, got {ndim!r}")
    return int(ndim)


def _check_size(spatial: Sequence[int], levels: int) -> None:
    """Every transformed axis needs at least 2^J samples (Section III-B)."""
    n = min(int(x) for x in spatial)
    if n >= 2**levels:
        return
    best = n.bit_length() - 1  # largest J with 2^J <= n
    hint = (
        f"use levels <= {best}"
        if best >= 1
        else "every axis needs at least 2 samples"
    )
    raise ValueError(
        f"levels={levels} needs at least 2^levels = {2**levels} samples "
        f"along every transformed axis, got {tuple(spatial)}: the closed "
        "forms of PURE assume that the filters of the coarsest level do "
        f"not wrap around the image; {hint}"
    )


def _check_nonneg(name: str, value: object) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a number >= 0, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return v


def _check_positive(name: str, value: object) -> float:
    v = _check_nonneg(name, value)
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return v


# -- transform ---------------------------------------------------------------
def _split(x: Tensor, ndim: int, s: int) -> Tensor:
    """All 2^d channels of one à trous Haar level, valid part only.

    `x` is `(*batch, *spatial)`; the result is `(2^d, *batch, *spatial′)`
    with every spatial extent reduced by `s`. Channel `c` has, along the
    i-th spatial axis, the high-pass filter if bit i of `c` is set and
    the low-pass filter otherwise; channel 0 is the low-pass one.
    """
    out = x.unsqueeze(0)
    for i in range(ndim):
        dim = out.ndim - ndim + i
        n = out.shape[dim] - s
        a, b = out.narrow(dim, 0, n), out.narrow(dim, s, n)
        out = torch.cat([(a + b) * _R, (a - b) * _R], 0)
    return out


def _lowpass_adjoint(u: Tensor, ndim: int, shifts: Sequence[int]) -> Tensor:
    """Adjoint of the low-pass cascade (full convolution, zero outside)."""
    for s in shifts:
        u = _adjoint_step(u, ndim, s, None)
    return u


def _adjoint_axis(u: Tensor, dim: int, s: int, sign: Tensor | None) -> Tensor:
    """Adjoint of one 1-D Haar step of `_split` along `dim` (zero outside).

    `sign` holds ±1 per leading channel (broadcast to `u`; −1 for the
    high-pass filter (1, −1)/√2); None means low-pass.
    """
    n = u.shape[dim]
    shape = list(u.shape)
    shape[dim] = n + s
    out = u.new_empty(shape)
    out.narrow(dim, n, s).zero_()
    out.narrow(dim, 0, n).copy_(u)
    late = out.narrow(dim, s, n)
    if sign is None:
        late.add_(u)
    else:
        late.addcmul_(u, sign)
    return out.mul_(_R)


def _axis_sign(signs: Tensor | None, i: int, u: Tensor) -> Tensor | None:
    if signs is None:
        return None
    return signs[:, i].to(u).reshape((-1,) + (1,) * (u.ndim - 1))


def _adjoint_step(
    u: Tensor, ndim: int, s: int, signs: Tensor | None
) -> Tensor:
    """Adjoint of `_split` for one channel per leading entry.

    `u` is `(C, *batch, *spatial)` and `signs` `(C, ndim)` holds ±1 (the
    high-pass filter is (1, −1)/√2); None means all low-pass.
    """
    for i in range(ndim):
        u = _adjoint_axis(u, u.ndim - ndim + i, s, _axis_sign(signs, i, u))
    return u


def _channel_signs(ndim: int) -> Tensor:
    """±1 per band (channels 1 … 2^d − 1 of `_split`) and axis."""
    c = torch.arange(1, 2**ndim)
    bits = torch.stack([(c >> i) & 1 for i in range(ndim)], dim=1)
    return 1 - 2 * bits.to(torch.float64)


def haar_uwt(
    data: Tensor, levels: int, *, ndim: int = 2
) -> list[tuple[Tensor, Tensor]]:
    """Undecimated separable Haar wavelet transform, valid part.

    The J-band undecimated filterbank of Luisier et al. (2011, Fig. 3)
    with the unit-norm Haar filters H(z) = (1 + z⁻¹)/√2 and
    G(z) = (1 − z⁻¹)/√2, applied separably along the `ndim` trailing
    axes (Section III: "separable filters"). At level j = 1 … J, with
    s = 2^(j−1), the analysis filters are

        H̄_j(z) = H(z)·H(z²)⋯H(z^s)          (scaling, w̄ = D̄_j y)
        G̃_j(z) = H(z)·H(z²)⋯H(z^(s/2))·G(z^s)   (wavelet, w = D_j y)

    in each 1-D direction; in d dimensions a level has 2^d − 1 wavelet
    bands (at least one direction high-pass). For the Haar filters the
    group-delay compensation filter of Eq. (17) is Q(z) = 1
    (H(z⁻¹) = z·H(z), n₀ = 1): the scaling coefficients of a level are
    aligned with its wavelet coefficients without further filtering. Each
    coefficient at index l depends on the samples l … l + 2^j − 1 along
    every axis; only coefficients with their whole support inside `data`
    are returned (no boundary extension: pad `data` first, e.g.
    periodically).

    Parameters
    ----------
    data : Tensor
        `(*batch, *spatial)` with `ndim` spatial axes.
    levels : int
        Number of levels J.
    ndim : int
        Number of transformed (trailing) axes d.

    Returns
    -------
    list of (Tensor, Tensor)
        Per level j: the wavelet bands `(2^d − 1, *batch, *spatial_j)` and
        the scaling coefficients `(*batch, *spatial_j)`, where each
        spatial extent is reduced by 2^j − 1.
    """
    levels, ndim = _check_levels(levels), _check_ndim(ndim)
    out = []
    low = data
    for j in range(1, levels + 1):
        c = _split(low, ndim, 2 ** (j - 1))
        low = c[0]
        out.append((c[1:], low))
    return out


# -- thresholding function ---------------------------------------------------
_RCOND = 1e-10  # relative eigenvalue cutoff of the PURE system
_U_MAX = 2.0  # |w|/(3t) beyond which exp(−(w/3t)^8) < e^(−256)


def let_threshold(
    w: Tensor,
    wbar: Tensor,
    beta: float,
    variance: Tensor | float,
    sharpness: float = 100.0,
    *,
    derivatives: bool = True,
) -> tuple[Tensor, ...]:
    """Nonlinear LET function of Eq. (20) and its partial derivatives.

    θ(w, w̄) = w·exp(−(w/(3·t(w̄)))⁸), with the signal-dependent threshold
    of Eq. (19),

        t(w̄) = √(β·|w̄| + σ²),

    where |w̄| is replaced by the smooth tanh(κ·w̄)·w̄ (Section III-C,
    where it is called k; κ = `sharpness` = 100) so that the derivatives
    of Corollary 1 exist. Returns θ and Θ₁ = ∂θ/∂w, Θ₂ = ∂θ/∂w̄,
    Θ₁₁ = ∂²θ/∂w², Θ₂₂ = ∂²θ/∂w̄², Θ₁₂ = ∂²θ/∂w∂w̄ (the vectors of
    Corollary 1), in closed form. With u = w/(3t) and E = exp(−u⁸):

        Θ₁ = E·(1 − 8u⁸),     Θ₁₁ = −(8u⁷/(3t))·E·(9 − 8u⁸),
        Θ₂ = (8w·u⁸·E/t)·t′,  Θ₁₂ = (8u⁸·E/t)·(9 − 8u⁸)·t′,
        Θ₂₂ = (8w·u⁸·E/t²)·(8u⁸ − 9)·t′² + (8w·u⁸·E/t)·t″,

    with t′ = β·m′/(2t), t″ = β·m″/(2t) − β²·m′²/(4t³) and
    m(w̄) = tanh(κ·w̄)·w̄.

    !!! note "Choices not specified in the paper"
        * t² is floored at the machine epsilon of the dtype (t = 0 occurs
          for pure Poisson noise on a zero background).
        * u is clipped to |u| ≤ 2, where E < e⁻²⁵⁶: θ and its derivatives
          are 0 to working precision there, and the clip avoids overflow.

    Parameters
    ----------
    w : Tensor
        Wavelet coefficients.
    wbar : Tensor
        Scaling coefficients of the same level (broadcast to `w`).
    beta : float
        Scale factor β_j = M_j^(−1/2) that turns w̄ into a local mean.
    variance : Tensor or float
        Gaussian variance σ² of the coefficients (a map for per-pixel
        noise parameters).
    sharpness : float
        κ of the smooth absolute value tanh(κ·w̄)·w̄.
    derivatives : bool
        Whether to compute the five derivatives; θ alone otherwise.

    Returns
    -------
    tuple of Tensor
        θ, Θ₁, Θ₂, Θ₁₁, Θ₂₂, Θ₁₂; `(θ,)` if `derivatives` is False.
    """
    kappa = sharpness
    th = torch.tanh(kappa * wbar)
    q = (beta * th * wbar + variance).clamp_min(torch.finfo(w.dtype).eps)
    t = q.sqrt()
    u = (w / (3 * t)).clamp_(-_U_MAX, _U_MAX)
    a = u.square()
    a.square_().square_()  # u⁸
    e = a.neg().exp_()
    theta = w * e
    if not derivatives:
        return (theta,)
    # derivatives of t(w̄): low-pass sized
    sech2 = 1 - th * th
    dm = th + kappa * wbar * sech2
    d2m = 2 * kappa * sech2 * (1 - kappa * wbar * th)
    dt = beta * dm / (2 * t)
    d2t = beta * d2m / (2 * t) - beta * beta * dm * dm / (4 * t * q)
    del sech2, dm, d2m, th
    # band-sized terms, with in-place updates to limit the temporaries
    a.mul_(8)  # 8u⁸
    d1 = torch.rsub(a, 1).mul_(e)  # E·(1 − 8u⁸)
    b = torch.rsub(a, 9)  # 9 − 8u⁸
    d11 = u.pow_(7).mul_(e).mul_(b).mul_(-8 / (3 * t))
    ge = a.mul_(e).div_(t)  # 8u⁸·E/t
    d12 = torch.mul(ge, b).mul_(dt)
    g = ge.mul_(w)  # ∂θ/∂t = 8w·u⁸·E/t
    d2 = g * dt
    d22 = g.mul_(b.mul_(-(dt * dt) / t).add_(d2t))
    return theta, d1, d2, d11, d22, d12


# -- LET basis and PURE terms ------------------------------------------------
def pure_let_halo(levels: int = 4) -> int:
    """Halo, in pixels, that makes block-wise PURE-LET exact.

    A function f_{j,k} of Eq. (21) at pixel n depends on the samples
    n − 2^j + 1 … n + 2^j − 1 along each axis (analysis by D_j and D̄_j,
    then reconstruction by R_j, filters of length 2^j), hence a halo of
    2^J − 1 pixels for J levels.

    Parameters
    ----------
    levels : int
        Number of levels J.

    Returns
    -------
    int
    """
    return 2 ** _check_levels(levels) - 1


def _crop(x: Tensor, ndim: int, start: int, sizes: Sequence[int]) -> Tensor:
    for i, n in enumerate(sizes):
        x = x.narrow(x.ndim - ndim + i, start, n)
    return x


def let_basis(
    data: Tensor,
    variance: Tensor | float,
    levels: int,
    *,
    ndim: int = 2,
    sharpness: float = 100.0,
    halo: int | None = None,
    derivatives: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """The LET functions f_{j,k}(y), the low-pass term and their PURE terms.

    Eq. (21): f(y) = Σ_j Σ_k a_{j,k}·f_{j,k}(y) + R_{J+1}·D_{J+1}·y with,
    for each wavelet band j of the undecimated Haar filterbank
    ([`haar_uwt`][optix.denoise.utils.haar_uwt]),

        f_{j,1}(y) = R_j·D_j·y,       f_{j,2}(y) = R_j·θ_j(D_j·y, D̄_j·y)

    (θ_{j,1}(w, w̄) = w and θ_{j,2} of Eq. (20), see
    [`let_threshold`][optix.denoise.utils.let_threshold]). The
    reconstruction is R_j = M_j⁻¹·D_jᵀ, M_j = 2^(d·j) (so that
    R·D = I and diag{D_j·R_j} = 1/M_j, Section III-B), with
    R_{J+1} = M_J⁻¹·D̄_Jᵀ for the low-pass residual, which is not
    thresholded.

    For each function, the derivative terms of the PURE of Eq. (6) are
    returned as one number, the term of ĉ in Eq. (22),

        δ_k = yᵀ·∂f_k(y) + Σ_n σ_n²·(∂f_{k,n} − ∂²f_{k,n})(y),

    where ∂f (∂²f) holds the first (second) derivatives ∂f_n/∂y_n
    (∂²f_n/∂y_n²). With a constant σ² the second sum is
    σ²·div{f_k − ∂f_k}, Eqs. (12)-(13). They are evaluated with the
    closed forms of Eqs. (11)-(13), in which, for the unit-norm Haar
    filters and periodic boundary conditions, the transform-dependent
    terms reduce to (v = y or the variance map σ², per coefficient l of
    band j)

        (D∙Rᵀ)v = M⁻³ᐟ²·D̄v,          (D̄∙Rᵀ)v = M⁻³ᐟ²·Dv,
        ((D∙D)∙Rᵀ)v = ((D̄∙D̄)∙Rᵀ)v = M⁻²·Dv,   ((D∙D̄)∙Rᵀ)v = M⁻²·D̄v,

    because d_{l,n}² = M⁻¹ on the support of the band's filter, the
    scaling filter is constant there (M⁻¹ᐟ²) and the wavelet filters have
    zero mean. This needs the 2^j taps of each filter to fall on distinct
    pixels, i.e. an image with at least 2^J samples along every
    transformed axis: on a smaller image the circulant D_j of Eq. (16)
    folds several taps onto one pixel and δ below is not the derivative
    term of PURE. Hence

        δ_k = M⁻³ᐟ²·Σ_l [Θ₁·D̄(y + σ²) + Θ₂·D(y + σ²)]
              − M⁻²·Σ_l [(Θ₁₁ + Θ₂₂)·Dσ² + 2·Θ₁₂·D̄σ²],

    which for a constant σ² is Eqs. (11)-(13) with diag{DR} = 1/M,
    diag{D̄R} = diag{(D∙D)R} = diag{(D̄∙D̄)R} = 0 and
    diag{(D∙D̄)R} = M⁻³ᐟ². The Gaussian variance of the threshold of
    Eq. (19) is that of the coefficient, β_j·D̄σ² (σ² for a constant
    variance).

    `data` holds a block with a halo of `halo` ≥ 2^J − 1 samples on each
    side of every transformed axis (pad the whole image periodically,
    the boundary condition of Section III-B): the functions are returned
    on the core, and δ sums over the coefficients l of the core, so that
    sums over blocks partitioning an image are the whole-image sums.

    Parameters
    ----------
    data : Tensor
        Normalised data y (Poisson counts plus Gaussian noise),
        `(*batch, *spatial)`, core plus halo.
    variance : Tensor or float
        Gaussian variance σ² in the units of y: a scalar, or a map of the
        shape of `data` (per-pixel noise parameters).
    levels : int
        Number of levels J.
    ndim : int
        Number of transformed (trailing) axes d.
    sharpness : float
        κ of the smooth absolute value tanh(κ·w̄)·w̄ (Section III-C,
        where it is called k).
    halo : int, optional
        Halo of `data` (default: 2^J − 1).
    derivatives : bool
        Whether to compute δ.

    Returns
    -------
    functions : Tensor
        f_{j,k}(y) on the core, `(K, *batch, *core)`, K = 2·(2^d − 1)·J,
        ordered by level, band, then k.
    lowpass : Tensor
        R_{J+1}·D_{J+1}·y on the core.
    delta : Tensor or None
        δ_k, `(K, *batch)`.
    """
    levels, ndim = _check_levels(levels), _check_ndim(ndim)
    h = pure_let_halo(levels) if halo is None else int(halo)
    if h < pure_let_halo(levels):
        raise ValueError(f"halo must be >= 2^levels - 1, got {h}")
    spatial = data.shape[-ndim:]
    core = [n - 2 * h for n in spatial]
    if min(core) < 1:
        raise ValueError(f"data of shape {tuple(data.shape)} has no core")
    smap = isinstance(variance, Tensor) and variance.ndim > 0
    if isinstance(variance, Tensor) and not smap:
        variance = float(variance)
    nb = 2**ndim - 1
    signs = _channel_signs(ndim).repeat_interleave(2, dim=0)
    sums = tuple(range(-ndim, 0))
    funcs = data.new_empty((2 * nb * levels, *data.shape[:-ndim], *core))
    deltas = []
    low, low_s = data, variance if smap else None
    for j in range(1, levels + 1):
        s = 2 ** (j - 1)
        m = 2.0 ** (ndim * j)
        beta = m**-0.5
        # coef[:, 0] holds the bands w, coef[:, 1] their θ(w, w̄)
        c = _split(low, ndim, s)
        coef = c.new_empty((nb, 2, *c.shape[1:]))
        coef[:, 0] = c[1:]
        low = c[0].clone()
        del c
        if smap:
            cs = _split(low_s, ndim, s)
            bands_s, low_s = cs[1:], cs[0]
        var = beta * low_s if smap else variance
        if derivatives:
            lo = _crop(low, ndim, h, core)
            if smap:
                bs = _crop(bands_s, ndim, h, core)
                ls = _crop(low_s, ndim, h, core)
            else:
                ls = variance * m**0.5  # D̄σ² for a constant σ²
            lys = lo + ls
            lin = lys.sum(sums).expand(nb, *lys.shape[:-ndim])
            nonlin = lin.new_empty(lin.shape)
        # one band at a time, to bound the temporaries of the threshold
        for i in range(nb):
            th, *dth = let_threshold(
                coef[i, 0], low, beta, var, sharpness, derivatives=derivatives
            )
            coef[i, 1] = th
            del th
            if not derivatives:
                continue
            # δ: sums over the coefficients l of the core
            d1, d2, d11, d22, d12 = (_crop(t, ndim, h, core) for t in dth)
            del dth
            bys = _crop(coef[i, 0], ndim, h, core)
            if smap:
                bys = bys + bs[i]
            t1 = (d1.mul_(lys) + d2.mul_(bys)).sum(sums)
            if smap:
                t2 = (d11.add_(d22).mul_(bs[i]) + d12.mul_(2 * ls)).sum(sums)
            else:
                t2 = 2 * ls * d12.sum(sums)
            del d1, d2, d11, d22, d12, bys
            nonlin[i] = t1 * m**-1.5 - t2 / (m * m)
        if derivatives:
            deltas.append(
                torch.stack([lin * m**-1.5, nonlin], 1).flatten(0, 1)
            )
            del lys
        # R_j: one axis at a time, each step freeing its input
        f = coef.flatten(0, 1)
        del coef
        for i in range(ndim):
            f = _adjoint_axis(f, f.ndim - ndim + i, s, _axis_sign(signs, i, f))
        for t in range(j - 2, -1, -1):
            for i in range(ndim):
                f = _adjoint_axis(f, f.ndim - ndim + i, 2**t, None)
        torch.div(
            _crop(f, ndim, h, core), m, out=funcs[2 * nb * (j - 1) :][: 2 * nb]
        )
        del f
    shifts = [2**i for i in range(levels - 1, -1, -1)]
    lowpass = _lowpass_adjoint(low.unsqueeze(0), ndim, shifts)[0]
    lowpass = _crop(lowpass, ndim, h, core) / 2.0 ** (ndim * levels)
    delta = torch.cat(deltas, 0) if derivatives else None
    return funcs, lowpass, delta


def _function_levels(levels: int, ndim: int) -> Tensor:
    """Level j of each LET function, in the order of `let_basis`."""
    per = 2 * (2**ndim - 1)
    return torch.arange(1, levels + 1).repeat_interleave(per)


@dataclass
class PureSums:
    """Sufficient statistics of PURE for a LET, mergeable over blocks.

    With f(y) = Σ_k a_k·f_k(y) + f_low(y) (Eq. (21)), the PURE estimate of
    the MSE in the Taylor form of Eq. (6), generalised to a per-pixel
    Gaussian variance σ_n² (Stein's lemma applies per pixel),

        ε̂ = (1/N)·(‖f − y‖² − 1ᵀy − Σσ_n²)
            + (2/N)·(yᵀ∂f + Σ_n σ_n²·(∂f_n − ∂²f_n)),

    is quadratic in a:

        N·ε̂(a) = aᵀ·G·a − 2·aᵀ·(b − δ) + ‖f_low − y‖² − 1ᵀy − Σσ_n²
                 + 2·M_J⁻¹·Σ_n (y_n + σ_n²),

    with G = [f_kᵀ f_l] (the matrix M of Eq. (22)), b = [f_kᵀ(y − f_low)]
    and δ_k of [`let_basis`][optix.denoise.utils.let_basis] (f_low is
    linear: ∂f_low = M_J⁻¹·1, ∂²f_low = 0). Its minimiser solves the
    linear system of Eq. (22), G·a = ĉ = b − δ. All terms are sums over
    pixels or coefficients, accumulated in float64.

    Attributes
    ----------
    gram, cross, delta : Tensor
        G `(K, K)`, b and δ `(K,)`.
    resid : float
        ‖f_low − y‖².
    sum_y, sum_y2, sum_s : float
        Σy, Σy² and Σσ_n².
    count : int
        Number of pixels N.
    levels, ndim : int
        J and d.
    """

    gram: Tensor
    cross: Tensor
    delta: Tensor
    resid: float
    sum_y: float
    sum_y2: float
    sum_s: float
    count: int
    levels: int
    ndim: int

    @classmethod
    def from_basis(
        cls,
        functions: Tensor,
        lowpass: Tensor,
        delta: Tensor,
        data: Tensor,
        variance: Tensor | float,
        *,
        levels: int,
        ndim: int = 2,
    ) -> "PureSums":
        """Statistics of one block (one image or volume, core only).

        Parameters
        ----------
        functions, lowpass, delta : Tensor
            From [`let_basis`][optix.denoise.utils.let_basis] (no batch
            dims).
        data : Tensor
            y on the core.
        variance : Tensor or float
            σ² on the core, or a constant.
        levels, ndim : int
            J and d.

        Returns
        -------
        PureSums
        """
        k = functions.shape[0]
        f = functions.reshape(k, -1)
        y = data.reshape(-1).double()
        low = lowpass.reshape(-1).double()
        r = y - low
        n = y.numel()
        # float64 sums over pixel chunks: a float64 copy of f of at most
        # the size of one image
        gram = torch.zeros(k, k, dtype=torch.float64, device=f.device)
        cross = torch.zeros(k, dtype=torch.float64, device=f.device)
        step = -(-n // k)
        for i in range(0, n, step):
            fi = f[:, i : i + step].double()
            gram += fi @ fi.T
            cross += fi @ r[i : i + step]
        del fi
        if isinstance(variance, Tensor) and variance.ndim > 0:
            sum_s = float(variance.double().sum())
        else:
            sum_s = float(variance) * n
        return cls(
            gram=gram,
            cross=cross,
            delta=delta.double().reshape(k),
            resid=float(r.square().sum()),
            sum_y=float(y.sum()),
            sum_y2=float(y.square().sum()),
            sum_s=sum_s,
            count=n,
            levels=levels,
            ndim=ndim,
        )

    def __add__(self, other: "PureSums") -> "PureSums":
        if (self.levels, self.ndim) != (other.levels, other.ndim):
            raise ValueError("Cannot merge sums of different transforms")
        return PureSums(
            self.gram + other.gram.to(self.gram),
            self.cross + other.cross.to(self.cross),
            self.delta + other.delta.to(self.delta),
            self.resid + other.resid,
            self.sum_y + other.sum_y,
            self.sum_y2 + other.sum_y2,
            self.sum_s + other.sum_s,
            self.count + other.count,
            self.levels,
            self.ndim,
        )

    @property
    def energy(self) -> float:
        """E_mean = (‖y‖² − 1ᵀy)/N − σ², an unbiased estimate of ‖x‖²/N."""
        return (self.sum_y2 - self.sum_y - self.sum_s) / self.count

    def active(self, reliability: float = 10.0) -> Tensor:
        """Functions kept in the LET (Section III-D).

        f_{j,1} is always kept; f_{j,2} only when T_j = M_j·E_mean > T
        (T = `reliability`, 10 in the paper), where the first-order Taylor
        approximation of PURE is reliable.

        Parameters
        ----------
        reliability : float
            T.

        Returns
        -------
        Tensor
            Boolean `(K,)`.
        """
        j = _function_levels(self.levels, self.ndim).double()
        kind2 = torch.arange(j.numel()) % 2 == 1
        tj = 2.0 ** (self.ndim * j) * self.energy
        return (~kind2) | (tj > reliability)

    def risk(self, coefficients: Tensor) -> float:
        """PURE ε̂(a) for the LET coefficients a (zero for unused ones).

        Parameters
        ----------
        coefficients : Tensor
            a, `(K,)`.

        Returns
        -------
        float
        """
        a = coefficients.to(self.gram)
        mj = 2.0 ** (self.ndim * self.levels)
        total = (
            a @ self.gram @ a
            - 2 * a @ (self.cross - self.delta)
            + self.resid
            - self.sum_y
            - self.sum_s
            + 2 * (self.sum_y + self.sum_s) / mj
        )
        return float(total) / self.count

    def solve(self, reliability: float = 10.0) -> Tensor:
        """LET coefficients minimising PURE: G·a = b − δ, Eq. (22).

        Restricted to the [`active`][optix.denoise.utils.PureSums.active]
        functions; the others get a = 0. Not specified in the paper: the
        system is scaled to a unit diagonal and solved by its
        eigendecomposition, discarding eigenvalues below 10⁻¹⁰ times the
        largest. This is the exact solution of a well-conditioned
        system, and the minimum-norm minimiser of PURE when some f_k are
        (numerically) linearly dependent, as f_k = 0 for constant data.
        The sums must come from an image with at least 2^J samples along
        every transformed axis (as
        [`pure_let`][optix.denoise.pure_let] and
        [`PURELET`][optix.denoise.PURELET] require): on a smaller image
        δ is not the derivative term of PURE.

        Parameters
        ----------
        reliability : float
            T of Section III-D.

        Returns
        -------
        Tensor
            a, `(K,)`, float64.
        """
        keep = self.active(reliability).to(self.gram.device)
        g = self.gram[keep][:, keep]
        c = (self.cross - self.delta)[keep]
        a = torch.zeros_like(self.cross)
        norm = g.diagonal().clamp_min(0).sqrt()
        scale = torch.where(norm > 0, 1 / norm, torch.zeros_like(norm))
        lam, vec = torch.linalg.eigh(scale[:, None] * g * scale)
        cut = _RCOND * lam.abs().max().clamp_min(torch.finfo(lam.dtype).tiny)
        inv = torch.where(lam > cut, 1 / lam, torch.zeros_like(lam))
        sol = scale * (vec @ (inv * (vec.T @ (scale * c))))
        a[keep] = sol.to(a)
        return a


def _combine(functions: Tensor, lowpass: Tensor, a: Tensor) -> Tensor:
    """f = Σ_k a_k·f_k + f_low."""
    return torch.tensordot(a.to(functions), functions, 1).add_(lowpass)


# -- functional API ----------------------------------------------------------
def _wrap(x: Tensor, ndim: int, h: int) -> Tensor:
    """Periodic extension by `h` samples on both sides of each axis."""
    for i in range(ndim):
        dim = x.ndim - ndim + i
        n = x.shape[dim]
        idx = torch.arange(-h, n + h, device=x.device) % n
        x = x.index_select(dim, idx)
    return x


def _as_map(
    value: Tensor | float, spatial: Sequence[int], like: Tensor
) -> Tensor | float:
    if isinstance(value, Tensor):
        if value.ndim == 0 or value.numel() == 1:
            return float(value.reshape(()))
        return value.to(like).broadcast_to(tuple(spatial))
    return float(value)


def pure_let(
    data: Tensor,
    sigma: Tensor | float = 0.0,
    *,
    gain: Tensor | float = 1.0,
    offset: Tensor | float = 0.0,
    levels: int = 4,
    ndim: int = 2,
    reliability: float = 10.0,
    sharpness: float = 100.0,
) -> Tensor:
    """PURE-LET denoising of mixed Poisson-Gaussian noise.

    Model (Luisier et al. 2011, Eq. (1) and Section V): the data are
    `offset + gain·y`, y = z + b, z ~ 𝒫(x), b ~ 𝒩(0, σ²/gain²), with x
    the noise-free intensity in photons. With the paper's notation
    α = `gain`, δ = `offset` and σ² in units of y. Each image is
    normalised, y = (data − δ)/α, denoised in the unit-norm undecimated
    Haar wavelet domain (UWT PURE-LET, Section III) and mapped back,
    α·x̂ + δ:

    1. **Transform.** J levels of the separable undecimated Haar
       filterbank with periodic boundary conditions
       ([`haar_uwt`][optix.denoise.utils.haar_uwt]); w = D_j·y are the
       wavelet coefficients of band j, w̄ = D̄_j·y the scaling
       coefficients of the same level (no group-delay compensation is
       needed for Haar, Section III-A).
    2. **Thresholding (LET).** Eq. (20), per band,
       θ_j(w, w̄) = a_{j,1}·w + a_{j,2}·w·exp(−(w/(3·t_j(w̄)))⁸) with
       t_j(w̄) = √(β_j·|w̄| + σ²), β_j = M_j^(−1/2) (Eq. (19));
       the low-pass residual is kept: Eq. (21),
       f(y) = Σ_j Σ_k a_{j,k}·R_j·θ_{j,k}(D_j·y, D̄_j·y) + R_{J+1}·D_{J+1}·y.
    3. **PURE.** The coefficients a minimise the first-order Taylor
       approximation of the PURE of Theorem 1, Eq. (6), an unbiased
       estimate of the MSE, which is quadratic in a: a solves the K × K
       linear system of Eq. (22), M·a = ĉ, K = 2·(2^d − 1)·J (one system
       per image, no iterations; see
       [`PureSums`][optix.denoise.utils.PureSums]).
    4. **Reliability (Section III-D).** f_{j,2} is dropped (a_{j,2} = 0)
       when T_j = M_j·E_mean ≤ T, E_mean = (‖y‖² − 1ᵀy)/N − σ², T = 10.

    In d dimensions a level has 2^d − 1 bands, each with its own
    (a_{j,1}, a_{j,2}); M_j = 2^(d·j) is the redundancy factor of level
    j (2^j in the 1-D derivation of the paper) and β_j = M_j^(−1/2).

    !!! note "Choices not specified in the paper"
        * The number of levels J (default 4; the paper does not give it,
          and its Fig. 6 shows the levels 1 and 3). The result at a pixel
          depends on the data within 2^J − 1 pixels. Every transformed
          axis needs at least 2^J samples: the closed forms of PURE
          (Section III-B) assume that the filters of the coarsest level
          do not wrap around the image, and a `ValueError` naming the
          largest valid J is raised otherwise.
        * Each of the 2^d − 1 orientation bands of a level is a band of
          the filterbank with its own LET coefficients.
        * Boundaries are periodic, the case in which Section III-B gives
          the transform-dependent terms in closed form.
        * The result is returned in the units of `data`, α·x̂ + δ.
        * The numerical safeguards of
          [`let_threshold`][optix.denoise.utils.let_threshold] and
          [`PureSums.solve`][optix.denoise.utils.PureSums.solve].

    !!! note "Extensions"
        * `gain`, `offset` and `sigma` may be maps (broadcastable to the
          spatial shape), e.g. the per-pixel parameters of an sCMOS
          camera: y and σ_n² = σ_n²/α_n² are then per pixel, PURE uses
          Stein's lemma per pixel (see
          [`PureSums`][optix.denoise.utils.PureSums]) and the threshold
          uses the local Gaussian variance of each coefficient.
        * `ndim` = 3 denoises volumes (or space-time stacks) with the
          separable 3-D transform; the paper derives the method in 1-D
          and applies it to images (d = 2).

    !!! note "Cost and memory"
        The peak memory is about K + 4·2^d + 16 images of the size of
        the input (periodically padded by 2^J − 1 pixels), with
        K = 2·(2^d − 1)·J: the K LET functions, plus the coefficients of
        one level and their reconstruction (56 for an image and 104 for
        a volume with J = 4; 2^d + 4 more with per-pixel noise maps).
        The cost is linear in the number of pixels.
        [`PURELET`][optix.denoise.PURELET] streams large arrays in
        blocks.

    !!! references "References"
        1. Luisier F, Blu T, Unser M. *Image denoising in mixed
           Poisson–Gaussian noise.* IEEE Trans Image Process
           **20**(3):696–708, 2011.
           [doi:10.1109/TIP.2010.2073477](https://doi.org/10.1109/TIP.2010.2073477)

        ??? quote "BibTeX"
            ```bibtex
            @article{luisier2011image,
              author  = {Luisier, Florian and Blu, Thierry and
                         Unser, Michael},
              title   = {Image denoising in mixed {Poisson--Gaussian}
                         noise},
              journal = {IEEE Transactions on Image Processing},
              volume  = {20}, number = {3}, pages = {696--708},
              year    = {2011}, doi = {10.1109/TIP.2010.2073477}
            }
            ```
    Parameters
    ----------
    data : Tensor
        `(..., *spatial)` with `ndim` spatial axes; each image (volume) is
        denoised separately.
    sigma : Tensor or float
        Standard deviation of the Gaussian noise, in data units (0: pure
        Poisson noise).
    gain : Tensor or float
        α, data units per photon.
    offset : Tensor or float
        δ, in data units.
    levels : int
        Number of levels J.
    ndim : int
        Number of spatial axes d.
    reliability : float
        T of Section III-D.
    sharpness : float
        κ of the smooth absolute value tanh(κ·w̄)·w̄ (Section III-C,
        where it is called k).

    Returns
    -------
    Tensor
        Of the shape of `data`.
    """
    levels, ndim = _check_levels(levels), _check_ndim(ndim)
    reliability = _check_nonneg("reliability", reliability)
    sharpness = _check_positive("sharpness", sharpness)
    if data.ndim < ndim:
        raise ValueError(f"data needs at least {ndim} dims, got {data.ndim}")
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    spatial = tuple(data.shape[-ndim:])
    _check_size(spatial, levels)
    alpha = _as_map(gain, spatial, data)
    delta = _as_map(offset, spatial, data)
    sd = _as_map(sigma, spatial, data)
    _check_params(alpha, delta, sd)
    h = pure_let_halo(levels)
    s = sd * sd / (alpha * alpha)
    s_pad = _wrap(s, ndim, h) if isinstance(s, Tensor) else s
    out = []
    for x in data.reshape(-1, *spatial):
        y = (x - delta) / alpha
        f, low, d = let_basis(
            _wrap(y, ndim, h),
            s_pad,
            levels,
            ndim=ndim,
            sharpness=sharpness,
        )
        sums = PureSums.from_basis(f, low, d, y, s, levels=levels, ndim=ndim)
        a = sums.solve(reliability)
        out.append(_combine(f, low, a) * alpha + delta)
    return torch.stack(out).reshape(data.shape)


def _check_params(
    alpha: Tensor | float, delta: Tensor | float, sd: Tensor | float
) -> None:
    for name, v, positive in (
        ("gain", alpha, True),
        ("offset", delta, False),
        ("sigma", sd, False),
    ):
        t = torch.as_tensor(v, dtype=torch.float64)
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} must be finite")
        if positive and not bool((t > 0).all()):
            raise ValueError(f"{name} must be > 0")
        if name == "sigma" and not bool((t >= 0).all()):
            raise ValueError("sigma must be >= 0")


# -- streaming ---------------------------------------------------------------
def _runs(lo: int, hi: int, n: int) -> list[tuple[int, int]]:
    """Contiguous runs of the indices lo … hi − 1 taken modulo n."""
    out = []
    i = lo
    while i < hi:
        m = i % n
        k = min(hi - i, n - m)
        out.append((m, m + k))
        i += k
    return out


def _read_wrapped(
    source: ArrayLike,
    index: list,
    spatial: Sequence[int],
    runs: Sequence[Sequence[tuple[int, int]]],
    perm: Sequence[int],
    like: Tensor,
) -> Tensor:
    """Read the product of index runs along the spatial axes and join it."""
    if len(runs) == 0:
        t = as_tensor(
            source[tuple(index)], dtype=like.dtype, device=like.device
        )
        return t.permute(list(perm))
    axis, i = spatial[len(spatial) - len(runs)], len(spatial) - len(runs)
    parts = []
    for start, stop in runs[0]:
        index = list(index)
        index[axis] = slice(start, stop)
        parts.append(
            _read_wrapped(source, index, spatial, runs[1:], perm, like)
        )
    return torch.cat(parts, i) if len(parts) > 1 else parts[0]


def _periodic(
    chunks: object, source: ArrayLike, layout: Layout, h: int
) -> Iterator[tuple[object, Tensor]]:
    """Blocks of `chunks` with a periodic halo of `h` on every axis.

    Interior blocks are those loaded by `chunks`; blocks whose halo
    crosses the border of the image are re-read with indices taken
    modulo the image size.
    """
    shape = tuple(int(n) for n in source.shape)
    _, _, spatial = layout.positions(len(shape))
    order = sorted(spatial)
    perm = [order.index(a) for a in spatial]
    for chunk in chunks:
        bounds = [
            (core.start - h, core.stop + h, shape[a])
            for core, a in zip(chunk.spatial, spatial, strict=True)
        ]
        if all(lo >= 0 and hi <= n for lo, hi, n in bounds):
            yield chunk, chunk.data
            continue
        runs = [_runs(lo, hi, n) for lo, hi, n in bounds]
        x = _read_wrapped(
            source, list(chunk.index), spatial, runs, perm, chunk.data
        )
        yield chunk, x


def _wrap_map(
    t: Tensor | float,
    chunk: object,
    shape: Sequence[int],
    h: int,
    like: Tensor,
) -> Tensor | float:
    """A `(*spatial)` map (or scalar) on the block of `chunk`, periodic."""
    if not isinstance(t, Tensor):
        return t
    for i, (core, n) in enumerate(zip(chunk.spatial, shape, strict=True)):
        if t.shape[i] == 1:
            continue
        idx = torch.arange(core.start - h, core.stop + h) % n
        t = t.index_select(i, idx.to(t.device))
    return t.to(like)


def _select(t: Tensor, group: tuple[int, ...], ndim: int) -> Tensor:
    """A `(*group, *spatial)` map at one group, with `ndim` spatial dims."""
    nlead = t.ndim - ndim
    if nlead > 0:
        if len(group) >= nlead:
            t = t[tuple(group[:nlead])]
        elif all(n == 1 for n in t.shape[:nlead]):
            t = t.reshape(t.shape[nlead:])
        else:
            raise ValueError(
                f"A map of shape {tuple(t.shape)} does not match the layout"
            )
    return t.reshape((1,) * (ndim - t.ndim) + tuple(t.shape))


def _constant(t: Tensor) -> bool:
    return t.numel() <= 1 or bool((t == t.reshape(-1)[0]).all())


# -- estimator ---------------------------------------------------------------
class PURELET(Estimator):
    """PURE-LET denoising of Poisson-Gaussian noise (Luisier et al. 2011).

    Applies [`pure_let`][optix.denoise.pure_let] to each image (the
    spatial axes of the layout; other axes are samples or groups, and
    each frame gets its own LET coefficients). The noise parameters are
    a [`SensorEstimate`][optix.estimates.SensorEstimate] given to `fit`
    (gain α, offset δ, read variance in ADU²), or `sigma`, `gain` and
    `offset` given at construction.

    [`apply`][optix.Estimator.apply] reads the data twice, in blocks with
    a periodic halo of [`pure_let_halo`][optix.denoise.pure_let_halo]
    pixels (wrapped around the image, the boundary condition of the
    paper): the first pass accumulates, per frame, the 2·(2^d − 1)·J
    × 2·(2^d − 1)·J linear system of PURE
    ([`PureSums`][optix.denoise.utils.PureSums]: sums over pixels and
    coefficients, so blocks give the whole-image system), which is
    solved once per frame; the second pass reconstructs each block with
    the frame's coefficients. Arrays larger than memory (zarr, memmap)
    thus give the whole-image result.

    !!! note
        Faithful mode is the UWT PURE-LET of the paper (Haar, Section
        III), with its T = 10 and κ = 100 (called k in the paper).
        `pixelwise` (per-pixel noise maps), `ndim` ≠ 2 (volumes,
        space-time stacks) and other values of `reliability` or
        `sharpness` are extensions; `levels` (not given in the paper)
        and `block` do not change the method. Maps in the
        `SensorEstimate` need `pixelwise=True`. Every spatial axis
        needs at least 2^J samples (see
        [`pure_let`][optix.denoise.pure_let]).

    !!! references "References"
        1. Luisier F, Blu T, Unser M. *Image denoising in mixed
           Poisson–Gaussian noise.* IEEE Trans Image Process
           **20**(3):696–708, 2011.
           [doi:10.1109/TIP.2010.2073477](https://doi.org/10.1109/TIP.2010.2073477)

        ??? quote "BibTeX"
            ```bibtex
            @article{luisier2011image,
              author  = {Luisier, Florian and Blu, Thierry and
                         Unser, Michael},
              title   = {Image denoising in mixed {Poisson--Gaussian}
                         noise},
              journal = {IEEE Transactions on Image Processing},
              volume  = {20}, number = {3}, pages = {696--708},
              year    = {2011}, doi = {10.1109/TIP.2010.2073477}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.denoise import PURELET
    >>> g = torch.Generator().manual_seed(0)
    >>> clean = 20 + 15 * torch.sin(torch.arange(64.0) / 5).expand(64, 64)
    >>> noisy = torch.poisson(clean, generator=g)
    >>> noisy = noisy + 2 * torch.randn(64, 64, generator=g)
    >>> den = PURELET(sigma=2.0).fit().apply(noisy)
    >>> bool((den - clean).square().mean() < (noisy - clean).square().mean())
    True
    """

    tags = Tags(
        targets=("denoised",),
        streaming="halo",
        noise="poisson-gaussian",
        reference="doi:10.1109/TIP.2010.2073477",
    )
    paper = {
        "sigma": None,
        "gain": None,
        "offset": None,
        "levels": 4,
        "reliability": 10.0,
        "sharpness": 100.0,
        "ndim": 2,
        "pixelwise": False,
        "block": None,
    }
    recommended = dict(paper)

    def __init__(
        self,
        *,
        sigma: float | None = None,
        gain: float | None = None,
        offset: float | None = None,
        levels: int | None = None,
        reliability: float | None = None,
        sharpness: float | None = None,
        ndim: int | None = None,
        pixelwise: bool | None = None,
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
            Standard deviation of the Gaussian (read) noise, in data
            units; 0 for pure Poisson noise. Omit it to give a
            `SensorEstimate` to `fit` instead.
        gain : float, optional
            α, data units per photon (default 1). Ignored with a
            `SensorEstimate`.
        offset : float, optional
            δ, in data units (default 0). Ignored with a
            `SensorEstimate`.
        levels : int, optional
            Number of levels J of the wavelet transform (not given in the
            paper). Default: 4.
        reliability : float, optional
            T of Section III-D (paper: 10).
        sharpness : float, optional
            κ of the smooth absolute value tanh(κ·w̄)·w̄ of Section III-C
            (called k there; paper: 100).
        ndim : int, optional
            Extension when ≠ 2: number of spatial axes transformed (3 for
            volumes, or for space-time with time declared spatial in the
            layout). Default: 2.
        pixelwise : bool, optional
            Extension: use the maps of a `SensorEstimate` (per-pixel
            gain, offset and read variance of sCMOS cameras) pixel by
            pixel. Default: False.
        block : sequence of int or None, optional
            Core block size per spatial axis (default: from the memory
            budget). The result does not depend on it.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        if o["sigma"] is not None:
            o["sigma"] = _check_nonneg("sigma", o["sigma"])
        if o["gain"] is not None:
            o["gain"] = _check_positive("gain", o["gain"])
        if o["offset"] is not None:
            o["offset"] = float(o["offset"])
        o["levels"] = _check_levels(o["levels"])
        o["ndim"] = _check_ndim(o["ndim"])
        o["reliability"] = _check_nonneg("reliability", o["reliability"])
        o["sharpness"] = _check_positive("sharpness", o["sharpness"])
        o["pixelwise"] = bool(o["pixelwise"])
        self.is_faithful = (
            o["reliability"] == 10.0
            and o["sharpness"] == 100.0
            and o["ndim"] == 2
            and not o["pixelwise"]
        )
        self._params: tuple[Tensor | float, ...] | None = None

    # -- fit -----------------------------------------------------------------
    def fit(
        self,
        data: ArrayLike | None = None,
        *,
        layout: Layout | None = None,
        init: object = None,
        **kwargs: object,
    ) -> Self:
        """Set the noise model.

        The LET coefficients are not fitted here: PURE is minimised for
        each frame of the data given to [`apply`][optix.Estimator.apply].

        Parameters
        ----------
        data : array-like, optional
            Only its layout is recorded.
        layout : Layout, optional
        init : SensorEstimate or Estimator, optional
            Noise parameters (gain α in ADU per photon, offset δ, read
            variance in ADU²; scalars, or maps with `pixelwise=True`).
            Default: `sigma`, `gain` and `offset` of the constructor.
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
        o = self.options
        sensor = self._pick(init, SensorEstimate) if init is not None else None
        if init is not None:
            self._warn_unused(init, SensorEstimate)
        if sensor is None:
            if o["sigma"] is None:
                raise ValueError(
                    "PURELET needs sigma= or fit(init=SensorEstimate)"
                )
            gain = 1.0 if o["gain"] is None else o["gain"]
            offset = 0.0 if o["offset"] is None else o["offset"]
            sensor = SensorEstimate(
                offset=torch.tensor(offset, dtype=torch.float64),
                variance=torch.tensor(o["sigma"] ** 2, dtype=torch.float64),
                gain=torch.tensor(gain, dtype=torch.float64),
            )
        for name in ("offset", "variance", "gain"):
            if getattr(sensor, name) is None:
                raise ValueError(f"init SensorEstimate lacks {name}")
        self.estimates = {"sensor": sensor}
        self._set_sensor(sensor)

    def _set_sensor(self, sensor: SensorEstimate) -> None:
        alpha = sensor.gain.double()
        delta = sensor.offset.double()
        var = sensor.variance.double()
        _check_params(alpha, delta, var.clamp_min(0).sqrt())
        if (var < 0).any():
            raise ValueError("the read variance must be >= 0")
        params = (alpha, delta, var)
        if all(_constant(t) for t in params):
            self._params = tuple(float(t.reshape(-1)[0]) for t in params)
        elif not self.options["pixelwise"]:
            raise ValueError(
                "The SensorEstimate has per-pixel maps: pass pixelwise=True "
                "(an extension of the paper, whose noise parameters are "
                "scalars)"
            )
        else:
            self._params = params

    # -- apply ---------------------------------------------------------------
    def halo(self) -> int:
        """Halo of the block-wise processing, 2^J − 1.

        Returns
        -------
        int
        """
        return pure_let_halo(self.options["levels"])

    def _block_params(
        self, chunk: object, x: Tensor, shape: Sequence[int], h: int
    ) -> tuple[Tensor | float, Tensor | float, Tensor | float]:
        """α, δ and σ² on a block (periodic), and on its core."""
        ndim = self.options["ndim"]
        out = []
        for t in self._params:
            if isinstance(t, Tensor):
                t = _wrap_map(
                    _select(t, chunk.group, ndim), chunk, shape, h, x
                )
            out.append(t)
        return tuple(out)

    def _frames(self, data: Labeled) -> tuple[object, Sequence[int], int]:
        o = self.options
        _, _, spatial = data.layout.positions(len(data.shape))
        if len(spatial) != o["ndim"]:
            raise ValueError(
                f"PURELET needs {o['ndim']} spatial axes, got {len(spatial)}"
            )
        shape = [int(data.shape[a]) for a in spatial]
        return spatial, shape, self.halo()

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if self._params is None:
            raise RuntimeError("Call fit before apply")
        o = self.options
        ndim, levels = o["ndim"], o["levels"]
        if not isinstance(data, Labeled):
            data = self._labeled(data, self._layout_of(data, ndim))
        _, shape, h = self._frames(data)
        _check_size(shape, levels)
        raw = data.array
        target = out.array if isinstance(out, Labeled) else out
        if out is not None and _shares_memory(target, raw):
            raise ValueError(
                "In-place processing with a halo is not possible: use a "
                "separate output."
            )
        out = prepare_out(out, tuple(data.shape), like=raw, dtype=self.dtype)
        maps = any(isinstance(t, Tensor) for t in self._params)
        chunks = blocks(
            data,
            o["block"],
            h,
            overhead=_block_overhead(ndim, levels, maps),
            contiguous=True,
            align=chunks_of(out),
            dtype=self.dtype,
            device=self.device,
        )
        kw = dict(ndim=ndim, sharpness=o["sharpness"], halo=h)
        # pass 1: the PURE system of each frame
        systems: dict[tuple, PureSums] = {}
        for chunk, x in _periodic(chunks, raw, data.layout, h):
            x = self._float(x)
            alpha, delta, s = self._normalisation(chunk, x, shape, h)
            y = (x - delta) / alpha
            f, low, d = let_basis(y, s, levels, **kw)
            core = [sl.stop - sl.start for sl in chunk.spatial]
            sums = PureSums.from_basis(
                f,
                low,
                d,
                _crop(y, ndim, h, core),
                _crop(s, ndim, h, core) if isinstance(s, Tensor) else s,
                levels=levels,
                ndim=ndim,
            )
            key = _frame_key(chunk)
            systems[key] = sums if key not in systems else systems[key] + sums
            # free the block before the next one is read
            del x, y, s, alpha, delta, f, low, d, sums
        coefs = {k: v.solve(o["reliability"]) for k, v in systems.items()}
        # pass 2: reconstruction
        for chunk, x in _periodic(chunks, raw, data.layout, h):
            x = self._float(x)
            alpha, delta, s = self._normalisation(chunk, x, shape, h)
            f, low, _ = let_basis(
                (x - delta) / alpha, s, levels, derivatives=False, **kw
            )
            core = [sl.stop - sl.start for sl in chunk.spatial]
            est = _combine(f, low, coefs[_frame_key(chunk)])
            a_c, d_c = (
                _crop(t, ndim, h, core) if isinstance(t, Tensor) else t
                for t in (alpha, delta)
            )
            chunk.store(out, est * a_c + d_c)
            del x, s, alpha, delta, f, low, est, a_c, d_c
        flush(out)
        return out

    def _float(self, x: Tensor) -> Tensor:
        if not x.is_floating_point():
            x = x.to(self._backend(x)["dtype"])
        return x

    def _normalisation(
        self, chunk: object, x: Tensor, shape: Sequence[int], h: int
    ) -> tuple[Tensor | float, Tensor | float, Tensor | float]:
        alpha, delta, var = (
            t.expand_as(x) if isinstance(t, Tensor) else t
            for t in self._block_params(chunk, x, shape, h)
        )
        return alpha, delta, var / (alpha * alpha)

    def _load_extra_state(self, state: dict[str, object]) -> None:
        sensor = self.estimates.get("sensor")
        if sensor is not None:
            self._set_sensor(sensor)


def _block_overhead(ndim: int, levels: int, maps: bool) -> int:
    """Peak memory of `PURELET.apply`, in copies of one block with halo.

    The K = 2·(2^d − 1)·J LET functions of the block, plus the transient
    of one level (its 2·(2^d − 1) coefficient bands and their
    reconstruction) and the block itself, measured with the torch
    profiler: K + 4·2^d + 16, and 2^d + 4 more with per-pixel noise maps.
    """
    nfun = 2 * (2**ndim - 1) * levels
    return nfun + 4 * 2**ndim + 16 + ((2**ndim + 4) if maps else 0)


def _frame_key(chunk: object) -> tuple:
    return tuple(chunk.group) + tuple(s.start for s in chunk.samples)
