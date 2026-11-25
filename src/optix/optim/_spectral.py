"""Spectral maps of packed symmetric 2 × 2 and 3 × 3 matrices, in closed form.

A field of symmetric d × d matrices is packed as in
[`Hessian`][optix.optim.Hessian]: the d diagonal entries, then √2 times
the entries above the diagonal, C = d(d + 1)/2 components. A *spectral
map* replaces the eigenvalues μᵢ of every matrix X = U diag(μ) Uᵀ by
g(μ)ᵢ and keeps the eigenvectors, X̂ = U diag(g(μ)) Uᵀ. All the maps used
here (Schatten-norm proxes and projections onto Schatten balls) apply
one scalar function to every eigenvalue, either a soft threshold
sign(μ)·max(|μ| − γ, 0) or a clip min(max(μ, −c), c), with a level γ or c
that may depend on all the eigenvalues of the matrix. These functions are
non-decreasing and 1-Lipschitz, which the reconstructions below rely on.

**2 × 2** (Lefkimmiatis, Bourquard & Unser 2012, Eq. 15). With
X = [[a, b], [b, d]], m = (a + d)/2 and r = √(((a − d)/2)² + b²), the
eigenvalues are m ± r and

    X̂ = m̂·I + (r̂/r)·(X − m·I),   m̂ = (ĝ₊ + ĝ₋)/2,   r̂ = (ĝ₊ − ĝ₋)/2.

**3 × 3** (closed form; not in the papers, which are 2-D). Write
X = m·I + s·B with m = tr(X)/3 and s the largest absolute entry of
X − m·I, so B is traceless with entries in [−1, 1] and q = tr(B²) ≥ 1.

1. *The isolated eigenvalue.* The eigenvalues β of B are the roots of
   β³ − (q/2)β − det B. With ρ = det(B)/(2(q/6)^{3/2}) ∈ [−1, 1] and
   φ = arccos(|ρ|)/3 ∈ [0, π/6], the root λ = sign(ρ)·2√(q/6)·cos φ is
   the one farthest from the other two (the trigonometric solution;
   Smith 1961): its gap to them is at least √(q/2) ≥ 1/√2. It is
   refined by one Newton step on the cubic, which also gives it the
   exact derivative of the implicit-function theorem.
2. *Its eigenvector* u is the largest of the three cross products of
   the rows of B − λI, normalised; the gap bound keeps it well
   conditioned (the cross products are the columns of the adjugate of
   B − λI, (β₂ − λ)(β₃ − λ)·u uᵀ).
3. *The other two* are those of P B P on the plane P = I − u uᵀ, a
   2 × 2 problem solved as above without forming a basis of the plane:
   m₂ = (tr B − uᵀBu)/2, D = P B P − m₂·P, r₂ = ‖D‖_F/√2, eigenvalues
   m₂ ± r₂.

Then X̂ = ĝ_u·u uᵀ + m̂·P + (r̂/r)·s·D, with m̂ and r̂ from the pair.
A double eigenvalue is always the pair, which is solved by the 2 × 2
formula and needs no eigenvectors; the vector u belongs to a simple
eigenvalue separated from the others by at least half the spread, so
every step is accurate to a few ulps of |m| + s.

Two cases map X to g(m)·I:

- *s below `tiny/eps`* (about 1e-31 in float32, 1e-292 in float64),
  where B could not be formed to full precision. The absolute error is
  then below that threshold, so the "few ulps" bound holds relative to
  |m| + s only above it: a float32 matrix diag(1e-33, 0, 0) gets the
  eigenvalues (m, m, m).
- *A rounding breakdown.* The rounded mean m leaves tr B = O(eps·|m|/s),
  which is not small when s is a few ulps of |m|. λ is then not an
  eigenvalue of B, and B − λI can have rank 1 (B = c·(I − w wᵀ), every
  cross product 0). For an exact λ the largest cross product has squared
  norm (β₂ − λ)²(β₃ − λ)²·maxₖ uₖ² ≥ gap⁴/3 ≥ 1/12, so a squared norm
  below 10⁻³ only occurs in this case. There s = O(eps·|m|) and g(m)·I is
  within a few ulps of |m| + s of the exact map, which is non-expansive.

!!! note "Why a closed form and not `torch.linalg.eigh`"
    The map is applied at every iteration of the Hessian–Schatten
    solvers. On 3 × 3 fields with a mix of distinct, repeated, nearly
    repeated and zero eigenvalues (2·10⁵ matrices, one CPU thread), the
    closed form computes the soft-threshold map about 3 times faster
    than batched `eigh` with the reconstruction U diag(g) Uᵀ in float32
    (1.6 times in float64). Its eigenvalues and maps are within 5 ulps
    of the matrix scale of a float64 `eigh` reference in float32, and
    within about 10 ulps in float64, where the reference's own error is of
    the same order (float32 `eigh`: 16 ulps). It is element-wise,
    processed in chunks of voxels to bound its temporaries, and has
    finite derivatives everywhere, where those of `eigh` are NaN at
    repeated eigenvalues.

!!! references "References"
    1. Smith OK. *Eigenvalues of a symmetric 3 × 3 matrix.* Commun ACM
       **4**(4):168, 1961.
       [doi:10.1145/355578.366316](https://doi.org/10.1145/355578.366316)
    2. Lefkimmiatis S, Bourquard A, Unser M. *Hessian-based norm
       regularization for image restoration with biomedical
       applications.* IEEE Trans Image Process **21**(3):983-995, 2012.
       [doi:10.1109/TIP.2011.2168232](https://doi.org/10.1109/TIP.2011.2168232)
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor

_SQ2 = math.sqrt(2)
#: Voxels per chunk of the element-wise eigen-decomposition (bounds its
#: few dozen temporaries to about this many voxels each).
_CHUNK = 1 << 16
#: Squared norm of the largest cross product of the rows of B − λI below
#: which the 3 × 3 solver treats the matrix as m·I. An exact isolated
#: eigenvalue gives at least 1/12; the value only falls below this when
#: the rounding of m leaves B far from traceless (module docstring).
_BREAKDOWN = 1e-3

#: A scalar map of the eigenvalues: `fn(mu, t)` with `mu` `(d, n)` and
#: `t` a float or `(n,)` tensor returns `(g, slope, piece)`, the mapped
#: eigenvalues, the slope of the scalar function at each eigenvalue
#: (0 or 1) and an integer label of its linear piece.
ScalarMap = Callable[[Tensor, float | Tensor], tuple[Tensor, Tensor, Tensor]]


def packed_ndim(components: int) -> int:
    """d such that d(d + 1)/2 = C."""
    d = round((math.sqrt(8 * components + 1) - 1) / 2)
    if d < 1 or d * (d + 1) // 2 != components:
        raise ValueError(
            f"{components} components do not pack a symmetric matrix"
        )
    return d


# -- element-wise helpers --------------------------------------------------
def _where(cond: Tensor, a: Tensor | float, b: Tensor) -> Tensor:
    if not isinstance(a, Tensor):
        a = torch.full_like(b, a)
    return torch.where(cond, a, b)


def _safe_sqrt(sq: Tensor) -> Tensor:
    """√sq with a zero (not NaN) derivative at sq = 0."""
    pos = sq > 0
    return _where(
        pos, torch.sqrt(_where(pos, sq, torch.ones_like(sq))), sq * 0
    )


def _hypot(a: Tensor, b: Tensor) -> Tensor:
    """√(a² + b²) without overflow, with a zero derivative at 0."""
    with torch.no_grad():
        s = torch.maximum(a.abs(), b.abs())
        pos = s > 0
        s = _where(pos, s, torch.ones_like(s))
    an, bn = a / s, b / s
    return s * _safe_sqrt(an.square() + bn.square())


# -- 2 × 2 ---------------------------------------------------------------
def eig2(v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """m, r and X − m·I of packed 2 × 2 matrices `(3, n)`.

    Eigenvalues m ± r (r ≥ 0); the third output is packed like `v`.
    """
    a, d, c = v.unbind(0)
    m = (a + d) / 2
    h = (a - d) / 2
    r = _hypot(h, c / _SQ2)
    dev = torch.stack([h, -h, c])
    return m, r, dev


# -- 3 × 3 ---------------------------------------------------------------
class _Eig3(NamedTuple):
    """Closed-form spectral decomposition of packed 3 × 3 matrices.

    X = m·I + s·(λ u uᵀ + m₂ P + D + E), with P = I − u uᵀ, D the
    traceless part of P B P on the plane (eigenvalues ±r₂) and E ≈ 0
    the rounding of the eigenvector. `s` is 0 where X is mapped as m·I.
    """

    m: Tensor
    s: Tensor
    lam: Tensor
    u: tuple[Tensor, Tensor, Tensor]
    m2: Tensor
    r2: Tensor
    dev: tuple[Tensor, ...]  # D: d11, d22, d33, d12, d13, d23

    def eigenvalues(self) -> Tensor:
        """(μ_u, μ₊, μ₋), `(3, n)`: the isolated one, then the pair."""
        m, s = self.m, self.s
        return torch.stack(
            [
                m + s * self.lam,
                m + s * (self.m2 + self.r2),
                m + s * (self.m2 - self.r2),
            ]
        )


def eig3(v: Tensor) -> _Eig3:
    """Closed-form spectral decomposition of packed 3 × 3 matrices `(6, n)`.

    See the module docstring for the method.
    """
    x11, x22, x33, c12, c13, c23 = v.unbind(0)
    m = (x11 + x22 + x33) / 3
    b = [x11 - m, x22 - m, x33 - m, c12 / _SQ2, c13 / _SQ2, c23 / _SQ2]
    finfo = torch.finfo(v.dtype)
    with torch.no_grad():
        s = torch.stack([e.abs() for e in b]).amax(0)
        deg = s <= finfo.tiny / finfo.eps
        scale = _where(deg, 1.0, s)
        s = _where(deg, 0.0, s)
    # B = (X − m·I)/s; a fixed non-degenerate B where X = m·I (s = 0)
    safe = (1.0, 0.0, -1.0, 0.0, 0.0, 0.0)
    n11, n22, n33, n12, n13, n23 = (
        _where(deg, f, e / scale) for f, e in zip(safe, b, strict=True)
    )
    q = (
        n11.square()
        + n22.square()
        + n33.square()
        + 2 * (n12.square() + n13.square() + n23.square())
    )
    det = (
        n11 * n22 * n33
        + 2 * n12 * n13 * n23
        - n11 * n23.square()
        - n22 * n13.square()
        - n33 * n12.square()
    )
    with torch.no_grad():
        p = torch.sqrt(q / 6)
        rho = (det / (2 * p * p * p)).clamp(-1, 1)
        sign = _where(rho < 0, -1.0, torch.ones_like(rho))
        lam0 = sign * 2 * p * torch.cos(torch.arccos(rho.abs()) / 3)
    # one Newton step on β³ − (q/2)β − det B (the derivative is at least
    # the squared gap, ≥ 1/2)
    lam = lam0 - (lam0 * (lam0.square() - q / 2) - det) / (
        3 * lam0.square() - q / 2
    )
    a11, a22, a33 = n11 - lam, n22 - lam, n33 - lam
    crosses = (
        (n12 * n23 - n13 * a22, n13 * n12 - a11 * n23, a11 * a22 - n12 * n12),
        (n12 * a33 - n13 * n23, n13 * n13 - a11 * a33, a11 * n23 - n12 * n13),
        (a22 * a33 - n23 * n23, n23 * n13 - n12 * a33, n12 * n23 - a22 * n13),
    )
    with torch.no_grad():
        norms = torch.stack(
            [sum(e.square() for e in w) for w in crosses]  # type: ignore[misc]
        )
        best = norms.argmax(0)
        # Breakdown of the rounded mean: if s is a few ulps of |m|, B is
        # not traceless, λ need not be an eigenvalue of B and B − λI can
        # have rank 1 (B = c·(I − w wᵀ) makes every cross product 0). For
        # an exact λ the largest squared cross product is at least
        # gap⁴/3 ≥ 1/12, so a value below _BREAKDOWN only occurs there,
        # where X maps to g(m)·I within O(s) = O(eps·|m|).
        bad = norms.amax(0) < _BREAKDOWN
        s = _where(bad, 0.0, s)
    w = [
        _where(
            bad,
            1.0 if i == 0 else 0.0,
            torch.where(
                best == 0,
                crosses[0][i],
                torch.where(best == 1, crosses[1][i], crosses[2][i]),
            ),
        )
        for i in range(3)
    ]
    norm = torch.sqrt(w[0].square() + w[1].square() + w[2].square())
    u1, u2, u3 = w[0] / norm, w[1] / norm, w[2] / norm
    # B u, uᵀ B u and the pair on the plane P = I − u uᵀ
    bu1 = n11 * u1 + n12 * u2 + n13 * u3
    bu2 = n12 * u1 + n22 * u2 + n23 * u3
    bu3 = n13 * u1 + n23 * u2 + n33 * u3
    beta = u1 * bu1 + u2 * bu2 + u3 * bu3
    m2 = (n11 + n22 + n33 - beta) / 2
    u = (u1, u2, u3)
    bu = (bu1, bu2, bu3)
    nb = {(0, 0): n11, (1, 1): n22, (2, 2): n33, (0, 1): n12, (0, 2): n13}
    nb[(1, 2)] = n23
    dev = []
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        uu = u[i] * u[j]
        pbp = nb[(i, j)] - u[i] * bu[j] - bu[i] * u[j] + beta * uu
        proj = (1 - uu) if i == j else -uu
        dev.append(pbp - m2 * proj)
    d11, d22, d33, d12, d13, d23 = dev
    frob = (
        d11.square()
        + d22.square()
        + d33.square()
        + 2 * (d12.square() + d13.square() + d23.square())
    )
    r2 = _safe_sqrt(frob / 2)
    return _Eig3(m, s, lam, u, m2, r2, tuple(dev))


# -- maps ------------------------------------------------------------------
def soft(mu: Tensor, gamma: float | Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Soft threshold sign(μ)·max(|μ| − γ, 0), with slopes and pieces."""
    a = mu.abs()
    sign = torch.sign(mu)
    on = a > gamma
    g = sign * (a - gamma).clamp(min=0)
    return g, on.to(mu.dtype), _where(on, sign, torch.zeros_like(mu))


def clip(mu: Tensor, c: float | Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Clip min(max(μ, −c), c), with slopes and pieces."""
    if not isinstance(c, Tensor):
        c = torch.full_like(mu, c)
    inside = mu.abs() <= c
    g = torch.minimum(torch.maximum(mu, -c), c)
    piece = _where(inside, torch.zeros_like(mu), torch.sign(mu))
    return g, inside.to(mu.dtype), piece


def l1_level(a: Tensor, radius: float | Tensor) -> Tensor:
    """γ ≥ 0 with ∑ᵢ max(aᵢ − γ, 0) = r, or 0 if ∑ aᵢ ≤ r (a ≥ 0, `(d, n)`).

    γ = max(0, maxₖ (Sₖ − r)/k), with Sₖ the sum of the k largest aᵢ: the
    level of the projection onto the ℓ1 ball of radius r, and the cap of
    the prox of r‖·‖∞ (Moreau). For d = 2 this is Eq. (36) of
    Lefkimmiatis, Ward & Unser (2013) at radius 1.
    """
    d = a.shape[0]
    total = a.sum(0)
    best = (total - radius) / d
    if d >= 2:
        top = a.amax(0)
        best = torch.maximum(best, top - radius)
    if d == 3:
        best = torch.maximum(best, (total - a.amin(0) - radius) / 2)
    return best.clamp(min=0)


def _ratio(
    gp: Tensor,
    gm: Tensor,
    den: Tensor,
    slope: Tensor,
    piece: Tensor,
    scale: Tensor,
) -> Tensor:
    """r̂/r for a pair μ± with μ₊ − μ₋ = `den` ≥ 0, in [0, 1].

    The slope of the scalar function when both eigenvalues lie on one
    linear piece (or are too close to tell), the divided difference
    (ĝ₊ − ĝ₋)/(μ₊ − μ₋) otherwise.
    """
    eps = torch.finfo(den.dtype).eps
    same = piece[0] == piece[1]
    near = den <= 4 * eps * scale
    use = same | near
    fallback = _where(same, slope[0], (slope[0] + slope[1]) / 2)
    safe = _where(use, torch.ones_like(den), den)
    return _where(use, fallback, ((gp - gm) / safe).clamp(0, 1))


def spectral_map(v: Tensor, fn: ScalarMap, t: float | Tensor) -> Tensor:
    """X̂ = U diag(g(μ)) Uᵀ for packed matrices `(C, n)`, C ∈ {1, 3, 6}."""
    c = v.shape[0]
    if c == 1:
        return fn(v, t)[0]
    if c == 3:
        m, r, dev = eig2(v)
        g, slope, piece = fn(torch.stack([m + r, m - r]), t)
        ratio = _ratio(g[0], g[1], 2 * r, slope, piece, m.abs() + r)
        mean = (g[0] + g[1]) / 2
        iso = torch.stack([mean, mean, torch.zeros_like(mean)])
        return iso + ratio * dev
    e = eig3(v)
    mu = e.eigenvalues()
    g, slope, piece = fn(mu, t)
    ratio = _ratio(
        g[1], g[2], 2 * e.s * e.r2, slope[1:], piece[1:], e.m.abs() + e.s
    )
    mean = (g[1] + g[2]) / 2
    lift = g[0] - mean
    k = ratio * e.s
    u = e.u
    out = []
    for i, j, dij in zip(
        (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2), e.dev, strict=True
    ):
        z = lift * u[i] * u[j] + k * dij
        out.append(z + mean if i == j else _SQ2 * z)
    return torch.stack(out)


def eigenvalues(v: Tensor) -> Tensor:
    """Eigenvalues of packed matrices `(C, n)` → `(d, n)` (unsorted)."""
    c = v.shape[0]
    if c == 1:
        return v
    if c == 3:
        m, r, _ = eig2(v)
        return torch.stack([m + r, m - r])
    return eig3(v).eigenvalues()


def voxelwise(
    kernel: Callable[[Tensor, float | Tensor], Tensor],
    v: Tensor,
    t: float | Tensor,
    dim: int,
    *,
    components: bool,
) -> Tensor:
    """Apply `kernel((C, n), t)` to the voxels of `v`, in chunks.

    `t` is a float or a tensor that broadcasts against `v` with a size-1
    `dim`. The kernel returns `(C, n)` (`components=True`) or `(n,)`.
    Chunks bound the temporaries of the kernel; they are skipped when
    `v` or `t` requires a gradient.
    """
    dim = dim % v.ndim
    vm = v.movedim(dim, 0)
    c, rest = vm.shape[0], vm.shape[1:]
    n = math.prod(rest)
    flat = vm.reshape(c, n)
    tf: float | Tensor = t
    if isinstance(t, Tensor):
        shape = list(v.shape)
        shape[dim] = 1
        tf = torch.broadcast_to(t, shape).movedim(dim, 0).reshape(n)
    grad = v.requires_grad or (isinstance(t, Tensor) and t.requires_grad)
    if grad or n <= _CHUNK:
        out = kernel(flat, tf)
    else:
        out = flat.new_empty((c, n) if components else (n,))
        for lo in range(0, n, _CHUNK):
            hi = min(lo + _CHUNK, n)
            tc = tf[lo:hi] if isinstance(tf, Tensor) else tf
            out[..., lo:hi] = kernel(flat[:, lo:hi], tc)
    if components:
        return out.reshape(out.shape[0], *rest).movedim(0, dim)
    return out.reshape(rest)
