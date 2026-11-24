"""Test helpers of the shared-signal (tile overlap) model.

A sampler from the generative model, and float64 brute-force oracles of
the per-voxel integral and of the total bound.
"""

import hashlib
import math

import numpy as np
import torch

import optix
from optix.io import Layout
from optix.shading.mix import GammaMixture

__all__ = [
    "digest",
    "fit_tiles",
    "oracle_loglik",
    "oracle_moments",
    "reference_fit",
    "sample_scene",
]


def digest(t: torch.Tensor) -> str:
    """sha256 of the bytes of a tensor (exact comparison of results)."""
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode()).hexdigest()


def sample_scene(
    pos,
    tile=(4, 10, 8),
    *,
    seed=0,
    mu=(150.0,),
    nu=(2.0,),
    frac=(0.6,),
    gain=2.0,
    tile_gain=None,
    offset=100.0,
    read_sd=3.0,
    logm=None,
    dtype=torch.float64,
    integer=True,
):
    """Tiles `(T, *tile)` drawn from the shared-signal generative model.

    Every scene voxel z has one class k_z and one signal u_z (0 in the
    background); tile t at position `pos[t]` sees y = c + b_t·u +
    noise with b_t = g_t·m and variance v + γ·b_t·u (Gaussian).

    Returns (tiles, truth) with truth a dict of u, k, logm, tile_gain.
    """
    pos = np.asarray(pos, dtype=np.int64)
    T = len(pos)
    lo = pos.min(0)
    shape = tuple((pos.max(0) - lo + np.asarray(tile)).tolist())
    gen = torch.Generator().manual_seed(seed)
    K = len(mu)
    probs = torch.tensor([*frac, 1 - sum(frac)], dtype=torch.float64)
    k = torch.multinomial(probs, math.prod(shape), True, generator=gen)
    k = k.reshape(shape)
    u = torch.zeros(shape, dtype=torch.float64)
    for j in range(K):
        sh = torch.full(shape, float(nu[j]), dtype=torch.float64)
        draw = torch._standard_gamma(sh, generator=gen) * mu[j] / nu[j]
        u = torch.where(k == j, draw, u)
    H, W = tile[-2:]
    if logm is None:
        yy = torch.linspace(-1, 1, H, dtype=torch.float64)[:, None]
        xx = torch.linspace(-1, 1, W, dtype=torch.float64)[None]
        logm = 0.3 * yy - 0.2 * yy**2 + 0.15 * xx * yy
        logm = logm - logm.mean()
    g = torch.ones(T, dtype=torch.float64)
    if tile_gain is not None:
        g = torch.as_tensor(tile_gain, dtype=torch.float64)
    out = []
    for t in range(T):
        box = tuple(
            slice(int(p - o), int(p - o + n))
            for p, o, n in zip(pos[t], lo, tile, strict=True)
        )
        b = g[t] * logm.exp()
        mean = offset + b * u[box]
        var = read_sd**2 + gain * b * u[box]
        y = mean + var.sqrt() * torch.randn(
            mean.shape, generator=gen, dtype=torch.float64
        )
        out.append(y.round() if integer else y)
    tiles = torch.stack(out).to(dtype)
    return tiles, dict(u=u, k=k, logm=logm, tile_gain=g, origin=lo)


def _grid(n=20001, lo=-12.0, hi=9.0):
    return torch.linspace(lo, hi, n, dtype=torch.float64)


def _log_integrand(Y, C, V, B, gain, mu, nu, dq, full, tau):
    """l(τ) + log normalisers, on a grid of τ = log u (last dim)."""
    u = tau.exp()
    beta = nu / mu
    lg = nu * math.log(beta) - math.lgamma(nu) + nu * tau - beta * u
    tot = lg
    for y, c, v, b in zip(Y, C, V, B, strict=True):
        s2 = v + (gain * b * u if full else 0 * u)
        r = y - c - b * u
        tot = tot - 0.5 * ((r**2 + dq) / s2 + s2.log() + math.log(2 * math.pi))
    return tot


def oracle_loglik(Y, C, V, B, gain, mu, nu, *, dq=0.0, full=True, n=20001):
    """log ∫ Gamma(u | ν, ν/μ) ∏_t N(y_t | c_t + b_t u, σ_t²(u)) du.

    Dense trapezoid in τ = log u, float64; Y, C, V, B are sequences of
    floats (one per tile seeing the voxel).
    """
    tau = _grid(n)
    lt = _log_integrand(Y, C, V, B, gain, mu, nu, dq, full, tau)
    m = lt.max()
    return (m + torch.trapezoid((lt - m).exp(), tau).log()).item()


def oracle_moments(Y, C, V, B, gain, mu, nu, *, dq=0.0, full=True, n=20001):
    """(log p, E[u], E[log u]) of the voxel integral by the same rule."""
    tau = _grid(n)
    lt = _log_integrand(Y, C, V, B, gain, mu, nu, dq, full, tau)
    m = lt.max()
    w = (lt - m).exp()
    Z = torch.trapezoid(w, tau)
    EU = torch.trapezoid(w * tau.exp(), tau) / Z
    ElogU = torch.trapezoid(w * tau, tau) / Z
    return (m + Z.log()).item(), EU.item(), ElogU.item()


def fit_tiles(data, overlap, it=40, **kw):
    """The fit used by the tile tests ((tile, x, y, z) data)."""
    import warnings

    with warnings.catch_warnings(), optix.config(compile=False):
        warnings.simplefilter("ignore")
        est = GammaMixture(
            nclass=1,
            gain=2.0,
            order=(4, 4),
            max_iterations=it,
            overlap=overlap,
            **kw,
        )
        return est.fit(data, layout=Layout("txyz", spatial="yz"))


def reference_fit(seed=1, it=6):
    """overlap=None fit whose results are pinned to e7d4cf7."""
    tiles, _ = sample_scene(
        [[0, 0, 0], [0, 6, 0], [1, 12, 0]],
        tile=(5, 12, 10),
        seed=seed,
        mu=(80.0, 400.0),
        nu=(3.0, 1.5),
        frac=(0.4, 0.2),
        dtype=torch.float32,
    )
    with optix.config(compile=False):
        est = GammaMixture(nclass=2, gain=2.0, order=(3, 3), max_iterations=it)
        est.fit(tiles, layout=Layout("txyz", spatial="yz"))
    return est
