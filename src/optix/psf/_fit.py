"""Sub-voxel centres of bead crops."""

import torch
from torch import Tensor

__all__ = ["fit_gaussian", "threshold_centroid"]


def _grids(shape: tuple[int, ...], like: Tensor) -> list[Tensor]:
    """Voxel coordinates of each axis, broadcastable to (M, *shape)."""
    out = []
    for i, n in enumerate(shape):
        view = [1] * (len(shape) + 1)
        view[i + 1] = n
        out.append(
            torch.arange(n, dtype=like.dtype, device=like.device).view(view)
        )
    return out


def _initial_sigma(x: Tensor, center: Tensor, grids: list[Tensor]) -> Tensor:
    """Second moments of the part of each crop above 20 % of its peak."""
    peak = x.flatten(1).amax(1).view([-1] + [1] * len(grids))
    w = (x - 0.2 * peak).clamp(min=0)
    total = w.flatten(1).sum(1).clamp(min=torch.finfo(x.dtype).tiny)
    out = []
    for i, g in enumerate(grids):
        c = center[:, i].view([-1] + [1] * len(grids))
        var = (w * (g - c) ** 2).flatten(1).sum(1) / total
        out.append(var.sqrt())
    return torch.stack(out, -1).clamp(min=0.5)


def fit_gaussian(
    x: Tensor, *, n_iter: int = 50
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Least-squares fit of an axis-aligned Gaussian plus a constant.

        m(r) = A · exp(−∑ᵢ (rᵢ − cᵢ)² / (2 sᵢ²)) + B,

    fitted to each crop by Levenberg–Marquardt, with the damping adapted
    per crop (multiplied by 3 on a rejected step, divided by 3 on an
    accepted one). Each crop is scaled to unit peak before the fit.

    Parameters
    ----------
    x : (M, *K) Tensor
        Crops (one bead each), background-subtracted or not.
    n_iter : int
        Number of iterations.

    Returns
    -------
    center : (M, D) Tensor
        Centres c in voxel coordinates of the crop.
    sigma : (M, D) Tensor
        Standard deviations s, in voxels.
    amplitude : (M,) Tensor
        A, in the units of `x`.
    offset : (M,) Tensor
        B, in the units of `x`.
    """
    shape = tuple(x.shape[1:])
    ndim = len(shape)
    m = x.shape[0]
    grids = _grids(shape, x)
    scale = x.flatten(1).amax(1).clamp(min=torch.finfo(x.dtype).tiny)
    y = (x / scale.view([-1] + [1] * ndim)).flatten(1)
    center = torch.tensor([(n - 1) / 2 for n in shape], dtype=x.dtype)
    center = center.to(x.device).expand(m, ndim).clone()
    sigma = _initial_sigma(y.view(x.shape), center, grids)
    theta = torch.cat(
        [
            torch.ones(m, 1, **_like(x)),
            torch.zeros(m, 1, **_like(x)),
            center,
            sigma,
        ],
        dim=1,
    )
    lo = torch.cat(
        [
            torch.tensor([0.0, -float("inf")], **_like(x)),
            torch.zeros(ndim, **_like(x)),
            torch.full((ndim,), 0.25, **_like(x)),
        ]
    )
    hi = torch.cat(
        [
            torch.full((2,), float("inf"), **_like(x)),
            torch.tensor([n - 1.0 for n in shape], **_like(x)),
            torch.tensor([2.0 * n for n in shape], **_like(x)),
        ]
    )

    def residual(theta: Tensor) -> tuple[Tensor, Tensor]:
        a, b = theta[:, 0], theta[:, 1]
        c, s = theta[:, 2 : 2 + ndim], theta[:, 2 + ndim :]
        g = torch.ones(x.shape, **_like(x))
        diffs = []
        for i, grid in enumerate(grids):
            view = [-1] + [1] * ndim
            d = (grid - c[:, i].view(view)) / s[:, i].view(view)
            diffs.append(d)
            g = g * torch.exp(-0.5 * d**2)
        av = a.view([-1] + [1] * ndim)
        model = av * g + b.view([-1] + [1] * ndim)
        jac = [g, torch.ones_like(g)]
        jac += [
            av * g * d / s[:, i].view([-1] + [1] * ndim)
            for i, d in enumerate(diffs)
        ]
        jac += [
            av * g * d**2 / s[:, i].view([-1] + [1] * ndim)
            for i, d in enumerate(diffs)
        ]
        jac = torch.stack([j.flatten(1) for j in jac], -1)
        return model.flatten(1) - y, jac

    lam = torch.full((m,), 1e-3, **_like(x))
    r, jac = residual(theta)
    cost = (r**2).sum(1)
    eye = torch.eye(theta.shape[1], **_like(x))
    for _ in range(n_iter):
        h = jac.transpose(1, 2) @ jac
        grad = (jac.transpose(1, 2) @ r[..., None])[..., 0]
        diag = torch.diagonal(h, dim1=1, dim2=2)
        damp = lam[:, None, None] * torch.diag_embed(diag) + 1e-12 * eye
        step = torch.linalg.solve(h + damp, -grad[..., None])[..., 0]
        trial = torch.maximum(torch.minimum(theta + step, hi), lo)
        r_new, jac_new = residual(trial)
        cost_new = (r_new**2).sum(1)
        better = cost_new < cost
        theta = torch.where(better[:, None], trial, theta)
        r = torch.where(better[:, None], r_new, r)
        jac = torch.where(better[:, None, None], jac_new, jac)
        cost = torch.where(better, cost_new, cost)
        lam = torch.where(better, lam / 3, lam * 3).clamp(1e-9, 1e9)
    return (
        theta[:, 2 : 2 + ndim],
        theta[:, 2 + ndim :],
        theta[:, 0] * scale,
        theta[:, 1] * scale,
    )


def threshold_centroid(x: Tensor, *, level: float = 0.25) -> Tensor:
    """Centroid of the part of each crop above a fraction of its peak.

        c = ∑ᵣ r · w(r) / ∑ᵣ w(r),   w = max(x − level · max x, 0).

    For a PSF symmetric about its centre, the clipped weights are too,
    so the centroid is not pulled towards the crop centre.

    Parameters
    ----------
    x : (M, *K) Tensor
        Background-subtracted crops.
    level : float
        Fraction of the peak subtracted before taking the centroid.

    Returns
    -------
    (M, D) Tensor
        Centres in voxel coordinates of the crop.
    """
    shape = tuple(x.shape[1:])
    grids = _grids(shape, x)
    peak = x.flatten(1).amax(1).view([-1] + [1] * len(shape))
    w = (x - level * peak).clamp(min=0)
    total = w.flatten(1).sum(1).clamp(min=torch.finfo(x.dtype).tiny)
    return torch.stack([(w * g).flatten(1).sum(1) / total for g in grids], -1)


def _like(x: Tensor) -> dict:
    return {"dtype": x.dtype, "device": x.device}
