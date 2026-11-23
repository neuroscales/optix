"""Tests of optix.deconv.RichardsonLucyTV (Dey et al. 2006)."""

import math

import pytest
import torch
from test_deconv import blur, gaussian_psf, poisson_image

from optix import load
from optix.deconv import (
    RichardsonLucy,
    RichardsonLucyTV,
    richardson_lucy,
    richardson_lucy_tv,
)
from optix.deconv.utils import PaddedConvolution, rl_block, tv_curvature
from optix.estimates import SensorEstimate
from optix.io import Layout

F64 = torch.float64


def idiv(truth, est):
    """I-divergence of Dey et al. (2006, Eq. 12), per voxel."""
    est = est.clamp(min=1e-12)
    return float((truth * torch.log(truth / est) - (truth - est)).mean())


def boxes_2d(seed=0):
    truth = torch.full((96, 96), 10.0, dtype=F64)
    truth[20:60, 25:70] = 120.0
    truth[50:80, 50:85] = 60.0
    psf = gaussian_psf(11, 2.0)
    gen = torch.Generator().manual_seed(seed)
    return truth, psf, torch.poisson(blur(truth, psf), generator=gen)


def cylinder_3d(seed=3):
    """Binary cylinder, as in Fig. 1 of the paper (scaled down)."""
    z, y, x = torch.meshgrid(
        torch.arange(24.0),
        torch.arange(32.0),
        torch.arange(32.0),
        indexing="ij",
    )
    truth = torch.full((24, 32, 32), 10.0, dtype=F64)
    inside = ((y - 15.5) ** 2 + (x - 15.5) ** 2 <= 64) & (z >= 6) & (z < 18)
    truth[inside] = 255.0
    k = torch.arange(9.0, dtype=F64) - 4
    kz, ky, kx = torch.meshgrid(k, k, k, indexing="ij")
    psf = torch.exp(-(kx**2 + ky**2) / 2 - kz**2 / 8)
    psf = psf / psf.sum()
    gen = torch.Generator().manual_seed(seed)
    return truth, psf, torch.poisson(blur(truth, psf), generator=gen)


# -- TV term ----------------------------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
def test_tv_curvature_is_minus_tv_gradient(ndim):
    """div(∇f/|∇f|) = −∂TV/∂f for the forward-difference TV."""
    gen = torch.Generator().manual_seed(0)
    f = torch.rand((7,) * ndim, generator=gen, dtype=F64) + 1

    def tv(f):
        dims = range(-ndim, 0)
        g = torch.stack([torch.roll(f, -1, d) - f for d in dims])
        return g.square().sum(0).sqrt().sum()

    x = f.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(tv(x), x)
    torch.testing.assert_close(tv_curvature(f, ndim), -grad)


@pytest.mark.parametrize("ndim", [2, 3])
def test_tv_curvature_bound(ndim):
    """|div n| ≤ d + √d, so λ_TV < 1/(d + √d) keeps the denominator > 0."""
    gen = torch.Generator().manual_seed(1)
    f = torch.rand((9,) * ndim, generator=gen, dtype=F64)
    f[f > 0.5] = 0  # many flat zero regions (vanishing normals)
    div = tv_curvature(f, ndim)
    assert float(div.abs().max()) <= ndim + math.sqrt(ndim) + 1e-12
    assert torch.isfinite(div).all()
    # constant image: no curvature
    assert float(tv_curvature(torch.ones((5,) * ndim), ndim).abs().max()) == 0


def test_lambda_bound_is_enforced():
    with pytest.raises(ValueError, match="lambda_tv"):
        RichardsonLucyTV(gaussian_psf(5, 1.0), lambda_tv=0.3)
    with pytest.raises(ValueError, match="lambda_tv"):
        RichardsonLucyTV(gaussian_psf(5, 1.0, ndim=3), lambda_tv=0.22)
    with pytest.raises(ValueError, match="lambda_tv"):
        RichardsonLucyTV(gaussian_psf(5, 1.0), lambda_tv=-1e-3)
    RichardsonLucyTV(gaussian_psf(5, 1.0), lambda_tv=0.29)


# -- reduction to Richardson–Lucy -------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
def test_lambda_zero_is_richardson_lucy(ndim):
    psf = gaussian_psf(5, 1.0, ndim=ndim)
    shape = (40, 36) if ndim == 2 else (16, 18, 14)
    _, data = poisson_image(shape, psf, seed=2)
    rl = richardson_lucy(data, psf, n_iter=15)
    tv = richardson_lucy_tv(
        data, psf, n_iter=15, lambda_tv=0.0, start="uniform"
    )
    assert torch.equal(rl, tv)
    # the hook with a unit denominator is the plain update, bit for bit
    op = PaddedConvolution(psf, data.shape)
    a, _ = rl_block(data, op, float(data.mean()), n_iter=7)
    b, _ = rl_block(
        data, op, float(data.mean()), n_iter=7, penalty=torch.ones_like
    )
    assert torch.equal(a, b)


# -- the paper's behaviour --------------------------------------------------
def test_tv_suppresses_noise_amplification_2d():
    truth, psf, data = boxes_2d()
    flat = (slice(25, 55), slice(30, 45))  # inside the bright box
    for n in (100, 300):
        rl = richardson_lucy(data, psf, n_iter=n)
        tv = richardson_lucy_tv(data, psf, n_iter=n)
        assert float(tv[flat].std()) < 0.6 * float(rl[flat].std())
        assert idiv(truth, tv) < idiv(truth, rl)


def test_cylinder_3d_paper_behaviour():
    """RL–TV beats RL at RL's best iteration count (paper, Fig. 1)."""
    truth, psf, data = cylinder_3d()
    counts = (10, 20, 50, 100, 200)
    rl = [idiv(truth, richardson_lucy_tv(data, psf, n_iter=n, lambda_tv=0))
          for n in counts]  # fmt: skip
    tv = [idiv(truth, richardson_lucy_tv(data, psf, n_iter=n)) for n in counts]
    # λ_TV = 0.002 (paper): better at every count, and at the optimum
    assert all(t < r for t, r in zip(tv, rl, strict=True))
    assert min(tv) < min(rl)
    # RL diverges with the iterations, RL–TV much more slowly
    assert rl[-1] - min(rl) > 2 * (tv[-1] - min(tv))
    # a stronger weight (the paper uses 0.005 on real data) stabilises
    # the iteration well below RL's best
    strong = richardson_lucy_tv(data, psf, n_iter=200, lambda_tv=0.01)
    assert idiv(truth, strong) < 0.85 * min(rl)


def test_relative_change_stopping_rule():
    psf = gaussian_psf(7, 1.2)
    truth, _ = poisson_image((40, 40), psf, seed=6)
    data = blur(truth, psf)  # noiseless: the iteration settles
    est = RichardsonLucyTV(psf, tol=1e-3, start="uniform").fit(data)
    est.apply(data)
    n = int(est.history.extras["iterations"][0])
    assert 1 < n < 1000
    op = PaddedConvolution(psf, data.shape)
    pen = est._penalty(2)
    a, _ = rl_block(data, op, float(data.mean()), n_iter=n - 1, penalty=pen)
    b, _ = rl_block(data, op, float(data.mean()), n_iter=n, penalty=pen)
    c, _ = rl_block(data, op, float(data.mean()), n_iter=n - 2, penalty=pen)
    rel = float((op.crop(b) - op.crop(a)).norm() / op.crop(a).norm())
    prev = float((op.crop(a) - op.crop(c)).norm() / op.crop(c).norm())
    assert rel <= 1e-3 < prev


# -- blocks, layouts, lazy input ---------------------------------------------
def test_blocks_match_whole_image_2d():
    _, psf, data = boxes_2d(seed=1)
    whole = richardson_lucy_tv(data, psf, n_iter=10, block=(96, 96))
    parts = richardson_lucy_tv(data, psf, n_iter=10, block=(32, 24))
    err = float((parts - whole).abs().max() / whole.abs().max())
    assert err < 1e-3


def test_blocks_match_whole_image_3d():
    psf = gaussian_psf(5, 1.0, ndim=3)
    _, data = poisson_image((20, 24, 22), psf, seed=2)
    whole = richardson_lucy_tv(data, psf, n_iter=10, block=(20, 24, 22))
    parts = richardson_lucy_tv(data, psf, n_iter=10, block=(10, 12, 11))
    err = float((parts - whole).abs().max() / whole.abs().max())
    assert err < 1e-3


def test_halo_must_cover_psf_and_stencil():
    psf = gaussian_psf(9, 1.5)
    data = torch.ones(32, 32, dtype=F64)
    est = RichardsonLucyTV(psf, n_iter=1, halo=9).fit(data)
    with pytest.raises(ValueError, match="halo"):
        est.apply(data)
    assert RichardsonLucyTV(psf)._halo(psf) == (20, 20)


def test_stack_with_layout():
    psf = gaussian_psf(7, 1.2)
    imgs = [poisson_image((32, 28), psf, seed=s)[1] for s in range(3)]
    stack = torch.stack(imgs, dim=-1)
    layout = Layout("yxt", spatial="yx")
    out = richardson_lucy_tv(stack, psf, n_iter=5, layout=layout)
    for i, img in enumerate(imgs):
        torch.testing.assert_close(
            out[..., i], richardson_lucy_tv(img, psf, n_iter=5)
        )


def test_lazy_zarr_input_and_output(tmp_path):
    zarr = pytest.importorskip("zarr")
    psf = gaussian_psf(7, 1.2)
    data = torch.stack(
        [poisson_image((64, 48), psf, seed=s)[1] for s in (4, 5)]
    )
    kw = {"shape": data.shape, "chunks": (1, 16, 16), "dtype": "f4"}
    src = zarr.open_array(str(tmp_path / "in.zarr"), mode="w", **kw)
    src[:] = data.numpy().astype("f4")
    dst = zarr.open_array(str(tmp_path / "out.zarr"), mode="w", **kw)
    opts = {"n_iter": 8, "block": (32, 16), "dtype": torch.float32}
    richardson_lucy_tv(src, psf, out=dst, **opts)
    ref = richardson_lucy_tv(data.float(), psf, **opts)
    torch.testing.assert_close(torch.from_numpy(dst[:]), ref)


def test_float32_matches_float64():
    _, psf, data = boxes_2d(seed=5)
    a = richardson_lucy_tv(data, psf, n_iter=30, dtype=F64)
    b = richardson_lucy_tv(
        data.float(), psf.float(), n_iter=30, dtype="float32"
    )
    assert float((a - b).abs().max() / a.max()) < 1e-3


# -- options ----------------------------------------------------------------
def test_faithful_flags_and_extensions():
    psf = gaussian_psf(5, 1.0)
    data = torch.poisson(torch.full((24, 24), 20.0, dtype=F64))
    est = RichardsonLucyTV(psf)
    assert est.is_faithful
    assert est.options["lambda_tv"] == 0.002
    assert est.options["tol"] == 1e-5
    assert est.options["start"] == "data"
    assert not RichardsonLucyTV(psf, start="uniform").is_faithful
    assert not RichardsonLucyTV(psf, epsilon=1e-3).is_faithful
    with pytest.raises(ValueError, match="start"):
        RichardsonLucyTV(psf, start="zeros")
    sensor = SensorEstimate(offset=100.0, gain=2.0, variance=1.0)
    est = RichardsonLucyTV(psf, n_iter=3).fit(100 + 2 * data, init=sensor)
    assert not est.is_faithful
    torch.testing.assert_close(
        est.apply(100 + 2 * data),
        richardson_lucy_tv(data, psf, n_iter=3),
    )
    assert "alpha" not in est.options
    assert isinstance(est, RichardsonLucy)


def test_save_load(tmp_path):
    psf = gaussian_psf(5, 1.0)
    data = torch.poisson(torch.full((24, 24), 20.0, dtype=F64))
    est = RichardsonLucyTV(psf, n_iter=4, lambda_tv=0.01).fit(data)
    path = tmp_path / "rltv.pt"
    est.save(path)
    back = load(path)
    assert back.options["lambda_tv"] == 0.01
    torch.testing.assert_close(back.apply(data), est.apply(data))
