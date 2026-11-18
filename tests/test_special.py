import math

import numpy as np
import pytest
import torch
from torch.distributions import Dirichlet, Gamma, Normal, kl_divergence

from optix.special import (
    dirichlet_elog,
    fit_dirichlet,
    fit_gamma_shape,
    gauss_hermite,
    inv_digamma,
    kl_dirichlet,
    kl_gamma,
    kl_normal,
)


@pytest.mark.parametrize("n", [1, 2, 5, 20, 60])
def test_gauss_hermite_vs_numpy(n):
    nodes, logw = gauss_hermite(n, dtype=torch.float64)
    ref_x, ref_w = np.polynomial.hermite.hermgauss(n)
    np.testing.assert_allclose(nodes.numpy(), ref_x, atol=1e-10)
    np.testing.assert_allclose(logw.exp().numpy(), ref_w, rtol=1e-8)


def test_gauss_hermite_large_n_no_underflow():
    _, logw = gauss_hermite(200, dtype=torch.float64)
    assert torch.isfinite(logw).all()


@pytest.mark.parametrize("n", [3, 8])
def test_gauss_hermite_polynomial_exactness(n):
    nodes, logw = gauss_hermite(n, dtype=torch.float64)
    w = logw.exp()
    for degree in range(2 * n):
        approx = (w * nodes**degree).sum().item()
        if degree % 2:
            exact = 0.0
        else:  # Gamma((d + 1) / 2)
            exact = math.gamma((degree + 1) / 2)
        assert approx == pytest.approx(exact, rel=1e-9, abs=1e-9)


def test_gauss_hermite_backend(device, dtype):
    nodes, logw = gauss_hermite(4, dtype=dtype, device=device)
    assert nodes.dtype == dtype and logw.dtype == dtype
    assert nodes.device.type == device.type


def test_kl_vs_torch_distributions():
    g = torch.Generator().manual_seed(0)
    a = torch.rand(6, 4, generator=g, dtype=torch.float64) * 5 + 0.1
    b = torch.rand(6, 4, generator=g, dtype=torch.float64) * 5 + 0.1
    torch.testing.assert_close(
        kl_dirichlet(a, b), kl_divergence(Dirichlet(a), Dirichlet(b))
    )
    sq, rq, sp, rp = (
        torch.rand(10, generator=g, dtype=torch.float64) * 4 + 0.1
        for _ in range(4)
    )
    torch.testing.assert_close(
        kl_gamma(sq, rq, sp, rp), kl_divergence(Gamma(sq, rq), Gamma(sp, rp))
    )
    mq, mp = torch.randn(2, 10, generator=g, dtype=torch.float64)
    vq, vp = torch.rand(2, 10, generator=g, dtype=torch.float64) + 0.1
    torch.testing.assert_close(
        kl_normal(mq, vq, mp, vp),
        kl_divergence(Normal(mq, vq.sqrt()), Normal(mp, vp.sqrt())),
    )


@pytest.mark.parametrize("dt", [torch.float32, torch.float64])
def test_fit_gamma_shape_residual(dt):
    nu_true = torch.logspace(-2, 5, 50, dtype=torch.float64)
    c = nu_true.log() - torch.digamma(nu_true)
    nu = fit_gamma_shape(c.to(dt))
    assert nu.dtype == dt
    rtol = 1e-8 if dt == torch.float64 else 1e-3
    torch.testing.assert_close(nu.double(), nu_true, rtol=rtol, atol=0)


def test_fit_gamma_shape_extremes():
    c = torch.tensor([-1.0, 0.0, 1e-30, 50.0, 1e4])
    nu = fit_gamma_shape(c)
    assert torch.isfinite(nu).all() and (nu > 0).all()


@pytest.mark.parametrize("dt", [torch.float32, torch.float64])
def test_fit_gamma_shape_huge_c_finite(dt):
    c = torch.tensor([1e8, 1e10, 1e12, 1e20, 1e30, float("inf")], dtype=dt)
    nu = fit_gamma_shape(c)
    assert nu.dtype == dt
    assert torch.isfinite(nu).all() and (nu > 0).all()


def test_fit_gamma_shape_from_samples():
    g = torch.Generator().manual_seed(1)
    x = torch._standard_gamma(
        torch.full((200_000,), 3.0, dtype=torch.float64), generator=g
    )
    c = x.mean().log() - x.log().mean()
    assert fit_gamma_shape(c).item() == pytest.approx(3.0, rel=0.02)


@pytest.mark.parametrize("dt", [torch.float32, torch.float64])
def test_inv_digamma(dt):
    x = torch.logspace(-3, 4, 40, dtype=dt)
    torch.testing.assert_close(inv_digamma(torch.digamma(x)), x)


@pytest.mark.parametrize("dt", [torch.float32, torch.float64])
def test_fit_dirichlet_recovers_alpha(dt):
    alpha = torch.tensor(
        [[0.5, 2.0, 10.0], [0.05, 0.05, 0.1], [100.0, 300.0, 50.0]],
        dtype=dt,
    )
    fit = fit_dirichlet(dirichlet_elog(alpha), torch.ones_like(alpha))
    assert (fit > 0).all()
    torch.testing.assert_close(fit, alpha, rtol=2e-3, atol=0)


@pytest.mark.parametrize("niter", [16, 64, 500])
def test_fit_dirichlet_float32_cancellation_stays_finite(niter):
    # Regression: a float32 cancellation in the Newton step used to give NaN.
    alpha = torch.tensor([[2e7, 3e4]], dtype=torch.float64)
    elogpi = dirichlet_elog(alpha).float()
    fit = fit_dirichlet(elogpi, torch.ones(2), niter=niter)
    assert fit.dtype == torch.float32
    assert torch.isfinite(fit).all() and (fit > 0).all()


def test_fit_dirichlet_broadcast_init():
    alpha = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    fit = fit_dirichlet(dirichlet_elog(alpha), torch.ones(2))
    torch.testing.assert_close(fit, alpha)
