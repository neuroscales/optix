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


# -- cancellation-free building blocks ------------------------------------


def test_stirling_residual_vs_lgamma():
    from optix.special import stirling_residual

    a = torch.cat(
        [torch.logspace(-3, 1, 400), torch.logspace(1, 8, 400)]
    ).double()
    direct = torch.lgamma(a) - (
        (a - 0.5) * a.log() - a + 0.5 * math.log(2 * math.pi)
    )
    small = a < 1e3  # direct difference still accurate in float64
    torch.testing.assert_close(stirling_residual(a)[small], direct[small])
    big = torch.tensor([1e6, 1e8], dtype=torch.float32)
    torch.testing.assert_close(stirling_residual(big), 1 / (12 * big))
    x = torch.tensor([1e-6, 0.5, 10.0, 1e7], requires_grad=True)
    (g,) = torch.autograd.grad(stirling_residual(x).sum(), x)
    assert torch.isfinite(g).all()


def test_ratio_minus_log():
    from optix.special import ratio_minus_log

    y = torch.cat(
        [torch.logspace(-30, 30, 2001), 1 + torch.linspace(-0.3, 0.3, 2001)]
    ).double()
    x = y - 1
    series = sum((-1) ** k * x**k / k for k in range(2, 8))
    ref = torch.where(x.abs() < 1e-3, series, x - torch.log1p(x))
    ref = torch.where(x.abs() > 0.5, y - 1 - y.log(), ref)
    torch.testing.assert_close(ratio_minus_log(y), ref, rtol=1e-12, atol=0)
    # float32 near 1: relative precision kept (naive y - 1 - log y is 0)
    num = torch.tensor([1.0001, 0.9999, 1.0], dtype=torch.float32)
    out = ratio_minus_log(num, torch.ones(3))
    ref = ratio_minus_log(num.double(), torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=0)
    # ratios too small for 1 + x stay finite
    tiny = ratio_minus_log(torch.tensor([1e-30]), torch.tensor([1.0]))
    torch.testing.assert_close(tiny, torch.tensor([1e-30 - 1 + 69.07755]))
    y = torch.tensor([0.5, 1.0, 1.1, 3.0], requires_grad=True)
    (g,) = torch.autograd.grad(ratio_minus_log(y).sum(), y)
    torch.testing.assert_close(g, 1 - 1 / y.detach())


@pytest.mark.parametrize("big", [1e4, 1e6])
def test_kl_float32_large_parameters(big):
    f = torch.float32
    aq = torch.tensor([big, 3 * big], dtype=f)
    ap = (aq.double() * 1.002).float()
    bq = torch.tensor([1.0, big], dtype=f)
    bp = (bq.double() * 0.998).float()
    ref = kl_divergence(
        Gamma(aq.double(), bq.double()), Gamma(ap.double(), bp.double())
    )
    out = kl_gamma(aq, bq, ap, bp)
    assert out.dtype == f
    torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-5)
    al = torch.tensor([[1.0, 2.0, 5.0]]) * big
    be = (al.double() * torch.tensor([1.001, 0.998, 1.0])).float()
    ref = kl_divergence(Dirichlet(al.double()), Dirichlet(be.double()))
    torch.testing.assert_close(
        kl_dirichlet(al, be).double(), ref, rtol=1e-5, atol=1e-5
    )
    vq = torch.tensor([1e-4, 1.0, 1e4], dtype=f)
    vp = (vq.double() * 1.0004).float()
    ref = kl_divergence(
        Normal(0.0, vq.double().sqrt()), Normal(0.0, vp.double().sqrt())
    )
    torch.testing.assert_close(
        kl_normal(torch.zeros(3), vq, torch.zeros(3), vp).double(),
        ref,
        rtol=1e-4,
        atol=1e-10,
    )
