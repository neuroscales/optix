"""Building blocks of Foi et al. (2008), checked against the paper."""

import math

import pytest
import torch

from optix.sensor._foi_clip import E_m, E_r, S_m, S_r, clipped_curve
from optix.sensor._foi_fit import nelder_mead
from optix.sensor._foi_local import (
    d_factor,
    d_mad_factor,
    daubechies3,
    kappa,
    kappa_mad,
    level_labels,
    local_pairs,
    wavelet_coefficients,
)

F64 = torch.float64


def test_daubechies_filters_match_eq10():
    psi1, phi1 = daubechies3()
    eq10_psi = [0.035, 0.085, -0.135, -0.460, 0.807, -0.333]
    eq10_phi = [0.025, -0.060, -0.095, 0.325, 0.571, 0.235]
    torch.testing.assert_close(
        psi1, torch.tensor(eq10_psi, dtype=F64), atol=5e-4, rtol=0
    )
    torch.testing.assert_close(
        phi1, torch.tensor(eq10_phi, dtype=F64), atol=5e-4, rtol=0
    )
    # Section III-A: sum psi = 0, ||psi||_2 = 1, sum phi = 1
    assert psi1.sum().item() == pytest.approx(0, abs=1e-15)
    assert psi1.square().sum().item() == pytest.approx(1, abs=1e-15)
    assert phi1.sum().item() == pytest.approx(1, abs=1e-15)
    assert phi1.square().sum().item() == pytest.approx(0.5, abs=1e-15)


def test_wavelet_noise_statistics():
    # Eqs. (7)-(8): std{z^wdet} = sigma, std{z^wapp} = ||phi||_2 sigma
    g = torch.Generator().manual_seed(0)
    z = 0.3 + 0.05 * torch.randn(1, 512, 512, generator=g, dtype=F64)
    wapp, wdet = wavelet_coefficients(z)
    n = wdet.numel()
    # 5 standard errors of a sample standard deviation, sd / sqrt(2n)
    tol = 5 * 0.05 / math.sqrt(2 * n)
    assert wdet.std().item() == pytest.approx(0.05, abs=tol)
    assert wapp.std().item() == pytest.approx(0.025, abs=tol)
    assert wapp.mean().item() == pytest.approx(0.3, abs=1e-3)


def test_kappa_eq17():
    n = torch.arange(2, 60, dtype=F64)
    direct = torch.exp(
        0.5 * torch.log(2 / (n - 1))
        + torch.lgamma(n / 2)
        - torch.lgamma((n - 1) / 2)
    )
    torch.testing.assert_close(kappa(n), direct, rtol=1e-12, atol=0)
    big = torch.tensor([1e3, 1e5, 1e8], dtype=F64)
    series = 1 - 1 / (4 * big) - 7 / (32 * big**2)
    torch.testing.assert_close(kappa(big), series, rtol=1e-9, atol=0)
    d = d_factor(big)
    torch.testing.assert_close(
        d, 1 / (2 * big) + 5 / (8 * big**2), rtol=1e-5, atol=0
    )  # Eq. (19)
    torch.testing.assert_close(
        d_factor(n), (1 - direct**2) / direct**2, rtol=1e-10, atol=0
    )


def test_kappa_mad_table_ii_monte_carlo():
    # E{median |X|} over n i.i.d. N(0,1), the definition of Table II
    g = torch.Generator().manual_seed(0)
    reps = 200_000
    for n in (1, 2, 3, 4, 7, 12):
        x = torch.randn(reps, n, generator=g, dtype=F64).abs()
        s = x.sort(-1).values
        med = (s[:, (n - 1) // 2] + s[:, n // 2]) / 2
        # 5 standard errors plus the rounding of the table (3 decimals)
        tol = 5 * med.std().item() / math.sqrt(reps) + 5e-4
        assert kappa_mad(torch.tensor(n)).item() == pytest.approx(
            med.mean().item(), abs=tol
        )
    assert kappa_mad(torch.tensor(1000.0)).item() == pytest.approx(
        0.6744897501960817 + 1 / 5000
    )
    assert d_mad_factor(torch.tensor(8.5)).item() == pytest.approx(0.135)


def test_clipped_expectation_eq32_33():
    mu = torch.linspace(-4, 6, 41, dtype=F64)
    Phi = torch.special.ndtr(mu)
    phi = torch.exp(-mu * mu / 2) / math.sqrt(2 * math.pi)
    torch.testing.assert_close(E_m(mu), Phi * mu + phi)
    var = Phi + phi * mu - phi**2 + Phi * mu * (mu - Phi * mu - 2 * phi)
    torch.testing.assert_close(
        S_m(mu), var.clamp_min(0).sqrt(), atol=1e-7, rtol=1e-7
    )
    # Table I, last row: S_m(mu) for mu = -1, -0.5, ..., 3
    table = [0.262, 0.413, 0.584, 0.744, 0.867, 0.943, 0.980, 0.994, 0.999]
    mus = torch.tensor([-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3], dtype=F64)
    torch.testing.assert_close(
        S_m(mus), torch.tensor(table, dtype=F64), atol=5e-4, rtol=0
    )


def test_table_iv_interpolants_invert_eq32_33():
    # E_r and S_r (Table IV) must invert the exact E_m, S_m on the
    # accuracy range of Section VIII-D: checks the transcription.
    mu = torch.linspace(-1.5, 4, 56, dtype=F64)
    e, s = E_m(mu), S_m(mu)
    rho = e / s
    ok = (rho > 0.65) & (rho < 11)
    torch.testing.assert_close(S_r(rho[ok]), s[ok], atol=1e-3, rtol=0)
    torch.testing.assert_close(E_r(rho[ok]), mu[ok] / e[ok], atol=5e-3, rtol=0)


def test_table_v_interpolants():
    # S_m^mad against Table III, last row (n = 500000)
    mus = torch.tensor([-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 5], dtype=F64)
    table = [
        0.154,
        0.329,
        0.536,
        0.731,
        0.873,
        0.953,
        0.986,
        0.997,
        0.999,
        1.000,
    ]
    torch.testing.assert_close(
        S_m(mus, robust=True),
        torch.tensor(table, dtype=F64),
        atol=1.5e-3,
        rtol=0,
    )
    # E_r^mad and S_r^mad invert (E_m, S_m^mad) where well-conditioned
    mu = torch.linspace(0.5, 3, 11, dtype=F64)
    e, s = E_m(mu), S_m(mu, robust=True)
    rho = e / s
    torch.testing.assert_close(S_r(rho, robust=True), s, atol=2e-3, rtol=0)
    torch.testing.assert_close(
        E_r(rho, robust=True), mu / e, atol=5e-3, rtol=0
    )


def test_clipped_curve_limits():
    y = torch.tensor([0.3, 0.5, 0.7], dtype=F64)
    s = torch.full_like(y, 1e-3)
    yt, st = clipped_curve(y, s)
    torch.testing.assert_close(yt, y)
    torch.testing.assert_close(st, s)
    # single-sided clipping of N(y, s^2) at 0 (Eqs. 34-35)
    y = torch.tensor([0.0, 0.02], dtype=F64)
    s = torch.full_like(y, 0.05)
    g = torch.Generator().manual_seed(0)
    z = (
        y[:, None] + s[:, None] * torch.randn(2, 10**6, generator=g, dtype=F64)
    ).clamp(0, 1)
    yt, st = clipped_curve(y, s)
    torch.testing.assert_close(yt, z.mean(1), atol=5 * 0.05 / 1e3, rtol=0)
    torch.testing.assert_close(st, z.std(1), atol=5 * 0.05 / 1e3, rtol=0)


def test_level_labels_eq13():
    delta = 0.25  # u_j = 0.25 j, j = 1..4
    zsmo = torch.tensor([0.1, 0.13, 0.37, 0.38, 0.99, 1.2, 0.6])
    mask = torch.tensor([True] * 6 + [False])
    labels = level_labels(zsmo, mask, delta)
    assert labels.tolist() == [-1, 0, 0, 1, 3, -1, -1]


def test_local_pairs():
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(-1, 5, (1000,), generator=g)
    labels[labels == 4] = 3
    labels[0] = 4  # a singleton set, dropped
    labels[labels == 4] = 4
    wapp = torch.rand(1000, generator=g, dtype=F64)
    wdet = torch.randn(1000, generator=g, dtype=F64)
    r = local_pairs(labels, wapp, wdet, 5, robust=True)
    assert len(r["count"]) == 4
    for i in range(4):
        sel = labels == i
        assert r["count"][i].item() == sel.sum().item()
        assert r["mean"][i].item() == pytest.approx(wapp[sel].mean().item())
        assert r["std"][i].item() == pytest.approx(wdet[sel].std().item())
        x = wdet[sel].abs().sort().values
        n = len(x)
        med = (x[(n - 1) // 2] + x[n // 2]) / 2
        assert r["mad"][i].item() == pytest.approx(med.item())


def test_nelder_mead_quadratic():
    target = torch.tensor([3.0, -2.0], dtype=F64)

    def f(x):
        d = x - target
        return float(d[0] ** 2 + 10 * d[1] ** 2 + d[0] * d[1])

    x, fx, ok, _ = nelder_mead(
        f,
        torch.zeros(2, dtype=F64),
        torch.ones(2, dtype=F64),
        max_iter=1000,
        tol=1e-12,
    )
    assert ok
    torch.testing.assert_close(x, target, atol=1e-5, rtol=0)
