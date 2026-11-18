"""Building blocks of Foi et al. (2008), checked against the paper."""

import math

import pytest
import torch

from optix.sensor._foi_clip import E_m, E_r, S_m, S_r, clipped_curve
from optix.sensor._foi_fit import Pairs, _curve, log_likelihood, nelder_mead
from optix.sensor._foi_local import (
    d_factor,
    d_mad_factor,
    daubechies3,
    edge_kernels,
    kappa,
    kappa_mad,
    kappa_mad_variance,
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


# -- Tables I and III: estimators on clipped data, finite samples ----------
# Footnote 8: Monte Carlo over constant images clipped from below, with
# the n samples of z^wdet contiguous in the level set (here: along rows).
TABLE_I = {  # E{sigma_hat} / sigma for mu = -1, 0, 1
    2: (0.226, 0.572, 0.868),
    10: (0.247, 0.580, 0.867),
}
TABLE_III = {  # E{sigma_hat^mad} / sigma for mu = -1, 0, 1
    2: (0.215, 0.566, 0.869),
    10: (0.168, 0.543, 0.872),
}


@pytest.mark.parametrize("robust", [False, True], ids=["std", "mad"])
@pytest.mark.parametrize("n", [2, 10])
def test_tables_i_iii_finite_samples(n, robust):
    g = torch.Generator().manual_seed(n)
    table = (TABLE_III if robust else TABLE_I)[n]
    for mu, expected in zip((-1.0, 0.0, 1.0), table, strict=True):
        z = (mu + torch.randn(16, 512, 512, generator=g, dtype=F64)).clamp(0)
        _, wdet = wavelet_coefficients(z)
        w = wdet[..., : wdet.shape[-1] // n * n].reshape(-1, n)
        nn = torch.tensor(float(n), dtype=F64)
        if robust:
            x = w.abs().sort(-1).values
            med = (x[:, (n - 1) // 2] + x[:, n // 2]) / 2
            s = med / kappa_mad(nn)  # Eq. (59)
        else:
            s = w.std(-1) / kappa(nn)  # Eq. (16)
        # 5 Monte Carlo standard errors plus the rounding of the table
        tol = 5 * s.std().item() / math.sqrt(s.numel()) + 5e-4
        assert s.mean().item() == pytest.approx(expected, abs=tol), mu


def test_kappa_mad_variance_factor():
    # E{(kappa_check sigma_hat^mad)^2} = sigma^2 should hold. The paper's
    # 1 + 1/(5n) (Section VI-A.2) is kept as the faithful default, but
    # 1/(1 + d^mad) (mad_correction="exact") is the unbiased factor.
    g = torch.Generator().manual_seed(0)
    reps = 200_000
    for n in (5, 11, 51):
        x = torch.randn(reps, n, generator=g, dtype=F64).abs()
        nn = torch.tensor(float(n), dtype=F64)
        s2 = (x.median(-1).values / kappa_mad(nn)) ** 2
        tol = 5 * s2.std().item() / math.sqrt(reps)
        paper = kappa_mad_variance(nn).item()
        exact = kappa_mad_variance(nn, exact=True).item()
        assert paper == pytest.approx(1 + 1 / (5 * n))
        assert exact == pytest.approx(1 / (1 + 1.35 / (n + 1.5)))
        # Eq. (58) is itself an approximation: allow 1 % on top of MC
        assert exact * s2.mean().item() == pytest.approx(1, abs=tol + 0.01)
        assert paper * s2.mean().item() > 1 + 0.002  # biased upwards


# -- kernels -----------------------------------------------------------------
def test_daubechies_orthogonality():
    # psi1 and the scaling filter h = sqrt(2) phi1 (reversed: Eq. (10)
    # prints phi1 in the opposite order) form an orthonormal pair:
    # orthogonal at all even shifts; psi1 has 3 vanishing moments
    psi1, phi1 = daubechies3()
    h = phi1.flip(0) * math.sqrt(2)
    assert h.square().sum().item() == pytest.approx(1, abs=1e-15)
    assert (h @ psi1).item() == pytest.approx(0, abs=1e-15)
    for k in (2, 4):
        for u, v in ((psi1, psi1), (h, h), (h, psi1), (psi1, h)):
            assert (u[k:] @ v[:-k]).item() == pytest.approx(0, abs=1e-15)
    t = torch.arange(6, dtype=F64)
    for p in range(3):
        assert (psi1 @ t**p).item() == pytest.approx(0, abs=1e-12)


def test_edge_kernels():
    lap, d_row, d_col = edge_kernels()
    assert lap.shape == d_row.shape == d_col.shape == (9, 9)  # Eq. (12)
    assert lap.sum().item() == pytest.approx(0, abs=1e-15)
    torch.testing.assert_close(lap, lap.T)
    torch.testing.assert_close(d_col, d_row.T)
    torch.testing.assert_close(d_row, -d_row.flip(0))
    # response at the centre of a 9x9 patch (convolution = flipped dot)
    t = torch.arange(-4, 5, dtype=F64)
    r, c = torch.meshgrid(t, t, indexing="ij")

    def conv(k, f):
        return (k.flip(0, 1) * f).sum().item()

    # first derivatives of a plane are exact up to the kernel's mass
    gain = conv(d_row, r)
    assert gain == pytest.approx(1, abs=0.05)
    assert conv(d_row, c) == pytest.approx(0, abs=1e-14)
    assert conv(d_col, c) == pytest.approx(gain)
    # the Laplacian of r^2 + c^2 is 4, and constant/linear terms vanish
    assert conv(lap, r * r + c * c) / 4 == pytest.approx(1, abs=0.1)
    assert conv(lap, 3 + r - 2 * c) == pytest.approx(0, abs=1e-12)


# -- likelihood (Eqs. 23, 53) against a dense quadrature --------------------
def _dense_log_integral(a, b, yhat, shat, c, d, clipped, robust, eps=1e-6):
    """log of the integral of Eq. (22)/(52) by zoomed trapezoid rules."""
    if clipped:
        upper = 1.0
        for _ in range(8):
            upper = 1 + 10 * math.sqrt(max(eps * eps, a * upper + b))
    else:
        upper = 1.0
    lo, hi = 0.0, upper
    for _ in range(4):
        y = torch.linspace(lo, hi, 100_001, dtype=F64)
        yt, st = _curve(a, b, y, eps, clipped, robust)
        var = (st * st).clamp_min(torch.finfo(F64).tiny)
        lp = (
            -math.log(2 * math.pi)
            - 0.5 * math.log(c * d)
            - torch.log(var)
            - (yhat - yt) ** 2 / (2 * var * c)
            - (shat - st) ** 2 / (2 * var * d)
        )
        h = (hi - lo) / 100_000
        w = torch.zeros_like(lp)
        w[0] = w[-1] = math.log(0.5)
        value = (torch.logsumexp(lp + w, 0) + math.log(h)).item()
        k = torch.nonzero(lp > lp.max() - 80).squeeze(-1)
        lo = max(lo, y[k[0]].item() - 2 * h)
        hi = min(hi, y[k[-1]].item() + 2 * h)
    return value


@pytest.mark.parametrize(
    "clipped, robust",
    [(True, False), (False, False), (True, True)],
    ids=["clipped", "basic", "clipped-mad"],
)
def test_likelihood_against_dense_quadrature(clipped, robust):
    # level sets from barely to heavily clipped (mu = (1 - y)/sigma down
    # to -3.5), small to large n, estimates off their expectation, and
    # parameters off the truth
    a, b = 0.01, 0.04**2  # Fig. 2
    s1 = math.sqrt(a + b)
    if not clipped:
        ys = (0.02, 0.3, 0.98)
    elif robust:  # S_m^mad (Table V) vanishes below mu = -2.5
        ys = (0.02, 0.5, 1.0, 1 + 1.5 * s1, 1 + 2.0 * s1)
    else:
        ys = (0.0, 0.5, 1.0, 1 + 1.5 * s1, 1 + 2.5 * s1, 1 + 3.5 * s1)
    worst = 0.0
    for n in (20, 2000, 200_000):
        nn = torch.tensor([float(n)], dtype=F64)
        c = 0.25 / nn
        d = d_mad_factor(nn) if robust else d_factor(nn)
        for y in ys:
            yt, st = _curve(
                a, b, torch.tensor([y], dtype=F64), 1e-6, clipped, robust
            )
            for dy, ds in ((0.0, 0.0), (1.0, -1.0), (-2.0, 1.5)):
                pairs = Pairs(
                    yhat=yt + dy * st * c.sqrt(),
                    shat=st * (1 + ds * d.sqrt()),
                    vhat=st**2,
                    c=c,
                    d=d,
                    count=nn,
                )
                for fa, fb in ((1.0, 1.0), (1.03, 0.95)):
                    kw = {"clipped": clipped, "robust": robust, "eps": 1e-6}
                    got = log_likelihood(a * fa, b * fb, pairs, **kw)
                    ref = _dense_log_integral(
                        a * fa,
                        b * fb,
                        pairs.yhat.item(),
                        pairs.shat.item(),
                        c.item(),
                        d.item(),
                        clipped,
                        robust,
                    )
                    worst = max(worst, abs(got - ref))
    assert worst < 1e-6


# -- Fig. 12: the clipped test case of the paper ------------------------------
def _fig2_like(seed, size=512):
    """A piecewise smooth image in [0, 1] clipped at both ends, as Fig. 2.

    The image of Fig. 2 is not available; this one has the same size,
    noise (chi = 100, so a = 0.01, and b = 0.04^2) and clipping: smooth
    background, a saturated disc, a dark ramp clipped at 0 and a bright
    ramp across 1.
    """
    t = torch.linspace(0, 1, size, dtype=F64)
    r, c = torch.meshgrid(t, t, indexing="ij")
    y = 0.15 + 0.55 * c + 0.1 * torch.sin(3 * r)
    disc = (r - 0.3) ** 2 + (c - 0.3) ** 2
    y = torch.where(disc < 0.15**2, 1.1 - 0.4 * disc / 0.15**2, y)
    dark = (r > 0.6) & (r < 0.9) & (c > 0.1) & (c < 0.4)
    y = torch.where(dark, 0.02 + 0.1 * (c - 0.1), y)
    bright = (r > 0.55) & (r < 0.85) & (c > 0.6) & (c < 0.9)
    y = torch.where(bright, 0.95 + 0.5 * (r - 0.55), y)
    g = torch.Generator().manual_seed(seed)
    chi = 100.0  # Eq. (2): z = chi^-1 P(chi y) + N(0, b)
    z = torch.poisson(chi * y, generator=g) / chi
    z = z + 0.04 * torch.randn(y.shape, generator=g, dtype=F64)
    return z.clamp(0, 1)[None]


def test_fig12_clipped_single_image():
    # faithful defaults on a 512 x 512 image clipped on both sides: the
    # data range is [0, 1] by itself. Fig. 12 reports a = 0.00995 and
    # b = 0.001552 for a = 0.01, b = 0.0016 (errors of 0.5 % and 3 %).
    from optix.sensor import PoissonGaussian

    z = _fig2_like(0)
    assert z.min() == 0 and z.max() == 1
    est = PoissonGaussian().fit(z)
    assert est.is_faithful and est.history.converged
    s = est.estimates["sensor"]
    se = est.covariance[()].diagonal().sqrt()
    # the single image measures a to about 1.5 % and b to about 4 %
    assert se[0] < 0.02 * 0.01 and se[1] < 0.05 * 0.0016
    assert abs(s.gain.item() - 0.01) < 3 * se[0]
    assert abs(s.variance.item() - 0.0016) < 3 * se[1]
