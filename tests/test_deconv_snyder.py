"""Tests of optix.deconv.Snyder (Snyder, Hammoud & White 1993)."""

import warnings

import numpy as np
import pytest
import torch
from test_deconv import blur, gaussian_psf, poisson_image, scene

from optix.deconv import RichardsonLucy, Snyder, richardson_lucy
from optix.deconv.utils import PaddedConvolution, rl_block, rl_step
from optix.estimates import SensorEstimate, ShadingEstimate
from optix.special import poisson_normal_logpdf, poisson_normal_mean

F64 = torch.float64


def ccd(truth, psf, gen, *, flat=None, dark=0.0, var=0.0, gain=1.0, off=0.0):
    """ADU data off + gain·(Poisson(β·H λ + dark) + N(0, var))."""
    mean = blur(truth, psf)
    if flat is not None:
        mean = flat * mean
    e = torch.poisson(mean + dark, generator=gen)
    noise = var**0.5 * torch.randn(e.shape, generator=gen, dtype=F64)
    return off + gain * (e + noise)


def obj(shape, level):
    """Extended object: plateaus, a bright bar and a dark gap."""
    h, w = shape
    t = torch.full(shape, 0.2 * level, dtype=F64)
    t[h // 5 : -h // 5, w // 5 : -w // 5] = level
    t[h // 3 : h // 3 + 6, w // 4 : -w // 4] = 3 * level
    t[-h // 3 : -h // 3 + 3, w // 3 : -w // 3] = 0.0
    return t


def eq15(r, op, f, beta, mu0):
    """One step of Snyder et al. (1993), Eq. (15), written out."""
    bbar = op.adjoint(beta)
    valid = op.norm > op.norm.max() * torch.finfo(F64).eps * 16
    new = f * op.adjoint(beta * r / (beta * op.forward(f) + mu0))
    return torch.where(valid, new / bbar, 0.0)


def start(op, level):
    valid = op.norm > op.norm.max() * torch.finfo(F64).eps * 16
    return torch.where(valid, level, 0.0).to(F64)


# -- Eq. (15) ---------------------------------------------------------------
def test_eq15_with_nonuniform_flat():
    """σ = 0: Eq. (15) with β(j), μ₀(j) and β̄ = Hᵀβ, two steps."""
    psf = gaussian_psf(7, 1.2)
    truth, gen = scene((40, 36), seed=1, base=3.0)
    beta = 0.3 + 0.7 * torch.rand(40, 36, generator=gen, dtype=F64)
    mu0 = 2.0 + torch.rand(40, 36, generator=gen, dtype=F64)
    r = ccd(truth, psf, gen, flat=beta, dark=mu0)
    for noise in ("exact", "shifted-poisson"):
        est = Snyder(
            psf, n_iter=2, flat=beta, background=mu0, noise=noise, dtype=F64
        ).fit(r)
        out = est.apply(r)
        level = float((r - mu0).sum() / beta.sum())
        torch.testing.assert_close(float(est.level), level)
        op = PaddedConvolution(psf, r.shape)
        f = start(op, level)
        for _ in range(2):
            f = eq15(r, op, f, beta, mu0)
        torch.testing.assert_close(out, op.crop(f))


def test_shifted_poisson_eq15_units():
    """Snyder et al. (1993), Eq. (15) with μ₀ → μ₀ + σ², in electrons.

    ADU data y = m + g n + e, e ~ N(0, v): the shifted data are
    (y − m)/g + v/g² and the background b/g + v/g², with a flat field.
    """
    psf = gaussian_psf(7, 1.2)
    truth, gen = scene((40, 36), seed=11, base=3.0)
    gain = torch.linspace(1.5, 2.5, 36, dtype=F64).expand(40, 36)
    offset = torch.full((40, 36), 100.0, dtype=F64)
    var_adu = torch.tensor(80.0, dtype=F64)
    b_adu = 12.0
    beta = 0.5 + 0.5 * torch.rand(40, 36, generator=gen, dtype=F64)
    electrons = torch.poisson(
        beta * blur(truth, psf) + b_adu / gain, generator=gen
    )
    noise = var_adu.sqrt() * torch.randn(electrons.shape, generator=gen)
    adu = offset + gain * electrons + noise.to(F64)
    sensor = SensorEstimate(offset=offset, gain=gain, variance=var_adu)
    est = Snyder(
        psf,
        n_iter=1,
        noise="shifted-poisson",
        flat=beta,
        background=b_adu,
        dtype=F64,
    ).fit(adu, init=sensor)
    out = est.apply(adu)
    sigma2 = var_adu / gain.square()
    shifted = ((adu - offset) / gain + sigma2).clamp(min=0)
    mu0 = b_adu / gain + sigma2
    level = float((shifted - mu0).sum() / beta.sum())
    torch.testing.assert_close(float(est.level), level)
    op = PaddedConvolution(psf, adu.shape)
    f1 = eq15(shifted, op, start(op, level), beta, mu0)
    torch.testing.assert_close(out, op.crop(f1))


def test_exact_eq18():
    """Eq. (18): Eq. (15) with r replaced by F[r, μₖ, σ] of Eq. (21)."""
    psf = gaussian_psf(7, 1.2)
    truth, gen = scene((32, 30), seed=2, base=1.0, peak=30.0)
    beta = 0.4 + 0.6 * torch.rand(32, 30, generator=gen, dtype=F64)
    var = 0.5
    r = ccd(truth, psf, gen, flat=beta, dark=1.0, var=var)
    sensor = SensorEstimate(offset=0.0, gain=1.0, variance=var)
    shading = ShadingEstimate(flat=3 * beta, dark=torch.tensor(1.0))
    est = Snyder(psf, n_iter=3, dtype=F64).fit(r, init=[sensor, shading])
    out = est.apply(r)
    op = PaddedConvolution(psf, r.shape)
    b = beta / beta.mean()  # the shading flat is rescaled to unit mean
    f = start(op, float((r - 1.0).sum() / b.sum()))
    for _ in range(3):
        mu = b * op.forward(f) + 1.0
        f = eq15(poisson_normal_mean(r, mu, var), op, f, b, 1.0)
    torch.testing.assert_close(out, op.crop(f))


# -- reductions -------------------------------------------------------------
@pytest.mark.parametrize("noise", ["exact", "shifted-poisson"])
def test_reduces_to_richardson_lucy(noise):
    """β = 1, μ₀ = 0, σ = 0: the Richardson–Lucy iteration and stop."""
    psf = gaussian_psf(7, 1.2)
    _, data = poisson_image((40, 36), psf, seed=6)
    sensor = SensorEstimate(offset=0.0, gain=1.0, variance=0.0)
    for n_iter in (10, None):
        ref = RichardsonLucy(psf, n_iter=n_iter, dtype=F64).fit(data)
        out = Snyder(
            psf, n_iter=n_iter, noise=noise, flat=torch.ones(()), dtype=F64
        ).fit(data, init=sensor)
        torch.testing.assert_close(out.apply(data), ref.apply(data))
        assert (
            out.history.extras["iterations"]
            == (ref.history.extras["iterations"])
        )


def test_small_read_noise_tends_to_eq15():
    """σ → 0: F → r − m and Eq. (18) becomes Eq. (15) on r − m."""
    psf = gaussian_psf(7, 1.2)
    truth, gen = scene((32, 32), seed=3, base=2.0)
    r = ccd(truth, psf, gen, off=10.0)  # integer counts plus a bias
    ref = Snyder(psf, n_iter=5, dtype=F64).fit(
        r, init=SensorEstimate(offset=10.0, gain=1.0)
    )
    tiny = Snyder(psf, n_iter=5, dtype=F64).fit(
        r, init=SensorEstimate(offset=10.0, gain=1.0, variance=1e-4)
    )
    torch.testing.assert_close(tiny.apply(r), ref.apply(r))


# -- EM ---------------------------------------------------------------------
def test_exact_likelihood_increases():
    """EM: the log-likelihood of Eq. (10) never decreases."""
    psf = gaussian_psf(7, 1.5)
    truth, gen = scene((32, 32), seed=4, base=0.5, peak=20.0)
    beta = 0.3 + 0.7 * torch.rand(32, 32, generator=gen, dtype=F64)
    mu0, var = 0.7, 0.8
    r = ccd(truth, psf, gen, flat=beta, dark=mu0, var=var)
    op = PaddedConvolution(psf, r.shape)
    bbar = op.adjoint(beta)
    valid = bbar > bbar.max() * 1e-12
    inv = torch.where(valid, 1 / bbar.clamp(min=1e-300), 0)
    f = start(op, float((r - mu0).sum() / beta.sum()))
    prev = -np.inf
    for _ in range(30):
        mu = beta * op.forward(f) + mu0
        ll = float(poisson_normal_logpdf(r, mu, var).sum())
        assert ll >= prev - 1e-9 * abs(ll)
        prev = ll
        data = poisson_normal_mean(r, mu, var)
        f = rl_step(f, data, op, inv_norm=inv, background=mu0, flat=beta)
        assert (f >= 0).all()
    # a constrained maximum: the likelihood gain has levelled off
    mu = beta * op.forward(f) + mu0
    assert float(poisson_normal_logpdf(r, mu, var).sum()) - prev < 5.0


# -- recovery ---------------------------------------------------------------
def test_exact_beats_shifted_poisson_at_low_counts():
    """A few tenths of an electron per pixel, σ = 0.5 e⁻."""
    psf = gaussian_psf(7, 1.2)
    inner = (slice(7, -7),) * 2
    errs = {"exact": 0.0, "shifted-poisson": 0.0}
    for seed in range(3):
        gen = torch.Generator().manual_seed(seed)
        truth = obj((48, 48), 0.2)
        r = ccd(truth, psf, gen, var=0.25)
        sensor = SensorEstimate(offset=0.0, gain=1.0, variance=0.25)
        for noise in errs:
            est = Snyder(psf, n_iter=100, noise=noise, dtype=F64)
            out = est.fit(r, init=sensor).apply(r)
            errs[noise] += float((out - truth)[inner].norm())
    assert errs["exact"] < 0.9 * errs["shifted-poisson"]


def test_flat_in_model_beats_flat_division():
    """β in the Poisson mean, not divided out of the data (Section 1)."""
    psf = gaussian_psf(9, 1.5)
    inner = (slice(9, -9),) * 2
    model = division = 0.0
    for seed in range(3):
        gen = torch.Generator().manual_seed(seed)
        truth = obj((64, 64), 2.0)
        beta = 0.3 + 0.7 * torch.rand(64, 64, generator=gen, dtype=F64)
        dark, var, gain, off = 0.5, 0.25, 2.0, 100.0
        adu = ccd(truth, psf, gen, flat=beta, dark=dark, var=var, gain=gain)
        adu = adu + off
        sensor = SensorEstimate(offset=off, gain=gain, variance=var * gain**2)
        shading = ShadingEstimate(flat=beta, dark=off + gain * dark)
        out = Snyder(psf, n_iter=50, flat=beta, dtype=F64)
        out = out.fit(adu, init=[sensor, shading]).apply(adu)
        model += float((out - truth)[inner].norm())
        pre = ((adu - off) / gain - dark) / beta
        flat = SensorEstimate(offset=0.0, gain=1.0, variance=var)
        out = Snyder(psf, n_iter=50, dtype=F64).fit(pre, init=flat).apply(pre)
        division += float((out - truth)[inner].norm())
    assert model < 0.85 * division


# -- camera terms -----------------------------------------------------------
def test_dark_is_poisson_background():
    """ShadingEstimate.dark includes the offset: (dark − offset)/gain → μ₀."""
    psf = gaussian_psf(7, 1.2)
    truth, gen = scene((32, 32), seed=5, base=2.0)
    gain, off = 2.0, 100.0
    d = 1.0 + torch.rand(32, 32, generator=gen, dtype=F64)  # electrons
    adu = ccd(truth, psf, gen, dark=d, var=1.0, gain=gain, off=off)
    sensor = SensorEstimate(offset=off, gain=gain, variance=gain**2)
    shading = ShadingEstimate(dark=off + gain * d)
    a = Snyder(psf, n_iter=5, dtype=F64).fit(adu, init=[sensor, shading])
    b = Snyder(psf, n_iter=5, background=gain * d, dtype=F64)
    b = b.fit(adu, init=sensor)
    torch.testing.assert_close(a.apply(adu), b.apply(adu))
    # a flat in the shading estimate is rescaled to unit mean
    beta = 0.5 + torch.rand(32, 32, generator=gen, dtype=F64)
    shading = ShadingEstimate(flat=7 * beta, dark=off + gain * d)
    a = Snyder(psf, n_iter=5, dtype=F64).fit(adu, init=[sensor, shading])
    b = Snyder(
        psf, n_iter=5, background=gain * d, flat=beta / beta.mean(), dtype=F64
    ).fit(adu, init=sensor)
    torch.testing.assert_close(a.apply(adu), b.apply(adu))
    # the explicit flat takes precedence, the baseline is ignored
    shading = shading.replace(baseline=torch.zeros(1))
    with pytest.warns(RuntimeWarning, match="baseline"):
        c = Snyder(psf, n_iter=5, flat=beta / beta.mean(), dtype=F64)
        c.fit(adu, init=[sensor, shading])
    torch.testing.assert_close(c.apply(adu), b.apply(adu))


def test_sieve_blurs_final_iterate():
    """Eq. (30): d = s * λ, the final iterate blurred by the sieve."""
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((40, 40), psf, seed=7)
    sieve = gaussian_psf(5, 1.0)
    plain = Snyder(psf, n_iter=5, dtype=F64).fit(data).apply(data)
    out = Snyder(psf, n_iter=5, sieve=sieve, dtype=F64).fit(data).apply(data)
    ref = torch.nn.functional.conv2d(
        plain[None, None], sieve.flip((0, 1))[None, None], padding=2
    )[0, 0]
    inner = (slice(2, -2),) * 2
    torch.testing.assert_close(out[inner], ref[inner])
    delta = torch.zeros(3, 3, dtype=F64)
    delta[1, 1] = 1
    same = Snyder(psf, n_iter=5, sieve=delta, dtype=F64).fit(data)
    torch.testing.assert_close(same.apply(data), plain)
    with pytest.raises(ValueError, match="sieve"):
        Snyder(psf, sieve=torch.ones(11, 11))


# -- blocks, dtypes, options ------------------------------------------------
@pytest.mark.parametrize("noise", ["exact", "shifted-poisson"])
def test_blocks_match_whole_image(noise):
    psf = gaussian_psf(9, 1.5)
    truth, gen = scene((96, 80), seed=8, base=1.0, peak=50.0)
    beta = 0.5 + 0.5 * torch.rand(96, 80, generator=gen, dtype=F64)
    r = ccd(truth, psf, gen, flat=beta, dark=0.5, var=0.6)
    sensor = SensorEstimate(offset=0.0, gain=1.0, variance=0.6)
    kw = {"n_iter": 10, "noise": noise, "flat": beta, "background": 0.5}
    whole = Snyder(psf, block=(96, 80), dtype=F64, **kw).fit(r, init=sensor)
    parts = Snyder(psf, block=(32, 24), dtype=F64, **kw).fit(r, init=sensor)
    a, b = whole.apply(r), parts.apply(r)
    assert float((a - b).abs().max() / a.abs().max()) < 1e-3


def test_float32_matches_float64():
    psf = gaussian_psf(7, 1.2)
    truth, gen = scene((40, 40), seed=9, base=1.0, peak=40.0)
    r = ccd(truth, psf, gen, dark=0.5, var=0.5)
    sensor = SensorEstimate(offset=0.0, gain=1.0, variance=0.5)
    kw = {"n_iter": 15, "background": 0.5}
    a = Snyder(psf, dtype=F64, **kw).fit(r, init=sensor).apply(r)
    b = Snyder(psf.float(), dtype=torch.float32, **kw)
    b = b.fit(r.float(), init=sensor).apply(r.float())
    assert b.dtype == torch.float32
    assert float((a - b).abs().max() / a.max()) < 1e-4


def test_chi2_stop_uses_read_noise_variance():
    psf = gaussian_psf(7, 1.2)
    truth, gen = scene((48, 48), seed=10, base=2.0, peak=60.0)
    r = ccd(truth, psf, gen, var=4.0)
    sensor = SensorEstimate(offset=0.0, gain=1.0, variance=4.0)
    est = Snyder(psf, dtype=F64).fit(r, init=sensor)
    est.apply(r)
    n = int(est.history.extras["iterations"][0])
    assert 1 <= n < 100
    op = PaddedConvolution(psf, r.shape)
    level = float(est.level)

    def expect(mu):
        return poisson_normal_mean(r, mu, torch.full_like(r, 4.0))

    for k, ok in ((n, True), (n - 1, False)):
        f, _ = rl_block(r, op, level, n_iter=k, expect=expect)
        mu = op.forward(f)
        chi2 = float(((r - mu).square() / (mu + 4.0)).sum())
        p = float(
            torch.special.gammaincc(
                torch.tensor(r.numel() - 1) / 2, torch.tensor(chi2) / 2
            )
        )
        assert (p >= 0.05) is ok


def test_options_and_faithful():
    psf = gaussian_psf(5, 1.0)
    assert Snyder(psf).is_faithful
    assert Snyder(psf).options["noise"] == "exact"
    beta = torch.ones(16, 16)
    for kw in (
        {"noise": "shifted-poisson"},
        {"flat": beta},
        {"background": 1.0},
        {"sieve": gaussian_psf(3, 1.0)},
    ):
        assert Snyder(psf, **kw).is_faithful, kw
    assert Snyder(psf, n_iter=3).is_faithful is False
    data = torch.ones(16, 16, dtype=F64)
    sensor = SensorEstimate(offset=0.0, gain=1.0, variance=1.0)
    assert Snyder(psf).fit(data, init=sensor).is_faithful
    with pytest.raises(ValueError, match="noise"):
        Snyder(psf, noise="gaussian")
    with pytest.raises(ValueError, match="flat"):
        Snyder(psf, n_iter=1, flat=-beta).fit(data)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Snyder(psf, n_iter=1).fit(data, init=[sensor, ShadingEstimate()])


def test_save_load(tmp_path):
    from optix import load

    psf = gaussian_psf(5, 1.0)
    truth, gen = scene((24, 24), seed=12, base=2.0)
    beta = 0.5 + torch.rand(24, 24, generator=gen, dtype=F64)
    r = ccd(truth, psf, gen, flat=beta, dark=1.0, var=0.5, off=10.0)
    sensor = SensorEstimate(offset=10.0, gain=1.0, variance=0.5)
    shading = ShadingEstimate(flat=beta, dark=torch.tensor(11.0))
    est = Snyder(psf, n_iter=4).fit(r, init=[sensor, shading])
    est.save(tmp_path / "snyder.pt")
    new = load(tmp_path / "snyder.pt")
    assert isinstance(new, Snyder)
    torch.testing.assert_close(new.apply(r), est.apply(r))


def test_richardson_lucy_shortcut_unchanged():
    psf = gaussian_psf(5, 1.0)
    _, data = poisson_image((24, 24), psf, seed=13)
    out = Snyder(psf, n_iter=3).fit(data).apply(data)
    torch.testing.assert_close(out, richardson_lucy(data, psf, n_iter=3))
