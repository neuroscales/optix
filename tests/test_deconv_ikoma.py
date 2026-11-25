"""Tests of optix.deconv.Ikoma (Ikoma et al. 2018)."""

import math
import warnings

import pytest
import torch

import optix
from optix import load
from optix.deconv import Ikoma, ikoma
from optix.deconv.utils import PaddedConvolution, transfer_function
from optix.estimates import SensorEstimate, ShadingEstimate
from optix.io import Layout
from optix.models import simulate_sensor
from optix.optim import Box, GroupL2, Hessian, ShiftedPoisson, Term, admm_fft

F64 = torch.float64
GAIN, READ, OFFSET = 2.0, 3.0, 100.0  # the paper's simulation: γ = 2, σ = 3


def gaussian2d(n=9, s=1.4):
    r = torch.arange(n, dtype=F64) - n // 2
    g = torch.exp(-(r[:, None] ** 2 + r[None] ** 2) / (2 * s * s))
    return g / g.sum()


def widefield3d(nz=5, nxy=7, s0=1.0, zr=1.2):
    """Gaussian beam: every xy-plane has the same energy."""
    z = torch.arange(nz, dtype=F64) - nz // 2
    r = torch.arange(nxy, dtype=F64) - nxy // 2
    s = s0 * torch.sqrt(1 + (z / zr) ** 2)
    d2 = r[None, :, None] ** 2 + r[None, None, :] ** 2
    g = torch.exp(-d2 / (2 * s[:, None, None] ** 2))
    return g / g.sum(dim=(-2, -1), keepdim=True)


def scene(shape, seed=0):
    """Blobs and filaments, max 1."""
    gen = torch.Generator().manual_seed(seed)
    grids = torch.meshgrid(
        *[torch.arange(n, dtype=F64) for n in shape], indexing="ij"
    )
    x = torch.zeros(shape, dtype=F64)
    for _ in range(10):
        c = [float(torch.rand(1, generator=gen)) * n for n in shape]
        rad = 1.5 + 2.5 * float(torch.rand(1, generator=gen))
        d2 = sum((g - ci) ** 2 for g, ci in zip(grids, c, strict=True))
        x += torch.exp(-d2 / (2 * rad**2))
    pts = torch.stack([g.reshape(-1) for g in grids], 1)
    for _ in range(5):
        p0 = torch.tensor(
            [float(torch.rand(1, generator=gen)) * n for n in shape],
            dtype=F64,
        )
        d = torch.randn(len(shape), generator=gen, dtype=F64)
        d /= d.norm()
        rel = pts - p0
        perp = rel - (rel @ d)[:, None] * d
        x += 0.8 * torch.exp(-perp.square().sum(1).reshape(shape) / 0.98)
    return x / x.max()


def sensor():
    return SensorEstimate(
        offset=torch.tensor(OFFSET, dtype=F64),
        gain=torch.tensor(GAIN, dtype=F64),
        variance=torch.tensor(READ**2, dtype=F64),
    )


def simulate(shape, psf, peak, seed=0):
    """x in photoelectrons (max of A x = peak) and ADU data."""
    x = scene(shape, seed)
    op = PaddedConvolution(psf, shape)
    ax = op.forward(op.pad(x))
    x = x * (peak / float(ax.max()))
    data = simulate_sensor(ax * (peak / float(ax.max())), sensor(), seed=seed)
    return x, data


def snr(est, x):
    return 10 * math.log10(float(x.square().sum() / (est - x).square().sum()))


# -- the paper's method -----------------------------------------------------
def test_defaults_and_paper_rules():
    psf = gaussian2d()
    x, data = simulate((48, 48), psf, 50.0, seed=1)
    est = Ikoma(psf, dtype=F64).fit(data, init=sensor())
    assert est.is_faithful
    y = ((data - OFFSET) / GAIN).clamp(min=-(READ**2) / GAIN**2)
    assert float(est.peak) == pytest.approx(float(y.max()))
    q = float(torch.quantile(y.flatten(), 0.999))
    assert float(est.robust_peak) == pytest.approx(q, rel=1e-12)
    nu = 0.1 / q  # ∑S = 1 for a 2-D PSF
    assert float(est.nu) == pytest.approx(nu)
    assert float(est.rho) == pytest.approx(6000 * nu / float(y.max()))
    est.apply(data)
    extras = est.history.extras
    for key in ("primal_residual", "dual_residual"):
        assert len(extras[key]) == 1 and math.isfinite(extras[key][0])
    none = Ikoma(psf, regulariser="none", dtype=F64).fit(data, init=sensor())
    assert float(none.rho) == pytest.approx(6 / float(y.max()))
    assert none.is_faithful


def test_stopping_rule():
    """Stops at the first iterate with relative change ≤ 10⁻³, before
    the cap of 150; one iteration fewer is above the tolerance."""
    psf = gaussian2d()
    x, data = simulate((64, 64), psf, 100.0, seed=3)
    est = Ikoma(psf, dtype=F64).fit(data, init=sensor())
    est.apply(data)
    n = int(est.history.extras["iterations"][0])
    change = est.history.extras["change"][0]
    assert 2 < n < 150, n
    assert change <= 1e-3
    short = Ikoma(psf, max_iter=n - 1, dtype=F64).fit(data, init=sensor())
    short.apply(data)
    assert short.history.extras["iterations"][0] == n - 1
    assert short.history.extras["change"][0] > 1e-3
    # the cap: max_iter iterations when the tolerance is not reached
    capped = Ikoma(psf, max_iter=5, dtype=F64).fit(data, init=sensor())
    capped.apply(data)
    assert capped.history.extras["iterations"][0] == 5
    assert capped.history.extras["change"][0] > 1e-3


def test_robust_peak_streamed():
    """The 99.9th percentile is exact when fit streams small chunks, and
    a hot pixel changes max ỹ (so ρ) but not the default ν."""
    psf = gaussian2d(5, 1.0)
    x, data = simulate((40, 50), psf, 40.0, seed=11)
    stack = torch.stack([data, 2 * data - OFFSET])
    ref = Ikoma(psf, dtype=F64).fit(stack, init=sensor())
    with optix.config(memory=8 * 40 * 7):
        small = Ikoma(psf, dtype=F64).fit(stack, init=sensor())
    torch.testing.assert_close(small.robust_peak, ref.robust_peak)
    for i in range(2):
        y = ((stack[i] - OFFSET) / GAIN).clamp(min=-(READ**2) / GAIN**2)
        q = float(torch.quantile(y.flatten(), 0.999))
        assert float(ref.robust_peak[i]) == pytest.approx(q, rel=1e-12)
    hot = data.clone()
    hot[3, 4] += 1e4
    a = Ikoma(psf, dtype=F64).fit(data, init=sensor())
    b = Ikoma(psf, dtype=F64).fit(hot, init=sensor())
    assert float(b.peak) > 10 * float(a.peak)
    assert float(b.nu) == pytest.approx(float(a.nu), rel=0.02)


def test_matches_admm_written_out():
    """The block solve is admm_fft on SI Eqs. (S8)–(S19), x-first."""
    psf = gaussian2d(7, 1.2)
    x, data = simulate((24, 20), psf, 30.0, seed=2)
    nu = 0.01
    est = Ikoma(psf, nu=nu, max_iter=40, tol=None, dtype=F64)
    out = est.fit(data, init=sensor()).apply(data)
    var = READ**2 / GAIN**2
    y = ((data - OFFSET) / GAIN).clamp(min=-var)
    op = PaddedConvolution(psf, y.shape)
    region = (slice(0, 24), slice(0, 20))
    terms = [
        Term(op.otf, ShiftedPoisson(y, var), region),
        Term(None, Box(0.0)),
        Term(Hessian(2), GroupL2(nu)),
    ]
    res = admm_fft(
        terms, op.grid, rho=6000 * nu / float(y.max()), max_iter=40, tol=None
    )
    torch.testing.assert_close(out, op.crop(res.x))


def test_s16_s19_written_out_3d():
    """3-D, independent of admm_fft: SI Eqs. (S9)–(S19) with full complex
    FFTs, the Hessian symbols built here, and the literal Eq. (S17)."""
    psf = widefield3d(3, 5)
    shape = (6, 8, 7)
    x, data = simulate(shape, psf, 30.0, seed=12)
    nu, k = 0.02, 25
    est = Ikoma(psf, nu=nu, max_iter=k, tol=None, dtype=F64)
    out = est.fit(data, init=sensor()).apply(data)
    var = READ**2 / GAIN**2
    y = ((data - OFFSET) / GAIN).clamp(min=-var)
    rho = 6000 * nu / float(y.max())
    grid = PaddedConvolution(psf, shape).grid
    # OTF: the PSF centred at index K // 2, moved to the origin
    h = torch.zeros(grid, dtype=F64)
    h[tuple(slice(0, n) for n in psf.shape)] = psf
    h = torch.roll(h, [-(n // 2) for n in psf.shape], dims=(0, 1, 2))
    a_hat = torch.fft.fftn(h)
    # D̂ₐ = e^{2πikₐ/Nₐ} − 1; H packs D_aa, then √2·D_ab (a < b) (S5, S7)
    d_hat = []
    for a, n in enumerate(grid):
        w = torch.exp(2j * math.pi * torch.arange(n, dtype=F64) / n) - 1
        shp = [1, 1, 1]
        shp[a] = n
        d_hat.append(w.reshape(shp))
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    h_hat = [
        (1 if a == b else math.sqrt(2)) * d_hat[a] * d_hat[b] for a, b in pairs
    ]

    def op(sym, v):
        return torch.fft.ifftn(sym * torch.fft.fftn(v)).real

    def opt(sym, v):
        return torch.fft.ifftn(sym.conj() * torch.fft.fftn(v)).real

    den = a_hat.abs() ** 2 + 1 + sum(s.abs() ** 2 for s in h_hat)
    obs = tuple(slice(0, n) for n in shape)
    z1 = torch.zeros(grid, dtype=F64)
    z2 = torch.zeros(grid, dtype=F64)
    z3 = torch.zeros((6, *grid), dtype=F64)
    u1, u2, u3 = (
        torch.zeros_like(z1),
        torch.zeros_like(z2),
        torch.zeros_like(z3),
    )
    for _ in range(k):
        rhs = (
            a_hat.conj() * torch.fft.fftn(z1 - u1)
            + torch.fft.fftn(z2 - u2)
            + sum(
                s.conj() * torch.fft.fftn(z3[c] - u3[c])
                for c, s in enumerate(h_hat)
            )
        )
        xk = torch.fft.ifftn(rhs / den).real  # (S16)
        v1 = op(a_hat, xk) + u1
        b = (1 - rho * (v1[obs] - var)) / (2 * rho)
        z1 = v1.clone()  # (S17): identity outside C
        z1[obs] = -b + torch.sqrt(b**2 + (y + rho * var * v1[obs]) / rho)
        z2 = (xk + u2).clamp(min=0)  # (S18)
        v3 = torch.stack([op(s, xk) for s in h_hat]) + u3
        norm = v3.norm(dim=0)
        z3 = v3 * (1 - (nu / rho) / norm.clamp(min=1e-300)).clamp(
            min=0
        )  # (S19)
        u1 = u1 + op(a_hat, xk) - z1
        u2 = u2 + xk - z2
        u3 = u3 + torch.stack([op(s, xk) for s in h_hat]) - z3
    torch.testing.assert_close(
        out, xk[obs], rtol=1e-8, atol=1e-8 * float(xk.abs().max())
    )


def test_recovery_and_table1_ordering():
    """Shifted-Poisson data (γ = 2, σ = 3): Hessian > none > data, and
    shifted Poisson ≥ Poisson without regulariser at low counts."""
    psf = gaussian2d()
    for peak in (10.0, 100.0):
        x, data = simulate((64, 64), psf, peak, seed=3)
        hess = Ikoma(psf, dtype=F64).fit(data, init=sensor()).apply(data)
        none = Ikoma(psf, regulariser="none", dtype=F64)
        none = none.fit(data, init=sensor()).apply(data)
        tv = Ikoma(psf, regulariser="tv", dtype=F64)
        tv = tv.fit(data, init=sensor()).apply(data)
        poisson = SensorEstimate(
            offset=torch.tensor(OFFSET, dtype=F64),
            gain=torch.tensor(GAIN, dtype=F64),
            variance=torch.tensor(0.0, dtype=F64),
        )
        pois = Ikoma(psf, regulariser="none", dtype=F64)
        pois = pois.fit(data, init=poisson).apply(data)
        raw = (data - OFFSET) / GAIN
        s = {
            k: snr(v, x)
            for k, v in dict(
                hess=hess, none=none, tv=tv, pois=pois, raw=raw
            ).items()
        }
        assert s["hess"] > s["none"] + 1.0, s
        assert s["hess"] > s["raw"] + 1.0, s
        assert s["hess"] > s["tv"] - 0.3, s
        if peak == 10.0:
            assert s["none"] >= s["pois"] - 0.05, s


def test_default_nu_across_intensity_ranges():
    """The default ν is close to the best ν of a grid at 10, 100 and 1000
    photoelectrons, and better than no regulariser (not in the paper)."""
    psf = gaussian2d()
    for peak in (10.0, 100.0, 1000.0):
        x, data = simulate((64, 64), psf, peak, seed=4)
        est = Ikoma(psf, dtype=F64).fit(data, init=sensor())
        default = snr(est.apply(data), x)
        nu = float(est.nu)
        grid = [
            snr(
                Ikoma(psf, nu=nu * f, dtype=F64)
                .fit(data, init=sensor())
                .apply(data),
                x,
            )
            for f in (1 / 9, 1 / 3, 3, 9)
        ]
        none = Ikoma(psf, regulariser="none", dtype=F64)
        none = snr(none.fit(data, init=sensor()).apply(data), x)
        assert default >= max(grid) - 1.5, (peak, default, grid)
        assert default > none, (peak, default, none)


def test_unknown_boundary():
    """Left and right edges of the field of view differ: the masked
    fidelity on the extended grid beats a periodic solve at the edges."""
    psf = gaussian2d(9, 1.6)
    full = 5 + 100 * scene((60, 60), 3)
    full[:, :30] += 60.0
    op = PaddedConvolution(psf, full.shape)
    mean = op.forward(op.pad(full))
    gen = torch.Generator().manual_seed(5)
    window = (slice(10, 50), slice(15, 45))
    data = torch.poisson(mean, generator=gen)[window]
    x = full[window]
    nu = 5e-4
    out = Ikoma(psf, nu=nu, dtype=F64).fit(data).apply(data)
    # the same problem, periodic on the observed grid (no extension)
    otf = transfer_function(psf, data.shape)
    terms = [
        Term(otf, ShiftedPoisson(data)),
        Term(None, Box(0.0)),
        Term(Hessian(2), GroupL2(nu)),
    ]
    rho = 6000 * nu / float(data.max())
    per = admm_fft(terms, data.shape, rho=rho, max_iter=150, tol=1e-3).x
    band = torch.zeros(data.shape, dtype=torch.bool)
    band[:, :4] = band[:, -4:] = True
    err = float((out - x)[band].square().mean())
    err_per = float((per - x)[band].square().mean())
    assert err < 0.5 * err_per, (err, err_per)


def test_blocks_vs_whole_image():
    psf = gaussian2d(7, 1.2)
    x, data = simulate((64, 60), psf, 50.0, seed=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        whole = Ikoma(psf, dtype=F64).fit(data, init=sensor()).apply(data)
        one = Ikoma(psf, block=(None, None), dtype=F64)
        one = one.fit(data, init=sensor()).apply(data)
    torch.testing.assert_close(whole, one)
    diffs = []
    for halo in (7, 21):
        est = Ikoma(psf, block=(32, 30), halo=halo, dtype=F64)
        est.fit(data, init=sensor())
        with pytest.warns(RuntimeWarning, match="blocks"):
            out = est.apply(data)
        assert len(est.history.extras["iterations"]) == 4
        inner = (slice(16, 48), slice(15, 45))  # around the seams
        diffs.append(float((out - whole)[inner].abs().max()))
    assert diffs[1] < diffs[0]
    assert diffs[1] < 0.05 * float(whole.max())


def test_memory_budget_on_the_fft_grid():
    """Blocks are sized on the FFT grid, and a budget that leaves blocks
    of mostly halo raises with the memory needed."""
    psf = gaussian2d(9, 1.6)
    data = torch.poisson(torch.full((256, 256), 20.0, dtype=F64))
    est = Ikoma(psf, max_iter=5, dtype=F64).fit(data)
    grid = PaddedConvolution(psf, data.shape).grid
    need = 8 * (est._grid_buffers() * math.prod(grid) + 4 * data.numel())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with optix.config(memory=int(need * 1.01)):
            est.apply(data)  # whole image
    assert est.history.extras["iterations"] == [5.0]
    with optix.config(memory=int(need * 0.99)):
        with pytest.warns(RuntimeWarning, match="blocks"):
            est.apply(data)  # 2 × 1 blocks of 128 × 256 (+ halo)
    assert len(est.history.extras["iterations"]) == 1 + 2
    # 3-D: the core would be a small part of each loaded block
    psf3 = widefield3d(5, 9)
    stack = torch.poisson(torch.full((12, 64, 64), 20.0, dtype=F64))
    est = Ikoma(psf3, max_iter=3, dtype=F64).fit(stack)
    with optix.config(memory="4MB"):
        with pytest.raises(ValueError, match="whole image needs about"):
            est.apply(stack)
        # an explicit block is respected
        out = Ikoma(psf3, block=(None, 32, 32), max_iter=3, dtype=F64)
        with pytest.warns(RuntimeWarning, match="blocks"):
            out.fit(stack).apply(stack)


def test_float32_matches_float64():
    psf = gaussian2d()
    for peak in (1.0, 100.0, 1e4):
        x, data = simulate((40, 40), psf, peak, seed=7)
        out = {}
        for dt in (torch.float32, F64):
            est = Ikoma(psf, dtype=dt, tol=None, max_iter=60)
            out[dt] = est.fit(data.to(dt), init=sensor()).apply(data.to(dt))
        scale = float(out[F64].abs().max())
        err = float((out[torch.float32].double() - out[F64]).abs().max())
        assert err < 1e-3 * scale, (peak, err, scale)


def test_sensor_units_and_shading():
    psf = gaussian2d(7, 1.2)
    gen = torch.Generator().manual_seed(8)
    e = torch.poisson(20 * scene((32, 36), 8) + 1, generator=gen)
    ref = Ikoma(psf, dtype=F64).fit(e).apply(e)
    s = SensorEstimate(
        offset=torch.tensor(50.0, dtype=F64), gain=torch.tensor(3.0, dtype=F64)
    )
    out = Ikoma(psf, dtype=F64).fit(e * 3 + 50, init=s).apply(e * 3 + 50)
    torch.testing.assert_close(out, ref)
    # shading: (y − d) / (f / median f) / gain, d includes the offset
    flat = 0.5 + torch.rand(32, 36, generator=gen, dtype=F64)
    dark = 60 + torch.rand(32, 36, generator=gen, dtype=F64)
    raw = (e * 3) * flat + dark
    sh = ShadingEstimate(flat=flat, dark=dark)
    out = ikoma(raw, psf, sensor=s, shading=sh, dtype=F64)
    manual = (raw - dark) / (flat / flat.median()) / 3
    ref = Ikoma(psf, dtype=F64).fit(manual).apply(manual)
    torch.testing.assert_close(out, ref)
    with pytest.warns(RuntimeWarning, match="baseline"):
        Ikoma(psf).fit(raw, init=[s, sh.replace(baseline=torch.zeros(1))])


def test_read_noise_with_flat_field():
    """σ̃² = variance / (gain·f)² with a flat f (unit median)."""
    psf = gaussian2d(7, 1.2)
    gen = torch.Generator().manual_seed(10)
    e = torch.poisson(20 * scene((32, 36), 10) + 1, generator=gen)
    flat = 0.5 + torch.rand(32, 36, generator=gen, dtype=F64)
    dark = 60 + torch.rand(32, 36, generator=gen, dtype=F64)
    gain, var = 3.0, 16.0
    raw = e * gain + 4 * torch.randn(32, 36, generator=gen, dtype=F64)
    raw = raw * flat + dark
    s = SensorEstimate(
        offset=torch.tensor(50.0, dtype=F64),
        gain=torch.tensor(gain, dtype=F64),
        variance=torch.tensor(var, dtype=F64),
    )
    sh = ShadingEstimate(flat=flat, dark=dark)
    out = ikoma(raw, psf, sensor=s, shading=sh, dtype=F64, max_iter=30)
    f = flat / flat.median()
    corrected = (raw - dark) / (f * gain)
    noise = var / (gain * f) ** 2
    ref_sensor = SensorEstimate(
        offset=torch.tensor(0.0, dtype=F64),
        gain=torch.tensor(1.0, dtype=F64),
        variance=noise,
    )
    ref = ikoma(corrected, psf, sensor=ref_sensor, dtype=F64, max_iter=30)
    torch.testing.assert_close(out, ref)
    # without the flat² factor the result differs
    flat_only = SensorEstimate(
        offset=torch.tensor(0.0, dtype=F64),
        gain=torch.tensor(1.0, dtype=F64),
        variance=torch.tensor(var / gain**2, dtype=F64),
    )
    other = ikoma(corrected, psf, sensor=flat_only, dtype=F64, max_iter=30)
    assert float((other - out).abs().max()) > 1e-3 * float(out.abs().max())


def test_psf_norm_plane_vs_sum():
    """x_plane = x_sum / K_z and ν_plane = K_z ν_sum, at convergence."""
    psf = widefield3d()
    kz = psf.shape[0]
    x, data = simulate((8, 16, 16), psf / kz, 40.0, seed=9)
    plane = Ikoma(psf, dtype=F64).fit(data, init=sensor())
    total = Ikoma(psf, psf_norm="sum", dtype=F64).fit(data, init=sensor())
    assert float(plane.nu) == pytest.approx(kz * float(total.nu))
    assert plane.is_faithful and not total.is_faithful
    nu = 0.02
    opts = dict(tol=1e-8, max_iter=1500, dtype=F64)
    a = Ikoma(psf, nu=kz * nu, rho=0.05, **opts)
    a = a.fit(data, init=sensor()).apply(data)
    b = Ikoma(psf, nu=nu, psf_norm="sum", rho=0.05 / kz, **opts)
    b = b.fit(data, init=sensor()).apply(data)
    torch.testing.assert_close(a * kz, b, rtol=0, atol=2e-3 * float(b.max()))


def test_layout_and_state(tmp_path):
    psf = gaussian2d(5, 1.0)
    imgs = [simulate((24, 20), psf, 30.0, seed=s)[1] for s in range(3)]
    stack = torch.stack(imgs, dim=-1)
    layout = Layout("yxt", spatial="yx")
    est = Ikoma(psf, max_iter=20, dtype=F64, layout=layout)
    out = est.fit(stack, init=sensor()).apply(stack)
    assert est.nu.shape == (3,)
    for i, img in enumerate(imgs):
        ref = Ikoma(psf, max_iter=20, dtype=F64).fit(img, init=sensor())
        torch.testing.assert_close(out[..., i], ref.apply(img))
    est.save(tmp_path / "ikoma.pt")
    new = load(tmp_path / "ikoma.pt")
    new.dtype = F64
    torch.testing.assert_close(new.apply(stack), out)
    torch.testing.assert_close(new.nu, est.nu)


def test_options_and_faithfulness():
    psf = gaussian2d(5, 1.0)
    assert Ikoma(psf).is_faithful
    assert Ikoma(psf, nu=0.01).is_faithful
    assert Ikoma(psf, regulariser="none").is_faithful
    assert not Ikoma(psf, regulariser="tv").is_faithful
    assert not Ikoma(psf, max_iter=10).is_faithful
    assert not Ikoma(psf, weights=(1.0, 0.5)).is_faithful
    # halo and block are memory settings, not method choices
    assert Ikoma(psf, halo=12).is_faithful
    assert Ikoma(psf, block=(8, None)).is_faithful
    est = Ikoma(psf, block=(10, 10), halo=10)
    data = torch.poisson(torch.full((20, 20), 5.0))
    with pytest.warns(RuntimeWarning, match="blocks"):
        est.fit(data).apply(data)
    assert est.is_faithful
    for bad in (
        dict(regulariser="l1"),
        dict(psf_norm="max"),
        dict(nu=-1.0),
        dict(rho=0.0),
        dict(weights=(1.0,)),
    ):
        with pytest.raises(ValueError):
            Ikoma(psf, **bad)
    with pytest.raises(ValueError, match="positive maximum"):
        Ikoma(psf).fit(torch.zeros(16, 16))
    data = torch.poisson(torch.full((20, 20), 5.0))
    out = Ikoma(psf, weights=(1.0, 0.5), max_iter=5).fit(data).apply(data)
    assert out.shape == data.shape
