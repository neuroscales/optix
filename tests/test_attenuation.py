import itertools
import math

import pytest
import torch

from optix import estimators, load
from optix.attenuation import DepthResolved, compensate, depth_resolved
from optix.attenuation.utils import (
    compensation_gain,
    smith_attenuation,
    tail_sums,
    vermeer_attenuation,
    vermeer_gain,
    weighted_tv,
)
from optix.estimates import SensorEstimate, ShadingEstimate
from optix.io import Layout, labeled
from optix.models import simulate_sensor


def profile(mu: torch.Tensor, dz: float, passes: int = 2, scale=1.0):
    """Pixel-integrated model I[i] = ∫ μ e^{−p∫μ} over pixel i (piecewise μ).

    Exact for μ constant within each pixel: the integral over pixel i is
    (1/p)·(A_i − A_{i+1}) with A_i = exp(−p·Δ·∑_{j<i} μ[j]).
    """
    mu = mu.to(torch.float64)
    cum = torch.cat([torch.zeros(1, dtype=mu.dtype), mu.cumsum(0)]) * dz
    a = torch.exp(-passes * cum)
    return scale * (a[:-1] - a[1:]) / passes


def test_constant_mu_recovered(dtype):
    dz, mu = 0.004, 5.0
    data = profile(torch.full((500,), mu), dz).to(dtype)
    est = depth_resolved(data, pixel_size=dz)
    assert est.dtype == dtype
    # exact up to where the light left in the profile matters (Fig. 1b):
    # e^{-2·μ·(2 mm - 0.6 mm)} ≈ 1e-6 of the light remains
    torch.testing.assert_close(
        est[:150], torch.full((150,), mu, dtype=dtype), rtol=1e-5, atol=0
    )


def test_depth_varying_mu_recovered():
    dz = 0.004
    # layered profile as in Vermeer et al. Fig. 2 (dashed), then a ramp
    mu = torch.cat(
        [
            torch.full((90,), 0.5),
            torch.full((90,), 2.0),
            torch.full((90,), 5.0),
            torch.linspace(5.0, 10.0, 230),
        ]
    ).double()
    # Eq. (17) inverts the model exactly up to the truncation of the
    # integral at the end of the profile: append a deep thick absorber so
    # that the truncation error is negligible in the first 500 pixels
    full = torch.cat([mu, torch.full((2000,), 10.0, dtype=torch.float64)])
    data = profile(full, dz)
    est = depth_resolved(data, pixel_size=dz)[: len(mu)]
    torch.testing.assert_close(est, mu, rtol=1e-9, atol=0)


def test_float32_matches_float64():
    dz = 0.004
    mu = torch.linspace(1.0, 8.0, 400).double()
    data = profile(mu, dz)
    ref = depth_resolved(data, pixel_size=dz)[:-1]
    est = depth_resolved(data.float(), pixel_size=dz)[:-1]
    torch.testing.assert_close(est.double(), ref, rtol=1e-5, atol=0)


def test_passes_one_single_pass_model():
    dz, mu = 0.01, 2.0
    data = profile(torch.full((3000,), mu), dz, passes=1)
    est = depth_resolved(data, pixel_size=dz, passes=1)
    torch.testing.assert_close(
        est[:400], torch.full((400,), mu, dtype=est.dtype), rtol=1e-9, atol=0
    )
    # same data, paper's p = 2: half the single-pass coefficient
    half = depth_resolved(data, pixel_size=dz, passes=2)
    torch.testing.assert_close(half, est / 2)


def test_end_of_profile_matches_paper():
    dz, mu = 0.004, 1.0
    data = profile(torch.full((500,), mu), dz)
    est = depth_resolved(data, pixel_size=dz)
    # the sum of deeper pixels is empty at the last pixel: log(1 + I/0)
    assert math.isinf(est[-1].item()) and est[-1] > 0
    # Eq. (17) with the sum truncated at the end of the profile, i.e. the
    # paper's formula evaluated literally
    for i in (0, 250, 497, 498):
        deeper = data[i + 1 :].sum()
        expected = math.log1p(data[i] / deeper) / (2 * dz)
        assert est[i].item() == pytest.approx(expected, rel=1e-12)
    # overestimation growing towards the end when light remains (Fig. 2b)
    tail = est[300:-1]
    assert (tail[1:] >= tail[:-1]).all()
    assert tail[-1] > 10 * mu
    # zero signal and zero tail: undefined
    padded = torch.cat([data, torch.zeros(3, dtype=data.dtype)])
    est = depth_resolved(padded, pixel_size=dz)
    assert math.isinf(est[499].item()) and torch.isnan(est[-3:]).all()


def test_linearisation_eq18_close_for_small_ratio():
    dz = 0.004
    data = profile(torch.full((500,), 2.0), dz)
    est = vermeer_attenuation(data, pixel_size=dz)
    lin = data / (2 * dz * tail_sums(data, inclusive=False))
    torch.testing.assert_close(est[:100], lin[:100], rtol=0.02, atol=0)


def test_compensation_flattens_uniform_emitter(dtype):
    dz = 0.004
    mu = torch.full((500,), 5.0)
    data = profile(mu, dz, scale=1e3).to(dtype)
    out = compensate(data, pixel_size=dz)
    assert out.dtype == dtype
    ref = out[0]
    torch.testing.assert_close(out[:150], ref.expand(150), rtol=1e-5, atol=0)
    # not flat before compensation
    assert data[299] < 0.01 * data[0]


def test_compensation_closed_form_equals_cumulated_mu():
    dz = 0.004
    mu = torch.linspace(1.0, 6.0, 300).double()
    # deep absorber: negligible truncation of the sums in the first 300
    data = profile(torch.cat([mu, torch.full((3000,), 6.0).double()]), dz)
    est = depth_resolved(data, pixel_size=dz)
    gain = compensation_gain(est[:-1], pixel_size=dz)
    torch.testing.assert_close(
        vermeer_gain(data)[:-1], gain, rtol=1e-9, atol=0
    )
    # and gives T[0]·(1 − exp(−p·Δ·μ)) ≈ T[0]·p·Δ·μ: proportional to μ
    out = compensate(data, pixel_size=dz)[:300]
    expected = data.sum() * -torch.expm1(-2 * dz * est[:300])
    torch.testing.assert_close(out, expected, rtol=1e-9, atol=0)


def test_compensation_independent_of_passes_and_pixel_size():
    data = profile(torch.full((200,), 3.0), 0.01)
    a = compensate(data, pixel_size=0.01, passes=2)
    b = compensate(data, pixel_size=1.0, passes=1)
    torch.testing.assert_close(a, b)
    # Smith: λ ∝ (2/p)/Δ keeps the TV strength on the scale of μ̂
    data, sensor, _, dz = _noisy_bscan(n=16, m=120)
    kw = dict(method="smith", tv_axis=0, background=sensor)
    a = compensate(data, pixel_size=dz, passes=2, **kw)
    b = compensate(data, pixel_size=1.0, passes=1, **kw)
    c = compensate(data, pixel_size=dz, passes=2, eta=0.0, **kw)
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    assert (a - c).abs().max() > 1e-3  # the TV term is active


def test_smith_noisy_tail_is_finite():
    # zero-mean noise in the tail after background subtraction: tail sums
    # near 0 or negative must not overflow the gain
    data, sensor, _, dz = _noisy_bscan(n=16, m=120)
    tail = torch.randn(
        16, 200, dtype=data.dtype, generator=torch.Generator().manual_seed(1)
    )
    noisy = torch.cat([data, 100.0 + 2.0 * tail], dim=-1)
    for kw in (dict(eta=0.0), dict(), dict(noise_power=1.0)):
        for dtype in (torch.float32, torch.float64):
            x = noisy.to(dtype)
            mu = depth_resolved(
                x, pixel_size=dz, background=100.0, method="smith", **kw
            )
            assert torch.isfinite(mu).all()
            out = compensate(
                x, pixel_size=dz, background=100.0, method="smith", **kw
            )
            assert torch.isfinite(out).all()
            xi = (x - 100.0).double()
            c = kw.get("noise_power", 0.0)
            t = tail_sums(xi * xi**2 / (xi**2 + c) if c else xi)
            assert (out[t.to(out.dtype) <= 0] == 0).all()
    # μ̂ = 0 where the tail sum is ≤ 0
    x = torch.tensor([1.0, 2.0, -5.0, 1.0], dtype=torch.float64)
    est = smith_attenuation(x)
    assert est[0] == 0 and est[1] == 0 and est[2] == 0 and est[3] == 0.5
    # the exponent is capped: finite gain
    assert torch.isfinite(
        compensation_gain(torch.full((4,), 1e300, dtype=torch.float64))
    ).all()


def test_vermeer_gain_zero_when_profile_sum_not_positive():
    x = torch.tensor([-5.0, 1.0, 2.0], dtype=torch.float64)
    assert (vermeer_gain(x) == 0).all()


def test_max_gain_caps():
    data = profile(torch.full((500,), 5.0), 0.004)
    out = compensate(data, pixel_size=0.004, max_gain=10.0)
    assert (out <= 10.0 * data + 1e-15).all()
    with pytest.raises(ValueError, match="max_gain"):
        compensate(data, max_gain=0.5)


def test_background_value_and_estimates():
    dz = 0.004
    data = profile(torch.full((400,), 4.0), dz, scale=1e3)
    ref = depth_resolved(data, pixel_size=dz)
    for bg in (
        100.0,
        torch.tensor(100.0),
        SensorEstimate(offset=torch.tensor(100.0)),
        ShadingEstimate(dark=torch.tensor(100.0)),
    ):
        est = depth_resolved(data + 100.0, pixel_size=dz, background=bg)
        torch.testing.assert_close(est, ref)
    with pytest.raises(ValueError, match="offset"):
        depth_resolved(data, background=SensorEstimate())


def _noisy_bscan(seed=0, n=64, m=200, dz=0.01):
    mu = torch.cat([torch.full((80,), 1.0), torch.full((m - 80,), 3.0)])
    clean = profile(torch.cat([mu, torch.full((400,), 3.0)]), dz)[:m]
    clean = clean.double() / clean.max() * 2000
    signal = clean.expand(n, m).clone()  # (lateral, depth)
    sensor = SensorEstimate(
        offset=torch.tensor(100.0),
        variance=torch.tensor(4.0),
        gain=torch.tensor(1.0),
    )
    data = simulate_sensor(signal, sensor, seed=seed)
    return data, sensor, mu.double(), dz


def test_smith_reduces_noise_in_mu():
    data, sensor, mu, dz = _noisy_bscan()
    kw = dict(axis=-1, pixel_size=dz, background=sensor)
    vermeer = depth_resolved(data, **kw)
    plain = depth_resolved(data, method="smith", eta=0.0, **kw)
    smith = depth_resolved(data, method="smith", tv_axis=0, **kw)
    sl = slice(130, 170)  # second layer, low intensity (heavier smoothing)
    lat_std = lambda x: x[:, sl].std(0).mean()  # noqa: E731
    assert lat_std(smith) < 0.6 * lat_std(plain)
    assert lat_std(smith) < 0.6 * lat_std(vermeer)
    # and stays close to the truth
    err = (smith[:, sl].mean() - mu[sl].mean()).abs() / mu[sl].mean()
    assert err < 0.1
    assert (smith >= 0).all()


def test_smith_eq10_and_restoration_filter():
    dz = 0.01
    data = profile(torch.full((300,), 2.0), dz)
    est = smith_attenuation(data, pixel_size=dz)
    expected = data / (2 * dz * tail_sums(data))
    torch.testing.assert_close(est, expected)
    assert est[-1].item() == pytest.approx(1 / (2 * dz))
    filt = smith_attenuation(data, pixel_size=dz, noise_power=1e-6)
    x = data * data**2 / (data**2 + 1e-6)
    torch.testing.assert_close(filt, x / (2 * dz * tail_sums(x)))


def _kkt_brute_force(y, bound, nonneg):
    """Weighted TV prox by enumeration of the KKT active sets.

    Unknowns x (n), dual p (n − 1) and multipliers ν ≥ 0 of x ≥ 0 (n):
    x − y + Dᵀp − ν = 0, and for every difference k either p_k = ±bound_k
    (with the matching sign of (Dx)_k) or (Dx)_k = 0, for every pixel
    either x_i = 0 or ν_i = 0. The strongly convex problem has a single
    solution, the x of any feasible combination.
    """
    n = y.numel()
    d = torch.diff(torch.eye(n, dtype=torch.float64), dim=0)
    m = 3 * n - 1
    pixels = itertools.product((0, 1), repeat=n) if nonneg else [(1,) * n]
    for zeros in pixels:
        for signs in itertools.product((-1, 0, 1), repeat=n - 1):
            a = torch.zeros(m, m, dtype=torch.float64)
            r = torch.zeros(m, dtype=torch.float64)
            a[:n, :n] = torch.eye(n)
            a[:n, n : 2 * n - 1] = d.T
            a[:n, 2 * n - 1 :] = -torch.eye(n)
            r[:n] = y
            for k, s in enumerate(signs):
                if s == 0:
                    a[n + k, :n] = d[k]
                else:
                    a[n + k, n + k] = 1
                    r[n + k] = s * bound[k]
            for i, free in enumerate(zeros):
                a[2 * n - 1 + i, i if free == 0 else 2 * n - 1 + i] = 1
            z, info = torch.linalg.solve_ex(a, r)
            if info != 0:
                continue
            x, p, nu = z[:n], z[n : 2 * n - 1], z[2 * n - 1 :]
            dx, e = d @ x, 1e-10
            ok = bool((p.abs() <= bound + e).all() and (nu >= -e).all())
            ok = ok and bool((x >= -e).all() or not nonneg)
            ok = ok and all(
                (s != 1 or dx[k] >= -e) and (s != -1 or dx[k] <= e)
                for k, s in enumerate(signs)
            )
            if ok:
                return x
    raise AssertionError("no KKT point found")


def _fista_reference(y, bound, iterations=100000, tol=1e-14):
    """Dual accelerated projected gradient for the nonneg weighted TV."""

    def d(v):
        return v.diff(dim=-1)

    def dt(p):
        zero = torch.zeros_like(p[..., :1])
        return -torch.cat([zero, p, zero], -1).diff(dim=-1)

    def primal(p):
        return (y - dt(p)).clamp_min(0)

    p = q = torch.zeros_like(bound)
    t = 1.0
    for it in range(1, iterations + 1):
        p_new = torch.clamp(q + d(primal(q)) / 4, -bound, bound)
        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
        q = p_new + ((t - 1) / t_new) * (p_new - p)
        p, t = p_new, t_new
        if it % 50 == 0:
            x = primal(p)
            fit = 0.5 * (x - y).square().sum(-1)
            obj = fit + (bound * d(x).abs()).sum(-1)
            gap = (obj - fit - (p * d(x)).sum(-1)) / obj.clamp_min(1e-300)
            if bool((gap <= tol).all()):
                break
    return primal(p)


def test_weighted_tv_exact_against_brute_force():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        n = int(torch.randint(2, 6, (), generator=g))
        y = torch.randn(n, dtype=torch.float64, generator=g)
        w = torch.rand(n - 1, dtype=torch.float64, generator=g) * 1.5
        w[torch.rand(n - 1, generator=g) < 0.2] = 0.0  # zero weights
        lam = 0.7
        x = weighted_tv(y, w, lam, nonneg=False)
        ref = _kkt_brute_force(y, lam * w, nonneg=False)
        torch.testing.assert_close(x, ref, rtol=0, atol=1e-12)
        # nonneg: max(prox_TV, 0), against the constrained KKT system
        x = weighted_tv(y, w, lam)
        ref = _kkt_brute_force(y, lam * w, nonneg=True)
        torch.testing.assert_close(x, ref, rtol=0, atol=1e-12)


def test_weighted_tv_matches_fista_and_dims():
    g = torch.Generator().manual_seed(1)
    y = torch.rand(30, 64, dtype=torch.float64, generator=g) * 3 - 0.5
    w = torch.rand(30, 64, dtype=torch.float64, generator=g) + 0.1
    x = weighted_tv(y, w, 0.4)  # N weights: the last is ignored
    ref = _fista_reference(y, 0.4 * w[:, :-1])
    torch.testing.assert_close(x, ref, rtol=0, atol=1e-9)
    # along another dim, float32 in and out
    xt = weighted_tv(y.T.float(), w.T[:-1].float(), 0.4, dim=0)
    assert xt.dtype == torch.float32
    torch.testing.assert_close(xt.T.double(), x, rtol=0, atol=1e-5)
    # lam = 0 is the identity (projected on x >= 0)
    torch.testing.assert_close(weighted_tv(y, w, 0.0), y.clamp_min(0))
    with pytest.raises(ValueError, match="weights"):
        weighted_tv(y, -w, 0.4)


def test_axis_handling():
    dz = 0.004
    p = profile(torch.full((300,), 5.0), dz)
    vol = p[None, :, None].expand(3, 300, 4).clone()  # propagation on dim 1
    est = depth_resolved(vol, axis=1, pixel_size=dz)
    ref = depth_resolved(p, pixel_size=dz)
    torch.testing.assert_close(est, ref[None, :, None].expand(3, 300, 4))
    out = compensate(vol, axis=-2, pixel_size=dz)
    torch.testing.assert_close(
        out, compensate(p, pixel_size=dz)[None, :, None].expand(3, 300, 4)
    )
    with pytest.raises(ValueError, match="out of range"):
        depth_resolved(vol, axis=3)
    with pytest.raises(ValueError, match="tv_axis"):
        depth_resolved(vol, axis=0, method="smith")


def test_estimator_axis_labels_and_layout():
    dz = 0.004
    p = profile(torch.full((100,), 5.0), dz)
    # storage (z, y, x), propagation along x, sample z
    vol = p[None, None, :].expand(5, 6, 100).clone().float()
    layout = Layout("zyx", spatial="yx", sample="z")
    est = DepthResolved(
        axis="x", pixel_size=dz, layout=layout, dtype=torch.float32
    )
    assert est.is_faithful
    out = est.apply(vol)
    torch.testing.assert_close(out, compensate(vol, axis=2, pixel_size=dz))
    mu = est.attenuation(labeled(vol, layout))
    torch.testing.assert_close(mu, depth_resolved(vol, pixel_size=dz))
    # propagation along y (dim 1)
    vol_y = vol.transpose(1, 2).contiguous()
    est = DepthResolved(axis="y", pixel_size=dz, dtype="float32").fit(
        labeled(vol_y, layout)
    )
    torch.testing.assert_close(
        est.attenuation(vol_y), depth_resolved(vol_y, axis=1, pixel_size=dz)
    )
    with pytest.raises(ValueError, match="label"):
        DepthResolved(axis="q", layout=layout).apply(vol)


def test_estimator_background_map_placed_by_layout():
    dz = 0.004
    p = profile(torch.full((50,), 5.0), dz, scale=1e3)
    # storage (x, t, y): propagation x, frames t, spatial (y, x)
    offset = torch.arange(7.0)[:, None] + 10 * torch.arange(50.0)  # (y, x)
    data = p[:, None, None].expand(50, 3, 7) + offset.T[:, None, :]
    layout = Layout("xty", spatial="yx", sample="t")
    est = DepthResolved(
        axis="x", pixel_size=dz, background=SensorEstimate(offset=offset)
    )
    assert not est.is_faithful
    out = est.fit(labeled(data, layout)).apply(data)
    clean = p[:, None, None].expand(50, 3, 7)
    torch.testing.assert_close(
        out,
        compensate(clean, axis=0, pixel_size=dz).to(out.dtype),
        rtol=1e-5,
        atol=1e-3,
    )


def test_estimator_faithful_flags_and_registry():
    assert "DepthResolved" in estimators()
    assert DepthResolved(axis=0, pixel_size=0.5).is_faithful
    assert not DepthResolved(method="smith").is_faithful
    assert not DepthResolved(passes=1).is_faithful
    assert not DepthResolved(max_gain=5.0).is_faithful
    # recommended: single-pass light-sheet excitation
    rec = DepthResolved(faithful=False)
    assert rec.options["passes"] == 1 and not rec.is_faithful
    with pytest.raises(ValueError, match="method"):
        DepthResolved(method="other")


def test_estimator_save_load(tmp_path):
    est = DepthResolved(axis=-2, pixel_size=0.1, max_gain=20.0)
    est.save(tmp_path / "e.pt")
    new = load(tmp_path / "e.pt")
    assert new.options == est.options


def test_streaming_blocks_and_zarr_parity():
    zarr = pytest.importorskip("zarr")
    import optix

    data, sensor, _, dz = _noisy_bscan(n=24, m=120)
    vol = data.reshape(4, 6, 120)  # (z, y, x), propagation along x
    ref = compensate(vol.float(), pixel_size=dz, background=sensor)
    est = DepthResolved(pixel_size=dz, background=sensor, dtype="float32")
    src = zarr.zeros(vol.shape, chunks=(1, 3, 120), dtype="f4")
    src[:] = vol.float().numpy()
    dst = zarr.zeros(vol.shape, chunks=(1, 3, 120), dtype="f4")
    # tiny budget: one block per (z, y-chunk), whole profiles
    # float64 accumulation, 8 block-sized arrays
    with optix.config(memory=3 * 120 * 8 * 8):
        est.apply(src, out=dst)
    torch.testing.assert_close(torch.from_numpy(dst[:]), ref)
    # Smith: whole extent of the TV axis (y) as well
    est = DepthResolved(
        pixel_size=dz, background=sensor, method="smith", dtype="float32"
    )
    ref = depth_resolved(
        vol.float(), pixel_size=dz, background=sensor, method="smith"
    )
    with optix.config(memory=6 * 120 * 8 * 24):
        mu = est.attenuation(src)
    torch.testing.assert_close(mu, ref, rtol=1e-4, atol=1e-4)
