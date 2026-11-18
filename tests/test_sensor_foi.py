import math

import numpy as np
import pytest
import torch

import optix
import optix.sensor._poisson_gaussian as PG
from optix.estimates import SensorEstimate
from optix.io import Layout, labeled
from optix.models import simulate_sensor
from optix.sensor import (
    PoissonGaussian,
    SegmentStats,
    clipped_normal_moments,
    from_normalized,
    group_frames,
    integer_range,
    sensor_estimate,
    to_normalized,
)

# (gain a [ADU/e-], read sd [ADU], offset [ADU]) settings of the harness
SETTINGS = [(0.5, 1.5, 100.0), (2.0, 5.0, 50.0), (0.1, 3.0, 0.0)]


def scene(shape=(256, 256), levels=8, peak=2000.0, seed=0):
    """Piecewise-constant scene (electrons): vertical bands plus a ramp."""
    g = torch.Generator().manual_seed(seed)
    H, W = shape
    band = torch.arange(W) * levels // W
    values = torch.rand(levels, generator=g, dtype=torch.float64) * peak
    img = values[band].expand(H, W).clone()
    img[: H // 4] = torch.linspace(0, peak, W, dtype=torch.float64)
    return img


def synthetic(a, read_sd, offset, *, clip=None, frames=1, seed=0, **kw):
    """Poisson-Gaussian data with ground-truth ``(a, b)`` in ADU."""
    sensor = SensorEstimate(
        offset=torch.tensor(offset, dtype=torch.float64),
        variance=torch.tensor(read_sd**2, dtype=torch.float64),
        gain=torch.tensor(a, dtype=torch.float64),
    )
    signal = scene(seed=seed, **kw).expand(frames, -1, -1)
    y = simulate_sensor(signal, sensor, seed=seed + 1)
    if clip is not None:
        y = y.clamp(*clip)
    b = read_sd**2 - a * offset
    return y, a, b, sensor


@pytest.mark.parametrize("a, read_sd, offset", SETTINGS)
def test_harness_ground_truth(a, read_sd, offset):
    # temporal variance per pixel follows sigma^2(y) = a y + b
    y, a, b, sensor = synthetic(a, read_sd, offset, frames=100)
    mean, var = y.mean(0), y.var(0)
    expected = a * mean + b
    rel = (var / expected).mean().item()
    assert rel == pytest.approx(1, abs=0.02)
    est = sensor_estimate(torch.tensor(a, dtype=torch.float64), b, offset)
    torch.testing.assert_close(est.noise_variance(mean), expected)
    torch.testing.assert_close(est.variance, sensor.variance)


def test_sensor_estimate_without_offset():
    # not estimated -> None, never a silent zero; b goes into variance
    est = sensor_estimate(torch.tensor(0.5, dtype=torch.float64), -40.0)
    assert est.meta["offset_estimated"] is False
    assert est.offset is None
    assert est.gain.item() == 0.5 and est.variance.item() == -40.0
    with pytest.raises(ValueError, match="offset"):
        est.noise_variance(torch.tensor([100.0]))


def test_normalized_units_round_trip():
    a, b, low, scale = 0.5, -47.75, 90.0, 4000.0
    an, bn = to_normalized(a, b, low, scale)
    y = torch.linspace(100, 3000, 5, dtype=torch.float64)
    yn = (y - low) / scale
    torch.testing.assert_close((an * yn + bn) * scale**2, a * y + b)
    assert from_normalized(an, bn, low, scale) == pytest.approx((a, b))


def test_integer_range():
    assert integer_range(torch.uint16) == (0.0, 65535.0)
    assert integer_range("uint8") == (0.0, 255.0)


@pytest.mark.parametrize(
    "mu, sd, low, high",
    [
        (0.5, 0.05, 0.0, 1.0),
        (0.02, 0.05, 0.0, 1.0),
        (0.98, 0.05, 0.0, 1.0),
        (-0.1, 0.1, 0.0, None),
        (1.2, 0.3, None, 1.0),
        (0.5, 2.0, 0.0, 1.0),
    ],
)
def test_clipped_normal_moments_monte_carlo(mu, sd, low, high):
    g = torch.Generator().manual_seed(0)
    z = mu + sd * torch.randn(2_000_000, generator=g, dtype=torch.float64)
    z = z.clamp(min=low, max=high)
    m, s = clipped_normal_moments(
        torch.tensor(mu, dtype=torch.float64),
        torch.tensor(sd, dtype=torch.float64),
        low,
        high,
    )
    tol = 4 * sd / 2_000_000**0.5
    assert m.item() == pytest.approx(z.mean().item(), abs=tol)
    assert s.item() == pytest.approx(z.std().item(), abs=2 * tol)


def test_clipped_normal_moments_limits(dtype):
    mu = torch.linspace(0.2, 0.8, 7, dtype=dtype)
    sd = torch.full_like(mu, 1e-3)
    m, s = clipped_normal_moments(mu, sd)
    torch.testing.assert_close(m, mu)
    torch.testing.assert_close(s, sd)
    # saturated: mean at the bound, (almost) no spread
    m, s = clipped_normal_moments(mu + 5, sd)
    torch.testing.assert_close(m, torch.ones_like(mu))
    assert (s >= 0).all() and (s < 1e-6).all()


def _pairs(n=20_000, segments=7, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(-1, segments + 1, (n,), generator=g)
    value = 1e4 + torch.randn(n, generator=g, dtype=torch.float64)
    noise = labels.clamp(0) * 0.1 + torch.randn(
        n, generator=g, dtype=torch.float64
    ) * (1 + labels.clamp(0))
    return labels, value.to(dtype), noise.to(dtype)


def test_segment_stats_matches_direct():
    labels, value, noise = _pairs()
    st = SegmentStats(7).update(labels, value, noise)
    r = st.result()
    for k, seg in enumerate(r["segment"].tolist()):
        sel = labels == seg
        assert r["count"][k].item() == sel.sum().item()
        assert r["mean"][k].item() == pytest.approx(value[sel].mean().item())
        assert r["std"][k].item() == pytest.approx(noise[sel].std().item())
    assert set(r["segment"].tolist()) == set(range(7))


def test_segment_stats_chunking_and_merge():
    labels, value, noise = _pairs()
    mask = torch.arange(len(labels)) % 5 != 0
    whole = SegmentStats(7).update(labels, value, noise, mask).result()
    parts = [SegmentStats(7) for _ in range(3)]
    for p, sl in zip(parts, torch.arange(len(labels)).chunk(3), strict=True):
        p.update(labels[sl], value[sl], noise[sl], mask[sl])
    merged = parts[0].copy().merge(parts[1]).merge(parts[2]).result()
    streamed = SegmentStats(7)
    for sl in torch.arange(len(labels)).chunk(5):
        streamed.update(labels[sl], value[sl], noise[sl], mask[sl])
    for other in (merged, streamed.result()):
        for key in whole:
            torch.testing.assert_close(other[key], whole[key])


def test_segment_stats_float32():
    labels, value, noise = _pairs()
    r64 = SegmentStats(7).update(labels, value, noise).result()
    r32 = SegmentStats(7).update(labels, value.float(), noise.float()).result()
    torch.testing.assert_close(r32["mean"], r64["mean"], rtol=1e-6, atol=0)
    torch.testing.assert_close(r32["std"], r64["std"], rtol=1e-5, atol=0)


def test_segment_stats_errors():
    with pytest.raises(ValueError):
        SegmentStats(0)
    with pytest.raises(ValueError):
        SegmentStats(3).result()
    with pytest.raises(ValueError):
        SegmentStats(3).merge(SegmentStats(4))


def test_group_frames_layout():
    x = np.arange(2 * 3 * 4 * 5, dtype="f4").reshape(2, 3, 4, 5)  # c t y x
    lay = Layout("ctyx", spatial="xy", group="c")
    out = list(group_frames(labeled(x, lay), frame=1))
    assert [g for g, _ in out] == [(0,), (1,)]
    for (g,), img in out:
        assert img.shape == (5, 4)
        np.testing.assert_array_equal(img.numpy(), x[g, 1].T)
    # default layout: (*samples, H, W)
    ((g, img),) = group_frames(torch.as_tensor(x), frame=4)
    assert g == () and img.shape == (4, 5)
    np.testing.assert_array_equal(img.numpy(), x.reshape(6, 4, 5)[4])
    with pytest.raises(IndexError):
        next(group_frames(x, frame=6))


# Tolerances of the recovery tests: K observed-information standard
# errors (inverse Hessian of -log L at the ML estimate) plus a small
# absolute tolerance for the known model approximations (normal
# approximation of the estimates, Table IV/V interpolants, rounding).
# Calibration: a Monte Carlo of 60 fits (the three SETTINGS clipped at
# 1000 ADU, 20 independent scenes and noise each) gave z-scores
# (estimate - truth) / SE with mean +0.42..+0.47 for a and -0.26..-0.41
# for b, standard deviation 0.90..1.10 and max |z| = 2.97.
K = 3
WIDE = (-200.0, 4500.0)  # data range enclosing unclipped harness data


def check_recovery(est, a, b, group=(), rtol=0.005, atol_b=0.5):
    s = est.estimates["sensor"]
    se = est.covariance[group].diagonal().sqrt()
    gain, var = s.gain[group].item(), s.variance[group].item()
    assert s.offset is None
    assert est.estimates["sensor"].meta["covariance_ok"][str(list(group))]
    assert se[0] < 0.1 * a  # the test has power
    tol_a = K * se[0].item() + rtol * a
    tol_b = K * se[1].item() + rtol * abs(b) + atol_b
    assert abs(gain - a) < tol_a, (gain, a, se[0].item())
    assert abs(var - b) < tol_b, (var, b, se[1].item())
    return se


@pytest.mark.parametrize("clip", [None, (0.0, 1000.0)])
@pytest.mark.parametrize("a, read_sd, offset", SETTINGS)
def test_recovery(a, read_sd, offset, clip):
    y, a, b, _ = synthetic(a, read_sd, offset, clip=clip)
    est = PoissonGaussian(data_range=clip or WIDE).fit(y)
    assert est.is_faithful is False  # data_range differs from the preset
    assert est.history.converged
    s = est.estimates["sensor"]
    assert s.gain.shape == (1, 1) and s.meta["offset_estimated"] is False
    check_recovery(est, a, b)


@pytest.mark.parametrize("a, read_sd, offset", SETTINGS)
def test_recovery_basic_model(a, read_sd, offset):
    # clipped=False: the basic model of Section III (Eq. 23), on data
    # that are not clipped
    y, a, b, _ = synthetic(a, read_sd, offset)
    est = PoissonGaussian(data_range=WIDE, clipped=False).fit(y)
    assert est.history.converged
    check_recovery(est, a, b)
    # away from the bounds both models agree
    both = PoissonGaussian(data_range=WIDE).fit(y).estimates["sensor"]
    se = est.covariance[()].diagonal().sqrt()
    assert abs(both.gain.item() - est.estimates["sensor"].gain.item()) < (
        0.1 * se[0]
    )


def test_recovery_float_range_from_data():
    # faithful defaults on float data: the range is the minimum and
    # maximum of the frame. The data are clipped on both sides by
    # themselves (nothing planted), so these are the clipping bounds.
    y, a, b, _ = synthetic(0.1, 3.0, 0.0, clip=(0.0, 150.0))
    assert y.min().item() == 0 and y.max().item() == 150
    assert (y == 0).double().mean() > 0.001
    assert (y == 150).double().mean() > 0.01
    est = PoissonGaussian().fit(y)
    assert est.is_faithful
    assert est.estimates["sensor"].meta["data_range"] == {"[]": [0.0, 150.0]}
    check_recovery(est, a, b)


def test_recovery_float32_matches_float64():
    y, a, b, _ = synthetic(2.0, 5.0, 50.0, clip=(0.0, 1000.0))
    e64 = PoissonGaussian(data_range=(0, 1000), dtype=torch.float64).fit(y)
    e32 = PoissonGaussian(data_range=(0, 1000), dtype=torch.float32).fit(y)
    se = e64.covariance[()].diagonal().sqrt()
    s64, s32 = e64.estimates["sensor"], e32.estimates["sensor"]
    assert s32.gain.dtype == torch.float32
    # float32 round-off of the coefficients must be far below the noise
    assert abs(s32.gain.item() - s64.gain.item()) < 0.05 * se[0].item()
    assert abs(s32.variance.item() - s64.variance.item()) < 0.05 * se[1]


def test_z_scores_over_seeds():
    # Calibration of the standard errors: over independent scenes and
    # noise, z = (estimate - truth) / SE must be about N(0, 1). With M
    # seeds, the mean of z has SE 1/sqrt(M) and the sample variance of
    # z is about chi2(M-1)/(M-1). A 20-seed run of this setting gave
    # mean z = +0.47 (a) and -0.41 (b), sd 1.07 and 1.10.
    zs = []
    for seed in range(8):
        y, a, b, _ = synthetic(
            0.5,
            1.5,
            100.0,
            clip=(0.0, 1000.0),
            seed=100 + seed,
            shape=(128, 128),
        )
        est = PoissonGaussian(data_range=(0, 1000)).fit(y)
        s = est.estimates["sensor"]
        se = est.covariance[()].diagonal().sqrt()
        zs.append(
            [
                (s.gain.item() - a) / se[0].item(),
                (s.variance.item() - b) / se[1].item(),
            ]
        )
    z = torch.tensor(zs, dtype=torch.float64)
    m = len(zs)
    # 3.5 standard errors of the mean; sd within the 0.1 % and 99.9 %
    # quantiles of the sample sd of 8 standard normals, 0.29 and 1.87
    assert (z.mean(0).abs() < 3.5 / m**0.5).all(), z
    assert ((z.std(0) > 0.29) & (z.std(0) < 1.87)).all(), z


def test_robust_mad():
    y, a, b, _ = synthetic(0.5, 1.5, 100.0, clip=(0.0, 1000.0))
    est = PoissonGaussian(data_range=(0, 1000), robust=True).fit(y)
    check_recovery(est, a, b)
    # the exact variance factor (extension) only moves the initial point
    exact = PoissonGaussian(
        data_range=(0, 1000), robust=True, mad_correction="exact"
    ).fit(y)
    assert not exact.is_faithful
    se = est.covariance[()].diagonal().sqrt()
    d = exact.estimates["sensor"].gain.item() - est.estimates["sensor"].gain
    assert abs(d.item()) < 0.05 * se[0].item()
    with pytest.raises(ValueError, match="mad_correction"):
        PoissonGaussian(mad_correction="other")


def test_pooled_frames_and_chunking():
    y, a, b, _ = synthetic(0.5, 1.5, 100.0, clip=(0.0, 1000.0), frames=6)
    single = PoissonGaussian(data_range=(0, 1000)).fit(y)
    pooled = PoissonGaussian(data_range=(0, 1000), pooled=True).fit(y)
    se = check_recovery(pooled, a, b)
    # six frames: about sqrt(6) times smaller standard errors
    assert se[0] < 0.6 * single.covariance[()][0, 0].sqrt()
    # streaming in batches of 1, 4 or all frames gives the same level-set
    # statistics, hence the same fit
    ref = pooled.pairs[()]
    for batch in (4, 1):
        with optix.config(batch=batch):
            est = PoissonGaussian(data_range=(0, 1000), pooled=True)
            ((pairs, *_),) = est._pooled_pairs(y).values()
        for key in ("yhat", "shat", "count"):
            torch.testing.assert_close(
                getattr(pairs, key), getattr(ref, key), rtol=1e-9, atol=0
            )


@pytest.mark.parametrize("pooled", [False, True])
def test_tau_doubled_until_two_level_sets(pooled):
    # footnote 2: with a tiny tau, X^smo is (nearly) empty; tau is doubled
    # until there are at least two level sets, in both paths
    y, a, b, _ = synthetic(2.0, 5.0, 50.0, frames=2, shape=(128, 128))
    est = PoissonGaussian(data_range=WIDE, tau=1e-3, pooled=pooled).fit(y)
    tau = est.estimates["sensor"].meta["tau"]["[]"]
    assert tau > 1e-3
    assert len(est.pairs[()].yhat) >= 2


def test_temporal_differencing():
    y, a, b, _ = synthetic(2.0, 5.0, 50.0, clip=(0.0, 1000.0), frames=5)
    est = PoissonGaussian(data_range=(0, 1000), differencing=True).fit(y)
    check_recovery(est, a, b)
    pooled = PoissonGaussian(
        data_range=(0, 1000), differencing=True, pooled=True
    )
    with optix.config(batch=3):  # pairs split across chunks
        pooled.fit(y)
    check_recovery(pooled, a, b)


def test_groups_and_layout():
    ya, a1, b1, _ = synthetic(0.5, 1.5, 100.0, seed=0)
    yb, a2, b2, _ = synthetic(2.0, 5.0, 50.0, seed=3)
    data = torch.stack([ya[0], yb[0]], -1)  # y x c
    lay = Layout("yxc", spatial="yx", group="c")
    est = PoissonGaussian(data_range=WIDE).fit(labeled(data, lay))
    s = est.estimates["sensor"]
    assert s.gain.shape == (2, 1, 1)
    check_recovery(est, a1, b1, (0,))
    check_recovery(est, a2, b2, (1,))


def test_options_and_errors():
    assert PoissonGaussian(faithful=False).options["pooled"] is True
    with pytest.raises(ValueError, match="incompatible"):
        PoissonGaussian(robust=True, pooled=True)
    with pytest.raises(ValueError, match="data_range"):
        PoissonGaussian(data_range="full")
    y = torch.zeros(1, 4, 4, 4)
    with pytest.raises(ValueError, match="two spatial"):
        PoissonGaussian().fit(y, layout=Layout("tzyx", spatial="zyx"))
    with pytest.raises(ValueError, match="empty data range"):
        PoissonGaussian().fit(torch.zeros(1, 32, 32))


# -- integer data: data_range is required ------------------------------------
def test_integer_data_range_required():
    y, *_ = synthetic(0.5, 1.5, 100.0, clip=(0.0, 1000.0))
    for data in (y.round().to(torch.int16), y.round().numpy().astype("u2")):
        with pytest.raises(ValueError, match=r"data_range=\(0, 4095\)"):
            PoissonGaussian().fit(data)
    with pytest.raises(ValueError, match="integer"):
        PoissonGaussian(data_range="auto").fit(y)


def test_integer_int16_clipped():
    # int16 data that saturate at 1000 ADU: the clip is the data range
    y, a, b, _ = synthetic(0.5, 1.5, 100.0, clip=(0.0, 1000.0))
    data = y.round().to(torch.int16)
    est = PoissonGaussian(data_range=(0, 1000)).fit(data)
    check_recovery(est, a, b + 1 / 12)  # rounding adds 1/12 ADU^2


def test_integer_12_bit_in_uint16():
    # 12-bit raw data stored as uint16 and saturating at 4095
    y, a, b, _ = synthetic(2.0, 5.0, 50.0, peak=2400.0, clip=(0.0, 4095.0))
    data = y.round().numpy().astype("uint16")
    assert (data == 4095).mean() > 0.05
    explicit = PoissonGaussian(data_range=(0, 4095)).fit(data)
    check_recovery(explicit, a, b + 1 / 12)
    with pytest.warns(UserWarning, match=r"\(0, 4095\)"):
        auto = PoissonGaussian(data_range="auto").fit(data)
    assert auto.estimates["sensor"].meta["data_range"] == {"[]": [0.0, 4095.0]}
    assert auto.estimates["sensor"].allclose(explicit.estimates["sensor"])


def test_bit_depth_range():
    from optix.sensor._poisson_gaussian import bit_depth_range

    assert bit_depth_range(4000) == 4095
    assert bit_depth_range(4095) == 4095
    assert bit_depth_range(4096) == 8191
    assert bit_depth_range(255.0) == 255
    assert bit_depth_range(0) == 1


# -- covariance (observed information) ---------------------------------------
def _fake_likelihood(monkeypatch, value):
    """Replace -log L by `value(a, b)`; the LS start by the minimum."""
    monkeypatch.setattr(PG, "least_squares_init", lambda *a, **k: (1.0, 2.0))
    monkeypatch.setattr(
        PG, "log_likelihood", lambda a, b, *args, **kw: -value(a, b)
    )


def test_covariance_survives_jitter(monkeypatch):
    # -log L = quadratic with standard errors (0.01, 0.02), plus a ripple
    # of 0.008 nats with the period of four steps of the initial finite
    # differences (0.1 % of the start): the jitter of a quadrature. Its
    # curvature makes the Hessian of those small steps indefinite; steps
    # of half a standard error average it out.
    se = (0.01, 0.02)
    lam = (4e-3, 8e-3)

    def value(a, b):
        q = ((a - 1) / se[0]) ** 2 / 2 + ((b - 2) / se[1]) ** 2 / 2
        ripple = math.cos(2 * math.pi * (a - 1) / lam[0])
        ripple *= math.cos(2 * math.pi * (b - 2) / lam[1])
        return q + 0.008 * ripple

    _fake_likelihood(monkeypatch, value)
    y = synthetic(0.5, 1.5, 100.0, shape=(64, 64))[0] / 1000
    est = PoissonGaussian(data_range=(0.0, 1.0)).fit(y)
    cov = est.covariance[()]
    assert torch.linalg.eigvalsh(cov).min() > 0
    sd = cov.diagonal().sqrt()  # (a, b) in data units = normalised here
    torch.testing.assert_close(
        sd, torch.tensor(se, dtype=torch.float64), rtol=0.1, atol=0
    )
    assert est.estimates["sensor"].meta["covariance_ok"] == {"[]": True}


def test_covariance_not_positive_definite(monkeypatch):
    # a flat likelihood has no curvature: NaN covariance, flag and warning
    _fake_likelihood(monkeypatch, lambda a, b: 0.0)
    y = synthetic(0.5, 1.5, 100.0, shape=(64, 64))[0] / 1000
    with pytest.warns(RuntimeWarning, match="positive definite"):
        est = PoissonGaussian(data_range=(0.0, 1.0)).fit(y)
    assert est.covariance[()].isnan().all()
    assert est.estimates["sensor"].meta["covariance_ok"] == {"[]": False}


# -- threads ---------------------------------------------------------------
def test_fit_leaves_torch_threads_alone():
    # the fit must not change the global number of intra-op threads, not
    # even while it runs (it is visible to callbacks and other threads)
    seen = []
    y, *_ = synthetic(0.5, 1.5, 100.0, shape=(64, 64))
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        PoissonGaussian(data_range=WIDE).fit(
            y, callback=lambda est: seen.append(torch.get_num_threads())
        )
    finally:
        torch.set_num_threads(before)
    assert seen and set(seen) == {2}
