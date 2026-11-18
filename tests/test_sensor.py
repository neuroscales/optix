import math

import numpy as np
import pytest
import torch

import optix
from optix.estimates import SensorEstimate
from optix.io import Layout, labeled
from optix.models import random_sensor, simulate_sensor
from optix.sensor import (
    DefectivePixels,
    GeneralizedAnscombe,
    PhotonTransferCurve,
    SCMOSCalibration,
    anscombe,
    bimodality,
    gain_regression,
    inverse_anscombe,
    pixel_stats,
    replace_defective,
)

F64 = torch.float64


def _stack(sensor, electrons, n, shape, seed, quantize=None):
    signal = torch.full((n, *shape), float(electrons), dtype=F64)
    return simulate_sensor(signal, sensor, seed=seed, quantize=quantize)


@pytest.fixture
def sensor():
    return random_sensor(
        (16, 12),
        offset=100.0,
        offset_sd=3.0,
        read_sd=1.5,
        gain=0.5,
        gain_sd=0.05,
        seed=0,
        dtype=F64,
    )


# -- pixel statistics --------------------------------------------------------


def test_pixel_stats_groups_and_median():
    g = torch.Generator().manual_seed(0)
    y = torch.randn(3, 20, 5, 4, generator=g, dtype=F64) + torch.arange(
        3, dtype=F64
    ).view(3, 1, 1, 1)
    st = pixel_stats(labeled(y, Layout(spatial=(-2, -1), group=0)))
    assert st.mean.shape == (3, 5, 4)
    torch.testing.assert_close(st.mean, y.mean(1).to(st.mean))
    torch.testing.assert_close(st.var, y.var(1).to(st.var))
    med = pixel_stats(y[0], statistic="median")
    torch.testing.assert_close(med.mean, y[0].median(0).values.to(med.mean))


# -- sCMOS -------------------------------------------------------------------


@pytest.mark.parametrize("robust", ["none", "huber", "theil-sen"])
def test_scmos_recovers_maps(sensor, robust):
    shape = sensor.offset.shape
    dark = _stack(sensor, 0, 400, shape, seed=1)
    levels = [
        _stack(sensor, e, 300, shape, seed=10 + i)
        for i, e in enumerate([50, 150, 300, 600, 1000])
    ]
    est = SCMOSCalibration(robust=robust, dtype=F64).fit(levels, dark=dark)
    s = est.estimates["sensor"]
    assert est.is_faithful == (robust == "none")
    # offset: SE = 1.5 / sqrt(400) = 0.075
    assert (s.offset - sensor.offset).abs().max() < 0.4
    # read variance: relative SE sqrt(2 / 400) = 7 %
    rel_var = (s.variance / sensor.variance - 1).abs()
    assert rel_var.mean() < 0.1
    rel_gain = s.gain / sensor.gain - 1
    assert rel_gain.abs().mean() < 0.08
    assert rel_gain.mean().abs() < 0.02
    # apply converts to electrons
    e = est.apply(levels[2][:5])
    assert abs(e.mean().item() - 300) < 10


def test_scmos_confidence_mask(sensor):
    shape = sensor.offset.shape
    dark = _stack(sensor, 0, 100, shape, seed=1)
    levels = [_stack(sensor, e, 100, shape, seed=e) for e in (100, 400, 900)]
    est = SCMOSCalibration(confidence=1e-6).fit(levels, dark=dark)
    assert est.estimates["sensor"].defective.all()
    est = SCMOSCalibration(confidence=10.0).fit(levels, dark=dark)
    assert not est.estimates["sensor"].defective.any()
    assert not est.is_faithful


def test_gain_regression_robust_outlier():
    a = torch.tensor([10.0, 20, 30, 40, 50, 60], dtype=F64)[:, None]
    b = 0.5 * a
    b[5] = 100.0  # outlier
    g_ols, _ = gain_regression(a, b)
    g_hub, _ = gain_regression(a, b, robust="huber")
    g_ts, _ = gain_regression(a, b, robust="theil-sen")
    assert abs(g_ols.item() - 0.5) > 0.1
    assert abs(g_hub.item() - 0.5) < 0.05
    assert abs(g_ts.item() - 0.5) < 1e-12


def test_scmos_streams_zarr(sensor, tmp_path):
    zarr = pytest.importorskip("zarr")
    shape = sensor.offset.shape
    dark = _stack(sensor, 0, 50, shape, seed=1).numpy()
    lev = [_stack(sensor, e, 50, shape, seed=e).numpy() for e in (100, 500)]

    def z(x):
        arr = zarr.create_array(
            store={}, shape=x.shape, chunks=(7, *shape), dtype="f8"
        )
        arr[:] = x
        return arr

    a = SCMOSCalibration(dtype=F64).fit([z(x) for x in lev], dark=z(dark))
    b = SCMOSCalibration(dtype=F64).fit(lev, dark=dark)
    assert a.estimates["sensor"].allclose(b.estimates["sensor"], rtol=1e-8)


def test_scmos_layout_groups():
    sensor = random_sensor((6, 5), gain=0.5, seed=0, dtype=F64)

    # storage (y, x, c, t): channels as groups, frames last
    def stack(e, seed):
        y = _stack(sensor, e, 2 * 100, (6, 5), seed=seed)
        y = y.reshape(2, 100, 6, 5)
        return y.permute(2, 3, 0, 1).contiguous()

    layout = Layout("yxct", spatial="yx", group="c", sample="t")
    est = SCMOSCalibration(layout=layout, dtype=F64).fit(
        [stack(200, 2), stack(800, 3)], dark=stack(0, 1)
    )
    s = est.estimates["sensor"]
    assert s.gain.shape == (2, 6, 5)
    assert (s.gain.mean() / 0.5 - 1).abs() < 0.05


# -- photon transfer ---------------------------------------------------------


def test_ptc_recovers_gain_and_read_noise():
    shape = (128, 128)
    k, read_sd = 0.4, 2.0
    sensor = random_sensor(
        shape,
        offset=200.0,
        offset_sd=4.0,
        read_sd=read_sd,
        gain=k,
        gain_sd=0.0,
        seed=0,
        dtype=F64,
    )
    q = torch.int32
    dark = _stack(sensor, 0, 2, shape, seed=1, quantize=q)
    levels = [
        _stack(sensor, e, 2, shape, seed=10 + i, quantize=q)
        for i, e in enumerate(np.linspace(100, 5000, 10))
    ]
    est = PhotonTransferCurve(dtype=F64).fit(levels, dark=dark)
    s = est.estimates["sensor"]
    assert est.is_faithful
    # relative SE of the variance per level: sqrt(2 / 128^2) = 1.1 %
    assert abs(s.gain.item() / k - 1) < 0.03
    # read-noise variance (DN^2) with the quantisation removed
    assert abs(s.variance.item() / read_sd**2 - 1) < 0.05
    assert abs(s.meta["read_noise_e"] - read_sd / k) < 0.3
    # DSNU: offset_sd / K electrons; no PRNU (gain_sd = 0)
    assert abs(s.meta["dsnu_e"] - 4.0 / k) < 1.0
    assert s.meta["prnu"] < 0.01
    assert est.curve["used"].sum() == 7  # up to 70 % of the top level


def test_ptc_prnu_and_per_pixel():
    shape = (64, 64)
    sensor = random_sensor(
        shape, gain=0.5, gain_sd=0.02, read_sd=1.0, seed=3, dtype=F64
    )
    dark = _stack(sensor, 0, 20, shape, seed=1)
    levels = [_stack(sensor, e, 20, shape, seed=e) for e in (500, 1000, 2000)]
    est = PhotonTransferCurve(quantization=0.0, per_pixel=True)
    est.fit(levels, dark=dark)
    s = est.estimates["sensor"]
    assert s.gain.shape == shape
    assert abs(s.meta["prnu"] - 0.02 / 0.5) < 0.006
    assert not est.is_faithful


def test_ptc_needs_two_frames(sensor):
    one = _stack(sensor, 10, 1, sensor.offset.shape, seed=0)
    with pytest.raises(ValueError, match="two frames"):
        PhotonTransferCurve().fit([one], dark=one)


# -- defective pixels --------------------------------------------------------


def test_defective_pixels_detects_each_kind():
    shape = (40, 40)
    sensor = random_sensor(
        shape, offset_sd=1.0, read_sd=1.5, seed=0, dtype=F64
    )
    n = 400
    y = _stack(sensor, 0, n, shape, seed=1)
    g = torch.Generator().manual_seed(5)
    y[:, 3, 4] += 60  # hot
    y[:, 10, 10] += 8 * torch.randn(n, generator=g, dtype=F64)  # noisy
    y[:, 20, 30] += 20 * (torch.rand(n, generator=g, dtype=F64) < 0.5)  # RTS
    est = DefectivePixels().fit(y)
    m = est.masks
    assert m["offset"][3, 4] and m["variance"][10, 10] and m["rts"][20, 30]
    d = est.estimates["sensor"].defective
    assert d.sum() <= 6
    # replacement by the median of the good neighbours
    out = est.apply(y[:3])
    for i, j in zip(*d.nonzero(as_tuple=True), strict=True):
        nb = y[:3, i - 1 : i + 2, j - 1 : j + 2].reshape(3, -1)
        ok = ~d[i - 1 : i + 2, j - 1 : j + 2].reshape(-1)
        ok[4] = False
        ref = nb[:, ok].nanmedian(-1).values
        torch.testing.assert_close(out[:, i, j], ref.to(out))
    good = ~d
    torch.testing.assert_close(out[:, good], y[:3, good].to(out))


def test_bimodality_values():
    assert abs(bimodality(torch.tensor(0.0), torch.tensor(3.0)) - 1 / 3) < 1e-7
    assert abs(bimodality(torch.tensor(0.0), torch.tensor(1.0)) - 1) < 1e-7


def test_replace_defective_corner():
    y = torch.arange(9.0).reshape(1, 3, 3)
    d = torch.zeros(3, 3, dtype=torch.bool)
    d[0, 0] = d[0, 1] = True
    out = replace_defective(y, d)
    # good neighbours of (0, 0): 3, 4 and of (0, 1): 2, 3, 4, 5 (lower
    # medians)
    assert out[0, 0, 0] == 3 and out[0, 0, 1] == 3
    assert torch.equal(out[0, 1:], y[0, 1:])


# -- generalised Anscombe ----------------------------------------------------


def test_anscombe_stabilises_variance():
    g = torch.Generator().manual_seed(0)
    alpha, sigma, mu = 2.0, 3.0, 10.0
    for y in (20.0, 100.0, 1000.0):
        z = (
            alpha
            * torch.poisson(
                torch.full((200000,), y / alpha, dtype=F64), generator=g
            )
            + mu
            + sigma * torch.randn(200000, generator=g, dtype=F64)
        )
        assert abs(anscombe(z, alpha, sigma, mu).var().item() - 1) < 0.03


@pytest.mark.parametrize("sigma", [0.0, 0.5, 2.0])
def test_exact_inverse_is_unbiased(sigma):
    """inverse(E[f(z) | y]) == y, by Monte Carlo."""
    g = torch.Generator().manual_seed(1)
    alpha, mu = 1.5, 5.0
    ys = torch.tensor([0.5, 2.0, 5.0, 20.0, 100.0, 800.0], dtype=F64)
    n = 400000
    ef = []
    for y in ys.tolist():
        z = (
            alpha
            * torch.poisson(
                torch.full((n,), y / alpha, dtype=F64), generator=g
            )
            + mu
            + sigma * torch.randn(n, generator=g, dtype=F64)
        )
        ef.append(anscombe(z, alpha, sigma, mu).mean())
    ef = torch.stack(ef)
    exact = inverse_anscombe(ef, alpha, sigma, mu) - mu
    alg = inverse_anscombe(ef, alpha, sigma, mu, method="algebraic") - mu
    # MC SE of E[f] is 1/sqrt(n); dy/dD = alpha * D / 2
    tol = 4 * alpha * ef / 2 / math.sqrt(n) + 1e-3
    assert ((exact - ys).abs() < tol).all(), (exact, ys)
    assert ((alg - ys).abs() > tol)[:3].any()  # the algebraic one is biased
    asym = inverse_anscombe(ef, alpha, sigma, mu, method="asymptotic") - mu
    assert ((asym - ys).abs() < tol)[-2:].all()


def test_inverse_round_trip_and_float32():
    z = torch.linspace(0, 500, 101, dtype=F64)
    d = anscombe(z, 1.0, 1.0, 0.0)
    back = inverse_anscombe(d, 1.0, 1.0, 0.0, method="algebraic")
    torch.testing.assert_close(back[d > 0], z[d > 0])
    e64 = inverse_anscombe(d, 1.0, 1.0, 0.0)
    e32 = inverse_anscombe(d.float(), 1.0, 1.0, 0.0)
    torch.testing.assert_close(e32.double(), e64, atol=2e-3, rtol=1e-5)


def test_generalized_anscombe_estimator(sensor):
    shape = sensor.offset.shape
    y = _stack(sensor, 200, 4, shape, seed=1)
    gat = GeneralizedAnscombe().fit(init=sensor)
    s = gat.estimates["sensor"]
    assert s.gain.ndim == 0  # scalar parameters by default
    d = gat.apply(y)
    torch.testing.assert_close(
        d, anscombe(y, s.gain, s.variance.sqrt(), s.offset).to(d)
    )
    back = gat.inverse(d)
    assert back.shape == y.shape
    pp = GeneralizedAnscombe(per_pixel=True).fit(init=sensor)
    assert pp.estimates["sensor"].gain.shape == shape and not pp.is_faithful
    back = pp.inverse(pp.apply(y))
    assert (back.mean(0) - y.mean(0)).abs().mean() < 3
    direct = GeneralizedAnscombe(alpha=2.0, sigma=1.0, mu=3.0)
    assert direct.estimates["sensor"].gain.item() == 2.0
    with pytest.raises(ValueError, match="init"):
        GeneralizedAnscombe().fit(None)


def test_registered_and_saved(sensor, tmp_path):
    names = optix.estimators()
    for cls in (SCMOSCalibration, PhotonTransferCurve, DefectivePixels):
        assert cls.__name__ in names
    shape = sensor.offset.shape
    dark = _stack(sensor, 0, 30, shape, seed=1)
    est = DefectivePixels().fit(dark)
    est.save(tmp_path / "d.pt")
    back = optix.load(tmp_path / "d.pt")
    assert back.estimates["sensor"].allclose(est.estimates["sensor"])
    assert set(back.masks) == set(est.masks)
    assert isinstance(back.estimates["sensor"], SensorEstimate)
