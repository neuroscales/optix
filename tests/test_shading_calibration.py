import pytest
import torch

from optix.estimates import SensorEstimate, ShadingEstimate
from optix.io import Layout, labeled
from optix.models import random_sensor, random_shading, simulate_sensor
from optix.shading import (
    DarkFlatCalibration,
    IlluminationFunction,
    fit_polynomial,
    gaussian_smooth,
    median_smooth,
    robust_minimum,
)

F64 = torch.float64


@pytest.fixture
def setup():
    shape = (32, 24)
    sensor = random_sensor(shape, offset=100.0, read_sd=1.0, seed=0, dtype=F64)
    shading = random_shading(shape, strength=0.2, seed=1, dtype=F64)
    return shape, sensor, shading


@pytest.mark.parametrize("statistic", ["mean", "median"])
def test_dark_flat_recovers_flat(setup, statistic):
    shape, sensor, shading = setup
    n = 200
    dark = simulate_sensor(torch.zeros(n, *shape, dtype=F64), sensor, seed=2)
    flats = simulate_sensor(
        2000 * shading.flat.expand(n, *shape), sensor, seed=3
    )
    est = DarkFlatCalibration(statistic=statistic, dtype=F64)
    est.fit(flats, dark=dark)
    sh = est.estimates["shading"]
    assert sh.meta["flat_norm"] == "mean"
    # flat SE ~ sqrt(gain^2 * 2000 + 1) / (gain * 2000 * sqrt(n)) ~ 0.2 %
    assert (sh.flat / shading.flat - 1).abs().max() < 0.02
    assert (sh.dark - sensor.offset).abs().max() < 0.5
    se = est.estimates["sensor"]
    assert (se.offset - sensor.offset).abs().max() < 0.5
    # correction makes a flat image uniform
    img = simulate_sensor(
        500 * shading.flat.expand(100, *shape), sensor, seed=4
    )
    out = est.apply(img)
    corrected = out.mean(0)
    # shot noise alone: sqrt(500 / 100) / 500 = 0.45 %
    assert corrected.std() / corrected.mean() < 0.01


def test_dark_flat_dark_inputs(setup):
    shape, sensor, shading = setup
    flats = simulate_sensor(
        1000 * shading.flat.expand(50, *shape), sensor, seed=3
    )
    a = DarkFlatCalibration(normalize="none").fit(flats, dark=sensor.offset)
    b = DarkFlatCalibration(normalize="none").fit(
        flats, dark=SensorEstimate(offset=sensor.offset)
    )
    c = DarkFlatCalibration(normalize="none").fit(
        flats, dark=ShadingEstimate(dark=sensor.offset)
    )
    assert a.estimates["shading"].allclose(b.estimates["shading"])
    assert a.estimates["shading"].allclose(c.estimates["shading"])
    assert "sensor" not in a.estimates
    assert not a.is_faithful
    # unnormalised flat: about gain * 1000 electrons
    assert abs(a.estimates["shading"].flat.mean().item() - 500) < 20


def test_dark_flat_layout_groups(setup):
    shape, sensor, shading = setup
    # (c, t, y, x) with one flat per channel
    flats = torch.stack(
        [
            simulate_sensor(
                s * shading.flat.expand(20, *shape), sensor, seed=s
            )
            for s in (500, 1000)
        ]
    )
    darks = torch.stack(
        [
            simulate_sensor(torch.zeros(20, *shape, dtype=F64), sensor, seed=s)
            for s in (1, 2)
        ]
    )
    layout = Layout("ctyx", spatial="yx", group="c")
    est = DarkFlatCalibration(layout=layout).fit(flats, dark=darks)
    assert est.estimates["shading"].flat.shape == (2, *shape)
    out = est.apply(labeled(flats, layout))
    assert out.shape == flats.shape


def test_illumination_function_recovers_smooth_flat():
    shape = (128, 96)
    sensor = random_sensor(shape, offset=100.0, seed=0, dtype=F64)
    shading = random_shading(shape, strength=0.2, seed=1, dtype=F64)
    g = torch.Generator().manual_seed(0)
    # random sparse "cells" on a flat background, many images
    n = 400
    x = 100 + 400 * (torch.rand(n, *shape, generator=g, dtype=F64) > 0.97)
    y = shading.flat * x + sensor.offset
    for smoothing in ("gaussian", "median", "none"):
        est = IlluminationFunction(smoothing=smoothing, size=8).fit(
            y, dark=sensor.offset
        )
        f = est.estimates["shading"].flat
        f = f / f.mean()
        tol = 0.05 if smoothing != "none" else 0.2
        assert (f / shading.flat - 1).abs().max() < tol, smoothing
    # a quadratic flat is recovered by the polynomial fit
    i, j = torch.meshgrid(
        torch.linspace(-1, 1, shape[0], dtype=F64),
        torch.linspace(-1, 1, shape[1], dtype=F64),
        indexing="ij",
    )
    quad = 1 - 0.2 * i * i - 0.1 * j * j + 0.05 * i * j
    est = IlluminationFunction(smoothing="polynomial").fit(quad * x)
    f = est.estimates["shading"].flat
    assert (f / f.mean() / (quad / quad.mean()) - 1).abs().max() < 0.03
    med = IlluminationFunction(statistic="median").fit(y, dark=sensor.offset)
    f = med.estimates["shading"].flat
    assert (f / f.mean() / shading.flat - 1).abs().max() < 0.05
    assert med.is_faithful is False
    out = med.apply(y[:2])
    assert out.shape == (2, *shape)


def test_illumination_rescale_and_faithful():
    y = 1 + torch.rand(10, 20, 20, dtype=F64)
    est = IlluminationFunction(rescale="robust-min", smoothing="none").fit(y)
    f = est.estimates["shading"].flat
    assert f.min() >= 1 and est.estimates["shading"].meta["flat_norm"] == (
        "robust-min"
    )
    assert IlluminationFunction().is_faithful


def test_smoothers():
    x = torch.ones(30, 20, dtype=F64)
    torch.testing.assert_close(gaussian_smooth(x, 3.0), x)
    torch.testing.assert_close(median_smooth(x, 2), x)
    i, j = torch.meshgrid(
        torch.linspace(-1, 1, 30, dtype=F64),
        torch.linspace(-1, 1, 20, dtype=F64),
        indexing="ij",
    )
    q = 1 + 0.3 * i * i - 0.2 * i * j + 0.1 * j
    torch.testing.assert_close(fit_polynomial(q), q)
    z = torch.zeros(9, 9)
    z[4, 4] = 100
    assert median_smooth(z, 1)[4, 4] == 0
    r = robust_minimum(torch.arange(1.0, 101.0))
    assert r.min() == 1
