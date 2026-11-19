import numpy as np
import pytest
import torch

from optix.estimates import SensorEstimate, ShadingEstimate
from optix.models import (
    random_sensor,
    random_shading,
    simulate_sensor,
    simulate_shading,
)
from optix.stats import Moments, reduce


def test_random_sensor(device, dtype):
    s = random_sensor(
        (64, 32),
        gain_sd=0.1,
        defective=0.1,
        seed=0,
        dtype=dtype,
        device=device,
    )
    assert isinstance(s, SensorEstimate)
    assert s.offset.shape == (64, 32) and s.offset.dtype == dtype
    assert s.offset.device.type == device.type
    assert (s.gain > 0).all()
    assert s.defective.dtype == torch.bool
    assert 0.05 < s.defective.float().mean().item() < 0.15
    torch.testing.assert_close(s.variance, torch.full_like(s.variance, 2.25))
    again = random_sensor(
        (64, 32),
        gain_sd=0.1,
        defective=0.1,
        seed=0,
        dtype=dtype,
        device=device,
    )
    assert again.allclose(s)


def test_simulate_sensor_statistics():
    s = random_sensor((8, 8), seed=1, dtype=torch.float64)
    signal = torch.full((4000, 8, 8), 50.0, dtype=torch.float64)
    y = simulate_sensor(signal, s, seed=2)
    mean = s.offset + s.gain * 50
    torch.testing.assert_close(y.mean(0), mean, atol=0.2, rtol=0)
    torch.testing.assert_close(
        y.var(0), s.noise_variance(mean), rtol=0.15, atol=0
    )
    q = simulate_sensor(signal[:3], s, seed=2, quantize=torch.uint8)
    assert q.dtype == torch.uint8
    with pytest.raises(ValueError):
        simulate_sensor(signal, s.replace(gain=None))


def test_random_shading():
    sh = random_shading((32, 24), strength=0.2, dark=5.0, seed=3)
    assert isinstance(sh, ShadingEstimate)
    assert sh.flat.mean().item() == pytest.approx(1, rel=1e-5)
    assert sh.flat.log().std().item() == pytest.approx(0.2, rel=1e-3)
    assert sh.meta["flat_norm"] == "mean"
    assert sh.dark.eq(5).all()
    assert random_shading((32, 24), seed=3).allclose(
        random_shading((32, 24), seed=3)
    )


def test_simulate_shading():
    sh = random_shading((4, 5), dark=2.0, seed=0)
    sh = sh.replace(baseline=torch.arange(3.0))
    x = torch.ones(3, 4, 5)
    y = simulate_shading(x, sh)
    torch.testing.assert_close(
        y, sh.flat + 2 + torch.arange(3.0)[:, None, None]
    )
    torch.testing.assert_close(simulate_shading(x, ShadingEstimate()), x)


def test_end_to_end_zarr_recovers_sensor():
    zarr = pytest.importorskip("zarr")
    sensor = random_sensor((16, 12), read_sd=3.0, seed=4, dtype=torch.float64)
    shading = random_shading((16, 12), seed=5, dtype=torch.float64)
    # dark frames: the scene is zero, so only the sensor contributes
    scene = torch.zeros(2, 300, 16, 12, dtype=torch.float64)
    electrons = simulate_shading(scene, shading.replace(dark=None))
    y = simulate_sensor(electrons, sensor, seed=6, quantize=torch.uint16)
    z = zarr.create_array(
        store={}, shape=y.shape, chunks=(1, 50, 16, 12), dtype="u2"
    )
    z[:] = y.numpy()
    (m,) = reduce(z, Moments(), batch=25, dtype=torch.float64)
    torch.testing.assert_close(m.mean, sensor.offset, atol=0.6, rtol=0)
    # rounding adds 1/12 ADU^2 of variance
    torch.testing.assert_close(
        m.var(), sensor.variance + 1 / 12, rtol=0.25, atol=0
    )
    np.testing.assert_array_equal(z[0, 0], y[0, 0].numpy())


def test_simulate_gamma_mixture():
    from optix.models import simulate_gamma_mixture

    draw = simulate_gamma_mixture(
        (50, 32, 32), [10.0, 100.0], [3.0, 2.0], concentration=5.0, seed=0
    )
    assert draw.signal.shape == (50, 32, 32)
    assert draw.proportions.shape == (50, 3)
    bg = draw.labels == 2
    assert (draw.signal[bg] == 0).all() and (draw.signal[~bg] > 0).all()
    for k, (mean, shape) in enumerate([(10.0, 3.0), (100.0, 2.0)]):
        u = draw.signal[draw.labels == k].double()
        assert abs(u.mean() / mean - 1) < 0.03
        assert abs(u.mean() ** 2 / u.var() / shape - 1) < 0.1
    again = simulate_gamma_mixture(
        (50, 32, 32), [10.0, 100.0], [3.0, 2.0], concentration=5.0, seed=0
    )
    assert torch.equal(again.signal, draw.signal)
