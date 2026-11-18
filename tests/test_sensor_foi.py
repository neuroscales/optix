import numpy as np
import pytest
import torch

from optix.estimates import SensorEstimate
from optix.io import Layout, labeled
from optix.models import simulate_sensor
from optix.sensor import (
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
    y, a, b, sensor = synthetic(a, read_sd, offset, frames=400)
    mean, var = y.mean(0), y.var(0)
    expected = a * mean + b
    rel = (var / expected).mean().item()
    assert rel == pytest.approx(1, abs=0.02)
    est = sensor_estimate(torch.tensor(a, dtype=torch.float64), b, offset)
    torch.testing.assert_close(est.noise_variance(mean), expected)
    torch.testing.assert_close(est.variance, sensor.variance)


def test_sensor_estimate_without_offset():
    est = sensor_estimate(torch.tensor(0.5, dtype=torch.float64), -40.0)
    assert est.meta["offset_estimated"] is False
    y = torch.tensor([100.0, 500.0], dtype=torch.float64)
    torch.testing.assert_close(est.noise_variance(y), 0.5 * y - 40)


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


@pytest.mark.skip(
    reason="blocked: the Foi et al. (2008) method needs the paper "
    "(doi:10.1109/TIP.2008.2001399), not yet available"
)
@pytest.mark.parametrize("clip", [None, (0.0, 1000.0)])
@pytest.mark.parametrize("a, read_sd, offset", SETTINGS)
def test_recovery(a, read_sd, offset, clip):  # pragma: no cover
    y, a, b, _ = synthetic(a, read_sd, offset, clip=clip)
    raise NotImplementedError
