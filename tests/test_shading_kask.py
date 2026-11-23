import warnings

import pytest
import torch

import optix
from optix.io import Layout, labeled
from optix.models import random_sensor, simulate_sensor
from optix.shading import KaskFlatField
from optix.shading._kask import (
    _CURVATURES,
    _background_mask,
    _global_fit,
    _Grid,
    _initial_profile,
    _mode,
    _Model,
    _noise_image,
    _outliers,
    _radial,
    _random_error,
)

SHAPE = (128, 136)
F64 = torch.float64


def dome(c, cy, cx, shape=SHAPE):
    """Normalised paraboloid 1 − c/2 · ((x − cx)² + (y − cy)²)."""
    y = torch.linspace(-1, 1, shape[0], dtype=F64)[:, None] - cy
    x = torch.linspace(-1, 1, shape[1], dtype=F64)[None, :] - cx
    p = 1 - 0.5 * c * (x.square() + y.square())
    return p / p.mean()


BACKGROUND = dome(0.3, 0.05, -0.05)
FOREGROUND = dome(0.5, -0.15, 0.2)


def simulate(
    n,
    *,
    seed=0,
    density=0.003,
    amp=(2000.0, 2500.0),
    full=(),
    level=(80.0, 40.0),
    sensor=None,
):
    """Images N = S_f · U + B · S_b with bright disks (U).

    Frames listed in `full` are covered by a bright textured layer (no
    background). B is uniform in [level[0], level[0] + level[1]]. The
    noise is Poisson (gain 1, no offset), or that of `sensor`
    ([`simulate_sensor`][optix.models.simulate_sensor], U and B in
    electrons).
    """
    g = torch.Generator().manual_seed(seed)
    H, W = SHAPE
    yy = torch.arange(H, dtype=F64)[:, None]
    xx = torch.arange(W, dtype=F64)[None, :]
    frames, signals, levels = [], [], []
    for k in range(n):
        u = torch.zeros(SHAPE, dtype=F64)
        m = int(torch.poisson(torch.tensor(density * H * W), generator=g))
        for _ in range(m):
            cy, cx = (torch.rand(2, generator=g, dtype=F64) * 1.0).tolist()
            r = 3 + 2 * float(torch.rand(1, generator=g))
            a = amp[0] + (amp[1] - amp[0]) * float(torch.rand(1, generator=g))
            disk = (yy - cy * H).square() + (xx - cx * W).square() < r * r
            u = torch.where(disk, a, u)
        if k in full:
            u = 500 + 400 * torch.rand(SHAPE, generator=g, dtype=F64)
        b = level[0] + level[1] * float(torch.rand(1, generator=g))
        rate = FOREGROUND * u + b * BACKGROUND
        if sensor is None:
            frames.append(torch.poisson(rate, generator=g))
        else:
            frames.append(simulate_sensor(rate, sensor, seed=seed * 1000 + k))
        signals.append(u)
        levels.append(b)
    return (
        torch.stack(frames),
        torch.stack(signals),
        torch.tensor(levels, dtype=F64),
    )


def rms(a, b):
    return float((a / b - 1).square().mean().sqrt())


@pytest.fixture(scope="module")
def fitted():
    data, signal, levels = simulate(256, seed=1)
    est = KaskFlatField(dtype=F64).fit(data)
    return est, data, signal, levels


# -- building blocks -------------------------------------------------------
def test_grid_roundtrip():
    grid = _Grid((32, 34), (4, 5), "cpu")
    x = torch.ones(32, 34, dtype=F64)
    assert grid.sum(x).sum() == 32 * 34
    # a plane is reproduced by the bin averages and their interpolation
    u = torch.arange(32, dtype=F64)[:, None] + 2 * torch.arange(34)[None]
    avg = grid.sum(u) / grid.sum(x)
    torch.testing.assert_close(grid.upsample(avg), u)
    assert grid.expand(avg).shape == (32, 34)


def test_noise_image_constant():
    """Eq. 4: var/mean = κ for scaled Poisson data, whatever the profile."""
    g = torch.Generator().manual_seed(0)
    profile = BACKGROUND
    for gain in (1.0, 3.0):
        x = gain * torch.poisson(500 * profile, generator=g)
        kappa = _mode(_noise_image(x, profile, 9))
        assert float(kappa) == pytest.approx(gain, rel=0.1)


def test_initial_curvature():
    """The centred curvature minimising the noise level is found."""
    rho2 = _radial(SHAPE, torch.zeros((), dtype=F64))
    g = torch.Generator().manual_seed(0)
    x = torch.poisson(1e5 * (1 - 0.3 * rho2), generator=g)
    best = _initial_profile(x, 9)
    c = [float((1 - best)[0, 0]) / float(rho2[0, 0])]
    assert c[0] == pytest.approx(0.3, abs=0.051)
    assert min(_CURVATURES) == 0


def test_background_mask():
    """Edges and bright flat objects are excluded (noise and intensity)."""
    data, signal, _ = simulate(1, seed=3, density=0.002)
    opts = KaskFlatField().options
    mask, kappa = _background_mask(data[0], BACKGROUND, opts)
    assert float(kappa) == pytest.approx(1.0, rel=0.15)
    objects = signal[0] > 0
    assert not bool((mask & objects).any())
    assert float(mask[~objects].to(F64).mean()) > 0.6


def test_regularization_eq5():
    """R = 0 reproduces the data; bilinear surfaces cost nothing."""
    grid = _Grid((64, 72), (8, 9), "cpu")  # equal bins: uniform centres
    u, v = grid.coords(full=False)
    plane = (1 + 0.2 * u + 0.1 * v + 0.05 * u * v).expand(8, 9)
    g = torch.Generator().manual_seed(0)
    noisy = plane + 0.01 * torch.randn(8, 9, generator=g, dtype=F64)
    w = torch.ones(8, 9, dtype=F64)
    exact = _Model(grid, "regularization", 0).smooth(noisy, w)[0]
    torch.testing.assert_close(exact, noisy)
    torch.testing.assert_close(
        _Model(grid, "regularization", 1e4).smooth(plane, w)[0], plane
    )
    smooth = _Model(grid, "regularization", 1e6).smooth(noisy, w)[0]
    assert float((smooth - plane).abs().max()) < 0.01


def test_polynomial_model():
    grid = _Grid((64, 68), (8, 9), "cpu")
    u, v = grid.coords(full=False)
    quartic = (1 + 0.1 * u**4 - 0.2 * u * v**3 + 0.3 * v**2).expand(8, 9)
    model = _Model(grid, "polynomial", 0)
    t, params = model.smooth(quartic, torch.ones(8, 9, dtype=F64))
    torch.testing.assert_close(t, quartic)
    full = model.full(t, params)
    uu, vv = grid.coords(full=True)
    truth = 1 + 0.1 * uu**4 - 0.2 * uu * vv**3 + 0.3 * vv**2
    torch.testing.assert_close(full, truth / truth.mean())


def test_global_fit_scales():
    """E_k = a_k · T is fitted exactly, with undefined pixels."""
    grid = _Grid((64, 68), (8, 9), "cpu")
    T = grid.sum(BACKGROUND[:64, :68]) / grid.sum(torch.ones(64, 68))
    T = (T / T.mean()).to(F64)
    a = torch.tensor([50.0, 100.0, 200.0], dtype=F64)
    E = a[:, None, None] * T
    E[0, :3, :3] = float("nan")
    fit, scale, _ = _global_fit(E, _Model(grid, "polynomial", 0))
    torch.testing.assert_close(fit, T, rtol=1e-3, atol=0)
    torch.testing.assert_close(scale, a, rtol=1e-3, atol=0)


def test_outlier_filter_and_random_error():
    grid = _Grid((64, 68), (8, 9), "cpu")
    model = _Model(grid, "regularization", 0.2)
    g = torch.Generator().manual_seed(0)
    T = torch.ones(8, 9, dtype=F64)
    E = 100 * (T + 0.01 * torch.randn(20, 8, 9, generator=g, dtype=F64))
    E[7] = E[7] * (1 + 0.3 * torch.linspace(-1, 1, 9, dtype=F64))
    keep = _outliers(E, model, 3.0)
    assert keep.sum() == 19 and not keep[7]
    small = _random_error(E[keep], model, 4)
    large = _random_error(E[keep] * (1 + 3 * (E[keep] / 100 - 1)), model, 4)
    assert 0 < small < large
    assert _random_error(E[:3], model, 4) == float("inf")


# -- estimator -------------------------------------------------------------
def test_recovers_both_profiles(fitted):
    est, *_ = fitted
    s = est.estimates["shading"]
    background = s.dark / s.dark.mean()
    assert rms(background, BACKGROUND) < 0.005
    assert rms(s.flat, FOREGROUND) < 0.02
    # the two profiles are told apart
    assert rms(s.flat, FOREGROUND) < 0.5 * rms(s.flat, BACKGROUND)
    assert float(s.flat.mean()) == pytest.approx(1.0)
    assert s.meta["flat_norm"] == "mean"
    assert s.meta["dark_scaling"] == "per-frame"
    assert s.meta["background_error"] < 0.02
    assert s.meta["foreground_error"] < 0.04
    assert s.meta["foreground_groups"] == 16
    assert est.is_faithful


def test_background_levels(fitted):
    est, _, _, levels = fitted
    torch.testing.assert_close(
        est.background_level_, levels, rtol=0.01, atol=0
    )
    s = est.estimates["shading"]
    assert float(s.dark.mean()) == pytest.approx(float(levels.mean()), 0.01)


def test_apply_eq3(fitted):
    """Û = (N − B̂ S_b)/S_f + B̂ restores U + B."""
    est, _, _, _ = fitted
    data, signal, levels = simulate(8, seed=7)
    out = est.apply(data)
    expected = signal + levels[:, None, None]
    bg = signal == 0
    assert float((out - expected)[bg].mean().abs()) < 1.0
    rel = (out - expected)[~bg] / expected[~bg]
    assert float(rel.mean().abs()) < 0.02
    # the corrected background is flat: no residual profile
    resid = torch.where(bg, out - levels[:, None, None], 0).sum(0)
    resid = resid / bg.sum(0).clamp_min(1)
    grid = _Grid(SHAPE, (4, 4), "cpu")
    blocks = grid.sum(resid) / grid.sum(torch.ones(SHAPE, dtype=F64))
    assert float(blocks.abs().max()) < 2.0  # ≪ the ±30 ADU of shading


def test_low_background_images_discarded():
    data, *_ = simulate(40, seed=2, full=(3, 11, 20))
    est = KaskFlatField(dtype=F64, max_error=(None, None)).fit(data)
    meta = est.estimates["shading"].meta
    assert meta["background_images"] == 37
    # B̄ averages the levels of the fitted images only
    used = torch.ones(40, dtype=torch.bool)
    used[[3, 11, 20]] = False
    expected = float(est.background_level_[used].nanmean())
    assert meta["background_level"] == pytest.approx(expected, rel=1e-9)


def test_noise_constant_series():
    """κ is one constant per group, estimated from the series (paper)."""
    data, *_ = simulate(24, seed=8)
    est = KaskFlatField(dtype=F64, max_error=(None, None)).fit(data)
    kappa = est.estimates["shading"].meta["noise_constant"]
    assert kappa == pytest.approx(1.0, rel=0.1)  # Poisson, gain 1
    assert est.options["kappa"] == KaskFlatField.paper["kappa"] == "series"
    fixed = KaskFlatField(dtype=F64, max_error=(None, None), kappa=kappa)
    a = est.estimates["shading"]
    b = fixed.fit(data).estimates["shading"]
    torch.testing.assert_close(a.dark, b.dark, rtol=0.01, atol=0)
    per = KaskFlatField(dtype=F64, max_error=(None, None), kappa="per-frame")
    per.fit(data)
    assert not per.is_faithful
    assert per.estimates["shading"].meta["noise_constant"] is None
    assert per.apply(data[:2]).shape == data[:2].shape
    with pytest.raises(ValueError):
        KaskFlatField(kappa="global")
    with pytest.raises(ValueError):
        KaskFlatField(kappa=-1.0)


def test_camera_sensor():
    """Gain ≠ 1, a camera offset and read noise (optix.models)."""
    gain, offset, level = 2.0, 100.0, 60.0
    sensor = random_sensor(
        SHAPE, offset=offset, offset_sd=0.0, gain=gain, seed=0, dtype=F64
    )
    data, signal, _ = simulate(
        96, seed=9, amp=(1000.0, 1250.0), level=(level, 0.0), sensor=sensor
    )
    est = KaskFlatField(dtype=F64, max_error=(None, None)).fit(data)
    s = est.estimates["shading"]
    # var/mean of the background: g · B S_b / (o + g · B S_b) < g
    kappa = s.meta["noise_constant"]
    assert 0.5 * gain < kappa < gain
    dark = offset + gain * level * BACKGROUND  # in ADU
    assert rms(s.dark, dark) < 0.005
    assert rms(s.flat, FOREGROUND) < 0.03


def test_quality_control():
    data, *_ = simulate(20, seed=4)
    with pytest.warns(RuntimeWarning, match="foreground profile"):
        est = KaskFlatField(dtype=F64, max_error=(None, 0.04)).fit(data)
    s = est.estimates["shading"]
    assert bool((s.flat == 1).all())
    assert s.meta["foreground_error"] is None  # inf: fewer than 4 averages
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est = KaskFlatField(dtype=F64, max_error=(None, None)).fit(data)
    assert not bool((est.estimates["shading"].flat == 1).all())
    assert not est.is_faithful


def test_background_failure_skips_foreground():
    data, *_ = simulate(20, seed=4)
    with pytest.warns(RuntimeWarning, match="background profile"):
        est = KaskFlatField(dtype=F64, max_error=(1e-9, None)).fit(data)
    s = est.estimates["shading"]
    assert bool((s.flat == 1).all())
    assert torch.allclose(s.dark, s.dark.mean().expand_as(s.dark))
    assert s.meta["foreground_groups"] == 0


def test_polynomial_profiles(fitted):
    _, data, _, _ = fitted
    est = KaskFlatField(dtype=F64, model="polynomial").fit(data)
    s = est.estimates["shading"]
    assert rms(s.dark / s.dark.mean(), BACKGROUND) < 0.005
    assert rms(s.flat, FOREGROUND) < 0.02


# -- streaming, layouts, precision ----------------------------------------
def test_streaming_parity(fitted):
    est, data, _, _ = fitted
    with optix.config(memory=SHAPE[0] * SHAPE[1] * 8 * 8 * 3):
        small = KaskFlatField(dtype=F64).fit(data)
    a, b = est.estimates["shading"], small.estimates["shading"]
    torch.testing.assert_close(a.flat, b.flat, rtol=0, atol=1e-12)
    torch.testing.assert_close(a.dark, b.dark, rtol=0, atol=1e-9)


def test_group_layout():
    data, *_ = simulate(40, seed=5)
    other, *_ = simulate(40, seed=6)
    both = torch.stack([data, other])  # (channel, frame, H, W)
    layout = Layout(spatial=(2, 3), sample=(1,), group=(0,))
    kw = {"dtype": F64, "max_error": (None, None)}
    est = KaskFlatField(**kw).fit(labeled(both, layout))
    s = est.estimates["shading"]
    assert s.flat.shape == (2, *SHAPE)
    assert est.background_level_.shape == (2, 40)
    one = KaskFlatField(**kw).fit(other).estimates["shading"]
    torch.testing.assert_close(s.flat[1], one.flat)
    torch.testing.assert_close(s.dark[1], one.dark)
    assert len(s.meta["background_error"]) == 2
    out = est.apply(labeled(both[:, :2], layout))
    assert out.shape == (2, 2, *SHAPE)


def test_float32_matches_float64(fitted):
    est, data, _, _ = fitted
    est32 = KaskFlatField(dtype=torch.float32).fit(data.float())
    a, b = est.estimates["shading"], est32.estimates["shading"]
    assert a.flat.dtype == torch.float64 and b.flat.dtype == torch.float32
    torch.testing.assert_close(b.flat.double(), a.flat, rtol=0, atol=0.01)
    torch.testing.assert_close(b.dark.double(), a.dark, rtol=0.01, atol=0)


def test_save_load(fitted, tmp_path):
    est, data, _, _ = fitted
    path = tmp_path / "kask.pt"
    est.save(path)
    new = optix.load(path)
    torch.testing.assert_close(new.background_level_, est.background_level_)
    new.to(dtype=F64)
    torch.testing.assert_close(new.apply(data[:2]), est.apply(data[:2]))


def test_options_validated():
    with pytest.raises(ValueError):
        KaskFlatField(model="spline")
    with pytest.raises(ValueError):
        KaskFlatField(window=8)
    with pytest.raises(ValueError):
        KaskFlatField(bins=(2, 17))
    with pytest.raises(RuntimeError):
        KaskFlatField().apply(torch.zeros(1, *SHAPE))
