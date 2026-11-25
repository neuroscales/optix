"""ACsN (Mandracchia et al. 2020), on simulated sCMOS data."""

import math

import numpy as np
import pytest
import torch

import optix
from optix.denoise import ACsN, acsn, acsn_halo
from optix.denoise import _acsn as A
from optix.denoise import utils as U
from optix.estimates import SensorEstimate
from optix.io import Layout
from optix.models import phantoms, simulate_sensor
from optix.models.blur import gaussian_psf

F64 = dict(dtype=torch.float64)
# the simulation parameters of the paper's Methods
NA, LAM, PX = 1.4, 0.7, 0.1
FC = 2 * NA / LAM


def otf(shape):
    """Incoherent OTF of a circular pupil, cutoff 2·NA/λ."""
    fy = torch.fft.fftfreq(shape[0], d=PX, **F64)
    fx = torch.fft.fftfreq(shape[1], d=PX, **F64)
    r = (torch.sqrt(fy[:, None] ** 2 + fx**2) / FC).clamp(max=1)
    return (2 / math.pi) * (torch.acos(r) - r * torch.sqrt(1 - r * r))


def scene(shape=(96, 96), peak=60.0, bg=10.0, seed=0):
    """Filaments and beads imaged through the OTF, in electrons."""
    x = phantoms.filaments(shape, 8, width=1.0, seed=seed, **F64)
    x = x / x.max()
    b = phantoms.beads(shape, 12, sigma=0.5, brightness=1.0, seed=seed, **F64)
    x = x + b[0] / b[0].max()
    x = torch.fft.ifft2(torch.fft.fft2(x) * otf(shape)).real.clamp(min=0)
    return bg + peak * x / x.max()


def camera(shape, seed=1):
    """Per-pixel offset, read variance and gain maps (ADU, ADU/e⁻)."""
    g = torch.Generator().manual_seed(seed)

    def randn():
        return torch.randn(shape, generator=g, **F64)

    return SensorEstimate(
        offset=100 + 5 * randn(),
        variance=(3 * torch.exp(0.5 * randn())) ** 2,
        gain=2.0 * (1 + 0.1 * randn()),
    )


def simulate(shape=(96, 96), seed=3, frames=None):
    clean = scene(shape)
    sensor = camera(shape)
    if frames is not None:
        clean = clean.expand(frames, *shape)
    raw = simulate_sensor(clean, sensor, seed=seed)
    return raw, clean, sensor


def electron_variance(clean, sensor):
    """Variance of M = (Z − β)/γ: shot noise plus read noise / γ²."""
    return clean + sensor.variance / sensor.gain**2


def psnr(x, ref):
    return 10 * math.log10(
        float(ref.max()) ** 2 / float((x - ref).square().mean())
    )


# -- noise estimate -----------------------------------------------------------
def test_otf_cutoff_and_ratio():
    assert U.otf_cutoff(1.4, 0.7) == pytest.approx(4.0)
    # Eq. (11), f̄ = 1.1/(2Δ) = 5.5 (the ACsN code), f_c = 4 → √(5.5/1.5)
    assert U.noise_ratio((64, 64), PX, FC) == pytest.approx(
        math.sqrt(5.5 / 1.5)
    )
    # f̄ = 1/(2Δ) = 5 → √(5/1)
    assert U.noise_ratio((64, 64), PX, FC, camera_cutoff=1.0) == pytest.approx(
        math.sqrt(5)
    )
    # 2-D: share of the DFT frequencies outside the disc
    f = torch.sqrt(
        torch.fft.fftfreq(64, d=PX)[:, None] ** 2
        + torch.fft.fftfreq(64, d=PX) ** 2
    )
    n_out = int((f > FC).sum())
    assert U.noise_ratio((64, 64), PX, FC, ratio="area") == pytest.approx(
        math.sqrt(64 * 64 / n_out)
    )
    with pytest.raises(ValueError, match="0.1375"):  # undersampled
        U.noise_ratio((64, 64), 0.2, FC)  # largest: 0.7/(4·1.4)·1.1
    # near Nyquist: finite with 1.1, an error with 1.0
    assert math.isfinite(U.noise_ratio((64, 64), 0.13, FC))
    with pytest.raises(ValueError, match="0.125"):
        U.noise_ratio((64, 64), 0.13, FC, camera_cutoff=1.0)
    # the 2-D grid follows the dtype of `like`
    like = torch.empty(0, **F64)
    assert U.noise_ratio(
        (64, 64), PX, FC, ratio="area", like=like
    ) == pytest.approx(math.sqrt(64 * 64 / n_out))
    with pytest.raises(ValueError):
        U.noise_ratio((64, 64), PX, FC, ratio="2d")


def test_highpass_removes_the_signal():
    clean = scene((64, 64))
    h = U.highpass(clean, PX, FC)
    assert h.dtype == clean.dtype
    assert float(h.abs().max()) < 1e-9 * float(clean.max())
    # on white noise it keeps the frequencies outside the OTF (Parseval)
    g = torch.Generator().manual_seed(0)
    n = torch.randn(64, 64, generator=g, **F64)
    hn = U.highpass(n, PX, FC)
    mask = (
        torch.sqrt(
            torch.fft.fftfreq(64, d=PX, **F64)[:, None] ** 2
            + torch.fft.fftfreq(64, d=PX, **F64) ** 2
        )
        > FC
    )
    spec = torch.fft.fft2(n).abs().square()
    expected = float((spec * mask).sum()) / 64**2
    assert float(hn.square().sum()) == pytest.approx(expected, rel=1e-10)


def test_gaussian_fit_is_exact_on_a_gaussian():
    x = torch.linspace(0, 4, 10, **F64)
    y = 900 * torch.exp(-0.5 * (x / 1.3) ** 2)
    assert A._gauss_fit(x, y, 800.0, 0.6) == pytest.approx(1.3)


def test_gaussian_fit_flat_histogram_does_not_raise():
    """Nearly flat counts drive σ up and JᵀJ towards singularity."""
    g = torch.Generator().manual_seed(3)
    h = torch.rand(64, 64, generator=g, dtype=torch.float64) - 0.5
    s = U.sigma_from_highpass(h)
    assert math.isfinite(s) and s >= 0
    for seed in range(40):
        g = torch.Generator().manual_seed(seed)
        h = torch.rand(64, 64, generator=g, dtype=torch.float64) - 0.5
        assert math.isfinite(U.sigma_from_highpass(h))


def test_constant_highpass_gives_zero():
    assert U.sigma_from_highpass(torch.zeros(32, 32, **F64)) == 0.0
    assert U.sigma_from_highpass(torch.full((32, 32), 3.0, **F64)) == 0.0


@pytest.mark.parametrize("method", ["histogram", "std"])
def test_sigma_from_highpass(method):
    g = torch.Generator().manual_seed(1)
    h = 2.5 * torch.randn(256, 256, generator=g, **F64)
    tol = 0.01 if method == "std" else 0.05
    assert U.sigma_from_highpass(h, method) == pytest.approx(2.5, rel=tol)
    with pytest.raises(ValueError):
        U.sigma_from_highpass(h, "mad")


def test_noise_estimate_white_noise():
    """On white noise the 2-D ratio is unbiased; Eq. (11) is a fixed
    multiple of it."""
    g = torch.Generator().manual_seed(2)
    m = scene((128, 128)) + 3.0 * torch.randn(128, 128, generator=g, **F64)
    area = U.acsn_noise(m, PX, FC, ratio="area", sigma_fit="std")
    assert area == pytest.approx(3.0, rel=0.02)
    area_ratio = U.noise_ratio(m.shape, PX, FC, ratio="area")
    for cc, expected in ((1.1, 1.35), (1.0, 1.577)):
        one = U.acsn_noise(
            m, PX, FC, ratio="1d", sigma_fit="std", camera_cutoff=cc
        )
        factor = U.noise_ratio(m.shape, PX, FC, camera_cutoff=cc) / area_ratio
        assert one == pytest.approx(area * factor, rel=1e-12)
        assert factor == pytest.approx(expected, abs=0.01)


def test_noise_estimate_scmos():
    """Per-pixel offset, variance and gain: σ_N² is the mean variance of
    M = (Z − β)/γ (shot + read noise)."""
    raw, clean, sensor = simulate((128, 128))
    m = (raw - sensor.offset) / sensor.gain
    truth = float(electron_variance(clean, sensor).mean().sqrt())
    std = U.acsn_noise(m, PX, FC, ratio="area", sigma_fit="std")
    hist = U.acsn_noise(m, PX, FC, ratio="area")
    assert std == pytest.approx(truth, rel=0.03)
    assert hist == pytest.approx(truth, rel=0.06)


def test_hotspot_removal():
    raw, clean, sensor = simulate((64, 64))
    m = (raw - sensor.offset) / sensor.gain
    hot = m.clone()
    hot[10, 20] += 5000
    hot[40, 33] += 3000
    fixed = A._remove_hotspots(hot, PX, FC)
    assert float(fixed[10, 20]) < float(m.max())
    assert float(fixed[40, 33]) < float(m.max())
    # other pixels flagged, if any, take their 3 × 3 median
    changed = fixed != hot
    med = A._median3(hot)
    torch.testing.assert_close(fixed[changed], med[changed])
    assert int(changed.sum()) < 0.02 * m.numel()
    est = ACsN(na=NA, wavelength=LAM, pixel_size=PX, hotspot=True, **F64)
    assert not est.is_faithful


@pytest.mark.parametrize(
    "kw",
    [
        dict(hotspot=True),
        dict(sigma_fit="std"),
        dict(hotspot=True, sigma_fit="std"),
    ],
)
def test_extensions_through_acsn_and_apply(kw):
    raw, clean, sensor = simulate((64, 64))
    raw = raw.clone()
    raw[10, 20] += 4000  # a hot pixel
    m = (raw - sensor.offset) / sensor.gain
    out = acsn(raw, NA, LAM, PX, offset=sensor.offset, gain=sensor.gain, **kw)
    est = ACsN(na=NA, wavelength=LAM, pixel_size=PX, **kw, **F64)
    torch.testing.assert_close(
        est.fit(init=sensor).apply(raw), out, rtol=0, atol=0
    )
    assert not est.is_faithful
    src = A._remove_hotspots(m, PX, FC) if kw.get("hotspot") else m
    assert float(est.sigma_) == pytest.approx(
        U.acsn_noise(src, PX, FC, sigma_fit=kw.get("sigma_fit", "histogram"))
    )
    if kw.get("hotspot"):
        assert float(out[10, 20]) < float(clean.max()) + 20
        truth = float(electron_variance(clean, sensor).mean().sqrt())
        assert float(est.sigma_) < 2 * truth


def test_hot_pixel_breaks_the_faithful_estimate():
    """The warning of the docstrings: one hot pixel inflates σ_N."""
    raw, clean, sensor = simulate((128, 128))
    m = (raw - sensor.offset) / sensor.gain
    base = U.acsn_noise(m, PX, FC, ratio="area")
    hot = m.clone()
    hot[50, 60] += 1000
    assert U.acsn_noise(hot, PX, FC, ratio="area") > 3 * base
    fixed = A._remove_hotspots(hot, PX, FC)
    assert U.acsn_noise(fixed, PX, FC, ratio="area") < 1.1 * base


# -- sparse filter ------------------------------------------------------------
def test_profile():
    ht, wie = U.acsn_profile()
    assert (ht.size, ht.group, ht.step, ht.search) == (8, 8, 3, 19)
    assert (ht.tau, ht.lam, ht.beta) == (2500.0, 4.0, None)
    assert ht.pixel_distance and not ht.sigma_weight and not ht.keep_dc
    assert (wie.size, wie.group, wie.step, wie.search) == (8, 8, 3, 19)
    assert wie.lam is None and wie.beta is None and wie.strict
    assert U.stage_halo(ht) + U.stage_halo(wie) == 50
    assert acsn_halo() == 50
    assert ACsN(na=NA, wavelength=LAM, pixel_size=PX).halo() == 50


def test_stage_rejects_pixel_distance_with_lam2d():
    with pytest.raises(ValueError):
        U.Stage("dct", 8, 8, 3, 9, 2500.0, 4.0, 2.0, pixel_distance=True)


def test_pixel_distance_matches_orthonormal_spectra():
    """Eq. (12) on pixels equals the distance between DCT spectra."""
    g = torch.Generator().manual_seed(0)
    x = 255 * torch.rand(40, 40, generator=g, **F64)
    a = U.Stage("dct", 8, 8, 3, 9, 2500.0, 4.0, pixel_distance=True)
    b = U.Stage("dct", 8, 8, 3, 9, 2500.0, 4.0)
    torch.testing.assert_close(
        U.run_stage(x, 20.0, a), U.run_stage(x, 20.0, b), rtol=0, atol=1e-9
    )


def test_hard_threshold_weight_eq15():
    s = torch.tensor([[[10.0, 0.5, -9.0, 0.1]], [[0.1, 0.2, 0.0, 0.3]]])
    out, w = U.hard_threshold(s, 2.0, 2.0, False, sigma_weight=False)
    torch.testing.assert_close(out[0, 0], torch.tensor([10.0, 0, -9.0, 0]))
    torch.testing.assert_close(w, torch.tensor([0.5, 1.0]))
    _, w = U.hard_threshold(s, 2.0, 2.0, False)
    torch.testing.assert_close(w, torch.tensor([0.125, 1.0]))


def test_fixed_pattern_correction():
    """Eq. (4): offset and gain maps are removed before filtering."""
    m = scene((48, 48))
    sensor = camera((48, 48))
    z = sensor.offset + sensor.gain * m
    a = acsn(z, NA, LAM, PX, offset=sensor.offset, gain=sensor.gain, sigma=2)
    b = acsn(m, NA, LAM, PX, sigma=2)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-9)
    # a frame without noise changes by less than the noise level σ = 2
    assert float((b - m).square().mean().sqrt()) < 1.0


@pytest.mark.parametrize("ratio", ["1d", "area"])
def test_psnr_gain(ratio):
    raw, clean, sensor = simulate()
    m = (raw - sensor.offset) / sensor.gain
    out = acsn(
        raw, NA, LAM, PX, offset=sensor.offset, gain=sensor.gain, ratio=ratio
    )
    before, after = psnr(m, clean), psnr(out, clean)
    assert after > before + 8, (before, after)


def test_quality_score():
    raw, clean, sensor = simulate()
    m = (raw - sensor.offset) / sensor.gain
    out = acsn(raw, NA, LAM, PX, offset=sensor.offset, gain=sensor.gain)
    k = gaussian_psf((15, 15), 4.0, **F64)[None, None]
    pad = torch.nn.functional.pad(m[None, None], (7,) * 4, mode="reflect")
    blurred = torch.nn.functional.conv2d(pad, k)[0, 0]
    q = U.quality_score(m, out)
    assert -1 <= q <= 1
    assert q > U.quality_score(m, blurred) + 0.3
    # invariant to an affine change of the intensities
    assert U.quality_score(3 * m + 7, 3 * out + 7) == pytest.approx(q)
    assert U.quality_score(m, torch.full_like(m, 5.0)) == 0.0


def test_temporal_psnr():
    g = torch.Generator().manual_seed(0)
    x = 10 + torch.randn(8, 16, 16, generator=g, **F64)
    p = U.temporal_psnr(x)
    assert p.shape == (3,)
    ref = x[1:7].mean(0)
    expected = 10 * math.log10(
        float(x[1].max()) ** 2 / float((x[1] - ref).square().mean())
    )
    assert float(p[1]) == pytest.approx(expected)
    with pytest.raises(ValueError):
        U.temporal_psnr(x[:5])


# -- estimator and streaming --------------------------------------------------
def test_estimator_options():
    est = ACsN(na=NA, wavelength=LAM, pixel_size=PX)
    assert est.is_faithful and est.options["ratio"] == "1d"
    assert est.options["camera_cutoff"] == 1.1 and not est.options["hotspot"]
    assert not ACsN(
        na=NA, wavelength=LAM, pixel_size=PX, camera_cutoff=1.0
    ).is_faithful
    rec = ACsN(na=NA, wavelength=LAM, pixel_size=PX, faithful=False)
    assert rec.options["ratio"] == "area" and not rec.is_faithful
    assert rec.options["hotspot"] and rec.options["camera_cutoff"] == 1.1
    assert not ACsN(na=NA, wavelength=LAM, pixel_size=PX, sigma=2).is_faithful
    assert ACsN(
        na=NA, wavelength=LAM, pixel_size=PX, block=(32, 32)
    ).is_faithful
    with pytest.raises(ValueError):
        ACsN(wavelength=LAM, pixel_size=PX)
    with pytest.raises(ValueError):
        ACsN(na=NA, wavelength=LAM, pixel_size=PX, sigma_fit="mad")
    with pytest.raises(RuntimeError):
        est.apply(torch.zeros(16, 16))
    with pytest.raises(ValueError):  # undersampled
        ACsN(na=NA, wavelength=LAM, pixel_size=0.2).fit().apply(
            torch.rand(32, 32)
        )


def test_frames_are_independent_and_streaming_matches_whole():
    raw, clean, sensor = simulate((70, 64), frames=2)
    whole = torch.stack(
        [
            acsn(z, NA, LAM, PX, offset=sensor.offset, gain=sensor.gain)
            for z in raw
        ]
    )
    est = ACsN(na=NA, wavelength=LAM, pixel_size=PX, block=(40, 32), **F64)
    est.fit(init=sensor)
    out = est.apply(raw.numpy())  # numpy in, numpy out, 2 × 2 blocks
    np.testing.assert_allclose(np.asarray(out), whole.numpy(), atol=1e-9)
    stacked = acsn(raw, NA, LAM, PX, offset=sensor.offset, gain=sensor.gain)
    torch.testing.assert_close(stacked, whole, rtol=0, atol=0)
    # σ_N per frame
    m = (raw - sensor.offset) / sensor.gain
    assert est.sigma_.shape == est.quality_.shape == (2,)
    assert float(est.quality_[0]) == pytest.approx(
        U.quality_score(m[0], whole[0])
    )
    for i in range(2):
        assert float(est.sigma_[i]) == pytest.approx(
            U.acsn_noise(m[i], PX, FC)
        )


def test_lazy_zarr_and_group_maps():
    zarr = pytest.importorskip("zarr")
    raw, _, sensor = simulate((48, 52), frames=2)
    other = camera((48, 52), seed=5)
    raw = torch.stack([raw[0], raw[1] - sensor.offset + other.offset])
    maps = SensorEstimate(
        offset=torch.stack([sensor.offset, other.offset]),
        variance=sensor.variance,
        gain=torch.stack([sensor.gain, sensor.gain]),
    )
    src = zarr.zeros(raw.shape, chunks=(1, 24, 26), dtype="f8")
    src[:] = raw.numpy()
    dst = zarr.zeros(raw.shape, chunks=(1, 24, 26), dtype="f8")
    lay = Layout("gyx", spatial="yx", group="g")
    est = ACsN(na=NA, wavelength=LAM, pixel_size=PX, layout=lay, **F64)
    est.fit(init=maps).apply(src, out=dst)
    for i, off in enumerate((sensor.offset, other.offset)):
        expected = acsn(raw[i], NA, LAM, PX, offset=off, gain=sensor.gain)
        np.testing.assert_allclose(dst[i], expected.numpy(), atol=1e-9)
    assert est.sigma_.shape == (2,)


def test_faithful_false_differs_only_through_sigma():
    """With σ_N given, the faithful run and the recommended one (whose
    hot-pixel removal is also applied here) are the same filter."""
    raw, _, sensor = simulate((64, 64))
    kw = dict(offset=sensor.offset, gain=sensor.gain)
    fai = ACsN(na=NA, wavelength=LAM, pixel_size=PX, **F64).fit(init=sensor)
    rec = ACsN(na=NA, wavelength=LAM, pixel_size=PX, faithful=False, **F64)
    rec.fit(init=sensor)
    a, b = fai.apply(raw), rec.apply(raw)
    assert float(rec.sigma_) < float(fai.sigma_)
    s_a, s_b = float(fai.sigma_), float(rec.sigma_)
    ref_a = acsn(raw, NA, LAM, PX, sigma=s_a, **kw)
    ref_b = acsn(raw, NA, LAM, PX, sigma=s_b, hotspot=True, **kw)
    torch.testing.assert_close(a, ref_a, rtol=0, atol=0)
    torch.testing.assert_close(b, ref_b, rtol=0, atol=0)


def test_quality_is_nan_below_11():
    g = torch.Generator().manual_seed(0)
    x = 20 + 3 * torch.randn(10, 12, generator=g, **F64)
    est = ACsN(na=NA, wavelength=LAM, pixel_size=PX, **F64).fit()
    est.apply(x)
    assert math.isnan(float(est.quality_))
    assert math.isfinite(float(est.sigma_))
    with pytest.raises(ValueError):
        U.quality_score(x, x)


def test_quality_score_float32():
    raw, clean, sensor = simulate()
    m = (raw - sensor.offset) / sensor.gain
    out = acsn(raw, NA, LAM, PX, offset=sensor.offset, gain=sensor.gain)
    q64 = U.quality_score(m, out)
    q32 = U.quality_score(m.float(), out.float())
    assert q32 == pytest.approx(q64, abs=1e-4)


def test_partial_blocks_match_whole_frame():
    """Windows (core + 2 × halo) smaller than the frame on both axes."""
    raw, _, sensor = simulate((120, 150))
    kw = dict(offset=sensor.offset, gain=sensor.gain)
    whole = acsn(raw, NA, LAM, PX, **kw)
    parts = acsn(raw, NA, LAM, PX, block=(16, 20), **kw)
    torch.testing.assert_close(parts, whole, rtol=0, atol=1e-9)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_blocks_match_whole_frame(dtype):
    raw, _, sensor = simulate((80, 90))
    raw = raw.to(dtype)
    kw = dict(offset=sensor.offset.to(dtype), gain=sensor.gain.to(dtype))
    whole = acsn(raw, NA, LAM, PX, **kw)
    parts = acsn(raw, NA, LAM, PX, block=(37, 41), **kw)
    assert parts.dtype == dtype
    atol = 1e-9 if dtype == torch.float64 else 4 * torch.finfo(dtype).eps * 100
    torch.testing.assert_close(parts, whole, rtol=0, atol=atol)


def test_float32_matches_float64():
    raw, clean, sensor = simulate((64, 64))
    kw = dict(offset=sensor.offset, gain=sensor.gain)
    ref = acsn(raw, NA, LAM, PX, **kw)
    out = acsn(
        raw.float(), NA, LAM, PX, **{k: v.float() for k, v in kw.items()}
    )
    assert out.dtype == torch.float32
    assert abs(psnr(out.double(), clean) - psnr(ref, clean)) < 0.05


def test_save_load(tmp_path):
    raw, _, sensor = simulate((32, 32))
    est = ACsN(na=NA, wavelength=LAM, pixel_size=PX, **F64).fit(init=sensor)
    path = tmp_path / "acsn.pt"
    est.save(path)
    back = optix.load(path).to(dtype=torch.float64)
    assert isinstance(back, ACsN) and back.is_faithful
    torch.testing.assert_close(back.apply(raw), est.apply(raw))
    assert est.sigma_.shape == ()
    # a single frame stored (x, y); maps are in layout order (y, x)
    lay = Layout("xy", spatial="yx")
    t = ACsN(na=NA, wavelength=LAM, pixel_size=PX, layout=lay, **F64)
    out = t.fit(init=sensor).apply(raw.T.numpy().copy())
    np.testing.assert_allclose(out, est.apply(raw).T.numpy(), atol=1e-12)
