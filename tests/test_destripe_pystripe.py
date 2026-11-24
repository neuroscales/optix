"""pystripe extensions of the wavelet–FFT filter (`WaveletFFT`)."""

import json
import os

import numpy as np
import pytest
import torch

import optix
from optix.destripe import WaveletFFT, wavelet_fft
from optix.estimates import ShadingEstimate
from optix.stats import otsu_threshold

FIXTURES = os.path.join(
    os.path.dirname(__file__), "data", "pystripe_fixtures.npz"
)

# pystripe's filter_streaks, in optix's terms
PYSTRIPE = dict(
    wavelet="db3",
    level="max",
    axis=-1,
    log_transform=True,
    sigma_units="pystripe",
    pad_even=True,
)
CASES = [
    "single",
    "dual",
    "dual_otsu",
    "foreground",
    "background",
    "flat",
    "odd",
    "u16",
]
F64 = dict(dtype=torch.float64)


@pytest.fixture(scope="module")
def ref():
    return np.load(FIXTURES)


def translate(args: dict) -> dict:
    """pystripe's `filter_streaks` arguments as `wavelet_fft` arguments."""
    args = dict(args)
    s1, s2 = args.pop("sigma")
    kw = dict(PYSTRIPE)
    kw["wavelet"] = args.pop("wavelet", "db3")
    kw["level"] = args.pop("level", 0) or "max"
    if s1 == s2:
        kw["sigma"] = s1
    elif s2 == 0:
        kw.update(bands="foreground", sigma=s1)
    elif s1 == 0:
        kw.update(bands="background", sigma_background=s2)
    else:
        kw.update(bands="dual", sigma=s1, sigma_background=s2)
    for name in ("threshold", "crossover", "dark"):
        if name in args:
            kw[name] = args.pop(name)
    assert not args, args
    return kw


def case_input(ref, case):
    args = json.loads(str(ref[f"args_{case}"]))
    image = ref[args.pop("image")]
    kw = translate(args)
    if "dark" in kw:
        kw["flat"] = torch.as_tensor(ref["flat"])
    if image.dtype == np.uint16:
        image = image.astype(np.int32)
    return torch.as_tensor(image), kw


def make_frames(
    n=2, shape=(48, 40), seed=0, dtype=torch.float64, stripes=True
):
    """Bright blobs on a background near 100, horizontal stripes."""
    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(
        torch.arange(shape[0], dtype=torch.float64),
        torch.arange(shape[1], dtype=torch.float64),
        indexing="ij",
    )
    frames = []
    with_stripes = stripes
    for _ in range(n):
        img = 100 + 20 * torch.sin(x / 7 + y / 11)
        for _ in range(3):
            c = torch.rand(2, generator=g, dtype=torch.float64)
            r2 = (y - c[0] * shape[0]).square() + (
                x - c[1] * shape[1]
            ).square()
            img = img + 800 * torch.exp(-r2 / 20)
        stripes = 15 * torch.randn(
            shape[0], 1, generator=g, dtype=torch.float64
        )
        frames.append(img + stripes * float(with_stripes))
    return torch.stack(frames).to(dtype)


# -- against pystripe ---------------------------------------------------------
def test_otsu_matches_skimage(ref):
    image = torch.as_tensor(ref["image"])
    assert float(otsu_threshold(image)) == pytest.approx(
        float(ref["otsu_float"]), rel=1e-14
    )
    u16 = torch.as_tensor(ref["image_u16"].astype(np.int32))
    assert float(otsu_threshold(u16, bins=None)) == float(ref["otsu_u16"])


@pytest.mark.parametrize(
    "case, sigma, level, wavelet",
    [("a", 8.0, "max", "db3"), ("b", 4.0, 3, "db4")],
)
def test_subband_matches_pystripe(ref, case, sigma, level, wavelet):
    """pystripe's `filter_subband`: log domain and bandwidth mapping."""
    kw = {**PYSTRIPE, "sigma": sigma, "level": level, "wavelet": wavelet}
    out = wavelet_fft(torch.as_tensor(ref["image"]), **kw)
    expected = torch.as_tensor(ref[f"subband_{case}"])
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("case", CASES)
def test_filter_streaks_matches_pystripe(ref, case):
    image, kw = case_input(ref, case)
    with optix.config(dtype=torch.float64):  # integer input is converted
        out = wavelet_fft(image, **kw)
    h, w = out.shape
    # before pystripe's final clip (padded for the odd case)
    expected = torch.as_tensor(ref[f"streaks_{case}"])[:h, :w]
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-10)
    # and its uint16 output
    u16 = np.clip(out.numpy(), 0, 2**16 - 1).astype(np.uint16)
    np.testing.assert_array_equal(u16, ref[f"streaks_{case}_u16"])


def test_estimator_matches_pystripe(ref):
    """Flat and dark through `fit(init=ShadingEstimate)`, streamed."""
    image, kw = case_input(ref, "flat")
    flat, dark = kw.pop("flat"), kw.pop("dark")
    est = WaveletFFT(**kw, **F64)
    assert not est.is_faithful
    shading = ShadingEstimate(flat=flat, dark=torch.tensor(dark))
    est.fit(init=shading)
    assert "shading" in est.estimates
    stack = image.expand(3, *image.shape).numpy()
    with optix.config(batch=2):
        out = torch.as_tensor(est.apply(stack))
    expected = torch.as_tensor(ref["streaks_flat"])
    for frame in out:
        torch.testing.assert_close(frame, expected, rtol=1e-12, atol=1e-10)


def test_estimator_integer_input_uses_integer_otsu(ref):
    """A uint16 array is thresholded with one bin per integer value."""
    _, kw = case_input(ref, "u16")
    est = WaveletFFT(**kw, **F64)
    out = torch.as_tensor(est.apply(ref["image_u16"][None]))
    expected = torch.as_tensor(ref["streaks_u16"])
    torch.testing.assert_close(out[0], expected, rtol=1e-12, atol=1e-10)


# -- unit tests ---------------------------------------------------------------
def test_defaults_unchanged():
    """The extensions are off by default: the paper's filter."""
    data = make_frames()
    off = dict(
        bands="single",
        sigma_background=None,
        threshold=None,
        crossover=10.0,
        smoothing=1.0,
        log_transform=False,
        sigma_units="band",
        pad_even=False,
        flat=None,
        dark=None,
    )
    torch.testing.assert_close(wavelet_fft(data), wavelet_fft(data, **off))
    est = WaveletFFT(**F64)
    assert est.is_faithful
    assert {k: est.options[k] for k in off if k in est.options} == {
        k: v for k, v in off.items() if k not in ("flat", "dark")
    }
    # options of the dual-band modes are inert with a single band
    assert WaveletFFT(threshold=200.0, crossover=3.0).is_faithful
    for kw in (
        dict(bands="foreground"),
        dict(log_transform=True),
        dict(sigma_units="pystripe"),
        dict(pad_even=True),
    ):
        assert not WaveletFFT(**kw).is_faithful
    est = WaveletFFT().fit(init=ShadingEstimate(dark=torch.tensor(1.0)))
    assert not est.is_faithful


def test_dual_with_equal_bandwidths_is_single():
    data = make_frames()
    kw = dict(wavelet="db4", level=3, axis=-1)
    torch.testing.assert_close(
        wavelet_fft(data, bands="dual", sigma=2.0, sigma_background=2.0, **kw),
        wavelet_fft(data, sigma=2.0, **kw),
    )


def test_band_limits():
    """A far threshold leaves one band: the blend is that band's filter."""
    data = make_frames()
    kw = dict(wavelet="db4", level=3, axis=-1, smoothing=0.0, crossover=1e-3)
    single = wavelet_fft(data, sigma=2.0, wavelet="db4", level=3, axis=-1)
    # all foreground: the foreground filter, unclipped
    fg = wavelet_fft(data, bands="foreground", sigma=2.0, threshold=-1e9, **kw)
    torch.testing.assert_close(fg, single)
    # all background: the background filter
    bg = wavelet_fft(
        data, bands="background", sigma_background=2.0, threshold=1e9, **kw
    )
    torch.testing.assert_close(bg, single)
    # the foreground mode leaves the background untouched, and vice versa
    torch.testing.assert_close(
        wavelet_fft(data, bands="foreground", sigma=2.0, threshold=1e9, **kw),
        data,
    )


def test_dual_band_reduces_stripes():
    """Bright blobs ring under a single band; the dual band spares them."""
    shape = (128, 96)
    data = make_frames(shape=shape)
    clean = make_frames(shape=shape, stripes=False)

    def error(x):
        return (x - clean).square().mean().sqrt()

    single = wavelet_fft(data, **PYSTRIPE, sigma=8.0)
    dual = wavelet_fft(
        data, **PYSTRIPE, bands="dual", sigma=1.0, sigma_background=8.0
    )
    assert error(dual) < 0.5 * error(data)
    assert error(dual) < 0.75 * error(single)


def test_horizontal_is_transposed_vertical():
    data = make_frames()
    kw = dict(PYSTRIPE, bands="dual", sigma=4.0, sigma_background=12.0)
    out = wavelet_fft(data, **kw)
    kw["axis"] = -2
    out_t = wavelet_fft(data.mT.contiguous(), **kw).mT
    torch.testing.assert_close(out_t, out)


def test_float32_matches_float64():
    data = make_frames()
    kw = dict(PYSTRIPE, bands="dual", sigma=4.0, sigma_background=12.0)
    kw["threshold"] = 300.0
    expected = wavelet_fft(data, **kw)
    out = wavelet_fft(data.float(), **kw)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.double(), expected, rtol=1e-4, atol=1e-3)


def test_flat_dark_correction():
    data = make_frames()
    flat = torch.linspace(0.5, 1.0, 40, dtype=torch.float64).expand(48, 40)
    kw = dict(wavelet="db4", level=3)
    out = wavelet_fft(data, flat=flat, dark=7.0, **kw)
    torch.testing.assert_close(out, (wavelet_fft(data, **kw) - 7.0) / flat)


def test_pad_even_crops_back():
    data = make_frames(shape=(45, 39))
    out = wavelet_fft(data, wavelet="db3", level=2, pad_even=True)
    assert out.shape == data.shape
    # equal to filtering the padded frames and cropping
    padded = torch.cat([data, data[..., -1:, :]], -2)
    padded = torch.cat([padded, padded[..., -1:]], -1)
    torch.testing.assert_close(
        out, wavelet_fft(padded, wavelet="db3", level=2)[..., :45, :39]
    )


def test_level_max():
    data = make_frames()
    # min(48, 40) = 40 and db3 (6 taps): ⌊log₂(40/5)⌋ = 3
    torch.testing.assert_close(
        wavelet_fft(data, wavelet="db3", level="max"),
        wavelet_fft(data, wavelet="db3", level=3),
    )
    with pytest.raises(ValueError, match="too small"):
        wavelet_fft(data[..., :4, :4], wavelet="db10", level="max")


def test_otsu_threshold():
    data = make_frames(3)
    t = otsu_threshold(data)
    assert t.shape == (3,)
    for i in range(3):
        assert t[i] == otsu_threshold(data[i])
        assert data[i].min() < t[i] < data[i].max()
    # two-level image: the threshold separates the levels
    two = torch.zeros(10, 10, dtype=torch.float64)
    two[:, 5:] = 1.0
    t = otsu_threshold(two)
    assert 0 <= t < 1
    assert otsu_threshold(torch.full((4, 4), 3.0)) == 3.0
    with pytest.raises(ValueError, match="integer-valued"):
        otsu_threshold(two + 0.5, bins=None)
    with pytest.raises(ValueError, match="bins"):
        otsu_threshold(two, bins=0)


def test_invalid_options():
    data = make_frames()
    with pytest.raises(ValueError, match="bands"):
        wavelet_fft(data, bands="both")
    with pytest.raises(ValueError, match="needs sigma_background"):
        wavelet_fft(data, bands="dual")
    with pytest.raises(ValueError, match="needs sigma_background"):
        WaveletFFT(bands="background")
    with pytest.raises(ValueError, match="crossover"):
        wavelet_fft(data, bands="foreground", crossover=0.0)
    with pytest.raises(ValueError, match="smoothing"):
        wavelet_fft(data, smoothing=-1.0)
    with pytest.raises(ValueError, match="sigma_units"):
        wavelet_fft(data, sigma_units="pixels")
    with pytest.raises(ValueError, match="level"):
        wavelet_fft(data, level="min")


def test_save_load_keeps_shading(tmp_path):
    est = WaveletFFT(**PYSTRIPE, bands="dual", sigma=4.0, sigma_background=8.0)
    flat = torch.linspace(0.5, 1.0, 40, dtype=torch.float64).expand(48, 40)
    est.fit(init=ShadingEstimate(flat=flat.contiguous()))
    est.save(tmp_path / "w.pt")
    back = optix.load(tmp_path / "w.pt")
    assert back.options == est.options and not back.is_faithful
    data = make_frames()
    torch.testing.assert_close(back.apply(data), est.apply(data))
