"""Wavelet–FFT stripe removal (Münch et al. 2009)."""

import math
import os

import numpy as np
import pytest
import torch

import optix
from optix.destripe import WaveletFFT, utils, wavelet_fft
from optix.io import Layout, labeled
from optix.models import random_shading

FIXTURES = os.path.join(
    os.path.dirname(__file__), "data", "destripe_fixtures.npz"
)


@pytest.fixture(scope="module")
def ref():
    return np.load(FIXTURES)


def make_frames(n=3, shape=(64, 80), *, seed=0, dtype=torch.float64):
    """Smooth shading plus sharp blobs: content without stripes."""
    g = torch.Generator().manual_seed(seed)
    frames = []
    for i in range(n):
        flat = random_shading(shape, seed=seed + i, dtype=dtype).flat
        y, x = torch.meshgrid(
            torch.arange(shape[0], dtype=dtype),
            torch.arange(shape[1], dtype=dtype),
            indexing="ij",
        )
        c = torch.rand(2, generator=g, dtype=torch.float64).to(dtype)
        blob = (
            (y - c[0] * shape[0]).square() + (x - c[1] * shape[1]).square()
        ) < 60
        frames.append(10 * flat + 2 * blob.to(dtype))
    return torch.stack(frames)


def make_stripes(n, shape, *, seed=1, dtype=torch.float64):
    """Ideal vertical stripes (Eq. (8)): offsets constant along y."""
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(n, *shape, dtype=dtype)
    for i in range(n):
        for _ in range(6):
            x0 = int(torch.randint(0, shape[1] - 8, (1,), generator=g))
            w = int(torch.randint(1, 6, (1,), generator=g))
            a = float(torch.randn(1, generator=g, dtype=torch.float64))
            out[i, :, x0 : x0 + w] += a
    return out


# -- filter banks -------------------------------------------------------------
def test_daubechies_matches_pywavelets(ref):
    for n in range(1, 39):
        w = utils.get_wavelet(f"db{n}")
        np.testing.assert_allclose(
            w.rec_lo, ref[f"db{n}_rec_lo"], rtol=0, atol=1e-15
        )
    assert utils.get_wavelet("haar").rec_lo == utils.get_wavelet("db1").rec_lo


@pytest.mark.parametrize("n", [1, 2, 5, 15, 30, 42, 45])
def test_daubechies_orthonormal(n):
    h = np.array(utils.get_wavelet(f"db{n}").rec_lo)
    assert len(h) == 2 * n
    assert h.sum() == pytest.approx(math.sqrt(2), abs=1e-14)
    for m in range(n):
        dot = h[2 * m :] @ h[: len(h) - 2 * m]
        assert dot == pytest.approx(float(m == 0), abs=1e-13)
    # N vanishing moments of the high-pass filter (checked where the
    # moments are well conditioned)
    g = np.array(utils.get_wavelet(f"db{n}").rec_hi)
    k = np.arange(len(g), dtype=np.float64)
    for p in range(min(n, 6)):
        moment = (k**p * g).sum() / (k**p * np.abs(g)).sum()
        assert abs(moment) < 1e-10


def test_unknown_wavelet():
    for name in ("db0", "db46", "sym4", "bior1.3"):
        with pytest.raises(ValueError, match="Unknown wavelet"):
            utils.get_wavelet(name)


# -- transform ----------------------------------------------------------------
@pytest.mark.parametrize("name", ["db1", "db2", "db7"])
def test_wavedec2_matches_pywavelets(ref, name):
    x = torch.tensor(ref["image"])
    coeffs = utils.wavedec2(x, name, 3)
    flat = [coeffs[0]] + [c for det in coeffs[1:] for c in det]
    for i, c in enumerate(flat):
        np.testing.assert_allclose(
            c.numpy(), ref[f"dwt_{name}_{i}"], rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("name", ["haar", "db2", "db4", "db15", "db45"])
@pytest.mark.parametrize("shape", [(32, 32), (37, 50), (9, 7)])
def test_perfect_reconstruction(name, shape, device, dtype):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, *shape, generator=g, dtype=torch.float64)
    x = x.to(device=device, dtype=dtype)
    tol = 1e-12 if dtype == torch.float64 else 2e-5
    for level in (1, 2, 4):
        coeffs = utils.wavedec2(x, name, level)
        back = utils.waverec2(coeffs, name, shape=shape)
        assert back.shape == x.shape and back.dtype == dtype
        assert (back - x).abs().max() <= tol * x.abs().max()
    # 1-D, along either dim
    for dim in (-1, -2):
        a, d = utils.dwt(x, name, dim)
        back = utils.idwt(a, d, name, dim).narrow(dim, 0, x.shape[dim])
        assert (back - x).abs().max() <= tol * x.abs().max()


def test_transform_is_orthogonal():
    # orthogonal filters: energy is kept up to the boundary coefficients;
    # with periodic-like data on an even grid and haar, exactly.
    x = torch.randn(16, 16, dtype=torch.float64)
    a, det = utils.dwt2(x, "haar")
    energy = a.square().sum() + sum(d.square().sum() for d in det)
    assert float(energy) == pytest.approx(float(x.square().sum()), rel=1e-12)


def test_max_level():
    assert utils.max_level(64, "db1") == 6
    assert utils.max_level(64, "db4") == 3  # log2(64 / 7)
    assert utils.max_level(4, "db10") == 0


# -- the filter ---------------------------------------------------------------
def test_damping():
    g = utils.stripe_damping(9, 1.5, dtype=torch.float64)
    k = torch.arange(5, dtype=torch.float64)
    torch.testing.assert_close(g, 1 - torch.exp(-(k**2) / 4.5))
    g0 = utils.stripe_damping(8, 0.0)
    assert g0[0] == 0 and bool((g0[1:] == 1).all())


@pytest.mark.parametrize("case", ["a", "b", "c"])
def test_matches_paper_code(ref, case):
    """Fig. 5 of the paper, transcribed with PyWavelets (fixture)."""
    x = torch.tensor(ref["image"])
    wavelet, level, sigma, axis = {
        "a": ("db4", 3, 1.5, -2),
        "b": ("db10", 2, 4.0, -2),
        "c": ("db4", 3, 1.5, -1),
    }[case]
    out = wavelet_fft(x, wavelet=wavelet, level=level, sigma=sigma, axis=axis)
    np.testing.assert_allclose(
        out.numpy(), ref[f"fig5_{case}"], rtol=0, atol=1e-12
    )


def test_matches_algotom(ref):
    """algotom's `remove_stripe_based_wavelet_fft` (fixture).

    Parameter mapping: `level` → level, `size` → sigma, `wavelet_name` →
    wavelet. algotom pads the image (mean along rows, edge along columns)
    before filtering and crops after: the padding is reproduced here.
    """
    image = ref["algotom_image"]
    pad = int(ref["algotom_pad"])
    padded = np.pad(image, ((pad, pad), (0, 0)), mode="mean")
    padded = np.pad(padded, ((0, 0), (pad, pad)), mode="edge")
    out = wavelet_fft(torch.tensor(padded), wavelet="db3", level=2, sigma=1.5)[
        pad:-pad, pad:-pad
    ]
    np.testing.assert_allclose(out.numpy(), ref["algotom"], rtol=0, atol=1e-12)


def _relative_energy(truth, estimate):
    """Eq. (11): ∑|s_o − s_f|² / ∑|s_o|²."""
    return float((truth - estimate).square().sum() / truth.square().sum())


def _tv(profile):
    return float(profile.diff(dim=-1).abs().sum())


def test_removes_ideal_stripes():
    content = make_frames(3)
    stripes = make_stripes(3, content.shape[-2:])
    data = content + stripes
    out = wavelet_fft(data, wavelet="db10", level=4, sigma=0.0)
    residual = out - content
    centred = stripes - stripes.mean((-2, -1), keepdim=True)
    # most of the stripe energy is gone; what is left is the part of the
    # stripes wider than the coarsest filtered scale (in the low-pass band)
    assert residual.square().sum() < 0.3 * centred.square().sum()
    # the sharp stripe edges are gone
    assert _tv(residual.mean(-2)) < 0.3 * _tv(stripes.mean(-2))
    # closer to the content than the data were (Eq. (11))
    assert _relative_energy(content, out) < 0.3 * _relative_energy(
        content, data
    )
    # the residual stripe energy decreases with the level (Section 3.6)
    left = [
        float(wavelet_fft(stripes, wavelet="db10", level=L, sigma=0.0).var())
        for L in (1, 2, 3, 4, 5)
    ]
    assert all(b < a for a, b in zip(left, left[1:], strict=False))


def test_ideal_filter_changes_only_stripes():
    """σ → 0 removes ŷ = 0 only: the change is constant along y (Fig. 7)."""
    data = make_frames(2) + make_stripes(2, (64, 80))
    for axis in (-2, -1):
        out = wavelet_fft(data, wavelet="db7", level=4, sigma=0.0, axis=axis)
        change = out - data
        spread = change - change.mean(axis, keepdim=True)
        assert spread.abs().max() < 1e-10 * change.abs().max()


def test_keeps_content_without_stripes():
    content = make_frames(2)
    out = wavelet_fft(content, wavelet="db10", level=4, sigma=1.5)
    assert _relative_energy(content, out) < 1e-3


def test_float32_matches_float64():
    content = make_frames(2)
    data = content + make_stripes(2, content.shape[-2:])
    kw = dict(wavelet="db12", level=4, sigma=1.5)
    hi = wavelet_fft(data, **kw)
    lo = wavelet_fft(data.float(), **kw)
    assert lo.dtype == torch.float32
    assert (lo.double() - hi).abs().max() < 1e-5 * data.abs().max()


def test_integer_input():
    data = (100 * make_frames(1)).round().to(torch.int32)
    out = wavelet_fft(data, wavelet="db2", level=2, sigma=1.0)
    assert out.is_floating_point()


def test_levels_and_per_level_sigma():
    data = make_frames(1) + make_stripes(1, (64, 80))
    kw = dict(wavelet="db4", level=3)
    same = wavelet_fft(data, sigma=[2.0, 2.0, 2.0], **kw)
    torch.testing.assert_close(same, wavelet_fft(data, sigma=2.0, **kw))
    # no level filtered: the identity (perfect reconstruction)
    ident = wavelet_fft(data, sigma=2.0, levels=[], **kw)
    torch.testing.assert_close(ident, data, rtol=0, atol=1e-12)
    # filtering only level 1 differs from filtering all
    one = wavelet_fft(data, sigma=2.0, levels=[1], **kw)
    assert not torch.allclose(one, same)
    with pytest.raises(ValueError, match="levels"):
        wavelet_fft(data, sigma=2.0, levels=[4], **kw)
    with pytest.raises(ValueError, match="sigma has"):
        wavelet_fft(data, sigma=[1.0, 2.0], **kw)
    with pytest.raises(ValueError, match="sigma must"):
        wavelet_fft(data, sigma=-1.0, **kw)
    with pytest.raises(ValueError, match="level must"):
        wavelet_fft(data, sigma=1.0, wavelet="db4", level=0)


# -- orientation --------------------------------------------------------------
def test_horizontal_is_transposed_vertical():
    data = make_frames(2) + make_stripes(2, (64, 80))
    kw = dict(wavelet="db6", level=3, sigma=1.0)
    vertical = wavelet_fft(data, axis=-2, **kw)
    horizontal = wavelet_fft(
        data.transpose(-1, -2).contiguous(), axis=-1, **kw
    )
    torch.testing.assert_close(horizontal.transpose(-1, -2), vertical)
    # positive axes are accepted
    torch.testing.assert_close(wavelet_fft(data, axis=1, **kw), vertical)


def test_wrong_orientation_keeps_stripes():
    content = make_frames(1)
    stripes = make_stripes(1, content.shape[-2:])
    out = wavelet_fft(
        content + stripes, wavelet="db10", level=5, sigma=0.0, axis=-1
    )
    assert (out - content).square().sum() > 0.5 * stripes.square().sum()


def test_estimator_axis_labels():
    content = make_frames(2)
    stripes = make_stripes(2, content.shape[-2:])
    data = content + stripes  # (z, y, x), stripes along y
    kw = dict(wavelet="db8", level=4, sigma=0.5)
    expected = wavelet_fft(data, axis=-2, **kw)
    f64 = dict(kw, dtype=torch.float64)
    layout = Layout("zyx", spatial="yx")
    for axis in ("y", 0, -2):
        out = WaveletFFT(axis=axis, layout=layout, **f64).apply(data)
        torch.testing.assert_close(out, expected)
    # spatial axes given as (x, y): stripes along y are now the 2nd axis
    swapped = Layout("zyx", spatial="xy")
    out = WaveletFFT(axis="y", layout=swapped, **f64).apply(data)
    torch.testing.assert_close(out, expected)
    out = WaveletFFT(axis=1, **f64).apply(labeled(data, swapped))
    torch.testing.assert_close(out, expected)
    # storage (y, z, x) with frames along z
    perm = data.permute(1, 0, 2).contiguous()
    out = WaveletFFT(axis="y", **f64).apply(
        labeled(perm, Layout("yzx", spatial="yx"))
    )
    torch.testing.assert_close(out.permute(1, 0, 2), expected)
    with pytest.raises(ValueError, match="not a spatial axis"):
        WaveletFFT(axis="z", layout=layout, **f64).apply(data)
    with pytest.raises(ValueError, match="2 spatial axes"):
        WaveletFFT(layout=Layout("zyx", spatial="zyx"), **f64).apply(data)


# -- estimator ----------------------------------------------------------------
def test_estimator_is_stateless():
    est = WaveletFFT(wavelet="db4", level=3, sigma=1.5)
    assert est.is_faithful
    assert est.fit() is est and est.estimates == {}
    data = make_frames(2)
    assert est.fit(data) is est and est.estimates == {}
    assert "WaveletFFT" in optix.estimators()
    assert est.tags.streaming == "per-frame"
    with pytest.raises(ValueError, match="must not be None"):
        wavelet_fft(data, wavelet=None)
    with pytest.raises(ValueError, match="Unknown wavelet"):
        WaveletFFT(wavelet="sym4", level=2, sigma=1.0)


def test_defaults():
    """The Section 3.8 suggestions: db10, L = 4, σ = 1.5."""
    data = make_frames(2, (64, 48))
    explicit = dict(wavelet="db10", level=4, sigma=1.5)
    expected = wavelet_fft(data, **explicit)
    torch.testing.assert_close(wavelet_fft(data), expected)
    f64 = dict(dtype=torch.float64)
    est = WaveletFFT(**f64)
    assert est.is_faithful
    assert {k: est.options[k] for k in explicit} == explicit
    assert WaveletFFT.paper == WaveletFFT.recommended
    torch.testing.assert_close(est.apply(data), expected)
    torch.testing.assert_close(
        WaveletFFT(faithful=False, **f64).apply(data), expected
    )
    torch.testing.assert_close(
        WaveletFFT(**explicit, **f64).apply(data), expected
    )
    # the three parameters stay free: other values are faithful
    assert WaveletFFT(wavelet="db4", level=2, sigma=0.5).is_faithful


def test_estimator_save_load(tmp_path):
    est = WaveletFFT(wavelet="db4", level=3, sigma=[1.0, 2.0, 3.0], axis=-1)
    est.save(tmp_path / "w.pt")
    back = optix.load(tmp_path / "w.pt")
    assert isinstance(back, WaveletFFT) and back.options == est.options
    data = make_frames(2)
    torch.testing.assert_close(back.apply(data), est.apply(data))


def test_stream_matches_batch():
    content = make_frames(5, (48, 40))
    data = content + make_stripes(5, (48, 40))
    kw = dict(wavelet="db5", level=3, sigma=1.0)
    expected = wavelet_fft(data, **kw)
    est = WaveletFFT(**kw, dtype=torch.float64)
    with optix.config(batch=2):
        out = est.apply(data.numpy())
    torch.testing.assert_close(torch.as_tensor(out), expected)
    # one frame at a time, into a preallocated output
    out = np.zeros(data.shape)
    with optix.config(batch=1):
        est.apply(data.numpy(), out=out)
    torch.testing.assert_close(torch.as_tensor(out), expected)
    # grouped layout: (c, t, y, x), frames per channel
    stack = data.reshape(1, 5, 48, 40).expand(2, 5, 48, 40).contiguous()
    lay = Layout("ctyx", spatial="yx", group="c")
    with optix.config(batch=3):
        out = WaveletFFT(layout=lay, **kw, dtype=torch.float64).apply(stack)
    torch.testing.assert_close(out, expected.expand(2, 5, 48, 40))


def test_stream_zarr():
    zarr = pytest.importorskip("zarr")
    data = make_frames(4, (32, 36)) + make_stripes(4, (32, 36))
    kw = dict(wavelet="db3", level=2, sigma=1.0)
    src = zarr.zeros(data.shape, chunks=(1, 32, 36), dtype="f8")
    src[:] = data.numpy()
    dst = zarr.zeros(data.shape, chunks=(1, 32, 36), dtype="f8")
    with optix.config(batch=1):
        WaveletFFT(**kw, dtype=torch.float64).apply(src, out=dst)
    torch.testing.assert_close(
        torch.as_tensor(dst[:]), wavelet_fft(data, **kw)
    )
