"""Tests of the linear deconvolution filters (Wiener, Tikhonov)."""

import math

import numpy as np
import pytest
import torch

from optix.deconv import Tikhonov, Wiener, tikhonov, wiener
from optix.deconv.utils import (
    PaddedConvolution,
    frequencies,
    laplacian_power,
    smooth_periodic_extension,
    transfer_function,
)
from optix.estimates import SensorEstimate
from optix.io import Layout

F64 = torch.float64


def gaussian_psf(size, sigma, ndim=2, dtype=F64):
    x = torch.arange(size, dtype=dtype) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    out = g
    for _ in range(ndim - 1):
        out = out[..., None] * g
    return out / out.sum()


def three_tap(ndim, c=0.6):
    """Separable [a, c, a] PSF, a = (1 − c)/2.

    H(ν) = ∏ (c + (1 − c) cos 2πν).
    """
    a = (1 - c) / 2
    g = torch.tensor([a, c, a], dtype=F64)
    out = g
    for _ in range(ndim - 1):
        out = out[..., None] * g
    return out


def circular_blur(truth, psf):
    dims = tuple(range(truth.ndim))
    otf = transfer_function(psf, truth.shape)
    return torch.fft.irfftn(
        torch.fft.rfftn(truth, dim=dims) * otf, s=truth.shape, dim=dims
    )


def scene(shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    truth = torch.full(shape, 5.0, dtype=F64)
    n = math.prod(shape) // 60
    idx = torch.randint(0, math.prod(shape), (n,), generator=gen)
    truth.view(-1)[idx] += 100.0
    return truth, gen


# -- building blocks ----------------------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
def test_transfer_function_matches_padded_convolution(ndim):
    psf = gaussian_psf(5, 1.0, ndim=ndim)
    shape = (12, 10, 9)[:ndim]
    op = PaddedConvolution(psf, shape)
    torch.testing.assert_close(op.otf, transfer_function(psf, op.grid))
    with pytest.raises(ValueError):
        transfer_function(psf, (3,) * ndim)


def test_smooth_periodic_extension():
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(2, 7, 5, generator=gen, dtype=F64)
    ext = smooth_periodic_extension(x, (12, 9))
    assert ext.shape == (2, 12, 9)
    torch.testing.assert_close(ext[:, :7, :5], x)
    # wrapped differences stay small: no jump at the seams
    rolled = torch.roll(ext, 1, dims=-2)
    jump = (ext - rolled).abs().amax()
    assert float(jump) <= float((x[:, 1:] - x[:, :-1]).abs().amax()) + 1


def test_laplacian_power():
    freqs = frequencies((8, 6), dtype=F64)
    p = laplacian_power(freqs)
    lap = torch.zeros(8, 6, dtype=F64)
    lap[0, 0] = -4
    lap[1, 0] = lap[-1, 0] = lap[0, 1] = lap[0, -1] = 1
    ref = torch.fft.rfftn(lap).abs().square()
    torch.testing.assert_close(p.expand_as(ref), ref)


# -- exact inversion and frequency response ----------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("kind", ["wiener", "tikhonov"])
def test_exact_inversion_of_circular_blur(ndim, kind):
    psf = three_tap(ndim)  # |H| ≥ 0.2³, invertible
    shape = (24, 20, 16)[:ndim]
    truth, _ = scene(shape)
    data = circular_blur(truth, psf)
    kw = {"boundary": "periodic", "block": shape, "dtype": F64}
    if kind == "wiener":
        out = wiener(data, psf, nsr=1e-14, **kw)
    else:
        out = tikhonov(data, psf, lam=1e-14, operator="laplacian", **kw)
    torch.testing.assert_close(out, truth, rtol=0, atol=1e-6)


@pytest.mark.parametrize("operator", ["identity", "laplacian"])
def test_analytic_frequency_response(operator):
    shape = (16, 12)
    c, lam = 0.6, 0.05
    psf = three_tap(2, c)
    delta = torch.zeros(shape, dtype=F64)
    delta[0, 0] = 1
    out = tikhonov(
        delta, psf, lam=lam, operator=operator, boundary="periodic", dtype=F64
    )
    nu = frequencies(shape, dtype=F64)
    h = (c + (1 - c) * torch.cos(2 * math.pi * nu[0])) * (
        c + (1 - c) * torch.cos(2 * math.pi * nu[1])
    )
    s = (
        4 * torch.sin(math.pi * nu[0]) ** 2
        + 4 * torch.sin(math.pi * nu[1]) ** 2
    )
    reg = 1.0 if operator == "identity" else s**2
    expected = h / (h**2 + lam * reg)
    torch.testing.assert_close(torch.fft.rfftn(out), expected + 0j)
    # Wiener with a frequency-dependent K(ν) = Sₙ/S_f
    k = lambda f: 0.01 * (1 + 50 * (f[0] ** 2 + f[1] ** 2))  # noqa: E731
    out = wiener(delta, psf, nsr=k, boundary="periodic", dtype=F64)
    expected = h / (h**2 + k(nu))
    torch.testing.assert_close(torch.fft.rfftn(out), expected + 0j)


def test_wiener_constant_equals_tikhonov_identity():
    psf = gaussian_psf(7, 1.2)
    truth, gen = scene((40, 36))
    data = circular_blur(truth, psf) + torch.randn(40, 36, generator=gen)
    torch.testing.assert_close(
        wiener(data, psf, nsr=0.03), tikhonov(data, psf, lam=0.03)
    )


# -- regularisation trade-off -------------------------------------------------
@pytest.mark.parametrize("operator", ["identity", "laplacian"])
def test_lambda_trades_noise_against_blur(operator):
    psf = gaussian_psf(9, 1.5)
    shape = (64, 64)
    gen = torch.Generator().manual_seed(3)
    noise = torch.randn(shape, generator=gen, dtype=F64)
    point = torch.zeros(shape, dtype=F64)
    point[32, 32] = 1.0
    blurred = circular_blur(point, psf)
    lams = [1e-4, 1e-3, 1e-2, 1e-1]
    gain, peak = [], []
    for lam in lams:
        kw = {"lam": lam, "operator": operator, "boundary": "periodic"}
        gain.append(float(tikhonov(noise, psf, **kw).std()))
        peak.append(float(tikhonov(blurred, psf, **kw)[32, 32]))
    # more regularisation: less noise amplification, less resolution
    assert all(a > b for a, b in zip(gain, gain[1:], strict=False))
    assert all(a > b for a, b in zip(peak, peak[1:], strict=False))
    assert peak[0] > 2 * float(blurred[32, 32])
    # and an intermediate λ restores noisy data best
    truth, gen = scene(shape, seed=4)
    data = circular_blur(truth, psf) + 2 * torch.randn(
        shape, generator=gen, dtype=F64
    )
    errs = [
        float((tikhonov(data, psf, lam=lam, operator=operator) - truth).std())
        for lam in (1e-6, 1e-2, 10.0)
    ]
    assert errs[1] < errs[0] and errs[1] < errs[2]


def test_gcv_picks_a_reasonable_lambda():
    psf = gaussian_psf(9, 1.5)
    shape = (64, 64)
    truth, gen = scene(shape, seed=5)
    data = circular_blur(truth, psf) + 2 * torch.randn(
        shape, generator=gen, dtype=F64
    )
    est = Tikhonov(psf, lam="gcv", operator="laplacian").fit(data)
    assert est.is_faithful is False
    assert 1e-8 < est.strength < 10
    out = est.apply(data)
    err = float((out - truth).std())
    for lam in (1e-8, 100.0):
        assert err < float((tikhonov(data, psf, lam=lam) - truth).std())
    # block-wise GCV pools the blocks
    parts = Tikhonov(psf, lam="gcv", operator="laplacian", block=(32, 32))
    ratio = parts.fit(data).strength / est.strength
    assert 0.1 < ratio < 10


# -- blocks, dims, lazy input -------------------------------------------------
def test_blocks_match_whole_image_2d():
    psf = gaussian_psf(9, 1.5)
    truth, gen = scene((96, 80), seed=1)
    data = circular_blur(truth, psf) + torch.randn(96, 80, generator=gen)
    data = data.double()
    for fn, kw in ((wiener, {"nsr": 1e-2}), (tikhonov, {"lam": 1e-2})):
        errs = []
        for halo in (18, 27, 36):
            kw.update(halo=halo, dtype=F64)
            whole = fn(data, psf, block=(96, 80), **kw)
            parts = fn(data, psf, block=(32, 24), **kw)
            errs.append(float((parts - whole).abs().max() / whole.abs().max()))
        assert errs[0] < 1e-3, errs
        assert errs[2] < 1e-6, errs
        assert errs[0] > errs[1] > errs[2]


def test_blocks_match_whole_image_3d():
    psf = gaussian_psf(5, 1.0, ndim=3)
    truth, gen = scene((20, 24, 22), seed=2)
    data = circular_blur(truth, psf) + torch.randn(
        20, 24, 22, generator=gen, dtype=F64
    )
    whole = tikhonov(data, psf, lam=1e-2, block=(20, 24, 22))
    parts = tikhonov(data, psf, lam=1e-2, block=(10, 12, 11))
    err = float((parts - whole).abs().max() / whole.abs().max())
    assert err < 1e-2


def test_taper_reduces_edge_ringing():
    """A ramp, not periodic: tapering avoids ringing from the wrap jump."""
    psf = gaussian_psf(7, 1.2)
    x = torch.linspace(0, 100, 48, dtype=F64)
    truth = x[:, None] + x[None, :]
    op = PaddedConvolution(psf, truth.shape)
    # blur with the object extended by replication (no wrap-around)
    big = torch.nn.functional.pad(truth[None, None], (8, 8, 8, 8), "replicate")
    data = circular_blur(big[0, 0], psf)[8:-8, 8:-8]
    assert op.shape == data.shape
    taper = wiener(data, psf, nsr=1e-3)
    periodic = wiener(data, psf, nsr=1e-3, boundary="periodic")
    err_t = float((taper - truth).abs().max())
    err_p = float((periodic - truth).abs().max())
    assert err_t < 0.5 * err_p


def test_stack_with_layout():
    psf = gaussian_psf(7, 1.2)
    imgs = [circular_blur(scene((32, 28), seed=s)[0], psf) for s in range(3)]
    stack = torch.stack(imgs, dim=-1)
    out = wiener(stack, psf, nsr=1e-2, layout=Layout("yxt", spatial="yx"))
    for i, img in enumerate(imgs):
        torch.testing.assert_close(out[..., i], wiener(img, psf, nsr=1e-2))


def test_lazy_zarr_input_and_output(tmp_path):
    zarr = pytest.importorskip("zarr")
    psf = gaussian_psf(7, 1.2)
    data = torch.stack(
        [circular_blur(scene((64, 48), seed=s)[0], psf) for s in (4, 5)]
    )
    kw = {"chunks": (1, 16, 16), "dtype": "f4", "shape": data.shape}
    src = zarr.open_array(str(tmp_path / "in.zarr"), mode="w", **kw)
    src[:] = data.numpy().astype("f4")
    dst = zarr.open_array(str(tmp_path / "out.zarr"), mode="w", **kw)
    opts = {"lam": 1e-2, "block": (32, 16), "dtype": torch.float32}
    tikhonov(src, psf, out=dst, **opts)
    ref = tikhonov(data.float(), psf, **opts)
    torch.testing.assert_close(torch.from_numpy(dst[:]), ref)


def test_float32_matches_float64():
    psf = gaussian_psf(9, 1.5)
    truth, gen = scene((48, 48), seed=6)
    data = circular_blur(truth, psf) + torch.randn(48, 48, generator=gen)
    for kw in ({"lam": 1e-3, "operator": "laplacian"}, {"lam": 1e-2}):
        a = tikhonov(data, psf, dtype=F64, **kw)
        b = tikhonov(data.float(), psf.float(), dtype="float32", **kw)
        assert float((a - b).abs().max() / a.abs().max()) < 1e-4


# -- options, sensor, persistence ---------------------------------------------
def test_faithful_flags_and_validation():
    psf = gaussian_psf(5, 1.0)
    assert Wiener(psf).is_faithful
    assert Tikhonov(psf).is_faithful
    assert Wiener(psf, nsr=0.5).is_faithful is False
    assert Tikhonov(psf, lam="gcv").is_faithful is False
    data = torch.ones(16, 16, dtype=F64)
    sensor = SensorEstimate(offset=torch.tensor(0.0), gain=torch.tensor(1.0))
    assert Wiener(psf).fit(data, init=sensor).is_faithful is False
    for bad in (
        lambda: Wiener(psf, nsr=-1.0),
        lambda: Tikhonov(psf, lam=-1.0),
        lambda: Tikhonov(psf, lam="auto"),
        lambda: Tikhonov(psf, operator="gradient"),
        lambda: Wiener(psf, boundary="mirror"),
        lambda: Wiener(torch.ones(5)),
    ):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(RuntimeError):
        Wiener(psf).apply(data)
    with pytest.raises(ValueError, match="halo"):
        Wiener(psf, halo=2).fit(data).apply(data)


def test_sensor_converts_to_electrons():
    psf = gaussian_psf(7, 1.2)
    electrons = circular_blur(scene((40, 36), seed=7)[0], psf)
    gain = torch.linspace(1.5, 2.5, 36, dtype=F64).expand(40, 36)
    offset = torch.tensor(100.0, dtype=F64)
    sensor = SensorEstimate(offset=offset, gain=gain)
    out = wiener(
        offset + gain * electrons,
        psf,
        sensor=sensor,
        block=(20, 18),
        dtype=F64,
    )
    torch.testing.assert_close(
        out, wiener(electrons, psf, block=(20, 18), dtype=F64)
    )


def test_save_load(tmp_path):
    from optix import load

    psf = gaussian_psf(5, 1.0)
    data = circular_blur(scene((24, 24), seed=8)[0], psf)
    est = Tikhonov(psf, lam="gcv", operator="laplacian").fit(data)
    est.save(tmp_path / "t.pt")
    new = load(tmp_path / "t.pt")
    assert new.strength == est.strength
    torch.testing.assert_close(new.apply(data), est.apply(data))


def test_numpy_input():
    psf = gaussian_psf(5, 1.0)
    data = circular_blur(scene((24, 20), seed=9)[0], psf)
    out = wiener(data.numpy(), psf)
    np.testing.assert_allclose(np.asarray(out), wiener(data, psf).numpy())
