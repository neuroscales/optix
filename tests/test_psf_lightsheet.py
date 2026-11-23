"""Tests of optix.psf light-sheet illumination and effective PSF."""

import math

import pytest
import torch

from optix.psf import BeadPSF, gaussian_sheet, lightsheet_psf
from optix.psf.utils import (
    beam_radius,
    fwhm,
    rayleigh_range,
    sheet_intensity,
)

F64 = torch.float64
W0 = 2.0  # µm
LAMBDA = 0.488  # µm
N = 1.33


def detection_psf(shape, sigma):
    out = torch.ones(shape, dtype=F64)
    for i, (n, s) in enumerate(zip(shape, sigma, strict=True)):
        view = [1] * len(shape)
        view[i] = n
        g = (torch.arange(n, dtype=F64) - n // 2) / s
        out = out * torch.exp(-0.5 * g**2).view(view)
    return out / out.sum()


def test_rayleigh_range_and_radius():
    zr = rayleigh_range(W0, LAMBDA, N)
    assert zr == pytest.approx(math.pi * W0**2 * N / LAMBDA)
    w = beam_radius(
        torch.tensor([0.0, zr, -zr, 3 * zr], dtype=F64),
        waist=W0,
        wavelength=LAMBDA,
        refractive_index=N,
    )
    expected = W0 * torch.tensor([1.0, 2**0.5, 2**0.5, 10**0.5], dtype=F64)
    assert torch.allclose(w, expected, rtol=1e-12)


def test_sheet_intensity_analytic_values():
    zr = rayleigh_range(W0, LAMBDA, N)
    kw = {"waist": W0, "wavelength": LAMBDA, "refractive_index": N}
    assert float(sheet_intensity(0.0, 0.0, **kw)) == pytest.approx(1.0)
    # 1/e² at the waist radius
    assert float(sheet_intensity(W0, 0.0, **kw)) == pytest.approx(math.exp(-2))
    # peak falls as W₀ / W(z): 1/√2 at the Rayleigh range
    assert float(sheet_intensity(0.0, zr, **kw)) == pytest.approx(2**-0.5)
    w = W0 * 2**0.5
    value = float(sheet_intensity(w, zr, **kw))
    assert value == pytest.approx(2**-0.5 * math.exp(-2))
    # power per unit length is conserved along the propagation axis
    t = torch.linspace(-60, 60, 200001, dtype=F64)
    for z in (0.0, zr, 4 * zr):
        power = torch.trapezoid(sheet_intensity(t, z, **kw), t)
        assert float(power) == pytest.approx(W0 * math.sqrt(math.pi / 2))


def test_sheet_fwhm_thickness():
    zr = rayleigh_range(W0, LAMBDA, N)
    sheet = gaussian_sheet(
        (401, 3),
        detection=0,
        propagation=1,
        waist=W0,
        wavelength=LAMBDA,
        refractive_index=N,
        spacing=(0.02, zr),
        dtype=F64,
    )
    widths = [float(fwhm(sheet[:, j], [200])[0]) * 0.02 for j in range(3)]
    k = math.sqrt(2 * math.log(2))
    assert widths[1] == pytest.approx(W0 * k, rel=1e-4)
    assert widths[0] == pytest.approx(W0 * 2**0.5 * k, rel=1e-4)
    assert widths[2] == pytest.approx(widths[0], rel=1e-9)
    assert float(sheet.max()) == pytest.approx(1.0)


def test_gaussian_sheet_axes_labels_and_uniformity():
    kw = {"waist": W0, "wavelength": LAMBDA, "refractive_index": N}
    a = gaussian_sheet(
        (9, 11, 13),
        detection="z",
        propagation="x",
        axes="zyx",
        **kw,
        dtype=F64,
    )
    b = gaussian_sheet(
        (9, 11, 13), detection=0, propagation=2, **kw, dtype=F64
    )
    assert torch.equal(a, b)
    # uniform along y, symmetric about the centre in z and x
    assert torch.equal(a, a[:, :1].expand_as(a))
    assert torch.allclose(a, a.flip(0))
    assert torch.allclose(a, a.flip(2))
    i = torch.arange(9, dtype=F64) - 4
    k = torch.arange(13, dtype=F64) - 6
    expected = sheet_intensity(i[:, None], k[None, :], **kw)
    assert torch.allclose(a[:, 5], expected)
    # swapping the roles transposes the pattern
    c = gaussian_sheet(
        (13, 11, 9), detection=2, propagation=0, **kw, dtype=F64
    )
    assert torch.allclose(c, a.permute(2, 1, 0))
    with pytest.raises(ValueError, match="must differ"):
        gaussian_sheet((4, 4), detection=0, propagation=-2, **kw)
    with pytest.raises(ValueError, match="label"):
        gaussian_sheet((4, 4), detection="z", propagation="x", **kw)


def test_offset_and_focus_shift_the_sheet():
    kw = {"waist": W0, "wavelength": LAMBDA, "refractive_index": N}
    s = gaussian_sheet(
        (21, 31),
        detection=0,
        propagation=1,
        offset=1.5,
        focus=-4.0,
        spacing=0.5,
        dtype=F64,
        **kw,
    )
    t = (torch.arange(21, dtype=F64) - 10) * 0.5 - 1.5
    z = (torch.arange(31, dtype=F64) - 15) * 0.5 + 4.0
    assert torch.allclose(s, sheet_intensity(t[:, None], z[None], **kw))
    assert tuple(map(int, divmod(int(s.argmax()), 31))) == (13, 7)


def test_effective_psf_is_the_product():
    shape = (33, 15, 17)
    spacing = (0.25, 0.1, 0.1)
    h = detection_psf(shape, (6.0, 2.0, 2.0))
    kw = {"waist": W0, "wavelength": LAMBDA, "refractive_index": N}
    zr = rayleigh_range(W0, LAMBDA, N)
    out = lightsheet_psf(
        h, detection="z", axes="zyx", spacing=spacing, position=zr, **kw
    )
    t = (torch.arange(33, dtype=F64) - 16) * 0.25
    expected = h * sheet_intensity(t, zr, **kw).view(-1, 1, 1)
    assert torch.allclose(out, expected, rtol=1e-12)
    # same as the product with the rendered sheet at the point's position
    sheet = gaussian_sheet(
        (33, 3),
        detection=0,
        propagation=1,
        spacing=(0.25, zr),
        dtype=F64,
        **kw,
    )[:, 2]
    assert torch.allclose(out, h * sheet.view(-1, 1, 1), rtol=1e-12)
    # the sheet narrows the axial extent only
    w_det, w_eff = fwhm(h), fwhm(out)
    assert float(w_eff[0]) < float(w_det[0])
    assert torch.allclose(w_eff[1:], w_det[1:])
    norm = lightsheet_psf(h, detection=0, normalize=True, **kw)
    assert float(norm.sum()) == pytest.approx(1.0)


def test_effective_psf_detection_axis_choice():
    h = detection_psf((15, 21, 15), (2.0, 5.0, 2.0))
    kw = {"waist": 1.0, "wavelength": LAMBDA, "refractive_index": N}
    a = lightsheet_psf(h, detection=1, **kw)
    b = lightsheet_psf(h.permute(1, 0, 2), detection=0, **kw).permute(1, 0, 2)
    assert torch.allclose(a, b)
    ratio = a / h
    assert torch.allclose(ratio, ratio[:1, :, :1].expand_as(ratio))
    with pytest.raises(ValueError, match="out of range"):
        lightsheet_psf(h, detection=3, **kw)


def test_far_from_waist_sheet_matters_less():
    h = detection_psf((41, 9, 9), (8.0, 1.5, 1.5))
    kw = {"waist": 1.0, "wavelength": LAMBDA, "refractive_index": N}
    zr = rayleigh_range(1.0, LAMBDA, N)
    widths = [
        float(fwhm(lightsheet_psf(h, detection=0, position=p, **kw))[0])
        for p in (0.0, zr, 5 * zr, 500 * zr)
    ]
    assert widths == sorted(widths)
    assert widths[-1] == pytest.approx(float(fwhm(h)[0]), rel=0.02)


def test_measured_sheet_profile():
    h = detection_psf((11, 7), (3.0, 1.5))
    profile = torch.linspace(0.2, 1.0, 11, dtype=F64)
    out = lightsheet_psf(h, detection=0, sheet=profile)
    assert torch.allclose(out, h * profile[:, None])
    full = torch.rand(11, 7, dtype=F64)
    assert torch.allclose(lightsheet_psf(h, detection=0, sheet=full), h * full)
    with pytest.raises(ValueError, match="samples"):
        lightsheet_psf(h, detection=0, sheet=torch.ones(7))
    with pytest.raises(TypeError, match="waist"):
        lightsheet_psf(h, detection=0)


def test_float32_matches_float64():
    h = detection_psf((21, 9, 9), (4.0, 1.5, 1.5))
    kw = {"waist": 1.2, "wavelength": LAMBDA, "refractive_index": N}
    a = lightsheet_psf(h.float(), detection=0, spacing=0.2, position=7.0, **kw)
    b = lightsheet_psf(h, detection=0, spacing=0.2, position=7.0, **kw)
    assert a.dtype == torch.float32
    assert torch.allclose(a.double(), b, rtol=1e-5, atol=1e-9)


def test_with_bead_psf():
    gen = torch.Generator().manual_seed(0)
    shape = (40, 60, 60)
    h = detection_psf((17, 11, 11), (3.0, 1.4, 1.4))
    expected = torch.full(shape, 10.0, dtype=F64)
    for z, y, x in [(20, 15, 15), (20, 15, 45), (20, 45, 15), (20, 45, 45)]:
        expected[z - 8 : z + 9, y - 5 : y + 6, x - 5 : x + 6] += 5e5 * h
    data = torch.poisson(expected, generator=gen)
    measured = BeadPSF(radius=(8, 5, 5)).fit(data).estimates["psf"].psf
    out = lightsheet_psf(
        measured, detection=0, waist=1.0, wavelength=LAMBDA, spacing=0.3
    )
    assert out.shape == measured.shape
    assert float(fwhm(out)[0]) < float(fwhm(measured)[0])
