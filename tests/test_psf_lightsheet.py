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


def test_waist_is_the_minimum_radius_and_fixes_the_rayleigh_range():
    # W(z) is smallest at z = 0, where it equals W₀; W(z_R)² = 2 W₀²
    zr = rayleigh_range(W0, LAMBDA, N)
    z = torch.linspace(-5 * zr, 5 * zr, 1001, dtype=F64)
    kw = {"waist": W0, "wavelength": LAMBDA, "refractive_index": N}
    w = beam_radius(z, **kw)
    assert int(w.argmin()) == 500
    assert float(w.min()) == pytest.approx(W0, rel=1e-12)
    w_zr = beam_radius(torch.tensor(zr, dtype=F64), **kw)
    assert float(w_zr) ** 2 == pytest.approx(2 * W0**2, rel=1e-12)
    # far field: W(z) → θ₀ z with θ₀ = λ / (π W₀), λ = λ₀ / n
    far = float(beam_radius(torch.tensor(1e6 * zr, dtype=F64), **kw))
    theta = LAMBDA / N / (math.pi * W0)
    assert far / (1e6 * zr) == pytest.approx(theta, rel=1e-9)


def test_sheet_fwhm_closed_form_along_propagation():
    zr = rayleigh_range(W0, LAMBDA, N)
    positions = (0.0, 0.5, 2.0)
    kw = {"waist": W0, "wavelength": LAMBDA, "refractive_index": N}
    t = torch.linspace(-20, 20, 4001, dtype=F64)
    k = math.sqrt(2 * math.log(2))
    for p in positions:
        profile = sheet_intensity(t, p * zr, **kw)
        width = float(fwhm(profile, [2000])[0]) * 0.01
        w = float(beam_radius(torch.tensor(p * zr, dtype=F64), **kw))
        assert width == pytest.approx(k * w, rel=1e-5)


def test_rotation_zero_is_identity():
    kw = {"waist": W0, "wavelength": LAMBDA, "refractive_index": N}
    grid = {"detection": 0, "propagation": 2, "spacing": (0.5, 1.0, 2.0)}
    ref = gaussian_sheet(
        (9, 5, 11), offset=0.7, focus=-3.0, dtype=F64, **grid, **kw
    )
    for rot in (0.0, torch.eye(3, dtype=F64)):
        out = gaussian_sheet(
            (9, 5, 11),
            offset=0.7,
            focus=-3.0,
            rotation=rot,
            dtype=F64,
            **grid,
            **kw,
        )
        assert torch.allclose(out, ref, rtol=1e-12)
    h = detection_psf((15, 9, 9), (3.0, 1.5, 1.5))
    base = lightsheet_psf(h, detection=0, position=4.0, offset=0.3, **kw)
    for rot, prop in ((0.0, 2), (torch.eye(3), None)):
        out = lightsheet_psf(
            h,
            detection=0,
            propagation=prop,
            rotation=rot,
            position=4.0,
            offset=0.3,
            **kw,
        )
        assert torch.allclose(out, base, rtol=1e-12)
    # a 1-D measured profile at the samples of the detection axis
    profile = torch.linspace(0.1, 1.0, 15, dtype=F64)
    out = lightsheet_psf(h, detection=0, sheet=profile, rotation=torch.eye(3))
    assert torch.allclose(out, h * profile[:, None, None], rtol=1e-12)


def test_rotation_by_90_degrees_swaps_the_axes():
    kw = {"waist": W0, "wavelength": LAMBDA, "refractive_index": N}
    shape = (21, 7, 21)
    rotated = gaussian_sheet(
        shape,
        detection=0,
        propagation=2,
        rotation=math.pi / 2,
        offset=0.5,
        dtype=F64,
        **kw,
    )
    swapped = gaussian_sheet(
        shape, detection=2, propagation=0, offset=0.5, dtype=F64, **kw
    )
    # propagation turns onto −e_det: the waist is symmetric (focus = 0)
    assert torch.allclose(rotated, swapped, rtol=1e-9, atol=1e-14)
    # the same rotation as a matrix (columns: sheet axes in the grid)
    mat = torch.zeros(3, 3, dtype=F64)
    mat[2, 0], mat[1, 1], mat[0, 2] = 1.0, 1.0, -1.0
    by_matrix = gaussian_sheet(
        shape,
        detection=0,
        propagation=2,
        rotation=mat,
        offset=0.5,
        dtype=F64,
        **kw,
    )
    assert torch.allclose(by_matrix, swapped, rtol=1e-12)
    # effective PSF: a 90° rotation moves the sheet factor to the other axis
    h = detection_psf((15, 9, 15), (3.0, 1.5, 3.0))
    a = lightsheet_psf(
        h, detection=0, propagation=2, rotation=math.pi / 2, **kw
    )
    b = lightsheet_psf(h, detection=2, **kw)
    assert torch.allclose(a, b, rtol=1e-9, atol=1e-16)


def test_rotation_errors():
    kw = {"waist": W0, "wavelength": LAMBDA}
    h = detection_psf((9, 9), (2.0, 2.0))
    with pytest.raises(ValueError, match="propagation"):
        lightsheet_psf(h, detection=0, rotation=0.3, **kw)
    with pytest.raises(ValueError, match="orthogonal"):
        lightsheet_psf(h, detection=0, rotation=2 * torch.eye(2), **kw)
    with pytest.raises(ValueError, match="matrix"):
        gaussian_sheet(
            (5, 5), detection=0, propagation=1, rotation=torch.eye(3), **kw
        )
    with pytest.raises(ValueError, match="1-D"):
        lightsheet_psf(
            h, detection=0, rotation=torch.eye(2), sheet=torch.ones(9, 9)
        )


def gaussian_on_grid(shape, precision, spacing):
    """exp(−½ rᵀ Q r) on the physical grid, 0 at index n // 2."""
    coords = torch.meshgrid(
        *[
            (torch.arange(n, dtype=F64) - n // 2) * s
            for n, s in zip(shape, spacing, strict=True)
        ],
        indexing="ij",
    )
    r = torch.stack(coords, -1)
    return torch.exp(-0.5 * torch.einsum("...i,ij,...j->...", r, precision, r))


def test_product_with_gaussian_detection_psf_closed_form():
    # σ_eff⁻² = σ⁻² + 4 / W(z)², amplitude W₀ / W(z)
    kw = {"waist": 1.5, "wavelength": LAMBDA, "refractive_index": N}
    zr = rayleigh_range(1.5, LAMBDA, N)
    spacing = (0.1, 0.2, 0.2)
    sigma = (0.8, 0.3, 0.3)
    shape = (61, 15, 15)
    q = torch.diag(torch.tensor([s**-2 for s in sigma], dtype=F64))
    h = gaussian_on_grid(shape, q, spacing)
    for z in (0.0, zr, 3 * zr):
        w = float(beam_radius(torch.tensor(z, dtype=F64), **kw))
        out = lightsheet_psf(h, detection=0, spacing=spacing, position=z, **kw)
        q_eff = q.clone()
        q_eff[0, 0] += 4 / w**2
        expected = (1.5 / w) * gaussian_on_grid(shape, q_eff, spacing)
        assert torch.allclose(out, expected, rtol=1e-10, atol=1e-15)
        s_eff = (sigma[0] ** -2 + 4 / w**2) ** -0.5
        width = float(fwhm(out)[0]) * spacing[0]
        assert width == pytest.approx(
            math.sqrt(8 * math.log(2)) * s_eff, rel=2e-3
        )


def test_rotated_product_closed_form():
    # Σ_eff⁻¹ = Σ⁻¹ + (4 / W²) n nᵀ with n = R e_det
    kw = {"waist": 1.0, "wavelength": LAMBDA, "refractive_index": N}
    theta = math.radians(30)
    spacing = (0.15, 0.15, 0.15)
    shape = (31, 31, 9)
    q = torch.diag(torch.tensor([1 / 0.9**2, 1 / 0.6**2, 4.0], dtype=F64))
    h = gaussian_on_grid(shape, q, spacing)
    out = lightsheet_psf(
        h,
        detection=0,
        propagation=1,
        rotation=theta,
        spacing=spacing,
        position=2.0,
        **kw,
    )
    w = float(beam_radius(torch.tensor(2.0, dtype=F64), **kw))
    n = torch.tensor([math.cos(theta), math.sin(theta), 0.0], dtype=F64)
    q_eff = q + (4 / w**2) * torch.outer(n, n)
    expected = (1.0 / w) * gaussian_on_grid(shape, q_eff, spacing)
    assert torch.allclose(out, expected, rtol=1e-10, atol=1e-15)
    # float32 agrees
    out32 = lightsheet_psf(
        h.float(),
        detection=0,
        propagation=1,
        rotation=theta,
        spacing=spacing,
        position=2.0,
        **kw,
    )
    assert torch.allclose(out32.double(), out, rtol=1e-5, atol=1e-8)


def test_rotated_measured_profile_matches_gaussian_sheet():
    # a 1-D profile of the Gaussian sheet, finely sampled, reproduces
    # the rotated Gaussian factor up to the interpolation error
    kw = {"waist": 1.0, "wavelength": LAMBDA, "refractive_index": N}
    h = detection_psf((21, 21), (3.0, 3.0))
    t = (torch.arange(401, dtype=F64) - 200) * 0.01
    profile = sheet_intensity(t, 0.0, **kw)
    rot = {"detection": 0, "propagation": 1, "rotation": 0.4}
    a = lightsheet_psf(h, sheet=profile, spacing=(0.01, 0.01), **rot)
    b = lightsheet_psf(h, spacing=(0.01, 0.01), **rot, **kw)
    assert torch.allclose(a, b, atol=1e-6)


def _grid_coords(shape, spacing):
    """1-D physical coordinates, 0 at index n // 2, as broadcast views."""
    out = []
    for i, (n, s) in enumerate(zip(shape, spacing, strict=True)):
        view = [1] * len(shape)
        view[i] = n
        out.append(((torch.arange(n, dtype=F64) - n // 2) * s).view(view))
    return out


def test_rotated_sheet_with_focus_and_offset():
    # t = n · r − offset, z = p · r − focus at 30°, anisotropic spacing;
    # focus ≠ 0 makes the sign of p = R e_prop observable
    kw = {"waist": W0, "wavelength": LAMBDA, "refractive_index": N}
    theta = math.radians(30)
    shape, spacing = (17, 5, 23), (0.4, 1.0, 1.5)
    offset, focus = 0.8, -4.0
    out = gaussian_sheet(
        shape,
        detection=0,
        propagation=2,
        rotation=theta,
        spacing=spacing,
        offset=offset,
        focus=focus,
        dtype=F64,
        **kw,
    )
    r0, _, r2 = _grid_coords(shape, spacing)
    c, s = math.cos(theta), math.sin(theta)
    t = c * r0 + s * r2 - offset
    z = -s * r0 + c * r2 - focus
    expected = sheet_intensity(t, z, **kw).expand(shape)
    assert torch.allclose(out, expected, rtol=1e-12, atol=1e-15)
    # the opposite sign of p (or of the angle) gives a different sheet
    flipped = gaussian_sheet(
        shape,
        detection=0,
        propagation=2,
        rotation=-theta,
        spacing=spacing,
        offset=offset,
        focus=focus,
        dtype=F64,
        **kw,
    )
    assert not torch.allclose(flipped, out, rtol=1e-3)


def _orthogonal_3d():
    # a non-planar rotation: about axis 2 by 25°, then about axis 0 by 40°
    a, b = math.radians(40), math.radians(25)
    rx = torch.tensor(
        [
            [1, 0, 0],
            [0, math.cos(a), -math.sin(a)],
            [0, math.sin(a), math.cos(a)],
        ],
        dtype=F64,
    )
    rz = torch.tensor(
        [
            [math.cos(b), -math.sin(b), 0],
            [math.sin(b), math.cos(b), 0],
            [0, 0, 1],
        ],
        dtype=F64,
    )
    return rx @ rz


def test_non_planar_rotation_closed_form():
    kw = {"waist": 1.0, "wavelength": LAMBDA, "refractive_index": N}
    mat = _orthogonal_3d()
    spacing = (0.15, 0.15, 0.15)
    shape = (31, 31, 31)
    q = torch.diag(
        torch.tensor([1 / 0.9**2, 1 / 0.6**2, 1 / 0.5**2], dtype=F64)
    )
    h = gaussian_on_grid(shape, q, spacing)
    out = lightsheet_psf(
        h, detection=0, rotation=mat, spacing=spacing, position=1.5, **kw
    )
    w = float(beam_radius(torch.tensor(1.5, dtype=F64), **kw))
    n = mat[:, 0]
    assert int((n.abs() > 1e-3).sum()) == 3  # not in a coordinate plane
    q_eff = q + (4 / w**2) * torch.outer(n, n)
    expected = (1.0 / w) * gaussian_on_grid(shape, q_eff, spacing)
    assert torch.allclose(out, expected, rtol=1e-10, atol=1e-15)


def test_rotated_psf_with_offset_and_anisotropic_spacing():
    kw = {"waist": 1.0, "wavelength": LAMBDA, "refractive_index": N}
    mat = _orthogonal_3d()
    shape, spacing = (21, 15, 11), (0.1, 0.2, 0.3)
    h = detection_psf(shape, (4.0, 3.0, 2.0))
    out = lightsheet_psf(
        h,
        detection=1,
        rotation=mat,
        spacing=spacing,
        position=3.0,
        offset=0.25,
        **kw,
    )
    coords = _grid_coords(shape, spacing)
    n = mat[:, 1]
    t = sum(float(n[i]) * coords[i] for i in range(3)) - 0.25
    expected = h * sheet_intensity(t, torch.tensor(3.0, dtype=F64), **kw)
    assert torch.allclose(out, expected, rtol=1e-12, atol=1e-18)
    # the offset moves the sheet along n, away from the point source
    centred = lightsheet_psf(
        h, detection=1, rotation=mat, spacing=spacing, position=3.0, **kw
    )
    assert not torch.allclose(out, centred, rtol=1e-3)


def test_improper_rotation_is_accepted():
    # det R = −1: negating an unused column leaves the sheet unchanged
    kw = {"waist": W0, "wavelength": LAMBDA, "refractive_index": N}
    mirror = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=F64))
    assert float(torch.linalg.det(mirror)) == pytest.approx(-1.0)
    grid = {"detection": 0, "propagation": 2, "spacing": (0.5, 1.0, 2.0)}
    ref = gaussian_sheet(
        (9, 5, 11), offset=0.7, focus=-3.0, dtype=F64, **grid, **kw
    )
    out = gaussian_sheet(
        (9, 5, 11),
        offset=0.7,
        focus=-3.0,
        rotation=mirror,
        dtype=F64,
        **grid,
        **kw,
    )
    assert torch.allclose(out, ref, rtol=1e-12)
    h = detection_psf((15, 9, 9), (3.0, 1.5, 1.5))
    base = lightsheet_psf(h, detection=0, position=4.0, offset=0.3, **kw)
    out = lightsheet_psf(
        h, detection=0, rotation=mirror, position=4.0, offset=0.3, **kw
    )
    assert torch.allclose(out, base, rtol=1e-12)


def test_empty_sheet_raises():
    h = detection_psf((9, 9), (2.0, 2.0))
    for rot in (None, torch.eye(2)):
        with pytest.raises(ValueError, match="empty"):
            lightsheet_psf(h, detection=0, rotation=rot, sheet=torch.ones(0))


def test_gradient_through_tensor_angle():
    kw = {"waist": W0, "wavelength": LAMBDA, "refractive_index": N}
    grid = {"detection": 0, "propagation": 1, "spacing": 0.5, "focus": 2.0}

    def total(theta):
        return gaussian_sheet(
            (15, 15), rotation=theta, dtype=F64, **grid, **kw
        ).sum()

    theta = torch.tensor(0.3, dtype=F64, requires_grad=True)
    (grad,) = torch.autograd.grad(total(theta), theta)
    eps = 1e-6
    numeric = (total(0.3 + eps) - total(0.3 - eps)) / (2 * eps)
    assert float(grad) == pytest.approx(float(numeric), rel=1e-5)
    h = detection_psf((15, 15), (3.0, 3.0))
    theta = torch.tensor(0.3, dtype=F64, requires_grad=True)
    out = lightsheet_psf(
        h, detection=0, propagation=1, rotation=theta, spacing=0.5, **kw
    )
    (grad,) = torch.autograd.grad(out.sum(), theta)
    assert torch.isfinite(grad) and float(grad) != 0.0
