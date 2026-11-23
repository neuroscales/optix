"""Tests of optix.psf.richards_wolf (Richards & Wolf 1959)."""

import math

import pytest
import torch

from optix.psf import richards_wolf
from optix.psf.utils import (
    fwhm,
    richards_wolf_field,
    richards_wolf_integrals,
)
from optix.special import bessel_j012

F64 = torch.float64
OPTICS = {
    "wavelength": 0.52,
    "numerical_aperture": 1.4,
    "refractive_index": 1.518,
}
SPACING = (0.1, 0.04, 0.04)  # µm, (z, y, x)


def _i0_focus(alpha):
    """I₀(0, 0), Eq. (3.18)."""
    c = math.cos(alpha)
    return 16 / 15 * (1 - 5 / 8 * c**1.5 * (1 + 3 / 5 * c))


# --------------------------------------------------------------------------
# Special functions and quadrature
# --------------------------------------------------------------------------


def test_bessel_j012_against_mpmath():
    mpmath = pytest.importorskip("mpmath")
    x = torch.cat(
        [
            torch.linspace(0, 30, 301, dtype=F64),
            torch.tensor([1e-8, 4.9, 5.0, 5.1, 24.99, 25.0, 80.0, 1e3]),
            -torch.tensor([0.3, 7.0, 40.0], dtype=F64),
        ]
    )
    out = bessel_j012(x)
    for n, j in enumerate(out):
        ref = torch.tensor(
            [float(mpmath.besselj(n, float(v))) for v in x], dtype=F64
        )
        assert (j - ref).abs().max() < 5e-15


def test_bessel_j012_float32(device):
    x = torch.linspace(0, 60, 601, dtype=F64)
    ref = bessel_j012(x)
    out = bessel_j012(x.float().to(device))
    for a, b in zip(out, ref, strict=True):
        assert a.dtype == torch.float32
        assert (a.cpu().double() - b).abs().max() < 2e-6


@pytest.mark.parametrize("alpha", [0.1, math.radians(60), math.pi / 2])
def test_value_at_focus(alpha):
    """I₀(0, 0) has the closed form (3.18); I₁ = I₂ = 0 on the axis."""
    zero = torch.zeros((), dtype=F64)
    i0, i1, i2 = richards_wolf_integrals(zero, zero, alpha)
    assert float(i0.real) == pytest.approx(_i0_focus(alpha), rel=1e-13)
    assert abs(float(i0.imag)) < 1e-15
    kz = torch.linspace(-50, 50, 11, dtype=F64)
    _, i1, i2 = richards_wolf_integrals(torch.zeros_like(kz), kz, alpha)
    assert i1.abs().max() < 1e-15 and i2.abs().max() < 1e-15


@pytest.mark.parametrize("alpha", [0.3, math.radians(60), math.pi / 2])
def test_quadrature_convergence(alpha):
    """Exponential convergence in the number of nodes; the automatic
    choice reaches the floating-point floor."""
    kr = torch.linspace(0, 80, 41, dtype=F64)
    kz = torch.linspace(-80, 80, 21, dtype=F64)
    kr, kz = torch.meshgrid(kr, kz, indexing="ij")
    ref = richards_wolf_integrals(kr, kz, alpha, nodes=600)
    peak = float(ref[0].abs().max())

    def error(nodes):
        out = richards_wolf_integrals(kr, kz, alpha, nodes=nodes)
        return (
            max(
                float((a - b).abs().max())
                for a, b in zip(out, ref, strict=True)
            )
            / peak
        )

    errors = [error(n) for n in (16, 32, 64)]
    assert errors[0] > errors[1] > errors[2] or errors[2] < 1e-12
    assert error(None) < 1e-12


def test_conjugate_symmetry():
    """Iₙ(−u, v) = Iₙ*(u, v), Eq. (3.3)."""
    alpha = math.radians(75)
    kr = torch.linspace(0, 20, 9, dtype=F64)[:, None]
    kz = torch.linspace(0.5, 20, 7, dtype=F64)[None]
    plus = richards_wolf_integrals(kr, kz, alpha)
    minus = richards_wolf_integrals(kr, -kz, alpha)
    for a, b in zip(plus, minus, strict=True):
        torch.testing.assert_close(b, a.conj(), atol=1e-14, rtol=0)


# --------------------------------------------------------------------------
# Paper values
# --------------------------------------------------------------------------

# Table 1 of the paper: v at which the quantity, normalised to 100 at the
# focus, falls to 80, 60, 40, 20, then its first and second minima, for
# α → 0, 30°, 60°, 75°, → 90°. None: no entry in the table.
_ALPHAS = (0.0, 30.0, 60.0, 75.0, 90.0)
_TABLE1 = {
    "we_phi0": [
        (0.93, 0.98, 1.16, 1.33, 1.48),
        (1.40, 1.47, 1.73, 1.95, 2.10),
        (1.83, 1.94, 2.27, 2.50, 2.63),
        (2.36, 2.50, 2.93, 3.14, 3.23),
        (3.83, 4.15, 4.75, 4.75, 4.70),
        (7.02, None, 8.10, 8.10, 8.00),  # α = 30°: 7.03 in the table
    ],
    "we_phi90": [
        (0.93, 0.93, 0.89, 0.86, 0.84),
        (1.40, 1.38, 1.32, 1.30, 1.26),
        (1.83, 1.82, 1.76, 1.70, 1.64),
        (2.36, 2.33, 2.22, 2.16, 2.10),
        (3.83, 3.77, 3.56, 3.40, 3.29),
        (7.02, 6.95, 6.85, 6.70, 6.55),
    ],
    "unpolarized": [
        (0.93, 0.95, 1.00, 1.03, 1.05),
        (1.40, 1.42, 1.50, 1.55, 1.58),
        (1.83, 1.87, 1.99, 2.05, 2.10),
        (2.36, 2.41, 2.56, 2.66, 2.73),
        (3.83, 3.95, None, None, None),
        (7.02, None, None, None, None),
    ],
    "poynting_z": [
        (0.93, 0.93, 0.93, 0.93, 0.91),
        (1.40, 1.40, 1.39, 1.38, 1.36),
        (1.83, 1.83, 1.82, 1.82, 1.79),
        (2.36, 2.36, 2.34, 2.33, 2.27),
        (3.83, None, 3.80, 3.76, 3.70),  # α = 30°: 3.80 in the table
        (7.02, 7.02, 7.02, 7.02, 6.95),
    ],
}


def _features(profile, v):
    """Level crossings (0.8 … 0.2) and the first two local minima."""
    out = []
    dv = float(v[1] - v[0])
    for level in (0.8, 0.6, 0.4, 0.2):
        i = int(torch.nonzero(profile < level)[0])
        a, b = float(profile[i - 1]), float(profile[i])
        out.append(float(v[i - 1]) + (a - level) / (a - b) * dv)
    d = profile[1:] - profile[:-1]
    minima = torch.nonzero((d[:-1] < 0) & (d[1:] >= 0)).flatten() + 1
    out += [float(v[m]) for m in minima[:2]]
    return out


@pytest.mark.parametrize("column", range(5))
def test_table1(column):
    """Table 1 (focal plane, u = 0): within 0.03 in v (the table is
    given to 0.01 and its α → 0 column departs from the exact Airy values
    by up to 0.01)."""
    deg = _ALPHAS[column]
    alpha = math.radians(deg) if deg > 0 else 1e-3
    v = torch.linspace(0, 9, 9001, dtype=F64)
    i0, i1, i2 = richards_wolf_integrals(v / math.sin(alpha), 0.0, alpha)
    a0, a1, a2 = i0.abs() ** 2, i1.abs() ** 2, i2.abs() ** 2
    cross = (i0 * i2.conj()).real
    focus = float(a0[0])
    curves = {  # Eqs. (3.14), (5.2), (3.21), at φ = 0 and φ = π/2
        "we_phi0": (a0 + 4 * a1 + a2 + 2 * cross) / focus,
        "we_phi90": (a0 + a2 - 2 * cross) / focus,
        "unpolarized": (a0 + 2 * a1 + a2) / focus,
        "poynting_z": (a0 - a2) / focus,
    }
    for name, rows in _TABLE1.items():
        found = _features(curves[name], v)
        for row, (expected, got) in enumerate(zip(rows, found, strict=False)):
            if expected[column] is None:
                continue
            tol = 0.025 if row < 4 else 0.03
            assert got == pytest.approx(expected[column], abs=tol), (
                name,
                row,
            )


def test_table1_low_aperture_is_airy():
    """α → 0: the exact Airy levels and zeros (the table's first column)."""
    v = torch.linspace(0, 9, 90001, dtype=F64)
    alpha = 1e-4
    i0, _, _ = richards_wolf_integrals(v / math.sin(alpha), 0.0, alpha)
    found = _features(i0.abs() ** 2 / float(i0[0].abs() ** 2), v)
    exact = [0.935937, 1.398698, 1.840019, 2.361810, 3.831706, 7.015587]
    for a, b in zip(found, exact, strict=True):
        assert a == pytest.approx(b, abs=2e-4)


@pytest.mark.parametrize("deg", [30, 60, 90])
def test_energy_conservation(deg):
    """Flux of ⟨S_z⟩ (Eq. 3.21) through the focal plane equals the power
    entering the pupil: ∫₀^∞ (|I₀|² − |I₂|²) v dv = 2 sin⁴α (Parseval on
    Eq. 2.2, with |a|² = f² l₀² cos θ from Eq. 2.13). The truncated
    integral converges as 1/V; the tail is extrapolated."""
    alpha = math.radians(deg)
    big = 200.0
    v = torch.linspace(0, big, 20001, dtype=F64)
    i0, _, i2 = richards_wolf_integrals(v / math.sin(alpha), 0.0, alpha)
    f = (i0.abs() ** 2 - i2.abs() ** 2) * v
    cum = torch.cumsum((f[1:] + f[:-1]) / 2 * float(v[1] - v[0]), 0)
    upper = v[1:]
    keep = upper > big / 2
    design = torch.stack([torch.ones_like(upper[keep]), -1 / upper[keep]], 1)
    total = torch.linalg.lstsq(design, cum[keep, None]).solution[0, 0]
    assert float(total) == pytest.approx(2 * math.sin(alpha) ** 4, rel=1e-4)


# --------------------------------------------------------------------------
# Low-aperture limit
# --------------------------------------------------------------------------


def test_low_aperture_airy_and_lommel():
    """α ≪ 1: Airy pattern in the focal plane (Eq. 4.12) and
    [sin(u/4) / (u/4)]² on the axis (Eqs. 4.4, 4.8)."""
    na, lam = 0.02, 0.5
    alpha = math.asin(na)
    k = 2 * math.pi / lam
    step = 0.5 / (k * na)  # Δv = 0.5
    plane = richards_wolf(
        (81, 81),
        wavelength=lam,
        numerical_aperture=na,
        spacing=step,
        normalize="focus",
        dtype=F64,
    )
    v = (torch.arange(81, dtype=F64) - 40) * 0.5
    j0, j1, _ = bessel_j012(v)
    safe = torch.where(v == 0, torch.ones_like(v), v)
    airy = torch.where(v == 0, torch.ones_like(v), (2 * j1 / safe) ** 2)
    torch.testing.assert_close(plane[40], airy, atol=1e-3, rtol=0)
    torch.testing.assert_close(plane[:, 40], airy, atol=1e-3, rtol=0)

    dz = 1.0 / (k * math.sin(alpha) ** 2)  # Δu = 1
    axis = richards_wolf(
        (81, 1, 1),
        wavelength=lam,
        numerical_aperture=na,
        spacing=(dz, 1.0, 1.0),
        normalize="focus",
        dtype=F64,
    )[:, 0, 0]
    u = torch.arange(81, dtype=F64) - 40
    sinc = torch.special.sinc(u / (4 * math.pi)) ** 2
    torch.testing.assert_close(axis, sinc, atol=1e-3, rtol=0)


# --------------------------------------------------------------------------
# Grids, symmetries and options
# --------------------------------------------------------------------------


def test_field_gives_intensity():
    """⟨w_e⟩ = |e|² / 16π (Eqs. 2.30, 3.13, 3.14)."""
    shape = (9, 16, 17)
    e = richards_wolf_field(shape, **OPTICS, spacing=SPACING, dtype=F64)
    w = richards_wolf(shape, **OPTICS, spacing=SPACING, dtype=F64)
    assert e.shape == (3, *shape) and e.is_complex()
    torch.testing.assert_close(
        e.abs().square().sum(0) / (16 * math.pi), w, atol=1e-15, rtol=1e-12
    )
    # On the axis the field is linearly polarised along x (Section 3.1).
    centre = e[:, :, 8, 8]
    assert centre[1:].abs().max() < 1e-15


def test_value_at_focus_matches_paper():
    """(A²/16π) I₀(0,0)² with A = 1 (Eq. 3.19)."""
    w = richards_wolf((5, 5, 5), **OPTICS, spacing=SPACING, dtype=F64)
    alpha = math.asin(1.4 / 1.518)
    expected = _i0_focus(alpha) ** 2 / (16 * math.pi)
    assert float(w[2, 2, 2]) == pytest.approx(expected, rel=1e-13)
    assert int(w.argmax()) == w.numel() // 2


def test_symmetries():
    shape = (16, 32, 32)
    kw = {**OPTICS, "spacing": SPACING, "dtype": F64}
    w = richards_wolf(shape, **kw)
    inner = w[1:, 1:, 1:]  # symmetric about index n // 2
    # Symmetric about the focal plane (Eq. 3.15) and about the x and y axes.
    torch.testing.assert_close(inner, inner.flip(0), atol=1e-14, rtol=1e-10)
    torch.testing.assert_close(inner, inner.flip(1), atol=1e-14, rtol=1e-10)
    torch.testing.assert_close(inner, inner.flip(2), atol=1e-14, rtol=1e-10)
    # Not radially symmetric: elongated along the polarisation (x).
    focal = w[8]
    assert (focal - focal.T).abs().max() > 0.05 * focal.max()
    width_y, width_x = fwhm(focal).tolist()
    assert width_x > 1.1 * width_y
    # y polarisation and an angle of π/2: the transpose.
    wy = richards_wolf(shape, **kw, polarization="y")
    torch.testing.assert_close(wy, w.transpose(1, 2), atol=1e-15, rtol=1e-12)
    wa = richards_wolf(shape, **kw, polarization=math.pi / 2)
    torch.testing.assert_close(wa, wy, atol=1e-15, rtol=1e-12)
    # Unpolarised (Eq. 5.2): the mean of x and y, radially symmetric.
    wu = richards_wolf(shape, **kw, polarization="unpolarized")
    torch.testing.assert_close(wu, (w + wy) / 2, atol=1e-15, rtol=1e-12)
    torch.testing.assert_close(wu, wu.transpose(1, 2), atol=1e-15, rtol=1e-12)


def test_float32_matches_float64(device, dtype):
    shape = (8, 24, 24)
    ref = richards_wolf(shape, **OPTICS, spacing=SPACING, dtype=F64)
    out = richards_wolf(
        shape, **OPTICS, spacing=SPACING, dtype=dtype, device=device
    )
    assert out.dtype == dtype and out.device.type == device.type
    err = (out.cpu().double() - ref).abs().max() / ref.max()
    assert err < (2e-6 if dtype == torch.float32 else 1e-12)


def test_focal_plane_shape():
    kw = {**OPTICS, "dtype": F64}
    vol = richards_wolf((5, 12, 13), **kw, spacing=SPACING)
    plane = richards_wolf((12, 13), **kw, spacing=SPACING[1:])
    torch.testing.assert_close(plane, vol[2], atol=0, rtol=1e-12)


def test_normalize():
    kw = {**OPTICS, "spacing": SPACING, "dtype": F64}
    shape = (12, 20, 20)
    raw = richards_wolf(shape, **kw)
    unit = richards_wolf(shape, **kw, normalize="sum")
    focus = richards_wolf(shape, **kw, normalize="focus")
    assert float(unit.sum()) == pytest.approx(1.0, rel=1e-12)
    torch.testing.assert_close(unit, raw / raw.sum())
    assert float(focus[6, 10, 10]) == pytest.approx(1.0, rel=1e-12)
    torch.testing.assert_close(focus, raw / raw[6, 10, 10])


def test_pupil_phase():
    """ψ = k δ cos θ turns e^{ikz cos θ} into e^{ik(z + δ) cos θ}: the
    pattern moves by −δ along the axis."""
    lam, n = 0.52, 1.518
    k = 2 * math.pi * n / lam
    kw = {
        "wavelength": lam,
        "numerical_aperture": 1.2,
        "refractive_index": n,
        "spacing": 0.1,
        "dtype": F64,
    }
    shape = (21, 9, 9)
    base = richards_wolf(shape, **kw)
    same = richards_wolf(shape, **kw, pupil_phase=lambda t: 0.7 + 0 * t)
    torch.testing.assert_close(same, base, atol=1e-15, rtol=1e-12)
    delta = 0.3  # three voxels
    moved = richards_wolf(
        shape, **kw, pupil_phase=lambda t: k * delta * torch.cos(t)
    )
    torch.testing.assert_close(moved[:-3], base[3:], atol=1e-15, rtol=1e-10)


def test_emitters():
    kw = {**OPTICS, "spacing": SPACING, "dtype": F64}
    shape = (10, 18, 18)
    unpol = richards_wolf(shape, **kw, polarization="unpolarized")
    iso = richards_wolf(shape, **kw, emitter="isotropic")
    torch.testing.assert_close(iso, 2 / 3 * unpol, atol=1e-15, rtol=1e-12)
    px = richards_wolf(shape, **kw, emitter=(2.0, 0, 0))
    py = richards_wolf(shape, **kw, emitter=(0, 1, 0))
    pz = richards_wolf(shape, **kw, emitter=(0, 0, 1))
    # Orientation average: (h_x + h_y + h_z) / 3.
    torch.testing.assert_close((px + py + pz) / 3, iso, atol=1e-15, rtol=1e-12)
    # x dipole: the x components of the x- and y-polarised fields.
    e = richards_wolf_field(shape, **kw)
    ex_x = e[0].abs().square()
    ex_y = e[1].transpose(-1, -2).abs().square()  # e⁽ʸ⁾_x(x, y)
    torch.testing.assert_close(
        px, (ex_x + ex_y) / (16 * math.pi), atol=1e-15, rtol=1e-12
    )
    # An axial dipole is dark on the axis (a doughnut).
    assert pz[:, 9, 9].abs().max() < 1e-15 * pz.max()
    assert float(pz.max()) > 0
    # Normalisations.
    iso_f = richards_wolf(shape, **kw, emitter="isotropic", normalize="focus")
    assert float(iso_f[5, 9, 9]) == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(ValueError, match="vanishes"):
        richards_wolf(shape, **kw, emitter=(0, 0, 1), normalize="focus")


def test_errors():
    kw = {"wavelength": 0.5, "spacing": 0.1}
    with pytest.raises(ValueError, match="numerical_aperture"):
        richards_wolf((4, 4), numerical_aperture=1.2, **kw)
    with pytest.raises(ValueError, match="shape"):
        richards_wolf((4,), numerical_aperture=0.5, **kw)
    with pytest.raises(ValueError, match="normalize"):
        richards_wolf((4, 4), numerical_aperture=0.5, normalize="max", **kw)
    with pytest.raises(ValueError, match="polarization"):
        richards_wolf((4, 4), numerical_aperture=0.5, polarization="z", **kw)
    with pytest.raises(ValueError, match="polarization"):
        richards_wolf(
            (4, 4),
            numerical_aperture=0.5,
            polarization="y",
            emitter="isotropic",
            **kw,
        )
    with pytest.raises(ValueError, match="emitter"):
        richards_wolf((4, 4), numerical_aperture=0.5, emitter="dipole", **kw)
    with pytest.raises(ValueError, match="aperture"):
        richards_wolf_integrals(0.0, 0.0, 2.0)
