"""Tests of the Gibson–Lanni PSF (optix.psf.gibson_lanni)."""

import math
import os
import time

import numpy as np
import pytest
import torch

from optix.estimates import PSFEstimate
from optix.psf import gibson_lanni
from optix.psf.utils import gibson_lanni_opd, gibson_lanni_profile
from optix.special import bessel_j0, bessel_j1, gauss_legendre

F64 = torch.float64
LAMBDA = 0.53  # µm, emission of the beads of Gibson & Lanni
NA = 1.3
N_OIL = 1.515
# Aberration-free: every layer at the immersion index
MATCHED = dict(
    numerical_aperture=NA,
    specimen_index=N_OIL,
    design_coverslip_index=N_OIL,
    design_immersion_index=N_OIL,
)
# Gibson & Lanni Figs. 4, 11 and 13: source in water under the design
# coverslip (1.522, 170 µm) and oil (1.515)
WATER = dict(numerical_aperture=NA, specimen_index=1.33)
# Li et al. Sec. 3A
LI = dict(
    numerical_aperture=1.4,
    specimen_index=1.33,
    design_coverslip_index=1.5,
    design_immersion_index=1.5,
    design_coverslip_thickness=170.0,
    design_immersion_thickness=150.0,
)
FIXTURES = os.path.join(
    os.path.dirname(__file__), "data", "microscpsf_fixtures.npz"
)


def airy(r, wavelength=LAMBDA, na=NA):
    v = 2 * math.pi / wavelength * na * r
    safe = torch.where(v > 0, v, torch.ones_like(v))
    return torch.where(v > 0, (2 * bessel_j1(safe) / safe) ** 2, 1.0)


def axial_exact(dz, wavelength=LAMBDA, na=NA, n=N_OIL):
    """|2 ∫₀¹ exp(ik Δz (√(n² − NA²ρ²) − n)) ρ dρ|², closed form."""
    k = 2 * math.pi / wavelength
    c1 = math.sqrt(n * n - na * na)
    out = []
    for d in dz.tolist():
        if abs(d) < 1e-9:
            out.append(1.0)
            continue
        al = k * d

        def prim(c, al=al):
            return complex(math.cos(al * (c - n)), math.sin(al * (c - n))) * (
                c / (1j * al) + 1 / al**2
            )

        out.append(abs(2 * (prim(n) - prim(c1)) / na**2) ** 2)
    return torch.tensor(out, dtype=F64)


# ----------------------------------------------------------------------
# Special functions
# ----------------------------------------------------------------------


def test_bessel_accurate_in_float64():
    # Series of J₀ and J₁ summed in exact rational arithmetic
    from fractions import Fraction

    def series(order, x, terms=120):
        xf = Fraction(x)
        total, term = Fraction(0), (xf / 2) ** order / math.factorial(order)
        for m in range(terms):
            total += term
            term *= -((xf / 2) ** 2) / ((m + 1) * (m + 1 + order))
        return float(total)

    xs = [0.0, 0.3, 2.404825557695773, 5.5, 7.0, 12.3, 19.9, 24.9, -6.1]
    x = torch.tensor(xs, dtype=F64)
    ref0 = torch.tensor([series(0, v) for v in xs], dtype=F64)
    ref1 = torch.tensor([series(1, v) for v in xs], dtype=F64)
    assert (bessel_j0(x) - ref0).abs().max() < 2e-15
    assert (bessel_j1(x) - ref1).abs().max() < 2e-15
    # continuity with torch above the switch point; float32 unchanged
    y = torch.tensor([25.0, 40.0, 300.0], dtype=F64)
    assert torch.equal(bessel_j0(y), torch.special.bessel_j0(y))
    z = torch.linspace(0, 30, 31)
    assert torch.equal(bessel_j1(z), torch.special.bessel_j1(z))


def test_gauss_legendre():
    for n in (1, 2, 7, 64, 300):
        x, w = gauss_legendre(n, dtype=F64)
        xr, wr = np.polynomial.legendre.leggauss(n)
        assert np.abs(x.numpy() - xr).max() < 1e-14
        assert np.abs(w.numpy() - wr).max() < 1e-13
    # exact for degree 2n − 1
    x, w = gauss_legendre(5, dtype=F64)
    for d in range(10):
        exact = (1 - (-1) ** (d + 1)) / (d + 1)
        assert float((w * x**d).sum()) == pytest.approx(exact, abs=1e-14)
    with pytest.raises(ValueError):
        gauss_legendre(0)


# ----------------------------------------------------------------------
# Optical path difference (Eqs. 2–4)
# ----------------------------------------------------------------------


def test_opd_equals_eq2_with_eq3():
    """Eq. (4) is Eq. (2) with t_oil eliminated through Eq. (3)."""
    p = dict(
        numerical_aperture=1.35,
        depth=12.0,
        specimen_index=1.38,
        coverslip_index=1.53,
        coverslip_thickness=160.0,
        immersion_index=1.51,
        design_coverslip_index=1.522,
        design_coverslip_thickness=170.0,
        design_immersion_index=1.515,
        design_immersion_thickness=140.0,
    )
    rho = torch.linspace(0, 1, 101, dtype=F64)
    for dz in (-3.0, 0.0, 2.5):
        ns, ng, ni = 1.38, 1.53, 1.51
        ng0, ni0, tg0, ti0 = 1.522, 1.515, 170.0, 140.0
        ts, tg = 12.0, 160.0
        toil = ni * (dz / ni - ts / ns - tg / ng + tg0 / ng0 + ti0 / ni0)

        def c(n):
            return torch.sqrt(1 - (1.35 * rho / n) ** 2)

        eq2 = (
            ns * ts * c(ns)
            + ng * tg * c(ng)
            + ni * toil * c(ni)
            - ng0 * tg0 * c(ng0)
            - ni0 * ti0 * c(ni0)
        )
        ours = gibson_lanni_opd(rho, dz, **p)
        torch.testing.assert_close(ours, eq2 - eq2[0], rtol=0, atol=1e-11)


def test_opd_paraxial_terms():
    """Δz = 0 is the paraxial focus: the ρ² term is −NA² ρ² Δz / (2 n_oil)."""
    rho = torch.tensor([1e-3, 2e-3], dtype=F64)
    for dz in (0.0, 1.7):
        opd = gibson_lanni_opd(rho, dz, depth=30.0, **WATER)
        quad = -(NA**2) * rho**2 * dz / (2 * N_OIL)
        # what is left is O(ρ⁴)
        assert ((opd - quad) / rho**4).abs().max() < 50


def test_coverslip_tolerance():
    """Gibson & Lanni p. 1609: ±30 µm of coverslip gives < λ/4 of
    spherical aberration (NA 1.3); with no other mismatch the OPD at
    Δz = 0 is pure higher-order aberration."""
    rho = torch.linspace(0, 1, 201, dtype=F64)
    for dt in (-30.0, 30.0):
        opd = gibson_lanni_opd(
            rho,
            0.0,
            numerical_aperture=NA,
            specimen_index=1.522,
            coverslip_thickness=170.0 + dt,
        )
        spread = float(opd.max() - opd.min())
        assert LAMBDA / 8 < spread < LAMBDA / 4


# ----------------------------------------------------------------------
# Aberration-free limits
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["quadrature", "li2017"])
def test_airy_in_focus(method):
    r = torch.linspace(0, 3, 121, dtype=F64)
    out = gibson_lanni_profile(
        r, 0.0, wavelength=LAMBDA, method=method, dtype=F64, **MATCHED
    )[0]
    # J₀(σₘρ) cannot represent a constant phase exactly: σ₁ > 0
    tol = 1e-12 if method == "quadrature" else 1e-4
    torch.testing.assert_close(out, airy(r), rtol=0, atol=tol)


def test_axial_exact_and_paraxial_sinc():
    dz = torch.linspace(-4, 4, 161, dtype=F64)
    out = gibson_lanni_profile(
        0.0, dz, wavelength=LAMBDA, dtype=F64, **MATCHED
    )
    torch.testing.assert_close(out[:, 0], axial_exact(dz), rtol=0, atol=1e-12)
    # paraxial limit (Born & Wolf): sinc²(u/4), u = k NA² Δz / n
    na = 0.25
    low = dict(MATCHED, numerical_aperture=na)
    dz = torch.linspace(-40, 40, 161, dtype=F64)
    out = gibson_lanni_profile(0.0, dz, wavelength=LAMBDA, dtype=F64, **low)
    u = 2 * math.pi / LAMBDA * na**2 * dz / N_OIL
    sinc = torch.sinc(u / (4 * math.pi)) ** 2
    torch.testing.assert_close(out[:, 0], sinc, rtol=0, atol=2e-2)


def test_grid_matches_airy_and_unit_sum():
    psf = gibson_lanni(
        (1, 33, 33), spacing=0.05, wavelength=LAMBDA, dtype=F64, **MATCHED
    )
    assert psf.shape == (1, 33, 33)
    assert float(psf.sum()) == pytest.approx(1.0, abs=1e-12)
    y = (torch.arange(33, dtype=F64) - 16) * 0.05
    r = torch.hypot(y[:, None], y[None, :])
    ref = airy(r)
    torch.testing.assert_close(
        psf[0] / psf[0, 16, 16], ref, atol=1e-12, rtol=0
    )


# ----------------------------------------------------------------------
# Symmetries
# ----------------------------------------------------------------------


def test_symmetries():
    kw = dict(spacing=(0.2, 0.08, 0.08), wavelength=LAMBDA, dtype=F64)
    free = gibson_lanni((21, 25, 25), **kw, **MATCHED)
    # lateral mirror and transpose; axial mirror without aberration
    torch.testing.assert_close(free, free.flip(1), rtol=0, atol=1e-15)
    torch.testing.assert_close(free, free.flip(2), rtol=0, atol=1e-15)
    torch.testing.assert_close(free, free.transpose(1, 2), rtol=0, atol=1e-15)
    torch.testing.assert_close(free, free.flip(0), rtol=1e-9, atol=1e-15)
    assert int(free.flatten().argmax()) == free.numel() // 2
    # depth in water: laterally symmetric, axially asymmetric
    deep = gibson_lanni((21, 25, 25), depth=10.0, **kw, **WATER)
    torch.testing.assert_close(deep, deep.flip(2), rtol=0, atol=1e-15)
    assert (deep - deep.flip(0)).abs().max() > 0.1 * deep.max()


def test_anisotropic_pixels():
    kw = dict(wavelength=LAMBDA, dtype=F64, depth=3.0, **WATER)
    psf = gibson_lanni((5, 13, 9), spacing=(0.3, 0.06, 0.09), **kw)
    y = (torch.arange(13, dtype=F64) - 6) * 0.06
    x = (torch.arange(9, dtype=F64) - 4) * 0.09
    z = (torch.arange(5, dtype=F64) - 2) * 0.3
    r = torch.hypot(y[:, None], x[None, :])
    prof = gibson_lanni_profile(
        r.flatten(), z, wavelength=LAMBDA, dtype=F64, **WATER | {"depth": 3.0}
    )
    ref = prof.view(5, 13, 9)
    torch.testing.assert_close(psf, ref / ref.sum(), rtol=1e-12, atol=0)


# ----------------------------------------------------------------------
# Quadrature accuracy
# ----------------------------------------------------------------------


def test_quadrature_converges():
    """Successive refinement at large radius, defocus and depth."""
    r = torch.tensor([0.0, 2.0, 8.0], dtype=F64)
    dz = torch.tensor([-15.0, 0.0, 15.0], dtype=F64)
    kw = dict(wavelength=LAMBDA, dtype=F64, depth=40.0, **WATER)
    ref = gibson_lanni_profile(r, dz, nodes=4096, **kw)
    errors = []
    for n in (32, 64, 128, 256):
        out = gibson_lanni_profile(r, dz, nodes=n, **kw)
        errors.append(float((out - ref).abs().max() / ref.max()))
    assert errors[-1] < 1e-12
    assert all(b < a for a, b in zip(errors, errors[1:], strict=False))
    # automatic node count meets its target
    auto = gibson_lanni_profile(r, dz, rtol=1e-8, **kw)
    assert float((auto - ref).abs().max() / ref.max()) < 1e-7


def test_truncated_aperture_converges():
    """NA 1.4 > n_s = 1.33: the pupil ends at ρ = n_s/NA."""
    r = torch.tensor([0.0, 0.5, 3.0], dtype=F64)
    dz = torch.tensor([-2.0, 0.0, 2.0], dtype=F64)
    kw = dict(wavelength=0.61, dtype=F64, depth=5.0, **LI)
    ref = gibson_lanni_profile(r, dz, nodes=4096, **kw)
    out = gibson_lanni_profile(r, dz, nodes=128, **kw)
    assert float((out - ref).abs().max() / ref.max()) < 1e-12


def test_float32_matches_float64():
    kw = dict(spacing=(0.25, 0.1, 0.1), wavelength=0.61, depth=10.0, nodes=512)
    ref = gibson_lanni((33, 48, 48), dtype=F64, **kw, **LI)
    for method in ("quadrature", "li2017"):
        out = gibson_lanni(
            (33, 48, 48), dtype=torch.float32, method=method, **kw, **LI
        )
        assert out.dtype == torch.float32
        rel = float((out.double() - ref).abs().max() / ref.max())
        assert rel < (1e-5 if method == "quadrature" else 2e-2)


# ----------------------------------------------------------------------
# Depth-induced aberration (Gibson & Lanni Figs. 4, 11, 13)
# ----------------------------------------------------------------------


def test_depth_trends():
    """Deeper sources in water: dimmer, longer, shifted, asymmetric."""
    dz = torch.arange(-12, 12.001, 0.02, dtype=F64)
    peaks, shifts, widths = [], [], []
    for depth in (0.0, 4.0, 10.0, 20.0, 60.0):
        axial = gibson_lanni_profile(
            0.0, dz, wavelength=LAMBDA, dtype=F64, depth=depth, **WATER
        )[:, 0]
        k = int(axial.argmax())
        above = (axial >= axial[k] / 2).nonzero().flatten()
        peaks.append(float(axial[k]))
        shifts.append(float(dz[k]))
        widths.append(float(dz[above.max()] - dz[above.min()]))
    assert peaks[0] == pytest.approx(1.0)
    assert all(b < a for a, b in zip(peaks, peaks[1:], strict=False))
    assert all(b > a for a, b in zip(widths, widths[1:], strict=False))
    # best focus moves from the paraxial focus towards a thinner
    # immersion layer (Δz < 0), more so for deeper sources
    assert shifts[0] == pytest.approx(0.0, abs=1e-12)
    assert all(b < a for a, b in zip(shifts, shifts[1:], strict=False))
    assert peaks[-1] < 0.15


# ----------------------------------------------------------------------
# Bessel series of Li et al. (2017)
# ----------------------------------------------------------------------


def test_li2017_matches_quadrature_within_paper_accuracy():
    """Li et al. Sec. 3A settings; RSE below the paper's estimate
    RSE(M, K) = (M/45)^−6.5 (K/80)^−1.5 % for M = 100, K = 1000."""
    kw = dict(spacing=(0.25, 0.1, 0.1), wavelength=0.61, dtype=F64, depth=2.0)
    shape = (64, 128, 128)
    t0 = time.perf_counter()
    quad = gibson_lanni(shape, **kw, **LI)
    t1 = time.perf_counter()
    series = gibson_lanni(shape, method="li2017", **kw, **LI)
    t2 = time.perf_counter()
    exact = gibson_lanni(shape, nodes=2048, **kw, **LI)
    rse = lambda a: float(((a - exact) ** 2).sum() / (exact**2).sum() * 100)  # noqa: E731
    estimate = (100 / 45) ** -6.5 * (1000 / 80) ** -1.5
    assert rse(quad) < 1e-12
    assert rse(series) < estimate
    print(
        f"\nGibson–Lanni {shape}: quadrature {t1 - t0:.3f} s, "
        f"li2017 {t2 - t1:.3f} s (×{(t1 - t0) / (t2 - t1):.1f}); "
        f"RSE li2017 {rse(series):.2e} % (paper estimate {estimate:.2e} %)"
    )


def test_li2017_error_decreases_with_basis():
    r = torch.linspace(0, 2, 41, dtype=F64)
    dz = torch.linspace(-3, 3, 31, dtype=F64)
    kw = dict(wavelength=0.61, dtype=F64, depth=2.0, **LI)
    ref = gibson_lanni_profile(r, dz, nodes=1024, **kw)
    errs = []
    for m in (30, 50, 100):
        out = gibson_lanni_profile(
            r, dz, method="li2017", basis=m, samples=1000, **kw
        )
        errs.append(float(((out - ref) ** 2).sum() / (ref**2).sum() * 100))
    assert errs[0] > errs[1] > errs[2]
    # within a factor 10 of the paper's fit, Eq. (6)
    for m, e in zip((30, 50, 100), errs, strict=True):
        assert e < 10 * (m / 45) ** -6.5 * (1000 / 80) ** -1.5


def test_matches_microscpsf():
    """MicroscPSF-Py 0.2 (MIT), Li et al.'s Bessel series, Δz grid."""
    data = np.load(FIXTURES)
    radius = torch.as_tensor(data["radius"])
    dz = torch.as_tensor(data["defocus"])
    for depth, ref in zip(data["depths"], data["intensity"], strict=True):
        ref = torch.as_tensor(ref) * 4
        kw = dict(wavelength=0.61, dtype=F64, depth=float(depth), **LI)
        series = gibson_lanni_profile(radius, dz, method="li2017", **kw)
        quad = gibson_lanni_profile(radius, dz, **kw)
        assert float((series - ref).abs().max() / ref.max()) < 1e-4
        rse = float(((quad - ref) ** 2).sum() / (ref**2).sum() * 100)
        assert rse < 5e-3


# ----------------------------------------------------------------------
# Sampling and API
# ----------------------------------------------------------------------


def test_oversampling_integrates_voxels():
    kw = dict(wavelength=LAMBDA, dtype=F64, depth=2.0, **WATER)
    psf = gibson_lanni(
        (3, 5, 5), spacing=(0.3, 0.12, 0.12), oversample=(3, 2, 2), **kw
    )
    # brute force: average of the 3·2·2 sub-voxel points of each voxel
    sub_z = (torch.arange(3, dtype=F64) - 1) * 0.3
    off_z = (torch.arange(3, dtype=F64) - 1) * 0.1
    off_xy = torch.tensor([-0.03, 0.03], dtype=F64)
    xy = (torch.arange(5, dtype=F64) - 2) * 0.12
    zz = (sub_z[:, None] + off_z[None]).flatten()
    yy = (xy[:, None] + off_xy[None]).flatten()
    r = torch.hypot(yy[:, None], yy[None, :])
    prof = gibson_lanni_profile(r.flatten(), zz, **kw).view(9, 10, 10)
    ref = prof.view(3, 3, 5, 2, 5, 2).mean((1, 3, 5))
    torch.testing.assert_close(psf, ref / ref.sum(), rtol=1e-12, atol=0)
    # an even size keeps the source at index K // 2
    even = gibson_lanni(
        (4, 6, 6), spacing=0.1, wavelength=LAMBDA, dtype=F64, **MATCHED
    )
    assert even[2, 3, 3] == even.max()


def test_defocus_shifts_slices():
    kw = dict(spacing=(0.2, 0.1, 0.1), wavelength=LAMBDA, dtype=F64, depth=5.0)
    a = gibson_lanni((9, 9, 9), defocus=0.4, **kw, **WATER)
    b = gibson_lanni((13, 9, 9), **kw, **WATER)
    torch.testing.assert_close(a / a.sum(), b[4:13] / b[4:13].sum())


def test_return_estimate_and_validation():
    est = gibson_lanni(
        (9, 15, 15),
        spacing=(0.25, 0.1),
        wavelength=LAMBDA,
        return_estimate=True,
        dtype=F64,
        **MATCHED,
    )
    assert isinstance(est, PSFEstimate)
    assert est.psf.shape == (9, 15, 15)
    assert est.meta["model"] == "gibson_lanni"
    assert est.meta["nodes"] >= 32
    assert est.fwhm.shape == (3,)
    # lateral FWHM of the Airy disc, 0.5145 λ/NA
    assert float(est.fwhm[1]) * 0.1 == pytest.approx(
        0.5145 * LAMBDA / NA, rel=0.05
    )
    with pytest.raises(ValueError, match="method"):
        gibson_lanni(
            (3, 3, 3), spacing=0.1, wavelength=LAMBDA, method="x", **MATCHED
        )
    with pytest.raises(ValueError, match="shape"):
        gibson_lanni((3, 3), spacing=0.1, wavelength=LAMBDA, **MATCHED)
    with pytest.raises(TypeError, match="Unknown"):
        gibson_lanni_opd(0.5, numerical_aperture=1.0, refractive_index=1.0)


def test_device(device):
    kw = dict(spacing=(0.25, 0.1, 0.1), wavelength=LAMBDA, depth=4.0)
    out = gibson_lanni(
        (5, 9, 9), device=device, dtype=torch.float32, **kw, **WATER
    )
    assert out.device.type == device.type
    ref = gibson_lanni((5, 9, 9), dtype=torch.float32, **kw, **WATER)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-7)


def test_water_dipping_objective():
    """Light-sheet detection: water-dipping NA 1.0, no coverslip."""
    dip = dict(
        numerical_aperture=1.0,
        design_immersion_index=1.333,
        design_coverslip_thickness=0.0,
        design_coverslip_index=1.333,
    )
    r = torch.linspace(0, 2, 41, dtype=F64)
    kw = dict(wavelength=0.52, dtype=F64)
    matched = gibson_lanni_profile(
        r, 0.0, specimen_index=1.333, depth=200.0, **kw, **dip
    )[0]
    torch.testing.assert_close(matched, airy(r, 0.52, 1.0), rtol=0, atol=1e-12)
    # tissue of index 1.38 at 200 µm: dimmer, with the best focus moved
    dz = torch.arange(-6, 6.001, 0.02, dtype=F64)
    axial = gibson_lanni_profile(
        0.0, dz, specimen_index=1.38, depth=200.0, **kw, **dip
    )[:, 0]
    assert float(axial.max()) < 0.9
    assert abs(float(dz[axial.argmax()])) > 0.1
    psf = gibson_lanni(
        (16, 32, 32),
        spacing=(0.5, 0.2, 0.2),
        specimen_index=1.38,
        depth=200.0,
        **kw,
        **dip,
    )
    assert torch.isfinite(psf).all() and float(psf.sum()) == pytest.approx(1)
