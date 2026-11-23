"""Tests of pupil functions, their phase retrieval and Zernike fit."""

import math

import numpy as np
import pytest
import torch

from optix.estimates import Estimate, PSFEstimate, PupilEstimate
from optix.psf import (
    PhaseRetrieval,
    index_mismatch_aberration,
    phase_retrieval,
    pupil_psf,
    zernike_psf,
)
from optix.psf.utils import (
    bead_otf,
    fit_otf_rescaling,
    noll_index,
    pupil_coordinates,
    unwrap_phase,
    vectorial_factors,
    zernike,
    zernike_fit,
    zernike_pupil,
)

F64 = torch.float64
LAM, NA, N = 0.6, 1.3, 1.515
SHAPE, SPACING = (25, 128, 128), (0.25, 0.1, 0.1)
FOCUS = SHAPE[0] // 2


def optics(**extra):
    return dict(wavelength=LAM, na=NA, n=N, **extra)


def truth(dtype=F64):
    """Magnitude and phase Zernike coefficients of the test pupil."""
    phase = torch.zeros(15, dtype=dtype)
    phase[4], phase[6], phase[8], phase[10] = 0.4, -0.3, 0.2, 0.35
    mag = torch.zeros(4, dtype=dtype)
    mag[0], mag[3] = 1.0, 0.1
    return mag, phase


def synthetic(photons=None, seed=0, dtype=F64):
    mag, phase = truth(dtype)
    psf = zernike_psf(mag, phase, shape=SHAPE, spacing=SPACING, **optics())
    if photons is None:
        return psf
    gen = torch.Generator().manual_seed(seed)
    return torch.poisson(psf * photons, generator=gen)


# -- Zernike polynomials ---------------------------------------------------


def test_noll_index_table():
    expected = [
        (0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1),
        (3, 1), (3, -3), (3, 3), (4, 0), (4, 2), (4, -2), (4, 4),
        (4, -4), (5, 1), (5, -1),
    ]  # fmt: skip
    assert [noll_index(j) for j in range(1, 18)] == expected
    # 79 terms = all orders up to 11 plus the radial one of order 12
    assert noll_index(77) == (11, -11)
    assert noll_index(78) == (11, 11)
    assert noll_index(79) == (12, 0)
    with pytest.raises(ValueError):
        noll_index(0)


def _radial_explicit(n, m, rho):
    out = torch.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        c = (-1) ** s * math.factorial(n - s)
        c /= (
            math.factorial(s)
            * math.factorial((n + m) // 2 - s)
            * math.factorial((n - m) // 2 - s)
        )
        out = out + c * rho ** (n - 2 * s)
    return out


def test_zernike_matches_explicit_formula():
    rho = torch.linspace(0, 1, 101, dtype=F64)
    theta = torch.linspace(-math.pi, math.pi, 101, dtype=F64)
    js = list(range(1, 80))
    values = zernike(js, rho, theta)
    for k, j in enumerate(js):
        n, m = noll_index(j)
        r = _radial_explicit(n, abs(m), rho)
        if m == 0:
            ref = math.sqrt(n + 1) * r
        elif m > 0:
            ref = math.sqrt(2 * (n + 1)) * r * torch.cos(m * theta)
        else:
            ref = math.sqrt(2 * (n + 1)) * r * torch.sin(-m * theta)
        torch.testing.assert_close(values[:, k], ref, atol=1e-9, rtol=0)


def test_zernike_float32_is_stable():
    rho = torch.linspace(0, 1, 201, dtype=F64)
    theta = torch.zeros_like(rho)
    js = list(range(1, 92))
    ref = zernike(js, rho, theta)
    low = zernike(js, rho.float(), theta.float()).double()
    assert (low - ref).abs().max() < 2e-5


def test_zernike_orthonormal_under_noll_normalisation():
    """(1/π) ∫∫ Z_i Z_j ρ dρ dθ = δ_ij, by exact quadrature."""
    x, w = np.polynomial.legendre.leggauss(40)
    rho = torch.tensor((x + 1) / 2, dtype=F64)
    wr = torch.tensor(w / 2, dtype=F64) * rho
    m = 64
    theta = torch.arange(m, dtype=F64) * 2 * math.pi / m
    z = zernike(list(range(1, 80)), rho[:, None], theta[None, :])
    weights = (wr[:, None] * (2 * math.pi / m)).expand(-1, m)
    gram = torch.einsum("ab,abi,abj->ij", weights, z, z) / math.pi
    torch.testing.assert_close(
        gram, torch.eye(79, dtype=F64), atol=1e-12, rtol=0
    )


def test_zernike_fit_recovers_coefficients():
    rho, theta = pupil_coordinates(
        (96, 96), (0.1, 0.1), wavelength=LAM, na=NA, dtype=F64
    )
    coef = torch.randn(
        2, 21, generator=torch.Generator().manual_seed(1), dtype=F64
    )
    values = torch.einsum(
        "bj,yxj->byx", coef, zernike(list(range(1, 22)), rho, theta)
    )
    fit = zernike_fit(values, rho, theta, terms=21)
    torch.testing.assert_close(fit, coef, atol=1e-10, rtol=0)
    with pytest.raises(ValueError, match="pupil pixels"):
        zernike_fit(
            values[:, 40:56, 40:56],
            rho[40:56, 40:56] * 0.01,
            theta[40:56, 40:56],
            terms=400,
        )


def test_unwrap_phase_recovers_smooth_phase():
    rho, theta = pupil_coordinates(
        (64, 64), (0.1, 0.1), wavelength=LAM, na=NA, dtype=F64
    )
    support = rho <= 1
    phase = 6.0 * zernike(4, rho, theta) + 4.0 * zernike(8, rho, theta)
    phase = (phase - phase[32, 32]) * support
    assert phase.abs().max() > 3 * math.pi
    wrapped = torch.remainder(phase + math.pi, 2 * math.pi) - math.pi
    out = unwrap_phase(torch.stack([wrapped, wrapped]), support)
    torch.testing.assert_close(out[1], phase, atol=1e-9, rtol=0)
    assert (out[0][~support] == 0).all()


# -- imaging model ---------------------------------------------------------


def test_pupil_coordinates_centre_and_edge():
    rho, theta = pupil_coordinates(
        (64, 80), (0.1, 0.12), wavelength=LAM, na=NA, dtype=F64
    )
    assert rho[32, 40] == 0
    dk = 1 / (80 * 0.12)
    torch.testing.assert_close(
        rho[32, 41], torch.tensor(dk * LAM / NA, dtype=F64)
    )
    assert theta[32, 41] == 0
    with pytest.raises(ValueError):
        pupil_coordinates((8, 8), (0.1, 0.1), wavelength=-1, na=NA)


def test_aberration_free_focus_is_airy():
    """In focus, |FT of the disk|² is (2 J₁(v)/v)², v = 2π NA r / λ."""
    nx, dx = 256, 0.05
    pupil = zernike_pupil(
        None,
        None,
        shape=(nx, nx),
        spacing=(dx, dx),
        wavelength=LAM,
        na=NA,
        dtype=F64,
    )
    psf = pupil_psf(pupil, 0.0, spacing=(dx, dx), **optics(), normalize=False)
    psf = psf[0] / psf[0].max()
    r = torch.arange(0, 40, dtype=F64) * dx
    v = 2 * math.pi * NA * r / LAM
    vs = torch.where(v > 0, v, 1)
    airy = torch.where(v > 0, 2 * torch.special.bessel_j1(vs) / vs, 1) ** 2
    profile = psf[nx // 2, nx // 2 : nx // 2 + 40]
    assert (profile - airy).abs().max() < 5e-3
    # the first dark ring at 0.61 λ / NA
    first_zero = int(torch.argmin(profile[:9]))
    assert abs(first_zero * dx - 0.6098 * LAM / NA) <= dx


def test_aberration_free_defocus():
    """On-axis intensity follows ∫ e^{2πi k_z z} dk; I(z) = I(−z)."""
    nx, dx = 256, 0.05
    pupil = zernike_pupil(
        None,
        None,
        shape=(nx, nx),
        spacing=(dx, dx),
        wavelength=LAM,
        na=NA,
        dtype=F64,
    )
    z = torch.linspace(-2, 2, 41, dtype=F64)
    psf = pupil_psf(pupil, z, spacing=(dx, dx), **optics(), normalize=False)
    torch.testing.assert_close(psf, psf.flip(0), atol=1e-12, rtol=1e-9)
    axial = psf[:, nx // 2, nx // 2]
    axial = axial / axial[20]
    big, cut = N / LAM, NA / LAM
    u = torch.linspace(math.sqrt(big**2 - cut**2), big, 20001, dtype=F64)
    phase = 2 * math.pi * (u[None] - big) * z[:, None]
    amp = torch.trapezoid(u * torch.exp(1j * phase), u, dim=-1)
    ref = amp.abs().square()
    ref = ref / ref[20]
    assert (axial - ref).abs().max() < 1e-2
    # paraxial first axial zero, z = λ / (n − √(n² − NA²))
    assert axial[25:].min() < 0.05


def test_pupil_psf_is_unit_sum_and_centred():
    mag, phase = truth()
    psf = zernike_psf(
        mag,
        torch.zeros(1, dtype=F64),
        shape=(5, 64, 64),
        spacing=(0.25, 0.1, 0.1),
        **optics(),
    )
    torch.testing.assert_close(psf.sum(), torch.tensor(1.0, dtype=F64))
    assert divmod(int(psf[2].argmax()), 64) == (32, 32)


def test_vectorial_factors_sum_of_squares():
    f = vectorial_factors((64, 64), (0.1, 0.1), **optics(), dtype=F64)
    rho, _ = pupil_coordinates((64, 64), (0.1, 0.1), wavelength=LAM, na=NA)
    total = f.square().sum(0)
    support = rho <= 1
    torch.testing.assert_close(
        total[support], torch.full_like(total[support], 2.0)
    )
    assert (total[~support] == 0).all()
    # at small NA the vectorial PSF is the scalar one
    pupil = zernike_pupil(
        None,
        None,
        shape=(64, 64),
        spacing=(0.1, 0.1),
        wavelength=LAM,
        na=0.2,
        dtype=F64,
    )
    kw = dict(spacing=(0.1, 0.1), wavelength=LAM, na=0.2, n=N)
    scalar = pupil_psf(pupil, [0.0, 1.0], **kw)
    vector = pupil_psf(pupil, [0.0, 1.0], vectorial=True, **kw)
    assert (scalar - vector).abs().max() < 1e-2 * scalar.max()


def test_bead_otf():
    k = torch.tensor([0.0, 1e-4, 0.5, 1.0, math.pi], dtype=F64)
    b = bead_otf(k / math.pi, 1.0)  # x = k
    assert b[0] == 1
    closed = 3 * (
        torch.sin(k[1:]) / k[1:] ** 3 - torch.cos(k[1:]) / k[1:] ** 2
    )
    torch.testing.assert_close(b[2:], closed[1:], atol=1e-12, rtol=0)
    torch.testing.assert_close(b[-1], torch.tensor(3 / math.pi**2, dtype=F64))
    # series and closed form join continuously at x = 1
    x = torch.tensor([1 - 1e-9, 1 + 1e-9], dtype=F64) / math.pi
    assert abs(float(bead_otf(x, 1.0).diff())) < 1e-8
    # single precision is accurate near 0 (no cancellation)
    small = torch.logspace(-4, 0, 50, dtype=F64)
    err = bead_otf(small.float(), 1.0).double() - bead_otf(small, 1.0)
    assert err.abs().max() < 1e-6


def test_otf_corrections_and_rescaling_fit():
    mag, phase = truth()
    kw = dict(shape=(32, 64, 64), spacing=(0.2, 0.1, 0.1), **optics())
    plain = zernike_psf(mag, phase, **kw)
    bead = zernike_psf(mag, phase, bead_diameter=0.2, **kw)
    torch.testing.assert_close(bead.sum(), torch.tensor(1.0, dtype=F64))
    assert bead.min() > -1e-12
    assert bead.max() < plain.max()  # convolution with the bead blurs
    sigma = 2.5
    measured = zernike_psf(mag, phase, otf_sigma=sigma, **kw)
    fitted = fit_otf_rescaling(
        measured,
        plain,
        spacing=(0.2, 0.1, 0.1),
        wavelength=LAM,
        na=1.15,
        n=N,
    )
    assert abs(fitted - sigma) < 1e-3 * sigma
    assert math.isinf(
        fit_otf_rescaling(
            plain,
            plain,
            spacing=(0.2, 0.1, 0.1),
            wavelength=LAM,
            na=1.15,
            n=N,
        )
    )


def test_index_mismatch_aberration():
    grid = dict(shape=(64, 64), spacing=(0.1, 0.1), wavelength=LAM, na=NA)
    rho, _ = pupil_coordinates(**grid)
    support = rho <= 1
    psi = index_mismatch_aberration(**grid, depth=10.0, n1=N, n2=N, dtype=F64)
    torch.testing.assert_close(psi, support.to(psi.dtype))
    # Eqs. 4-8 as printed, off axis
    n1, n2, d = 1.515, 1.47, 12.0
    psi = index_mismatch_aberration(**grid, depth=d, n1=n1, n2=n2, dtype=F64)
    kx = (torch.arange(64, dtype=F64) - 32) / 6.4
    for i in (33, 40, 45):
        t1 = math.asin(LAM * float(kx[i]) / n1)
        t2 = math.asin(n1 * math.sin(t1) / n2)
        a_t = (math.sin(t1) * math.cos(t2) / math.sin(t1 + t2)) * (
            1 + 1 / math.cos(t2 - t1)
        )
        a_w = n1 * math.tan(t2) / (n2 * math.tan(t1))
        op = d * (n2 * math.cos(t2) - n1 * math.cos(t1))
        ref = (
            a_t
            * a_w
            * complex(
                math.cos(2 * math.pi * op / LAM),
                math.sin(2 * math.pi * op / LAM),
            )
        )
        assert abs(complex(psi[32, i]) - ref) < 1e-10
    # on axis: normal-incidence transmittance and (n1/n2)² compression
    expected = 2 * n2 / (n1 + n2) * (n1 / n2) ** 2
    assert abs(float(psi[32, 32].abs()) - expected) < 1e-12
    # beyond the critical angle (water) the rays are dropped
    psi = index_mismatch_aberration(**grid, depth=d, n1=n1, n2=1.2)
    s1 = rho * NA / n1
    assert (psi[support & (n1 * s1 > 1.2)] == 0).all()
    assert (psi[support & (n1 * s1 < 1.19)] != 0).all()


# -- phase retrieval -------------------------------------------------------


def _manual_iteration(stack, focus, sections, dz, dx):
    """Steps 1-6 of Fig. 2, written directly from Eq. 2 (piston kept)."""
    ny, nx = stack.shape[-2:]
    k = (torch.arange(nx, dtype=F64) - nx // 2) / (nx * dx)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    support = k2 <= (NA / LAM) ** 2
    kz = torch.sqrt(((N / LAM) ** 2 - k2).clamp(min=0))
    meas = stack[sections].clamp(min=0)
    meas = meas * support.sum() / meas.sum((-2, -1)).mean()
    pupil = support.to(torch.complex128)  # step 1
    total = 0
    for i, s in enumerate(sections):
        defocus = torch.exp(2j * math.pi * kz * (s - focus) * dz)
        shifted = torch.fft.ifftshift(pupil * defocus)  # step 2
        field = torch.fft.fftshift(torch.fft.ifft2(shifted, norm="ortho"))
        field = meas[i].sqrt() * torch.exp(1j * field.angle())  # step 3
        back = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(field), norm="ortho")
        )
        total = total + back / defocus  # step 4
    pupil = total / len(sections)  # step 5
    pupil = pupil * support  # step 6
    s = pupil.sum()
    return pupil * s.conj() / s.abs()


def test_one_iteration_follows_the_paper_steps():
    data = synthetic()
    est = PhaseRetrieval(
        spacing=SPACING, **optics(), focus=FOCUS, iterations=1, dtype=F64
    ).fit(data)
    sections = est.estimates["pupil"].meta["sections"][0]
    assert sections == [FOCUS - 12, FOCUS - 4, FOCUS + 4, FOCUS + 12]
    ref = _manual_iteration(data, FOCUS, sections, SPACING[0], SPACING[2])
    torch.testing.assert_close(
        est.estimates["pupil"].pupil, ref, atol=1e-10, rtol=0
    )


def test_starting_guess_is_unit_disk():
    data = synthetic()
    est = PhaseRetrieval(spacing=SPACING, **optics(), iterations=0).fit(data)
    pupil = est.estimates["pupil"].pupil
    rho, _ = pupil_coordinates(SHAPE[1:], SPACING[1:], wavelength=LAM, na=NA)
    torch.testing.assert_close(pupil, (rho <= 1).to(pupil.dtype))
    assert len(est.history.objective) == 1


def test_default_focus_is_the_brightest_section():
    data = synthetic()
    shifted = torch.roll(data, 2, 0)
    est = PhaseRetrieval(
        spacing=SPACING, **optics(), offsets=(-1.0, 1.0), iterations=0
    )
    est.fit(shifted)
    meta = est.estimates["pupil"].meta
    assert meta["focus"] == [int(shifted.flatten(1).amax(1).argmax())]
    with pytest.raises(ValueError, match="outside the stack"):
        PhaseRetrieval(spacing=SPACING, **optics(), offsets=(-3.0, 3.0)).fit(
            shifted
        )


@pytest.mark.parametrize("photons", [None, 2.5e7])
def test_retrieval_recovers_zernike_coefficients(photons):
    data = synthetic(photons)
    est = PhaseRetrieval(
        spacing=SPACING, **optics(), focus=FOCUS, iterations=200, dtype=F64
    ).fit(data)
    e = est.estimates["pupil"]
    mag, phase = truth()
    tol = 2e-3 if photons is None else 1e-2
    got = e.zernike_phase[1:15]
    assert (got - phase[1:15]).abs().max() < tol
    rel = e.zernike_magnitude / e.zernike_magnitude[0]
    assert (rel[1:4] - mag[1:4]).abs().max() < 2 * tol
    assert rel[4:].abs().max() < 2 * tol
    assert e.zernike_phase.shape == (79,)
    assert e.meta["correlation"][0] > 0.999


def test_error_metrics_decrease():
    data = synthetic(2.5e7)
    est = PhaseRetrieval(
        spacing=SPACING, **optics(), focus=FOCUS, dtype=F64
    ).fit(data)
    h = est.history
    assert len(h.objective) == 26
    amp = h.extras["amplitude_error"]
    assert all(
        b <= a * (1 + 1e-9) for a, b in zip(amp[:-1], amp[1:], strict=True)
    )
    imse = h.extras["imse"]
    assert imse[-1] < 0.05 * imse[0]
    assert h.extras["correlation"][-1] > h.extras["correlation"][0]
    torch.testing.assert_close(-torch.tensor(h.objective), torch.tensor(imse))


def test_retrieved_pupil_layout():
    data = synthetic()
    e = phase_retrieval(data, spacing=SPACING, **optics(), focus=FOCUS)
    assert isinstance(e, PupilEstimate)
    rho, _ = pupil_coordinates(SHAPE[1:], SPACING[1:], wavelength=LAM, na=NA)
    assert (e.pupil[rho > 1] == 0).all()
    total = e.pupil.sum()
    assert abs(float(total.imag)) < 1e-5 * float(total.abs())
    assert float(total.real) > 0
    assert e.meta["zernike"] == "noll"
    assert e.meta["spacing"] == list(SPACING)


def test_float32_matches_float64():
    data = synthetic(2.5e7)
    kw = dict(spacing=SPACING, **optics(), focus=FOCUS, iterations=50)
    lo = PhaseRetrieval(**kw, dtype=torch.float32).fit(data.float())
    hi = PhaseRetrieval(**kw, dtype=F64).fit(data)
    a = lo.estimates["pupil"].zernike_phase.double()
    b = hi.estimates["pupil"].zernike_phase
    assert (a - b).abs().max() < 1e-3


def test_inputs_estimate_groups_and_layout():
    data = synthetic()
    kw = dict(spacing=SPACING, **optics(), focus=FOCUS, iterations=5)
    ref = PhaseRetrieval(**kw, dtype=F64).fit(data).estimates["pupil"]
    from_estimate = PhaseRetrieval(**kw, dtype=F64).fit(PSFEstimate(psf=data))
    torch.testing.assert_close(
        from_estimate.estimates["pupil"].pupil, ref.pupil
    )

    groups = PhaseRetrieval(**kw, dtype=F64).fit(torch.stack([data, 2 * data]))
    e = groups.estimates["pupil"]
    assert e.pupil.shape == (2, *SHAPE[1:])
    assert e.zernike_phase.shape == (2, 79)
    torch.testing.assert_close(e.pupil[1], ref.pupil)  # scale-free
    from optix.io import Layout, labeled

    xzy = labeled(data.permute(2, 0, 1), Layout("xzy", spatial="zyx"))
    lab = PhaseRetrieval(**kw, dtype=F64).fit(xzy).estimates["pupil"]
    torch.testing.assert_close(lab.pupil, ref.pupil)
    with pytest.raises(ValueError, match="sample axes"):
        PhaseRetrieval(**kw).fit(
            labeled(torch.stack([data, data]), Layout("tzyx", spatial="zyx"))
        )


def test_bead_psf_estimator_input():
    from optix.psf import BeadPSF

    data = synthetic()
    est = BeadPSF()
    est.estimates = {"psf": PSFEstimate(psf=data)}
    kw = dict(spacing=SPACING, **optics(), focus=FOCUS, iterations=3)
    a = PhaseRetrieval(**kw, dtype=F64).fit(est).estimates["pupil"].pupil
    b = PhaseRetrieval(**kw, dtype=F64).fit(data).estimates["pupil"].pupil
    torch.testing.assert_close(a, b)


def test_options_presets_and_validation():
    est = PhaseRetrieval(spacing=SPACING, **optics())
    assert est.is_faithful
    assert est.options["iterations"] == 25
    assert est.options["offsets"] == (-3.0, -1.0, 1.0, 3.0)
    assert est.options["zernike_terms"] == 79
    smooth = PhaseRetrieval(spacing=SPACING, **optics(), smoothing=1.0)
    assert not smooth.is_faithful
    e = smooth.fit(synthetic()).estimates["pupil"]
    assert torch.isfinite(e.zernike_phase).all()
    with pytest.raises(ValueError, match="needs `wavelength`"):
        PhaseRetrieval(spacing=SPACING, na=NA, n=N)
    with pytest.raises(ValueError):
        PhaseRetrieval(spacing=SPACING, wavelength=LAM, na=1.6, n=N)
    with pytest.raises(NotImplementedError):
        est.apply(synthetic())


def test_pupil_estimate_save_load(tmp_path):
    e = phase_retrieval(synthetic(), spacing=SPACING, **optics(), iterations=2)
    path = tmp_path / "pupil.pt"
    e.save(path)
    back = Estimate.load(path)
    assert isinstance(back, PupilEstimate)
    assert back.pupil.is_complex()
    assert back.allclose(e)
    assert back.meta["wavelength"] == LAM


def test_truncated_zernike_psf_on_another_grid():
    """Coefficients render a PSF on any grid; truncation keeps it close."""
    data = synthetic()
    e = phase_retrieval(
        data,
        spacing=SPACING,
        **optics(),
        focus=FOCUS,
        iterations=200,
        dtype=F64,
    )
    mag, phase = truth()
    grid = dict(shape=(9, 80, 96), spacing=(0.5, 0.08, 0.07), **optics())
    ref = zernike_psf(mag, phase, **grid)
    full = zernike_psf(e.zernike_magnitude, e.zernike_phase, **grid)
    short = zernike_psf(e.zernike_magnitude[:15], e.zernike_phase[:15], **grid)
    assert ref.shape == (9, 80, 96)
    for psf in (full, short):
        assert (psf - ref).abs().max() < 2e-2 * ref.max()
