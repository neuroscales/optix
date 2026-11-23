"""Scalar pupil functions and their phase retrieval (Hanser et al. 2004)."""

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from .._estimator import Estimator, Tags
from ..estimates import PSFEstimate, PupilEstimate
from ..io import (
    ArrayLike,
    Labeled,
    WritableArrayLike,
    as_tensor,
    chunk_length,
)
from ._ops import gaussian_filter
from ._zernike import unwrap_phase, zernike, zernike_fit

__all__ = [
    "PhaseRetrieval",
    "bead_otf",
    "fit_otf_rescaling",
    "index_mismatch_aberration",
    "phase_retrieval",
    "pupil_coordinates",
    "pupil_psf",
    "vectorial_factors",
    "zernike_pupil",
    "zernike_psf",
]

_DOI = "10.1111/j.0022-2720.2004.01393.x"
# frequencies where the calculated OTF is below this fraction of its value
# at k = 0 are left out of the rescaling fit (the ratio is noise there)
_OTF_FLOOR = 1e-3


# -- grids and transforms ------------------------------------------------


def _check_optics(wavelength: float, na: float, n: float | None) -> None:
    if not wavelength > 0:
        raise ValueError(f"wavelength must be positive, got {wavelength}")
    if not na > 0:
        raise ValueError(f"numerical_aperture must be positive, got {na}")
    if n is not None and not na < n:
        raise ValueError(
            f"numerical_aperture ({na}) must be smaller than the "
            f"refractive index ({n})"
        )


class _Grid:
    """Pupil-plane frequencies conjugate to a lateral image grid.

    k = 0 sits at index `K // 2` (the FFT-shifted layout); the frequency
    step is 1 / (K · spacing).
    """

    def __init__(
        self,
        shape: Sequence[int],
        spacing: Sequence[float],
        wavelength: float,
        na: float,
        n: float | None,
        *,
        dtype: torch.dtype,
        device: torch.device | None,
    ) -> None:
        _check_optics(wavelength, na, n)
        (ny, nx), (dy, dx) = shape, spacing
        ky = (torch.arange(ny, dtype=dtype, device=device) - ny // 2) / (
            ny * dy
        )
        kx = (torch.arange(nx, dtype=dtype, device=device) - nx // 2) / (
            nx * dx
        )
        self.ky, self.kx = ky[:, None], kx[None, :]
        self.k2 = self.ky.square() + self.kx.square()
        self.cutoff = na / wavelength
        self.support = self.k2 <= self.cutoff**2
        self.rho = self.k2.sqrt() / self.cutoff
        self.theta = torch.atan2(self.ky, self.kx).expand_as(self.k2)
        self.wavelength, self.na, self.n = wavelength, na, n

    def defocus(self, z: Tensor) -> Tensor:
        """exp(2πi (k_z − n/λ) z) on the grid, `(*z.shape, Ky, Kx)`.

        k_z − n/λ = −k² / (n/λ + k_z): the piston n z / λ is dropped
        (it cancels between the forward and the inverse defocus) and the
        difference is computed without cancellation.
        """
        big = self.n / self.wavelength
        kz = (big**2 - self.k2).clamp(min=0).sqrt()
        shift = torch.where(self.support, -self.k2 / (big + kz), 0)
        angle = 2 * math.pi * z[..., None, None] * shift
        return torch.polar(torch.ones_like(angle), angle)


def _fft2c(x: Tensor) -> Tensor:
    dims = (-2, -1)
    y = torch.fft.fft2(torch.fft.ifftshift(x, dim=dims), norm="ortho")
    return torch.fft.fftshift(y, dim=dims)


def _ifft2c(x: Tensor) -> Tensor:
    dims = (-2, -1)
    y = torch.fft.ifft2(torch.fft.ifftshift(x, dim=dims), norm="ortho")
    return torch.fft.fftshift(y, dim=dims)


def _complex(x: Tensor) -> Tensor:
    if x.is_complex():
        return x
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    return x.to(
        torch.complex128 if x.dtype == torch.float64 else torch.complex64
    )


def pupil_coordinates(
    shape: Sequence[int],
    spacing: Sequence[float],
    *,
    wavelength: float,
    numerical_aperture: float,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Normalised polar coordinates of the pupil grid of an image grid.

    The pupil grid is the discrete Fourier grid of the lateral image
    grid: k = (i − K // 2) / (K · Δ) along each axis, so that k = 0 is at
    index `K // 2`. The normalised radius is ρ = |k| / (NA/λ), 1 on the
    pupil edge, and θ = atan2(k_y, k_x).

    Parameters
    ----------
    shape : (int, int)
        Lateral image shape `(Y, X)`.
    spacing : (float, float)
        Lateral pixel size `(Δy, Δx)`, in the units of `wavelength`.
    wavelength : float
        Vacuum wavelength λ of the detected light.
    numerical_aperture : float
        Numerical aperture.
    dtype, device : optional
        Floating dtype and device (default: torch's default dtype, CPU).

    Returns
    -------
    rho, theta : Tensor
        `(Y, X)` maps.
    """
    grid = _Grid(
        shape,
        spacing,
        wavelength,
        numerical_aperture,
        None,
        dtype=dtype or torch.get_default_dtype(),
        device=None if device is None else torch.device(device),
    )
    return grid.rho, grid.theta


def vectorial_factors(
    shape: Sequence[int],
    spacing: Sequence[float],
    *,
    wavelength: float,
    numerical_aperture: float,
    refractive_index: float,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Pupil factors of the six dipole-to-polarisation components.

    With sin θ = λ|k|/n (sine condition) and φ the azimuth of k
    (Hanser et al. 2004, Eq. 3):

        Pₓ → Eₓ: cos θ cos² φ + sin² φ
        Pₓ → E_y: (cos θ − 1) sin φ cos φ
        P_y → Eₓ: (cos θ − 1) sin φ cos φ
        P_y → E_y: cos θ sin² φ + cos² φ
        P_z → Eₓ: sin θ cos φ
        P_z → E_y: sin θ sin φ

    The sum of their squares is 2 everywhere.

    !!! references "References"
        1. Hanser BM, Gustafsson MGL, Agard DA, Sedat JW.
           *Phase-retrieved pupil functions in wide-field fluorescence
           microscopy.* J Microsc **216**(1):32–48, 2004.
           [doi:10.1111/j.0022-2720.2004.01393.x](https://doi.org/10.1111/j.0022-2720.2004.01393.x)

        ??? quote "BibTeX"
            ```bibtex
            @article{hanser2004phase,
              author  = {Hanser, B. M. and Gustafsson, M. G. L. and
                         Agard, D. A. and Sedat, J. W.},
              title   = {Phase-retrieved pupil functions in wide-field
                         fluorescence microscopy},
              journal = {Journal of Microscopy},
              volume  = {216}, number = {1}, pages = {32--48},
              year    = {2004}, doi = {10.1111/j.0022-2720.2004.01393.x}
            }
            ```

    Parameters
    ----------
    shape, spacing : (int, int), (float, float)
        Lateral image grid (see
        [`pupil_coordinates`][optix.psf.utils.pupil_coordinates]).
    wavelength, numerical_aperture, refractive_index : float
        Vacuum wavelength, numerical aperture and immersion refractive
        index.
    dtype, device : optional

    Returns
    -------
    Tensor
        `(6, Y, X)`, in the order above, zero outside the pupil.
    """
    grid = _Grid(
        shape,
        spacing,
        wavelength,
        numerical_aperture,
        refractive_index,
        dtype=dtype or torch.get_default_dtype(),
        device=None if device is None else torch.device(device),
    )
    return _vectorial(grid)


def _vectorial(grid: _Grid) -> Tensor:
    sin_t = (grid.k2.sqrt() * grid.wavelength / grid.n).clamp(max=1)
    cos_t = (1 - sin_t.square()).sqrt()
    k = grid.k2.sqrt()
    safe = torch.where(k > 0, k, 1)
    cos_p = torch.where(k > 0, grid.kx / safe, 1)
    sin_p = torch.where(k > 0, grid.ky / safe, 0)
    cross = (cos_t - 1) * sin_p * cos_p
    out = torch.stack(
        [
            cos_t * cos_p.square() + sin_p.square(),
            cross,
            cross,
            cos_t * sin_p.square() + cos_p.square(),
            sin_t * cos_p,
            sin_t * sin_p,
        ]
    )
    return out * grid.support


def bead_otf(k: Tensor, diameter: float) -> Tensor:
    """Fourier transform of a uniformly fluorescent spherical bead.

        b(k) = 3 h(π k d),   h(x) = sin(x)/x³ − cos(x)/x²,

    with k the magnitude of the 3D spatial frequency (cycles per unit
    length) and d the bead diameter (Hanser et al. 2004); b(0) = 1. For
    x < 1 the power series 3h(x) = ∑ₖ 3(−1)ᵏ(2k + 2) x²ᵏ / (2k + 3)! is
    used instead of the closed form, which cancels catastrophically near
    0.

    !!! references "References"
        1. Hanser BM, Gustafsson MGL, Agard DA, Sedat JW.
           *Phase-retrieved pupil functions in wide-field fluorescence
           microscopy.* J Microsc **216**(1):32–48, 2004.
           [doi:10.1111/j.0022-2720.2004.01393.x](https://doi.org/10.1111/j.0022-2720.2004.01393.x)

        ??? quote "BibTeX"
            ```bibtex
            @article{hanser2004phase,
              author  = {Hanser, B. M. and Gustafsson, M. G. L. and
                         Agard, D. A. and Sedat, J. W.},
              title   = {Phase-retrieved pupil functions in wide-field
                         fluorescence microscopy},
              journal = {Journal of Microscopy},
              volume  = {216}, number = {1}, pages = {32--48},
              year    = {2004}, doi = {10.1111/j.0022-2720.2004.01393.x}
            }
            ```

    Parameters
    ----------
    k : Tensor
        Spatial frequency magnitudes, ≥ 0.
    diameter : float
        Bead diameter d, in the inverse units of `k`.

    Returns
    -------
    Tensor
        b(k), same shape.
    """
    x = math.pi * diameter * k
    small = x < 1
    xs = torch.where(small, x, 1)
    x2 = xs.square()
    series = torch.zeros_like(x)
    power = torch.ones_like(x)
    for i in range(9):
        series = (
            series
            + 3 * (-1) ** i * (2 * i + 2) / math.factorial(2 * i + 3) * power
        )
        power = power * x2
    xl = torch.where(small, 1, x)
    closed = 3 * (torch.sin(xl) / xl**3 - torch.cos(xl) / xl**2)
    return torch.where(small, series, closed)


def _correct_otf(
    psf: Tensor,
    spacing: Sequence[float],
    bead_diameter: float | None,
    otf_sigma: float | None,
) -> Tensor:
    """Multiply the 3D OTF of `(..., Z, Y, X)` PSFs by b(k) and a Gaussian."""
    if bead_diameter is None and (otf_sigma is None or math.isinf(otf_sigma)):
        return psf
    shape = psf.shape[-3:]
    freqs = [
        torch.fft.fftfreq(s, d, dtype=psf.dtype, device=psf.device)
        for s, d in zip(shape, spacing, strict=True)
    ]
    k2 = (
        freqs[0][:, None, None].square()
        + freqs[1][None, :, None].square()
        + freqs[2][None, None, :].square()
    )
    factor = torch.ones_like(k2)
    if bead_diameter is not None:
        factor = factor * bead_otf(k2.sqrt(), float(bead_diameter))
    if otf_sigma is not None and not math.isinf(otf_sigma):
        factor = factor * torch.exp(-0.5 * k2 / float(otf_sigma) ** 2)
    otf = torch.fft.fftn(psf, dim=(-3, -2, -1))
    return torch.fft.ifftn(otf * factor, dim=(-3, -2, -1)).real


def _uniform_step(z: Tensor) -> float:
    if z.numel() < 2:
        raise ValueError("OTF corrections need at least two z positions")
    steps = z.diff()
    step = float(steps.mean())
    if not torch.allclose(steps, torch.full_like(steps, step), rtol=1e-4):
        raise ValueError("OTF corrections need uniformly spaced z positions")
    return step


def pupil_psf(
    pupil: Tensor,
    z: float | Sequence[float] | Tensor,
    *,
    spacing: Sequence[float],
    wavelength: float,
    numerical_aperture: float,
    refractive_index: float,
    vectorial: bool = False,
    bead_diameter: float | None = None,
    otf_sigma: float | None = None,
    normalize: bool = True,
) -> Tensor:
    """Intensity PSF sections computed from a pupil function.

    The amplitude PSF at defocus z is the 2D Fourier transform of the
    pupil P(k_x, k_y) times the defocus phase function (Hanser et al.
    2004, Eq. 2):

        PSF_A(x, y, z) = ∬ P(k) e^{2πi(k_x x + k_y y)} e^{2πi k_z z} dk,

        k_z = √((n/λ)² − k_x² − k_y²),   |k| ≤ NA/λ,

    evaluated by a discrete Fourier transform (unitary), and the
    intensity PSF is |PSF_A|². The piston n z/λ of k_z only adds a
    global phase per section and is dropped.

    Optional corrections, which the paper applies to PSFs calculated
    after phase retrieval:

    - `vectorial`: six component PSFs, the pupil times each factor of
      [`vectorial_factors`][optix.psf.utils.vectorial_factors] (Eq. 3),
      added incoherently;
    - `bead_diameter`: the 3D OTF is multiplied by
      [`bead_otf`][optix.psf.utils.bead_otf] (a convolution with the
      bead);
    - `otf_sigma`: the 3D OTF is multiplied by the empirical Gaussian
      rescaling exp(−|k|²/(2σ²)) (see
      [`fit_otf_rescaling`][optix.psf.utils.fit_otf_rescaling]).

    The OTF corrections are circular on the computed grid and need
    uniformly spaced `z`.

    !!! note "Orientation"
        z increases with the section index. Flipping the axis (z → −z)
        changes the pupil into its complex conjugate rotated by 180°
        about index `K // 2`, i.e. P(k) → P*(−k). For even K the row and
        column at index 0 have no partner: the rotation is
        `torch.roll(P.flip(-2, -1), (1, 1), (-2, -1))`, not a plain flip.

    !!! references "References"
        1. Hanser BM, Gustafsson MGL, Agard DA, Sedat JW.
           *Phase-retrieved pupil functions in wide-field fluorescence
           microscopy.* J Microsc **216**(1):32–48, 2004.
           [doi:10.1111/j.0022-2720.2004.01393.x](https://doi.org/10.1111/j.0022-2720.2004.01393.x)

        ??? quote "BibTeX"
            ```bibtex
            @article{hanser2004phase,
              author  = {Hanser, B. M. and Gustafsson, M. G. L. and
                         Agard, D. A. and Sedat, J. W.},
              title   = {Phase-retrieved pupil functions in wide-field
                         fluorescence microscopy},
              journal = {Journal of Microscopy},
              volume  = {216}, number = {1}, pages = {32--48},
              year    = {2004}, doi = {10.1111/j.0022-2720.2004.01393.x}
            }
            ```

    Parameters
    ----------
    pupil : Tensor
        `(..., Y, X)` complex pupil, k = 0 at index `K // 2` (see
        [`pupil_coordinates`][optix.psf.utils.pupil_coordinates]); values
        outside |k| ≤ NA/λ are ignored.
    z : float or sequence of float or Tensor
        Defocus of each section, in the units of `wavelength`.
    spacing : (float, float)
        Lateral pixel size `(Δy, Δx)` of the image grid.
    wavelength, numerical_aperture, refractive_index : float
        Vacuum wavelength λ, numerical aperture and immersion refractive
        index.
    vectorial : bool
        Use the vectorial model of Eq. 3. Default False (scalar).
    bead_diameter : float, optional
        Convolve with a bead of this diameter.
    otf_sigma : float, optional
        Width σ of the Gaussian OTF rescaling (cycles per unit length).
    normalize : bool
        Divide the result by its sum (unit-sum PSF). Default True.

    Returns
    -------
    Tensor
        `(..., Z, Y, X)` intensity PSF, centred at index `K // 2`
        laterally.
    """
    pupil = _complex(torch.as_tensor(pupil))
    real = pupil.real.dtype
    grid = _Grid(
        pupil.shape[-2:],
        spacing,
        wavelength,
        numerical_aperture,
        refractive_index,
        dtype=real,
        device=pupil.device,
    )
    z = torch.as_tensor(z, dtype=real, device=pupil.device).reshape(-1)
    field = (pupil * grid.support)[..., None, :, :] * grid.defocus(z)
    if vectorial:
        psf = torch.zeros(field.shape, dtype=real, device=pupil.device)
        for factor in _vectorial(grid):
            psf = psf + _ifft2c(field * factor).abs().square()
    else:
        psf = _ifft2c(field).abs().square()
    if bead_diameter is not None or otf_sigma is not None:
        step = _uniform_step(z)
        psf = _correct_otf(psf, (step, *spacing), bead_diameter, otf_sigma)
    if normalize:
        psf = psf / psf.sum((-3, -2, -1), keepdim=True)
    return psf


def zernike_pupil(
    magnitude: Tensor | Sequence[float] | None,
    phase: Tensor | Sequence[float] | None,
    *,
    shape: Sequence[int],
    spacing: Sequence[float],
    wavelength: float,
    numerical_aperture: float,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Pupil function from Zernike expansions of its magnitude and phase.

        P(ρ, θ) = (∑_j a_j Z_j(ρ, θ)) · exp(i ∑_j b_j Z_j(ρ, θ)),  ρ ≤ 1,

    with Noll-indexed [`zernike`][optix.psf.utils.zernike] polynomials
    (j = 1, 2, ...) on the pupil grid of a lateral image grid, and P = 0
    outside the pupil. Truncate the coefficient vectors to truncate the
    expansion.

    Parameters
    ----------
    magnitude : Tensor or sequence of float, optional
        `(..., J₁)` coefficients a_j of the magnitude; None for a unit
        magnitude.
    phase : Tensor or sequence of float, optional
        `(..., J₂)` coefficients b_j of the phase, in radians; None for
        a flat phase.
    shape, spacing : (int, int), (float, float)
        Lateral image grid `(Y, X)` and pixel size.
    wavelength, numerical_aperture : float
        Vacuum wavelength and numerical aperture.
    dtype, device : optional
        Floating dtype (default: that of the coefficients, else torch's
        default) and device.

    Returns
    -------
    Tensor
        `(..., Y, X)` complex pupil.
    """
    like = next((c for c in (magnitude, phase) if isinstance(c, Tensor)), None)
    if dtype is None:
        dtype = (
            like.dtype
            if like is not None and like.is_floating_point()
            else torch.get_default_dtype()
        )
    if device is None and like is not None:
        device = like.device
    device = None if device is None else torch.device(device)
    grid = _Grid(
        shape,
        spacing,
        wavelength,
        numerical_aperture,
        None,
        dtype=dtype,
        device=device,
    )

    def expand(coef: object) -> Tensor:
        c = torch.as_tensor(coef, dtype=dtype, device=device)
        basis = zernike(list(range(1, c.shape[-1] + 1)), grid.rho, grid.theta)
        return torch.einsum("...j,yxj->...yx", c, basis)

    mag = torch.ones_like(grid.rho) if magnitude is None else expand(magnitude)
    angle = torch.zeros_like(grid.rho) if phase is None else expand(phase)
    mag, angle = torch.broadcast_tensors(mag, angle)
    return torch.polar(mag * grid.support, angle * grid.support)


def zernike_psf(
    magnitude: Tensor | Sequence[float] | None,
    phase: Tensor | Sequence[float] | None,
    *,
    shape: Sequence[int],
    spacing: Sequence[float],
    wavelength: float,
    numerical_aperture: float,
    refractive_index: float,
    vectorial: bool = False,
    bead_diameter: float | None = None,
    otf_sigma: float | None = None,
    normalize: bool = True,
) -> Tensor:
    """Intensity PSF on any grid from Zernike coefficients of a pupil.

    The pupil of [`zernike_pupil`][optix.psf.utils.zernike_pupil] is
    built on the pupil grid of the lateral image grid and propagated by
    [`pupil_psf`][optix.psf.pupil_psf] to the planes
    z = (i − Z // 2) Δz, so that the PSF is centred at index `K // 2`
    along every axis. Typical input: the coefficients of a
    [`PupilEstimate`][optix.estimates.PupilEstimate].

    !!! references "References"
        1. Hanser BM, Gustafsson MGL, Agard DA, Sedat JW.
           *Phase-retrieved pupil functions in wide-field fluorescence
           microscopy.* J Microsc **216**(1):32–48, 2004.
           [doi:10.1111/j.0022-2720.2004.01393.x](https://doi.org/10.1111/j.0022-2720.2004.01393.x)

        ??? quote "BibTeX"
            ```bibtex
            @article{hanser2004phase,
              author  = {Hanser, B. M. and Gustafsson, M. G. L. and
                         Agard, D. A. and Sedat, J. W.},
              title   = {Phase-retrieved pupil functions in wide-field
                         fluorescence microscopy},
              journal = {Journal of Microscopy},
              volume  = {216}, number = {1}, pages = {32--48},
              year    = {2004}, doi = {10.1111/j.0022-2720.2004.01393.x}
            }
            ```

    Parameters
    ----------
    magnitude, phase : Tensor or sequence of float, optional
        Noll-ordered Zernike coefficients of the pupil magnitude and
        phase (see [`zernike_pupil`][optix.psf.utils.zernike_pupil]).
    shape : (int, int, int)
        Output shape `(Z, Y, X)`.
    spacing : (float, float, float)
        Voxel size `(Δz, Δy, Δx)`, in the units of `wavelength`.
    wavelength, numerical_aperture, refractive_index : float
        Vacuum wavelength, numerical aperture and immersion refractive
        index.
    vectorial, bead_diameter, otf_sigma, normalize
        See [`pupil_psf`][optix.psf.pupil_psf].

    Returns
    -------
    Tensor
        `(..., Z, Y, X)` intensity PSF.
    """
    nz, ny, nx = shape
    dz, dy, dx = spacing
    pupil = zernike_pupil(
        magnitude,
        phase,
        shape=(ny, nx),
        spacing=(dy, dx),
        wavelength=wavelength,
        numerical_aperture=numerical_aperture,
    )
    real = pupil.real.dtype
    z = (torch.arange(nz, dtype=real, device=pupil.device) - nz // 2) * dz
    return pupil_psf(
        pupil,
        z,
        spacing=(dy, dx),
        wavelength=wavelength,
        numerical_aperture=numerical_aperture,
        refractive_index=refractive_index,
        vectorial=vectorial,
        bead_diameter=bead_diameter,
        otf_sigma=otf_sigma,
        normalize=normalize,
    )


def fit_otf_rescaling(
    measured: Tensor,
    calculated: Tensor,
    *,
    spacing: Sequence[float],
    wavelength: float,
    numerical_aperture: float,
    refractive_index: float,
) -> float:
    """Width of the empirical Gaussian OTF rescaling.

    The point-wise ratio of the magnitudes of the measured and the
    calculated 3D OTFs, each normalised to 1 at k = 0, is fitted by
    least squares with

        g(k) = exp(−|k|² / (2σ²))

    over the OTF support of a wide-field microscope of numerical
    aperture `numerical_aperture`, i.e. |k_r| ≤ 2 NA/λ and
    |k_z| ≤ √((n/λ)² − (|k_r| − NA/λ)²) − √((n/λ)² − (NA/λ)²).
    Hanser et al. (2004) fit on the support of a smaller aperture than
    the objective's (NA 1.15 for an NA 1.3 objective), where the
    measured OTF is above the noise, and multiply the magnitude of
    calculated OTFs by g ([`pupil_psf`][optix.psf.pupil_psf] with
    `otf_sigma=σ`).

    !!! note "Interpretation"
        The paper fits "a Gaussian" "as a function of spatial
        frequency"; its exact form is not specified. Here g is isotropic
        in the 3D frequency |k|, with unit value at k = 0 (so the PSF sum
        is kept), and σ is found by a golden-section search of the
        least-squares criterion. Frequencies where the calculated OTF is
        below 10⁻³ of its value at k = 0 are left out of the fit, since
        the ratio is dominated by noise there (not specified in the
        paper, which restricts the fit to the support of a smaller
        aperture for the same reason).

    !!! references "References"
        1. Hanser BM, Gustafsson MGL, Agard DA, Sedat JW.
           *Phase-retrieved pupil functions in wide-field fluorescence
           microscopy.* J Microsc **216**(1):32–48, 2004.
           [doi:10.1111/j.0022-2720.2004.01393.x](https://doi.org/10.1111/j.0022-2720.2004.01393.x)

        ??? quote "BibTeX"
            ```bibtex
            @article{hanser2004phase,
              author  = {Hanser, B. M. and Gustafsson, M. G. L. and
                         Agard, D. A. and Sedat, J. W.},
              title   = {Phase-retrieved pupil functions in wide-field
                         fluorescence microscopy},
              journal = {Journal of Microscopy},
              volume  = {216}, number = {1}, pages = {32--48},
              year    = {2004}, doi = {10.1111/j.0022-2720.2004.01393.x}
            }
            ```

    Parameters
    ----------
    measured, calculated : Tensor
        `(Z, Y, X)` measured and calculated intensity PSFs on the same
        grid.
    spacing : (float, float, float)
        Voxel size `(Δz, Δy, Δx)`.
    wavelength, numerical_aperture, refractive_index : float
        Vacuum wavelength, numerical aperture of the fit support, and
        immersion refractive index.

    Returns
    -------
    float
        σ in cycles per unit length (`inf` when no attenuation fits).
    """
    _check_optics(wavelength, numerical_aperture, refractive_index)
    measured = torch.as_tensor(measured, dtype=torch.float64)
    calculated = torch.as_tensor(calculated, dtype=torch.float64)
    if measured.shape != calculated.shape or measured.ndim != 3:
        raise ValueError("measured and calculated must be equal 3D shapes")
    om = torch.fft.fftn(measured.cpu())
    oc = torch.fft.fftn(calculated.cpu())
    om, oc = om.abs() / om[0, 0, 0].abs(), oc.abs() / oc[0, 0, 0].abs()
    fz, fy, fx = [
        torch.fft.fftfreq(s, d, dtype=torch.float64)
        for s, d in zip(measured.shape, spacing, strict=True)
    ]
    kr = (fy[None, :, None].square() + fx[None, None, :].square()).sqrt()
    kz = fz[:, None, None].abs()
    big, edge = refractive_index / wavelength, numerical_aperture / wavelength
    bound = (big**2 - (kr - edge).square()).clamp(min=0).sqrt() - math.sqrt(
        big**2 - edge**2
    )
    mask = (kr <= 2 * edge) & (kz <= bound)
    mask &= oc > _OTF_FLOOR
    ratio = (om / torch.where(mask, oc, 1))[mask]
    k2 = (kr.square() + kz.square()).expand_as(mask)[mask]
    kmax = float(k2.max()) if k2.numel() else 1.0
    if kmax == 0:
        return math.inf

    def cost(t: float) -> float:
        return float((ratio - torch.exp(-t * k2)).square().sum())

    lo, hi = 0.0, 50.0 / kmax
    golden = (math.sqrt(5) - 1) / 2
    a, b = hi - golden * (hi - lo), lo + golden * (hi - lo)
    fa, fb = cost(a), cost(b)
    for _ in range(200):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - golden * (hi - lo)
            fa = cost(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + golden * (hi - lo)
            fb = cost(b)
    t = 0.5 * (lo + hi)
    if cost(0.0) <= cost(t):
        return math.inf
    return 1.0 / math.sqrt(2 * t)


def index_mismatch_aberration(
    shape: Sequence[int],
    spacing: Sequence[float],
    *,
    depth: float,
    wavelength: float,
    numerical_aperture: float,
    immersion_index: float,
    specimen_index: float,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Pupil aberration function of a refractive-index mismatch.

    A point source at depth d in a medium of index n₂, imaged by an
    objective designed for an immersion medium of index n₁ (Hanser et
    al. 2004, Eqs. 4–8):

        OP = d (n₂ cos θ₂ − n₁ cos θ₁),
        A_t = [sin θ₁ cos θ₂ / sin(θ₁ + θ₂)] [1 + 1/cos(θ₂ − θ₁)],
        A_w = n₁ tan θ₂ / (n₂ tan θ₁),
        sin θ₁ = (λ/n₁) |k|,   n₁ sin θ₁ = n₂ sin θ₂,
        ψ = A_t A_w e^{2πi OP/λ}.

    A_t is the mean of the s and p Fresnel transmittances, A_w the
    wavefront compression. Multiplying an unaberrated pupil by ψ gives
    the pupil of the microscope under the mismatched conditions (Eq. 9).
    They are evaluated in the equivalent forms
    A_t = cos θ₂ [1 + 1/cos(θ₂ − θ₁)] / (cos θ₂ + (n₁/n₂) cos θ₁) and
    A_w = (n₁/n₂)² cos θ₁ / cos θ₂, which are finite on axis.

    !!! note
        Rays beyond the critical angle (n₁ sin θ₁ > n₂) do not reach the
        pupil and get ψ = 0; this case is not specified in the paper.
        Eq. 9 of the paper also carries a constant factor 1/λ, which
        only scales the PSF and is omitted here.

    !!! references "References"
        1. Hanser BM, Gustafsson MGL, Agard DA, Sedat JW.
           *Phase-retrieved pupil functions in wide-field fluorescence
           microscopy.* J Microsc **216**(1):32–48, 2004.
           [doi:10.1111/j.0022-2720.2004.01393.x](https://doi.org/10.1111/j.0022-2720.2004.01393.x)

        ??? quote "BibTeX"
            ```bibtex
            @article{hanser2004phase,
              author  = {Hanser, B. M. and Gustafsson, M. G. L. and
                         Agard, D. A. and Sedat, J. W.},
              title   = {Phase-retrieved pupil functions in wide-field
                         fluorescence microscopy},
              journal = {Journal of Microscopy},
              volume  = {216}, number = {1}, pages = {32--48},
              year    = {2004}, doi = {10.1111/j.0022-2720.2004.01393.x}
            }
            ```

    Parameters
    ----------
    shape, spacing : (int, int), (float, float)
        Lateral image grid (see
        [`pupil_coordinates`][optix.psf.utils.pupil_coordinates]).
    depth : float
        Depth d of the source below the cover glass, in the units of
        `wavelength`.
    wavelength, numerical_aperture : float
        Vacuum wavelength and numerical aperture.
    immersion_index : float
        Refractive index n₁ of the immersion medium the objective is
        designed for.
    specimen_index : float
        Refractive index n₂ of the specimen (mounting medium).
    dtype, device : optional

    Returns
    -------
    Tensor
        `(Y, X)` complex aberration function ψ, 0 outside the pupil.
    """
    grid = _Grid(
        shape,
        spacing,
        wavelength,
        numerical_aperture,
        immersion_index,
        dtype=dtype or torch.get_default_dtype(),
        device=None if device is None else torch.device(device),
    )
    n1, n2 = float(immersion_index), float(specimen_index)
    s1 = (grid.k2.sqrt() * wavelength / n1).clamp(max=1)
    s2 = n1 * s1 / n2
    ok = grid.support & (s2 < 1)
    s2 = torch.where(ok, s2, 0)
    c1 = (1 - s1.square()).sqrt()
    c2 = (1 - s2.square()).sqrt()
    cos_diff = c2 * c1 + s2 * s1
    a_t = c2 * (1 + 1 / cos_diff) / (c2 + (n1 / n2) * c1)
    a_w = (n1 / n2) ** 2 * c1 / c2
    path = depth * (n2 * c2 - n1 * c1)
    amplitude = torch.where(ok, a_t * a_w, 0)
    return torch.polar(amplitude, 2 * math.pi * path / wavelength * ok)


# -- phase retrieval -----------------------------------------------------

_DEFAULTS: dict[str, object] = {
    "offsets": (-3.0, -1.0, 1.0, 3.0),
    "focus": None,
    "iterations": 25,
    "smoothing": None,
    "zernike_terms": 79,
    "unwrap": True,
}


def _metrics(calc: Tensor, meas: Tensor, amp: Tensor) -> Tensor:
    """Per group: iMSE, correlation coefficient, amplitude error."""
    dims = tuple(range(1, calc.ndim))
    imse = (calc - meas).square().sum(dims) / meas.square().sum(dims)
    c = calc - calc.mean(dims, keepdim=True)
    m = meas - meas.mean(dims, keepdim=True)
    corr = (c * m).sum(dims) / (
        c.square().sum(dims).sqrt() * m.square().sum(dims).sqrt()
    )
    amp_err = (calc.sqrt() - amp).square().sum(dims) / amp.square().sum(dims)
    return torch.stack([imse, corr, amp_err], -1)


class _StackReader:
    """Reads z-slabs of one group of a `(*group, Z, Y, X)` stack.

    Only basic slices are taken from the storage (any array-like, or the
    array of a [`Labeled`][optix.io.Labeled] whose layout gives the axis
    roles), so a zarr-backed stack is never loaded whole.
    """

    def __init__(self, data: object, backend: dict) -> None:
        self.backend = backend
        if isinstance(data, Labeled):
            raw, layout = data.array, data.layout
            ndim = len(raw.shape)
            group, sample, spatial = layout.positions(ndim)
            if sample:
                raise ValueError(
                    "PhaseRetrieval takes one stack per group: average the "
                    "sample axes first (e.g. with BeadPSF)"
                )
            if len(spatial) != 3:
                raise ValueError("PhaseRetrieval needs 3 spatial axes")
            perm = layout.permutation(ndim)
        else:
            raw = data
            ndim = len(raw.shape)
            if ndim < 3:
                raise ValueError("PhaseRetrieval needs a (…, Z, Y, X) stack")
            group, spatial = (
                tuple(range(ndim - 3)),
                (ndim - 3, ndim - 2, ndim - 1),
            )
            perm = tuple(range(ndim))
        self.raw, self.perm = raw, perm
        self.group_axes, self.z_axis = group, spatial[0]
        self.groups = tuple(int(raw.shape[a]) for a in group)
        self.ngroups = math.prod(self.groups)
        self.shape = tuple(int(raw.shape[a]) for a in spatial)
        probe = self.read(0, 0, 1)
        self.dtype, self.device = probe.dtype, probe.device

    def read(self, g: int, z0: int, z1: int) -> Tensor:
        """Sections `z0:z1` of group `g` (flat index), `(z1 − z0, Y, X)`."""
        index = [slice(None)] * len(self.raw.shape)
        rest = g
        pairs = list(zip(self.group_axes, self.groups, strict=True))
        for axis, size in reversed(pairs):
            rest, i = divmod(rest, size)
            index[axis] = slice(i, i + 1)
        index[self.z_axis] = slice(z0, z1)
        x = as_tensor(self.raw[tuple(index)], **self.backend)
        x = x.permute(self.perm)
        return x.reshape(x.shape[-3:])


class _Agreement:
    """Mergeable sums of the iMSE and the correlation of two images.

    Means and centred (co)moments are merged pairwise (Chan et al.), in
    float64, so the correlation does not cancel catastrophically.
    """

    def __init__(self) -> None:
        self.n = 0
        self.mc = self.mm = 0.0
        self.cc = self.ccm = self.cm = 0.0
        self.diff2 = self.meas2 = 0.0

    def update(self, calc: Tensor, meas: Tensor) -> None:
        c, m = calc.double().flatten(), meas.double().flatten()
        nb = c.numel()
        if nb == 0:
            return
        mc, mm = float(c.mean()), float(m.mean())
        dc, dm = c - mc, m - mm
        cc, mmm, cm = (
            float(dc.square().sum()),
            float(dm.square().sum()),
            float((dc * dm).sum()),
        )
        na, n = self.n, self.n + nb
        ddc, ddm = mc - self.mc, mm - self.mm
        w = na * nb / n
        self.cc += cc + ddc * ddc * w
        self.ccm += mmm + ddm * ddm * w
        self.cm += cm + ddc * ddm * w
        self.mc += ddc * nb / n
        self.mm += ddm * nb / n
        self.n = n
        self.diff2 += float((c - m).square().sum())
        self.meas2 += float(m.square().sum())

    @property
    def imse(self) -> float:
        return self.diff2 / self.meas2 if self.meas2 > 0 else math.nan

    @property
    def correlation(self) -> float:
        den = math.sqrt(self.cc * self.ccm)
        return self.cm / den if den > 0 else math.nan


class PhaseRetrieval(Estimator):
    """Pupil function retrieved from a bead z-stack, and its Zernike fit.

    The pupil function P(k_x, k_y) of a wide-field microscope is
    estimated from a few defocused sections of a measured intensity PSF
    by the Gerchberg–Saxton-type iteration of Hanser et al. (2004),
    built on the imaging model of
    [`pupil_psf`][optix.psf.pupil_psf] (Eq. 2):

    1. Start from P = 1 inside the pupil |k| ≤ NA/λ, 0 outside.
    2. For each selected section j at defocus z_j, multiply P by the
       defocus phase function e^{2πi k_z z_j} and Fourier transform it
       to the amplitude PSF section g_j.
    3. Replace the magnitude of g_j by the square root of the measured
       intensity, keeping its phase.
    4. Transform back and refocus by e^{−2πi k_z z_j}.
    5. Average the refocused pupils over the sections.
    6. Set P = 0 outside the pupil (NA constraint), and smooth it (see
       the warning below).
    7. Repeat from 2 for a fixed number of iterations (25 in the paper).

    The magnitude and the phase of the final pupil are then expanded in
    Zernike polynomials by linear least squares (79 terms: all radial
    orders up to 11, plus the radial polynomial of order 12), after the
    phase is unwrapped.

    **Sections.** The paper uses about four sections offset by −3, −1,
    +1 and +3 µm from the peak-intensity section; `offsets` selects the
    sections nearest to these offsets (in the units of `spacing`) from
    the `focus` section (default: the one holding the brightest voxel),
    and z_j = (i_j − focus) Δz.

    **Corrections.** The paper corrects calculated PSFs after retrieval
    for the bead size, the vectorial nature of light and an empirical
    OTF rescaling; these are options of
    [`pupil_psf`][optix.psf.pupil_psf] /
    [`zernike_psf`][optix.psf.zernike_psf], with
    [`fit_otf_rescaling`][optix.psf.utils.fit_otf_rescaling] for the
    rescaling.

    **Quality.** The intensity mean-square error (iMSE) and the
    correlation coefficient between the measured sections and those of
    the current pupil are recorded at each iteration in `history`
    (`extras["imse"]`, `extras["correlation"]`, plus the amplitude error
    ∑(|g_j| − √I_j)² / ∑ I_j, which the iteration never increases without
    smoothing); `history.objective` is −iMSE. The final values over the
    whole stack are in `meta["imse"]` / `meta["correlation"]`.

    **Memory.** The stack is read in z-slabs, one group at a time: a pass
    for the per-section maxima (only when `focus` is None), the selected
    sections, and a pass for the whole-stack iMSE and correlation, which
    are accumulated from mergeable sums. Only the selected sections and
    the pupils of all groups are held in memory during the iteration.

    !!! note "Choices not specified in the paper"
        - iMSE = ∑(I_calc − I_meas)² / ∑ I_meas² over the voxels compared.
        - The data are scaled so that the mean energy of the selected
          sections equals that of the starting pupil (its number of
          pixels); the retrieved magnitude is then close to 1.
        - Negative intensities (after background subtraction) are set to
          0 before the square root.
        - The paper's optional stopping criterion is not implemented
          (fixed number of iterations).
        - The arbitrary global phase of the result is set so that ∑P is
          real and positive.
        - Zernike indexing and normalisation are Noll's (1976), whom the
          paper cites; the 79 terms are Noll indices 1 … 79.
        - The phase is unwrapped by
          [`unwrap_phase`][optix.psf.utils.unwrap_phase] (path not
          specified in the paper); the fits are unweighted over the
          pupil pixels.

    !!! warning "Departure from the paper: smoothing"
        Hanser et al. (2004) applied a smoothing filter to the pupil at
        every iteration (step 6) to suppress noise, noting that less
        frequent or no smoothing may suit low-noise data. The paper does
        not specify the filter, so it cannot be reproduced: the default
        (`smoothing=None`, also in faithful mode) runs without smoothing.
        `smoothing=σ` applies a Gaussian of standard deviation σ pupil
        pixels to the real and imaginary parts of the pupil at each
        iteration; it is an extension (`is_faithful` is then False).

    !!! note "Sampling"
        The pupil grid is the Fourier grid of the lateral field of view:
        the pupil radius is NA · X · Δx / λ pixels, and the most
        defocused sections must fit in the field of view. Pad the data
        laterally (after background subtraction) for a finer pupil.
        Preprocessing (bad pixels, background, lamp flicker) is left to
        the caller, e.g. [`BeadPSF`][optix.psf.BeadPSF].

    !!! references "References"
        1. Hanser BM, Gustafsson MGL, Agard DA, Sedat JW.
           *Phase-retrieved pupil functions in wide-field fluorescence
           microscopy.* J Microsc **216**(1):32–48, 2004.
           [doi:10.1111/j.0022-2720.2004.01393.x](https://doi.org/10.1111/j.0022-2720.2004.01393.x)

        ??? quote "BibTeX"
            ```bibtex
            @article{hanser2004phase,
              author  = {Hanser, B. M. and Gustafsson, M. G. L. and
                         Agard, D. A. and Sedat, J. W.},
              title   = {Phase-retrieved pupil functions in wide-field
                         fluorescence microscopy},
              journal = {Journal of Microscopy},
              volume  = {216}, number = {1}, pages = {32--48},
              year    = {2004}, doi = {10.1111/j.0022-2720.2004.01393.x}
            }
            ```
    """

    tags = Tags(
        targets=("pupil",),
        streaming="whole-stack",
        modality=("widefield",),
        noise="none",
        reference=f"doi:{_DOI}",
    )
    paper = dict(_DEFAULTS)
    recommended = dict(_DEFAULTS)

    def __init__(
        self,
        *,
        spacing: Sequence[float] | None = None,
        wavelength: float | None = None,
        numerical_aperture: float | None = None,
        refractive_index: float | None = None,
        offsets: Sequence[float] | str | None = None,
        focus: int | None = None,
        iterations: int | None = None,
        smoothing: float | None = None,
        zernike_terms: int | None = None,
        unwrap: bool | None = None,
        faithful: bool = True,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        layout: object = None,
    ) -> None:
        """Set the optics and the retrieval options.

        Parameters
        ----------
        spacing : (float, float, float)
            Voxel size `(Δz, Δy, Δx)` of the stack, in the units of
            `wavelength` (µm for the default `offsets`). Required.
        wavelength : float
            Vacuum wavelength λ of the detected light. Required.
        numerical_aperture : float
            Numerical aperture. Required.
        refractive_index : float
            Refractive index of the immersion medium. Required.
        offsets : sequence of float or "all"
            Defocus of the sections used, relative to `focus`; each
            selects the nearest section. Paper: (−3, −1, 1, 3) µm.
            `"all"`: every section.
        focus : int, optional
            Index of the in-focus section. Default (paper): the section
            holding the brightest voxel.
        iterations : int
            Number of iterations. Paper: 25.
        smoothing : float, optional
            Standard deviation, in pupil pixels, of a Gaussian smoothing
            of the pupil at each iteration. Default: none. The paper
            smooths at each iteration with a filter it does not specify,
            so faithful mode leaves smoothing off (see the warning
            above); any value is an extension.
        zernike_terms : int
            Number of Noll-ordered Zernike terms fitted. Paper: 79.
        unwrap : bool
            Unwrap the phase before its Zernike fit. Paper: True.
        faithful : bool
            Both presets are the paper's settings.
        dtype, device : optional
            See [`Estimator`][optix.Estimator].
        layout : Layout, optional
            Axis roles: three spatial axes `(z, y, x)`, the first along
            the optical axis, and optional group axes (one pupil each).
            Default: the trailing three dims are spatial, leading dims
            are groups.
        """
        super().__init__(faithful=faithful, dtype=dtype, device=device)
        self._resolve(faithful, locals())
        opt = self.options
        for name in (
            "spacing",
            "wavelength",
            "numerical_aperture",
            "refractive_index",
        ):
            if opt[name] is None:
                raise ValueError(f"PhaseRetrieval needs `{name}`")
        if len(opt["spacing"]) != 3:
            raise ValueError("spacing must be (Δz, Δy, Δx)")
        _check_optics(
            float(opt["wavelength"]),
            float(opt["numerical_aperture"]),
            opt["refractive_index"],
        )
        if isinstance(opt["offsets"], str):
            if opt["offsets"] != "all":
                raise ValueError(
                    f"offsets must be numbers or 'all': {offsets!r}"
                )
        else:
            opt["offsets"] = tuple(float(o) for o in opt["offsets"])
            self.is_faithful = all(
                opt.get(k) == v for k, v in self.paper.items()
            )
        if int(opt["iterations"]) < 0:
            raise ValueError("iterations must be ≥ 0")

    def _resolve(self, faithful: bool, params: dict) -> None:
        super()._resolve(faithful, params)
        # the optics describe the data, not the algorithm
        self.is_faithful = all(
            self.options.get(k) == v for k, v in self.paper.items()
        )

    # -- data -----------------------------------------------------------
    def _reader(self, data: object) -> "_StackReader":
        backend = self._backend(data)
        if isinstance(data, Estimator):
            data = data.estimates.get("psf")
            if data is None:
                raise ValueError("the estimator has no 'psf' estimate")
        if isinstance(data, PSFEstimate):
            data = data.psf
        return _StackReader(data, backend)

    def _sections(self, focus: int, nz: int) -> list[int]:
        opt = self.options
        dz = float(opt["spacing"][0])
        if not 0 <= focus < nz:
            raise ValueError(f"focus {focus} is outside the stack (Z={nz})")
        if opt["offsets"] == "all":
            return list(range(nz))
        idx = [focus + round(float(o) / dz) for o in opt["offsets"]]
        if any(not 0 <= i < nz for i in idx):
            raise ValueError(
                f"offsets {opt['offsets']} from section {focus} fall "
                f"outside the stack (Z={nz}, Δz={dz})"
            )
        if len(set(idx)) != len(idx):
            raise ValueError(f"offsets select repeated sections: {idx}")
        return idx

    # -- fit ------------------------------------------------------------
    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        self._warn_unused(init)
        opt = self.options
        reader = self._reader(data)
        real, device = reader.dtype, reader.device
        nz, ny, nx = reader.shape
        dz, dy, dx = (float(s) for s in opt["spacing"])
        grid = _Grid(
            (ny, nx),
            (dy, dx),
            float(opt["wavelength"]),
            float(opt["numerical_aperture"]),
            float(opt["refractive_index"]),
            dtype=real,
            device=device,
        )
        support = grid.support
        complex_dtype = _complex(torch.zeros((), dtype=real)).dtype
        # z-slab length of the streaming passes (a few complex temporaries)
        slab = chunk_length(
            (ny, nx), dtype=complex_dtype, overhead=6, maximum=nz
        )

        # sections: one group at a time, reading only the chosen ones
        focus, chosen, meas = [], [], []
        for g in range(reader.ngroups):
            if opt["focus"] is None:
                peak = torch.cat(
                    [
                        reader.read(g, z0, min(z0 + slab, nz)).amax((-2, -1))
                        for z0 in range(0, nz, slab)
                    ]
                )
                f = int(peak.argmax())
            else:
                f = int(opt["focus"])
            idx = self._sections(f, nz)
            focus.append(f)
            chosen.append(idx)
            meas.append(torch.cat([reader.read(g, i, i + 1) for i in idx]))
        meas = torch.stack(meas).clamp(min=0)  # (G, J, Y, X)
        ng = meas.shape[0]
        index = torch.tensor(chosen, device=device)
        energy = meas.sum((-2, -1)).mean(-1)
        scale = support.sum().to(real) / energy.clamp(
            min=torch.finfo(real).tiny
        )
        meas = meas * scale[:, None, None, None]
        amp = meas.sqrt()
        offsets = index - torch.tensor(focus, device=device)[:, None]
        defocus = grid.defocus(offsets.to(real) * dz)  # (G, J, Y, X)
        pupil = support.to(defocus.dtype).expand(ng, -1, -1).clone()
        sigma = opt["smoothing"]
        iterations = int(opt["iterations"])
        for t in range(iterations + 1):
            field = _ifft2c(pupil[:, None] * defocus)
            calc = field.abs().square()
            stats = _metrics(calc, meas, amp).mean(0).tolist()
            self.history.record(
                -stats[0],
                imse=stats[0],
                correlation=stats[1],
                amplitude_error=stats[2],
            )
            if verbose:
                print(
                    f"PhaseRetrieval {t}: iMSE {stats[0]:.4g}, "
                    f"correlation {stats[1]:.4f}"
                )
            if t == iterations:
                break
            field = torch.polar(amp, field.angle())
            pupil = (_fft2c(field) * defocus.conj()).mean(1) * support
            if sigma:
                s = (0.0, float(sigma), float(sigma))
                pupil = (
                    torch.complex(
                        gaussian_filter(pupil.real, s),
                        gaussian_filter(pupil.imag, s),
                    )
                    * support
                )
            if callback is not None:
                callback(self)
        self.history.reason = f"{iterations} iterations"
        del field, calc, amp, meas, defocus

        total = pupil.sum((-2, -1))
        norm = total.abs()
        rotate = torch.where(
            norm > 0, total.conj() / norm.clamp(min=1e-300), 1
        )
        pupil = pupil * rotate[:, None, None]

        # final quality over the whole stack, streamed in z-slabs
        imse, corr = [], []
        for g in range(ng):
            acc = _Agreement()
            for z0 in range(0, nz, slab):
                z1 = min(z0 + slab, nz)
                whole = reader.read(g, z0, z1).clamp(min=0) * scale[g]
                zs = torch.arange(z0, z1, device=device) - focus[g]
                calc = _ifft2c(pupil[g] * grid.defocus(zs.to(real) * dz))
                acc.update(calc.abs().square(), whole)
            imse.append(acc.imse)
            corr.append(acc.correlation)

        terms = int(opt["zernike_terms"])
        phase = pupil.angle()
        if opt["unwrap"]:
            phase = unwrap_phase(phase, support)
        mag_coef = zernike_fit(
            pupil.abs(), grid.rho, grid.theta, terms=terms, support=support
        )
        phase_coef = zernike_fit(
            phase, grid.rho, grid.theta, terms=terms, support=support
        )
        groups = reader.groups
        self.estimates = {
            "pupil": PupilEstimate(
                pupil=pupil.reshape(groups + (ny, nx)),
                zernike_magnitude=mag_coef.reshape(groups + (terms,)),
                zernike_phase=phase_coef.reshape(groups + (terms,)),
                meta={
                    "wavelength": float(opt["wavelength"]),
                    "numerical_aperture": float(opt["numerical_aperture"]),
                    "refractive_index": float(opt["refractive_index"]),
                    "spacing": [dz, dy, dx],
                    "focus": focus,
                    "sections": chosen,
                    "scale": scale.tolist(),
                    "imse": imse,
                    "correlation": corr,
                    "zernike": "noll",
                },
            )
        }

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        raise NotImplementedError(
            "PhaseRetrieval has no apply(): render PSFs with pupil_psf or "
            "zernike_psf"
        )


def phase_retrieval(
    data: ArrayLike,
    **options: object,
) -> PupilEstimate:
    """Pupil function of a microscope from a bead z-stack.

    Shortcut for `PhaseRetrieval(**options).fit(data)`; see
    [`PhaseRetrieval`][optix.psf.PhaseRetrieval] for the algorithm and
    the options.

    !!! references "References"
        1. Hanser BM, Gustafsson MGL, Agard DA, Sedat JW.
           *Phase-retrieved pupil functions in wide-field fluorescence
           microscopy.* J Microsc **216**(1):32–48, 2004.
           [doi:10.1111/j.0022-2720.2004.01393.x](https://doi.org/10.1111/j.0022-2720.2004.01393.x)

        ??? quote "BibTeX"
            ```bibtex
            @article{hanser2004phase,
              author  = {Hanser, B. M. and Gustafsson, M. G. L. and
                         Agard, D. A. and Sedat, J. W.},
              title   = {Phase-retrieved pupil functions in wide-field
                         fluorescence microscopy},
              journal = {Journal of Microscopy},
              volume  = {216}, number = {1}, pages = {32--48},
              year    = {2004}, doi = {10.1111/j.0022-2720.2004.01393.x}
            }
            ```

    Parameters
    ----------
    data : array-like, Labeled, PSFEstimate or BeadPSF
        `(…, Z, Y, X)` measured intensity PSF (background subtracted),
        or a fitted [`BeadPSF`][optix.psf.BeadPSF] or its estimate.
    **options
        Options of [`PhaseRetrieval`][optix.psf.PhaseRetrieval]
        (`spacing`, `wavelength`, `numerical_aperture` and
        `refractive_index` are required).

    Returns
    -------
    PupilEstimate
        The pupil and its Zernike coefficients.
    """
    est = PhaseRetrieval(**options).fit(data)  # type: ignore[arg-type]
    return est.estimates["pupil"]
