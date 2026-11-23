"""Gibson–Lanni scalar PSF of a widefield microscope with layer mismatch."""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .._config import get_config, resolve_backend
from ..estimates import PSFEstimate
from ..special import bessel_j0, bessel_j1, gauss_legendre
from ._ops import fwhm

__all__ = [
    "gibson_lanni",
    "gibson_lanni_opd",
    "gibson_lanni_profile",
]

#: Reference wavelength λ_ref (µm) of the Bessel scaling factors of Li et
#: al., σₘ = NA (3m − 2) λ_ref/λ₀ (their λ₀; λ₀ is here the wavelength).
LI2017_WAVELENGTH = 0.436

_METHODS = ("quadrature", "li2017")
_MAX_NODES = 2**16


@dataclass(frozen=True)
class _Optics:
    """Layers of the Gibson–Lanni model (lengths in µm)."""

    na: float
    ts: float  # depth of the source below the coverslip
    ns: float
    ng: float
    tg: float
    ni: float
    ng0: float
    tg0: float
    ni0: float
    ti0: float

    @property
    def n_min(self) -> float:
        """Smallest index of the layers of non-zero thickness.

        The actual immersion layer always counts; the specimen, the
        coverslips and the design immersion layer only if their
        thickness is > 0 (a layer of zero thickness has no term in the
        OPD).
        """
        layers = [(self.ni, 1.0), (self.ns, self.ts), (self.ng, self.tg)]
        layers += [(self.ng0, self.tg0), (self.ni0, self.ti0)]
        return min(n for n, t in layers if t > 0)

    @property
    def aperture(self) -> float:
        """Upper limit a of the normalised pupil radius ρ."""
        return min(1.0, self.n_min / self.na)


def _optics(
    numerical_aperture: float,
    depth: float = 0.0,
    specimen_index: float = 1.33,
    coverslip_index: float | None = None,
    coverslip_thickness: float | None = None,
    immersion_index: float | None = None,
    design_coverslip_index: float = 1.522,
    design_coverslip_thickness: float = 170.0,
    design_immersion_index: float = 1.515,
    design_immersion_thickness: float = 150.0,
) -> _Optics:
    p = _Optics(
        na=float(numerical_aperture),
        ts=float(depth),
        ns=float(specimen_index),
        ng=float(
            design_coverslip_index
            if coverslip_index is None
            else coverslip_index
        ),
        tg=float(
            design_coverslip_thickness
            if coverslip_thickness is None
            else coverslip_thickness
        ),
        ni=float(
            design_immersion_index
            if immersion_index is None
            else immersion_index
        ),
        ng0=float(design_coverslip_index),
        tg0=float(design_coverslip_thickness),
        ni0=float(design_immersion_index),
        ti0=float(design_immersion_thickness),
    )
    if not p.na > 0:
        raise ValueError(f"numerical_aperture must be > 0, got {p.na}")
    for name in ("ns", "ng", "ni", "ng0", "ni0"):
        if not getattr(p, name) > 0:
            raise ValueError("Refractive indices must be > 0")
    for name in ("ts", "tg", "tg0", "ti0"):
        if getattr(p, name) < 0:
            raise ValueError("Depth and thicknesses must be >= 0")
    return p


def _hi(device: torch.device) -> torch.dtype:
    """Highest precision available on `device`."""
    return torch.float32 if device.type == "mps" else torch.float64


def _dn(n: float, s2: Tensor) -> Tensor:
    """√(n² − s²) − n, without cancellation."""
    return -s2 / (torch.sqrt((n * n - s2).clamp_min(0)) + n)


def _opd(rho: Tensor, defocus: Tensor, p: _Optics) -> Tensor:
    """OPD(ρ, Δz) − OPD(0, Δz) of Gibson & Lanni Eq. (4), z_d = z_d*.

    `rho` (..., Q) and `defocus` broadcast against each other.
    """
    s2 = (p.na * rho) ** 2
    di = _dn(p.ni, s2)

    def layer(n: float) -> Tensor:
        # n t {[1 − (NAρ/n)²]^½ − (n_oil/n)² [1 − (NAρ/n_oil)²]^½} / t,
        # minus its value at ρ = 0
        return _dn(n, s2) - (p.ni / n) * di

    static = p.ts * layer(p.ns) + p.tg * layer(p.ng)
    static = static - p.tg0 * layer(p.ng0) - p.ti0 * layer(p.ni0)
    return defocus * di + static


def _check_optics_keys(optics: dict) -> None:
    allowed = set(_optics.__code__.co_varnames[: _optics.__code__.co_argcount])
    unknown = set(optics) - allowed
    if unknown:
        raise TypeError(f"Unknown optical parameters: {sorted(unknown)}")
    if "numerical_aperture" not in optics:
        raise TypeError("`numerical_aperture` is required")


def gibson_lanni_opd(
    rho: float | Tensor,
    defocus: float | Tensor = 0.0,
    **optics: float | None,
) -> Tensor:
    """Optical path difference of the Gibson–Lanni model.

        OPD(ρ; Δz) = n_oil Δz C_oil(ρ)
                   + n_s t_s [C_s(ρ) − (n_oil/n_s)² C_oil(ρ)]
                   + n_g t_g [C_g(ρ) − (n_oil/n_g)² C_oil(ρ)]
                   − n_g* t_g* [C_g*(ρ) − (n_oil/n_g*)² C_oil(ρ)]
                   − n_oil* t_oil* [C_oil*(ρ) − (n_oil/n_oil*)² C_oil(ρ)],
        C_n(ρ) = [1 − (NA ρ / n)²]^½,

    Gibson & Lanni Eq. (4) with the detector at its design distance
    (z_d = z_d*), minus its value at ρ = 0 (a constant phase, which does
    not change the intensity). Starred quantities are the design values;
    t_s is the depth of the source below the coverslip and Δz the
    defocus of Eq. (3),

        Δz/n_oil = t_s/n_s + t_g/n_g + t_oil/n_oil − t_g*/n_g* − t_oil*/n_oil*,

    which eliminates the actual immersion thickness t_oil from Eq. (2):
    Δz = 0 is the paraxial (geometric) focus, where the ρ² term of the
    OPD vanishes. Each square root is evaluated as
    √(n² − s²) − n = −s² / (√(n² − s²) + n), s = NA ρ, so that no large
    terms cancel.

    The OPD is defined for |ρ| ≤ n_min/NA, with n_min the smallest
    refractive index of the layers of non-zero thickness (see
    [`gibson_lanni`][optix.psf.gibson_lanni]); beyond, a square root is
    imaginary (the ray is evanescent in that layer) and NaN is returned.
    The pupil itself is ρ ≤ a = min(1, n_min/NA).

    !!! references "References"

        1. Gibson SF, Lanni F. *Experimental test of an analytical model
           of aberration in an oil-immersion objective lens used in
           three-dimensional light microscopy.* J Opt Soc Am A
           **8**(10):1601–1613, 1991.
           [doi:10.1364/JOSAA.8.001601](https://doi.org/10.1364/JOSAA.8.001601)

        ??? quote "BibTeX"

            ```bibtex
            @article{gibson1991experimental,
              author  = {Gibson, Sarah Frisken and Lanni, Frederick},
              title   = {Experimental test of an analytical model of
                         aberration in an oil-immersion objective lens
                         used in three-dimensional light microscopy},
              journal = {Journal of the Optical Society of America A},
              volume  = {8},
              number  = {10},
              pages   = {1601--1613},
              year    = {1991},
              doi     = {10.1364/JOSAA.8.001601}
            }
            ```

    Parameters
    ----------
    rho : float or Tensor
        Normalised pupil radius ρ = n sin θ / NA (1 at the edge of the
        aperture of the objective), |ρ| ≤ n_min/NA.
    defocus : float or Tensor
        Δz; broadcast against `rho`.
    **optics
        `numerical_aperture` (required), `depth`, refractive indices and
        thicknesses, as in [`gibson_lanni`][optix.psf.gibson_lanni].

    Returns
    -------
    Tensor
        OPD, in the units of the lengths (µm), in float64 (float32 on
        MPS); NaN where |ρ| > n_min/NA.
    """
    _check_optics_keys(optics)
    p = _optics(**optics)
    rho = torch.as_tensor(rho)
    hi = _hi(rho.device)
    rho = rho.to(hi)
    defocus = torch.as_tensor(defocus, dtype=hi, device=rho.device)
    out = _opd(rho, defocus, p)
    outside = (rho.abs() * p.na > p.n_min).expand_as(out)
    return out.masked_fill(outside, math.nan)


# ----------------------------------------------------------------------
# Evaluation of the pupil integral
# ----------------------------------------------------------------------


def _pupil_nodes(
    n: int, p: _Optics, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Gauss–Legendre nodes ρ_q and weights w_q, ∑ w_q f(ρ_q) ≈ ∫₀^a f ρ dρ.

    The pupil radius is parametrised by the ray angle in the layer of
    lowest index, ρ = (n_min/NA) sin θ, which removes the square-root
    singularity of C_min at ρ = n_min/NA when the aperture is truncated
    there (and its near-singularity when NA is close to n_min).
    """
    hi = _hi(device)
    x, w = gauss_legendre(int(n), dtype=torch.float64, device="cpu")
    c = p.n_min / p.na
    theta_max = math.asin(min(1.0, p.na / p.n_min))
    theta = 0.5 * theta_max * (x + 1)
    w = 0.5 * theta_max * w
    rho = c * torch.sin(theta)
    weight = w * c * c * torch.sin(theta) * torch.cos(theta)
    return rho.to(device, hi), weight.to(device, hi)


def _quadrature_amplitude(
    radius: Tensor,
    defocus: Tensor,
    p: _Optics,
    k0: float,
    n: int,
    dtype: torch.dtype,
    budget: int,
) -> tuple[Tensor, Tensor]:
    """Re and Im (Z, R) of 2 ∫₀^a J₀(k₀ NA r ρ) exp(i k₀ OPD) ρ dρ."""
    device = radius.device
    hi = _hi(device)
    rho, weight = _pupil_nodes(n, p, device)
    phase = k0 * _opd(rho, defocus.to(hi).unsqueeze(-1), p)  # (Z, Q)
    cos = (2 * weight * torch.cos(phase)).to(dtype)
    sin = (2 * weight * torch.sin(phase)).to(dtype)
    scale = k0 * p.na * rho  # (Q,)
    itemsize = torch.finfo(hi).bits // 8
    step = max(1, budget // (3 * itemsize * len(rho)))
    re, im = [], []
    for start in range(0, len(radius), step):
        r = radius[start : start + step].to(hi)
        # argument in high precision, J₀ in the working dtype
        bessel = bessel_j0((r.unsqueeze(-1) * scale).to(dtype))
        re.append(cos @ bessel.T)
        im.append(sin @ bessel.T)
    return torch.cat(re, -1), torch.cat(im, -1)


def _auto_nodes(
    radius_max: float, defocus: Tensor, p: _Optics, k0: float, rtol: float
) -> int:
    """Smallest power-of-two node count whose result changes by ≤ `rtol`.

    The amplitude is computed on a probe set (17 radii from 0 to the
    largest radius; every defocus value, or 257 of them evenly spaced in
    index, including both ends, when there are more) with
    n and 2n nodes, doubling n from 32 until the largest change, relative
    to the largest amplitude, is at most `rtol`.
    """
    cpu = torch.device("cpu")
    radius = torch.linspace(0, radius_max, 17, dtype=torch.float64)
    z = defocus.detach().to(cpu, torch.float64).flatten()
    if len(z) > 257:
        z = z[torch.linspace(0, len(z) - 1, 257).round().long()]
    budget = 64 * 2**20
    n = 32
    prev = _quadrature_amplitude(radius, z, p, k0, n, torch.float64, budget)
    while 2 * n <= _MAX_NODES:
        cur = _quadrature_amplitude(
            radius, z, p, k0, 2 * n, torch.float64, budget
        )
        peak = torch.hypot(*cur).max()
        err = torch.hypot(cur[0] - prev[0], cur[1] - prev[1]).max()
        n *= 2
        if err <= rtol * peak:
            return n
        prev = cur
    warnings.warn(
        f"Gibson–Lanni quadrature did not reach rtol={rtol:g} with "
        f"{_MAX_NODES} nodes; the PSF may be inaccurate",
        RuntimeWarning,
        stacklevel=3,
    )
    return n


def _lommel(sigma: Tensor, beta: Tensor, a: float) -> Tensor:
    """∫₀^a ρ J₀(σρ) J₀(βρ) dρ (Li et al. Eq. 3), broadcast.

    For σ = β the integral is (a²/2) [J₀(σa)² + J₁(σa)²]. The general
    form loses ~ε/(|σ − β| a) to cancellation (ε of the dtype), and the
    σ = β form taken at the midpoint (σ + β)/2 is off by
    O((σ − β)² a²) relative (the integral is symmetric in σ and β), so
    the switch is at |σ − β| a = ε^⅓, where both are ~ε^⅔.
    """
    j0, j1 = bessel_j0, bessel_j1
    sa, ba = sigma * a, beta * a
    num = sigma * j1(sa) * j0(ba) - beta * j0(sa) * j1(ba)
    diff = sigma - beta
    near = diff.abs() * a < torch.finfo(diff.dtype).eps ** (1 / 3)
    den = torch.where(near, torch.ones_like(diff), diff) * (sigma + beta)
    out = a * num / den
    mid = 0.5 * (sa + ba)
    same = 0.5 * a * a * (j0(mid) ** 2 + j1(mid) ** 2)
    return torch.where(near, same, out)


def _li2017_amplitude(
    radius: Tensor,
    defocus: Tensor,
    p: _Optics,
    k0: float,
    wavelength: float,
    basis: int,
    samples: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Bessel-series amplitude of Li et al. (2017), Eqs. (3)–(5).

    The fit runs in float64 on the CPU; the closed-form integrals and
    their product with the coefficients in float64 on the working
    device, or on the CPU when the device has no float64 (MPS).
    """
    device = radius.device
    f64 = device if _hi(device) == torch.float64 else torch.device("cpu")
    a = p.aperture
    rho = torch.linspace(0, a, samples, dtype=torch.float64)
    phase = k0 * _opd(rho, defocus.detach().cpu().double().unsqueeze(-1), p)
    target = torch.cat([torch.cos(phase), torch.sin(phase)], 0).T  # (K, 2Z)
    m = torch.arange(1, basis + 1, dtype=torch.float64)
    sigma = p.na * (3 * m - 2) * LI2017_WAVELENGTH / wavelength
    design = bessel_j0(rho.unsqueeze(-1) * sigma)  # (K, M)
    coef = torch.linalg.lstsq(design, target, driver="gelsd").solution
    coef = coef.to(f64)  # (M, 2Z)
    beta = k0 * p.na * radius.to(f64, torch.float64)
    basis_r = _lommel(sigma.to(f64), beta.unsqueeze(-1), a)  # (R, M)
    amp = 2 * (basis_r @ coef).T.to(device, dtype)  # (2Z, R)
    nz = len(defocus)
    return amp[:nz], amp[nz:]


def _profile(
    radius: Tensor,
    defocus: Tensor,
    p: _Optics,
    wavelength: float,
    *,
    method: str,
    nodes: int | None,
    rtol: float,
    basis: int,
    samples: int,
    dtype: torch.dtype,
) -> tuple[Tensor, dict]:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if not wavelength > 0:
        raise ValueError(f"wavelength must be > 0, got {wavelength}")
    k0 = 2 * math.pi / wavelength
    info: dict[str, object] = {"method": method, "aperture": p.aperture}
    if method == "quadrature":
        if nodes is None:
            rmax = float(radius.max()) if radius.numel() else 0.0
            nodes = _auto_nodes(rmax, defocus, p, k0, rtol)
        if int(nodes) < 1:
            raise ValueError(f"nodes must be >= 1, got {nodes}")
        info["nodes"] = int(nodes)
        budget = get_config().memory
        re, im = _quadrature_amplitude(
            radius, defocus, p, k0, int(nodes), dtype, budget
        )
    else:
        if int(basis) < 1 or int(samples) < int(basis):
            raise ValueError("Need 1 <= basis <= samples")
        info["basis"], info["samples"] = int(basis), int(samples)
        re, im = _li2017_amplitude(
            radius, defocus, p, k0, wavelength, int(basis), int(samples), dtype
        )
    return re * re + im * im, info


def gibson_lanni_profile(
    radius: float | Tensor,
    defocus: float | Tensor = 0.0,
    *,
    wavelength: float,
    method: str = "quadrature",
    nodes: int | None = None,
    rtol: float = 1e-6,
    basis: int = 100,
    samples: int = 1000,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
    **optics: float | None,
) -> Tensor:
    """Gibson–Lanni intensity at radii r and defocus values Δz.

        I(r, Δz) = | 2 ∫₀^a J₀(k₀ NA r ρ) exp(i k₀ OPD(ρ; Δz)) ρ dρ |²,

    the Kirchhoff integral of Gibson & Lanni Eq. (5) with the
    [`gibson_lanni_opd`][optix.psf.utils.gibson_lanni_opd], in object
    space, k₀ = 2π/λ₀. I = 1 at the focus of an aberration-free system
    with a = 1.
    See [`gibson_lanni`][optix.psf.gibson_lanni] for the methods and
    the parameters.

    !!! references "References"

        1. Gibson SF, Lanni F. *Experimental test of an analytical model
           of aberration in an oil-immersion objective lens used in
           three-dimensional light microscopy.* J Opt Soc Am A
           **8**(10):1601–1613, 1991.
           [doi:10.1364/JOSAA.8.001601](https://doi.org/10.1364/JOSAA.8.001601)
        2. Li J, Xue F, Blu T. *Fast and accurate three-dimensional point
           spread function computation for fluorescence microscopy.*
           J Opt Soc Am A **34**(6):1029–1034, 2017.
           [doi:10.1364/JOSAA.34.001029](https://doi.org/10.1364/JOSAA.34.001029)

        ??? quote "BibTeX"

            ```bibtex
            @article{gibson1991experimental,
              author  = {Gibson, Sarah Frisken and Lanni, Frederick},
              title   = {Experimental test of an analytical model of
                         aberration in an oil-immersion objective lens
                         used in three-dimensional light microscopy},
              journal = {Journal of the Optical Society of America A},
              volume  = {8},
              number  = {10},
              pages   = {1601--1613},
              year    = {1991},
              doi     = {10.1364/JOSAA.8.001601}
            }
            @article{li2017fast,
              author  = {Li, Jizhou and Xue, Feng and Blu, Thierry},
              title   = {Fast and accurate three-dimensional point spread
                         function computation for fluorescence microscopy},
              journal = {Journal of the Optical Society of America A},
              volume  = {34},
              number  = {6},
              pages   = {1029--1034},
              year    = {2017},
              doi     = {10.1364/JOSAA.34.001029}
            }
            ```

    Parameters
    ----------
    radius : float or Tensor
        (R,) lateral distances r from the source, in µm.
    defocus : float or Tensor
        (Z,) defocus values Δz, in µm.
    wavelength : float
        Emission wavelength λ₀ in vacuum, in µm.
    method, nodes, rtol, basis, samples
        See [`gibson_lanni`][optix.psf.gibson_lanni].
    dtype, device : optional
        Working dtype and device. Default: configuration.
    **optics
        `numerical_aperture` (required), `depth`, refractive indices and
        thicknesses, as in [`gibson_lanni`][optix.psf.gibson_lanni].

    Returns
    -------
    (Z, R) Tensor
        Intensity.
    """
    _check_optics_keys(optics)
    p = _optics(**optics)
    backend = resolve_backend(dtype, device)
    hi = _hi(backend["device"])
    radius = torch.as_tensor(radius, dtype=hi, device=backend["device"])
    defocus = torch.as_tensor(defocus, dtype=hi, device=backend["device"])
    out, _ = _profile(
        radius.flatten(),
        defocus.flatten(),
        p,
        float(wavelength),
        method=method,
        nodes=nodes,
        rtol=rtol,
        basis=basis,
        samples=samples,
        dtype=backend["dtype"],
    )
    return out


# ----------------------------------------------------------------------
# Sampling on a voxel grid
# ----------------------------------------------------------------------


def _triple(value: object, name: str, cast: type) -> list:
    if isinstance(value, int | float):
        return [cast(value)] * 3
    out = [cast(v) for v in value]  # type: ignore[union-attr]
    if len(out) == 2:
        out = [out[0], out[1], out[1]]
    if len(out) != 3:
        raise ValueError(f"{name} needs 1, 2 or 3 entries, got {len(out)}")
    return out


def _half_steps(n: int, s: int) -> Tensor:
    """Sub-voxel positions in units of step / (2 s), 0 at index n // 2.

    Voxel i is split into s samples at (i − n//2) + (j + ½)/s − ½.
    """
    i = torch.arange(n).unsqueeze(-1)
    j = torch.arange(s)
    return (2 * s * (i - n // 2) + 2 * j + 1 - s).flatten()


def gibson_lanni(
    shape: Sequence[int],
    *,
    spacing: float | Sequence[float],
    wavelength: float,
    numerical_aperture: float,
    depth: float = 0.0,
    specimen_index: float = 1.33,
    coverslip_index: float | None = None,
    coverslip_thickness: float | None = None,
    immersion_index: float | None = None,
    design_coverslip_index: float = 1.522,
    design_coverslip_thickness: float = 170.0,
    design_immersion_index: float = 1.515,
    design_immersion_thickness: float = 150.0,
    defocus: float = 0.0,
    method: str = "quadrature",
    nodes: int | None = None,
    rtol: float = 1e-6,
    basis: int = 100,
    samples: int = 1000,
    oversample: int | Sequence[int] = 1,
    return_estimate: bool = False,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor | PSFEstimate:
    """Gibson–Lanni scalar PSF of a widefield microscope, on a voxel grid.

    The PSF of an objective used outside its design conditions: the
    source lies at a depth t_s in a specimen of index n_s, below a
    coverslip and an immersion layer whose indices and thicknesses may
    differ from the design ones (Gibson & Lanni 1991, Fig. 1). The
    intensity at lateral distance r from the source and defocus Δz is the
    Kirchhoff integral (Eq. 5),

        I(r, Δz) ∝ | ∫₀^a J₀(k₀ NA r ρ) exp(i k₀ OPD(ρ; Δz)) ρ dρ |²,

    with k₀ = 2π/λ₀ the wave number in vacuum and OPD the optical path
    difference between the actual and the design rays (Eq. 4, see
    [`gibson_lanni_opd`][optix.psf.utils.gibson_lanni_opd]):

        OPD(ρ; Δz) = n_oil Δz C_oil(ρ)
                   + ∑ₗ ± nₗ tₗ [Cₗ(ρ) − (n_oil/nₗ)² C_oil(ρ)],
        C_n(ρ) = [1 − (NA ρ / n)²]^½,

    summed over the specimen (t_s, n_s), the actual coverslip (+) and the
    design coverslip and immersion layer (−). Δz is the defocus of
    Eq. (3), measured in object space as the stage movement from the
    paraxial focus of the source (Δz = 0, where the ρ² term of the OPD
    vanishes). Δz > 0 is a thicker immersion layer (objective moved away
    from the coverslip): the focal plane then lies between the source and
    the objective.

    The detector is at its design distance (z_d = z_d*), so the
    tube-length terms of Eq. (4) vanish. (As printed, the separate
    detector term of Eq. (4) has n_oil in its denominator, which would
    leave a ρ² term at Δz = 0, against the text of the paper; the term
    that cancels it is a²ρ²(z_d* − z_d)/(2 z_d* z_d). With z_d = z_d*
    both vanish, so the PSF computed here does not depend on this.) The
    lateral coordinate is in object space: Gibson & Lanni's
    J₀(k₀ a ρ r_d / z_d) on the detector equals J₀(k₀ NA ρ r) with
    r = r_d / M up to a factor [1 − (NA/M)²]^−½, taken as 1 (as in Li et
    al.). Since a/z_d is tan θ′ in image space and the sine condition
    gives sin θ′ = NA/M, the factor 1 is the exact sine-condition form,
    not a further approximation. The constant amplitude C / z_d is
    dropped: the PSF is normalised to unit sum.

    **Aperture.** The upper limit is a = 1, or a = n_min / NA when the
    numerical aperture exceeds n_min, the smallest refractive index of
    the layers of non-zero thickness: the actual immersion layer, and
    the specimen, the actual and design coverslips and the design
    immersion layer when their thickness is > 0. Beyond ρ = n_min/NA the
    ray is evanescent in that layer and the model's square roots stop
    being real; Gibson & Lanni do not specify the truncation (their
    design system has no specimen layer, and its PSF integrates over the
    whole aperture). Li et al. truncate at the smallest index of all
    layers, the specimen included at any depth; here a layer of zero
    thickness, which has no term in the OPD, does not truncate, so that
    with `depth=0` an NA 1.4 oil objective has its full aperture,
    whatever `specimen_index`. For t_s > 0 the super-critical rays are
    dropped (they would be attenuated as
    exp(−k₀ t_s √((NA ρ)² − n_s²))), the large-depth limit; the
    aperture therefore jumps from 1 to n_s/NA as the depth leaves 0. The
    value of a is `meta["aperture"]` of the returned estimate.

    **Evaluation.** With `method="quadrature"` (default) the 1-D pupil
    integral is computed for each (r, Δz) by Gauss–Legendre quadrature
    (as in Gibson & Lanni) over the ray angle θ in the layer of lowest
    index, ρ = (n_min/NA) sin θ, which keeps the integrand smooth up to
    the edge of the aperture. Without `nodes`, the number of nodes is
    doubled from 32 until the amplitude, evaluated with n and 2n nodes on
    a probe set, changes by at most `rtol` relative to its peak. The
    probe set is 17 radii evenly spaced from 0 to the largest radius of
    the grid, and the defocus values of the grid, or 257 of them evenly
    spaced in index (both ends included) when there are more.
    Oscillations at large r (Bessel function) or large defocus or depth
    (phase) need more nodes. The phase is computed in float64 and each
    square root as √(n² − s²) − n = −s² / (√(n² − s²) + n), so large
    constant optical paths never cancel. The integral is one matrix
    product per chunk of radii, [cos(k₀ OPD), sin(k₀ OPD)] · J₀ᵀ, on the
    working device.

    With `method="li2017"` the phase factor is expanded on M rescaled
    Bessel functions (Li et al. Eq. 4),
    exp(i k₀ OPD(ρ)) ≈ ∑ₘ cₘ J₀(σₘ ρ), σₘ = NA (3m − 2) λ_ref/λ₀ with
    the reference wavelength λ_ref = 0.436 µm of Li et al. (their λ₀),
    fitted by least squares on K samples ρ_k uniform in [0, a], so that
    the integral is the closed form

        ∫₀^a ρ J₀(σρ) J₀(βρ) dρ
            = a [σ J₁(σa) J₀(βa) − β J₀(σa) J₁(βa)] / (σ² − β²),

    β = k₀ NA r (Eqs. 3 and 5), and (a²/2)[J₀(σa)² + J₁(σa)²] for σ = β
    (printed as (a²/2) J₁(σa)² under Eq. 3 of Li et al., which lacks the
    J₀² term). The coefficients are found once for all Δz with an SVD
    least-squares solver, equal to (JᵀJ)⁻¹JᵀF in exact arithmetic. Li et
    al. report a relative squared error RSE ≈ (M/45)^−6.5 (K/80)^−1.5 %
    against the exact model for 30 ≤ M ≤ 100, 50 ≤ K ≤ 1000. Both
    methods evaluate each voxel at its exact radius (Li et al.
    interpolate a 2× oversampled radial profile).

    !!! warning "Depth error of `li2017`"
        When NA ≥ n_s and the source is below the surface, the aperture
        ends at ρ = a = n_s/NA, where the specimen term of the phase
        varies as k₀ t_s n_s √(1 − (ρ/a)²): its slope is infinite at the
        edge, and the uniform samples of Li et al. cannot follow it. The
        fit is aliased there, whatever M, and the error grows with the
        depth. It is inherent to the published method (MicroscPSF has
        the same), and it is reduced by `samples` (K), not by `basis`
        (M). For NA 1.4, λ₀ = 0.61 µm, a source in water at t_s = 10 µm
        (r ≤ 3.2 µm, |Δz| ≤ 4 µm, M = 100), the largest error is about
        7·10⁻³ of the peak with K = 1000, 10⁻³ with K = 3000 and 2·10⁻⁴
        with K = 10000, and about 4·10⁻² at 50 µm with K = 1000; with
        NA 1.2 it is 10⁻¹³ at 10 µm. Raise `samples` with the depth, or
        use the quadrature.

    **Sampling.** Voxel (k, j, i) is at Δz = `defocus` + (k − K_z//2) d_z
    and lateral offset ((j − K_y//2) d_y, (i − K_x//2) d_x) from the
    source: the PSF is centred at index `K // 2`, the convention of
    `optix.deconv`. Slices are in order of increasing Δz along axis 0,
    i.e. of the objective moving away from the specimen; flip axis 0 for
    a stack acquired with the focus moving deeper. Point sampling is the
    model; `oversample` > 1 instead averages s points per axis inside
    each voxel (integration over the voxel; an extension, not in Gibson
    & Lanni).

    **Relation to the z of Li et al.** Li et al., MicroscPSF and PSF
    Generator scan the actual immersion thickness t_oil = t_oil* + z, so
    their z is the stage position from the design one (focus at the
    coverslip), not the defocus from the paraxial focus of the source.
    By Eq. (3),

        Δz = z + t_oil* + n_oil (t_s/n_s + t_g/n_g − t_g*/n_g*
                                 − t_oil*/n_oil*),

    which is Δz = z + n_oil t_s/n_s when the coverslip and the immersion
    medium match the design.

    !!! references "References"

        1. Gibson SF, Lanni F. *Experimental test of an analytical model
           of aberration in an oil-immersion objective lens used in
           three-dimensional light microscopy.* J Opt Soc Am A
           **8**(10):1601–1613, 1991.
           [doi:10.1364/JOSAA.8.001601](https://doi.org/10.1364/JOSAA.8.001601)
        2. Li J, Xue F, Blu T. *Fast and accurate three-dimensional point
           spread function computation for fluorescence microscopy.*
           J Opt Soc Am A **34**(6):1029–1034, 2017.
           [doi:10.1364/JOSAA.34.001029](https://doi.org/10.1364/JOSAA.34.001029)

        ??? quote "BibTeX"

            ```bibtex
            @article{gibson1991experimental,
              author  = {Gibson, Sarah Frisken and Lanni, Frederick},
              title   = {Experimental test of an analytical model of
                         aberration in an oil-immersion objective lens
                         used in three-dimensional light microscopy},
              journal = {Journal of the Optical Society of America A},
              volume  = {8},
              number  = {10},
              pages   = {1601--1613},
              year    = {1991},
              doi     = {10.1364/JOSAA.8.001601}
            }
            @article{li2017fast,
              author  = {Li, Jizhou and Xue, Feng and Blu, Thierry},
              title   = {Fast and accurate three-dimensional point spread
                         function computation for fluorescence microscopy},
              journal = {Journal of the Optical Society of America A},
              volume  = {34},
              number  = {6},
              pages   = {1029--1034},
              year    = {2017},
              doi     = {10.1364/JOSAA.34.001029}
            }
            ```

    Parameters
    ----------
    shape : sequence of int
        (K_z, K_y, K_x) grid shape.
    spacing : float or sequence of float
        Voxel size (d_z, d_y, d_x) in object space, in µm; two values
        mean (d_z, d_xy).
    wavelength : float
        Emission wavelength λ₀ in vacuum, in µm.
    numerical_aperture : float
        NA of the objective.
    depth : float
        t_s, depth of the source below the coverslip, in µm.
    specimen_index : float
        n_s, refractive index of the specimen.
    coverslip_index, coverslip_thickness : float, optional
        Actual n_g and t_g (µm). Default: the design values.
    immersion_index : float, optional
        Actual n_oil. Default: the design value. The actual immersion
        thickness t_oil is set by the defocus (Eq. 3).
    design_coverslip_index, design_coverslip_thickness : float
        n_g* and t_g* (µm), defaults from the experiments of Gibson &
        Lanni (1.522, 170 µm).
    design_immersion_index : float
        n_oil*, default from Gibson & Lanni (1.515).
    design_immersion_thickness : float
        t_oil* (µm), the working distance. It only matters when
        `immersion_index` differs from the design. Not specified in
        Gibson & Lanni; the default, 150 µm, is the value of Li et al.
    defocus : float
        Δz of the central slice `K_z // 2`, in µm.
    method : {"quadrature", "li2017"}
        Evaluation of the pupil integral (see above).
    nodes : int, optional
        Number of Gauss–Legendre nodes. Default: chosen to meet `rtol`.
    rtol : float
        Accuracy target of the automatic node count, relative to the
        peak amplitude (author choice; not specified in Gibson & Lanni).
    basis, samples : int
        M and K of `method="li2017"`; defaults M = 100, K = 1000 as in
        Li et al. Fig. 3.
    oversample : int or sequence of int
        Samples per voxel along (z, y, x); 1 = point sampling.
    return_estimate : bool
        Return a [`PSFEstimate`][optix.estimates.PSFEstimate] (with the
        FWHM, in voxels, and the model parameters in `meta`) instead of
        the tensor. `meta` also holds `"method"`, `"aperture"` (the upper
        limit a of ρ, 1 or n_min/NA; see above), and `"nodes"` (the
        number of quadrature nodes, chosen or given) or `"basis"` and
        `"samples"` (`li2017`).
    dtype, device : optional
        Working dtype and device. Default: configuration.

    Returns
    -------
    (K_z, K_y, K_x) Tensor or PSFEstimate
        PSF normalised to unit sum, centred at `K // 2`.
    """
    optics = {
        "numerical_aperture": numerical_aperture,
        "depth": depth,
        "specimen_index": specimen_index,
        "coverslip_index": coverslip_index,
        "coverslip_thickness": coverslip_thickness,
        "immersion_index": immersion_index,
        "design_coverslip_index": design_coverslip_index,
        "design_coverslip_thickness": design_coverslip_thickness,
        "design_immersion_index": design_immersion_index,
        "design_immersion_thickness": design_immersion_thickness,
    }
    p = _optics(**optics)
    shape = [int(n) for n in shape]
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError(f"shape must be 3 positive ints, got {shape}")
    step = _triple(spacing, "spacing", float)
    if min(step) <= 0:
        raise ValueError(f"spacing must be > 0, got {spacing}")
    over = _triple(oversample, "oversample", int)
    if min(over) < 1:
        raise ValueError(f"oversample must be >= 1, got {oversample}")
    backend = resolve_backend(dtype, device)
    dev, out_dtype = backend["device"], backend["dtype"]
    hi = _hi(dev)

    # Lateral sample positions, in half-steps; unique radii by integer key
    qy = _half_steps(shape[1], over[1]).abs()
    qx = _half_steps(shape[2], over[2]).abs()
    hy, hx = step[1] / (2 * over[1]), step[2] / (2 * over[2])
    if hy == hx:
        key = qy.unsqueeze(-1) ** 2 + qx**2
    else:
        key = qy.unsqueeze(-1) * (int(qx.max()) + 1) + qx
    key, inverse = torch.unique(key, return_inverse=True)
    if hy == hx:
        radius = hy * key.double().sqrt()
    else:
        ky, kx = key // (int(qx.max()) + 1), key % (int(qx.max()) + 1)
        radius = torch.hypot(hy * ky.double(), hx * kx.double())
    radius = radius.to(dev, hi)
    inverse = inverse.to(dev)
    qz = _half_steps(shape[0], over[0])
    z = float(defocus) + qz.double() * (step[0] / (2 * over[0]))
    z = z.to(dev, hi)

    prof, info = _profile(
        radius,
        z,
        p,
        float(wavelength),
        method=method,
        nodes=nodes,
        rtol=rtol,
        basis=basis,
        samples=samples,
        dtype=out_dtype,
    )  # (Z·s_z, R)

    sz, sy, sx = over
    ny, nx = shape[1] * sy, shape[2] * sx
    per_slice = ny * nx * torch.finfo(out_dtype).bits // 8
    chunk = max(1, get_config().memory // max(1, per_slice * sz))
    psf = torch.empty(shape, **backend)
    for start in range(0, shape[0], chunk):
        stop = min(shape[0], start + chunk)
        block = prof[start * sz : stop * sz][:, inverse]  # (zc·sz, ny, nx)
        block = block.view(stop - start, sz, shape[1], sy, shape[2], sx)
        psf[start:stop] = block.mean((1, 3, 5))
    psf = psf / psf.sum()
    if not return_estimate:
        return psf
    meta: dict[str, object] = {
        "units": "voxels",
        "psf_center": "K // 2",
        "model": "gibson_lanni",
        "wavelength": float(wavelength),
        "spacing": step,
        "defocus": float(defocus),
        "oversample": over,
        **{k_: v for k_, v in optics.items() if v is not None},
        **info,
    }
    return PSFEstimate(psf=psf, fwhm=fwhm(psf), meta=meta)
