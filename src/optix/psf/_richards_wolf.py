"""Vectorial focal field of an aplanatic system (Richards & Wolf, 1959)."""

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import Tensor

from .._config import get_config, resolve_backend
from ..special import bessel_j012

__all__ = [
    "richards_wolf",
    "richards_wolf_field",
    "richards_wolf_integrals",
]


# Automatic number of Gauss-Legendre nodes:
#     _NODES_MIN + ⌈_NODES_SLOPE · Φ + _NODES_EDGE · (k r sin α)^⅓⌉,
# with Φ the largest phase excursion of the integrand over the aperture
# (k r sin α + |kz| (1 − cos α) + the total variation of ψ). The cube-root
# term is the transition margin of J_n(k r sin θ) near θ = 90°, where the
# Bessel argument behaves as 2 k r √(1 − s) in s = √cos θ.
_NODES_MIN = 24
_NODES_SLOPE = 0.5
_NODES_EDGE = 4.0
# Samples of ψ on [0, α] used to measure its total variation.
_PHASE_SAMPLES = 4097
# Complex work arrays per (point, node, Bessel sample) in the quadrature:
# the Bessel argument, its cosine and sine and two products.
_WORK_ARRAYS = 5
_BESSEL_SAMPLES = 17

PupilPhase = Callable[[Tensor], Tensor]


# --------------------------------------------------------------------------
# Quadrature
# --------------------------------------------------------------------------


def _gauss_legendre(
    n: int, lower: float, upper: float, backend: dict
) -> tuple[Tensor, Tensor]:
    """Gauss-Legendre nodes and weights on [lower, upper]."""
    x, w = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (upper - lower)
    nodes = torch.as_tensor(lower + half * (x + 1), dtype=torch.float64)
    weights = torch.as_tensor(half * w, dtype=torch.float64)
    return nodes.to(**backend), weights.to(**backend)


def _eval_phase(pupil_phase: PupilPhase, theta: Tensor) -> Tensor:
    """ψ(θ), checked to be real, finite and broadcastable to θ."""
    psi = pupil_phase(theta)
    psi = torch.as_tensor(psi, device=theta.device)
    if psi.is_complex():
        raise TypeError(
            "pupil_phase must return a real phase ψ(θ), in radians, got a "
            "complex tensor"
        )
    try:
        shape = torch.broadcast_shapes(psi.shape, theta.shape)
    except RuntimeError:
        shape = None
    if shape != theta.shape:
        raise ValueError(
            "pupil_phase must return a tensor broadcastable to the shape "
            f"of θ, {tuple(theta.shape)}, got {tuple(psi.shape)}"
        )
    psi = psi.to(theta.dtype).expand(theta.shape)
    if not bool(torch.isfinite(psi).all()):
        raise ValueError("pupil_phase returned non-finite values")
    return psi


def _phase_variation(
    aperture: float, pupil_phase: PupilPhase | None, backend: dict
) -> float:
    """Total variation of ψ(θ) on [0, α], on a dense grid of θ."""
    if pupil_phase is None:
        return 0.0
    theta = torch.linspace(0, aperture, _PHASE_SAMPLES, **backend)
    psi = _eval_phase(pupil_phase, theta)
    return float((psi[1:] - psi[:-1]).abs().sum())


def _auto_nodes(
    aperture: float, radial: float, axial: float, variation: float = 0.0
) -> int:
    """Number of nodes for the largest k r and |kz| of a request, and the
    total variation of ψ."""
    lateral = radial * math.sin(aperture)
    phase = lateral + axial * (1 - math.cos(aperture)) + variation
    extra = _NODES_SLOPE * phase + _NODES_EDGE * lateral ** (1 / 3)
    return _NODES_MIN + math.ceil(extra)


def _chunk_size(nodes: int, dtype: torch.dtype) -> int:
    """Number of points per chunk of the quadrature, from the budget."""
    itemsize = torch.finfo(dtype).bits // 8
    per_point = _WORK_ARRAYS * itemsize * nodes * _BESSEL_SAMPLES
    return max(1, get_config().memory // per_point)


def _check_aperture(aperture: float) -> float:
    aperture = float(aperture)
    if not 0 < aperture <= math.pi / 2 + 1e-12:
        raise ValueError(
            f"The aperture half-angle must be in (0, π/2], got {aperture}"
        )
    return min(aperture, math.pi / 2)


def _pupil(
    aperture: float,
    nodes: int,
    pupil_phase: PupilPhase | None,
    backend: dict,
) -> dict[str, Tensor]:
    """Quadrature in s = √cos θ, with the weights of I₀, I₁ and I₂.

    With cos θ = s², sin θ dθ = −2 s ds and the integrands of (2.32)
    become 2 s² (1 + s²) J₀, 2 s² sin θ J₁ and 2 s² (1 − s²) J₂, all
    analytic in s on [√cos α, 1]: J₀(b sin θ), sin θ J₁(b sin θ) and
    J₂(b sin θ) are even in sin θ, hence functions of sin²θ = 1 − s⁴.
    The √cos θ apodisation is absorbed by the change of variable, so
    Gauss-Legendre converges exponentially up to α = 90°.
    """
    lower = math.sqrt(max(math.cos(aperture), 0.0))
    s, w = _gauss_legendre(nodes, lower, 1.0, backend)
    cos = s * s
    sin = torch.sqrt((1 - cos * cos).clamp_min(0))
    base = 2 * cos * w
    out = {
        "cos": cos,
        "sin": sin,
        "g0": base * (1 + cos),
        "g1": base * sin,
        "g2": base * (1 - cos),
    }
    if pupil_phase is not None:
        theta = torch.acos(cos.clamp(-1, 1))
        out["psi"] = _eval_phase(pupil_phase, theta)
    else:
        out["psi"] = torch.zeros_like(cos)
    return out


# --------------------------------------------------------------------------
# The integrals I₀, I₁, I₂
# --------------------------------------------------------------------------


def richards_wolf_integrals(
    radial: Tensor | float,
    axial: Tensor | float,
    aperture: float,
    *,
    nodes: int | None = None,
    pupil_phase: PupilPhase | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """The integrals I₀, I₁, I₂ of Richards & Wolf (1959), Eq. (2.32).

        I₀ = ∫₀^α √cos θ sin θ (1 + cos θ) J₀(k r sin θ) e^{ikz cos θ} dθ,
        I₁ = ∫₀^α √cos θ sin²θ J₁(k r sin θ) e^{ikz cos θ} dθ,
        I₂ = ∫₀^α √cos θ sin θ (1 − cos θ) J₂(k r sin θ) e^{ikz cos θ} dθ,

    with α the angular semi-aperture on the image side, k = n k₀ the wave
    number in the image space (k₀ = 2π / λ₀, λ₀ the vacuum wavelength,
    n the refractive index), r = √(x² + y²) the distance from the axis
    and z the distance from the Gaussian focus along the axis, positive
    away from the exit pupil (k r = k r_P sin θ_P and kz = k r_P cos θ_P
    in the paper). In the optical coordinates of Eq. (3.1a),
    u = kz sin²α and v = k r sin α, these are the integrals of
    Eq. (3.2).

    The integrals are computed by Gauss-Legendre quadrature in the
    variable s = √cos θ, in which the three integrands are analytic on
    [√cos α, 1] (the √cos θ factor, singular at θ = 90°, is absorbed by
    the change of variable), so that the quadrature converges
    exponentially once the nodes resolve the oscillations of the
    integrand. Bessel functions are from
    [`bessel_j012`][optix.special.bessel_j012].

    !!! references "References"

        1. Richards B, Wolf E. *Electromagnetic diffraction in optical
           systems II. Structure of the image field in an aplanatic
           system.* Proc R Soc Lond A **253**(1274):358–379, 1959.
           [doi:10.1098/rspa.1959.0200](https://doi.org/10.1098/rspa.1959.0200)

        ??? quote "BibTeX"

            ```bibtex
            @article{richards1959electromagnetic,
              author  = {Richards, B. and Wolf, E.},
              title   = {Electromagnetic diffraction in optical systems
                         {II}. {Structure} of the image field in an
                         aplanatic system},
              journal = {Proceedings of the Royal Society of London.
                         Series A, Mathematical and Physical Sciences},
              volume  = {253},
              number  = {1274},
              pages   = {358--379},
              year    = {1959},
              doi     = {10.1098/rspa.1959.0200}
            }
            ```

    Parameters
    ----------
    radial : Tensor or float
        k r, the radial optical distance (≥ 0).
    axial : Tensor or float
        kz, the axial optical distance. Broadcast with `radial`.
    aperture : float
        Angular semi-aperture α, in radians, in (0, π/2].
    nodes : int, optional
        Number of quadrature nodes. Default:
        24 + ⌈½ Φ + 4 (max k r · sin α)^⅓⌉, with
        Φ = max k r · sin α + max |kz| · (1 − cos α) + TV(ψ) the largest
        phase excursion of the integrand and TV(ψ) the total variation
        of `pupil_phase` on [0, α] (measured on 4097 values of θ). This
        keeps the error near 10⁻¹² of I₀(0, 0) in float64, up to
        α = 90°, for a smooth ψ (see below).
    pupil_phase : callable, optional
        Extension (not in the aplanatic model of the paper): a phase
        ψ(θ) = kΦ(θ), in radians, multiplied into the integrands as
        e^{iψ(θ)}. Φ is the aberration function of Eq. (2.2), which the
        paper sets to zero for an aplanatic system (Eq. 2.7); ψ must
        depend on θ only (not on the azimuth), so that the azimuthal
        integration (2.29) still applies. Called with a tensor of polar
        angles θ, it returns a real tensor broadcastable to the shape of
        θ (a `TypeError` or `ValueError` is raised otherwise). It is a
        phase only: it carries an optical path difference (e.g. that of
        an index mismatch), but not amplitudes that differ between the s
        and p components, such as the Fresnel transmission coefficients
        of an interface, which enter I₀, I₁ and I₂ with different
        weights. The quadrature converges exponentially only if ψ is a
        smooth function of sin²θ (even in θ, e.g. a polynomial in cos θ
        or sin²θ); a term odd in θ, such as ∝ sin θ, makes the
        convergence algebraic, and `nodes` must then be raised by hand.

    Returns
    -------
    I0, I1, I2 : Tensor
        Complex tensors of the broadcast shape of `radial` and `axial`.
        The points are processed in chunks within the memory budget of
        `optix.config`: each point and node holds 17 Bessel samples in
        about five work arrays.
    """
    aperture = _check_aperture(aperture)
    radial = torch.as_tensor(radial)
    axial = torch.as_tensor(axial, device=radial.device)
    dtype = torch.promote_types(radial.dtype, axial.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    radial, axial = torch.broadcast_tensors(radial.to(dtype), axial.to(dtype))
    backend = {"dtype": dtype, "device": radial.device}
    if nodes is None:
        max_r = float(radial.abs().max()) if radial.numel() else 0.0
        max_z = float(axial.abs().max()) if axial.numel() else 0.0
        variation = _phase_variation(aperture, pupil_phase, backend)
        nodes = _auto_nodes(aperture, max_r, max_z, variation)
    nodes = int(nodes)
    q = _pupil(aperture, nodes, pupil_phase, backend)
    shape = radial.shape
    radial, axial = radial.reshape(-1), axial.reshape(-1)
    chunk = _chunk_size(nodes, dtype)
    out: list[list[Tensor]] = [[], [], []]
    # One pass at least, so that an empty request returns empty tensors.
    for start in range(0, max(radial.numel(), 1), chunk):
        r = radial[start : start + chunk, None]
        phase = axial[start : start + chunk, None] * q["cos"] + q["psi"]
        expo = torch.polar(torch.ones_like(phase), phase)
        js = bessel_j012(r * q["sin"])
        for n, (g, j) in enumerate(zip(("g0", "g1", "g2"), js, strict=True)):
            out[n].append((expo * (q[g] * j)).sum(-1))
    i0, i1, i2 = (torch.cat(o).reshape(shape) for o in out)
    return i0, i1, i2


def _grid_integrals(
    axial: Tensor,
    radial: Tensor,
    aperture: float,
    nodes: int,
    pupil_phase: PupilPhase | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """I₀, I₁, I₂ on the outer product (kz, k r), as matrix products.

    Iₙ(z, r) = ∑_q e^{i(kz cos θ_q + ψ_q)} · gₙ(θ_q) Jₙ(k r sin θ_q), i.e.
    a (Z, N) by (N, R) product, computed in chunks of R.
    """
    backend = {"dtype": radial.dtype, "device": radial.device}
    q = _pupil(aperture, nodes, pupil_phase, backend)
    phase = axial[:, None] * q["cos"] + q["psi"]
    cos_p, sin_p = torch.cos(phase), torch.sin(phase)  # (Z, N)
    # The Bessel evaluation holds a few (N, chunk, 17) work arrays.
    chunk = _chunk_size(nodes, radial.dtype)
    out: list[list[Tensor]] = [[], [], []]
    for start in range(0, radial.numel(), chunk):
        r = radial[start : start + chunk]
        js = bessel_j012(q["sin"][:, None] * r)  # (N, R) each
        for n, (g, j) in enumerate(zip(("g0", "g1", "g2"), js, strict=True)):
            b = q[g][:, None] * j
            out[n].append(torch.complex(cos_p @ b, sin_p @ b))
    return tuple(torch.cat(o, dim=-1) for o in out)  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------


def _shape_spacing(
    shape: Sequence[int], spacing: float | Sequence[float]
) -> tuple[tuple[int, int, int], list[float], int]:
    shape = tuple(int(n) for n in shape)
    if len(shape) not in (2, 3):
        raise ValueError(
            f"shape must be (Y, X) or (Z, Y, X), got {len(shape)} dims"
        )
    ndim = len(shape)
    if isinstance(spacing, int | float):
        step = [float(spacing)] * ndim
    else:
        step = [float(s) for s in spacing]
        if len(step) != ndim:
            raise ValueError(f"spacing needs {ndim} entries, got {len(step)}")
    if ndim == 2:
        shape = (1, *shape)
        step = [1.0, *step]
    return shape, step, ndim  # type: ignore[return-value]


def _optics(
    wavelength: float, numerical_aperture: float, refractive_index: float
) -> tuple[float, float]:
    if wavelength <= 0:
        raise ValueError(f"wavelength must be positive, got {wavelength}")
    if refractive_index <= 0:
        raise ValueError(
            f"refractive_index must be positive, got {refractive_index}"
        )
    ratio = numerical_aperture / refractive_index
    if not 0 < ratio <= 1 + 1e-12:
        raise ValueError(
            "Need 0 < numerical_aperture ≤ refractive_index, got "
            f"{numerical_aperture} and {refractive_index}"
        )
    aperture = math.asin(min(ratio, 1.0))
    k = 2 * math.pi * refractive_index / wavelength
    return k, aperture


class _Grid:
    """Cylindrical coordinates of a (Z, Y, X) grid centred at `n // 2`."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        step: Sequence[float],
        k: float,
        backend: dict,
    ) -> None:
        nz, ny, nx = shape
        z = (torch.arange(nz, **backend) - nz // 2) * step[0]
        y = (torch.arange(ny, **backend) - ny // 2) * step[1]
        x = (torch.arange(nx, **backend) - nx // 2) * step[2]
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        r2 = (yy * yy + xx * xx).reshape(-1)
        # Distinct radii only: the quadrant (and, for square pixels, the
        # diagonal) symmetries of the grid make most of them repeat.
        unique, self.inverse = torch.unique(r2, return_inverse=True)
        r_unique = unique.sqrt()
        self.axial = k * z
        self.radial = k * r_unique
        r_flat = r2.sqrt()
        safe = torch.where(r_flat > 0, r_flat, torch.ones_like(r_flat))
        cos = torch.where(r_flat > 0, xx.reshape(-1) / safe, 1.0)
        sin = torch.where(r_flat > 0, yy.reshape(-1) / safe, 0.0)
        self.cos, self.sin = cos, sin
        self.cos2, self.sin2 = cos * cos - sin * sin, 2 * cos * sin
        self.shape = shape


def _integrals_on_grid(
    grid: _Grid,
    aperture: float,
    nodes: int | None,
    pupil_phase: PupilPhase | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if nodes is None:
        max_r = float(grid.radial.max())
        max_z = float(grid.axial.abs().max())
        backend = {"dtype": grid.radial.dtype, "device": grid.radial.device}
        variation = _phase_variation(aperture, pupil_phase, backend)
        nodes = _auto_nodes(aperture, max_r, max_z, variation)
    i0, i1, i2 = _grid_integrals(
        grid.axial, grid.radial, aperture, int(nodes), pupil_phase
    )
    return i0[:, grid.inverse], i1[:, grid.inverse], i2[:, grid.inverse]


# --------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------


def richards_wolf_field(
    shape: Sequence[int],
    *,
    wavelength: float,
    numerical_aperture: float,
    refractive_index: float = 1.0,
    spacing: float | Sequence[float] = 1.0,
    nodes: int | None = None,
    pupil_phase: PupilPhase | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Electric field near the focus of an aplanatic system, on a grid.

        e_x = −iA (I₀ + I₂ cos 2φ),
        e_y = −iA I₂ sin 2φ,
        e_z = −2A I₁ cos φ,

    (Richards & Wolf 1959, Eq. 2.30), for a linearly polarised incident
    wave with its electric vector along x, with A = 1 (A = π f l₀ / λ,
    Eq. 2.28, scales the whole field), φ the azimuth of the point from
    the x axis and I₀, I₁, I₂ the
    [`richards_wolf_integrals`][optix.psf.utils.richards_wolf_integrals].
    The time dependence is e^{−iωt} and z increases away from the
    lens.

    !!! references "References"

        1. Richards B, Wolf E. *Electromagnetic diffraction in optical
           systems II. Structure of the image field in an aplanatic
           system.* Proc R Soc Lond A **253**(1274):358–379, 1959.
           [doi:10.1098/rspa.1959.0200](https://doi.org/10.1098/rspa.1959.0200)

        ??? quote "BibTeX"

            ```bibtex
            @article{richards1959electromagnetic,
              author  = {Richards, B. and Wolf, E.},
              title   = {Electromagnetic diffraction in optical systems
                         {II}. {Structure} of the image field in an
                         aplanatic system},
              journal = {Proceedings of the Royal Society of London.
                         Series A, Mathematical and Physical Sciences},
              volume  = {253},
              number  = {1274},
              pages   = {358--379},
              year    = {1959},
              doi     = {10.1098/rspa.1959.0200}
            }
            ```

    Parameters
    ----------
    shape, wavelength, numerical_aperture, refractive_index, spacing
        See [`richards_wolf`][optix.psf.richards_wolf].
    nodes, pupil_phase
        See
        [`richards_wolf_integrals`][optix.psf.utils.richards_wolf_integrals].
    dtype, device : optional
        Real working dtype and device. Default: configuration.

    Returns
    -------
    Tensor
        Complex field, `(3, *shape)`: the components (e_x, e_y, e_z).
    """
    full, step, ndim = _shape_spacing(shape, spacing)
    k, aperture = _optics(wavelength, numerical_aperture, refractive_index)
    backend = resolve_backend(dtype, device)
    grid = _Grid(full, step, k, backend)
    i0, i1, i2 = _integrals_on_grid(grid, aperture, nodes, pupil_phase)
    minus_i = -1j
    ex = minus_i * (i0 + i2 * grid.cos2)
    ey = minus_i * (i2 * grid.sin2)
    ez = -2 * i1 * grid.cos
    field = torch.stack([ex, ey, ez]).reshape(3, *full)
    return field if ndim == 3 else field[:, 0]


def _emitter_vector(emitter: object) -> tuple[float, float, float] | None:
    if isinstance(emitter, str):
        if emitter != "isotropic":
            raise ValueError(
                f"emitter must be 'isotropic' or a 3-vector, got {emitter!r}"
            )
        return None
    p = [float(v) for v in emitter]  # type: ignore[attr-defined]
    if len(p) != 3:
        raise ValueError(f"A dipole needs 3 components, got {len(p)}")
    norm = math.sqrt(sum(v * v for v in p))
    if norm == 0:
        raise ValueError("The dipole orientation must be non-zero")
    return p[0] / norm, p[1] / norm, p[2] / norm


def richards_wolf(
    shape: Sequence[int],
    *,
    wavelength: float,
    numerical_aperture: float,
    refractive_index: float = 1.0,
    spacing: float | Sequence[float] = 1.0,
    polarization: str | float = "x",
    emitter: str | Sequence[float] | None = None,
    pupil_phase: PupilPhase | None = None,
    normalize: str | None = None,
    nodes: int | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Vectorial PSF of an aplanatic system of high aperture.

    **Model (faithful).** A linearly polarised monochromatic wave
    focused by an aplanatic system (axially stigmatic, obeying the sine
    condition) of angular semi-aperture α on the image side. The
    time-averaged electric energy density near the focus is

        ⟨w_e⟩ = (A² / 16π) {|I₀|² + 4 |I₁|² cos²φ + |I₂|²
                            + 2 cos 2φ · ℜ(I₀ I₂*)}

    (Richards & Wolf 1959, Eq. 3.14), with φ the azimuth of the point
    measured from the electric vector of the incident wave, and I₀, I₁,
    I₂ the integrals of Eq. (2.32) (see
    [`richards_wolf_integrals`][optix.psf.utils.richards_wolf_integrals]),
    which carry the √cos θ apodisation of an aplanatic system
    (Eqs. 2.12–2.13). It is ⟨E²⟩ / 8π, i.e. |e|² / 16π, with e the
    field of [`richards_wolf_field`][optix.psf.utils.richards_wolf_field]
    (Eq. 2.30). The output is ⟨w_e⟩ with A = 1 (A = π f l₀ / λ,
    Eq. 2.28, depends on the focal length f and the incident amplitude
    l₀). At the focus ⟨w_e⟩ = (A² / 16π) I₀(0, 0)², with
    I₀(0, 0) = (16/15){1 − (5/8) cos^{3/2}α (1 + (3/5) cos α)}
    (Eqs. 3.18–3.19).

    The distribution is symmetric about the focal plane (Eq. 3.15) but,
    for α > 0, not about the axis: in the focal plane it is elongated
    along the polarisation direction (Figs. 3–4). As α → 0 it tends to
    the Airy pattern ∝ (2 J₁(v) / v)² in the focal plane (Eq. 4.12) and
    to the Lommel–Struve distribution near the focus (Eq. 4.8).

    `polarization` selects the incident wave, all within the paper:

    - `"x"` (default) or `"y"`: linear polarisation along x or y;
    - an angle φ₀ in radians: linear polarisation at azimuth φ₀ from
      the x axis, i.e. φ replaced by φ − φ₀ above (Eq. 5.1);
    - `"unpolarized"`: the average of Eq. (5.1) over φ₀ (Section 5,
      Eq. 5.2), (A² / 16π) {|I₀|² + 2 |I₁|² + |I₂|²}, radially
      symmetric.

    **Extension: fluorescence emitters** (`emitter`, not in the paper).
    The detection PSF of a dipole emitter imaged by the same system,
    obtained by reciprocity: the signal at an image point from a dipole
    p at a point r of the object space, as a function of r, is
    ∑_j |p · e⁽ʲ⁾(r)|², where e⁽ʲ⁾ is the focal field of Eq. (2.30)
    produced in the object space by a point source at the image point
    (a wave of uniform amplitude on the pupil when the other side of the
    system has a low aperture) and j runs over the two orthogonal
    polarisations x and y, since a camera records both. Derivation:

    1. e⁽ˣ⁾ = −iA (I₀ + I₂ cos 2φ, I₂ sin 2φ, −2i I₁ cos φ) (Eq. 2.30),
       and e⁽ʸ⁾, its rotation by 90° about the axis,
       e⁽ʸ⁾ = −iA (I₂ sin 2φ, I₀ − I₂ cos 2φ, −2i I₁ sin φ);
    2. `emitter=(p_x, p_y, p_z)` (a fixed dipole, normalised to unit
       length): (1/16π) {|p · e⁽ˣ⁾|² + |p · e⁽ʸ⁾|²};
    3. `emitter="isotropic"` (a freely rotating dipole, or an
       unpolarised emitter): the average over orientations p uniform on
       the sphere, using ⟨|p · e|²⟩ = |e|² / 3 and
       |e⁽ˣ⁾|² + |e⁽ʸ⁾|² = 2A² {|I₀|² + 2 |I₁|² + |I₂|²}:
       (1/16π) · (2/3) {|I₀|² + 2 |I₁|² + |I₂|²}, proportional to
       Eq. (5.2).

    The √cos θ apodisation is kept: a dipole radiates an intensity per
    unit solid angle that the sine condition (h = f sin θ, Eq. 2.8)
    spreads over a pupil area f² cos θ dΩ, so the pupil amplitude is
    ∝ 1/√cos θ, and the pupil area element contributes cos θ, leaving
    √cos θ, as in Eq. (2.32). The model also assumes that:

    - the emitter is in a homogeneous medium of index
      `refractive_index` (no interface);
    - a fixed dipole is excited independently of its orientation (no
      excitation selection by a polarised illumination);
    - the result is the detection PSF only: a confocal or light-sheet
      system multiplies it by an illumination PSF.

    The prefactor 1/16π is kept from Eq. (3.14) for continuity with the
    focal energy density; it has no physical meaning for an emitter (the
    2/3 of the isotropic case is not a collection efficiency).

    **Extension: pupil phase** (`pupil_phase`, not in the paper): an
    aberration phase e^{iψ(θ)} multiplied into the integrands. It is a
    phase only (no s- and p-dependent amplitudes, such as Fresnel
    coefficients), and ψ should be a smooth function of sin²θ; see
    [`richards_wolf_integrals`][optix.psf.utils.richards_wolf_integrals].

    **Evaluation.** The integrals are computed by Gauss-Legendre
    quadrature in s = √cos θ, for each axial position and each distinct
    distance from the axis on the grid (as matrix products), with Bessel
    functions accurate to ~10⁻¹⁵ ([`bessel_j012`][optix.special.bessel_j012]).
    The default number of nodes, which accounts for the grid extent and
    for the total variation of ψ, keeps the error near 10⁻¹² of the
    peak in float64. The Bessel work arrays are chunked within the
    memory budget of `optix.config`; I₀, I₁, I₂ are then gathered onto
    every voxel, so the peak memory is about ten times the size of the
    output. It runs on any torch device.

    !!! references "References"

        1. Richards B, Wolf E. *Electromagnetic diffraction in optical
           systems II. Structure of the image field in an aplanatic
           system.* Proc R Soc Lond A **253**(1274):358–379, 1959.
           [doi:10.1098/rspa.1959.0200](https://doi.org/10.1098/rspa.1959.0200)

        ??? quote "BibTeX"

            ```bibtex
            @article{richards1959electromagnetic,
              author  = {Richards, B. and Wolf, E.},
              title   = {Electromagnetic diffraction in optical systems
                         {II}. {Structure} of the image field in an
                         aplanatic system},
              journal = {Proceedings of the Royal Society of London.
                         Series A, Mathematical and Physical Sciences},
              volume  = {253},
              number  = {1274},
              pages   = {358--379},
              year    = {1959},
              doi     = {10.1098/rspa.1959.0200}
            }
            ```

    Parameters
    ----------
    shape : sequence of int
        `(Z, Y, X)`, with z the optical axis, or `(Y, X)` for the focal
        plane only. The Gaussian focus is at index `n // 2` of each
        axis; the x axis is the default polarisation direction.
    wavelength : float
        Vacuum wavelength λ₀, in the units of `spacing`.
    numerical_aperture : float
        NA = n sin α, at most `refractive_index`.
    refractive_index : float
        Refractive index n of the image space (immersion medium); the
        wave number is k = n k₀, with k₀ = 2π / λ₀. The paper takes
        n = 1.
    spacing : float or sequence of float
        Voxel size, per axis.
    polarization : {"x", "y", "unpolarized"} or float
        Incident wave (see above; `"unpolarised"` is also accepted).
        Not used with `emitter`.
    emitter : {"isotropic"} or sequence of 3 floats, optional
        Extension: return the detection PSF of a fluorescent emitter
        (see above) instead of the focal energy density.
    pupil_phase : callable, optional
        Extension: aberration phase ψ(θ), in radians (real, smooth in
        sin²θ; see above).
    normalize : {None, "focus", "sum"}
        None (default): ⟨w_e⟩ with A = 1, the paper's quantity;
        `"focus"`: divided by its value at the Gaussian focus (1 there;
        the paper's tables and figures use 100); `"sum"`: unit sum, the
        convention of `optix.deconv`.
    nodes : int, optional
        Number of quadrature nodes (default: automatic, see
        [`richards_wolf_integrals`][optix.psf.utils.richards_wolf_integrals]).
    dtype, device : optional
        Default: configuration.

    Returns
    -------
    Tensor
        Real tensor of shape `shape`.
    """
    full, step, ndim = _shape_spacing(shape, spacing)
    k, aperture = _optics(wavelength, numerical_aperture, refractive_index)
    if normalize not in (None, "focus", "sum"):
        raise ValueError(
            f"normalize must be None, 'focus' or 'sum', got {normalize!r}"
        )
    if emitter is not None and polarization != "x":
        raise ValueError("`polarization` does not apply with `emitter`")
    dipole = None if emitter is None else _emitter_vector(emitter)
    backend = resolve_backend(dtype, device)
    grid = _Grid(full, step, k, backend)
    i0, i1, i2 = _integrals_on_grid(grid, aperture, nodes, pupil_phase)
    a0, a1, a2 = i0.abs().square(), i1.abs().square(), i2.abs().square()
    scale = 1 / (16 * math.pi)
    if emitter is None:
        if polarization in ("unpolarized", "unpolarised"):
            out = a0 + 2 * a1 + a2
        else:
            if polarization == "x":
                phi0 = 0.0
            elif polarization == "y":
                phi0 = math.pi / 2
            elif isinstance(polarization, int | float):
                phi0 = float(polarization)
            else:
                raise ValueError(
                    "polarization must be 'x', 'y', 'unpolarized' or an "
                    f"angle, got {polarization!r}"
                )
            c0, s0 = math.cos(2 * phi0), math.sin(2 * phi0)
            cos2 = grid.cos2 * c0 + grid.sin2 * s0  # cos 2(φ − φ₀)
            cross = (i0 * i2.conj()).real
            out = a0 + 2 * a1 * (1 + cos2) + a2 + 2 * cos2 * cross
    elif dipole is None:
        out = (2 / 3) * (a0 + 2 * a1 + a2)
    else:
        px, py, pz = dipole
        z1 = -2j * i1
        ex = px * (i0 + i2 * grid.cos2) + py * i2 * grid.sin2
        ex = ex + pz * z1 * grid.cos
        ey = px * i2 * grid.sin2 + py * (i0 - i2 * grid.cos2)
        ey = ey + pz * z1 * grid.sin
        out = ex.abs().square() + ey.abs().square()
    out = (scale * out).reshape(full)
    if ndim == 2:
        out = out[0]
    if normalize == "focus":
        # The Gaussian focus is the voxel at index n // 2 on every axis.
        at_focus = out[tuple(n // 2 for n in out.shape)]
        if not bool(at_focus > 0):
            raise ValueError(
                "The PSF vanishes at the focus (e.g. an axial dipole); use "
                "normalize='sum'"
            )
        out = out / at_focus
    elif normalize == "sum":
        out = out / out.sum()
    return out
