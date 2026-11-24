"""Light-sheet illumination and the effective PSF of light-sheet imaging."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .._config import resolve_backend

__all__ = [
    "beam_radius",
    "gaussian_sheet",
    "lightsheet_psf",
    "rayleigh_range",
    "sheet_intensity",
]


def rayleigh_range(
    waist: float | Tensor, wavelength: float, refractive_index: float = 1.0
) -> float | Tensor:
    """Rayleigh range of a Gaussian beam.

        z_R = π W₀² / λ,   λ = λ₀ / n,

    with W₀ the waist radius, λ₀ the vacuum wavelength and n the
    refractive index of the medium (Saleh & Teich, Sec. 3.1). The beam
    radius grows to √2 W₀ at z = ± z_R.

    Parameters
    ----------
    waist : float or Tensor
        Waist radius W₀ (1/e² intensity half-width at the focus).
    wavelength : float
        Vacuum wavelength λ₀, in the units of `waist`.
    refractive_index : float
        Refractive index n of the immersion medium.

    Returns
    -------
    float or Tensor
        z_R, in the units of `waist`.

    !!! references "References"

        1. Saleh BEA, Teich MC. *Fundamentals of Photonics*, 2nd ed.,
           chapter 3 (Beam Optics), section 3.1 (The Gaussian Beam).
           Wiley, 2007.

        ??? quote "BibTeX"

            ```bibtex
            @book{saleh2007fundamentals,
              author    = {Saleh, Bahaa E. A. and Teich, Malvin Carl},
              title     = {Fundamentals of Photonics},
              edition   = {2},
              chapter   = {3},
              publisher = {Wiley},
              year      = {2007}
            }
            ```
    """
    return math.pi * waist**2 * refractive_index / wavelength


def beam_radius(
    z: float | Tensor,
    *,
    waist: float,
    wavelength: float,
    refractive_index: float = 1.0,
) -> Tensor:
    """Radius of a Gaussian beam along its propagation axis.

        W(z) = W₀ √(1 + (z / z_R)²)

    (Saleh & Teich, Sec. 3.1), with z measured from the waist and z_R
    the [`rayleigh_range`][optix.psf.utils.rayleigh_range].

    Parameters
    ----------
    z : float or Tensor
        Distance from the waist along the propagation axis.
    waist, wavelength, refractive_index
        See [`rayleigh_range`][optix.psf.utils.rayleigh_range].

    Returns
    -------
    Tensor
        W(z), the 1/e² intensity half-width.

    !!! references "References"

        1. Saleh BEA, Teich MC. *Fundamentals of Photonics*, 2nd ed.,
           chapter 3 (Beam Optics), section 3.1 (The Gaussian Beam).
           Wiley, 2007.

        ??? quote "BibTeX"

            ```bibtex
            @book{saleh2007fundamentals,
              author    = {Saleh, Bahaa E. A. and Teich, Malvin Carl},
              title     = {Fundamentals of Photonics},
              edition   = {2},
              chapter   = {3},
              publisher = {Wiley},
              year      = {2007}
            }
            ```
    """
    z = torch.as_tensor(z)
    zr = rayleigh_range(waist, wavelength, refractive_index)
    return waist * torch.hypot(torch.ones_like(z), z / zr)


def sheet_intensity(
    transverse: float | Tensor,
    propagation: float | Tensor,
    *,
    waist: float,
    wavelength: float,
    refractive_index: float = 1.0,
) -> Tensor:
    """Intensity of a Gaussian light sheet.

        I(t, z) = (W₀ / W(z)) · exp(−2 t² / W(z)²),

    with t the distance from the sheet's central plane (along the
    detection axis), z the distance from the waist along the propagation
    axis and W(z) the [`beam_radius`][optix.psf.utils.beam_radius]. The
    transverse profile is that of a Gaussian beam (Saleh & Teich,
    Sec. 3.1) focused along t only, as by a cylindrical lens; the factor
    W₀ / W(z) conserves the power per unit length of the sheet,
    ∫ I dt = W₀ √(π / 2) for every z. A circular Gaussian beam scanned
    along the sheet plane has the same time-averaged profile. I = 1 at
    the centre of the waist.

    The full width at half maximum of the sheet is W(z) √(2 ln 2).

    Parameters
    ----------
    transverse : float or Tensor
        t, distance from the central plane.
    propagation : float or Tensor
        z, distance from the waist along the propagation axis.
    waist, wavelength, refractive_index
        See [`rayleigh_range`][optix.psf.utils.rayleigh_range].

    Returns
    -------
    Tensor
        I(t, z), broadcast over `transverse` and `propagation`.

    !!! references "References"

        1. Saleh BEA, Teich MC. *Fundamentals of Photonics*, 2nd ed.,
           chapter 3 (Beam Optics), section 3.1 (The Gaussian Beam).
           Wiley, 2007.

        ??? quote "BibTeX"

            ```bibtex
            @book{saleh2007fundamentals,
              author    = {Saleh, Bahaa E. A. and Teich, Malvin Carl},
              title     = {Fundamentals of Photonics},
              edition   = {2},
              chapter   = {3},
              publisher = {Wiley},
              year      = {2007}
            }
            ```
    """
    t = torch.as_tensor(transverse)
    z = torch.as_tensor(propagation, device=t.device)
    dtype = torch.promote_types(t.dtype, z.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    t, z = t.to(dtype), z.to(dtype)
    w = beam_radius(
        z,
        waist=waist,
        wavelength=wavelength,
        refractive_index=refractive_index,
    )
    return (waist / w) * torch.exp(-2 * (t / w) ** 2)


def _axis(axis: int | str, ndim: int, axes: str | Sequence[str] | None) -> int:
    if isinstance(axis, str):
        if axes is None:
            raise ValueError(
                f"Axis {axis!r} is a label, but `axes` is not given"
            )
        labels = list(axes)
        if len(labels) != ndim:
            raise ValueError(
                f"axes {axes!r} has {len(labels)} labels, the data {ndim} dims"
            )
        if axis not in labels:
            raise ValueError(f"Unknown axis {axis!r}; axes are {axes!r}")
        return labels.index(axis)
    if not -ndim <= axis < ndim:
        raise ValueError(f"Axis {axis} is out of range for {ndim} dims")
    return axis % ndim


def _spacing(spacing: float | Sequence[float], ndim: int) -> list[float]:
    if isinstance(spacing, int | float):
        return [float(spacing)] * ndim
    spacing = [float(s) for s in spacing]
    if len(spacing) != ndim:
        raise ValueError(f"spacing needs {ndim} entries, got {len(spacing)}")
    return spacing


def _coordinate(n: int, step: float, shift: float, backend: dict) -> Tensor:
    """Physical coordinate of each index, 0 at index n // 2, minus `shift`."""
    return (torch.arange(n, **backend) - n // 2) * step - shift


def _rotation(
    rotation: float | Tensor | Sequence[Sequence[float]],
    ndim: int,
    det: int,
    prop: int | None,
    backend: dict,
) -> Tensor:
    """Matrix R (ndim × ndim) of the sheet frame in grid coordinates.

    A scalar is an angle θ (radians) in the (detection, propagation)
    plane: R eₜ = cos θ eₜ + sin θ e_z, R e_z = −sin θ eₜ + cos θ e_z.
    """
    if isinstance(rotation, int | float) or (
        isinstance(rotation, Tensor) and rotation.ndim == 0
    ):
        if prop is None:
            raise ValueError(
                "A rotation angle needs the `propagation` axis (the plane "
                "of the rotation); give a matrix otherwise"
            )
        theta = float(rotation)
        c, s = math.cos(theta), math.sin(theta)
        mat = torch.eye(ndim, dtype=torch.float64)
        mat[det, det], mat[prop, det] = c, s
        mat[det, prop], mat[prop, prop] = -s, c
        return mat.to(**backend)
    mat = torch.as_tensor(rotation, dtype=torch.float64)
    if mat.shape != (ndim, ndim):
        raise ValueError(
            f"rotation must be an angle or a ({ndim}, {ndim}) matrix, "
            f"got shape {tuple(mat.shape)}"
        )
    if not torch.allclose(
        mat.T @ mat, torch.eye(ndim, dtype=mat.dtype), atol=1e-5
    ):
        raise ValueError("rotation must be an orthogonal matrix")
    return mat.to(**backend)


def _project(
    direction: Tensor, coords: Sequence[Tensor], shift: float
) -> Tensor:
    """d · r − shift on the grid, broadcast from the 1-D coordinates."""
    ndim = len(coords)
    out = None
    for i, (d, r) in enumerate(zip(direction.tolist(), coords, strict=True)):
        if d == 0:
            continue
        view = [1] * ndim
        view[i] = -1
        term = d * r.view(view)
        out = term if out is None else out + term
    if out is None:
        out = coords[0].new_zeros([1] * ndim)
    return out - shift


def gaussian_sheet(
    shape: Sequence[int],
    *,
    detection: int | str,
    propagation: int | str,
    waist: float,
    wavelength: float,
    refractive_index: float = 1.0,
    spacing: float | Sequence[float] = 1.0,
    offset: float = 0.0,
    focus: float = 0.0,
    rotation: float | Tensor | Sequence[Sequence[float]] | None = None,
    axes: str | Sequence[str] | None = None,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Gaussian light-sheet intensity sampled on a grid.

        I(r) = (W₀ / W(z)) · exp(−2 t² / W(z)²),
        t = r_det − `offset`,   z = r_prop − `focus`,

    the [`sheet_intensity`][optix.psf.utils.sheet_intensity] across the
    `detection` axis, with a thickness that varies along the
    `propagation` axis; it is uniform along the other axes. Coordinates
    r are physical (index times `spacing`), with 0 at index `n // 2`.

    With a `rotation` R, the sheet frame is rotated relative to the
    grid (e.g. a view of a multiview acquisition sampled on the common
    grid): its normal is n = R e_det and its propagation direction
    p = R e_prop, and

        t = n · r − `offset`,   z = p · r − `focus`.

    R = 1 (or an angle 0) gives the sheet as without rotation; an angle of 90°
    turns the sheet normal onto the `propagation` axis of the grid and
    the propagation direction onto −e_det.

    Parameters
    ----------
    shape : sequence of int
        Grid shape.
    detection : int or str
        Axis of the detection objective, across the sheet (an index, or
        a label of `axes`).
    propagation : int or str
        Axis along which the illumination propagates.
    waist, wavelength, refractive_index
        See [`rayleigh_range`][optix.psf.utils.rayleigh_range]; lengths
        in the units of `spacing`.
    spacing : float or sequence of float
        Voxel size, per axis.
    offset : float
        Position of the sheet's central plane along `detection`.
    focus : float
        Position of the waist along `propagation`.
    rotation : float or Tensor, optional
        Orientation of the sheet frame on the grid: an angle θ (radians)
        in the (`detection`, `propagation`) plane, from the detection
        axis towards the propagation axis, or an orthogonal matrix R,
        `(ndim, ndim)` in the order of the grid axes, whose columns are
        the sheet-frame axes in grid coordinates (r = R s). `offset`
        and `focus` are then measured along n and p from the grid
        origin. Default: no rotation.
    axes : str or sequence of str, optional
        Axis labels (e.g. `"zyx"`), to name `detection` and
        `propagation`.
    dtype, device : optional
        Default: configuration.

    Returns
    -------
    Tensor
        Intensity of shape `shape`, 1 at the centre of the waist.

    !!! references "References"

        1. Saleh BEA, Teich MC. *Fundamentals of Photonics*, 2nd ed.,
           chapter 3 (Beam Optics), section 3.1 (The Gaussian Beam).
           Wiley, 2007.

        ??? quote "BibTeX"

            ```bibtex
            @book{saleh2007fundamentals,
              author    = {Saleh, Bahaa E. A. and Teich, Malvin Carl},
              title     = {Fundamentals of Photonics},
              edition   = {2},
              chapter   = {3},
              publisher = {Wiley},
              year      = {2007}
            }
            ```
    """
    shape = tuple(int(n) for n in shape)
    ndim = len(shape)
    det = _axis(detection, ndim, axes)
    prop = _axis(propagation, ndim, axes)
    if det == prop:
        raise ValueError("The detection and propagation axes must differ")
    backend = resolve_backend(dtype, device)
    step = _spacing(spacing, ndim)
    kw = {
        "waist": waist,
        "wavelength": wavelength,
        "refractive_index": refractive_index,
    }
    if rotation is not None:
        mat = _rotation(rotation, ndim, det, prop, backend)
        coords = [
            _coordinate(n, h, 0.0, backend)
            for n, h in zip(shape, step, strict=True)
        ]
        t = _project(mat[:, det], coords, offset)
        z = _project(mat[:, prop], coords, focus)
        return sheet_intensity(t, z, **kw).expand(shape).contiguous()
    t = _coordinate(shape[det], step[det], offset, backend)
    z = _coordinate(shape[prop], step[prop], focus, backend)
    view_t = [1] * ndim
    view_t[det] = -1
    view_z = [1] * ndim
    view_z[prop] = -1
    intensity = sheet_intensity(
        t.view(view_t),
        z.view(view_z),
        waist=waist,
        wavelength=wavelength,
        refractive_index=refractive_index,
    )
    return intensity.expand(shape).contiguous()


def lightsheet_psf(
    psf: Tensor,
    *,
    detection: int | str,
    waist: float | None = None,
    wavelength: float | None = None,
    refractive_index: float = 1.0,
    spacing: float | Sequence[float] = 1.0,
    position: float = 0.0,
    offset: float = 0.0,
    sheet: Tensor | None = None,
    rotation: float | Tensor | Sequence[Sequence[float]] | None = None,
    propagation: int | str | None = None,
    axes: str | Sequence[str] | None = None,
    normalize: bool = False,
) -> Tensor:
    """Effective PSF of light-sheet imaging: detection PSF times the sheet.

        h_eff(r) = h_det(r) · I(r_det − `offset`, `position`),

    with I the Gaussian [`sheet_intensity`][optix.psf.utils.sheet_intensity]
    and r the offset from the point source, with 0 at index `K // 2`.
    A point source at distance z from the waist along the propagation
    axis is excited by I(t, z), with t its distance from the sheet's
    central plane: as the sheet (or the sample) is stepped along the
    detection axis, the image of the point is the detection PSF weighted
    by the excitation at each step. The sheet thickness is therefore
    taken at the point's own position z = `position`, and the factor
    varies only along the `detection` axis; for a field of view,
    evaluate the PSF at several positions.

    **Rotated sheet** (multiview). When `psf` is sampled on a grid that
    is rotated relative to the view (as the PSFs of
    [`MultiviewRL`][optix.deconv.MultiviewRL], given in the
    orientation of the common grid), the sheet factor varies along the
    view's detection direction n = R e_det:

        h_eff(r) = h_det(r) · I(n · r − `offset`, `position`).

    It is evaluated on the grid of `psf` only, so its cost is that of
    one PSF-sized tensor. With a Gaussian detection PSF of covariance Σ
    and a Gaussian sheet, h_eff is Gaussian with covariance
    (Σ⁻¹ + (4 / W²) n nᵀ)⁻¹, W = W(`position`).

    Parameters
    ----------
    psf : Tensor
        Detection PSF h_det, centred at index `K // 2` (any model, or a
        measured PSF such as the output of
        [`bead_psf`][optix.psf.bead_psf]).
    detection : int or str
        Axis of the detection objective (an index, or a label of
        `axes`).
    waist, wavelength, refractive_index
        Gaussian sheet (see
        [`rayleigh_range`][optix.psf.utils.rayleigh_range]); lengths in
        the units of `spacing`. Not used with `sheet`.
    spacing : float or sequence of float
        Voxel size, per axis (only the `detection` entry is used
        without `rotation`).
    position : float
        Distance z of the point source from the waist, along the
        propagation axis.
    offset : float
        Position of the sheet's central plane along `detection`,
        relative to the point source.
    sheet : Tensor, optional
        Measured sheet profile, used instead of the Gaussian sheet:
        either a 1-D profile along `detection` with `K` samples on the
        grid of `psf` (centred at `K // 2`), or a tensor that broadcasts
        to `psf`. With `rotation`, a 1-D profile is interpolated
        linearly at t = n · r − `offset`, its end values held beyond its
        range (not specified in any reference).
    rotation : float or Tensor, optional
        Orientation of the sheet frame relative to the axes of `psf`:
        an angle θ (radians) in the (`detection`, `propagation`) plane,
        or an orthogonal matrix R, `(ndim, ndim)`, whose columns are the
        sheet-frame axes in grid coordinates (see
        [`gaussian_sheet`][optix.psf.gaussian_sheet]). `detection` then
        names the sheet-frame axis that R maps to n. Default:
        no rotation.
    propagation : int or str, optional
        Propagation axis, needed only for a rotation angle.
    axes : str or sequence of str, optional
        Axis labels of `psf` (e.g. `"zyx"`), to name `detection` and
        `propagation`.
    normalize : bool
        Rescale the result to unit sum.

    Returns
    -------
    Tensor
        h_eff, same shape, dtype and device as `psf`.

    !!! references "References"

        1. Saleh BEA, Teich MC. *Fundamentals of Photonics*, 2nd ed.,
           chapter 3 (Beam Optics), section 3.1 (The Gaussian Beam).
           Wiley, 2007.

        ??? quote "BibTeX"

            ```bibtex
            @book{saleh2007fundamentals,
              author    = {Saleh, Bahaa E. A. and Teich, Malvin Carl},
              title     = {Fundamentals of Photonics},
              edition   = {2},
              chapter   = {3},
              publisher = {Wiley},
              year      = {2007}
            }
            ```
    """
    ndim = psf.ndim
    det = _axis(detection, ndim, axes)
    view = [1] * ndim
    view[det] = -1
    if rotation is not None:
        out = psf * _rotated_factor(
            psf,
            det,
            None if propagation is None else _axis(propagation, ndim, axes),
            rotation,
            waist=waist,
            wavelength=wavelength,
            refractive_index=refractive_index,
            spacing=spacing,
            position=position,
            offset=offset,
            sheet=sheet,
        )
    elif sheet is not None:
        sheet = torch.as_tensor(sheet).to(psf)
        if sheet.ndim == 1:
            if len(sheet) != psf.shape[det]:
                raise ValueError(
                    f"A 1-D sheet profile needs {psf.shape[det]} samples "
                    f"(the detection axis of the PSF), got {len(sheet)}"
                )
            sheet = sheet.view(view)
        out = psf * sheet
    else:
        if waist is None or wavelength is None:
            raise TypeError("Give `waist` and `wavelength`, or a `sheet`")
        step = _spacing(spacing, ndim)[det]
        backend = {"dtype": psf.dtype, "device": psf.device}
        t = _coordinate(psf.shape[det], step, offset, backend)
        intensity = sheet_intensity(
            t,
            torch.tensor(float(position), **backend),
            waist=waist,
            wavelength=wavelength,
            refractive_index=refractive_index,
        )
        out = psf * intensity.view(view)
    if normalize:
        out = out / out.sum()
    return out


def _rotated_factor(
    psf: Tensor,
    det: int,
    prop: int | None,
    rotation: float | Tensor | Sequence[Sequence[float]],
    *,
    waist: float | None,
    wavelength: float | None,
    refractive_index: float,
    spacing: float | Sequence[float],
    position: float,
    offset: float,
    sheet: Tensor | None,
) -> Tensor:
    """Sheet factor I(n · r − offset, position) on the grid of `psf`."""
    ndim = psf.ndim
    if prop == det:
        raise ValueError("The detection and propagation axes must differ")
    backend = {"dtype": psf.dtype, "device": psf.device}
    mat = _rotation(rotation, ndim, det, prop, backend)
    step = _spacing(spacing, ndim)
    coords = [
        _coordinate(n, h, 0.0, backend)
        for n, h in zip(psf.shape, step, strict=True)
    ]
    t = _project(mat[:, det], coords, offset)
    if sheet is None:
        if waist is None or wavelength is None:
            raise TypeError("Give `waist` and `wavelength`, or a `sheet`")
        return sheet_intensity(
            t,
            torch.tensor(float(position), **backend),
            waist=waist,
            wavelength=wavelength,
            refractive_index=refractive_index,
        )
    sheet = torch.as_tensor(sheet).to(psf)
    if sheet.ndim != 1:
        raise ValueError(
            "With `rotation`, `sheet` must be a 1-D profile along the "
            "sheet normal"
        )
    k = len(sheet)
    # profile samples at t_k = (k − K // 2) · spacing[det]
    u = (t / step[det] + k // 2).clamp(0, k - 1)
    lo = u.floor().clamp(max=k - 2) if k > 1 else u.floor()
    frac = u - lo
    lo = lo.long()
    hi = (lo + 1).clamp(max=k - 1)
    return sheet[lo] * (1 - frac) + sheet[hi] * frac
