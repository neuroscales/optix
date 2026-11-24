# Light-sheet PSF

!!! info "At a glance"
    - **Estimates:** the effective PSF tensor ([`lightsheet_psf`][optix.psf.lightsheet_psf]) and the sheet intensity on a grid ([`gaussian_sheet`][optix.psf.gaussian_sheet]); no estimate object
    - **Noise model:** none (a deterministic optical model)
    - **Streaming:** none: one PSF-sized tensor is computed from the optical parameters, not from data
    - **Needs many frames:** no
    - **Reference:** Gaussian beam: Saleh and Teich, *Fundamentals of Photonics*, 2nd ed., Wiley, 2007, Sec. 3.1, [Wiley](https://www.wiley.com/en-us/Fundamentals+of+Photonics%2C+2nd+Edition-p-9780471358329); the sheet and the effective PSF are derived in optix

## Overview

In light-sheet microscopy the specimen is lit by a thin sheet of light, and the
fluorescence is collected by an objective whose axis is perpendicular to the
sheet. A point of the specimen is only imaged where the sheet reaches it, so the
effective PSF of the microscope is the PSF of the detection objective weighted by
the excitation: it is narrower along the detection axis than the detection PSF
alone, by an amount set by the thickness of the sheet.

[`lightsheet_psf`][optix.psf.lightsheet_psf] multiplies a detection PSF by the
intensity of a Gaussian sheet. The detection PSF is any model of this section,
for instance [Gibson–Lanni](gibson-lanni.md) or
[Richards–Wolf](richards-wolf.md), or a measured PSF such as the output of
[Bead PSF](bead-psf.md). Use it to give a deconvolution
estimator the PSF of a light-sheet acquisition, or, with the `rotation` option,
the PSF of each view of a multiview acquisition expressed on the common grid, as
[`MultiviewRL`][optix.deconv.MultiviewRL] expects. A measured sheet profile can
replace the Gaussian one.

The building blocks are functions, not estimators: there is no `fit` and no
`apply`. They return tensors. [`gaussian_sheet`][optix.psf.gaussian_sheet] samples
the sheet intensity on a grid, and the three functions of `optix.psf.utils`
([`rayleigh_range`][optix.psf.utils.rayleigh_range],
[`beam_radius`][optix.psf.utils.beam_radius] and
[`sheet_intensity`][optix.psf.utils.sheet_intensity]) evaluate the Gaussian beam
and the sheet at given points.

## Image formation (optics)

The model has three parts: the Gaussian beam, the sheet made from it, and the
effective PSF. Lengths share one unit (the unit of `spacing`). See
[Image formation and noise](../../concepts/image-formation.md) for the imaging
model the PSF enters.

**Gaussian beam.** This part is taken from the reference
([Saleh and Teich, Sec. 3.1](https://www.wiley.com/en-us/Fundamentals+of+Photonics%2C+2nd+Edition-p-9780471358329)).
A beam of waist radius $W_0$ (the half-width at $1/e^2$ of the intensity at the
focus) and vacuum wavelength $\lambda_0$, in a medium of index $n$, has the
wavelength $\lambda = \lambda_0 / n$ and the Rayleigh range, the distance at
which the beam radius reaches $\sqrt{2}\,W_0$,

=== "optix"

    $$
    z_R = \frac{\pi W_0^2}{\lambda}, \qquad
    W(z) = W_0 \sqrt{1 + \left(\frac{z}{z_R}\right)^2}, \qquad
    \theta_0 = \frac{\lambda}{\pi W_0},
    $$

    with $\lambda = \lambda_0 / n$ and $z$ the distance from the waist along the
    propagation axis.

=== "Saleh and Teich 2007"

    $$
    z_0 = \frac{\pi W_0^2}{\lambda}, \qquad
    W(z) = W_0 \sqrt{1 + \left(\frac{z}{z_0}\right)^2}, \qquad
    \theta_0 = \frac{\lambda}{\pi W_0},
    $$

    with $\lambda = \lambda_0 / n$ the wavelength in the medium.

$\theta_0$ is the divergence half-angle of the beam. It is not needed to compute
the PSF; [`rayleigh_range`][optix.psf.utils.rayleigh_range] and
[`beam_radius`][optix.psf.utils.beam_radius] implement $z_R$ and $W(z)$.

**Sheet intensity (derived in optix, not from a reference).** The sheet is the
beam focused along one axis only, as by a cylindrical lens: Gaussian across the
sheet (along the detection axis, at distance $t$ from its central plane) and
uniform along the other axes. Its intensity at distance $z$ from the waist along
the propagation axis is

$$
I(t, z) = \frac{W_0}{W(z)}\, \exp\!\left(-\frac{2 t^2}{W(z)^2}\right),
$$

where the factor $W_0 / W(z)$ conserves the power per unit length of the sheet,
$\int I\,\mathrm{d}t = W_0 \sqrt{\pi/2}$ for every $z$, and $I = 1$ at the centre
of the waist. A circular Gaussian beam scanned along the sheet plane has the same
time-averaged profile. The full width at half maximum of the sheet is, from
$I(t, z) = I(0, z)/2$,

$$
\mathrm{FWHM}(z) = \sqrt{2 \ln 2}\; W(z) \approx 1.177\, W(z).
$$

**Effective PSF (derived in optix, not from a reference).** A point source at
distance $z_p$ from the waist is excited by $I(t, z_p)$, where $t$ is its
distance from the central plane of the sheet. As the sheet (or the sample) is
stepped along the detection axis, the image of the point is the detection PSF
weighted by the excitation at each step:

=== "optix"

    $$
    h_{\mathrm{eff}}(\mathbf{r}) = h_{\mathrm{det}}(\mathbf{r})\;
    I\bigl(\mathbf{n}\cdot\mathbf{r} - t_0,\; z_p\bigr),
    $$

    with $\mathbf{r}$ the offset from the point source, $\mathbf{n} = \mathbf{e}_{\mathrm{det}}$
    (the detection axis) when the sheet is not rotated, $t_0$ the position of the
    central plane relative to the point source (`offset`) and $z_p$ the distance
    of the point from the waist (`position`).

=== "Saleh and Teich 2007"

    The reference gives the beam only, so there is no second form of this
    equation: $I$ is built from $W_0$, $W(z)$ and $z_0$ as above.

| optix | Saleh and Teich 2007 | Meaning | Shape |
|---|---|---|---|
| $W_0$ | $W_0$ | waist radius (`waist`) | scalar |
| $\lambda_0$ | | vacuum wavelength (`wavelength`) | scalar |
| $n$ | | refractive index of the medium (`refractive_index`) | scalar |
| $\lambda$ | $\lambda$ | wavelength in the medium, $\lambda_0 / n$ | scalar |
| $z_R$ | $z_0$ | Rayleigh range | scalar |
| $W(z)$ | $W(z)$ | beam radius at distance $z$ from the waist | scalar |
| $\theta_0$ | $\theta_0$ | divergence half-angle | scalar |
| $I(t, z)$ | | sheet intensity | scalar field |
| $t$ | | distance from the central plane of the sheet, along $\mathbf{n}$ | scalar |
| $z$, $z_p$ | $z$ | distance from the waist, along the propagation axis; of the point source | scalar |
| $t_0$, $z_f$ | | position of the central plane (`offset`); of the waist (`focus`) | scalar |
| $\mathbf{h}_{\mathrm{det}}$, $\mathbf{h}_{\mathrm{eff}}$ | | detection PSF; effective PSF, on a grid of $K$ voxels centred at index `K // 2` | $\mathbb{R}^{K}$ |
| $\mathbf{n}$, $\mathbf{p}$ | | normal of the sheet, propagation direction (unit vectors; not the index $n$ nor the dipole $\mathbf{p}$ of Richards–Wolf) | $\mathbb{R}^d$ |
| $\mathbf{R}$ | | rotation of the sheet frame on the grid | $\mathbb{R}^{d\times d}$ |

There is no paper for the sheet and the effective PSF, so the second tab shows
the notation of the reference for what it covers. The reference writes the
Rayleigh range $z_0$; optix writes $z_R$.

**Rotation (not in any reference).** A view of a multiview acquisition is sampled
on a common grid that is rotated relative to its sheet frame. The orientation is
an orthogonal matrix $\mathbf{R}$ whose columns are the axes of the sheet frame
in grid coordinates, $\mathbf{r} = \mathbf{R}\,\mathbf{s}$ with $\mathbf{s}$ the
coordinates in the sheet frame. The normal and the propagation direction of the
sheet on the grid are

$$
\mathbf{n} = \mathbf{R}\,\mathbf{e}_{\mathrm{det}}, \qquad
\mathbf{p} = \mathbf{R}\,\mathbf{e}_{\mathrm{prop}},
$$

and the sheet of [`gaussian_sheet`][optix.psf.gaussian_sheet] is
$I(\mathbf{n}\cdot\mathbf{r} - t_0,\; \mathbf{p}\cdot\mathbf{r} - z_f)$. A scalar
`rotation` is an angle $\theta$ in the (detection, propagation) plane, from the
detection axis towards the propagation axis:

$$
\mathbf{R}\,\mathbf{e}_{\mathrm{det}} = \cos\theta\,\mathbf{e}_{\mathrm{det}} + \sin\theta\,\mathbf{e}_{\mathrm{prop}},
\qquad
\mathbf{R}\,\mathbf{e}_{\mathrm{prop}} = -\sin\theta\,\mathbf{e}_{\mathrm{det}} + \cos\theta\,\mathbf{e}_{\mathrm{prop}}.
$$

An angle of $90^\circ$ turns the sheet normal onto the propagation axis of the
grid and the propagation direction onto $-\mathbf{e}_{\mathrm{det}}$. The offset
$t_0$ and the focus $z_f$ are measured along $\mathbf{n}$ and $\mathbf{p}$ from
the grid origin (index `n // 2` of each axis), not from the point source.

## Probabilistic model

This is a deterministic optical model: no noise and no prior. The assumptions
are the following.

- The illumination is a Gaussian beam in a homogeneous medium of index $n$,
  focused along one axis (a sheet), with the intensity above. The Gaussian beam
  and its parameters are from the reference; the sheet and the power
  normalisation are derived in optix.
- The detection and the excitation are independent: the effective PSF is the
  product of the detection PSF and the sheet intensity (derived in optix).
- **The thickness is taken at the point's own $z$ (not in any reference).** The
  sheet thickness is evaluated at $z_p$ (`position`) and held constant across the
  support of the PSF. The factor therefore varies along the detection axis only
  (or along $\mathbf{n}$ for a rotated sheet), not along the propagation axis.
  For a field of view, evaluate the PSF at several positions.
- **A measured profile is a 1-D function of $t$ (not in any reference).** A
  measured sheet profile replaces $I(\cdot, z_p)$. It is read by linear
  interpolation between its samples, and its end values are held beyond its range.

**Closed form with a Gaussian detection PSF (derived in optix).** With a Gaussian
detection PSF of covariance $\boldsymbol{\Sigma}$, the product with the Gaussian
sheet is Gaussian, with covariance

$$
\boldsymbol{\Sigma}_{\mathrm{eff}} = \left(\boldsymbol{\Sigma}^{-1} + \frac{4}{W^2}\,\mathbf{n}\mathbf{n}^\top\right)^{-1}, \qquad W = W(z_p).
$$

Along the detection axis of a PSF with standard deviation $\sigma$ there, and for
a sheet that is not rotated and centred on the point ($t_0 = 0$), this reads

$$
\frac{1}{\sigma_{\mathrm{eff}}^2} = \frac{1}{\sigma^2} + \frac{4}{W^2}.
$$

The width along the other axes is unchanged. A sheet thinner than the detection
PSF ($W \ll \sigma$) gives $\sigma_{\mathrm{eff}} \approx W/2$: the axial
resolution is set by the sheet.

## Inference / derivation

There is no inference. The effective PSF is computed in closed form by one
pointwise product:

1. The coordinate $t = \mathbf{n}\cdot\mathbf{r} - t_0$ of every voxel of the PSF
   grid, in physical units (index times `spacing`, 0 at index `K // 2`).
2. The sheet intensity $I(t, z_p)$ at those coordinates, from the Gaussian
   beam: $W(z_p)$, then the Gaussian in $t$.
3. The product with the detection PSF, optionally rescaled to unit sum.

The building blocks follow the same pattern: [`gaussian_sheet`][optix.psf.gaussian_sheet]
evaluates $I(\mathbf{n}\cdot\mathbf{r} - t_0, \mathbf{p}\cdot\mathbf{r} - z_f)$
on a grid, with a thickness that varies along the propagation axis (a sheet
image, not a PSF).

## Algorithm

**Input:** detection PSF $\mathbf{h}_{\mathrm{det}}$ (centred at index `K // 2`);
`detection` axis; `waist` $W_0$, `wavelength` $\lambda_0$, `refractive_index` $n$
(or a measured `sheet`); `spacing`; `position` $z_p$; `offset` $t_0$;
optionally `rotation` and `propagation`; `normalize`. \
**Output:** $\mathbf{h}_{\mathrm{eff}}$, same shape, dtype and device as
$\mathbf{h}_{\mathrm{det}}$.

1. **Beam radius** $W = W(z_p)$ from $W_0$, $z_R = \pi W_0^2 n / \lambda_0$ and
   $z_p$.
2. **Coordinate** of each voxel along the sheet normal: $t = \mathbf{n}\cdot\mathbf{r} - t_0$,
   with $\mathbf{n} = \mathbf{e}_{\mathrm{det}}$ (one 1-D coordinate) or
   $\mathbf{n} = \mathbf{R}\,\mathbf{e}_{\mathrm{det}}$ (a PSF-sized tensor).
3. **Sheet factor**: $I(t, z_p) = (W_0/W)\exp(-2t^2/W^2)$, or the measured
   profile read at $t$ (a 1-D profile along `detection`, with $K$ samples,
   interpolated linearly with held ends when `rotation` is given; a tensor that
   broadcasts to the PSF otherwise).
4. **Multiply**: $\mathbf{h}_{\mathrm{eff}} = \mathbf{h}_{\mathrm{det}} \odot I$.
5. **Normalise** to unit sum when `normalize=True`.

## Faithful mode and extensions

There is no reference publication for the sheet and the effective PSF, and the
functions have no `faithful` argument. The Gaussian beam equations follow
Saleh and Teich; the rest are choices of optix, listed here.

| Choice | What it does | Default |
|---|---|---|
| sheet shape | Gaussian across the sheet, power per unit length conserved along $z$ | always |
| thickness at `position` | sheet thickness at the point's own $z_p$, constant over the PSF support | always |
| `sheet` | measured profile instead of the Gaussian sheet (1-D along `detection`, or a tensor that broadcasts to the PSF) | `None` |
| `rotation` | orientation of the sheet frame on the grid: an angle (radians; needs `propagation`) or an orthogonal matrix | `None` (no rotation) |
| `normalize` | rescale the result to unit sum | `False` |
| `offset`, `focus` | position of the central plane along $\mathbf{n}$ (relative to the point source in `lightsheet_psf`, to the grid origin in `gaussian_sheet`); position of the waist along $\mathbf{p}$ | `0.0` |

The rotation conventions (the angle in the (detection, propagation) plane, its
sign from the detection axis towards the propagation axis, $\mathbf{r} = \mathbf{R}\,\mathbf{s}$,
and the measure of `offset` and `focus` from the grid origin) are not in any
reference.

**Limitations.**

- For a strongly rotated view, $z$ also varies across the support of the PSF
  (through $\mathbf{p}\cdot\mathbf{r}$), but the thickness is taken at the single
  $z_p$ of the point. The approximation is good when the PSF is small compared to
  the Rayleigh range $z_R$; evaluate at several positions otherwise. Only
  [`gaussian_sheet`][optix.psf.gaussian_sheet], which does not use a point
  source, varies the thickness with $\mathbf{p}\cdot\mathbf{r}$.
- Reflections are accepted: the model is that of a Gaussian beam in a
  homogeneous medium and does not describe reflections, scattering,
  absorption or refraction of the sheet by the specimen.
- A measured profile is a function of the distance to the central plane only; it
  does not change with $z_p$.

## Memory and streaming

The effective PSF is computed on the grid of the detection PSF only. Without
rotation the sheet factor is a 1-D vector along the detection axis, broadcast
over the PSF; with a rotation it is one PSF-sized tensor ($K$ voxels, the cost of
the PSF itself). [`gaussian_sheet`][optix.psf.gaussian_sheet] returns one tensor
of the grid shape. Nothing is read in chunks: the PSFs are small compared to the
data.

## Usage

The example builds the PSF of a Gaussian detection objective (a stand-in for a
measured or modelled PSF) and weights it by a sheet, then checks the closed form.

```pycon
>>> import math
>>> import torch
>>> from optix.psf import gaussian_sheet, lightsheet_psf
>>> from optix.psf.utils import beam_radius, rayleigh_range, sheet_intensity

>>> # A water-immersion sheet: waist 1 µm, wavelength 0.5 µm, n = 1.33.
>>> sheet = dict(waist=1.0, wavelength=0.5, refractive_index=1.33)
>>> zr = rayleigh_range(1.0, 0.5, 1.33)
>>> round(float(zr), 2)
8.36
>>> round(float(beam_radius(zr, **sheet)), 4), round(math.sqrt(2), 4)
(1.4142, 1.4142)

>>> # The full width at half maximum is sqrt(2 ln 2) W(z): the intensity is 1/2 there.
>>> fwhm = float(beam_radius(0.0, **sheet)) * math.sqrt(2 * math.log(2))
>>> round(fwhm, 3), round(float(sheet_intensity(fwhm / 2, 0.0, **sheet)), 3)
(1.177, 0.5)

>>> # A Gaussian detection PSF, sigma = 0.4 µm, on a 0.1 µm grid (axis 0 is the detection axis).
>>> x = (torch.arange(65) - 32) * 0.1
>>> r2 = x.view(-1, 1, 1) ** 2 + x.view(1, -1, 1) ** 2 + x.view(1, 1, -1) ** 2
>>> det = torch.exp(-r2 / (2 * 0.4**2))
>>> det = det / det.sum()

>>> # The effective PSF at the waist: narrower along the detection axis only.
>>> eff = lightsheet_psf(
...     det, detection=0, spacing=0.1, position=0.0, normalize=True, **sheet
... )
>>> def std(h, axis):
...     p = h.sum(dim=[a for a in range(3) if a != axis])
...     p = p / p.sum()
...     m = (p * x).sum()
...     return float((p * (x - m) ** 2).sum().sqrt())
>>> w = float(beam_radius(0.0, **sheet))
>>> (
...     round(std(eff, 0), 3),
...     round((1 / 0.4**2 + 4 / w**2) ** -0.5, 3),
...     round(std(eff, 1), 3),
... )
(0.312, 0.312, 0.4)

>>> # The same view rotated by 30 degrees in the (detection, propagation) plane.
>>> rot = lightsheet_psf(
...     det,
...     detection=0,
...     propagation=1,
...     rotation=math.pi / 6,
...     spacing=0.1,
...     normalize=True,
...     **sheet,
... )
>>> rot.shape, round(float(rot.sum()), 3)
(torch.Size([65, 65, 65]), 1.0)

>>> # The sheet intensity on a grid, rotated by 90 degrees.
>>> gaussian_sheet(
...     (9, 9), detection=0, propagation=1, rotation=math.pi / 2, **sheet
... ).shape
torch.Size([9, 9])

>>> # A measured profile along the detection axis in place of the Gaussian sheet.
>>> profile = torch.exp(-2 * (x / 0.7) ** 2)
>>> lightsheet_psf(det, detection=0, sheet=profile).shape
torch.Size([65, 65, 65])
```

Common variations: `axes="zyx"` to name `detection` and `propagation` by label;
`position` for the PSF at another distance from the waist; `offset` to centre the
sheet away from the point; an orthogonal matrix as `rotation`, for a view whose
frame differs from the grid by more than one angle; `dtype` and `device` of
[`gaussian_sheet`][optix.psf.gaussian_sheet]. The effective PSF can be given to
[`RichardsonLucy`](../deconv/richardson-lucy.md) and, for each view, to
[`MultiviewRL`][optix.deconv.MultiviewRL]. See the
[API reference](../../reference/psf.md).

## References

!!! references "References"

    1. Saleh BEA, Teich MC. *Fundamentals of Photonics*, 2nd ed., chapter 3
       (Beam Optics), section 3.1 (The Gaussian Beam). Wiley, 2007.
       [Wiley](https://www.wiley.com/en-us/Fundamentals+of+Photonics%2C+2nd+Edition-p-9780471358329)

    ??? quote "BibTeX"

        ```bibtex
        @book{saleh2007fundamentals,
          author    = {Saleh, Bahaa E. A. and Teich, Malvin Carl},
          title     = {Fundamentals of Photonics},
          edition   = {2},
          chapter   = {3},
          publisher = {Wiley},
          year      = {2007},
        }
        ```
