# Gibson–Lanni PSF

!!! info "At a glance"
    **Estimates:** `PSFEstimate` (`psf`, `fwhm`, `meta`) with `return_estimate=True`; otherwise the PSF tensor \
    **Noise model:** none (a deterministic optical model) \
    **Streaming:** none: a PSF is computed from the optical parameters, not from data \
    **Needs many frames:** no \
    **Reference:** Gibson and Lanni, J Opt Soc Am A, 1991, [doi:10.1364/JOSAA.8.001601](https://doi.org/10.1364/JOSAA.8.001601); fast evaluation: Li, Xue and Blu, J Opt Soc Am A, 2017, [doi:10.1364/JOSAA.34.001029](https://doi.org/10.1364/JOSAA.34.001029)

## Overview

[`gibson_lanni`][optix.psf.gibson_lanni] computes the point-spread function (PSF)
of a widefield fluorescence microscope whose objective is used *outside its
design conditions*: the source lies at some depth in a specimen whose refractive
index differs from that of the immersion oil, and the coverslip and the immersion
layer may differ in index and thickness from the values the objective was
designed for. These mismatches add a phase aberration that makes the PSF
asymmetric along the optical axis, dimmer and longer, and increasingly so with
depth. The model is scalar (no polarisation), needs only the numerical aperture
and the indices and thicknesses of the layers, and gives a PSF on a voxel grid
that can be passed to the [deconvolution](../../reference/deconv.md) estimators.

Use it when no bead acquisition is available (otherwise
[Bead PSF](bead-psf.md) measures the PSF), or to model the depth dependence of
the PSF. At high numerical aperture, where polarisation matters, a vectorial
model is more accurate; the scalar model is simpler and cheaper. It is a
function, not an estimator: there is no `fit` and no `apply`. With
`return_estimate=True` it returns a [`PSFEstimate`][optix.estimates.PSFEstimate],
with the FWHM (in voxels) and the parameters used in `meta`; by default it
returns the tensor. The building blocks
[`gibson_lanni_opd`][optix.psf.utils.gibson_lanni_opd] (the optical path
difference) and [`gibson_lanni_profile`][optix.psf.utils.gibson_lanni_profile]
(the intensity at given radii and defocus values) are in `optix.psf.utils`.

## Image formation (optics)

The microscope is treated as linear and shift-invariant over the field, as in
[Image formation and noise](../../concepts/image-formation.md): the recorded
image is the latent image convolved with the PSF. This page is about the PSF
itself, which in this model is axially shift-variant: it depends on the depth
$t_s$ of the source. One PSF is therefore valid for one depth.

[Gibson and Lanni (1991, Fig. 1)](https://doi.org/10.1364/JOSAA.8.001601)
compare two rays that leave the point source and enter the objective at the same
angle: one through the *actual* layers (specimen of index $n_s$ and thickness
$t_s$, coverslip $n_g$, $t_g$, immersion layer $n_i$, $t_i$) and one through the
*design* layers (coverslip $n_g^*$, $t_g^*$, immersion layer $n_i^*$, $t_i^*$),
with the source directly below the coverslip. The wave in the back focal plane is
spherical under the design conditions, so the phase difference between the two
rays, $k_0$ times their optical path difference (OPD), is the aberration. With the
sine condition, the normalised pupil radius $\rho$ and
$C_n(\rho) = [1 - (\mathrm{NA}\,\rho/n)^2]^{1/2}$, the OPD is
([Eqs. 2–4](https://doi.org/10.1364/JOSAA.8.001601)), for a detector at its
design distance,

=== "optix"

    $$
    \begin{aligned}
    \mathrm{OPD}(\rho;\Delta z) ={}& n_i\,\Delta z\, C_{n_i}(\rho)
      + n_s t_s \bigl[C_{n_s}(\rho) - (n_i/n_s)^2\, C_{n_i}(\rho)\bigr] \\
    &+ n_g t_g \bigl[C_{n_g}(\rho) - (n_i/n_g)^2\, C_{n_i}(\rho)\bigr] \\
    &- n_g^* t_g^* \bigl[C_{n_g^*}(\rho) - (n_i/n_g^*)^2\, C_{n_i}(\rho)\bigr] \\
    &- n_i^* t_i^* \bigl[C_{n_i^*}(\rho) - (n_i/n_i^*)^2\, C_{n_i}(\rho)\bigr],
    \end{aligned}
    $$

    with the defocus $\Delta z$ defined by

    $$
    \frac{\Delta z}{n_i} = \frac{t_s}{n_s} + \frac{t_g}{n_g} + \frac{t_i}{n_i}
      - \frac{t_g^*}{n_g^*} - \frac{t_i^*}{n_i^*} .
    $$

=== "Gibson and Lanni 1991"

    $$
    \begin{aligned}
    \mathrm{OPD} \approx{}& n_{\mathrm{oil}}\Bigl[\Delta z + \tfrac{(z_d^* - z_d)\,a^2 n_{\mathrm{oil}}}{z_d^* z_d\,\mathrm{NA}^2}\Bigr]
      \Bigl[1 - \bigl(\tfrac{\mathrm{NA}\rho}{n_{\mathrm{oil}}}\bigr)^2\Bigr]^{1/2}
      + \tfrac{a^2\rho^2 (z_d^* - z_d)}{2 n_{\mathrm{oil}} z_d^* z_d} \\
    &+ n_s t_s \Bigl\{\bigl[1 - (\tfrac{\mathrm{NA}\rho}{n_s})^2\bigr]^{1/2}
      - (\tfrac{n_{\mathrm{oil}}}{n_s})^2 \bigl[1 - (\tfrac{\mathrm{NA}\rho}{n_{\mathrm{oil}}})^2\bigr]^{1/2}\Bigr\} \\
    &+ n_g t_g \{\cdots\} - n_{g^*} t_{g^*} \{\cdots\} - n_{\mathrm{oil}^*} t_{\mathrm{oil}^*} \{\cdots\},
    \end{aligned}
    $$

    with $\Delta z / n_{\mathrm{oil}} = t_s/n_s + t_g/n_g + t_{\mathrm{oil}}/n_{\mathrm{oil}} - t_{g^*}/n_{g^*} - t_{\mathrm{oil}^*}/n_{\mathrm{oil}^*}$
    (Eq. 3), and each remaining brace of the same form as the $n_s t_s$ one.

| optix | Gibson and Lanni 1991 |
|---|---|
| $\mathrm{NA}$ | NA |
| $\lambda_0$, $k_0 = 2\pi/\lambda_0$ | $\lambda$, $k = 2\pi/\lambda$ |
| $n_s$, $t_s$ | $n_s$, $t_s$ |
| $n_g$, $t_g$; $n_g^*$, $t_g^*$ | $n_g$, $t_g$; $n_{g^*}$, $t_{g^*}$ |
| $n_i$, $t_i$; $n_i^*$, $t_i^*$ | $n_{\mathrm{oil}}$, $t_{\mathrm{oil}}$; $n_{\mathrm{oil}^*}$, $t_{\mathrm{oil}^*}$ |
| $\Delta z$ | $\Delta z$ |
| $\rho$ | $\rho$ |
| $r$ | $r_d / M$ ($r_d$ the radius on the detector, $M$ the magnification) |

The defocus $\Delta z$ is the distance the stage has moved from the position in
which the source is at the paraxial focus. $\Delta z = 0$ is the best geometric
focus: the term of the OPD in $\rho^2$ vanishes there, and what remains is
higher-order spherical aberration. Positive $\Delta z$ corresponds to a thicker
immersion layer $t_i$, that is, the objective moved away from the coverslip.
Eq. 3 eliminates $t_i$, which is rarely known, in favour of $\Delta z$, which is
what the experimenter moves. Only the *differences* between the actual and the
design layers matter: with specimen index equal to the immersion index, the
specimen term vanishes whatever the depth.

Written as it is, each square root of the OPD loses precision when the optical
paths are tens of micrometres long and the corrections are a fraction of a
wavelength. optix computes each $n\,[C_n(\rho) - 1]$ as
$-(\mathrm{NA}\rho)^2 / (\sqrt{n^2 - (\mathrm{NA}\rho)^2} + n)$ and drops the
constant value of the OPD at $\rho = 0$, a phase common to all rays that does not
change the intensity. The result agrees with the direct evaluation of the
equations to $10^{-11}$ (µm).

## Probabilistic model

The model has no noise and no prior: it is a deterministic image-formation model.
Its assumptions are those of
[Gibson and Lanni (1991)](https://doi.org/10.1364/JOSAA.8.001601):

- the source is a point on the optical axis, at depth $t_s$ in a homogeneous
  specimen layer; the PSF is radially symmetric about the axis;
- all the aberration comes from the layers between the source and the front
  element (specimen, coverslip, immersion), not from the objective, which is
  assumed to obey the sine condition and to be free of aberration under its
  design conditions;
- the light is a scalar wave, the amplitude is constant over the pupil, and
  the pupil is imaged by Kirchhoff diffraction.

The intensity at lateral distance $r$ and defocus $\Delta z$ is the squared
modulus of the Kirchhoff integral
([Gibson and Lanni 1991, Eq. 5](https://doi.org/10.1364/JOSAA.8.001601)),

=== "optix"

    $$
    h(r, \Delta z) \propto \Bigl|\int_0^a J_0\bigl(k_0\,\mathrm{NA}\, r\,\rho\bigr)\,
    e^{\,i k_0\, \mathrm{OPD}(\rho;\,\Delta z)}\,\rho\,\mathrm{d}\rho\Bigr|^2 ,
    $$

    with $J_0$ the Bessel function of the first kind and order 0, and $a$ the
    upper limit of the pupil (below).

=== "Gibson and Lanni 1991"

    $$
    I(x_d, y_d, z_d) = \Bigl|\frac{C}{z_d}\int_0^1
    J_0\Bigl[k a \rho \frac{(x_d^2 + y_d^2)^{1/2}}{z_d}\Bigr]
    e^{\,j W(\rho)}\,\rho\,\mathrm{d}\rho\Bigr|^2 ,
    \qquad W = k\,\mathrm{OPD},
    $$

    with $C$ a constant complex amplitude and $a$ the radius of the aperture
    projected on the back focal plane.

| optix | Gibson and Lanni 1991 |
|---|---|
| $h(r, \Delta z)$ | $I(x_d, y_d, z_d)$ |
| $k_0\,\mathrm{NA}\, r$ | $k a (x_d^2 + y_d^2)^{1/2} / z_d$ |
| $e^{i k_0\,\mathrm{OPD}}$ | $e^{jW(\rho)}$ |
| upper limit $a$ | upper limit 1 |

The letter $a$ has two meanings. In the Gibson and Lanni tab it is the radius of
the aperture projected on the back focal plane, a length. In the optix tab it is
the upper limit of the dimensionless pupil radius $\rho$, as in Li et al. (see
[Faithful mode and extensions](#faithful-mode-and-extensions)); the two are not
the same quantity.

The detector coordinate $x_d$ is mapped to object space: the paper's argument
$k a \rho\, r_d / z_d$ equals $k_0\,\mathrm{NA}\,\rho\, r$ with $r = r_d/M$, up to a
factor $[1 - (\mathrm{NA}/M)^2]^{-1/2}$ that is close to 1 when $\mathrm{NA} \ll M$ (a few $10^{-4}$ away from 1 for NA 1.4 at
60$\times$). The
constant $C / z_d$ is dropped because the PSF is normalised to unit sum. Samples
on the voxel grid are the intensity at $\Delta z$ = `defocus` + $(k - K_z//2)\,d_z$
and lateral offset $(j - K_y//2)\, d_y$, $(i - K_x//2)\, d_x$.

## Inference / derivation

There is nothing to infer: the PSF is the evaluation of the integral above on
the voxel grid, one integral per distinct (lateral radius, axial position)
pair. Because the PSF is radially symmetric, it is computed on the *distinct*
radii of the grid only and looked up for every lateral voxel. Two evaluation
methods are provided.

**Quadrature** (`method="quadrature"`, the default). The integral is computed
by Gauss–Legendre quadrature, as in the paper, which says only that Bessel
functions and Gaussian quadrature were used (not the node count, nor the
variable). The pupil radius is parametrised by the ray angle $\theta$ in the layer
of lowest index, $\rho = (n_{\min}/\mathrm{NA}) \sin\theta$, which keeps the
integrand smooth up to the edge of the aperture. For all $\Delta z$ at once, the
real and imaginary parts of the integral are the matrix products
$[\cos(k_0\,\mathrm{OPD}), \sin(k_0\,\mathrm{OPD})]\, J_0^\top$ of the
nodes-by-defocus phases and the nodes-by-radii Bessel values. The number of
nodes $Q$ is chosen automatically: starting from 32, it doubles until the
amplitude computed with $Q$ and $2Q$ nodes (on 17 radii evenly spaced from 0 to the
largest radius of the grid, and on the defocus values of the grid, or 257 of them
evenly spaced in index when there are more) differs by at most `rtol` of its peak.
Many oscillations of the Bessel function at large $r$, or of the phase at large
depth, need more nodes. The phase is always computed in double precision.

**Bessel series** (`method="li2017"`).
[Li, Xue and Blu (2017)](https://doi.org/10.1364/JOSAA.34.001029) expand the
phase factor on rescaled Bessel functions,

=== "optix"

    $$
    e^{\,i k_0\,\mathrm{OPD}(\rho;\Delta z)} \approx \sum_{m=1}^{M} c_m(\Delta z)\, J_0(\sigma_m \rho),
    \qquad
    \sigma_m = \mathrm{NA}\,(3m - 2)\,\frac{\lambda_{\mathrm{ref}}}{\lambda_0},
    $$

    so that, with $\beta = k_0\,\mathrm{NA}\, r$,

    $$
    \int_0^a \rho\, J_0(\sigma\rho)\, J_0(\beta\rho)\,\mathrm{d}\rho =
    a\,\frac{\sigma J_1(\sigma a) J_0(\beta a) - \beta J_0(\sigma a) J_1(\beta a)}{\sigma^2 - \beta^2} ,
    $$

    and the PSF is $\bigl|\sum_m c_m R_m\bigr|^2$ with $R_m$ the integral above at
    $\sigma = \sigma_m$.

=== "Li et al. 2017"

    $$
    \exp(i W) = \sum_{m} c_m(z)\, J_0(\sigma_m \rho), \qquad
    \sigma_m = \mathrm{NA}\,(3m - 2)\,\lambda_0/\lambda,
    \quad \lambda_0 = 436\ \mathrm{nm},
    $$

    $$
    R_m(r; \mathbf{p}) = \frac{\sigma_m J_1(\sigma_m a) J_0(\beta a) a - \beta J_0(\sigma_m a) J_1(\beta a) a}{\sigma_m^2 - \beta^2},
    \qquad \beta = k r\,\mathrm{NA} .
    $$

| optix | Li et al. 2017 |
|---|---|
| $\lambda_0$ | $\lambda$ |
| $\lambda_{\mathrm{ref}} = 0.436$ µm | $\lambda_0$ |
| $\Delta z$ | $\Delta z = z + t_i^* + n_i\bigl(\tfrac{t_s}{n_s} + \tfrac{t_g}{n_g} - \tfrac{t_g^*}{n_g^*} - \tfrac{t_i^*}{n_i^*}\bigr)$, i.e. $z + n_i t_s/n_s$ for matched layers; $z$ is the stage displacement from the design position |
| $a$, $M$ | $a$, $M$ |
| $r$ | $r$ |
| $n_s, n_g, n_g^*, n_i, n_i^*$ | $n_s, n_g, n_g^*, n_i, n_i^*$ |
| $t_s$ | $z_p$ |

The coefficients $c_m$ are fitted by least squares on $K$ samples $\rho_k$
uniform in $[0, a]$, the same for all radii: the matrix of Bessel values is
formed once and one solve gives the coefficients of every axial position. The
lateral basis $R_m$ is then evaluated at every distinct radius. optix solves the
least-squares problem with an SVD-based solver (`gelsd`), which equals the
normal-equation solution $(J^\top J)^{-1} J^\top F$ of the paper in exact
arithmetic and does not square the condition number.

The error of this method grows with depth, and the cause is the pupil edge, not
the number of functions. When $\mathrm{NA} \ge n_s$ the aperture is truncated
at $a = n_s/\mathrm{NA}$ and the phase varies there as
$k_0 t_s n_s \sqrt{1 - (\rho/a)^2}$, which the uniform samples $\rho_k$ cannot
follow. The error is therefore reduced by `samples` ($K$), not by `basis`
($M$). With $M = 100$, NA 1.4, $\lambda_0 = 0.61$ µm and $t_s = 10$ µm, the
largest error against quadrature, relative to the peak, is $6.8\times 10^{-3}$,
$9.2\times 10^{-4}$ and $2.4\times 10^{-4}$ for $K = 1000$, 3000 and 10000;
$M = 500$ with $K = 1000$ gives $5.2\times 10^{-3}$. It is $10^{-13}$ when
$\mathrm{NA} < n_s$ (NA 1.2 in water), where the aperture is not truncated.
The defaults are those of the paper. Raise `samples` with depth, or use
quadrature.

## Algorithm

```text
Input: shape (K_z, K_y, K_x), spacing (d_z, d_y, d_x), wavelength λ₀, NA,
       layers (depth, indices, thicknesses, design values), defocus,
       method, rtol, oversample
Radii:    the distinct values of √(Δy² + Δx²) over the lateral grid
Defocus:  Δz_k = defocus + (k − K_z//2) d_z, for the K_z slices
a = min(1, n_min / NA)   n_min = smallest index of the layers that have a thickness
if method = "quadrature":
    Q = nodes, or the smallest 32·2^j for which amplitudes with Q and 2Q nodes agree to rtol
    nodes ρ_q, weights w_q: Gauss–Legendre on θ, ρ = (n_min/NA) sin θ
    amplitude(r, Δz) = Σ_q w_q J_0(k₀ NA r ρ_q) exp(i k₀ OPD(ρ_q; Δz))
else:
    fit c_m(Δz) for m = 1..M on K samples ρ_k; amplitude = Σ_m c_m R_m(r)
I(r, Δz) = |amplitude|²
Look up I at each voxel (averaging oversample points per voxel axis if > 1)
Return I / ΣI, centred at index K // 2
```

Defaults: `method="quadrature"`, `rtol=1e-6`, `basis` $M = 100$ and `samples`
$K = 1000$ for `li2017`, `oversample=1`, `specimen_index=1.33`, design values
$n_g^* = 1.522$, $t_g^* = 170$ µm, $n_i^* = 1.515$ and $t_i^* = 150$ µm.

## Faithful mode and extensions

`gibson_lanni` is a function, with no `faithful` switch: the defaults are the
model of [Gibson and Lanni (1991)](https://doi.org/10.1364/JOSAA.8.001601), and
the opt-in keywords below are separate. The model is the paper's under the
following choices, made where the paper leaves a setting open or prints an
equation that cannot be used as is.

**Choices not specified in, or differing from, the paper**

- **Detector at its design distance.** The detector terms of Eq. 4 are left out
  ($z_d = z_d^*$): the PSF is that of a microscope with the objective-to-detector
  distance at its design value (the tube length is correct). As printed, the
  detector term $a^2\rho^2(z_d^* - z_d)/(2 n_{\mathrm{oil}} z_d^* z_d)$ would add
  a $\rho^2$ term even at $\Delta z = 0$, in contradiction with the text of the
  paper, which states that $\Delta z = 0$ removes the term in $\rho^2$;
  the printed term has a stray $n_{\mathrm{oil}}$ in the denominator. The term
  that cancels the $\rho^2$ term of the first line of Eq. 4 at $\Delta z = 0$ is
  $a^2\rho^2(z_d^* - z_d)/(2 z_d^* z_d)$, which is also its physical value (the
  detector is in air). With $z_d = z_d^*$ both vanish, so the PSF computed here
  does not depend on it. The PSF for a detector at another distance (Figs. 5 and 6
  of the paper) is not provided.
- **Lateral factor.** $[1 - (\mathrm{NA}/M)^2]^{-1/2}$ is taken as 1, as in
  [Li et al. (2017)](https://doi.org/10.1364/JOSAA.34.001029), so that the lateral
  coordinate is in object space and the magnification does not appear. This is
  the exact sine-condition form, not a further approximation: the paper's
  $a/z_d$ is $\tan\theta'$ in image space, and the sine condition gives
  $\sin\theta' = \mathrm{NA}/M$.
- **Upper limit of the integral.** The paper integrates to 1. Here
  $a = \min(1, n_{\min}/\mathrm{NA})$, because beyond $\rho = n_{\min}/\mathrm{NA}$
  the ray is evanescent in the layer of lowest index and the square roots of the
  OPD stop being real. $n_{\min}$ is the smallest refractive index among the
  layers *that have a thickness*: the actual immersion layer always, and the
  specimen, the actual and design coverslips and the design immersion layer only
  when their thickness is positive. Li et al. truncate at the smallest index of
  all layers, the specimen included at any depth; here a layer of zero thickness
  has no term in the OPD and does not truncate, so that with `depth=0` an NA 1.4
  oil objective keeps its full aperture whatever `specimen_index`. For
  $t_s > 0$ the super-critical rays are dropped (physically they are attenuated
  as $\exp(-k_0 t_s \sqrt{(\mathrm{NA}\rho)^2 - n_s^2})$, so this is the
  large-depth limit), and the aperture jumps from 1 to $n_s/\mathrm{NA}$ as the
  depth leaves 0. The value of $a$ is `meta["aperture"]` of the estimate.
- **Design immersion thickness** $t_i^* = 150$ µm, the value of Li et al.; the
  paper gives none. It matters only when the immersion index differs from the
  design index, since otherwise the $t_i^*$ term is cancelled by $\Delta z$.
- **Design indices and thickness of the coverslip and oil**, 1.522, 170 µm and
  1.515: the values of the experiments of the paper, not a prescription.
- **Quadrature order and node variable**, and the tolerance `rtol`, are choices
  of the implementation (the paper does not specify them).
- **Sign of the defocus.** The PSF follows Eqs. 3 and 4 as written: for a source
  in water the peak and the side lobes lie at $\Delta z < 0$, the side where the
  objective is closer to the specimen. Fig. 13A of the paper plots the profiles
  against a stage defocus measured from the in-focus plane and of opposite sign
  (side lobes at positive values). The slices (axis 0) are in order of increasing
  $\Delta z$, that is, of the objective moving away from the specimen; flip
  axis 0 for a stack acquired with the focus moving deeper.
- **Normalisation.** The PSF is normalised to unit sum, so that the constant
  amplitude $C/z_d$ is irrelevant.
- **Bessel series, $\sigma = \beta$.** Li et al. print $(a^2/2)[J_1(\sigma a)]^2$
  for the case $\sigma = \beta$ of Eq. 3; the limit of the general expression is
  $(a^2/2)[J_0(\sigma a)^2 + J_1(\sigma a)^2]$, which is what is used (for
  $|\sigma - \beta|\,a$ below the cube root of the machine epsilon of the working precision, using the value at $(\sigma + \beta)/2$).
- **Bessel functions.** $J_0$ and $J_1$ come from `optix.special`
  ([`bessel_j0`][optix.special.bessel_j0], [`bessel_j1`][optix.special.bessel_j1]),
  accurate to about $10^{-15}$; the Bessel functions of the PyTorch library
  are only accurate to about $5\times 10^{-7}$ in double precision for
  arguments below 25, which is enough to spoil the Bessel-series fit.
- **Radial evaluation.** Each voxel is evaluated at its own radius; Li et al.
  interpolate a twice oversampled radial profile on the Cartesian grid.

**Opt-in extension (not in the papers)**

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `oversample` | averages $s$ points per axis inside every voxel instead of sampling at its centre: the PSF integrated over the voxel | `1` (point sampling) | `1`; use $2$ or $3$ when the voxel is larger than about a third of the resolution |
| `method` | `"li2017"` replaces quadrature by the Bessel series of Li et al. (a separate published method, not the paper's own evaluation) | `"quadrature"` | `"quadrature"`; `"li2017"` for speed near the surface |
| `rtol` | accuracy target of the automatic node count | `1e-6` | `1e-6` |
| `nodes` | fixes the node count instead | `None` (automatic) | `None` |
| `basis`, `samples` | $M$ and $K$ of the Bessel series | `100`, `1000` | `100`, `1000` |

## Memory and streaming

There is no input data to stream: the PSF is generated from its parameters. The
cost is that of one matrix product per chunk of radii, of size
(distinct radii) $\times$ (nodes) $\times$ (axial slices); the radii are
processed in chunks that respect the configured memory budget
([`optix.config`][optix.config]), and the axial slices of the output are filled
the same way. The number of distinct radii is much smaller than $K_y K_x$, because the PSF is
radially symmetric; it grows with `oversample`, which multiplies the number of
samples by $s$ along each axis. The cost grows with the number of nodes, and therefore with depth,
defocus and the lateral extent of the grid. The working dtype and device follow
the configuration or the `dtype` and `device` arguments; the phase and the
arguments of the Bessel functions are always computed in double precision (single
on devices without it), and only the products are done in the working dtype.

## Usage

```pycon
>>> import torch
>>> from optix.psf import gibson_lanni
>>> kw = dict(spacing=(0.25, 0.1), wavelength=0.61, numerical_aperture=1.4)
>>> shape = (33, 65, 65)

>>> # An oil-immersion objective at its design conditions: aberration-free,
>>> # with the full NA 1.4 aperture.
>>> ideal = gibson_lanni(shape, **kw)
>>> ideal.shape, round(float(ideal.sum()), 6)
(torch.Size([33, 65, 65]), 1.0)
>>> gibson_lanni(shape, return_estimate=True, **kw).meta["aperture"]
1.0

>>> # The same objective imaging a point 10 µm deep in water (n = 1.33).
>>> deep = gibson_lanni(shape, depth=10.0, **kw)
>>> float(deep.max() / ideal.max()) < 0.5
True

>>> # The estimate carries the FWHM (in voxels) and the parameters used; the
>>> # aperture is cut at n_s / NA = 0.95 once the specimen has a thickness.
>>> est = gibson_lanni(shape, depth=10.0, return_estimate=True, **kw)
>>> [round(float(v), 1) for v in est.fwhm]
[5.8, 2.9, 2.9]
>>> est.meta["method"], est.meta["nodes"], round(est.meta["aperture"], 2)
('quadrature', 128, 0.95)

>>> # The Bessel series of Li et al. is faster; compare the two.
>>> fast = gibson_lanni(shape, depth=10.0, method="li2017", **kw)
>>> bool((fast - deep).abs().max() < 0.01 * deep.max())
True

>>> # The optical path difference, in µm, at the edge of the aperture (ρ = 0.95);
>>> # it is defined for ρ ≤ n_s / NA only, and is NaN beyond.
>>> from optix.psf.utils import gibson_lanni_opd
>>> opd = gibson_lanni_opd(
...     0.95, 0.0, numerical_aperture=1.4, depth=10.0, specimen_index=1.33
... )
>>> round(float(opd), 2)
-4.3
```

Common variations: a 2-value `spacing` $(d_z, d_{xy})$; `defocus` to centre the
stack away from the paraxial focus (Δz of the central slice); `coverslip_index`
and `coverslip_thickness` for a non-standard coverslip (for instance
`coverslip_thickness=140.0`); `immersion_index` for another oil or water
immersion, together with the `design_*` values of the objective (a water-dipping
objective, which has no coverslip, is `design_immersion_index=1.333` with
`design_coverslip_thickness=0.0`; a coverslip-corrected water objective keeps
the glass values of the coverslip);
`oversample=2`; and `dtype`, `device` for the working precision and device. The
PSF can be given directly to [`RichardsonLucy`](../deconv/richardson-lucy.md) and
the other deconvolution estimators. See the
[API reference](../../reference/psf.md).

## References

!!! references "References"

    1. Gibson SF, Lanni F. *Experimental test of an analytical model of aberration
       in an oil-immersion objective lens used in three-dimensional light
       microscopy.* J Opt Soc Am A **8**(10):1601-1613, 1991.
       [doi:10.1364/JOSAA.8.001601](https://doi.org/10.1364/JOSAA.8.001601)
    2. Li J, Xue F, Blu T. *Fast and accurate three-dimensional point spread
       function computation for fluorescence microscopy.* J Opt Soc Am A
       **34**(6):1029-1034, 2017.
       [doi:10.1364/JOSAA.34.001029](https://doi.org/10.1364/JOSAA.34.001029)

    ??? quote "BibTeX"

        ```bibtex
        @article{gibson1991experimental,
          author  = {Gibson, Sarah Frisken and Lanni, Frederick},
          title   = {Experimental test of an analytical model of aberration in an oil-immersion objective lens used in three-dimensional light microscopy},
          journal = {Journal of the Optical Society of America A},
          volume  = {8},
          number  = {10},
          pages   = {1601--1613},
          year    = {1991},
          doi     = {10.1364/JOSAA.8.001601},
        }

        @article{li2017fast,
          author  = {Li, Jizhou and Xue, Feng and Blu, Thierry},
          title   = {Fast and accurate three-dimensional point spread function computation for fluorescence microscopy},
          journal = {Journal of the Optical Society of America A},
          volume  = {34},
          number  = {6},
          pages   = {1029--1034},
          year    = {2017},
          doi     = {10.1364/JOSAA.34.001029},
        }
        ```
