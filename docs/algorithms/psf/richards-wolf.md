# Richards–Wolf

!!! info "At a glance"
    **Estimates:** none; a model, not an estimator. [`richards_wolf`][optix.psf.richards_wolf] returns the PSF as a tensor of the requested shape \
    **Noise model:** none (a noise-free forward model) \
    **Streaming:** none (the whole PSF is computed at once; see [Memory and streaming](#memory-and-streaming)) \
    **Needs many frames:** no \
    **Reference:** Richards and Wolf, Proc R Soc Lond A, 1959, [doi:10.1098/rspa.1959.0200](https://doi.org/10.1098/rspa.1959.0200)

## Overview

[`richards_wolf`][optix.psf.richards_wolf] computes the point-spread function (PSF) of a
high-aperture objective from electromagnetic diffraction theory. A point source is
imaged by an aplanatic lens, one that is free of spherical aberration and obeys the
sine condition. The function evaluates the time-averaged electric energy density near
the focus, in the form given by
[Richards and Wolf (1959, Eq. 3.14)](https://doi.org/10.1098/rspa.1959.0200). Unlike
the scalar (Airy, Gibson–Lanni) models, it keeps the vector nature of light: at a
numerical aperture (NA) above about 0.7 the longitudinal field component and the
polarisation of the illumination change the shape of the spot, which becomes wider
along the direction of polarisation.

Use it for a high-NA objective in a homogeneous medium, in the absence of aberrations.
For the scalar PSF of an objective with an immersion-to-sample interface, see the
sibling model Gibson–Lanni (`gibson_lanni`), and for a PSF measured on beads see
[Bead PSF](bead-psf.md). The result is a real tensor, centred at index `n // 2` of
every axis, that can be given directly to the [deconvolution](../../reference/deconv.md)
estimators (with `normalize="sum"`).

The paper treats a linearly polarised or an unpolarised incident wave. Two opt-in
extensions go beyond it: the detection PSF of a fluorescent dipole emitter
(`emitter`) and a user-supplied aberration phase (`pupil_phase`). Both are described
separately [below](#faithful-mode-and-extensions).

## Image formation (optics)

The PSF is the image of a point source at infinity, formed by an objective of
numerical aperture $\mathrm{NA}$ in an immersion medium of refractive index $n$, at
vacuum wavelength $\lambda_0$. The symbols of this page are in the
[optics table of the notation page](../../concepts/notation.md#optics). The
aperture on the image side is the half-angle

$$
\alpha = \arcsin\frac{\mathrm{NA}}{n},
\qquad k = \frac{2\pi n}{\lambda_0},
$$

and a point near the focus is given in cylindrical coordinates: the distance $\rho$
from the optical axis, the azimuth $\varphi$ measured from the electric vector of the
incident wave, and the distance $z$ from the Gaussian focus along the axis, positive
away from the lens. The time dependence is $e^{-i\omega t}$. A recorded image
is the convolution of this PSF with the specimen, as in
[Image formation and noise](../../concepts/image-formation.md).

Light is focused through a spherical wavefront centred on the Gaussian focus. The
sine condition maps a ray entering at height $h$ onto the focal sphere of radius $f$
at the same height, $h = f \sin\theta$. Conservation of energy through the area
element then gives an amplitude that grows as $\sqrt{\cos\theta}$ towards the edge of
the pupil (the *apodisation* of an aplanatic system), and the polarisation vector
of every ray stays in its meridional plane. Integrating the plane waves of the
aperture over the solid angle gives the field in terms of three integrals
$I_0$, $I_1$, $I_2$.

=== "optix"

    $$
    \begin{aligned}
    I_0 &= \int_0^\alpha \sqrt{\cos\theta}\,\sin\theta\,(1+\cos\theta)\,
           J_0(k\rho\sin\theta)\,e^{ikz\cos\theta}\,\mathrm{d}\theta,\\
    I_1 &= \int_0^\alpha \sqrt{\cos\theta}\,\sin^2\theta\,
           J_1(k\rho\sin\theta)\,e^{ikz\cos\theta}\,\mathrm{d}\theta,\\
    I_2 &= \int_0^\alpha \sqrt{\cos\theta}\,\sin\theta\,(1-\cos\theta)\,
           J_2(k\rho\sin\theta)\,e^{ikz\cos\theta}\,\mathrm{d}\theta,
    \end{aligned}
    $$

    and the electric field, for incident light linearly polarised along $x$ (with
    $A = 1$), is

    $$
    \mathbf{e}(\rho, \varphi, z) =
    \begin{pmatrix}
    -i\,(I_0 + I_2 \cos 2\varphi)\\
    -i\,I_2 \sin 2\varphi\\
    -2\,I_1 \cos\varphi
    \end{pmatrix}.
    $$

=== "Richards and Wolf 1959"

    $$
    \begin{aligned}
    I_0 &= \int_0^\alpha \cos^{\frac12}\theta\,\sin\theta\,(1+\cos\theta)\,
           J_0(k r_P \sin\theta \sin\theta_P)\,e^{ikr_P\cos\theta\cos\theta_P}\,\mathrm{d}\theta,\\
    I_1 &= \int_0^\alpha \cos^{\frac12}\theta\,\sin^2\theta\,
           J_1(k r_P \sin\theta \sin\theta_P)\,e^{ikr_P\cos\theta\cos\theta_P}\,\mathrm{d}\theta,\\
    I_2 &= \int_0^\alpha \cos^{\frac12}\theta\,\sin\theta\,(1-\cos\theta)\,
           J_2(k r_P \sin\theta \sin\theta_P)\,e^{ikr_P\cos\theta\cos\theta_P}\,\mathrm{d}\theta,
    \end{aligned}
    \qquad (2.32)
    $$

    $$
    e_x = -iA\,(I_0 + I_2 \cos 2\phi_P),\quad
    e_y = -iA\,I_2 \sin 2\phi_P,\quad
    e_z = -2A\,I_1 \cos\phi_P
    \qquad (2.30)
    $$

    with $A = \pi f l_0/\lambda$ (2.28).

| optix | Richards and Wolf 1959 |
|---|---|
| $\lambda_0$ (`wavelength`) | $\lambda$ (the refractive index of the image space is 1) |
| $n$ (`refractive_index`) | 1 (assumed) |
| $\alpha$ (from `numerical_aperture`) | $\alpha$ |
| $k = 2\pi n/\lambda_0$ | $k = 2\pi/\lambda$ (2.5) |
| $\rho$, $z$ | $r_P \sin\theta_P$, $r_P \cos\theta_P$ |
| $\varphi$ | $\phi_P$ |
| $k\rho$ (`radial`), $kz$ (`axial`) | $k r_P \sin\theta_P$, $k r_P \cos\theta_P$ |
| $\mathbf{e}$ | $\mathbf{e}$, with $A = 1$ |
| $\theta$ (integration variable) | $\theta$ |

The paper also writes the integrals in the optical coordinates
$u = kz\sin^2\alpha$ and $v = k\rho\sin\alpha$ (3.1a). optix works with $k\rho$ and
$kz$ directly, so that the refractive index does not enter anywhere else.

## Probabilistic model

The model is deterministic: the PSF is a function of the optics and of the position,
and no noise is modelled. The detector is assumed to record the electric energy
density, which for a time-harmonic field is $\langle w_e\rangle = |\mathbf{e}|^2/16\pi$
(Eq. 3.13). Substituting the field of Eq. 2.30 gives, for light polarised along $x$,

=== "optix"

    $$
    \mathrm{PSF}(\rho, \varphi, z) = \frac{1}{16\pi}
    \Bigl\{ |I_0|^2 + 4\,|I_1|^2 \cos^2\varphi + |I_2|^2
    + 2\cos 2\varphi\;\mathrm{Re}\bigl(I_0 I_2^*\bigr) \Bigr\}
    $$

=== "Richards and Wolf 1959"

    $$
    \langle w_e\rangle = \frac{A^2}{16\pi}
    \Bigl\{ |I_0|^2 + 4\,|I_1|^2 \cos^2\phi_P + |I_2|^2
    + 2\cos 2\phi_P\;\mathscr{R}\bigl(I_0 I_2^*\bigr) \Bigr\}
    \qquad (3.14)
    $$

The two forms agree with $A = 1$. For light polarised at an azimuth $\varphi_0$ from
the $x$ axis, $\varphi$ is replaced by $\varphi - \varphi_0$ (Eq. 5.1). For
unpolarised light the energy density is averaged over $\varphi_0$ (Eq. 5.2), which
makes the spot radially symmetric:

$$
\overline{\mathrm{PSF}} = \frac{1}{16\pi}\bigl\{ |I_0|^2 + 2\,|I_1|^2 + |I_2|^2 \bigr\}.
$$

Properties that follow from the equations, and that the paper states:

- $I_n(-z) = I_n(z)^*$ (Eq. 3.3), so that the PSF is symmetric about the focal plane
  (Eq. 3.15), but, for $\alpha > 0$, not about the axis.
- On the axis, $I_1 = I_2 = 0$ and $\mathrm{PSF} = |I_0|^2/16\pi$ (Eq. 3.17). At the
  focus
  $I_0 = \tfrac{16}{15}\bigl\{1 - \tfrac58\cos^{3/2}\alpha\,(1 + \tfrac35\cos\alpha)\bigr\}$
  (Eq. 3.18).
- For $\alpha \to 0$, the PSF tends to the Airy pattern $\propto (2J_1(v)/v)^2$ in the
  focal plane (Eq. 4.12), and to the Lommel–Struve distribution near the focus
  (Eq. 4.8).

## Inference / derivation

Nothing is estimated. What remains is to evaluate the three integrals on every
voxel. With $s = \sqrt{\cos\theta}$, so that $\sin\theta\,\mathrm{d}\theta = -2s\,\mathrm{d}s$,
the integrals become

$$
\begin{aligned}
I_0 &= \int_{\sqrt{\cos\alpha}}^{1} 2s^2(1+s^2)\,J_0(k\rho\sin\theta)\,e^{ikz s^2}\,\mathrm{d}s,\\
I_1 &= \int_{\sqrt{\cos\alpha}}^{1} 2s^2\sin\theta\,J_1(k\rho\sin\theta)\,e^{ikz s^2}\,\mathrm{d}s,\\
I_2 &= \int_{\sqrt{\cos\alpha}}^{1} 2s^2(1-s^2)\,J_2(k\rho\sin\theta)\,e^{ikz s^2}\,\mathrm{d}s,
\end{aligned}
\qquad \sin\theta = \sqrt{1 - s^4}.
$$

The functions $J_0(b\sin\theta)$, $\sin\theta\,J_1(b\sin\theta)$ and
$J_2(b\sin\theta)$ are even in $\sin\theta$, hence analytic functions of
$\sin^2\theta = 1 - s^4$. All three integrands are therefore analytic on the whole
interval, up to $\alpha = 90°$ where the factor $\sqrt{\cos\theta}$ of the original
form is singular. This is why Gauss–Legendre quadrature in $s$ converges
exponentially: the error drops to the level of floating-point precision as soon as
the nodes resolve the oscillations of the integrand, which are set by the largest
$k\rho\sin\alpha$ and $|kz|(1 - \cos\alpha)$ of the request.

The Bessel functions are evaluated by [`bessel_j012`][optix.special.bessel_j012],
accurate to about $10^{-15}$: below an argument of 25 by a trapezoidal rule on
Bessel's integral. The azimuthal dependence is analytic and needs no integration, since
Eq. 2.29 has already been used to bring it to $\cos 2\varphi$, $\sin 2\varphi$ and
$\cos\varphi$.

## Algorithm

```text
Input: shape (Z, Y, X) or (Y, X), wavelength λ₀, NA, n = 1, spacing,
       polarization = "x", emitter = None, pupil_phase = None, normalize = None
α = arcsin(NA / n);   k = 2π n / λ₀
Grid: z, y, x = (index − size // 2) · spacing;   ρ = √(x² + y²),  φ = atan2(y, x)
nodes N = 24 + ⌈½ (max kρ · sin α + max |kz| · (1 − cos α))⌉     (unless given)
Gauss–Legendre nodes s_q and weights w_q on [√cos α, 1];   cos θ_q = s_q²
For every z and every distinct ρ of the grid:
    Iₙ = Σ_q  gₙ(θ_q) · Jₙ(kρ sin θ_q) · exp(i (kz cos θ_q + ψ(θ_q)))     (n = 0, 1, 2)
    # gₙ: the weights 2 s² w_q (1 + s²), 2 s² w_q sin θ, 2 s² w_q (1 − s²);  ψ = 0 by default
Gather Iₙ back onto the voxels
PSF = (1 / 16π) · energy density of the chosen case (see below)
Normalise:  None (A = 1), "focus" (value 1 at the Gaussian focus), or "sum" (unit sum)
Return PSF, shape `shape`
```

The energy density of the chosen case is, with $a_n = |I_n|^2$ and
$c = \cos 2(\varphi - \varphi_0)$:

| Case | Value inside the $1/16\pi$ |
|---|---|
| linear, azimuth $\varphi_0$ (`"x"`: 0, `"y"`: $\pi/2$) | $a_0 + 2a_1(1 + c) + a_2 + 2c\,\mathrm{Re}(I_0 I_2^*)$ |
| `"unpolarized"` | $a_0 + 2a_1 + a_2$ |

Here $4|I_1|^2\cos^2(\varphi - \varphi_0) = 2a_1(1 + c)$.

## Faithful mode and extensions

`richards_wolf` is a function, not an estimator, and has no `faithful` flag: its
defaults compute the quantity of the paper, and every extension is an explicit
keyword that is off by default.

**Faithful.** The default output is the time-averaged electric energy density
$\langle w_e\rangle$ of Eq. 3.14 for light polarised along $x$, with $A = 1$, for the
field of Eq. 2.30 and the integrals of Eq. 2.32. The paper's other cases are options
that remain within it: `polarization` set to `"y"`, to an angle $\varphi_0$ in radians
(Eq. 5.1) or to `"unpolarized"` (Eq. 5.2); `normalize="focus"` or `"sum"`, which only
rescale the result.

**Choices that the paper does not specify.**

- *Amplitude.* The overall factor $A^2 = (\pi f l_0/\lambda)^2$ depends on the focal
  length and on the amplitude of the incident wave. The output uses $A = 1$.
- *Refractive index.* The paper takes the image space as vacuum ($n = 1$). For
  $n \neq 1$ the same formulas are used in a homogeneous medium, with
  $\alpha = \arcsin(\mathrm{NA}/n)$ and $k = 2\pi n/\lambda_0$.
- *Grid.* The Gaussian focus is the voxel at index `n // 2` of each axis, the first
  spatial axis of a 3D shape is $z$ (positive away from the lens), and the $x$ axis
  is the last one. A 2D shape is the focal plane.
- *Conventions.* The time dependence is $e^{-i\omega t}$.
- *Quadrature.* The scheme and the number of nodes (given above) are optix's.

**Keywords.** `emitter` and `pupil_phase` are the extensions, which are not in the
paper. The others stay within it, except that `refractive_index` other than 1 and
`nodes` are choices of optix (see above).

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `emitter` | `(px, py, pz)` or `"isotropic"`: the detection PSF of a dipole emitter, instead of the energy density of an incident wave | `None` | `None` for an illumination PSF; `"isotropic"` for the image of a fluorescent point |
| `pupil_phase` | a callable $\psi(\theta)$, in radians, multiplied into the integrands as $e^{i\psi(\theta)}$ | `None` | `None` |
| `polarization` | `"x"`, `"y"`, an angle or `"unpolarized"` (the paper's cases) | `"x"` | the incident polarisation (not used with `emitter`) |
| `normalize` | `None`, `"focus"` or `"sum"` | `None` | `"sum"` for deconvolution |
| `refractive_index` | $n$ of the image space | `1.0` | the immersion medium |
| `nodes` | number of quadrature nodes | `None` (automatic) | `None` |

**Emitter.** The signal at an image point from a dipole $\mathbf{p}$ at a point of
the object space is, by reciprocity, $\sum_j |\mathbf{p}\cdot\mathbf{e}^{(j)}|^2$, with
$\mathbf{e}^{(j)}$ the focal field of Eq. 2.30 for a point source at the image point
and $j$ running over the two orthogonal polarisations $x$ and $y$ (a camera records
both). The field for $y$ is that of $x$ rotated by $90°$ about the axis:

$$
\mathbf{e}^{(x)} = -i\,(I_0 + I_2\cos 2\varphi,\; I_2\sin 2\varphi,\; -2i\,I_1\cos\varphi),
\qquad
\mathbf{e}^{(y)} = -i\,(I_2\sin 2\varphi,\; I_0 - I_2\cos 2\varphi,\; -2i\,I_1\sin\varphi),
$$

and for a unit dipole $\mathbf{p}$ (normalised from `emitter`) the PSF is

$$
\mathrm{PSF}_{\mathbf{p}} = \frac{1}{16\pi}\Bigl\{
|\mathbf{p}\cdot\mathbf{e}^{(x)}|^2 + |\mathbf{p}\cdot\mathbf{e}^{(y)}|^2\Bigr\}.
$$

Averaging over orientations uniform on the sphere, with
$\langle|\mathbf{p}\cdot\mathbf{e}|^2\rangle = |\mathbf{e}|^2/3$, gives for
`emitter="isotropic"`

$$
\mathrm{PSF}_\mathrm{iso} = \frac{1}{16\pi}\cdot\frac23
\bigl\{ |I_0|^2 + 2\,|I_1|^2 + |I_2|^2 \bigr\},
$$

proportional to Eq. 5.2. The $\sqrt{\cos\theta}$ apodisation is kept: a dipole
radiates an intensity per unit solid angle that the sine condition spreads over a
pupil area $f^2\cos\theta\,\mathrm{d}\Omega$, so the pupil amplitude is
$\propto 1/\sqrt{\cos\theta}$ and the pupil area element contributes
$\cos\theta$, leaving $\sqrt{\cos\theta}$ as in Eq. 2.32. The derivation assumes a
low-aperture image side, a camera that sums both polarisations and a homogeneous
medium without interface (no refractive-index mismatch).

**Pupil phase.** `pupil_phase` is the aberration function $\Phi$ of Eq. 2.2, which
the paper sets to zero for an aplanatic system (Eq. 2.7), restricted to a function of
$\theta$ alone, as $\psi(\theta) = k\Phi(\theta)$. It must not depend on the
azimuth, so that the azimuthal integration of Eq. 2.29 still holds. It is the hook
that couples the model with, for instance, the defocus and index-mismatch phase of
Gibson–Lanni.

## Memory and streaming

The function is not streamed: it returns the whole PSF, of the requested `shape`, and
is meant for kernels of a few hundred voxels per axis. Two measures keep the cost low.

- The integrals depend on a voxel only through $z$ and $\rho$. They are computed for
  each axial position and each *distinct* radius of the grid (the quadrant symmetry
  of the grid, and the diagonal symmetry for square pixels, make most radii repeat)
  and gathered back onto the voxels.
- For each axial position the sum over nodes is a matrix product, $(Z, N)$ by
  $(N, R)$, with $R$ the number of distinct radii. The $R$ axis is processed in
  chunks sized from the memory budget of [`optix.config`][optix.config], so that the
  Bessel work arrays stay within it.

The cost grows as $Z \cdot R \cdot N$, with $N$ growing with the optical size of the
grid. A $64 \times 256^2$ PSF takes about a second on one CPU thread. The
function runs on any torch device; `dtype` and `device` default to the
configuration. In float32 the result agrees with float64 to about $4 \cdot 10^{-7}$
of the peak; in float64 the quadrature error is about $10^{-13}$ of the peak.

## Usage

The PSF of an oil-immersion objective (NA 1.4, $n = 1.518$) at 520 nm, on a grid of
50 nm lateral and 100 nm axial voxels. Lengths are in micrometres, the unit of the
wavelength.

```pycon
>>> from optix.psf import richards_wolf
>>> kw = dict(wavelength=0.52, numerical_aperture=1.4, refractive_index=1.518)
>>> psf = richards_wolf(
...     (33, 65, 65), spacing=(0.1, 0.05, 0.05), normalize="sum", **kw
... )
>>> psf.shape, round(float(psf.sum()), 6)
(torch.Size([33, 65, 65]), 1.0)
>>> tuple(int(i) for i in (psf == psf.max()).nonzero()[0])
(16, 32, 32)
```

The maximum is at the Gaussian focus, index `(16, 32, 32)`. For light polarised along
$x$, the focal plane is wider along $x$ than along $y$:

```pycon
>>> plane = richards_wolf((65, 65), spacing=0.05, normalize="focus", **kw)
>>> round(float(plane[32, 32 + 3]), 3), round(float(plane[32 + 3, 32]), 3)
(0.351, 0.1)
```

With `polarization="y"` the pattern is transposed, and unpolarised light gives a
radially symmetric spot:

```pycon
>>> unpol = richards_wolf(
...     (65, 65), spacing=0.05, polarization="unpolarized", normalize="focus", **kw
... )
>>> round(float(unpol[32, 32 + 6]), 6) == round(float(unpol[32 + 6, 32]), 6)
True
```

The building blocks are in `optix.psf.utils`: the complex field of Eq. 2.30 on a
grid, and the integrals at given optical distances $k\rho$ and $kz$. The value of
$I_0$ at the focus can be compared with the closed form of Eq. 3.18:

```pycon
>>> import math
>>> from optix.psf.utils import richards_wolf_field, richards_wolf_integrals
>>> field = richards_wolf_field((65, 65), spacing=0.05, **kw)
>>> field.shape, field.dtype
(torch.Size([3, 65, 65]), torch.complex64)
>>> alpha = math.asin(1.4 / 1.518)
>>> i0, i1, i2 = richards_wolf_integrals(0.0, 0.0, alpha)
>>> exact = (
...     16
...     / 15
...     * (1 - 5 / 8 * math.cos(alpha) ** 1.5 * (1 + 3 / 5 * math.cos(alpha)))
... )
>>> abs(float(i0.real) - exact) < 1e-5
True
```

The extensions are keywords. The detection PSF of a freely rotating fluorescent
molecule, and a PSF with a pupil aberration (here a small spherical-aberration term):

```pycon
>>> iso = richards_wolf(
...     (33, 33, 33), spacing=0.05, emitter="isotropic", normalize="sum", **kw
... )
>>> iso.shape
torch.Size([33, 33, 33])
>>> sph = richards_wolf(
...     (33, 33, 33),
...     spacing=0.05,
...     normalize="sum",
...     **kw,
...     pupil_phase=lambda theta: 2.0 * (theta / alpha) ** 4,
... )
>>> sph.shape
torch.Size([33, 33, 33])
```

**Validation.** The implementation is checked against closed forms and limits of the
paper: the value at the focus (Eq. 3.18), conjugate symmetry (Eq. 3.3), the Airy
pattern in the focal plane (Eq. 4.12) and the $\mathrm{sinc}^2(u/4)$ profile on the axis
at low NA, symmetry about the focal plane (Eq. 3.15), the transposition between the
two linear polarisations and the radial symmetry of unpolarised light. Table 1 of the
paper is reproduced to within 0.03 on its normalised scale of 100 at the focus, except
for two entries: the second minimum at $\alpha = 30°$ (7.03 in the paper, 7.25 here,
where the $S_z$ row of the same table reads 7.02) and the first minimum of the $S_z$
row at $\alpha = 30°$ (3.80 in the paper, 3.83 here). The paper's $\alpha \to 0$
column also differs from the exact Airy values by up to 0.01. The axial energy flow through the focal plane equals the power entering the
pupil, $\int_0^\infty (|I_0|^2 - |I_2|^2)\,v\,\mathrm{d}v = 2\sin^4\alpha$ with
$v = k\rho\sin\alpha$ (from Eq. 3.21 and Parseval's theorem on Eq. 2.2; the identity is
not stated in the paper), and the implementation satisfies it. The emitter extension is validated only through its limits, not against another
implementation.

## References

!!! references "References"

    1. Richards B, Wolf E. *Electromagnetic diffraction in optical systems II.
       Structure of the image field in an aplanatic system.* Proc R Soc Lond A
       **253**(1274):358-379, 1959.
       [doi:10.1098/rspa.1959.0200](https://doi.org/10.1098/rspa.1959.0200)

    ??? quote "BibTeX"

        ```bibtex
        @article{richards1959electromagnetic,
          author  = {Richards, B. and Wolf, E.},
          title   = {Electromagnetic diffraction in optical systems {II}. {Structure} of the image field in an aplanatic system},
          journal = {Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences},
          volume  = {253},
          number  = {1274},
          pages   = {358--379},
          year    = {1959},
          doi     = {10.1098/rspa.1959.0200},
        }
        ```
