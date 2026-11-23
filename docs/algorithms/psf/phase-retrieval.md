# Phase retrieval

!!! info "At a glance"
    **Estimates:** `PupilEstimate` (`pupil`, `zernike_magnitude`, `zernike_phase`) \
    **Noise model:** none (a noise-free forward model; noise is reduced by the 2D pupil constraint) \
    **Streaming:** whole-stack (the sections used and the pupil of each group are in memory) \
    **Needs many frames:** no (a z-stack of one bead, or of an averaged bead; at least two sections) \
    **Reference:** Hanser, Gustafsson, Agard and Sedat, J Microsc, 2004, [doi:10.1111/j.0022-2720.2004.01393.x](https://doi.org/10.1111/j.0022-2720.2004.01393.x)

## Overview

[`PhaseRetrieval`][optix.psf.PhaseRetrieval] estimates the **pupil function** of a
wide-field fluorescence microscope from a measured three-dimensional PSF. The pupil
function is the magnitude and the phase of the wavefront that a point source produces
at the exit pupil of the objective. It is a compact description: a 2D complex image
instead of a 3D PSF, so that the noise of the measurement is suppressed, and it can be
modified, for instance by multiplying it with the aberration of a refractive-index
mismatch, to predict the PSF under other imaging conditions. The magnitude and the
phase are also expanded in Zernike polynomials, which gives a short list of
aberration coefficients.

The intensity PSF is a squared magnitude, so the phase of the pupil is not measured
and must be retrieved. The method iterates between the pupil plane and a few defocused
sections of the PSF, imposing the measured magnitude in the image and the numerical
aperture (NA) in the pupil (Fig. 2 of the paper).

Use it to turn a bead acquisition into an aberration-aware PSF, shift-invariant over
the region sampled, that [`pupil_psf`][optix.psf.pupil_psf] and
[`zernike_psf`][optix.psf.zernike_psf] can render on any grid. Compared with the
empirical PSF of [Bead PSF](bead-psf.md), it is smooth and noise-free, can be changed
(defocus, depth, aberrations) and is described by a few numbers, but it assumes a
scalar, non-saturated wide-field model and does not keep structure that a pupil cannot
represent. The result, a [`PupilEstimate`][optix.estimates.PupilEstimate], has no
`apply`: the PSF is rendered from it. The input may be a raw stack, a labelled array,
a `PSFEstimate` or a fitted [`BeadPSF`][optix.psf.BeadPSF]; leading dimensions are
groups, with one pupil each.

## Image formation (optics)

The image of a point source is the squared magnitude of an amplitude PSF
$h_{\mathrm A}$, and a recorded image is the convolution of the intensity PSF
$h = |h_{\mathrm A}|^2$ with the specimen (see
[Image formation and noise](../../concepts/image-formation.md)). The Fourier transform
of $h_{\mathrm A}$ is non-zero only on a spherical cap of radius $n/\lambda_0$ and
angular half-aperture $\alpha$, where $n$ is the refractive index of the immersion
medium and $\lambda_0$ the vacuum wavelength. The cap is described by its projection
on the lateral frequency plane, the pupil, of radius $\mathrm{NA}/\lambda_0$, and the
shape of the cap is carried by the axial frequency
$\nu_z(\boldsymbol{\nu}) = \sqrt{(n/\lambda_0)^2 - |\boldsymbol{\nu}|^2}$. The symbols
of this page are in the [pupil function table](../../concepts/notation.md#pupil-function)
of the notation page.

### Imaging model

=== "optix"

    $$
    h_{\mathrm A}(\mathbf{r}, z) = \iint_{|\boldsymbol{\nu}| \le \mathrm{NA}/\lambda_0}
    p(\boldsymbol{\nu})\, e^{2\pi i\, \nu_z(\boldsymbol{\nu})\, z}\,
    e^{2\pi i\, \boldsymbol{\nu} \cdot \mathbf{r}}\, \mathrm{d}\boldsymbol{\nu},
    \qquad h = |h_{\mathrm A}|^2 ,
    $$

    with $\mathbf{r} = (x, y)$ the lateral position and $\boldsymbol{\nu} = (\nu_x, \nu_y)$.

=== "Hanser et al. 2004"

    $$
    PSF_A(x, y, z) = \iint_{\text{pupil}} P(k_x, k_y)\, e^{2\pi i (k_x x + k_y y)}\,
    e^{2\pi i k_z(k_x, k_y)\, z}\, \mathrm{d}k_x\, \mathrm{d}k_y
    \qquad (2)
    $$

    with $k_z = \sqrt{(n/\lambda)^2 - (k_x^2 + k_y^2)}$.

| optix | Hanser et al. 2004 |
|---|---|
| $h_{\mathrm A}$, $h = \lvert h_{\mathrm A}\rvert^2$ | $PSF_A$, $PSF$ |
| $p(\boldsymbol{\nu})$ | $P(k_x, k_y)$ |
| $\boldsymbol{\nu}$, $\nu_z$ (cycles per unit length) | $(k_x, k_y)$, $k_z$ |
| $\lambda_0$ | $\lambda$ |
| $n$, $\mathrm{NA}$ | $n$, $NA$ |
| $z$ | $z$ |

The paper's $k$ is a frequency in cycles per unit length, not the wave number
$2\pi n/\lambda_0$ of the notation page; it is written $\boldsymbol{\nu}$ here.

The factor $e^{2\pi i \nu_z z}$ is the *defocus phase function*: it refocuses the pupil
by a distance $z$, and the expression is exact for all apertures because it only
states that the OTF lies on a sphere. Three consequences are used by the algorithm.

- **Sections.** One lateral section at depth $z_j$ is a 2D Fourier transform
  of $p$ times the defocus phase function.
- **Discretisation.** The pupil is sampled on the discrete Fourier grid of the
  lateral image grid, $\nu = (i - K/2)/(K \Delta)$ for $K$ pixels of size $\Delta$ along
  an axis, with $\nu = 0$ at index `K // 2`. The radius of the pupil is
  $\mathrm{NA}\, K \Delta / \lambda_0$ pixels, so the field of view sets the
  sampling of the pupil. The transform is unitary.
- **Piston.** The term $n z/\lambda_0$ of $\nu_z$ is a global phase per section that
  cancels between the forward and the backward defocus, so it is dropped, and
  $\nu_z - n/\lambda_0 = -|\boldsymbol{\nu}|^2/(n/\lambda_0 + \nu_z)$ is computed in
  this form, which has no cancellation.

### Corrections applied after the retrieval

The retrieval itself assumes a point source and a scalar field. Three refinements
of the paper bring the calculated PSF closer to a measured one, and the paper applies
them after the retrieval (they could also be put inside the loop). They are options of
[`pupil_psf`][optix.psf.pupil_psf] and [`zernike_psf`][optix.psf.zernike_psf].

**Bead size.** A bead of diameter $D$, uniformly fluorescent, multiplies the
intensity OTF by the Fourier transform of a sphere:

=== "optix"

    $$
    b(\nu) = 3\, s(\pi D \nu), \qquad
    s(x) = \frac{\sin x}{x^3} - \frac{\cos x}{x^2}
    $$

    with $\nu$ the magnitude of the 3D frequency and $b(0) = 1$ (`bead_diameter`).

=== "Hanser et al. 2004"

    $$
    b(k) = 3\, h(\pi k d), \qquad h(x) = \frac{\sin x}{x^3} - \frac{\cos x}{x^2}
    $$

    (page 36; $d$ is the bead diameter)

**Vectorial nature of light.** The emitters are modelled as three incoherent dipoles,
along $x$, $y$ and $z$. Each contributes the intensity of two field components
polarised along $x$ and $y$ in the camera plane, so the PSF is the sum of six
intensity PSFs, each computed from the pupil times a real factor of the polar angle
$\theta$ ($\sin\theta = \lambda_0 |\boldsymbol{\nu}|/n$, the sine condition) and the
azimuth $\varphi$ of $\boldsymbol{\nu}$ (`vectorial=True`):

=== "optix"

    $$
    \begin{aligned}
    q_{xx} &= \cos\theta \cos^2\varphi + \sin^2\varphi, &
    q_{xy} = q_{yx} &= (\cos\theta - 1) \sin\varphi \cos\varphi, \\
    q_{yy} &= \cos\theta \sin^2\varphi + \cos^2\varphi, &
    q_{zx} &= \sin\theta \cos\varphi, \quad q_{zy} = \sin\theta \sin\varphi,
    \end{aligned}
    \qquad
    h = \sum_{c} \bigl|\,\mathbf{F}\,(\mathbf{q}_c \odot \mathbf{p} \odot \mathbf{u}_z)\bigr|^2 ,
    $$

    where $\mathbf{q}_c$ runs over the six factors, $\mathbf{u}_z$ is the defocus phase function
    and $\mathbf{F}$ the lateral transform (Eq. 2).

=== "Hanser et al. 2004"

    $$
    \begin{array}{ll}
    P_x \to E_x: \cos\theta \cos^2\varphi + \sin^2\varphi &
    P_x \to E_y,\ P_y \to E_x: (\cos\theta - 1) \sin\varphi \cos\varphi \\
    P_y \to E_y: \cos\theta \sin^2\varphi + \cos^2\varphi &
    P_z \to E_x: \sin\theta \cos\varphi, \quad P_z \to E_y: \sin\theta \sin\varphi
    \end{array}
    \qquad (3)
    $$

The factor $1/\sqrt{\cos\theta}$ of the wavefront compression is absorbed in the
pupil function, and the six squared factors add up to 2 everywhere, so the
overall scale is unchanged. The factors are the same as in the classical calculation of
Richards and Wolf.

**Empirical OTF rescaling.** The paper found that the high spatial frequencies of
phase-retrieved PSFs are systematically over-emphasised, and attenuates them with
a Gaussian function of the spatial frequency, fitted to the ratio between the OTF of a
measured PSF and that of the PSF calculated from its pupil (`otf_sigma`,
[`fit_otf_rescaling`][optix.psf.utils.fit_otf_rescaling]):

$$
g(\nu) = \exp\!\Bigl(-\frac{\nu^2}{2\sigma^2}\Bigr).
$$

### Depth-dependent index mismatch

A point source at depth $d$ below the cover slip, in a medium of index $n_2$, imaged by
an objective designed for an immersion index $n_1$ ($= n$), is aberrated: the pupil
function is multiplied by a complex function of the pupil position. A ray that
reaches the pupil at the angle $\theta_1$ left the source at the angle $\theta_2$, with
$n_1 \sin\theta_1 = n_2 \sin\theta_2$ (Snell's law). The phase is the change of the
optical path, the magnitude combines the Fresnel transmission of the interface
(averaged over the two polarisations) and the *wavefront compression* that
conserves energy:

=== "optix"

    $$
    \begin{aligned}
    \psi(\theta_1) &= \frac{2\pi}{\lambda_0}\, d\,\bigl(n_2 \cos\theta_2 - n_1 \cos\theta_1\bigr),\\
    A_{\mathrm t} &= \frac{\sin\theta_1 \cos\theta_2}{\sin(\theta_1 + \theta_2)}
    \Bigl[1 + \frac{1}{\cos(\theta_2 - \theta_1)}\Bigr],\qquad
    A_{\mathrm w} = \frac{n_1 \tan\theta_2}{n_2 \tan\theta_1},\\
    p'(\boldsymbol{\nu}) &= p(\boldsymbol{\nu})\, A_{\mathrm t} A_{\mathrm w}\, e^{i\psi(\theta_1)},
    \qquad \sin\theta_1 = \frac{\lambda_0}{n_1}\, |\boldsymbol{\nu}| .
    \end{aligned}
    $$

=== "Hanser et al. 2004"

    $$
    \begin{aligned}
    OP &= d\,\bigl(n_2 \cos\theta_2 - n_1 \cos\theta_1\bigr) &(4)\\
    A_1 &= \frac{\sin\theta_1 \cos\theta_2}{\sin(\theta_1 + \theta_2)}
    \Bigl[1 + \frac{1}{\cos(\theta_2 - \theta_1)}\Bigr] &(5)\\
    A_w &= \frac{n_1 \tan\theta_2}{n_2 \tan\theta_1} &(6)\\
    \sin\theta_1 &= \frac{\lambda}{n_1}\sqrt{k_x^2 + k_y^2} &(7)\\
    \psi(k_x, k_y, d, n_1, n_2) &= A\, e^{i 2\pi OP/\lambda},\quad A = A_1 A_w &(8)\\
    P' &= P\, \psi / \lambda &(9)
    \end{aligned}
    $$

The aberration is computed by
[`index_mismatch_aberration`][optix.psf.index_mismatch_aberration], which returns the
complex factor $A_{\mathrm t} A_{\mathrm w} e^{i\psi}$ on the pupil grid.

| optix | Hanser et al. 2004 |
|---|---|
| $d$ | $d$ (depth) |
| $n_1$, $n_2$ | $n_1$, $n_2$ |
| $\theta_1$, $\theta_2$ | $\theta_1$, $\theta_2$ |
| $\psi(\theta_1)$ (phase) | $2\pi OP/\lambda$ |
| $A_{\mathrm t}$, $A_{\mathrm w}$ | $A_1$, $A_w$ |
| $A_{\mathrm t} A_{\mathrm w} e^{i\psi}$ | $\psi(k_x, k_y, d, n_1, n_2)$ (Eq. 8) |

The paper uses $\psi$ for the whole complex function; here $\psi$ is the phase, as in
the optics symbols of the notation page. The expressions are evaluated in equivalent
forms that are finite on the axis ($\theta_1 = 0$):
$A_{\mathrm t} = \cos\theta_2\,[1 + 1/\cos(\theta_2 - \theta_1)]/(\cos\theta_2 + (n_1/n_2)\cos\theta_1)$
and $A_{\mathrm w} = (n_1/n_2)^2 \cos\theta_1/\cos\theta_2$. Rays beyond the critical angle
($n_1 \sin\theta_1 > n_2$) do not reach the pupil, and get $\psi = 0$ and a zero magnitude.
The constant $1/\lambda$ of Eq. 9 only scales the PSF and is omitted.

## Probabilistic model

The model is deterministic: the data are sections of the intensity PSF, assumed
noise-free, and the unknown is the pupil function. Noise is not modelled; it is
suppressed because the 3D PSF is described by a 2D function.

Let $\mathbf{y}_j$, $j = 1, \ldots, J$, be the measured sections (background
subtracted, at least two, about four in the paper) at the defocus $z_j$ from the
in-focus section, $\mathbf{u}_j$ the defocus phase function of $z_j$ on the pupil grid,
$\mathbf{F}$ the unitary lateral transform from the pupil plane to the image plane, and
$\mathbf{S}$ the projection on the pupil, $|\boldsymbol{\nu}| \le \mathrm{NA}/\lambda_0$
(multiplication by its indicator). The pupil function must satisfy

=== "optix"

    $$
    \bigl|\mathbf{F}\,(\mathbf{u}_j \odot \mathbf{p})\bigr|^2 = \mathbf{y}_j,
    \quad j = 1, \ldots, J,
    \qquad \mathbf{p} = \mathbf{S}\,\mathbf{p},
    $$

    and it is searched as a minimiser of the magnitude misfit under the support constraint

    $$
    \min_{\mathbf{p} = \mathbf{S}\mathbf{p}}\ \sum_{j=1}^{J}
    \bigl\| \lvert \mathbf{F}(\mathbf{u}_j \odot \mathbf{p}) \rvert - \sqrt{\mathbf{y}_j} \bigr\|^2 .
    $$

=== "Hanser et al. 2004"

    The same problem is stated in words: the pupil function is estimated from
    measured sections of the PSF such that the magnitude of
    $PSF_A$ (Eq. 2) equals the square root of the measured intensity, with a
    non-zero pupil only inside the objective NA (Fig. 2). The paper gives no objective.

| optix | Hanser et al. 2004 |
|---|---|
| $\mathbf{y}_j$ | measured PSF section (Fig. 2A) |
| $\mathbf{p}$ | pupil function (Fig. 2B, 2F, 2G) |
| $\mathbf{u}_j$ | defocus phase function $e^{i 2\pi k_z z}$ |
| $\mathbf{F}$ | FT (Fig. 2) |
| $\mathbf{S}$ | NA constraint (Fig. 2G) |

After the retrieval the magnitude and the phase of the pupil are expanded
in Zernike polynomials:

=== "optix"

    $$
    p(r, \varphi) = m(r, \varphi)\, e^{i\psi(r, \varphi)},
    \qquad
    m = \sum_{j=1}^{J_{\mathrm Z}} a_j Z_j(r, \varphi),
    \qquad
    \psi = \sum_{j=1}^{J_{\mathrm Z}} b_j Z_j(r, \varphi),
    \qquad r \le 1,
    $$

    with $r = |\boldsymbol{\nu}|\lambda_0/\mathrm{NA}$ the radius normalised to 1 at the
    edge of the pupil, and $\varphi$ the azimuth.

=== "Hanser et al. 2004"

    "The basis chosen was the Zernike polynomials. We used 79 basis
    functions, consisting of all the Zernike polynomials up through radial order 11,
    plus the purely radial polynomial of order 12. The magnitude and phase components
    of the pupil functions were fitted separately to the polynomials using a linear
    least-squares fitting routine." (page 37)

| optix | Hanser et al. 2004 / code |
|---|---|
| $a_j$, $b_j$ | `zernike_magnitude`, `zernike_phase` |
| $Z_j(r, \varphi)$ | Zernike polynomial of Noll index $j$ (`zernike(j, rho, theta)`) |
| $r$, $\varphi$ | `rho`, `theta` in the code (the radius and azimuth of the pupil grid) |
| $J_{\mathrm Z}$ | `zernike_terms` (79) |

The polynomials are those of Noll (1976), whom the paper cites; they are
orthonormal on the unit disc, $\tfrac{1}{\pi}\iint Z_i Z_j\, r\, \mathrm{d}r\, \mathrm{d}\varphi = \delta_{ij}$, so
a coefficient is the root-mean-square contribution of its term. With the radial order $n$ and
the azimuthal frequency $m$ of the Noll index $j$
([`noll_index`][optix.psf.utils.noll_index]),

$$
Z_j = \begin{cases}
\sqrt{n+1}\, R_n^0(r) & m = 0,\\
\sqrt{2(n+1)}\, R_n^{|m|}(r) \cos m\varphi & m > 0 \ (j \text{ even}),\\
\sqrt{2(n+1)}\, R_n^{|m|}(r) \sin |m|\varphi & m < 0 \ (j \text{ odd}),
\end{cases}
$$

with $R_n^m$ the radial polynomials. The 79 terms are the indices $1 \ldots 79$: the
$12 \times 13 / 2 = 78$ polynomials up to radial order 11 and the first polynomial of order
12, which is the radial one.

## Inference / derivation

The magnitude constraint is non-convex, and the paper follows the family of
Gerchberg–Saxton algorithms, extended by Hanser et al. (2003) to high-NA optics
through the exact defocus phase function. The iteration alternates between the two
constraint sets of the problem above, and each step has a closed form.

1. **Magnitude projection.** For each section, the pupil of the current iterate
   $\mathbf{p}^{(t)}$ gives an amplitude section $\mathbf{g}_j = \mathbf{F}(\mathbf{u}_j \odot \mathbf{p}^{(t)})$.
   The closest field with the measured magnitude keeps the phase of $\mathbf{g}_j$ (0
   where $g_j = 0$) and takes the magnitude $\sqrt{y_j}$:

    $$
    \mathbf{g}_j' = \sqrt{\mathbf{y}_j}\ \odot\ \frac{\mathbf{g}_j}{\lvert\mathbf{g}_j\rvert}.
    $$

2. **Back to the pupil.** The field is transformed back and refocused for zero defocus,
   $\mathbf{u}_j^{*} \odot \mathbf{F}^{\dagger} \mathbf{g}_j'$ (the defocus is a
   unit-magnitude factor, so its inverse is its conjugate).

3. **Consensus and support.** The $J$ pupils are averaged and cut at the aperture,

    $$
    \mathbf{p}^{(t+1)} = \mathbf{S}\ \frac{1}{J} \sum_{j=1}^{J}
    \mathbf{u}_j^{*} \odot \mathbf{F}^{\dagger} \mathbf{g}_j' .
    $$

    Since $\mathbf{p} \mapsto \mathbf{F}(\mathbf{u}_j \odot \mathbf{p})$ is an isometry on
    the pupils supported in $\mathbf{S}$, this average is the least-squares solution of
    $\min_{\mathbf{p} = \mathbf{S}\mathbf{p}} \sum_j \lVert \mathbf{F}(\mathbf{u}_j \odot \mathbf{p}) - \mathbf{g}_j'\rVert^2$.
    Each iteration is therefore an alternating projection, and without smoothing the
    amplitude error $\sum_j \lVert\,|\mathbf{g}_j| - \sqrt{\mathbf{y}_j}\rVert^2 / \sum_j \lVert\mathbf{y}_j\rVert_1$
    cannot increase.

4. **Smoothing (optional).** The paper allows a smoothing filter at this point, whose
   form it does not give (see [below](#faithful-mode-and-extensions)).

The starting guess is $\mathbf{p}^{(0)} = \mathbf{S}\mathbf{1}$: one inside the aperture,
zero outside (Fig. 2B). The iteration stops after a fixed number of iterations
(25 in the paper).

**Quality measures.** At each iteration, with $\hat{\mathbf{y}}_j = |\mathbf{g}_j|^2$ the
calculated sections and all voxels of the $J$ sections compared,

$$
\mathrm{iMSE} = \frac{\sum (\hat{\mathbf{y}} - \mathbf{y})^2}{\sum \mathbf{y}^2},
$$

the intensity mean-square error, and the correlation coefficient between $\hat{\mathbf{y}}$ and
$\mathbf{y}$, a normalisation-independent supplementary measure, are recorded
(`history.extras["imse"]`, `["correlation"]`; `history.objective` is $-\mathrm{iMSE}$).
The final values, over the whole stack, are in `meta["imse"]` and `meta["correlation"]`.
The paper reports the iMSE but does not give its normalisation.

**Zernike fit.** After the loop the global phase of the pupil is arbitrary: it is
fixed so that $\sum_{\boldsymbol\nu} p$ is real and positive. The magnitude is fitted as it is. The
phase is first unwrapped by [`unwrap_phase`][optix.psf.utils.unwrap_phase], following
the paper's "simple linear extrapolation method": each pixel is moved by a multiple of
$2\pi$ to the value closest to $2\psi_1 - \psi_2$, the linear extrapolation of the two
previous pixels of its path. The paths start at the centre pixel, run along the central
column, then outwards along each row. Each map $v$ is then fitted by

$$
\hat{\mathbf{c}} = \arg\min_{\mathbf{c}} \sum_{r \le 1}
\Bigl(v(r, \varphi) - \sum_{j=1}^{J_{\mathrm Z}} c_j Z_j(r, \varphi)\Bigr)^2 ,
$$

unweighted over the pixels of the pupil, by linear least squares
([`zernike_fit`][optix.psf.utils.zernike_fit]). The fit fails with an error when the
pupil has fewer pixels than terms: a larger lateral field of view, or lateral padding
(after background subtraction), gives a finer pupil.

## Algorithm

```text
Input: stack (…, Z, Y, X), spacing (Δz, Δy, Δx), λ₀, NA, n,
       offsets = (−3, −1, 1, 3), focus = brightest section, iterations = 25,
       smoothing = none, zernike_terms = 79, unwrap = True
Per group:
    j₀ = focus; sections i_j = j₀ + round(offset_j / Δz); z_j = (i_j − j₀) Δz
    y_j = max(section i_j, 0), scaled so that the mean energy of the sections
          equals the number of pixels of the pupil
    p = indicator of |ν| ≤ NA/λ₀                         # starting guess
    for t = 0 … iterations:
        g_j = F (u_j ⊙ p)                                # u_j: defocus phase function of z_j
        record iMSE, correlation, amplitude error
        if t = iterations: stop
        g'_j = √y_j · g_j / |g_j|                        # keep the phase
        p = S · mean_j ( conj(u_j) ⊙ Fᴴ g'_j )           # refocus, average, NA support
        if smoothing: p = S · Gaussian(p)                # optional, off by default
    rotate p by a constant phase so that Σ p > 0
    quality (iMSE, correlation) over the whole stack
    unwrap the phase; fit magnitude and unwrapped phase by 79 Zernike terms
Return PupilEstimate(pupil, zernike_magnitude, zernike_phase), meta, history
```

## Faithful mode and extensions

`faithful=True` reproduces the procedure of the paper with its published settings: the
starting guess, the four defocused sections, 25 iterations, no smoothing in
the loop, the Zernike fit of 79 terms after phase unwrapping. Both presets
(`faithful=True` and `False`) are the paper's settings; the estimator is reported as
faithful (`is_faithful`) as long as no option below is changed.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `offsets` | defocus of the sections used, relative to `focus`, each selecting the nearest section; `"all"` uses every section | `(-3, -1, 1, 3)` (the unit of `spacing`, µm) | same |
| `focus` | index of the in-focus section | `None`: the section with the brightest voxel | same |
| `iterations` | number of iterations | `25` | `25` |
| `smoothing` | standard deviation, in pupil pixels, of a Gaussian smoothing of the pupil at every iteration (extension) | `None` | `None` |
| `zernike_terms` | Noll-ordered terms in each fit | `79` | `79` |
| `unwrap` | unwrap the phase before its fit | `True` | `True` |

Only `spacing`, `wavelength`, `na` and `n` are required; they describe the data and do
not change the faithful status.

The rendering options of [`pupil_psf`][optix.psf.pupil_psf] and
[`zernike_psf`][optix.psf.zernike_psf] implement the corrections of the paper, all
off by default and meant to be applied after the retrieval:

| Keyword | Changes | Default |
|---|---|---|
| `vectorial` | six dipole components added incoherently (Eq. 3) instead of a scalar PSF | `False` |
| `bead_diameter` | multiplies the 3D OTF by the OTF of a spherical bead; needs uniformly spaced `z` | `None` |
| `otf_sigma` | multiplies the 3D OTF by the Gaussian rescaling of width $\sigma$ (cycles per unit length) | `None` |
| `normalize` | divides the PSF by its sum | `True` |

### Choices the paper does not specify

Each of the following is a choice of the implementation.

- **Smoothing filter.** The paper has an optional smoothing step but does not describe
  the filter. Faithful mode does not smooth; the Gaussian of `smoothing` pupil pixels,
  applied to the real and imaginary parts, is an opt-in extension. The optional
  stopping criterion of the paper is not implemented: the number of iterations is
  fixed.
- **iMSE normalisation.** $\sum(\hat{\mathbf{y}} - \mathbf{y})^2/\sum \mathbf{y}^2$ over the voxels compared.
- **Data scaling and sign.** The sections are scaled so that their mean energy equals that of the
  starting pupil (its number of pixels), so that the retrieved magnitude is close to 1.
  Negative intensities, left by the background subtraction, are set to 0 before the
  square root.
- **Section selection.** The sections nearest to the offsets; it is an error when two
  offsets select the same section or when a section falls outside the stack.
- **Global phase of the pupil.** Fixed so that the sum of the pupil is real and positive.
- **Phase-unwrapping path.** The linear extrapolation is described; its path (centre,
  central column, then rows) is chosen.
- **Zernike convention and weighting.** Noll's indexing and normalisation; the fits are
  unweighted over the pupil pixels.
- **OTF-rescaling Gaussian.** The paper fits "a Gaussian" of the spatial frequency. It is
  taken isotropic in the 3D frequency, equal to 1 at $\nu = 0$ (so the PSF sum is
  kept), and its width is found by a golden-section search on the least-squares
  criterion over the OTF support of an objective of the numerical aperture given to
  [`fit_otf_rescaling`][optix.psf.utils.fit_otf_rescaling]. The paper fitted on the
  support of NA 1.15 for an NA 1.3 objective, where the measured OTF is above the noise.
- **Rays beyond the critical angle.** $\psi = 0$ and a zero magnitude in the index-mismatch
  aberration.
- **Constant factor $1/\lambda$ of Eq. 9.** Omitted.
- **Numerical form of the defocus factor.** Written as above without cancellation; a
  test checks that it equals the direct form.

## Memory and streaming

The algorithm is *whole-stack*: `PhaseRetrieval` reads the data once and in one piece
(`data[...]`), then uses only the $J$ selected sections. The iteration holds, per group,
$J$ complex fields of the size of the lateral image, $K_y \times K_x$, plus
the defocus phase function of each; the final quality is computed on all $Z$ sections, at
the cost of one more complex array of $Z K_y K_x$ per group. Each iteration costs
$2J$ 2D FFTs per group, and the Zernike fit solves a least-squares problem of
(pupil pixels) $\times$ 79 once per group. Groups are processed together in a single
batch. To lower the memory for a large stack, crop it laterally around the bead, to the
size that the pupil sampling requires, and pass only the sections needed with `focus`
set to the in-focus section of the cropped stack. [`pupil_psf`][optix.psf.pupil_psf] and
[`zernike_psf`][optix.psf.zernike_psf] hold the whole $(Z, Y, X)$ PSF.

## Usage

```pycon
>>> import torch
>>> from optix.psf import PhaseRetrieval, zernike_psf
>>> optics = dict(wavelength=0.52, na=1.3, n=1.515)  # µm, oil objective
>>> spacing = (1.0, 0.0566, 0.0566)  # (Δz, Δy, Δx) in µm

>>> # 1. A synthetic stack: 0.5 rad of coma (Noll j = 8), 0.3 rad of spherical (j = 11).
>>> phase = torch.zeros(79)
>>> phase[7], phase[10] = 0.5, 0.3
>>> data = zernike_psf(None, phase, shape=(9, 128, 128), spacing=spacing, **optics)

>>> # 2. Retrieve the pupil and its Zernike expansion.
>>> est = PhaseRetrieval(spacing=spacing, **optics).fit(data)
>>> pupil = est.estimates["pupil"]
>>> pupil.pupil.shape
torch.Size([128, 128])
>>> [round(float(pupil.zernike_phase[j - 1]), 2) for j in (8, 11)]
[0.49, 0.3]

>>> # 3. Quality: the iMSE of the pupil of the last iteration, and the number of iterations.
>>> est.history.extras["imse"][-1] < 1e-3
True
>>> len(est.history.objective), est.history.reason
(26, '25 iterations')
```

Rendering a PSF from the result, on a new grid, with the corrections of the paper:

```pycon
>>> psf = zernike_psf(
...     pupil.zernike_magnitude,
...     pupil.zernike_phase,
...     shape=(33, 33, 33),
...     spacing=spacing,
...     vectorial=True,
...     bead_diameter=0.121,
...     **optics,
... )
>>> psf.shape, round(float(psf.sum()), 4)
(torch.Size([33, 33, 33]), 1.0)
```

The pupil of the microscope under an index mismatch, a source $5\,\mu$m deep in water
($n_2 = 1.33$) for an oil objective, is the retrieved pupil times the aberration:

```pycon
>>> from optix.psf import index_mismatch_aberration, pupil_psf
>>> aberration = index_mismatch_aberration(
...     (128, 128),
...     spacing[1:],
...     depth=5.0,
...     n1=1.515,
...     n2=1.33,
...     wavelength=0.52,
...     na=1.3,
... )
>>> deep = pupil_psf(
...     pupil.pupil * aberration,
...     [-1.0, 0.0, 1.0],
...     spacing=spacing[1:],
...     **optics,
... )
>>> deep.shape
torch.Size([3, 128, 128])
```

Common variations: leading dimensions of the stack are groups (one pupil per channel
or per bead) and are kept in the `(*group, Ky, Kx)` shape of `pupil`; a fitted
[`BeadPSF`][optix.psf.BeadPSF] or its `PSFEstimate` is accepted in place of an array
(the averaged PSF of beads; its voxel size is given with `spacing`); `offsets="all"`
to use every section; `smoothing=1.0` to smooth the pupil at every iteration (not
faithful); the shortcut `phase_retrieval(data, ...)`, which returns the
`PupilEstimate`; `fit_otf_rescaling(measured, calculated, ...)` to calibrate
`otf_sigma`. A [`PupilEstimate`][optix.estimates.PupilEstimate] records the optics
and the sections used in `meta`. See the [API reference](../../reference/psf.md).

The retrieval is validated on synthetic aberrations with and without Poisson noise
(recovery of the Zernike coefficients), against the Airy pattern for an aberration-free
pupil, against the steps of Fig. 2 written out by hand (one iteration, to 1e-10), and
for the orthonormality of the Zernike basis and the agreement of single and double
precision.

## References

!!! references "References"

    1. Hanser BM, Gustafsson MGL, Agard DA, Sedat JW. *Phase-retrieved pupil functions
       in wide-field fluorescence microscopy.* J Microsc **216**(1):32-48, 2004.
       [doi:10.1111/j.0022-2720.2004.01393.x](https://doi.org/10.1111/j.0022-2720.2004.01393.x)
    2. Hanser BM, Gustafsson MGL, Agard DA, Sedat JW. *Phase retrieval for high-numerical-aperture
       optical systems.* Opt Lett **28**(10):801-803, 2003.
       [doi:10.1364/OL.28.000801](https://doi.org/10.1364/OL.28.000801)
    3. Noll RJ. *Zernike polynomials and atmospheric turbulence.* J Opt Soc Am
       **66**(3):207-211, 1976.
       [doi:10.1364/JOSA.66.000207](https://doi.org/10.1364/JOSA.66.000207)

    ??? quote "BibTeX"

        ```bibtex
        @article{hanser2004phase,
          author  = {Hanser, B. M. and Gustafsson, M. G. L. and Agard, D. A. and Sedat, J. W.},
          title   = {Phase-retrieved pupil functions in wide-field fluorescence microscopy},
          journal = {Journal of Microscopy},
          volume  = {216},
          number  = {1},
          pages   = {32--48},
          year    = {2004},
          doi     = {10.1111/j.0022-2720.2004.01393.x},
        }
        @article{hanser2003phase,
          author  = {Hanser, B. M. and Gustafsson, M. G. L. and Agard, D. A. and Sedat, J. W.},
          title   = {Phase retrieval for high-numerical-aperture optical systems},
          journal = {Optics Letters},
          volume  = {28},
          number  = {10},
          pages   = {801--803},
          year    = {2003},
          doi     = {10.1364/OL.28.000801},
        }
        @article{noll1976zernike,
          author  = {Noll, R. J.},
          title   = {Zernike polynomials and atmospheric turbulence},
          journal = {Journal of the Optical Society of America},
          volume  = {66},
          number  = {3},
          pages   = {207--211},
          year    = {1976},
          doi     = {10.1364/JOSA.66.000207},
        }
        ```
