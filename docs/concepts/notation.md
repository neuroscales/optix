# Notation

One notation is used across every page of these docs and every docstring of
optix, whatever the algorithm and whatever the paper it comes from. This page
states the rules, lists the symbols, and says how the quantities that the
algorithms estimate relate to the physics of the microscope. Each algorithm
page shows its equations in this notation and, in a second tab, in the notation
of its paper.

## Typography

| Kind | Style | Examples |
|---|---|---|
| scalar | italic, lowercase | $g$, $o$, $\sigma^2$, $\lambda$, $b_n$ |
| vector: an image flattened over its pixels, or a set of per-frame values | bold lowercase | $\mathbf{y}_n$, $\mathbf{x}_n$, $\mathbf{f}$, $\mathbf{d}$, $\mathbf{b}$ |
| matrix or linear operator | bold uppercase | $\mathbf{A}$, $\boldsymbol{\Phi}$, $\mathbf{W}$, $\mathbf{Y}$ |
| count, size | plain uppercase, or the size of a set | $N$, $K$, $H$, $W$, $\lvert\Omega\rvert$ |
| Greek vector or matrix | bold Greek | $\boldsymbol{\alpha}$, $\boldsymbol{\pi}$, $\boldsymbol{\eta}$ |
| set, distribution family | calligraphic or upright | $\mathcal{N}$, $\mathrm{Gamma}$, $\mathcal{D}$ |

- Images are vectors. The pixels of a picture form the grid $\Omega$, with
  $\lvert\Omega\rvert = HW$ pixels in 2D ($DHW$ in 3D); pixel $i$ is
  written $i \in \Omega$. A picture is a vector
  $\mathbf{y}_n \in \mathbb{R}^{\lvert\Omega\rvert}$, and a stack of $N$ frames
  is the matrix
  $\mathbf{Y} = [\mathbf{y}_1, \ldots, \mathbf{y}_N] \in \mathbb{R}^{\lvert\Omega\rvert \times N}$,
  one frame per column. Pixel $i$ of frame $n$ is $y_{ni}$.
- Products of images are pixel-wise and written $\odot$; division of images
  is pixel-wise too. A scalar added to an image is broadcast:
  $\mathbf{x}_n + b_n$ means $\mathbf{x}_n + b_n \mathbf{1}$.
- The position of a pixel is rarely needed. When it is, it is written
  $\mathbf{r}$, and a pixel of an image $\mathbf{f}(\mathbf{r})$ or $f_i$.
- Hats mark estimates ($\hat{\mathbf{f}}$), bars means ($\bar{x}$).
- Probability distributions follow Bishop's [*Pattern Recognition and Machine
  Learning*](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) (2006). Some letters therefore have a second meaning there (the rate $b$
  of a Gamma is not a baseline); the context always tells which is meant.

### In docstrings

Docstrings write maths in UTF-8, not LaTeX, so that they read well in
`help()`, in an editor and in the API reference alike (see the
[style guide](../contributing/style.md)). Bold is not available there, so the
case carries the meaning: `yₙ = f ⊙ (xₙ + bₙ) + d`, `A x`, `Φ θ`, and the
number of pixels is `|Ω|`. The
symbols are the same as on this page.

Indices and exponents use Unicode subscripts and superscripts wherever they
exist: `xₙ`, `bₙ`, `fᵢ`, `σ²`, `Φᵀ`, `τ⁻¹`, `x̂ₙ`, `rₙₖ`, `α₀`, `10⁻⁶`. Superscripts
cover the digits, `⁺ ⁻ ⁼ ⁽ ⁾`, almost every lowercase letter and `ᵀ`.
Subscripts cover only the digits, `₊ ₋ ₌ ₍ ₎` and the letters
`ₐ ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ`; for any other index (`y_{ng}`,
`x_{max}`, `b_g`), fall back to `_` with braces when it spans several
characters. Do not mix the two styles within one symbol: `r_{nk}` or `rₙₖ`,
never `rₙ_k`.

## Image formation

| Symbol | Meaning | Unit | In code |
|---|---|---|---|
| $\mathbf{y}_n$, $\mathbf{Y}$ | recorded frame $n$; stack of frames | ADU | the data `Y` |
| $\mathbf{x}_n$ | latent image: frame $n$ without shading | ADU | result of `apply` |
| $\mathbf{f}$ | flat field (multiplicative) | dimensionless | `ShadingEstimate.flat` |
| $\mathbf{d}$ | dark field (additive, camera offset included) | ADU | `ShadingEstimate.dark` |
| $\mathbf{s}_b$ | background profile, mean 1, when the background has its own profile | dimensionless | `ShadingEstimate.dark` ÷ its mean |
| $\beta_n$, $\bar\beta$ | mean background intensity of frame $n$; its mean over frames | ADU | `KaskFlatField.background_level_` |
| $\kappa_{\text{n}}$ | noise constant of a channel: local variance over local mean on the background; it equals the gain $g$ only when the offset $o$ is 0, and is $g\,(\bar y - o)/\bar y$ otherwise (written $\kappa$ in docstrings) | ADU | `meta["noise_constant"]` |
| $b_n$, $\mathbf{b}$ | baseline of frame $n$; all baselines | ADU | `ShadingEstimate.baseline` |
| $\lambda$ | expected number of photoelectrons | e⁻ | `signal` of `simulate_sensor` |
| $e$ | actual number of photoelectrons (Poisson) | e⁻ | |
| $g$ | gain | ADU / e⁻ | `SensorEstimate.gain` |
| $o$ | camera offset | ADU | `SensorEstimate.offset` |
| $\sigma^2$ | read-noise variance | ADU² | `SensorEstimate.variance` |
| $\varepsilon$ | read noise, $\mathcal{N}(0, \sigma^2)$ | ADU | |
| $\Delta$ | quantisation step (1 ADU) | ADU | |

A camera model with one value per pixel, as for an sCMOS sensor, makes the
camera parameters vectors: $\mathbf{g}$, $\mathbf{o}$, $\boldsymbol{\sigma}^2$.

The two models are

$$
y = g\,e + o + \varepsilon, \quad e \sim \mathrm{Poisson}(\lambda)
\qquad\text{and}\qquad
\mathbf{y}_n = \mathbf{f} \odot (\mathbf{x}_n + b_n) + \mathbf{d},
$$

and correcting a frame means

$$
\hat{\mathbf{x}}_n = \frac{\mathbf{y}_n - \mathbf{d}}{\mathbf{f}} - b_n .
$$

The baseline sits *inside* the flat field: it is a uniform background of the
specimen (autofluorescence, fluorescence of the medium), lit by the same
uneven illumination as the signal.

### What the fields mean physically

The flat field, the dark field and the baseline are defined statistically:
the flat field is whatever multiplies every frame alike, the dark field
whatever is added to every frame alike, the baseline whatever is constant over
one frame. Written out, the expected recorded value of pixel $i$ of frame $n$
is

$$
\mathbb{E}[y_{ni}] =
g_i \Bigl(
\underbrace{s_i\, c_i\, \eta_i}_{\text{multiplicative}}\,(u_{ni} + a_{ni})
+ \underbrace{\ell_i + j_i\, t_{\text{exp}}}_{\text{additive}}
\Bigr) + o_i ,
$$

and each physical quantity lands in one of the estimated terms:

| Physical quantity | Symbol | Lands in |
|---|---|---|
| illumination profile | $s_i$ | flat $\mathbf{f}$ |
| collection efficiency, vignetting | $c_i$ | flat $\mathbf{f}$ |
| quantum efficiency and gain of each pixel (photo-response non-uniformity) | $\eta_i$, $g_i$ | flat $\mathbf{f}$ |
| specimen signal | $u_{ni}$ | latent $\mathbf{x}_n$ |
| autofluorescence, medium, out-of-focus haze | $a_{ni}$ | baseline $b_n$ where uniform, latent $\mathbf{x}_n$ otherwise |
| stray or ambient light, outside the illumination path | $\ell_i$ | dark $\mathbf{d}$ |
| dark current, over exposure time $t_{\text{exp}}$ | $j_i\, t_{\text{exp}}$ | dark $\mathbf{d}$ |
| camera offset, including its fixed pattern | $o_i$ | dark $\mathbf{d}$ |
| overall scale: lamp power, mean efficiency, mean gain | | shared by $\mathbf{f}$ and $\mathbf{x}_n$; fixed by `meta["flat_norm"]` |

So the flat field is not the illumination: it is illumination × collection ×
pixel response, which is why it overlaps with the gain map of an sCMOS camera.
And the dark field is not the camera offset: it is offset + dark current +
stray light. This is the reason why corrections never subtract both
`ShadingEstimate.dark` and `SensorEstimate.offset`.

## Optics

The symbols of the models of the optics (`optix.psf`), such as the
[Gibson–Lanni](../algorithms/psf/gibson-lanni.md) and Richards–Wolf PSFs.
Lengths share one unit; the Gibson–Lanni defaults are in µm.

| Symbol | Meaning | Unit | In code | Used by |
|---|---|---|---|---|
| $\mathrm{NA}$ | numerical aperture of the objective, $n \sin\alpha$ | | `numerical_aperture` | both |
| $\lambda_0$ | wavelength of the light in vacuum | length | `wavelength` | both |
| $k_0$ | wave number in vacuum, $2\pi/\lambda_0$ | 1 / length | | Gibson–Lanni |
| $k$ | wave number in a medium of index $n$, $n k_0$ | 1 / length | | Richards–Wolf |
| $n$ | refractive index of the medium in which the focus is formed | | `refractive_index` | Richards–Wolf |
| $n_s$, $n_g$, $n_i$ | refractive index of the specimen, the coverslip, the immersion medium | | `specimen_index`, `coverslip_index`, `immersion_index` | Gibson–Lanni |
| $n_g^*$, $n_i^*$ | the same, under the design conditions of the objective | | `design_coverslip_index`, `design_immersion_index` | Gibson–Lanni |
| $t_s$ | depth of the source below the coverslip | length | `depth` | Gibson–Lanni |
| $t_g$, $t_i$ | thickness of the coverslip, of the immersion layer | length | `coverslip_thickness` | Gibson–Lanni |
| $t_g^*$, $t_i^*$ | the same, under the design conditions | length | `design_coverslip_thickness`, `design_immersion_thickness` | Gibson–Lanni |
| $\alpha$ | angular semi-aperture of the objective, $\arcsin(\mathrm{NA}/n)$ | rad | | Richards–Wolf |
| $\theta$, $\varphi$ | angle of a ray to the optical axis; azimuth around the axis | rad | | both |
| $\rho$ | radius in the pupil (back focal plane), normalised to 1 at the edge of the aperture: $\rho = n \sin\theta / \mathrm{NA}$ | | `rho` | Gibson–Lanni |
| $r$ | lateral distance from the optical axis, in object space | length | `radius`; `radial` is $k r$ | both |
| $z$ | position along the optical axis, from the Gaussian focus, positive away from the lens | length | `axial` is $k z$ | Richards–Wolf |
| $\Delta z$ | defocus: displacement of the stage from the paraxial focus of the source, positive for a thicker immersion layer | length | `defocus` | Gibson–Lanni |
| $\mathrm{OPD}$ | optical path difference between the actual and the design ray | length | `gibson_lanni_opd` | Gibson–Lanni |
| $\psi$ | phase aberration of the pupil; $\psi = k_0\,\mathrm{OPD}$ | rad | `pupil_phase` | both |
| $\mathbf{e}$ | electric field near the focus | | `richards_wolf_field` | Richards–Wolf |
| $I_0$, $I_1$, $I_2$ | the diffraction integrals of the focal field | | `richards_wolf_integrals` | Richards–Wolf |
| $u$, $v$ | optical coordinates, $u = kz\sin^2\alpha$, $v = k r \sin\alpha$ | | | Richards–Wolf |
| $\varphi_0$ | azimuth of the electric vector of the incident wave, from the $x$ axis | rad | `polarization` | Richards–Wolf |
| $\mathbf{p}$ | orientation of an emitting dipole (unit vector) | | `emitter` | Richards–Wolf |
| $h$, $\mathbf{h}$ | the PSF as a function, $h(r, \varphi, z)$ or $h(r, \Delta z)$; the PSF sampled on the grid, centred at index `K // 2` | | the returned PSF | both |
| $K_z$, $K_y$, $K_x$ | shape of the PSF grid | | `shape` | both |
| $d_z$, $d_y$, $d_x$ | voxel size of the PSF grid | length | `spacing` | both |

Some letters have another meaning elsewhere on this page: here $n$ is a
refractive index, not a number of photoelectrons; $\lambda_0$ is a wavelength,
where $\lambda$ alone is the expected number of photoelectrons; $\theta$,
$\rho$, $\psi$, $\varphi$ and $\alpha$ are not the basis coefficients, the Stirling
residual, the digamma function, the function $\varphi(u)$ and the Dirichlet
concentration. A paper that writes the wavelength $\lambda$ and the vacuum
wave number $k$ is mapped to $\lambda_0$ and $k_0$ on its page. Docstrings
write `λ₀`, `k₀`.

## Bases and regularisation

| Symbol | Meaning | In code |
|---|---|---|
| $L$ | number of samples along an axis | |
| $\boldsymbol{\Phi}$ | DCT basis matrix, one function per column | `dct_basis` |
| $\boldsymbol{\theta}$, $\boldsymbol{\Theta}$ | coefficients in the basis (1D, 2D) | `project`, `reconstruct` |
| $\kappa_j$ | eigenvalue of the $j$-th basis function for the Laplacian | `dct_bending` |
| $h$, $\ell$ | grid spacing; order of the energy $\mathbf{f}^\top(-\boldsymbol{\Delta}_h)^\ell \mathbf{f}$ | `dct_bending(order=)` |
| $\mathbf{A}$ | forward operator (blur) | |
| $\lambda_{\mathrm{TV}}$ | weight of the total-variation penalty $\sum_p \lVert\nabla\mathbf{x}\rVert_p$ | `lambda_tv` |
| $\nabla$, $\operatorname{div}$ | gradient and divergence of an image (forward and backward differences) | `tv_curvature` |
| $\mathbf{W}$ | pixel weights | |
| $\lambda$ | weight of a regulariser (a subscript tells several apart: $\lambda_f$, $\lambda_d$); distinct from the photoelectron count by context | `regularization` |
| $\mathbf{L}_x$, $\mathbf{L}_y$ | second-difference operators along $x$ and $y$ | |

## Shift-variant blur

| Symbol | Meaning | In code |
|---|---|---|
| $S$, $s$ | number of terms of the blur, term index | |
| $\mathbf{h}_s$, $\mathbf{H}_s$ | kernel of term $s$; convolution with it | `ShiftVariantPSF.kernels` |
| $\mathbf{w}_s$, $\mathbf{W}_s$ | pixel weights of term $s$; $\operatorname{diag}(\mathbf{w}_s)$ | `ShiftVariantConvolution.weights` |
| $\mathbf{r}_s$ | position of the grid node of term $s$ | `centres` |
| $\mathbf{K}$ | matrix of the un-shifted PSFs, one column per source pixel | |
| $\boldsymbol{\varphi}_s$ | (multi)linear interpolation weights of node $s$ | `interpolation_weights` |
| $\mathbf{C}$ | matrix of the kernels $\mathbf{h}_s$ of the optimal local approximation, one per column | `DenisPSF.kernels` |
| $\tau$, $\mathbf{C}_\tau$ | truncation level; block-circulant preconditioner of Nagy and O'Leary, not to be confused with $\mathbf{C}$ | `tau`, `RegionPreconditioner` |
| $\mathbf{e}_k$ | residual of the conjugate-gradient iteration $k$ | |

The letter $\tau$ has a second meaning in the distributions below, the
precision of a Normal; the context tells which is meant.

## Morphology

| Symbol | Meaning | In code |
|---|---|---|
| $\boldsymbol{\beta}$ | background of an image, the opening of $\mathbf{y}$ | `output="background"` |
| $\mathbf{u}$ | offset (integer displacement) within a structuring element | |
| $B(\mathbf{u})$ | structuring element: its grey level at offset $\mathbf{u}$, $-\infty$ outside its support | `ball`, `footprint` |
| $\rho$ | radius of the ball, in pixels and intensity units | `radius` |
| $\ominus$, $\oplus$ | grey-scale erosion and dilation | `grey_erosion`, `grey_dilation` |

## Attenuation

Along the propagation axis $z$ of the illumination, with the index $j$ of the
pixels of a profile (here, and only here, $j$ counts depth pixels).

| Symbol | Meaning | In code |
|---|---|---|
| $y(z)$, $y_j$ | signal at depth $z$; value of pixel $j$ of a profile | the data of `DepthResolved` |
| $y^{\mathrm{c}}_j$ | compensated signal, $y_j\,\gamma_j$ | `DepthResolved.apply`, `compensate` |
| $y^{\mathrm{r}}_j$ | signal after the restoration filter of Smith et al. | `smith_attenuation` |
| $\mu(z)$, $\mu_j$ | attenuation coefficient, in the inverse of the length unit of $z$ | `DepthResolved.attenuation`, `depth_resolved` |
| $\hat{\mu}_j$, $\tilde{\mu}_q$ | estimate of $\mu$ of Smith et al., before and after the lateral denoising | `smith_attenuation`, `weighted_tv` |
| $s$ | number of times the light crosses the medium (2 for OCT, 1 for a light sheet); written $p$ in the docstrings | `passes` |
| $\Delta$ | pixel size along $z$ (also the quantisation step above) | `pixel_size` |
| $\xi$ | constant of the model: recorded fraction, conversion factor and incident irradiance | |
| $T_j$ | sum of the intensities from pixel $j$ to the end of the profile | `tail_sums` |
| $\gamma_j$ | compensation gain of pixel $j$ | `compensate` |
| $q$, $Q$ | lateral position and number of positions of the total-variation term | `tv_axis` |
| $w_q$ | weight of the difference between positions $q+1$ and $q$ in the total variation, $1/(y_q + \epsilon)$ | `weighted_tv` |
| $\eta$, $\epsilon$ | weight of the total variation, and floor of its weights; $\eta$ is also the natural parameters of a distribution, and $\epsilon$ (this glyph) differs from the read noise $\varepsilon$ | `eta`, `epsilon` |
| $\sigma^2$ | average noise power of the restoration filter (also the read-noise variance) | `noise_power` |

## Tiles

| Symbol | Meaning | In code |
|---|---|---|
| $I$ | number of tiles of a mosaic | `T` in docstrings |
| $i$, $j$ | tile indices (frame indices elsewhere) | |
| $\mathbf{y}_i$, $y_i(\mathbf{r})$ | tile $i$; its value at grid pixel $\mathbf{r}$ (the common pixel grid of the mosaic) | the tiles `data` |
| $a_i$, $c_i$ | scale and shift of tile $i$, $\hat{\mathbf{x}}_i = a_i \mathbf{y}_i + c_i$ | `TileIntensityEstimate.scale`, `.shift` |
| $\gamma_i$, $\omega_i$ | gain (dimensionless) and offset (ADU) of tile $i$ in a simulation, $\mathbf{y}_i = \gamma_i \mathbf{x} + \omega_i$; not the camera $g$, $o$ | `simulate_tiles(tile_gain=, tile_offset=)` |
| $v$, $w$ | a value of tile $i$ and the transformed value of a neighbour at the same pixel | |
| $a'_i$, $c'_i$, $c''_i$ | linear and additive sub-models of tile $i$ | |
| $\lambda_1$, $\lambda_2$, $\lambda_3$ | weights of the linear, additive and identity sub-models (not the photoelectron rate $\lambda$) | `weights` |
| $O_{ij}$ | overlap of tiles $i$ and $j$ | `overlap_box` |
| $F$ | set of fixed tiles | `fixed` |
## Intensity histograms

| Symbol | Meaning | In code |
|---|---|---|
| $G$ | number of grey levels | `levels` |
| $h(n)$, $p(n)$ | histogram of the grey levels $n = 1 \ldots G$, and its normalisation; $h$ is local to this section (it is the grid spacing elsewhere) | |
| $H(\cdot)$ | Shannon entropy of a histogram, in nats; an operator name in upright type, not a set | `meta["entropy"]` |
| $\mathbf{a}$, $\mathbf{m}$ | coefficients of the polynomial dark and flat fields: the paper's notation, not to be confused with the mean $m$ of a Normal | `meta["coefficients"]` |

## Stripes

| Symbol | Meaning | In code |
|---|---|---|
| $\mathbf{s}$ | stripe component of a frame, $\mathbf{y} = \mathbf{x} + \mathbf{s}$ | |
| $\theta$ | direction of the stripes, from the stripe axis towards the other in-plane axis | `angle` |
| $\nabla$, $\nabla_\theta$ | forward differences along the axes; along the stripe direction $\theta$ | `lattice_diff` |
| $\lVert \nabla \mathbf{x} \rVert_{2,1}$ | isotropic total variation: sum over pixels of the Euclidean norm of the gradient | |
| $\mu_1$, $\mu_2$ | weights of the total variation and of the sparsity of $\mathbf{s}$ (the Gamma mean $\mu$ is unrelated) | `mu1`, `mu2` |
| $\rho_z$ | weight of the differences along $z$ | `rho_z` |
| $\iota_{S}$ | indicator of a set $S$: $0$ inside, $\infty$ outside | |

## Distributions

These follow [Bishop (2006)](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/).

| Symbol | Meaning | In code |
|---|---|---|
| $\mathcal{N}(m, \tau^{-1})$ | Normal with mean $m$ and precision $\tau = 1/v$ | `Normal` |
| $\mathrm{Gamma}(a, b)$ | Gamma with shape $a$ and rate $b$ | `Gamma` |
| $\mu$, $\nu$ | Gamma mean $a/b$ and shape, in the form $(\mu, \nu)$ | `Gamma(mean=, shape=)` |
| $\boldsymbol{\alpha}$, $\alpha_0$ | Dirichlet concentration, and $\alpha_0 = \sum_k \alpha_k$ | `Dirichlet` |
| $\boldsymbol{\pi}$ | class probabilities | `Categorical.probs` |
| $z_n$, $r_{nk}$ | class of observation $n$; responsibility of class $k$ for it | |
| $T$, $\boldsymbol{\eta}$, $A$ | sufficient statistics, natural parameters, log-normaliser | `expected_stats`, `natural_params`, `log_normalizer` |
| $\bar{x}$, $M$ | weighted mean, and scatter about it | `NormalStats.mean`, `.scatter` |
| $\chi(x)$ | base measure of an exponential family | |
| $\psi$, $\Gamma$ | digamma and gamma functions | |
| $\rho(a)$, $\delta(a)$, $\varphi(u)$ | Stirling residual, $\log a - \psi(a)$, $u - 1 - \log u$ | `stirling_residual`, `log_minus_digamma`, `ratio_minus_log` |
| $\theta$ | difference of two means, in the pairwise merge | |
| $\mathbb{E}[\cdot]$, $\mathrm{KL}(q \Vert p)$ | expectation; Kullback-Leibler divergence | `kl_divergence(q, p)` |

## Pupil function

Used by the [phase retrieval](../algorithms/psf/phase-retrieval.md) page, with the
[optics symbols](#optics) above: the normalised pupil radius $\rho$, the pupil phase
$\psi$, the PSF $h$, the angles $\theta$, $\varphi$, the voxel size $d_z, d_y, d_x$ and the
indices $n$, $n_s$, $n_i$. The symbols below are specific to it.

| Symbol | Meaning | Unit | In code |
|---|---|---|---|
| $\boldsymbol{\nu}$, $\nu_z$ | lateral spatial frequency $(\nu_x, \nu_y)$, and axial frequency $\sqrt{(n/\lambda_0)^2 - \lvert\boldsymbol{\nu}\rvert^2}$, in cycles (not the wave number $k$) | 1 / length | |
| $h_{\mathrm A}$ | amplitude PSF (complex), $h = \lvert h_{\mathrm A}\rvert^2$ | | `pupil_psf` |
| $p(\boldsymbol{\nu})$, $\mathbf{p}$ | complex pupil function, $p = A\,e^{i\psi}$, non-zero for $\lvert\boldsymbol{\nu}\rvert \le \mathrm{NA}/\lambda_0$; its samples | | `PupilEstimate.pupil` |
| $A$ | magnitude of the pupil | | |
| $\mathbf{u}_j$ | defocus phase function $e^{2\pi i \nu_z z_j}$ of the section $z_j$ | | |
| $Z_j$ | Zernike polynomial of Noll index $j$, with radial order $q$ and azimuthal frequency $l$ | | `zernike` |
| $a_j$, $b_j$ | Zernike coefficients of $A$ and of $\psi$ | rad (for $b_j$) | `zernike_magnitude`, `zernike_phase` |
| $\delta$ | phase of the index-mismatch aberration, $k_0\,\mathrm{OPD}$ | rad | `index_mismatch_aberration` |
| $D$ | bead diameter | length | `bead_diameter` |
## Stationary noise

Noise made of known patterns convolved with white noise, as in
[VSNR](../algorithms/destripe/vsnr.md). The first rows are the stripe symbols
shared with the other destriping pages.

| Symbol | Meaning | In code |
|---|---|---|
| $\mathbf{s}$ | stripe or stationary noise component of a frame, $\mathbf{y} = \mathbf{x} + \mathbf{s}$ | |
| $\theta$ | direction of the stripes (in degrees in code), not the difference of two means nor the coefficients $\boldsymbol{\Theta}$ | `angle` |
| $\nabla$ | forward differences along the two axes of a frame | |
| $\lVert \nabla \mathbf{x} \rVert_{2,1}$ | isotropic total variation: sum over pixels of the Euclidean norm of the gradient | |
| $\iota_{S}$ | indicator of a set $S$: $0$ inside, $\infty$ outside | |
| $\mathbf{k}_i$ | known pattern $i$ of the noise: a kernel centred on its middle pixel (unrelated to the digamma function $\psi$) | `filters` |
| $\boldsymbol{\lambda}_i$ | white noise convolved with $\mathbf{k}_i$, so that $\mathbf{s} = \sum_i \boldsymbol{\lambda}_i * \mathbf{k}_i$ (unrelated to the photoelectrons $\lambda$) | `VSNRSolution.weights` |
| $*$ | convolution with periodic boundaries | |
| $\phi_i$, $\alpha_i$ | prior of $\boldsymbol{\lambda}_i$ and its weight | `prior`, `alpha` |
| $\epsilon$, $h_\epsilon$ | Huber parameter and function; $\lVert \cdot \rVert_{2,1}^{\epsilon}$ is the total variation with $h_\epsilon$ in place of the norm (unrelated to the read noise $\varepsilon$) | `eps` |
| $\vartheta$ | over-relaxation of a primal-dual scheme (unrelated to $\theta$) | |
## Views

| Symbol | Meaning | In code |
|---|---|---|
| $v$, $V$ | view index; number of views | `MultiviewRL(psfs)` |
| $\mathbf{y}_v$ | recorded image of view $v$ | the views `data` |
| $\mathbf{h}_v$, $\mathbf{A}_v$ | PSF of view $v$; its blur | `psfs` |
| $\check{\mathbf{h}}_v$ | mirrored PSF, $\check{h}(\mathbf{r}) = h(-\mathbf{r})$ (kernel of $\mathbf{A}_v^\top$) | |
| $\mathbf{h}_{b,v}$, $\mathbf{B}_v$ | back-projection kernel of view $v$ and its convolution | `MultiviewRL.kernels` |
| $\mathbf{c}_v$ | multiplicative correction of view $v$ | |
| $\mathbf{w}_v$, $k$ | weight image of view $v$; sum of the weights over the views | `weights`, `weight_sum` |
| $\rho_v$ | reach of view $v$ on the margin of the object domain | |

## Optimisation

| Symbol | Meaning | In code |
|---|---|---|
| $f_k$, $\mathbf{A}_k$ | function and circulant operator of term $k$ in $\sum_k f_k(\mathbf{A}_k\mathbf{x})$ | `Term.function`, `Term.operator` |
| $\mathbf{z}_k$ | split variable, $\mathbf{z}_k = \mathbf{A}_k\mathbf{x}$ | `AdmmResult.splits` |
| $\mathbf{u}_k$ | scaled dual variable of term $k$ | `AdmmResult.duals` |
| $\rho$ | ADMM penalty parameter | `rho` |
| $\nu$ | weight of a regulariser, $\nu\sum_j\lVert(\mathbf{H}\mathbf{x})_j\rVert_2$ | `Ikoma.nu` |
| $t$ | step size of a proximity operator | `t` of `prox` |
| $\mathrm{prox}_{tf}$, $\iota_{\mathcal{S}}$ | proximity operator of $f$; indicator function of a set ($0$ inside, $+\infty$ outside) | `optix.optim` |
| $\mathbf{D}_a$, $\mathbf{D}_{ab}$ | forward difference along axis $a$; second difference $\mathbf{D}_a\mathbf{D}_b$ | `diff` |
| $\nabla$, $\mathbf{H}$ | stacked first differences; Hessian, its distinct second differences with $\sqrt{2}$ on the off-diagonal ones | `Gradient`, `Hessian` |
| $\mathcal{C}$, $\Omega$ | observed voxels, where the data term applies; voxels where the regulariser applies | `Term.region` |
| $\tilde{\mathbf{y}}$, $\tilde\sigma^2$ | data in electrons, $(\mathbf{y} - \mathbf{d}) / (\mathbf{f}\,g)$; read-noise variance in electrons² (shifted-Poisson and Poisson–Gaussian models), $\sigma^2 / (g\,f)^2$ (pixel-wise, $f \equiv 1$ without a flat field) | `variance` of `ShiftedPoisson`, `PoissonGaussian` |
| $\kappa$ | constant of the default weight, $\nu = \kappa \sum\mathbf{h} / q_{99.9}(\tilde{y})$ for `Ikoma`, $\tau = \kappa / (q_{99.9}(\tilde{y}) - q_{50}(\tilde{y}))$ for `HSPIRAL`, and $\lambda = \kappa / (q_{99.9}(\tilde{y}) - q_{50}(\tilde{y}))$ with $\kappa_{\mathrm{TV}}$, $\kappa_{\mathrm{H}}$ for `Chouzenoux` (a scalar; unrelated to the eigenvalues $\kappa_j$ of the basis) | default of `nu`, `tau` |
| $\xi_a$ | frequency along axis $a$, $k_a / N_a$ | `rfft_frequencies` |
| $\tau$ | weight of the Schatten penalty, $\tau\sum_j\lVert\mathbf{X}_j\rVert_{S_p}$ (the role of $\nu$ for `HSPIRAL`) | `HSPIRAL.tau` |
| $\varphi_i$, $\bar\varphi_i$, $\xi_i$, $\eta_i$ | negative log-likelihood of pixel $i$ of the exact Poisson–Gaussian model, and its continuation for $u < 0$; its first and second derivative terms, $\varphi_i' = 1 - \xi_i$, $\varphi_i'' = \eta_i$ ($\varphi$ is also the function $\varphi(u)$ of the distributions section; $\eta$ also the natural parameters) | `PoissonGaussian`, `poisson_normal_ratio` |
| $h$, $\mu$ | smooth term of a primal–dual problem and the Lipschitz constant of its gradient (not the PSF $h$, nor the Gamma mean $\mu$) | `Smooth.lipschitz`, `Chouzenoux.lipschitz` |
| $\psi_0$, $\psi_r$, $\mathbf{V}_r$, $\mathbf{v}_r$ | constraint function, penalty $r$, its linear operator and its dual variable in `primal_dual` (not the digamma $\psi$) | `constraint`, `terms`, `PrimalDualResult.duals` |
| $\mathbf{y}_1$, $\mathbf{p}_1$, $\mathbf{q}_1$, $\mathbf{y}_2$, $\mathbf{p}_2$, $\mathbf{q}_2$ | intermediate points of one primal–dual iteration (not the data $\mathbf{y}$) | |
| $\gamma$, $\delta$, $\epsilon$ | step of the primal–dual iteration, $\delta = \mu + (\sum_r\lVert\mathbf{V}_r\rVert^2)^{1/2}$, and the margin in $\gamma = (1 - \epsilon)/\delta$ (unrelated to the tile gain $\gamma_i$ or the read noise $\varepsilon$) | `step`, `primal_dual_step(eps=)` |
| $\lambda_{\mathrm{H}}$ | weight of the Frobenius norm of the Hessian, next to $\lambda_{\mathrm{TV}}$ | `Chouzenoux.hessian` |
| $C$, $x_+$ | range of the image, $C = [0, x_+]^N$, and its upper bound | `upper` |
| $\mathbf{X}_j$ | symmetric Hessian matrix of the image at voxel $j$, packed in $(\mathbf{H}\mathbf{x})_j$ | |
| $\lVert\cdot\rVert_{S_p}$ | Schatten norm of order $p \in \{1, 2, \infty\}$: the $\ell_p$ norm of the singular values; nuclear, Frobenius and spectral norm | `schatten_norm`, `Schatten` |
| $\mu_\pm$, $m$, $r$ | eigenvalues of a symmetric $2 \times 2$ matrix, $\mu_\pm = m \pm r$, with $m$ the mean of the diagonal and $r \ge 0$ half the gap | `prox_schatten` |

The shifted Poisson law of $\tilde{\mathbf{y}}$ is $\tilde{y} + \tilde\sigma^2 \sim
\mathrm{Poisson}(\lambda + \tilde\sigma^2)$, which has the mean and variance of the
Poisson-Gaussian model. The letters $\rho$ and $\nu$ have other meanings in the
sections on morphology and distributions (the radius of a ball, the shape of a
Gamma); the context always tells which is meant. $\mu_\pm$ are eigenvalues of a
Hessian matrix, not a mean; $m$ and $r$ there are the centre and half-gap of the
eigenvalues, not the mean of a Normal; and $\tau$, here the weight of a penalty, is also
the truncation level of Nagy and O'Leary and the precision of a Normal. $\mathbf{H}$ in bold is the
Hessian operator, $H$ in plain type a count of rows.

## Indices and layouts

| Symbol | Ranges over |
|---|---|
| $i$ | pixels, $i \in \Omega$, $\lvert\Omega\rvert$ of them |
| $n$ | frames (samples of a stack), $1 \ldots N$; more generally the observations fed to a distribution |
| $k$ | classes, $1 \ldots K$ |
| $j$ | basis functions |
| $t$ | iterations; $L_t$ is the objective at iteration $t$ (`History.objective`) |
| $G_1, \ldots$; $X_1, \ldots$ | sizes of the group and spatial axes of an array |

In prose, *sample* means one picture of a stack, as in the
[layout](axes-and-layouts.md) roles, and *observation* means one value fed to
a distribution. Shapes are written in code font, as in `(*group, *spatial)`.

## Native notations

Every algorithm page writes its model twice, in tabs: in optix notation, and
in the notation of its paper, followed by a table that maps one onto the
other. For BaSiC, for instance:

=== "optix"

    $$
    \mathbf{y}_n = \mathbf{f} \odot (\mathbf{x}_n + b_n) + \mathbf{d}
    $$

=== "Peng et al. 2017"

    $$
    I_n^{\text{meas}}(x) = S^F(x)\,\bigl(B_n + I_n^R(x)\bigr) + S^D(x)
    $$

| optix | [Peng et al. 2017](https://doi.org/10.1038/ncomms14836) | CIDRE ([Smith et al. 2015](https://doi.org/10.1038/nmeth.3323)) |
|---|---|---|
| $\mathbf{y}_n$ | $I_n^{\text{meas}}$ | $I_o$ |
| $\mathbf{x}_n$ | $I_n^R$ | $I$ |
| $\mathbf{f}$ | $S^F$ | $v$ |
| $\mathbf{d}$ | $S^D$ | $z$ |
| $b_n$ | $B_n$ | — |

!!! references "References"

    1. Bishop CM. *Pattern Recognition and Machine Learning.* Springer, 2006.
       [Microsoft Research](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
    2. Peng T, Thorn K, Schroeder T, Wang L, Theis FJ, Marr C, Navab N.
       *A BaSiC tool for background and shading correction of optical
       microscopy images.* Nat Commun **8**:14836, 2017.
       [doi:10.1038/ncomms14836](https://doi.org/10.1038/ncomms14836)
    3. Smith K, Li Y, Piccinini F, Csucs G, Balazs C, Bevilacqua A, Horvath P.
       *CIDRE: an illumination-correction method for optical microscopy.*
       Nat Methods **12**(5):404-406, 2015.
       [doi:10.1038/nmeth.3323](https://doi.org/10.1038/nmeth.3323)

    ??? quote "BibTeX"

        ```bibtex
        @book{bishop2006pattern,
          author    = {Bishop, Christopher M.},
          title     = {Pattern Recognition and Machine Learning},
          publisher = {Springer},
          address   = {New York},
          year      = {2006},
          isbn      = {978-0-387-31073-2},
          url       = {https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/},
        }
        @article{peng2017basic,
          author  = {Peng, Tingying and Thorn, Kurt and Schroeder, Timm and Wang, Lichao and Theis, Fabian J. and Marr, Carsten and Navab, Nassir},
          title   = {A {BaSiC} tool for background and shading correction of optical microscopy images},
          journal = {Nature Communications},
          volume  = {8},
          pages   = {14836},
          year    = {2017},
          doi     = {10.1038/ncomms14836},
        }
        @article{smith2015cidre,
          author  = {Smith, Kevin and Li, Yunpeng and Piccinini, Filippo and Csucs, Gabor and Balazs, Csaba and Bevilacqua, Alessandro and Horvath, Peter},
          title   = {{CIDRE}: an illumination-correction method for optical microscopy},
          journal = {Nature Methods},
          volume  = {12},
          number  = {5},
          pages   = {404--406},
          year    = {2015},
          doi     = {10.1038/nmeth.3323},
        }
        ```
