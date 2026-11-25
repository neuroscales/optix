# NCS

!!! info "At a glance"
    - **Estimates:** noise-corrected image, same shape as the data, in electrons (or in ADU with `units="adu"`)
    - **Noise model:** Poisson–Gaussian with one offset, gain and read-noise variance per pixel (sCMOS), as a shifted Poisson
    - **Streaming:** halo
    - **Needs many frames:** no
    - **Reference:** Liu et al., Nat Methods, 2017, [doi:10.1038/nmeth.4379](https://doi.org/10.1038/nmeth.4379)

## Overview

`NCS` removes the pixel-dependent noise of an sCMOS camera from a single image
of a microscope ([Liu et al. 2017](https://doi.org/10.1038/nmeth.4379)). Each pixel
of an sCMOS sensor has its own offset, gain and read-noise variance, and the
variance spans two or three orders of magnitude across the chip, so a few
pixels flicker far more than their neighbours even when they see the same
light. The correction uses what is known about the camera, the three per-pixel
maps of a [`SensorEstimate`](../../reference/estimates.md), and about the
optics: a diffraction-limited image has no content beyond the cut-off frequency
$2\,\mathrm{NA}/\lambda_0$ of the optical transfer function (OTF), while the noise
is white. The estimate is the image that agrees with the data under the
camera's noise model and has little energy where only noise can live.

Use it on widefield or similar images recorded at low photon counts, where the
read noise of the noisiest pixels is comparable to the signal, when the maps of
the camera are known (they come from the manufacturer, or from
[sCMOS calibration](../sensor/scmos-calibration.md)) and the numerical aperture,
the wavelength and the pixel size are known. It needs no other frames: every
image is corrected alone. It is not a general denoiser: the penalty only helps
if the image is sampled finely enough for part of its spectrum to lie beyond the
OTF (the pixel must be smaller than about $0.35\,\lambda_0/\mathrm{NA}$ for any
frequency of the image to be there, and the paper recommends a sampling rate,
$0.61\,\lambda_0/\mathrm{NA}$ over the pixel size, above 2; Supp. Note 11), and it
does not deconvolve: the blur is kept. For
Gaussian noise of one level see [BM3D and BM4D](bm3d.md) and
[Non-local means](nl-means.md), and for deconvolution with the same shifted-Poisson
likelihood see [Ikoma](../deconv/ikoma.md) and [Snyder](../deconv/snyder.md).

The estimator is [`NCS`][optix.denoise.NCS], with the function
[`ncs`][optix.denoise.ncs] for in-memory images (see the
[API reference](../../reference/denoise.md)). `fit(init=sensor)` sets the camera
maps, `apply` corrects, and large arrays are streamed in blocks with a halo,
giving the result of the whole-image computation.

## Image formation (optics)

The camera follows the per-pixel model of
[Image formation and noise](../../concepts/image-formation.md): the recorded value
of pixel $i$ is its offset, plus its gain times a Poisson count of
photoelectrons, plus Gaussian read noise of its own variance. The photoelectron
rate $x_i$ is the image formed by the microscope. For an ideal system with
numerical aperture $\mathrm{NA}$ at wavelength $\lambda_0$ the pupil has radius
$\mathrm{NA}/\lambda_0$ and the OTF, its autocorrelation, vanishes beyond
$2\,\mathrm{NA}/\lambda_0$ ([Supp. Note 4](https://doi.org/10.1038/nmeth.4379)), so the
spectrum of $\mathbf{x}$ is confined to the disc $k_r \le 2\,\mathrm{NA}/\lambda_0$.
The Poisson and read noise have no such limit.

=== "optix"

    $$
    y_i = g_i\,e_i + o_i + \varepsilon_i,
    \quad e_i \sim \mathrm{Poisson}(x_i),
    \quad \varepsilon_i \sim \mathcal{N}(0, \sigma_i^2),
    \qquad
    (\mathbf{F}\mathbf{x})(\mathbf{k}) = 0 \ \text{ for } \ k_r > \frac{2\,\mathrm{NA}}{\lambda_0}
    $$

=== "Liu et al. 2017"

    $$
    S_{\mathrm{sCMOS}} = g\,S_{\mathrm{Poisson}} + G(0, \sigma_r^2) + o,
    \qquad
    \hat\mu(k_x, k_y) = 0 \ \text{ for } \ k_r > \frac{2\,NA}{\lambda}
    $$

The data are first brought to electrons, which is the pre-correction of the
paper, and the read noise is expressed in electrons too:

=== "optix"

    $$
    \tilde{y}_i = \frac{y_i - o_i}{g_i},
    \qquad
    \tilde{y}_i \leftarrow 10^{-6} \ \text{ where } \ \tilde{y}_i \le 0,
    \qquad
    \tilde\sigma_i^2 = \frac{\sigma_i^2}{g_i^2}
    $$

=== "Liu et al. 2017"

    $$
    D_i = \frac{A_i - o_i}{g_i},
    \qquad
    D_i \leftarrow 10^{-6} \ \text{ where } \ D_i \le 0,
    \qquad
    \gamma_i = \frac{\sigma_i^2}{g_i^2}
    $$

([Supp. Notes 2, 3 and 6, Eq. 1](https://doi.org/10.1038/nmeth.4379).) The
pre-corrected pixel $\tilde{y}_i$ has mean $x_i$ and variance
$x_i + \tilde\sigma_i^2$. The floor $10^{-6}$ keeps $\tilde{y}_i > 0$, which the
likelihood below needs; it is part of the method, and it acts on every pixel
whose offset-corrected value is zero or negative, which in the dark parts of a
low-light image are many.

| optix | Liu et al. 2017 | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}$ | $A$ ($S_{\mathrm{sCMOS}}$ for simulated data) | raw image, in ADU | $\mathbb{R}^{\lvert\Omega\rvert}$, per image |
| $\mathbf{o}$, $\mathbf{g}$, $\boldsymbol{\sigma}^2$ | $o$, $g$, $\sigma^2$ ($\sigma_r^2$ in Note 14) | offset (ADU), gain (ADU per electron), read-noise variance (ADU²) | $\mathbb{R}^{\lvert\Omega\rvert}$ each; one value per pixel (maps), shared by all images of a group |
| $\mathbf{e}$ | $S_{\mathrm{Poisson}}$ | photoelectrons collected, Poisson | $\mathbb{R}^{\lvert\Omega\rvert}$, per image |
| $\mathbf{x}$ | $\mu$ | image in electrons that the camera sees; the variable of the estimation | $\mathbb{R}^{\lvert\Omega\rvert}$, per image |
| $\tilde{\mathbf{y}}$ | $D$ | pre-corrected image, in electrons | $\mathbb{R}^{\lvert\Omega\rvert}$, per image |
| $\tilde{\boldsymbol\sigma}^2$ | $\gamma$ | read-noise variance in electrons² | $\mathbb{R}^{\lvert\Omega\rvert}$, shared by all images of a group |
| $\mathrm{NA}$, $\lambda_0$ | $NA$, $\lambda$ | numerical aperture; detection wavelength | scalars, one per call |
| $d$ | $\Delta x = \Delta y$ | pixel size in the object plane, in the unit of $\lambda_0$ | scalar, one per call |
| $\mathbf{F}$ | the hat, $\hat\mu = \mathcal{F}\mu$ | unitary 2-D discrete Fourier transform of a segment | $\mathbb{C}^{L \times L}$ for a segment of $L \times L$ pixels |
| $\mathbf{k} = (k_x, k_y)$, $k_r$ | $(k_x, k_y)$, $k_r$ | spatial frequency in cycles per unit length, $k_x = p/(L\,d)$, $p \in [-L/2, L/2 - 1]$, and its modulus | one value per pixel of the spectrum |

The image $\mathbf{x}$ is not the specimen: it is the specimen as seen through
the microscope, in electrons per exposure, and it is what the corrected image
estimates. The symbol $\mathbf{x}$ is used for it as in
[Notation](../../concepts/notation.md#image-formation); the paper's $\mu$ is its
"effective photon count".

## Probabilistic model

The assumptions are the camera model above (independent pixels, Gaussian read
noise of known variance, known gain and offset), the band limit of $\mathbf{x}$,
and that the images are corrected one at a time (Supp. Note 8: the likelihood
assumes no correlation of the noise between frames).

**Shifted-Poisson likelihood.** The exact distribution of a pre-corrected pixel
is a Poisson law convolved with a Gaussian, an infinite sum. Following
[Huang et al. (2013)](https://doi.org/10.1038/nmeth.2488), the paper replaces it
by a *shifted Poisson* law ([Snyder et al. 1993](https://doi.org/10.1364/JOSAA.10.001014)):
$\tilde{y}_i + \tilde\sigma_i^2$ is taken as Poisson with mean $x_i + \tilde\sigma_i^2$,
which has the mean $x_i$ and the variance $x_i + \tilde\sigma_i^2$ of the
Poisson–Gaussian model. Its likelihood and negative logarithm, with Stirling's
formula for the gamma function, are

=== "optix"

    $$
    p(\tilde{\mathbf{y}} \mid \mathbf{x}) \approx
    \prod_{i} \frac{e^{-(x_i + \tilde\sigma_i^2)}\,(x_i + \tilde\sigma_i^2)^{\tilde{y}_i + \tilde\sigma_i^2}}
    {\Gamma(\tilde{y}_i + \tilde\sigma_i^2 + 1)},
    \qquad
    \mathrm{LL}_{\mathrm{s}}(\mathbf{x}) = \sum_i x_i - (\tilde{y}_i + \tilde\sigma_i^2)\ln(x_i + \tilde\sigma_i^2)
    $$

=== "Liu et al. 2017"

    $$
    L(\mu, \gamma \mid D, \gamma) = \prod_{i=1}^{N}
    \frac{e^{-(\mu_i + \gamma_i)}\,(\mu_i + \gamma_i)^{(D_i + \gamma_i)}}{\Gamma(D_i + \gamma_i + 1)},
    \qquad
    LL_S = \sum_{i=1}^{N} \bigl[\mu_i - (D_i + \gamma_i)\ln(\mu_i + \gamma_i)\bigr]
    $$

([Supp. Note 6, Eqs. 5 and 7](https://doi.org/10.1038/nmeth.4379).) Between them
is Eq. 6, $-\ln L \simeq \sum_i [\mu_i - D_i - (D_i + \gamma_i)\ln(\mu_i + \gamma_i)
+ (D_i + \gamma_i)\ln(D_i + \gamma_i)]$, from Stirling's approximation
$\ln\Gamma(a + 1) \approx a\ln a - a$; the terms that do not depend on $\mu$ are
dropped to give the simplified negative log-likelihood $\mathrm{LL}_{\mathrm{s}}$ of
Eq. 7. It is the data term of
[`ShiftedPoisson`](../../concepts/optimisation.md) with data $\tilde{\mathbf{y}}$ and
variance $\tilde{\boldsymbol\sigma}^2$, the same as in [Ikoma](../deconv/ikoma.md).
Its minimum is at $\mathbf{x} = \tilde{\mathbf{y}}$, which is why the weight $\alpha = 0$
returns the pre-corrected image (Fig. SS6).

**Noise contribution outside the OTF.** The energy of $\mathbf{x}$ at the
frequencies where the OTF leaves no signal is the *noise contribution*. It is
measured through a mask $\mathbf{m}$ of the spectrum, between 0 (signal can be
there: not penalised) and 1 (only noise can be there):

=== "optix"

    $$
    E_{\mathrm{N}}(\mathbf{x}) = \sum_{\mathbf{k}} \bigl\lvert m(\mathbf{k})\,(\mathbf{F}\mathbf{x})(\mathbf{k}) \bigr\rvert^2
    = \mathbf{x}^\top \mathbf{Q}\,\mathbf{x},
    \qquad
    \mathbf{Q} = \mathbf{F}^{\dagger} \operatorname{diag}(\mathbf{m}^{\odot 2})\,\mathbf{F}
    $$

=== "Liu et al. 2017"

    $$
    \sigma_N = \sum_{k_x, k_y} \bigl\lvert \hat\mu(k_x, k_y)\, M(k_x, k_y) \bigr\rvert^2,
    \qquad
    \hat\mu(k_x, k_y) = \frac{1}{L}\sum_{m, n = -L/2}^{L/2 - 1} \mu(x, y)\, e^{-i 2\pi k_x x}\, e^{-i 2\pi k_y y}
    $$

([Supp. Note 4, Eqs. 2 and 3](https://doi.org/10.1038/nmeth.4379).) With the
factor $1/L$ the transform is unitary, so by Parseval's theorem $E_{\mathrm{N}}$ is
the squared norm of the image filtered by $\mathbf{m}$. As the mask is real and
even in $\mathbf{k}$, $\mathbf{Q}$ is a real symmetric, positive semi-definite
operator, and $E_{\mathrm{N}}$ is a convex quadratic form.

**The masks.** The mask is a high-pass raised-cosine filter of the radial
frequency $k_r = \sqrt{k_x^2 + k_y^2}$, with a roll-off $\beta \in [0, 1]$ and a
period $T$:

=== "optix"

    $$
    m(k_r) =
    \begin{cases}
    0 & k_r < k_{\mathrm{lo}} \\[0.5ex]
    \dfrac{1}{2} - \dfrac{1}{2}\cos\!\Bigl[\dfrac{\pi T}{\beta}\,(k_r - k_{\mathrm{lo}})\Bigr] & k_{\mathrm{lo}} \le k_r \le k_{\mathrm{hi}} \\[1.5ex]
    1 & k_r > k_{\mathrm{hi}}
    \end{cases}
    \qquad
    k_{\mathrm{lo}} = \frac{1 - \beta}{2T},
    \quad
    k_{\mathrm{hi}} = \frac{1 + \beta}{2T}
    $$

=== "Liu et al. 2017"

    $$
    M(k_x, k_y) =
    \begin{cases}
    1 & k_r > \dfrac{1 + \beta}{2T} \\[1.5ex]
    0 & k_r < \dfrac{1 - \beta}{2T} \\[1ex]
    \dfrac{1}{2} + \dfrac{1}{2}\cos\!\Bigl[\dfrac{\pi T}{\beta}\Bigl(k_r - \dfrac{1 - \beta}{2T}\Bigr)\Bigr] & \text{otherwise}
    \end{cases}
    $$

([Supp. Note 5, Eq. 4](https://doi.org/10.1038/nmeth.4379).) The sign of the cosine
differs between the tabs: the paper prints a plus, optix uses a minus (see
[choice 3](#choices-not-in-the-paper) for the reason). The paper uses two masks,
which set the parameters from the cut-off $2\,\mathrm{NA}/\lambda_0$:

| Mask (`mask=`) | $\beta$ | $T$ | $m = 0$ below | $m = 1$ above | Penalises |
|---|---|---|---|---|---|
| OTF-weighted (`"otf-weighted"`, default) | 1 | $\lambda_0 / (5.6\,\mathrm{NA})$ | 0 | $5.6\,\mathrm{NA}/\lambda_0$ | increasingly from the centre of the spectrum, the signal band lightly ($m \approx 0.28$ at $2\,\mathrm{NA}/\lambda_0$) |
| noise-only (`"noise-only"`) | 0.2 | $(1 - \beta)\,\lambda_0 / (4\,\mathrm{NA})$ | $2\,\mathrm{NA}/\lambda_0$ | $3\,\mathrm{NA}/\lambda_0$ | only the frequencies at and beyond the OTF boundary |

The OTF-weighted mask is the one that the paper recommends: it removes more
noise but gives up a little of the signal band (a loss of resolution of a few
nanometres in the paper's simulations), while the noise-only mask leaves the
signal band untouched and corrects less (Supp. Notes 5 and 9). The mask is
evaluated on the frequency grid of the segment being solved, so that it follows
its size (the grid of `torch.fft.fftfreq`, zero frequency first).

**The cost.** The estimate minimises the sum of the data term and the weighted
noise contribution, per segment (below):

=== "optix"

    $$
    J(\mathbf{x}) = \mathrm{LL}_{\mathrm{s}}(\mathbf{x}) + \alpha\,E_{\mathrm{N}}(\mathbf{x}),
    \qquad
    \hat{\mathbf{x}} = \operatorname*{arg\,min}_{\mathbf{x}\,:\,x_i + \tilde\sigma_i^2 > 0} J(\mathbf{x}),
    \qquad
    \alpha = 0.2
    $$

=== "Liu et al. 2017"

    $$
    f = LL_S + \alpha\,\sigma_N,
    \qquad
    \alpha = 0.2
    $$

([Supp. Note 2, step 3](https://doi.org/10.1038/nmeth.4379); $\alpha$ is "an empirical weight
factor", set to 0.2 from Fig. SS6, where a larger value removes more noise at the
price of resolution.) In Bayesian terms $J$ is a penalised likelihood: the
posterior of $\mathbf{x}$ under an improper zero-mean Gaussian prior on the masked
spectrum, with precision $2\alpha\,\mathbf{m}^{\odot 2}$, which leaves the
frequencies where $m = 0$ free. This reading is not the paper's, which only calls
$\alpha$ empirical, and nothing below depends on it.

## Inference / derivation

The estimate is a *penalised maximum likelihood* (or *maximum a posteriori*, in the
reading above) estimate under the shifted-Poisson approximation. The variables
are the $\lvert\Omega\rvert$ pixels of the image, a large problem; the paper
splits the image into small independent *segments*, each estimated on its own
(Supp. Note 7), and the same is done here.

**Segmentation.** An image of $n$ pixels along an axis is cut into segments of
$M$ pixels from the first pixel (Supp. Note 2, step 2; the recommended $M$ is 8
to 32). Each segment of $M \times M$ pixels is padded by one pixel on every side
to $(M + 2) \times (M + 2)$ before it is estimated: a pad pixel takes the value
of the adjacent pixel of the image, and a pad pixel beyond the border of the image takes
the value of the border pixel itself. The pad is there to avoid the edge effect of the discrete Fourier
transform, which treats the segment as periodic: without it the abrupt
change between the two edges would put spurious energy at all frequencies, which
$E_{\mathrm{N}}$ would try to remove. With the pad, the wrap-around joins pad pixels, which are
discarded after the estimate; only the $M \times M$ interior is kept (the edge
effect is then removed everywhere except at the border of the image, Supp. Note 7).
The pad pixels are variables of the problem like the others. In
equations, with $s$ a segment and $\mathbf{C}_s$ the operator that cuts and pads it,

$$
\hat{\mathbf{x}}_s = \operatorname*{arg\,min}_{\mathbf{x}} \; J_s(\mathbf{x}),
\qquad
J_s(\mathbf{x}) = \sum_{i}\Bigl[x_i - (\tilde{y}_{s,i} + \tilde\sigma_{s,i}^2)\ln(x_i + \tilde\sigma_{s,i}^2)\Bigr] + \alpha\,E_{\mathrm{N}}(\mathbf{x}),
\qquad
\tilde{\mathbf{y}}_s = \mathbf{C}_s\tilde{\mathbf{y}},
$$

with the mask $\mathbf{m}$ of the padded size $(M + 2) \times (M + 2)$. The
segments are independent of each other, and the result in a segment depends only on the pixels of
the segment and of its pad (the consequence for streaming is below).

**Minimisation.** The paper leaves the optimiser open ("a generic minimization
routine such as `fminunc`", started from $\mu^0 = D$, Supp. Note 2). The cost has
a unique minimiser: $\mathrm{LL}_{\mathrm{s}}$ is a sum of strictly convex terms on
$x_i + \tilde\sigma_i^2 > 0$ (its second derivative is
$(\tilde{y}_i + \tilde\sigma_i^2)/(x_i + \tilde\sigma_i^2)^2 > 0$, positive because
$\tilde{y}_i \ge 10^{-6}$), and $E_{\mathrm{N}}$ is a positive semi-definite quadratic
form. Its gradient and Hessian are

$$
\nabla J = \mathbf{1} - \frac{\tilde{\mathbf{y}} + \tilde{\boldsymbol\sigma}^2}{\mathbf{x} + \tilde{\boldsymbol\sigma}^2} + 2\alpha\,\mathbf{Q}\mathbf{x},
\qquad
\nabla^2 J = \operatorname{diag}\!\left(\frac{\tilde{\mathbf{y}} + \tilde{\boldsymbol\sigma}^2}{(\mathbf{x} + \tilde{\boldsymbol\sigma}^2)^{\odot 2}}\right) + 2\alpha\,\mathbf{Q}.
$$

The problem is solved by a damped Newton method (a *truncated Newton* or
Newton–CG method): each Newton system $\nabla^2 J\,\mathbf{p} = -\nabla J$ is solved
by conjugate gradients, which only needs products with $\mathbf{Q}$, that is two
FFTs, and a line search shortens the step to keep the iterate in the domain and to
decrease the cost. A strictly convex problem solved to convergence gives the same
answer whatever the algorithm, so the choice of the optimiser changes the cost of
the computation, not its result (the solution agrees with an independent
ADMM solver of [`optix.optim`](../../reference/optim.md), see
[below](#results-on-synthetic-data)).

## Algorithm

**Input:** raw images $\mathbf{y}$ in ADU; maps $\mathbf{o}$, $\boldsymbol\sigma^2$, $\mathbf{g}$
(a `SensorEstimate`); $\mathrm{NA}$, $\lambda_0$, $d$; `alpha` $= 0.2$,
`mask = "otf-weighted"`, segment size $M = 16$ (`segment`), `units = "electrons"`,
`max_iter` $= 50$, `tol` (default $10^{-10}$ in float64, $1.2 \cdot 10^{-5}$ in float32) \
**Output:** the corrected images $\hat{\mathbf{x}}$, of the shape of $\mathbf{y}$, in electrons
(in ADU: $\mathbf{g} \odot \hat{\mathbf{x}} + \mathbf{o}$).

1. **Pre-correct** every image: $\tilde{\mathbf{y}} = (\mathbf{y} - \mathbf{o})/\mathbf{g}$, with
   the floor $10^{-6}$ where it is not positive, and
   $\tilde{\boldsymbol\sigma}^2 = \boldsymbol\sigma^2/\mathbf{g}^2$.
2. **Segment** each axis of the image in segments of $M$ pixels (the last segment of an axis takes
   the pixels that remain when $M$ does not divide the length, see
   [choice 6](#choices-not-in-the-paper)), and pad every segment by one pixel as above.
3. **Estimate** every padded segment, independently of the others and in batches
   of segments of the same size:
    1. build the mask $\mathbf{m}$ of the padded size (once per size);
    2. start at $\mathbf{x}^{(0)} = \tilde{\mathbf{y}}_s$ (the paper's initial value);
    3. **repeat** until $\max\lvert\nabla J\rvert \le$ `tol`, the line search fails
       (round-off), or `max_iter` iterations: solve $\nabla^2 J\,\mathbf{p} = -\nabla J$
       by preconditioned conjugate gradients, then set
       $\mathbf{x} \leftarrow \mathbf{x} + t\,\mathbf{p}$ with the largest step
       $t \in \{1, \tfrac12, \tfrac14, \ldots\}$ that keeps $x_i + \tilde\sigma_i^2$
       positive and decreases $J$;
    4. drop the pad.
4. **Assemble** the interiors of the segments into the image.
5. **Convert** to ADU if `units="adu"`: $\hat{\mathbf{x}} \leftarrow \mathbf{g} \odot \hat{\mathbf{x}} + \mathbf{o}$.

Steps 1 and 3 are those of Supp. Note 2 (Fig. SS1), where step 3 reads "obtain an
estimate by minimizing a cost function". The details of step 3.3 are given in
the next section, as they are not in the paper.

## Faithful mode and extensions

`faithful=True` (the default) reproduces the cost of the paper: the
pre-correction with its floor, the shifted-Poisson likelihood, one of the two
masks, $\alpha = 0.2$, the segmentation with the padding, and the initial value
$\mathbf{x}^{(0)} = \tilde{\mathbf{y}}$, with a segment size in the range that the paper
recommends. The paper leaves a number of choices open, and the code makes them as
follows.

### Choices not in the paper

1. **The optimiser.** The paper uses "a generic minimization routine such as
   `fminunc`" (MATLAB) and runs it for a fixed number of iterations: its software
   defaults to 15, and the Supplementary Notes say that the regression converges
   within 20 (Supp. Notes 2 and 13). The cost is strictly convex, so it has one
   minimiser, which a fixed small number of iterations only approaches. optix runs
   a damped Newton method with preconditioned conjugate gradients until the
   gradient is small: $\max\lvert\nabla J\rvert \le$ `tol` (default
   $\max(100\,\epsilon, 10^{-10})$ for the machine epsilon $\epsilon$ of the dtype,
   so $10^{-10}$ in float64 and $1.2 \cdot 10^{-5}$ in float32; the gradient is
   dimensionless), at most `max_iter` $= 50$ Newton iterations. The
   conjugate-gradient solve is preconditioned by the diagonal of the Hessian
   (the diagonal of $\mathbf{Q}$ is the mean of $\mathbf{m}^{\odot 2}$) and stops at a
   residual of $\min(\tfrac12, \lVert\nabla J\rVert^{1/2})\,\lVert\nabla J\rVert$;
   the line search starts at $t = 1$, is capped to 99 % of the distance to the
   boundary $x_i = -\tilde\sigma_i^2$ of the domain, and halves $t$ until the
   Armijo condition holds with a constant of $10^{-4}$. A segment stops when
   the gradient is small, when the line search fails, or after `max_iter`
   iterations. The result is therefore the minimiser of the paper's cost, not the
   iterate of the authors' software after 15 iterations, whose distance from the
   minimiser depends on the optimiser's settings that the paper does not give. The
   gradient and the stopping rule are in
   [`ncs_solve`][optix.denoise.utils.ncs_solve].
2. **The segment size.** The paper recommends $M$ between 8 and 32 and does not
   give a default. The default here is $M = 16$ (`segment`).
3. **The sign of the cosine in Eq. 4.** The paper prints $\tfrac12 + \tfrac12\cos[\cdots]$
   in the transition band. With that sign the mask is 0 below the band, jumps to 1
   at its lower edge $k_{\mathrm{lo}}$, decreases through the band to 0 at its upper
   edge $k_{\mathrm{hi}}$, and jumps back to 1 above it. That is not the
   high-pass raised-cosine filter that the text describes, which is 0 inside and 1
   outside. The code uses $\tfrac12 - \tfrac12\cos[\cdots]$, which is continuous at both
   ends and increasing. The evidence is the paper's own figures:
    - Fig. SS4B plots the radial profile of the masks: the default (OTF-weighted)
      curve rises from 0 at the centre to about 0.28 at the OTF boundary $2\,\mathrm{NA}/\lambda_0$
      and about 0.7 at $3.5\,\mathrm{NA}/\lambda_0$, and the noise-only curve is 0 up to
      $2\,\mathrm{NA}/\lambda_0$ and 1 from $3\,\mathrm{NA}/\lambda_0$. The minus sign gives 0.283 and
      0.691 for the first mask, the plus sign 0.717 and 0.309, a mask that
      weighs the signal band more than the noise band (see the example in
      [Usage](#usage));
    - Supp. Note 11 gives the height of the OTF-weighted mask as about 0.2 at a
      sampling rate of 1.4 (the Rayleigh criterion $0.61\,\lambda_0/\mathrm{NA}$ over the
      pixel size); the minus sign gives 0.19 at the corner of the spectrum, the
      largest frequency of the image.
4. **ADU units.** The paper's output is the "effective photon count", in
   electrons. `units="adu"` returns $\mathbf{g} \odot \hat{\mathbf{x}} + \mathbf{o}$ instead,
   with the maps of the camera, so that the corrected image can be compared with
   the raw one. It only converts the result; the estimation is the same.
5. **The faithful flag.** `is_faithful` is true when $\alpha = 0.2$ exactly, `mask`
   is `"otf-weighted"` or `"noise-only"`, and $8 \le M \le 32$. It is computed
   from the values of the options (the defaults are all the paper's, so
   `faithful=False` changes none of them). Any other $\alpha$, a callable
   mask or a segment size outside 8 to 32 makes the estimator non-faithful. The
   options `units`, `max_iter`, `tol` and `block` do not affect the flag: they do
   not change the cost.
6. **Segments that do not tile the image.** When $M$ does not divide the length $n$
   of an axis, the last segment of the axis also takes the remaining
   $n \bmod M$ pixels, so every segment has between $M$ and $2M - 1$ pixels (the
   whole axis when $n < M$). The paper does not say what to do with the remainder.
   Segments need not be square or of even length: the frequencies of each axis are
   those of its own length (the paper's Eq. 2 assumes an even $L$).
7. **The domain.** The paper constrains $D > 0$ so that $\mu \ge 0$ (Supp. Note 6),
   but the cost only requires $\mu_i + \gamma_i > 0$, and that is the constraint
   the solver keeps. An estimate may therefore be slightly negative, down to
   $-\gamma_i$, in a pixel of large read noise.

### Extensions (opt-in, not faithful)

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `alpha` | weight $\alpha$ of the noise contribution; larger values correct more and cost resolution (Fig. SS6) | 0.2 | 0.2 |
| `mask` (callable) | a user-defined mask: a function of the radial frequency $k_r$ (a tensor of the shape of the spectrum, in cycles per unit of `wavelength`) that returns $\mathbf{m}$ | `"otf-weighted"` or `"noise-only"` | `"otf-weighted"` |
| `segment` outside 8 to 32 | segment size $M$ | 16 (any $M$ from 8 to 32 is faithful) | 16 |

The paper's software also offers a *user-defined OTF mask* (Supp. Note 9), for
aberrated systems whose effective cut-off is below $2\,\mathrm{NA}/\lambda_0$; the callable is
the optix counterpart of it, for radially symmetric masks. A function of $k_r$ is even in $\mathbf{k}$,
as the mask must be. A mask that is zero everywhere makes the noise contribution
vanish, and the estimate is then the pre-corrected image. The other keywords do
not change the method:

| Keyword | Changes | Faithful value |
|---|---|---|
| `numerical_aperture`, `wavelength`, `pixel_size` | the optics, in a common unit of length; required | none |
| `units` | `"electrons"` or `"adu"` (choice 4) | `"electrons"` |
| `max_iter`, `tol` | stopping rule of the solver (choice 1) | 50, `None` (dtype dependent) |
| `block` | core block size per spatial axis of the streaming; the result does not depend on it | `None` (memory budget) |

### Not implemented

- The *adjustable* mask of Supp. Note 9, given by a width $w$ and a height $h$, for
  which the paper gives no formula. The callable above can express a mask of this
  kind.
- The multiple gain and variance maps selected by the ADU level of each pixel
  (Supp. Note 10). One set of maps per group is used.
- The estimation of the OTF boundary from a bead image (Supp. Note 9). The cut-off
  is $2\,\mathrm{NA}/\lambda_0$, or whatever the callable mask encodes.
- A cost across frames for correlated readout noise (Supp. Note 8). The images are
  corrected one by one.
- Volumes: the images have two spatial axes.

## Memory and streaming

The tag is `halo`. A segment, with its pad, is estimated from the pixels it
covers and nothing else, and an axis has segments of up to $2M - 1$ pixels (the
last one takes the remainder), so the result at a pixel depends on the data within
$2M - 1$ pixels of it: a segment that meets a block has at most $2M - 1$ pixels
and a pad of one, so it reaches $2M - 1$ pixels beyond the block.
[`ncs_halo`][optix.denoise.ncs_halo]`(segment)` and `NCS.halo()` give this halo,
31 for $M = 16$.

`apply` reads the data with the [`blocks`](../../reference/io.md) iterator: a core
of `block` pixels per spatial axis, sized by default so that the block with its
halo, times a working factor of about 24, fits the memory budget of
[`optix.config`](../../reference/configuration.md), with this halo around it. The
segments are those of the whole image (the grid starts at the first pixel of the
image, not of the block), the segments that meet the core of a block are
estimated, and the core is written to the output. The result therefore equals the
whole-image computation, up to floating-point rounding, for any `block`, including
blocks that do not align with the segments. The data may be a zarr array or a
memory-mapped file larger than memory; the output cannot be the input (an
in-place call is refused, as a halo needs the unchanged neighbours). `fit` does not
read the data. The camera maps may be scalars, an $(H, W)$ map, or maps of shape
`(*group, H, W)` for a layout with groups (one camera per channel, for instance);
they are cropped to each block. Images of a stack are independent, and so are
groups.

The cost is linear in the number of pixels: every segment is a small problem, and
the segments of a block (of the same size) are solved together as a batch, so the
number of pixels in the batch, not the size of the image, sets the memory. Each
Newton iteration needs several conjugate-gradient steps of two FFTs of
$(M + 2)^2$ pixels each. A larger $M$ makes the mask finer and the padding
overhead $(M + 2)^2 / M^2$ smaller (1.27 for $M = 16$, 1.56 for $M = 8$, 1.13 for
$M = 32$), but each solve harder; the default is a compromise, and the paper does
not recommend segments below 8 pixels (Supp. Note 13). `max_iter` and `tol` only
matter for speed: on synthetic segments with $M = 8$, 16 and 32, in float32 and
float64, every segment converged to the default `tol` in 10 Newton iterations or
fewer. In float32 the
result agrees with float64 to about $10^{-4}$ in electrons.

## Usage

The first example builds a synthetic movie with known truth, in the manner of
Supp. Note 14: filaments seen through the OTF of an ideal circular pupil
(NA 1.4, wavelength 0.7 µm, pixel 0.1 µm), recorded by a camera with an offset, a
gain and a log-normally distributed read-noise variance per pixel, eight frames
of the same scene. `ncs` corrects them, and the error of the corrected frames is
compared with that of the pre-corrected frames in two bands of the spectrum.

```pycon
>>> import math
>>> import torch
>>> from optix.denoise import NCS, ncs, ncs_halo
>>> from optix.denoise import utils
>>> from optix.estimates import SensorEstimate
>>> from optix.models import phantoms, simulate_sensor
>>> optics = dict(numerical_aperture=1.4, wavelength=0.7, pixel_size=0.1)
>>> cutoff = 2 * 1.4 / 0.7  # OTF radius 2NA/λ, in cycles per µm
>>> k = torch.fft.fftfreq(64, d=0.1)
>>> kr = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
>>> rho = (kr / cutoff).clamp(max=1)
>>> otf = (2 / math.pi) * (torch.acos(rho) - rho * torch.sqrt(1 - rho**2))
>>> x = phantoms.filaments((64, 64), 6, width=1.0, intensity=100, seed=0)
>>> truth = torch.fft.ifft2(torch.fft.fft2(x) * otf).real + 10
>>> g = torch.Generator().manual_seed(1)
>>> sensor = SensorEstimate(
...     offset=100 + 5 * torch.randn(64, 64, generator=g),
...     variance=20 * torch.exp(1.5 * torch.randn(64, 64, generator=g)),
...     gain=2 + 0.2 * torch.randn(64, 64, generator=g),
... )
>>> raw = simulate_sensor(truth.expand(8, 64, 64), sensor, seed=2)
>>> pre_corrected = (raw - sensor.offset) / sensor.gain
>>> corrected = ncs(raw, sensor, **optics)
>>> def energy(image, band):
...     spectrum = torch.fft.fft2(image, norm="ortho")
...     return float((spectrum.abs().square() * band).sum((-2, -1)).mean())
>>> outside, inside = kr > cutoff, kr <= 0.8 * cutoff
>>> def ratios(image):
...     return [
...         round(
...             energy(image - truth, band) / energy(pre_corrected - truth, band),
...             1,
...         )
...         for band in (outside, inside)
...     ]
>>> ratios(corrected)  # error energy beyond the OTF, and well inside it
[0.1, 0.6]
>>> noise_only = ncs(raw, sensor, mask="noise-only", **optics)
>>> outside_ratio, inside_ratio = ratios(noise_only)
>>> outside_ratio, 0.7 < inside_ratio < 1
(0.3, True)
```

The error energy beyond $2\,\mathrm{NA}/\lambda_0$ falls to about a tenth with the
OTF-weighted mask and to about 0.3 with the noise-only mask, the error well
inside the OTF falls less (about 0.6, and 0.8 to 0.9), the temporal standard deviation of the
10 % noisiest pixels falls from about 10 to about 3 electrons, and the
mean of the corrected frames keeps the signal inside the OTF (a bias of less than
1 % of its energy). The estimator form streams the images in blocks with a halo
and gives the whole-image result to rounding. Here blocks of $23 \times 41$ pixels
cut the $64 \times 64$ images across the segments of 8 pixels, and the halo is
$2 \cdot 8 - 1 = 15$:

```pycon
>>> est = NCS(segment=8, block=(23, 41), **optics).fit(init=sensor)
>>> est.halo(), ncs_halo(8)
(15, 15)
>>> whole = ncs(raw, sensor, segment=8, **optics)
>>> torch.allclose(est.apply(raw), whole, rtol=1e-5, atol=1e-5)
True
```

The output is in electrons, or in ADU with `units="adu"`, which only applies the
gain and offset of each pixel to the result:

```pycon
>>> adu = NCS(units="adu", **optics).fit(init=sensor).apply(raw)
>>> torch.allclose(
...     adu, corrected * sensor.gain + sensor.offset, rtol=1e-5, atol=1e-3
... )
True
```

The sign of the cosine (choice 3) is visible on the mask itself. At $2$, $3.5$ and
$5.6\,\mathrm{NA}/\lambda_0$ the continuous high-pass form gives 0.283, 0.691 and 1,
as in Fig. SS4B, while the printed sign gives 0.717, 0.309 and 0. The height of
the mask at the corner of the spectrum, at a sampling rate of 1.4, is about 0.2
(Supp. Note 11):

```pycon
>>> na_l = 1.4 / 0.7
>>> k = torch.tensor([2 * na_l, 3.5 * na_l, 5.6 * na_l])
>>> utils.raised_cosine(k, beta=1.0, period=0.7 / (5.6 * 1.4))
tensor([0.2831, 0.6913, 1.0000])
>>> 0.5 + 0.5 * torch.cos(math.pi * 0.7 / (5.6 * 1.4) * k)
tensor([0.7169, 0.3087, 0.0000])
>>> pixel = 0.61 * 0.7 / (1.4 * 1.4)  # sampling rate 1.4
>>> mask = utils.ncs_mask((64, 64), **dict(optics, pixel_size=pixel))
>>> round(float(mask.max()), 1)
0.2
```

Segments follow `segment_bounds`, and a user-defined mask is an extension, so it
makes the estimator non-faithful:

```pycon
>>> utils.segment_bounds(40, 16)
[(0, 16), (16, 40)]
>>> hard = lambda kr: (kr > 0.8 * 2 * 1.4 / 0.7).to(kr.dtype)
>>> NCS(mask=hard, **optics).is_faithful, NCS(**optics).is_faithful
(False, True)
>>> NCS(alpha=0.5, **optics).is_faithful, NCS(segment=64, **optics).is_faithful
(False, False)
>>> bool(ncs(raw, sensor, mask=hard, **optics).isfinite().all())
True
```

Common variations: `fit(init=calibration)` with a fitted
[`SCMOSCalibration`](../sensor/scmos-calibration.md) in place of a `SensorEstimate`;
`layout=` to say which axes are spatial when the array has frames or channels, with maps of shape
`(*group, H, W)` for one camera per group; a large zarr array in place of the
tensor, with `out=` a separate array; `mask="noise-only"` to leave the signal
band untouched; `units="adu"`. A fitted estimator can be saved and loaded like
any other.

### Results on synthetic data

The synthetic movie above (filaments through the OTF of an ideal pupil, with a
log-normal read-noise variance map, eight frames) is used by the test suite to
check that the estimate behaves as the method intends. These numbers describe
that scene only; no real data and no published figures are used. No
comparison with the authors' implementation is given: its licence (GPL-3.0) is
not compatible with that of optix.

| Quantity | OTF-weighted mask | Noise-only mask |
|---|---|---|
| error energy beyond $2\,\mathrm{NA}/\lambda_0$, relative to the pre-corrected frames | about 0.1 | about 0.3 |
| error energy up to $0.8 \cdot 2\,\mathrm{NA}/\lambda_0$, relative to the pre-corrected frames | about 0.6 | 0.8 to 0.9 |
| squared error in the image, relative to the pre-corrected frames | about 0.3 | about 0.6 |
| temporal standard deviation of the 10 % noisiest pixels | from about 10 to about 3 electrons | from about 10 to about 5 electrons |
| bias of the mean over frames, in the band inside the OTF | below 1 % of the signal energy | below 1 % of the signal energy |

Further checks of the implementation, all on synthetic segments with known
ground truth:

- **Optimality.** At the result $\max\lvert\nabla J\rvert < 10^{-9}$ in float64 for
  $M = 8$, 16 and 32; random perturbations of the result raise the cost; the
  analytic gradient equals the gradient by automatic differentiation; the cost
  equals the direct sum $\mathrm{LL}_{\mathrm{s}} + \alpha \sum_{\mathbf{k}} \lvert m\,\mathbf{F}\mathbf{x}\rvert^2$
  of Eqs. 3 and 7.
- **Another solver.** The result agrees to $10^{-6}$ with the ADMM solver of
  `optix.optim` (`admm_fft`) applied to the same cost.
- **Limits.** With $\alpha = 0$ the result is the pre-corrected image, to the
  bit.
- **Segments.** A segment of the corrected image equals the solution of its padded
  segment alone, in the interior of the image (neighbours as pad) and in a corner
  (the border pixel as pad). The result of an image does not depend on the other
  images of a batch, to rounding.
- **Streaming.** Block-wise `apply` equals the whole-image `ncs` for blocks that do
  not align with the segments: exactly in float64, to a relative $4 \cdot 10^{-7}$
  in float32.
- **Precision.** float32 and float64 agree to $10^{-4}$ in electrons.

## References

!!! references "References"

    1. Liu S, Mlodzianoski MJ, Hu Z, Ren Y, McElmurry K, Suter DM, Huang F.
       *sCMOS noise-correction algorithm for microscopy images.* Nat Methods
       **14**(8):760-761, 2017.
       [doi:10.1038/nmeth.4379](https://doi.org/10.1038/nmeth.4379)
    2. Huang F, Hartwich TMP, Rivera-Molina FE, Lin Y, Duim WC, Long JJ,
       Uchil PD, Myers JR, Baird MA, Mothes W, Davidson MW, Toomre D,
       Bewersdorf J. *Video-rate nanoscopy using sCMOS camera-specific
       single-molecule localization algorithms.* Nat Methods
       **10**(7):653-658, 2013.
       [doi:10.1038/nmeth.2488](https://doi.org/10.1038/nmeth.2488)
    3. Snyder DL, Hammoud AM, White RL. *Image recovery from data acquired with a
       charge-coupled-device camera.* J Opt Soc Am A **10**(5):1014-1023, 1993.
       [doi:10.1364/JOSAA.10.001014](https://doi.org/10.1364/JOSAA.10.001014)

    ??? quote "BibTeX"

        ```bibtex
        @article{liu2017scmos,
          author  = {Liu, Sheng and Mlodzianoski, Michael J. and Hu, Zhenhua and Ren, Yuan and McElmurry, Kristi and Suter, Daniel M. and Huang, Fang},
          title   = {{sCMOS} noise-correction algorithm for microscopy images},
          journal = {Nature Methods},
          volume  = {14},
          number  = {8},
          pages   = {760--761},
          year    = {2017},
          doi     = {10.1038/nmeth.4379},
        }
        @article{huang2013video,
          author  = {Huang, Fang and Hartwich, Tobias M. P. and Rivera-Molina, Felix E. and Lin, Yu and Duim, Whitney C. and Long, Jane J. and Uchil, Pradeep D. and Myers, Jordan R. and Baird, Michelle A. and Mothes, Walther and Davidson, Michael W. and Toomre, Derek and Bewersdorf, Joerg},
          title   = {Video-rate nanoscopy using {sCMOS} camera-specific single-molecule localization algorithms},
          journal = {Nature Methods},
          volume  = {10},
          number  = {7},
          pages   = {653--658},
          year    = {2013},
          doi     = {10.1038/nmeth.2488},
        }
        @article{snyder1993image,
          author  = {Snyder, Donald L. and Hammoud, Abed M. and White, Richard L.},
          title   = {Image recovery from data acquired with a charge-coupled-device camera},
          journal = {Journal of the Optical Society of America A},
          volume  = {10},
          number  = {5},
          pages   = {1014--1023},
          year    = {1993},
          doi     = {10.1364/JOSAA.10.001014},
        }
        ```
