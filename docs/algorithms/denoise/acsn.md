# ACsN

!!! info "At a glance"
    - **Estimates:** denoised frames, in photoelectrons, with the noise level `sigma_` and the quality score `quality_` of each frame
    - **Noise model:** Poisson–Gaussian sCMOS noise, treated as white Gaussian after the offset and gain maps are removed
    - **Streaming:** per-frame (the sparse filter runs in blocks with a halo inside each frame)
    - **Needs many frames:** no
    - **Reference:** Mandracchia et al., Nat Commun, 2020, [doi:10.1038/s41467-019-13841-8](https://doi.org/10.1038/s41467-019-13841-8)

## Overview

[`ACsN`][optix.denoise.ACsN] (automatic correction of sCMOS-related noise) removes
the noise of fluorescence images recorded with an sCMOS camera
([Mandracchia et al. 2020](https://doi.org/10.1038/s41467-019-13841-8)). It
works in three steps. The offset and gain maps of the camera are removed, so that
each pixel is a count of photoelectrons. The standard deviation $\sigma_{\mathrm{N}}$ of the
noise is then estimated from the frame itself: a microscope cannot transmit
frequencies above the cutoff of its optical transfer function (OTF), $2\,\mathrm{NA}/\lambda_0$,
so whatever the frame holds beyond that cutoff is noise, and its level gives the
level of the noise at all frequencies. Last, a sparse filter of the BM3D type
(block matching and collaborative filtering; see [BM3D and BM4D](bm3d.md))
removes the noise, with $\sigma_{\mathrm{N}}$ as its only noise parameter and a set of
parameters of its own.

Use it for widefield, light-sheet or other images of an sCMOS camera, when the
numerical aperture, the emission wavelength and the pixel size in the image
plane are known, the pixels sample the OTF finely enough (the pixel is smaller
than $\lambda_0 / (4\,\mathrm{NA})$, so that the Nyquist frequency of the frame lies above the OTF
cutoff), and a calibration of the camera (offset and gain per pixel, for
instance from [`SCMOSCalibration`](../sensor/scmos-calibration.md)) is at hand.
It does not need the noise level: it estimates it. It is not a
deconvolution (see [Richardson–Lucy](../deconv/richardson-lucy.md)), it assumes the
noise is uncorrelated between pixels (see [VSNR](../destripe/vsnr.md) for stripes),
and it does not remove blur. Each frame is treated on its own. The filtering of
video along time of the paper (spatio-temporal grouping with motion tracking) is
not implemented; only its trigger is (see [What is not implemented](#what-is-not-implemented)).

The estimator is [`ACsN`][optix.denoise.ACsN] and the function
[`acsn`][optix.denoise.acsn] its shortcut (see the
[API reference](../../reference/denoise.md)). `fit` takes the offset and gain maps
as a [`SensorEstimate`](../../reference/estimates.md), `apply` returns the
denoised frames in electrons, and sets `sigma_` and `quality_`. The noise
estimate, the profile of the filter and the quality score are available as
building blocks in `optix.denoise.utils`.

## Image formation (optics)

The camera follows the model of
[Image formation and noise](../../concepts/image-formation.md), with a gain,
an offset and a read noise per pixel, as for the
[sCMOS calibration](../sensor/scmos-calibration.md). With $\lambda_i$ the expected
number of photoelectrons of pixel $i$ in the exposure:

=== "optix"

    $$
    \mathbf{y} = \mathbf{g} \odot \mathbf{e} + \mathbf{o} + \boldsymbol{\varepsilon},
    \qquad
    e_i \sim \mathrm{Poisson}(\lambda_i),
    \quad
    \varepsilon_i \sim \mathcal{N}(0, \sigma_i^2)
    $$

=== "Mandracchia et al. 2020"

    $$
    Z_p = \gamma_p \bigl\{ \mathrm{Pois}\{S_p(\tau)\} + \mathcal{N}(0, \sigma_{R}) \bigr\} + \beta_p
    $$

The fixed-pattern noise, the part of the noise that depends on the camera and
not on the scene, is that of the offset $\mathbf{o}$ and the gain $\mathbf{g}$.
They are measured once, and give the frame in photoelectrons
([Mandracchia et al. 2020, Supplementary Eqs. 3 and 4](https://doi.org/10.1038/s41467-019-13841-8)):

=== "optix"

    $$
    \mathbf{m} = \frac{\mathbf{y} - \mathbf{o}}{\mathbf{g}}
    = \mathbf{e} + \frac{\boldsymbol{\varepsilon}}{\mathbf{g}}
    $$

=== "Mandracchia et al. 2020"

    $$
    M_p = \frac{Z_p - \beta_p}{\gamma_p}
    $$

| optix | Mandracchia et al. 2020 | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}$ | $Z$ | recorded frame, in ADU | $\mathbb{R}^{\lvert\Omega\rvert}$, per frame |
| $\mathbf{g}$ | $\gamma$ | gain, in ADU per electron | $\mathbb{R}^{\lvert\Omega\rvert}$, shared by all frames of a group |
| $\mathbf{o}$ | $\beta$ | offset, in ADU | $\mathbb{R}^{\lvert\Omega\rvert}$, shared by all frames of a group |
| $\mathbf{e}$ | $\mathrm{Pois}\{S(\tau)\}$ | photoelectrons collected | $\mathbb{R}^{\lvert\Omega\rvert}$, per frame |
| $\boldsymbol{\lambda}$ | $S(\tau)$ | expected photoelectrons (the signal) | $\mathbb{R}^{\lvert\Omega\rvert}$, per frame |
| $\boldsymbol{\varepsilon}$, $\sigma_i$ | $\gamma\,\mathcal{N}(0, \sigma_R)$, $\gamma\sigma_R$ | read noise of the pixel and its standard deviation, in ADU | $\mathbb{R}^{\lvert\Omega\rvert}$ |
| $\sigma_i / g_i$ | $\sigma_R$ | read-noise standard deviation, in electrons | $\mathbb{R}^{\lvert\Omega\rvert}$ |
| $\mathbf{m}$ | $M$ | frame corrected for offset and gain, in electrons | $\mathbb{R}^{\lvert\Omega\rvert}$, per frame |

The paper writes the gain $\gamma$ and the offset $\beta$, and uses $\gamma$ also for
the hard-threshold operator of the filter (below).

The optics enter through the frequency content of the signal. The image of a
fluorescent specimen is the specimen convolved with the PSF, so its Fourier
transform is that of the specimen times the OTF, which vanishes for frequencies
above the cutoff

$$
\nu_c = \frac{2\,\mathrm{NA}}{\lambda_0}
$$

(the radius of the OTF of an incoherent circular-aperture system, computed from
the numerical aperture and the emission wavelength:
[Mandracchia et al. 2020, Supplementary Note 2.3](https://doi.org/10.1038/s41467-019-13841-8);
[`otf_cutoff`][optix.denoise.utils.otf_cutoff]). The camera samples the image on a grid of
pixel size $d_{\mathrm{px}}$ (the camera pixel divided by the magnification), which
records frequencies up to the Nyquist frequency $1/(2 d_{\mathrm{px}})$ along each
axis. When $\nu_c$ is below it, the frame holds a band of frequencies, between
$\nu_c$ and the camera cutoff $\bar\nu$, where there is no signal. Aberrations
reduce the effective cutoff, so the theoretical value is conservative.

## Probabilistic model

ACsN is not derived from a probabilistic model: it is a filter, built on three
assumptions.

1. **The noise is white and Gaussian.** The photoelectron count is Poisson,
   which is well approximated by a Gaussian of variance $\lambda_i$ when the
   flux exceeds a few photoelectrons per pixel (more than 3 for the shape of the
   distribution, more than 5 for a standard deviation within 1 %;
   [Supplementary Note 2.3, Supplementary Fig. 13](https://doi.org/10.1038/s41467-019-13841-8)).
   The read noise is Gaussian. The sum of the two independent Gaussians is a
   Gaussian of variance $\sigma_{\mathrm{N}}^2 = \sigma_{R}^2 + \sigma_{G}^2$
   (Supplementary Eq. 5), with a power spectral density that is constant over
   the frequencies. The shot noise has a variance $\lambda_i$ that varies over the
   frame, so $\sigma_{\mathrm{N}}^2$ stands for its mean over the pixels,
   $\overline{\lambda_i + \sigma_i^2 / g_i^2}$.
2. **The signal lies within the OTF.** The spectrum of $\mathbf{x}$ vanishes for
   $\lvert\boldsymbol{\nu}\rvert > \nu_c$, so the spectrum of $\mathbf{m}$ there is that of
   the noise.
3. **Nonlocal sparsity**, as for [BM3D](bm3d.md): a group of similar blocks has
   a sparse spectrum under a separable transform of the blocks followed by a
   transform across the group, and the noise stays white in that spectrum.

The corrected frame is then

$$
\mathbf{m} = \mathbf{x} + \boldsymbol{\eta},
\qquad
\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \sigma_{\mathrm{N}}^2 \mathbf{I}),
\qquad
\mathbf{x} = \boldsymbol{\lambda},
$$

and the task is to estimate $\mathbf{x}$ and the level $\sigma_{\mathrm{N}}$.

The symbols of the sparse filter are those of the
[notation page](../../concepts/notation.md#block-matching-denoising), and those of the
noise estimate are on the same page, in
[sCMOS noise estimation](../../concepts/notation.md#scmos-noise-estimation).

## Inference / derivation

The estimate is made in two parts: the level $\sigma_{\mathrm{N}}$ of the noise is
estimated from the high frequencies, then the filter removes noise of that level.
There is no iteration. In statistical terms the first part is a *method of
moments* for the noise variance, on the band of frequencies where the
spectrum holds only noise, and the second is the same kind of plug-in filter as
[BM3D](bm3d.md): hard thresholding is $\ell_0$-penalised least squares in an
orthonormal basis, and the Wiener stage is the linear minimum-mean-squared-error
estimator with the signal power of the first estimate plugged in.

### Noise estimate

Let $\mathbf{m}_{\mathrm{hp}}$ be the high-pass image that keeps the frequencies
outside the OTF, an ideal (step) filter on the DFT frequencies of the frame,

$$
\mathbf{m}_{\mathrm{hp}} = \mathcal{F}^{-1}\bigl[\,\mathbb{1}(\lvert\boldsymbol{\nu}\rvert > \nu_c) \odot \mathcal{F}\mathbf{m}\,\bigr],
$$

([`highpass`][optix.denoise.utils.highpass]). It holds noise only, so its standard
deviation $\sigma_{\mathrm{hp}}$ measures the noise above $\nu_c$ (the mean is zero). The noise
$\sigma_{\mathrm{N}}$ of the whole band follows from the white spectrum. With
$N_0$ the power per unit bandwidth, the camera recording up to the cutoff $\bar\nu$:

=== "optix"

    $$
    \sigma_{\mathrm{N}}^2 = \int_0^{\bar\nu} N_0\, d\nu = N_0\,\bar\nu,
    \qquad
    \sigma_{\mathrm{hp}}^2 = \int_{\nu_c}^{\bar\nu} N_0\, d\nu = N_0\,(\bar\nu - \nu_c),
    \qquad
    \sigma_{\mathrm{N}} = \underbrace{\sqrt{\frac{\bar\nu}{\bar\nu - \nu_c}}}_{\varrho}\,\sigma_{\mathrm{hp}}
    $$

=== "Mandracchia et al. 2020"

    $$
    \sigma_N^2 = \int_{0}^{\bar f} N_0 \lvert H(f)\rvert^2 df,
    \qquad
    \sigma_c^2 = \int_{f_c}^{\bar f} N_0 \lvert H(f)\rvert^2 df,
    \qquad
    \sigma_N = \sqrt{\frac{\bar f}{\bar f - f_c}}\;\sigma_c
    $$

([Mandracchia et al. 2020, Supplementary Eqs. 6–11](https://doi.org/10.1038/s41467-019-13841-8);
$H(f) = 1$ up to $\bar f$ is the transfer function of the camera,
Supplementary Eq. 7. Implemented in [`noise_ratio`][optix.denoise.utils.noise_ratio]
and [`acsn_noise`][optix.denoise.utils.acsn_noise].)

| optix | Mandracchia et al. 2020 | Meaning | Shape |
|---|---|---|---|
| $\nu$, $\boldsymbol{\nu}$ | $f$ | spatial frequency, in cycles per unit of length | scalar; two components per DFT frequency |
| $\nu_c$ | $f_c$ | cutoff of the OTF, $2\,\mathrm{NA}/\lambda_0$ | scalar, one per call |
| $\bar\nu$ | $\bar f$ | cutoff of the camera, `camera_cutoff` times the Nyquist frequency $1/(2 d_{\mathrm{px}})$ | scalar, one per call |
| $N_0$ | $N_0$ | power of the white noise per unit bandwidth | scalar |
| $\sigma_{\mathrm{hp}}$ | $\sigma_c$ | standard deviation of the high-pass image | scalar, per frame |
| $\sigma_{\mathrm{N}}$ | $\sigma_N$ | standard deviation of the noise of $\mathbf{m}$, in electrons | scalar, per frame (`sigma_`) |
| $\varrho$ | $\sqrt{\bar f/(\bar f - f_c)}$ | noise ratio $\sigma_{\mathrm{N}} / \sigma_{\mathrm{hp}}$ | scalar, one per call |

The noise ratio $\varrho$ depends only on the frequencies, not on the data.
The derivation is one-dimensional: it counts the frequencies on a line. In a
2-D frame the frequencies fill a plane, and the formula overestimates
$\sigma_{\mathrm{N}}$: see [The 2-D noise ratio](#the-2-d-noise-ratio).

The standard deviation $\sigma_{\mathrm{hp}}$ is not the root mean square of
$\mathbf{m}_{\mathrm{hp}}$ but the width of a Gaussian fitted to the histogram of
its magnitudes (choice 1 below); the root mean square is the extension
`sigma_fit="std"`.

### Sparse filter

The collaborative filter is that of [BM3D](bm3d.md), with different parameters
and two departures in the matching and the weights. It runs twice: hard
thresholding, then an empirical Wiener filter whose gain is computed from the
first estimate. The steps and equations of
[BM3D](bm3d.md#inference-derivation) apply (grouping, transform, shrinkage,
aggregation); only what differs is given here, with the equations of the paper.

**Grouping.** The reference blocks lie on a grid of step 3. For each, the
blocks of a $19 \times 19$ search window are compared with the distance between
their pixels, and the closest, up to 8, are kept
([Supplementary Eq. 12](https://doi.org/10.1038/s41467-019-13841-8)):

=== "optix"

    $$
    d(\mathbf{r}, \mathbf{r}') = \frac{1}{K}\,\lVert \mathbf{p}_{\mathbf{r}} - \mathbf{p}_{\mathbf{r}'} \rVert_2^2,
    \qquad
    S_{\mathbf{r}} = \bigl\{\mathbf{r}' : d(\mathbf{r}, \mathbf{r}') \le \tau \bigr\}
    $$

=== "Mandracchia et al. 2020"

    $$
    d(P, Q) = \frac{\lVert P - Q \rVert_2^2}{k^2}
    $$

**Hard thresholding.** The group is transformed, the coefficients below
$\lambda_{\mathrm{ht}}\,\sigma_{\mathrm{N}}$ are zeroed, and the block estimates are averaged
with a weight that is the inverse of the number of coefficients that remain
([Supplementary Eqs. 13–15](https://doi.org/10.1038/s41467-019-13841-8)):

=== "optix"

    $$
    \hat{\boldsymbol{\Theta}} = \boldsymbol{\Theta} \odot \mathbb{1}\bigl[\lvert\boldsymbol{\Theta}\rvert > \lambda_{\mathrm{ht}}\sigma_{\mathrm{N}}\bigr],
    \qquad
    w_{\mathbf{r}} =
    \begin{cases}
    1 / N_{\mathrm{har}} & N_{\mathrm{har}} \ge 1 \\
    1 & \text{otherwise}
    \end{cases}
    $$

=== "Mandracchia et al. 2020"

    $$
    \mathcal{P}^h(P) = {\tau^h_{3D}}^{-1}\Bigl(\gamma\bigl(\tau^h_{3D}(\mathcal{P}[P])\bigr)\Bigr),
    \quad
    \gamma(x) = \begin{cases} 0 & \lvert x\rvert \le \lambda_\gamma \sigma_N \\ x & \text{otherwise}\end{cases},
    \quad
    w_P = \begin{cases} (N_P)^{-1} & N_P \ge 1 \\ 1 & \text{otherwise}\end{cases}
    $$

**Wiener filtering.** The group of the first estimate (the "intermediate image"),
at positions found on that image, gives the gain
([Supplementary Eqs. 16–17](https://doi.org/10.1038/s41467-019-13841-8)):

=== "optix"

    $$
    \mathbf{W} = \frac{(\boldsymbol{\Theta}^{\mathrm{b}})^{2}}{(\boldsymbol{\Theta}^{\mathrm{b}})^{2} + \sigma_{\mathrm{N}}^2},
    \qquad
    \hat{\boldsymbol{\Theta}} = \mathbf{W} \odot \boldsymbol{\Theta}
    $$

=== "Mandracchia et al. 2020"

    $$
    \mathcal{P}^w(P) = {\tau^w_{3D}}^{-1}\bigl(c_P \cdot \tau^w_{3D}(\mathcal{P}[P])\bigr),
    \qquad
    c_P = \frac{\lvert\tau^w_{3D}(\mathcal{P}[P])\rvert^2}{\lvert\tau^w_{3D}(\mathcal{P}[P])\rvert^2 + \sigma_N^2}
    $$

In these equations the gain $c_P$ is computed from the group of the
intermediate image and applied to the group of the noisy image; the paper writes
both as $\mathcal{P}[P]$.

| optix | Mandracchia et al. 2020 | Meaning | Shape |
|---|---|---|---|
| $N$ | $k$ | side of a block | scalar, 8 |
| $N_S$ | $m$ | side of the search window | scalar, 19 |
| $M$ | none | largest number of blocks in a group | scalar, 8 |
| $N_{\text{step}}$ | none | step between reference blocks | scalar, 3 |
| $\mathbf{p}_{\mathbf{r}}$, $\mathbf{p}_{\mathbf{r}'}$ | $P$, $Q$ | reference block and candidate block | $\mathbb{R}^{K}$, $K = N^2$ |
| $d$, $\tau$ | $d(P, Q)$, none | distance and matching threshold (2500 for the first run) | scalar |
| $\mathbf{G}$ | $\mathcal{P}[P]$ | group of the matched blocks | $\mathbb{R}^{m \times K}$, per reference |
| $\boldsymbol{\Theta}$ | $\tau_{3D}(\mathcal{P}[P])$ | spectrum of the group | $\mathbb{R}^{m \times K}$, per reference |
| $\boldsymbol{\Theta}^{\mathrm{b}}$ | $\tau^w_{3D}$ of the intermediate image | spectrum of the group of the first estimate | $\mathbb{R}^{m \times K}$, per reference |
| $\lambda_{\mathrm{ht}}$ | $\lambda_\gamma$ | hard threshold, in units of $\sigma_{\mathrm{N}}$ | scalar, 4.0 |
| $N_{\mathrm{har}}$ | $N_P$ | coefficients that survive the threshold | scalar, per reference |
| $w_{\mathbf{r}}$ | $w_P$ | weight of the group in the aggregation | scalar, per reference |
| $\mathbf{W}$ | $c_P$ | empirical Wiener gain | $\mathbb{R}^{m \times K}$, per reference |

The paper has no symbol for the matching threshold, "the minimum distance under
which two blocks are assumed similar" (Note 3.1), nor for the largest group and the
step. Its values for $\lambda_{\mathrm{ht}}$ (4.0, chosen by PSNR and SSIM in
Supplementary Table 6), $\tau$, the block, the window and the group are in
[Supplementary Note 3.1](https://doi.org/10.1038/s41467-019-13841-8).

#### Differences from BM3D

The parameters are in the units of 8-bit images, as those of BM3D (choice 3 below).
BM3D is taken at its Normal Profile for $\tilde\sigma \le 40$ ([Dabov et al. 2007, Table I](https://doi.org/10.1109/TIP.2007.901238)).

| | ACsN | BM3D (Normal Profile, $\tilde\sigma \le 40$) |
|---|---|---|
| Block size $N$ | 8 | 8 |
| Search window $N_S$ | 19 | 39 |
| Largest group $M$ | 8, in both runs | 16 (first run), 32 (second run) |
| Step $N_{\text{step}}$ | 3 | 3 |
| Hard threshold $\lambda_{\mathrm{ht}}$ | 4.0 | 2.7 |
| Matching of the first run | distance between the pixels of the blocks | distance between their bior1.5 coefficients |
| $\tau$, first run | 2500 | 2500 |
| $\tau$, second run | 400 | 400 |
| Weight of the first run | $1/N_{\mathrm{har}}$ | $1/(\sigma^2 N_{\mathrm{har}})$ |
| Weight of the second run | $1/(\sigma_{\mathrm{N}}^2\lVert\mathbf{W}\rVert_2^2)$ | $1/(\sigma^2\lVert\mathbf{W}\rVert_2^2)$ |
| Window of the aggregation | none | Kaiser, $\beta = 2$ |
| Parameters for $\tilde\sigma > 40$ | none: the same for every $\sigma_{\mathrm{N}}$ | a second profile |
| Intensity range | each frame mapped to $[0, 255]$ by its minimum and maximum | `data_range` (default: estimated) |
| Noise level | estimated from each frame | given, or from a `SensorEstimate` |

Without the factor $\sigma^2$, the weight of a group with one coefficient left is
the weight 1 of a group with none; with it, the second weighs far more than the first (BM3D,
[choice 8](bm3d.md#choices-not-in-the-papers)). The distance on pixels equals
that of the spectra when the transform is orthonormal, which bior1.5 is not.

### Quality score and video trigger

The paper checks the result with a score that needs no reference image, and
decides from the temporal PSNR whether video filtering would be worth running.

**Q-score** ([Supplementary Note 7, Eqs. 17–19](https://doi.org/10.1038/s41467-019-13841-8),
after [Kong et al. 2013](https://doi.org/10.1109/ICCV.2013.359);
[`quality_score`][optix.denoise.utils.quality_score]). The shot noise and the read
noise are independent of the structures of the image, so the *method noise*
$\mathbf{m} - \hat{\mathbf{x}}$ (what the filter removed) should have no structure of the image
in it, while the denoised image should. Two maps of local structure similarity
measure this, between the noisy image $\mathbf{m}$ and the method noise, and between
$\mathbf{m}$ and the denoised image. Where the second is large (structure), the
first should be small, and the reverse; the score is the negative of the
correlation between the maps, so that it is high for a good result:

=== "optix"

    $$
    S(\mathbf{a}, \mathbf{b}) = \frac{\sigma_{ab} + c}{\sigma_a \sigma_b + c},
    \quad
    \mathbf{s}_{\mathrm{N}} = S(\mathbf{m},\, \mathbf{m} - \hat{\mathbf{x}}),
    \quad
    \mathbf{s}_{\mathrm{P}} = S(\mathbf{m},\, \hat{\mathbf{x}}),
    \quad
    Q = -\operatorname{corr}(\mathbf{s}_{\mathrm{N}}, \mathbf{s}_{\mathrm{P}})
    $$

=== "Mandracchia et al. 2020"

    $$
    M_I = I - \hat I,
    \quad
    \rho(A, B) = \frac{\sigma_{AB} + c}{\sigma_A \sigma_B + c},
    \quad
    N = \rho(I, M_I),
    \quad
    P = \rho(I, \hat I),
    \quad
    Q\text{-score} = -\rho(N, P)
    $$

The statistics $\sigma_a$, $\sigma_b$, $\sigma_{ab}$ are local (see choice 6 below), the correlation is the
Pearson coefficient over the pixels of the maps, and $Q \in [-1, 1]$. A higher score is a better
result. It is computed for each frame, on $\mathbf{m}$ and the denoised frame, and stored
in `quality_`.

**Video trigger** ([Supplementary Note 2.5](https://doi.org/10.1038/s41467-019-13841-8);
[`temporal_psnr`][optix.denoise.utils.temporal_psnr]). For a sequence of denoised
frames $\hat{\mathbf{x}}_t$, the paper compares each frame to the average of
itself and its neighbours. With the frames $t, \ldots, t + 5$:

$$
\mathrm{PSNR}_t = 10 \log_{10} \frac{\max(\hat{\mathbf{x}}_t)^2}{\operatorname{mean}\bigl[(\hat{\mathbf{x}}_t - \bar{\mathbf{x}}_t)^2\bigr]},
\qquad
\bar{\mathbf{x}}_t = \frac{1}{6} \sum_{s=t}^{t+5} \hat{\mathbf{x}}_s,
$$

and the video filtering is run when the average PSNR is below 35 dB (empirical
threshold). The function only computes the values of the PSNR; nothing runs
from it (see [What is not implemented](#what-is-not-implemented)).

## Algorithm

**Input:** frames $\mathbf{y} \in \mathbb{R}^{\lvert\Omega\rvert}$ (ADU), each of $H \times W$ pixels with
$H, W \ge 8$; offset $\mathbf{o}$ and gain $\mathbf{g}$ (scalars or maps; default 0 and 1,
for data already in electrons); $\mathrm{NA}$, $\lambda_0$, $d_{\mathrm{px}}$;
`ratio = "1d"`, `sigma_fit = "histogram"`, `camera_cutoff = 1` \
**Output:** denoised frames $\hat{\mathbf{x}}$, in electrons, of the shape of the data;
$\sigma_{\mathrm{N}}$ and $Q$ of each frame.

Each frame, on its own:

1. **Correct** the fixed pattern: $\mathbf{m} = (\mathbf{y} - \mathbf{o}) / \mathbf{g}$.
2. **Estimate the noise.**
    1. $\nu_c = 2\,\mathrm{NA}/\lambda_0$ and $\bar\nu = \texttt{camera\_cutoff} / (2 d_{\mathrm{px}})$
       (an error if $\nu_c \ge \bar\nu$);
    2. high-pass: $\mathbf{m}_{\mathrm{hp}}$ keeps the DFT frequencies with $\lvert\boldsymbol{\nu}\rvert > \nu_c$;
    3. $\sigma_{\mathrm{hp}}$: the magnitudes $\lvert\mathbf{m}_{\mathrm{hp}}\rvert$ are binned in 10 equal bins between their minimum and maximum, and
       $a_0 \exp(-x^2 / (2\sigma_{\mathrm{hp}}^2))$ is fitted to the counts at the centres
       of the bins by nonlinear least squares;
    4. $\sigma_{\mathrm{N}} = \varrho\,\sigma_{\mathrm{hp}}$ with $\varrho = \sqrt{\bar\nu / (\bar\nu - \nu_c)}$.
3. **Scale** to the range of 8 bits: $s = 255 / (\max\mathbf{m} - \min\mathbf{m})$,
   $\tilde{\mathbf{m}} = s\,(\mathbf{m} - \min\mathbf{m})$, $\tilde\sigma = s\,\sigma_{\mathrm{N}}$.
   A constant frame, or $\sigma_{\mathrm{N}} \le 0$, is returned unchanged.
4. **First estimate.** For each reference block on the grid of step 3:
    1. **group**: the blocks of the $19 \times 19$ window with $d \le 2500$, the
       reference first, at most 8, their number rounded down to a power of two;
    2. **transform** the group: bior1.5 on the blocks, Haar across the group;
    3. **hard-threshold** at $4.0\,\tilde\sigma$ and set $w_{\mathbf{r}} = 1/N_{\mathrm{har}}$ (1 if none remains);
    4. **invert** and **accumulate** the weighted block estimates, without a window.

    Then $\hat{\mathbf{x}}^{\mathrm{b}} = \mathbf{n} / \mathbf{d}$, the weighted average at each pixel.
5. **Final estimate.** For each reference block:
    1. **group** with $d < 400$ computed between the blocks of $\hat{\mathbf{x}}^{\mathrm{b}}$; cut the
       noisy group and the group of $\hat{\mathbf{x}}^{\mathrm{b}}$ at these positions;
    2. **transform** both: DCT on the blocks, Haar across the group;
    3. **Wiener-filter** the noisy spectrum with the gain $\mathbf{W}$ of the other, and set
       $w_{\mathbf{r}} = 1 / (\tilde\sigma^2 \lVert\mathbf{W}\rVert_2^2)$;
    4. **invert** and **accumulate** as above.

    Then $\hat{\mathbf{x}} = \mathbf{n} / \mathbf{d}$.
6. **Unscale**: $\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} / s + \min\mathbf{m}$.
7. **Score**: $Q = Q(\mathbf{m}, \hat{\mathbf{x}})$ (not defined, and set to NaN, for frames smaller than $11 \times 11$).

## Faithful mode and extensions

`faithful=True` (the default) is the method of the paper: the noise ratio of
Supplementary Eq. 11, $\sigma_{\mathrm{hp}}$ from the histogram, $\bar\nu$ at the
Nyquist frequency, no hot-pixel removal, $\sigma_{\mathrm{N}}$ estimated, and the
filter of the table above. The paper leaves a number of choices open. The code
makes them as follows, in two lists: those taken from the authors' code, and the
others.

### Choices from the ACsN code, not in the paper

The code of the authors ([ShuJiaLab/ACsN](https://github.com/ShuJiaLab/ACsN),
BSD 3-Clause; its notice is kept in the header of `_acsn.py`) was read to settle these
details. Its sparse filter is not used: the filtering runs on the BM3D engine of
optix.

1. **The fit of $\sigma_{\mathrm{hp}}$.** The paper takes it from "the intensity
   distribution of the pixel values in a high-pass filtered image" (Supplementary
   Fig. 12b), without saying how. The magnitudes $\lvert\mathbf{m}_{\mathrm{hp}}\rvert$ are binned in 10 equal
   bins between their minimum and maximum, and a Gaussian
   $a_0\exp(-x^2/(2\sigma_{\mathrm{hp}}^2))$ is fitted to the counts at the bin centres
   by Levenberg–Marquardt, starting from $a_0$ equal to the largest count and
   $\sigma_{\mathrm{hp}}$ equal to the centre of bin $\mathrm{round}(k/2)$, with $k$ the
   (1-based) bin of the smallest count.
2. **The high-pass filter is an ideal step** at $\nu_c$ on the DFT frequencies;
   the paper does not give the shape of the filter.
3. **The mapping to 8 bits.** The paper takes $\tau = 2500$ for $\sigma_{\mathrm{N}} < 40$, the
   value of BM3D for 8-bit images
   ([Dabov et al. 2007](https://doi.org/10.1109/TIP.2007.901238)), and does not say how
   the data are brought to that range. Each frame is mapped to $[0, 255]$ by its own
   minimum and maximum before the sparse filter, with $\sigma_{\mathrm{N}}$ scaled alike,
   and mapped back afterwards. The result therefore depends on the extremes of the frame.
4. **Hot-pixel removal.** The code replaces the pixels whose high-pass magnitude
   exceeds $\operatorname{mean}\lvert\mathbf{m}_{\mathrm{hp}}\rvert + 3\operatorname{sd}\lvert\mathbf{m}_{\mathrm{hp}}\rvert$ by the $3 \times 3$ median
   of the frame, before the noise estimate. It is on by default there and is not
   in the paper. Here it is the extension `hotspot=True`.
5. **The camera cutoff.** The paper calls $\bar f$ the cutoff of the camera and
   gives no value. The code uses $1.1/(2 d_{\mathrm{px}})$, 1.1 times the Nyquist
   frequency. optix defaults to the Nyquist frequency (`camera_cutoff=1`);
   `camera_cutoff=1.1` is the value of the code. For NA 1.4, $\lambda_0 = 0.7$ µm and
   $d_{\mathrm{px}} = 0.1$ µm it changes $\varrho$ from 2.24 to 1.91.
6. **Clamps.** The code clamps the gain map to values $\ge 1$ and the corrected
   frame to values $> 0$. Neither is done here: the gain of optix is in ADU per
   electron and may be below 1, and a negative value of $\mathbf{m}$ (a pixel
   below the offset) is data. A gain $\le 0$ is an error.
7. **The neighbours of the video trigger.** The paper compares a frame with "its 5 nearest
   neighbors"; in the code they are the 5 *following* frames
   ([`temporal_psnr`][optix.denoise.utils.temporal_psnr]).

### Other choices not in the paper

1. **Transforms and the second run.** The paper does not name the transforms
   $\tau^h_{3D}$ and $\tau^w_{3D}$, nor the parameters of the second run beyond the
   group of 8 patches. They are those of BM3D's Normal Profile, which the paper
   cites: bior1.5 on the blocks of the first run and DCT on those of the second, the Haar
   transform across the group (the construction is described in
   [choice 2 of BM3D](bm3d.md#choices-not-in-the-papers)), $\tau = 400$ with
   a strict test $d < \tau$ in the second run, and the weight
   $1/(\sigma_{\mathrm{N}}^2\lVert\mathbf{W}\rVert_2^2)$ of Eq. 11 of
   [Dabov et al. (2007)](https://doi.org/10.1109/TIP.2007.901238).
2. **Group sizes.** The group has at most 8 blocks; its size is rounded down to a
   power of two, as the Haar transform needs.
3. **Reference grid, search window and ties.** As in
   [BM3D](bm3d.md#choices-not-in-the-papers) (choices 3 and 4): the reference
   blocks lie on the global grid of step 3 plus the last position along each axis, the search window is
   centred on the reference and clipped at the border, and blocks at equal
   distance keep the raster order of the window. The norm $\lVert\mathbf{W}\rVert_2$ is
   floored at the smallest positive floating-point number (choice 6 of BM3D).
4. **The DC coefficient.** Eq. 14 thresholds every coefficient and the paper does not
   exempt the DC, so it is thresholded like the others, as in BM3D (choice 8 there).
   The method is not invariant to an offset of the intensities; after the mapping of
   choice 3 of the list above, the darkest pixel of the frame is 0.
5. **The camera cutoff $\bar\nu$ is the Nyquist frequency** (see the choice 5 above for the code's value).
6. **The window of the Q-score.** The paper does not give the local
   statistics of Eq. 18. They use the window of the SSIM index
   ([Wang et al. 2004](https://doi.org/10.1109/TIP.2003.819861)): a Gaussian of
   $11 \times 11$ pixels with $\sigma = 1.5$, at the positions where the window lies
   entirely inside the frame, and $c = (0.03\,L)^2 / 2$, with $L$ the range of
   $\mathbf{m}$. A constant map has no correlation and gives $Q = 0$.
7. **A noise level that varies over the frame.** The paper has one $\sigma_{\mathrm{N}}$. With a
   gain and a read noise per pixel and a shot noise that follows the signal,
   $\sigma_{\mathrm{N}}^2$ is the mean variance over the frame, estimated from the high
   frequencies of the whole frame. The filter then uses this level at
   every pixel.
8. **Whole frames.** The noise estimate needs the Fourier transform of a whole
   frame, so a frame is read at once (see
   [Memory and streaming](#memory-and-streaming)). A frame smaller than $8 \times 8$ raises an error.

### The 2-D noise ratio

Supplementary Eq. 11 is derived in one dimension. The power $N_0$ is spread over
a *line* of frequencies from 0 to $\bar\nu$, and the fraction above $\nu_c$ is
the length $(\bar\nu - \nu_c)/\bar\nu$. A frame has two frequency axes. Its white
noise has the same power at every point of the frequency plane, so the fraction of
the noise above $\nu_c$ is the *area* of the plane outside the circle of radius
$\nu_c$, which is larger than the length fraction because the area of a ring grows
with its radius. With $n$ the number of DFT frequencies of the frame
and $n_c$ the number with $\lvert\boldsymbol{\nu}\rvert > \nu_c$, the exact ratio for white
noise is $\sigma_{\mathrm{N}} = \sqrt{n / n_c}\;\sigma_{\mathrm{hp}}$
([`noise_ratio`][optix.denoise.utils.noise_ratio] with `ratio="area"`).

The 1-D ratio is larger than this one, and the factor grows with $\nu_c / \bar\nu$
(a frame of $256^2$ pixels, Nyquist frequency taken for $\bar\nu$):

| $\nu_c / \bar\nu$ | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|
| $\varrho$, 1-D (`"1d"`) | 1.20 | 1.41 | 1.83 | 2.24 | 3.16 |
| $\varrho$, 2-D (`"area"`) | 1.04 | 1.12 | 1.28 | 1.42 | 1.66 |
| overestimate by the 1-D ratio | 1.15 | 1.27 | 1.43 | 1.58 | 1.91 |

For $\nu_c / \bar\nu = 0.8$ (NA 1.4, $\lambda_0 = 0.7$ µm, $d_{\mathrm{px}} = 0.1$ µm, the
setting of the paper's simulations and of the tests), the 1-D formula
overestimates $\sigma_{\mathrm{N}}$ by a factor of about 1.5 on simulated white noise
(see [Results on synthetic data](#results-on-synthetic-data)). The overestimate is
a property of the formula, not of the fit of $\sigma_{\mathrm{hp}}$. The faithful
mode uses the 1-D ratio, as the paper does. `ratio="area"` uses the exact 2-D value. The
filter does not need the noise level to be exact: a $\sigma_{\mathrm{N}}$ 1.5 times
too large thresholds more coefficients and smooths more, and on the simulated
frames the PSNR of the result changes by less than 0.6 dB (table below); but
the quantitative use of $\sigma_{\mathrm{N}}$ (`sigma_`) is off by that factor.

### Extensions (opt-in, not faithful)

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `ratio` | the noise ratio $\varrho$: `"1d"` (Supplementary Eq. 11) or `"area"` (exact for 2-D white noise) | `"1d"` | `"area"` |
| `sigma_fit` | $\sigma_{\mathrm{hp}}$: `"histogram"` (the code) or `"std"`, the root mean square of $\mathbf{m}_{\mathrm{hp}}$ | `"histogram"` | `"histogram"` |
| `camera_cutoff` | $\bar\nu$, in units of the Nyquist frequency (1.1 in the code) | 1.0 | 1.0 |
| `hotspot` | hot-pixel removal of the code (choice 4 above) | `False` | `False` |
| `sigma` | a known $\sigma_{\mathrm{N}}$, in electrons, in place of the estimate | `None` | `None` |

`faithful=False` selects `ratio="area"` and keeps the other values. Any value that
differs from the faithful one sets `is_faithful` to false. The other keywords do
not change the method:

| Keyword | Changes | Value |
|---|---|---|
| `na`, `wavelength`, `pixel_size` | $\mathrm{NA}$, $\lambda_0$ (the unit of length of `pixel_size`) and $d_{\mathrm{px}}$ | required |
| `block` | core size of the blocks of the sparse filter; the result does not depend on it | `None` (memory budget) |

### What is not implemented

**Video and 3-D filtering.** For sequences, the paper adds a second sparse
filtering that also searches for similar blocks in the neighbouring frames: a
search volume of $m \times m \times t$ with $t = 9$, along trajectories that follow the motion
of the scene, found on the sequence down-sampled by 2 with a diamond search, in
areas of $q \times q = 200 \times 200$ pixels
([Supplementary Notes 2.5 and 3.1](https://doi.org/10.1038/s41467-019-13841-8)).
It follows the spatio-temporal filters V-BM4D
([Maggioni et al. 2012](https://doi.org/10.1109/TIP.2012.2199324)) and RF3D
([Maggioni et al. 2014](https://doi.org/10.1109/TIP.2014.2345261)), which are not part of optix.
Frames are therefore denoised one at a time, and the temporal self-similarity of a
time-lapse is not used. `temporal_psnr` gives the value that the paper uses to decide when
the video filtering is needed; to apply a spatio-temporal filter afterwards is
left to the user.

## Memory and streaming

The tag is `per-frame`. Each frame is processed on its own, with batches of frames
read by the [`frame_batches`](../../reference/io.md) iterator and the result
written back as they are denoised; the data may be a zarr array or a memory-mapped file larger than memory, and `out=` a
separate array. Images of a stack, and groups (with a gain and offset map of
their own, as maps of `(*group, *spatial)`), are independent. The estimator keeps $\sigma_{\mathrm{N}}$ and $Q$ of every frame, two numbers
per frame.

A frame is held whole: the noise estimate takes its Fourier transform, and the
masks, the complex spectrum and the histogram take a few copies of its size. The
sparse filter runs in blocks of the frame with a halo, with the same engine as
[BM3D](bm3d.md#memory-and-streaming): the reach of a run is
$2\lfloor N_S/2\rfloor + N - 1 = 25$ pixels, and the halo of the two runs is 50 pixels.
The core of a block is sized from the memory budget of
[`optix.config`](../../reference/configuration.md), or set with `block`, and the result equals
the computation on the whole frame, up to floating-point rounding, for any
`block`. Within a block, the spectra of all blocks are kept: 64 values per pixel,
and twice that in the second run.

The cost per pixel and run is about $N_S^2 N^2 / N_{\text{step}}^2 \approx 2600$
operations, a quarter of the Normal Profile of BM3D, whose window is larger. The noise estimate costs an FFT and a
fit of 10 counts, and is negligible. The computation is vectorised and runs on any torch device; on the CPU
it is deterministic.

## Usage

The first example simulates an sCMOS frame: filaments imaged through the OTF of
a NA 1.4 objective ($\lambda_0 = 0.7$ µm, pixels of 0.1 µm, so
$\nu_c = 4$ cycles/µm against a Nyquist frequency of 5), with a signal of 5 to
60 electrons, a gain of 2 ADU per electron, an offset of 100 ADU and a read noise of 3 ADU.

```pycon
>>> import math
>>> import torch
>>> from optix.denoise import ACsN, acsn
>>> from optix.denoise.utils import (
...     acsn_profile,
...     noise_ratio,
...     otf_cutoff,
...     stage_halo,
... )
>>> from optix.estimates import SensorEstimate
>>> from optix.models import phantoms, simulate_sensor
>>> na, wavelength, pixel = 1.4, 0.7, 0.1
>>> nu_c = otf_cutoff(na, wavelength)
>>> nu_c
4.0
>>> f = torch.fft.fftfreq(128, d=pixel)
>>> r = (torch.hypot(f[:, None], f) / nu_c).clamp(max=1)
>>> otf = (2 / math.pi) * (torch.acos(r) - r * torch.sqrt(1 - r * r))
>>> x = phantoms.filaments((128, 128), 8, width=1.0, seed=0)
>>> x = torch.fft.ifft2(torch.fft.fft2(x) * otf).real.clamp(min=0)
>>> x = 5 + 55 * x / x.max()  # expected photoelectrons
>>> sensor = SensorEstimate(
...     offset=torch.tensor(100.0),
...     variance=torch.tensor(9.0),
...     gain=torch.tensor(2.0),
... )
>>> raw = simulate_sensor(x, sensor, seed=1)  # ADU
>>> true_sigma = float((x + 9.0 / 2.0**2).mean().sqrt())  # shot + read noise
>>> round(true_sigma, 1)
3.2
```

The two noise ratios, for this OTF and this pixel size:

```pycon
>>> round(noise_ratio((128, 128), pixel, nu_c), 2)  # 1-D, the paper
2.24
>>> round(noise_ratio((128, 128), pixel, nu_c, ratio="area"), 2)  # 2-D
1.42
```

The estimator is given the offset and gain maps through the `SensorEstimate`. It
returns electrons, and the estimated noise level of each frame is in `sigma_`:

```pycon
>>> est = ACsN(na=na, wavelength=wavelength, pixel_size=pixel).fit(init=sensor)
>>> est.is_faithful
True
>>> den = est.apply(raw)
>>> noisy = (raw - 100.0) / 2.0
>>> def psnr(a):
...     return float(10 * torch.log10(x.max() ** 2 / (a - x).square().mean()))
>>> round(psnr(noisy)), round(psnr(den))
(25, 36)
>>> round(float(est.sigma_) / true_sigma, 1)  # the 1-D ratio: about 1.5 too large
1.5
```

With `faithful=False` the exact 2-D ratio is used, and $\sigma_{\mathrm{N}}$ is recovered:

```pycon
>>> area = ACsN(na=na, wavelength=wavelength, pixel_size=pixel, faithful=False)
>>> den_area = area.fit(init=sensor).apply(raw)
>>> bool(abs(float(area.sigma_) / true_sigma - 0.95) < 0.05)
True
>>> round(psnr(den_area))
35
```

Several frames are denoised one by one, each with its own noise level and
score. The frames have the same expected signal here, so their levels are
close. `temporal_psnr` gives the values of the trigger of the video filtering
of the paper, here far above its threshold of 35 dB:

```pycon
>>> from optix.denoise.utils import temporal_psnr
>>> stack = torch.stack([simulate_sensor(x, sensor, seed=s) for s in range(8)])
>>> est = ACsN(na=na, wavelength=wavelength, pixel_size=pixel, faithful=False)
>>> frames = est.fit(init=sensor).apply(stack)
>>> frames.shape, est.sigma_.shape, est.quality_.shape
(torch.Size([8, 128, 128]), torch.Size([8]), torch.Size([8]))
>>> bool(((est.sigma_ / true_sigma - 0.95).abs() < 0.1).all())
True
>>> bool((temporal_psnr(frames) > 35).all())
True
```

The sparse filter is streamed in blocks with a halo. Here a frame of
$128 \times 128$ pixels is cut into blocks of $50 \times 60$ pixels; the result
does not change, up to rounding, with the size of the blocks.

```pycon
>>> sum(stage_halo(s) for s in acsn_profile())
50
>>> whole = acsn(raw, na, wavelength, pixel, offset=100.0, gain=2.0)
>>> parts = acsn(
...     raw, na, wavelength, pixel, offset=100.0, gain=2.0, block=(50, 60)
... )
>>> bool((whole - parts).abs().max() < 1e-3)
True
```

Common variations: `layout=` to say which axes are spatial when the array has
frames or channels, with the offset and gain given as maps per group; a large zarr array in place of the
tensor, with `out=` a separate array; `sigma=` to give a known $\sigma_{\mathrm{N}}$; `hotspot=True` for a camera
with hot pixels. Without `init`, the offset is 0 and the gain 1, for data already in
electrons. A fitted estimator can be saved and loaded like any other.

### Results on synthetic data

Frames of $128 \times 128$ pixels: filaments and beads imaged through the incoherent OTF
(NA 1.4, $\lambda_0 = 0.7$ µm, pixels of 0.1 µm), over a background of one sixth of the signal peak, with a
different offset (about $100 \pm 5$ ADU), gain (about $2 \pm 0.2$ ADU/e⁻) and read noise
(about 3 ADU, with a log-normal spread) at every pixel. The peak is the signal above the
background, in electrons. Each cell is the mean over 5 realisations of the
noise, in float64. $\sigma$ is the root mean square of the true noise of $\mathbf{m}$, the shot
noise of the signal plus the read noise in electrons; PSNR is in dB, with the maximum of the
noise-free frame as the peak; "true $\sigma$" is the filter run with $\sigma_{\mathrm{N}} = \sigma$ given by `sigma=`.

| Peak (e⁻) | $\hat\sigma/\sigma$, `"1d"` | $\hat\sigma/\sigma$, `"area"` | PSNR, noisy | PSNR, `"1d"` | PSNR, `"area"` | PSNR, true $\sigma$ | $Q$, `"1d"` | $Q$, `"area"` |
|---|---|---|---|---|---|---|---|---|
| 10 | 1.50 | 0.95 | 13.5 | 27.9 | 28.5 | 28.5 | −0.55 | −0.52 |
| 30 | 1.52 | 0.97 | 20.6 | 33.5 | 33.2 | 33.3 | −0.18 | −0.07 |
| 100 | 1.52 | 0.96 | 27.2 | 37.5 | 37.3 | 37.4 | 0.10 | 0.18 |

The standard deviation over the realisations is 0.01 to 0.02 for $\hat\sigma/\sigma$ and below 0.6 dB for the PSNR.
The 1-D ratio overestimates the noise by 1.50 to 1.52 on every row, and the 2-D ratio is 3 to 5 % below the true level. The
PSNR of the three variants is within 0.6 dB, and the 1-D ratio gives the lowest
PSNR at the lowest peak and the highest at the others, where more smoothing helps on this
scene. These numbers describe the synthetic frames only. In `float32` and in `float64` the PSNR
agrees to 0.05 dB, and the blocks of the sparse filter give the result of the whole frame to 1e-9 in `float64`.

## References

!!! references "References"

    1. Mandracchia B, Hua X, Guo C, Son J, Urner T, Jia S. *Fast and accurate
       sCMOS noise correction for fluorescence microscopy.* Nat Commun
       **11**:94, 2020.
       [doi:10.1038/s41467-019-13841-8](https://doi.org/10.1038/s41467-019-13841-8)
    2. Dabov K, Foi A, Katkovnik V, Egiazarian K. *Image denoising by sparse
       3-D transform-domain collaborative filtering.* IEEE Trans Image Process
       **16**(8):2080–2095, 2007.
       [doi:10.1109/TIP.2007.901238](https://doi.org/10.1109/TIP.2007.901238)
    3. Maggioni M, Boracchi G, Foi A, Egiazarian K. *Video denoising,
       deblocking, and enhancement through separable 4-D nonlocal spatiotemporal
       transforms.* IEEE Trans Image Process **21**:3952–3966, 2012.
       [doi:10.1109/TIP.2012.2199324](https://doi.org/10.1109/TIP.2012.2199324)
    4. Maggioni M, Sánchez-Monge E, Foi A. *Joint removal of random and
       fixed-pattern noise through spatiotemporal video filtering.* IEEE Trans
       Image Process **23**:4282–4296, 2014.
       [doi:10.1109/TIP.2014.2345261](https://doi.org/10.1109/TIP.2014.2345261)
    5. Kong X, Li K, Yang Q, Wenyin L, Yang M-H. *A new image quality metric for
       image auto-denoising.* Proc IEEE Int Conf Comput Vis, 2888–2895, 2013.
       [doi:10.1109/ICCV.2013.359](https://doi.org/10.1109/ICCV.2013.359)
    6. Wang Z, Bovik AC, Sheikh HR, Simoncelli EP. *Image quality assessment:
       from error visibility to structural similarity.* IEEE Trans Image Process
       **13**(4):600–612, 2004.
       [doi:10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861)

    ??? quote "BibTeX"

        ```bibtex
        @article{mandracchia2020fast,
          author  = {Mandracchia, Biagio and Hua, Xuanwen and Guo, Changliang and Son, Jeonghwan and Urner, Tara and Jia, Shu},
          title   = {Fast and accurate {sCMOS} noise correction for fluorescence microscopy},
          journal = {Nature Communications},
          volume  = {11},
          pages   = {94},
          year    = {2020},
          doi     = {10.1038/s41467-019-13841-8},
        }
        @article{dabov2007image,
          author  = {Dabov, Kostadin and Foi, Alessandro and Katkovnik, Vladimir and Egiazarian, Karen},
          title   = {Image denoising by sparse {3-D} transform-domain collaborative filtering},
          journal = {IEEE Transactions on Image Processing},
          volume  = {16},
          number  = {8},
          pages   = {2080--2095},
          year    = {2007},
          doi     = {10.1109/TIP.2007.901238},
        }
        @article{maggioni2012video,
          author  = {Maggioni, Matteo and Boracchi, Giacomo and Foi, Alessandro and Egiazarian, Karen},
          title   = {Video denoising, deblocking, and enhancement through separable {4-D} nonlocal spatiotemporal transforms},
          journal = {IEEE Transactions on Image Processing},
          volume  = {21},
          pages   = {3952--3966},
          year    = {2012},
          doi     = {10.1109/TIP.2012.2199324},
        }
        @article{maggioni2014joint,
          author  = {Maggioni, Matteo and S{\'a}nchez-Monge, Enrique and Foi, Alessandro},
          title   = {Joint removal of random and fixed-pattern noise through spatiotemporal video filtering},
          journal = {IEEE Transactions on Image Processing},
          volume  = {23},
          pages   = {4282--4296},
          year    = {2014},
          doi     = {10.1109/TIP.2014.2345261},
        }
        @inproceedings{kong2013new,
          author    = {Kong, Xiangwei and Li, Kuan and Yang, Qingxiong and Wenyin, Liu and Yang, Ming-Hsuan},
          title     = {A new image quality metric for image auto-denoising},
          booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
          pages     = {2888--2895},
          year      = {2013},
          doi       = {10.1109/ICCV.2013.359},
        }
        @article{wang2004image,
          author  = {Wang, Zhou and Bovik, Alan C. and Sheikh, Hamid R. and Simoncelli, Eero P.},
          title   = {Image quality assessment: from error visibility to structural similarity},
          journal = {IEEE Transactions on Image Processing},
          volume  = {13},
          number  = {4},
          pages   = {600--612},
          year    = {2004},
          doi     = {10.1109/TIP.2003.819861},
        }
        ```
