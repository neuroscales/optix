# BM3D and BM4D

!!! info "At a glance"
    - **Estimates:** denoised image (`BM3D`) or volume (`BM4D`), same shape as the data
    - **Noise model:** Gaussian (white, known $\sigma$); Poisson–Gaussian through the generalised Anscombe transform
    - **Streaming:** halo
    - **Needs many frames:** no
    - **Reference:** Dabov et al., IEEE Trans Image Process, 2007, [doi:10.1109/TIP.2007.901238](https://doi.org/10.1109/TIP.2007.901238) (BM3D); Maggioni et al., IEEE Trans Image Process, 2013, [doi:10.1109/TIP.2012.2210725](https://doi.org/10.1109/TIP.2012.2210725) (BM4D)

## Overview

`BM3D` removes noise from images, and `BM4D` from volumes, by block matching and
collaborative filtering ([Dabov et al. 2007](https://doi.org/10.1109/TIP.2007.901238),
[Maggioni et al. 2013](https://doi.org/10.1109/TIP.2012.2210725)). Every small
block of the image is compared with the blocks around it. The most similar ones are
stacked into a group, the group is transformed, its small coefficients are
shrunk, and the filtered blocks are returned to their places, where the many
overlapping estimates of each pixel are averaged. The method is *nonlocal*: it
exploits the similar structures that recur in an image (edges, filaments, cells
of one kind), not only the smoothness of a pixel's neighbourhood. It runs
twice: a first estimate by hard thresholding guides the second, a Wiener filter.
`BM4D` is the same method on cubes of voxels, grouped into four-dimensional
arrays.

Use it when the noise is white and Gaussian with a known standard deviation, or
when the data are Poisson–Gaussian (a camera) and a [`SensorEstimate`](../../reference/estimates.md)
is available: the data are then stabilised by the
[generalised Anscombe transform](../sensor/generalized-anscombe.md), denoised
with $\sigma = 1$ and transformed back by its exact unbiased inverse. It is not
a deconvolution: it does not know about the optics, and it keeps blur as it
finds it (see [Richardson–Lucy](../deconv/richardson-lucy.md) and
[Wiener](../deconv/wiener.md) for that). It is slow compared with linear
filters (seconds for a $128^2$ image on one CPU thread), and the noise must be
uncorrelated between pixels: stripes or other structured noise are matched as
if they were signal (see [VSNR](../destripe/vsnr.md) for those).

The estimators are [`BM3D`][optix.denoise.BM3D] and [`BM4D`][optix.denoise.BM4D],
with the functions `bm3d` and `bm4d` as shortcuts for Gaussian noise (see the
[API reference](../../reference/denoise.md)). `fit` fixes the noise model and
the intensity range, `apply` denoises, and large arrays are streamed in blocks
with a halo, giving the result of the whole-image computation.

## Image formation (optics)

Only the noise is modelled; the optics are not (see
[Image formation and noise](../../concepts/image-formation.md)). The recorded
image is the latent image plus white Gaussian noise:

=== "optix"

    $$
    \mathbf{y} = \mathbf{x} + \boldsymbol{\varepsilon},
    \qquad
    \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
    $$

=== "Dabov et al. 2007"

    $$
    z(x) = y(x) + \eta(x), \quad x \in X,
    \qquad \eta(\cdot) \sim \mathcal{N}(0, \sigma^2)\ \text{i.i.d.}
    $$

=== "Maggioni et al. 2013"

    $$
    z(x) = y(x) + \eta(x), \quad x \in X \subset \mathbb{Z}^3,
    \qquad \eta(\cdot) \sim \mathcal{N}(0, \sigma^2)\ \text{i.i.d.}
    $$

| optix | Dabov, Maggioni | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}$ | $z$ | noisy image or volume | $\mathbb{R}^{\lvert\Omega\rvert}$, per image or volume |
| $\mathbf{x}$ | $y$ | latent (noise-free) image or volume | $\mathbb{R}^{\lvert\Omega\rvert}$, per image or volume |
| $\boldsymbol{\varepsilon}$ | $\eta$ | white Gaussian noise | $\mathbb{R}^{\lvert\Omega\rvert}$, per image or volume |
| $\sigma$ | $\sigma$ | noise standard deviation, in the units of the data | scalar, one per call (one per fit) |
| $\Omega$ | $X$ | pixel (voxel) grid, in 2D for `BM3D`, in 3D for `BM4D` | $\lvert\Omega\rvert = HW$ or $DHW$ |

Both papers call the noisy image $z$ and the noise-free one $y$: the reverse of
optix, where $\mathbf{y}$ is what the camera recorded. Images of a stack, or of
different groups, are denoised one by one, each with the same $\sigma$.

Poisson–Gaussian data, the camera model $y = g\,e + o + \varepsilon$ with
$e \sim \mathrm{Poisson}(\lambda)$, are brought to this model by the
generalised Anscombe transform $f$ of
[Mäkitalo & Foi (2013)](https://doi.org/10.1109/TIP.2012.2202675)
(see the [page of the transform](../sensor/generalized-anscombe.md) for $f$ and
its inverse): the noise of $f(\mathbf{y})$ is approximately $\mathcal{N}(0, 1)$,
whatever the signal, so the data are denoised with $\sigma = 1$.

## Probabilistic model

BM3D and BM4D are not derived from a probabilistic model: they are filters,
built on two assumptions. First, the noise is white and Gaussian with a known
$\sigma$ (above). Second, *nonlocal sparsity*: a block of the latent image has
several similar blocks elsewhere, and a group of similar blocks has a sparse
spectrum under a separable transform of the blocks followed by a 1-D transform
across the group. The transforms are orthonormal in the stage that needs it
(the Haar transform across the group, the DCT of the Wiener stage; see the
[Inference](#inference-derivation) section), so white noise of variance
$\sigma^2$ stays white with variance $\sigma^2$ in the spectrum, and the large
coefficients are signal.

Notation, with the symbols of the [notation page](../../concepts/notation.md#block-matching-denoising)
and the papers' own:

- A *block* ($N \times N$ pixels in 2D, a *cube* of $N \times N \times N$ voxels
  in 3D) is a vector $\mathbf{p}_{\mathbf{r}} = \mathbf{P}_{\mathbf{r}}\,\mathbf{y} \in \mathbb{R}^{K}$,
  $K = N^2$ ($N^3$), cut from the image at position $\mathbf{r}$ (its first
  corner) by the operator $\mathbf{P}_{\mathbf{r}}$.
- A *group* is a matrix $\mathbf{G} \in \mathbb{R}^{m \times K}$ whose rows are the $m$
  blocks matched to a *reference* block at $\mathbf{r}$, at positions $S_{\mathbf{r}}$.
- The *spectrum* of a group is $\boldsymbol{\Theta} = \mathbf{H}_m\,\mathbf{G}\,\mathbf{T}^\top$:
  the block transform $\mathbf{T} \in \mathbb{R}^{K \times K}$ (separable: the same
  1-D transform along every axis of the block) applied to each block, then the
  Haar transform $\mathbf{H}_m \in \mathbb{R}^{m \times m}$ across the group. Its coefficient
  $\Theta_{00}$ is the DC, proportional to the mean of the group.

The estimate of a group is a shrinkage of its spectrum, either a *hard threshold*
or an *empirical Wiener* gain, and the estimate of the image is a weighted average
of the estimated blocks.

=== "optix"

    $$
    \hat{\boldsymbol{\Theta}} = \mathbf{M} \odot \boldsymbol{\Theta},
    \qquad
    \hat{\mathbf{x}} = \frac{\mathbf{n}}{\mathbf{d}},
    \quad
    \mathbf{n} = \sum_{\mathbf{r}} w_{\mathbf{r}} \sum_{\mathbf{r}' \in S_{\mathbf{r}}}
    \mathbf{P}_{\mathbf{r}'}^\top (\boldsymbol{\omega} \odot \hat{\mathbf{p}}^{\mathbf{r}}_{\mathbf{r}'}),
    \quad
    \mathbf{d} = \sum_{\mathbf{r}} w_{\mathbf{r}} \sum_{\mathbf{r}' \in S_{\mathbf{r}}}
    \mathbf{P}_{\mathbf{r}'}^\top \boldsymbol{\omega}
    $$

=== "Dabov et al. 2007"

    $$
    \hat{\mathbf{Y}}_{S_{x_R}} = \mathcal{T}_{3\mathrm{D}}^{-1}\bigl(\Upsilon(\mathcal{T}_{3\mathrm{D}}(\mathbf{Z}_{S_{x_R}}))\bigr),
    \qquad
    \hat{y}(x) = \frac{\sum_{x_R} \sum_{x_m \in S_{x_R}} w_{x_R}\, \hat{Y}^{x_R}_{x_m}(x)}
    {\sum_{x_R} \sum_{x_m \in S_{x_R}} w_{x_R}\, \chi_{x_m}(x)}
    $$

=== "Maggioni et al. 2013"

    $$
    \hat{\mathbf{G}}^y_{S_{x_R}} = \mathcal{T}_{4D}^{-1}\bigl(\Upsilon(\mathcal{T}_{4D}(\mathbf{G}^z_{S_{x_R}}))\bigr),
    \qquad
    \hat{y}(x) = \frac{\sum_{x_R} \sum_{x_i \in S_{x_R}} w_{x_R}\, \hat{C}^y_{x_i}(x)}
    {\sum_{x_R} \sum_{x_i \in S_{x_R}} w_{x_R}\, \chi_{x_i}(x)}
    $$

Here $\mathbf{M}$ is a mask (hard thresholding) or a gain (Wiener filtering),
both defined below, $w_{\mathbf{r}}$ is a weight per group, $\boldsymbol{\omega}$ a
window on the block (equal to $\mathbf{1}$ for `BM4D`), and
$\hat{\mathbf{p}}^{\mathbf{r}}_{\mathbf{r}'}$ the estimate, from the group of
reference $\mathbf{r}$, of the block at $\mathbf{r}'$. The division is pixel-wise,
and $\mathbf{P}^\top$ puts a block back in place, zero elsewhere (the papers'
zero-padding and characteristic function $\chi$).

| optix | Dabov et al. 2007 | Maggioni et al. 2013 | Meaning | Shape |
|---|---|---|---|---|
| $\mathbf{r}$, $\mathbf{r}'$ | $x_R$, $x_m$ | $x_R$, $x_i$ | position of a reference block; of a block of its group | one per spatial axis |
| $\mathbf{p}_{\mathbf{r}}$ | $Z_x$ | $C^z_{x}$ | noisy block (cube) at $\mathbf{r}$ | $\mathbb{R}^{K}$, $K = N^2$ ($N^3$) |
| $N$ | $N_1$ ($N_1^{\mathrm{ht}}$, $N_1^{\mathrm{wie}}$) | $L$ | side of the block (cube) | scalar, one per stage |
| $M$ | $N_2$ | $M$ | largest group size | scalar, one per stage |
| $m$ | $\lvert S_{x_R}\rvert$ | $\lvert S_{x_R}\rvert$ | group size: the number of matched blocks, capped at $M$ and rounded down to a power of two | scalar, per reference |
| $N_{\text{step}}$ | $N_{\mathrm{step}}$ | $N_{\mathrm{step}}$ | step between reference blocks | scalar, one per stage |
| $N_S$ | $N_S$ | $N_S$ | side of the (cubic) search window | scalar, one per stage |
| $\tau$ | $\tau_{\mathrm{match}}$ | $\tau_{\mathrm{match}}$ | matching threshold on the distance $d$ | scalar, one per stage |
| $S_{\mathbf{r}}$ | $S_{x_R}$ | $S_{x_R}$ | positions of the blocks matched to $\mathbf{r}$ | $m$ positions |
| $\mathbf{G}$ | $\mathbf{Z}_{S_{x_R}}$ | $\mathbf{G}^z_{S_{x_R}}$ | noisy group | $\mathbb{R}^{m \times K}$, per reference |
| $\mathbf{T}$ | $\mathcal{T}_{\mathrm{2D}}$ | the spatial part of $\mathcal{T}_{4D}$ | block transform (bior1.5 or DCT) | $\mathbb{R}^{K \times K}$, one per stage |
| $\mathbf{H}_m$ | $\mathcal{T}_{\mathrm{1D}}$ | the Haar part of $\mathcal{T}_{4D}$ | 1-D Haar transform across the group | $\mathbb{R}^{m \times m}$ |
| $\boldsymbol{\Theta}$ | $\mathcal{T}_{\mathrm{3D}}(\mathbf{Z}_{S_{x_R}})$ | $\mathcal{T}_{4D}(\mathbf{G}^z_{S_{x_R}})$ | spectrum of a group, DC at $\Theta_{00}$ | $\mathbb{R}^{m \times K}$, per reference |
| $\lambda_{\mathrm{ht}}$ | $\lambda_{\mathrm{3D}}$ | $\lambda_{\mathrm{4D}}$ | hard threshold, in units of $\sigma$ | scalar, one per profile |
| $\lambda_{\mathrm{pre}}$ | $\lambda_{\mathrm{2D}}$ | none | threshold of the pre-filter of the matching, in units of $\sigma$ | scalar, one per profile |
| $\boldsymbol{\omega}$, $\beta$ | Kaiser window, $\beta$ | none | window of the aggregation and its parameter | $\mathbb{R}^{K}$ |
| $w_{\mathbf{r}}$ | $w_{x_R}$ | $w_{x_R}$ | weight of the group of $\mathbf{r}$ | scalar, per reference |

## Inference / derivation

Both methods run the same three steps twice: grouping by block matching,
collaborative filtering of the group, aggregation. The first run thresholds the
spectrum of the noisy groups, and the second runs a Wiener filter whose gain is
estimated from the first run. In statistical terms, hard thresholding is
$\ell_0$-penalised least squares in an orthonormal basis (it solves
$\min_{\boldsymbol\theta} \lVert\boldsymbol\theta - \boldsymbol\Theta\rVert^2 + \lambda^2\sigma^2 \lVert\boldsymbol\theta\rVert_0$);
the empirical Wiener filter is the linear minimum-mean-squared-error estimator
with the signal power of the first estimate plugged in; and the aggregation is an
inverse-variance weighted mean of overlapping estimates.

### Grouping by block matching

Reference blocks are visited on a grid of step $N_{\text{step}}$. For each
reference at $\mathbf{r}$, the blocks at positions $\mathbf{r}'$ of a search window of
$N_S \times N_S$ ($N_S^3$) positions around it are compared by the normalised
squared distance between their spectra, and those within $\tau$ are kept.

=== "optix"

    $$
    d(\mathbf{r}, \mathbf{r}') = \frac{1}{K}
    \bigl\lVert \Upsilon'(\mathbf{T}\mathbf{p}_{\mathbf{r}}) - \Upsilon'(\mathbf{T}\mathbf{p}_{\mathbf{r}'}) \bigr\rVert_2^2,
    \qquad
    S_{\mathbf{r}} = \bigl\{\mathbf{r}' : d(\mathbf{r}, \mathbf{r}') \le \tau \bigr\}
    $$

=== "Dabov et al. 2007"

    $$
    d(Z_{x_R}, Z_x) = \frac{\bigl\lVert \Upsilon'(\mathcal{T}^{\mathrm{ht}}_{\mathrm{2D}}(Z_{x_R})) - \Upsilon'(\mathcal{T}^{\mathrm{ht}}_{\mathrm{2D}}(Z_x)) \bigr\rVert_2^2}{(N_1^{\mathrm{ht}})^2},
    \qquad
    S^{\mathrm{ht}}_{x_R} = \{x \in X : d(Z_{x_R}, Z_x) \le \tau^{\mathrm{ht}}_{\mathrm{match}}\}
    $$

=== "Maggioni et al. 2013"

    $$
    d(C^z_{x_i}, C^z_{x_j}) = \frac{\lVert C^z_{x_i} - C^z_{x_j}\rVert_2^2}{L^3},
    \qquad
    S^z_{x_R} = \{x_i \in X : d(C^z_{x_R}, C^z_{x_i}) \le \tau^{\mathrm{ht}}_{\mathrm{match}}\}
    $$

($\Upsilon'$ hard-thresholds at $\lambda_{\mathrm{pre}}\sigma$: [Dabov et al. (2007, Eqs. 4–5)](https://doi.org/10.1109/TIP.2007.901238);
[Maggioni et al. (2013, Eqs. 2–3)](https://doi.org/10.1109/TIP.2012.2210725), without pre-filter.) The
pre-filter makes the distance robust to noise at high noise levels: for the noisy distance, the
expected value is the ideal distance plus $2\sigma^2$, and its variance grows
as $\sigma^4$ ([Dabov et al. 2007, Eqs. 2–3](https://doi.org/10.1109/TIP.2007.901238)).
The reference is always in its own group ($d = 0$). The group size is rounded
down to a power of two, as the Haar transform needs
([Dabov et al. 2007, §IV-B](https://doi.org/10.1109/TIP.2007.901238);
[Maggioni et al. 2013, §III-A](https://doi.org/10.1109/TIP.2012.2210725)), and capped at $M$. In the second
run, the distance is taken between the blocks of the *first estimate*
$\hat{\mathbf{x}}^{\mathrm{b}}$, without pre-filter, and the test is strict,
$d < \tau$
([Dabov et al. 2007, Eq. 7](https://doi.org/10.1109/TIP.2007.901238);
[Maggioni et al. 2013, Eq. 9](https://doi.org/10.1109/TIP.2012.2210725)).
It gives the positions $S_{\mathbf{r}}$ of two groups of the same blocks: the noisy
$\mathbf{G}$ and the pilot $\mathbf{G}^{\mathrm{b}}$ cut from $\hat{\mathbf{x}}^{\mathrm{b}}$.

### Collaborative filtering

**Hard thresholding** (first run) keeps the coefficients larger than
$\lambda_{\mathrm{ht}}\sigma$ and zeroes the others; the weight of the group is
the inverse of the variance that remains in it, proportional to the number
$N_{\mathrm{har}}$ of retained coefficients:

=== "optix"

    $$
    \hat{\boldsymbol{\Theta}} = \boldsymbol{\Theta} \odot \mathbb{1}\bigl[\lvert\boldsymbol{\Theta}\rvert > \lambda_{\mathrm{ht}}\sigma\bigr],
    \qquad
    w_{\mathbf{r}} =
    \begin{cases}
    \dfrac{1}{\sigma^2 N_{\mathrm{har}}} & N_{\mathrm{har}} \ge 1 \\[1ex]
    1 & \text{otherwise}
    \end{cases}
    $$

=== "Dabov et al. 2007"

    $$
    \Upsilon(c) = c\,\mathbb{1}[\lvert c\rvert > \lambda_{\mathrm{3D}}\sigma],
    \qquad
    w^{\mathrm{ht}}_{x_R} =
    \begin{cases}
    \dfrac{1}{\sigma^2 N^{x_R}_{\mathrm{har}}} & N^{x_R}_{\mathrm{har}} \ge 1 \\[1ex]
    1 & \text{otherwise}
    \end{cases}
    $$

=== "Maggioni et al. 2013"

    $$
    \Upsilon(c) = c\,\mathbb{1}[\lvert c\rvert > \lambda_{4D}\sigma]\ \ (\text{DC kept}),
    \qquad
    w^{\mathrm{ht}}_{x_R} = \frac{1}{\sigma^2 N^{\mathrm{ht}}_{x_R}}
    $$

([Dabov et al. 2007, Eqs. 6 and 10](https://doi.org/10.1109/TIP.2007.901238);
[Maggioni et al. 2013, Eqs. 5–8](https://doi.org/10.1109/TIP.2012.2210725).)
$N_{\mathrm{har}}$ counts the non-zero coefficients of $\hat{\boldsymbol\Theta}$.
In `BM4D` the DC is never thresholded, so $N_{\mathrm{har}} \ge 1$ and the
second case does not occur. The filtered spectrum is transformed back,
$\hat{\mathbf{G}} = \mathbf{H}_m^\top \hat{\boldsymbol\Theta}\,\mathbf{T}^{-\top}$, to the $m$
block estimates $\hat{\mathbf{p}}$ (the inverse of $\mathbf{T}$ is computed by matrix
inversion: bior1.5 is biorthogonal, not orthogonal).

**Empirical Wiener filtering** (second run) shrinks the spectrum of the noisy
group by a gain computed from the spectrum $\boldsymbol\Theta^{\mathrm{b}}$ of the pilot
group, taken as the true signal:

=== "optix"

    $$
    \mathbf{W} = \frac{(\boldsymbol{\Theta}^{\mathrm{b}})^{2}}{(\boldsymbol{\Theta}^{\mathrm{b}})^{2} + \sigma^2},
    \qquad
    \hat{\boldsymbol{\Theta}} = \mathbf{W} \odot \boldsymbol{\Theta},
    \qquad
    w_{\mathbf{r}} = \frac{1}{\sigma^2 \lVert \mathbf{W} \rVert_2^2}
    $$

=== "Dabov et al. 2007"

    $$
    \mathbf{W}_{S_{x_R}} = \frac{\lvert\mathcal{T}^{\mathrm{wie}}_{3\mathrm{D}}(\hat{\mathbf{Y}}^{\mathrm{basic}}_{S_{x_R}})\rvert^2}
    {\lvert\mathcal{T}^{\mathrm{wie}}_{3\mathrm{D}}(\hat{\mathbf{Y}}^{\mathrm{basic}}_{S_{x_R}})\rvert^2 + \sigma^2},
    \qquad
    w^{\mathrm{wie}}_{x_R} = \sigma^{-2} \lVert \mathbf{W}_{S_{x_R}} \rVert_2^{-2}
    $$

=== "Maggioni et al. 2013"

    $$
    \mathbf{W}_{S^{\hat y^{\mathrm{ht}}}_{x_R}} = \frac{\lvert\mathcal{T}^{\mathrm{wie}}_{4D}(\mathbf{G}^{\hat y^{\mathrm{ht}}}_{S})\rvert^2}
    {\lvert\mathcal{T}^{\mathrm{wie}}_{4D}(\mathbf{G}^{\hat y^{\mathrm{ht}}}_{S})\rvert^2 + \sigma^2},
    \qquad
    w^{\mathrm{wie}}_{x_R} = \sigma^{-2} \lVert \mathbf{W} \rVert_2^{-2}
    $$

([Dabov et al. 2007, Eqs. 8, 9 and 11](https://doi.org/10.1109/TIP.2007.901238);
[Maggioni et al. 2013, Eqs. 10–12](https://doi.org/10.1109/TIP.2012.2210725).) The
squares and the product with $\boldsymbol\Theta$ are element-wise.
The pilot is an estimate with much less noise than the data, so its spectrum
is a better guide to which coefficients carry signal than the noisy one, and the
weight is again the inverse of the residual variance of the group,
$\sigma^2 \lVert\mathbf{W}\rVert_2^2$. Where the pilot spectrum is zero, the gain is
zero and the coefficient is removed.

### Aggregation

Blocks overlap, so each pixel has several estimates. They are averaged with the
group weights $w_{\mathbf{r}}$ and a window $\boldsymbol{\omega}$ over the block
(the equation of the [Probabilistic model](#probabilistic-model) section;
[Dabov et al. 2007, Eq. 12](https://doi.org/10.1109/TIP.2007.901238),
[Maggioni et al. 2013, Eq. 7](https://doi.org/10.1109/TIP.2012.2210725)). In
`BM3D` the window is a Kaiser window of parameter $\beta = 2$, which gives less
weight to the border of a block, where the transform is least reliable
([Dabov et al. 2007, §IV](https://doi.org/10.1109/TIP.2007.901238)); `BM4D` uses
no window ([Maggioni et al. 2013, §II-B](https://doi.org/10.1109/TIP.2012.2210725)). The numerator
$\mathbf{n}$ and the denominator $\mathbf{d}$ are accumulated in two buffers of the size of
the image, and divided at the end. A pixel covered by no block keeps the value
that the run received as input: this occurs only when $N_{\text{step}} > N$.

### Poisson–Gaussian data

With `fit(init=sensor)`, the three parameters $g$, $\sigma$, $o$ of the
[`SensorEstimate`](../../reference/estimates.md) define the generalised
Anscombe transform $f$, pixel by pixel when they are maps (such as the offset
and variance maps of an sCMOS camera), and the estimate is

$$
\hat{\mathbf{x}} = g\,\mathcal{I}_{\sigma/g}\bigl(\operatorname{BM3D}(f(\mathbf{y}),\ \sigma = 1)\bigr) + o,
$$

with $\mathcal{I}_{\sigma/g}$ the exact unbiased inverse of
[Mäkitalo & Foi (2013)](https://doi.org/10.1109/TIP.2012.2202675), the
default; the keyword `inverse` selects the closed-form, asymptotic or algebraic
inverse instead. This chaining follows the approach of
[Maggioni et al. (2013, Eq. 14)](https://doi.org/10.1109/TIP.2012.2210725) for Rician
data, with the transform of Poisson–Gaussian noise. The papers on BM3D and BM4D
do not treat Poisson–Gaussian noise: this chaining is an application of the
transform, not part of either method. Since the stabilised noise has unit
variance, `sigma` is not needed and the profile switch of `BM3D` ($\sigma > 40$)
is decided after the scaling to the range $[0, 255]$ (see below).

## Algorithm

**Input:** noisy images $\mathbf{y} \in \mathbb{R}^{\lvert\Omega\rvert}$ (volumes for `BM4D`);
noise standard deviation $\sigma$ (or a `SensorEstimate`); `data_range`; `profile = "normal"` \
**Output:** denoised images $\hat{\mathbf{x}} \in \mathbb{R}^{\lvert\Omega\rvert}$ (the first estimate
$\hat{\mathbf{x}}^{\mathrm{b}}$ with `output="basic"`).

1. **Scale** the data to the range of the paper: $\tilde{\mathbf{y}} = s\,\mathbf{y}$,
   $\tilde\sigma = s\,\sigma$, with $s = 255 / \texttt{data\_range}$ (`BM3D`) or
   $s = 1 / \texttt{data\_range}$ (`BM4D`). With a `SensorEstimate`,
   first replace $\mathbf{y}$ by $f(\mathbf{y})$ and $\sigma$ by $1$.
2. **Select the parameters** of the stage from the profile
   (Tables 1 and 2 below; `BM3D` switches on $\tilde\sigma > 40$).
3. **First estimate.** For each reference position $\mathbf{r}$ on the grid of step
   $N_{\text{step}}$:
    1. **group**: compute $d(\mathbf{r}, \mathbf{r}')$ over the search window,
       keep the $m$ closest blocks with $d \le \tau$, the reference first,
       with $m$ the largest power of two not above $\min(M, \lvert S_{\mathbf{r}}\rvert)$;
    2. **transform** the group, $\boldsymbol\Theta = \mathbf{H}_m \mathbf{G}\mathbf{T}^\top$, with bior1.5
       as the block transform;
    3. **hard-threshold** at $\lambda_{\mathrm{ht}}\tilde\sigma$, and set $w_{\mathbf{r}}$;
    4. **invert** the transforms and **accumulate** the weighted, windowed block
       estimates in $\mathbf{n}$ and $\mathbf{d}$.

    Then $\hat{\mathbf{x}}^{\mathrm{b}} = \mathbf{n}/\mathbf{d}$.
4. **Final estimate** (unless `output="basic"`). For each reference position:
    1. **group** with $d < \tau$ computed between the blocks of
       $\hat{\mathbf{x}}^{\mathrm{b}}$; cut the noisy group $\mathbf{G}$ and the pilot group
       $\mathbf{G}^{\mathrm{b}}$ at these positions;
    2. **transform** both, with the DCT as the block transform;
    3. **Wiener-filter** $\boldsymbol\Theta$ with the gain $\mathbf{W}$ of the pilot, and set $w_{\mathbf{r}}$;
    4. **invert** and **accumulate** as above.

    Then $\hat{\mathbf{x}} = \mathbf{n}/\mathbf{d}$.
5. **Unscale**: divide by $s$. With a `SensorEstimate`, apply the inverse of $f$.

### Profiles

The default parameters are the *Normal Profile* of Table I of each paper. The
sizes are in pixels (voxels), $\tilde\sigma$ is the noise level in the units of
the paper ($[0, 255]$ for `BM3D`, $[0, 1]$ for `BM4D`).

**Table 1: BM3D** ([Dabov et al. 2007, Table I](https://doi.org/10.1109/TIP.2007.901238), Normal Profile)

| Run | $\tilde\sigma$ | $\mathbf{T}$ | $N$ | $M$ | $N_{\text{step}}$ | $N_S$ | $\beta$ | $\lambda_{\mathrm{pre}}$ | $\lambda_{\mathrm{ht}}$ | $\tau$ |
|---|---|---|---|---|---|---|---|---|---|---|
| first (hard thresholding) | $\le 40$ | bior1.5 | 8 | 16 | 3 | 39 | 2 | 0 | 2.7 | 2500 |
| second (Wiener) | $\le 40$ | DCT | 8 | 32 | 3 | 39 | 2 | | | 400 |
| first (hard thresholding) | $> 40$ | DCT | 12 | 16 | 4 | 39 | 2 | 2 | 2.8 | 5000 |
| second (Wiener) | $> 40$ | DCT | 11 | 32 | 6 | 39 | 2 | | | 3500 |

**Table 2: BM4D** ([Maggioni et al. 2013, Table I](https://doi.org/10.1109/TIP.2012.2210725); data in $[0, 1]$)

| Run | Profile | $N$ | $M$ | $N_{\text{step}}$ | $N_S$ | $\lambda_{\mathrm{ht}}$ | $\tau$ (as printed) |
|---|---|---|---|---|---|---|---|
| first (hard thresholding) | normal | 4 | 16 | 3 | 11 | 2.7 | 2.9 |
| second (Wiener) | normal | 4 | 32 | 3 | 11 | | 0.4 |
| first (hard thresholding) | modified | 4 | 32 | 3 | 11 | 2.8 | 24.6 |
| second (Wiener) | modified | 5 | 32 | 3 | 11 | | 6.7 |

The block transform of `BM4D` is a 3-D bior1.5 (first run) or 3-D DCT (second
run), and the transform across the group is the Haar transform. The "modified"
profile is the one that Maggioni et al. use in their experiments. The
printed $\tau$ is applied as explained in the
[next section](#faithful-mode-and-extensions) (choice 7). The paper also
defines a *Fast Profile* for `BM3D` ($N_{\mathrm{step}} = 6$, with a predictive
search); it is not implemented.

## Faithful mode and extensions

`faithful=True` (the default) reproduces the Normal Profile of Table I of the
paper: the exhaustive search ($N_{FS} = 1$), the two runs, the weights of
Eqs. 10 and 11 for `BM3D` and 8 and 12 for `BM4D`, with the group sizes rounded
down to powers of two. The papers leave a number of choices open, and the
code makes them as follows.

### Choices not in the papers

1. **Data scaling and `data_range`.** The parameters of Table I are given for
   images with values in $[0, 255]$ (`BM3D`) or $[0, 1]$ (`BM4D`), and they
   include thresholds on the distance and on $\sigma$. The data and $\sigma$ are
   therefore multiplied by $255/\texttt{data\_range}$ ($1/\texttt{data\_range}$), and the result divided by
   it. The default `data_range` is an estimate of the range of the clean
   signal ([`clean_range`][optix.denoise.utils.clean_range]). The extremes of
   noisy data overshoot those of the signal by about $\sqrt{2\ln n}\,\sigma$
   each, for $n$ samples, so the estimate is the range of the data minus
   $2\sqrt{2\ln n}\,\sigma$, and at least half the range of the data:
   $\hat r = \max\bigl(\max\mathbf{y} - \min\mathbf{y} - 2\sqrt{2\ln n}\,\sigma,\ (\max\mathbf{y} - \min\mathbf{y})/2\bigr)$.
   This is not in the papers, whose images have a known range. The range of the
   noisy data itself would keep $\tilde\sigma$ below about 39, since the noise
   alone spans 6 to 9 $\sigma$, so that the $\tilde\sigma > 40$ rows of Table 1
   would not be reached. With a `SensorEstimate` the estimate is that of the
   stabilised data, with $\sigma = 1$. Give the range of the clean signal when
   it is known. On a $160^2$ synthetic scene of range 255, a `data_range` twice
   too large changes the PSNR by at most 0.2 dB for $\sigma$ from 20 to 80; one
   40 % too small (150 for 255) loses up to 0.3 dB for $\sigma \le 50$,
   and 1.3 dB at $\sigma = 80$.
2. **The bior1.5 transform.** The paper names the wavelet; the transform of a
   block of $N$ samples is built here as the full dyadic decomposition with
   periodic extension, iterated on the approximation down to one coefficient
   (the DC first), with the analysis low-pass filter
   $\tfrac{\sqrt2}{256}[3, -3, -22, 22, 128, 128, 22, -22, -3, 3]$ and the Haar
   high-pass filter. Each row is scaled to unit $\ell^2$ norm, so that white noise of
   variance $\sigma^2$ gives coefficients of variance $\sigma^2$.
   For $N = 4$ it is the orthonormal Haar transform, as
   [Maggioni et al. (2013, §III-A)](https://doi.org/10.1109/TIP.2012.2210725) state.
   The paper mentions "periodized wavelets" (§IV) and "full dyadic"
   decompositions (§IV-B). At $N = 8$, the block size of the paper, the
   periodic extension cancels the outer taps: taps $-4$ and $4$ ($+3$ and $-3$)
   fall on the same sample, and so do taps $-3$ and $5$. The effective low-pass
   filter is the 6-tap $\tfrac{\sqrt2}{256}[-22, 22, 128, 128, 22, -22]$, and the
   two coarser levels are Haar. A symmetric extension would keep the
   $\pm 3/256$ taps; Table II of Dabov et al. suggests that this matters little
   (29.42 dB with bior1.3 against 29.43 dB with bior1.5).
   The matching distance of the first run is taken between these
   coefficients, the literal Eq. 4 (and footnote 3 of Dabov et al.). As
   $\mathbf{T}$ is not orthogonal, $\tau = 2500$ then applies to
   $\lVert\mathbf{T}(\mathbf{p}_{\mathbf{r}} - \mathbf{p}_{\mathbf{r}'})\rVert^2/K$,
   not to the mean squared difference of the pixels.
3. **The reference grid.** References sit on a global grid, at the multiples of
   $N_{\text{step}}$ from the first pixel, plus the last position along each axis, so that
   every pixel is covered. Streamed blocks use the grid of the whole image.
4. **Search windows and ties.** The $N_S \times N_S$ ($N_S^3$) window is
   centred on the reference (offsets from $-\lfloor (N_S - 1)/2 \rfloor$ to
   $\lfloor N_S/2 \rfloor$) and clipped at the border of the image; only
   positions inside the image are candidates. Blocks at equal distance keep the
   raster order of the window, and the reference comes first.
5. **The Kaiser window of `BM3D`.** The papers prescribe a 2-D Kaiser
   window of parameter $\beta$, not its shape: it is the outer product of two
   1-D Kaiser windows of $N$ samples,
   $\omega(i_1, i_2) = \kappa_\beta(i_1)\,\kappa_\beta(i_2)$ with
   $\kappa_\beta(i) = I_0\bigl(\beta\sqrt{1 - (2i/(N-1) - 1)^2}\bigr) / I_0(\beta)$.
6. **The $\lVert\mathbf{W}\rVert^2$ floor.** The weight of Eq. 11 divides by
   $\lVert\mathbf{W}\rVert_2^2$, which is zero when the pilot group is zero.
   The norm is clamped at the smallest positive floating-point number.
7. **The $\tau$ of `BM4D`.** Read literally, the values of $\tau_{\mathrm{match}}$ of
   Table I (2.9, 0.4, 24.6, 6.7) are larger than any
   distance of Eq. 2 between cubes of values in $[0, 1]$, where $d \le 1$, and so
   every cube would match. They are applied to the sum of squared differences
   $\lVert C_{x_i} - C_{x_j}\rVert_2^2$, that is $d \le \tau / L^3$. Maggioni et al. reuse
   "most of the settings originally optimized for BM3D" (§III-A), and in 8-bit
   units, $\tau \cdot 255^2 / L^3$, this reading gives the thresholds of `BM3D`
   (Table 1):

    | Table 2 | $L$ | $\tau \cdot 255^2 / L^3$ | `BM3D` |
    |---|---|---|---|
    | 0.4 (Wiener, normal) | 4 | 406 | 400 (second run, $\tilde\sigma \le 40$) |
    | 6.7 (Wiener, modified) | 5 | 3485 | 3500 (second run, $\tilde\sigma > 40$) |
    | 2.9 (hard thresholding, normal) | 4 | 2946 | 2500 (first run, $\tilde\sigma \le 40$) |
    | 24.6 (hard thresholding, modified) | 4 | 24994 | none |

    The two Wiener values, $0.4/64$ and $6.7/125$, are the $400/255^2$ and
    $3500/255^2$ of `BM3D` to rounding, each with the $L$ of its profile.
8. **The DC coefficient in `BM3D`.** `BM4D` never thresholds the DC
   ([Maggioni et al. 2013, §II-B](https://doi.org/10.1109/TIP.2012.2210725)). The
   `BM3D` paper does not say, and its Eq. 10, with the case $N_{\mathrm{har}} = 0$ (all coefficients,
   the DC included, thresholded), implies that the DC may be: it is thresholded
   like the others. `BM3D` is therefore not invariant to an offset of the
   intensities. The DC of a group of $m$ blocks, $\sqrt{mK}$ times its mean, is
   removed when the mean is below $\lambda_{\mathrm{ht}}\tilde\sigma/\sqrt{mK}$ in
   absolute value (about $0.08\,\tilde\sigma$ for $m = 16$), which occurs on a
   background close to 0, as in dark-subtracted microscopy data. On a filament
   phantom with a background of 0 ($96^2$ pixels, range $[0, 255]$,
   $\sigma = 20$, `search=15`), denoising $\mathbf{y}$ and $\mathbf{y} + 100$
   differs by up to 22 grey levels in the first estimate and 25 in the final
   one. The weight 1 of a group with $N_{\mathrm{har}} = 0$ is not scale-free,
   unlike $1/(\sigma^2 N_{\mathrm{har}})$: at $\sigma = 20$ in 8-bit units it is at
   least 400 times any other weight, so such a group, whose estimate is 0,
   dominates the pixels it covers. The extension `keep_dc` keeps the DC, as
   `BM4D` does: the first estimate is then invariant to an offset, and the final
   one nearly so (2.5 grey levels in the case above), since the Wiener gain of
   the DC still depends on the mean.

### Extensions (opt-in, not faithful)

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `search` | side $N_S$ of the search window, in both runs | `None` ($N_S = 39$ for `BM3D`, 11 for `BM4D`) | `None` |
| `step` | step $N_{\text{step}}$ between reference blocks, in both runs | `None` (3, or 4 and 6 for `BM3D` with $\tilde\sigma > 40$) | `None` |
| `keep_dc` (`BM3D` only) | never thresholds the DC of a group in the first run, as `BM4D` does (choice 8) | `False` | `False` |

`search` and `step` trade quality for speed: the cost per pixel and run is about
$N_S^2 N^2 / N_{\text{step}}^2$ operations. Giving any of the three sets
`is_faithful` to false. They are not a "fast profile" of the paper, which also changes the
transforms and uses a predictive search: that profile is not implemented.
Table IV of [Maggioni et al. (2013)](https://doi.org/10.1109/TIP.2012.2210725)
studies the effect of $N_S$ and $N_{\text{step}}$ on `BM4D`.

The other keywords do not change the method:

| Keyword | Changes | Faithful value |
|---|---|---|
| `sigma` | noise standard deviation $\sigma$; omit it to give a `SensorEstimate` to `fit` | `None` |
| `data_range` | intensity range mapped to $[0, 255]$ ($[0, 1]$) | `None` (estimated from the data, choice 1) |
| `profile` | `"normal"`; `BM4D` also has `"modified"` | `"normal"` |
| `output` | `"final"` (second run) or `"basic"` (first run only) | `"final"` |
| `inverse` | inverse of the Anscombe transform, for Poisson–Gaussian data | `"exact"` |
| `block` | core block size per spatial axis of the streaming; the result does not depend on it | `None` (memory budget) |

## Memory and streaming

The tag is `halo`. The result at a pixel depends on the data within a finite
distance: a block covers a pixel if it is within $N - 1$ of it, its group was
formed by a reference within $\lfloor N_S/2 \rfloor$ of the block, and the search
window of that reference reaches $\lfloor N_S/2 \rfloor$ further, so each run reaches
$2\lfloor N_S/2 \rfloor + N - 1$ pixels, and the final estimate adds both runs.
`bm3d_halo` and `bm4d_halo` compute it, and `BM3D.halo()` and `BM4D.halo()` do for a
fitted estimator:

| | first run only | both runs |
|---|---|---|
| `BM3D`, $\tilde\sigma \le 40$ | 45 | 90 |
| `BM3D`, $\tilde\sigma > 40$ | 49 | 97 |
| `BM4D`, normal | 13 | 26 |
| `BM4D`, modified | 13 | 27 |

`apply` reads the data with the [`blocks`](../../reference/io.md) iterator: a
core of `block` pixels per spatial axis (by default sized from the memory budget,
z-slabs for a volume), with this halo around it. Each block is denoised, the core is
kept and written to the output, and the reference grid is the one of the whole
image. The result therefore equals the whole-image computation, up to
floating-point rounding, for any `block`. The data may be a zarr array or a
memory-mapped file larger than memory; the output cannot be the input (an
in-place call is refused, as a halo needs the unchanged neighbours). `fit` reads
the data once, only when `data_range` is not given, to estimate its range. Images of
a stack, and groups, are independent. Maps of a `SensorEstimate` are cropped to
each block.

Within a block, the spectra of all blocks of the image are kept: $N^2$ values per
pixel ($N^3$ per voxel), and twice that in the second run, which also holds the
spectra of the first estimate. The cost per pixel is about
$N_S^2 N^2 / N_{\text{step}}^2$ operations per run
([Dabov et al. 2007, §IV-A](https://doi.org/10.1109/TIP.2007.901238)), linear in the
size of the image; it is the search size $N_S$ that dominates, and the one to
lower first, with `search` and `step` (extensions), to go faster. The memory
budget of [`optix.config`](../../reference/configuration.md) is shared: half for the
spectra (which set the default `block`), and a quarter each for a batch of
block matching and a batch of filtering, which run one after the other. The computation is vectorised over batches of references
with gathers, sorts and scatter-adds, and runs on any torch device. On the CPU
it is deterministic; on CUDA the aggregation uses atomic additions, so the result may differ in the last bits.

## Usage

The first example denoises a synthetic image of filaments with Gaussian noise
of standard deviation 25 on a range of 255, and compares the first and the
final estimate with the truth.

```pycon
>>> import torch
>>> from optix.denoise import BM3D, bm3d
>>> from optix.models import phantoms
>>> _ = torch.manual_seed(0)
>>> x = phantoms.filaments((64, 64), intensity=200.0, seed=0)
>>> x = 255 * (x - x.min()) / (x.max() - x.min())
>>> y = x + 25 * torch.randn_like(x)
>>> def psnr(a):
...     return round(float(10 * torch.log10(255**2 / (a - x).square().mean())), 1)
>>> psnr(y)
20.2
>>> psnr(bm3d(y, sigma=25, data_range=255, output="basic"))
28.8
>>> psnr(bm3d(y, sigma=25, data_range=255))
29.5
```

The estimator form streams the image in blocks with a halo, and gives the
whole-image result: exactly in float64, and to rounding in float32, where
the order in which the blocks are summed can move a value by one ulp. Here a
$150 \times 170$ image is cut into blocks of $31 \times 37$ pixels; `search=9`
only shortens the example, and reduces the halo from 90 to 30 pixels.

```pycon
>>> big = 255 * phantoms.filaments((150, 170), intensity=1.0, seed=1)
>>> noisy_big = big + 20 * torch.randn_like(big)
>>> est = BM3D(sigma=20, data_range=255, search=9, block=(31, 37)).fit()
>>> est.halo()
30
>>> out = est.apply(noisy_big)  # 25 blocks
>>> ref = bm3d(noisy_big, sigma=20, data_range=255, search=9)
>>> bool((out - ref).abs().max() < 1e-3)
True
```

Poisson–Gaussian data: a camera with gain 2, offset 100 and read noise 3 ADU,
looking at 1 to 26 electrons. The estimator is given the
`SensorEstimate`, and denoises after the Anscombe transform.

```pycon
>>> from optix.estimates import SensorEstimate
>>> from optix.models import simulate_sensor
>>> sensor = SensorEstimate(offset=100.0, variance=9.0, gain=2.0)
>>> rate = 1 + 0.1 * x
>>> z = simulate_sensor(rate, sensor, seed=0)
>>> den = BM3D().fit(z, init=sensor).apply(z)
>>> truth = 2.0 * rate + 100.0
>>> round(float((z - truth).square().mean().sqrt()), 1)
5.0
>>> round(float((den - truth).square().mean().sqrt()), 1)
2.3
```

Volumes use `BM4D` in the same way, here with the "modified" profile for data in $[0, 1]$.

```pycon
>>> from optix.denoise import bm4d
>>> axes = torch.meshgrid(
...     *[torch.linspace(0, 1, n) for n in (16, 32, 32)], indexing="ij"
... )
>>> ball = ((sum((a - 0.5) ** 2 for a in axes)) < 0.1).float()
>>> noisy = ball + 0.15 * torch.randn_like(ball)
>>> den = bm4d(noisy, sigma=0.15, data_range=1.0, profile="modified")
>>> bool((den - ball).square().mean() < (noisy - ball).square().mean() / 10)
True
```

Common variations: `output="basic"` for the first estimate; `layout=` to say
which axes are spatial when the array has frames or channels; `inverse=` to
choose another inverse of the Anscombe transform; a large zarr array in place of
the tensor, with `out=` a separate array; `search=` and `step=` (extensions) to go
faster. A fitted estimator can be saved and loaded like any other.

### Results on synthetic data

PSNR of the estimates of the synthetic scenes of the test suite (cells and
filaments on a smooth background, one realisation of the noise each), in dB. `BM3D`: $128^2$ pixels, range
$[0, 255]$; `BM4D`: $24 \times 48^2$ voxels, range $[0, 1]$.

| Case | Noisy | Basic | Final |
|---|---|---|---|
| `BM3D`, $\sigma = 10$ | 28.14 | 39.93 | 40.47 |
| `BM3D`, $\sigma = 25$ | 20.18 | 33.53 | 34.53 |
| `BM3D`, $\sigma = 50$ | 14.16 | 29.00 | 29.77 |
| `BM4D`, $\sigma = 0.05$ | 26.02 | | 40.34 (normal), 40.76 (modified) |
| `BM4D`, $\sigma = 0.15$ | 16.48 | | 34.03 (normal), 34.66 (modified) |

These numbers describe the synthetic scenes only, not the standard test images
of the papers. In `float32` and in `float64` the PSNR agrees to 0.01 dB. No
comparison with the authors' implementation is given: its licence
(non-commercial use) is not compatible with that of optix.

## References

!!! references "References"

    1. Dabov K, Foi A, Katkovnik V, Egiazarian K. *Image denoising by sparse
       3-D transform-domain collaborative filtering.* IEEE Trans Image Process
       **16**(8):2080–2095, 2007.
       [doi:10.1109/TIP.2007.901238](https://doi.org/10.1109/TIP.2007.901238)
    2. Maggioni M, Katkovnik V, Egiazarian K, Foi A. *A nonlocal
       transform-domain filter for volumetric data denoising and
       reconstruction.* IEEE Trans Image Process
       **22**(1):119–133, 2013.
       [doi:10.1109/TIP.2012.2210725](https://doi.org/10.1109/TIP.2012.2210725)
    3. Mäkitalo M, Foi A. *Optimal inversion of the generalized Anscombe
       transformation for Poisson-Gaussian noise.* IEEE Trans Image Process
       **22**(1):91-103, 2013.
       [doi:10.1109/TIP.2012.2202675](https://doi.org/10.1109/TIP.2012.2202675)

    ??? quote "BibTeX"

        ```bibtex
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
        @article{maggioni2013nonlocal,
          author  = {Maggioni, Matteo and Katkovnik, Vladimir and Egiazarian, Karen and Foi, Alessandro},
          title   = {A nonlocal transform-domain filter for volumetric data denoising and reconstruction},
          journal = {IEEE Transactions on Image Processing},
          volume  = {22},
          number  = {1},
          pages   = {119--133},
          year    = {2013},
          doi     = {10.1109/TIP.2012.2210725},
        }
        @article{makitalo2013optimal,
          author  = {M{\"a}kitalo, Markku and Foi, Alessandro},
          title   = {Optimal inversion of the generalized {A}nscombe transformation for {P}oisson-{G}aussian noise},
          journal = {IEEE Transactions on Image Processing},
          volume  = {22},
          number  = {1},
          pages   = {91--103},
          year    = {2013},
          doi     = {10.1109/TIP.2012.2202675},
        }
        ```
