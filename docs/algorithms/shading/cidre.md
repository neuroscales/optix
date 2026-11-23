# CIDRE

!!! info "At a glance"
    - **Estimates:** `ShadingEstimate` (`flat`, `dark`)
    - **Noise model:** none
    - **Streaming:** per-pixel-stats
    - **Needs many frames:** yes
    - **Reference:** Smith et al., Nature Methods, 2015, [doi:10.1038/nmeth.3323](https://doi.org/10.1038/nmeth.3323)

## Overview

CIDRE (corrected intensity distributions using regularised energy
minimisation) estimates the illumination gain, a flat field $\mathbf{f}$, and
an additive zero-light term, a dark field $\mathbf{d}$, from many images
acquired under the same conditions, without calibration images
([Smith et al. 2015](https://doi.org/10.1038/nmeth.3323)). It works on the
distribution of intensities at each pixel rather than on the images
themselves, so the order of the frames does not matter.

Use it on large sets of images (tens to thousands of fields of view of a
screen) when a dark field matters. It does not estimate a per-frame baseline:
for time-lapse data with a drifting background, use [BaSiC](basic.md).

It returns a `ShadingEstimate` in `estimates["shading"]`, plus the zero-light
point `zero_light_`, the target distribution `target_` and the weight
`lambda_r_` used.

## Image formation (optics)

CIDRE uses the shading model of
[Image formation and noise](../../concepts/image-formation.md) without a
baseline:

=== "optix"

    $$
    \mathbf{y}_i = \mathbf{f} \odot \mathbf{x}_i + \mathbf{d}
    $$

=== "Smith et al. 2015"

    $$
    I_o(x) = v(x)\, I(x) + z(x) \tag{1}
    $$

$\mathbf{d}$ includes the camera offset. $\mathbf{f}$ is relative to a target
distribution (below) and is stored as produced (`meta["flat_norm"] == "none"`).
[`apply`](../../reference/shading.md) reverses the model and, by default,
rescales by the mean gain $\bar{f}$ ("dynamic range corrected"):

$$
\hat{\mathbf{x}}_i
= \underbrace{\frac{\mathbf{y}_i - \mathbf{d}}{\mathbf{f}}}_{\text{direct}}
\,\underbrace{\bar{f}}_{\text{dynamic range}}
\;\underbrace{+\;\bar{d}}_{\text{zero light}} .
$$

`mode="direct"` keeps only the first factor, `"dynamic_range"` (the default)
multiplies it by $\bar{f}$, and `"zero_light"` also adds $\bar{d}$, the mean
dark field.

Frames have $P$ pixels, flattened into vectors; products and divisions of
vectors are pixel-wise.

| optix | [Smith et al. 2015](https://doi.org/10.1038/nmeth.3323) | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}_i$ | $I_o$ | recorded image $i$ | $\mathbb{R}^P$, one per frame |
| $\mathbf{x}_i$ | $I$ | latent (shading-free) image $i$ | $\mathbb{R}^P$, one per frame |
| $\mathbf{f}$ | $v$ | illumination gain (flat field) | $\mathbb{R}^P$, shared by all frames of a group |
| $\mathbf{d}$ | $z$ | zero-light term (dark field) | $\mathbb{R}^P$, shared by all frames of a group |
| $\bar{f}$, $\bar{d}$ | | spatial means of $\mathbf{f}$ and $\mathbf{d}$ | scalars, per group |
| $\mathbf{r}$ | $x$ | pixel position | 2-D coordinate |

## Probabilistic model

**Assumption.** Objects appear anywhere with equal probability. Then the
sorted intensities $\mathbf{q}(\mathbf{r})$ at every pixel are a linear
transform of one distribution $\mathbf{Q}$ common to all pixels:

=== "optix"

    $$
    q_n(\mathbf{r}) = f(\mathbf{r})\,Q_n + c(\mathbf{r}), \qquad
    \mathbf{c} = \mathbf{d} - \mathbf{f}\,Z_Q
    $$

=== "Smith et al. 2015"

    $$
    q(x) = v(x)\,Q + b(x) \tag{7}
    $$

    $$
    b(x) = z(x) - v(x)\,Z_Q \tag{8}
    $$

with $n = 1 \ldots K$ the quantile levels ($K = 100$ by default),
$\mathbf{q}(\mathbf{r}) \in \mathbb{R}^K$ and $\mathbf{Q} \in \mathbb{R}^K$, and
$(Z_Q, Z_q)$ the zero-light point,
through which every line passes. $\mathbf{Q}$ is the robust mean of the
$\mathbf{q}(\mathbf{r})$ of the pixels whose mean lies in the central band
$[\rho_{1/2-\omega}, \rho_{1/2+\omega}]$ of all pixel means (Eqs. 5–6).

The fields minimise an energy, not a likelihood:

=== "optix"

    $$
    E = E_f + \lambda_r E_r + \lambda_z E_z + \lambda_b E_b ,
    $$

    $$
    E_f = \sum_{\mathbf{r}} \sum_n \tfrac{w^2}{2}
    \log\Bigl(1 + \bigl(\tfrac{Q_n f(\mathbf{r}) + c(\mathbf{r}) - q_n(\mathbf{r})}{w}\bigr)^2\Bigr),
    \qquad
    E_r = \sum_{t \in T} \Bigl\lVert \tfrac{1}{\lvert T \rvert}\, t^2\, \nabla^2 G_t * \mathbf{f} \Bigr\rVert^2 ,
    $$

    $$
    E_z = \sum_{\mathbf{r}} \bigl(f(\mathbf{r}) Z_Q + c(\mathbf{r}) - Z_q\bigr)^2,
    \qquad
    E_b = \beta(Z_Q) + \beta(Z_q).
    $$

=== "Smith et al. 2015"

    $$
    E(v, b, Z) = E_f(v, b) + \lambda_r E_r(v) + \lambda_z E_z(v, b, Z) + \lambda_b E_b(Z) \tag{9}
    $$

    with the Cauchy fit $E_f$ (Eq. 10), the multi-scale Laplacian-of-Gaussian
    regulariser $E_r$ (Eq. 11), the zero-light term $E_z$ (Eq. 12) and the
    barrier $E_b$ (Eqs. 13–14).

Here $\mathbf{r}$ runs over the $P_w \le 128^2$ pixels of the working grid,
on which $\mathbf{f}, \mathbf{c} \in \mathbb{R}^{P_w}$; $\lvert T \rvert$ is
the number of scales.

- $E_f$ is a robust (Cauchy, width $w$) fit of the $K$ quantiles at every pixel.
- $E_r$ makes $\mathbf{f}$ smooth: scale-normalised Laplacian of Gaussian at
  scales $T$ (standard deviations).
- $E_z$ pulls every line through the zero-light point.
- $E_b$ keeps the zero-light point in $[\alpha_1, \alpha_2]$, by default
  $[0, \min \mathbf{q}]$: $\beta$ is zero inside and $((x - \alpha)/w_b)^2$
  outside.

| optix | [Smith et al. 2015](https://doi.org/10.1038/nmeth.3323) | Meaning | Shape |
|---|---|---|---|
| $\mathbf{q}(\mathbf{r})$ | $q(x)$ | sorted intensities (quantiles) at pixel $\mathbf{r}$ | $\mathbb{R}^K$, per pixel |
| $\mathbf{Q}$ | $Q$ | target distribution common to all pixels | $\mathbb{R}^K$, per group |
| $\mathbf{c}$ | $b$ | intercept of the per-pixel line | $\mathbb{R}^{P_w}$, per group |
| $(Z_Q, Z_q)$ | $(Z_Q, Z_q)$ | zero-light point shared by all lines | two scalars, per group |
| $\omega$ | $\omega$ | half-width of the central band of pixel means | scalar |
| $w$, $w_b$ | | widths of the Cauchy fit and of the barrier | scalars |
| $T$ | | scales (standard deviations) of the regulariser | set of scalars |
| $\lambda_r, \lambda_z, \lambda_b$ | $\lambda_r, \lambda_z, \lambda_b$ | weights of $E_r$, $E_z$, $E_b$ | scalars, per group |

## Inference / derivation

The strategy reduces the stack to per-pixel statistics, then fits one smooth
gain and one intercept per pixel:

1. **Summarise.** Each frame is resampled to a working grid (long side 128) and
   each pixel is summarised by $K = 100$ quantiles.
2. **Target.** $\mathbf{Q}$ is computed from the central band of pixels.
3. **Minimise.** $E$ is minimised jointly over $\mathbf{f}$, $\mathbf{c}$ and
   $(Z_Q, Z_q)$ by L-BFGS, in a pivot space where $\mathbf{Q}$ and
   $\mathbf{q}$ are shifted by their medians (Eqs. 19–20), which only
   reparametrises $\mathbf{c}$.
4. **Recover.** $\mathbf{d} = \mathbf{c} + \mathbf{f}\,Z_Q$ (Eq. 7), and both
   maps are resampled back to the image size.

**Statistical counterpart.** $E_f$ is, up to constants, the negative
log-likelihood of Cauchy (Student-t with one degree of freedom) residuals, so
the fit is a robust M-estimation of each per-pixel line; $E_r$ and $E_z$ act
as Gaussian priors, which makes $\hat{\mathbf{f}}, \hat{\mathbf{c}}$ a MAP
estimate. Matching quantiles to $\mathbf{Q}$ is a quantile-based method of
moments for the affine transform at each pixel.

The ambiguities are resolved by the model itself: the scale of $\mathbf{f}$ is
that of $\mathbf{Q}$ (a gain of 1 means "like the typical pixel"), and the
split of the intercept into dark field and $\mathbf{f}\,Z_Q$ is fixed by the
zero-light point, constrained by $E_z$ and $E_b$.

## Algorithm

**Input:** frames $\mathbf{y}_1, \ldots, \mathbf{y}_N \in \mathbb{R}^P$,
optionally in groups. \
**Output:** `ShadingEstimate` with $\hat{\mathbf{f}}$ and $\hat{\mathbf{d}}$.

For each group:

1. **Resample** each frame to the working grid (long side 128, $P_w$ pixels).
2. **Summarise** each pixel by $K = 100$ quantiles $\mathbf{q}(\mathbf{r})$
   (exact, or from per-pixel histograms) and divide by the maximum intensity.
3. **Target** $\mathbf{Q}$: robust mean over the central band, $\omega = 0.25$.
4. **Weights:** $\lambda_r$ from 9.5 for one image to 6 at 200 images (6
   beyond), $\lambda_z = 0.5$, $\lambda_b = 1$.
5. **Initialise** $\mathbf{f} = \mathbf{1}$, lines through the pivot,
   $Z_Q = Z_q = (\alpha_1 + \alpha_2)/2$.
6. **Minimise** $E$ by L-BFGS (at most 500 iterations, history 20,
   tolerance $10^{-9}$).
7. **Recover** $\mathbf{d} = \mathbf{c} + \mathbf{f}\,Z_Q$ and resample
   $\mathbf{f}$, $\mathbf{d}$ to the image size.

## Faithful mode and extensions

Both presets are the paper's method: `faithful=True` and `faithful=False`
give the same values for every keyword.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `normalize_fit` | divide $E_f$ by $K$ (the $\lambda$ then weigh $1/K$ as much data); extension | `False` | `False` |
| `lambda_r` | weight of $E_r$ | schedule of Supplementary Note 3 | same |
| `lambda_z` | weight of $E_z$ | `0.5` | `0.5` |

Interpretations of the paper and choices it leaves open:

- $\mathbf{Q}$ is the mean over the central band (Eq. 6 prints a sum; the text
  calls it a robust mean).
- The Laplacian of Gaussian is normalised by $t^2$ with $t$ the standard
  deviation (Supplementary Note 3 calls $t$ the variance but multiplies by
  $t^2$, which is scale-invariant only for the standard deviation).
- The scale-space pyramid used by the authors when few images are available is
  not implemented: the paper does not give the function from entropy to the
  number of octaves.
- Not specified in the paper, with their defaults: `working_size=128`,
  `quantiles=100`, `cauchy_width=0.05` and `barrier_width=1e-3` (fractions of
  the maximum intensity), `lambda_b=1.0`, `band=0.25`,
  `scales=(1, 2, 4, 8)`, the L-BFGS settings, the borders of the filters
  (point reflection, so that linear trends are not penalised), the
  initialisation (`init_method="pivot"`), and the division of the intensities
  by their maximum.

## Memory and streaming

CIDRE is tagged `per-pixel-stats`. The fit makes one pass over the data, plus
a second one when the quantiles come from per-pixel histograms
(`quantile_method="histogram"`, or `"auto"` when the resampled stack does not
fit the memory budget). Memory is then $O(P_w \cdot \texttt{bins})$ on the
working grid ($P_w \le 128^2$), independent of the number of frames; the
energy has at most $K$ terms per pixel. [`apply`](../../reference/shading.md)
works on frame batches. Tune `working_size`, `quantiles` and `bins`.

## Usage

```python
import torch
from optix.models import random_shading, simulate_shading
from optix.shading import CIDRE

# 1. A synthetic stack with known truth.
truth = random_shading((64, 64), dark=20.0, seed=0)
x = 100 + 400 * torch.rand(50, 64, 64)
data = simulate_shading(x, truth)

# 2. Fit.
cidre = CIDRE().fit(data)
shading = cidre.estimates["shading"]

# 3. Correct (dynamic range corrected by default).
corrected = cidre.apply(data)
direct = cidre.apply(data, mode="direct")

# 4. Compare to the truth: f is relative to Q, so normalise it first.
flat = shading.flat / shading.flat.mean()
zero_light = cidre.zero_light_  # (Z_Q, Z_q)
```

See the [API reference](../../reference/shading.md) for every option.

## References

!!! references "References"

    1. Smith K, Li Y, Piccinini F, Csucs G, Balazs C, Bevilacqua A, Horvath P.
       *CIDRE: an illumination-correction method for optical microscopy.*
       Nat Methods **12**(5):404–406, 2015.
       [doi:10.1038/nmeth.3323](https://doi.org/10.1038/nmeth.3323)

    ??? quote "BibTeX"

        ```bibtex
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
