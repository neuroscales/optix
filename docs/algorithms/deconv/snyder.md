# Snyder

!!! info "At a glance"
    - **Estimates:** deconvolved image (the latent image $\mathbf{x}$, same shape as the data)
    - **Noise model:** Poisson-Gaussian (Poisson counts plus Gaussian read noise)
    - **Streaming:** halo
    - **Needs many frames:** no
    - **Reference:** Snyder, Hammoud and White, J Opt Soc Am A, 1993, [doi:10.1364/JOSAA.10.001014](https://doi.org/10.1364/JOSAA.10.001014)

## Overview

`Snyder` is maximum-likelihood deconvolution of camera data that keeps the
camera in the model ([Snyder, Hammoud and White 1993](https://doi.org/10.1364/JOSAA.10.001014)).
Where [Richardson–Lucy](richardson-lucy.md) assumes the data are pure Poisson
counts, `Snyder` models the pixel-to-pixel detection efficiency (the flat
field), Poisson background counts (dark current, stray light) and Gaussian
read noise, instead of subtracting or dividing them out of the data first,
which would leave data that are no longer Poisson. Use it on raw camera
frames, together with a [`SensorEstimate`](../../reference/estimates.md) and
optionally a `ShadingEstimate`, especially at low counts where read noise
matters. On data already in photo-electrons with negligible read noise it
reduces to Richardson–Lucy.

## Image formation (optics)

The object is blurred by a known, shift-invariant PSF $\mathbf{h}$ of unit sum
(the operator $\mathbf{A}$), detected with a per-pixel efficiency
$\mathbf{f}$, and recorded by a camera that adds background counts, a bias and
read noise (see [Image formation and noise](../../concepts/image-formation.md)).
In electrons, after the offset has been subtracted and the gain divided out,

=== "optix"

    $$
    \mathbf{r} = \frac{\mathbf{y} - \mathbf{o}}{g}
    = \mathbf{e} + \mathbf{e}_0 + \boldsymbol{\varepsilon},
    \qquad
    \mathbf{e} \sim \mathrm{Poisson}\bigl(\mathbf{f} \odot \mathbf{A}\mathbf{x}\bigr),
    \quad
    \mathbf{e}_0 \sim \mathrm{Poisson}(\mathbf{b}),
    \quad
    \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}).
    $$

    Here $\mathbf{y}, \mathbf{r}, \mathbf{e}, \mathbf{e}_0,
    \boldsymbol{\varepsilon} \in \mathbb{R}^{\lvert\Omega\rvert}$ for one image of $\lvert\Omega\rvert$ pixels,
    $\mathbf{x} \in \mathbb{R}^Q$ is the latent image on the domain extended
    by the PSF support ($Q \ge \lvert\Omega\rvert$ pixels) and
    $\mathbf{A} \in \mathbb{R}^{\lvert\Omega\rvert \times Q}$.

=== "Snyder et al. 1993"

    $$
    r(j) = n_{\mathrm{obj}}(j) + n_0(j) + g(j),
    \qquad
    n_{\mathrm{obj}}(j) \sim \mathrm{Poisson}\Bigl(\beta(j) \sum_i p(j \mid i)\, \lambda(i)\Bigr),
    $$

    $$
    n_0(j) \sim \mathrm{Poisson}\bigl(\mu_0(j)\bigr),
    \qquad
    g(j) \sim \mathcal{N}(m, \sigma^2)
    \quad \text{(Eqs. 5–11)}.
    $$

| optix | [Snyder et al. 1993](https://doi.org/10.1364/JOSAA.10.001014) | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}$ | none (the paper works in electrons) | recorded image, in ADU | $\lvert\Omega\rvert$, one per image |
| $\mathbf{r}$ | $r(j)$ | data in electrons | $\lvert\Omega\rvert$, one per image |
| $\mathbf{x}$ | $\lambda(i)$ | latent image | $Q$, one per image |
| $\mathbf{h}$, $\mathbf{A}$ | $p(j \mid i)$ | PSF and blur operator | $K$ and $\lvert\Omega\rvert \times Q$, one for the whole stack |
| $\mathbf{f}$ | $\beta(j)$ | flat field (detection efficiency) | $\lvert\Omega\rvert$, one per pixel, shared by all images |
| $\mathbf{b}$ | $\mu_0(j)$ | mean background counts | $\lvert\Omega\rvert$, one per pixel, shared by all images |
| $\mathbf{o}$ | $m$ (bias) | camera offset, in ADU | $\lvert\Omega\rvert$ or scalar, one per pixel, shared by all images |
| $\boldsymbol{\varepsilon}$, $\sigma^2$ | $g(j)$, $\sigma^2$ | read noise and its variance | $\lvert\Omega\rvert$ per image; $\sigma^2$ per pixel or scalar, shared |
| $g$ | none | camera gain, ADU per electron | $\lvert\Omega\rvert$ or scalar, shared by all images |
| $\mathbf{e}$, $\mathbf{e}_0$ | $n_{\mathrm{obj}}(j)$, $n_0(j)$ | object and background photo-electron counts | $\lvert\Omega\rvert$, one per image |

The paper's $g(j)$ is the read noise, not the gain. Since the offset is
subtracted in optix, its bias is $m = 0$ in electrons.

## Probabilistic model

Assumptions: the PSF is known and shift-invariant; the photo-electrons from the
object and the background counts are independent Poisson variables; the read
noise is Gaussian, independent between pixels and of the counts; the flat
field, background, offset, gain and read-noise variance are known (from the
calibration estimates); the object is non-negative. The likelihood of each
pixel is a Poisson distribution of mean

$$
\boldsymbol{\mu} = \mathbf{f} \odot \mathbf{A}\mathbf{x} + \mathbf{b}
$$

convolved with a Gaussian of variance $\sigma^2$, and the estimate is

$$
\hat{\mathbf{x}} = \arg\max_{\mathbf{x} \ge 0}\; \log p(\mathbf{r} \mid \mathbf{x})
\quad \text{(Eq. 10)}.
$$

There is no prior; regularisation, when wanted, is by the method of sieves
(below).

## Inference / derivation

The strategy is expectation–maximisation with the Poisson counts
$\mathbf{e} + \mathbf{e}_0$ as the missing data. The E step replaces each
noisy data value by the conditional mean of the counts given it,

$$
F(r, \mu, \sigma) = \mathbb{E}[\,e + e_0 \mid r\,],
$$

a ratio of two sums over the possible counts (Eqs. 19 and 21) that pulls the
data towards the current prediction $\mu$ where read noise dominates. The M
step is then a Richardson–Lucy update on these "cleaned" counts, with the flat
field in the forward and back projections:

=== "optix"

    $$
    \mathbf{x}_{k+1} = \frac{\mathbf{x}_k}{\mathbf{A}^\top \mathbf{f}} \odot
    \mathbf{A}^\top\!\left(\mathbf{f} \odot
    \frac{F(\mathbf{r}, \boldsymbol{\mu}_k, \sigma)}{\boldsymbol{\mu}_k}\right),
    \qquad
    \boldsymbol{\mu}_k = \mathbf{f} \odot \mathbf{A}\mathbf{x}_k + \mathbf{b}.
    $$

=== "Snyder et al. 1993"

    $$
    \lambda_{k+1}(i) = \frac{\lambda_k(i)}{\bar\beta(i)}
    \sum_j \beta(j)\, p(j \mid i)\,
    \frac{F[r(j), \mu_k(j), \sigma]}{\mu_k(j)},
    \qquad
    \bar\beta(i) = \sum_j \beta(j)\, p(j \mid i)
    \quad \text{(Eqs. 18, 20)}.
    $$

| optix | [Snyder et al. 1993](https://doi.org/10.1364/JOSAA.10.001014) | Meaning | Shape |
|---|---|---|---|
| $\mathbf{x}_k$ | $\lambda_k(i)$ | iterate after $k$ updates | $Q$, one per image |
| $\boldsymbol{\mu}_k$ | $\mu_k(j)$ | predicted mean counts | $\lvert\Omega\rvert$, one per image |
| $\mathbf{A}^\top \mathbf{f}$ | $\bar\beta(i)$ | back-projected flat, the normaliser | $Q$, shared by all images |
| $F(\mathbf{r}, \boldsymbol{\mu}_k, \sigma)$ | $F[r(j), \mu_k(j), \sigma]$ | conditional mean of the counts given the data | $\lvert\Omega\rvert$, one per image |

As for any EM scheme, every iterate is non-negative and the likelihood never
decreases. Without read noise ($\sigma \to 0$), $F = r$ and the update is
Richardson–Lucy with a flat field and background (Eq. 15); with
$\mathbf{f} = \mathbf{1}$, $\mathbf{b} = \mathbf{0}$ it is Richardson–Lucy
itself.

Two variants of the paper are available. The **shifted-Poisson
approximation** (Section 3.C, `noise="shifted-poisson"`) avoids evaluating
$F$: for large $\sigma^2$, Gaussian read noise is close to a Poisson variable
of mean $\sigma^2$, so $\mathbf{r} + \sigma^2$ is treated as Poisson with mean
$\boldsymbol{\mu} + \sigma^2$ and Eq. 15 is used (the shifted data are clipped
at 0, which the paper does not specify). This is EM for the approximate model
$\mathbf{r} + \sigma^2 \sim \mathrm{Poisson}(\boldsymbol{\mu} + \sigma^2)$,
in which the read noise becomes one more Poisson background of mean
$\sigma^2$: the E step needs no sum over counts, and the update is
Richardson–Lucy with background $\mathbf{b} + \sigma^2$. The **method of sieves**
(Section 4, Eq. 30, `sieve=s`) blurs the final iterate with a kernel
$\mathbf{s}$, which estimates the smoothed object $\mathbf{s} * \mathbf{x}$.

## Algorithm

**Input:** an image $\mathbf{y} \in \mathbb{R}^{\lvert\Omega\rvert}$ in ADU, the PSF
$\mathbf{h}$, the sensor (offset $\mathbf{o}$, gain $g$, read-noise variance),
optional flat $\mathbf{f}$ and background; hyper-parameters `noise` $=$
`"exact"`, `alpha` $= 0.05$, `max_iter` $= 100$, `n_iter` $=$ `None`. \
**Output:** the deconvolved image $\hat{\mathbf{x}}$, cropped to the observed
region.

1. **Convert to electrons:** $\mathbf{r} = (\mathbf{y} - \mathbf{o}) / g$ and
   $\sigma^2 = \text{variance} / g^2$.
2. **Camera maps:** $\mathbf{f}$ as given, else the shading flat rescaled to
   unit mean, else $\mathbf{1}$; $\mathbf{b}$ from the dark field $\mathbf{d}$ of the shading estimate,
   $(\mathbf{d} - \mathbf{o})/g$ clipped at 0, plus the extra background
   divided by $g$.
3. **Initialise** $\mathbf{x}_0$ uniform, at
   $\sum_{i\in\Omega} (r_i - b_i) / \sum_{i\in\Omega} f_i$ per image.
4. **Repeat** for $k = 0, 1, \ldots$ until `n_iter` (if given) or `max_iter`
   iterations:
    1. **Predict** $\boldsymbol{\mu}_k = \mathbf{f} \odot \mathbf{A}\mathbf{x}_k + \mathbf{b}$;
    2. **Test** (only when `n_iter` is `None`): stop if the χ² statistic
       $\sum_{i\in\Omega} (r_i - \mu_{k,i})^2 / (\mu_{k,i} + \sigma^2)$ is no longer
       significant at level `alpha`;
    3. **E step:** $\mathbf{c} = F(\mathbf{r}, \boldsymbol{\mu}_k, \sigma)$
       (exact), or use Eq. 15 on $\mathbf{r} + \sigma^2$ (shifted Poisson);
    4. **M step:** $\mathbf{x}_{k+1} = \mathbf{x}_k / (\mathbf{A}^\top\mathbf{f}) \odot \mathbf{A}^\top(\mathbf{f} \odot \mathbf{c} / \boldsymbol{\mu}_k)$.
5. **Sieve** (if `sieve` $= \mathbf{s}$ is given): $\mathbf{x} \leftarrow \mathbf{s} * \mathbf{x}$.
6. **Crop** $\mathbf{x}$ to the observed region and return it.

The start and the stopping rule are not in the paper (its example runs 600
iterations); they are those of [Richardson–Lucy](richardson-lucy.md), with the
χ² test of [Lucy (1974)](https://doi.org/10.1086/111605) using the
Poisson-Gaussian variance $\mu_k + \sigma^2$.

## Faithful mode and extensions

The camera terms (`flat`, `background`) and the paper's own variants (`noise`,
`sieve`) are part of the model and keep the estimate faithful whatever their
value. Without a `SensorEstimate` the data are taken to be electrons with
$\sigma = 0$; a per-frame `baseline` of a `ShadingEstimate` is not part of the
model and is ignored with a warning.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `noise` | read-noise treatment: `"exact"` conditional mean $F$, or `"shifted-poisson"` (Section 3.C) | `"exact"` (either is faithful) | `"exact"` |
| `flat` | flat field $\mathbf{f}$, used as given | `None` (shading flat, else 1) | `None` |
| `background` | extra background counts in ADU, offset excluded, added to $\mathbf{b}$ | `None` | `None` |
| `sieve` | kernel blurring the final iterate (Section 4) | `None` | `None` |
| `n_iter` | fixed number of iterations instead of the χ² rule | `None` | `None` |
| `alpha` | level of the χ² stopping test | `0.05` | `0.05` |
| `max_iter` | iteration cap of the χ² rule | `100` | `100` |
| `halo` | halo per spatial axis, at least the PSF size | `None` (twice the PSF) | `None` |
| `block` | core block size per spatial axis | `None` (memory budget) | `None` |

The paper specifies none of `n_iter`, `alpha`, `max_iter`, `halo`, `block`;
the faithful and recommended presets coincide.

## Memory and streaming

As for [Richardson–Lucy](richardson-lucy.md#memory-and-streaming): each block
of the image, with a halo of at least the PSF size, runs all its iterations on
its own, in one pass over the data. The camera maps (offset, gain, variance,
flat, background) are read for each block alongside the data. The exact
conditional mean $F$ is a sum over counts at every pixel and iteration, which
makes an iteration dearer than one of Richardson–Lucy;
`noise="shifted-poisson"` costs the same as Richardson–Lucy.

## Usage

The example images a [phantom](../../reference/models.md#phantoms) of filaments through a Gaussian PSF
and a simulated camera with read noise.

```python
import torch
from optix.deconv import RichardsonLucy, Snyder
from optix.models import (
    gaussian_psf,
    phantoms,
    random_sensor,
    simulate_blur,
    simulate_sensor,
)

# 1. Filaments, blurred and imaged by a camera with read noise.
psf = gaussian_psf((15, 15), sigma=2.0)
x = phantoms.filaments((64, 64), intensity=40.0, seed=0) + 2
sensor = random_sensor((64, 64), gain=2.0, read_sd=6.0, offset_sd=0.0, seed=0)
y = simulate_sensor(simulate_blur(x, psf), sensor, seed=0)

# 2. Fit Snyder with the camera model, and Richardson–Lucy to compare.
sn = Snyder(psf, n_iter=30).fit(y, init=sensor)
sp = Snyder(psf, n_iter=30, noise="shifted-poisson").fit(y, init=sensor)
rl = RichardsonLucy(psf, n_iter=30).fit(y, init=sensor)

# 3. Apply, and 4. compare to the truth (electrons):
# mean error about 3.7, 3.7 and 4.0.
for est in (sn, sp, rl):
    print(float((est.apply(y) - x).abs().mean()))
```

Common variations: `fit(y, init=[sensor, shading])` to take the flat and dark
fields from a `ShadingEstimate`; `flat=` for a quantum-efficiency map;
`sieve=` for a smoothed estimate; no `n_iter` for the χ² stopping rule. See the
[API reference](../../reference/deconv.md).

## References

!!! references "References"

    1. Snyder DL, Hammoud AM, White RL. *Image recovery from data acquired
       with a charge-coupled-device camera.* J Opt Soc Am A
       **10**(5):1014-1023, 1993.
       [doi:10.1364/JOSAA.10.001014](https://doi.org/10.1364/JOSAA.10.001014)
    2. Lucy LB. *An iterative technique for the rectification of observed
       distributions.* Astron J **79**(6):745-754, 1974.
       [doi:10.1086/111605](https://doi.org/10.1086/111605)

    ??? quote "BibTeX"

        ```bibtex
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
        @article{lucy1974iterative,
          author  = {Lucy, L. B.},
          title   = {An iterative technique for the rectification of observed distributions},
          journal = {The Astronomical Journal},
          volume  = {79},
          number  = {6},
          pages   = {745--754},
          year    = {1974},
          doi     = {10.1086/111605},
        }
        ```
