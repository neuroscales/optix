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
    = \mathbf{n} + \mathbf{n}_0 + \boldsymbol{\varepsilon},
    \qquad
    \mathbf{n} \sim \mathrm{Poisson}\bigl(\mathbf{f} \odot \mathbf{A}\mathbf{x}\bigr),
    \quad
    \mathbf{n}_0 \sim \mathrm{Poisson}(\mathbf{b}),
    \quad
    \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}).
    $$

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

| optix | [Snyder et al. 1993](https://doi.org/10.1364/JOSAA.10.001014) |
|---|---|
| $\mathbf{y}$ (recorded image, ADU) | — (the paper works in electrons) |
| $\mathbf{r}$ (data in electrons) | $r(j)$ |
| $\mathbf{x}$ (latent image) | $\lambda(i)$ |
| $\mathbf{h}$, $\mathbf{A}$ (PSF, blur) | $p(j \mid i)$ |
| $\mathbf{f}$ (flat field) | $\beta(j)$ |
| $\mathbf{b}$ (background counts) | $\mu_0(j)$ |
| $\mathbf{o}$ (offset) | $m$ (bias), in ADU |
| $\boldsymbol{\varepsilon}$, $\sigma^2$ (read noise) | $g(j)$, $\sigma^2$ |
| $g$ (gain) | — |
| $\mathbf{n}$, $\mathbf{n}_0$ | $n_{\mathrm{obj}}(j)$, $n_0(j)$ |

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
$\mathbf{n} + \mathbf{n}_0$ as the missing data. The E step replaces each
noisy data value by the conditional mean of the counts given it,

$$
F(r, \mu, \sigma) = \mathbb{E}[\,n + n_0 \mid r\,],
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

| optix | [Snyder et al. 1993](https://doi.org/10.1364/JOSAA.10.001014) |
|---|---|
| $\mathbf{x}_k$ | $\lambda_k(i)$ |
| $\boldsymbol{\mu}_k$ | $\mu_k(j)$ |
| $\mathbf{A}^\top \mathbf{f}$ | $\bar\beta(i)$ |
| $F(\mathbf{r}, \boldsymbol{\mu}_k, \sigma)$ | $F[r(j), \mu_k(j), \sigma]$ |

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
at 0, which the paper does not specify). The **method of sieves**
(Section 4, Eq. 30, `sieve=s`) blurs the final iterate with a kernel
$\mathbf{s}$, which estimates the smoothed object $\mathbf{s} * \mathbf{x}$.

## Algorithm

```text
Input: data Y (ADU), PSF h, sensor (o, g, σ²), optional flat f and background b,
       noise = "exact", alpha = 0.05, max_iter = 100
r = (y − o) / g;   σ² ← variance / g²
f: given, else the shading flat rescaled to unit mean, else 1
b: (dark − o)/g clipped at 0, plus background / g
Initialise x(0): uniform, Σ(r − b) / Σ f per image
Repeat:
    μ_k = f ⊙ A x_k + b
    if n_iter is None and the χ² test Σ (r − μ_k)² / (μ_k + σ²) is not significant: stop
    c = F(r, μ_k, σ)            (exact)   or   Eq. 15 on r + σ²  (shifted-poisson)
    x_{k+1} = x_k / (Aᵀ f) ⊙ Aᵀ(f ⊙ c / μ_k)
If sieve s is given: x ← s * x
Return x, cropped to the observed region
```

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

```python
import torch
from optix.deconv import Snyder, RichardsonLucy
from optix.models import gaussian_psf, simulate_blur
from optix.estimates import SensorEstimate

torch.manual_seed(0)

# 1. A Gaussian PSF and a test pattern, imaged by a camera with read noise.
psf = gaussian_psf((15, 15), sigma=2.0)
x = torch.full((64, 64), 2.0)
x[16:48:8, 16:48] = 40.0
gain, offset, sigma = 2.0, 100.0, 3.0
counts = torch.poisson(simulate_blur(x, psf))
y = offset + gain * counts + sigma * torch.randn(64, 64)
sensor = SensorEstimate(offset=offset, gain=gain, variance=sigma**2)

# 2. Snyder with the camera model; Richardson–Lucy for comparison.
sn = Snyder(psf, n_iter=30).fit(y, init=sensor)
x_sn = sn.apply(y)
x_sp = (
    Snyder(psf, n_iter=30, noise="shifted-poisson")
    .fit(y, init=sensor)
    .apply(y)
)
x_rl = RichardsonLucy(psf, n_iter=30).fit(y, init=sensor).apply(y)

# 3. Errors against the truth (in electrons).
for est in (x_sn, x_sp, x_rl):
    print(float((est - x).abs().mean()))
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
