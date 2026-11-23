# Richardson–Lucy

!!! info "At a glance"
    - **Estimates:** deconvolved image (the latent image $\mathbf{x}$, same shape as the data)
    - **Noise model:** Poisson (Poisson-Gaussian as an extension)
    - **Streaming:** halo
    - **Needs many frames:** no
    - **Reference:** Richardson, J Opt Soc Am, 1972, [doi:10.1364/JOSA.62.000055](https://doi.org/10.1364/JOSA.62.000055); Lucy, Astron J, 1974, [doi:10.1086/111605](https://doi.org/10.1086/111605)

## Overview

`RichardsonLucy` removes a known blur from an image: given the recorded image
$\mathbf{y}$ and the point spread function (PSF) $\mathbf{h}$, it estimates the
latent image $\mathbf{x}$ that the microscope blurred. It is the
maximum-likelihood deconvolution under Poisson noise, keeps the estimate
non-negative, and has no regularisation: the number of iterations acts as the
regulariser, chosen by a χ² stopping rule by default. Use it for photon-counting
data with a measured or modelled PSF; for faster convergence use
[Biggs–Andrews](biggs-andrews.md), the same iteration accelerated. Every image of
a stack is deconvolved on its own; `apply` returns the deconvolved images.

## Image formation (optics)

The microscope is linear and shift-invariant over the field: the recorded image
is the latent image convolved with the PSF (see
[Image formation and noise](../../concepts/image-formation.md)). The PSF has 2 or
3 dims, is centred at index `K // 2` and is normalised to unit sum. Light from
outside the field of view is blurred into it, so the latent image is estimated on
a domain extended by the PSF support, and the blur crops back to the observed
region. Shading and camera offset are assumed removed, or handled by the
extensions below.

## Probabilistic model

=== "optix"

    $$
    \mathbf{y} \sim \mathrm{Poisson}(\mathbf{A}\mathbf{x} + \mathbf{b}),
    \qquad
    (\mathbf{A}\mathbf{x})_p = \sum_q h(p - q)\, x_q ,
    \qquad \mathbf{x} \ge 0,
    $$

    with $\mathbf{y} \in \mathbb{R}^P$ one recorded image of $P$ pixels, in
    counts (photons or electrons); $\mathbf{x} \in \mathbb{R}^Q$ the latent
    image on the extended domain of $Q \ge P$ pixels; $\mathbf{h}$ the PSF of
    $K$ voxels; $\mathbf{A} \in \mathbb{R}^{P \times Q}$ the blur
    ($\mathbf{h} \circledast$, cropped to the observed region); and
    $\mathbf{b} \in \mathbb{R}^P$ an optional background ($\mathbf{0}$ in
    the papers).

=== "Richardson 1972"

    $$
    H_k = \sum_i S_{i,k}\, W_i ,
    $$

    the blurred image $H$ of the object $W$, with $S_{i,k}$ the PSF: the
    probability that a photon from object point $i$ lands at image point $k$.

=== "Lucy 1974"

    $$
    \psi(x) = \int \phi(\xi)\, P(x \mid \xi)\, d\xi ,
    $$

    the observed distribution $\tilde\psi$ is a finite sample of $\psi$, the
    blur of the sought distribution $\phi$ by the kernel $P(x \mid \xi)$.

| optix | [Richardson 1972](https://doi.org/10.1364/JOSA.62.000055) | [Lucy 1974](https://doi.org/10.1086/111605) | Meaning | Shape |
|---|---|---|---|---|
| $\mathbf{y}$ | $H$ (recorded) | $\tilde\psi$ | recorded image, in counts | $P$, one per image |
| $\mathbf{A}\mathbf{x}$ | $H$ (model) | $\psi$ | blurred latent image, the noise-free prediction | $P$, one per image |
| $\mathbf{x}$ | $W$ | $\phi$ | latent (deconvolved) image | $Q$, one per image |
| $\mathbf{h}$, $\mathbf{A}$ | $S$ | $P(x \mid \xi)$ | PSF and the blur operator it defines | $K$ and $P \times Q$, one for the whole stack |
| $\mathbf{b}$ | none | none | optional background | $P$ or scalar, one for the whole stack |
| $p$, $q$ | $k$, $i$ | $x$, $\xi$ | image and object pixel indices | scalars |
| $t$ (iteration) | $r$ | $r$ | iteration counter | scalar |

## Inference / derivation

The estimate maximises the Poisson likelihood of $\mathbf{y}$ over
$\mathbf{x} \ge 0$ by a fixed-point iteration: each step multiplies the
current estimate by the back-projected ratio of data to prediction. This
iteration is exactly the expectation-maximisation (EM) algorithm for the
maximum-likelihood estimate
([Shepp and Vardi 1982](https://doi.org/10.1109/TMI.1982.4307558)): the
complete data are the unobserved photon counts $z_{pq}$ emitted at object
pixel $q$ and detected at pixel $p$, each $\mathrm{Poisson}(A_{pq} x_q)$; the
E-step splits each $y_p$ among the $q$ in proportion to $A_{pq} x_q$, and the
M-step sets $x_q$ to the expected counts it emitted divided by
$\sum_p A_{pq}$. This keeps
$\mathbf{x}$ non-negative, increases the likelihood at every step and conserves
the flux. Because the unregularised maximum fits the noise, the iteration is
stopped early, by
[Lucy's (1974, §IV)](https://doi.org/10.1086/111605) rule: stop as soon as the χ²
statistic of the fit, $\sum_p (y_p - \mu_p)^2 / \mu_p$ with
$\boldsymbol{\mu} = \mathbf{A}\mathbf{x}^{(t)} + \mathbf{b}$ and $P - 1$ degrees of
freedom, is no longer significant.

=== "optix"

    $$
    \mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} \odot
    \frac{\mathbf{A}^\top\!\left[\mathbf{y} / (\mathbf{A}\mathbf{x}^{(t)} + \mathbf{b})\right]}{\mathbf{A}^\top \mathbf{1}}
    $$

=== "Richardson 1972"

    $$
    W_{i,r+1} = W_{i,r} \sum_k \frac{S_{i,k}\, H_k}{\sum_j S_{j,k}\, W_{j,r}}
    $$

=== "Lucy 1974"

    $$
    \phi^{r+1}(\xi) = \phi^{r}(\xi) \int \frac{\tilde\psi(x)}{\psi^{r}(x)}\, P(x \mid \xi)\, dx
    $$

The normaliser $\mathbf{A}^\top\mathbf{1}$ is 1 inside the field and falls below 1
near its edges, where the extended domain receives less data.

## Algorithm

**Input:** an image $\mathbf{y} \in \mathbb{R}^P$ in counts, the PSF
$\mathbf{h}$; hyper-parameters `alpha` $= 0.05$, `max_iter` $= 100$, `n_iter`
$=$ `None`. \
**Output:** the deconvolved image $\hat{\mathbf{x}}$, cropped to the observed
region.

1. **Initialise** $\mathbf{x}^{(0)}$ uniform, at the mean intensity of
   $\mathbf{y} - \mathbf{b}$ (one level per image).
2. **Repeat** for $t = 0, 1, \ldots$ until `n_iter` iterations (if given) or
   `max_iter` iterations:
    1. **Test** (only when `n_iter` is `None`): compute
       $\chi^2 = \sum_p (y_p - \mu_p)^2 / \mu_p$ with
       $\boldsymbol{\mu} = \mathbf{A}\mathbf{x}^{(t)} + \mathbf{b}$; stop if
       $\Pr(\chi^2_{P-1} > \chi^2) \ge$ `alpha`;
    2. **Update** $\mathbf{x}^{(t+1)}$ by the multiplicative Richardson–Lucy
       step above.
3. **Crop** $\mathbf{x}$ to the observed region and return it.

The number of iterations per block is recorded in
`history.extras["iterations"]`.

## Faithful mode and extensions

`faithful=True` (the default) reproduces the papers: uniform start, Lucy's χ²
stopping rule, no background, Poisson likelihood, full correction. The defaults of
`faithful=False` are the same. Fitting with a camera model,
`fit(data, init=sensor)`, converts the data to electrons, $(\mathbf{y} - o)/g$,
and makes the estimator non-faithful.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `n_iter` | fixed number of iterations instead of the χ² rule | `None` | `None` |
| `alpha` | level of the χ² stopping test | 0.05 | 0.05 |
| `max_iter` | cap of the χ² rule (not in the papers) | 100 | 100 |
| `background` | $\mathbf{b}$ in $\mathrm{Poisson}(\mathbf{A}\mathbf{x} + \mathbf{b})$, data units | `None` | `None` |
| `poisson_gaussian` | shifted Poisson: $\mathbf{y} + \sigma^2 \sim \mathrm{Poisson}(\mathbf{A}\mathbf{x} + \mathbf{b} + \sigma^2)$, $\sigma^2$ from the sensor estimate | `False` | `False` |
| `damping` | fraction $\varepsilon \in (0, 1]$ of the correction applied, as in [Lucy (1974, §III)](https://doi.org/10.1086/111605) | 1.0 | 1.0 |
| `halo` | halo width per spatial axis, at least the PSF size | `None` (twice the PSF size) | `None` |
| `block` | core block size per spatial axis | `None` (from the memory budget) | `None` |

## Memory and streaming

Large images are cut into blocks with a halo (`blocks`, halo streaming). Each
block, core plus halo, is deconvolved as an image of its own by FFT on a padded
grid ([`PaddedConvolution`](../../reference/deconv.md)), and only its core is kept;
the seams fade with the halo width, and along image edges the block-wise and
whole-image results coincide. A block needs about twelve working buffers of its
size; `block` and `halo` trade memory for seam quality. Cost is one forward and
one adjoint FFT per iteration. `fit` makes one pass over the data to compute the
start levels.

## Usage

The example blurs a test pattern with a Gaussian PSF from
[`optix.models`](../../reference/models.md), using the same forward model as
the estimator.

```python
import torch
from optix.deconv import RichardsonLucy
from optix.models import gaussian_psf, simulate_blur

torch.manual_seed(0)

# 1. A Gaussian PSF and a blurred, Poisson-noisy test pattern.
psf = gaussian_psf((15, 15), sigma=2.0)
x = torch.full((96, 96), 5.0)
x[20:76:8, 20:76] = 200.0
y = torch.poisson(simulate_blur(x, psf))

# 2. Fit and deconvolve (χ² stopping rule).
rl = RichardsonLucy(psf).fit(y)
x_hat = rl.apply(y)

# 3. Compare to the truth.
print(rl.history.extras["iterations"])  # iterations used
print(float((y - x).abs().mean()), float((x_hat - x).abs().mean()))
```

Common variations: `RichardsonLucy(psf, n_iter=20)` for a fixed number of
iterations; `fit(y, init=sensor)` with a
[`SensorEstimate`](../../reference/estimates.md) for raw camera data, adding
`poisson_gaussian=True` to account for read noise; `background=b`; a `zarr`
array as input and `out`, processed block by block. The function
`richardson_lucy(data, psf, ...)` is a one-line shortcut. See the
[API reference](../../reference/deconv.md).

## References

!!! references "References"

    1. Richardson WH. *Bayesian-based iterative method of image restoration.*
       J Opt Soc Am **62**(1):55-59, 1972.
       [doi:10.1364/JOSA.62.000055](https://doi.org/10.1364/JOSA.62.000055)
    2. Lucy LB. *An iterative technique for the rectification of observed
       distributions.* Astron J **79**(6):745-754, 1974.
       [doi:10.1086/111605](https://doi.org/10.1086/111605)
    3. Shepp LA, Vardi Y. *Maximum likelihood reconstruction for emission
       tomography.* IEEE Trans Med Imaging **1**(2):113-122, 1982.
       [doi:10.1109/TMI.1982.4307558](https://doi.org/10.1109/TMI.1982.4307558)

    ??? quote "BibTeX"

        ```bibtex
        @article{richardson1972bayesian,
          author  = {Richardson, William Hadley},
          title   = {Bayesian-Based Iterative Method of Image Restoration},
          journal = {Journal of the Optical Society of America},
          volume  = {62},
          number  = {1},
          pages   = {55--59},
          year    = {1972},
          doi     = {10.1364/JOSA.62.000055},
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
        @article{shepp1982maximum,
          author  = {Shepp, L. A. and Vardi, Y.},
          title   = {Maximum Likelihood Reconstruction for Emission Tomography},
          journal = {IEEE Transactions on Medical Imaging},
          volume  = {1},
          number  = {2},
          pages   = {113--122},
          year    = {1982},
          doi     = {10.1109/TMI.1982.4307558},
        }
        ```
