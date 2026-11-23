# Wiener

!!! info "At a glance"
    - **Estimates:** deconvolved image (the latent image $\mathbf{x}$, same shape as the data)
    - **Noise model:** Gaussian (additive, stationary)
    - **Streaming:** halo
    - **Needs many frames:** no
    - **Reference:** Gonzalez and Woods, *Digital Image Processing*; Bertero and Boccacci, *Introduction to Inverse Problems in Imaging*, 1998 (textbooks, no DOI)

## Overview

`Wiener` deconvolves an image with a known PSF by a single linear filter, the
minimum-mean-squared-error estimate for a stationary object and noise. It is
one FFT per block, so it is the fastest deconvolution in optix: a preview, or
a starting point for iterative methods such as
[Richardson–Lucy](richardson-lucy.md). The result is linear in the data, so it
can be negative and its noise is correlated over the PSF scale. With a
constant noise-to-signal ratio it is the zero-order
[Tikhonov](tikhonov.md) filter; its Wiener inverse of the OTF is also the
first factor of the [Wiener–Butterworth](wiener-butterworth.md) back
projector.

## Image formation (optics)

A linear, shift-invariant blur $\mathbf{A}$ by a known PSF $\mathbf{h}$ of unit
sum (so $|\hat{h}(\boldsymbol{\nu})| \le 1$ and $\hat{h}(\mathbf{0}) = 1$), plus
additive noise (see [Image formation and noise](../../concepts/image-formation.md)).
One image (or one block of it) of $\lvert\Omega\rvert$ pixels is a vector
$\mathbf{y} \in \mathbb{R}^{\lvert\Omega\rvert}$, and the blur a matrix
$\mathbf{A} \in \mathbb{R}^{\lvert\Omega\rvert\times\lvert\Omega\rvert}$:

=== "optix"

    $$
    \mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon}
    = \mathbf{h} * \mathbf{x} + \boldsymbol{\varepsilon}
    $$

=== "Gonzalez & Woods"

    $$
    g = H f + \eta = h * f + \eta
    $$

| optix | Gonzalez & Woods | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}$ | $g$ | recorded image | $\mathbb{R}^{\lvert\Omega\rvert}$, per image |
| $\mathbf{x}$ | $f$ | latent (sharp) image | $\mathbb{R}^{\lvert\Omega\rvert}$, per image |
| $\mathbf{h}$ | $h$ | PSF, unit sum | kernel, one per group |
| $\mathbf{A}$ | $H$ | convolution by $\mathbf{h}$ | $\mathbb{R}^{\lvert\Omega\rvert\times\lvert\Omega\rvert}$, one per group |
| $\boldsymbol{\varepsilon}$ | $\eta$ | additive noise | $\mathbb{R}^{\lvert\Omega\rvert}$, per image |
| $\hat{h}(\boldsymbol{\nu})$ | $H(u, v)$ | OTF, Fourier transform of the PSF | scalar per frequency, one per group |
| $\boldsymbol{\nu}$ | $(u, v)$ | spatial frequency, cycles per pixel | one per spatial axis |

The textbooks' $f$ is the object, not the optix flat field. The class
docstring writes $y = Hf + n$ with $S$ the PSF.

## Probabilistic model

Assumptions: the PSF is known and shift-invariant; object and noise are
zero-mean-independent, wide-sense stationary random fields with power spectra
$S_x(\boldsymbol{\nu})$ and $S_\varepsilon(\boldsymbol{\nu})$; the image is
periodic (the FFT model; see the boundary handling below). The estimate is the
linear filter $\mathbf{R}$ that minimises the expected squared error
$\mathbb{E}\,\lVert \mathbf{x} - \mathbf{R}\mathbf{y} \rVert^2$. No
non-negativity is imposed.

## Inference / derivation

Because blur, object and noise are all stationary, the optimal filter is a
convolution and decouples over frequencies; each frequency is a scalar
least-squares problem, solved in closed form:

=== "optix"

    $$
    \hat{\mathbf{x}}(\boldsymbol{\nu}) =
    \frac{\hat{h}^*(\boldsymbol{\nu})}{|\hat{h}(\boldsymbol{\nu})|^2 + K(\boldsymbol{\nu})}\,
    \hat{\mathbf{y}}(\boldsymbol{\nu}),
    \qquad
    K(\boldsymbol{\nu}) = \frac{S_\varepsilon(\boldsymbol{\nu})}{S_x(\boldsymbol{\nu})} .
    $$

=== "Gonzalez & Woods"

    $$
    \hat{F}(u, v) = \frac{H^*(u, v)}{|H(u, v)|^2 + S_\eta(u, v) / S_f(u, v)}\, G(u, v).
    $$

| optix | Gonzalez & Woods | Meaning | Shape |
|---|---|---|---|
| $\hat{\mathbf{x}}(\boldsymbol{\nu})$, $\hat{\mathbf{y}}(\boldsymbol{\nu})$ | $\hat{F}(u, v)$, $G(u, v)$ | Fourier transforms of estimate and data | $\mathbb{C}^{\lvert\Omega\rvert}$, per image |
| $S_x$, $S_\varepsilon$ | $S_f$, $S_\eta$ | power spectra of object and noise | scalar per frequency, whole stack |
| $K$ (`nsr`) | $S_\eta / S_f$ (often a constant $K$) | noise-to-signal ratio | scalar, or per frequency; whole stack |

In statistical terms this is the linear minimum-mean-squared-error estimator. When object and noise are Gaussian,
$\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_x)$ and
$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\varepsilon)$
with circulant covariances $\boldsymbol{\Sigma}_x, \boldsymbol{\Sigma}_\varepsilon \in \mathbb{R}^{\lvert\Omega\rvert\times\lvert\Omega\rvert}$
(diagonalised by the FFT, eigenvalues $S_x$ and $S_\varepsilon$), it is also
the posterior mean, and the MAP estimate, of $\mathbf{x}$ given $\mathbf{y}$.
A constant $K$ is the MAP estimate under white noise of variance $\sigma^2$
and a white Gaussian prior of variance
$\sigma^2/K$, which is [Tikhonov](tikhonov.md) with $\lambda = K$.

The power spectra are rarely known, so $K$ is in practice a constant
noise-to-signal ratio tuned by hand (`nsr`), or a function of frequency given
by the user. $K \to 0$ gives the inverse filter $\hat{\mathbf{y}}/\hat{h}$;
larger $K$ suppresses the frequencies the PSF transmits poorly, trading
resolution for noise. Where both $|\hat{h}|^2$ and $K$ vanish the filter is
set to 0.

The FFT treats each block as periodic. With `boundary="taper"`, each block
(core plus halo) is extended past its edges by its mirror image over the halo
width, and the two reflections are joined by a raised-cosine blend, so the
periodic continuation has no jump and the edges do not ring; only the core is
kept. With `boundary="periodic"` the block is filtered as is, which exactly
inverts a circular blur on a single block.

## Algorithm

**Input:** images $\mathbf{y} \in \mathbb{R}^{\lvert\Omega\rvert}$; PSF $\mathbf{h}$ of unit sum;
`nsr` $K = 0.01$; `boundary = "taper"` \
**Output:** the deconvolved images $\hat{\mathbf{x}} \in \mathbb{R}^{\lvert\Omega\rvert}$.

1. **Split** each image into blocks, each a core plus a halo of at least the
   PSF size (default twice).
2. **For each block:**
    1. **extend** it smoothly to an FFT grid (`"taper"`), or keep it as is
       (`"periodic"`);
    2. **filter**:
       $\hat{\mathbf{x}}(\boldsymbol{\nu}) = \dfrac{\hat{h}^*(\boldsymbol{\nu})}{\lvert\hat{h}(\boldsymbol{\nu})\rvert^2 + K(\boldsymbol{\nu})}\,\hat{\mathbf{y}}(\boldsymbol{\nu})$;
    3. **invert** the FFT and keep the core.
3. **Return** the assembled $\hat{\mathbf{x}}$.

## Faithful mode and extensions

The method comes from textbooks, which leave $K$ to be tuned by hand; the
faithful and recommended presets coincide.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `nsr` | noise-to-signal ratio $K \ge 0$, or a callable $K(\boldsymbol{\nu})$ of the block frequencies | `0.01` | `0.01` |
| `boundary` | `"taper"` (smooth periodic extension) or `"periodic"` | `"taper"` | `"taper"` |
| `halo` | halo per spatial axis, at least the PSF size | `None` (twice the PSF) | `None` |
| `block` | core block size per spatial axis | `None` (memory budget) | `None` |

Extension: `fit(y, init=sensor)` with a `SensorEstimate` converts the data to
electrons, $(\mathbf{y} - \mathbf{o})/g$, first. An estimator with a callable
`nsr` cannot be saved.

## Memory and streaming

One pass over the data, block by block with a halo; images along sample and
group axes are independent. Each block costs two FFTs on its extended grid;
the filter is computed once per block shape. The block-wise result approaches
the whole-image one geometrically as the halo grows, more slowly for weak
regularisation (small $K$), whose impulse response is longer: increase `halo`
then.

## Usage

```python
import torch
from optix.deconv import Wiener
from optix.models import gaussian_psf, simulate_blur

torch.manual_seed(0)

# 1. A Gaussian PSF and a blurred, noisy test pattern.
psf = gaussian_psf((15, 15), sigma=2.0)
x = torch.full((96, 96), 5.0)
x[20:76:8, 20:76] = 200.0
y = torch.poisson(simulate_blur(x, psf))

# 2. Wiener filters with a strong and a weak regularisation.
for nsr in (1e-1, 1e-2, 1e-3):
    x_w = Wiener(psf, nsr=nsr).fit(y).apply(y)
    print(nsr, float((x_w - x).abs().mean()))
```

Common variations: `nsr=lambda nu: ...` for a frequency-dependent ratio;
`boundary="periodic"` for a circular blur; `wiener(data, psf, ...)` as a
shortcut. See the [API reference](../../reference/deconv.md).

## References

!!! references "References"

    1. Gonzalez RC, Woods RE. *Digital Image Processing*, 4th ed., chapter
       "Image Restoration and Reconstruction". Pearson, 2018. (No DOI.)
    2. Bertero M, Boccacci P. *Introduction to Inverse Problems in Imaging.*
       IOP Publishing, 1998. (No DOI.)

    ??? quote "BibTeX"

        ```bibtex
        @book{gonzalez2018digital,
          author    = {Gonzalez, Rafael C. and Woods, Richard E.},
          title     = {Digital Image Processing},
          edition   = {4},
          publisher = {Pearson},
          year      = {2018},
        }
        @book{bertero1998introduction,
          author    = {Bertero, Mario and Boccacci, Patrizia},
          title     = {Introduction to Inverse Problems in Imaging},
          publisher = {IOP Publishing},
          year      = {1998},
        }
        ```
