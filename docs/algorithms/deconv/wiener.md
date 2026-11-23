# Wiener

!!! info "At a glance"
    **Estimates:** deconvolved image (the latent image $\mathbf{x}$, same shape as the data) \
    **Noise model:** Gaussian (additive, stationary) \
    **Streaming:** halo \
    **Needs many frames:** no \
    **Reference:** Gonzalez and Woods, *Digital Image Processing*; Bertero and Boccacci, *Introduction to Inverse Problems in Imaging*, 1998 (textbooks, no DOI)

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
additive noise (see [Image formation and noise](../../concepts/image-formation.md)):

=== "optix"

    $$
    \mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon}
    = \mathbf{h} * \mathbf{x} + \boldsymbol{\varepsilon}
    $$

=== "Gonzalez & Woods"

    $$
    g = H f + \eta = h * f + \eta
    $$

| optix | Gonzalez & Woods |
|---|---|
| $\mathbf{y}$ (recorded image) | $g$ |
| $\mathbf{x}$ (latent image) | $f$ |
| $\mathbf{h}$, $\mathbf{A}$ (PSF, blur) | $h$, $H$ |
| $\boldsymbol{\varepsilon}$ (noise) | $\eta$ |
| $\hat{h}(\boldsymbol{\nu})$ (OTF) | $H(u, v)$ |
| $\boldsymbol{\nu}$ (frequency, cycles per pixel) | $(u, v)$ |

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

| optix | Gonzalez & Woods |
|---|---|
| $\hat{\mathbf{x}}(\boldsymbol{\nu})$, $\hat{\mathbf{y}}(\boldsymbol{\nu})$ | $\hat{F}(u, v)$, $G(u, v)$ |
| $S_x$, $S_\varepsilon$ | $S_f$, $S_\eta$ |
| $K$ (`nsr`) | $S_\eta / S_f$ (often a constant $K$) |

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

```text
Input: data Y, PSF h (unit sum), nsr = 0.01, boundary = "taper"
For each block (core + halo ≥ PSF size, default twice):
    extend the block smoothly to an FFT grid (taper), or not (periodic)
    R(ν) = ĥ*(ν) / (|ĥ(ν)|² + K(ν))
    x = IFFT(R · FFT(y)), cropped to the core
Return x
```

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
from optix.deconv.utils import PaddedConvolution

torch.manual_seed(0)

# 1. A Gaussian PSF and a blurred, noisy test pattern.
r = torch.arange(-7, 8, dtype=torch.float32)
psf = torch.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * 2.0**2))
psf = psf / psf.sum()
x = torch.full((96, 96), 5.0)
x[20:76:8, 20:76] = 200.0
op = PaddedConvolution(psf, (96, 96))
y = torch.poisson(op.forward(op.pad(x)))

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
