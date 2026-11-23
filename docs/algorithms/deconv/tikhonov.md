# Tikhonov

!!! info "At a glance"
    - **Estimates:** deconvolved image (the latent image $\mathbf{x}$, same shape as the data)
    - **Noise model:** Gaussian (least squares)
    - **Streaming:** halo
    - **Needs many frames:** no
    - **Reference:** Gonzalez and Woods, *Digital Image Processing*; Bertero and Boccacci, *Introduction to Inverse Problems in Imaging*, 1998 (textbooks, no DOI)

## Overview

`Tikhonov` deconvolves an image with a known PSF by regularised least
squares, solved in closed form by one FFT filter per block. The penalty is
either the energy of the image (zero order, identity operator; the same filter
as [Wiener](wiener.md) with a constant noise-to-signal ratio) or its
roughness (Laplacian operator, the constrained least-squares filter). As an
extension, the regularisation weight can be chosen automatically by
generalised cross-validation. Like Wiener, it is fast and linear, so the
result may be negative; use [Richardson–Lucy](richardson-lucy.md) or
[Wiener–Butterworth](wiener-butterworth.md) for a non-negative estimate under
Poisson noise.

## Image formation (optics)

As for [Wiener](wiener.md#image-formation-optics): a linear, shift-invariant
blur $\mathbf{A}$ by a known PSF $\mathbf{h}$ of unit sum, plus additive noise
(see [Image formation and noise](../../concepts/image-formation.md)).

=== "optix"

    $$
    \mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon}
    $$

=== "Bertero & Boccacci 1998"

    $$
    g = A f + w
    $$

| optix | Bertero & Boccacci 1998 |
|---|---|
| $\mathbf{y}$ (recorded image) | $g$ |
| $\mathbf{x}$ (latent image) | $f$ |
| $\mathbf{A}$ (blur), $\hat{h}$ (OTF) | $A$, $\hat{K}$ |
| $\boldsymbol{\varepsilon}$ (noise) | $w$ |
| $\lambda$ | $\mu$ |

The book's $f$ is the object, not the optix flat field. The class docstring
writes $y = Hf + n$ with $S$ the PSF and $\lambda$ the weight.

## Probabilistic model

Assumptions: the PSF is known and shift-invariant; the noise is additive and
white (so least squares is the right data term); the object is a priori small
(identity) or smooth (Laplacian); the image is periodic (FFT model, with the
boundary handling of [Wiener](wiener.md#inference-derivation)). The estimate
minimises

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}}\;
\lVert \mathbf{y} - \mathbf{A}\mathbf{x} \rVert^2 + \lambda \lVert \mathbf{L}\mathbf{x} \rVert^2 ,
$$

with $\mathbf{L}$ the identity or the discrete Laplacian (the $2d+1$-point
stencil), equivalently the MAP estimate under a Gaussian likelihood and a
Gaussian prior $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, (\lambda\mathbf{L}^\top\mathbf{L})^{-1})$
(up to the noise variance). No non-negativity is imposed.

## Inference / derivation

Both $\mathbf{A}$ and $\mathbf{L}$ are convolutions, diagonal in the Fourier
basis, so the normal equations
$(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{L}^\top\mathbf{L})\mathbf{x} = \mathbf{A}^\top\mathbf{y}$
decouple over frequencies and are solved in one step:

=== "optix"

    $$
    \hat{\mathbf{x}}(\boldsymbol{\nu}) =
    \frac{\hat{h}^*(\boldsymbol{\nu})}{|\hat{h}(\boldsymbol{\nu})|^2 + \lambda |\hat{l}(\boldsymbol{\nu})|^2}\,
    \hat{\mathbf{y}}(\boldsymbol{\nu}),
    \qquad
    |\hat{l}(\boldsymbol{\nu})|^2 =
    \begin{cases}
    1 & \text{identity,}\\
    \bigl(\sum_a 4\sin^2(\pi\nu_a)\bigr)^2 & \text{Laplacian.}
    \end{cases}
    $$

=== "Bertero & Boccacci 1998"

    $$
    \hat{f}_\mu(\omega) = \frac{\hat{K}^*(\omega)}{|\hat{K}(\omega)|^2 + \mu\, |\hat{P}(\omega)|^2}\, \hat{g}(\omega).
    $$

| optix | Bertero & Boccacci 1998 |
|---|---|
| $\hat{\mathbf{x}}$, $\hat{\mathbf{y}}$ | $\hat{f}_\mu$, $\hat{g}$ |
| $\hat{h}$ | $\hat{K}$ |
| $\lambda$ (`lam`) | $\mu$ |
| $\hat{l}$ (transfer function of $\mathbf{L}$) | $\hat{P}$ |
| $\boldsymbol{\nu}$ | $\omega$ |

$\lambda \to 0$ gives the inverse filter; larger $\lambda$ trades resolution
for noise. With the Laplacian, the mean of the image ($\boldsymbol{\nu} = 0$)
is not penalised and is preserved; with the identity it is scaled by
$1/(1 + \lambda)$.

**Choosing λ (extension).** `lam="gcv"` picks $\lambda$ at `fit` by
generalised cross-validation ([Golub, Heath and Wahba 1979](https://doi.org/10.1080/00401706.1979.10489751)):
with $r(\boldsymbol{\nu}) = |\hat{h}|^2/(|\hat{h}|^2 + \lambda|\hat{l}|^2)$
the eigenvalues of the influence matrix, the score

$$
V(\lambda) = n\,\frac{\sum_{\boldsymbol{\nu}} |(1 - r)\,\hat{\mathbf{y}}|^2}{\bigl(\sum_{\boldsymbol{\nu}} (1 - r)\bigr)^2},
$$

summed over all blocks and images, is minimised over a grid of $\lambda$ from
$10^{-10}$ to $10^{2}$, four values per decade. GCV estimates the
prediction error without knowing the noise level; a choice at an end of the
grid is flagged with a warning.

## Algorithm

```text
Input: data Y, PSF h (unit sum), lam = 0.01, operator = "identity", boundary = "taper"
If lam = "gcv": scan λ on 10⁻¹⁰ … 10² (49 values), keep the minimiser of V(λ)
For each block (core + halo ≥ PSF size, default twice):
    extend the block smoothly to an FFT grid (taper), or not (periodic)
    R(ν) = ĥ*(ν) / (|ĥ(ν)|² + λ |l̂(ν)|²)
    x = IFFT(R · FFT(y)), cropped to the core
Return x; the λ used is in `strength`
```

## Faithful mode and extensions

The textbooks leave $\lambda$ to be tuned by hand; the faithful and
recommended presets coincide.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `lam` | weight $\lambda \ge 0$, or `"gcv"` (extension) | `0.01` | `0.01` |
| `operator` | $\mathbf{L}$: `"identity"` or `"laplacian"` | `"identity"` | `"identity"` |
| `boundary` | `"taper"` (smooth periodic extension) or `"periodic"` | `"taper"` | `"taper"` |
| `halo` | halo per spatial axis, at least the PSF size | `None` (twice the PSF) | `None` |
| `block` | core block size per spatial axis | `None` (memory budget) | `None` |

Extension: `fit(y, init=sensor)` converts the data to electrons first, as for
[Wiener](wiener.md#faithful-mode-and-extensions).

## Memory and streaming

As for [Wiener](wiener.md#memory-and-streaming): one pass over the data, block
by block with a halo, two FFTs per block. `lam="gcv"` adds one pass at `fit`,
over blocks without halo, with one FFT each and the score of all 49 values of
$\lambda$ accumulated in float64.

## Usage

```python
import torch
from optix.deconv import Tikhonov
from optix.models import gaussian_psf, simulate_blur

torch.manual_seed(0)

# 1. A Gaussian PSF and a blurred, noisy test pattern.
psf = gaussian_psf((15, 15), sigma=2.0)
x = torch.full((96, 96), 5.0)
x[20:76:8, 20:76] = 200.0
y = torch.poisson(simulate_blur(x, psf))

# 2. A fixed λ with each operator, and λ chosen by GCV.
for operator in ("identity", "laplacian"):
    x_t = Tikhonov(psf, lam=1e-2, operator=operator).fit(y).apply(y)
    print(operator, float((x_t - x).abs().mean()))
gcv = Tikhonov(psf, lam="gcv", operator="laplacian").fit(y)
print(gcv.strength, float((gcv.apply(y) - x).abs().mean()))
```

Common variations: `boundary="periodic"` for a circular blur;
`tikhonov(data, psf, ...)` as a shortcut. See the
[API reference](../../reference/deconv.md).

## References

!!! references "References"

    1. Gonzalez RC, Woods RE. *Digital Image Processing*, 4th ed., chapter
       "Image Restoration and Reconstruction". Pearson, 2018. (No DOI.)
    2. Bertero M, Boccacci P. *Introduction to Inverse Problems in Imaging.*
       IOP Publishing, 1998. (No DOI.)
    3. Golub GH, Heath M, Wahba G. *Generalized cross-validation as a method
       for choosing a good ridge parameter.* Technometrics **21**(2):215-223,
       1979.
       [doi:10.1080/00401706.1979.10489751](https://doi.org/10.1080/00401706.1979.10489751)

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
        @article{golub1979generalized,
          author  = {Golub, Gene H. and Heath, Michael and Wahba, Grace},
          title   = {Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter},
          journal = {Technometrics},
          volume  = {21},
          number  = {2},
          pages   = {215--223},
          year    = {1979},
          doi     = {10.1080/00401706.1979.10489751},
        }
        ```
