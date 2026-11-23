# Wavelet-FFT

!!! info "At a glance"
    - **Estimates:** stripe-free image
    - **Noise model:** none
    - **Streaming:** per-frame
    - **Needs many frames:** no
    - **Reference:** Münch et al., Opt Express, 2009, [doi:10.1364/OE.17.008567](https://doi.org/10.1364/OE.17.008567)

## Overview

`WaveletFFT` removes straight stripes from images, such as the shadows cast
along the light sheet in light-sheet microscopy, with the combined
wavelet–Fourier filter of [Münch et al. (2009)](https://doi.org/10.1364/OE.17.008567).
It is a fixed linear filter, applied to every frame independently: there is
nothing to fit, so `fit` only records the layout and `apply` returns the
destriped frames. It suits stripes that run the whole frame in one known
direction; it does not model their physical cause (attenuation, absorption), and
it slightly smooths genuine structure aligned with the stripes.

## Image formation (optics)

Stripes are additive offsets that are constant along one axis of the frame, the
stripe axis, over the whole frame ([Münch et al. 2009, Eq. 8](https://doi.org/10.1364/OE.17.008567)).
In the frequency domain along that axis, their energy sits on the zero frequency;
in a wavelet decomposition, it sits in the detail bands that are low-pass along
the stripes and high-pass across them. Shading and sensor effects are not
modelled (see [Image formation and noise](../../concepts/image-formation.md)).

## Probabilistic model

There is no noise model; the filter rests on a deterministic signal model:

=== "optix"

    $$
    \mathbf{y} = \mathbf{x} + \mathbf{s},
    \qquad s(r_1, r_2) = s(r_2),
    $$

    a frame $\mathbf{y}$ is the stripe-free image $\mathbf{x}$ plus stripes
    $\mathbf{s}$ constant along the stripe axis $r_1$ (the first spatial axis by
    default).

=== "Münch et al. 2009"

    $$
    f(x, y) = f_0(x, y) + s(x),
    $$

    stripes along $y$: offsets depending on $x$ only (Eq. 8).

| optix | [Münch et al. 2009](https://doi.org/10.1364/OE.17.008567) |
|---|---|
| $\mathbf{y}$ (recorded frame) | $f$ |
| $\mathbf{x}$ (stripe-free) | $f_0$ |
| $\mathbf{s}$ | $s$ |
| $r_1$, $r_2$ (stripe axis, across) | $y$, $x$ |
| $\mathbf{c}_l$, $\hat{\mathbf{c}}_l$ | $c_{v,l}$, $\hat c_{v,l}$ |
| $\omega$ | $\hat y$ |
| $L$, $\sigma$ | `decNum`, $\sigma$ |

## Inference / derivation

The strategy is to isolate the stripes in two steps and suppress only where they
are concentrated. A wavelet decomposition separates scales and directions, so the
stripe information of width up to about $2^L$ pixels lands in the vertical detail
bands $\mathbf{c}_l$, $l = 1, \ldots, L$; a 1-D Fourier transform along the stripe
axis then condenses it onto the zero-frequency line of each band, which is damped
by a Gaussian band-stop:

=== "optix"

    $$
    \hat{\mathbf{c}}'_l(r_2, \omega) = \hat{\mathbf{c}}_l(r_2, \omega)
    \left(1 - e^{-\omega^2 / (2\sigma^2)}\right)
    $$

=== "Münch et al. 2009"

    $$
    \hat c'_{v,l}(x, \hat y) = \hat c_{v,l}(x, \hat y)
    \left(1 - e^{-\hat y^2 / (2\sigma^2)}\right)
    \quad \text{(Eq. 9)}
    $$

The approximation and the other detail bands are left untouched, so the image
content outside the stripe direction and scale is preserved exactly. $\sigma = 0$
is the limit of ideal stripes: only $\omega = 0$ is removed.

## Algorithm

```text
Input: frame y, wavelet = db10, L = 4, sigma = 1.5
1. multilevel 2-D DWT of y down to level L (symmetric boundary extension)
2. for each chosen level l: FFT of the vertical detail band along the stripe axis,
   multiply by 1 − exp(−ω²/(2σ²)), inverse FFT
3. inverse DWT
Return the destriped frame
```

The FFT runs over the whole band, without padding or windowing (the paper
specifies none). The Daubechies filters db1–db45 are tabulated in optix.

## Faithful mode and extensions

The paper leaves its three parameters to be tuned to the data (Section 4.4); the
defaults are its suggestions for typical data (Section 3.8), and any value is
faithful. `faithful=False` uses the same defaults.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `wavelet` | `"dbN"` (N = 1…45) or `"haar"`; db10–db15 recommended (Section 3.8) | `"db10"` | `"db10"` |
| `level` | highest level $L$, sets the widest stripes removed (Section 3.6) | 4 | 4 |
| `sigma` | damping width $\sigma \ge 0$, one value or one per level, finest first | 1.5 | 1.5 |
| `axis` | stripe axis: position among the two spatial axes, or a layout label | -2 (first spatial axis) | -2 |
| `levels` | detail levels to filter, 1 the finest (Section 3.6) | `None` (all) | `None` |

Heavier or wider stripes need a larger $L$ or $\sigma$ (the paper uses
$\sigma = 7$, $L = 8$ for its FIB-nt image, Section 4.1).

## Memory and streaming

Frames are processed independently, in batches sized from the memory budget
(per-frame streaming), so arrays larger than memory (`zarr`, memmap) can be
filtered, and `apply` works in place. The cost is $O(PF)$ per frame of $P$ pixels
with a filter of length $F$; the working memory is a few copies of a batch.

## Usage

```python
import torch
from optix.destripe import WaveletFFT

torch.manual_seed(0)

# 1. A frame with stripes along the first axis (one offset per column).
x = torch.rand(128, 128) * 10 + 100
stripes = 5 * torch.randn(128)
y = (x + stripes[None, :])[None]  # (frames, H, W)

# 2. Filter (nothing is fitted).
y_hat = WaveletFFT().fit(y).apply(y)

# 3. Residual stripes, before and after.
print(float((y[0] - x).std()), float((y_hat[0] - x).std()))
```

Common variations: `axis=-1` for horizontal stripes, `level=` and `sigma=` for
wider or stronger stripes, and the function
`optix.destripe.wavelet_fft(tensor, ...)` for a tensor `(..., H, W)` without a
layout. See the [API reference](../../reference/destripe.md).

## References

!!! references "References"

    1. Münch B, Trtik P, Marone F, Stampanoni M. *Stripe and ring artifact
       removal with combined wavelet — Fourier filtering.* Opt Express
       **17**(10):8567-8591, 2009.
       [doi:10.1364/OE.17.008567](https://doi.org/10.1364/OE.17.008567)

    ??? quote "BibTeX"

        ```bibtex
        @article{munch2009stripe,
          author  = {Münch, Beat and Trtik, Pavel and Marone, Federica and Stampanoni, Marco},
          title   = {Stripe and ring artifact removal with combined wavelet --- {F}ourier filtering},
          journal = {Optics Express},
          volume  = {17},
          number  = {10},
          pages   = {8567--8591},
          year    = {2009},
          doi     = {10.1364/OE.17.008567},
        }
        ```
