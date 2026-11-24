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
it slightly smooths genuine structure aligned with the stripes. The
options of pystripe ([Swaney et al. 2019](https://doi.org/10.1101/576595)), such
as separate bandwidths for background and foreground, are available as opt-in
[extensions](#pystripe-extensions-opt-in-not-faithful).

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

    a frame $\mathbf{y} \in \mathbb{R}^{L_1\times L_2}$ ($L_1$ pixels along
    the stripe axis, $L_2$ across) is the stripe-free image $\mathbf{x}$ of
    the same shape plus stripes
    $\mathbf{s}$ constant along the stripe axis $r_1$ (the first spatial axis by
    default).

=== "Münch et al. 2009"

    $$
    f(x, y) = f_0(x, y) + s(x),
    $$

    stripes along $y$: offsets depending on $x$ only (Eq. 8).

| optix | [Münch et al. 2009](https://doi.org/10.1364/OE.17.008567) | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}$ | $f$ | recorded frame | $\mathbb{R}^{L_1\times L_2}$, per frame |
| $\mathbf{x}$ | $f_0$ | stripe-free frame | $\mathbb{R}^{L_1\times L_2}$, per frame |
| $\mathbf{s}$ | $s$ | stripe offsets, constant along $r_1$ | $\mathbb{R}^{L_2}$, per frame |
| $r_1$, $r_2$ | $y$, $x$ | pixel index along and across the stripes | scalars |
| $\mathbf{c}_l$ | $c_{v,l}$ | vertical detail band at level $l$ | about $\mathbb{R}^{L_1/2^l\times L_2/2^l}$, per frame |
| $\hat{\mathbf{c}}_l$ | $\hat c_{v,l}$ | its 1-D FFT along the stripe axis | same shape as $\mathbf{c}_l$, complex |
| $\omega$ | $\hat y$ | frequency along the stripe axis | scalar |
| $L$ | `decNum` | number of wavelet levels | scalar, whole stack |
| $\sigma$ | $\sigma$ | width of the Gaussian band-stop | scalar or one per level, whole stack |

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

Nothing is estimated statistically: the filter is a fixed linear projection.
For $\sigma = 0$ it is the least-squares estimate of $\mathbf{s}$ restricted
to the chosen bands (the mean of each band column along the stripe axis is
taken as stripe and subtracted); $\sigma > 0$ softens this projection into a
shrinkage of the low frequencies along the stripes.

## Algorithm

**Input:** a frame $\mathbf{y} \in \mathbb{R}^{L_1\times L_2}$;
`wavelet = "db10"`; $L = 4$; $\sigma = 1.5$ \
**Output:** the destriped frame $\hat{\mathbf{x}}$, same shape.

1. **Decompose** $\mathbf{y}$ by a multilevel 2-D DWT down to level $L$
   (symmetric boundary extension).
2. **For each chosen level** $l$:
    1. FFT the vertical detail band $\mathbf{c}_l$ along the stripe axis;
    2. multiply by $1 - e^{-\omega^2/(2\sigma^2)}$;
    3. invert the FFT.
3. **Reconstruct** $\hat{\mathbf{x}}$ by the inverse DWT.

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

### PyStripe extensions (opt-in, not faithful)

[pystripe](https://github.com/chunglabmit/pystripe) is a Python implementation
of the Münch filter by [Swaney et al. (2019)](https://doi.org/10.1101/576595),
for light-sheet microscopy. `WaveletFFT` offers its options as opt-in
extensions. They are off by default, and any of them makes the filter
non-faithful (`is_faithful` is `False`). The paper only names the dual-band
mode, the crossover and the flat-field correction, and gives no equation for
them. Every item below is therefore **from the pystripe code (MIT licence), not
in the paper**, and the equations are those the code implements, with no
counterpart in the paper's notation to tabulate. The code of `optix` that
follows pystripe is adapted from it, under its licence (Copyright (c) 2018 Chung
Lab).

Write $W(\mathbf{y}; \sigma)$ for the Münch filter above with bandwidth
$\sigma$ (one value, or one per level), and $W_0$ for the same filter when it is
the only one in play.

- **Log domain** (`log_transform=True`), from the pystripe code, not in the
  paper. The filter runs on $\log(1 + \mathbf{y})$:

    $$
    W(\mathbf{y}; \sigma) = \exp\!\big(W_0(\log(1 + \mathbf{y}); \sigma)\big) - 1 .
    $$

    It is non-linear: stripes become multiplicative.

- **Dual band** (`bands`), from the pystripe code, not in the paper. A
  threshold $t$ (Otsu's threshold of each frame,
  [`otsu_threshold`][optix.stats.otsu_threshold], unless `threshold` is given)
  splits the frame into a background $\min(\mathbf{y}, t)$ and a foreground
  $\max(\mathbf{y}, t)$. They are filtered with their own bandwidths
  $\sigma_{\text{bg}}$ (`sigma_background`) and $\sigma_{\text{fg}}$ (`sigma`),
  then blended with the foreground fraction $\boldsymbol{\chi}$:

    $$
    \boldsymbol{\chi} = G_{s} * \operatorname{sigmoid}\!\Big(\frac{\mathbf{y} - t}{c}\Big),
    \qquad
    \hat{\mathbf{x}} = \boldsymbol{\chi} \odot W(\max(\mathbf{y}, t); \sigma_{\text{fg}})
    + (1 - \boldsymbol{\chi}) \odot W(\min(\mathbf{y}, t); \sigma_{\text{bg}}) .
    $$

    Here $c$ is the `crossover` (the intensity range of the transition), and
    $G_s$ a Gaussian of width $s$ pixels (`smoothing`, 1 in pystripe; mirror
    boundaries, truncated at $\lfloor 4s + 0.5 \rfloor$ samples as in
    `scipy.ndimage.gaussian_filter`). With `bands="foreground"` the background
    term is $\mathbf{y}$ itself (only the foreground is filtered), and with
    `bands="background"` the foreground term is. With `bands="dual"` and
    $\sigma_{\text{fg}} = \sigma_{\text{bg}}$ the frame is not split: the result is
    the single-band filter, as in pystripe.

- **Bandwidth mapping** (`sigma_units="pystripe"`), from the pystripe code, not
  in the paper. With the default `"band"`, $\sigma$ counts frequency samples of
  each band, as in Eq. (9). With `"pystripe"` it is given in pixels of the
  frame, and level $l$ uses $\sigma_l = \sigma\, n_l / n$, with $n_l$ and $n$ the
  sizes of the band and of the frame across the stripes. The damping of Eq. (9)
  is evaluated on the index $j = 0, \ldots, m - 1$ of a real FFT of length $m$
  stored in the packed order of `scipy.fftpack.rfft`: the real part of the
  frequency $k$ is damped by $g(2k - 1)$ and its imaginary part by $g(2k)$,
  with $g(j) = 1 - e^{-j^2/(2\sigma_l^2)}$.

- **Even padding** (`pad_even=True`), from the pystripe code, not in the paper.
  Frames of odd size are padded to even size by repeating their last row or
  column, and cropped back after filtering. The threshold is computed before
  padding.

- **Flat field and dark offset** (`flat`, `dark`), from the pystripe code, not in
  the paper. After destriping,

    $$
    \hat{\mathbf{x}} \leftarrow (\hat{\mathbf{x}} - \mathbf{d}) \oslash \mathbf{f},
    $$

    with the flat field $\mathbf{f}$ and the dark offset $\mathbf{d}$ of a
    [`ShadingEstimate`](../../reference/estimates.md) given to `fit` as `init`
    (`wavelet_fft` takes them as `flat` and `dark`). `flat` is used as given:
    pystripe's command line divides it by its maximum first, so pass
    $\mathbf{f} / \max \mathbf{f}$ to get its result. The per-frame baseline of
    the estimate is not used.

- **Floating-point output.** pystripe clips its result to $[0, 2^{16} - 1]$ and
  casts it to `uint16`; here the output stays floating point, unclipped.

!!! warning "Ringing with bright signals"
    Swaney et al. note that with very bright signals the hybrid wavelet-FFT
    filter may introduce ringing artefacts. Their remedies are the dual-band
    mode, which filters the background and the foreground with separate
    bandwidths, a smaller bandwidth, or no destriping.  The paper does not say how much
    each helps on a given image.

The options map to those of the pystripe command line (and of its function
`filter_streaks`) as follows. The paper's examples are `-s1 256 -s2 256 -w db10
-c 3 -x 10` at full resolution, and 32 for the two bandwidths at 8× downsampling,
with `sigma_units="pystripe"` for both.

| pystripe | optix | Default in optix | Note |
|---|---|---|---|
| `-s1`, `sigma1` (foreground bandwidth, pixels) | `sigma` | 1.5 (`"band"` units) | with `sigma_units="pystripe"` for pystripe's meaning |
| `-s2`, `sigma2` (background bandwidth, pixels) | `sigma_background` | `None` | needed by `bands="dual"` and `"background"` |
| dual-band filtering | `bands` | `"single"` | `"dual"`, `"foreground"` or `"background"` |
| `-w`, wavelet | `wavelet` | `"db10"` | pystripe's default is `"db3"` |
| `level` | `level` | 4 | pystripe's default is the maximum level: `level="max"` |
| `-x`, crossover | `crossover` | 10 | only used by the dual-band modes |
| threshold (Otsu by default) | `threshold` | `None` | `None` is Otsu's threshold of each frame |
| smoothing of the fraction (1 pixel) | `smoothing` | 1 | `0` for none |
| log transform | `log_transform` | `False` | pystripe filters $\log(1 + \mathbf{y})$ |
| frame padded to even size | `pad_even` | `False` | |
| stripe direction (horizontal stripes) | `axis=-1` | `-2` | optix defaults to the paper's direction |
| `-f`, `--flat` | `flat`, or `ShadingEstimate.flat` | `None` | used as given, not divided by its maximum |
| `-d`, `--dark` | `dark`, or `ShadingEstimate.dark` | `None` | an image or a scalar in `wavelet_fft`, a map in the estimate |
| `-c`, compression | not applicable | | file output option of pystripe |
| cast to `uint16` | not applied | | the output stays floating point |

With pystripe's settings (db3, `level="max"`, `log_transform=True`,
`sigma_units="pystripe"`, `pad_even=True`, `axis=-1`), the filter matches
pystripe's `filter_streaks` up to the final cast. In the example, bright cells
make the faithful filter ring (its error exceeds that of the unfiltered frame),
and a smaller foreground bandwidth in dual-band mode reduces it:

```pycon
>>> import torch
>>> from optix.destripe import WaveletFFT
>>> from optix.estimates import ShadingEstimate
>>> from optix.models import phantoms
>>> shape = (1, 128, 128)
>>> x = phantoms.cells(shape, intensity=500.0, seed=0)
>>> x = x + phantoms.background(shape, level=100.0, seed=1)
>>> y = x + phantoms.stripes(shape, strength=10.0, seed=2)
>>> def error(estimator):
...     return round(float((estimator.fit(y).apply(y) - x).std()), 1)
>>> round(float((y - x).std()), 1)  # unfiltered
8.9
>>> error(WaveletFFT())  # the paper's filter
14.5
>>> dual = WaveletFFT(
...     faithful=False, bands="dual", sigma=0.5, sigma_background=1.5
... )
>>> error(dual)
10.6
>>> dual.is_faithful
False
```

A flat field and a dark offset are given as a `ShadingEstimate`:

```pycon
>>> shading = ShadingEstimate(
...     flat=torch.full((128, 128), 0.5), dark=torch.full((128, 128), 20.0)
... )
>>> corrected = WaveletFFT().fit(y, init=shading).apply(y)
>>> corrected.dtype  # floating point, as is the input
torch.float32
```

## Memory and streaming

Frames are processed independently, in batches sized from the memory budget
(per-frame streaming), so arrays larger than memory (`zarr`, memmap) can be
filtered, and `apply` works in place. The cost is $O(\lvert\Omega\rvert F)$ per frame of $\lvert\Omega\rvert$ pixels
with a filter of length $F$; the working memory is a few copies of a batch.

## Usage

The example adds stripes to a [phantom](../../reference/models.md#phantoms) of cells on a smooth background.

```python
from optix.destripe import WaveletFFT
from optix.models import phantoms

# 1. Cells on a background (the truth), plus stripes along the first axis.
shape = (1, 128, 128)  # (frames, H, W)
x = phantoms.cells(shape, intensity=50.0, seed=0)
x = x + phantoms.background(shape, level=100.0, seed=1)
y = x + phantoms.stripes(shape, strength=10.0, seed=2)

# 2. Fit (nothing is learned) and 3. filter.
y_hat = WaveletFFT().fit(y).apply(y)

# 4. Compare to the truth: error std about 8.9 before, 3.5 after.
print(float((y - x).std()), float((y_hat - x).std()))
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
    2. Swaney J, Kamentsky L, Evans NB, Xie K, Park Y-G, Drummond G, Yun DH,
       Chung K. *Scalable image processing techniques for quantitative analysis
       of volumetric biological images from light-sheet microscopy.* bioRxiv
       576595, 2019.
       [doi:10.1101/576595](https://doi.org/10.1101/576595). Code:
       [pystripe](https://github.com/chunglabmit/pystripe) (MIT licence).

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
        
        @article{swaney2019scalable,
          author  = {Swaney, Justin and Kamentsky, Lee and Evans, Nicholas B. and Xie, Katherine and Park, Young-Gyun and Drummond, Gabrielle and Yun, Dae Hee and Chung, Kwanghun},
          title   = {Scalable image processing techniques for quantitative analysis of volumetric biological images from light-sheet microscopy},
          journal = {bioRxiv},
          pages   = {576595},
          year    = {2019},
          doi     = {10.1101/576595},
        }
        ```
