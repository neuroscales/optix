# sCMOS calibration

!!! info "At a glance"
    - **Estimates:** `SensorEstimate` (`offset`, `variance`, `gain`, and `defective` with `confidence=`)
    - **Noise model:** Poisson-Gaussian, one set of parameters per pixel
    - **Streaming:** per-pixel-stats
    - **Needs many frames:** yes
    - **Reference:** Huang F et al., Nat Methods, 2013, [doi:10.1038/nmeth.2488](https://doi.org/10.1038/nmeth.2488)

## Overview

[`SCMOSCalibration`](../../reference/sensor.md) measures, for every pixel of an
sCMOS camera, its offset $o_p$, its read-noise variance $\sigma^2_p$ and its gain
$g_p$, following the camera characterisation of
[Huang et al. 2013 (Supplementary Note, §2)](https://doi.org/10.1038/nmeth.2488).
It needs a stack of dark frames and a few stacks of uniform illumination at
different levels. Its `apply` converts data to photoelectrons,
$(\mathbf{y} - \mathbf{o}) / \mathbf{g}$.

Use it when the gain and read noise vary from pixel to pixel (sCMOS) and you can
record calibration stacks. For a camera with one gain and one read noise, or
when only the data themselves are available, a global noise-model fit such as
the photon transfer curve or a Poisson-Gaussian fit is enough.

## Image formation (optics)

The camera model of [Image formation and noise](../../concepts/image-formation.md),
with one value per pixel. The illumination stacks only need to be uniform *in
time* at each pixel: the calibration is per pixel, so spatial non-uniformity of
the light does not matter. Within one level, every frame of a pixel has the same
expected number of photoelectrons $\lambda_p^k$.

## Probabilistic model

Each pixel $p$ of each frame follows

$$
y_p = g_p\, n_p + o_p + \varepsilon_p, \qquad
n_p \sim \mathrm{Poisson}(\lambda_p), \qquad
\varepsilon_p \sim \mathcal{N}(0, \sigma^2_p),
$$

so that, over the frames of one illumination level $k$,

$$
\mathbb{E}[y_p] = g_p \lambda_p^k + o_p, \qquad
\mathrm{Var}[y_p] = g_p^2 \lambda_p^k + \sigma^2_p
= g_p\,\bigl(\mathbb{E}[y_p] - o_p\bigr) + \sigma^2_p .
$$

The variance excess is linear in the mean excess, with slope $g_p$ and no
intercept. In vector form, with $\bar{\mathbf{y}}^k$ and $\mathbf{v}^k$ the
temporal mean and variance at level $k$:

=== "optix"

    $$
    \mathbf{o} = \frac{1}{M}\sum_{m=1}^{M} \mathbf{y}_m^{\text{dark}}, \qquad
    \boldsymbol{\sigma}^2 = \frac{1}{M - \delta}\sum_{m=1}^{M} \bigl(\mathbf{y}_m^{\text{dark}} - \mathbf{o}\bigr)^2,
    $$

    $$
    \hat{g}_p = \arg\min_{g} \sum_{k=1}^{K}
    \Bigl( (v_p^k - \sigma^2_p) - g\,(\bar{y}_p^k - o_p) \Bigr)^2
    = \frac{\sum_k (\bar{y}_p^k - o_p)(v_p^k - \sigma^2_p)}{\sum_k (\bar{y}_p^k - o_p)^2}
    $$

=== "Huang et al. 2013"

    $$
    o = \frac{1}{M}\sum_{m=1}^{M} s^m, \qquad
    \mathrm{var} = \frac{1}{M}\sum_{m=1}^{M} (s^m)^2 - o^2
    \qquad \text{(Eq. 2.1, 2.2)}
    $$

    $$
    \hat{g} = \arg\min_g \sum_{k} \bigl( (v^k - \mathrm{var}) - g\,(\bar{D}^k - o) \bigr)^2
    = (B B^\top)^{-1} B A^\top
    \qquad \text{(Eq. 2.3, 2.5)}
    $$

    with $A = (v^k - \mathrm{var})_k$ and $B = (\bar{D}^k - o)_k$.

| optix | [Huang et al. 2013](https://doi.org/10.1038/nmeth.2488) |
|---|---|
| $\mathbf{y}_m^{\text{dark}}$ | $s^m$ (dark frame $m$) |
| $\mathbf{o}$ | $o$ |
| $\boldsymbol{\sigma}^2$ | $\mathrm{var}$ |
| $\mathbf{g}$ | $g$ |
| $\bar{\mathbf{y}}^k$ | $\bar{D}^k$ |
| $\mathbf{v}^k$ | $v^k$ |
| $M$, $K$ | number of frames, number of levels |
| $\delta$ (`ddof`) | $0$ |

## Inference / derivation

Strategy: moments, then a regression. The dark stack ($\lambda_p = 0$) gives the
offset and read variance directly as its temporal mean and variance. Removing
them from the moments of each illuminated level leaves $K$ points per pixel on a
line through the origin whose slope is the gain, fitted by least squares with
equal weight per level (closed form above). Nothing is ambiguous: the dark
stack pins the intercept. The standard error of the slope is kept as
`gain_stderr`. Extensions swap the regression for a robust one (Huber
reweighting, or the Theil-Sen median of per-level slopes).

## Algorithm

```text
Input: dark stack (M frames), K illumination stacks, robust, confidence, ddof
1. Stream the dark stack once: per-pixel mean o and variance σ² (divisor M − ddof).
2. For each level k, stream its stack once: per-pixel mean ȳᵏ and variance vᵏ.
3. a_k = ȳᵏ − o,  b_k = vᵏ − σ²  (per pixel).
4. g = Σₖ a_k b_k / Σₖ a_k²   (or Huber IRLS / Theil-Sen median of b_k / a_k).
5. If confidence is set: defective = not (g finite and g > 0 and stderr/g ≤ confidence).
Return SensorEstimate(offset=o, variance=σ², gain=g, defective)
```

The paper used 60,000 dark frames and 15 levels of 20,000 frames, from about 20
to 200 photons per pixel.

## Faithful mode and extensions

`faithful=True` reproduces the Supplementary Note: ordinary least squares, no
defect mask, variances divided by $M$.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `robust` | regression of the gain: `"none"`, `"huber"`, `"theil-sen"` | `"none"` | `"none"` |
| `confidence` | flag pixels whose relative standard error of the gain exceeds it | `None` | `None` |
| `ddof` | divisor $M - \delta$ of every temporal variance | `0` | `1` |

## Memory and streaming

Each stack is read once, in frame batches, with Welford/Pébay running moments
(`per-pixel-stats`). Memory is a few maps of shape `(*group, *spatial)` per
level, independent of the number of frames, so stacks of tens of thousands of
frames can be `zarr` arrays. Every level needs at least two samples per group.

## Usage

```python
import torch
from optix.models import random_sensor, simulate_sensor
from optix.sensor import SCMOSCalibration

truth = random_sensor((32, 32), gain=0.5, gain_sd=0.05, seed=0)
dark = simulate_sensor(torch.zeros(500, 32, 32), truth, seed=1)
levels = [
    simulate_sensor(torch.full((500, 32, 32), lam), truth, seed=2 + k)
    for k, lam in enumerate([20.0, 60.0, 120.0, 200.0])
]

est = SCMOSCalibration().fit(levels, dark=dark)
sensor = est.estimates["sensor"]
rel_err = ((sensor.gain - truth.gain).abs() / truth.gain).mean()  # about 0.04
electrons = est.apply(levels[0])  # about 20 e⁻ per pixel
```

Variations: `SCMOSCalibration(faithful=False)` for unbiased variances,
`robust="huber"` when some levels contain outliers, `confidence=0.1` to flag
pixels whose gain is poorly determined.

## References

!!! references "References"

    1. Huang F, Hartwich TMP, Rivera-Molina FE, Lin Y, Duim WC, Long JJ,
       Uchil PD, Myers JR, Baird MA, Mothes W, Davidson MW, Toomre D,
       Bewersdorf J. *Video-rate nanoscopy using sCMOS camera-specific
       single-molecule localization algorithms.* Nat Methods
       **10**(7):653-658, 2013.
       [doi:10.1038/nmeth.2488](https://doi.org/10.1038/nmeth.2488)

    ??? quote "BibTeX"

        ```bibtex
        @article{huang2013video,
          author  = {Huang, Fang and Hartwich, Tobias M. P. and Rivera-Molina, Felix E. and Lin, Yu and Duim, Whitney C. and Long, Jane J. and Uchil, Pradeep D. and Myers, Jordan R. and Baird, Michelle A. and Mothes, Walther and Davidson, Michael W. and Toomre, Derek and Bewersdorf, Joerg},
          title   = {Video-rate nanoscopy using {sCMOS} camera-specific single-molecule localization algorithms},
          journal = {Nature Methods},
          volume  = {10},
          number  = {7},
          pages   = {653--658},
          year    = {2013},
          doi     = {10.1038/nmeth.2488},
        }
        ```
