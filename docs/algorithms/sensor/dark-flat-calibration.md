# Dark/flat calibration

!!! info "At a glance"
    - **Estimates:** `ShadingEstimate` (`flat`, `dark`) and, with a dark stack, `SensorEstimate` (`offset`, `variance`)
    - **Noise model:** none
    - **Streaming:** per-pixel-stats
    - **Needs many frames:** yes
    - **Reference:** none: standard practice, optix convention

## Overview

[`DarkFlatCalibration`](../../reference/shading.md) is prospective flat-field
correction: record dark frames (shutter closed) and flat frames (a uniform
sample), average them per pixel, and divide the data by the dark-subtracted
flat. It is exported from both `optix.sensor` and `optix.shading`. It fills a
`ShadingEstimate`, so its result can be used anywhere a retrospective shading
estimate (BaSiC, CIDRE) would be, and, when given a dark stack, a
`SensorEstimate` with the offset and read variance.

Use it when reference stacks can be recorded under the same conditions as the
data. When they cannot, estimate the shading retrospectively from the data.

## Image formation (optics)

The shading model of [Image formation and noise](../../concepts/image-formation.md),
without baseline. The flat reference is a sample with $\mathbf{x}$ uniform, so
its mean is $\mathbf{f}\,c + \mathbf{d}$; the dark reference has $\mathbf{x} = 0$
and measures $\mathbf{d}$: camera offset, dark current and stray light. As the
[physical-mapping table](../../concepts/notation.md#what-the-fields-mean-physically)
explains, the flat measured this way includes the per-pixel gain and quantum
efficiency of the camera, so it overlaps with the gain map of an sCMOS
calibration: do not correct by both.

## Probabilistic model

There is no noise model: the references are reduced per pixel and the noise is
assumed to average out. With $P$ the number of pixels, frames
$\mathbf{y}_i, \mathbf{x}_i \in \mathbb{R}^P$ and $M$ frames per reference stack,

$$
\mathbf{y}_i = \mathbf{f} \odot \mathbf{x}_i + \mathbf{d}, \qquad
\hat{\mathbf{d}} = \hat{\mathbf{y}}^{\text{dark}}, \qquad
\hat{\mathbf{f}} = \frac{\hat{\mathbf{y}}^{\text{flat}} - \hat{\mathbf{d}}}{\frac{1}{P}\sum_{p=1}^{P}\bigl(\hat{y}^{\text{flat}}_p - \hat{d}_p\bigr)},
$$

where $\hat{\mathbf{y}}$ is the per-pixel mean or median over the frames of a
reference stack: the sample mean
$\bar{\mathbf{y}} = \frac{1}{M}\sum_{m=1}^{M}\mathbf{y}_m$ with
`statistic="mean"`, or the sample median
$\tilde{\mathbf{y}} = \operatorname{median}_m \mathbf{y}_m$ (pixel-wise) with
`statistic="median"`. There is no paper, so no native-notation tab.

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{y}_i$ | recorded frame $i$ (ADU) | $\mathbb{R}^P$, per frame |
| $\mathbf{x}_i$ | latent image of frame $i$, without shading | $\mathbb{R}^P$, per frame |
| $\mathbf{f}$ | flat field: relative sensitivity, mean $1$ | $\mathbb{R}^P$, per pixel, shared by the frames of a group |
| $\mathbf{d}$ | dark: offset, dark current and stray light (ADU) | $\mathbb{R}^P$, per pixel, shared by the frames of a group |
| $\mathbf{y}_m^{\text{dark}}$, $\mathbf{y}_m^{\text{flat}}$ | frame $m$ of the dark and flat reference stacks | $\mathbb{R}^P$, per frame |
| $\hat{\mathbf{y}}^{\text{dark}}$, $\hat{\mathbf{y}}^{\text{flat}}$ | per-pixel mean or median of each reference stack | $\mathbb{R}^P$, per pixel |
| $P$, $M$ | numbers of pixels and of reference frames | scalars, whole stack |

## Inference / derivation

The correction is
$\hat{\mathbf{x}}_i = (\mathbf{y}_i - \hat{\mathbf{d}}) / \hat{\mathbf{f}}
= \dfrac{\mathbf{y}_i - \hat{\mathbf{d}}}{\hat{\mathbf{y}}^{\text{flat}} - \hat{\mathbf{d}}}\,\frac{1}{P}\sum_p \bigl(\hat{y}^{\text{flat}}_p - \hat{d}_p\bigr)$
(pixel-wise division). Statistically, the mean reduction is the method-of-moments
(and Gaussian maximum-likelihood) estimate of each pixel's expected value, and
the median is its robust counterpart, the maximum-likelihood estimate under
Laplace noise, insensitive to a few outlying frames.
The overall scale shared by $\mathbf{f}$ and $\mathbf{x}_i$ is fixed by dividing
the flat by its spatial mean (per group), which keeps corrected intensities in
ADU; `meta["flat_norm"]` records `"mean"` or `"none"`.

## Algorithm

**Input:** flat frames $\mathbf{y}_1^{\text{flat}}, \ldots, \mathbf{y}_M^{\text{flat}} \in \mathbb{R}^P$;
`dark` (a stack, a map, a `ShadingEstimate`, a `SensorEstimate` or none);
`statistic="mean"`, `normalize="mean"`. \
**Output:** `ShadingEstimate` with $\hat{\mathbf{f}}$ and $\hat{\mathbf{d}}$ and, for
a dark stack, `SensorEstimate` with offset $\hat{\mathbf{d}}$ and read variance.

1. **Flat reduction.** $\hat{\mathbf{y}}^{\text{flat}}$ is the per-pixel mean
   $\bar{\mathbf{y}}$ or median $\tilde{\mathbf{y}}$ of the flat stack.
2. **Dark.** $\hat{\mathbf{d}}$ is the same statistic of the dark stack; or the
   given map; or `ShadingEstimate.dark`; or `SensorEstimate.offset`; or $\mathbf{0}$.
3. **Flat.** $\hat{\mathbf{f}} = \hat{\mathbf{y}}^{\text{flat}} - \hat{\mathbf{d}}$;
   if `normalize="mean"`, divide it by its spatial mean (per group).
4. **Sensor** (dark stack only): the read variance is the per-pixel temporal
   variance of the dark stack.

## Faithful mode and extensions

There is no paper: `faithful=True` and `faithful=False` both use the optix
defaults.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `statistic` | per-pixel reduction: `"mean"` or `"median"` (robust to cosmic rays and moving dust) | `"mean"` | `"mean"` |
| `normalize` | divide the flat by its spatial mean, or not (`"none"`) | `"mean"` | `"mean"` |

## Memory and streaming

With `"mean"`, each stack is streamed once in frame batches (Welford/Pébay
moments); memory is a few `(*group, *spatial)` maps. With `"median"`, the exact
per-pixel median is computed over row tiles, so each tile of the full stack is
held in memory at once.

## Usage

```python
import torch
from optix.models import random_sensor, simulate_sensor
from optix.sensor import DarkFlatCalibration

camera = random_sensor((32, 32), seed=0)
illumination = torch.linspace(0.5, 1.0, 32).expand(32, 32)
darks = simulate_sensor(torch.zeros(500, 32, 32), camera, seed=1)
flats = simulate_sensor(1000 * illumination.expand(50, 32, 32), camera, seed=2)

cal = DarkFlatCalibration().fit(flats, dark=darks)
shading = cal.estimates["shading"]  # flat has mean 1
sensor = cal.estimates["sensor"]  # offset and read variance
corrected = cal.apply(flats)  # about uniform
```

Variations: pass a precomputed dark map, or a `SensorEstimate` from an
[sCMOS calibration](scmos-calibration.md) (`dark=sensor`), in which case only
`"shading"` is returned; `statistic="median"` for flats with dust or cosmic
rays.

## References

No publication defines dark/flat correction; it is standard practice, and the
reductions and normalisation above are documented optix choices.
