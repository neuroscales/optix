# Defective pixels

!!! info "At a glance"
    - **Estimates:** `SensorEstimate` (`defective`, `offset`, `variance`)
    - **Noise model:** Gaussian
    - **Streaming:** per-pixel-stats
    - **Needs many frames:** yes
    - **Reference:** none: optix convention

## Overview

[`DefectivePixels`](../../reference/sensor.md) finds hot, cold, noisy and
random-telegraph-signal (RTS) pixels from a stack of dark frames, and its
`apply` replaces them by the median of their good neighbours
([`replace_defective`](../../reference/sensor.md)). The individual masks are kept
in the `masks` attribute.

Use it before any algorithm that assumes every pixel follows the camera model.
An [sCMOS calibration](scmos-calibration.md) with `confidence=` flags pixels
whose gain is poorly determined, which is a complementary test.

## Image formation (optics)

In the dark, $\lambda = 0$ and a good pixel records
$y_p = o_p + \varepsilon_p$ (camera model of
[Image formation and noise](../../concepts/image-formation.md)): Gaussian
read noise about its offset. A defective pixel departs from this: an offset far
from the others (hot or cold, including high dark current), a large variance,
or a non-Gaussian distribution, the two-level jumps of an RTS pixel in
particular.

## Probabilistic model

Good pixels: $y_{mp} \sim \mathcal{N}(o_p, \sigma^2_p)$ for the dark frames
$m = 1 \ldots M$ and pixels $p = 1 \ldots P$, with $o_p$ and $\sigma^2_p$
similar across pixels. From the temporal moments of each pixel, the dark mean
$\mathbf{o} \in \mathbb{R}^P$, the variance $\boldsymbol{\sigma}^2$, the
skewness $\mathbf{g}_1$ and the non-excess kurtosis $\boldsymbol{\beta}_2$ (all
in $\mathbb{R}^P$), a pixel is flagged when

$$
\lvert z(o_p) \rvert > t, \qquad z(\sigma^2_p) > t, \qquad \lvert z(\beta_{2,p}) \rvert > t,
\qquad\text{or}\qquad \frac{g_{1,p}^2 + 1}{\beta_{2,p}} > b,
$$

with the robust z-score over the pixels of a group

$$
z(x_p) = \frac{x_p - \operatorname{med}_p x}{1.4826\,\operatorname{MAD}_p x}.
$$

| Symbol | Meaning | Shape |
|---|---|---|
| $y_{mp}$ | dark frame $m$ at pixel $p$ (ADU) | scalar, per pixel and frame |
| $\mathbf{o}$ | temporal mean of the dark frames: offset (ADU) | $\mathbb{R}^P$, per pixel |
| $\boldsymbol{\sigma}^2$ | temporal variance: read-noise variance (ADU²) | $\mathbb{R}^P$, per pixel |
| $\mathbf{g}_1$ | temporal skewness | $\mathbb{R}^P$, per pixel |
| $\boldsymbol{\beta}_2$ | temporal kurtosis, not excess ($3$ for a Gaussian) | $\mathbb{R}^P$, per pixel |
| $z(\cdot)$ | robust z-score across the pixels of a group | $\mathbb{R}^P$; median and MAD per group |
| $t$, $b$ | thresholds (`threshold`, `bimodality`) | scalars, whole stack |
| $M$, $P$ | numbers of dark frames and of pixels | scalars, whole stack |

The last test is Sarle's bimodality coefficient: $1/3$ for a Gaussian, $5/9$
for a uniform distribution, $1$ for a two-level (RTS) signal. There is no paper,
so no native-notation tab.

## Inference / derivation

There is nothing to infer: the four masks are threshold tests on per-pixel
moments, and `defective` is their union. Statistically, $\mathbf{o}$ and
$\boldsymbol{\sigma}^2$ are method-of-moments (Gaussian maximum-likelihood)
estimates, and each test is outlier detection against the population of
pixels, whose centre and spread are estimated robustly (median and MAD, the
high-breakdown counterparts of mean and standard deviation). The median and MAD ignore non-finite
values. Replacement takes, for each defective pixel, the (lower) median of its
$3^d - 1$ neighbours that are in bounds and not defective; a pixel without such
neighbours is left unchanged.

## Algorithm

**Input:** dark frames $\mathbf{y}_1, \ldots, \mathbf{y}_M \in \mathbb{R}^P$;
threshold $t = 5$, bimodality $b = 5/9$. \
**Output:** `SensorEstimate` with $\mathbf{o}$, $\boldsymbol{\sigma}^2$ and
`defective`; the individual masks in `masks`.

1. **Moments.** Stream the stack once: per-pixel mean $\mathbf{o}$, variance
   $\boldsymbol{\sigma}^2$, skewness $\mathbf{g}_1$ and kurtosis $\boldsymbol{\beta}_2$.
2. **Tests.** Per pixel:
    1. `offset`: $\lvert z(o_p) \rvert > t$;
    2. `variance`: $z(\sigma^2_p) > t$;
    3. `kurtosis`: $\lvert z(\beta_{2,p}) \rvert > t$;
    4. `rts`: $(g_{1,p}^2 + 1) / \beta_{2,p} > b$.
3. **Union.** `defective` is the union of the four masks.
4. **Apply.** Replace each defective pixel by the median of its good neighbours.

## Faithful mode and extensions

There is no paper: `faithful=True` and `faithful=False` both use the optix
defaults.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `threshold` | robust z-score threshold $t$ | `5.0` | `5.0` |
| `bimodality` | threshold $b$ of the bimodality coefficient | `5/9` | `5/9` |

## Memory and streaming

One streaming pass of fourth-order moments over frame batches
(`per-pixel-stats`); memory is a few `(*group, *spatial)` maps whatever the
number of frames. `apply` works chunk by chunk and needs about three times the
memory of a chunk.

## Usage

```python
import torch
from optix.models import random_sensor, simulate_sensor
from optix.sensor import DefectivePixels

camera = random_sensor((32, 32), defective=0.01, seed=3)
camera.variance[camera.defective] *= 100  # make the labelled pixels noisy
darks = simulate_sensor(torch.zeros(500, 32, 32), camera, seed=4)

det = DefectivePixels().fit(darks)
found = det.estimates["sensor"].defective
hits = (found & camera.defective).sum()  # all 10 labelled pixels
noisy = det.masks["variance"]
repaired = det.apply(darks)
```

`simulate_sensor` treats `defective` as a label only, which is why the example
inflates their variance. For a single frame or a map, use
`replace_defective(y, defective)` directly.

## References

No publication defines this detector: the tests, thresholds and replacement
rule are documented optix choices.
