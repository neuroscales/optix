# Defective pixels

!!! info "At a glance"
    **Estimates:** `SensorEstimate` (`defective`, `offset`, `variance`) \
    **Noise model:** Gaussian \
    **Streaming:** per-pixel-stats \
    **Needs many frames:** yes \
    **Reference:** none: optix convention

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
$m = 1 \ldots M$, with $o_p$ and $\sigma^2_p$ similar across pixels. From the
temporal moments of each pixel, the dark mean $\mathbf{o}$, the variance
$\boldsymbol{\sigma}^2$, the skewness $\mathbf{g}_1$ and the non-excess kurtosis
$\boldsymbol{\beta}_2$, a pixel is flagged when

$$
|z(o_p)| > t, \qquad z(\sigma^2_p) > t, \qquad |z(\beta_{2,p})| > t,
\qquad\text{or}\qquad \frac{g_{1,p}^2 + 1}{\beta_{2,p}} > b,
$$

with the robust z-score over the pixels of a group

$$
z(x_p) = \frac{x_p - \operatorname{med}_p x}{1.4826\,\operatorname{MAD}_p x}.
$$

The last test is Sarle's bimodality coefficient: $1/3$ for a Gaussian, $5/9$
for a uniform distribution, $1$ for a two-level (RTS) signal. There is no paper,
so no native-notation tab.

## Inference / derivation

There is nothing to infer: the four masks are threshold tests on per-pixel
moments, and `defective` is their union. The median and MAD ignore non-finite
values. Replacement takes, for each defective pixel, the (lower) median of its
$3^d - 1$ neighbours that are in bounds and not defective; a pixel without such
neighbours is left unchanged.

## Algorithm

```text
Input: dark stack, threshold t = 5, bimodality b = 5/9
1. Stream the stack once: per-pixel mean, variance, skewness, kurtosis.
2. masks = {offset: |z(mean)| > t, variance: z(var) > t,
            kurtosis: |z(kurtosis)| > t, rts: (skew² + 1)/kurtosis > b}
3. defective = union of the masks.
Return SensorEstimate(offset=mean, variance=var, defective)
apply: replace each defective pixel by the median of its good neighbours.
```

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
