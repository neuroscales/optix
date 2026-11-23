# Dark/flat calibration

!!! info "At a glance"
    **Estimates:** `ShadingEstimate` (`flat`, `dark`) and, with a dark stack, `SensorEstimate` (`offset`, `variance`) \
    **Noise model:** none \
    **Streaming:** per-pixel-stats \
    **Needs many frames:** yes \
    **Reference:** none: standard practice, optix convention

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
assumed to average out.

$$
\mathbf{y}_i = \mathbf{f} \odot \mathbf{x}_i + \mathbf{d}, \qquad
\hat{\mathbf{d}} = \operatorname{stat}_m \mathbf{y}_m^{\text{dark}}, \qquad
\hat{\mathbf{f}} = \frac{\mathbf{F} - \hat{\mathbf{d}}}{\operatorname{mean}_p(\mathbf{F} - \hat{\mathbf{d}})},
$$

with $\mathbf{F} = \operatorname{stat}_m \mathbf{y}_m^{\text{flat}}$ and
$\operatorname{stat}$ the per-pixel temporal mean or median. There is no paper,
so no native-notation tab.

## Inference / derivation

The correction is
$\hat{\mathbf{x}}_i = (\mathbf{y}_i - \hat{\mathbf{d}}) / \hat{\mathbf{f}}
= \dfrac{\mathbf{y}_i - \mathbf{D}}{\mathbf{F} - \mathbf{D}}\,\operatorname{mean}(\mathbf{F} - \mathbf{D})$.
The overall scale shared by $\mathbf{f}$ and $\mathbf{x}_i$ is fixed by dividing
the flat by its spatial mean (per group), which keeps corrected intensities in
ADU; `meta["flat_norm"]` records `"mean"` or `"none"`.

## Algorithm

```text
Input: flat stack, dark (stack, map, ShadingEstimate, SensorEstimate or none), statistic, normalize
1. F = per-pixel statistic of the flat stack.
2. D = per-pixel statistic of the dark stack; or the given map;
   or ShadingEstimate.dark; or SensorEstimate.offset; or 0.
3. f = F − D; if normalize == "mean": f = f / spatial mean of f (per group).
Return ShadingEstimate(flat=f, dark=D) and, for a dark stack,
       SensorEstimate(offset=D, variance=temporal variance of the dark stack)
```

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
