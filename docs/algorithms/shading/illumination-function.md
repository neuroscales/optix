# Illumination function

!!! info "At a glance"
    - **Estimates:** `ShadingEstimate` (`flat`; `dark` when one is given)
    - **Noise model:** none
    - **Streaming:** per-pixel-stats
    - **Needs many frames:** yes
    - **Reference:** Singh S, Bray M-A, Jones TR, Carpenter AE, J Microsc, 2014, [doi:10.1111/jmi.12178](https://doi.org/10.1111/jmi.12178)

## Overview

`IlluminationFunction` computes the *illumination correction function* (ICF)
of [Singh et al. 2014](https://doi.org/10.1111/jmi.12178): the per-pixel
average of every image of a batch (a plate and a channel, typically), smoothed
heavily, and used as a flat field. Each image is corrected by dividing it by
that function.

It is the simplest retrospective method: one pass over the data, no
iterations, no noise model. It works when the batch holds many images whose
content is randomly placed, so that the specimen averages out and only the
illumination remains. It does not estimate a dark field or a baseline; when
the dark field matters, pass one, or use [BaSiC](basic.md) or
[CIDRE](cidre.md). It returns a `ShadingEstimate` whose `flat` is the
smoothed average, unnormalised by default (`meta["flat_norm"] == "none"`).

## Image formation (optics)

The method assumes the shading model of
[Image formation and noise](../../concepts/image-formation.md) with no
baseline, and, in the paper, no dark field:

=== "optix"

    $$
    \mathbf{y}_i = \mathbf{f} \odot \mathbf{x}_i + \mathbf{d},
    \qquad
    \hat{\mathbf{x}}_i = \frac{\mathbf{y}_i - \mathbf{d}}{\hat{\mathbf{f}}}
    $$

    with $\mathbf{d} = \mathbf{0}$ unless a dark field is given (extension).

=== "Singh et al. 2014"

    $$
    I_{\text{corrected}} = \frac{I}{F}
    $$

| optix | [Singh et al. 2014](https://doi.org/10.1111/jmi.12178) |
|---|---|
| $\mathbf{y}_i$ | $I$ (an image of the batch) |
| $\hat{\mathbf{x}}_i$ | $I_{\text{corrected}}$ |
| $\mathbf{f}$ | $F$ (the illumination correction function) |
| $\mathbf{d}$ | — (not modelled) |

Because the flat is never normalised in faithful mode, the corrected images
are rescaled by the mean brightness of the batch: $\hat{\mathbf{x}}_i$ is the
latent image divided by the average signal.

## Probabilistic model

There is no likelihood. The assumption is statistical: over the batch, the
expected latent image is spatially uniform,
$\mathbb{E}_i[\mathbf{x}_i] = \bar{x}\,\mathbf{1}$, so that the average image
is proportional to the flat field, and the flat field is smooth, so that
whatever specimen structure survives the average is removed by a smoothing
operator $\mathcal{S}$:

=== "optix"

    $$
    \hat{\mathbf{f}} = \mathcal{S}\!\left(\frac{1}{N}\sum_{i=1}^{N} (\mathbf{y}_i - \mathbf{d})\right)
    \;\approx\; \bar{x}\,\mathbf{f}
    $$

=== "Singh et al. 2014"

    $$
    F = \operatorname{median}_{500}\!\left(\frac{1}{N}\sum_{n=1}^{N} I_n\right)
    $$

| optix | [Singh et al. 2014](https://doi.org/10.1111/jmi.12178) |
|---|---|
| $\frac{1}{N}\sum_i \mathbf{y}_i$ | average image of the batch |
| $\mathcal{S}$ | median filter, window of 500 pixels |
| $\hat{\mathbf{f}}$ | $F$ |

## Inference / derivation

The strategy is *average, then smooth*. The per-pixel mean (or, as an
extension, the median) is accumulated in one streaming pass. Smoothing then
removes residual structure; the paper uses a large median filter, robust to
bright objects that the mean did not wash out. The scale ambiguity between
$\mathbf{f}$ and $\mathbf{x}_i$ is left unresolved in the paper (the ICF keeps
the units of the data); the `"robust-min"` extension resolves it as
CellProfiler does, by dividing by the 2nd percentile of the positive values
after clipping below it, so that $\hat{\mathbf{f}} \ge 1$.

## Algorithm

```text
Input: stack Y (N frames), optional dark d, options statistic, smoothing, size, rescale
1. a = per-pixel mean (or median) of Y over frames     one streaming pass
2. if d is given: a = a − d                             extension
3. f = S(a)                                            median of side `size` (paper: 500)
4. if rescale == "robust-min": f = max(f, q₀.₀₂) / q₀.₀₂ extension
Return ShadingEstimate(flat=f, dark=d)
apply: x̂ = (y − d) / f
```

Each group of a layout gets its own function.

## Faithful mode and extensions

`faithful=True` reproduces the paper: mean, square median window of side 500,
no rescaling, no dark field. The extensions are ported from CellProfiler's
CorrectIlluminationCalculate module (BSD-3-Clause), in which the paper
implements the method. `faithful=False` currently uses the same values.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `statistic` | per-pixel average: `"mean"` (streaming moments) or `"median"` (exact, by row tiles) | `"mean"` | `"mean"` |
| `smoothing` | `"median"` (square window of side `size`); `"median-disk"` (disk of radius $\max(1, \operatorname{round}(\sigma))$); `"gaussian"` (normalised, std $\sigma$); `"polynomial"` (least-squares quadric); `"none"`; with $\sigma = \text{size}/2.35$ | `"median"` | `"median"` |
| `size` | window side, or CellProfiler's filter size (FWHM); `"auto"` gives $\min(30, \max(\text{shape})/40)$ | `500` | `500` |
| `rescale` | `"robust-min"` divides by the 2nd percentile after clipping below it | `"none"` | `"none"` |
| `dark` (in `fit`) | stack, map, `ShadingEstimate` (`dark`) or `SensorEstimate` (`offset`) subtracted from the average before smoothing | not given | not given |

The paper's "median filter with window size = 500 pixels" is read as a square
window of side 500, pixels outside the image being ignored.

## Memory and streaming

The fit needs per-pixel statistics only: the mean is accumulated over frame
batches in memory proportional to one frame; `statistic="median"` reads the
stack by strips of rows holding every frame. The smoothing works on one frame
per group. The exact square median costs $O(\text{size}^2)$ per pixel, so the
paper's 500-pixel window on a 1280 × 1024 image is very slow; a smaller
`size`, or `smoothing="gaussian"`, is much faster. `apply` is pointwise.

## Usage

```python
import torch
import optix
from optix.models import (
    random_sensor,
    random_shading,
    simulate_sensor,
    simulate_shading,
)
from optix.shading import IlluminationFunction

# 1. A synthetic stack with known truth: random content, smooth flat field.
truth = random_shading((64, 64), strength=0.2, seed=0)
gen = torch.Generator().manual_seed(1)
x = 50.0 + 400.0 * torch.rand(200, 64, 64, generator=gen)
sensor = random_sensor((64, 64), gain=1.0, offset=0.0, offset_sd=0.0, seed=2)
Y = simulate_sensor(simulate_shading(x, truth), sensor, seed=3)

# 2. Fit (a 31-pixel window instead of the paper's 500, for a small image).
est = IlluminationFunction(size=31).fit(Y)

# 3. Correct.
corrected = est.apply(Y)

# 4. The flat field is the truth up to a scale.
ratio = est.estimates["shading"].flat / truth.flat
print(float(ratio.std() / ratio.mean()))  # small
```

Common variations: `IlluminationFunction(smoothing="gaussian", size="auto",
rescale="robust-min", faithful=False)` for CellProfiler's defaults;
`fit(Y, dark=sensor_estimate)` to subtract the camera offset first. See the
[API reference](../../reference/shading.md).

## References

!!! references "References"

    1. Singh S, Bray M-A, Jones TR, Carpenter AE. *Pipeline for illumination
       correction of images for high-throughput microscopy.* J Microsc
       **256**(3):231–236, 2014.
       [doi:10.1111/jmi.12178](https://doi.org/10.1111/jmi.12178)

    ??? quote "BibTeX"

        ```bibtex
        @article{singh2014pipeline,
          author  = {Singh, S. and Bray, M.-A. and Jones, T. R. and Carpenter, A. E.},
          title   = {Pipeline for illumination correction of images for high-throughput microscopy},
          journal = {Journal of Microscopy},
          volume  = {256},
          number  = {3},
          pages   = {231--236},
          year    = {2014},
          doi     = {10.1111/jmi.12178},
        }
        ```
