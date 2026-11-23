# Photon-transfer curve

!!! info "At a glance"
    - **Estimates:** `SensorEstimate` (`offset`, `variance`, `gain`; DSNU, PRNU, linearity in `meta`)
    - **Noise model:** Poisson-Gaussian
    - **Streaming:** per-pixel-stats
    - **Needs many frames:** no (two images per level, a dark stack)
    - **Reference:** European Machine Vision Association, EMVA Standard 1288 Release 4.0 Linear, 2021, [emva.org](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf)

## Overview

`PhotonTransferCurve` calibrates a camera the way the
[EMVA 1288 standard](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf) prescribes: from a dark stack and pairs of images of
a uniform illumination at a series of levels, it plots the temporal variance
against the mean and reads the gain from the slope of this line. It also
reports the read noise, the saturation, the linearity of the curve, the dark
signal non-uniformity (DSNU), the photo-response non-uniformity (PRNU) and,
given the photon exposure of each level, the responsivity and quantum efficiency.

Use it when you can record calibration data; it is the reference against which
the [single-image fit](poisson-gaussian.md) is checked. Unlike that fit, it
estimates the offset (the per-pixel dark mean), so `variance` is the read-noise
variance $\sigma^2$. Per-pixel gains (sCMOS) are an extension (`per_pixel=True`).
The result is a [`SensorEstimate`](../../reference/sensor.md).

## Image formation (optics)

The camera model of [Image formation and noise](../../concepts/image-formation.md),
plus the quantisation noise of the analog-to-digital converter. The
illumination is uniform and stable within each level.

=== "optix"

    $$
    y = g\,n + o + \varepsilon, \qquad n \sim \operatorname{Poisson}(\lambda),\quad
    \varepsilon \sim \mathcal{N}(0, \sigma^2), \qquad \text{plus rounding to } \Delta = 1
    $$

=== "EMVA 1288"

    $$
    \mu_y = K\,(\mu_d + \mu_e), \qquad \mu_e = \eta\,\mu_p, \qquad
    \sigma_y^2 = K^2 \sigma_d^2 + \sigma_q^2 + K^2 \sigma_e^2, \quad \sigma_e^2 = \mu_e
    $$

## Probabilistic model

The photon-transfer curve ([EMVA 1288 R4.0 Linear, Eq. (15)](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf)):

=== "optix"

    $$
    \operatorname{var}[y] = \sigma^2 + \tfrac{1}{12} + g\,\bigl(\mathbb{E}[y] - o\bigr)
    $$

=== "EMVA 1288"

    $$
    \sigma_y^2 = K^2 \sigma_d^2 + \sigma_q^2 + K\,(\mu_y - \mu_{y.\text{dark}}),
    \qquad \sigma_q^2 = \tfrac{1}{12}\ \text{DN}^2
    $$

| optix | [EMVA 1288](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf) | meaning |
|---|---|---|
| $ | $ | overall system gain (ADU = DN per e⁻) |
| $ | $\mu_{y.\text{dark}}$ | dark mean (offset) |
| $\mathbb{E}[y]$, $\operatorname{var}[y]$ | $\mu_y$, $\sigma_y^2$ | mean and temporal variance |
| $\sigma^2$ | ^2 \sigma_d^2$ ($\sigma_d$ in e⁻) | read-noise variance |
| $\sigma^2 + \tfrac{1}{12}$ | $\sigma_{y.\text{dark}}^2$ | dark temporal variance |
| $\lambda$ | $\mu_e = \eta\,\mu_p$ | mean photoelectrons |

Assumptions: linear response below saturation, noise independent between
pixels and frames, dark values at the same exposure time as the levels, one
gain per group (channel).

## Inference / derivation

The strategy is a straight-line fit on summary statistics.

- **Temporal statistics** per level and for the dark: mean and variance from the
  first two images, the variance from their difference (which cancels the
  fixed pattern) with the standard's mean-difference correction.
- **Saturation** is the brightest level with at most 0.2 % of pixels at the
  maximum value.
- **Gain** $: least squares of $\sigma_y^2 - \sigma_{y.\text{dark}}^2$ on
  $\mu_y - \mu_{y.\text{dark}}$, with free slope and intercept, over levels up
  to 70 % of saturation (Eq. (50)). A cubic B-spline fit checks that the local
  slopes stay within 3 % of $ (Eq. (51)).
- **Read noise** $\sigma^2 = \sigma_{y.\text{dark}}^2 - \sigma_q^2$ (Eq. (53)).
  When $\sigma_{y.\text{dark}}^2 < 0.24$ DN², only an upper limit is reported (Eq. (54)).
- **Non-uniformities** from highpass-filtered mean images, with the temporal
  noise of the mean subtracted: DSNU from the dark, PRNU from the level nearest
  50 % saturation (Section 8).

## Algorithm

```text
Input: levels (>= 2 images each), dark stack, optional exposure per level
Stream every stack once: pair statistics and per-pixel moments
mu_i, var_i: dark-subtracted mean and temporal variance of each level
Saturation: brightest level with <= saturation_fraction pixels at max
K: OLS of var_i on mu_i for mu_i <= fit_range * saturation
Linearity: B-spline slopes (spline_intervals) vs K
Read noise, DSNU, PRNU; with exposure: responsivity, QE, linearity error
Return offset = dark mean image, variance = read noise (DN^2), gain = K
```

## Faithful mode and extensions

`faithful=True` and `faithful=False` are identical; every extension is opt-in.
Choices the standard leaves open (saturation picked among the measured levels,
highpass border crop, a single dark exposure time) are listed in the class docstring.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `fit_range` | upper end of the gain fit, fraction of saturation | `0.7` | `0.7` |
| `quantization` | $\sigma_q^2$ (DN²); `0` for unquantised data | `1/12` | `1/12` |
| `max_value` | maximum gray value for the saturation rule | `None` (from the data) | `None` |
| `saturation_fraction` | largest fraction at the maximum below saturation | `0.002` | `0.002` |
| `saturation` | `"histogram"` (R4.0) or `"variance"` (R3.0 rule) | `"histogram"` | `"histogram"` |
| `temporal` | `"pair"` (Eq. (18)) or `"stack"` (all images) | `"pair"` | `"pair"` |
| `highpass` | highpass the non-uniformity images | `True` | `True` |
| `spline_intervals` | B-spline intervals of the linearity check | `6` | `6` |
| `per_pixel` | also fit a per-pixel gain map | `False` | `False` |

## Memory and streaming

Per-pixel-stats: each stack is streamed once, accumulating the pair statistics
and per-pixel moments (a few images' worth of memory per level). The fits work
on one number per level and group.

## Usage

```python
import torch
from optix.models import random_sensor, simulate_sensor
from optix.sensor import PhotonTransferCurve

truth = random_sensor((64, 64), offset=100.0, read_sd=2.0, gain=0.5, seed=0)
electrons = [100.0 * 2**k for k in range(8)] + [40000.0]
levels = [
    simulate_sensor(
        torch.full((2, 64, 64), e), truth, seed=i, quantize=torch.uint16
    )
    for i, e in enumerate(electrons)
]
dark = simulate_sensor(
    torch.zeros(16, 64, 64), truth, seed=99, quantize=torch.uint16
)

ptc = PhotonTransferCurve(max_value=65535).fit(levels, dark=dark)
s = ptc.estimates["sensor"]
print(float(s.gain), float(s.variance))  # about 0.5 and 4.0
print(s.meta["read_noise_e"])  # sigma / K, about 4 e-
```

Pass `exposure=` (photons per pixel for each level) for the responsivity and
quantum efficiency, and `per_pixel=True` for an sCMOS gain map.

## References

!!! references "References"

    1. European Machine Vision Association. *EMVA Standard 1288: Standard for
       Characterization of Image Sensors and Cameras, Release 4.0 Linear.*
       16 June 2021. [emva.org](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf)
    2. European Machine Vision Association. *EMVA Standard 1288: Standard for
       Characterization of Image Sensors and Cameras, Release 4.0 General.*
       16 June 2021. [emva.org](https://www.emva.org/wp-content/uploads/EMVA1288General_4.0Release.pdf)

    ??? quote "BibTeX"

        ```bibtex
        @techreport{emva1288linear40,
          author      = {{European Machine Vision Association}},
          title       = {{EMVA} Standard 1288: Standard for Characterization of Image Sensors and Cameras, Release 4.0 Linear},
          institution = {EMVA},
          year        = {2021},
          month       = jun,
          url         = {https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf},
        }
        @techreport{emva1288general40,
          author      = {{European Machine Vision Association}},
          title       = {{EMVA} Standard 1288: Standard for Characterization of Image Sensors and Cameras, Release 4.0 General},
          institution = {EMVA},
          year        = {2021},
          month       = jun,
          url         = {https://www.emva.org/wp-content/uploads/EMVA1288General_4.0Release.pdf},
        }
        ```
