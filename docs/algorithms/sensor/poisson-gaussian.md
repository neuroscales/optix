# Poisson-Gaussian noise fit

!!! info "At a glance"
    **Estimates:** `SensorEstimate` (`gain`, `variance`; `offset = None`) \
    **Noise model:** Poisson-Gaussian, optionally clipped \
    **Streaming:** per-frame \
    **Needs many frames:** no (one image; pooling over frames is an extension) \
    **Reference:** Foi, Trimeche, Katkovnik, Egiazarian, IEEE Trans Image Process, 2008, [doi:10.1109/TIP.2008.2001399](https://doi.org/10.1109/TIP.2008.2001399)

## Overview

`PoissonGaussian` fits the noise variance function $\operatorname{var}[y] = a\,\mathbb{E}[y] + b$ of a
camera from a single raw image of an arbitrary scene, following
[Foi et al. 2008](https://doi.org/10.1109/TIP.2008.2001399). No flat or dark
frames are needed: the scene itself supplies many intensity levels. Use it when
calibration stacks are not available, or to check a calibration on real data.
When a dark stack and a series of flat fields are available, the
[photon-transfer curve](photon-transfer-curve.md) is more accurate and also
gives the offset.

It returns a [`SensorEstimate`](../../reference/sensor.md) with `gain` $= a$
and `variance` $= b$, the variance extrapolated to 0 ADU. The offset is
`None`: $(a, b)$ alone do not determine it, so `variance` is **not** the
read-noise variance $\sigma^2$ unless the offset is zero (see below).

## Image formation (optics)

The camera model of [Image formation and noise](../../concepts/image-formation.md):
each pixel counts $n \sim \operatorname{Poisson}(\lambda)$ photoelectrons,
converted with gain $g$ and offset $o$ and corrupted by Gaussian read noise. The
image may be clipped to the range of the format (saturation, and zero for
offset-free data). Nothing about the optics is assumed: the scene only needs
smooth regions spanning several intensities.

=== "optix"

    $$
    y = g\,n + o + \varepsilon, \qquad n \sim \operatorname{Poisson}(\lambda),\quad
    \varepsilon \sim \mathcal{N}(0, \sigma^2)
    $$

=== "Foi et al. 2008"

    $$
    z(x) = y(x) + \eta_p\bigl(y(x)\bigr) + \eta_g(x), \qquad
    \chi\,\bigl(y(x) + \eta_p\bigl(y(x)\bigr)\bigr) \sim \mathcal{P}\bigl(\chi\, y(x)\bigr),\quad
    \eta_g(x) \sim \mathcal{N}(0, b),
    \tilde z = \max\bigl(0, \min(z, 1)\bigr)
    $$

## Probabilistic model

Both parametrisations give a variance that is affine in the expectation:

=== "optix"

    $$
    \operatorname{var}[y] = g\,\mathbb{E}[y] + \sigma^2 - g\,o
    $$

=== "Foi et al. 2008"

    $$
    \sigma^2\bigl(y(x)\bigr) = a\,y(x) + b, \qquad a = \chi^{-1}
    $$

| optix | [Foi et al. 2008](https://doi.org/10.1109/TIP.2008.2001399) | meaning |
|---|---|---|
| $y$ | $z$ (clipped: $\tilde z$) | observation |
| $\mathbb{E}[y]$ | $y$ | expectation of the observation |
| $g\,(n - \lambda)$, $\varepsilon$ | $\eta_p$, $\eta_g$ | Poisson and Gaussian parts |
| $g$ | $a = \chi^{-1}$ | gain |
| $\sigma^2 - g\,o$ | $b$ | variance at zero expectation |
| data range | $[0, 1]$ after normalisation (Section IV-A) | |

Assumptions: the noise is independent between pixels and its variance depends
only on the expectation, with the same $(a, b)$ everywhere (one value per
group); clipping, if present, is at the known bounds `data_range`. optix works in
normalised units $z' = (z - \text{low})/(\text{high} - \text{low})$, as the
paper does, and maps $(a, b)$ back to ADU at the end.

## Inference / derivation

The strategy has two stages.

1. **Local estimates.** A one-level wavelet decomposition separates a smoothed
   image (approximation coefficients) from noise (detail coefficients). Pixels
   near edges, where the detail coefficients exceed a threshold $\tau$, are
   discarded. The remaining smooth pixels are grouped into level sets of width
   $\Delta$ of the smoothed image. Each set gives a pair
   $(\hat y_i, \hat\sigma_i)$: its mean expectation and the standard deviation
   of its detail coefficients (sample std, or MAD in the robust variant).
2. **Global fit.** The pairs are treated as noisy samples of the curve
   $\sigma(y) = \sqrt{a y + b}$ (or its clipped counterpart
   $\tilde\sigma(\tilde y)$), with variances known from the set sizes. $(a, b)$
   are found by maximum likelihood over all pairs, from a weighted
   least-squares starting point. With clipping, the curve is the expectation
   and standard deviation of a censored normal, which bends near the bounds and
   lets the fit use saturated regions instead of being biased by them.

The covariance of $(a, b)$ is the inverse Hessian of $-\log L$ (an optix
addition). The offset cannot be separated from $b$, so it is left unestimated.

## Algorithm

```text
Input: image(s) Y, data_range, delta, tau
Normalise Y to [0, 1] with data_range
Wavelet-transform; mask edge coefficients (|detail| > tau)
Segment the smooth pixels into level sets of width delta
   (double tau while fewer than two sets are found)
For each set: pair (y_i, sigma_i) and its variance factors
Initial (a, b): weighted least squares on the pairs
(a, b): maximise log L with Nelder-Mead (max_iter, tol)
Covariance from the Hessian at the optimum
Return gain = a, variance = b (ADU), offset = None
```

## Faithful mode and extensions

`faithful=True` uses one frame per group, sample standard deviations and the
clipped model (Sections III and V of the paper). `faithful=False` only adds
pooling over all frames.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `data_range` | normalised range and clipping bounds | `None` (float data: min/max of the data; integer data must pass it or `"auto"`) | same |
| `clipped` | clipped observation model | `True` | `True` |
| `robust` | MAD estimates (Section VI); incompatible with `pooled` | `False` | `False` |
| `mad_correction` | factor of the robust LS start: `"paper"` or `"exact"` | `"paper"` | `"paper"` |
| `delta` | width $\Delta$ of the level sets | `1/300` | `1/300` |
| `tau` | edge threshold (not in the paper) | `0.25` | `0.25` |
| `eps_reg` | regularisation $\varepsilon_{\text{reg}}$ of Eq. (21) | `1e-6` | `1e-6` |
| `frame` | frame used by the single-image fit | `0` | `0` |
| `pooled` | merge level-set statistics over all frames | `False` | `True` |
| `differencing` | $\hat\sigma$ from differences of consecutive frames | `False` | `False` |
| `max_iter`, `tol` | Nelder-Mead budget | `500`, `1e-10` | `500`, `1e-10` |

## Memory and streaming

Per-frame: the single-image fit loads one frame per group. With `pooled=True`
the frames are streamed in batches and per-segment sufficient statistics are
merged exactly, so memory is that of a few frames whatever the stack length.
The ML fit works on the pairs only (a few hundred numbers).

## Usage

```python
import torch
from optix.models import random_sensor, simulate_sensor
from optix.sensor import PoissonGaussian

truth = random_sensor((128, 128), offset=100.0, offset_sd=0.0,
                      read_sd=2.0, gain=0.5, seed=0)
ramp = 20.0 + 3000.0 * torch.linspace(0, 1, 128).expand(128, 128)
y = simulate_sensor(ramp.expand(4, 128, 128), truth, seed=1)

est = PoissonGaussian(faithful=False).fit(y)       # pooled over the 4 frames
s = est.estimates["sensor"]
print(float(s.gain))                                 # about 0.5
print(float(s.variance), 2.0**2 - 0.5 * 100.0)      # b = sigma^2 - g o, about -46
```

Pass `data_range=(0, 4095)` (or `"auto"`) for integer data. To obtain the
read-noise variance, add a known offset: $\sigma^2 = b + g\,o$.

## References

!!! references "References"

    1. Foi A, Trimeche M, Katkovnik V, Egiazarian K. *Practical
       Poissonian-Gaussian noise modeling and fitting for single-image raw-data.*
       IEEE Trans Image Process **17**(10):1737-1754, 2008.
       [doi:10.1109/TIP.2008.2001399](https://doi.org/10.1109/TIP.2008.2001399)
    2. Nelder JA, Mead R. *A simplex method for function minimization.*
       Comput J **7**(4):308-313, 1965.
       [doi:10.1093/comjnl/7.4.308](https://doi.org/10.1093/comjnl/7.4.308)

    ??? quote "BibTeX"

        ```bibtex
        @article{foi2008practical,
          author  = {Foi, Alessandro and Trimeche, Mejdi and Katkovnik, Vladimir and Egiazarian, Karen},
          title   = {Practical {P}oissonian-{G}aussian noise modeling and fitting for single-image raw-data},
          journal = {IEEE Transactions on Image Processing},
          volume  = {17},
          number  = {10},
          pages   = {1737--1754},
          year    = {2008},
          doi     = {10.1109/TIP.2008.2001399},
        }
        @article{nelder1965simplex,
          author  = {Nelder, J. A. and Mead, R.},
          title   = {A simplex method for function minimization},
          journal = {The Computer Journal},
          volume  = {7},
          number  = {4},
          pages   = {308--313},
          year    = {1965},
          doi     = {10.1093/comjnl/7.4.308},
        }
        ```
