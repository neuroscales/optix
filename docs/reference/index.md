---
icon: fontawesome/solid/code
---

# API reference

The reference documents every public name, generated from the docstrings. The
[Concepts](../concepts/index.md) pages explain how the pieces fit together; use
the reference to look up a signature.

| Module | Contents |
|---|---|
| [`optix`](configuration.md) | `config`, `set_config` and `get_config`, the options of a computation |
| [`optix`](estimator.md) | `Estimator`, `Tags`, `History`, `estimators` and `load` |
| [`optix.estimates`](estimates.md) | `SensorEstimate`, `ShadingEstimate`, `Classification`, `PSFEstimate`: the shared result types |
| [`optix.io`](io.md) | `Layout`, `labeled` and the chunk iterators `frame_batches`, `row_tiles` and `blocks` |
| [`optix.stats`](stats.md) | streaming reducers (`Moments`, `Histogram`, ...) and per-pixel order statistics |
| [`optix.filters`](filters.md) | spatial neighbourhood filters (`median_filter`, `median_disk`) |
| [`optix.basis`](basis.md) | DCT bases and their smoothness penalties |
| [`optix.special`](special.md) | quadrature, Newton solvers and cancellation-free KL divergences |
| [`optix.optim`](optim.md) | proximity operators, difference operators and the ADMM solver `admm_fft` |
| [`optix.sensor`](sensor.md) | sensor calibration estimators and variance-stabilising transforms |
| [`optix.shading`](shading.md) | shading estimators and `correct` |
| [`optix.deconv`](deconv.md) | deconvolution: Richardson–Lucy and its variants, Wiener, Tikhonov, `Ikoma` |
| [`optix.background`](background.md) | `RollingBall`, `TopHat`, `rolling_ball` and `top_hat`: background subtraction by grey-scale opening |
| [`optix.psf`](psf.md) | point spread functions: `BeadPSF`, `bead_psf` and `lightsheet_psf` |
| [`optix.models`](models.md) | simulators of the camera and shading models |
| [`optix.distributions`](distributions.md) | `Gamma`, `Normal`, `Dirichlet`, `Categorical`, statistics and conjugate updates |
| [`optix.distributions.functional`](functional.md) | the pure tensor kernels behind the distributions |

Conventions that apply throughout:

- Tensor shapes in docstrings use `(*group, *spatial)` for parameter maps and
  `(n, *spatial)` for a batch of frames, as explained in
  [Axes and layouts](../concepts/axes-and-layouts.md).
- Optional `dtype` and `device` arguments default to the active
  [configuration](configuration.md).
- Names that start with an underscore are private and may change without notice,
  except where the [Estimators](../concepts/estimators.md#writing-an-estimator)
  page documents them for authors of new algorithms.
