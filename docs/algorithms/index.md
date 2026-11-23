---
icon: fontawesome/solid/gears
---

# Algorithms

One page for every estimator in optix. Each page states the model in the common
[notation](../concepts/notation.md), shows it next to the notation of the paper
it comes from, and goes from the model to the algorithm, its options and an
example. The sections follow the modules of the
[API reference](../reference/index.md).

| Module | Algorithm | Estimates |
|---|---|---|
| [`optix.sensor`](../reference/sensor.md) | [Poisson-Gaussian fit](sensor/poisson-gaussian.md) (`PoissonGaussian`) | gain, offset and read noise from one image |
| | [Generalized Anscombe](sensor/generalized-anscombe.md) (`GeneralizedAnscombe`) | variance-stabilising transform and its inverses |
| | [Photon-transfer curve](sensor/photon-transfer-curve.md) (`PhotonTransferCurve`) | gain and read noise from a series of exposures (EMVA 1288) |
| | [sCMOS calibration](sensor/scmos-calibration.md) (`SCMOSCalibration`) | per-pixel offset, gain and read noise |
| | [Dark and flat calibration](sensor/dark-flat-calibration.md) (`DarkFlatCalibration`) | dark field and flat field from calibration stacks |
| | [Defective pixels](sensor/defective-pixels.md) (`DefectivePixels`) | hot, dead and noisy pixels |
| [`optix.shading`](../reference/shading.md) | [BaSiC](shading/basic.md) (`BaSiC`) | flat field, dark field and baseline |
| | [CIDRE](shading/cidre.md) (`CIDRE`) | flat field and dark field |
| | [Kask](shading/kask.md) (`KaskFlatField`) | foreground and background flat fields |
| | [Illumination function](shading/illumination-function.md) (`IlluminationFunction`) | flat field |
| | [Gamma mixture](shading/gamma-mixture.md) (`GammaMixture`) | flat field and classes |
| [`optix.background`](../reference/background.md) | [Rolling ball](background/rolling-ball.md) (`RollingBall`, `TopHat`) | background, by grey-scale opening |
| [`optix.deconv`](../reference/deconv.md) | [Richardson–Lucy](deconv/richardson-lucy.md) (`RichardsonLucy`) | deconvolved image |
| | [Biggs–Andrews](deconv/biggs-andrews.md) (`BiggsAndrews`) | deconvolved image, accelerated |
| | [Snyder](deconv/snyder.md) (`Snyder`) | deconvolved image |
| | [Wiener](deconv/wiener.md) (`Wiener`) | deconvolved image, in one linear step |
| | [Tikhonov](deconv/tikhonov.md) (`Tikhonov`) | deconvolved image, in one linear step |
| | [Wiener–Butterworth](deconv/wiener-butterworth.md) (`WienerButterworth`) | deconvolved image |
| | [Multiview Richardson–Lucy](deconv/multiview.md) (`MultiviewRL`) | one deconvolved image from several registered views |
| [`optix.destripe`](../reference/destripe.md) | [Wavelet-FFT](destripe/wavelet-fft.md) (`WaveletFFT`) | stripe-free image |
| [`optix.psf`](../reference/psf.md) | [Bead PSF](psf/bead-psf.md) (`BeadPSF`) | point-spread function from images of beads |
| | [Gibson–Lanni PSF](psf/gibson-lanni.md) (`gibson_lanni`) | scalar PSF model with refractive-index and thickness mismatch |

The rest of this page explains how the section is organised and what an
algorithm page contains.

## Organised by what is estimated

The same method can be filed under many headings: low-rank plus sparse, an
iterative scheme, a family of wavelets. But people come to the documentation
with a *problem*: the illumination is uneven, the image has stripes, the camera
noise must be removed. And two algorithms can only be compared, or chained, when
they produce the same kind of result. So algorithms are grouped by **what they
estimate**, such as:

- the **sensor**: offset, gain and read noise of the camera, defective pixels;
- the **shading**: flat field, dark field and per-frame baseline;
- **stripes and shadows** of a light sheet, and the **attenuation** of light along
  its path;
- the **background** of an image;
- **noise**, to be removed;
- the **point-spread function** and the **deconvolution** that uses it;
- the **fusion** of tiles and views.

Each target has a *result type*, a small dataclass from
[`optix.estimates`](../reference/estimates.md), that every algorithm for that
target returns. `ShadingEstimate`, for instance, holds the flat field, the dark
field and the baseline, whichever algorithm estimated them. Algorithms that
estimate several things at once, such as a joint model of sensor, shading and
tissue, are filed under their main target and return the result type of every
target they estimate. These shared types are how algorithms are compared, and
how one is started from another: the result of a simple method can initialise a
sophisticated one (see [Starting from another estimate](../concepts/estimators.md#starting-from-another-estimate)).

## Assumptions are tags

An algorithm is more than its target. It assumes a noise model, needs
many frames or few, runs in constant memory or needs the whole stack. These are
properties to compare, but they cannot form a second tree of folders. They are
declared as **tags** on the estimator, in the `tags` attribute of the
class (see [`optix.Tags`](../reference/estimator.md)), and are stored with the
code so they cannot drift from it:

| Tag | Values | Meaning |
|---|---|---|
| `targets` | `"sensor"`, `"shading"`, ... | what it estimates |
| `streaming` | `"pointwise"`, `"per-frame"`, `"halo"`, `"per-pixel-stats"`, `"whole-stack"` | how it accesses the data (below) |
| `modality` | `"lightsheet"`, `"any"`, ... | the imaging modalities it is meant for |
| `noise` | `"none"`, `"gaussian"`, `"poisson"`, `"poisson-gaussian"`, ... | the noise model it assumes |
| `many_frames` | true or false | whether it needs many frames to work |
| `reference` | a citation or DOI | the publication it implements |

The streaming classes say what an algorithm needs from the data, and therefore
how much of it can be in memory at once:

- **pointwise**: each output pixel depends on one input pixel only;
- **per-frame**: each frame is processed on its own;
- **halo**: each pixel depends on its neighbourhood, so data are processed in
  blocks with an overlap;
- **per-pixel-stats**: the algorithm needs per-pixel summaries over the frames,
  which are accumulated in one pass;
- **whole-stack**: the algorithm needs the stack all at once.

The registry of estimators (`optix.estimators()`) lets the tags be read by
code. The documentation will use them to build an *assumption by target* table:
one row for every algorithm, with the noise model, streaming class and
modality in columns, so that you can find the algorithms that suit your data.

## The page template

Every algorithm page has the same nine sections, in the same order, so that a
reader who has seen one knows where to look in the next. A ready-to-copy
skeleton is in the [page template](_template.md). Sections are as long as their
topic needs: a method that is a few lines of arithmetic gets a short
derivation, and one that is a research contribution gets a long one.

1. **Overview.** What the algorithm estimates, in a paragraph that a user
   without the paper can follow, when to use it, and when not to. Names the
   [tags](#assumptions-are-tags) in words, and links to the result type it
   returns.

2. **Image formation (optics).** The physical process behind the data, as far as
   the algorithm assumes it: how light, sample, optics and detector combine into
   the image. Links to the shared models in [Concepts](../concepts/index.md)
   rather than re-deriving them, and states only what is particular to this
   method.

3. **Probabilistic model.** The assumptions turned into equations: the
   generative model, the noise, the priors or the regularisers, with every
   symbol defined where it first appears and with the notation of the
   [Concepts](../concepts/notation.md). For a method that is not
   probabilistic (a filter, say), this section states the objective or the
   operator instead.

4. **Inference / derivation.** How the estimates follow from the model: the
   objective, the update equations, the steps in between. This is the section a
   reviewer checks line by line, so intermediate steps are shown, and
   conventions that fix an ambiguity (a normalisation, for example) are stated.

5. **Algorithm.** Pseudocode of the whole procedure, with the order of the
   steps, the stopping rule and the default values of the hyper-parameters.

6. **Faithful mode and extensions.** What `faithful=True` does, step by step,
   and how it differs from the publication where it must (with the reason).
   Then every extension, as a table: its keyword, what it changes, its default
   and why it might help. This is also the place to record what the reference
   implementation did that the paper did not say.

7. **Memory and streaming.** How the algorithm reads the data (which of the
   chunk iterators), what it holds in memory, how the cost grows with the size
   of the stack, and what to tune.

8. **Usage.** A short example that runs: fit, inspect, apply, on synthetic data
   from `optix.models` where possible, followed by the common variations
   (groups, initialisation from another estimate, a large zarr array).

9. **References.** The publication the algorithm implements, and others it relies
   on, each with its DOI, verified against the publisher's page, in the
   [reference admonition](../contributing/style.md#references) with a BibTeX entry. Also the
   reference implementation the results were validated against, if there is one.

Every algorithm is validated in two ways: against the reference implementation
on stored fixtures where one exists, and on synthetic data drawn from
`optix.models` where the truth is known. The usage section says which.
