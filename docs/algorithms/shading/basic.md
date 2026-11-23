# BaSiC

!!! info "At a glance"
    **Estimates:** `ShadingEstimate` (`flat`, `dark` with `darkfield=True`, `baseline` with `timelapse=True`) \
    **Noise model:** none \
    **Streaming:** per-frame \
    **Needs many frames:** yes \
    **Reference:** Peng et al., Nature Communications, 2017, [doi:10.1038/ncomms14836](https://doi.org/10.1038/ncomms14836)

## Overview

BaSiC estimates the shading of a microscope, a smooth flat field
$\mathbf{f}$ and optionally a smooth dark field $\mathbf{d}$, from many images
acquired under the same conditions, without calibration images. For time-lapse
data it also estimates a per-frame baseline $b_i$, a uniform background that
drifts over time (photobleaching of the medium, for instance).
It treats the stack as a low-rank matrix (the shading shared by all frames)
plus a sparse residual (the specimen), following
[Peng et al. 2017](https://doi.org/10.1038/ncomms14836).

Use it on tiles of a mosaic or on a time-lapse, with tens to hundreds of frames
whose content varies from frame to frame. When the frames are few, or the
specimen covers the same pixels in every frame, the decomposition cannot tell
specimen from shading. [CIDRE](cidre.md) fits the per-pixel intensity
distributions instead, and also estimates a dark field.

It returns a `ShadingEstimate` in `estimates["shading"]`, and
[`apply`](../../reference/shading.md) corrects frames with it.

## Image formation (optics)

BaSiC uses the shading model of
[Image formation and noise](../../concepts/image-formation.md) as it is:

=== "optix"

    $$
    \mathbf{y}_i = \mathbf{f} \odot (\mathbf{x}_i + b_i) + \mathbf{d},
    \qquad
    \hat{\mathbf{x}}_i = \frac{\mathbf{y}_i - \mathbf{d}}{\mathbf{f}} - b_i + b_{\text{norm}}
    $$

=== "Peng et al. 2017"

    $$
    I_i^{\text{meas}}(x) = S(x)\,\bigl(B_i + I_i^R(x)\bigr) + D(x)
    $$

The flat field is smooth with mean 1, the dark field smooth, and the baseline
$b_i$ constant over a frame. $b_{\text{norm}}$ (`baseline_norm`) is an
arbitrary background added back after correction: 0 removes the background
(fluorescence), $\bar{b}$ (`"mean"`) keeps the raw intensity range (bright
field).

| optix | [Peng et al. 2017](https://doi.org/10.1038/ncomms14836) | Meaning |
|---|---|---|
| $\mathbf{y}_i$ | $I_i^{\text{meas}}$ ($I$) | recorded frame |
| $\mathbf{x}_i$ | $I_i^R$ ($F$) | latent image |
| $\mathbf{f}$ | $S^F$ ($S$) | flat field |
| $\mathbf{d}$ | $S^D$ ($D$) | dark field |
| $b_i$ | $B_i$ | baseline of frame $i$ |
| $\mathbf{L}$ | $I_B$ | low-rank part |
| $\mathbf{E}$ | $I_R$ | sparse residual |

## Probabilistic model

BaSiC has no explicit noise model: it is a regularised decomposition. The
fitting frames, sorted at each pixel over the frames and resized to the working
size (128 × 128), form the columns of the measurement matrix $\mathbf{Y}$.
It is split into a low-rank part and a sparse residual,

=== "optix"

    $$
    \mathbf{Y} = \mathbf{L} + \mathbf{E}, \qquad
    \mathbf{L} = \mathbf{f}\,\mathbf{b}^\top + \mathbf{z}\,\mathbf{1}^\top, \qquad
    \mathbf{z} = d_Z (\mathbf{1} - \mathbf{f}) + \mathbf{d}_R ,
    $$

    $$
    \min_{\mathbf{f}, \mathbf{b}, d_Z, \mathbf{d}_R, \mathbf{E}}\;
    \lVert \mathbf{W} \odot \mathbf{E} \rVert_1
    + \lambda_f \lVert \boldsymbol{\Phi}^\top \mathbf{f} \rVert_1
    + \lambda_d \lVert \boldsymbol{\Phi}^\top \mathbf{d}_R \rVert_1
    + \lambda_d \lVert \mathbf{d}_R \rVert_1
    \quad \text{s.t.}\quad \mathbf{Y} = \mathbf{L} + \mathbf{E},\;
    d_Z \in [0, \min \mathbf{Y}] ,
    $$

=== "Peng et al. 2017"

    $$
    I = I_B + I_R, \qquad I_B = S\,B^\top + D_Z (1 - S) + D_R ,
    $$

    $$
    \min\; \lVert W \circ I_R \rVert_1
    + \lambda_s \lVert \mathcal{F}(S) \rVert_1
    + \lambda_d \bigl(\lVert \mathcal{F}(D_R) \rVert_1 + \lVert D_R \rVert_1\bigr)
    $$

    with the reweighting of Eq. (9) and the default $\lambda$ of
    Supplementary Note 4.

with $\boldsymbol{\Phi}$ the orthonormal 2-D DCT basis, so that
$\boldsymbol{\Phi}^\top \mathbf{f}$ are the DCT coefficients of $\mathbf{f}$;
the L1 norms on them make the fields smooth. The dark field is
$\mathbf{d} = d_Z + \mathbf{d}_R$. The weights are reweighted-L1,

$$
\mathbf{W} = \frac{1}{\lvert \mathbf{E} / \mathbf{L} \rvert + \epsilon},
$$

scaled to mean 1. By default
$\lambda_f = \lVert \boldsymbol{\Phi}^\top \bar{\mathbf{y}} \rVert_1 / 800$ and
$\lambda_d = \lVert \boldsymbol{\Phi}^\top \bar{\mathbf{y}} \rVert_1 / 2000$,
with $\bar{\mathbf{y}}$ the per-pixel mean frame divided by its mean.

| optix | [Peng et al. 2017](https://doi.org/10.1038/ncomms14836) |
|---|---|
| $\mathbf{Y}$ | $I$ (measurement matrix) |
| $\boldsymbol{\Phi}^\top \cdot$ | $\mathcal{F}(\cdot)$ (described as a Fourier transform, a DCT in the authors' code) |
| $\lambda_f$, $\lambda_d$ | $\lambda_s$, $\lambda_d$ |
| $d_Z$, $\mathbf{d}_R$ | $D_Z$, $D_R$ |
| $\mathbf{W}$, $\epsilon$ | $W$, $\varepsilon$ |

## Inference / derivation

The strategy is a convex low-rank + sparse decomposition, made robust by
reweighting:

- For fixed weights $\mathbf{W}$, the problem is convex and is solved by the
  linearised augmented Lagrangian method (LADM), which alternates
  soft-thresholding of $\mathbf{E}$, of the DCT coefficients of $\mathbf{f}$
  and $\mathbf{d}_R$, and an update of $\mathbf{b}$ and $d_Z$.
- The weights are then recomputed from the current $\mathbf{E}$ and
  $\mathbf{L}$ and the problem is solved again, so that pixels occupied by the
  specimen weigh less and less (reweighted L1).
- Sorting each pixel over the frames makes every column a quantile of the
  intensity distribution, which makes the low-rank model fit better when
  objects move from frame to frame.
- The scale shared by $\mathbf{f}$ and $\mathbf{x}_i$ is fixed by normalising
  $\mathbf{f}$ to mean 1 (`meta["flat_norm"] == "mean"`); the fields are then
  resized back to the image size.

**Baseline.** After sorting, the fitted $b_i$ belong to sorted frames, not to
real ones. With `timelapse=True`, each $b_i$ is re-estimated on its own,
unsorted frame with $\mathbf{f}$ and $\mathbf{d}$ fixed, by a reweighted
$\min_b \lVert \mathbf{W} \odot (\mathbf{y}_i - b\,\mathbf{f} - \mathbf{d}) \rVert_1$
(a weighted median). It is in units of the flat-corrected signal
(`meta["baseline_units"] == "signal"`).

## Algorithm

```text
Input: stack Y
For each group:
    Pick max_frames = 512 fitting frames; sort each pixel; resize to 128 x 128
    λ_f, λ_d from the mean frame (unless given); W = 1
    Repeat (at most max_reweight = 10 times, until f and d change by < 10⁻³):
        Solve the weighted problem by LADM (tol = 10⁻³, at most 500 iterations)
        W = 1 / (|E / L| + ε), ε = 0.1, scaled to mean 1
    Normalise f to mean 1; resize f, d to the image size
    If timelapse: one pass over the stack to estimate every b_i
Return ShadingEstimate(flat=f, dark=d, baseline=b)
```

## Faithful mode and extensions

`faithful=True` (the default) is the `paper` preset. The two presets differ in
one keyword only:

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `sort_intensity` | sort each pixel over the fitting frames before the fit | `True` | `False` |

`faithful=False` fits the unsorted frames, which is more accurate when the
background drifts between frames; `fit_baseline_` then holds the baselines
of the fitting frames from the decomposition itself.

Interpretations of the paper, used in both presets:

- $\mathcal{F}$ is the orthonormal 2-D DCT: the "Fourier transform" of the
  Supplementary Information refers to the DCT of the authors' implementation.
- The weights of Eq. (9) are scaled to mean 1, which the $\lambda$ of
  Supplementary Note 4 are taken to assume.
- The regions "$S \ge 1$" and "$S < 1$" of the dark-field update are read as
  $\mathbf{f} \ge \operatorname{mean}(\mathbf{f})$ and
  $\mathbf{f} < \operatorname{mean}(\mathbf{f})$, $\mathbf{f}$ being
  unnormalised there.
- The frames are sorted at full resolution and then resized to the working
  size.

Extensions, off in both presets:

| Keyword | Changes | Default |
|---|---|---|
| `variant="basicpy"` | the solver of [BaSiCPy](https://github.com/peng-lab/BaSiCPy) 2.0.0 (`fitting_mode="approximate"`): weights divided by the frame mean of $\mathbf{L}$, $b_i \ge 0$, $\mathbf{f}$ started at 1, dark field fitted on frames with $b_i < 1$, $\lambda_d = \lambda_f / 10$. With `sort_intensity=False` and `reweight_tol=1e-2` it reproduces BaSiCPy | `"paper"` |
| `baseline_method="basicpy"` | BaSiCPy's `fit_only_baseline`, a reweighted-L1 ALM on the pixels below each frame's 10 % quantile; biased low where the flat field varies (about −11 % for a flat of ±25 %) | `"paper"` |
| `sampling="reservoir"` | random fitting frames, seeded by `seed` | `"stride"` |
| `normalize_weights=False` | the weights of Eq. (9) without scaling | `True` |

## Memory and streaming

BaSiC is tagged `per-frame`. The fit reads the `max_frames` fitting frames of
each group, plus one pass over the whole stack with `timelapse=True`. Sorting
needs all fitting frames of a strip of rows at once: the strips are sized to
the memory budget, and each fitting frame is read once per strip. The
decomposition itself works on a $P \times K$ matrix at the working size
($P = 128^2$, $K \le 512$), so its cost does not grow with the image size or
the length of the stack. [`apply`](../../reference/shading.md) works on frame
batches. To trade accuracy for speed, lower `working_size` or `max_frames`.

## Usage

```python
import torch
from optix.models import random_shading, simulate_shading
from optix.shading import BaSiC

# 1. A synthetic stack with known truth.
truth = random_shading((64, 64), seed=0)
x = 100 + 50 * torch.rand(40, 64, 64)
data = simulate_shading(x, truth)

# 2. Fit.
basic = BaSiC(working_size=64).fit(data)
flat = basic.estimates["shading"].flat

# 3. Correct.
corrected = basic.apply(data)

# 4. Compare to the truth (both flats have mean 1).
error = ((flat - truth.flat).abs() / truth.flat).max()
```

Common variations:

```python
# Dark field and time-lapse baseline, unsorted frames.
basic = BaSiC(faithful=False, darkfield=True, timelapse=True).fit(data)
baseline = basic.estimates["shading"].baseline

# Reproduce BaSiCPy.
basic = BaSiC(variant="basicpy", sort_intensity=False, reweight_tol=1e-2)
```

See the [API reference](../../reference/shading.md) for every option.

## References

!!! references "References"

    1. Peng T, Thorn K, Schroeder T, Wang L, Theis FJ, Marr C, Navab N.
       *A BaSiC tool for background and shading correction of optical
       microscopy images.* Nat Commun **8**:14836, 2017.
       [doi:10.1038/ncomms14836](https://doi.org/10.1038/ncomms14836)

    ??? quote "BibTeX"

        ```bibtex
        @article{peng2017basic,
          author  = {Peng, Tingying and Thorn, Kurt and Schroeder, Timm and Wang, Lichao and Theis, Fabian J. and Marr, Carsten and Navab, Nassir},
          title   = {A {BaSiC} tool for background and shading correction of optical microscopy images},
          journal = {Nature Communications},
          volume  = {8},
          pages   = {14836},
          year    = {2017},
          doi     = {10.1038/ncomms14836},
        }
        ```
