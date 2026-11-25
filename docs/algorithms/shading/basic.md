# BaSiC

!!! info "At a glance"
    - **Estimates:** `ShadingEstimate` (`flat`, `dark` with `darkfield=True`, `baseline` with `timelapse=True`)
    - **Noise model:** none
    - **Streaming:** per-frame
    - **Needs many frames:** yes
    - **Reference:** Peng et al., Nature Communications, 2017, [doi:10.1038/ncomms14836](https://doi.org/10.1038/ncomms14836)

## Overview

BaSiC estimates the shading of a microscope, a smooth flat field
$\mathbf{f}$ and optionally a smooth dark field $\mathbf{d}$, from many images
acquired under the same conditions, without calibration images. For time-lapse
data it also estimates a per-frame baseline $b_n$, a uniform background that
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
    \mathbf{y}_n = \mathbf{f} \odot (\mathbf{x}_n + b_n) + \mathbf{d},
    \qquad
    \hat{\mathbf{x}}_n = \frac{\mathbf{y}_n - \mathbf{d}}{\mathbf{f}} - b_n + b_{\text{norm}}
    $$

=== "Peng et al. 2017"

    $$
    I_i^{\text{meas}}(x) = S(x)\,\bigl(B_i + I_i^R(x)\bigr) + D(x)
    $$

The flat field is smooth with mean 1, the dark field smooth, and the baseline
$b_n$ constant over a frame. $b_{\text{norm}}$ (`baseline_norm`) is an
arbitrary background added back after correction: 0 removes the background
(fluorescence), $\bar{b}$ (`"mean"`) keeps the raw intensity range (bright
field).

Frames have $\lvert\Omega\rvert$ pixels, flattened into vectors, and the stack (or group) has
$N$ frames; products and divisions of vectors are pixel-wise.

| optix | [Peng et al. 2017](https://doi.org/10.1038/ncomms14836) | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}_n$ | $I_i^{\text{meas}}$ ($I$) | recorded frame $n$ | $\mathbb{R}^{\lvert\Omega\rvert}$, one per frame |
| $\mathbf{x}_n$ | $I_i^R$ ($F$) | latent (shading-free) image of frame $n$ | $\mathbb{R}^{\lvert\Omega\rvert}$, one per frame |
| $\mathbf{f}$ | $S^F$ ($S$) | flat field, multiplicative gain | $\mathbb{R}^{\lvert\Omega\rvert}$, shared by all frames of a group |
| $\mathbf{d}$ | $S^D$ ($D$) | dark field, additive offset | $\mathbb{R}^{\lvert\Omega\rvert}$, shared by all frames of a group |
| $b_n$ | $B_i$ | baseline (uniform background) of frame $n$ | scalar, one per frame |
| $b_{\text{norm}}$ | | background added back after correction | scalar, one for the whole stack |

## Probabilistic model

BaSiC has no explicit noise model: it is a regularised decomposition. The
$K \le N$ fitting frames, sorted at each pixel over the frames and resized to
the working size (128 × 128, so $\lvert\Omega\rvert = 128^2$ here), form the columns of the
measurement matrix $\mathbf{Y} \in \mathbb{R}^{\lvert\Omega\rvert \times K}$.
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

with $\mathbf{b} \in \mathbb{R}^K$ the baselines of the fitting frames,
$\mathbf{1}$ a vector of ones, and
$\boldsymbol{\Phi} \in \mathbb{R}^{\lvert\Omega\rvert \times \lvert\Omega\rvert}$ the orthonormal 2-D DCT basis, so that
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

| optix | [Peng et al. 2017](https://doi.org/10.1038/ncomms14836) | Meaning | Shape |
|---|---|---|---|
| $\mathbf{Y}$ | $I$ | measurement matrix, one fitting frame per column | $\mathbb{R}^{\lvert\Omega\rvert \times K}$, per group |
| $\mathbf{L}$ | $I_B$ | low-rank part, the shading of every frame | $\mathbb{R}^{\lvert\Omega\rvert \times K}$, per group |
| $\mathbf{E}$ | $I_R$ | sparse residual, the specimen | $\mathbb{R}^{\lvert\Omega\rvert \times K}$, per group |
| $\mathbf{b}$ | $B$ | baselines of the fitting frames | $\mathbb{R}^K$, one entry per frame |
| $\boldsymbol{\Phi}^\top \cdot$ | $\mathcal{F}(\cdot)$ | DCT (described as a Fourier transform, a DCT in the authors' code) | $\mathbb{R}^{\lvert\Omega\rvert \times \lvert\Omega\rvert}$, fixed |
| $\lambda_f$, $\lambda_d$ | $\lambda_s$, $\lambda_d$ | smoothness weights of flat and dark fields | scalars, per group |
| $d_Z$ | $D_Z$ | uniform part of the dark field | scalar, per group |
| $\mathbf{d}_R$ | $D_R$ | spatially varying part of the dark field | $\mathbb{R}^{\lvert\Omega\rvert}$, per group |
| $\mathbf{W}$ | $W$ | reweighting of the residual | $\mathbb{R}^{\lvert\Omega\rvert \times K}$, per group |
| $\epsilon$ | $\varepsilon$ | floor of the reweighting | scalar |

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
- The scale shared by $\mathbf{f}$ and $\mathbf{x}_n$ is fixed by normalising
  $\mathbf{f}$ to mean 1 (`meta["flat_norm"] == "mean"`); the fields are then
  resized back to the image size.

**Statistical counterpart.** For fixed weights the objective is a MAP
estimate: an L1 data term is a Laplace likelihood for the specimen residual,
and the L1 penalties are Laplace (sparsity) priors on the DCT coefficients of
the fields. Reweighted L1 minimises a log-sum
penalty by successive linear upper bounds; equivalently, EM for a Laplace scale mixture, each weight being the
expected inverse scale of its residual (an IRLS-type robust M-estimator).

**Baseline.** After sorting, the fitted $b_n$ belong to sorted frames, not to
real ones. With `timelapse=True`, each $b_n$ is re-estimated on its own,
unsorted frame with $\mathbf{f}$ and $\mathbf{d}$ fixed, by a reweighted
$\min_b \lVert \mathbf{W} \odot (\mathbf{y}_n - b\,\mathbf{f} - \mathbf{d}) \rVert_1$
(a weighted median, the maximum-likelihood location under a Laplace
likelihood). It is in units of the flat-corrected signal
(`meta["baseline_units"] == "signal"`).

## Algorithm

**Input:** frames $\mathbf{y}_1, \ldots, \mathbf{y}_N \in \mathbb{R}^{\lvert\Omega\rvert}$,
optionally in groups. \
**Output:** `ShadingEstimate` with $\hat{\mathbf{f}}$, $\hat{\mathbf{d}}$ and,
with `timelapse=True`, $\hat b_1, \ldots, \hat b_N$.

For each group:

1. **Select** $K \le$ `max_frames` $= 512$ fitting frames, sort each pixel over
   them (`sort_intensity=True`) and resize them to 128 × 128, giving
   $\mathbf{Y} \in \mathbb{R}^{\lvert\Omega\rvert \times K}$.
2. **Initialise** $\lambda_f$, $\lambda_d$ from the mean frame (unless given)
   and $\mathbf{W} = \mathbf{1}$.
3. **Reweight**, at most `max_reweight` $= 10$ times, until $\mathbf{f}$ and
   $\mathbf{d}$ change by less than $10^{-3}$:
    1. **solve** the weighted problem by LADM (`tol` $= 10^{-3}$, at most 500
       iterations);
    2. **update** $\mathbf{W} = 1 / (\lvert \mathbf{E} / \mathbf{L} \rvert + \epsilon)$,
       $\epsilon = 0.1$, scaled to mean 1.
4. **Normalise** $\mathbf{f}$ to mean 1 and resize $\mathbf{f}$, $\mathbf{d}$
   to the image size.
5. **Baseline** (with `timelapse=True`): one pass over the stack, estimating
   each $b_n$ by the weighted median above.

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
| `variant="basicpy"` | the solver of [BaSiCPy](https://github.com/peng-lab/BaSiCPy) 2.0.0 (`fitting_mode="approximate"`): weights divided by the frame mean of $\mathbf{L}$, $b_n \ge 0$, $\mathbf{f}$ started at 1, dark field fitted on frames with $b_n < 1$, $\lambda_d = \lambda_f / 10$. With `sort_intensity=False` and `reweight_tol=1e-2` it reproduces BaSiCPy | `"paper"` |
| `baseline_method="basicpy"` | BaSiCPy's `fit_only_baseline`, a reweighted-L1 ALM on the pixels below each frame's 10 % quantile; biased low where the flat field varies (about −11 % for a flat of ±25 %) | `"paper"` |
| `sampling="reservoir"` | random fitting frames, seeded by `seed` | `"stride"` |
| `normalize_weights=False` | the weights of Eq. (9) without scaling | `True` |

## Memory and streaming

BaSiC is tagged `per-frame`. The fit reads the `max_frames` fitting frames of
each group, plus one pass over the whole stack with `timelapse=True`. Sorting
needs all fitting frames of a strip of rows at once: the strips are sized to
the memory budget, and each fitting frame is read once per strip. The
decomposition itself works on a $\lvert\Omega\rvert \times K$ matrix at the working size
($\lvert\Omega\rvert = 128^2$, $K \le 512$), so its cost does not grow with the image size or
the length of the stack. [`apply`](../../reference/shading.md) works on frame
batches. To trade accuracy for speed, lower `working_size` or `max_frames`.

## Usage

```python
from optix.models import phantoms, random_shading, simulate_shading
from optix.shading import BaSiC

# 1. A synthetic stack with known truth: 40 frames of cells, one flat field.
truth = random_shading((64, 64), seed=0)
x = phantoms.cells((40, 64, 64), seed=1) + 20  # on autofluorescence
data = simulate_shading(x, truth)

# 2. Fit.
basic = BaSiC(working_size=64).fit(data)
flat = basic.estimates["shading"].flat

# 3. Correct.
corrected = basic.apply(data)

# 4. Compare to the truth (both flats have mean 1).
error = ((flat - truth.flat).abs() / truth.flat).max()  # about 0.006
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
