# Kask foreground/background flat field

!!! info "At a glance"
    **Estimates:** `ShadingEstimate` (`flat` = foreground profile, `dark` = background profile scaled to the mean background level) \
    **Noise model:** Poisson (used to find the background) \
    **Streaming:** per-frame \
    **Needs many frames:** yes (tens for the background, hundreds for the foreground) \
    **Reference:** Kask et al., Journal of Microscopy, 2016, [doi:10.1111/jmi.12404](https://doi.org/10.1111/jmi.12404)

## Overview

`KaskFlatField` estimates two shading profiles of a fluorescence channel from
the sample images themselves: a **background profile** $\mathbf{s}_b$, which
shapes the out-of-focus and medium signal, and a **foreground profile**
$\mathbf{f}$, which shapes the signal of in-focus objects
([Kask et al. 2016](https://doi.org/10.1111/jmi.12404)). The authors find that
the two are significantly different. Each frame is segmented into background
and foreground pixels, the two sets are accumulated separately, and each
profile is fitted to its own data, the background first.

Use it on high-content screens: many fields of view of cells in one channel,
with some background visible in most of them. It needs a stable optical path,
because one pair of profiles is shared by all frames. The background profile
needs a few tens of images; the foreground profile, which is much harder to
measure, needs hundreds. It is not suited to confluent samples with no
visible background, where the background profile cannot be estimated. For a
baseline that drifts frame to frame inside the flat field, use
[BaSiC](basic.md).

It returns a `ShadingEstimate` in `estimates["shading"]`, plus the background
level of every fitted frame in `background_level_`. The random error of each
profile, the numbers of images used, the mean background level and the noise
constant (`noise_constant`) are in `meta`.

## Image formation (optics)

The model is the two-profile form of the shading model of
[Image formation and noise](../../concepts/image-formation.md): the
background is additive and shaded by its own profile $\mathbf{s}_b$, the
signal is multiplicative and shaded by $\mathbf{f}$:

=== "optix"

    $$
    \mathbf{y}_i = \mathbf{f} \odot \mathbf{x}_i + \beta_i\, \mathbf{s}_b
    $$

=== "Kask et al. 2016"

    $$
    N(x, y) = U(x, y)\, S_M(x, y) + S_A(x, y) \tag{1}
    $$

    with $S_A = B\,S_b$ and $S_M = S_f$ in the normalised form below.

Both profiles are normalised to mean 1, $\bar{\mathbf{f}} = \bar{\mathbf{s}}_b = 1$,
and $\beta_i$ is the mean background intensity of frame $i$, in ADU
(camera offset included). The correction is

=== "optix"

    $$
    \hat{\mathbf{x}}_i = \frac{\mathbf{y}_i - \hat\beta_i\, \mathbf{s}_b}{\mathbf{f}} + \hat\beta_i
    $$

=== "Kask et al. 2016"

    $$
    \hat{U}(x, y) = \frac{N(x, y) - \hat{B}\,\hat{S}_b(x, y)}{\hat{S}_f(x, y)} + \hat{B} \tag{3}
    $$

    (Eq. 2 is the same without the term $\hat{B}$ added back and with the
    profiles not normalised.)

The background is corrected additively and the foreground multiplicatively.
The constant $\hat\beta_i$ added back keeps the corrected image positive and
restores the original background level.

| optix | [Kask et al. 2016](https://doi.org/10.1111/jmi.12404) | Meaning |
|---|---|---|
| $\mathbf{y}_i$ | $N$ | recorded image |
| $\mathbf{x}_i$ | $U$, $\hat{U}$ | latent and corrected image |
| $\mathbf{f}$ | $S_f$ (normalised $S_M$) | foreground profile, `flat` |
| $\mathbf{s}_b$ | $S_b$ (normalised $S_A$) | background profile |
| $\beta_i$, $\hat\beta_i$ | $B$, $\hat{B}$ | mean background intensity of frame $i$ |
| $\bar\beta$ | | mean of $\beta_i$ over the images used in the background fit |

The estimate stores $\mathbf{f}$ in `flat` (`meta["flat_norm"] == "mean"`) and
$\mathbf{d} = \bar\beta\,\mathbf{s}_b$ in `dark`, in ADU. Unlike the shading
model of the other estimators, the background here is *not* lit through
$\mathbf{f}$, and its amplitude changes per frame:
`meta["dark_scaling"] == "per-frame"` records that `apply` re-estimates
$\beta_i$ for every image, whereas
[`ShadingEstimate.correct`](../../reference/shading.md) subtracts `dark` as it
is, that is, with $\beta_i = \bar\beta$.

## Probabilistic model

The method has no likelihood. It rests on four assumptions:

- **Stable optics.** Both normalised profiles are the same for all images of
  the data set (all wells and fields).
- **Poisson-like noise.** For a pure counting process,
  $\operatorname{var}(n) = \langle n \rangle$ (Eq. 4). The ratio of local
  variance to local mean, the *noise image*, is therefore a constant
  $\kappa_{\text{n}}$ of the detection channel on the background, whatever the local
  intensity, and much larger on objects.
- **Smooth profiles.** Each profile is a smooth function of the position,
  imposed on a binned image ($17 \times 16$ in the paper) by regularisation, or by a
  fourth-order polynomial.
- **Background present.** A frame is useful for the background profile only
  if at least 15 % of its pixels are background.

Smoothing is the minimisation, over the binned grid, of

=== "optix"

    $$
    F(\mathbf{t}) = \sum_{i,j} (e_{ij} - t_{ij})^2
    + \lambda \sum_{i,j} \bigl[(\mathbf{L}_x \mathbf{t})_{ij}^2 + (\mathbf{L}_y \mathbf{t})_{ij}^2\bigr]
    $$

=== "Kask et al. 2016"

    $$
    F = \frac{1}{N}\Bigl[\sum_{i,j} (E_{ij} - T_{ij})^2
    + R \sum_{i,j} \bigl(Lxx_{ij}^2 + Lyy_{ij}^2\bigr)\Bigr] \tag{5}
    $$

| optix | [Kask et al. 2016](https://doi.org/10.1111/jmi.12404) | Meaning |
|---|---|---|
| $\mathbf{e}$ | $E$ | measured binned image |
| $\mathbf{t}$ | $T$ | smooth solution |
| $\lambda$ | $R$ | regularisation weight |
| $\mathbf{L}_x$, $\mathbf{L}_y$ | $Lxx$, $Lyy$ (applied to $T$) | second-difference operators along $x$ and $y$ |
| $\kappa_{\text{n}}$ | | noise constant of the channel, $\operatorname{var}/\text{mean}$ on the background (written $\kappa$ in the docstrings) |
| $N$ (paper) | | number of pixels; $1/N$ does not change the minimiser |
| $\bar{\mathbf{e}}$ | | mean over the images of $e_k/a_k$ |
| $W_{ij}$ | | number of images that define bin $ij$ |

The paper writes Eq. 5 for one image $E$ ("a noisy image") and says that
"a data set is usually an image of the size 17 × 16", which can be read as
a single combined image. With many images, the code uses the **summed
reading** instead: the deviation term is summed over the $K$ images of a
set, each divided by its own level $a_k$ (scale differences between images
are not roughness),

$$
F(\mathbf{t}, \mathbf{a}) = \sum_{k=1}^{K} \sum_{i,j}
\bigl(e_{kij}/a_k - t_{ij}\bigr)^2
+ \lambda \sum_{i,j} \bigl[(\mathbf{L}_x \mathbf{t})_{ij}^2 + (\mathbf{L}_y \mathbf{t})_{ij}^2\bigr]
\quad \text{s.t.}\ \bar{\mathbf{t}} = 1 ,
$$

summed over the bins defined in each image (a bin without segmented pixel is
undefined). Both readings are consistent with Eq. 5; this one is an
interpretation, stated again under
[Faithful mode](#faithful-mode-and-extensions).

**Why the summed reading.** Let $\bar{\mathbf{e}}$ be the mean of the
$e_k/a_k$ and $W_{ij}$ the number of images that define bin $ij$. Then the
deviation term equals $\sum_{ij} W_{ij} (\bar e_{ij} - t_{ij})^2$ plus a
constant, so the summed reading is a weighted fit of $\bar{\mathbf{e}}$ with
weight $\lambda$ on the roughness. When every image defines every bin,
$W_{ij} = K$ and this is the fit of $\bar{\mathbf{e}}$ with the coefficient
$\lambda/K$. The noise in $\bar{\mathbf{e}}$ has variance $\sigma^2/K$, so the
best $\lambda$ of the summed reading depends on the noise $\sigma^2$ of one
image and not on $K$. This agrees with the paper, which fixes one value per
profile for the whole study (0.2 and 20, "in the rest of this study") while
the number of images ranges from about 10 to several hundred (Fig. 8), and
which finds that "the optimal value of the regularization coefficient tends to
be bigger for data sets with bigger noise" (Fig. 7). Fitting
$\bar{\mathbf{e}}$ with a fixed $\lambda$ would smooth more and more as $K$
grows.

## Inference / derivation

Profiles are estimated in one pass in this order:

1. **Segment every frame**, with the current background profile as the
   reference (before there is one, a centred profile
   $1 - c\rho^2$ whose curvature $c$ minimises the noise level).
    - *Noise image.* $\mathbf{y}/\mathbf{p}$ is the frame divided by the
      reference profile $\mathbf{p}$, which removes the contribution of the
      slope of the profile to the local variance. Its local variance over
      local mean in a $9 \times 9$ window, multiplied by $\mathbf{p}$, is
      $\approx \kappa_{\text{n}}$ on the background. The paper treats
      $\kappa_{\text{n}}$ as a constant of the detection channel, estimated
      from a series of images containing background; by default
      (`kappa="series"`) it is one value per group, the median over the
      frames of the mode of each frame's noise image, a running median while
      streaming (an optix choice). A float sets it directly; `"per-frame"`
      takes the mode of each frame (an extension).
    - *Background pixels* are those where the noise image is below an upper
      threshold in units of $\kappa_{\text{n}}$ and where a local-intensity threshold
      removes flat bright regions inside objects.
    - *Foreground pixels* are those above, in each block of the binning grid,
      the block mean plus $5$ expected standard deviations
      $\sqrt{\kappa_{\text{n}} \cdot \text{mean}}$, without looking at neighbouring
      blocks.
2. **Bin** each segmented image by block averaging into $17 \times 16$
   images.
3. **Background profile.** Discard frames with under 15 % background. Fit the
   rest globally ($e_k \approx a_k \mathbf{t}$ with smoothing), characterise
   each frame by its mean square deviation from the fit, discard outliers
   and repeat until none remain.
4. **Foreground profile.** Take consecutive sets of 16 binned foreground
   images, filter outliers in each and average it; subtract the background
   $\bar\beta\,\mathbf{s}_b$ from each average; fit the averages globally.
5. **Quality control.** Split the data into subsets, fit each (with $\lambda$
   scaled by the share $n_s/n$ of the images of the subset, so that it smooths
   as much as the full fit), and take the spread of their profiles as the
   range of the random error. A profile
   above its limit is replaced by 1, with a warning, and the foreground
   profile is only estimated when the background profile passes.

Each global fit alternates two steps until the profile changes by less than
$10^{-9}$. It is not coordinate descent on one objective, and the paper does
not give the objective of the global fit:

- *Levels.* $a_k = \arg\min_{a} \sum_{ij} (e_{kij} - a\,t_{ij})^2$, by least
  squares over the defined bins.
- *Profile.* The Eq. 5 smooth fit $\mathbf{t}$ of $\bar{\mathbf{e}}$, with
  weights $W_{ij}$, renormalised to mean 1. That fixes the scale ambiguity
  between $a_k$ and $\mathbf{t}$.
 Profiles go from bins to pixels by
linear interpolation between bin centres (the polynomial is evaluated at the
pixels).

Applying the correction needs $\hat\beta_i$: the frame is segmented with the
fitted background profile and $\hat\beta_i = \sum y / \sum s_b$ over its
background pixels (the mean level of the fit if the frame has none).

## Algorithm

```text
Input: stack Y (frames of H × W), per group of frames
Bins (16, 17); window 9; noise threshold 1.5 κn; intensity threshold 1.25
κn: median over the frames of the mode of the noise image (running median)
For each frame y, in storage order:
    profile p = current background profile (or 1 - c ρ², c minimising noise)
    noise image = p · var(y/p) / mean(y/p) over a 9 × 9 window; mode → κn
    background mask = noise < 1.5 κn and local level < 1.25 × median level
    foreground mask = y > block mean + 5 √(κn · block mean)
    bin y over each mask into (16, 17) images; keep only these
    if ≥ 15 % background: count the frame for the background profile
    after 10, 20, 40, ... background frames: refit the provisional p
Background: outlier-filtered global fit (λ = 0.2), repeated until no outlier
    (outlier: mean square deviation > 3 × median over the images)
Levels: β_i = Σ background / Σ (s_b over background bins), per frame
    β̄ = mean of β_i over the images used in the background fit
Foreground: for each set of 16 binned images: filter outliers, average,
    subtract β̄ s_b; fit all averages globally (λ = 20)
Random error: 4 contiguous subsets, λ scaled by n_s / n; background ≤ 2 %, foreground ≤ 4 %
Return flat = f, dark = β̄ s_b, background_level_ = β_i, κn
```

## Faithful mode and extensions

`faithful=True` and `faithful=False` give the same values for every keyword,
and both follow the paper's method. The table lists the keywords that
choose between variants or set limits described in the paper.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `model` | profile model: image regularisation (Eq. 5) or fourth-order polynomial | `"regularization"` | `"regularization"` |
| `regularization` | weight $\lambda$ of Eq. 5 for the (background, foreground) profile | `(0.2, 20)` | `(0.2, 20)` |
| `kappa` | noise constant: `"series"` (one value per group), a float, or `"per-frame"` (extension) | `"series"` | `"series"` |
| `max_error` | limits on the range of the random error (None disables) | `(0.02, 0.04)` | same |

### The polynomial model

The paper compares a second-order polynomial, a fourth-order polynomial and
regularisation (Table 1), reports that, on background data, the fourth-order polynomial and
regularisation have "more or less equal performance", and retains
regularisation for the study. The estimator's paper preset follows that
selection. The fourth-order polynomial is therefore a method of the paper but
not the preset, and `model="polynomial"` makes `is_faithful` false:

```pycon
>>> from optix.shading import KaskFlatField
>>> KaskFlatField().is_faithful
True
>>> KaskFlatField(model="polynomial").is_faithful
False
```

With `model="polynomial"`, `regularization` is not used.

### Interpretations and choices not specified in the paper

- **Eq. 5 with many images (summed reading).** Eq. 5 is written for one
  image. Here the deviation term is summed over the $K$ images of a set, each
  divided by its level, which is a weighted fit of $\bar{\mathbf{e}}$ (the
  mean of $e_k/a_k$) with weights $W_{ij}$; it equals the fit with
  coefficient $\lambda/K$ only when every image defines every bin. The other
  reading, fitting one combined image with the coefficient $\lambda$ itself,
  smooths $K$ times as strongly for the same $\lambda$. The reasons for the
  choice are given [above](#probabilistic-model). It is not a keyword.
- **Global fit.** The paper does not give the objective of the global fit;
  the alternation of levels and smooth fit described above is an optix choice.
- **Noise constant.** The paper takes $\kappa_{\text{n}}$ as a constant of
  the detection channel, estimated from a series of images. `kappa="series"`
  does so with the median of the per-frame modes (a running median while
  streaming), which are optix choices; `kappa="per-frame"` is an extension.
- **Second differences** are taken between neighbouring bins, at interior bins
  only, on the binned grid.
- **Stopping.** The paper stops collecting images when the random error is
  below the limit. Here all images are used and the error is checked once at
  the end.
- **Foreground selection.** The block threshold (mean + 5 standard
  deviations) excludes dim objects in dark corners, which biases the
  foreground profile there; this is a property of the paper's segmentation.
  The authors report that, in their tests on synthetic images with a known
  profile and objects of varying intensity, "distortions of the estimated
  profile are negligible"; the effect can matter for deep foreground profiles
  or dim objects.
- **Not implemented:** the method of overlapping fields is the authors'
  evaluation of the correction, which needs specially acquired images, not
  an estimation step.
- Not specified in the paper, with their defaults: `noise_threshold=1.5`,
  `intensity_threshold=1.25`, `outlier_factor=3`, `subsets=4` (contiguous),
  `max_error` of the foreground (the paper gives 2 % as an example for the
  background), the mode of the histogram of the logarithm of the noise image as the
  per-frame $\kappa_{\text{n}}$, the number of bins (the paper fixes the
  block size, 80 × 64 or 64 × 64 pixels; `bins=(16, 17)` gives these blocks on
  1360 × 1024 frames only, and $\lambda$ keeps its per-bin meaning on other
  sizes), the curvatures tried for the initial
  profile ($c \in \{0, 0.05, \ldots, 0.6\}$), the refits of the provisional
  profile at 10, 20, 40, ... background frames, the mean-square-deviation
  statistic of the outlier filter, the handling of bins without segmented
  pixels, and `group_size`: a trailing incomplete set is not used.

## Memory and streaming

The estimator is tagged `per-frame`. Frames are segmented one at a time, in
storage order, and only their binned images are kept: the sums and the pixel
counts of the background and of the foreground, $4hw$ values per frame (the
running median of $\kappa_{\text{n}}$ keeps one number per frame).
Memory is therefore $O(N\,hw)$ for $N$ frames and $h \times w$ bins, not
$O(N\,HW)$; only one batch of frames and its masks is in memory at a time. The
provisional background profile is refitted at fixed counts of background
frames, so the result depends on the frame order but not on the batch size.
All smoothing is dense and small ($hw = 272$ unknowns). Cost is dominated by
the segmentation, which grows with the number of pixels, and by the repeated
global fits of the outlier filters.

With a layout that has group axes, each group (for instance each channel) has
its own pair of profiles, as the paper estimates a separate set per detection
channel. [`apply`](../../reference/shading.md) works on frame batches and
re-estimates $\beta_i$ per image.

## Usage

```pycon
>>> import torch
>>> from optix.shading import KaskFlatField
>>> # 1. Poisson images of bright disks, with distinct background and
>>> #    foreground profiles.
>>> H, W, N = 128, 136, 96
>>> y = torch.linspace(-1, 1, H)[:, None]
>>> x = torch.linspace(-1, 1, W)[None, :]
>>> def dome(depth):
...     p = 1 - 0.5 * depth * (x**2 + y**2)
...     return p / p.mean()
>>> s_b, s_f = dome(0.3), dome(0.5)
>>> g = torch.Generator().manual_seed(0)
>>> rows, cols = torch.arange(H)[:, None], torch.arange(W)[None, :]
>>> frames = []
>>> for _ in range(N):
...     u = torch.zeros(H, W)
...     for _ in range(40):
...         cy, cx = torch.rand(2, generator=g) * torch.tensor([H, W])
...         disk = (rows - cy) ** 2 + (cols - cx) ** 2 < 25
...         u = torch.where(disk, 2000.0, u)
...     level = 80 + 40 * torch.rand(1, generator=g)
...     frames.append(torch.poisson(s_f * u + level * s_b, generator=g))
>>> data = torch.stack(frames)
>>> # 2. Fit, then correct.
>>> kask = KaskFlatField(group_size=8).fit(data)
>>> shading = kask.estimates["shading"]
>>> corrected = kask.apply(data)
>>> shading.flat.shape, shading.meta["dark_scaling"]
(torch.Size([128, 136]), 'per-frame')
>>> kask.background_level_.shape
torch.Size([96])
>>> shading.meta["noise_constant"] > 0
True
>>> # 3. Compare the profiles to the truth (relative rms error).
>>> bool((shading.flat / s_f - 1).square().mean().sqrt() < 0.05)
True
>>> dark = shading.dark / shading.dark.mean()
>>> bool((dark / s_b - 1).square().mean().sqrt() < 0.01)
True
```

Common variations: `model="polynomial"` for the fourth-order polynomial,
`regularization=(0.5, 20)` to smooth the background more, `max_error=(None,
None)` to apply the profiles whatever their random error, and
`group_size=16` (the paper's value) when hundreds of images are available. See
the [API reference](../../reference/shading.md) for every option.

## References

!!! references "References"

    1. Kask P, Palo K, Hinnah C, Pommerencke T. *Flat field correction for
       high-throughput imaging of fluorescent samples.* J Microsc
       **263**(3):328–340, 2016.
       [doi:10.1111/jmi.12404](https://doi.org/10.1111/jmi.12404)

    ??? quote "BibTeX"

        ```bibtex
        @article{kask2016flat,
          author  = {Kask, Peet and Palo, Kaupo and Hinnah, Chris and Pommerencke, Thora},
          title   = {Flat field correction for high-throughput imaging of fluorescent samples},
          journal = {Journal of Microscopy},
          volume  = {263},
          number  = {3},
          pages   = {328--340},
          year    = {2016},
          doi     = {10.1111/jmi.12404},
        }
        ```
