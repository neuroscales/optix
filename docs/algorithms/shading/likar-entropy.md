# Likar (entropy minimisation)

!!! info "At a glance"
    **Estimates:** `ShadingEstimate` (`flat`, `dark`) \
    **Noise model:** none \
    **Streaming:** per-pixel-stats \
    **Needs many frames:** no \
    **Reference:** Likar et al., Journal of Microscopy, 2000, [doi:10.1046/j.1365-2818.2000.00669.x](https://doi.org/10.1046/j.1365-2818.2000.00669.x)

## Overview

`Likar` estimates a smooth flat field $\mathbf{f}$ and a smooth dark field
$\mathbf{d}$ from the image to be corrected, with no calibration image and no
other frame. It looks for the two quadratic surfaces that, once removed,
leave the corrected image with the lowest Shannon entropy of its grey-level
histogram, that is, the most uniform distribution of intensities
([Likar et al. 2000](https://doi.org/10.1046/j.1365-2818.2000.00669.x)). It
handles an additive and a multiplicative component together, which filtering
and surface-fitting methods cannot.

Use it on a single image, or on a few, when there is no calibration stack and
the shading is slowly varying (reflection or transmission microscopy of
objects of limited size, for instance). With many frames of a mosaic or a
time-lapse, [BaSiC](basic.md) and [CIDRE](cidre.md) use the redundancy between
frames and do not restrict the shading to a quadratic. The method has a
limit of its own: when the structure of the image is itself smooth and
large-scale, the entropy minimum can absorb that structure into the shading.

It returns a `ShadingEstimate` in `estimates["shading"]` with `flat` and `dark`
(no baseline), and [`apply`](../../reference/shading.md) corrects frames with
it.

## Image formation (optics)

`Likar` uses the shading model of
[Image formation and noise](../../concepts/image-formation.md) without a
baseline, as [CIDRE](cidre.md) does:

=== "optix"

    $$
    \mathbf{y}_i = \mathbf{f} \odot \mathbf{x}_i + \mathbf{d},
    \qquad
    \hat{\mathbf{x}}_i = \frac{\mathbf{y}_i - \hat{\mathbf{d}}}{\hat{\mathbf{f}}}
    $$

=== "Likar et al. 2000"

    $$
    N(x, y) = U(x, y)\, S_M(x, y) + S_A(x, y) \tag{3}
    $$

    $$
    \hat{U}(x, y) = \frac{N(x, y) - \hat{S}_A(x, y)}{\hat{S}_M(x, y)} \tag{5}
    $$

| optix | [Likar et al. 2000](https://doi.org/10.1046/j.1365-2818.2000.00669.x) | Meaning |
|---|---|---|
| $\mathbf{y}_i$ | $N$ | recorded image |
| $\mathbf{x}_i$ | $U$ | latent image |
| $\mathbf{f}$ | $S_M$ | multiplicative shading (flat field) |
| $\mathbf{d}$ | $S_A$ | additive shading (dark field) |
| $\mathbf{r} = (x, y)$ | $(x, y)$ | pixel position, origin at the image centre |
| $H \times W$ | $H \times W$ | image height and width |
| $G$ | $G$ | number of grey levels (`levels`) |

$\mathbf{d}$ is the additive part of the shading only: it has zero mean over the
image (below), so a constant offset is not estimated and stays in the corrected
image.

## Probabilistic model

**Assumptions.** The shading components are second-order polynomials of the
position, neutralised so that they change neither the mean level nor the
mean contrast of the image, and the corrected image is the one with the most
uniform intensity distribution.

The polynomials and their neutralisation (Eqs. 12–17), with
$\mathbf{r} = (x, y)$ measured from the image centre:

=== "optix"

    $$
    \begin{aligned}
    d(\mathbf{r}) &= a_1 x + a_2 y + a_3 x y
      + a_4 \bigl(x^2 - \tfrac{W^2}{12}\bigr)
      + a_5 \bigl(y^2 - \tfrac{H^2}{12}\bigr), \\
    f(\mathbf{r}) &= 1 + m_1 x + m_2 y + m_3 x y
      + m_4 \bigl(x^2 - \tfrac{W^2}{12}\bigr)
      + m_5 \bigl(y^2 - \tfrac{H^2}{12}\bigr),
    \end{aligned}
    $$

    so that $\sum_p d_p \approx 0$ and $\bar{f} \approx 1$.

=== "Likar et al. 2000"

    $$
    \begin{aligned}
    \hat{S}_A &= a_1 x + a_2 y + a_3 x y + a_4 \bigl(x^2 - \tfrac{W^2}{12}\bigr) + a_5 \bigl(y^2 - \tfrac{H^2}{12}\bigr) && (16)\\
    \hat{S}_M &= 1 + m_1 x + m_2 y + m_3 x y + m_4 \bigl(x^2 - \tfrac{W^2}{12}\bigr) + m_5 \bigl(y^2 - \tfrac{H^2}{12}\bigr) && (17)
    \end{aligned}
    $$

    obtained from the general quadratics (12, 13) under
    $\iint \hat{S}_A = 0$ (14) and $\tfrac{1}{WH}\iint \hat{S}_M = 1$ (15).

The coefficients $\mathbf{a}$ and $\mathbf{m}$ of the paper are the same symbols
in optix: $(a_1, \ldots, a_5)$ for the dark field, in intensity units per
pixel, and $(m_1, \ldots, m_5)$ for the flat field.

The objective is the entropy of the corrected image. With $h(n)$ the
histogram of the grey levels $n = 1 \ldots G$ and $p(n)$ its normalisation:

=== "optix"

    $$
    \{\mathbf{a}, \mathbf{m}\} = \arg\min_{\mathbf{a},\, \mathbf{m}}
    \; \mathcal{H}\!\left(\frac{\mathbf{y} - \mathbf{d}(\mathbf{a})}{\mathbf{f}(\mathbf{m})}\right),
    \qquad
    \mathcal{H} = -\sum_{n} p(n) \log p(n)
    $$

=== "Likar et al. 2000"

    $$
    \{\mathbf{a}_o, \mathbf{m}_o\} = \arg\min_{\{\mathbf{a}, \mathbf{m}\}}
    \left\{ H\!\left(\frac{N - \hat{S}_A}{\hat{S}_M}\right) \right\} \tag{18}
    $$

    $$
    H(N) = -\sum_{n=1}^{G} p(n) \log p(n) \tag{19}
    $$

The histogram is built so that the entropy is a smooth function of the
parameters (Eqs. 19–20, text):

- **Partial intensity interpolation.** A corrected value $g'$ between the
  integers $k$ and $k+1$ adds $k + 1 - g'$ to the bin $k$ and $g' - k$ to the
  bin $k + 1$.
- **Wider range.** The correction can move values out of $[1, G]$, so the
  histogram covers $[-G, 2G]$.
- **Blur.** The histogram is convolved with a triangular window,

  $$
  {}^{*}h(n) = \sum_{i=-t}^{t} h(n + i)\,(t + 1 - |i|), \qquad t = 2, \tag{20}
  $$

  and normalised afterwards, which gives $p(n)$.
- **Marginal grey values.** The pixels of the input image at the grey values 1
  and $G$ are ignored, since they may be saturated.

## Inference / derivation

The entropy has no closed-form minimiser, so the ten parameters are found by
direct search: Powell's direction-set method, which minimises along one
direction at a time with Brent's one-dimensional method
([Press et al. 1992](https://numerical.recipes/)), as in the paper. No gradient is used.

The two ambiguities of the model are resolved by the neutralisation conditions
alone. Without them, a constant added to $\mathbf{d}$ or a constant factor in
$\mathbf{f}$ would only translate or stretch the histogram; with them the mean
level and the mean contrast of the image are those of the input. The result is
therefore stored with `meta["flat_norm"] == "mean"`.

A global property is minimised, not a local one, so the method needs no
selection of control points, but it also cannot tell shading from a structure
of the image that looks like shading (the paper states the same limit of every
retrospective method).

## Algorithm

```text
Input: image Y (H × W), G = 256, t = 2, tol = 10⁻⁶, max_iterations = 200
Grey level of each pixel:  g = round((y − lo) / δ) + 1,  δ = (hi − lo) / (G − 1)
Discard the pixels at g = 1 and g = G
θ = 0 (no shading); Powell directions = the 10 unit vectors of the rescaled parameters
Repeat until the relative change of the entropy is below tol:
    for each direction: minimise H along it (Brent, tolerance 2·10⁻⁴)
    update the directions (Powell)
    # H(θ): corrected values u = (y − d) / f  (rejected if f ≤ 0 anywhere)
    #       histogram of u on [−G, 2G] with partial interpolation,
    #       blur by the triangular window, normalise, entropy in nats
Return f = 1 + m·Φ,  d = (hi − lo) · a·Φ  (neutralised quadratics, Eqs. 16–17)
```

## Faithful mode and extensions

`faithful=True` and `faithful=False` select the same preset, which is the
paper's method. The keywords are the same in both.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `levels` | number $G$ of grey levels | `256` | `256` |
| `window` | half-width $t$ of the histogram blur; `0` disables it | `2` | `2` |
| `tol`, `max_iterations` | stopping rule of Powell's method | `1e-6`, `200` | same |
| `components` | `"both"` fits $\mathbf{f}$ and $\mathbf{d}$; extension: `"additive"` or `"multiplicative"` fits one | `"both"` | `"both"` |
| `sampling` | extension: use every `sampling`-th pixel along each spatial axis in the fit | `1` | `1` |
| `data_range` | data values $[lo, hi]$ of the grey levels 1 and $G$ | observed minimum and maximum | same |

`data_range` describes the data, not the method, and does not affect
`is_faithful`. Fitting a single component and sparse sampling are two of the
modifications that the discussion of the paper suggests; they are not part
of the method as evaluated.

Choices that the paper does not specify:

- **Data range.** The paper uses 8-bit images, whose grey levels are the
  integer values. For other data, the grey level of a value $v$ is
  $\mathrm{round}((v - lo)/\delta) + 1$, with $[lo, hi]$ the observed minimum and
  maximum of each group by default. With `data_range=(0, 255)` and
  `levels=256`, 8-bit values map to their own grey levels exactly. The model
  and the correction are in data units.
- **Optimiser settings.** `tol`, `max_iterations`, the tolerance of Brent's
  method, the start at $\boldsymbol{\theta} = \mathbf{0}$, the initial
  directions of Powell's method (the unit vectors of parameters rescaled to
  $\tilde{x} = 2x/W$, $\tilde{y} = 2y/H \in [-1, 1]$, and additive ones in
  units of $hi - lo$, which changes the scale of the search and not the model)
  and the use of pixel-centre coordinates.
- **Out-of-range values.** Corrected values outside $[-G, 2G]$ are clamped to
  the end bins.
- **Invalid parameters.** Parameters that make $\hat{\mathbf{f}} \le 0$ at any
  pixel are rejected, their objective being larger than the maximum entropy.

### Open choices

Two choices are not settled and may change.

- **Several frames.** The paper corrects one image at a time. Given several
  frames in a group, `Likar` fits one shading per group by minimising the
  entropy of the pooled histogram of all the corrected frames. Fitting each
  frame separately is possible by calling `fit` on each frame. A per-frame
  mode inside one call is not provided: `ShadingEstimate` has no per-frame
  flat field.
- **Default intensity range for integer data.** The default is the observed
  minimum and maximum of each group. The alternative is the full range of the
  data type (0 to 255 for `uint8`), which keeps the grey levels comparable
  between groups and runs, and makes the correction independent of which
  values happen to occur. Pass `data_range` to select it explicitly.

## Memory and streaming

`Likar` is tagged `per-pixel-stats`. The statistic is the number of times each
pixel takes each grey level, a sparse count that is exactly mergeable across
chunks, with at most $\min(N, G)$ entries per pixel for $N$ frames. The fit
reads the data once when `data_range` is given, and twice otherwise (first for
the minimum and maximum). The entropy is then evaluated on these counts: the
cost of one evaluation depends on the number of distinct (pixel, grey level)
pairs and not directly on $N$, and it grows with the image size. Use `sampling`
to fit on a subset of the pixels; the fields are always returned at full
size. [`apply`](../../reference/shading.md) streams frame batches.

## Usage

```pycon
>>> import torch
>>> from optix.estimates import ShadingEstimate
>>> from optix.models import simulate_shading
>>> from optix.shading import Likar
>>> # 1. An 8-bit image with known quadratic shading.
>>> g = torch.Generator().manual_seed(0)
>>> yy, xx = torch.meshgrid(torch.arange(96), torch.arange(96), indexing="ij")
>>> scene = torch.full((96, 96), 180.0)
>>> for k in range(30):
...     cy, cx = torch.randint(0, 96, (2,), generator=g).tolist()
...     r = float(torch.randint(4, 10, (1,), generator=g))
...     scene[(yy - cy) ** 2 + (xx - cx) ** 2 < r**2] = (60.0, 100.0, 140.0)[k % 3]
>>> scene = scene + 5 * torch.randn(96, 96, generator=g)
>>> u, v = 2 * xx / 96 - 1, 2 * yy / 96 - 1
>>> truth = ShadingEstimate(
...     flat=1 + 0.2 * u + 0.15 * (v**2 - 1 / 3), dark=15 * u * v
... )
>>> data = simulate_shading(scene[None], truth).clamp(0, 255).round()
>>> # 2. Fit, with the 8-bit range, and correct.
>>> likar = Likar(data_range=(0, 255)).fit(data)
>>> shading = likar.estimates["shading"]
>>> corrected = likar.apply(data)
>>> # 3. Compare the estimate to the truth.
>>> float((shading.flat - truth.flat).abs().max()) < 0.05
True
>>> float((shading.dark - truth.dark).abs().mean()) < 1
True
>>> likar.is_faithful, likar.history.converged
(True, True)
```

Common variations:

- `Likar(components="multiplicative")` fits the flat field only, for images
  without an additive shading (an extension).
- `Likar(sampling=2)` fits on every second pixel in each direction, which is
  faster (an extension).
- For several channels, give a layout with a group axis: each channel gets its
  own shading, and `meta["coefficients"]` holds $(a_1 \ldots a_5)$ and
  $(m_1 \ldots m_5)$ of each, in pixel coordinates and data units.
- `meta["entropy"]` holds the final entropy of each group, in nats, and
  `meta["data_range"]` the range used.
- For a large `zarr` array, pass it as it is: the data are read in batches.

See the [API reference](../../reference/shading.md) for every option.

## References

!!! references "References"

    1. Likar B, Maintz JBA, Viergever MA, Pernuš F. *Retrospective shading
       correction based on entropy minimization.* J Microsc
       **197**(3):285–295, 2000.
       [doi:10.1046/j.1365-2818.2000.00669.x](https://doi.org/10.1046/j.1365-2818.2000.00669.x)
    2. Press WH, Teukolsky SA, Vetterling WT, Flannery BP. *Numerical Recipes
       in C.* 2nd edn. Cambridge University Press, 1992.
       [Cambridge University Press](https://numerical.recipes/)

    ??? quote "BibTeX"

        ```bibtex
        @article{likar2000retrospective,
          author  = {Likar, B. and Maintz, J. B. A. and Viergever, M. A. and Pernu{\v{s}}, F.},
          title   = {Retrospective shading correction based on entropy minimization},
          journal = {Journal of Microscopy},
          volume  = {197},
          number  = {3},
          pages   = {285--295},
          year    = {2000},
          doi     = {10.1046/j.1365-2818.2000.00669.x},
        }
        @book{press1992numerical,
          author    = {Press, William H. and Teukolsky, Saul A. and Vetterling, William T. and Flannery, Brian P.},
          title     = {Numerical Recipes in {C}},
          edition   = {2nd},
          publisher = {Cambridge University Press},
          year      = {1992},
          url       = {https://numerical.recipes/},
        }
        ```
