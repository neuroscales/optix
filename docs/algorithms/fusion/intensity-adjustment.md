# Intensity adjustment

!!! info "At a glance"
    **Estimates:** `TileIntensityEstimate` (`scale`, `shift`) \
    **Noise model:** none (least squares on the overlaps) \
    **Streaming:** pointwise (`apply`); the fit reads each overlap once \
    **Needs many frames:** no, but the tiles must overlap \
    **Reference:** Hörl et al., Nature Methods, 2019, Supplementary Note 17, [doi:10.1038/s41592-019-0501-0](https://doi.org/10.1038/s41592-019-0501-0)

## Overview

`IntensityAdjustment` makes the brightness and contrast of the tiles of a mosaic
agree. Bleaching, drift of the laser power or of the exposure make neighbouring
tiles differ by a gain and an offset, and the seams show in the fused image.
Given the tiles and the position of each on a common pixel grid, it estimates
one linear transform per tile,
$\hat{\mathbf{x}}_i = a_i\,\mathbf{y}_i + c_i$, such that the transformed
tiles agree where they overlap. It follows the brightness and contrast
adjustment of BigStitcher,
[Hörl et al. 2019 (Supplementary Note 17)](https://doi.org/10.1038/s41592-019-0501-0).

Use it after the shading of every tile has been corrected (with
[BaSiC](../shading/basic.md) or [CIDRE](../shading/cidre.md), for instance) and
before the tiles are fused: it does not model the smooth variation of
illumination across a tile, only a global gain and offset per tile. It does not
register the tiles: the positions are inputs.

It returns a [`TileIntensityEstimate`](../../reference/estimates.md) in
`estimates["intensity"]`, with one `scale` $a_i$ and one `shift` $c_i$ for each
tile (and each group), and `apply` harmonises the tiles with it.

## Image formation (optics)

The shading model of
[Image formation and noise](../../concepts/image-formation.md) is assumed to be
inverted already, so that a tile differs from its neighbours by two numbers only:
the intensity of the same specimen point, recorded in tile $i$, is

$$
\mathbf{y}_i = g_i\,\mathbf{x}|_i + o_i + \boldsymbol{\varepsilon}_i ,
$$

where $\mathbf{x}|_i$ is the part of the (unknown) uniform image $\mathbf{x}$
that the tile covers, $g_i$ and $o_i$ its gain and offset, and
$\boldsymbol{\varepsilon}_i$ noise. The harmonised tile
$\hat{\mathbf{x}}_i = a_i \mathbf{y}_i + c_i$ undoes this when
$a_i = 1/g_i$ and $c_i = -o_i/g_i$; this is the model of
`optix.models.simulate_tiles`. Only the *relative* gains and offsets can be
recovered from the overlaps, since $\mathbf{x}$ is known up to a common affine
map: the estimate is anchored by the identity term of the model below, or by
fixing tiles.

The tiles are placed on the common grid by integer translations: the position
of the first pixel of tile $i$ is an input. Two pixels *correspond* when they
land on the same grid pixel. The overlap of tiles $i$ and $j$ is written
$O_{ij}$, and $y_i(\mathbf{r})$ is the value of tile $i$ at the grid pixel
$\mathbf{r} \in O_{ij}$.

## Probabilistic model

The method is a least-squares fit and has no noise model. The objective is the
squared difference of the intensities of all corresponding pixels:

=== "optix"

    $$
    \min_{\{a_i,\,c_i\}}\;
    \sum_{i}\sum_{j \ne i}\sum_{\mathbf{r} \in O_{ij}}
    \bigl[(a_j\,y_j(\mathbf{r}) + c_j) - (a_i\,y_i(\mathbf{r}) + c_i)\bigr]^2 ,
    $$

    with the identity transform for the fixed tiles, $a_i = 1$ and
    $c_i = 0$ for $i \in F$.

=== "Hörl et al. 2019"

    $$
    \arg\min_{\alpha,\,\beta}\;
    \sum_{I_A \in \mathcal{I}}\;\sum_{I_B \in \mathcal{I}\setminus I_A}\;
    \sum_{(x_A,\,x_B) \in O_{AB}}
    \Bigl[I_B(x_B) - \bigl(\alpha_{I_A} I_A(x_A) + \beta_{I_A}\bigr)\Bigr]^2
    \qquad\text{(Eq. 8)}
    $$

The objective has the trivial solution $a_i = 0$, $c_i = 0$ for all tiles. The
paper removes it by writing each transform as a weighted average of three
transforms: a linear one, an additive one and the identity.

=== "optix"

    $$
    a_i\,y + c_i
    = \lambda_1\,(a'_i\,y + c'_i) + \lambda_2\,(y + c''_i) + \lambda_3\,y ,
    \qquad \lambda_1 + \lambda_2 + \lambda_3 = 1 ,\;\; \lambda_k \ge 0 .
    $$

=== "Hörl et al. 2019"

    $$
    \alpha I(x) + \beta
    = \lambda_1\,\bigl(\alpha' I(x) + \beta_1\bigr)
    + \lambda_2\,\bigl(I(x) + \beta_2\bigr) + \lambda_3\,I(x)
    \qquad\text{(Eq. 9)}
    $$

The weights are user parameters. With $\lambda_2, \lambda_3 > 0$ the solution
is pulled towards the identity and the trivial solution is excluded.

| optix | Hörl et al. 2019 | Meaning |
|---|---|---|
| $\mathbf{y}_i$, $y_i(\mathbf{r})$ | $I_A$, $I_B$; $I_A(x_A)$ | tile, and its value at a grid pixel |
| $a_i$, $c_i$ | $\alpha_{I_A}$, $\beta_{I_A}$ | scale and shift of tile $i$ |
| $a'_i$, $c'_i$ | $\alpha'$, $\beta_1$ | linear model of tile $i$ |
| $c''_i$ | $\beta_2$ | additive model of tile $i$ |
| $\lambda_1$, $\lambda_2$, $\lambda_3$ | $\lambda_1$, $\lambda_2$, $\lambda_3$ | weights of the linear, additive and identity models |
| $O_{ij}$ | $O_{AB}$ | overlap of tiles $i$ and $j$ |
| $F$ | $F$ (Note 12, Eq. 3) | fixed tiles |
| $\mathbf{r}$ | $x_A$, $x_B$ | grid pixel; corresponding pixels of both tiles |

The tile index $i$ and the weights $\lambda_k$ are the ones of the
[notation](../../concepts/notation.md#tiles); the $\lambda_k$ carry a subscript and
are not the photoelectron rate of the camera model.

## Inference / derivation

The paper notes that, for one tile, the fit of Eq. 8 is a one-dimensional point
correspondence registration, and minimises it with the iterative optimiser of
its global registration (Supplementary Note 12): the transform of each tile is
fitted by least squares to its point correspondences, and the fits are repeated
until they agree. Here the correspondences of tile $i$ are the pairs

$$
\bigl(p,\, q\bigr) = \bigl(y_i(\mathbf{r}),\; a_j\,y_j(\mathbf{r}) + c_j\bigr),
\qquad \mathbf{r} \in O_{ij},\; j \in \mathcal{N}(i),
$$

the value of tile $i$ and the *current transformed* value of each neighbour
$j$ at the same pixel. Let $N_i$ be the number of pairs, $\bar p_i$, $\bar q_i$
their means, and $S_{pp}$, $S_{pq}$ the sums of squares and cross-products about
the means. Each model is a least-squares fit of $q$ on $p$:

$$
a'_i = \frac{S_{pq}}{S_{pp}},\quad c'_i = \bar q_i - a'_i\,\bar p_i,
\qquad
c''_i = \bar q_i - \bar p_i ,
$$

and the transform of the tile is their average,

$$
a_i = \lambda_1\,\frac{S_{pq}}{S_{pp}} + \lambda_2 + \lambda_3,
\qquad
c_i = \lambda_1\,(\bar q_i - a'_i\,\bar p_i) + \lambda_2\,(\bar q_i - \bar p_i).
$$

The transforms are the fixed point of these updates. The sums $\bar q_i$ and
$S_{pq}$ are linear in the $(a_j, c_j)$ of the neighbours, and $S_{pp}$ does not
depend on them, so the fixed point is the solution of one linear system of
size $2T$ for $T$ tiles, which is solved directly instead of iterated. Fixed
tiles enter as the equations $a_i = 1$, $c_i = 0$, and tiles without any
overlap keep the identity.

The system only needs, for each overlap, the number of pixel pairs, the two
means, and the scatter and co-scatter of the pairs. They are accumulated in
one streaming pass over the overlaps, with the pairwise merge of
[Chan et al. 1983](https://doi.org/10.1080/00031305.1983.10483115), so that the
fit does not suffer from the cancellation of raw sums. The system is solved in
float64 after centring the intensities.

**Uniqueness.** With $\lambda_3 > 0$ the system is non-singular. With
$\lambda_3 = 0$ the linear and the additive models are equivariant to a
common offset of all the tiles, so every connected set of overlapping tiles
needs a fixed tile (with $\lambda_2 = 0$ too, a common scale is also free);
otherwise `fit` raises a `ValueError`.

**Constant tiles.** When the values of a tile are constant over all its
overlaps, $S_{pp} = 0$ and the linear model is undefined: the additive model
replaces it for that tile, and a `RuntimeWarning` names the tiles.

**Reported error.** `meta["mse_before"]` and `meta["mse_after"]` hold, for each
group, the objective above divided by the number of pixel pairs, for the
identity and for the fitted transforms. They are computed from the overlap
statistics, without a second pass.

## Algorithm

```text
Input: tiles y_1..y_T, positions, fixed tiles F, weights λ, downsample, max_pixels
For each group (channel, ...):
    For each pair (i, j) of overlapping tiles, with overlap O_ij:
        Read O_ij of both tiles in blocks; bin by `downsample` pixels per axis
        Accumulate (n, mean_i, mean_j, M_ii, M_jj, M_ij) over the blocks
            (or, with max_pixels, over a uniform random sample of the pixel pairs)
    Assemble the 2T x 2T linear system of the fixed point from these statistics
    Solve it; set a_i = 1, c_i = 0 for the fixed tiles
    Record the mean squared difference before and after
Return TileIntensityEstimate(scale=a, shift=c)
```

`apply` then computes $\hat{\mathbf{x}}_i = a_i \mathbf{y}_i + c_i$ for every
tile.

## Faithful mode and extensions

`faithful=True` (the default) is the `paper` preset, and the `recommended`
preset is identical. The method has free parameters but no extension: a keyword
set to anything but its default makes `is_faithful` false. The paper leaves the
three values below open; the defaults are the choices of optix.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `weights` | $(\lambda_1, \lambda_2, \lambda_3)$ of Eq. 9 | $(0.8, 0.1, 0.1)$, not specified in the paper | same |
| `downsample` | bin size per spatial axis before the fit; the paper fits on precomputed downsampled images | 1 (no binning), the factor is not specified | same |
| `max_pixels` | cap on the pixel pairs per overlap, by random removal | `None` (all pixels), the cap is not specified | same |
| `seed` | seed of the random selection (no effect on faithfulness) | `None` | same |

Choices that the paper does not specify and that both presets make:

1. **Eq. 8 transforms both tiles.** Eq. 8 as printed transforms tile $A$ only
   and compares it with the raw tile $B$. Supplementary Note 17 equates the fit
   with point correspondence registration, whose objective (Note 12, Eq. 3)
   transforms both sets of points, so the objective here transforms both tiles
   of an overlap. Both readings agree when
   the neighbour is fixed.
2. **Direct solve.** The paper uses "the same iterative optimiser" as the
   registration. Since every sub-model is a linear least-squares fit, its fixed
   point is computed by one direct solve instead of by iterating.
3. **Integer translations.** The tiles are placed by integer translations;
   non-integer positions are rounded to the nearest pixel. Rotated or affinely
   registered tiles are not supported.
4. **Binning.** With `downsample`, both tiles of an overlap are averaged over
   the same bins, anchored at the corner of the overlap, so that binned values
   still correspond exactly; trailing partial bins average the pixels they
   hold.
5. **Cap on the pixels.** `max_pixels` keeps a uniform random sample without
   replacement of the pixel pairs of each overlap, so that large overlaps do not
   outweigh small ones, as the paper intends.
6. **No identity term.** The paper does not say what to do with
   $\lambda_3 = 0$. Fixed tiles are then required (see above).
7. **Constant tiles** use the additive model, with a warning.

## Memory and streaming

The fit reads only the overlaps, each once and in blocks sized to the memory
budget of [`optix.config`](../../reference/configuration.md), so lazy zarr arrays
work and no tile is loaded whole. What it keeps in memory is a handful of
numbers per group and per overlapping pair, plus, with `max_pixels`, at most
`max_pixels` pixel pairs for each of them. The $2T \times 2T$ system is dense:
its cost, $O(T^3)$ per group, is negligible for hundreds of tiles.

Group axes (channels, for instance) are fitted independently and the sample
axes (time points) share the transforms of the tile and pool their pixels.
`apply` is pointwise: tiles are processed in frame batches and written to `out`,
in place if needed.

Because the least squares regress the neighbour on the tile, noise on the tile
attenuates the slope $a'_i$ slightly (errors in variables). Binning reduces the
noise of the pixel pairs and therefore the bias.

## Usage

A synthetic scene is cut into $2 \times 3$ overlapping tiles with random gains
and offsets; `positions` are the positions of the first pixel of each tile.

```pycon
>>> import torch
>>> from optix.fusion import IntensityAdjustment
>>> from optix.models import simulate_tiles
>>> gen = torch.Generator().manual_seed(0)
>>> r, c = torch.meshgrid(torch.arange(96.0), torch.arange(128.0), indexing="ij")
>>> x = 100 + 40 * torch.sin(r / 7) * torch.cos(c / 9)
>>> x = x + 20 * torch.rand(96, 128, generator=gen)
>>> tiles, positions, truth = simulate_tiles(x, grid=(2, 3), overlap=0.25, seed=1)
>>> len(tiles), tiles[0].shape
(6, torch.Size([55, 52]))
```

With a fixed tile and $\lambda = (1, 0, 0)$ the transforms are recovered up to
the arbitrary choice of the fixed tile, whose gain and offset they inherit:

```pycon
>>> adj = IntensityAdjustment(weights=(1, 0, 0)).fit(tiles, positions, fixed=[0])
>>> estimate = adj.estimates["intensity"]
>>> expected_scale = truth.scale / truth.scale[0]
>>> expected_shift = (truth.shift - truth.shift[0]) / truth.scale[0]
>>> bool((estimate.scale - expected_scale).abs().max() < 1e-5)
True
>>> bool((estimate.shift - expected_shift).abs().max() < 1e-3)
True
>>> estimate.meta["mse_after"][0] < 1e-6 * estimate.meta["mse_before"][0]
True
>>> adj.is_faithful  # weights differ from the defaults
False
```

The default weights keep the tiles close to their original intensities, so the
agreement is good but not exact:

```pycon
>>> adj = IntensityAdjustment().fit(tiles, positions)
>>> adj.is_faithful
True
>>> harmonised = adj.apply(tiles)
>>> estimate = adj.estimates["intensity"]
>>> estimate.meta["mse_after"][0] < 0.05 * estimate.meta["mse_before"][0]
True
>>> bool(torch.equal(estimate.correct(tiles[1], 1), harmonised[1]))
True
```

Common variations:

```python
from optix.io import Layout

# Channels as a group axis: tiles of shape (c, y, x), one fit per channel.
layout = Layout("cyx", spatial="yx", group="c")
adj = IntensityAdjustment(layout=layout).fit(
    channel_tiles, positions
)  # scale: (c, T)

# No identity term: every connected set of tiles needs a fixed tile.
adj = IntensityAdjustment(weights=(0.5, 0.5, 0)).fit(
    tiles, positions, fixed=[0]
)

# Fit on binned data, with at most 10 000 pixel pairs per overlap.
adj = IntensityAdjustment(downsample=4, max_pixels=10_000, seed=0)

# Lazy tiles (zarr arrays) in and out.
adj = IntensityAdjustment().fit(zarr_tiles, positions)
adj.apply(zarr_tiles, out=zarr_outputs)
```

See the [API reference](../../reference/fusion.md) for every option, and for the
building blocks in `optix.fusion.utils` (overlap boxes, the statistics, the
solver).

Validation: on simulated tiles, with a fixed tile and $\lambda = (1, 0, 0)$, the
gains and offsets are recovered to about $10^{-9}$ in float64, in 2-D and 3-D,
with tiles of unequal sizes and with downsampling. The direct solve matches an
independent Gauss–Seidel iteration, written on raw pixel pairs, to $10^{-8}$ for
four settings of the weights, and float32 results agree with float64. There are
no fixtures from a reference implementation: BigStitcher is distributed under
the GPL, which is not compatible with the licence of optix.

## References

!!! references "References"

    1. Hörl D, Rojas Rusak F, Preusser F, Tillberg P, Randel N, Chhetri RK,
       Cardona A, Keller PJ, Harz H, Leonhardt H, Treier M, Preibisch S.
       *BigStitcher: reconstructing high-resolution image datasets of cleared
       and expanded samples.* Nat Methods **16**(9):870-874, 2019.
       [doi:10.1038/s41592-019-0501-0](https://doi.org/10.1038/s41592-019-0501-0)
    2. Chan TF, Golub GH, LeVeque RJ. *Algorithms for computing the sample
       variance: analysis and recommendations.* Am Stat **37**(3):242-247, 1983.
       [doi:10.1080/00031305.1983.10483115](https://doi.org/10.1080/00031305.1983.10483115)

    ??? quote "BibTeX"

        ```bibtex
        @article{horl2019bigstitcher,
          author  = {H{\"o}rl, David and Rojas Rusak, Fabio and Preusser, Friedrich and Tillberg, Paul and Randel, Nadine and Chhetri, Raghav K. and Cardona, Albert and Keller, Philipp J. and Harz, Hartmann and Leonhardt, Heinrich and Treier, Mathias and Preibisch, Stephan},
          title   = {{BigStitcher}: reconstructing high-resolution image datasets of cleared and expanded samples},
          journal = {Nature Methods},
          volume  = {16},
          number  = {9},
          pages   = {870--874},
          year    = {2019},
          doi     = {10.1038/s41592-019-0501-0},
        }
        @article{chan1983algorithms,
          author  = {Chan, Tony F. and Golub, Gene H. and LeVeque, Randall J.},
          title   = {Algorithms for computing the sample variance: analysis and recommendations},
          journal = {The American Statistician},
          volume  = {37},
          number  = {3},
          pages   = {242--247},
          year    = {1983},
          doi     = {10.1080/00031305.1983.10483115},
        }
        ```
