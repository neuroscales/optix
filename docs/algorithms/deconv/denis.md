# Denis

!!! info "At a glance"
    - **Estimates:** the shift-variant blur operator of a field, `DenisPSF`, used as the `psf` of the Richardson–Lucy family (deconvolved image)
    - **Noise model:** none (a blur model: the noise is that of the deconvolution it is given to)
    - **Streaming:** halo
    - **Needs many frames:** no
    - **Reference:** Denis, Thiébaut, Soulez, Becker and Mourya, Int J Comput Vis, 2015, [doi:10.1007/s11263-015-0817-x](https://doi.org/10.1007/s11263-015-0817-x)

## Overview

`DenisPSF` turns PSFs known on a coarse grid of positions into a fast
approximation of the blur of the whole field. The blur is a sum of a few
convolutions, each of a *weighted* copy of the object: the PSF seen from any
pixel is the weighted combination of a few kernels
([Denis et al. 2015, §2](https://doi.org/10.1007/s11263-015-0817-x)). Three
approximations are available, `model=`:

- `"interpolation"`: the kernels are the sampled PSFs and the weights those of a
  bilinear (trilinear in 3-D) interpolation, so the PSF at every pixel is the
  interpolation of the sampled PSFs;
- `"modes"`: the kernels are the principal modes of the sampled PSFs, of which
  the first `rank` are kept, and the weights are their coefficients interpolated
  between nodes (the practical scheme of Flicker and Rigaut 2005, which departs
  from the paper's equations, see below);
- `"optimal"`: kernels and weights are fitted to a PSF known at every pixel (a
  function `psf_model`), keeping the localised weights of the interpolation.

The paper recommends the PSF interpolation over the convolve-then-weight model of
[Nagy and O'Leary](nagy-oleary.md), which is not the blur of an interpolated PSF,
and the optimal approximation for the best accuracy at the cost of the
interpolation. Use `"interpolation"` when the PSFs are measured on a grid (beads,
for instance), `"modes"` when they vary in few independent ways, and `"optimal"` when
the PSF is known as a function of position and accuracy matters.

The model is the `psf` of [`RichardsonLucy`](richardson-lucy.md),
[`BiggsAndrews`](biggs-andrews.md) and [`Snyder`](snyder.md), which keep their
Poisson likelihood, stopping rule, non-negativity and block-wise processing. The
paper's own restoration (a total-variation penalty minimised by L-BFGS) is not
implemented.

## Image formation (optics)

The blur is linear but not shift invariant, as in
[Image formation and noise](../../concepts/image-formation.md): a point source
at $\mathbf{r}$ gives an image $\mathbf{h}_{\mathbf{r}}$ that depends on
$\mathbf{r}$ (aberrations, depth, a light sheet that thickens away from its
waist). The paper separates the change of the PSF from its translation with the
*un-shifted* PSF, the PSF re-centred on its source. Collecting them, one column
per source pixel, gives the matrix $\mathbf{K}$ of un-shifted PSFs. For a
shift-invariant blur, all columns are equal. The PSF has 2 or 3 dimensions, is
centred at index `K // 2` and, by default, normalised to unit sum. As for
[Richardson–Lucy](richardson-lucy.md), the object is estimated on a domain
extended by the PSF support, and the blur crops back to the observed region.

## Probabilistic model

Not probabilistic: the model is a blur operator, and a noise model comes with the
deconvolution method ([Poisson](richardson-lucy.md#probabilistic-model) for the RL family).
A separable approximation of the un-shifted PSFs, with $S$ terms indexed by $s$,
gives the operator as a sum of convolutions of weighted objects.

=== "optix"

    $$
    \mathbf{K} \approx \sum_{s=1}^{S} \mathbf{h}_s \mathbf{w}_s^\top
    \qquad\Rightarrow\qquad
    \mathbf{A} \approx \sum_{s=1}^{S} \mathbf{H}_s \mathbf{W}_s ,
    \qquad \mathbf{W}_s = \operatorname{diag}(\mathbf{w}_s),
    $$

    with $\mathbf{K} \in \mathbb{R}^{K \times Q}$ the un-shifted PSFs (one
    column of $K$ PSF voxels per pixel of the extended object domain of $Q$
    pixels), $\mathbf{h}_s \in \mathbb{R}^{K}$ the kernel of term $s$,
    $\mathbf{H}_s$ the convolution with it, $\mathbf{w}_s \in \mathbb{R}^{Q}$
    the weights of the term, an image over the object pixels, and
    $\mathbf{A} \in \mathbb{R}^{\lvert\Omega\rvert \times Q}$ the blur onto the
    $\lvert\Omega\rvert$ observed pixels.

=== "Denis et al. 2015"

    $$
    \mathbf{K} \approx \sum_{p} \mathbf{m}_p \cdot \mathbf{w}_p^\top \quad (13),
    \qquad
    \mathbf{H} \approx \sum_{p} \operatorname{conv}(\mathbf{m}_p) \cdot \operatorname{diag}(\mathbf{w}_p) \quad (14).
    $$

| optix | [Denis et al. 2015](https://doi.org/10.1007/s11263-015-0817-x) | Meaning | Shape |
|---|---|---|---|
| $\mathbf{A}$ | $\mathbf{H}$ | shift-variant blur operator | $\lvert\Omega\rvert \times Q$, one for the whole stack |
| $\mathbf{K}$ | $\mathbf{K}$ | un-shifted PSFs, one column per object pixel | $K \times Q$, one for the whole stack |
| $\mathbf{H}_s$, $\mathbf{h}_s$ | $\operatorname{conv}(\mathbf{m}_p)$, $\mathbf{m}_p$ | convolution with kernel $s$; the kernel (a component of the PSF model) | $\lvert\Omega\rvert \times Q$ and $K$, one per term |
| $\mathbf{W}_s$, $\mathbf{w}_s$ | $\operatorname{diag}(\mathbf{w}_p)$, $\mathbf{w}_p$ | weights of term $s$ over the object, as a diagonal matrix and as an image | $Q \times Q$ and $Q$, one per term |
| $S$, $s$ | $P$, $p$ | number of terms; term index | scalars |
| $\boldsymbol{\varphi}_s$ | $\boldsymbol{\varphi}_p$ | (multi)linear interpolation weights of node $s$ | $Q$, one per node |
| $K$ | $L$ | voxels of the PSF support | scalar |
| $Q$ | $M$ | pixels of the (extended) object domain | scalar |

The order matters: the weights multiply the object, then the kernel is convolved
(the adjoint is the convolution followed by the weights). The inverse order, in
[Nagy and O'Leary (1998)](nagy-oleary.md) and later works, is not the blur of an
interpolated PSF ([§2.2–2.3, Table 2](https://doi.org/10.1007/s11263-015-0817-x)).

## Inference / derivation

The three models differ in how the kernels and weights are obtained. The error of
an approximation is the squared Frobenius norm on the un-shifted PSFs, $E^2 =
\lVert\mathbf{K} - \sum_s \mathbf{h}_s \mathbf{w}_s^\top\rVert_F^2$ (Eq. 25), the sum of
squared distances between the true PSF and its approximation over all pixels.
Statistically, every model is a least-squares fit of a low-rank, separable
model to the PSFs: `"modes"` is a truncated SVD (principal component analysis
without centring) of the sampled PSFs, and `"optimal"` is alternating least
squares with a support constraint on the weights; no noise model is involved.

**Interpolation** ([§2.3](https://doi.org/10.1007/s11263-015-0817-x), Eqs. 21–22).
The PSFs are sampled at $S$ nodes of a rectangular grid. Each column of $\mathbf{K}$
is approximated by the weighted sum of the sampled columns $\mathbf{h}_s$,

$$
\mathbf{K} \approx \sum_{s} \mathbf{h}_s \boldsymbol{\varphi}_s^\top,
\qquad
\mathbf{A} \approx \sum_s \mathbf{H}_s \operatorname{diag}(\boldsymbol{\varphi}_s),
$$

where $\varphi_s(\mathbf{r})$ is the (multi)linear interpolation weight of node
$s$: non-zero only in the cells touching the node, a square twice the grid step
along each axis. The interpolated PSF is positive and keeps the normalisation and
the symmetries of the sampled PSFs ([§3.2](https://doi.org/10.1007/s11263-015-0817-x)).
Because the weights are zero outside their support, a block is convolved only with
the kernels of the nodes whose weights are non-zero in it.

**Modes** ([§2.4](https://doi.org/10.1007/s11263-015-0817-x), Eqs. 26–28; this
departs from the paper). In the paper, $\mathbf{K}$ holds the PSF of *every* pixel
($K \times Q$), its singular value decomposition
$\mathbf{K} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ gives the closest rank-$R$
approximation $\sum_{s \le R} \mathbf{u}_s\, \sigma_s \mathbf{v}_s^\top$ (Eq. 27, Eckart–Young),
and $\sigma_s \mathbf{v}_s$ are the weights over the whole field (Eq. 28), with no
interpolation. The model implements the practical scheme of Flicker and Rigaut
(2005): the SVD is that of the matrix $\mathbf{K}_0 \in \mathbb{R}^{K \times S_0}$ of
the $S_0$ *sampled* PSFs, the kernels are the first $R$ modes
$\mathbf{h}_s = \mathbf{u}_s$ (not positive nor normalised), and the weights are the
mode coefficients $\sigma_s \mathbf{v}_s$ of the nodes, interpolated between nodes
with the weights $\boldsymbol{\varphi}$. Eckart–Young then holds for the samples
only: the truncation is the best rank-$R$ approximation of $\mathbf{K}_0$, with
error $\sum_{s > R} \sigma_s^2$ on them, and not of the PSFs over the field. The
weights are not localised: each mode is convolved over the whole field. With
$R = S_0$ the model equals the interpolation.

**Optimal local approximation** ([§2.5](https://doi.org/10.1007/s11263-015-0817-x),
Eqs. 29–38). Kernels $\mathbf{C} = [\mathbf{c}_1 \cdots \mathbf{c}_S] \in \mathbb{R}^{K \times S}$
and weights $\mathbf{W} = [\mathbf{w}_1 \cdots \mathbf{w}_S] \in \mathbb{R}^{Q \times S}$ solve

$$
\min_{\mathbf{C}, \mathbf{W}} \lVert\mathbf{K} - \mathbf{C}\mathbf{W}^\top\rVert_F^2
\quad\text{s.t.}\quad \operatorname{supp}(\mathbf{w}_s) \subset \operatorname{supp}(\boldsymbol{\varphi}_s)
\qquad (30, 31),
$$

where $\mathbf{K}$ holds the PSF of every pixel. This is biconvex, solved by
alternating two least-squares steps, starting from the interpolation model. For
fixed weights, the kernels are

$$
\mathbf{C} = \mathbf{K}\mathbf{W}(\mathbf{W}^\top\mathbf{W})^{-1}
\qquad (35).
$$

For fixed kernels, the weights of pixel $j$ are the projection of its PSF
$\mathbf{K}_{\cdot j}$ on the $2^d$ kernels $\mathbf{C}'_j$ of its grid cell, in $d$
dimensions,

$$
\boldsymbol{\omega}_j = (\mathbf{C}'^\top_j \mathbf{C}'_j)^{-1} \mathbf{C}'^\top_j \mathbf{K}_{\cdot j}
\qquad (37),
$$

$\mathbf{C}'_j$ is the same for every pixel of a cell, so the matrix
$(\mathbf{C}'^\top_j \mathbf{C}'_j)^{-1}\mathbf{C}'^\top_j$ is computed once per cell; the weights
$\boldsymbol{\omega}_j$ themselves depend on the PSF of the pixel. Each alternation lowers the error, and ten are enough
(`n_alt=10`). The operator is then Eq. 38,
$\mathbf{A} \approx \sum_s \operatorname{conv}(\mathbf{c}_s)\operatorname{diag}(\mathbf{w}_s)$,
at the cost of the interpolation model, with errors about an order of magnitude
smaller.

## Algorithm

**Input:** PSFs sampled at the $S_0$ nodes of a grid, array `(*G, *K)`, and the
node positions; `model="interpolation"`, `rank=None`, `n_alt=10`; for
`"optimal"`, `psf_model` and `extent`. \
**Output:** the operator $\mathbf{A} \approx \sum_s \mathbf{H}_s \mathbf{W}_s$ of
any block, used as the `psf` of the Richardson–Lucy family.

1. **Build the terms** (kernels $\mathbf{h}_s$, weights $\mathbf{w}_s$):
    1. `"interpolation"`: $\mathbf{h}_s$ is the PSF at node $s$, and
       $\mathbf{w}_s = \boldsymbol{\varphi}_s$, the (multi)linear weights;
    2. `"modes"`: compute the SVD
       $\mathbf{K}_0 = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ of the sampled
       PSFs; $\mathbf{h}_s = \mathbf{u}_s$ for the first `rank` modes, and
       $\mathbf{w}_s$ interpolates the coefficients $\sigma_s \mathbf{v}_s$
       between nodes;
    3. `"optimal"`: start from the interpolation terms, evaluate `psf_model` at
       every pixel of `extent`, and repeat `n_alt` times:
        1. **Kernels:** $\mathbf{C} = \mathbf{K}\mathbf{W}(\mathbf{W}^\top\mathbf{W})^{+}$ (Eq. 35);
        2. **Weights:** $\boldsymbol{\omega}_j = (\mathbf{C}'_j)^{+}\, \mathbf{K}_{\cdot j}$ for every pixel $j$ (Eq. 37);

        then one more weight pass stores the $2^d$ weights of every pixel.
2. **Apply** for a block at a global position: evaluate the weights at the
   global pixel positions and compute
   $\mathbf{A}\mathbf{x} = \sum_s \mathbf{H}_s (\mathbf{w}_s \odot \mathbf{x})$
   by FFT on a padded grid; the adjoint is
   $\mathbf{A}^\top \mathbf{v} = \sum_s \mathbf{w}_s \odot (\mathbf{H}_s^\top \mathbf{v})$.
3. **Deconvolve:** give the operator to `RichardsonLucy`, `BiggsAndrews` or
   `Snyder`.

Here $^{+}$ is the pseudo-inverse (see the choices below).

## Faithful mode and extensions

`DenisPSF` is a model rather than an estimator and has no `faithful` flag; every
model of the paper is a value of `model`, and none is an extension.

| Keyword | Changes | Paper | Default |
|---|---|---|---|
| `model` | `"interpolation"` (Eqs. 21–22), `"modes"` (Eqs. 26–28, with the departure described above), `"optimal"` (Eqs. 29–38) | all three | `"interpolation"` |
| `rank` | number of modes kept (`"modes"`) | 5 and 13 in the experiments | `None` (all modes) |
| `psf_model` | callable giving the PSFs `(n, *K)` at positions `(n, d)`, for `"optimal"` | the PSF of every pixel | `None` |
| `extent` | image shape the `"optimal"` fit covers | the whole image | `None` |
| `n_alt` | alternations of Eqs. 35 and 37 (`"optimal"`) | 10 | 10 |
| `normalize` | divide each PSF by its sum | unit-sum PSFs | `True` |

Choices that the paper does not specify:

- outside the grid of nodes, the weights are constant (clamped to the nearest
  node);
- with `"modes"`, the SVD is that of the sampled PSFs (one column per node), not
  of the PSFs of every pixel as in Eqs. 26–28: the coefficients are interpolated
  bilinearly between nodes (Flicker and Rigaut 2005), and by default all modes are
  kept;
- with `"optimal"`, positions are clamped to the image before `psf_model` is
  evaluated, and the least-squares problems are solved with the pseudo-inverse
  (minimum-norm solution), so that identical neighbouring PSFs are allowed, and
  all $2^d$ nodes of the clamped cell of a pixel carry a weight (the paper's
  "weights of a cell"), which loosens the support constraint of Eq. 31 where
  $\varphi_s$ vanishes on a node of the cell: beyond the outermost centres, and
  exactly on grid lines;
- `normalize=False` keeps a PSF sum that varies over the field (vignetting,
  [§3.2.2](https://doi.org/10.1007/s11263-015-0817-x)).

!!! note "Costs of `\"optimal\"`"
    As in [§2.5.2](https://doi.org/10.1007/s11263-015-0817-x), the kernels and the
    weights $\boldsymbol{\omega}$ of every pixel are computed once, when the model is
    built, and stored ($2^d$ values per pixel of `extent`); operators only slice
    them, and `psf_model` is not called again. The fit evaluates `psf_model`
    $(n_\text{alt} + 1)$ times at every pixel, in batches sized from the memory
    budget, at a cost of about $(n_\text{alt} + 1)\, Q K\, 2^d$ operations ($Q$
    pixels, $K$ voxels of the PSF support): fine in 2-D, heavy for a large volume
    with large 3-D PSFs. The model can be saved without `psf_model`.

The paper's restoration (a total-variation penalty minimised by L-BFGS) is not
implemented; the model is given to the RL family instead. `RichardsonLucy` stays
faithful with a shift-variant model: [Lucy (1974)](https://doi.org/10.1086/111605)
states the iteration for any kernel $P(x \mid \xi)$, not only a convolution.
`WienerButterworth` needs a single kernel for its back projector and rejects
shift-variant models with a `TypeError`.

## Memory and streaming

The model is evaluated block by block, with a halo, by the deconvolution method it is given to
(see [Richardson–Lucy](richardson-lucy.md#memory-and-streaming)). The weights
are computed from the global pixel positions of a block, so the operator of a
block is the whole-image operator restricted to it, whatever the blocking, and
`BiggsAndrews`, with its acceleration parameter shared by the blocks, matches the
whole-image result. Memory per block is that of the single-PSF case plus the weights and
transfer functions of the $S_b$ terms that touch the block: $S_b$ is at most $2^d$ for a
block inside a grid cell, with `"interpolation"` and `"optimal"`, and $R$ for
`"modes"` (the weights are not localised). The cost of an iteration is $S_b$ forward
and $S_b$ adjoint FFT pairs. The fit of `"optimal"` visits every pixel of
`extent` $n_\text{alt} + 1$ times, in batches sized from the memory budget.

## Usage

The example blurs a [phantom](../../reference/models.md#phantoms) of cells with the interpolated PSFs of a
$3 \times 3$ grid, whose width grows from the top to the bottom of the field, and
restores it with Richardson–Lucy.

```python
import torch
from optix.deconv import DenisPSF, RichardsonLucy
from optix.models import gaussian_psf, phantoms

torch.manual_seed(0)

# 1. A 3 × 3 grid of PSFs widening down the field, and cells blurred by it.
centres = [[8.0, 32.0, 56.0], [8.0, 32.0, 56.0]]
sigma = lambda row: 1.0 + row / 32.0
row = lambda r: torch.stack([gaussian_psf((11, 11), sigma=sigma(r))] * 3)
psf = torch.stack([row(8.0), row(32.0), row(56.0)])  # (3, 3, 11, 11)
x = phantoms.cells((64, 64), intensity=200.0, seed=0) + 5
model = DenisPSF(psf, centres)
op = model.operator((0, 0), x.shape)
y = torch.poisson(op.forward(op.pad(x)))

# 2. Fit with the shift-variant model, and with the central PSF only.
rl = RichardsonLucy(model, n_iter=20).fit(y)
rl_one = RichardsonLucy(psf[1, 1], n_iter=20).fit(y)

# 3. Apply.
shift_variant, single = rl.apply(y), rl_one.apply(y)

# 4. Compare to the truth: relative error 0.25 data, 0.21 single, 0.15 shift-variant.
err = lambda a: float((a - x).norm() / x.norm())
print(err(y), err(single), err(shift_variant))
```

The three columns of the grid are equal, so the nine sampled PSFs span three
dimensions and the first three modes represent them exactly; the singular values
beyond the third vanish:

```pycon
>>> modes = DenisPSF(psf, centres, model="modes", rank=3)
>>> modes.kernels.shape
torch.Size([3, 11, 11])
>>> bool(modes.singular_values[3:].max() < 1e-12)
True
```

When the PSF is known at any position, the optimal approximation fits kernels and
weights to it. `psf_model` maps positions `(n, d)` to PSFs `(n, *K)`:

```pycon
>>> def psf_model(pos):
...     return torch.stack(
...         [
...             gaussian_psf((11, 11), sigma=float(sigma(p)), dtype=torch.float64)
...             for p in pos[:, 0]
...         ]
...     )
>>> optimal = DenisPSF(
...     psf, centres, model="optimal", psf_model=psf_model, extent=x.shape
... )
>>> optimal.kernels.shape
torch.Size([9, 11, 11])
>>> better = RichardsonLucy(optimal, n_iter=20).fit(y).apply(y)
>>> err(better) < err(single)
True
```

Common variations: `BiggsAndrews(model)` and `Snyder(model)` for acceleration and
camera data; a 3-D grid of 3-D PSFs; a `zarr` array as input, with `out`; the
same model for a whole stack. The estimators `RichardsonLucy`, `BiggsAndrews` and
`Snyder` run their usual stopping rules with the model. See the
[API reference](../../reference/deconv.md).

## References

!!! references "References"

    1. Denis L, Thiébaut É, Soulez F, Becker J-M, Mourya R. *Fast
       approximations of shift-variant blur.* Int J Comput Vis
       **115**(3):253-278, 2015.
       [doi:10.1007/s11263-015-0817-x](https://doi.org/10.1007/s11263-015-0817-x)
    2. Nagy JG, O'Leary DP. *Restoring images degraded by spatially variant
       blur.* SIAM J Sci Comput **19**(4):1063-1082, 1998.
       [doi:10.1137/S106482759528507X](https://doi.org/10.1137/S106482759528507X)
    3. Lucy LB. *An iterative technique for the rectification of observed
       distributions.* Astron J **79**(6):745-754, 1974.
       [doi:10.1086/111605](https://doi.org/10.1086/111605)

    ??? quote "BibTeX"

        ```bibtex
        @article{denis2015fast,
          author  = {Denis, Lo{\"\i}c and Thi{\'e}baut, {\'E}ric and Soulez, Ferr{\'e}ol and Becker, Jean-Marie and Mourya, Rahul},
          title   = {Fast Approximations of Shift-Variant Blur},
          journal = {International Journal of Computer Vision},
          volume  = {115},
          number  = {3},
          pages   = {253--278},
          year    = {2015},
          doi     = {10.1007/s11263-015-0817-x},
        }
        @article{nagy1998restoring,
          author  = {Nagy, James G. and O'Leary, Dianne P.},
          title   = {Restoring Images Degraded by Spatially Variant Blur},
          journal = {SIAM Journal on Scientific Computing},
          volume  = {19},
          number  = {4},
          pages   = {1063--1082},
          year    = {1998},
          doi     = {10.1137/S106482759528507X},
        }
        @article{lucy1974iterative,
          author  = {Lucy, L. B.},
          title   = {An iterative technique for the rectification of observed distributions},
          journal = {The Astronomical Journal},
          volume  = {79},
          number  = {6},
          pages   = {745--754},
          year    = {1974},
          doi     = {10.1086/111605},
        }
        ```
