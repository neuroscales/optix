# Multiview Richardson–Lucy

!!! info "At a glance"
    **Estimates:** deconvolved image (the latent image $\mathbf{x}$, one image for all the views, of the shape of one view) \
    **Noise model:** none assumed by the derivation; Poisson counts for the χ² extension \
    **Streaming:** halo \
    **Needs many frames:** no (several registered views of one specimen) \
    **Reference:** Preibisch et al., Nat Methods, 2014, [doi:10.1038/nmeth.2929](https://doi.org/10.1038/nmeth.2929)

## Overview

`MultiviewRL` fuses several views of one specimen into a single deconvolved
image. Each view is the specimen blurred by its own point spread function (PSF),
as in light-sheet microscopy where the sample is rotated and imaged from several
angles: one view resolves what another blurs, and the deconvolution combines
them. The method is the *efficient Bayesian-based multiview deconvolution* of
[Preibisch et al. (2014)](https://doi.org/10.1038/nmeth.2929): the
[Richardson–Lucy](richardson-lucy.md) iteration extended to several views, in
which the back projection of each view also accounts for what the other views
would see, and in which the views update the image one after the other. It
converges in several times fewer iterations than combining the views by
averaging their Richardson–Lucy updates.

Use it for views that are already registered and resampled onto a common grid,
with one PSF per view in the orientation of that grid. Registration and PSF
estimation are not part of the method. With one view it is exactly
[Richardson–Lucy](richardson-lucy.md). `apply` returns the deconvolved image,
not one image per view.

## Image formation (optics)

Every view is a linear, shift-invariant blur of the same latent image, with its
own PSF $\mathbf{h}_v$, for $v = 1, \ldots, V$ (see
[Image formation and noise](../../concepts/image-formation.md) and
[Richardson–Lucy](richardson-lucy.md#image-formation-optics)). The views are
registered, so that the latent image and all views live on one grid and the
PSFs are given in the coordinates of that grid, centred at index `K // 2` and
normalised to unit sum. Their shapes may differ; they are zero-padded to a common
odd shape. Light from outside the field of view is blurred into it, so the
latent image is estimated on a domain extended by the PSF support. Shading and
camera offset are assumed removed.

=== "optix"

    $$
    \mathbb{E}[\mathbf{y}_v] = \mathbf{A}_v \mathbf{x},
    \qquad
    (\mathbf{A}_v \mathbf{x})_p = \sum_q h_v(p - q)\, x_q ,
    \qquad v = 1, \ldots, V .
    $$

=== "Preibisch et al. 2014"

    $$
    \phi_v(x_v) = \int_\xi \psi(\xi)\, P_v(x_v \mid \xi)\, d\xi
    \quad \text{(Suppl. Note 1.2, Eqs. 12–15)}.
    $$

| optix | [Preibisch et al. 2014](https://doi.org/10.1038/nmeth.2929) |
|---|---|
| $\mathbf{y}_v$ (view $v$) | $\phi_v(x_v)$ |
| $\mathbf{x}$ (latent image) | $\psi(\xi)$ |
| $\mathbf{h}_v$, $\mathbf{A}_v$ (PSF, blur) | $P_v(x_v \mid \xi)$, convolution $\psi * P_v$ |
| $\check{\mathbf{h}}_v$, $\mathbf{A}_v^\top$ (mirrored PSF, adjoint) | $P_v^*$, convolution $\cdot * P_v^*$ |
| $V$, $v$ | $N = \lvert V \rvert$, $v \in V$ |
| $t$ (iteration) | $r$ |

The mirrored kernel is $\check{h}(\mathbf{r}) = h(-\mathbf{r})$. It is the kernel of
the adjoint $\mathbf{A}_v^\top$, which matters because the PSFs of rotated views
are not symmetric.

## Probabilistic model

The paper derives the iteration from Bayes' theorem on distributions, not from a
likelihood: every image is a probability distribution (non-negative, of unit
total mass), and the latent image is the distribution that explains all the
views through their PSFs. The derivation makes two assumptions that the model
inherits: the PSF of a view is the same everywhere in the field, and the views
are conditionally independent given the latent image. It does *not* assume that
the views are independent of each other ([Suppl. Note 3](https://doi.org/10.1038/nmeth.2929)), and this is what allows
the efficient variant: given an observation in view $v$, the PSF of view $w$
assigns a probability to where the same fluorophore appears in view $w$, which
defines a *virtual view* of $w$ computed from $v$ alone.

No noise model is part of the derivation. With one view it is the
maximum-likelihood EM iteration for Poisson counts,
$\mathbf{y} \sim \mathrm{Poisson}(\mathbf{A}\mathbf{x})$. With several, the plain
Bayesian-based multiplicative update converges to the same maximum-likelihood
solution on noise-free data (Suppl. Note 2), and the sequential update, the
default, is ordered-subset EM, without such a guarantee on noisy data; only
`update="additive"` is the EM iteration of the stacked Poisson model. The
paper reports the method to be robust to Poisson noise, the dominant noise of
light-sheet microscopy. As for
[Richardson–Lucy](richardson-lucy.md), the number of iterations is the only
regulariser.

## Inference / derivation

The strategy is a multiplicative fixed-point iteration, built in four steps.

**1. Plain Bayesian-based iteration.** The correction of view $v$ is the
single-view Richardson–Lucy correction,
$\mathbf{c}_v = \mathbf{A}_v^\top[\mathbf{y}_v / \mathbf{A}_v\mathbf{x}^{(t)}] /
\mathbf{A}_v^\top\mathbf{1}$, and the views are combined by *multiplying* their
corrections. The paper proves convergence to the maximum-likelihood solution for
noise-free data for the geometric mean of the corrections (the
$\lvert V \rvert$-th root, Eqs. 46–47), whereas the
EM derivation of [Shepp and Vardi (1982)](https://doi.org/10.1109/TMI.1982.4307558)
averages them arithmetically.

=== "optix"

    $$
    \mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} \odot \Bigl(\prod_{v=1}^{V} \mathbf{c}_v\Bigr)^{1/V} ,
    \qquad
    \mathbf{c}_v = \frac{\mathbf{A}_v^\top\!\left[\mathbf{y}_v / (\mathbf{A}_v \mathbf{x}^{(t)})\right]}{\mathbf{A}_v^\top \mathbf{1}} .
    $$

=== "Preibisch et al. 2014"

    $$
    \psi^{r+1} = \psi^{r} \prod_{v \in V} \frac{\phi_v}{\psi^{r} * P_v} * P_v^*
    \quad \text{(Eqs. 36–37)}.
    $$

The exponent $1/V$ is not optional: with the product as printed in Eq. 36 the
iteration diverges from about four views on. The normaliser $\mathbf{A}_v^\top\mathbf{1}$ is 1 inside the field and falls below
1 near its edges, as for Richardson–Lucy; the paper's kernels have unit mass and
do not write it.

**2. Sequential updates.** The product suggests applying the corrections one
after the other, each evaluated at the current estimate. This is ordered-subset
EM ([Hudson and Larkin 1994](https://doi.org/10.1109/42.363108)) with one view
per subset, and converges about $V$ times faster than combining the views
(Suppl. Note 5, Eqs. 104–112). One iteration is one pass over all the views.

**3. Efficient Bayesian-based iteration.** Adding the updates of the virtual
views to those of the observed views (Suppl. Note 4, Eq. 86) and
approximating $(f * g) \cdot (f * h) \approx f * (g \cdot h)$ twice (Eq. 91)
folds all of them into the back projection of view $v$, at no cost per
iteration. The mirrored PSF is replaced by the *compound kernel* $\mathbf{h}_{b,v}$,
a pixel-wise product of $\check{\mathbf{h}}_v$ with one convolved kernel per other
view:

=== "optix"

    $$
    \mathbf{c}_v = \frac{\mathbf{B}_v\!\left[\mathbf{y}_v / (\mathbf{A}_v \mathbf{x})\right]}{\mathbf{B}_v \mathbf{1}} ,
    \qquad
    \mathbf{h}_{b,v} \propto \check{\mathbf{h}}_v \odot \prod_{w \ne v}
    \bigl(\check{\mathbf{h}}_v * \mathbf{h}_w * \check{\mathbf{h}}_w\bigr) ,
    $$

    with $\mathbf{B}_v$ the convolution with $\mathbf{h}_{b,v}$ inside the
    field of view, and $\mathbf{B}_v = \mathbf{A}_v^\top$ on the margin of the
    padded domain (see the note on edge damping below).

=== "Preibisch et al. 2014"

    $$
    \psi^{r+1} = \psi^{r} \prod_{v \in V} \frac{\phi_v}{\psi^{r} * P_v} *
    \Bigl( P_v^* \prod_{w \in W_v} P_v^* * P_w * P_w^* \Bigr)
    \quad \text{(Eqs. 102–103)},
    $$

    with $W_v = V \setminus \{v\}$.

| optix | [Preibisch et al. 2014](https://doi.org/10.1038/nmeth.2929) |
|---|---|
| $\mathbf{h}_{b,v}$, $\mathbf{B}_v$ (back-projection kernel, operator) | $P_v^{\mathrm{compound}}$, convolution $\cdot * P_v^{\mathrm{compound}}$ |
| $\odot$, $\prod$ of kernels | $\cdot$, $\prod$ (pixel-wise) |
| $\mathbf{c}_v$ (correction of view $v$) | the factor of view $v$ in Eq. 102 |
| $w \ne v$ | $w \in W_v$ |

The approximation holds for Gaussian-like PSFs, which measured light-sheet PSFs
resemble ([Suppl. Fig. 3](https://doi.org/10.1038/nmeth.2929)); the convergence
proof covers the plain Bayesian-based kernels only.

**4. Ad hoc optimisations.** Two further simplifications sharpen the virtual
views and speed up convergence more (Suppl. Note 6). *Optimisation I* skips the
convolution with $\check{\mathbf{h}}_w$ (Eq. 114); *optimisation II* assumes
$\mathbf{h}_w$ constant, which leaves the pixel-wise power of the mirrored PSF
(Eq. 115):

=== "optix"

    $$
    \mathbf{h}_{b,v} \propto \check{\mathbf{h}}_v \odot \prod_{w \ne v}
    \bigl(\check{\mathbf{h}}_v * \mathbf{h}_w\bigr)
    \quad\text{(I)},
    \qquad
    \mathbf{h}_{b,v} \propto \check{\mathbf{h}}_v^{\,\odot V}
    \quad\text{(II)}.
    $$

=== "Preibisch et al. 2014"

    $$
    P_v^* \prod_{w \in W_v} P_v^* * P_w
    \quad \text{(Eq. 114)},
    \qquad
    \prod_{v,\, w \in W_v} P_v^*
    \quad \text{(Eq. 115)}.
    $$

The paper reports, for a simulated ground truth and depending on the number of
views (up to twelve in Suppl. Fig. 4), that
the efficient kernels converge 1.5–2.5 times faster than the plain ones,
optimisation I 2–4 times and optimisation II 3–8 times, and that sequential
updates add a further factor of about $V$ (Suppl. Note 7.1). Optimisation II
gives the lowest quality at low signal-to-noise ratio (Suppl. Note 7.5).

## Algorithm

```text
Input: views y_1 … y_V (registered, same shape), PSFs h_1 … h_V,
       kernel = "efficient", update = "sequential", n_iter = 10,
       optional weights w_1 … w_V and weight_sum k = 1
Pad every PSF to a common odd shape, normalise to unit sum
Compute the compound kernels h_b,1 … h_b,V (once)
Initialise x(0): constant, the mean intensity of all the views (one level per image)
For every block (core + halo) of the views, all views at once:
    Repeat n_iter times (or until the χ² rule stops, if chi2_stop):
        for v = 1 … V in turn:
            c_v = B_v[y_v / (A_v x)] / (B_v 1)           (1 where view v does not reach)
            x   = x ⊙ (1 − w_v + w_v c_v),  negative values set to 0
Return x, cropped to the observed region
```

The default `n_iter = 10` is not given in the paper, where the user chooses the
number of iterations. With `update="multiplicative"` the loop over views is
replaced by one update $\mathbf{x} \leftarrow \mathbf{x} \odot (\prod_v
\mathbf{c}_v)^{1/V}$, with every $\mathbf{c}_v$ evaluated at the same estimate; with
`update="additive"` by $\mathbf{x} \leftarrow \mathbf{x} \odot \sum_v
\mathbf{B}_v[\mathbf{y}_v / \mathbf{A}_v\mathbf{x}] / \sum_v \mathbf{B}_v\mathbf{1}$, the
maximum-likelihood EM of the stacked views that the paper compares against.

### View weights

Views that cover only part of the specimen are blended with weights
$w_v(\mathbf{r}) \in [0, 1]$, typically 1 in the central part of a view and
falling to 0 at its borders with a cosine, so that a view does not contribute
artefacts from its edges (Online Methods, Eq. 3). The sequential update of view
$v$ then applies the fraction $w_v$ of its correction.

=== "optix"

    $$
    \mathbf{x} \leftarrow \mathbf{x} \odot \bigl(\mathbf{1} - \mathbf{w}_v + \mathbf{w}_v \odot \mathbf{c}_v\bigr),
    \qquad
    \mathbf{w}_v \leftarrow \min\!\Bigl(\mathbf{1},\; \frac{k\, \mathbf{w}_v}{\sum_u \mathbf{w}_u}\Bigr)
    $$

=== "Preibisch et al. 2014"

    $$
    \psi^{r+1}(\xi) = \psi^{r}(\xi)\, w_v(\xi)\, f^{v}_{\mathrm{RL}}(\xi),
    \qquad \textstyle\sum_{v \in V} w_v(\xi) = 1
    \quad \text{(Online Methods, Eq. 3)}.
    $$

| optix | [Preibisch et al. 2014](https://doi.org/10.1038/nmeth.2929) |
|---|---|
| $\mathbf{w}_v$ (weight image of view $v$) | $w_v(\xi)$ |
| $\mathbf{c}_v$ (correction factor) | $f^v_{\mathrm{RL}}$ |
| $k$ (`weight_sum`) | the number the weights sum to; 1 by default |

By default the weights of every pixel sum to 1, which balances the update over
the image but gives no ordered-subset speed-up. Summing them to $k > 1$ recovers
the speed-up where $k$ views overlap, at the cost of an uneven deconvolution where
fewer do (Suppl. Fig. 10); the paper suggests the smallest number of overlapping
views. Every individual weight stays at most 1, otherwise the iteration becomes
unstable. The weights only gate the update: within a PSF half-width of the
border of a view, its missing pixels still enter the correction as observed
zeros, so the fall-off of the weights should be at least as wide as the PSF.
Weights are carried into the margin of the padded domain as
$\rho_v \cdot \mathrm{clamp}(\mathbf{B}_v[\mathbf{w}_v] / \mathbf{B}_v\mathbf{1}, 0, 1)$
(see the note on edge damping), so that a view without data beyond a border does
not erase the margin there.

## Faithful mode and extensions

`faithful=True` (the default) reproduces the paper: efficient kernels
(Eq. 102), sequential updates, a user-chosen fixed number of iterations, a
constant start at the mean intensity (Suppl. Eq. 11), and no weights unless
given. The defaults of `faithful=False` are the same.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `kernel` | `"efficient"` (Eq. 102), `"bayesian"` (Eq. 37), `"optimization-1"` (Eq. 114) or `"optimization-2"` (Eq. 115) | `"efficient"` | `"efficient"` |
| `update` | `"sequential"` (Eqs. 104–112), `"multiplicative"` (Eqs. 36, 46–47) or `"additive"` (maximum-likelihood EM) | `"sequential"` | `"sequential"` |
| `n_iter` | number of iterations (passes over all the views); not specified in the paper | `10` | `10` |
| `weights` (constructor) | one weight image per view, sequential update only | `None` | `None` |
| `weight_sum` | $k$, the sum of the weights over the views, per pixel | `1.0` | `1.0` |
| `chi2_stop` | extension: stop with Lucy's χ² test instead of after `n_iter` | `False` | `False` |
| `alpha` | level of the χ² test (with `chi2_stop`) | `0.05` | `0.05` |
| `max_iter` | cap of the χ² rule (not in the paper) | `100` | `100` |
| `halo` | halo width per spatial axis, at least the PSF size | `None` (twice the PSF size) | `None` |
| `block` | core block size per spatial axis | `None` (from the memory budget) | `None` |

The χ² stop is the only extension. Following
[Lucy (1974, §IV)](https://doi.org/10.1086/111605), it stops at the first
iteration for which $P(\chi^2 > \chi^2_{\mathrm{obs}}) \ge \alpha$, with
$\chi^2_{\mathrm{obs}} = \sum_v \sum_p (y_{vp} - \mu_{vp})^2 / \mu_{vp}$,
$\boldsymbol{\mu}_v = \mathbf{A}_v \mathbf{x}$ and $V P - 1$ degrees of freedom
($P$ pixels per view); it assumes counts.

!!! note "Choices the paper leaves open"
    These are fixed by the implementation and have no keyword:

    - **Back projection on the margin.** The compound kernels are used inside the
      block only. On the margin of the padded domain, where no view is observed
      directly, the back projection of every `kernel` is the adjoint
      $\mathbf{A}_v^\top$, since the narrower compound kernels would leave part of
      the margin without an update.
    - **Edge damping.** On that margin the update of view $v$ is applied with the
      weight $\rho_v = \mathbf{A}_v^\top\mathbf{1} / \max_u \mathbf{A}_u^\top\mathbf{1}$,
      the reach of the view relative to the view that sees the pixel best (also in
      the multiplicative update, as $\tilde{\mathbf{c}}_v = \mathbf{1} - \boldsymbol{\rho}_v +
      \boldsymbol{\rho}_v \odot \mathbf{c}_v$). The ordered-subset iteration assumes
      subsets of equal sensitivity, which fails on the margin, where the reach of the
      views differs by orders of magnitude; the damping restores that balance. Fixed
      points and positivity are unchanged. Without it, the noise-free data misfit is
      not monotone and stalls about ten times higher, in double precision too, and
      single and double precision results differ visibly. Inside the field of view
      the update rule is unchanged, and with one view $\rho_v = 1$; but the margin
      feeds the border pixels, so next to the image border the result differs from
      the undamped one, by up to about 12% of the maximum.
    - **Unit-sum kernels.** Each compound kernel is normalised to unit sum, and
      has the shape of the PSFs, since the product with $\check{\mathbf{h}}_v$
      restricts it to their support. The paper does not state how the compound
      kernels are normalised; Suppl. Fig. 3 compares both sides of Eq. 91 after
      normalisation.
    - **The $\lvert V \rvert$-th root of the multiplicative update.** Eq. 36 is
      printed as a plain product; the convergence proof (Eqs. 46–47) uses its
      geometric mean, which is what `update="multiplicative"` computes.
    - **Weights on the margin.** The weights of the margin are $\rho_v \cdot
      \mathrm{clamp}(\mathbf{B}_v[\mathbf{w}_v] / \mathbf{B}_v\mathbf{1}, 0, 1)$, the
      normalised block weights spread by the back projection; the paper has no margin.
    - **Reading of the weights.** Eq. 3 of the Online Methods multiplies the
      estimate by $w_v f^v_{\mathrm{RL}}$; the text says that $w_v$ is the fraction
      of the update that is applied, which is the form above ($1 - w_v + w_v c_v$).
      Rescaling to $k$ and capping each weight at 1 follows the text; the paper
      gives no formula. A pixel where all weights vanish is left unchanged.
    - **Exponent of optimisation II.** The product of Eq. 115 runs over $v$ and the
      $V - 1$ other views, which gives $V$ factors, $\check{\mathbf{h}}_v^{\,\odot V}$.
    - **Blocks.** The halo (twice the PSF size) is not specified in the paper.

!!! warning "Approximations"
    The virtual views rest on $Q(\xi \mid x_v) \approx P(x_v \mid \xi)$ (Eq. 76) and
    on Eq. 91, which hold for Gaussian-like PSFs; the convergence proof covers the
    plain Bayesian-based kernels with the multiplicative update only, not the
    efficient ones, the optimisations or the weights.

Not covered: the Tikhonov regularisation option of the paper, bead-based
registration, PSF estimation and the GPU implementation.

## Memory and streaming

The views are cut into blocks with a halo (`blocks`, halo streaming), the same
region of every view at once, as for
[Richardson–Lucy](richardson-lucy.md#memory-and-streaming). Each block runs all
its iterations on its own and only its core is kept, so the input is read once
and views can be lazy (`zarr`). A block holds, per voxel, the data, weight,
blurred estimate and ratio of each view plus a few shared buffers, about
$8 + 5V$ working buffers of its size: `block` and `halo` trade memory for seam
quality. Cost is two FFT convolutions per view and iteration (the blur and the
back projection), as for one Richardson–Lucy iteration per view; the compound
kernels are computed once, in double precision, and are as small as the PSFs.
`fit` makes one pass over the views to compute the starting level.

## Usage

The example images a test pattern through two anisotropic Gaussian PSFs, one
blurring along each axis, and compares the multiview result to the
Richardson–Lucy deconvolution of each view alone.

```pycon
>>> import torch
>>> from optix.deconv import MultiviewRL, RichardsonLucy, multiview_rl
>>> from optix.models import gaussian_psf, simulate_blur
>>> _ = torch.manual_seed(0)
>>> psfs = [
...     gaussian_psf((15, 15), sigma=(4.0, 1.0)),
...     gaussian_psf((15, 15), sigma=(1.0, 4.0)),
... ]
>>> x = torch.full((96, 96), 5.0)
>>> x[20:76:8, 20:76] = 200.0
>>> x[20:76, 30:76:12] = 200.0
>>> views = [torch.poisson(simulate_blur(x, psf)) for psf in psfs]
>>> mv = MultiviewRL(psfs, n_iter=10).fit(views)
>>> x_mv = mv.apply(views)
>>> x_mv.shape
torch.Size([96, 96])
>>> error = lambda image: float((image - x).abs().mean())
>>> single = [
...     RichardsonLucy(psf, n_iter=10).fit(y).apply(y)
...     for psf, y in zip(psfs, views)
... ]
>>> error(x_mv) < min(error(s) for s in single)
True
>>> mv.is_faithful
True
```

The kernels and the update schemes are compared by their keyword:

```pycon
>>> for kernel in ("bayesian", "efficient", "optimization-1", "optimization-2"):
...     print(
...         kernel,
...         round(error(multiview_rl(views, psfs, kernel=kernel, n_iter=5)), 1),
...     )
bayesian 13.8
efficient 13.8
optimization-1 13.8
optimization-2 13.2
>>> [tuple(k.shape) for k in mv.kernels()]
[(15, 15), (15, 15)]
```

Common variations: `update="multiplicative"` or `"additive"` for the
combined updates; `weights=[w1, w2]` with one weight image per view (lazy weights are not
stored by `save` and must be set again after `load`), and
`weight_sum=2.0` for the ordered-subset speed-up where two views overlap;
`chi2_stop=True` to stop by the χ² rule; views as `zarr` arrays, or with a layout
(time-lapse and group axes are iterated over, the views sharing one layout),
and `out`, processed block by block. The function `multiview_rl(data, psfs, ...)` is a
one-line shortcut, and the building blocks
[`compound_kernels`](../../reference/deconv.md) and
[`Multiview`](../../reference/deconv.md) are documented under Utilities. See the
[API reference](../../reference/deconv.md).

## References

!!! references "References"

    1. Preibisch S, Amat F, Stamataki E, Sarov M, Singer RH, Myers E,
       Tomancak P. *Efficient Bayesian-based multiview deconvolution.*
       Nat Methods **11**(6):645-648, 2014.
       [doi:10.1038/nmeth.2929](https://doi.org/10.1038/nmeth.2929)
    2. Hudson HM, Larkin RS. *Accelerated image reconstruction using ordered
       subsets of projection data.* IEEE Trans Med Imaging **13**(4):601-609,
       1994. [doi:10.1109/42.363108](https://doi.org/10.1109/42.363108)
    3. Shepp LA, Vardi Y. *Maximum likelihood reconstruction for emission
       tomography.* IEEE Trans Med Imaging **1**(2):113-122, 1982.
       [doi:10.1109/TMI.1982.4307558](https://doi.org/10.1109/TMI.1982.4307558)
    4. Lucy LB. *An iterative technique for the rectification of observed
       distributions.* Astron J **79**(6):745-754, 1974.
       [doi:10.1086/111605](https://doi.org/10.1086/111605)

    ??? quote "BibTeX"

        ```bibtex
        @article{preibisch2014efficient,
          author  = {Preibisch, Stephan and Amat, Fernando and Stamataki, Evangelia and
                     Sarov, Mihail and Singer, Robert H. and Myers, Eugene and
                     Tomancak, Pavel},
          title   = {Efficient {Bayesian}-based multiview deconvolution},
          journal = {Nature Methods},
          volume  = {11},
          number  = {6},
          pages   = {645--648},
          year    = {2014},
          doi     = {10.1038/nmeth.2929},
        }
        @article{hudson1994accelerated,
          author  = {Hudson, H. Malcolm and Larkin, Richard S.},
          title   = {Accelerated image reconstruction using ordered subsets of projection data},
          journal = {IEEE Transactions on Medical Imaging},
          volume  = {13},
          number  = {4},
          pages   = {601--609},
          year    = {1994},
          doi     = {10.1109/42.363108},
        }
        @article{shepp1982maximum,
          author  = {Shepp, Lawrence A. and Vardi, Yehuda},
          title   = {Maximum likelihood reconstruction for emission tomography},
          journal = {IEEE Transactions on Medical Imaging},
          volume  = {1},
          number  = {2},
          pages   = {113--122},
          year    = {1982},
          doi     = {10.1109/TMI.1982.4307558},
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
