# Nagy–O'Leary

!!! info "At a glance"
    **Estimates:** restored image (the latent image $\mathbf{x}$, same shape as the data), from an image blurred by a spatially variant PSF \
    **Noise model:** Gaussian (least squares, regularised by early stopping) \
    **Streaming:** halo \
    **Needs many frames:** no \
    **Reference:** Nagy and O'Leary, SIAM J Sci Comput, 1998, [doi:10.1137/S106482759528507X](https://doi.org/10.1137/S106482759528507X)

## Overview

`NagyOLeary` restores an image whose blur changes across the field, such as a
light-sheet or widefield image whose PSF widens towards the edges. The PSF is
measured (or modelled) on a coarse grid of positions; the blur at any pixel is a
weighted sum of the images blurred by the PSFs at the grid nodes, with weights
that are either constant over rectangles around each node or linearly
interpolated between nodes ([Nagy and O'Leary 1998, §2](https://doi.org/10.1137/S106482759528507X)).
The object is recovered by conjugate gradients on the least-squares problem,
stopped early to regularise, and preconditioned by a block-circulant
approximation of the blur that costs one FFT per region.

The blur model alone is available as
[`NagyOLearyPSF`](../../reference/deconv.md), and can be given as the `psf` of
[Richardson–Lucy](richardson-lucy.md), [Biggs–Andrews](biggs-andrews.md) and
[Snyder](snyder.md), which then keep their Poisson likelihood and
non-negativity. Use `NagyOLeary` for data with additive, roughly Gaussian noise
and a PSF that varies smoothly; use the RL family for photon-counting data. The
convolve-then-weight order of this model is not the blur of an interpolated PSF
([Denis et al. 2015, §2.3](https://doi.org/10.1007/s11263-015-0817-x)); the
[Denis model](denis.md) has the weighting before the convolution and
approximates a smoothly varying PSF better. `apply` returns the restored images.

## Image formation (optics)

The microscope is linear, as in
[Image formation and noise](../../concepts/image-formation.md), but not shift
invariant: the PSF depends on the position in the field (aberrations that grow
off-axis, a light sheet that thickens away from its waist, depth-dependent
blur). The PSFs $\mathbf{h}_s$ are known at $S$ nodes $\mathbf{r}_s$ of a
rectangular grid, in 2 or 3 dimensions; each is centred at index `K // 2` and
normalised to unit sum. As for [Richardson–Lucy](richardson-lucy.md), light from
outside the field of view is blurred into it: the object is estimated on a
domain extended by the PSF support, and the blur crops back to the observed
region.

## Probabilistic model

=== "optix"

    $$
    \mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon},
    \qquad
    \mathbf{A} = \sum_{s=1}^{S} \mathbf{W}_s \mathbf{H}_s ,
    \qquad
    \sum_s \mathbf{W}_s = \mathbf{I},
    $$

    with $\mathbf{H}_s$ the convolution with $\mathbf{h}_s$, and
    $\mathbf{W}_s = \operatorname{diag}(\mathbf{w}_s)$ a non-negative diagonal
    matrix of pixel weights applied to the *blurred* image.

=== "Nagy and O'Leary 1998"

    $$
    \mathbf{K} = \mathbf{D}_1 \mathbf{K}_1 + \cdots + \mathbf{D}_p \mathbf{K}_p
    \quad (2.1),
    \qquad \mathbf{K}\mathbf{f} \approx \mathbf{g},
    $$

    with $\mathbf{K}_i$ the block-Toeplitz convolution matrix of the PSF $k_i$
    and $\mathbf{D}_i$ a diagonal matrix with $\sum_i \mathbf{D}_i = \mathbf{I}$.

| optix | [Nagy and O'Leary 1998](https://doi.org/10.1137/S106482759528507X) |
|---|---|
| $\mathbf{y}$ | $\mathbf{g}$ |
| $\mathbf{x}$ | $\mathbf{f}$ |
| $\mathbf{A}$ | $\mathbf{K}$ |
| $\mathbf{H}_s$, $\mathbf{h}_s$ | $\mathbf{K}_i$, $k_i$ |
| $\mathbf{W}_s$, $\mathbf{w}_s$ | $\mathbf{D}_i$, $\delta_i$ |
| $\mathbf{r}_s$ | $s_i$ |
| $S$, $s$ | $p$, $i$ |
| $\boldsymbol{\varepsilon}$ | $\boldsymbol{\eta}$ |
| $\mathbf{C}_\tau$, $\tau$ | $C_\tau$, $\tau$ |

The weights $\mathbf{w}_s$ are chosen by `interpolation`:

- `"constant"` ([§3.2.1](https://doi.org/10.1137/S106482759528507X)): $\mathbf{w}_s$
  is the indicator of the rectangle of the pixels nearest to node $s$, whose
  boundaries lie half-way between neighbouring nodes;
- `"linear"` ([§3.2.2](https://doi.org/10.1137/S106482759528507X)): $\mathbf{w}_s$
  is the barycentric weight of node $s$ on a triangulation of the grid. Each
  rectangle between four nodes is split along its diagonal (Fig. 3.2); in 3-D, each
  box is split into six tetrahedra (Kuhn triangulation).

Beyond the outermost nodes, the weights are constant. Both are evaluated at the
global pixel positions, so the operator of any block of the image is the
whole-image operator restricted to that block.

## Inference / derivation

The object minimises the least-squares misfit, $\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{y}\|^2$,
by conjugate gradients on the normal equations (CGLS, Björck 1989), from $\mathbf{x}^{(0)} = \mathbf{0}$. The problem
is ill-posed and the noise enters the iterates after a few iterations, so
regularisation is by stopping early
([§1](https://doi.org/10.1137/S106482759528507X)): the paper uses no stopping
rule and compares solutions after a fixed number of iterations.

**Fast products** ([§3](https://doi.org/10.1137/S106482759528507X)). $\mathbf{A}\mathbf{x}$ is
evaluated with FFTs on a padded grid, as a sum of $S$ circular convolutions
followed by the pixel weights, cropped to the observed region. The adjoint is the
exact transpose,

$$
\mathbf{A}^\top \mathbf{r} = \sum_s \mathbf{H}_s^\top \mathbf{W}_s\, \mathbf{r},
$$

with $\mathbf{H}_s^\top$ the correlation with $\mathbf{h}_s$ (multiplication by the
conjugate transfer function).

**Preconditioner** ([§4](https://doi.org/10.1137/S106482759528507X)). Let
$\tilde{\mathbf{K}}_i$ be $\mathbf{K}_i$ restricted to region $i$ (Eq. 4.1). On a
region of $\rho$ pixels, $\tilde{\mathbf{K}}_i$ is block Toeplitz with Toeplitz
blocks; its closest block-circulant matrix with circulant blocks in the Frobenius
norm (Chan and Olkin 1993, as in [§4.1](https://doi.org/10.1137/S106482759528507X)) has first column

$$
c_j = \sum_{\mathbf{d} \equiv j \,(\mathrm{mod}\ \rho)} \prod_a \left(1 - \frac{|d_a|}{\rho_a}\right) h(\mathbf{d}),
$$

with $h(\mathbf{d})$ the PSF at offset $\mathbf{d}$ from its centre, and eigenvalues
$\lambda_i$ given by its DFT. The block-diagonal matrix $\mathbf{C}$ of these
blocks has its eigenvalues of magnitude below $\tau$ replaced by 1, which gives
$\mathbf{C}_\tau$ (after Hanke, Nagy and Plemmons 1993, as in [§4](https://doi.org/10.1137/S106482759528507X)): it matches
$\mathbf{A}$ on the large singular values (the signal subspace) and leaves the
small ones (the noise subspace) unchanged. CGLS then runs on the right-preconditioned
system, with $\mathbf{P} = \mathbf{C}_\tau^{-1}$ applied by one FFT pair per region:

$$
\begin{aligned}
\mathbf{s}_0 &= \mathbf{P}^\top \mathbf{A}^\top \mathbf{y}, \quad \mathbf{p}_0 = \mathbf{s}_0, \quad \mathbf{r}_0 = \mathbf{y}, \quad \mathbf{x}_0 = \mathbf{0},\\
\mathbf{t}_k &= \mathbf{P}\mathbf{p}_k, \quad \mathbf{q}_k = \mathbf{A}\mathbf{t}_k, \quad \alpha_k = \|\mathbf{s}_k\|^2 / \|\mathbf{q}_k\|^2,\\
\mathbf{x}_{k+1} &= \mathbf{x}_k + \alpha_k \mathbf{t}_k, \quad \mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k \mathbf{q}_k,\\
\mathbf{s}_{k+1} &= \mathbf{P}^\top \mathbf{A}^\top \mathbf{r}_{k+1}, \quad
\mathbf{p}_{k+1} = \mathbf{s}_{k+1} + \frac{\|\mathbf{s}_{k+1}\|^2}{\|\mathbf{s}_k\|^2}\,\mathbf{p}_k .
\end{aligned}
$$

With `interpolation="linear"`, the preconditioner uses on each region of the
piecewise-constant partition the PSF of that region, one of the vertex kernels
([§4.1.1](https://doi.org/10.1137/S106482759528507X)).

## Algorithm

```text
Input: data y, PSFs h_s on a grid with node positions r_s,
       interpolation = "constant", precondition = True, tau = 0, n_iter = 14
For each block of the image (the whole image when it fits in memory):
    Build A on the block from the weights at the global pixel positions
    If precondition: build C_tau, one circulant block per region of the block
    x(0) = 0; r = y; s = P' A' r; p = s
    Repeat n_iter times (or until s = 0 or q = 0):
        t = P p;  q = A t;  alpha = |s|² / |q|²
        x += alpha t;  r -= alpha q
        s_new = P' A' r;  p = s_new + (|s_new|² / |s|²) p;  s = s_new
    Keep the core of x, cropped to the observed region
Return the restored image
```

The number of iterations made per block is recorded in
`history.extras["iterations"]`.

## Faithful mode and extensions

`faithful=True` (the default) reproduces the paper: piecewise-constant weights,
preconditioned CG with $\tau = 0$ (all experiments of
[§5](https://doi.org/10.1137/S106482759528507X)), 14 iterations (the
preconditioned solutions of Fig. 5.8), no stopping criterion. The defaults of
`faithful=False` are the same: the keywords below select the paper's
alternatives, and none is an extension.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `interpolation` | weights of the blur model: region indicators (§3.2.1) or linear (§3.2.2) | `"constant"` | `"constant"` |
| `precondition` | preconditioned CG, or plain CG (the paper's reference method) | `True` | `True` |
| `tau` | truncation level $\tau$ of the preconditioner | 0 | 0 |
| `n_iter` | number of iterations | 14 | 14 |
| `halo` | halo width per spatial axis, at least the PSF size | `None` (twice the PSF size) | `None` |
| `block` | core block size per spatial axis | `None` (from the memory budget) | `None` |

!!! warning "Smooth PSFs need $\tau > 0$"
    The PSFs of the paper have a sharp core, so the eigenvalues of $\mathbf{C}$
    stay away from 0 and $\tau = 0$ works (§4: "for moderately ill-posed
    problems with little noise, we can take $\tau \approx 0$"). For smooth PSFs
    such as a Gaussian, the eigenvalues of $\mathbf{C}$ decay to tiny values, and
    $\mathbf{C}_\tau^{-1}$ with $\tau = 0$ amplifies the mismatch between
    $\mathbf{C}$ and $\mathbf{A}$ at the region boundaries and the noise: the
    iteration diverges from the first iterations. Choose $\tau$ a few per cent of
    the largest eigenvalue (about 1 for unit-sum PSFs), or `precondition=False`.
    The paper also notes that the preconditioner does not help on smooth images,
    and is less effective as the number of regions grows.

Choices that the paper does not specify:

- the number of iterations is that of Fig. 5.8; there is no stopping rule;
- CGLS starts from $\mathbf{x}^{(0)} = \mathbf{0}$;
- the object outside the observed region (the margin of the padded grid) is
  extended by the PSF support and is not preconditioned;
- region boundaries lie half-way between neighbouring node positions, and the
  weights are constant beyond the outermost nodes;
- each PSF is normalised to unit sum;
- eigenvalues that vanish to round-off (below the machine epsilon times the
  largest magnitude) are also replaced by 1, so that $\tau = 0$ is defined;
- the halo of blocks is twice the PSF size.

## Memory and streaming

Large images are cut into blocks with a halo (`blocks`, halo streaming), as for
[Richardson–Lucy](richardson-lucy.md#memory-and-streaming). Each block, core plus
halo, is restored as an image of its own and only its core is kept. The weights
come from global positions, so the blur of a block is exactly the whole-image
blur restricted to it. The preconditioner of a block uses the parts of the regions
that fall in it, and the CG inner products are computed per block, so
block-wise and whole-image results differ slightly away from the image edges. An
image that fits in the memory budget is restored as a single block, which is
then exact.

A block holds the weights and the transfer functions of the $S_b$ terms that touch
it (nodes with a non-zero weight in the block), and about sixteen working
buffers. Each term is convolved over the whole padded grid of the block (the
paper convolves region by region, which has the same result), so the cost of one
iteration is $S_b$ forward and $S_b$ adjoint FFT pairs, plus one FFT pair per region
for the preconditioner. Smaller blocks touch fewer terms; `block` and `halo`
trade memory for seam quality.

## Usage

The example restores an image blurred by Gaussian PSFs that widen from the top
to the bottom of the field, on a $3 \times 3$ grid of nodes, and compares it to the
blurred image and to a restoration with the central PSF only.

```pycon
>>> import torch
>>> from optix.deconv import NagyOLeary, NagyOLearyPSF
>>> from optix.models import gaussian_psf
>>> torch.manual_seed(0)
<torch._C.Generator object at ...>
>>> centres = [[16.0, 64.0, 112.0], [16.0, 64.0, 112.0]]
>>> row = lambda s: torch.stack([gaussian_psf((15, 15), sigma=s)] * 3)
>>> psf = torch.stack([row(1.0), row(2.0), row(3.0)])
>>> psf.shape
torch.Size([3, 3, 15, 15])
>>> # a test image: nine blobs on a background
>>> r, c = torch.meshgrid(torch.arange(128.0), torch.arange(128.0), indexing="ij")
>>> x = torch.full((128, 128), 5.0)
>>> for i in (24, 64, 104):
...     for j in (24, 64, 104):
...         x += 200 * torch.exp(-((r - i) ** 2 + (c - j) ** 2) / 8)
>>> # blur it with the model and add noise
>>> op = NagyOLearyPSF(psf, centres).operator((0, 0), x.shape)
>>> y = op.forward(op.pad(x))
>>> y = y + 0.05 * torch.randn_like(y)
>>> # restore: 14 preconditioned iterations (tau > 0 for Gaussian PSFs)
>>> est = NagyOLeary(psf, centres, tau=0.1).fit(y)
>>> x_hat = est.apply(y)
>>> est.history.extras["iterations"]
[14.0]
>>> # the same with only the central PSF
>>> one = NagyOLeary(psf[1:2, 1:2], [[64.0], [64.0]], tau=0.1).fit(y)
>>> err = lambda a: float((a - x).norm() / x.norm())
>>> err(x_hat) < err(y) and err(x_hat) < err(one.apply(y))
True
```

Common variations: `interpolation="linear"` for weights interpolated between
nodes; `precondition=False` for plain CG; a 3-D grid of 3-D PSFs, `psf` of shape
`(*G, *K)` with three grid and three kernel axes; a `zarr` array as input, with
`out`, processed block by block. The function
`nagy_oleary(data, psf, centres, ...)` is a one-line shortcut. To restore
Poisson data, give the blur model to the [RL family](richardson-lucy.md)
instead: `RichardsonLucy(NagyOLearyPSF(psf, centres))`. See the
[API reference](../../reference/deconv.md).

## References

!!! references "References"

    1. Nagy JG, O'Leary DP. *Restoring images degraded by spatially variant
       blur.* SIAM J Sci Comput **19**(4):1063-1082, 1998.
       [doi:10.1137/S106482759528507X](https://doi.org/10.1137/S106482759528507X)
    2. Denis L, Thiébaut É, Soulez F, Becker J-M, Mourya R. *Fast
       approximations of shift-variant blur.* Int J Comput Vis
       **115**(3):253-278, 2015.
       [doi:10.1007/s11263-015-0817-x](https://doi.org/10.1007/s11263-015-0817-x)
    3. Chan TF, Olkin JA. *Preconditioners for Toeplitz-block systems.*
       Numer Algorithms **6**:89-101, 1993.
    4. Hanke M, Nagy JG, Plemmons R. *Preconditioned iterative regularization.*
       In: Reichel L, Ruttan A, Varga RS (eds) Numerical Linear Algebra, de Gruyter,
       141-163, 1993.
    5. Björck Å. *Least squares methods.* In: Ciarlet P, Lions J (eds) Handbook of
       Numerical Analysis, vol 1, Elsevier/North Holland, 1989.

    ??? quote "BibTeX"

        ```bibtex
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
        @article{chan1993preconditioners,
          author  = {Chan, Tony F. and Olkin, Julia A.},
          title   = {Preconditioners for {T}oeplitz-block systems},
          journal = {Numerical Algorithms},
          volume  = {6},
          pages   = {89--101},
          year    = {1993},
        }
        @incollection{hanke1993preconditioned,
          author    = {Hanke, Martin and Nagy, James G. and Plemmons, Robert},
          title     = {Preconditioned iterative regularization},
          booktitle = {Numerical Linear Algebra},
          editor    = {Reichel, L. and Ruttan, A. and Varga, R. S.},
          publisher = {de Gruyter},
          address   = {Berlin},
          pages     = {141--163},
          year      = {1993},
        }
        @incollection{bjorck1989least,
          author    = {Bj{\"o}rck, {\AA}ke},
          title     = {Least squares methods},
          booktitle = {Handbook of Numerical Analysis},
          editor    = {Ciarlet, P. and Lions, J.},
          volume    = {1},
          publisher = {Elsevier/North Holland},
          address   = {Amsterdam},
          year      = {1989},
        }
        ```
