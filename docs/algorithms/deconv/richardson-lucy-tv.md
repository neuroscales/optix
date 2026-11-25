# Richardson–Lucy with total variation

!!! info "At a glance"
    **Estimates:** deconvolved image (the latent image $\mathbf{x}$, same shape as the data) \
    **Noise model:** Poisson \
    **Streaming:** halo \
    **Needs many frames:** no \
    **Reference:** Dey et al., Microsc Res Tech, 2006, [doi:10.1002/jemt.20294](https://doi.org/10.1002/jemt.20294)

## Overview

[`RichardsonLucyTV`](../../reference/deconv.md) removes a known blur from an
image, like [Richardson–Lucy](richardson-lucy.md), and adds a penalty on the
total variation (TV) of the estimate ([Dey et al. 2006](https://doi.org/10.1002/jemt.20294)).
Plain Richardson–Lucy amplifies noise as the iterations go on, and is stopped
early to avoid it. The TV term smooths the estimate along the level lines of the
image and not across them, so flat regions are denoised while edges stay sharp.
Use it for photon-counting data with a known PSF when a piecewise-smooth object is
a reasonable assumption, typically 3D confocal stacks. `RichardsonLucyTV` is a
subclass of `RichardsonLucy`: the blur, the boundary handling and the block-wise
streaming are the same, and every image of a stack is deconvolved on its own.
`apply` returns the deconvolved images.

## Image formation (optics)

As for [Richardson–Lucy](richardson-lucy.md#image-formation-optics): a
linear, shift-invariant blur by a PSF $\mathbf{h}$ of 2 or 3 dims, centred at
index `K // 2` and normalised to unit sum, followed by Poisson counting noise
(see [Image formation and noise](../../concepts/image-formation.md)). The paper
uses a model of the confocal PSF, which is not part of this estimator; pass the
PSF as an array (for instance a measured one). The paper has no background term,
since it argues that the regulariser makes one unnecessary.

## Probabilistic model

The data are Poisson counts of the blurred object, as in Richardson–Lucy, and
the object is penalised by its total variation:

=== "optix"

    $$
    \mathbf{y} \sim \mathrm{Poisson}(\mathbf{A}\mathbf{x}), \qquad \mathbf{x} \ge 0,
    $$

    $$
    J(\mathbf{x}) = \sum_p \bigl[(\mathbf{A}\mathbf{x})_p - y_p \log (\mathbf{A}\mathbf{x})_p\bigr]
    + \lambda_{\mathrm{TV}} \sum_p \lVert \nabla \mathbf{x} \rVert_p ,
    $$

    with $\mathbf{A}$ the blur and $\lVert \nabla \mathbf{x} \rVert_p$ the
    Euclidean norm of the gradient of $\mathbf{x}$ at pixel $p$. The estimate
    minimises $J$.

=== "Dey et al. 2006"

    $$
    i = \mathcal{P}(o \ast h), \qquad
    J_1(o) + J_{\mathrm{reg}}(o) =
    \sum_s \bigl[(o \ast h)(s) - i(s) \log (o \ast h)(s)\bigr]
    + \lambda_{\mathrm{TV}} \sum_s |\nabla o(s)|
    \quad \text{(Eqs. 1, 6, 9, 10)},
    $$

    with $\mathcal{P}$ the Poisson distribution.

| optix | [Dey et al. 2006](https://doi.org/10.1002/jemt.20294) |
|---|---|
| $\mathbf{y}$ | $i$ |
| $\mathbf{x}$ | $o$ |
| $\mathbf{h}$, $\mathbf{A}$ | $h$ ($o \ast h$ is $\mathbf{A}\mathbf{x}$) |
| $p$ | $s$ |
| $\lambda_{\mathrm{TV}}$ | $\lambda_{\mathrm{TV}}$ |
| $t$ (iteration) | $k$ |

The TV penalty is the prior that distinguishes this model from
[Richardson–Lucy](richardson-lucy.md): it is the $\ell_1$ norm of the gradient
magnitude, which favours piecewise-constant images, whereas a quadratic
(Tikhonov–Miller) penalty would blur the edges.

## Inference / derivation

The derivation of the multiplicative Richardson–Lucy update is on the
[Richardson–Lucy page](richardson-lucy.md#inference-derivation) and is not
repeated. With the penalty, the zero of the gradient of $J$ is approached by the
explicit, one-step-late scheme of
[Green (1990)](https://doi.org/10.1111/j.2517-6161.1990.tb01798.x): the
derivative of the penalty is evaluated at the current estimate and moved to the
denominator of the update. The derivative of the TV is
$-\lambda_{\mathrm{TV}} \operatorname{div}(\nabla\mathbf{x} / \lVert\nabla\mathbf{x}\rVert)$,
a non-linear curvature term.

=== "optix"

    $$
    \mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} \odot
    \frac{\mathbf{A}^\top\!\left[\mathbf{y} / \mathbf{A}\mathbf{x}^{(t)}\right] / \mathbf{A}^\top \mathbf{1}}
    {\mathbf{1} - \lambda_{\mathrm{TV}} \operatorname{div}\!\left(\nabla\mathbf{x}^{(t)} / \lVert\nabla\mathbf{x}^{(t)}\rVert\right)}
    $$

=== "Dey et al. 2006"

    $$
    o_{k+1}(s) = \left[\left(\frac{i}{o_k \ast h}\right) \star h\right](s)\,
    \frac{o_k(s)}{1 - \lambda_{\mathrm{TV}} \operatorname{div}\!\left(\frac{\nabla o_k(s)}{|\nabla o_k(s)|}\right)}
    \quad \text{(Eq. 11)},
    $$

    with $\star$ the correlation with $h$.

The numerator is the Richardson–Lucy update, normalised by $\mathbf{A}^\top\mathbf{1}$
(which is below 1 near the edges of the field; Eq. 11 is written for
$\mathbf{A}^\top\mathbf{1} = 1$). With $\lambda_{\mathrm{TV}} = 0$ the iteration
is Richardson–Lucy. Non-negativity holds as long as the denominator is positive,
which the paper obtains by keeping $\lambda_{\mathrm{TV}}$ small, and the
implementation by a bound on it (below).

### Discretisation

The paper does not give a discretisation of $\nabla$ and $\operatorname{div}$. Here,
on a periodic grid with unit voxel spacing along every axis, with $\mathbf{e}_k$ the
unit vector of axis $k$ and $d$ the number of spatial dims:

$$
(\nabla\mathbf{x})_k(\mathbf{r}) = x(\mathbf{r} + \mathbf{e}_k) - x(\mathbf{r}),
\qquad
\mathbf{n} = \frac{\nabla\mathbf{x}}{\max(\lVert\nabla\mathbf{x}\rVert, \varepsilon)},
\qquad
\operatorname{div}\mathbf{n}(\mathbf{r}) = \sum_{k=1}^{d} n_k(\mathbf{r}) - n_k(\mathbf{r} - \mathbf{e}_k),
$$

forward differences for the gradient and backward differences for the
divergence, which is minus the adjoint of the gradient. Then
$\operatorname{div}\mathbf{n}$ is exactly minus the gradient of the discrete TV.
The floor $\varepsilon$ of $\lVert\nabla\mathbf{x}\rVert$ is not in the paper
either. It only avoids a division of 0 by 0. By default it is the round-off level
of the working dtype, $\varepsilon = \mathrm{eps}\cdot\max\lvert\mathbf{x}\rvert$, with
$\mathrm{eps}$ the machine epsilon of the dtype, so that it does not depend on the
scale of the image. Where the gradient is below it, the normal vanishes with the
gradient.

Since $\lVert\mathbf{n}\rVert \le 1$, $\sum_k n_k(\mathbf{r}) \le \sqrt{d}$, and each
$-n_k(\mathbf{r} - \mathbf{e}_k) \le 1$ adds at most 1, so
$\lvert\operatorname{div}\mathbf{n}\rvert \le d + \sqrt{d}$ (the per-component bound
$\lvert n_k \rvert \le 1$ alone would only give $2d$). The
denominator stays positive when $\lambda_{\mathrm{TV}} < 1/(d + \sqrt{d})$, that is
0.29 in 2-D and 0.21 in 3-D. Larger values are refused with an error.

## Algorithm

```text
Input: data Y (counts), PSF h, lambda_TV = 0.002, tol = 1e-5, max_iter = 1000
Initialise x(0): the data (negative values set to 0); the mean of the data
    outside the observed region
Repeat:
    RL correction c = Aᵀ[y / A x] / Aᵀ1
    TV denominator d = 1 − lambda_TV · div(∇x / ‖∇x‖)
    x_new = x ⊙ c / d
    if n_iter is None and ‖x_new − x‖₂ ≤ tol · ‖x‖₂ on the observed block: stop
until n_iter (if given) or max_iter iterations
Return x, cropped to the observed region
```

The paper stops "at convergence, defined when the relative difference between
two images is less than a threshold", $10^{-5}$ in its experiments. The norm of
the relative difference is not specified, and neither is a cap on the iterations;
optix uses the $\ell_2$ norm on the observed region of each block, and 1000
iterations at most. The number of iterations per block is recorded in
`history.extras["iterations"]`. With the default $\varepsilon$ the threshold is
usually not reached: the one-step-late iteration keeps oscillating in flat noisy
regions, where the normals keep changing direction, and the relative change
plateaus near $1.5 \cdot 10^{-3}$, so `max_iter` is the effective stop. Use `n_iter`
to select the number of iterations directly, or a larger `epsilon` (an extension),
which smooths the TV term and gives a stop that does trigger.

### Starting estimate

The paper starts from "the measured 3D image, scaled by a value equal to the mean
of the image stack". This can be read in two ways, and the choice is open:

- **From the data** (the default): the measured image, with negative values set to
  0. Its scale does not matter, since both the Richardson–Lucy correction and the
  TV term are invariant to the scale of $\mathbf{x}$. Pixels that are 0 in the data
  stay 0 for ever, because the update is multiplicative.
- **From a uniform image** (`start="uniform"`): the uniform image with the mean
  intensity of the data, as in [Richardson–Lucy](richardson-lucy.md#algorithm).
  Every pixel can then become positive.

The two give different iterates and different results at a given iteration count.

## Faithful mode and extensions

`faithful=True` (the default) reproduces the paper: $\lambda_{\mathrm{TV}} = 0.002$
(the value of its experiments on synthetic data and on the real shell image of
its Fig. 4; it uses 0.005 for the real bead image of its Fig. 5), a start from the data, and the relative-change stopping rule with threshold
$10^{-5}$. The choices that the paper leaves open (discretisation, $\varepsilon$,
the norm and the cap of the stopping rule, the halo) are fixed as described above and
are the same in both modes. The defaults of `faithful=False` are the same. Any
keyword set to a value other than its faithful one makes
`is_faithful` False.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `lambda_tv` | weight $\lambda_{\mathrm{TV}}$ of the total variation, $0 \le \lambda_{\mathrm{TV}} < 1/(d + \sqrt{d})$ | 0.002 | 0.002 |
| `n_iter` | fixed number of iterations instead of the relative-change rule | `None` | `None` |
| `tol` | threshold of the relative change | $10^{-5}$ | $10^{-5}$ |
| `max_iter` | cap of the relative-change rule (not in the paper) | 1000 | 1000 |
| `epsilon` | floor $\varepsilon$ of $\lVert\nabla\mathbf{x}\rVert$ (not in the paper) | `None` (round-off level) | `None` |
| `start` | starting estimate: `"data"` or `"uniform"` | `"data"` | `"data"` |
| `halo` | halo width per spatial axis, at least the PSF size plus one | `None` (twice the PSF size plus one) | `None` |
| `block` | core block size per spatial axis | `None` (from the memory budget) | `None` |

Fitting with a camera model, `fit(data, init=sensor)` with a
[`SensorEstimate`](../../reference/estimates.md), converts the data to electrons,
$(\mathbf{y} - o)/g$, as in [Richardson–Lucy](richardson-lucy.md#faithful-mode-and-extensions),
and makes the estimator non-faithful. The background, shifted-Poisson and damping
options of Richardson–Lucy do not exist here. Anisotropic voxel spacing in the TV
term and a Hessian-based regulariser are not implemented: $\lambda_{\mathrm{TV}}$
acts per voxel, along every axis alike. The data of the paper are anisotropic
($30 \times 30 \times 50$ nm voxels for the synthetic images,
$89 \times 89 \times 230$ nm for the real ones), and the paper does not say whether
the differences along the axial axis were weighted.

### Float32 and float64

The normals $\nabla\mathbf{x}/\lVert\nabla\mathbf{x}\rVert$ are very sensitive to
round-off where the gradient is small against the noise, since they scale as
$1/\lVert\nabla\mathbf{x}\rVert$. The TV term, which is non-smooth at a vanishing
gradient and has $\varepsilon$ at round-off level, amplifies round-off by about
$10^3$ to $10^4$. A float32 and a float64 run therefore differ more than they do for
plain Richardson–Lucy: of the order of $10^{-3}$ of the peak, against about
$10^{-5}$. With a larger floor, `epsilon=1` for instance (an extension), the
difference drops to about $4 \cdot 10^{-6}$, the level of plain Richardson–Lucy.

## Memory and streaming

As in [Richardson–Lucy](richardson-lucy.md#memory-and-streaming): the image is cut into
blocks with a halo, each block is iterated on its own by FFT on a padded grid, and
only its core is written back, in one pass over the data. The TV stencil reaches
one voxel beyond the PSF, so the halo is at least the PSF size plus one (twice that
by default; a smaller `halo` raises an error). The relative-change test is applied
to each block separately. A block needs about nine working buffers more than for
Richardson–Lucy (gradient, norm, normals, divergence). Blocks and whole-image
results agree to about $10^{-3}$ of the peak.

## Usage

The example blurs a [phantom](../../reference/models.md#phantoms) of flat cells by a Gaussian PSF from
[`optix.models`](../../reference/models.md) and compares the error to the
truth, for the same number of iterations.

```python
import torch
from optix.deconv import RichardsonLucy, RichardsonLucyTV
from optix.models import gaussian_psf, phantoms, simulate_blur

torch.manual_seed(0)

# 1. A Gaussian PSF and a blurred, Poisson-noisy field of flat cells.
psf = gaussian_psf((11, 11), sigma=2.0)
x = phantoms.cells((96, 96), count=6, radius=12.0, seed=0) + 10
y = torch.poisson(simulate_blur(x, psf))

# 2. Fit the same number of iterations, without and with the TV term.
rl = RichardsonLucy(psf, n_iter=100).fit(y)
tv = RichardsonLucyTV(psf, n_iter=100).fit(y)

# 3. Apply.
x_rl, x_tv = rl.apply(y), tv.apply(y)

# 4. Compare to the truth: mean error about 8.5 (RL), 6.6 (TV).
print(float((x_rl - x).abs().mean()), float((x_tv - x).abs().mean()))
```

Common variations: `RichardsonLucyTV(psf)` alone for the paper's stopping rule
(up to 1000 iterations); `lambda_tv=0.005` for a stronger penalty;
`start="uniform"`; 3-D stacks with a 3-D PSF; `fit(data, init=sensor)` for raw
camera data; a `zarr` array as input and `out`, processed block by block. The
function `richardson_lucy_tv(data, psf, ...)` is a shortcut, and
[`tv_curvature`](../../reference/deconv.md) computes the curvature term alone. See the
[API reference](../../reference/deconv.md).

## References

!!! references "References"

    1. Dey N, Blanc-Féraud L, Zimmer C, Roux P, Kam Z, Olivo-Marin JC, Zerubia J.
       *Richardson–Lucy algorithm with total variation regularization for 3D
       confocal microscope deconvolution.* Microsc Res Tech **69**(4):260-266, 2006.
       [doi:10.1002/jemt.20294](https://doi.org/10.1002/jemt.20294)
    2. Green PJ. *On use of the EM algorithm for penalized likelihood estimation.*
       J R Stat Soc B **52**(3):443-452, 1990.
       [doi:10.1111/j.2517-6161.1990.tb01798.x](https://doi.org/10.1111/j.2517-6161.1990.tb01798.x)

    ??? quote "BibTeX"

        ```bibtex
        @article{dey2006richardson,
          author  = {Dey, Nicolas and Blanc-F{\'e}raud, Laure and Zimmer, Christophe and Roux, Pascal and Kam, Zvi and Olivo-Marin, Jean-Christophe and Zerubia, Josiane},
          title   = {{Richardson--Lucy} algorithm with total variation regularization for {3D} confocal microscope deconvolution},
          journal = {Microscopy Research and Technique},
          volume  = {69},
          number  = {4},
          pages   = {260--266},
          year    = {2006},
          doi     = {10.1002/jemt.20294},
        }
        @article{green1990use,
          author  = {Green, Peter J.},
          title   = {On use of the {EM} algorithm for penalized likelihood estimation},
          journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
          volume  = {52},
          number  = {3},
          pages   = {443--452},
          year    = {1990},
          doi     = {10.1111/j.2517-6161.1990.tb01798.x},
        }
        ```
