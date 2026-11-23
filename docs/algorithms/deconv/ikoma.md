# Ikoma

!!! info "At a glance"
    **Estimates:** deconvolved image (the latent image $\mathbf{x}$, in electrons, same shape as the data) \
    **Noise model:** Poisson-Gaussian, as a shifted Poisson \
    **Streaming:** halo \
    **Needs many frames:** no \
    **Reference:** Ikoma, Broxton, Kudo and Wetzstein, Sci Rep, 2018, [doi:10.1038/s41598-018-29768-x](https://doi.org/10.1038/s41598-018-29768-x)

## Overview

`Ikoma` deconvolves a widefield focal stack, or an image, recorded at a low
photon count: given the data $\mathbf{y}$ and the point spread function (PSF)
$\mathbf{h}$, it estimates the latent image $\mathbf{x}$
([Ikoma et al. 2018](https://doi.org/10.1038/s41598-018-29768-x)). It differs
from [Richardson–Lucy](richardson-lucy.md) in three ways. The noise model keeps
the read noise of the camera, as a shifted Poisson law. The estimate is
regularised by the Frobenius norm of the Hessian, which favours locally smooth
structures and preserves edges, and it is constrained to be non-negative.
And the problem is solved to convergence by the alternating direction method of
multipliers (ADMM), with no early stopping to tune. In the simulations of the
paper the shifted-Poisson fidelity beats the Gaussian and Poisson ones at every
photon count, and the Hessian penalty, with the Laplacian, beats the other
penalties it compares (squared $\ell_2$, $\ell_1$, total variation) at every
noise level, and the Laplacian in almost all cases.

Use it for images with peaks of a few to a few hundred photoelectrons, where
read noise matters and Richardson–Lucy amplifies noise. The price is one weight $\nu$ to choose (a default is
provided), and a heavier iteration than Richardson–Lucy. For a one-step linear
filter see [Wiener](wiener.md) and [Tikhonov](tikhonov.md). `apply` returns the
deconvolved images. The ADMM solver and the operators it is built from are
described in [Optimisation tools](../../concepts/optimisation.md).

## Image formation (optics)

The microscope blurs the latent image with a known, shift-invariant PSF, and the
camera adds shot noise and read noise (see
[Image formation and noise](../../concepts/image-formation.md)). The PSF has 2 or
3 dims and is centred at index `K // 2`. It is normalised so that each lateral
plane (the last two spatial axes) sums to 1, which expresses $\mathbf{x}$ in
photoelectrons collected during the exposure of *one* image of the focal stack
(SI §H): every plane of a widefield stack is a realisation of the same Poisson
process, so the total number of electrons in $\mathbf{x}$ is much lower than the
number recorded in the whole stack.

The data are first brought to electrons. The dark field (or, without a shading
estimate, the camera offset) is subtracted, the result is divided by the flat
field, rescaled to unit median, and by the gain; the read-noise variance
becomes $\tilde\sigma^2 = \sigma^2 / g^2$. Both come from the
[`SensorEstimate`](../../reference/estimates.md) and
[`ShadingEstimate`](../../reference/estimates.md) given to `fit`. Without a
sensor estimate the data are taken to be in electrons and $\tilde\sigma^2 = 0$.

=== "optix"

    $$
    \tilde{\mathbf{y}} = \frac{(\mathbf{y} - \mathbf{d}) / \mathbf{f}}{g},
    \qquad
    \tilde{\mathbf{y}} = \mathbf{n} + \tilde{\boldsymbol{\varepsilon}},
    \quad
    \mathbf{n} \sim \mathrm{Poisson}(\mathbf{A}\mathbf{x}),
    \quad
    \tilde{\boldsymbol{\varepsilon}} \sim \mathcal{N}(\mathbf{0}, \tilde\sigma^2),
    \qquad
    \tilde\sigma^2 = \frac{\sigma^2}{g^2},
    $$

    with $\mathbf{y}$ the recorded stack in ADU, $\mathbf{d}$ the dark field
    (offset included), $\mathbf{f}$ the flat field, $g$ the gain and
    $\mathbf{A}$ the convolution with the PSF.

=== "Ikoma et al. 2018"

    $$
    y_j = \gamma\, \mathcal{P}\bigl((\mathbf{A}\mathbf{x})_j\bigr) + \mathcal{N}(0, \sigma^2)
    \quad (3),
    \qquad
    \tilde{y}_j = y_j / \gamma,
    \quad
    \tilde{\sigma} = \sigma / \gamma ,
    $$

    after dark-field and flat-field correction, with $\mathcal{P}(\theta)$ a
    Poisson realisation of rate $\theta$ and $\gamma$ the camera gain.

| optix | [Ikoma et al. 2018](https://doi.org/10.1038/s41598-018-29768-x) |
|---|---|
| $\mathbf{y}$ (raw, ADU) | $y$ before the dark-field and flat-field correction |
| $\tilde{\mathbf{y}}$ (electrons) | $\tilde{y}$ |
| $g$ | $\gamma$ |
| $\sigma^2$ (ADU²) | $\sigma^2$ |
| $\tilde\sigma^2$ (e⁻²) | $\tilde\sigma^2 = (\sigma/\gamma)^2$ |
| $\mathbf{A}$, $\mathbf{h}$ | $A$, $h$ |
| $\mathbf{d}$, $\mathbf{f}$ | dark-field and flat-field images (Methods) |

The paper estimates $g$ and $\sigma$ from the focal stack itself with the method of
[Foi et al. (2008)](https://doi.org/10.1109/TIP.2008.2001399), after the flat-field
correction (SI §F); in optix that is the job of
[`PoissonGaussian`](../sensor/poisson-gaussian.md), and the result is passed as
`init=sensor`. As in the paper, these estimates should describe the
flat-corrected data: the code divides the variance by $g^2$ only.

## Probabilistic model

The sum of a Poisson and a Gaussian variable has no closed-form likelihood, so
the model is approximated by a shifted Poisson law, which has the same mean and
variance for every photon count:

$$
\tilde{y}_j + \tilde\sigma^2 \sim \mathrm{Poisson}\bigl((\mathbf{A}\mathbf{x})_j + \tilde\sigma^2\bigr)
\qquad (4).
$$

The image $\mathbf{x}$ does not live on the observed region only. It lives on a
larger grid of $P$ voxels (the paper's $N > M$), and the likelihood is evaluated
on the set $\mathcal{C}$ of the observed voxels, so that light blurred into the
field of view from outside it is modelled, with no padding or apodisation. The
estimate minimises the negative log-likelihood, a penalty on the Hessian and the
non-negativity constraint:

=== "optix"

    $$
    \min_{\mathbf{x}} \;
    \sum_{j \in \mathcal{C}} \Bigl[ (\mathbf{A}\mathbf{x})_j
      - (\tilde{y}_j + \tilde\sigma^2) \log\bigl((\mathbf{A}\mathbf{x})_j + \tilde\sigma^2\bigr) \Bigr]
    + \nu \sum_{j \in \Omega} \bigl\lVert (\mathbf{H}\mathbf{x})_j \bigr\rVert_2
    + \iota_{\ge 0}(\mathbf{x}),
    $$

    where $\mathbf{H}\mathbf{x}$ packs the second differences of $\mathbf{x}$
    (below), $\Omega$ is the whole grid and $\iota_{\ge 0}$ is the indicator
    function of the non-negative orthant (0 inside, $+\infty$ outside).

=== "Ikoma et al. 2018"

    $$
    \min_{\mathbf{x}} \; \ell(\mathbf{x}) + \nu\, R(\mathbf{x}) + I_{[0,+\infty)}(\mathbf{x}),
    \qquad
    R(\mathbf{x}) = \sum_{j \in \Omega} \lVert H_j \mathbf{x} \rVert_F
    \quad (5),
    $$

    with $\ell$ the shifted-Poisson negative log-likelihood (S3), $H_j$ the
    $3 \times 3$ Hessian at voxel $j$ and $\mathcal{C}$, $\Omega$ sets of voxel
    indices.

The data are clipped at $-\tilde\sigma^2$, $\tilde{y}_j \leftarrow \max(\tilde{y}_j, -\tilde\sigma^2)$,
which makes $\tilde{y}_j + \tilde\sigma^2 \ge 0$ and the fidelity convex (SI §B).
The second differences are compositions of forward differences,
$\mathbf{D}_{ab} = \mathbf{D}_a \mathbf{D}_b$ (SI Eq. S5), so that
$(\mathbf{D}_{aa}\mathbf{x})_i = x_{i+2} - 2x_{i+1} + x_i$ along axis $a$, and
they are packed as (S7)

$$
\mathbf{H}\mathbf{x} =
\bigl(\mathbf{D}_{11}\mathbf{x}, \dots, \mathbf{D}_{dd}\mathbf{x},\;
\sqrt{2}\,\mathbf{D}_{12}\mathbf{x}, \dots, \sqrt{2}\,\mathbf{D}_{d-1,d}\mathbf{x}\bigr),
$$

with $d = 2$ or $3$ the number of spatial axes, so that $\lVert (\mathbf{H}\mathbf{x})_j \rVert_2$ is the
Frobenius norm of the $d \times d$ Hessian matrix at voxel $j$ (the paper's
$\lVert H_j \mathbf{x} \rVert_F$, with $d(d+1)/2$ components instead of $d^2$).
The convolution and the differences are periodic on the extended grid.

| optix | [Ikoma et al. 2018](https://doi.org/10.1038/s41598-018-29768-x) |
|---|---|
| $\mathbf{x}$, $P$ (voxels of the grid) | $x$, $N$ |
| $\mathcal{C}$, $\lvert\mathcal{C}\rvert$ | $\mathcal{C}$, $M$ |
| $\mathbf{H}\mathbf{x}$, $(\mathbf{H}\mathbf{x})_j$ | $H\theta$ (S7), $H_j \theta$ (S5) |
| $\mathbf{D}_a$, $\mathbf{D}_{ab}$ | $D_x, D_y, D_z$, $D_{xy}, \dots$ |
| $\nu$, $\Omega$ | $\nu$, $\Omega$ |
| $\iota_{\ge 0}$ | $I_{[0,+\infty)}$ |
| $\mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3$ | $z_1, z_2, z_3$ |
| $\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$ | $u_1, u_2, u_3$ |

## Inference / derivation

The problem is convex and is solved by ADMM
([Boyd et al. 2011](https://doi.org/10.1561/2200000016)). Each non-smooth or
non-quadratic piece gets its own copy of the variable it acts on: the splits
$\mathbf{z}_1 = \mathbf{A}\mathbf{x}$ (fidelity), $\mathbf{z}_2 = \mathbf{x}$
(non-negativity) and $\mathbf{z}_3 = \mathbf{H}\mathbf{x}$ (penalty), tied to
$\mathbf{x}$ by scaled dual variables $\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$
and a single penalty $\rho$. One iteration updates $\mathbf{x}$ first, then the
splits, then the duals (SI Eqs. S9–S15):

=== "optix"

    $$
    \begin{aligned}
    \mathbf{x} &\leftarrow (\mathbf{A}^\top\!\mathbf{A} + \mathbf{I} + \mathbf{H}^\top\!\mathbf{H})^{-1}
      \bigl[\mathbf{A}^\top(\mathbf{z}_1 - \mathbf{u}_1) + (\mathbf{z}_2 - \mathbf{u}_2) + \mathbf{H}^\top(\mathbf{z}_3 - \mathbf{u}_3)\bigr] \\
    \mathbf{z}_1 &\leftarrow \mathrm{prox}_{\ell/\rho}(\mathbf{A}\mathbf{x} + \mathbf{u}_1)
      \text{ on } \mathcal{C}, \quad \mathbf{A}\mathbf{x} + \mathbf{u}_1 \text{ elsewhere} \\
    \mathbf{z}_2 &\leftarrow \max(\mathbf{x} + \mathbf{u}_2, 0) \\
    \mathbf{z}_3 &\leftarrow \max\!\Bigl(1 - \frac{\nu/\rho}{\lVert (\mathbf{H}\mathbf{x} + \mathbf{u}_3)_j \rVert_2}, 0\Bigr)
      (\mathbf{H}\mathbf{x} + \mathbf{u}_3)_j \\
    \mathbf{u}_k &\leftarrow \mathbf{u}_k + \mathbf{A}_k\mathbf{x} - \mathbf{z}_k,
      \quad \mathbf{A}_1 = \mathbf{A},\ \mathbf{A}_2 = \mathbf{I},\ \mathbf{A}_3 = \mathbf{H}
    \end{aligned}
    $$

=== "Ikoma et al. 2018"

    $$
    \begin{aligned}
    x^{k+1} &= (A^\top\!A + I^\top\! I + H^\top\!H)^{-1}
      \bigl[A^\top(z_1^k - u_1^k) + I^\top(z_2^k - u_2^k) + H^\top(z_3^k - u_3^k)\bigr] && (S16) \\
    z_{1,j}^{k+1} &= -b_j + \sqrt{b_j^2 + \dfrac{\tilde{y}_j + \rho\tilde\sigma^2 v_j}{\rho}},
      \quad v = A x^{k+1} + u_1^k,\ b_j = \dfrac{1 - \rho(v_j - \tilde\sigma^2)}{2\rho}
      && (S17,\ j \in \mathcal{C}) \\
    z_{2}^{k+1} &= \max(x^{k+1} + u_2^k, 0) && (S18) \\
    [z_3^{k+1}]_j &= \max\!\Bigl(1 - \dfrac{\nu/\rho}{\lVert [H x^{k+1} + u_3^k]_j \rVert_2}, 0\Bigr)
      [H x^{k+1} + u_3^k]_j && (S19)
    \end{aligned}
    $$

All four updates are closed-form:

- **$\mathbf{x}$.** $\mathbf{A}$, $\mathbf{I}$ and $\mathbf{H}$ are circulant on the periodic grid, so the
  matrix to invert is diagonal in the Fourier domain and the solve is one
  division, $|\hat{A}|^2 + 1 + \sum_c |\hat{H}_c|^2$ at each frequency. Neither the
  blur nor the differences are ever formed as matrices.
- **$\mathbf{z}_1$.** Setting the derivative of the penalised fidelity to 0 gives
  a quadratic in $w = z + \tilde\sigma^2$. With $c = \tilde{y} + \tilde\sigma^2$,
  $s = v + \tilde\sigma^2$, $q = \rho s - 1$ and $\Delta = \sqrt{q^2 + 4\rho c}$,
  its non-negative root is
  $w = (q + \Delta) / 2\rho$ when $q \ge 0$ and $w = 2c / (\Delta - q)$ otherwise,
  and $z_1 = w - \tilde\sigma^2$. This is algebraically (S17); the second branch
  avoids the cancellation of $q + \Delta$ when $q < 0$, which matters in
  single precision (see [`prox_shifted_poisson`](../../reference/optim.md)). Outside
  $\mathcal{C}$ the fidelity is absent and $\mathbf{z}_1 = \mathbf{A}\mathbf{x} + \mathbf{u}_1$.
- **$\mathbf{z}_2$.** Projection onto the non-negative orthant.
- **$\mathbf{z}_3$.** Block soft thresholding of the packed Hessian at each voxel,
  at the threshold $\nu/\rho$.

## Algorithm

```text
Input: data Y, PSF h, sensor (g, σ²) and shading (d, f) estimates, nu (optional)
Preprocess: ỹ = (y − d) / f / g;  σ̃² = σ² / g²;  ỹ ← max(ỹ, −σ̃²)
            h ← h / (sum of each xy-plane of h)
For each image:
    ν = nu, or κ · (sum of h) / max(ỹ) with κ = 0.1
    ρ = 6000 ν / max(ỹ)               (6 / max(ỹ) without a regulariser)
    Initialise x = z1 = z2 = z3 = u1 = u2 = u3 = 0 on the grid of size ≥ L + K − 1
    Repeat for k = 1 … 150:
        1. x  ← Fourier solve (S16)
        2. z1 ← shifted-Poisson prox on the observed voxels (S17)
           z2 ← max(x + u2, 0)                                  (S18)
           z3 ← block soft threshold of H x + u3 at ν / ρ        (S19)
        3. u_k ← u_k + A_k x − z_k
        4. stop if ‖x⁽ᵏ⁺¹⁾ − x⁽ᵏ⁾‖₂ < 10⁻³ ‖x⁽ᵏ⁾‖₂
Return x, cropped to the observed region
```

The number of iterations and the last relative change of every block are
recorded in `history.extras["iterations"]` and `history.extras["change"]`; the
values of $\nu$, $\rho$ and $\max \tilde{y}$ used for each image are the
attributes `nu`, `rho` and `peak` after `fit`.

## Faithful mode and extensions

`faithful=True` (the default) reproduces the paper: the Hessian penalty,
$\rho = 6000\nu / \max\tilde{y}$, the per-plane PSF normalisation, the start from
zero, the update of $\mathbf{x}$ first, and the stopping rule. The defaults of
`faithful=False` are the same. The weight `nu` is tuned per data set in the paper
and any value keeps the estimate faithful, as does `regulariser="none"`, one of
the paper's comparison variants (Table 1). Every other option marks the
estimate as not faithful.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `nu` | regularisation weight $\nu \ge 0$ | `None` (rule below, not in the paper) | `None` |
| `regulariser` | `"hessian"`: Frobenius norm of the Hessian; `"none"`: non-negativity only, $\rho = 6/\max\tilde{y}$; `"tv"`: isotropic total variation, extension | `"hessian"` | `"hessian"` |
| `rho` | ADMM penalty $\rho > 0$ | `None` ($6000\nu/\max\tilde{y}$) | `None` |
| `max_iter` | iteration cap | 150 | 150 |
| `tol` | relative-change tolerance; `None` runs `max_iter` iterations | $10^{-3}$ | $10^{-3}$ |
| `psf_norm` | `"plane"`: each xy-plane sums to 1; `"sum"`: the PSF sums to 1 (extension) | `"plane"` | `"plane"` |
| `weights` | per-axis weights $w_a$ of the differences, $\mathbf{D}_a \to w_a \mathbf{D}_a$ (extension) | `None` (unit weights) | `None` |
| `halo` | halo width per spatial axis, at least the PSF size | `None` (twice the PSF size) | `None` |
| `block` | core block size per spatial axis | `None` (memory budget) | `None` |

A variant with the Poisson likelihood ($\tilde\sigma^2 = 0$) is obtained by
fitting without a sensor estimate. The total-variation regulariser is one of
the paper's comparison penalties, but the paper gives its $\rho$ only for the
Hessian; `"tv"` uses the same rule, which is an author choice. The weights allow
for anisotropic sampling, which the paper mentions and does not use, for
instance $w_a = \min(s) / s_a$ for the voxel spacings $s_a$.

**Choices the paper does not specify.** These are made by the code, and flagged
as such in the docstrings:

- **Default $\nu$.** The paper tunes $\nu$ per data set (it scans 0.005 to 0.1) and gives
  no default. The default is $\nu = \kappa \sum\mathbf{h} / \max\tilde{y}$ with
  $\kappa = 0.1$, per image, where $\sum\mathbf{h}$ is the sum of the normalised PSF (1 for
  `psf_norm="sum"`, the number of planes for `"plane"`). The rule is
  scale-invariant: writing $\mathbf{x} = s\hat{\mathbf{x}}$ with $s = \max\tilde{y}$,
  the shifted-Poisson fidelity is, to second order, a weighted least squares with
  the noise variance of the data, and a prior that does not depend on the
  photon count, $\nu' \lVert \mathbf{H}\hat{\mathbf{x}} \rVert$, gives $\nu = \nu' / s$.
  The read noise drops out. The constant $\kappa$ was calibrated on 18 simulated cases
  (blobs, filaments and their mixture, in 2-D and 3-D, with peaks of 10, 100 and
  1000 photoelectrons, $\gamma = 2$, $\sigma = 3$ as in the paper): on average its
  SNR is 1.3 dB below that of the best $\nu$ of a fine grid, and 4 dB in the worst
  case, where the image content decides (smooth blobs want a larger $\nu$, thin
  filaments a smaller one). Pass `nu` to set it.
- **Grid.** $P$ is, along each axis, the observed size plus the PSF size minus one,
  rounded up to a fast FFT size. The paper says only that $N > M$.
- **Output.** The result is $\mathbf{x}$, which may be slightly negative, cropped
  to the observed region. The paper does not say whether $\mathbf{x}$ or $\mathbf{z}_2$ is returned.
- **Stopping rule.** The paper stops on the "relative change" of the iterates, without
  saying of which variable; it is $\mathbf{x}$, in float64 norms.
- **Halo.** Twice the PSF size.
- **Camera.** The dark field replaces the offset when a shading estimate is given.
  The flat field is rescaled to unit median, as in the Methods. A per-frame baseline
  of the shading estimate is ignored, with a warning.

!!! warning "PSF normalisation"
    The per-plane normalisation assumes that every plane of the PSF carries the same
    energy, as for a widefield PSF. For a PSF whose planes differ (confocal,
    light sheet) it mis-scales $\mathbf{x}$, and with it the default $\nu$; use
    `psf_norm="sum"`. For a PSF of $K_z$ planes of equal energy the two
    conventions differ by $\mathbf{x}_\text{plane} = \mathbf{x}_\text{sum} / K_z$ and
    $\nu_\text{plane} = K_z\, \nu_\text{sum}$. A PSF as large as the stack (as
    simulated in the paper) doubles the grid: crop it to its support.

## Memory and streaming

Large images are cut into blocks with a halo and each block is solved as an
image of its own, as for [Richardson–Lucy](richardson-lucy.md#memory-and-streaming)
(`blocks`, halo streaming, `PaddedConvolution`). `fit` makes one pass over the
data to find the maximum $\max\tilde{y}$ of every image, from which $\nu$ and $\rho$ follow,
so that every block of an image shares them. `apply` makes a second pass.

The solver is much hungrier than Richardson–Lucy. It holds, per voxel of the
loaded block and its padding, the grid, the splits and the duals: about
$17 + 6C$ buffers for $C$ regulariser components, that is 53 for the 3-D
Hessian ($C = 6$), 35 for the 2-D Hessian ($C = 3$), and 17 without a
regulariser. An iteration costs $5 + 2C$ FFTs of the grid, 17 for the 3-D Hessian. The block size is chosen from the
memory budget (`optix.config(memory=...)`), or set with `block`.

The $\mathbf{x}$-update is a global solve, so the block-wise result is not the
whole-image one near the seams, which Richardson–Lucy's is along image edges.
The seams fade with the halo; a `RuntimeWarning` says when an image is split.
Raise the memory budget or pass `block=` to avoid it.

## Usage

The example blurs a test pattern with a Gaussian PSF from
[`optix.models`](../../reference/models.md), records it with a camera of gain
2, offset 100 ADU and read noise of 3 ADU, and deconvolves it with the
`SensorEstimate` of that camera.

```pycon
>>> import torch
>>> from optix.deconv import Ikoma, RichardsonLucy
>>> from optix.estimates import SensorEstimate
>>> from optix.models import gaussian_psf, simulate_blur
>>> _ = torch.manual_seed(0)

>>> # 1. A Gaussian PSF and a noisy recording of a test pattern.
>>> psf = gaussian_psf((15, 15), sigma=2.0)
>>> x = torch.full((64, 64), 1.0)
>>> x[16:48:8, 16:48] = 20.0
>>> gain, offset, sigma = 2.0, 100.0, 3.0
>>> counts = torch.poisson(simulate_blur(x, psf))
>>> y = offset + gain * counts + sigma * torch.randn(64, 64)
>>> sensor = SensorEstimate(offset=offset, gain=gain, variance=sigma**2)

>>> # 2. Fit and deconvolve; Richardson-Lucy for comparison.
>>> ikoma = Ikoma(psf).fit(y, init=sensor)
>>> x_ikoma = ikoma.apply(y)
>>> x_rl = RichardsonLucy(psf).fit(y, init=sensor).apply(y)

>>> # 3. The weights that were used, and the errors against the truth.
>>> round(float(ikoma.nu), 4), round(float(ikoma.rho), 2)
(0.0078, 3.68)
>>> ikoma.history.extras["iterations"], ikoma.is_faithful
([150.0], True)
>>> [round(float((e - x).abs().mean()), 2) for e in (x_ikoma, x_rl)]
[1.38, 1.99]
```

Common variations: `Ikoma(psf, nu=0.02)` to set the weight, then
`fit(y, init=sensor)` again to see its effect; `regulariser="none"` for the
shifted-Poisson fit with non-negativity only; `fit(y, init=[sensor, shading])`
for a `ShadingEstimate` with the dark and flat fields; a 3-D PSF and a
`Layout` with three spatial axes for a focal stack; a `zarr` array as input
and `out`, processed block by block. The function `ikoma(data, psf, ...)` is a
one-line shortcut. See the [API reference](../../reference/deconv.md).

The implementation is checked against the ADMM written out from the equations of
the SI on small problems, and on synthetic stacks from `optix.models` for the
orderings of the paper's Table 1 (shifted Poisson over Poisson over Gaussian,
and the Hessian penalty over the others), for the gain from modelling the
unknown boundary, and for the agreement of float32 and float64.

## References

!!! references "References"

    1. Ikoma H, Broxton M, Kudo T, Wetzstein G. *A convex 3D deconvolution
       algorithm for low photon count fluorescence imaging.* Sci Rep
       **8**:11489, 2018.
       [doi:10.1038/s41598-018-29768-x](https://doi.org/10.1038/s41598-018-29768-x)
    2. Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. *Distributed
       optimization and statistical learning via the alternating direction
       method of multipliers.* Found Trends Mach Learn **3**(1):1-122, 2011.
       [doi:10.1561/2200000016](https://doi.org/10.1561/2200000016)
    3. Foi A, Trimeche M, Katkovnik V, Egiazarian K. *Practical
       Poissonian-Gaussian noise modeling and fitting for single-image
       raw-data.* IEEE Trans Image Process **17**(10):1737-1754, 2008.
       [doi:10.1109/TIP.2008.2001399](https://doi.org/10.1109/TIP.2008.2001399)

    ??? quote "BibTeX"

        ```bibtex
        @article{ikoma2018convex,
          author  = {Ikoma, Hayato and Broxton, Michael and Kudo, Takamasa and Wetzstein, Gordon},
          title   = {A convex {3D} deconvolution algorithm for low photon count fluorescence imaging},
          journal = {Scientific Reports},
          volume  = {8},
          pages   = {11489},
          year    = {2018},
          doi     = {10.1038/s41598-018-29768-x},
        }
        @article{boyd2011distributed,
          author  = {Boyd, Stephen and Parikh, Neal and Chu, Eric and Peleato, Borja and Eckstein, Jonathan},
          title   = {Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers},
          journal = {Foundations and Trends in Machine Learning},
          volume  = {3},
          number  = {1},
          pages   = {1--122},
          year    = {2011},
          doi     = {10.1561/2200000016},
        }
        @article{foi2008practical,
          author  = {Foi, Alessandro and Trimeche, Mejdi and Katkovnik, Vladimir and Egiazarian, Karen},
          title   = {Practical {P}oissonian-{G}aussian noise modeling and fitting for single-image raw-data},
          journal = {IEEE Transactions on Image Processing},
          volume  = {17},
          number  = {10},
          pages   = {1737--1754},
          year    = {2008},
          doi     = {10.1109/TIP.2008.2001399},
        }
        ```
