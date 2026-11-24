# HSPIRAL

!!! info "At a glance"
    **Estimates:** deconvolved image (the latent image $\mathbf{x}$, in photoelectrons, same shape as the data) \
    **Noise model:** Poisson \
    **Streaming:** halo \
    **Needs many frames:** no \
    **Reference:** Lefkimmiatis and Unser, IEEE Trans Image Process, 2013, [doi:10.1109/TIP.2013.2271852](https://doi.org/10.1109/TIP.2013.2271852)

## Overview

`HSPIRAL` deconvolves an image recorded at a low photon count: given the data
$\tilde{\mathbf{y}}$ and the point spread function (PSF) $\mathbf{h}$, it estimates
the latent image $\mathbf{x}$
([Lefkimmiatis and Unser 2013](https://doi.org/10.1109/TIP.2013.2271852)).
Like [Ikoma](ikoma.md) it uses a Poisson likelihood, a penalty on the Hessian of the
image and a non-negativity constraint, and it solves the problem by the alternating
direction method of multipliers (ADMM). It differs in two ways. The noise model is a
plain Poisson law, so read noise is ignored. And the penalty is a *Schatten norm*
of the $2 \times 2$ Hessian matrix at every pixel, of order $p = 1$ (nuclear norm),
$p = 2$ (Frobenius norm, the penalty of Ikoma) or $p = \infty$ (spectral norm).
All three are functions of the two eigenvalues of the Hessian (the principal
curvatures) and are invariant to a rotation of the image; they differ in how they
combine them: their sum ($p = 1$), their root sum of squares ($p = 2$) or the larger
one ($p = \infty$). In the simulations of the paper the Hessian
Schatten penalties almost always beat total variation and wavelet penalties, with
$p = 1$ slightly ahead of $p = 2$; $p = \infty$ is behind the other two but still
ahead of the alternatives in most cases (Table I of the paper, which reports the
first of its two algorithms, see [below](#faithful-mode-and-extensions)).

Use it for images with peaks of a few to a few hundred photoelectrons when the read
noise is negligible, or already accounted for, and an edge-preserving smooth prior
is wanted. When read noise matters, [Ikoma](ikoma.md) models it; for a
Poisson likelihood without a Hessian penalty see
[Richardson–Lucy](richardson-lucy.md) and
[Richardson–Lucy with TV](richardson-lucy-tv.md). `apply` returns the deconvolved
images. The ADMM solver, the Hessian operator and the Schatten proximity operators
are described in [Optimisation tools](../../concepts/optimisation.md).

## Image formation (optics)

The microscope blurs the latent image with a known, shift-invariant PSF, and the
detector counts photons (see [Image formation and noise](../../concepts/image-formation.md)).
The PSF has 2 or 3 dims, is centred at index `K // 2` and is normalised to unit sum.
The data are taken to be photon counts. With a
[`SensorEstimate`](../../reference/estimates.md) given to `fit` they are first
brought to electrons, $\tilde{\mathbf{y}} = (\mathbf{y} - \mathbf{o}) / g$; with a
[`ShadingEstimate`](../../reference/estimates.md) the dark field is subtracted and
the result divided by the flat field, rescaled to unit median, as for
[Ikoma](ikoma.md#image-formation-optics). Read noise is ignored, and negative values
of $\tilde{\mathbf{y}}$ are set to 0. Without an estimate the data are taken as
counts.

=== "optix"

    $$
    \tilde{y}_j \sim \mathrm{Poisson}\bigl((\mathbf{A}\mathbf{x})_j\bigr),
    \qquad \mathbf{x} \ge 0 ,
    $$

    with $\mathbf{A}$ the convolution with the PSF.

=== "Lefkimmiatis and Unser 2013"

    $$
    Y_m \sim \mathrm{Pois}\bigl(Y_m \mid [\mathbf{K}\mathbf{x}]_m\bigr)
    \quad (1\text{–}2),
    $$

    with $\mathbf{K}$ the blur matrix and $\mathbf{x} \in \mathbb{R}_+^N$.

| optix | [Lefkimmiatis and Unser 2013](https://doi.org/10.1109/TIP.2013.2271852) | Meaning |
|---|---|---|
| $\tilde{\mathbf{y}}$ | $\mathbf{y}$ | photon counts, $M$ values |
| $\mathbf{x}$ | $\mathbf{x}$ | latent image, $N$ values |
| $\mathbf{A}$, $\mathbf{h}$ | $\mathbf{K}$ | blur operator and its PSF |
| $\mathbf{H}$ | $\mathcal{H}$ | Hessian operator, see below |
| $\tau$ | $\tau$ | weight of the penalty |
| $\rho$ | $\alpha$ | ADMM penalty parameter |
| $\mathbf{u}_k$ | $\mathbf{s}_k$ | scaled dual variables |
| $\iota_{\ge 0}$ | $\iota_C$ | indicator of the non-negative orthant |

The paper works with counts and does not model a camera. The conversion of
the data to electrons and the neglect of read noise are the conventions of optix
(see [below](#faithful-mode-and-extensions)).

## Probabilistic model

The counts are independent Poisson variables, so the penalised maximum-likelihood
estimate minimises the negative log-likelihood plus a penalty on the Hessian
(Eq. 12 of the paper). The Hessian at pixel $j$ is the symmetric $2 \times 2$ matrix
$\mathbf{X}_j$ of the second differences of $\mathbf{x}$, and the penalty is the sum
over the pixels of its Schatten norm:

=== "optix"

    $$
    \min_{\mathbf{x}} \;
    \sum_{j} \Bigl[ (\mathbf{A}\mathbf{x})_j - \tilde{y}_j \log (\mathbf{A}\mathbf{x})_j \Bigr]
    + \tau \sum_{j} \bigl\lVert \mathbf{X}_j \bigr\rVert_{S_p}
    + \iota_{\ge 0}(\mathbf{x}),
    $$

    with $\mathbf{X}_j$ the Hessian matrix of $\mathbf{x}$ at pixel $j$, packed in
    $(\mathbf{H}\mathbf{x})_j$.

=== "Lefkimmiatis and Unser 2013"

    $$
    \min_{\mathbf{x}} \;
    \sum_{m=1}^{M} \Bigl( [\mathbf{K}\mathbf{x}]_m - y_m \log [\mathbf{K}\mathbf{x}]_m \Bigr)
    + \tau \lVert \mathcal{H}\mathbf{x} \rVert_{1,p} + \iota_C(\mathbf{x})
    \quad (12),
    $$

    with $\lVert \mathcal{H}\mathbf{x} \rVert_{1,p} = \sum_n \lVert [\mathcal{H}\mathbf{x}]_n \rVert_{S_p}$ (11).

The Schatten norm of order $p$ of a matrix is the $\ell_p$ norm of its singular
values (Eq. 5). The singular values of a symmetric matrix are the absolute values of
its eigenvalues $\mu_\pm$, so for the Hessian

$$
\lVert \mathbf{X}_j \rVert_{S_1} = |\mu_+| + |\mu_-|,
\qquad
\lVert \mathbf{X}_j \rVert_{S_2} = \lVert \mathbf{X}_j \rVert_F = \sqrt{\mu_+^2 + \mu_-^2},
\qquad
\lVert \mathbf{X}_j \rVert_{S_\infty} = \max(|\mu_+|, |\mu_-|).
$$

The second differences compose forward differences, $\mathbf{D}_{ab} = \mathbf{D}_a \mathbf{D}_b$,
which is the stencil of Eq. 21 of
[Lefkimmiatis, Bourquard and Unser (2012)](https://doi.org/10.1109/TIP.2011.2168232)
used by the paper (for instance
$f_{xx}[i, j] = f[i, j] - 2f[i+1, j] + f[i+2, j]$ and
$f_{xy}[i, j] = f[i, j] - f[i+1, j] - f[i, j+1] + f[i+1, j+1]$), with periodic
boundaries. `Hessian` packs the three distinct components of $\mathbf{X}_j$ as in
[Ikoma](ikoma.md#probabilistic-model),

$$
(\mathbf{H}\mathbf{x})_j = \bigl( (\mathbf{D}_{11}\mathbf{x})_j,\; (\mathbf{D}_{22}\mathbf{x})_j,\;
\sqrt{2}\,(\mathbf{D}_{12}\mathbf{x})_j \bigr),
$$

an isometry for the Frobenius inner product, so that $\lVert (\mathbf{H}\mathbf{x})_j \rVert_2 = \lVert \mathbf{X}_j \rVert_F$.
The penalty is the class `Schatten(tau, p)` of
[`optix.optim`](../../reference/optim.md); for $p = 2$ it equals the Hessian–Frobenius
penalty of Ikoma, `GroupL2`.

## Inference / derivation

The problem is convex and is solved by ADMM
([Boyd et al. 2011](https://doi.org/10.1561/2200000016)) with three splits,
$\mathbf{z}_1 = \mathbf{A}\mathbf{x}$ (fidelity), $\mathbf{z}_2 = \mathbf{H}\mathbf{x}$
(penalty) and $\mathbf{z}_3 = \mathbf{x}$ (non-negativity), tied to $\mathbf{x}$ by
scaled duals $\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$ and a single penalty $\rho$.
This is the second algorithm of the paper, HSPIRAL2 (Algorithm 2). Unlike
[Ikoma](ikoma.md#inference-derivation) it updates the splits first, from the current
$\mathbf{x}$ and duals, then $\mathbf{x}$, then the duals (`order="z-first"` of
[`admm_fft`](../../concepts/optimisation.md#admm-for-circulant-operators)):

=== "optix"

    $$
    \begin{aligned}
    \mathbf{z}_1 &\leftarrow \mathrm{prox}_{\ell/\rho}(\mathbf{A}\mathbf{x} + \mathbf{u}_1) \\
    \mathbf{z}_2 &\leftarrow \mathrm{prox}_{(\tau/\rho)\lVert\cdot\rVert_{S_p}}(\mathbf{H}\mathbf{x} + \mathbf{u}_2) \\
    \mathbf{z}_3 &\leftarrow \max(\mathbf{x} + \mathbf{u}_3, 0) \\
    \mathbf{x} &\leftarrow (\mathbf{A}^\top\!\mathbf{A} + \mathbf{H}^\top\!\mathbf{H} + \mathbf{I})^{-1}
      \bigl[\mathbf{A}^\top(\mathbf{z}_1 - \mathbf{u}_1) + \mathbf{H}^\top(\mathbf{z}_2 - \mathbf{u}_2) + (\mathbf{z}_3 - \mathbf{u}_3)\bigr] \\
    \mathbf{u}_k &\leftarrow \mathbf{u}_k + \mathbf{A}_k\mathbf{x} - \mathbf{z}_k,
      \quad \mathbf{A}_1 = \mathbf{A},\ \mathbf{A}_2 = \mathbf{H},\ \mathbf{A}_3 = \mathbf{I}
    \end{aligned}
    $$

=== "Lefkimmiatis and Unser 2013"

    $$
    \begin{aligned}
    \mathbf{z}_1^{t+1} &= \tfrac{1}{2}\Bigl(\mathbf{u}_1^t - \tfrac{1}{\alpha}
      + \sqrt{(\mathbf{u}_1^t - \tfrac{1}{\alpha})^2 + \tfrac{4\mathbf{y}}{\alpha}}\Bigr)
      && (27) \\
    \mathbf{z}_2^{t+1} &= \arg\min_{\mathbf{z}_2} \tfrac{1}{2}\lVert \mathbf{z}_2 - \mathbf{u}_2^t \rVert^2
      + \tfrac{\tau}{\alpha} \lVert \mathbf{z}_2 \rVert_{1,p}
      && (\text{Prop. 1}) \\
    \mathbf{z}_3^{t+1} &= \max(\mathbf{u}_3^t, 0) && (32) \\
    \mathbf{x}^{t+1} &= (\mathbf{K}^T\mathbf{K} + \mathcal{H}^*\mathcal{H} + \mathbf{I})^{-1}
      \bigl[\mathbf{K}^T(\mathbf{z}_1^{t+1} - \mathbf{s}_1^t) + \mathcal{H}^*(\mathbf{z}_2^{t+1} - \mathbf{s}_2^t)
      + (\mathbf{z}_3^{t+1} - \mathbf{s}_3^t)\bigr] && (42)
    \end{aligned}
    $$

    with $\mathbf{u}_1^t = \mathbf{K}\mathbf{x}^t + \mathbf{s}_1^t$,
    $\mathbf{u}_2^t = \mathcal{H}\mathbf{x}^t + \mathbf{s}_2^t$,
    $\mathbf{u}_3^t = \mathbf{x}^t + \mathbf{s}_3^t$, and
    $\mathbf{s}_k^{t+1} = \mathbf{s}_k^t + \mathbf{A}_k\mathbf{x}^{t+1} - \mathbf{z}_k^{t+1}$.

(The paper's $\mathbf{u}_k^t$ are the arguments of the proximity operators; they are
not the scaled duals, which are its $\mathbf{s}_k$.) All four updates are closed
form:

- **$\mathbf{x}$.** On the periodic grid $\mathbf{A}$, $\mathbf{H}$ and $\mathbf{I}$ are
  circulant, so the matrix to invert is diagonal in the Fourier domain, and the solve
  is one division by $|\hat{A}|^2 + \sum_c |\hat{H}_c|^2 + 1$ at each frequency.
- **$\mathbf{z}_1$.** The Poisson prox (Eq. 27), the positive root of a quadratic;
  it is the case $\tilde\sigma^2 = 0$ of the
  [shifted-Poisson prox](../../concepts/optimisation.md#the-shifted-poisson-prox),
  computed in the form that avoids cancellation.
- **$\mathbf{z}_2$.** The [Schatten prox](../../concepts/optimisation.md#schatten-norms)
  `prox_schatten`, applied to the three packed components at every pixel.
- **$\mathbf{z}_3$.** Projection onto the non-negative orthant (Eq. 32).

The Schatten prox is the key result of the paper (Proposition 1, Eq. 36): it acts on
the singular values only. If $\mathbf{Y} = \mathbf{U}\,\mathrm{diag}(\boldsymbol{\sigma})\,\mathbf{V}^H$, then

$$
\mathrm{prox}_{\tau\lVert\cdot\rVert_{S_p}}(\mathbf{Y})
= \mathbf{U}\,\mathrm{diag}\bigl(\mathrm{prox}_{\tau\lVert\cdot\rVert_p}(\boldsymbol{\sigma})\bigr)\,\mathbf{V}^H ,
$$

with closed-form $\ell_p$ proxes for $p = 1, 2, \infty$ (Eqs. 43–47). For a
symmetric $2 \times 2$ matrix the decomposition is never computed; see
[Schatten norms](../../concepts/optimisation.md#schatten-norms).

## Algorithm

```text
Input: data y (counts), PSF h, sensor and shading estimates (optional), tau (optional), p
Preprocess: ỹ = (y − d) / f / g, clipped at 0 (or y itself without estimates)
            h ← h / sum(h)
For each image:
    τ = tau, or κ / (q₉₉.₉(ỹ) − q₅₀(ỹ)) with κ = 1.6 (99.9th percentile minus median)
    ρ = 60 τ / max(ỹ)
    Initialise x = ỹ, u1 = u2 = u3 = 0, on the grid of the block
    Repeat for k = 1 … 400:
        1. z1 ← Poisson prox of A x + u1 at step 1/ρ                 (Eq. 27)
           z2 ← Schatten-p prox of H x + u2 at threshold τ/ρ          (Prop. 1)
           z3 ← max(x + u3, 0)                                        (Eq. 32)
        2. x  ← Fourier solve                                         (Eq. 42)
        3. u_k ← u_k + A_k x − z_k
        4. stop if ‖x⁽ᵏ⁺¹⁾ − x⁽ᵏ⁾‖₂ ≤ 10⁻⁵ ‖x⁽ᵏ⁾‖₂
Return x
```

The number of iterations, the last relative change and the last primal and dual
residuals of every block are recorded in `history.extras["iterations"]`,
`["change"]`, `["primal_residual"]` and `["dual_residual"]`. The values of $\tau$,
$\rho$, $\max \tilde{y}$ and the 99.9th percentile and median of $\tilde{y}$ used for each image
are the attributes `tau`, `rho`, `peak` and `robust_peak` after `fit`.

## Faithful mode and extensions

`faithful=True` (the default) reproduces HSPIRAL2: $\rho = 60\tau / M$, the start
from $\mathbf{x}^0 = \tilde{\mathbf{y}}$, the splits-first order of the updates, the
stopping rule of Section IV-B (a relative change of $10^{-5}$, or 400 iterations),
the periodic boundary, and the result $\mathbf{x}^t$ of the last iteration (not
projected onto $\mathbf{x} \ge 0$). The defaults of `faithful=False` are the same, except for the boundary, which is `"unknown"`.
The weight `tau` is tuned per image in the paper, and the three values of `p` are all
the method of the paper, so any of them keeps the estimate faithful. `halo` and
`block` are memory settings and keep it faithful too, even when they split an
image. Any other option marks the estimate as not faithful.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `tau` | weight $\tau \ge 0$ (any value is faithful) | `None` (rule below, not in the paper) | `None` |
| `p` | order of the Schatten norm: 1, 2 or `inf` (any value is faithful; 3-D needs 2) | 1 | 1 |
| `rho` | ADMM penalty $\rho > 0$ | `None` ($60\tau/M$) | `None` |
| `max_iter` | iteration cap | 400 | 400 |
| `tol` | relative-change tolerance; `None` runs `max_iter` iterations | $10^{-5}$ | $10^{-5}$ |
| `boundary` | `"periodic"`: blur and differences wrap around the block; `"unknown"`: $\mathbf{x}$ lives on a larger grid, extension | `"periodic"` | `"unknown"` |
| `halo` | halo width per spatial axis, at least the PSF size (memory setting, faithful) | `None` (twice the PSF size) | `None` |
| `block` | core block size per spatial axis (memory setting, faithful) | `None` (memory budget) | `None` |

**Extension: `boundary="unknown"`.** The paper assumes periodic (or mirror)
boundaries for the Fourier solve of Eq. 42 and does not say which its experiments
use. With `boundary="unknown"`, $\mathbf{x}$ lives on a larger grid (the observed
region plus the PSF size minus one, rounded up to a fast FFT size, as in
[Ikoma](ikoma.md#probabilistic-model)), the Poisson term sums over the observed
pixels only, and $\mathbf{x}^0$ is the data continued smoothly over the margin. It
avoids the wrap-around artefacts that the periodic model produces on real images.

**HSPIRAL1 is not implemented.** The paper has a second variant, HSPIRAL1
(Algorithm 1), which splits only $\mathbf{z}_1 = \mathbf{A}\mathbf{x}$ and
$\mathbf{z}_2 = \mathbf{x}$ and evaluates the prox of $\tau\lVert\mathcal{H}\cdot\rVert_{1,p} + \iota_C$
iteratively. Both variants minimise Eq. 12, and the paper reports the results of
HSPIRAL1 in its tables; HSPIRAL2 is the variant with a closed-form prox for every
split and no inner iterations.

**Choices the paper does not specify.** These are made by the code, and flagged
as such in the docstrings:

- **Scale $M$.** The paper uses the maximum intensity of the ground-truth image in
  $\alpha = 60\tau/M$, which is unknown on real data; here $M = \max\tilde{y}$ per image.
- **Order $p$.** The paper compares $p = 1, 2, \infty$ without a default. The default is
  $p = 1$, which has the best pooled PSNR in its Table I.
- **Dimension.** The paper is 2-D. In 3-D, $p = 2$ works unchanged (the Frobenius
  norm of the $3 \times 3$ Hessian needs no eigenvalues); $p = 1$ and $p = \infty$ raise
  `NotImplementedError`.
- **Default $\tau$.** The paper tunes $\tau$ per image for the best PSNR and gives no
  default. The default is $\tau = \kappa / (q_{99.9}(\tilde{y}) - q_{50}(\tilde{y}))$ with
  $\kappa = 1.6$, per image, where $q_{99.9}$ is the 99.9th percentile of the prepared data
  (computed exactly in the pass of `fit`, stored as `robust_peak`: unlike the maximum, a
  hot pixel barely changes it) and $q_{50}$ their median (from a uniform sample of 8192
  pixels per image taken in the same pass, exact for smaller images, stored as `median`).
  The difference is a robust range of the signal above the background: the Hessian ignores
  a constant, so it is the signal amplitude, not the intensity, that sets the scale of the
  prior. The $1/s$ scaling, with $s$ this range, is that of a MAP estimate whose prior on
  the normalised image $\mathbf{x}/s$ is the same at every photon count. The Poisson
  likelihood is homogeneous,
  $\ell(s\hat{\mathbf{x}}; s\hat{\mathbf{y}}) = s\,\ell(\hat{\mathbf{x}}; \hat{\mathbf{y}})$ up to a constant, so
  the objective is $\ell(\mathbf{x}) + (\kappa/s)\lVert \mathbf{H}\mathbf{x} \rVert$ and $\tau = \kappa/s$.
  `fit` raises an error when $q_{99.9} \le q_{50}$; pass `tau` then.
  The constant $\kappa$ was calibrated for the rules of the paper ($\rho$, stop) on
  simulated images with circular blur (a Gaussian of standard deviation 2 on
  $9 \times 9$ pixels, and a uniform $7 \times 7$ blur): four scene types (blobs,
  filaments, their mixture and piecewise-smooth scenes), peaks of 10, 100 and 1000
  photons on a constant background of 0, 1, 3 and 10 times the peak, $64 \times 64$ (two
  seeds) and $128 \times 128$ (one seed) pixels, so 288 cases, against a grid of $\tau$
  (factor $\sqrt{2}$, searched with $p = 1$). Of the scales $q_{99.9}$,
  $q_{99.9} - q_1$ and $q_{99.9} - q_{50}$, the last gives the smallest mean and worst-case
  loss; with $q_{99.9}$ alone the best $\kappa$ grows with the background (about 18 times
  from no background to 10 times the peak), and the loss reaches 19.7 dB. For smooth
  scenes the best $\kappa$ stays at about 1.3 to 2.9 at every photon count; for thin
  filaments it falls by up to 5 times between 10 and 1000 photons. With $\kappa = 1.6$ the
  PSNR is on average 0.41, 0.39 and 0.47 dB below that of the best $\tau$ of the grid, and
  at most 2.4, 2.7 and 3.2 dB, for $p = 1, 2, \infty$; it is at least 1.3 dB above that of
  the data (8 dB on average). Pass `tau` to set it.
- **Camera.** The conversion to electrons with the `SensorEstimate` and the
  `ShadingEstimate`, the neglect of read noise and the clipping of negative values at 0
  are not in the paper, which takes photon counts. The dark field replaces the offset
  when a shading estimate is given, and a per-frame baseline of the shading estimate is
  ignored, with a warning.
- **Halo.** Twice the PSF size.

!!! warning "Periodic boundary"
    With the faithful boundary the blur and the differences wrap around the image, so a
    structure near one edge can leak to the opposite edge. This is the model of the
    paper and is exact for data simulated with a circular blur; for real images prefer
    `boundary="unknown"` (the default of `faithful=False`), at the price of a larger grid. Each block of a split image
    wraps around too, so seams are worse than with `"unknown"`.

## Memory and streaming

Large images are cut into blocks with a halo and each block is solved as an image of
its own, as for [Ikoma](ikoma.md#memory-and-streaming) and
[Richardson–Lucy](richardson-lucy.md#memory-and-streaming) (`blocks`, halo streaming).
`fit` makes one pass over the data to find, for every image, $\max\tilde{y}$, the
99.9th percentile (exactly, by keeping the largest 0.1 % of its values) and the median (from
a uniform sample of 8192 pixels), from which
$\rho$ and $\tau$ follow, so that every block of an image shares them. `apply` makes a
second pass. With the periodic boundary a block needs to be at least as large as the PSF
along every axis.

The solver holds about $21 + 3.5C$ buffers per voxel of the FFT grid for $C$ Hessian
components (31.5 for the 2-D Hessian, $C = 3$; 42 for the 3-D one, $C = 6$), measured on
the peak use of the solver. The grid of a block is the loaded block (core and halo) for
the periodic boundary, and that block plus the PSF, rounded up, for
`boundary="unknown"`. The block size is chosen from the memory budget
(`optix.config(memory=...)`), or set with `block`; the rules of
[Ikoma](ikoma.md#memory-and-streaming) for a budget that leaves a small core apply.

The $\mathbf{x}$-update is a global solve, so the block-wise result differs from the
whole-image one near the seams, and the difference fades with the halo width. A
`RuntimeWarning` says when an image is split.

## Usage

The example blurs a test pattern with a Gaussian PSF from
[`optix.models`](../../reference/models.md), draws Poisson counts and deconvolves them,
with the Richardson–Lucy estimate for comparison.

```pycon
>>> import torch
>>> from optix.deconv import HSPIRAL, RichardsonLucy
>>> from optix.models import gaussian_psf, simulate_blur
>>> from optix.optim import prox_schatten, schatten_norm
>>> _ = torch.manual_seed(0)

>>> # 1. A Gaussian PSF and Poisson counts of a test pattern.
>>> psf = gaussian_psf((9, 9), sigma=2.0)
>>> x = torch.full((64, 64), 1.0)
>>> x[16:48:8, 16:48] = 20.0
>>> counts = torch.poisson(simulate_blur(x, psf))

>>> # 2. Fit and deconvolve; Richardson-Lucy for comparison.
>>> hspiral = HSPIRAL(psf).fit(counts)
>>> x_hs = hspiral.apply(counts)
>>> x_rl = RichardsonLucy(psf).fit(counts).apply(counts)

>>> # 3. The weights that were used, and the PSNR (peak M = 20) against the truth.
>>> round(float(hspiral.tau), 4), round(float(hspiral.rho), 4)
(0.1778, 0.8889)
>>> hspiral.history.extras["iterations"][0] <= 400, hspiral.is_faithful
(True, True)
>>> psnr = lambda e: round(
...     float(10 * torch.log10(x.max() ** 2 / ((e - x) ** 2).mean())), 2
... )
>>> [psnr(e) for e in (counts, x_hs, x_rl)]  # data, HSPIRAL, Richardson-Lucy
[16.04, 16.53, 16.26]

>>> # 4. The other orders of the norm, and the proximity operator on its own.
>>> HSPIRAL(psf, p=2).is_faithful, HSPIRAL(psf, boundary="unknown").is_faithful
(True, False)
>>> v = torch.tensor([[3.0], [-1.0], [0.0]])  # diag(3, -1), packed
>>> [float(schatten_norm(v, p)) for p in (1, 2, float("inf"))]
[4.0, 3.1622776985168457, 3.0]
>>> prox_schatten(v, 0.5, 1).T
tensor([[ 2.5000, -0.5000,  0.0000]])
```

Common variations: `HSPIRAL(psf, tau=0.05)` to set the weight, then `fit` again to
see its effect; `p=2` or `p=float("inf")` for the other norms; `boundary="unknown"` for
images that are not periodic; `fit(y, init=sensor)` to convert ADU to electrons with a
`SensorEstimate`, or `init=[sensor, shading]` with a `ShadingEstimate`; a 3-D PSF and a
`Layout` with three spatial axes for a focal stack (with `p=2`); a `zarr` array as input
and `out`, processed block by block. The function `hspiral(data, psf, ...)` is a
one-line shortcut. See the [API reference](../../reference/deconv.md).

The implementation is checked against the paper: `prox_schatten` against a
singular-value decomposition for each $p$; Algorithm 2 written out line by line, with
explicit FFTs, must give the same estimate as `admm_fft` for each $p$; the same
minimiser as the $\mathbf{x}$-first ADMM of [Ikoma](ikoma.md); the optimality conditions of the solution;
that each Hessian Schatten norm at its best $\tau$ beats the data and at least matches total variation, as in Table I of the paper; the default
$\tau$ across intensity ranges; and 3-D, blocks, float32 and sensor units.

## References

!!! references "References"

    1. Lefkimmiatis S, Unser M. *Poisson image reconstruction with Hessian
       Schatten-norm regularization.* IEEE Trans Image Process
       **22**(11):4314-4327, 2013.
       [doi:10.1109/TIP.2013.2271852](https://doi.org/10.1109/TIP.2013.2271852)
    2. Lefkimmiatis S, Bourquard A, Unser M. *Hessian-based norm regularization for
       image restoration with biomedical applications.* IEEE Trans Image Process
       **21**(3):983-995, 2012.
       [doi:10.1109/TIP.2011.2168232](https://doi.org/10.1109/TIP.2011.2168232)
    3. Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. *Distributed optimization
       and statistical learning via the alternating direction method of
       multipliers.* Found Trends Mach Learn **3**(1):1-122, 2011.
       [doi:10.1561/2200000016](https://doi.org/10.1561/2200000016)

    ??? quote "BibTeX"

        ```bibtex
        @article{lefkimmiatis2013poisson,
          author  = {Lefkimmiatis, Stamatios and Unser, Michael},
          title   = {Poisson Image Reconstruction With {Hessian} {Schatten}-Norm Regularization},
          journal = {IEEE Transactions on Image Processing},
          volume  = {22},
          number  = {11},
          pages   = {4314--4327},
          year    = {2013},
          doi     = {10.1109/TIP.2013.2271852},
        }
        @article{lefkimmiatis2012hessian,
          author  = {Lefkimmiatis, Stamatios and Bourquard, Aur{\'e}lien and Unser, Michael},
          title   = {Hessian-Based Norm Regularization for Image Restoration With Biomedical Applications},
          journal = {IEEE Transactions on Image Processing},
          volume  = {21},
          number  = {3},
          pages   = {983--995},
          year    = {2012},
          doi     = {10.1109/TIP.2011.2168232},
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
        ```
