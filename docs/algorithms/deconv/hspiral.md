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
of the Hessian matrix at every pixel ($2 \times 2$ for an image, $3 \times 3$ for a
focal stack), of order $p = 1$ (nuclear norm),
$p = 2$ (Frobenius norm, the penalty of Ikoma) or $p = \infty$ (spectral norm).
All three are functions of the two eigenvalues of the Hessian (the principal
curvatures) and are invariant to a rotation of the image; they differ in how they
combine them: their sum ($p = 1$), their root sum of squares ($p = 2$) or the larger
one ($p = \infty$). In the simulations of the paper the Hessian
Schatten penalties almost always beat total variation and wavelet penalties, with
$p = 1$ slightly ahead of $p = 2$; $p = \infty$ is behind the other two but still
ahead of the alternatives in most cases (Table I of the paper, which reports
HSPIRAL1, the first of its two algorithms, see [below](#inference-derivation)).

Use it for images with peaks of a few to a few hundred photoelectrons when the read
noise is negligible, or already accounted for, and an edge-preserving smooth prior
is wanted. When read noise matters, [Ikoma](ikoma.md) models it; for a
Poisson likelihood without a Hessian penalty see
[Richardson–Lucy](richardson-lucy.md) and
[Richardson–Lucy with TV](richardson-lucy-tv.md). `apply` returns the deconvolved
images. The paper gives two ADMM solvers for the same problem, selected with `variant`:
HSPIRAL1 (`variant=1`, the default) and HSPIRAL2 (`variant=2`). The ADMM solver, the
Hessian operator and the Schatten proximity operators are described in
[Optimisation tools](../../concepts/optimisation.md).

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
| $\mathbf{D}_{ab}$ | $\partial_{r_a r_b}$ | second difference along axes $a$, $b$ |
| $\tau$ | $\tau$ | weight of the penalty |
| $\rho$ | $\alpha$ | ADMM penalty parameter |
| $\mathbf{u}_k$ | $\mathbf{s}_k$ | scaled dual variables |
| $\iota_{\ge 0}$ | $\iota_C$ | indicator of the non-negative orthant |
| $\boldsymbol\Omega$, $\boldsymbol\Psi$, $t_n$ | $\boldsymbol\Omega$, $\boldsymbol\Psi$, $t_n$ | dual field, extrapolated dual field and momentum of the inner algorithm of HSPIRAL1 |
| $q$ | $q$ | conjugate order, $1/p + 1/q = 1$ |

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

The second differences are forward stencils (Section IV of the paper: "defined as in"
[Lefkimmiatis, Ward and Unser 2013](https://doi.org/10.1109/TIP.2013.2237919), whose Eq. 10
is given below). For instance, away from the border,
$f_{xx}[i, j] = f[i, j] - 2f[i+1, j] + f[i+2, j]$ and
$f_{xy}[i, j] = f[i, j] - f[i+1, j] - f[i, j+1] + f[i+1, j+1]$, the stencil of Eq. 21 of
[Lefkimmiatis, Bourquard and Unser (2012)](https://doi.org/10.1109/TIP.2011.2168232).
They differ at the last two rows and columns, where the boundary comes in, and the two
solvers use different boundaries.

- **HSPIRAL2** needs the Hessian to be circulant for its Fourier solve, so the stencil wraps
  around, $\mathbf{D}_{ab} = \mathbf{D}_a \mathbf{D}_b$ with periodic forward differences
  (`Hessian(boundary="periodic")`).
- **HSPIRAL1** applies the Hessian only inside its inner algorithm, which needs no Fourier
  solve of the Hessian, and uses the mirror stencil of Eq. 10 of the paper of
  Lefkimmiatis, Ward and Unser (`Hessian(boundary="mirror")`).

Along an axis of $N$ samples (1-based), the mirror stencil is:

=== "optix"

    $$
    (\mathbf{D}_{aa}\mathbf{x})_i =
    \begin{cases}
    x_{i+2} - 2x_{i+1} + x_i, & 1 \le i \le N - 2, \\
    x_{N-1} - x_N, & i \ge N - 1,
    \end{cases}
    \qquad
    (\mathbf{D}_{ab}\mathbf{x})_{i,j} =
    \begin{cases}
    x_{i+1,j+1} - x_{i+1,j} - x_{i,j+1} + x_{i,j}, & i < N_a,\ j < N_b, \\
    0, & \text{otherwise}.
    \end{cases}
    $$

    Equivalently, $\mathbf{x}$ is extended by half-sample symmetry,
    $x_{N+k} = x_{N+1-k}$, before the stencils are applied (the mixed terms are zero at the
    last index of either axis).

=== "Lefkimmiatis, Ward and Unser 2013"

    $$
    \partial_{r_1 r_1} x_{i,j} =
    \begin{cases}
    x_{i+2,j} - 2x_{i+1,j} + x_{i,j}, & 1 \le i \le N_x - 2, \\
    x_{N_x-1,j} - x_{N_x,j}, & i \ge N_x - 1,
    \end{cases}
    \quad (10\mathrm{a})
    $$

    and likewise $\partial_{r_2 r_2}$ along $j$ (10b), with $\partial_{r_1 r_2} x_{i,j}$ the
    mixed stencil for $1 \le i \le N_x - 1$ and $1 \le j \le N_y - 1$ and $0$ otherwise (10c).

The adjoint is that of Eq. 14 of that paper, $\mathbf{H}^*\mathbf{Y} = \sum_a \mathbf{D}_{aa}^*\mathbf{Y}^{(aa)}
+ \sum_{a<b} \mathbf{D}_{ab}^*(\mathbf{Y}^{(ab)} + \mathbf{Y}^{(ba)})$. `Hessian` packs the distinct
components of $\mathbf{X}_j$ as in [Ikoma](ikoma.md#probabilistic-model),

$$
(\mathbf{H}\mathbf{x})_j = \bigl( (\mathbf{D}_{11}\mathbf{x})_j,\; (\mathbf{D}_{22}\mathbf{x})_j,\;
\sqrt{2}\,(\mathbf{D}_{12}\mathbf{x})_j \bigr)
$$

in 2-D (three components; six in 3-D), an isometry for the Frobenius inner product, so that
$\lVert (\mathbf{H}\mathbf{x})_j \rVert_2 = \lVert \mathbf{X}_j \rVert_F$. With either stencil
$\lVert\mathbf{H}\rVert^2 \le (4d)^2$ for $d$ spatial axes (64 in 2-D, as in Proposition 2 of
that paper; 144 in 3-D). The penalty is the class `Schatten(tau, p)` of
[`optix.optim`](../../reference/optim.md) for HSPIRAL2, and `HessianSchatten(tau, p)`, which
adds the constraint $\mathbf{x} \ge 0$ and computes the prox iteratively, for HSPIRAL1; for
$p = 2$ the penalty equals the Hessian–Frobenius penalty of Ikoma, `GroupL2`.

## Inference / derivation

The problem is convex and is solved by ADMM
([Boyd et al. 2011](https://doi.org/10.1561/2200000016)) with scaled duals and a single
penalty $\rho$. Both variants of the paper update the splits first, from the current
$\mathbf{x}$ and duals, then $\mathbf{x}$, then the duals (`order="z-first"` of
[`admm_fft`](../../concepts/optimisation.md#admm-for-circulant-operators)), and differ in how
the problem is split.

### HSPIRAL1 (default)

HSPIRAL1 (Algorithm 1) has two splits, $\mathbf{z}_1 = \mathbf{A}\mathbf{x}$ (fidelity) and
$\mathbf{z}_2 = \mathbf{x}$, which carries the penalty and the non-negativity together,
$f_2(\mathbf{z}_2) = \tau \lVert \mathbf{H}\mathbf{z}_2 \rVert_{1,p} + \iota_{\ge 0}(\mathbf{z}_2)$.
It is the variant of the experiments of the paper (Section IV-B).

=== "optix"

    $$
    \begin{aligned}
    \mathbf{z}_1 &\leftarrow \mathrm{prox}_{\ell/\rho}(\mathbf{A}\mathbf{x} + \mathbf{u}_1) \\
    \mathbf{z}_2 &\leftarrow \mathrm{prox}_{f_2/\rho}(\mathbf{x} + \mathbf{u}_2) \\
    \mathbf{x} &\leftarrow (\mathbf{A}^\top\!\mathbf{A} + \mathbf{I})^{-1}
      \bigl[\mathbf{A}^\top(\mathbf{z}_1 - \mathbf{u}_1) + (\mathbf{z}_2 - \mathbf{u}_2)\bigr] \\
    \mathbf{u}_k &\leftarrow \mathbf{u}_k + \mathbf{A}_k\mathbf{x} - \mathbf{z}_k,
      \quad \mathbf{A}_1 = \mathbf{A},\ \mathbf{A}_2 = \mathbf{I}
    \end{aligned}
    $$

=== "Lefkimmiatis and Unser 2013"

    $$
    \begin{aligned}
    \mathbf{z}_1^{t+1} &= \text{Eq. (27) at } \mathbf{u}_1^t = \mathbf{K}\mathbf{x}^t + \mathbf{s}_1^t \\
    \mathbf{z}_2^{t+1} &= \arg\min_{\mathbf{x}\in C} \tfrac{1}{2}\lVert \mathbf{x} - \mathbf{u}_2^t \rVert^2
      + \tfrac{\tau}{\alpha} \lVert \mathcal{H}\mathbf{x} \rVert_{1,p},
      \quad \mathbf{u}_2^t = \mathbf{x}^t + \mathbf{s}_2^t
      && (28) \\
    \mathbf{x}^{t+1} &= (\mathbf{K}^T\mathbf{K} + \mathbf{I})^{-1}
      \bigl[\mathbf{K}^T(\mathbf{z}_1^{t+1} - \mathbf{s}_1^t) + \mathbf{z}_2^{t+1} - \mathbf{s}_2^t\bigr]
      && (29) \\
    \mathbf{s}_k^{t+1} &= \mathbf{s}_k^t + \mathbf{A}_k\mathbf{x}^{t+1} - \mathbf{z}_k^{t+1}
      && (23)
    \end{aligned}
    $$

$\mathbf{z}_1$ is the Poisson prox, as in HSPIRAL2 below, and the $\mathbf{x}$-update is one
division in the Fourier domain by $|\hat{A}|^2 + 1$. The prox of $f_2$ (Eq. 28) has no closed
form. The paper computes it approximately with 5 iterations of the dual algorithm of
[Lefkimmiatis, Ward and Unser (2013)](https://doi.org/10.1109/TIP.2013.2237919)
(Algorithm 2 there), a FISTA iteration
([Beck and Teboulle 2009](https://doi.org/10.1109/TIP.2009.2028250)) on the dual problem,
available as `prox_hessian_schatten` and, as a function with a prox, `HessianSchatten`. With
$\mathbf{v} = \mathbf{x} + \mathbf{u}_2$, $\tau' = \tau/\rho$ and $q$ the conjugate order
of $p$, $1/p + 1/q = 1$, it solves
$\min_{\mathbf{z} \ge 0} \tfrac12\lVert\mathbf{z} - \mathbf{v}\rVert^2 + \tau' \lVert\mathbf{H}\mathbf{z}\rVert_{1,p}$ by:

=== "optix"

    $$
    \begin{aligned}
    \boldsymbol\Omega_n &= P_{\mathcal{B}_q}\Bigl(\boldsymbol\Psi_n
      + \tfrac{1}{\tau' \lVert\mathbf{H}\rVert^2}\,\mathbf{H}\,P_{\ge 0}(\mathbf{v} - \tau'\mathbf{H}^*\boldsymbol\Psi_n)\Bigr) \\
    t_{n+1} &= \tfrac12\bigl(1 + \sqrt{1 + 4t_n^2}\bigr) \\
    \boldsymbol\Psi_{n+1} &= \boldsymbol\Omega_n + \tfrac{t_n - 1}{t_{n+1}}(\boldsymbol\Omega_n - \boldsymbol\Omega_{n-1})
    \end{aligned}
    $$

    and the result is $\mathbf{z}_2 = P_{\ge 0}(\mathbf{v} - \tau'\mathbf{H}^*\boldsymbol\Omega)$ at the last
    $\boldsymbol\Omega$. Here $P_{\ge 0}$ is the projection onto $\mathbf{x} \ge 0$, $P_{\mathcal{B}_q}$
    projects the matrix at every pixel onto the unit $S_q$ ball (`project_schatten`) and $\lVert\mathbf{H}\rVert^2$ is the bound 64 (2-D)
    or 144 (3-D).

=== "Lefkimmiatis, Ward and Unser 2013"

    $$
    \begin{aligned}
    \boldsymbol\Omega_n &= P_{B_{\infty,q}}\Bigl(\boldsymbol\Psi_n
      + \tfrac{1}{64\tau}\,\mathcal{H}\,P_C(\mathbf{z} - \tau\mathcal{H}^*\boldsymbol\Psi_n)\Bigr)
      && (\text{Alg. 2; Eqs. 30, 31}) \\
    \hat{\mathbf{x}} &= P_C(\mathbf{z} - \tau\mathcal{H}^*\hat{\boldsymbol\Omega}) && (29)
    \end{aligned}
    $$

    with the FISTA recursion for $t_n$ and $\boldsymbol\Psi_n$ of the algorithm, and
    $P_{B_{\infty,q}}$ the projection of every matrix onto the unit $S_q$ ball (Eqs. 33–36).

Here $\mathbf{z}$ and $\tau$ of the paper's denoising problem are $\mathbf{v}$ and $\tau'$. The
choices the paper does not specify are listed [below](#faithful-mode-and-extensions): the
dual $\boldsymbol\Omega$ is kept from one ADMM iteration to the next while the momentum
restarts ($t_1 = 1$, $\boldsymbol\Psi_1 = \boldsymbol\Omega_0$), and the inner prox is
therefore inexact and depends on the iterations before it.

### HSPIRAL2

HSPIRAL2 (Algorithm 2) has three splits,
$\mathbf{z}_1 = \mathbf{A}\mathbf{x}$ (fidelity), $\mathbf{z}_2 = \mathbf{H}\mathbf{x}$
(penalty) and $\mathbf{z}_3 = \mathbf{x}$ (non-negativity), tied to $\mathbf{x}$ by scaled duals
$\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$. Every prox is closed form, so there are no inner
iterations. Unlike [Ikoma](ikoma.md#inference-derivation) it updates the splits first:

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
  `prox_schatten`, applied to the packed components at every pixel.
- **$\mathbf{z}_3$.** Projection onto the non-negative orthant (Eq. 32).

The Schatten prox is the key result of the paper (Proposition 1, Eq. 36): it acts on
the singular values only. If $\mathbf{Y} = \mathbf{U}\,\mathrm{diag}(\boldsymbol{\sigma})\,\mathbf{V}^H$, then

$$
\mathrm{prox}_{\tau\lVert\cdot\rVert_{S_p}}(\mathbf{Y})
= \mathbf{U}\,\mathrm{diag}\bigl(\mathrm{prox}_{\tau\lVert\cdot\rVert_p}(\boldsymbol{\sigma})\bigr)\,\mathbf{V}^H ,
$$

with closed-form $\ell_p$ proxes for $p = 1, 2, \infty$ (Eqs. 43–47). For a
symmetric $2 \times 2$ or $3 \times 3$ matrix the decomposition is never computed; see
[Schatten norms](../../concepts/optimisation.md#schatten-norms). The same proposition, in the
paper of Lefkimmiatis, Ward and Unser (Eqs. 33–36), gives the projection onto the unit $S_q$
ball used by the inner algorithm of HSPIRAL1.

## Algorithm

```text
Input: data y (counts), PSF h, sensor and shading estimates (optional), tau (optional), p,
       variant (1 or 2), inner_iter (5)
Preprocess: ỹ = (y − d) / f / g, clipped at 0 (or y itself without estimates)
            h ← h / sum(h)
For each image:
    τ = tau, or κ / (q₉₉.₉(ỹ) − q₅₀(ỹ)) with κ = 1.6 (99.9th percentile minus median)
    ρ = 60 τ / max(ỹ)
    Initialise x = ỹ, all duals u_k = 0, on the grid of the block
    Repeat for k = 1 … 400:
        variant 1 (HSPIRAL1):
        1. z1 ← Poisson prox of A x + u1 at step 1/ρ                       (Eq. 27)
           z2 ← inner_iter iterations of the dual algorithm for the prox of
                (τ/ρ)‖H·‖₁,ₚ + ι_{≥0} at x + u2, started from the dual of
                the previous iteration, with mirror Hessian               (Eq. 28)
        2. x  ← Fourier solve with |Â|² + 1                                (Eq. 29)
        3. u_k ← u_k + A_k x − z_k, with A_1 = A, A_2 = I                  (Eq. 23)
        variant 2 (HSPIRAL2):
        1. z1 ← Poisson prox of A x + u1 at step 1/ρ                       (Eq. 27)
           z2 ← Schatten-p prox of H x + u2 at threshold τ/ρ               (Prop. 1)
           z3 ← max(x + u3, 0)                                             (Eq. 32)
        2. x  ← Fourier solve with |Â|² + ∑|Ĥ_c|² + 1                      (Eq. 42)
        3. u_k ← u_k + A_k x − z_k
        stop if ‖x⁽ᵏ⁺¹⁾ − x⁽ᵏ⁾‖₂ ≤ 10⁻⁵ ‖x⁽ᵏ⁾‖₂
Return x
```

The number of iterations, the last relative change and the last primal and dual
residuals of every block are recorded in `history.extras["iterations"]`,
`["change"]`, `["primal_residual"]` and `["dual_residual"]`. The values of $\tau$,
$\rho$, $\max \tilde{y}$ and the 99.9th percentile and median of $\tilde{y}$ used for each image
are the attributes `tau`, `rho`, `peak` and `robust_peak` after `fit`.

## Faithful mode and extensions

`faithful=True` (the default) reproduces the algorithms of the paper: $\rho = 60\tau / M$
(Sections IV-A and IV-C), the start from $\mathbf{x}^0 = \tilde{\mathbf{y}}$ with zero duals,
the splits-first order of the updates, the stopping rule of Section IV-B (a relative change
of $10^{-5}$, or 400 iterations), the periodic boundary of the blur, and the result
$\mathbf{x}^t$ of the last iteration (not projected onto $\mathbf{x} \ge 0$). The default
variant is HSPIRAL1, with 5 inner iterations, because the results of Section IV of the paper
(Table I and the figures) are those of HSPIRAL1; HSPIRAL2 is the other algorithm of the paper
and is equally faithful. The defaults of `faithful=False` are the same, except for the boundary, which is `"unknown"`.
The weight `tau` is tuned per image in the paper, the three values of `p` are all
the method of the paper, and so are the two variants, so any of them keeps the estimate faithful. `halo` and
`block` are memory settings and keep it faithful too, even when they split an
image. Any other option marks the estimate as not faithful.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `tau` | weight $\tau \ge 0$ (any value is faithful) | `None` (rule below, not in the paper) | `None` |
| `p` | order of the Schatten norm: 1, 2 or `inf` (any value is faithful) | 1 | 1 |
| `variant` | solver: 1 (HSPIRAL1) or 2 (HSPIRAL2); both are faithful | 1 | 1 |
| `inner_iter` | iterations of the inner prox of HSPIRAL1 per ADMM iteration; unused by HSPIRAL2 | 5 | 5 |
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

**Choices the paper does not specify.** These are made by the code, and flagged
as such in the docstrings:

- **Scale $M$.** The paper uses the maximum intensity of the ground-truth image in
  $\alpha = 60\tau/M$, which is unknown on real data; here $M = \max\tilde{y}$ per image.
- **Order $p$.** The paper compares $p = 1, 2, \infty$ without a default. The default is
  $p = 1$, which has the best pooled PSNR in its Table I.
- **Dimension.** The paper is 2-D. In 3-D both variants are unchanged, with the
  $3 \times 3$ Hessian (six components, $\lVert\mathbf{H}\rVert^2 \le 144$ in the inner
  algorithm of HSPIRAL1). The orders $p = 1$ and $p = \infty$ need the eigenvalues
  of $3 \times 3$ matrices, which `prox_schatten`, `project_schatten` and `schatten_norm`
  compute in closed form (not in the papers); $p = 2$ needs none.
- **Warm start of the inner algorithm.** The paper says only that the internal variables
  of the inner algorithm are initialised with those of the previous ADMM iteration
  (Section III-C). Here the dual $\boldsymbol\Omega$ is kept and the momentum is restarted
  at every iteration ($t_1 = 1$, $\boldsymbol\Psi_1 = \boldsymbol\Omega_0$). Each solve
  (each block of each image) starts from $\boldsymbol\Omega_0 = 0$.
- **Hessian boundary.** The paper points to the stencil of Eq. 10 of its companion paper
  for the Hessian, and does not say which boundary its experiments use. HSPIRAL1 uses that
  stencil (`boundary="mirror"`); HSPIRAL2 needs the periodic one for its Fourier solve.
  The two variants therefore differ in the Hessian at the last two rows and columns of the
  block, and minimise the same objective only up to that boundary.
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
  The constant $\kappa$ was calibrated with HSPIRAL2 for the rules of the paper ($\rho$, stop) on
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
  the data (8 dB on average). Pass `tau` to set it. It was not recalibrated for HSPIRAL1 (see
  [known limits](#known-limits)).
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
    wraps around too, so seams are worse than with `"unknown"`. With
    `boundary="unknown"` the mirror Hessian of HSPIRAL1 sits at the edge of the larger grid.

### Known limits

- **Slow outer convergence with 5 inner iterations.** The inner prox of HSPIRAL1 is
  inexact, and the outer iteration can need more iterations than HSPIRAL2 to meet the
  stopping rule. In the [example below](#usage) HSPIRAL1 stops at the cap of 400 iterations
  with a last relative change of $1.6 \times 10^{-5}$, where HSPIRAL2 stops after 270.
  Raising `inner_iter` (to 20 there, 144 iterations) or `max_iter` shortens or extends the
  run but marks the estimate as not faithful; `variant=2` has no inner iterations.
- **Linear ramps at the border.** The mirror stencil is not zero on a linear ramp at the
  last two samples of an axis, where $(\mathbf{D}_{aa}\mathbf{x})_i = x_{N-1} - x_N$ is minus
  the slope, while it vanishes inside. A ramp that reaches the border is therefore
  penalised there. The periodic stencil of HSPIRAL2 penalises the
  jump at the wrap-around in the same way.
- **Default $\tau$ calibrated with HSPIRAL2.** The constant $\kappa$ was found with HSPIRAL2
  on 2-D simulations. On one 3-D scene the default was about 4 times too large for HSPIRAL1,
  so check the result on a new kind of data, and pass `tau` to set it.

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

The solver holds about $21 + 3.5C$ buffers per voxel of the FFT grid for HSPIRAL2 and
$27 + 3C$ for HSPIRAL1 (which includes the dual of the inner algorithm), with $C$ the Hessian
components: 31.5 and 36 for the 2-D Hessian ($C = 3$), 42 and 45 for the 3-D one
($C = 6$), measured on the peak use of the solver. The grid of a block is the loaded block (core and halo) for
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

>>> # 2. Fit and deconvolve (HSPIRAL1, the default); HSPIRAL2 and
>>> # Richardson-Lucy for comparison.
>>> hspiral = HSPIRAL(psf).fit(counts)
>>> x_hs = hspiral.apply(counts)
>>> hspiral2 = HSPIRAL(psf, variant=2).fit(counts)
>>> x_hs2 = hspiral2.apply(counts)
>>> x_rl = RichardsonLucy(psf).fit(counts).apply(counts)

>>> # 3. The weights that were used, and the PSNR (peak M = 20) against the truth.
>>> round(float(hspiral.tau), 4), round(float(hspiral.rho), 4)
(0.1778, 0.8889)
>>> (
...     hspiral.history.extras["iterations"][0],
...     hspiral2.history.extras["iterations"][0],
... )
(400.0, 270.0)
>>> hspiral.is_faithful, hspiral2.is_faithful
(True, True)
>>> psnr = lambda e: round(
...     float(10 * torch.log10(x.max() ** 2 / ((e - x) ** 2).mean())), 2
... )
>>> [psnr(e) for e in (counts, x_hs, x_hs2, x_rl)]  # data, HSPIRAL1, 2, RL
[16.04, 16.52, 16.53, 16.26]

>>> # 4. The other orders of the norm, and the proximity operator on its own.
>>> HSPIRAL(psf, p=2).is_faithful, HSPIRAL(psf, boundary="unknown").is_faithful
(True, False)
>>> HSPIRAL(psf, inner_iter=2).is_faithful
False
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
`Layout` with three spatial axes for a focal stack (any `p`); `variant=2` for the solver without
inner iterations; a `zarr` array as input
and `out`, processed block by block. The function `hspiral(data, psf, ...)` is a
one-line shortcut. See the [API reference](../../reference/deconv.md).

The implementation is checked against the paper: `prox_schatten` and `project_schatten`
against a spectral decomposition for each $p$, in 2-D and 3-D; the mirror Hessian against
Eq. 10 written by indices, and its adjoint; the inner algorithm against Algorithm 2 of
Lefkimmiatis, Ward and Unser written out line by line, with its duality gap; Algorithms 1
and 2 written out line by line, with explicit FFTs, must give the same estimate as
`admm_fft` for each $p$; the same
minimiser as the $\mathbf{x}$-first ADMM of [Ikoma](ikoma.md); the optimality conditions of the solution;
that each Hessian Schatten norm at its best $\tau$ beats the data and at least matches total variation, as in Table I of the paper; the default
$\tau$ across intensity ranges; and 3-D, blocks, float32 and sensor units.

## References

!!! references "References"

    1. Lefkimmiatis S, Unser M. *Poisson image reconstruction with Hessian
       Schatten-norm regularization.* IEEE Trans Image Process
       **22**(11):4314-4327, 2013.
       [doi:10.1109/TIP.2013.2271852](https://doi.org/10.1109/TIP.2013.2271852)
    2. Lefkimmiatis S, Ward JP, Unser M. *Hessian Schatten-norm regularization for
       linear inverse problems.* IEEE Trans Image Process **22**(5):1873-1888, 2013.
       [doi:10.1109/TIP.2013.2237919](https://doi.org/10.1109/TIP.2013.2237919)
    3. Lefkimmiatis S, Bourquard A, Unser M. *Hessian-based norm regularization for
       image restoration with biomedical applications.* IEEE Trans Image Process
       **21**(3):983-995, 2012.
       [doi:10.1109/TIP.2011.2168232](https://doi.org/10.1109/TIP.2011.2168232)
    4. Beck A, Teboulle M. *Fast gradient-based algorithms for constrained total
       variation image denoising and deblurring problems.* IEEE Trans Image Process
       **18**(11):2419-2434, 2009.
       [doi:10.1109/TIP.2009.2028250](https://doi.org/10.1109/TIP.2009.2028250)
    5. Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. *Distributed optimization
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
        @article{lefkimmiatis2013hessian,
          author  = {Lefkimmiatis, Stamatios and Ward, John Paul and Unser, Michael},
          title   = {{Hessian} {Schatten}-Norm Regularization for Linear Inverse Problems},
          journal = {IEEE Transactions on Image Processing},
          volume  = {22},
          number  = {5},
          pages   = {1873--1888},
          year    = {2013},
          doi     = {10.1109/TIP.2013.2237919},
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
        @article{beck2009fast,
          author  = {Beck, Amir and Teboulle, Marc},
          title   = {Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems},
          journal = {IEEE Transactions on Image Processing},
          volume  = {18},
          number  = {11},
          pages   = {2419--2434},
          year    = {2009},
          doi     = {10.1109/TIP.2009.2028250},
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
