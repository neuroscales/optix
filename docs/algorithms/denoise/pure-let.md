# PURE-LET

!!! info "At a glance"
    - **Estimates:** denoised image (or volume), same shape and units as the data; `estimates["sensor"]` holds the noise parameters used
    - **Noise model:** Poisson–Gaussian (gain, offset and read noise known)
    - **Streaming:** halo
    - **Needs many frames:** no
    - **Reference:** Luisier, Blu & Unser, IEEE Trans Image Process, 2011, [doi:10.1109/TIP.2010.2073477](https://doi.org/10.1109/TIP.2010.2073477)

## Overview

`PURELET` removes noise from camera images, and from volumes, when the noise is
the mixture of photon (Poisson) noise and read (Gaussian) noise of a camera
([Luisier et al. 2011](https://doi.org/10.1109/TIP.2010.2073477)). The image
is analysed by an *undecimated* Haar wavelet transform. In each band, the
coefficients are replaced by a weighted sum of two functions of themselves: the
coefficient as it is, and the coefficient damped by a threshold that grows with
the local signal (photon noise is stronger where the signal is). The weights of
the sum are the unknowns. They are found by minimising PURE (Poisson's unbiased
risk estimate), an estimate of the mean squared error against the *unknown*
noise-free image that is computed from the noisy image alone. Because the
weights enter PURE quadratically, the minimisation is one small linear system
per image: no iterations, no step size, no tuning of a threshold.

Use it when the data are Poisson–Gaussian, in particular at low counts (a few
to a hundred photoelectrons per pixel), and the gain, the offset and the read
noise of the camera are known (see the
[Poisson-Gaussian fit](../sensor/poisson-gaussian.md) and
[sCMOS calibration](../sensor/scmos-calibration.md) for ways of finding them). It
is fast, a fraction of a second for a $512^2$ image on one CPU thread, and
treats the Poisson statistics directly, without the variance-stabilising
transform that [BM3D and BM4D](bm3d.md) need for such data. The Haar transform
suits piecewise-smooth images; the paper reports that an overcomplete
block-DCT dictionary does better on textured images at high counts, and that
dictionary is not implemented. It is not a deconvolution
(see [Richardson–Lucy](../deconv/richardson-lucy.md)), the noise must be
independent between pixels, and the result is a smoothed estimate of the
intensity, not a sharpened one.

The estimator is [`PURELET`][optix.denoise.PURELET], with the function
[`pure_let`][optix.denoise.pure_let] as a shortcut and
[`pure_let_halo`][optix.denoise.pure_let_halo] for the halo of the streaming (see
the [API reference](../../reference/denoise.md)). `fit` fixes the noise model,
`apply` denoises each image, and large arrays are streamed in two passes over
blocks with a halo, giving the result of the whole-image computation.

## Image formation (optics)

Only the noise is modelled; the optics are not (see
[Image formation and noise](../../concepts/image-formation.md)). The recorded
image is the camera model of that page. The method works on the data in
photoelectrons, $\tilde{\mathbf{y}} = (\mathbf{y} - o)/g$, where the noise is the
sum of a Poisson variable of mean $\lambda_i$ and a Gaussian of variance
$\tilde\sigma^2 = \sigma^2/g^2$:

=== "optix"

    $$
    \mathbf{y} = g\,\mathbf{e} + o\,\mathbf{1} + \boldsymbol{\varepsilon},
    \qquad
    \tilde{\mathbf{y}} = \frac{\mathbf{y} - o\,\mathbf{1}}{g} = \mathbf{e} + \tilde{\boldsymbol{\varepsilon}},
    \qquad
    \begin{cases}
    e_i \sim \mathrm{Poisson}(\lambda_i) \\
    \tilde{\boldsymbol{\varepsilon}} \sim \mathcal{N}(\mathbf{0}, \tilde\sigma^2 \mathbf{I})
    \end{cases}
    $$

=== "Luisier et al. 2011"

    $$
    \mathbf{y} = \mathbf{z} + \mathbf{b},
    \qquad
    \begin{cases}
    z_n \sim \mathcal{P}(x_n) \\
    \mathbf{b} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
    \end{cases}
    \qquad
    \text{raw data} = \delta\,\mathbf{1} + \alpha\,\mathbf{y}
    $$

| optix | Luisier et al. | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}$ | $\delta\mathbf{1} + \alpha\mathbf{y}$ | recorded image, in ADU | $\lvert\Omega\rvert$, per image |
| $\tilde{\mathbf{y}}$ | $\mathbf{y}$ | normalised data, in photoelectrons | $\lvert\Omega\rvert$, per image |
| $\boldsymbol{\lambda}$ | $\mathbf{x}$ | noise-free intensity: expected photoelectrons per pixel, a fixed (not random) vector | $\lvert\Omega\rvert$, per image |
| $\mathbf{e}$ | $\mathbf{z}$ | photoelectron counts | $\lvert\Omega\rvert$, per image |
| $\tilde{\boldsymbol{\varepsilon}}$ | $\mathbf{b}$ | read noise, divided by the gain | $\lvert\Omega\rvert$, per image |
| $g$ | $\alpha$ | gain, ADU per photoelectron | scalar (a map with `pixelwise=True`) |
| $o$ | $\delta$ | offset, ADU | scalar (a map with `pixelwise=True`) |
| $\sigma^2$ | $\alpha^2\sigma^2$ | read-noise variance, ADU² | scalar (a map with `pixelwise=True`) |
| $\tilde\sigma^2$ | $\sigma^2$ | read-noise variance, in photoelectrons² | scalar (a map with `pixelwise=True`) |
| $\lvert\Omega\rvert$ | $N$ | number of pixels (voxels) of an image | |

The paper's $\sigma^2$ is the variance in the normalised units, $\tilde\sigma^2$
here. The mean and the variance of $\tilde y_i$ are $\lambda_i$ and
$\lambda_i + \tilde\sigma^2$: the noise depends on the signal. The paper
([Luisier et al. 2011, §V](https://doi.org/10.1109/TIP.2010.2073477))
subtracts the offset, divides by the gain, and takes $\alpha$, $\delta$ and
$\sigma^2$ from a robust regression of local variances on local means; optix
takes them from the arguments `gain`, `offset` and `sigma`, or from a
[`SensorEstimate`](../../reference/estimates.md). Clipping and quantisation are
not modelled, and negative values of $\tilde{\mathbf{y}}$ are allowed. Images of
a stack, and of different groups, are denoised one by one.

## Probabilistic model

PURE-LET is not derived from a prior: the intensity $\boldsymbol\lambda$ is
treated as deterministic, and the method is built on an *unbiased estimate of
the mean squared error*,

$$
\mathrm{MSE} = \frac{1}{\lvert\Omega\rvert}\,\lVert \mathcal{F}(\tilde{\mathbf{y}}) - \boldsymbol{\lambda} \rVert^2
$$

of an estimator $\hat{\boldsymbol\lambda} = \mathcal{F}(\tilde{\mathbf{y}})$ ($\mathbf{f}$ in the paper;
the calligraphic letter avoids the flat field $\mathbf{f}$ of the other pages).
It rests on two identities that replace the unknown $\boldsymbol\lambda$ by
the data. For $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, \tilde\sigma^2\mathbf{I})$ independent of
$\mathbf{e}$, *Stein's lemma*,
$\mathbb{E}\{\mathbf{u}^\top \mathcal{F}(\mathbf{e} + \mathbf{u})\} = \tilde\sigma^2\,\mathbb{E}\{\operatorname{div}\mathcal{F}\}$,
and its Poisson counterpart,
$\mathbb{E}\{\boldsymbol\lambda^\top \mathcal{F}(\mathbf{e})\} = \mathbb{E}\{\mathbf{e}^\top \mathcal{F}^{-}(\mathbf{e})\}$
([Luisier et al. 2011, §II-A](https://doi.org/10.1109/TIP.2010.2073477)). With
$\mathcal{F}^{-}(\tilde{\mathbf{y}}) = [\mathcal{F}_i(\tilde{\mathbf{y}} - \mathbf{1}_i)]_{i}$, where $\mathbf{1}_i$ is the
vector with 1 at pixel $i$ and 0 elsewhere, they give *PURE*
([Luisier et al. 2011, Theorem 1, Eq. 2](https://doi.org/10.1109/TIP.2010.2073477)):

=== "optix"

    $$
    \hat\epsilon =
    \frac{1}{\lvert\Omega\rvert}\Bigl(
    \lVert \mathcal{F}(\tilde{\mathbf{y}}) \rVert^2
    - 2\,\tilde{\mathbf{y}}^\top \mathcal{F}^{-}(\tilde{\mathbf{y}})
    + 2\tilde\sigma^2 \operatorname{div}\mathcal{F}^{-}(\tilde{\mathbf{y}})
    \Bigr)
    + \frac{1}{\lvert\Omega\rvert}\bigl(\lVert \tilde{\mathbf{y}} \rVert^2 - \mathbf{1}^\top\tilde{\mathbf{y}}\bigr)
    - \tilde\sigma^2,
    \qquad
    \mathbb{E}\{\hat\epsilon\} = \mathbb{E}\{\mathrm{MSE}\}
    $$

=== "Luisier et al. 2011"

    $$
    \epsilon =
    \frac{1}{N}\Bigl(
    \lVert \mathbf{f}(\mathbf{y}) \rVert^2
    - 2\,\mathbf{y}^\top \mathbf{f}^{-}(\mathbf{y})
    + 2\sigma^2 \operatorname{div}\mathbf{f}^{-}(\mathbf{y})
    \Bigr)
    + \frac{1}{N}\bigl(\lVert \mathbf{y} \rVert^2 - \mathbf{1}^\top\mathbf{y}\bigr)
    - \sigma^2,
    \qquad
    \mathbb{E}_{\mathbf{y}}\{\epsilon\} = \frac{1}{N}\mathbb{E}_{\mathbf{y}}\{\lVert \mathbf{f}(\mathbf{y}) - \mathbf{x} \rVert^2\}
    $$

($\mathbf{f}^{-}(\mathbf{y}) = [f_n(\mathbf{y} - \mathbf{e}_n)]_n$, with $\mathbf{e}_n$ the canonical basis;
the paper's $\mathbf{e}_n$ is not the photoelectron count $\mathbf{e}$.) PURE needs the
function at $N$ shifted inputs, which is impractical for a denoiser. The paper
replaces $\mathcal{F}^{-}(\tilde{\mathbf{y}})$ by its first-order Taylor expansion,
$\mathcal{F}(\tilde{\mathbf{y}}) - \partial\mathcal{F}(\tilde{\mathbf{y}})$, with
$\partial\mathcal{F} = [\partial\mathcal{F}_i/\partial \tilde y_i]_i$
([Luisier et al. 2011, §II-C, Eq. 6](https://doi.org/10.1109/TIP.2010.2073477)):

=== "optix"

    $$
    \hat\epsilon =
    \frac{1}{\lvert\Omega\rvert}\Bigl(
    \lVert \mathcal{F} \rVert^2
    - 2\,\tilde{\mathbf{y}}^\top (\mathcal{F} - \partial\mathcal{F})
    + 2\tilde\sigma^2 \operatorname{div}\{\mathcal{F} - \partial\mathcal{F}\}
    \Bigr)
    + \frac{1}{\lvert\Omega\rvert}\bigl(\lVert \tilde{\mathbf{y}} \rVert^2 - \mathbf{1}^\top\tilde{\mathbf{y}}\bigr)
    - \tilde\sigma^2
    $$

=== "Luisier et al. 2011"

    $$
    \hat\epsilon =
    \frac{1}{N}\Bigl(
    \lVert \mathbf{f}(\mathbf{y}) \rVert^2
    - 2\,\mathbf{y}^\top (\mathbf{f}(\mathbf{y}) - \partial\mathbf{f}(\mathbf{y}))
    \Bigr)
    + \frac{1}{N}\Bigl(2\sigma^2 \operatorname{div}\{\mathbf{f}(\mathbf{y}) - \partial\mathbf{f}(\mathbf{y})\} + \lVert \mathbf{y} \rVert^2 - \mathbf{1}^\top\mathbf{y}\Bigr)
    - \sigma^2
    $$

The two estimates coincide when $\mathcal{F}$ is linear. This $\hat\epsilon$ is what
`PURELET` minimises (it is called "PURE" below), and
[`PureSums.risk`][optix.denoise.utils.PureSums.risk] evaluates it. It needs only
$\mathcal{F}$, its diagonal first and second derivatives and the data, and it is
accurate when each $\mathcal{F}_i$ varies slowly with $\tilde y_i$ (the reliability
rule of the [next section](#the-reliability-rule) controls where this fails).

## Inference / derivation

The strategy is *unbiased risk estimation*: PURE is a Poisson–Gaussian
counterpart of Stein's unbiased risk estimate (SURE), and minimising it over
the weights of a linear combination of fixed functions is a linear
least-squares problem, which is solved exactly. There is no likelihood and no
prior, and no iteration.

### Transform-domain processing and the LET

The estimator analyses the data by a linear transform $\mathbf{D}$, processes
each coefficient *pointwise* in a nonlinear way, and reconstructs by a linear
$\mathbf{R}$ with $\mathbf{R}\mathbf{D} = \mathbf{I}$
([Luisier et al. 2011, Eq. 7](https://doi.org/10.1109/TIP.2010.2073477)):

=== "optix"

    $$
    \mathcal{F}(\tilde{\mathbf{y}}) = \mathbf{R}\,\boldsymbol{\Theta}(\mathbf{w}, \bar{\mathbf{w}}),
    \qquad
    \mathbf{w} = \mathbf{D}\tilde{\mathbf{y}},
    \quad
    \bar{\mathbf{w}} = \bar{\mathbf{D}}\tilde{\mathbf{y}}
    $$

=== "Luisier et al. 2011"

    $$
    \hat{\mathbf{x}} = \mathbf{f}(\mathbf{y}) = \mathbf{R}\,\boldsymbol{\Theta}(\mathbf{D}\mathbf{y}, \bar{\mathbf{D}}\mathbf{y})
    $$

The coefficients $\mathbf{w}$ are processed with the help of a coarse map
$\bar{\mathbf{w}}$ of the signal, obtained by smoothing, from which the
signal-dependent noise variance of each coefficient is estimated. The
*linear expansion of thresholds* (LET) makes $\boldsymbol\Theta$ a linear
combination of $K$ fixed functions, with unknown weights $a_k$
([Luisier et al. 2011, Eqs. 14–15](https://doi.org/10.1109/TIP.2010.2073477)).
PURE is then quadratic in $\mathbf{a}$, and its minimiser solves a linear system:

=== "optix"

    $$
    \mathcal{F}(\tilde{\mathbf{y}}) = \sum_{k=1}^{K} a_k\,\mathcal{F}_k(\tilde{\mathbf{y}}),
    \qquad
    \mathbf{G}\,\mathbf{a} = \hat{\mathbf{c}},
    \qquad
    G_{kl} = \mathcal{F}_k^\top \mathcal{F}_l
    $$

=== "Luisier et al. 2011"

    $$
    \mathbf{f}(\mathbf{y}) = \sum_{k=1}^{K} a_k\,\underbrace{\mathbf{R}\boldsymbol{\Theta}_k(\mathbf{w}, \bar{\mathbf{w}})}_{\mathbf{f}_k(\mathbf{y})},
    \qquad
    \mathbf{M}\,\mathbf{a} = \hat{\mathbf{c}},
    \qquad
    [\mathbf{M}]_{k,l} = \mathbf{f}_k(\mathbf{y})^\top \mathbf{f}_l(\mathbf{y})
    $$

The paper's matrix $\mathbf{M}$ is called $\mathbf{G}$ (Gram matrix) here, since
$M_j$ is the redundancy of the transform (below). The right-hand side is given
after the transform has been chosen.

### The undecimated Haar filterbank

The transform is the $J$-level undecimated (à trous) Haar wavelet transform of
[Luisier et al. (2011, §III, Fig. 3)](https://doi.org/10.1109/TIP.2010.2073477), with
the unit-norm filters

$$
H(z) = \frac{1 + z^{-1}}{\sqrt{2}},
\qquad
G(z) = \frac{1 - z^{-1}}{\sqrt{2}}.
$$

Unit-norm filters keep white noise of variance $\tilde\sigma^2$ at variance
$\tilde\sigma^2$ in every band. At level $j = 1, \ldots, J$, the scaling and
wavelet filters along one axis are (up to the direction of the shifts)

$$
\bar H_j(z) = H(z)\,H(z^2)\cdots H(z^{2^{j-1}}),
\qquad
\tilde G_j(z) = H(z)\,H(z^2)\cdots H(z^{2^{j-2}})\,G(z^{2^{j-1}}),
$$

of length $2^j$. They are applied along each of the $d$ axes of the image
($d = 2$, or 3 for a volume), as separable filters
([Luisier et al. 2011, §III](https://doi.org/10.1109/TIP.2010.2073477)): the
scaling coefficients $\bar{\mathbf{w}}_j = \bar{\mathbf{D}}_j\tilde{\mathbf{y}}$ use $\bar H_j$ on every axis,
and a level has $2^d - 1$ wavelet bands $\mathbf{w}_{j,b} = \mathbf{D}_{j,b}\tilde{\mathbf{y}}$, one
for each choice of high-pass ($\tilde G_j$) or low-pass ($\bar H_j$) along
each axis other than all low-pass. The transform is undecimated: every band has
as many coefficients as the image. The scaling coefficients of a level are
aligned with its wavelet coefficients by the *group-delay compensation* filter
$Q(z)$ of [Luisier et al. (2011, Eq. 17)](https://doi.org/10.1109/TIP.2010.2073477),
which is $Q(z) = 1$ for the Haar filters ($H(z^{-1}) = z\,H(z)$), so that no
further filtering is needed. The reconstruction is

$$
\mathbf{R}_{j,b} = M_j^{-1}\,\mathbf{D}_{j,b}^\top,
\qquad
\mathbf{R}_{J+1} = M_J^{-1}\,\bar{\mathbf{D}}_J^\top,
\qquad
M_j = 2^{dj},
$$

with $M_j$ the redundancy of level $j$ (the paper's down-sampling factor
$M_j = 2^j$, in 1-D), so that
$\sum_{j,b}\mathbf{R}_{j,b}\mathbf{D}_{j,b} + \mathbf{R}_{J+1}\bar{\mathbf{D}}_J = \mathbf{I}$
(perfect reconstruction) and, with periodic boundaries,
$\operatorname{diag}\{\mathbf{D}_{j,b}\mathbf{R}_{j,b}\} = M_j^{-1}\mathbf{1}$
([Luisier et al. 2011, §III-B](https://doi.org/10.1109/TIP.2010.2073477)). Each
coefficient at index $l$ depends on the samples $l, \ldots, l + 2^j - 1$ along
every axis. [`haar_uwt`][optix.denoise.utils.haar_uwt] computes the transform,
for the part of the output that has its whole support inside the data.

### The thresholding function

Each band is processed by a pointwise function of its wavelet coefficient $w$
and of the scaling coefficient $\bar w$ at the same index and level. The
threshold follows the signal-dependent noise: the variance of the wavelet
coefficient is the Gaussian variance $\tilde\sigma^2$ plus the Poisson variance,
which is the local mean, estimated from $\bar w$
([Luisier et al. 2011, Eqs. 19–20](https://doi.org/10.1109/TIP.2010.2073477)):

=== "optix"

    $$
    \theta_{j,1}(w, \bar w) = w,
    \qquad
    \theta_{j,2}(w, \bar w) = w\,\exp\!\Bigl(-\Bigl(\frac{w}{3\,t_j(\bar w)}\Bigr)^{8}\Bigr),
    \qquad
    t_j(\bar w) = \sqrt{\beta_j\,\lvert\bar w\rvert + \tilde\sigma^2}
    $$

=== "Luisier et al. 2011"

    $$
    \theta_j(w, \bar w) = a_{j,1}\,\underbrace{w}_{\theta_{j,1}}
    + a_{j,2}\,\underbrace{w\exp\!\Bigl(-\Bigl(\frac{w}{3t_j(\bar w)}\Bigr)^{8}\Bigr)}_{\theta_{j,2}},
    \qquad
    t_j(\bar w) = \sqrt{\beta_j\lvert\bar w\rvert + \sigma^2}
    $$

with $\beta_j = M_j^{-1/2}$. For unit-norm filters with a constant value on their
support of $M_j$ samples, $\bar w = M_j^{1/2}$ times the local mean, so that
$\beta_j \bar w$ is the local mean of $\tilde{\mathbf{y}}$ (the Poisson variance). The
function $\theta_{j,2}$ is close to a hard threshold at $3\,t_j$ that adapts to
the noise of each coefficient (Fig. 5 of the paper). The derivatives that PURE
needs must exist, so $\lvert\bar w\rvert$ is replaced by the smooth
$\tanh(\kappa\bar w)\,\bar w$, with $\kappa = 100$
($k$ in the paper; [§III-C](https://doi.org/10.1109/TIP.2010.2073477)).
[`let_threshold`][optix.denoise.utils.let_threshold] returns $\theta$ and its first
and second partial derivatives in closed form.

### The LET and its linear system

Each wavelet band $(j, b)$ has two functions, $k = 1, 2$, and the low-pass
residual is not thresholded
([Luisier et al. 2011, Eq. 21](https://doi.org/10.1109/TIP.2010.2073477)):

=== "optix"

    $$
    \mathcal{F}(\tilde{\mathbf{y}})
    = \sum_{j=1}^{J}\sum_{b=1}^{2^d-1}\sum_{k=1}^{2}
    a_{j,b,k}\,\underbrace{\mathbf{R}_{j,b}\,\boldsymbol{\theta}_{j,k}(\mathbf{D}_{j,b}\tilde{\mathbf{y}}, \bar{\mathbf{D}}_j\tilde{\mathbf{y}})}_{\mathcal{F}_{j,b,k}(\tilde{\mathbf{y}})}
    + \underbrace{\mathbf{R}_{J+1}\bar{\mathbf{D}}_J\tilde{\mathbf{y}}}_{\mathcal{F}_{\mathrm{low}}(\tilde{\mathbf{y}})}
    $$

=== "Luisier et al. 2011"

    $$
    \mathbf{f}(\mathbf{y})
    = \sum_{j=1}^{J}\sum_{k=1}^{2}
    a_{j,k}\,\underbrace{\mathbf{R}_j\,\boldsymbol{\Theta}_{j,k}(\mathbf{D}_j\mathbf{y}, \bar{\mathbf{D}}_j\mathbf{y})}_{\mathbf{f}_{j,k}(\mathbf{y})}
    + \underbrace{\mathbf{R}_{J+1}\mathbf{D}_{J+1}\mathbf{y}}_{\text{lowpass}}
    $$

In one dimension there is one band per level and the paper has $K = 2J$
coefficients. In $d$ dimensions there are $2^d - 1$ bands per level, each with
its own pair $(a_{j,b,1}, a_{j,b,2})$, so $K = 2(2^d - 1)J$ ($24$ for an image
and $J = 4$, $56$ for a volume). Write $\mathcal{F}_k$ for the $K$ functions in one
index. With $\mathcal{F}_{\mathrm{low}}$ fixed, PURE of the sum is, up to a term that does
not depend on $\mathbf{a}$,

$$
\lvert\Omega\rvert\,\hat\epsilon(\mathbf{a}) = \mathbf{a}^\top \mathbf{G}\,\mathbf{a} - 2\,\mathbf{a}^\top\hat{\mathbf{c}} + \text{const},
$$

whose minimiser solves $\mathbf{G}\mathbf{a} = \hat{\mathbf{c}}$
([Luisier et al. 2011, Eq. 22](https://doi.org/10.1109/TIP.2010.2073477)):

=== "optix"

    $$
    G_{kl} = \mathcal{F}_k^\top \mathcal{F}_l,
    \qquad
    \hat c_k = \mathcal{F}_k^\top\bigl(\tilde{\mathbf{y}} - \mathcal{F}_{\mathrm{low}}\bigr)
    - \underbrace{\Bigl(\tilde{\mathbf{y}}^\top \partial\mathcal{F}_k + \tilde\sigma^2 \operatorname{div}\{\mathcal{F}_k - \partial\mathcal{F}_k\}\Bigr)}_{\delta_k}
    $$

=== "Luisier et al. 2011"

    $$
    [\mathbf{M}]_{2(k_1-1)+k_2,\,2(l_1-1)+l_2} = \mathbf{f}_{k_1,k_2}(\mathbf{y})^\top \mathbf{f}_{l_1,l_2}(\mathbf{y}),
    \qquad
    \hat c_{k_1,k_2} = \mathbf{f}_{k_1,k_2}^\top(\mathbf{I} - \mathbf{R}_{J+1}\mathbf{D}_{J+1})\mathbf{y}
    - \bigl(\mathbf{y}^\top \partial\mathbf{f}_{k_1,k_2} + \sigma^2 \operatorname{div}\{\mathbf{f}_{k_1,k_2} - \partial\mathbf{f}_{k_1,k_2}\}\bigr)
    $$

Here $\delta_k$ collects the derivative terms of PURE. The low-pass residual is
linear, with $\partial\mathcal{F}_{\mathrm{low}} = M_J^{-1}\mathbf{1}$, so it only adds a
constant to the risk. All the terms are sums over pixels or coefficients:
$\mathbf{G}$, $\hat{\mathbf{c}}$, the sums $\sum_i \tilde y_i$ and $\sum_i \tilde y_i^2$ and the
number of pixels are the *sufficient statistics* of the system, and they add
over blocks of an image (see
[Memory and streaming](#memory-and-streaming)). They are held by
[`PureSums`][optix.denoise.utils.PureSums], and
[`let_basis`][optix.denoise.utils.let_basis] returns the functions $\mathcal{F}_k$, the
low-pass term and $\delta_k$.

### The derivative terms for the Haar transform

The derivative terms of $\delta_k$ in terms of the nonlinearity are
([Luisier et al. 2011, Eq. 8, Corollary 1](https://doi.org/10.1109/TIP.2010.2073477)):

=== "optix"

    $$
    \begin{aligned}
    \delta_k = {} & \boldsymbol{\Theta}_{1}^\top (\mathbf{D}\bullet\mathbf{R}^\top)\tilde{\mathbf{y}}
    + \boldsymbol{\Theta}_{2}^\top (\bar{\mathbf{D}}\bullet\mathbf{R}^\top)\tilde{\mathbf{y}} \\
    & + \tilde\sigma^2\Bigl(\operatorname{diag}\{\mathbf{D}\mathbf{R}\}^\top\boldsymbol{\Theta}_{1} + \operatorname{diag}\{\bar{\mathbf{D}}\mathbf{R}\}^\top\boldsymbol{\Theta}_{2}\Bigr) \\
    & - \tilde\sigma^2\Bigl(\operatorname{diag}\{(\mathbf{D}\bullet\mathbf{D})\mathbf{R}\}^\top\boldsymbol{\Theta}_{11}
    + \operatorname{diag}\{(\bar{\mathbf{D}}\bullet\bar{\mathbf{D}})\mathbf{R}\}^\top\boldsymbol{\Theta}_{22}
    + 2\operatorname{diag}\{(\mathbf{D}\bullet\bar{\mathbf{D}})\mathbf{R}\}^\top\boldsymbol{\Theta}_{12}\Bigr)
    \end{aligned}
    $$

=== "Luisier et al. 2011"

    $$
    \begin{aligned}
    \hat\epsilon = {} & \frac{1}{N}\lVert \mathbf{f}(\mathbf{y}) - \mathbf{y}\rVert^2 - \frac{1}{N}\mathbf{1}^\top\mathbf{y} - \sigma^2
    + \frac{2}{N}\Bigl(\boldsymbol{\Theta}_{1}^\top (\mathbf{D}\bullet\mathbf{R}^\top)\mathbf{y} + \boldsymbol{\Theta}_{2}^\top (\bar{\mathbf{D}}\bullet\mathbf{R}^\top)\mathbf{y}\Bigr) \\
    & + \frac{2\sigma^2}{N}\Bigl(\operatorname{diag}\{\mathbf{D}\mathbf{R}\}^\top\boldsymbol{\Theta}_{1} + \operatorname{diag}\{\bar{\mathbf{D}}\mathbf{R}\}^\top\boldsymbol{\Theta}_{2}\Bigr) \\
    & - \frac{2\sigma^2}{N}\Bigl(\operatorname{diag}\{(\mathbf{D}\bullet\mathbf{D})\mathbf{R}\}^\top\boldsymbol{\Theta}_{11}
    + \operatorname{diag}\{(\bar{\mathbf{D}}\bullet\bar{\mathbf{D}})\mathbf{R}\}^\top\boldsymbol{\Theta}_{22}
    + 2\operatorname{diag}\{(\mathbf{D}\bullet\bar{\mathbf{D}})\mathbf{R}\}^\top\boldsymbol{\Theta}_{12}\Bigr)
    \end{aligned}
    $$

Here $\bullet$ is the Hadamard (element-wise) product of two matrices, and
$\boldsymbol\Theta_1$, $\boldsymbol\Theta_2$ and $\boldsymbol\Theta_{11}$,
$\boldsymbol\Theta_{22}$, $\boldsymbol\Theta_{12}$ (of the function $k$) are the vectors of first and second
partial derivatives of the function of each coefficient with respect to its
first argument $w$, its second argument $\bar w$, or both. For a band of level
$j$, with $M = M_j$, the unit-norm Haar filters have the same squared value
$M^{-1}$ at every tap of their support, the scaling filter is constant there
($M^{-1/2}$) and the wavelet filters have zero mean. The transform-dependent
terms then reduce, for a coefficient $l$ of the band (periodic boundaries),
to

$$
\operatorname{diag}\{\mathbf{D}\mathbf{R}\} = M^{-1},
\quad
\operatorname{diag}\{(\mathbf{D}\bullet\bar{\mathbf{D}})\mathbf{R}\} = M^{-3/2},
\quad
\operatorname{diag}\{\bar{\mathbf{D}}\mathbf{R}\} = \operatorname{diag}\{(\mathbf{D}\bullet\mathbf{D})\mathbf{R}\} = \operatorname{diag}\{(\bar{\mathbf{D}}\bullet\bar{\mathbf{D}})\mathbf{R}\} = 0,
$$

$$
(\mathbf{D}\bullet\mathbf{R}^\top)\tilde{\mathbf{y}} = M^{-3/2}\,\bar{\mathbf{w}},
\qquad
(\bar{\mathbf{D}}\bullet\mathbf{R}^\top)\tilde{\mathbf{y}} = M^{-3/2}\,\mathbf{w},
$$

and, for the nonlinear function $\theta_{j,b,2}$ with a constant $\tilde\sigma^2$, using
$\bar{\mathbf{D}}\tilde\sigma^2 = M^{1/2}\tilde\sigma^2$,

$$
\delta_{j,b,2} = M_j^{-3/2}\sum_{l}\Bigl[
\partial_w\theta\,\bigl(\bar w_l + M_j^{1/2}\tilde\sigma^2\bigr)
+ \partial_{\bar w}\theta\;w_l
- 2\tilde\sigma^2\,\partial_{w\bar w}\theta
\Bigr],
\qquad
\delta_{j,b,1} = M_j^{-3/2}\sum_{l}\bigl(\bar w_l + M_j^{1/2}\tilde\sigma^2\bigr),
$$

where the sums run over the coefficients $l$ of the band and the derivatives are
evaluated at $(w_l, \bar w_l)$. These closed forms need the periodic boundary
conditions, which is why the implementation extends the image periodically.

### The reliability rule

The Taylor approximation of PURE is poor for the nonlinear functions when the
signal is very low (typically when all the intensities are below 5
photoelectrons: the estimates $\hat c_{j,2}$ are then far from the true values). The paper
([Luisier et al. 2011, §III-D](https://doi.org/10.1109/TIP.2010.2073477)) keeps the nonlinear function of
level $j$ only if

$$
T_j = M_j\,E_{\text{mean}} > T,
\qquad
\hat E_{\text{mean}} = \frac{\lVert\tilde{\mathbf{y}}\rVert^2 - \mathbf{1}^\top\tilde{\mathbf{y}}}{\lvert\Omega\rvert} - \tilde\sigma^2,
$$

with $E_{\text{mean}} = \lVert\boldsymbol\lambda\rVert^2/\lvert\Omega\rvert$ the mean energy of the
intensity, estimated without bias by $\hat E_{\text{mean}}$
([Luisier et al. 2011, Eq. 5](https://doi.org/10.1109/TIP.2010.2073477)), and
$T = 10$ in all the experiments of the paper ("any value in $[5, 15]$ was
reasonable"). When a function is dropped, its coefficient is set to
$a = 0$. The linear function of each band is always kept. All bands of a level
share $T_j$, so they are kept or dropped together.
[`PureSums.active`][optix.denoise.utils.PureSums.active] returns the kept functions.

### Symbols of the method

| optix | Luisier et al. | Meaning | Shape |
|---|---|---|---|
| $\mathcal{F}$, $\hat{\boldsymbol\lambda}$ | $\mathbf{f}$, $\hat{\mathbf{x}}$ | the denoiser and its output, in photoelectrons | $\lvert\Omega\rvert$ |
| $\mathcal{F}_k$, $\mathcal{F}_{j,b,k}$ | $\mathbf{f}_k$, $\mathbf{f}_{j,k}$ | $k$-th function of the LET (level $j$, band $b$, linear or nonlinear $k = 1, 2$) | $\lvert\Omega\rvert$, $K$ of them |
| $\mathcal{F}_{\mathrm{low}}$ | lowpass | the low-pass residual, $\mathbf{R}_{J+1}\bar{\mathbf{D}}_J\tilde{\mathbf{y}}$ | $\lvert\Omega\rvert$ |
| $\mathbf{D}_{j,b}$, $\bar{\mathbf{D}}_j$ | $\mathbf{D}_j$, $\bar{\mathbf{D}}_j$ | analysis by the wavelet band $b$ and by the scaling filter of level $j$ | $\lvert\Omega\rvert \times \lvert\Omega\rvert$ (circulant) |
| $\mathbf{R}_{j,b}$ | $\mathbf{R}_j$ | reconstruction, $M_j^{-1}\mathbf{D}_{j,b}^\top$ | $\lvert\Omega\rvert \times \lvert\Omega\rvert$ |
| $\mathbf{w}$, $\bar{\mathbf{w}}$ | $\mathbf{w}$, $\bar{\mathbf{w}}$ | wavelet and scaling coefficients | $\lvert\Omega\rvert$ per band |
| $J$, $j$ | $J$, $j$ | number of levels (`levels`); level | scalar |
| $d$ | | number of transformed axes (`ndim`) | scalar |
| $b$ | | orientation band of a level, $1 \ldots 2^d - 1$ | |
| $M_j$ | $M_j$ | redundancy of level $j$, $2^{dj}$ ($2^j$ in the paper) | scalar |
| $\beta_j$ | $\beta_j$ | scale that turns $\bar w$ into a local mean, $M_j^{-1/2}$ | scalar |
| $t_j(\bar w)$ | $t_j(\bar w)$ | signal-dependent threshold | |
| $\theta_{j,k}$ | $\theta_{j,k}$, $\Theta_{j,k}$ | pointwise function of $(w, \bar w)$, and the vector of its values | |
| $\kappa$ | $k$ | sharpness of the smooth absolute value (`sharpness`) | scalar |
| $a_{j,b,k}$ | $a_{j,k}$ | LET coefficients | $K = 2(2^d - 1)J$ |
| $\mathbf{G}$, $\hat{\mathbf{c}}$ | $\mathbf{M}$, $\hat{\mathbf{c}}$ | matrix and right-hand side of the linear system | $K \times K$, $K$ |
| $\delta_k$ | | derivative terms of $\hat c_k$, $\tilde{\mathbf{y}}^\top\partial\mathcal{F}_k + \tilde\sigma^2\operatorname{div}\{\mathcal{F}_k - \partial\mathcal{F}_k\}$ | $K$ |
| $\hat\epsilon$ | $\epsilon$, $\hat\epsilon$ | PURE (Taylor form) | scalar |
| $T$, $T_j$, $E_{\text{mean}}$ | $T$, $T_j$, $E_{\text{mean}}$ | threshold of the reliability rule (`reliability`); statistic of level $j$; mean energy of the intensity | scalars |

## Algorithm

**Input:** noisy images $\mathbf{y} \in \mathbb{R}^{\lvert\Omega\rvert}$ (volumes with `ndim=3`); gain $g$,
offset $o$ and read-noise standard deviation $\sigma$ (or a `SensorEstimate`);
$J = 4$ levels, $T = 10$ (`reliability`), $\kappa = 100$ (`sharpness`) \
**Output:** denoised images $\hat{\mathbf{x}} \in \mathbb{R}^{\lvert\Omega\rvert}$, in the units of the data.

For each image (volume) of the data:

1. **Normalise:** $\tilde{\mathbf{y}} = (\mathbf{y} - o)/g$ and $\tilde\sigma^2 = \sigma^2/g^2$
   (pixel by pixel when they are maps).
2. **Extend** $\tilde{\mathbf{y}}$ periodically by $2^J - 1$ pixels on every side of
   every axis.
3. **For each level** $j = 1, \ldots, J$, with $M_j = 2^{dj}$ and $\beta_j = M_j^{-1/2}$:
    1. **analyse**: the $2^d - 1$ wavelet bands $\mathbf{w}_{j,b}$ and the scaling
       coefficients $\bar{\mathbf{w}}_j$ of the level;
    2. **threshold**: $\theta_{j,2}$ and its derivatives, with
       $t_j = \sqrt{\beta_j\lvert\bar w\rvert + \tilde\sigma^2}$;
    3. **reconstruct** the functions $\mathcal{F}_{j,b,1}$ and $\mathcal{F}_{j,b,2}$ of each band by
       $\mathbf{R}_{j,b}$, and compute their derivative terms $\delta_k$.
4. **Accumulate** the system: $\mathbf{G}$, $\mathbf{b} = [\mathcal{F}_k^\top(\tilde{\mathbf{y}} - \mathcal{F}_{\mathrm{low}})]_k$,
   $\boldsymbol\delta$, and the sums of $\tilde y$, $\tilde y^2$ and $\tilde\sigma^2$.
5. **Select** the functions: all $\mathcal{F}_{j,b,1}$, and the $\mathcal{F}_{j,b,2}$ of the levels
   with $M_j\hat E_{\text{mean}} > T$.
6. **Solve** $\mathbf{G}_{\text{kept}}\,\mathbf{a}_{\text{kept}} = (\mathbf{b} - \boldsymbol\delta)_{\text{kept}}$; $\mathbf{a} = 0$ for the others.
7. **Reconstruct** $\hat{\boldsymbol\lambda} = \sum_k a_k\mathcal{F}_k + \mathcal{F}_{\mathrm{low}}$ and **return**
   $\hat{\mathbf{x}} = g\,\hat{\boldsymbol\lambda} + o$.

Defaults: $J = 4$ (`levels`), $T = 10$ (`reliability`), $\kappa = 100$
(`sharpness`), $d = 2$ (`ndim`).

## Faithful mode and extensions

`faithful=True` (the default) is the UWT PURE-LET of the paper: the undecimated
Haar filterbank, the threshold of Eqs. 19–20 with the smooth absolute value of
$\kappa = 100$, the LET of Eq. 21, the Taylor-approximated PURE of Eq. 6 and the
reliability rule with $T = 10$. `is_faithful` is true when `reliability` is 10,
`sharpness` is 100, `ndim` is 2 and `pixelwise` is off.

### Choices not in the paper

The paper leaves a number of choices open, and the code makes them as follows.

1. **The number of levels $J$ is 4** (`levels`). The paper does not give it,
   and its Fig. 6 shows the levels 1 and 3. The result at a pixel depends on the
   data within $2^J - 1$ pixels along each axis.
2. **The 2-D (and 3-D) scaling is $M_j = 2^{dj}$** ($4^j$ for images, $8^j$ for
   volumes), with $\beta_j = M_j^{-1/2}$ and the reliability statistic
   $T_j = M_j E_{\text{mean}}$. The paper derives the method in 1-D, with
   $M_j = 2^j$ and $\beta_j = 2^{-j/2}$, and says that it extends to higher
   dimensions with separable filters (§III). With separable unit-norm filters,
   $\bar w$ is $M_j^{1/2}$ times a local mean over $M_j$ pixels, which is what
   $\beta_j$ undoes.
3. **Each orientation band has its own coefficients.** A level of the 2-D
   transform has three wavelet bands (high-pass along one axis, the other, or
   both), each with its own pair $(a_{j,b,1}, a_{j,b,2})$, so $K = 24$ for
   $J = 4$. The paper writes one pair per level, for a 1-D filterbank.
4. **Boundaries are periodic**, the case in which
   [Luisier et al. (2011, §III-B)](https://doi.org/10.1109/TIP.2010.2073477) give the
   transform-dependent terms in closed form. The image is extended by wrapping
   around, by $2^J - 1$ pixels. Structures near one border are filtered together
   with those near the opposite border.
5. **The output is in the units of the data**, $g\hat{\boldsymbol\lambda} + o$. The paper
   denoises the normalised data and does not map back.
6. **The noise parameters are inputs.** The paper estimates $\alpha$, $\delta$
   and $\sigma^2$ from the data by a robust regression; optix takes them as
   arguments, or from a `SensorEstimate` (the result of the
   [Poisson-Gaussian fit](../sensor/poisson-gaussian.md), for instance).
7. **Each image of a stack has its own coefficients**, solved from its own
   statistics, as for the single images of the paper.
8. **Numerical guards**, which do not change the method where the system is
   well posed:
    * $t^2$ is floored at the machine epsilon of the dtype ($t = 0$ occurs for
      pure Poisson noise, $\sigma = 0$, on a zero background);
    * $u = w/(3t)$ is clipped to $\lvert u\rvert \le 2$, where
      $\exp(-u^8) < e^{-256}$: the function and its derivatives are zero to
      working precision there, and the clip avoids overflow;
    * the sums of the system are accumulated in `float64`;
    * the system is scaled to a unit diagonal and solved by its eigenvalue
      decomposition, discarding the eigenvalues below $10^{-10}$ times the
      largest. For a well-conditioned system this is the exact solution. When
      some $\mathcal{F}_k$ are linearly dependent to working precision ($\mathcal{F}_k = 0$ for
      constant data, or coarse bands that wrap around an image smaller than
      their filters), it gives the minimum-norm minimiser of PURE.

**Not implemented.** The paper also uses a $16 \times 16$ overcomplete
block-DCT (BDCT) transform and a dictionary that combines it with the UWT
(Eq. 23), which does better on textured images. Only the UWT is implemented,
which is the algorithm used for the fluorescence-microscopy data of the paper.

### Extensions (opt-in, not faithful)

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `pixelwise` | uses the maps of a `SensorEstimate` (gain, offset, read variance of each pixel, as of an sCMOS camera) pixel by pixel | `False` | `False` |
| `ndim` | number of transformed axes: 3 for volumes, or for stacks in which time is declared spatial in the `layout` | `2` | `2` |
| `reliability` | threshold $T$ of the reliability rule | `10.0` | `10.0` |
| `sharpness` | $\kappa$ of the smooth absolute value | `100.0` | `100.0` |

Giving any value other than the faithful one sets `is_faithful` to false.

**Per-pixel maps.** The paper's noise parameters are scalars. With maps $g_i$,
$o_i$, $\sigma_i^2$, the normalised data $\tilde y_i = (y_i - o_i)/g_i$ are still
Poisson plus Gaussian noise of variance $\tilde\sigma_i^2 = \sigma_i^2/g_i^2$,
which now depends on the pixel. Stein's lemma holds pixel by pixel, so PURE is

$$
\hat\epsilon = \frac{1}{\lvert\Omega\rvert}\Bigl(\lVert\mathcal{F} - \tilde{\mathbf{y}}\rVert^2 - \mathbf{1}^\top\tilde{\mathbf{y}} - \sum_i\tilde\sigma_i^2\Bigr)
+ \frac{2}{\lvert\Omega\rvert}\Bigl(\tilde{\mathbf{y}}^\top\partial\mathcal{F} + \sum_i\tilde\sigma_i^2\bigl(\partial_i\mathcal{F}_i - \partial_i^2\mathcal{F}_i\bigr)\Bigr),
$$

with $\partial_i^2\mathcal{F}_i = \partial^2\mathcal{F}_i/\partial\tilde y_i^2$, which is
the expression of the [Probabilistic model](#probabilistic-model) section for a
constant $\tilde\sigma^2$. The threshold of a coefficient uses its own Gaussian
variance, $\beta_j\bar{\mathbf{D}}_j\tilde{\boldsymbol\sigma}^2$ (the mean of $\tilde\sigma_i^2$ over its
support), and the derivative terms become

$$
\delta_k = M^{-3/2}\sum_l\Bigl[\partial_w\theta\;\bar{\mathbf{D}}(\tilde{\mathbf{y}} + \tilde{\boldsymbol\sigma}^2)_l + \partial_{\bar w}\theta\;\mathbf{D}(\tilde{\mathbf{y}} + \tilde{\boldsymbol\sigma}^2)_l\Bigr]
- M^{-2}\sum_l\Bigl[(\partial_{ww}\theta + \partial_{\bar w\bar w}\theta)\,(\mathbf{D}\tilde{\boldsymbol\sigma}^2)_l + 2\,\partial_{w\bar w}\theta\,(\bar{\mathbf{D}}\tilde{\boldsymbol\sigma}^2)_l\Bigr],
$$

which reduces to the constant-variance expression above when
$\tilde{\boldsymbol\sigma}^2$ is constant. A `SensorEstimate` with maps needs
`pixelwise=True`; the function [`pure_let`][optix.denoise.pure_let] accepts maps
of `sigma`, `gain` and `offset` directly.

**Volumes.** With `ndim=3` the separable transform acts on three axes, each level has
seven bands ($K = 56$ for $J = 4$) and $M_j = 8^j$. The same applies to
space-time stacks, with time declared spatial in the `layout`. The paper
derives the method in 1-D and applies it to images ($d = 2$).

**`reliability` and `sharpness`.** `reliability` is the $T$ of the paper (any
value in $[5, 15]$ is reasonable there); 0 keeps every function as long as
the estimated energy is positive. `sharpness` is
the $\kappa$ of $\tanh(\kappa\bar w)\bar w$; larger values approach
$\lvert\bar w\rvert$ more closely, smaller ones make the threshold smoother.

The other keywords do not change the method:

| Keyword | Changes | Faithful value |
|---|---|---|
| `sigma` | standard deviation $\sigma$ of the read noise, in data units; 0 for pure Poisson noise; omit it to give a `SensorEstimate` to `fit` | `None` |
| `gain`, `offset` | $g$ and $o$ when no `SensorEstimate` is given (ignored with one) | `None` (1 and 0) |
| `levels` | number of levels $J$ (choice 1) | `4` |
| `block` | core block size per spatial axis of the streaming; the result does not depend on it | `None` (memory budget) |

## Memory and streaming

The tag is `halo`. A function $\mathcal{F}_{j,b,k}$ at a pixel depends on the data
within $2^j - 1$ pixels along each axis (analysis by filters of length $2^j$,
then reconstruction by filters of the same length), so $2^J - 1$ for $J$ levels.
[`pure_let_halo`][optix.denoise.pure_let_halo] computes it, and `PURELET.halo()` does
for a fitted estimator:

| $J$ | 1 | 2 | 3 | 4 (default) | 5 |
|---|---|---|---|---|---|
| halo (pixels) | 1 | 3 | 7 | 15 | 31 |

The coefficients $\mathbf{a}$ depend on the whole image, through the sums of the
system, so `apply` reads the data **twice**, in blocks with a periodic halo of
this size (the [`blocks`](../../reference/io.md) iterator; blocks whose halo
crosses the border of the image are re-read with indices wrapped around it):

1. **First pass.** For each block, the functions, the low-pass term and the
   derivative terms $\delta_k$ are computed on the core, and
   [`PureSums`][optix.denoise.utils.PureSums] adds the sums of the system. The
   statistics of the blocks of a frame are merged by addition: they are sums
   over the pixels and the coefficients of the core, so they give those of the
   whole image. Each frame then gets its coefficients from one $K \times K$
   system, solved once.
2. **Second pass.** Each block is processed again, without the derivative
   terms, reconstructed with the coefficients of its frame, mapped back to the
   units of the data and written to the output.

The result equals the whole-image computation for any `block`: to $10^{-10}$ in
`float64`, and to within 8 units in the last place in `float32` (the order
in which the blocks are summed can move a value). The data may be a zarr array or a
memory-mapped file larger than memory; the output cannot be the input (an
in-place call is refused, as a halo needs the unchanged neighbours). `fit` does not
read the data: the LET coefficients are not fitted but solved for each frame
by `apply`. Images of a stack, and groups, are independent, and the maps of a
`SensorEstimate` are cropped to each block.

The cost is linear in the number of pixels: each pass runs the $J$ levels of
the transform and the reconstruction of the $K$ functions, and the second pass
skips the derivative terms. In memory, a block holds the $K$ functions
and the intermediate transforms; the default `block` is chosen from the memory
budget of [`optix.config`](../../reference/configuration.md), assuming a
conservative number of copies of the block. Larger blocks reduce the overhead of
the halo, at the cost of memory. The statistics of all the frames are kept between the passes, $K^2$ values
of `float64` for each (about 5 kB per frame for images with $J = 4$, 25 kB for
volumes). The computation is vectorised and runs on any torch device.

## Usage

The first example denoises a synthetic image of filaments seen at low counts
(between 2 and 22 photoelectrons) with a read noise of 1.5, and compares it
with the truth. The data are in photoelectrons, so the gain is 1 and the offset
0.

```pycon
>>> import torch
>>> from optix.denoise import PURELET, pure_let, pure_let_halo
>>> from optix.models import phantoms
>>> _ = torch.manual_seed(0)
>>> x = phantoms.filaments((96, 96), intensity=1.0, seed=0)
>>> x = 20 * (x - x.min()) / (x.max() - x.min()) + 2
>>> y = torch.poisson(x) + 1.5 * torch.randn_like(x)
>>> def psnr(a):
...     return round(
...         float(10 * torch.log10(x.max() ** 2 / (a - x).square().mean())), 1
...     )
>>> psnr(y)
19.1
>>> psnr(pure_let(y, sigma=1.5))
27.9
>>> pure_let_halo()
15
```

The building blocks give access to the system. PURE evaluated at the solution
estimates the mean squared error, which here is known. The functions are
computed on an image extended periodically by the halo, as `pure_let` does.

```pycon
>>> import torch.nn.functional as F
>>> from optix.denoise.utils import PureSums, let_basis
>>> h = pure_let_halo(4)
>>> padded = F.pad(y[None, None].double(), (h,) * 4, mode="circular")[0, 0]
>>> funcs, low, delta = let_basis(padded, 1.5**2, 4)
>>> funcs.shape  # K = 2 * 3 * 4 functions
torch.Size([24, 96, 96])
>>> sums = PureSums.from_basis(funcs, low, delta, y.double(), 1.5**2, levels=4)
>>> a = sums.solve()
>>> estimate = (a[:, None, None] * funcs).sum(0) + low
>>> bool(torch.allclose(estimate.float(), pure_let(y, sigma=1.5), atol=1e-3))
True
>>> round(sums.risk(a), 2), round(float((estimate - x).square().mean()), 2)
(0.84, 0.79)
```

The reliability rule drops the nonlinear functions of the fine levels when the
signal is weak. At a peak of one photoelectron, only the coarsest level keeps its
three nonlinear functions, and a brighter image keeps all 24:

```pycon
>>> def n_kept(peak, sigma):
...     s2 = sigma**2
...     yy = torch.poisson(x * peak / x.max()) + sigma * torch.randn_like(x)
...     pad = F.pad(yy[None, None].double(), (h,) * 4, mode="circular")[0, 0]
...     f, lo, d = let_basis(pad, s2, 4)
...     return int(
...         PureSums.from_basis(f, lo, d, yy.double(), s2, levels=4).active().sum()
...     )
>>> n_kept(22, 1.5), n_kept(1, 0.1)
(24, 15)
```

Poisson–Gaussian data from a camera with gain 2, offset 100 and read noise 3 ADU
(variance 9). The estimator is given the `SensorEstimate`, and normalises and
maps back the data. Here a $96 \times 96$ image is cut into blocks of
$40 \times 40$ pixels; the blocks give the whole-image result, up to the
rounding of `float32`.

```pycon
>>> from optix.estimates import SensorEstimate
>>> from optix.models import simulate_sensor
>>> sensor = SensorEstimate(offset=100.0, variance=9.0, gain=2.0)
>>> z = simulate_sensor(x, sensor, seed=0)
>>> est = PURELET(block=(40, 40)).fit(z, init=sensor)
>>> est.halo(), est.is_faithful
(15, True)
>>> den = est.apply(z)  # 9 blocks, two passes
>>> truth = 2.0 * x + 100.0
>>> round(float((z - truth).square().mean().sqrt()), 1)
4.9
>>> round(float((den - truth).square().mean().sqrt()), 1)
1.8
>>> ref = pure_let(z, sigma=3.0, gain=2.0, offset=100.0)
>>> bool(torch.allclose(den, ref, rtol=1e-4, atol=1e-3))
True
```

Volumes (or space-time stacks) with `ndim=3`, an extension: here a $16 \times 32 \times 32$
volume of cells with three levels.

```pycon
>>> vol = phantoms.cells((16, 32, 32), 4, radius=4.0, ndim=3, seed=0)
>>> vol = 15 * vol / vol.max()
>>> noisy = torch.poisson(vol) + torch.randn_like(vol)
>>> den3 = pure_let(noisy, sigma=1.0, ndim=3, levels=3)
>>> bool((den3 - vol).square().mean() < (noisy - vol).square().mean() / 10)
True
>>> PURELET(sigma=1.0, ndim=3).is_faithful
False
```

The maps of an sCMOS camera (a gain, an offset and a read variance at each pixel)
need `pixelwise=True`, an extension; without it the estimator refuses them.

```pycon
>>> g = torch.Generator().manual_seed(1)
>>> var = 4 + 4 * torch.rand(96, 96, generator=g)
>>> gain = 1.5 + 0.5 * torch.rand(96, 96, generator=g)
>>> offset = 100 + torch.randn(96, 96, generator=g)
>>> maps = SensorEstimate(offset=offset, variance=var, gain=gain)
>>> zz = simulate_sensor(x, maps, seed=0)
>>> PURELET().fit(zz, init=maps)
Traceback (most recent call last):
    ...
ValueError: The SensorEstimate has per-pixel maps: pass pixelwise=True (an extension of the paper, whose noise parameters are scalars)
>>> den = PURELET(pixelwise=True).fit(zz, init=maps).apply(zz)
>>> truth = gain * x + offset
>>> round(float((zz - truth).square().mean().sqrt()), 1)
4.2
>>> round(float((den - truth).square().mean().sqrt()), 1)
1.5
```

Common variations: `layout=` to say which axes are spatial when the array has
frames or channels (each frame is denoised on its own, with its own
coefficients); a large zarr array in place of the tensor, with `out=` a separate
array; `levels=` for a larger support. A fitted estimator can be saved and loaded
like any other.

### Results on synthetic data

All checks use synthetic data with a known truth: a scene of cells and
filaments on a smooth background, with intensities in photoelectrons
(the scene of the test suite).

- **Closed forms.** The derivatives of the threshold, and the derivative terms
  $\delta_k$ of the Haar transform (for constant and per-pixel variance, 2-D and
  3-D), agree with finite differences and with a brute-force evaluation of
  Eq. 6.
- **PURE is unbiased (exact form).** With the LET coefficients fixed, the mean over
  150 realisations of PURE of Eq. 2, evaluated by brute force with the $N$
  shifted inputs of $\mathcal{F}^{-}$ (an $8 \times 8$ image of peak 20, $\sigma = 1$,
  $J = 2$), differs from the mean squared error by less than three standard errors of
  the mean. So does its first-order approximation, Eq. 6.
- **PURE is unbiased (Taylor form).** The table gives, on $256^2$ images with
  $J = 4$ and 100 realisations of the noise, the mean squared error and the mean
  difference $\hat\epsilon - \mathrm{MSE}$ (with its standard error), for the
  coefficients solved on an independent realisation (fixed) and for those solved
  on the same data (solved):

    | Peak / $\sigma$ | MSE | fixed coefficients | solved coefficients |
    |---|---|---|---|
    | 30 / 3 | 0.281 | $-0.0145 \pm 0.0071$ | $-0.0352 \pm 0.0072$ |
    | 5 / 0.5 | 0.0179 | $0.0003 \pm 0.0006$ | $-0.0013 \pm 0.0006$ |
    | 1 / 0.1 | 0.00228 | $-0.00002 \pm 0.00013$ | $-0.00017 \pm 0.00012$ |

    With fixed coefficients, the difference is within three standard errors of zero
    at the three intensities, and the risk estimate with a per-pixel
    $\tilde\sigma^2$ passes the same test. With the coefficients solved on the
    data themselves, PURE lies below the MSE (by 7 to 13 %): a risk estimate
    that is minimised over $K = 24$ weights on the data that it is evaluated on
    is optimistic, as any empirical risk is. This does not stop the minimiser
    from being nearly the best one (see the PSNR below).
- **The solution minimises PURE.** The solution of the system satisfies
  $\mathbf{G}\mathbf{a} = \hat{\mathbf{c}}$ to rounding, and random perturbations of $\mathbf{a}$
  increase PURE by the quadratic form of $\mathbf{G}$.
- **PSNR.** The mean over 10 noise realisations of the PSNR
  ($10\log_{10}(I_{\max}^2/\mathrm{MSE})$, in dB) of $256^2$ images, with a peak
  intensity of $I_{\max}$ photoelectrons and read noise $\sigma$. The *oracle
  LET* uses the same functions and the same kept set, with the coefficients that
  minimise the true MSE (least squares against the truth), which is the best
  that a LET can do:

    | Peak / $\sigma$ | Noisy | PURE-LET | Oracle LET |
    |---|---|---|---|
    | 30 / 3 | 18.28 | 35.10 | 35.32 |
    | 5 / 0.5 | 14.10 | 31.49 | 31.76 |
    | 1 / 0.1 | 8.10 | 26.41 | 26.60 |

    PURE-LET is within 0.3 dB of the oracle at the three intensities, without
    access to the truth.
- **Streaming.** Block-wise and whole-image results agree to $10^{-10}$ in
  `float64` and to within 8 units in the last place in `float32`, for any block
  size, any number of wraps of the halo around a small image, with a zarr array
  and with several frames.

These numbers describe the synthetic scenes only: they are not comparable with
the PSNRs of the standard test images of the paper.

## References

!!! references "References"

    1. Luisier F, Blu T, Unser M. *Image denoising in mixed Poisson–Gaussian
       noise.* IEEE Trans Image Process **20**(3):696–708, 2011.
       [doi:10.1109/TIP.2010.2073477](https://doi.org/10.1109/TIP.2010.2073477)

    ??? quote "BibTeX"

        ```bibtex
        @article{luisier2011image,
          author  = {Luisier, Florian and Blu, Thierry and Unser, Michael},
          title   = {Image denoising in mixed {Poisson--Gaussian} noise},
          journal = {IEEE Transactions on Image Processing},
          volume  = {20},
          number  = {3},
          pages   = {696--708},
          year    = {2011},
          doi     = {10.1109/TIP.2010.2073477},
        }
        ```
