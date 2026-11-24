# VSNR

!!! info "At a glance"
    - **Estimates:** stripe-free image (the stripes are the difference with the data)
    - **Noise model:** stationary noise: white noises convolved with known patterns
    - **Streaming:** per-frame
    - **Needs many frames:** no
    - **Reference:** Fehrenbach et al., IEEE Trans Image Process, 2012, [doi:10.1109/TIP.2012.2206037](https://doi.org/10.1109/TIP.2012.2206037)

## Overview

[`VSNR`][optix.destripe.VSNR], the variational stationary noise remover of
[Fehrenbach, Weiss and Lorenzo (2012)](https://doi.org/10.1109/TIP.2012.2206037),
removes structured noise from images, such as the stripes of light-sheet
microscopy, scanning electron microscopy or FIB-nanotomography. The noise is
described by a few known patterns $\mathbf{k}_j$ (a line, an elongated
Gaussian, a Dirac for white noise) that appear at random positions and with
random amplitudes. The method finds the positions and amplitudes that make the
rest of the image as smooth as possible, in the sense of total variation, and
subtracts the resulting noise.

The patterns and the strength of each noise component describe the data and are
chosen by the user; the paper gives no defaults for them. That makes VSNR the
most flexible stripe remover of the section, and the one that needs the most
tuning: it can remove stripes of any length and thickness, several kinds of
noise at once, and stripes that do not span the frame, which
[Wavelet-FFT](wavelet-fft.md) cannot. The paper reports that it also removes
the low-frequency part of the stripes better than the wavelet-FFT filter
(Section VI-A).

It is an iterative solver, run on every frame independently. Nothing is fitted:
`fit` only records the layout, and `apply` returns the denoised frames. The
function [`vsnr`][optix.destripe.vsnr] does the same on a tensor without a
layout, and [`vsnr_solve`][optix.destripe.utils.vsnr_solve] also returns the
noise, its white components and the convergence diagnostics.

## Image formation (optics)

The recorded frame is the sum of a noise-free image and a stationary noise. The
method assumes nothing about the cause of the noise: no noise model of the
camera, no shading (see [Image formation and noise](../../concepts/image-formation.md)).
In a pipeline, it is applied after the corrections for the sensor and for
shading. A noise is stationary when its distribution does not change if the
image is translated (with periodic boundary conditions); the paper restricts
itself to the noises obtained by convolving white noise with a pattern
(Hypothesis 3.1).

=== "optix"

    $$
    \mathbf{y} = \mathbf{x} + \mathbf{s},
    \qquad
    \mathbf{s} = \sum_{j=1}^{m} \boldsymbol{\lambda}_j * \mathbf{k}_j ,
    $$

    a frame $\mathbf{y} \in \mathbb{R}^{\lvert\Omega\rvert}$, with one value
    per pixel $i \in \Omega$ ($\lvert\Omega\rvert = HW$), is the sum of the stripe-free image
    $\mathbf{x}$ and a stationary noise $\mathbf{s}$. The noise is made of $m$
    known patterns $\mathbf{k}_j$, each convolved (with periodic
    boundaries) with an unknown white noise
    $\boldsymbol{\lambda}_j \in \mathbb{R}^{\lvert\Omega\rvert}$, one value
    per pixel.

=== "Fehrenbach et al. 2012"

    $$
    u_0 = u + b,
    \qquad
    b = \sum_{i=1}^{m} \lambda_i * \psi_i
    \quad \text{(Eq. 6 and 4)},
    $$

    the observed image $u_0$ is the sum of the image $u$ and a noise $b$
    independent of $u$.

| optix | [Fehrenbach et al. 2012](https://doi.org/10.1109/TIP.2012.2206037) | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}$ | $u_0$ | recorded frame | $\lvert\Omega\rvert$, one per frame |
| $\mathbf{x}$ | $u$ | stripe-free image | $\lvert\Omega\rvert$, one per frame |
| $\mathbf{s}$ | $b$ | stripes, the stationary noise | $\lvert\Omega\rvert$, one per frame |
| $\mathbf{k}_j$ | $\psi_i$ | pattern (kernel) of noise component $j$ | at most $H \times W$, shared by all frames |
| $\boldsymbol{\lambda}_j$ | $\lambda_i$ | white noise of component $j$ | $\lvert\Omega\rvert$, one per component and frame |
| $\lvert\Omega\rvert = HW$ | $n = n_x n_y$ | number of pixels of a frame | scalar |
| $i \in \Omega$ | $\mathbf{x}$ | pixel index | scalar |
| $m$, $j$ | $m$, $i$ | number of patterns; component index | scalars |
| $\phi_j$, $\alpha_j$ | $\phi_i$, $\alpha_i$ | prior of $\boldsymbol{\lambda}_j$; its weight | function and scalar, one per component |
| $\epsilon$, $h_\epsilon$ | $\epsilon$, $\phi_\epsilon$ | Huber parameter; Huber function | scalar and function, one for the stack |
| $\lVert \nabla \mathbf{x} \rVert_{2,1}^{\epsilon}$ | $\lVert \nabla u \rVert_{1,\epsilon}$ | smoothed isotropic total variation | scalar, one per frame |
| $C$ | $C$ | bound on $\lVert\boldsymbol{\lambda}\rVert_\infty$ | scalar, one per frame |

The model is the same additive form $\mathbf{y} = \mathbf{x} + \mathbf{s}$ as in
[Wavelet-FFT](wavelet-fft.md); the two methods differ in what they assume about
$\mathbf{s}$. With a single pattern that is a line as long as the frame, the
noise is a set of stripes that run over the whole frame, with independent
amplitudes (Section VI-A). With a Dirac as a second pattern, white noise is
removed as well. Shorter or thicker patterns give stripes of finite extent.

The patterns are $\mathbf{k}_j$ (the paper's $\psi_i$, not to be confused with the digamma function), kernels centred on
their middle pixel. [`dirac`][optix.destripe.utils.dirac],
[`line`][optix.destripe.utils.line] and
[`gaussian_stripe`][optix.destripe.utils.gaussian_stripe] build the ones of
the paper:

=== "optix"

    $$
    \mathbf{k}(\mathbf{r}) = \exp\!\left(
    -\frac{r_1'^2}{\sigma_1^2} - \frac{r_2'^2}{\sigma_2^2}\right),
    \quad
    \begin{aligned}
    r_1' &= r_1 \cos\theta + r_2 \sin\theta, \\
    r_2' &= -r_1 \sin\theta + r_2 \cos\theta,
    \end{aligned}
    $$

    with $r_1, r_2$ the coordinates along the first and second axis of the
    frame, centred on the kernel.

=== "Fehrenbach et al. 2012"

    $$
    \psi(x, y) = \exp\!\left(
    -\frac{x'^2}{\sigma_x^2} - \frac{y'^2}{\sigma_y^2}\right),
    \quad
    \begin{aligned}
    x' &= x \cos\theta + y \sin\theta, \\
    y' &= -x \sin\theta + y \cos\theta
    \end{aligned}
    \quad \text{(Section III-B)}
    $$

With $\theta = 0$, $\sigma_2 > \sigma_1$ gives stripes along the second axis and
$\sigma_1 > \sigma_2$ stripes along the first, the stripe axis; $\theta$ is in
degrees in `gaussian_stripe`, as the direction of the stripes is in the other
destriping pages. `gaussian_stripe` is the
filter of Section III-B. It is not the filter of the experiments on real images,
which use a Dirac and an anisotropic Gabor filter oriented along the stripes
(Section VI-B): the paper gives neither a formula nor parameters for the Gabor
filter, and optix has no helper for it. Any kernel can be passed in `filters`;
a Gabor-like kernel for stripes along the first axis is a stripe Gaussian times
a cosine across the stripes, whose period is to be chosen from the data:

```pycon
>>> import torch
>>> from optix.destripe.utils import gaussian_stripe
>>> r2 = (torch.arange(65) - 65 // 2)[None, :]
>>> envelope = gaussian_stripe((65, 65), (20.0, 4.0))
>>> gabor = envelope * torch.cos(2 * torch.pi * r2 / 8)
>>> gabor.shape
torch.Size([65, 65])
```

## Probabilistic model

The white noises $\boldsymbol{\lambda}_j$ are independent, with densities
$p(\boldsymbol{\lambda}_j) \propto \exp(-\phi_j(\boldsymbol{\lambda}_j))$ (Eq. 5),
and the image has the total-variation prior $p(\mathbf{x}) \propto
\exp(-\lVert \nabla \mathbf{x} \rVert_{2,1}^{\epsilon})$ (Eq. 8). The frame is
the sum $\mathbf{x} + \mathbf{s}$ without any other noise, so the maximum a
posteriori estimate of the noise components minimises

=== "optix"

    $$
    P(\boldsymbol{\lambda}) =
    \Bigl\lVert \nabla\Bigl(\mathbf{y} - \sum_{j=1}^m
    \boldsymbol{\lambda}_j * \mathbf{k}_j\Bigr)\Bigr\rVert_{2,1}^{\epsilon}
    + \sum_{j=1}^m \phi_j(\boldsymbol{\lambda}_j),
    \qquad
    \lVert \boldsymbol{\lambda} \rVert_\infty \le C,
    $$

    $$
    \lVert \mathbf{g} \rVert_{2,1}^{\epsilon} = \sum_{i \in \Omega}
    h_\epsilon\bigl(\lVert \mathbf{g}_i \rVert_2\bigr),
    \qquad
    h_\epsilon(t) = \begin{cases}
    t^2 / (2\epsilon) & t \le \epsilon, \\
    t - \epsilon/2 & \text{otherwise,}
    \end{cases}
    $$

    where $\mathbf{g}_i \in \mathbb{R}^2$ is the gradient at pixel $i \in \Omega$ and $\nabla$
    the forward differences along the two axes, zero at the last row and
    column. The estimates are $\hat{\mathbf{s}} = \sum_j \hat{\boldsymbol{\lambda}}_j
    * \mathbf{k}_j$ and $\hat{\mathbf{x}} = \mathbf{y} - \hat{\mathbf{s}}$.
    For $\epsilon = 0$, $h_\epsilon(t) = t$ and the prior is the isotropic total
    variation $\lVert \nabla \mathbf{x} \rVert_{2,1}$.

=== "Fehrenbach et al. 2012"

    $$
    P(\lambda) = \Bigl\lVert \nabla\Bigl(u_0 - \sum_{i=1}^m
    \lambda_i * \psi_i\Bigr)\Bigr\rVert_{1,\epsilon}
    + \sum_{i=1}^m \phi_i(\lambda_i)
    \quad \text{(Eq. 9)},
    \qquad \lambda \in \Xi = \{\lVert \lambda \rVert_\infty \le C\}
    \quad \text{(Eq. 11)},
    $$

    with $\lVert q \rVert_{1,\epsilon} = \sum_{\mathbf{x}}
    \phi_\epsilon(\lVert q(\mathbf{x}) \rVert_2)$ (Eq. 8) and $\phi_\epsilon$ the
    same Huber function.

The box constraint is not part of the model: it keeps the duality gap finite
(Section V-A), and $C$ is taken large enough for it to be inactive at the
minimiser, so that the solutions of (9) and (11) coincide.

Three priors are available for the white noises (Section III-C):

| `prior` | $\phi_j(\boldsymbol{\lambda}_j)$ | Noise $\boldsymbol{\lambda}_j$ |
|---|---|---|
| `"l1"` | $\alpha_j \lVert \boldsymbol{\lambda}_j \rVert_1$ | Laplace; a convex approximation of a sparse (Bernoulli) process |
| `"l2"` | $\tfrac{\alpha_j}{2} \lVert \boldsymbol{\lambda}_j \rVert_2^2$ | Gaussian |
| `"linf"` | $\iota_{\lVert \cdot \rVert_\infty \le \alpha_j}$ | uniform in $[-\alpha_j, \alpha_j]$ |

Here $\iota_S$ is the indicator of a set $S$: $0$ inside, $\infty$ outside. A
larger $\alpha_j$ makes the $j$-th component smaller and sparser (`"l1"`),
smaller (`"l2"`) or more bounded (`"linf"`). The weight depends on the amplitude
of the pattern $\mathbf{k}_j$: a pattern that is scaled by a factor needs
a weight scaled by the same factor to give the same result. The paper rescales
patterns and weights (Section VI-C) and leaves the choice to the user.

$\epsilon > 0$ smooths the total variation: the gradient magnitudes below
$\epsilon$ are penalised quadratically, by $t^2 / (2\epsilon)$, and the larger
ones linearly, as in the total variation, which is $\epsilon = 0$. When
$\epsilon$ exceeds every gradient magnitude, the prior is a Tikhonov term
$\tfrac{1}{2\epsilon} \lVert \nabla \mathbf{x} \rVert^2$, whose weight decreases
as $\epsilon$ grows. A positive $\epsilon$ makes the scheme converge faster, and
can give better results for some applications (Section IV-B).

## Inference / derivation

The estimate is a maximum a posteriori (MAP) estimate of the white noises
$\boldsymbol{\lambda}_j$ under the TV image prior, computed by convex
optimisation of its negative log-posterior.

The problem has a closed form only in the Tikhonov case ($\ell^2$ priors and
$\epsilon$ larger than every gradient magnitude at the solution, so that the
prior is quadratic), where it reduces to small linear systems solved by FFT
(Section V). In general it is solved to a given precision by the primal–dual
algorithm of [Chambolle and Pock (2011)](https://doi.org/10.1007/s10851-010-0251-1)
(Section V). Writing $\mathbf{A}$ for the linear operator
$\boldsymbol{\lambda} \mapsto \nabla\sum_j \boldsymbol{\lambda}_j *
\mathbf{k}_j$, $F(\mathbf{g}) = \lVert \nabla \mathbf{y} - \mathbf{g}
\rVert_{2,1}^{\epsilon}$ and $G(\boldsymbol{\lambda}) = \sum_j
\phi_j(\boldsymbol{\lambda}_j) + \iota_{\lVert \boldsymbol{\lambda}
\rVert_\infty \le C}$, the problem is $\min_{\boldsymbol{\lambda}}
F(\mathbf{A}\boldsymbol{\lambda}) + G(\boldsymbol{\lambda})$, which is recast as
the saddle-point problem

$$
\min_{\boldsymbol{\lambda}} \max_{\mathbf{q}}\;
\langle \mathbf{A}\boldsymbol{\lambda}, \mathbf{q} \rangle - F^*(\mathbf{q})
+ G(\boldsymbol{\lambda}),
$$

where $\mathbf{q}$ is a dual field with two values per pixel and $F^*$ the
Fenchel conjugate of $F$ (Eq. 13). Each iteration alternates a proximal step on
$\mathbf{q}$ and one on $\boldsymbol{\lambda}$, with an over-relaxation:

=== "optix"

    $$
    \begin{aligned}
    \mathbf{q}_{k+1} &= (\mathrm{Id} + \sigma_k \partial F^*)^{-1}
        (\mathbf{q}_k + \sigma_k \mathbf{A}\bar{\boldsymbol{\lambda}}_k), \\
    \boldsymbol{\lambda}_{k+1} &= (\mathrm{Id} + \tau_k \partial G)^{-1}
        (\boldsymbol{\lambda}_k - \tau_k \mathbf{A}^\top \mathbf{q}_{k+1}), \\
    \bar{\boldsymbol{\lambda}}_{k+1} &= \boldsymbol{\lambda}_{k+1}
        + \vartheta_k (\boldsymbol{\lambda}_{k+1} - \boldsymbol{\lambda}_k).
    \end{aligned}
    $$

=== "Fehrenbach et al. 2012"

    $$
    \begin{aligned}
    q_{k+1} &= (\mathrm{Id} + \sigma_k \partial F^*)^{-1}
        (q_k + \sigma_k A\bar\lambda_k), \\
    \lambda_{k+1} &= (\mathrm{Id} + \tau_k \partial G)^{-1}
        (\lambda_k - \tau_k A^* q_{k+1}), \\
    \bar\lambda_{k+1} &= \lambda_{k+1} + \theta_k (\lambda_{k+1} - \lambda_k)
    \end{aligned}
    \quad \text{(Algorithm 1)}
    $$

The over-relaxation parameter is written $\vartheta_k$ here, because $\theta$
is the stripe direction (see the [notation](../../concepts/notation.md)).

The two proximal steps have closed forms (Appendix A). The dual one is a
projection of $\tilde{\mathbf{q}} = \mathbf{q}_k + \sigma_k(\mathbf{A}
\bar{\boldsymbol{\lambda}}_k - \nabla\mathbf{y})$ onto the unit ball, at every
pixel, softened by $\epsilon$:

$$
\mathbf{q}_{k+1}(i) = \frac{\tilde{\mathbf{q}}(i)}
{\max\bigl(\lVert \tilde{\mathbf{q}}(i) \rVert_2,\; 1 + \sigma_k \epsilon\bigr)}
\quad \text{(Appendix A, item 4)} .
$$

The primal one acts independently on every pixel of every component, on
$t = \lambda_{k,j} - \tau_k (\mathbf{A}^\top \mathbf{q}_{k+1})_j$. For `"l1"`, it
is a soft threshold followed by a clip to $[-C, C]$:

$$
\lambda_{k+1,j} = \operatorname{sign}(t)\,
\min\bigl(\max(\lvert t \rvert - \tau_k \alpha_j, 0),\; C\bigr)
\quad \text{(Appendix A, item 5)},
$$

for `"l2"` it is $t / (\tau_k \alpha_j + 1)$ clipped to $[-C, C]$, and for
`"linf"` it is $t$ clipped to $[-\delta, \delta]$ with
$\delta = \min(C, \alpha_j)$.

**Cost per iteration.** The convolutions are periodic and computed with the FFT:
$\mathbf{A}$ and $\mathbf{A}^\top$ need one FFT per component and one for the
gradient, so an iteration costs $2m + 2$ FFTs of the frame size, with $m$ the
number of patterns. The paper reports less than 50 iterations to reduce the
duality gap by a factor $10^3$ when the steps are well chosen (Section VI-C).

**Stopping rule.** Each iteration gives a primal value $P(\boldsymbol{\lambda}_k)$
and a dual value $D(\mathbf{q}_k) = -F^*(\mathbf{q}_k) -
G^*(-\mathbf{A}^\top\mathbf{q}_k)$, whose difference is the duality gap
$\Delta(\boldsymbol{\lambda}_k, \mathbf{q}_k) = P - D$ (Eq. 15), an upper bound on
$P(\boldsymbol{\lambda}_k) - P(\boldsymbol{\lambda}^*)$. The iterations stop when
the gap has fallen to a fraction `tol` of its initial value, with the paper's
$10^{-3}$ (Section VI-C). For the components with an `"l1"` prior, $D$ is
evaluated at the dual-feasible point $\mathbf{q}_k / \kappa$, with
$\kappa = \max(1, \max_j \lVert (\mathbf{A}^\top \mathbf{q}_k)_j \rVert_\infty /
\alpha_j)$ over those components, and the conjugate of $\phi_j$ alone, which is
then zero. This is still an upper bound on $P(\boldsymbol{\lambda}_k) -
P(\boldsymbol{\lambda}^*)$, and it does not depend on $C$.

**Step sizes.** Convergence needs $\sigma_0 \tau_0 L^2 < 1$, with
$L = \lVert\lVert \mathbf{A} \rVert\rVert$ the norm of $\mathbf{A}$. With periodic
differences, it is computed exactly by the FFT (Appendix B):
$L^2 = \max_{\boldsymbol{\omega}} \Sigma(\boldsymbol{\omega})
(\lvert\hat d_1(\boldsymbol{\omega})\rvert^2 + \lvert\hat d_2(\boldsymbol{\omega})\rvert^2)$, with
$\Sigma = \sum_j \lvert\hat{\mathbf{k}}_j\rvert^2$ and
$\lvert\hat d_1(\boldsymbol{\omega})\rvert^2 = 4\sin^2(\pi \omega_1 / H)$ and
$\lvert\hat d_2(\boldsymbol{\omega})\rvert^2 = 4\sin^2(\pi \omega_2 / W)$ the transfer
functions of the forward differences along the two axes, for a frame of
$H \times W$ pixels. An over-estimate of $L$ slows the scheme down
and an under-estimate makes it diverge (Section VI-C). The rules of the paper
(Algorithm 2) depend on the strong convexity of $F^*$ (parameter $\epsilon$) and
of $G$ (parameter $\gamma = \min_j \alpha_j$, when all priors are `"l2"`):

| Priors | $\epsilon$ | Steps |
|---|---|---|
| all `"l2"` | $> 0$ | constant: $\mu = 2\sqrt{\gamma\epsilon}/L$, $\tau = \mu/(2\gamma)$, $\sigma = \mu/(2\epsilon)$, $\vartheta = 1/(1+\mu)$ |
| all `"l2"` | $0$ | $\vartheta_k = 1/\sqrt{1 + 2\gamma\tau_k}$, $\tau_{k+1} = \vartheta_k \tau_k$, $\sigma_{k+1} = \sigma_k / \vartheta_k$ |
| otherwise | any | constant $\tau$ and $\sigma$, $\vartheta = 1$ |

The convergence is accordingly $O(1/k)$ for non-quadratic priors, $O(1/k^2)$
for quadratic priors with $\epsilon = 0$ and linear for quadratic priors with
$\epsilon > 0$ (Proposition 5.3).

## Algorithm

**Input:** a frame $\mathbf{y} \in \mathbb{R}^{\lvert\Omega\rvert}$; patterns
$\mathbf{k}_1, \ldots, \mathbf{k}_m$ and weights $\alpha_1, \ldots, \alpha_m$
(no default); priors $\phi_j$ (`prior="l1"`), $\epsilon$ (`eps=0`), bound $C$,
`tol` $= 10^{-3}$, `max_iter` $= 1000$. \
**Output:** the stripe-free frame $\hat{\mathbf{x}} = \mathbf{y} - \sum_j \hat{\boldsymbol{\lambda}}_j * \mathbf{k}_j$.

1. **Compute** $L = \lVert \mathbf{A} \rVert$ from the patterns by FFT, and set
   $\sigma_0, \tau_0$ with $\sigma_0 \tau_0 L^2 < 1$.
2. **Initialise** $\boldsymbol{\lambda}_0 = \bar{\boldsymbol{\lambda}}_0 = \mathbf{0}$, $\mathbf{q}_0 = \mathbf{0}$.
3. **Repeat** for $k = 0, 1, \ldots$ until the duality gap is below `tol` times
   the initial gap, or `max_iter` is reached:
    1. **Dual step**, at every pixel $i$:
       $\mathbf{q}_{k+1}(i) = \tilde{\mathbf{q}}(i) / \max\bigl(\lVert \tilde{\mathbf{q}}(i) \rVert_2, 1 + \sigma_k \epsilon\bigr)$
       with $\tilde{\mathbf{q}} = \mathbf{q}_k + \sigma_k(\mathbf{A}\bar{\boldsymbol{\lambda}}_k - \nabla\mathbf{y})$;
    2. **Primal step**, at every pixel of every component $j$:
       $\boldsymbol{\lambda}_{k+1,j} = \operatorname{prox}_{\tau_k \phi_j + \iota_{[-C, C]}}\bigl(\boldsymbol{\lambda}_{k,j} - \tau_k (\mathbf{A}^\top \mathbf{q}_{k+1})_j\bigr)$;
    3. **Over-relax**: $\bar{\boldsymbol{\lambda}}_{k+1} = \boldsymbol{\lambda}_{k+1} + \vartheta_k (\boldsymbol{\lambda}_{k+1} - \boldsymbol{\lambda}_k)$;
    4. **Update** $\tau_k$, $\sigma_k$, $\vartheta_k$ by the rules above.
4. **Return** $\hat{\mathbf{x}} = \mathbf{y} - \sum_j \boldsymbol{\lambda}_j * \mathbf{k}_j$.

Frames are processed independently, and a frame stops
when its own gap has converged.

## Faithful mode and extensions

`VSNR` has no extension to the published method. Every keyword is a free
parameter of the paper, or a choice that the paper leaves open (below), so any
value is faithful, and `faithful=False` gives the same result.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `filters` | patterns $\mathbf{k}_j$, `(h, w)`, `(m, h, w)` or a list of kernels up to the frame size | required | required |
| `alpha` | weights $\alpha_j > 0$, one value or one per filter | required | required |
| `prior` | `"l1"`, `"l2"` or `"linf"`, one value or one per filter | `"l1"` | `"l1"` |
| `eps` | Huber parameter $\epsilon \ge 0$ | 0 | 0 |
| `bound` | box bound $C$ | derived from the frame | derived from the frame |
| `tol` | relative duality gap at which to stop | $10^{-3}$ | $10^{-3}$ |
| `max_iter` | iteration cap per frame | 1000 | 1000 |
| `tau`, `sigma` | initial steps $\tau_0$, $\sigma_0$ with $\sigma_0 \tau_0 L^2 < 1$ | derived from the frame | derived from the frame |

### Not specified in the paper

- **Patterns and weights.** The paper chooses them per application (Section III).
  There is no default.
- **Prior.** The paper uses the three priors; `"l1"`, the convex approximation of
  sparse noise, is the default here.
- **Split of the steps.** The paper asks for $\sigma_0 \tau_0 L^2 < 1$ and leaves
  the split between $\sigma_0$ and $\tau_0$ to experimental tuning (Section
  VI-C). The default is $\tau_0 = 0.99\, \bar g / L$ and $\sigma_0 = 0.99 /
  (\bar g L)$, with $\bar g$ the mean gradient magnitude of the frame (1 if the
  frame is constant). The noise $\boldsymbol{\lambda}$ scales with the data and
  $\mathbf{q}$ does not, so this makes the iterates equivariant to a rescaling of
  the intensities. If only one of `tau` and `sigma` is given, the other keeps
  the product at $0.99^2 / L^2$. In the constant-step case of `"l2"` priors with
  $\epsilon > 0$, the steps are set by Algorithm 2 and `tau` and `sigma` are
  ignored.
- **Box bound.** The paper asks for $C \ge \lVert \boldsymbol{\lambda}^*
  \rVert_\infty$. The default uses $\phi_j(\boldsymbol{\lambda}_j^*) \le
  P(\boldsymbol{\lambda}^*) \le P(\mathbf{0})$, which gives, per frame, the
  maximum over $j$ of $P(\mathbf{0})/\alpha_j$ (`"l1"`), $\sqrt{2P(\mathbf{0})/\alpha_j}$
  (`"l2"`) or $\alpha_j$ (`"linf"`). The box is then inactive at the minimiser.
  With `"l1"` priors this bound is loose, but it does not enter the stopping
  rule (see above): the gap does not depend on $C$.
- **Stopping at `max_iter`.** A `RuntimeWarning` is raised when some frames reach
  `max_iter` before the gap has fallen to `tol` (not when `tol=0` or
  `max_iter=0`). With `"l1"` priors and $\epsilon = 0$, the rate is only
  $O(1/k)$ (Proposition 5.3), so this is common.
- **Iteration cap.** `max_iter` is not in the paper, which stops on the gap only.
- **Boundaries and norm.** The convolutions are periodic (as in the paper). The
  differences are the forward ones of Section II, zero at the last row and
  column, but $L$ is computed with periodic differences, which gives an upper
  bound of the norm of $\mathbf{A}$ with the differences used: the step sizes
  are safe, and slightly conservative.
- **Sign of the conjugate.** Appendix A writes
  $F^*(\mathbf{q}) = \tfrac{\epsilon}{2}\lVert\mathbf{q}\rVert^2 - \langle \nabla u_0,
  \mathbf{q}\rangle$, which goes with $u = u_0 + \sum_j \lambda_j * \psi_j$. The
  code uses $+\langle \nabla \mathbf{y}, \mathbf{q}\rangle$, which goes with the
  model $u = u_0 - b$ of Eq. 6. The two differ by the sign of $\boldsymbol{\lambda}$
  only, and all the priors are even.
- **Layout.** The patterns are 2-D kernels over the two spatial axes of the
  layout, in layout order. The paper's first axis $x$ is the first spatial axis.

## Memory and streaming

Frames are independent: they are read in batches sized from the memory budget
(per-frame streaming), denoised and written back, so arrays larger than memory
(`zarr`, memmap) can be processed, and `apply` works in place. The batch is
solved at once, and a frame leaves the iterations when its gap is below `tol`.
The working memory is $16 + 9m$ buffers of the size of the chunk, with $m$ the
number of patterns: the iterates, their updated copies, the spectra of the
convolutions and the temporaries of the duality gap. The batch size accounts for
it. The cost is $2m + 2$ FFTs of the frame size per iteration; the paper reports
about 5 seconds for a 1024 × 1024 frame and one pattern on a 2012 laptop. The
patterns must fit in a frame; a pattern as long as the frame (stripes over the
whole frame) is a valid choice.

## Usage

```pycon
>>> import torch
>>> from optix.destripe import VSNR
>>> from optix.destripe.utils import line, vsnr_solve
>>> _ = torch.manual_seed(0)
>>> # 1. A frame (64 × 64) with stripes along the first axis: one offset per column.
>>> x = torch.zeros(64, 64)
>>> x[16:48, 16:48] = 100.0
>>> y = (x + 10 * torch.randn(1, 64))[None]  # (frames, H, W)
>>> # 2. One pattern: a line as long as the frame, along the first axis (the default).
>>> filters = line((64, 64))
>>> x_hat = VSNR(filters, alpha=10.0, prior="l2").fit(y).apply(y)
>>> # 3. The error before and after.
>>> print(f"{float((y - x).std()):.2f} {float((x_hat - x).std()):.2f}")
10.42 1.38
>>> # 4. The iterations run and the final duality gap, relative to the first.
>>> sol = vsnr_solve(y, filters, alpha=10.0, prior="l2", history=True)
>>> print(int(sol.iterations), f"{float(sol.gap[-1] / sol.gap[0]):.1e}")
170 9.6e-04
```

The default prior, `"l1"`, suits sparse stripes, but with $\epsilon = 0$ its
convergence rate is only $O(1/k)$: it often reaches `max_iter` before the gap
falls to `tol`, and a `RuntimeWarning` says so. Check the gap with `history=True`, raise
`max_iter`, or use `"l2"` (as above) or `eps > 0`, which converge faster.

Common variations: [`vsnr_solve`][optix.destripe.utils.vsnr_solve] with
`history=True` returns the noise, its components and the primal value and the
duality gap of every iteration, to check the convergence; `axis` of
[`line`][optix.destripe.utils.line] (a frame position, default $-2$, the first
spatial axis, or a layout label with `layout=`) and the `angle` of
[`gaussian_stripe`][optix.destripe.utils.gaussian_stripe] (in degrees) set the
stripe direction; `prior="l2"` and `eps > 0` use the accelerated schemes of
Algorithm 2; and the function `optix.destripe.vsnr(tensor, filters, alpha=...)`
works on a tensor `(..., H, W)` without a layout. See the
[API reference](../../reference/destripe.md).

## References

!!! references "References"

    1. Fehrenbach J, Weiss P, Lorenzo C. *Variational algorithms to remove
       stationary noise: applications to microscopy imaging.* IEEE Trans Image
       Process **21**(10):4420-4430, 2012.
       [doi:10.1109/TIP.2012.2206037](https://doi.org/10.1109/TIP.2012.2206037)
    2. Chambolle A, Pock T. *A first-order primal-dual algorithm for convex
       problems with applications to imaging.* J Math Imaging Vis
       **40**(1):120-145, 2011.
       [doi:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)

    ??? quote "BibTeX"

        ```bibtex
        @article{fehrenbach2012variational,
          author  = {Fehrenbach, J{\'e}r{\^o}me and Weiss, Pierre and Lorenzo, Corinne},
          title   = {Variational algorithms to remove stationary noise: applications to microscopy imaging},
          journal = {IEEE Transactions on Image Processing},
          volume  = {21},
          number  = {10},
          pages   = {4420--4430},
          year    = {2012},
          doi     = {10.1109/TIP.2012.2206037},
        }

        @article{chambolle2011first,
          author  = {Chambolle, Antonin and Pock, Thomas},
          title   = {A first-order primal-dual algorithm for convex problems with applications to imaging},
          journal = {Journal of Mathematical Imaging and Vision},
          volume  = {40},
          number  = {1},
          pages   = {120--145},
          year    = {2011},
          doi     = {10.1007/s10851-010-0251-1},
        }
        ```
