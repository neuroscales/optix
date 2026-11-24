# Chouzenoux

!!! info "At a glance"
    **Estimates:** deconvolved image (the latent image $\mathbf{x}$, in electrons, same shape as the data) \
    **Noise model:** Poisson-Gaussian, exact likelihood \
    **Streaming:** halo \
    **Needs many frames:** no \
    **Reference:** Chouzenoux, Jezierska, Pesquet and Talbot, SIAM J Imaging Sci, 2015, [doi:10.1137/15M1014395](https://doi.org/10.1137/15M1014395)

## Overview

`Chouzenoux` deconvolves an image recorded at a low photon count with a noticeable
read noise: given the data $\tilde{\mathbf{y}}$ and the point spread function (PSF)
$\mathbf{h}$, it estimates the latent image $\mathbf{x}$
([Chouzenoux et al. 2015](https://doi.org/10.1137/15M1014395)). It keeps the
Poisson–Gaussian noise model exactly: the likelihood of a pixel is the sum, over the
number of photoelectrons, of a Poisson probability times a Gaussian one, with no
approximation of the sum or of the noise. [Ikoma](ikoma.md) uses a shifted Poisson
law in its place, which has the mean and the variance of the Poisson–Gaussian model;
[HSPIRAL](hspiral.md) ignores the read noise. The paper shows that the exact negative
log-likelihood is convex and has a Lipschitz gradient, so that the estimate is the
minimiser of a convex problem, with a total-variation (TV) penalty, or with the
hybrid of TV and the Frobenius norm of the Hessian (TVH), and a non-negativity
constraint. It is found by a primal–dual splitting algorithm that needs no matrix
inversion. In the simulations of the paper the exact likelihood gives the best or
equal-best scores in most cases, and the shifted Poisson and exponential approximations
come within about 0.4 dB of it (Tables 3–6), at a higher cost: thousands of
iterations. The TVH prior gives the best scores of the priors it compares in nearly all
cases, and reduces the staircase effect of TV.

Use it when the read noise is high compared with the signal, at peaks of a few
tens of electrons, and the exactness of the likelihood is wanted. The method does not
work at high counts: the Lipschitz constant of the likelihood grows exponentially
with the peak over the read-noise variance, and with it the step of the solver
collapses (see [Faithful mode and extensions](#faithful-mode-and-extensions)). For
higher counts, or for a faster solver, use [Ikoma](ikoma.md). `apply` returns the
deconvolved images. The primal–dual solver and the likelihood function are described
in [Optimisation tools](../../concepts/optimisation.md#primaldual-splitting).

## Image formation (optics)

The microscope blurs the latent image with a known, shift-invariant PSF, and the
camera adds shot noise and read noise (see
[Image formation and noise](../../concepts/image-formation.md)). The PSF has 2 or 3
dims, is centred at index `K // 2` and is normalised to unit sum, so that
$\mathbf{x}$ is in electrons. The data are brought to electrons as for
[Ikoma](ikoma.md#image-formation-optics): the dark field (or, without a shading
estimate, the camera offset) is subtracted, the result is divided by the flat field,
rescaled to unit median, and by the gain, and the read-noise variance becomes
$\tilde\sigma^2 = \sigma^2 / (g\,f)^2$ pixel by pixel. The camera terms come from the
[`SensorEstimate`](../../reference/estimates.md) and
[`ShadingEstimate`](../../reference/estimates.md) given to `fit`. The paper takes the
data in electrons and does not model a camera; the conversion is an optix convention.
A sensor estimate with $\sigma^2 > 0$ is required (see
[below](#faithful-mode-and-extensions)).

=== "optix"

    $$
    \tilde{\mathbf{y}} = \frac{(\mathbf{y} - \mathbf{d}) / \mathbf{f}}{g},
    \qquad
    \tilde{y}_i = n_i + \tilde\varepsilon_i,
    \quad
    n_i \sim \mathrm{Poisson}\bigl((\mathbf{A}\mathbf{x})_i\bigr),
    \quad
    \tilde\varepsilon_i \sim \mathcal{N}(0, \tilde\sigma_i^2),
    \qquad
    \mathbf{x} \in C ,
    $$

    with $\mathbf{y}$ the recorded image in ADU, $\mathbf{d}$ the dark field (offset
    included), $\mathbf{f}$ the flat field, $g$ the gain, $\mathbf{A}$ the
    convolution with the PSF and $C = [0, x_+]^N$ the range of the image
    ($x_+ = +\infty$ by default).

=== "Chouzenoux et al. 2015"

    $$
    y = z(\overline{x}) + w,
    \qquad
    Z_i(\overline{x}) \sim \mathcal{P}\bigl([H\overline{x}]_i\bigr),
    \quad
    W_i \sim \mathcal{N}(0, \sigma^2),
    \qquad
    \overline{x} \in [0, +\infty)^N
    \quad (2.1\text{–}2.3),
    $$

    with $H$ the blur matrix and $\sigma$ the standard deviation of the read noise.

| optix | [Chouzenoux et al. 2015](https://doi.org/10.1137/15M1014395) | Meaning |
|---|---|---|
| $\tilde{\mathbf{y}}$, $\tilde{y}_i$ | $y$, $y_i$ | data in electrons, $Q$ values (pixels $i \in \mathcal{C}$) |
| $\mathbf{x}$ | $\overline{x}$, $x$ | latent image, $N$ values |
| $\mathbf{A}$, $\mathbf{h}$ | $H$ | blur operator and its PSF |
| $\tilde\sigma_i^2$ | $\sigma^2$ | read-noise variance in electrons² (one value per pixel in optix, one for the image in the paper) |
| $C$, $x_+$ | $C$, $x^+$ | range of the image; its upper bound |
| $n_i$, $\tilde\varepsilon_i$ | $z_i$, $w_i$ | photoelectron count and read noise |

## Probabilistic model

The pixels are independent. The density of $\tilde{y}_i$ given the mean $u = (\mathbf{A}\mathbf{x})_i$
is the convolution of a Poisson law with a Gaussian one (Eq. 2.4), an infinite sum,
and the negative log-likelihood of a pixel is (Eq. 2.7)

$$
\varphi_i(u) = -\log \sum_{m=0}^{\infty} \frac{u^m e^{-u}}{m!}\,
\frac{e^{-(\tilde{y}_i - m)^2 / 2\tilde\sigma_i^2}}{\sqrt{2\pi\tilde\sigma_i^2}},
\qquad u \ge 0 .
$$

Writing $s(a, b) = \sum_{m \ge 0} \frac{a^m}{m!}\, e^{-(b - m)^2 / 2\tilde\sigma_i^2}$ (Eq. 2.12),
its first two derivatives on $u > 0$ are $\varphi_i'(u) = 1 - \xi_i(u)$ and
$\varphi_i''(u) = \eta_i(u)$, with

$$
\xi_i(u) = \frac{s(u, \tilde{y}_i - 1)}{s(u, \tilde{y}_i)},
\qquad
\eta_i(u) = \frac{s(u, \tilde{y}_i - 1)^2 - s(u, \tilde{y}_i)\,s(u, \tilde{y}_i - 2)}{s(u, \tilde{y}_i)^2}
\quad (2.10\text{–}2.11).
$$

The paper proves that $\eta_i \ge 0$ (the function is convex) and that
$\eta_i(u) \le \eta_i(0)$, with (Eqs. 2.29–2.30)

$$
\xi_i(0) = e^{(2\tilde{y}_i - 1) / 2\tilde\sigma_i^2},
\qquad
\eta_i(0) = \bigl(1 - e^{-1/\tilde\sigma_i^2}\bigr)\, e^{(2\tilde{y}_i - 1)/\tilde\sigma_i^2},
$$

so that the gradient is Lipschitz with the constant $\mu = \lVert\mathbf{A}\rVert^2 \max_i \eta_i(0)$
(Theorem 2.1, Eq. 2.13), the factor $\lVert\mathbf{A}\rVert^2$ being 1 for a PSF of unit sum. To
define the function on the whole space, which the solver needs, it is continued
for $u < 0$ by the quadratic that matches its value, slope and curvature at 0
(Eq. 2.28),

$$
\bar\varphi_i(u) =
\begin{cases}
\varphi_i(u) & u \ge 0 \\
\tfrac12 \eta_i(0)\,u^2 + (1 - \xi_i(0))\,u + \varphi_i(0) & u < 0 ,
\end{cases}
$$

and the non-negativity (or the range of the image) is imposed by the indicator
function $\iota_C$ of $C$ (Eqs. 2.25–2.27). The estimate minimises this likelihood
plus a penalty on the gradient (isotropic total variation) and on the Hessian of
the image (Section 3.1; $\mathbf{H}$ packs the second differences with $\sqrt{2}$ on
the cross terms, as for [HSPIRAL](hspiral.md#probabilistic-model), so that its
Euclidean norm at a pixel is the Frobenius norm of the Hessian matrix):

=== "optix"

    $$
    \min_{\mathbf{x}} \;
    \sum_{i \in \mathcal{C}} \bar\varphi_i\bigl((\mathbf{A}\mathbf{x})_i\bigr)
    + \iota_C(\mathbf{x})
    + \lambda_{\mathrm{TV}} \sum_{i} \bigl\lVert (\nabla\mathbf{x})_i \bigr\rVert_2
    + \lambda_{\mathrm{H}} \sum_{i} \bigl\lVert (\mathbf{H}\mathbf{x})_i \bigr\rVert_2 ,
    $$

    with $\nabla$ the forward differences and $\mathbf{H}$ the Hessian operator
    ([`Gradient`, `Hessian`](../../concepts/optimisation.md#difference-operators)). With
    `hessian=0` the penalty is the total variation alone.

=== "Chouzenoux et al. 2015"

    $$
    \min_{x} \;
    h(x) + \psi_0(x) + \sum_{r=1}^{R} \psi_r(V_r x)
    \quad (3.1),
    \qquad
    h(x) = \sum_{i=1}^{Q} \varphi_i\bigl([Hx]_i\bigr)
    \quad (2.26),
    $$

    with $\psi_0 = \iota_C$, $V_1 = [(\Delta^{\mathrm h})^\top\ (\Delta^{\mathrm v})^\top]^\top$ and
    $\psi_1(V_1 x) = \lambda_{\mathrm{TV}} \sum_n \bigl([\Delta^{\mathrm h} x]_n^2 + [\Delta^{\mathrm v} x]_n^2\bigr)^{1/2}$,
    $V_2 = [(\Delta^{\mathrm{hh}})^\top\ \sqrt{2}(\Delta^{\mathrm{hv}})^\top\ (\Delta^{\mathrm{vv}})^\top]^\top$ and
    $\psi_2(V_2 x) = \lambda_{\mathrm{H}} \sum_n \bigl([\Delta^{\mathrm{hh}} x]_n^2 + 2[\Delta^{\mathrm{hv}} x]_n^2 + [\Delta^{\mathrm{vv}} x]_n^2\bigr)^{1/2}$.

| optix | [Chouzenoux et al. 2015](https://doi.org/10.1137/15M1014395) | Meaning |
|---|---|---|
| $\bar\varphi_i$ | $\varphi_i$ (on $\mathbb{R}$), $\Phi_i$ (on $[0, +\infty)$) | negative log-likelihood of pixel $i$, continued for $u < 0$ |
| $\xi_i$, $\eta_i$ | $\xi_i$, $\eta_i$ | first and second derivative terms of $\varphi_i$ |
| $\mu$ | $\mu$ | Lipschitz constant of the gradient of the likelihood term |
| $\lambda_{\mathrm{TV}}$, $\lambda_{\mathrm{H}}$ | $\lambda_{\mathrm{TV}}$, $\lambda_{\mathrm{H}}$ | weights of the TV and Hessian penalties (`tv`, `hessian`) |
| $\nabla$, $\mathbf{H}$ | $V_1$ ($\Delta^{\mathrm h}$, $\Delta^{\mathrm v}$), $V_2$ ($\Delta^{\mathrm{hh}}$, $\Delta^{\mathrm{hv}}$, $\Delta^{\mathrm{vv}}$) | stacked first and second differences |
| $\iota_C$ | $\psi_0$ | indicator of the range $C$ |

The penalties and the range are those of the TV and TVH experiments of the paper
(Section 4.1). The paper also uses a non-local TV prior and compares the exact
likelihood with five approximations of it (Table 1); neither is implemented here.

## Inference / derivation

The problem is convex, and the estimate is its minimiser: a maximum a posteriori
(MAP) estimate for the Poisson–Gaussian likelihood and the improper TV and
Hessian priors. The solver is the primal–dual splitting algorithm of
[Combettes and Pesquet (2012)](https://doi.org/10.1007/s11228-011-0191-y), a
forward–backward–forward scheme, as Algorithm 1 of the paper. It uses the gradient
of the smooth term $h$ and the proximity operator of $\psi_0$, and, for each
penalty, one dual variable $\mathbf{v}_r$ and the proximity operator of $\psi_r$, so
that no matrix is inverted and no linear system is solved. For $\psi_r =
\lambda\lVert\cdot\rVert_{2,1}$ the dual step is the projection of $\mathbf{y}_2$ onto the
$\ell_2$ balls of radius $\lambda$ (Moreau's identity,
[Optimisation tools](../../concepts/optimisation.md#proximity-operators)). One iteration
is, with $\nabla h(\mathbf{x}) = \mathbf{A}^\top(\mathbf{1} - \boldsymbol{\xi}(\mathbf{A}\mathbf{x}))$ (Eq. 2.8)
and $\mathbf{V}_r$ the operator of penalty $r$:

=== "optix"

    $$
    \begin{aligned}
    \mathbf{y}_1 &= \mathbf{x}_t - \gamma\Bigl(\nabla h(\mathbf{x}_t) + \sum_r \mathbf{V}_r^\top \mathbf{v}_r\Bigr),
      \qquad \mathbf{p}_1 = \mathrm{prox}_{\gamma\iota_C}(\mathbf{y}_1) \\
    &\text{for each } r: \\
    &\quad \mathbf{y}_{2} = \mathbf{v}_r + \gamma\,\mathbf{V}_r\mathbf{x}_t,
      \qquad \mathbf{p}_{2} = \mathbf{y}_{2} - \gamma\,\mathrm{prox}_{\psi_r/\gamma}(\mathbf{y}_{2}/\gamma) \\
    &\quad \mathbf{q}_{2} = \mathbf{p}_{2} + \gamma\,\mathbf{V}_r\mathbf{p}_1,
      \qquad \mathbf{v}_r \leftarrow \mathbf{v}_r - \mathbf{y}_{2} + \mathbf{q}_{2} \\
    \mathbf{q}_1 &= \mathbf{p}_1 - \gamma\Bigl(\nabla h(\mathbf{p}_1) + \sum_r \mathbf{V}_r^\top \mathbf{p}_{2,r}\Bigr),
      \qquad \mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{y}_1 + \mathbf{q}_1
    \end{aligned}
    $$

=== "Chouzenoux et al. 2015"

    $$
    \begin{aligned}
    y_{1,k} &= x_k - \gamma\Bigl(\nabla h(x_k) + \sum_r V_r^\top v_{r,k}\Bigr),
      \qquad p_{1,k} = \mathrm{prox}_{\gamma\psi_0}(y_{1,k}) \\
    y_{2,r,k} &= v_{r,k} + \gamma V_r x_k,
      \qquad p_{2,r,k} = y_{2,r,k} - \gamma\,\mathrm{prox}_{\gamma^{-1}\psi_r}(\gamma^{-1} y_{2,r,k}) \\
    q_{2,r,k} &= p_{2,r,k} + \gamma V_r p_{1,k},
      \qquad v_{r,k+1} = v_{r,k} - y_{2,r,k} + q_{2,r,k} \\
    q_{1,k} &= p_{1,k} - \gamma\Bigl(\nabla h(p_{1,k}) + \sum_r V_r^\top p_{2,r,k}\Bigr),
      \qquad x_{k+1} = x_k - y_{1,k} + q_{1,k}
    \end{aligned}
    $$

    (Algorithm 1, without the error terms $c_k$ and $e_k$.)

Here $\mathrm{prox}_{\gamma\iota_C}$ is the projection onto $C$. Convergence of $\mathbf{x}_t$ and
$\mathbf{p}_{1,t}$ to a minimiser is guaranteed for any step
$\gamma \in [\epsilon, (1 - \epsilon)/\delta]$ with $\epsilon \in (0, 1/(\delta + 1))$ and

$$
\delta = \mu + \Bigl(\sum_r \lVert\mathbf{V}_r\rVert^2\Bigr)^{1/2}
\quad (\text{Theorem 3.2 iii}),
$$

provided that the errors on the gradient are summable (Theorem 3.2 iv). The paper takes
for $\gamma$ "the maximum possible value". The gradient needs the series $s(a, b)$,
which are infinite: the paper sums them over a window of $m$ around
$m^\star = \tilde\sigma^2\,W\bigl(\tfrac{a}{\tilde\sigma^2} e^{b/\tilde\sigma^2}\bigr)$, with $W$ the
Lambert function, from $\lfloor m^\star - \Delta\tilde\sigma \rfloor$ to
$\lceil m^\star + \Delta\tilde\sigma \rceil$, and shows that $\Delta = 3$ is enough
(Section 3.4, Proposition 3.3, Eqs. 3.4–3.7, Table 2). The window used here is
different (see [below](#faithful-mode-and-extensions)).

## Algorithm

**Input:** data $\mathbf{y}$, PSF $\mathbf{h}$, a sensor estimate with $\sigma^2 > 0$, optionally a
shading estimate; optionally the weights $\lambda_{\mathrm{TV}}$, $\lambda_{\mathrm{H}}$, the
upper bound $x_+$ and the step $\gamma$. \
**Output:** the deconvolved image $\hat{\mathbf{x}}$, in electrons.

1. **Prepare.** $\tilde{\mathbf{y}} = (\mathbf{y} - \mathbf{d}) / \mathbf{f} / g$ and $\tilde\sigma^2 = \sigma^2 / (g\,\mathbf{f})^2$,
   without clipping; normalise $\mathbf{h}$ to unit sum.
2. **Scan the data.** Once per image (one pass over the data), find $\max\tilde{y}$, the mean
   $\bar{y}$, the 99.9th percentile $q_{99.9}(\tilde{y})$, the median $q_{50}(\tilde{y})$ and
   $\max_i \log \eta_i(0)$.
3. **Set the parameters.** $\lambda_{\mathrm{TV}} = 0.2 / (q_{99.9} - q_{50})$ and
   $\lambda_{\mathrm{H}} = 0.5 / (q_{99.9} - q_{50})$ unless given; $\mu = \max_i \eta_i(0)$;
   $\delta = \mu + \sqrt{\sum_r \lVert\mathbf{V}_r\rVert^2}$ with $\lVert\nabla\rVert^2 \le 4d$,
   $\lVert\mathbf{H}\rVert^2 \le (4d)^2$ in $d$ dimensions (a penalty with a weight of 0 is left
   out); $\epsilon = \min(10^{-3}, \tfrac{1}{2(\delta + 1)})$ and $\gamma = (1 - \epsilon)/\delta$ unless
   given. Warn if $(\max\tilde{y} + \max\tilde\sigma^2)/\gamma > 10^4$ (stop with an error with
   `on_step_collapse="raise"`, or when $\mu$ overflows and no step is left).
4. **Initialise.** $\mathbf{x}_0 = \bar{y}\,\mathbf{1}$ on the grid of the block, and all dual
   variables $\mathbf{v}_r = \mathbf{0}$.
5. **Repeat** for $t = 0, \ldots, 4999$:
    1. one iteration of the scheme above, with the gradient of the fidelity
       $\mathbf{A}^\top(\mathbf{1} - \boldsymbol{\xi}(\mathbf{A}\mathbf{x}))$ on the observed pixels;
    2. stop if $\lVert\mathbf{x}_{t+1} - \mathbf{x}_t\rVert_2 \le 10^{-5}\,\lVert\mathbf{x}_t\rVert_2$.
6. **Return** $\mathbf{p}_1$ of the last iteration, restricted to the observed region. It lies in $C$.

The number of iterations, the last relative change $\lVert\mathbf{x}_{t+1} - \mathbf{x}_t\rVert / \lVert\mathbf{x}_t\rVert$
and the last norm of the full forward–backward–forward residual
$\bigl(\lVert\mathbf{x}_t - \mathbf{p}_{1,t}\rVert^2 + \sum_r \lVert\mathbf{v}_{r,t} - \mathbf{p}_{2,r,t}\rVert^2\bigr)^{1/2}/\gamma$
of every block are recorded in `history.extras["iterations"]`, `["change"]` and
`["residual"]`. The residual tends to 0 at a solution and, unlike the relative change, does
not depend on the step: it is the diagnostic of convergence. The weights, the Lipschitz constant, the step and the
scalars of the scan are the attributes `tv`, `hessian`, `lipschitz`, `step`, `peak`,
`robust_peak`, `median` and `mean` after `fit`.

## Faithful mode and extensions

`faithful=True` (the default) reproduces the paper: the exact likelihood with the
quadratic continuation, the TV and TVH penalties, the algorithm of Theorem 3.2 with the
largest admissible step, and the result in $C$. The defaults of `faithful=False` are
the same. The weights `tv` and `hessian` are tuned per image in the paper, and both TV
alone (`hessian=0`) and TVH are its method, so any values keep the estimate faithful.
The upper bound `upper` ($x_+$) is the paper's when it is known. `max_iter` and `tol`
are not given in the paper, so any values keep the estimate faithful;
`on_step_collapse` only chooses between a warning and an error. `halo` and `block` are
memory settings and keep it faithful too, even when they split an image. A given `step`
marks the estimate as not faithful.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `tv` | weight $\lambda_{\mathrm{TV}} \ge 0$ (any value is faithful; 0 removes the penalty) | `None` (rule below, not in the paper) | `None` |
| `hessian` | weight $\lambda_{\mathrm{H}} \ge 0$ (any value is faithful; 0 gives the TV prior) | `None` (rule below, not in the paper) | `None` |
| `upper` | upper bound $x_+ > 0$ of $C$, in electrons (faithful) | `None` ($C = [0, +\infty)$) | `None` |
| `step` | step $\gamma > 0$; a given step skips the high-count check and the check of `apply`, and is not faithful | `None` (largest of Theorem 3.2 iii) | `None` |
| `max_iter` | iteration cap (any value is faithful) | 5000 (not in the paper, whose count is 5500) | 5000 |
| `tol` | relative-change tolerance; `0` runs `max_iter` iterations (any value is faithful) | $10^{-5}$ (not in the paper) | $10^{-5}$ |
| `on_step_collapse` | `"warn"`: a `RuntimeWarning` when $(\max\tilde{y} + \tilde\sigma^2)/\gamma > 10^4$, then the run as in the paper; `"raise"`: a `ValueError` (faithful) | `"warn"` | `"warn"` |
| `halo` | halo width per spatial axis, at least the PSF size (memory setting, faithful) | `None` (twice the PSF size) | `None` |
| `block` | core block size per spatial axis (memory setting, faithful) | `None` (memory budget) | `None` |

**Choices the paper does not specify.** These are made by the code, and flagged as such
in the docstrings:

- **Start.** $\mathbf{x}_0$ is the mean of the data of the image, a constant, and the dual
  variables start at 0. The paper takes any $x_0$ and $v_{r,0}$.
- **The $\epsilon$ of the step.** The step is $\gamma = (1 - \epsilon)/\delta$, with
  $\epsilon = \min(10^{-3}, \tfrac{1}{2(\delta + 1)})$, which lies in $(0, 1/(\delta + 1))$. The
  paper takes "the maximum possible value" without giving $\epsilon$.
- **Stopping rule.** The paper stops "when the difference in norm between two
  consecutive iterates becomes small". Its one iteration count for the exact likelihood
  is 5500, for $\bar{x}_1$ with the TVH prior (Figure 3). Here the run stops when
  $\lVert\mathbf{x}_{t+1} - \mathbf{x}_t\rVert_2 \le 10^{-5}\lVert\mathbf{x}_t\rVert_2$, or after 5000
  iterations, a cap below the paper's count. The result is $\mathbf{p}_1$ of the last
  iteration.
- **Boundary.** The paper does not say how the borders are treated. The blur is computed on
  a grid larger than the block (the block plus the PSF size, rounded up to a fast FFT
  size, as for [Ikoma](ikoma.md#probabilistic-model)), the likelihood sums over the
  observed pixels only, and the differences are periodic on that grid. $\mathbf{x}_0$ is
  constant over the whole grid.
- **No clipping.** The data are used as they are, negative values included: the exact
  likelihood is convex for any $\tilde{y}$. This differs from [Ikoma](ikoma.md), which clips
  the data at $-\tilde\sigma^2$. The paper truncates the data only for the likelihoods
  that need it (Section 4.1).
- **Series window.** The paper sums the series over the window of Lambert-function
  estimate, $\Delta = 3$ (Eqs. 3.4–3.7). Here $\xi$ and $\bar\varphi$ are evaluated by
  [`poisson_normal_ratio`](../../reference/special.md) and `poisson_normal_logpdf`, in
  float64, over a window of $m$ that is adapted to each pixel and leaves out the terms
  below $e^{-50}$ times the largest one. Both series of $\xi$ are summed over the same
  window, so that the ratio stays accurate when the mean is close to 0. The truncation
  errors are negligible, as the summability of Theorem 3.2 (iv) requires.
- **Read noise is required.** $\sigma^2 > 0$: fitting without a sensor estimate, or with a
  variance of 0, raises a `ValueError`. Without read noise the model is Poisson, whose
  gradient is not Lipschitz, and the step of the algorithm would be 0. Use
  [Richardson–Lucy](richardson-lucy.md) or [HSPIRAL](hspiral.md) then. The variance may
  be a map: $\mu$ and the likelihood then use the variance of each pixel.
- **Default weights.** The paper tunes $\lambda_{\mathrm{TV}}$ and $\lambda_{\mathrm{H}}$ by hand
  for the lowest mean absolute error (for instance $\lambda_{\mathrm{TV}} = 0.008$,
  $\lambda_{\mathrm{H}} = 0.296$ for its first image, Table 3) and gives no rule. The defaults
  are
  $\lambda_{\mathrm{TV}} = \kappa_{\mathrm{TV}} / (q_{99.9}(\tilde{y}) - q_{50}(\tilde{y}))$ and
  $\lambda_{\mathrm{H}} = \kappa_{\mathrm{H}} / (q_{99.9}(\tilde{y}) - q_{50}(\tilde{y}))$, per image, with
  $\kappa_{\mathrm{TV}} = 0.2$ and $\kappa_{\mathrm{H}} = 0.5$, where $q_{99.9}$ is the 99.9th
  percentile of the prepared data (exact, in the pass of `fit`) and $q_{50}$ their median
  (from a uniform sample of 8192 pixels per image taken in the same pass, exact for smaller
  images). The rule is that of [HSPIRAL](hspiral.md#faithful-mode-and-extensions): the
  priors ignore a constant, so the amplitude of the signal above the background sets their
  scale, and the factor $1/(q_{99.9} - q_{50})$ keeps the prior on the normalised image the
  same at every count (the likelihood is close to homogeneous). `fit` raises an error when
  $q_{99.9} \le q_{50}$; pass `tv` and `hessian` then. The ratio
  $\kappa_{\mathrm{TV}}/\kappa_{\mathrm{H}} = 0.4$ is within the flat optimum of a pilot
  grid of both weights, and the scale was calibrated for the stopping rule above on
  simulated images (a Gaussian PSF of standard deviation 1.6 on $9 \times 9$ pixels, blurred
  on a larger field and cropped): cells, filaments and their mixture; peaks of 12 ($\sigma^2 = 9$),
  30 ($\sigma^2 = 16$) and 60 ($\sigma^2 = 36$) electrons on a smooth background (0 or 1/2 of
  the peak at peaks 12 and 60, 0 or 1/3 at peak 30); $32 \times 32$ (two seeds) and
  $64 \times 64$ (one seed) pixels; the 47 of the 54 cases below the limit of $10^4$ (all 18 at
  peaks 12 and 60, 11 of 18 at peak 30), against seven scales of the weights from 1/4 to 4
  times the default. The
  default SNR is on average 0.23 dB below that of the best scale (at most 1.5 dB) and 2.4 to
  11 dB (6.6 dB on average) above that of the data. In two thirds of these runs the 5000
  iterations end before the tolerance is met, which the calibration includes.
- **Step collapse.** The step $\gamma \le 1/\mu$ and $\mu$ grows like
  $e^{2 \max\tilde{y} / \tilde\sigma^2}$, so that at high counts the step is tiny: an iteration
  moves $\mathbf{x}$ by about $\gamma$ per unit of gradient, and the number of iterations
  needed grows with $(\max\tilde{y} + \tilde\sigma^2)/\gamma$. The method suits low counts
  with a high read noise. When $(\max\tilde{y} + \tilde\sigma^2)/\gamma$ exceeds $10^4$ (a
  limit that is not in the paper), `fit` issues a `RuntimeWarning` that names $\mu$ (Eq. 2.13)
  and $\gamma$, warns that the run is likely to stop early, far from the minimiser, and points
  to [Ikoma](ikoma.md), whose shifted-Poisson approximation of the same noise model has no
  such limit; the run then goes ahead as in the paper. `on_step_collapse="raise"` makes it a
  `ValueError`. When $\mu$ overflows float64 there is no step left and `fit` raises in both
  modes. Of the paper's four settings, $\bar{x}_1$ (peak 12, $\sigma^2 = 9$, about
  $6 \cdot 10^3$) and $\bar{x}_4$ (peak 60, $\sigma^2 = 36$, about $2 \cdot 10^3$) are below the
  limit; $\bar{x}_2$ (peak 30, $\sigma^2 = 12$) and $\bar{x}_3$ (peak 20, $\sigma^2 = 9$) are
  above it, at about $5 \cdot 10^4$, and give the warning. Passing `step` skips the check.
- **Other data in `apply`.** The step is admissible for the data seen by `fit`. `apply`
  compares $\max_i \log\eta_i(0)$ of each block with that of the fitted image and issues a
  `RuntimeWarning` when it is larger, since the step is then no longer admissible; fit on
  those data instead.
- **Camera.** The conversion to electrons with the sensor and shading estimates is not in the
  paper. The dark field replaces the offset when a shading estimate is given, and a per-frame
  baseline of the shading estimate is ignored, with a warning.
- **Halo.** Twice the PSF size.

!!! warning "Limitations"
    The stop is a test of the relative change of $\mathbf{x}$ between two iterations, and
    that change is about $\gamma$ times the gradient: it depends on the step. At large
    intensity scales $\gamma$ is small, so a run can stop on the tolerance while still far
    from the minimiser; and below the limit of $10^4$ it can still end after 5000 iterations
    before the tolerance is met (as in two thirds of the calibration runs). Read
    `history.extras["residual"]`, which does not depend on the step, and set `tol=0` with a
    larger `max_iter` when the minimiser is wanted. The method is also slow: thousands of
    iterations are typical, each evaluating the series twice per pixel, about 10 to 15 ms per
    iteration for a $48 \times 48$ image and about 0.2 s for a $48^3$ volume on four CPU
    threads, so that 5000 iterations take about a minute in 2-D and about 16 minutes for a
    $48^3$ volume.

## Memory and streaming

Large images are cut into blocks with a halo and each block is solved as an image of its
own, as for [Ikoma](ikoma.md#memory-and-streaming) and
[Richardson–Lucy](richardson-lucy.md#memory-and-streaming) (`blocks`, halo streaming,
`PaddedConvolution`). `fit` makes one pass over the data to find, for every image,
$\max\tilde{y}$, the mean, the 99.9th percentile (exactly, by keeping the largest 0.1 %
of its values), the median (from a uniform sample of 8192 pixels) and
$\max_i \log\eta_i(0)$, from which the weights, $\mu$ and $\gamma$ follow, so that every
block of an image shares them. `apply` makes a second pass.

The memory budget of a block counts two parts. The solver holds about $16 + 5.5C$ buffers of
the working dtype per voxel of the FFT grid, with $C = d + d(d+1)/2$ the components of the
two penalties (43.5 for 2-D, $C = 5$; 65.5 for 3-D, $C = 9$). The series are evaluated in
float64 whatever the dtype, and their temporaries take about 20 float64 buffers (160 bytes)
per voxel of the loaded block. With the 4 buffers of the loaded data, the sum bounds the
measured peak of `apply` in float32 and float64, in 2-D and 3-D. The grid of a block is the
loaded block (core and halo) plus the PSF, rounded up. The block size is chosen from the memory
budget (`optix.config(memory=...)`), or set with `block`; the rules of
[Ikoma](ikoma.md#memory-and-streaming) for a budget that leaves a small core apply. Each
iteration evaluates the series twice, over every pixel, in float64 whatever the dtype.

Each block is solved on its own, so the block-wise result differs from the whole-image
one near the seams, and the difference fades with the halo width. A `RuntimeWarning`
says when an image is split.

## Usage

The example blurs a test pattern with a Gaussian PSF from
[`optix.models`](../../reference/models.md), adds Poisson counts and Gaussian read noise
of variance 9, and deconvolves them with the TVH and TV priors; [Ikoma](ikoma.md) is run for
comparison. The iterations are capped at 1000 to keep the example short; the default is
5000. The cap is not in the paper, so the estimate stays faithful.

```pycon
>>> import torch
>>> from optix.deconv import Chouzenoux, Ikoma
>>> from optix.estimates import SensorEstimate
>>> from optix.models import gaussian_psf, simulate_blur
>>> _ = torch.manual_seed(0)

>>> # 1. A Gaussian PSF, a test pattern, and electrons with read noise (σ² = 9).
>>> psf = gaussian_psf((7, 7), sigma=1.5)
>>> x = torch.full((32, 32), 1.0)
>>> x[8:24, 8:24] = 8.0
>>> x[14:18, 14:18] = 16.0
>>> data = torch.poisson(simulate_blur(x, psf)) + 3 * torch.randn(32, 32)
>>> sensor = SensorEstimate(
...     offset=torch.tensor(0.0),
...     gain=torch.tensor(1.0),
...     variance=torch.tensor(9.0),
... )

>>> # 2. Fit and deconvolve with TVH, then with TV only; Ikoma for comparison.
>>> est = Chouzenoux(psf, max_iter=1000).fit(data, init=sensor)
>>> x_tvh = est.apply(data)
>>> tv = Chouzenoux(psf, max_iter=1000, hessian=0.0).fit(data, init=sensor)
>>> x_tv = tv.apply(data)
>>> x_ik = Ikoma(psf).fit(data, init=sensor).apply(data)

>>> # 3. The weights and the step that were used, and the mean absolute error.
>>> [round(float(v), 4) for v in (est.tv, est.hessian, est.step)]
[0.01, 0.0249, 0.0371]
>>> round(float(est.lipschitz), 1), est.history.extras["iterations"][0]
(18.4, 1000.0)
>>> mae = lambda e: round(float((e - x).abs().mean()), 2)
>>> [mae(e) for e in (data, x_tvh, x_tv, x_ik)]  # data, TVH, TV, Ikoma
[2.82, 0.95, 1.05, 1.13]
>>> float(x_tvh.min()) >= 0, est.is_faithful  # max_iter keeps it faithful
(True, True)

>>> # 4. The high-count guard: the same scene at 20 times the counts gives a
>>> # warning that points to Ikoma, or an error with on_step_collapse="raise".
>>> import warnings
>>> big = torch.poisson(simulate_blur(20 * x, psf)) + 3 * torch.randn(32, 32)
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     _ = Chouzenoux(psf).fit(big, init=sensor)
>>> ["Use Ikoma" in str(w.message) for w in caught]
[True]
>>> try:
...     Chouzenoux(psf, on_step_collapse="raise").fit(big, init=sensor)
... except ValueError as e:
...     print("Use Ikoma" in str(e))
True
```

Common variations: `Chouzenoux(psf, tv=0.02, hessian=0.0)` or `hessian=0.05` to set the
weights, then `fit` again to see their effect (the values in use are `est.tv` and
`est.hessian`); `upper=60.0` to bound the image from above; `tol=0` to run all
`max_iter` iterations; `fit(data, init=[sensor, shading])` with a `ShadingEstimate`; a
3-D PSF and a `Layout` with three spatial axes for a focal stack; a `zarr` array as
input and `out`, processed block by block. The function `chouzenoux(data, psf, sensor=...)`
is a one-line shortcut. See the [API reference](../../reference/deconv.md).

The implementation is checked against the paper: the ratio $\xi$ against a brute-force
sum of the series; the likelihood, its gradient and the constant $\mu$ against
Eqs. 2.7–2.13, with the quadratic continuation; Algorithm 1 written out line by line,
which must give the same estimate as `Chouzenoux`; the step against Theorem 3.2 (iii);
`primal_dual` against the closed-form Tikhonov solution and against `admm_fft` on a
problem both solve (a shifted-Poisson fidelity with TV, Hessian and box terms); the
optimality conditions of the solution; the default weights across the intensity ranges
of the calibration; and blocks, float32, sensor units and variance maps.

## References

!!! references "References"

    1. Chouzenoux E, Jezierska A, Pesquet J-C, Talbot H. *A convex approach for image
       restoration with exact Poisson–Gaussian likelihood.* SIAM J Imaging Sci
       **8**(4):2662-2682, 2015.
       [doi:10.1137/15M1014395](https://doi.org/10.1137/15M1014395)
    2. Combettes PL, Pesquet J-C. *Primal-dual splitting algorithm for solving
       inclusions with mixtures of composite, Lipschitzian, and parallel-sum type
       monotone operators.* Set-Valued Var Anal **20**(2):307-330, 2012.
       [doi:10.1007/s11228-011-0191-y](https://doi.org/10.1007/s11228-011-0191-y)

    ??? quote "BibTeX"

        ```bibtex
        @article{chouzenoux2015convex,
          author  = {Chouzenoux, Emilie and Jezierska, Anna and Pesquet, Jean-Christophe and Talbot, Hugues},
          title   = {A Convex Approach for Image Restoration with Exact {Poisson}--{Gaussian} Likelihood},
          journal = {SIAM Journal on Imaging Sciences},
          volume  = {8},
          number  = {4},
          pages   = {2662--2682},
          year    = {2015},
          doi     = {10.1137/15M1014395},
        }
        @article{combettes2012primal,
          author  = {Combettes, Patrick L. and Pesquet, Jean-Christophe},
          title   = {Primal-Dual Splitting Algorithm for Solving Inclusions with Mixtures of Composite, {Lipschitzian}, and Parallel-Sum Type Monotone Operators},
          journal = {Set-Valued and Variational Analysis},
          volume  = {20},
          number  = {2},
          pages   = {307--330},
          year    = {2012},
          doi     = {10.1007/s11228-011-0191-y},
        }
        ```
