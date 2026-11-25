# ROF total-variation denoising

!!! info "At a glance"
    - **Estimates:** denoised image (or volume), same shape as the data
    - **Noise model:** Gaussian (white, known $\sigma$); Poisson–Gaussian through the generalised Anscombe transform or a Poisson data term
    - **Streaming:** per-frame (each image is solved whole)
    - **Needs many frames:** no
    - **Reference:** Rudin, Osher & Fatemi, Physica D, 1992, [doi:10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F)

## Overview

`ROF` removes noise from an image by total-variation (TV) denoising
([Rudin, Osher & Fatemi 1992](https://doi.org/10.1016/0167-2789(92)90242-F)). Among
all images that lie within the noise level of the data, it returns the one with the
smallest total variation, the sum over the pixels of the length of the gradient. Noise
adds a great deal of variation to an image, and edges add little, so the result is flat
where the data are flat, and keeps its sharp edges; the price is that gradual ramps turn
into small steps (the *staircase* effect) and fine texture of low contrast is smoothed away.
The method knows the noise level, not the amount of smoothing: it is given the standard
deviation $\sigma$ of the noise and finds how strongly to smooth by itself, through a
Lagrange multiplier that it also returns.

Use it when the noise is white and Gaussian with a known standard deviation, or when the
data are Poisson–Gaussian (a camera) and a [`SensorEstimate`](../../reference/estimates.md)
is available, and the object is made of regions of nearly constant intensity bounded by
sharp edges. It is a local, edge-preserving smoother, where [BM3D and BM4D](bm3d.md) also
exploit similar structures that recur in the image; its first-order solver needs a few
hundred iterations per image (see [Slow convergence](#slow-convergence)). It is not a deconvolution: it does not know about the optics
(see [Richardson–Lucy with TV](../deconv/richardson-lucy-tv.md) and
[Chouzenoux](../deconv/chouzenoux.md) for TV-regularised deconvolution). The noise must be
uncorrelated between pixels.

The estimator is [`ROF`][optix.denoise.ROF], with the function [`rof`][optix.denoise.rof]
as a shortcut and the solver [`tv_denoise`][optix.denoise.utils.tv_denoise], which also
returns the multiplier and the state of the solver (see the [API reference](../../reference/denoise.md)).
`fit` fixes the noise model and `apply` denoises, each image of a stack on its own. The
page also documents the options that go beyond the paper: a penalised data term, a Poisson
data term, pixel and axis weights, and 3-D total variation; they are listed apart from the
faithful method in [Faithful mode and extensions](#faithful-mode-and-extensions).

## Image formation (optics)

Only the noise is modelled; the optics are not (see
[Image formation and noise](../../concepts/image-formation.md)). The recorded image is the
latent image plus white Gaussian noise of zero mean and known standard deviation:

=== "optix"

    $$
    \mathbf{y} = \mathbf{x} + \boldsymbol{\varepsilon},
    \qquad
    \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
    $$

=== "Rudin et al. 1992"

    $$
    u_0(x, y) = u(x, y) + n(x, y),
    \qquad
    \int_\Omega n \,\mathrm{d}x\,\mathrm{d}y = 0,
    \quad
    \int_\Omega n^2 \,\mathrm{d}x\,\mathrm{d}y = \sigma^2
    $$

| optix | Rudin et al. 1992 | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}$ | $u_0$ | noisy image or volume | $\mathbb{R}^{\lvert\Omega\rvert}$, per image |
| $\mathbf{x}$ | $u$ | latent (noise-free) image, and its estimate $\hat{\mathbf{x}}$ | $\mathbb{R}^{\lvert\Omega\rvert}$, per image |
| $\boldsymbol{\varepsilon}$ | $n$ | white Gaussian noise | $\mathbb{R}^{\lvert\Omega\rvert}$, per image |
| $\sigma$ | $\sigma$ | noise standard deviation, in the units of the data | scalar, one per call (one per fit) |
| $\Omega$, $M$ | $\Omega$ | pixel grid, and its number of pixels $M = \lvert\Omega\rvert$ | $M = HW$ or $DHW$ |

The paper writes the noise statistics for a domain $\Omega$ of unit area (the grid of
Eq. 2.7a has $Nh = 1$ and the noise $\sigma\varphi$ of Eq. 2.7d has an $L^2$ norm
$\sigma$), so that its integrals are the averages over the $M$ pixels, and the
constraint on $\int(u - u_0)^2$ is $\sum_j (x_j - y_j)^2 = M\sigma^2$ with unit pixel
spacing. Images of a stack, and groups, are denoised one by one, each with the same
$\sigma$.

Poisson–Gaussian data, the camera model $y = g\,e + o + \varepsilon$ with
$e \sim \mathrm{Poisson}(\lambda)$, are brought to this model by the generalised Anscombe
transform of [Mäkitalo & Foi (2013)](https://doi.org/10.1109/TIP.2012.2202675) (see the
[page of the transform](../sensor/generalized-anscombe.md)): the noise of the transformed
image is approximately $\mathcal{N}(0, 1)$, whatever the signal, so the data are denoised with
$\sigma = 1$. The paper does not treat this noise; the chaining is an application of
the transform, not part of the method (see [`SensorEstimate` handling](#sensorestimate-handling)).

## Probabilistic model

TV denoising is not derived from a probabilistic model; it is a constrained optimisation
problem. The statistical reading of its penalised form (below) is a maximum a posteriori
estimate under Gaussian noise with a prior that charges the total variation of the image,
$p(\mathbf{x}) \propto \exp(-\beta\,\mathrm{TV}(\mathbf{x}))$; the constraint plays the role of Morozov's
discrepancy principle, which picks the amount of regularisation so that the residual has the
size of the noise.

**The total variation.** The image is smooth (a function of bounded variation) between
sharp edges, and the measure of its variation is the integral of the length of its
gradient, discretised by forward differences with a Neumann boundary: the difference along
axis $a$ is $(D_a\mathbf{x})_j = x_{j + \mathbf{e}_a} - x_j$, and zero at the last pixel of the
axis (the image is extended by replicating its border, Eqs. 2.8b and 2.9a of the paper).

=== "optix"

    $$
    \mathrm{TV}(\mathbf{x}) = \sum_{j \in \Omega} \lVert(\nabla\mathbf{x})_j\rVert_2
    = \sum_{j \in \Omega} \Bigl(\sum_{a} (D_a\mathbf{x})_j^2\Bigr)^{1/2}
    $$

=== "Rudin et al. 1992"

    $$
    \int_\Omega \sqrt{u_x^2 + u_y^2}\,\mathrm{d}x\,\mathrm{d}y
    \;\approx\;
    h \sum_{i,j} \sqrt{(\Delta^x_+ u_{ij})^2 + (\Delta^y_+ u_{ij})^2}
    $$

with the pixel spacing $h = 1$ in optix (the differences $\Delta_+$ of Eq. 2.9a are not divided
by $h$, so the area $h^2$ of a pixel and the $1/h$ of the gradient leave one factor $h$, as in
Eq. 2.9c). The sum over $a$ runs over the axes of the image:
two for an image, three for a volume (an extension, see below). This is the isotropic
total variation; the gradient is not smoothed at zero.

**The constrained model.** The method minimises the total variation of the image subject to
the mean and the standard deviation of the noise:

=== "optix"

    $$
    \hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \; \mathrm{TV}(\mathbf{x})
    \quad\text{subject to}\quad
    \mathbf{1}^\top\mathbf{x} = \mathbf{1}^\top\mathbf{y},
    \qquad
    \lVert\mathbf{x} - \mathbf{y}\rVert_2^2 \le M\sigma^2
    $$

=== "Rudin et al. 1992"

    $$
    \begin{aligned}
    &\text{minimise}\quad \int_\Omega \sqrt{u_x^2 + u_y^2}\,\mathrm{d}x\,\mathrm{d}y
    &&\text{(2.3a)}\\
    &\text{subject to}\quad \int_\Omega u\,\mathrm{d}x\,\mathrm{d}y = \int_\Omega u_0\,\mathrm{d}x\,\mathrm{d}y
    &&\text{(2.3b)}\\
    &\phantom{\text{subject to}}\quad \int_\Omega \tfrac12 (u - u_0)^2\,\mathrm{d}x\,\mathrm{d}y = \sigma^2
    &&\text{(2.3c)}
    \end{aligned}
    $$

The mean constraint (2.3b) needs no treatment: the paper drops it because its evolution
preserves the mean, and here the total variation is unchanged by an offset while the mean of
$\mathbf{y}$ minimises the residual, so the solution has the mean of the data (for pixel
weights equal to one). The noise constraint is the subject of the next two paragraphs.

**The factor of 2 in the constraint.** The paper states the noise constraint three
times, and not in the same way. Eq. 2.2c, for the first (quadratic) model, and the initial
data of Eq. 2.7d, built so that "the constraints are both satisfied initially", have
$\int(u - u_0)^2 = \sigma^2$. Eq. 2.3c has $\int\tfrac12(u - u_0)^2 = \sigma^2$, and the
multiplier of Eq. 2.6 has the factor $1/(2\sigma^2)$ that goes with it. The two differ by a
factor 2 in the variance that the residual is allowed to have. optix follows the first:
the residual is the noise, with variance $\sigma^2$, so that the constraint is
$\sum_j (x_j - y_j)^2 \le M\sigma^2$, which is the one that matches the model of the data
($\mathbb{E}\lVert\boldsymbol\varepsilon\rVert^2 = M\sigma^2$). Taking Eq. 2.3c literally allows a residual of variance
$2\sigma^2$ and smooths more than the noise warrants. The multiplier that the code returns
is accordingly that of the constraint without the factor: it is the factor $1/\sigma^2$, not
$1/(2\sigma^2)$, in Eq. 2.6.

**The equality relaxed to an inequality.** The constraint of the paper is an equality.
optix writes $\le$, a convex set. The two problems have the same solution unless
$M\sigma^2$ is at least $\lVert\mathbf{y} - \bar{y}\mathbf{1}\rVert^2$, the energy of the data around
their mean: then the constant image $\bar{y}\mathbf{1}$ is inside the ball and has the
smallest possible total variation, 0, whereas the equality has no solution of low
variation. In that case the minimiser is **not unique**: every constant $c\mathbf{1}$ with
$\lVert\mathbf{y} - c\mathbf{1}\rVert^2 \le M\sigma^2$ has a total variation of 0. optix checks for it
before iterating and returns the mean $\bar{y}\mathbf{1}$ (the $\omega$-weighted mean with pixel
weights) at once, with $\lambda = 0$, no iteration and a gap of 0. This is not rare: a frame
of pure noise denoised with its true $\sigma$ has $\lVert\mathbf{y} - \bar{y}\mathbf{1}\rVert^2 \approx (M - 1)\sigma^2$,
below $M\sigma^2$ about half of the time. The convex form is what the solver needs.

**The Lagrange multiplier.** The Euler–Lagrange equation of the problem with the
multipliers $\lambda_1$ (mean) and $\lambda_2$ (variance) is Eq. 2.4a, with the Neumann
condition of Eq. 2.4b. Since $\lambda_1 = 0$ here, and writing $\lambda$ for $\lambda_2$:

=== "optix"

    $$
    \lambda(\mathbf{x} - \mathbf{y}) = -\nabla^\top\mathbf{p},
    \qquad
    \mathbf{p}_j = \frac{(\nabla\mathbf{x})_j}{\lVert(\nabla\mathbf{x})_j\rVert_2}
    \ \text{ where } (\nabla\mathbf{x})_j \ne \mathbf{0},
    \quad
    \lVert\mathbf{p}_j\rVert_2 \le 1 \text{ elsewhere}
    $$

=== "Rudin et al. 1992"

    $$
    0 = \frac{\partial}{\partial x}\!\left(\frac{u_x}{\sqrt{u_x^2 + u_y^2}}\right)
    + \frac{\partial}{\partial y}\!\left(\frac{u_y}{\sqrt{u_x^2 + u_y^2}}\right)
    - \lambda_1 - \lambda_2\,(u - u_0)
    \quad\text{in } \Omega,
    \qquad
    \frac{\partial u}{\partial n} = 0 \text{ on } \partial\Omega
    $$

with $\nabla^\top = -\mathrm{div}$ the adjoint of the discrete gradient. This is the
optimality condition of the *penalised* problem

$$
\min_{\mathbf{x}} \; \mathrm{TV}(\mathbf{x}) + \frac{\lambda}{2}\lVert\mathbf{x} - \mathbf{y}\rVert_2^2 ,
$$

so the constrained solution is also the solution of the penalised problem with the weight
$\lambda$ that makes its residual equal to $M\sigma^2$: $\lambda$ is large when the noise is
small. Multiplying by $\mathbf{x} - \mathbf{y}$ and summing, as the paper does to obtain Eq. 2.6, gives

=== "optix"

    $$
    \lambda = -\frac{\langle\nabla^\top\mathbf{p},\ \mathbf{x} - \mathbf{y}\rangle}{\lVert\mathbf{x} - \mathbf{y}\rVert_2^2}
    = -\frac{\mathrm{TV}(\mathbf{x}) - \langle\mathbf{p}, \nabla\mathbf{y}\rangle}{M\sigma^2}
    $$

=== "Rudin et al. 1992"

    $$
    \lambda = -\frac{1}{2\sigma^2}\int_\Omega
    \left[\sqrt{u_x^2 + u_y^2} - \left(\frac{(u_0)_x u_x}{\sqrt{u_x^2 + u_y^2}} + \frac{(u_0)_y u_y}{\sqrt{u_x^2 + u_y^2}}\right)\right]\mathrm{d}x\,\mathrm{d}y
    \qquad\text{(2.6)}
    $$

and `tv_denoise` returns $\lambda$ computed from the first form with the residual actually
reached.

### Symbols of the solver

The solver (below) uses the notation of Chambolle and Pock for a problem
$\min_{\mathbf{x}} G(\mathbf{x}) + F(\mathbf{K}\mathbf{x})$, and the steps are written
$\tau_{\mathrm{p}}$ and $\tau_{\mathrm{d}}$ here to leave $\sigma$ to the noise.

| optix | Chambolle & Pock 2011 | Meaning | Shape |
|---|---|---|---|
| $\mathbf{K}$, $\nabla$ | $K$ | linear operator, here the discrete gradient with the Neumann boundary | $\mathbb{R}^{\lvert\Omega\rvert} \to \mathbb{R}^{d \times \lvert\Omega\rvert}$ |
| $F$ | $F$ | the norm $\sum_j\lVert\cdot_j\rVert_2$ (`GroupL2`), so that $\mathrm{TV}(\mathbf{x}) = F(\nabla\mathbf{x})$ | |
| $G$ | $G$ | the data term: the indicator of the ball $\lVert\mathbf{x} - \mathbf{y}\rVert^2 \le M\sigma^2$ (`L2Ball`) | |
| $\mathbf{v}$ | $y$ | dual variable, one vector of $d$ components per pixel, $\lVert\mathbf{v}_j\rVert_2 \le 1$ | $\mathbb{R}^{d \times \lvert\Omega\rvert}$ |
| $\bar{\mathbf{x}}$ | $\bar{x}$ | extrapolated primal iterate | $\mathbb{R}^{\lvert\Omega\rvert}$ |
| $\tau_{\mathrm{p}}$, $\tau_{\mathrm{d}}$ | $\tau$, $\sigma$ | primal and dual steps | scalars |
| $\theta$, $\gamma$ | $\theta$, $\gamma$ | extrapolation parameter ($1$); strong-convexity modulus of $G$ | scalars |
| $\kappa$ | | scale of the data term in data units, sets the split of the steps | scalar |
| $d$ | | number of axes (2 or 3) | |

## Inference / derivation

The problem is convex, with a non-smooth objective (the total variation) and a constraint.
Its statistical counterpart is the penalised maximum a posteriori problem of the previous
section, with the weight chosen by the discrepancy principle. The paper solves it by
**marching in time** the parabolic equation $u_t = \nabla\!\cdot\!(\nabla u/\lvert\nabla u\rvert) - \lambda(t)(u - u_0)$
(Eqs. 2.5a–c), the gradient-projection method of Rosen: an explicit scheme (Eqs. 2.8a–c) with
minmod-regularised differences, a step restricted by $\Delta t/h^2 \le c$ (Eq. 2.9d), and a
multiplier $\lambda^n$ recomputed at every step from Eq. 2.9c, that "appears to converge as
$t \to \infty$".

optix does not march in time. It solves the same model with the **primal–dual algorithm of
[Chambolle & Pock (2011)](https://doi.org/10.1007/s10851-010-0251-1)**, implemented as
[`chambolle_pock`][optix.optim.chambolle_pock] on the building blocks of
[`optix.optim`](../../concepts/optimisation.md). This is a choice of the implementation, not
of the paper, made for these reasons:

- the solution is the exact minimiser of the discrete convex problem, with a convergence
  theorem (Theorem 1 of Chambolle & Pock; $O(1/N)$ in the ergodic sense), where the
  time-marching scheme is an approximation of a steady state, with a step-size restriction,
  a regularised gradient and no stated stopping rule;
- the noise constraint is a convex set whose projection is explicit, so every iterate
  satisfies it, and $\lambda$ comes out of the dual variable at the end, with no
  estimate of it at every step;
- the gradient is not regularised: the norm $\lVert(\nabla\mathbf{x})_j\rVert_2$ is
  handled by its proximity operator, a projection of the dual variable, which is exact at
  zero gradient;
- convergence is measured by a primal–dual gap, a certified bound on the distance of the
  objective to its minimum, instead of a change of the image;
- the same iteration solves the extensions (penalised and Poisson data terms) by swapping
  the data term $G$.

The problem is written $\min_{\mathbf{x}} G(\mathbf{x}) + F(\nabla\mathbf{x})$, with
$F(\mathbf{z}) = \sum_j\lVert\mathbf{z}_j\rVert_2$ and
$G(\mathbf{x}) = \iota\{\lVert\mathbf{x} - \mathbf{y}\rVert^2 \le M\sigma^2\}$, the indicator of the
ball, equal to 0 inside and $+\infty$ outside. The conjugate $F^*$ is the indicator of
$\{\lVert\mathbf{v}_j\rVert_2 \le 1\ \forall j\}$, so the iteration of Chambolle & Pock (their
Algorithm 1, with $\theta = 1$) is:

=== "optix"

    $$
    \begin{aligned}
    \mathbf{v}_{k+1} &= \mathrm{proj}_{\{\lVert\mathbf{v}_j\rVert \le 1\}}\bigl(\mathbf{v}_k + \tau_{\mathrm{d}}\nabla\bar{\mathbf{x}}_k\bigr)\\
    \mathbf{x}_{k+1} &= \mathrm{proj}_{\{\lVert\mathbf{x} - \mathbf{y}\rVert^2 \le M\sigma^2\}}\bigl(\mathbf{x}_k - \tau_{\mathrm{p}}\nabla^\top\mathbf{v}_{k+1}\bigr)\\
    \bar{\mathbf{x}}_{k+1} &= 2\mathbf{x}_{k+1} - \mathbf{x}_k
    \end{aligned}
    $$

=== "Chambolle & Pock 2011"

    $$
    \begin{aligned}
    y^{n+1} &= (I + \sigma\,\partial F^*)^{-1}(y^n + \sigma K\bar{x}^n)\\
    x^{n+1} &= (I + \tau\,\partial G)^{-1}(x^n - \tau K^* y^{n+1})\\
    \bar{x}^{n+1} &= x^{n+1} + \theta\,(x^{n+1} - x^n)
    \end{aligned}
    $$

The first projection rescales each pixel's vector $\mathbf{v}_j$ to length 1 if it is longer
(`GroupL2` through the Moreau identity, `prox_conjugate`). The second is the radial
projection onto the ball of `L2Ball` ([`project_l2_ball`][optix.optim.project_l2_ball]):
$\mathbf{x} \leftarrow \mathbf{y} + (\mathbf{x} - \mathbf{y})\min\bigl(1, \sqrt{M}\sigma/\lVert\mathbf{x} - \mathbf{y}\rVert\bigr)$.
The iteration converges when $\tau_{\mathrm{p}}\tau_{\mathrm{d}}\lVert\nabla\rVert^2 < 1$ (Theorem 1;
`chambolle_pock` accepts $\le 1$), where
$\lVert\nabla\rVert^2 \le 4d$ for unit axis weights (8 for an image).

**Steps.** The paper of Chambolle & Pock leaves the split of $\tau_{\mathrm{p}}\tau_{\mathrm{d}}$ open. The
primal variable lives on the scale of the data and the dual variable on that of a unit
gradient, so the steps are balanced with the scale $\kappa$ of the data term in data
units: $\tau_{\mathrm{p}} = \kappa/\lVert\nabla\rVert$, $\tau_{\mathrm{d}} = 0.99/(\kappa\lVert\nabla\rVert)$, with
$\kappa = \sigma/(8\sqrt{\bar\omega})$ for the constraint ($\bar\omega$ the mean pixel weight, 1
by default; the factor is the module constant `SPLIT` = 1/8). The split is a choice of optix,
and it makes the iteration independent of the units of the data. The factor 1/8 was chosen
on synthetic scenes (see [Slow convergence](#slow-convergence)): it needs 3 to 8 times fewer
iterations than $\kappa = \sigma$ for the same gap.

**Stopping rule.** The relative primal–dual gap $(P - D)/P$ is evaluated every 10 iterations,
with the primal value $P = \mathrm{TV}(\mathbf{x})$ and the dual value (Section 5 of Chambolle & Pock) of the
dual variable $\mathbf{v}$ of the iteration, $\mathbf{z} = \nabla^\top\mathbf{v}$:

$$
D(\mathbf{v}) = \langle\mathbf{z}, \mathbf{y}\rangle - \sqrt{M}\,\sigma\,\lVert\mathbf{z}\rVert_2 ,
$$

which is the minimum over the ball of $\langle\mathbf{z}, \mathbf{x}\rangle$, and a lower bound on the
optimal value for any $\mathbf{v}$ with $\lVert\mathbf{v}_j\rVert \le 1$. The iteration stops when the gap is at most
`tol` = $10^{-4}$, or after `max_iter` = 2000 iterations, with a `RuntimeWarning` in that
case (see [Slow convergence](#slow-convergence)). These are not in the paper.

After the last iteration, the multiplier is computed from the dual variable by the
first form of the equation for $\lambda$ above.

## Algorithm

**Input:** a noisy image $\mathbf{y} \in \mathbb{R}^{\lvert\Omega\rvert}$ (or volume); the noise standard
deviation $\sigma$ (or a `SensorEstimate`); `max_iter` = 2000, `tol` = $10^{-4}$, check every 10 iterations \
**Output:** the denoised image $\hat{\mathbf{x}}$, and the multiplier $\lambda$ (with `tv_denoise`).

1. **Prepare.** With a `SensorEstimate`, first replace $\mathbf{y}$ by the generalised Anscombe
   transform of the data and take $\sigma = 1$. If a constant is feasible,
   $\lVert\mathbf{y} - \bar{y}\mathbf{1}\rVert^2 \le M\sigma^2$, return $\bar{y}\mathbf{1}$ and $\lambda = 0$.
   Compute $\kappa = \sigma/(8\sqrt{\bar\omega})$ ($\bar\omega = 1$ by default), $\lVert\nabla\rVert$
   (from `Gradient.norm2`), $\tau_{\mathrm{p}} = \kappa/\lVert\nabla\rVert$ and
   $\tau_{\mathrm{d}} = 0.99/(\kappa\lVert\nabla\rVert)$.
2. **Initialise** $\mathbf{x}_0 = \bar{\mathbf{x}}_0 = \mathbf{y}$ and $\mathbf{v}_0 = \mathbf{0}$. The start is the data
   (the paper starts from the data plus noise of the right size, Eq. 2.7d), and it is feasible.
3. **Repeat** for $k = 0, 1, \ldots$ until the relative gap is at most `tol` or `max_iter`
   is reached:
    1. update the dual variable, $\mathbf{v}_{k+1}$, by the projection onto the unit balls;
    2. update the image, $\mathbf{x}_{k+1}$, by the projection onto the noise ball;
    3. extrapolate, $\bar{\mathbf{x}}_{k+1} = 2\mathbf{x}_{k+1} - \mathbf{x}_k$;
    4. every 10 iterations, evaluate the gap $(\mathrm{TV}(\mathbf{x}_{k+1}) - D(\mathbf{v}_{k+1}))/\mathrm{TV}(\mathbf{x}_{k+1})$.
4. **Return** $\hat{\mathbf{x}} = \mathbf{x}_k$ (after the inverse Anscombe transform with a
   `SensorEstimate`), and $\lambda = -\langle\nabla^\top\mathbf{v}, \hat{\mathbf{x}} - \mathbf{y}\rangle / \lVert\hat{\mathbf{x}} - \mathbf{y}\rVert^2$.

Each image of the last `ndim` axes is denoised separately, each with its own iterations.

## Faithful mode and extensions

`faithful=True` (the default) solves the constrained model of Eqs. 2.3a–c for an image
(`ndim=2`), with a known $\sigma$ (`sigma=`). `faithful=False` changes nothing: the
estimator has no recommended settings that differ from the paper, and the extensions below
are chosen by their own keywords. `is_faithful` is `True` as long as none of them is used.

### Choices not in the paper

1. **The solver.** Chambolle & Pock replace the time-marching scheme (see
   [Inference / derivation](#inference-derivation)): the steps $\tau_{\mathrm{p}}$ and $\tau_{\mathrm{d}}$,
   the stopping rule (relative primal–dual gap at most $10^{-4}$, checked every 10
   iterations, at most 2000 iterations: `tol`, `max_iter`) and the start from the data are
   optix choices.
2. **The noise constraint is $\lVert\mathbf{x} - \mathbf{y}\rVert^2 \le M\sigma^2$.** The paper has
   a factor $\tfrac12$ in Eq. 2.3c and not in Eqs. 2.2c and 2.7d (see
   [above](#probabilistic-model)); optix takes the version without it.
3. **Equality relaxed to $\le$.** The solutions are the same, except when a constant image
   is feasible ($\sigma^2$ at least the variance of the data). Then the minimiser is not
   unique (every feasible constant has a total variation of 0), and the mean of the data is
   returned, without iterating.
4. **The discretisation.** The exact isotropic total variation of the forward differences with
   a Neumann boundary, minimised as such, and pixel spacing 1. The paper's scheme
   (Eq. 2.8a) uses a minmod-regularised stencil of the Euler–Lagrange equation, which is
   not the gradient of this functional.
5. **The mean constraint (2.3b)** is not imposed: it holds at the solution for unit pixel
   weights, as the paper notes for its evolution.
6. **The multiplier $\lambda$** is that of the noise constraint as written in item 2.
7. **Poisson–Gaussian data.** Given a `SensorEstimate`, the data are stabilised and
   denoised with $\sigma = 1$, then transformed back. The paper treats additive Gaussian
   noise only; this is an application of the generalised Anscombe transform, not part of the
   method. It leaves `is_faithful` at `True`: the model solved is the paper's, on
   transformed data. `sigma` and a `SensorEstimate` together are refused by `fit`: the
   stabilised data have $\sigma = 1$.

### Extensions

All are off by default and make `is_faithful` `False`.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `lam` | a penalised data term, $\min \mathrm{TV}(\mathbf{x}) + \tfrac\lambda2\sum_j\omega_j(x_j - y_j)^2$, with $\lambda > 0$ given instead of the constraint (`sigma`); the same equation as the multiplier above, with $\lambda$ given | `None` | `None` |
| `data_term` | `"kl"`: the Poisson (Kullback–Leibler) negative log-likelihood, $\min \mathrm{TV}(\mathbf{x}) + \lambda\sum_j\omega_j\bigl(x_j - (y_j + s^2)\log(x_j + s^2)\bigr)$, for counts (shifted Poisson with read variance $s^2$; $x_j \ge -s^2$); needs `lam` | `"gaussian"` | `"gaussian"` |
| `weights` | pixel weights $\omega_j > 0$ in the data term, $\sum_j\omega_j(x_j - y_j)^2 \le M\sigma^2$ or in the penalised terms, for instance inverse variances known per pixel; of shape `(*group, *spatial)` or broadcastable to it | `None` | `None` |
| `axis_weights` | a weight $w_a > 0$ on the difference along each spatial axis, $\mathrm{TV}(\mathbf{x}) = \sum_j(\sum_a (w_a D_a\mathbf{x})_j^2)^{1/2}$, for instance a smaller weight along an anisotropically sampled $z$ axis | `None` (all 1) | `None` |
| `ndim` | 3: three-dimensional total variation of volumes (three spatial axes) | 2 | 2 |
| `max_iter`, `tol` | solver settings (not in the paper) | 2000, $10^{-4}$ | 2000, $10^{-4}$ |
| `inverse` | inverse of the Anscombe transform with a `SensorEstimate`: `"exact"`, `"closed-form"`, `"asymptotic"` or `"algebraic"` | `"exact"` | `"exact"` |

With `lam` and a Gaussian data term the objective is strongly convex, with modulus
$\gamma = \lambda\min_j\omega_j$, and `chambolle_pock` uses its accelerated steps (Algorithm 2 of
Chambolle & Pock, $O(1/N^2)$); the scale is $\kappa = 1/(\lambda\bar\omega)$. For `"kl"` it is
$\kappa = \bar{c}/(\lambda\bar\omega)$, with $\bar{c}$ the mean of $y + s^2$, and the relative gap
is $(P - D)/(P - \min G)$. Values of `sigma` and `lam` together are refused, as is `"kl"`
without `lam`. With $\lambda$ from a constrained solution, the penalised problem has the same
minimiser.

### `SensorEstimate` handling

`fit(init=sensor)` takes a [`SensorEstimate`](../../reference/estimates.md) with its offset
$o$, variance $\sigma_{\mathrm{r}}^2$ and gain $g$ (scalars or maps; all three are required),
for Poisson–Gaussian data. The data are not read by `fit`. Maps apply pixel by pixel, and the map of each group is
selected for its images.

- **Gaussian data term** (the default): the data are stabilised by the
  [generalised Anscombe transform](../sensor/generalized-anscombe.md) $f$, which gives noise
  of unit variance, denoised with $\sigma = 1$ and transformed back by the inverse selected by
  `inverse`, as in [`BM3D`][optix.denoise.BM3D]:
  $\hat{\mathbf{x}} = g\,\mathcal{I}_{\sigma_{\mathrm{r}}/g}(\mathrm{ROF}(f(\mathbf{y}),\ \sigma = 1)) + o$,
  with $\mathcal{I}$ the exact unbiased inverse of
  [Mäkitalo & Foi (2013)](https://doi.org/10.1109/TIP.2012.2202675) by default. With `lam`
  and no `sigma`, the penalised data term applies to the stabilised data.
- **KL data term** (`data_term="kl"`, with `lam`): the data are converted to electrons,
  $(\mathbf{y} - o)/g$, with the read variance $\sigma_{\mathrm{r}}^2/g^2$ of the
  shifted-Poisson likelihood, denoised, and converted back, $g\,\hat{\mathbf{x}} + o$. Then
  `lam` is in electron units.

Without a `SensorEstimate`, `sigma` (or `lam`) is required: `fit` raises an error otherwise.

## Memory and streaming

The tag is `per-frame`, and each image is solved **whole**, with no spatial tiling and no
halo. `apply` reads the data with the [`blocks`](../../reference/io.md) iterator with one block
per image (the full extent of the `ndim` spatial axes; other axes one at a time), denoises it
and writes it to the output with `Chunk.store`. The data can be a zarr array or a memory-mapped
file larger than memory, as long as one image fits; the output cannot be the input (an
in-place call is refused). Images of a stack, and groups, are independent: a map of `weights`
or of a `SensorEstimate` is selected for each group.

The reason for not tiling is that the solution is not local. The noise constraint
$\lVert\mathbf{x} - \mathbf{y}\rVert^2 \le M\sigma^2$ and the multiplier $\lambda$ concern the whole
image: a tile solved on its own would satisfy its own budget, which is another model (a noise
budget per tile), and its multiplier would differ from tile to tile, so that the smoothing would
change at the seams. And the minimiser of the total variation has no finite range of dependence:
the value of a pixel in a large flat region depends on the data of the whole region (the
constraint couples all pixels), so no halo would give the whole-image result, as it does for
[BM3D](bm3d.md#memory-and-streaming). Tiling is left to the user, who then accepts the
change of model.

One image needs about $2d + 5$ times its own size ($d$ the number of spatial axes), measured
at the peak of the solver in float32: 7.3 times on a $2048^2$ image and 10.3 times on a
$64 \times 256^2$ volume. That is the image, its iterate and extrapolation, the dual variable
(with one component per axis) and temporaries for the operators; the sums of the gap are
accumulated in float64 without double-precision copies of the fields. The KL data term
evaluates its gap in float64 and needs about three times as much (26 times the image in 2-D).
`apply` raises a `MemoryError` when this estimate exceeds the memory budget of
[`optix.config`](../../reference/configuration.md) (`memory=`, 256 MiB by default, so about $2400^2$
pixels in float32 in 2-D). The cost of an iteration is linear in the number of pixels, with
$2d$ differences. The computation is vectorised in torch and runs on any device.

### Slow convergence

First-order methods on the total variation converge slowly: the constrained problem has no
strong convexity ($G$ is an indicator), so the rate is $O(1/N)$. In practice the split of the
steps matters more than the size of the image. With $\kappa = \sigma$ (equal scaling of the
primal and dual steps in data units) the iterations needed grow with the size of the image and
the noise level, up to 7790 for a $256^2$ image at $\sigma = 25$; with the default
$\kappa = \sigma/8$ they stay below 600. Iterations needed for a relative gap of $10^{-4}$, by
the factor of the split, on the scenes of the [results](#results-on-synthetic-data)
($128^2$; float64, one realisation of the noise each), and on a $32 \times 64^2$ volume of
cells in 3-D:

| Scene, $\sigma$ | $\kappa = \sigma$ | $\sigma/4$ | $\sigma/8$ (default) | $\sigma/16$ |
|---|---|---|---|---|
| cells, 10 | 1310 | 360 | 360 | 500 |
| cells, 25 | 2710 | 700 | 450 | 610 |
| cells, 50 | 4080 | 1070 | 530 | 660 |
| filaments, 10 | 720 | 190 | 150 | 190 |
| filaments, 25 | 1610 | 420 | 220 | 250 |
| filaments, 50 | 3110 | 810 | 400 | 310 |
| cells 3-D, 10 | 1670 | 440 | 340 | 440 |
| cells 3-D, 25 | 2950 | 790 | 430 | 510 |

With the default split the count hardly depends on the size of the image: on the cells scene
scaled with the image (radius $10 \times$ side$/128$), float32:

| Image | $\sigma = 10$ | $\sigma = 25$ | $\sigma = 50$ |
|---|---|---|---|
| $64^2$ | 360 | 450 | 490 |
| $128^2$ | 370 | 470 | 550 |
| $256^2$ | 340 | 420 | 500 |

The default `max_iter=2000` is therefore rarely reached. When it is, the returned image is the
last iterate, which still satisfies the noise constraint, and `rof` and `ROF.apply` emit one
`RuntimeWarning` per call with the number of images concerned and the worst gap
(`tv_denoise` warns too, and reports the gap in `TVResult.gap`). `ROF` also appends the
iterations and the final gap of every image to `history.extras["iterations"]` and
`history.extras["gap"]`. `max_iter` and `tol` are the knobs: raise `max_iter`, or relax `tol`
to go faster. A tolerance of $10^{-4}$ leaves differences of 0.1 to 0.5 grey levels (on a
range of 255) from a solution converged to $2 \times 10^{-6}$, with the same PSNR to 0.01 dB.
The penalised model (`lam`) is strongly convex and uses accelerated steps, but it is not faster
here: 630 iterations for the $128^2$ cells image with $\lambda = 0.017$, close to the
multiplier of the constrained solution at $\sigma = 25$. When many images of one kind are
denoised, the multiplier of one of them (`tv_denoise(...).lam`) can be reused with `lam=`, which
is a different, penalised, model.

## Usage

The first example denoises a synthetic image of cells with Gaussian noise of standard
deviation 20 on a range of 200, with `tv_denoise`, which returns the multiplier. The solution
has the residual that the constraint prescribes, and the mean of the data:

```pycon
>>> import torch
>>> from optix.denoise import ROF, rof
>>> from optix.denoise.utils import tv_denoise
>>> from optix.models import phantoms
>>> _ = torch.manual_seed(0)
>>> x = phantoms.cells((64, 64), 8, radius=8.0, ndim=2, seed=0)
>>> x = 200 * x / x.max()
>>> y = x + 20 * torch.randn_like(x)
>>> def psnr(a):
...     return round(float(10 * torch.log10(255**2 / (a - x).square().mean())), 1)
>>> psnr(y)
22.2
>>> res = tv_denoise(y, sigma=20)
>>> psnr(res.x)
28.2
>>> res.gap < 1e-4, res.n_iter < 2000
(True, True)
>>> abs(float((res.x - y).square().mean().sqrt()) - 20) < 0.05  # Mσ²
True
>>> abs(float(res.x.mean() - y.mean())) < 1e-2  # Eq. 2.3b
True
>>> 0.02 < res.lam < 0.03
True
```

The multiplier is that of the penalised problem: solving it with `lam=` gives the same image
(to a fraction of a grey level, for a range of 200). The estimator and the function are the
same computation:

```pycon
>>> pen = tv_denoise(y, lam=res.lam)
>>> float((pen.x - res.x).abs().max()) < 1
True
>>> est = ROF(sigma=20).fit()
>>> est.is_faithful, ROF(lam=0.03).is_faithful
(True, False)
>>> den = est.apply(y)
>>> (
...     bool((den - res.x).abs().max() < 1e-2),
...     bool((rof(y, 20) - res.x).abs().max() < 1e-2),
... )
(True, True)
```

Poisson–Gaussian data: a camera with gain 2, offset 100 and read noise 3 ADU, looking at
1 to 21 electrons. The estimator is given the `SensorEstimate` and denoises after the
Anscombe transform; no `sigma` is needed.

```pycon
>>> from optix.estimates import SensorEstimate
>>> from optix.models import simulate_sensor
>>> sensor = SensorEstimate(offset=100.0, variance=9.0, gain=2.0)
>>> rate = 1 + 0.1 * x
>>> z = simulate_sensor(rate, sensor, seed=0)
>>> den = ROF().fit(z, init=sensor).apply(z)
>>> truth = 2.0 * rate + 100.0
>>> rmse = lambda a: float((a - truth).square().mean().sqrt())
>>> rmse(den) < rmse(z) / 2
True
```

The extensions: a volume denoised in 3-D, with a smaller weight on the first (anisotropic) axis,
and Poisson counts with the Kullback–Leibler data term. In the volume, the 3-D total variation
uses the neighbours along the first axis that a slice-by-slice (2-D) denoising ignores.

```pycon
>>> vol = phantoms.cells((16, 32, 32), 5, radius=6.0, ndim=3, seed=0)
>>> vol = 100 * vol / vol.max()
>>> noisy = vol + 10 * torch.randn_like(vol)
>>> rmse3 = lambda a: float((a - vol).square().mean().sqrt())
>>> by_slice = ROF(sigma=10).fit().apply(noisy)
>>> full = ROF(sigma=10, ndim=3).fit().apply(noisy)
>>> weighted = ROF(sigma=10, ndim=3, axis_weights=(0.5, 1, 1)).fit().apply(noisy)
>>> rmse3(full) < rmse3(by_slice) < rmse3(noisy) / 2
True
>>> ROF(sigma=10, ndim=3).is_faithful
False
>>> mean = 20 * x / 200 + 2  # counts, 2 to 22
>>> counts = torch.poisson(mean)
>>> kl = tv_denoise(counts, lam=3.0, data_term="kl")
>>> mse = lambda a: float((a - mean).square().mean())
>>> kl.gap < 1e-4, mse(kl.x) < mse(counts) / 2
(True, True)
```

Common variations: `layout=` to say which axes are spatial when the array has frames or
channels (every image of a stack is solved on its own); `weights=` with a map of the shape of the
image; a large zarr array in place of the tensor, with `out=` a separate array; `max_iter=` and
`tol=` for the solver (see [Slow convergence](#slow-convergence)). A fitted estimator can be
saved and loaded like any other.

### Results on synthetic data

PSNR of the estimates of two synthetic $128^2$ scenes with a range of $[0, 255]$ (cells with a
radius of about 10 pixels, and filaments, `optix.models.phantoms`), one realisation of Gaussian
noise each, in dB, with the default solver settings (float32). The noise constraint holds: the
rms residual is $\sigma$ to four digits in every case.

| Scene | $\sigma$ | Noisy | `ROF` | Iterations | Gap |
|---|---|---|---|---|---|
| cells | 10 | 28.11 | 36.24 | 370 | $9.2 \times 10^{-5}$ |
| cells | 25 | 20.15 | 30.20 | 470 | $8.9 \times 10^{-5}$ |
| cells | 50 | 14.13 | 25.93 | 550 | $9.6 \times 10^{-5}$ |
| filaments | 10 | 28.11 | 33.06 | 140 | $9.6 \times 10^{-5}$ |
| filaments | 25 | 20.15 | 27.18 | 210 | $9.5 \times 10^{-5}$ |
| filaments | 50 | 14.13 | 23.31 | 340 | $9.8 \times 10^{-5}$ |

Every case reaches the gap tolerance well before `max_iter`. Solving the same noisy images in
float64 takes the same number of iterations, gives the same PSNR to 0.01 dB, and an image that
differs from the float32 one by less than $3 \times 10^{-4}$ grey levels. With `ROF` on a camera
image, the root mean square error of the Anscombe-stabilised estimate is less than half that of
the data (see the example above). These numbers describe the synthetic scenes only, not the
images of the paper, and no comparison with the figures of the paper or with another
implementation is given.

## References

!!! references "References"

    1. Rudin LI, Osher S, Fatemi E. *Nonlinear total variation based noise removal
       algorithms.* Physica D **60**(1–4):259–268, 1992.
       [doi:10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F)
    2. Chambolle A, Pock T. *A first-order primal-dual algorithm for convex problems
       with applications to imaging.* J Math Imaging Vis **40**(1):120–145, 2011.
       [doi:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)
    3. Mäkitalo M, Foi A. *Optimal inversion of the generalized Anscombe
       transformation for Poisson-Gaussian noise.* IEEE Trans Image Process
       **22**(1):91-103, 2013.
       [doi:10.1109/TIP.2012.2202675](https://doi.org/10.1109/TIP.2012.2202675)

    ??? quote "BibTeX"

        ```bibtex
        @article{rudin1992nonlinear,
          author  = {Rudin, Leonid I. and Osher, Stanley and Fatemi, Emad},
          title   = {Nonlinear total variation based noise removal algorithms},
          journal = {Physica D: Nonlinear Phenomena},
          volume  = {60},
          number  = {1--4},
          pages   = {259--268},
          year    = {1992},
          doi     = {10.1016/0167-2789(92)90242-F},
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
        @article{makitalo2013optimal,
          author  = {M{\"a}kitalo, Markku and Foi, Alessandro},
          title   = {Optimal inversion of the generalized {A}nscombe transformation for {P}oisson-{G}aussian noise},
          journal = {IEEE Transactions on Image Processing},
          volume  = {22},
          number  = {1},
          pages   = {91--103},
          year    = {2013},
          doi     = {10.1109/TIP.2012.2202675},
        }
        ```
