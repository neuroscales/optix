# General stripe remover

!!! info "At a glance"
    - **Estimates:** stripe-free image (the stripe component is the difference with the data)
    - **Noise model:** none
    - **Streaming:** per-frame
    - **Needs many frames:** no
    - **Reference:** Rottmayer et al., Opt Express, 2025, [doi:10.1364/OE.542868](https://doi.org/10.1364/OE.542868)

## Overview

[`GeneralStripeRemover`][optix.destripe.GeneralStripeRemover] splits every frame
into a stripe-free image and a stripe component with the variational method of
[Rottmayer et al. (2025)](https://doi.org/10.1364/OE.542868). It was designed
for light-sheet microscopy and tested on FIB-SEM and remote-sensing images as
well. The stripe-free image has few strong edges (total variation), the stripes
change little along their direction and are sparse, and the result stays in the
value range of the input. Two weights, $\mu_1$ (how strongly stripes are
removed) and $\mu_2$ (how much image structure is spared), tune the method; the
stripe direction can be oblique, several directions can be given, and a stack
can be processed as 3-D volumes.

It is an iterative solver rather than a filter, so it is slower than
[Wavelet-FFT](wavelet-fft.md), but it does not need the stripes to span the
whole frame or to have a fixed width. Like every stripe remover, it affects
image structures that are aligned with the stripes. Nothing is fitted: `fit`
only records the layout, and `apply` returns the destriped frames. The function
[`general_stripe_remover`][optix.destripe.general_stripe_remover] does the same
on a tensor without a layout.

## Image formation (optics)

Stripes are modelled as an additive component $\mathbf{s}$ of the recorded
frame, made of elongated structures that point in a common direction (in
light-sheet microscopy, shadows cast along the light sheet by absorbing or
scattering structures). The method assumes nothing about their cause: no noise
model, no shading and no camera model (see
[Image formation and noise](../../concepts/image-formation.md)). It is meant to
be applied to data in which the stripes are the main artefact, and, in a
pipeline, after the corrections for the sensor and for shading.

=== "optix"

    $$
    \mathbf{y} = \mathbf{x} + \mathbf{s},
    \qquad \mathbf{x} \in [0, 1]^{\lvert\Omega\rvert},
    \quad \mathbf{s} \in [-1, 1]^{\lvert\Omega\rvert} ,
    $$

    a frame (or volume) $\mathbf{y} \in [0, 1]^{\lvert\Omega\rvert}$ with
    pixels $i \in \Omega$, rescaled to $[0, 1]$, is the sum of the stripe-free
    image $\mathbf{x}$ and the stripes $\mathbf{s}$, all three of the same
    shape. Frames $n = 1, \ldots, N$ are processed independently, so the index
    $n$ is dropped.

=== "Rottmayer et al. 2025"

    $$
    u_0 = u + s,
    \qquad u \in [0, 1]^N,
    \quad s \in [-1, 1]^N ,
    $$

    the observed image $u_0 \in [0, 1]^N$ is the sum of the clean image $u$ and
    the stripes $s$ (Section 2.1.1).

The form is the same as in [Wavelet-FFT](wavelet-fft.md), $\mathbf{y} = \mathbf{x} + \mathbf{s}$;
the two methods differ in what they assume about $\mathbf{s}$.

## Probabilistic model

There is no noise model. The clean image and the stripes are the minimiser of an
objective made of a term for the image, $D(\mathbf{x})$, and a term for the
stripes, $N(\mathbf{s})$ (Eq. 1 and 2 of the paper), under the constraint
$\mathbf{x} + \mathbf{s} = \mathbf{y}$:

=== "optix"

    $$
    \operatorname*{argmin}_{\mathbf{x} + \mathbf{s} = \mathbf{y}}\;
    \underbrace{\mu_1 \lVert \nabla \mathbf{x} \rVert_{2,1}
    + \iota_{[0,1]^{\lvert\Omega\rvert}}(\mathbf{x})}_{D(\mathbf{x})}
    + \underbrace{\lVert \nabla_\theta \mathbf{s} \rVert_1
    + \mu_2 \lVert \mathbf{s} \rVert_1}_{N(\mathbf{s})},
    \qquad \mu_1, \mu_2 > 0,
    $$

    $$
    \lVert \nabla \mathbf{x} \rVert_{2,1} = \sum_{i \in \Omega}
    \sqrt{(\nabla_1 \mathbf{x})_i^2 + (\nabla_2 \mathbf{x})_i^2
    + (\rho_z \nabla_3 \mathbf{x})_i^2},
    $$

    with $\nabla_1, \nabla_2$ the differences along the two in-plane axes and
    $\nabla_3$ along $z$ (volumes only), so that
    $\nabla \mathbf{x} \in \mathbb{R}^{\lvert\Omega\rvert \times g}$ with
    $g = 2$ components ($g = 3$ for volumes with $\rho_z > 0$).

=== "Rottmayer et al. 2025"

    $$
    \operatorname*{argmin}_{u + s = u_0}\;
    \mu_1 \lVert \nabla u \rVert_{2,1} + \iota_{[0,1]^N}(u)
    + \lVert \nabla_y s \rVert_1 + \mu_2 \lVert s \rVert_1
    \quad \text{(Eq. 1)}
    $$

    $$
    \lVert \nabla u \rVert_{2,1} = \sum_{i,j,k}
    \sqrt{(\nabla_x u)_{i,j,k}^2 + (\nabla_y u)_{i,j,k}^2 + (\nabla_z u)_{i,j,k}^2}
    \quad \text{(Eq. 2)}
    $$

    with the stripes along $y$; $\nabla_z$ is replaced by $\rho_z \nabla_z$, and
    $\nabla_y s$ by $\nabla_\theta s$ for oblique stripes (Section 2.1.1).

| optix | Rottmayer et al. 2025 | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}$ | $u_0$ | recorded frame or volume, rescaled to $[0, 1]$ | $\lvert\Omega\rvert$, one per frame |
| $\mathbf{x}$ | $u$ | stripe-free image | $\lvert\Omega\rvert$, one per frame |
| $\mathbf{s}$ | $s$ | stripe component | $\lvert\Omega\rvert$, one per frame |
| $\lvert\Omega\rvert$ | $N = n_x n_y n_z$ | number of pixels of a frame or volume | scalar |
| $\mu_1$, $\mu_2$ | $\mu_1$, $\mu_2$ | weights of the image and sparsity terms | scalars, shared by all frames |
| $\rho_z$ | $\rho_z$ | weight of the differences along $z$ | scalar, shared by all frames |
| $\theta$, $\nabla_\theta$ | $\theta$, $\nabla_\theta$ | stripe direction and the difference along it | scalar; operator $\mathbb{R}^{\lvert\Omega\rvert} \to \mathbb{R}^{\lvert\Omega\rvert}$ |
| $\nabla$ | $\nabla$ | discrete gradient | operator $\mathbb{R}^{\lvert\Omega\rvert} \to \mathbb{R}^{\lvert\Omega\rvert \times g}$ |
| $\iota_{[0,1]^{\lvert\Omega\rvert}}$ | $\iota_{[0,1]^N}$ | indicator of the value range | function of $\mathbf{x}$ |
| first spatial axis | $y$ | axis along which the stripes run | none |

Each term has a role:

- $\lVert \nabla \mathbf{x} \rVert_{2,1}$ is the total variation. It favours
  images with few strong edges.
- $\lVert \nabla_\theta \mathbf{s} \rVert_1$ favours stripes that change little
  along their direction $\theta$ and a lot across it.
- $\lVert \mathbf{s} \rVert_1$ favours sparse stripes: usually only part of the
  image is affected.
- $\iota_{[0,1]^{\lvert\Omega\rvert}}$ is zero when $\mathbf{x}$ lies in $[0, 1]^{\lvert\Omega\rvert}$ and infinite
  otherwise. It keeps the result in the range of the input.

The weights act as follows (Section 2.1.1): a larger $\mu_1$ removes stripes
more strongly but can affect stripe-like structures and smooth the image; a
larger $\mu_2$ enforces sparser stripes, which spares image structures but
impairs the removal; scaling both by the same factor changes the amount of
removal while mostly keeping the effect on image structures. The weight
$\rho_z \in [0, 1]$ scales the differences along $z$: $\rho_z = 0$ decouples the
slices (2-D processing, as in all the results of the paper), and $0 < \rho_z < 1$
trusts the $z$ axis less, for instance because of a lower axial resolution.

Notation ($\mu_1$, $\mu_2$, $\rho_z$, $\nabla_\theta$, $\lVert\cdot\rVert_{2,1}$, $\iota$)
is added to the [notation page](../../concepts/notation.md#stripes).

## Inference / derivation

There is no likelihood in the usual sense: the data constraint
$\mathbf{x} + \mathbf{s} = \mathbf{y}$ is exact, which amounts to noise-free
data. Read statistically, the minimiser is the maximum a posteriori (MAP)
estimate of $(\mathbf{x}, \mathbf{s})$ given $\mathbf{y}$ under the improper
prior $p(\mathbf{x}, \mathbf{s}) \propto \exp\bigl(-D(\mathbf{x}) -
N(\mathbf{s})\bigr)$: a TV prior on $\mathbf{x}$ restricted to $[0, 1]$, and
Laplace priors on $\nabla_\theta \mathbf{s}$ and on $\mathbf{s}$. The paper
presents it as a variational problem only, and nothing in optix uses this
reading beyond interpretation.

The objective is convex but non-smooth, and not strictly convex, so a minimiser
exists but need not be unique. It is minimised with the primal-dual hybrid gradient method with
extrapolation of the dual variable (PDHGMp) of
[Burger et al. (2016)](https://doi.org/10.1007/978-3-319-41589-5_10), as in the
paper (Section 2.1.1), which converges to one of the minimisers. Writing the objective as a sum of terms that are each
simple after a linear map, $\mathbf{K}(\mathbf{x}, \mathbf{s}) =
(\nabla \mathbf{x},\ \nabla_\theta \mathbf{s},\ \mathbf{s})$, every term has a
cheap proximal operator in its dual form:

| Term | Dual variable | Meaning | Shape | Projection in the dual step |
|---|---|---|---|---|
| $\mu_1 \lVert \nabla \mathbf{x} \rVert_{2,1}$ | $\mathbf{p}_1$ | dual of the image gradient | $\lvert\Omega\rvert \times g$, one per frame | onto the Euclidean ball of radius $\mu_1$, per pixel |
| $\lVert \nabla_\theta \mathbf{s} \rVert_1$ | $\mathbf{p}_2$ | dual of the stripe differences | $\lvert\Omega\rvert$, one per frame and direction | onto $[-1, 1]$, per pixel |
| $\mu_2 \lVert \mathbf{s} \rVert_1$ | $\mathbf{p}_3$ | dual of the stripes | $\lvert\Omega\rvert$, one per frame | onto $[-\mu_2, \mu_2]$, per pixel |

The constraint $\mathbf{x} + \mathbf{s} = \mathbf{y}$ and the indicator are
handled in the primal step by a projection onto
$\{\mathbf{x} + \mathbf{s} = \mathbf{y},\ \mathbf{x} \in [0, 1]^{\lvert\Omega\rvert}\}$: the
residual $\mathbf{y} - \mathbf{x} - \mathbf{s}$ is shared equally between
$\mathbf{x}$ and $\mathbf{s}$, then the part of $\mathbf{x}$ outside $[0, 1]$ is
moved to $\mathbf{s}$. The cost of an iteration is a few differences and
clamps: it is linear in the number of pixels.

The paper names the method and the problem but not the discretisation, the step
sizes or the initialisation. The choices made are listed under
[Faithful mode and extensions](#faithful-mode-and-extensions).

## Algorithm

**Input:** frames $\mathbf{y}_n \in \mathbb{R}^{\lvert\Omega\rvert}$, $n = 1, \ldots, N$
(frames or volumes); weights $\mu_1 = 1/3$, $\mu_2 = 1/300$; direction
$\theta = 0$; $\rho_z = 0$; `max_iter` $= 25000$ (1000 recommended); steps
$\tau = \sigma = 0.35$ ($0.28$ when $\rho_z > 0$); `tol` $=$ `None`.

**Output:** stripe-free frames $\mathbf{x}_n \in \mathbb{R}^{\lvert\Omega\rvert}$.

1. For each frame $n = 1, \ldots, N$ (a batch at once, each with its own
   iterates), with $\mathbf{y} = \mathbf{y}_n$:
    1. **Rescale.** $\mathbf{y} \leftarrow (\mathbf{y} - \min_i y_i) /
       (\max_i y_i - \min_i y_i)$, so that $\mathbf{y} \in [0, 1]^{\lvert\Omega\rvert}$.
    2. **Initialise.** $\mathbf{x} = \mathbf{y}$, $\mathbf{s} = \mathbf{0}$,
       $\mathbf{p}_1 = \mathbf{p}_2 = \mathbf{p}_3 = \mathbf{0}$ and
       $\bar{\mathbf{p}}_j = \mathbf{p}_j$.
    3. For $k = 1, \ldots,$ `max_iter`, or until the duality gap has fallen to
       `tol` times its initial value (extension, checked every 10 iterations):
        1. **Primal step.** $\mathbf{x} \leftarrow \mathbf{x} - \tau \nabla^\top
           \bar{\mathbf{p}}_1$ and $\mathbf{s} \leftarrow \mathbf{s} - \tau
           (\nabla_\theta^\top \bar{\mathbf{p}}_2 + \bar{\mathbf{p}}_3)$.
        2. **Project onto the data constraint.** $\mathbf{r} = (\mathbf{y} -
           \mathbf{x} - \mathbf{s}) / 2$, then $\mathbf{x} \leftarrow
           \mathbf{x} + \mathbf{r}$ and $\mathbf{s} \leftarrow \mathbf{s} +
           \mathbf{r}$.
        3. **Project onto the range.** $\mathbf{x}' = \operatorname{clamp}(\mathbf{x},
           0, 1)$, then $\mathbf{s} \leftarrow \mathbf{s} + (\mathbf{x} -
           \mathbf{x}')$ and $\mathbf{x} \leftarrow \mathbf{x}'$.
        4. **Dual step.** $\mathbf{p}_1^+ = \mathbf{p}_1 + \sigma \nabla
           \mathbf{x}$, rescaled at each pixel $i$ to a norm at most $\mu_1$;
           $\mathbf{p}_2^+ = \operatorname{clamp}(\mathbf{p}_2 + \sigma
           \nabla_\theta \mathbf{s}, -1, 1)$;
           $\mathbf{p}_3^+ = \operatorname{clamp}(\mathbf{p}_3 + \sigma
           \mathbf{s}, -\mu_2, \mu_2)$.
        5. **Extrapolate.** $\bar{\mathbf{p}}_j = 2 \mathbf{p}_j^+ -
           \mathbf{p}_j$ and $\mathbf{p}_j \leftarrow \mathbf{p}_j^+$, for
           $j = 1, 2, 3$.
    4. **Map back.** $\mathbf{x}_n = \min_i y_{n,i} + (\max_i y_{n,i} -
       \min_i y_{n,i})\, \mathbf{x}$.
2. Return $\mathbf{x}_1, \ldots, \mathbf{x}_N$.

With several directions, step 1 sums one term
$\nabla_{\theta_i}^\top \bar{\mathbf{p}}_{2,i}$ per direction and step 4 has one
$\mathbf{p}_{2,i}$ per direction.

## Faithful mode and extensions

`faithful=True` solves Eq. 1 for the parameters $\mu_1$ and $\mu_2$, the stripe
direction and $\rho_z$ that the user gives, with the indicator constraint, for
25000 iterations. These parameters are meant to be tuned to the data
(Section 2.1.1), so any value is faithful; only the extensions below make
`is_faithful` false. The recommended values are the faithful ones, except for the
number of iterations.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `mu1` | strength of stripe removal $\mu_1 > 0$ | 1/3 (see [below](#default-value-of-the-first-weight)) | 1/3 |
| `mu2` | precaution towards image structures $\mu_2 > 0$ | 1/300 | 1/300 |
| `axis` | spatial axis along which stripes run: a position among the spatial axes, or a layout label | first of two (vertical stripes in `(y, x)` frames) | same |
| `angle` | stripe direction $\theta$ in degrees from `axis` towards the other in-plane axis; a sequence for several directions (see [below](#several-stripe-directions)) | 0 | 0 |
| `rho_z` | weight $\rho_z \in [0, 1]$ of the differences along $z$ (three spatial axes) | 0 | 0 |
| `normalize` | rescale each frame to $[0, 1]$ and back | `True` | `True` |
| `max_iter` | number of PDHGMp iterations (see [below](#not-specified-in-the-paper)) | 25000 | 1000 |
| `tol` | **extension:** stop a frame when its relative duality gap falls to `tol`, as for `tol` of `VSNR`; needs `box=True` | `None` | `None` |
| `box` | **extension:** `False` drops the indicator $\iota_{[0,1]^{\lvert\Omega\rvert}}$ | `True` | `True` |

With two spatial axes in the layout, frames are 2-D; with three, the first is
$z$ and each volume is solved as a whole.

With `tol`, a frame stops when $\Delta(\mathbf{x}_k, \mathbf{p}_k) \le
\texttt{tol} \cdot \Delta(\mathbf{x}_0, \mathbf{p}_0)$, where
$\Delta = E(\mathbf{x}) - D(\mathbf{p})$ is the gap between the objective of
Eq. 1 and its dual, and $\Delta_0 = \mu_1 \lVert \nabla \mathbf{y} \rVert_{2,1}$
is its initial value. The dual is bounded only with the indicator, hence
`box=True`. On synthetic $96 \times 96$ frames the relative gap is of the order
of $10^{-1}$ to $10^{-2}$ after 1000 iterations and $10^{-3}$ to $10^{-4}$ after
25000: 1000 iterations are not fully converged.

### Not specified in the paper

The following choices follow the authors' code, which is released under the MIT
licence ([General-Stripe-Removal](https://github.com/NiklasRottmayer/General-Stripe-Removal)),
except where stated, and are marked as such in the docstrings.

- **Step sizes and iterations.** $\tau = \sigma = 0.35$ for 2-D problems and
  $0.28$ when $\rho_z > 0$, the steps of the authors' current code. They are
  scaled down by $\sqrt{b / \lVert \mathbf{K} \rVert^2}$ when a bound on
  $\lVert \mathbf{K} \rVert^2$ (8 in 2-D, 12 in 3-D) is exceeded, which happens
  with several stripe directions, so that
  $\tau \sigma \lVert \mathbf{K} \rVert^2 < 1$ holds. The code used for the
  results of the paper weighted the stripe term by 30, with the same
  $\tau = \sigma = 0.35$: that is PDHGMp on Eq. 1 with $\tau = 10.5$ and
  $\sigma = 0.35 / 30 \approx 0.0117$, and the 25000 iterations of the paper
  (Section 3) refer to that balance. Both converge to a minimiser of Eq. 1,
  along different paths. With the balanced steps the authors suggest about 1000
  iterations, the recommended value; 25000 is kept as the faithful one.
- **Initialisation.** $\mathbf{x} = \mathbf{y}$, $\mathbf{s} = \mathbf{0}$, and all
  dual variables zero.
- **Boundary.** Differences are forward differences that are zero at the last
  pixel along their direction (a Neumann boundary).
- **Discretisation of $\theta$.** $\nabla_\theta$ is a forward difference along
  a lattice offset, and $\theta$ is rounded to the nearest of 0°, ±26.6°, ±45°,
  ±63.4° and 90° (modulo 180°), with offsets $(1, 0)$, $(2, \pm 1)$,
  $(1, \pm 1)$, $(1, \pm 2)$ and $(0, 1)$ along and across the stripe axis.
- **Normalisation.** The paper assumes data in $[0, 1]$. With `normalize`, each
  frame (or volume) is rescaled by its own minimum and maximum, and the result
  is mapped back to the input range. This is a choice of optix: the authors
  rescale the whole image or stack once. With `normalize=False`, the data must
  already lie in $[0, 1]$.
- **Weight $\rho_z$.** $\rho_z$ weights $\nabla_z$ in both $\nabla$ and
  $\nabla^\top$, as Eq. 2 implies. The authors' code applies it in the adjoint
  only, so 3-D results with $0 < \rho_z < 1$ differ from theirs. Stripe
  directions with a $z$ component are not supported.
- **Negative angles.** For $\theta < 0$, the authors' code mirrors the frame
  across the stripe axis and solves at $-\theta$. The stripe term is unchanged
  by the mirroring, but the total variation becomes a backward difference across
  the stripes. optix keeps forward differences for every angle, which gives a
  slightly different result with a lower objective, and keeps this behaviour.
  Flipping the input across the stripe axis, passing $-\theta$ and flipping the
  result back reproduces the authors' result.

### Default value of the first weight

The Discussion of the paper prints $\mu_1 = 1/30$ and $\mu_2 = 1/300$ as the
default values (the second weight is written $\mu$ there). The authors' code at
the time of publication (before the commit "Improved GSR" of 29 July 2025)
weights $\lVert \nabla_y s \rVert_1$ by $\lambda = 30$ and defaults to
$\mu = (10, 0.1)$; its processing script states that, compared to the paper, $\mu$
is rescaled by a factor of 30. In the units of Eq. 1 this is
$\mu_1 = 10/30 = 1/3$ and $\mu_2 = 0.1/30 = 1/300$. The second weight agrees
with the Discussion and the first does not, so $1/30$ is read as a typographical
error for $1/3$. The current code of the authors defaults to the same values
($0.33$ and $0.003$), and `GeneralStripeRemover` uses $1/3$ and $1/300$. A larger
$\mu_1$ removes stripes more strongly.

### Several stripe directions

Section 2.1.1 states that multi-directional destriping is possible "by including
a sum of penalisation terms $\lVert \nabla_{\theta_i} s \rVert_1$ with different
stripe directions $\theta_i$", and Supplement 1 (Section 4B) discusses the
limitations of this approach (incomplete removal and smoothing of image
structures). The sentence can be read in two ways:

1. **One shared stripe component.** All terms act on the same $\mathbf{s}$:
   $\sum_i \lVert \nabla_{\theta_i} \mathbf{s} \rVert_1$. This is the literal
   reading, and the one implemented (`angle=[θ₁, θ₂]`).
2. **One component per direction.** Each direction has its own stripes
   $\mathbf{s}_i$, with $\mathbf{y} = \mathbf{x} + \sum_i \mathbf{s}_i$ and the
   penalties $\lVert \nabla_{\theta_i} \mathbf{s}_i \rVert_1$. This is not
   implemented.

With a shared $\mathbf{s}$, each term also penalises the stripes of the other
directions, with weight 1, so stripes are removed only for $\mu_1$ of order 1,
larger than for one direction:

- For the orthogonal pair 0° and 90°, the differences along the two axes are the
  components of $\nabla$, so $\lVert D_y \mathbf{s} \rVert_1 + \lVert D_x
  \mathbf{s} \rVert_1 \ge \lVert \nabla \mathbf{s} \rVert_{2,1}$. For
  $\mu_1 \le 1$ the objective is then at least
  $\mu_1 \lVert \nabla \mathbf{y} \rVert_{2,1} + \mu_2 \lVert \mathbf{s}
  \rVert_1$, and $\mathbf{x} = \mathbf{y}$ is the unique minimiser: the output
  equals the input.
- For other pairs the threshold is lower and depends on the data. On synthetic
  frames, removal starts between $\mu_1 = 0.5$ and $0.6$ for 0° + 45° and for
  ±45°, while 0° + 26.6° and 0° + 63.4° remove almost nothing up to $0.7$. At
  $\mu_1 = 0.7$, 0° + 45° removes both families of stripes better than either
  direction alone.

A `UserWarning` is emitted when several directions are given and $\mu_1 \le 1$
for a 0°/90° pair, or $\mu_1 \le 1/2$ for other pairs. The paper's results use one
direction only, so multi-directional use is untested against the paper.

## Memory and streaming

Frames are processed independently (per-frame streaming), in batches sized from
the memory budget, so arrays larger than memory (`zarr`, memmap) can be
destriped, and `apply` can write in place. The solver works on preallocated
buffers. A batch holds about $11 + 5 g + 3 m$ frame-sized tensors, for $m$ stripe
directions and $g = 3$ gradient components for volumes with $\rho_z > 0$ and
$g = 2$ otherwise (the estimate used to size batches). A volume is solved as a
whole, so a batch is one or a few volumes.

The cost is $O(\lvert\Omega\rvert m)$ per iteration for a frame of
$\lvert\Omega\rvert$ pixels, so $O(\lvert\Omega\rvert m)$
times `max_iter` per frame. On one CPU thread in float32, with one direction, an
iteration takes about 1 ms at $256 \times 256$ and 25 ms at $1024 \times 1024$:
about 10 minutes for a $1024 \times 1024$ frame at 25000 iterations, and 25
seconds at the recommended 1000. It does not depend on $\mu_1$, $\mu_2$ or on the
amount of stripes, and grows with the number of directions. Reduce `max_iter`, or
set `tol`, for large frames, and check the result.

## Usage

The example adds stripes to a [phantom](../../reference/models.md#phantoms) of cells.

```pycon
>>> from optix.destripe import GeneralStripeRemover
>>> from optix.models import phantoms
>>> # 1. Cells on a constant level (the truth), stripes along the first axis.
>>> shape = (1, 48, 48)  # (frames, H, W)
>>> x = phantoms.cells(shape, count=3, radius=8.0, intensity=0.5, seed=0) + 0.2
>>> y = (x + phantoms.stripes(shape, strength=0.15, seed=1)).clamp(0, 1)
>>> # 2. Fit (nothing is learned; few iterations for the example) and 3. apply.
>>> remover = GeneralStripeRemover(max_iter=300)
>>> x_hat = remover.fit(y).apply(y)
>>> # 4. Compare to the truth: the error std before and after.
>>> print(f"{(y - x).std():.3f} -> {(x_hat - x).std():.3f}")
0.145 -> 0.048
>>> remover.is_faithful
True
```

Common variations:

- `angle=20` for oblique stripes, and `axis=-1` (or a layout label) for
  horizontal ones; a sequence such as `angle=[0, 45]` with `mu1=0.7` for several
  directions (see [above](#several-stripe-directions)).
- A layout with three spatial axes processes volumes; `rho_z` weights the
  differences along $z$.
- `tol=1e-3` stops each frame at a relative duality gap of 1e-3; `box=False` drops the range constraint. Both
  are extensions and make `is_faithful` false.
- The function `optix.destripe.general_stripe_remover(data, ...)` takes a tensor
  `(..., H, W)` without a layout. See the
  [API reference](../../reference/destripe.md).

The tests check the method on synthetic stripes from `optix.models`, where the
truth is known, and against stored results of the authors' Python code (commit
[`f1972aa`](https://github.com/NiklasRottmayer/General-Stripe-Removal/tree/f1972aa47fc3ecf6580984f72ce7ce630b1ea683)).
optix reproduces these to rounding error in 2-D and 3-D (with $\rho_z = 1$),
except for negative angles (see [above](#not-specified-in-the-paper)).

## References

!!! references "References"

    1. Rottmayer N, Redenbach C, Fahrbach FO. *A universal and effective
       variational method for destriping: application to light-sheet microscopy,
       FIB-SEM, and remote sensing images.* Opt Express **33**(3):5800-5809, 2025.
       [doi:10.1364/OE.542868](https://doi.org/10.1364/OE.542868)
    2. Burger M, Sawatzky A, Steidl G. *First order algorithms in variational
       image processing.* In: Glowinski R, Osher SJ, Yin W (eds), Splitting
       Methods in Communication, Imaging, Science, and Engineering, pp. 345-407.
       Springer, 2016.
       [doi:10.1007/978-3-319-41589-5_10](https://doi.org/10.1007/978-3-319-41589-5_10)
    3. Rottmayer N. *General stripe removal* (reference implementation, MIT
       licence), 2024.
       [github.com/NiklasRottmayer/General-Stripe-Removal](https://github.com/NiklasRottmayer/General-Stripe-Removal)

    ??? quote "BibTeX"

        ```bibtex
        @article{rottmayer2025universal,
          author  = {Rottmayer, Niklas and Redenbach, Claudia and Fahrbach, Florian O.},
          title   = {A universal and effective variational method for destriping: application to light-sheet microscopy, {FIB-SEM}, and remote sensing images},
          journal = {Optics Express},
          volume  = {33},
          number  = {3},
          pages   = {5800--5809},
          year    = {2025},
          doi     = {10.1364/OE.542868},
        }
        @incollection{burger2016first,
          author    = {Burger, Martin and Sawatzky, Alex and Steidl, Gabriele},
          title     = {First order algorithms in variational image processing},
          booktitle = {Splitting Methods in Communication, Imaging, Science, and Engineering},
          editor    = {Glowinski, Roland and Osher, Stanley J. and Yin, Wotao},
          pages     = {345--407},
          publisher = {Springer},
          year      = {2016},
          doi       = {10.1007/978-3-319-41589-5_10},
        }
        @misc{rottmayer2024general,
          author = {Rottmayer, Niklas},
          title  = {General Stripe Removal},
          year   = {2024},
          url    = {https://github.com/NiklasRottmayer/General-Stripe-Removal},
        }
        ```
