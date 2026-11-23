# General stripe remover

!!! info "At a glance"
    **Estimates:** stripe-free image (the stripe component is the difference with the data) \
    **Noise model:** none \
    **Streaming:** per-frame \
    **Needs many frames:** no \
    **Reference:** Rottmayer et al., Opt Express, 2025, [doi:10.1364/OE.542868](https://doi.org/10.1364/OE.542868)

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
    \qquad \mathbf{x} \in [0, 1]^P,
    \quad \mathbf{s} \in [-1, 1]^P ,
    $$

    a frame (or volume) $\mathbf{y}$ of $P$ pixels, rescaled to $[0, 1]$, is
    the sum of the stripe-free image $\mathbf{x}$ and the stripes $\mathbf{s}$.

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
    + \iota_{[0,1]^P}(\mathbf{x})}_{D(\mathbf{x})}
    + \underbrace{\lVert \nabla_\theta \mathbf{s} \rVert_1
    + \mu_2 \lVert \mathbf{s} \rVert_1}_{N(\mathbf{s})},
    \qquad \mu_1, \mu_2 > 0,
    $$

    $$
    \lVert \nabla \mathbf{x} \rVert_{2,1} = \sum_p
    \sqrt{(\nabla_1 x)_p^2 + (\nabla_2 x)_p^2 + (\rho_z \nabla_3 x)_p^2},
    $$

    with $\nabla_1, \nabla_2$ the differences along the two in-plane axes and
    $\nabla_3$ along $z$ (volumes only).

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

| optix | [Rottmayer et al. 2025](https://doi.org/10.1364/OE.542868) |
|---|---|
| $\mathbf{y}$ (data) | $u_0$ |
| $\mathbf{x}$ (stripe-free) | $u$ |
| $\mathbf{s}$ | $s$ |
| $P$ (pixels) | $N = n_x n_y n_z$ |
| $\mu_1$, $\mu_2$ | $\mu_1$, $\mu_2$ |
| $\rho_z$ | $\rho_z$ |
| $\theta$, $\nabla_\theta$ | $\theta$, $\nabla_\theta$ |
| $\iota_{[0,1]^P}$ | $\iota_{[0,1]^N}$ |
| first spatial axis (stripe axis) | $y$ |

Each term has a role:

- $\lVert \nabla \mathbf{x} \rVert_{2,1}$ is the total variation. It favours
  images with few strong edges.
- $\lVert \nabla_\theta \mathbf{s} \rVert_1$ favours stripes that change little
  along their direction $\theta$ and a lot across it.
- $\lVert \mathbf{s} \rVert_1$ favours sparse stripes: usually only part of the
  image is affected.
- $\iota_{[0,1]^P}$ is zero when $\mathbf{x}$ lies in $[0, 1]^P$ and infinite
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

The objective is convex but non-smooth, and it has a single minimiser for the
parameters used. It is minimised with the primal-dual hybrid gradient method with
extrapolation of the dual variable (PDHGMp) of
[Burger et al. (2016)](https://doi.org/10.1007/978-3-319-41589-5_10), as in the
paper (Section 2.1.1). Writing the objective as a sum of terms that are each
simple after a linear map, $\mathbf{K}(\mathbf{x}, \mathbf{s}) =
(\nabla \mathbf{x},\ \nabla_\theta \mathbf{s},\ \mathbf{s})$, every term has a
cheap proximal operator in its dual form:

| Term | Dual variable | Projection in the dual step |
|---|---|---|
| $\mu_1 \lVert \nabla \mathbf{x} \rVert_{2,1}$ | $\mathbf{p}_1$ (one vector per pixel) | onto the Euclidean ball of radius $\mu_1$, per pixel |
| $\lVert \nabla_\theta \mathbf{s} \rVert_1$ | $\mathbf{p}_2$ | onto $[-1, 1]$, per pixel |
| $\mu_2 \lVert \mathbf{s} \rVert_1$ | $\mathbf{p}_3$ | onto $[-\mu_2, \mu_2]$, per pixel |

The constraint $\mathbf{x} + \mathbf{s} = \mathbf{y}$ and the indicator are
handled in the primal step by a projection onto
$\{\mathbf{x} + \mathbf{s} = \mathbf{y},\ \mathbf{x} \in [0, 1]^P\}$: the
residual $\mathbf{y} - \mathbf{x} - \mathbf{s}$ is shared equally between
$\mathbf{x}$ and $\mathbf{s}$, then the part of $\mathbf{x}$ outside $[0, 1]$ is
moved to $\mathbf{s}$. The cost of an iteration is a few differences and
clamps: it is linear in the number of pixels.

The paper names the method and the problem but not the discretisation, the step
sizes or the initialisation. The choices made are listed under
[Faithful mode and extensions](#faithful-mode-and-extensions).

## Algorithm

```text
Input: data y (frame or volume), μ₁ = 1/30, μ₂ = 1/300, θ = 0, ρ_z = 0,
       max_iter = 25000, steps τ = σ
Rescale y to [0, 1] by its minimum and maximum
Initialise x = y, s = 0, p₁ = p₂ = p₃ = 0 (and their extrapolations p̄ = p)
Repeat max_iter times:
    1. Primal step:
           x ← x − τ ∇ᵀ p̄₁
           s ← s − τ (∇_θᵀ p̄₂ + p̄₃)
    2. Project onto x + s = y:
           r ← (y − x − s) / 2;  x ← x + r;  s ← s + r
    3. Project onto x ∈ [0, 1]:
           x' ← clamp(x, 0, 1);  s ← s + (x − x');  x ← x'
    4. Dual step:
           p₁⁺ ← p₁ + σ ∇x,   rescaled per pixel to norm ≤ μ₁
           p₂⁺ ← clamp(p₂ + σ ∇_θ s, −1, 1)
           p₃⁺ ← clamp(p₃ + σ s, −μ₂, μ₂)
    5. Extrapolate:  p̄ ← 2 p⁺ − p;   p ← p⁺
Map x back to the range of the data
Return x
```

Frames are independent, and a batch is solved at once, each frame with its own
minimum, maximum and iterates. With several directions, step 1 sums one term
$\nabla_{\theta_i}^\top \bar{\mathbf{p}}_{2,i}$ per direction and step 4 has one
$\mathbf{p}_{2,i}$ per direction.

## Faithful mode and extensions

`faithful=True` solves Eq. 1 for the parameters $\mu_1$ and $\mu_2$, the stripe
direction and $\rho_z$ that the user gives, with the indicator constraint, for
25000 iterations. These parameters are meant to be tuned to the data
(Section 2.1.1), so any value is faithful; only the extensions below make
`is_faithful` false. The recommended values are the same as the faithful ones.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `mu1` | strength of stripe removal $\mu_1 > 0$ | 1/30 (see [below](#default-value-of-the-first-weight)) | 1/30 |
| `mu2` | precaution towards image structures $\mu_2 > 0$ | 1/300 | 1/300 |
| `axis` | spatial axis along which stripes run: a position among the spatial axes, or a layout label | first of two (vertical stripes in `(y, x)` frames) | same |
| `angle` | stripe direction $\theta$ in degrees from `axis` towards the other in-plane axis; a sequence for several directions | 0 | 0 |
| `rho_z` | weight $\rho_z \in [0, 1]$ of the differences along $z$ (three spatial axes) | 0 | 0 |
| `normalize` | rescale each frame to $[0, 1]$ and back | `True` | `True` |
| `max_iter` | number of PDHGMp iterations | 25000 | 25000 |
| `tol` | **extension:** stop a frame when $\lVert \mathbf{x}_k - \mathbf{x}_{k-1} \rVert / \lVert \mathbf{x}_k \rVert <$ `tol`, checked every 10 iterations | `None` | `None` |
| `box` | **extension:** `False` drops the indicator $\iota_{[0,1]^P}$ | `True` | `True` |

With two spatial axes in the layout, frames are 2-D; with three, the first is
$z$ and each volume is solved as a whole.

### Not specified in the paper

The following choices follow the authors' code, which is released under the MIT
licence ([General-Stripe-Removal](https://github.com/NiklasRottmayer/General-Stripe-Removal)),
and are marked as such in the docstrings.

- **Step sizes.** $\tau = \sigma = 0.35$ for 2-D problems and $0.28$ when
  $\rho_z > 0$. They are scaled down by $\sqrt{b / \lVert \mathbf{K} \rVert^2}$
  when a bound on $\lVert \mathbf{K} \rVert^2$ (8 in 2-D, 12 in 3-D) is
  exceeded, which happens with several stripe directions, so that
  $\tau \sigma \lVert \mathbf{K} \rVert^2 < 1$ holds.
- **Initialisation.** $\mathbf{x} = \mathbf{y}$, $\mathbf{s} = \mathbf{0}$, and all
  dual variables zero.
- **Boundary.** Differences are forward differences that are zero at the last
  pixel along their direction (a Neumann boundary).
- **Discretisation of $\theta$.** $\nabla_\theta$ is a forward difference along
  a lattice offset, and $\theta$ is rounded to the nearest of 0°, ±26.6°, ±45°,
  ±63.4° and 90° (modulo 180°), with offsets $(1, 0)$, $(2, \pm 1)$,
  $(1, \pm 1)$, $(1, \pm 2)$ and $(0, 1)$ along and across the stripe axis.
- **Normalisation.** The paper assumes data in $[0, 1]$. With `normalize`, each
  frame (or volume) is rescaled by its minimum and maximum, as in the authors'
  scripts, and the result is mapped back to the input range. With
  `normalize=False`, the data must already lie in $[0, 1]$.

### Default value of the first weight

The text gives different values of $\mu_1$ in different places:

- the Discussion states that good results can be achieved for the default values
  $\mu_1 = 1/30$ and $\mu_2 = 1/300$ (the second weight is written $\mu$ there);
- the figures use $\mu_1 = 1/3$ for the synthetic data (Fig. 2), and between
  $1/10$ and $2/5$ for the real data (Fig. 3), with $\mu_2 = 1/300$ or $1/150$;
- the authors' code defaults to $\mu_1 = 0.33$ and $\mu_2 = 0.003$.

The value $1/30$ is therefore possibly a typographical error for $1/3$; the paper
does not say. `GeneralStripeRemover` uses $1/30$ as written. Pass `mu1=1/3` for
the value of the figures: a larger $\mu_1$ removes stripes more strongly.

### Several stripe directions

Section 2.1.1 states that multi-directional destriping is possible "by including
a sum of penalisation terms $\lVert \nabla_{\theta_i} s \rVert_1$ with different
stripe directions $\theta_i$", and Supplement 1 (Section 4B) discusses the
limitations of this approach. The sentence can be read in two ways:

1. **One shared stripe component.** All terms act on the same $\mathbf{s}$:
   $\sum_i \lVert \nabla_{\theta_i} \mathbf{s} \rVert_1$. This is the literal
   reading, and the one implemented (`angle=[θ₁, θ₂]`). Each term then also
   penalises the stripes of the other directions, which are not constant along
   $\theta_i$.
2. **One component per direction.** Each direction has its own stripes
   $\mathbf{s}_i$, with $\mathbf{y} = \mathbf{x} + \sum_i \mathbf{s}_i$ and the
   penalties $\lVert \nabla_{\theta_i} \mathbf{s}_i \rVert_1$. This is not
   implemented.

The paper's results use one direction only, so multi-directional use is
untested against the paper.

## Memory and streaming

Frames are processed independently (per-frame streaming), in batches sized from
the memory budget, so arrays larger than memory (`zarr`, memmap) can be
destriped, and `apply` can write in place. A batch holds a few frame-sized
tensors for each of the data, the iterates and the dual variables: about
$10 + 4 d + 2 m$ frames-worth for $d$ spatial axes and $m$ stripe directions
(the estimate used to size batches). A volume ($d = 3$) is solved as a whole, so
a batch is one or a few volumes.

The cost is $O(P)$ per iteration for a frame of $P$ pixels, hence
$O(P \times \text{max\_iter})$ per frame, 25000 iterations by default. This is
slow for large frames: reduce `max_iter`, or set `tol`, and check the result.
The cost does not depend on $\mu_1$, $\mu_2$ or the number of stripes.

## Usage

```pycon
>>> import torch
>>> from optix.destripe import GeneralStripeRemover
>>> _ = torch.manual_seed(0)
>>> # 1. A disc with stripes along the first axis (one offset per column).
>>> rows, cols = torch.meshgrid(torch.arange(48), torch.arange(48), indexing="ij")
>>> clean = 0.2 + 0.5 * (((rows - 24) ** 2 + (cols - 24) ** 2) < 14**2)
>>> stripes = 0.15 * torch.randn(48)
>>> data = (clean + stripes[None, :]).clamp(0, 1)[None]  # (frames, H, W)
>>> # 2. Destripe (nothing is fitted); few iterations for the example.
>>> remover = GeneralStripeRemover(mu1=1 / 3, max_iter=300)
>>> destriped = remover.fit(data).apply(data)
>>> # 3. Residual stripes, before and after.
>>> print(f"{(data[0] - clean).std():.3f} -> {(destriped[0] - clean).std():.3f}")
0.137 -> 0.031
>>> remover.is_faithful
True
```

Common variations:

- `angle=20` for oblique stripes, and `axis=-1` (or a layout label) for
  horizontal ones; a sequence such as `angle=[0, 45]` for several directions
  (see [above](#several-stripe-directions)).
- A layout with three spatial axes processes volumes; `rho_z` weights the
  differences along $z$.
- `tol=1e-4` stops each frame early; `box=False` drops the range constraint. Both
  are extensions and make `is_faithful` false.
- The function `optix.destripe.general_stripe_remover(data, ...)` takes a tensor
  `(..., H, W)` without a layout. See the
  [API reference](../../reference/destripe.md).

The tests check the method on synthetic stripes from `optix.models`, where the
truth is known; there are no stored fixtures from the authors' code.

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
