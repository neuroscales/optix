# Biggs–Andrews

!!! info "At a glance"
    - **Estimates:** deconvolved image (the latent image $\mathbf{x}$, same shape as the data)
    - **Noise model:** Poisson (Poisson-Gaussian as an extension)
    - **Streaming:** halo
    - **Needs many frames:** no
    - **Reference:** Biggs and Andrews, Appl Opt, 1997, [doi:10.1364/AO.36.001766](https://doi.org/10.1364/AO.36.001766)

## Overview

`BiggsAndrews` is [Richardson–Lucy](richardson-lucy.md) deconvolution
accelerated by vector extrapolation ([Biggs and Andrews 1997, "technique
II"](https://doi.org/10.1364/AO.36.001766)): each iteration is applied not to the
current estimate but to a point predicted from the last two, which typically
reaches a given fit in several times fewer iterations. Model, start, stopping
rule, blocks and extensions are those of `RichardsonLucy`. Use it when plain
Richardson–Lucy needs many iterations; accelerated iterations amplify noise
sooner, so the stopping rule or `n_iter` matters more.

## Image formation (optics)

As for [Richardson–Lucy](richardson-lucy.md#image-formation-optics): a linear,
shift-invariant blur by a known PSF $\mathbf{h}$, normalised to unit sum, with the
latent image estimated on a domain extended by the PSF support (see
[Image formation and noise](../../concepts/image-formation.md)).

## Probabilistic model

The same as Richardson–Lucy:

$$
\mathbf{y} \sim \mathrm{Poisson}(\mathbf{A}\mathbf{x} + \mathbf{b}),
\qquad \mathbf{x} \ge 0 ,
$$

with $\mathbf{y} \in \mathbb{R}^P$ one recorded image of $P$ pixels,
$\mathbf{x} \in \mathbb{R}^Q$ the latent image on the extended domain of
$Q \ge P$ pixels, $\mathbf{A} \in \mathbb{R}^{P \times Q}$ the blur by
$\mathbf{h}$ (one for the whole stack) and $\mathbf{b} \in \mathbb{R}^P$ an
optional background (none in the paper). The acceleration changes the path to the maximum-likelihood
estimate, not the model.

## Inference / derivation

Write $\psi$ for one Richardson–Lucy update. The strategy is to extrapolate along
the direction of the last step, by an amount set by how well successive changes
agree, and to apply $\psi$ at the extrapolated point:

=== "optix"

    $$
    \mathbf{v}_k = \mathbf{x}_k + \alpha_k(\mathbf{x}_k - \mathbf{x}_{k-1}),
    \qquad
    \mathbf{x}_{k+1} = \psi(\mathbf{v}_k),
    \qquad
    \alpha_k = \frac{\mathbf{g}_{k-1}^\top \mathbf{g}_{k-2}}{\mathbf{g}_{k-2}^\top \mathbf{g}_{k-2}},
    $$

    with $\mathbf{x}_k, \mathbf{v}_k, \mathbf{g}_k \in \mathbb{R}^Q$ and
    $\mathbf{g}_k = \psi(\mathbf{v}_k) - \mathbf{v}_k$ the change made by
    the $k$-th update.

=== "Biggs & Andrews 1997"

    $$
    \mathbf{y}_k = \mathbf{x}_k + \alpha_k \mathbf{h}_k,
    \quad \mathbf{h}_k = \mathbf{x}_k - \mathbf{x}_{k-1},
    \qquad
    \mathbf{x}_{k+1} = \mathbf{y}_k + \mathbf{g}_k,
    \quad \mathbf{g}_k = \psi(\mathbf{y}_k) - \mathbf{y}_k,
    $$

    $$
    \alpha_k = \frac{\sum \mathbf{g}_{k-1} \cdot \mathbf{g}_{k-2}}{\sum \mathbf{g}_{k-2} \cdot \mathbf{g}_{k-2}}
    \quad \text{(Eq. 10)}.
    $$

| optix | [Biggs & Andrews 1997](https://doi.org/10.1364/AO.36.001766) | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}$ | $g$ | recorded image, in counts | $P$, one per image |
| $\mathbf{x}_k$ | $\mathbf{x}_k$ | iterate after $k$ updates | $Q$, one per image |
| $\mathbf{v}_k$ | $\mathbf{y}_k$ | extrapolated (predicted) point | $Q$, one per image |
| $\mathbf{x}_k - \mathbf{x}_{k-1}$ | $\mathbf{h}_k$ | last step, the extrapolation direction | $Q$, one per image |
| $\mathbf{g}_k$ | $\mathbf{g}_k$ | change made by one update at $\mathbf{v}_k$ | $Q$, one per image |
| $\alpha_k$ | $\alpha_k$ | acceleration factor | scalar, per image (or per block) and iteration |
| $\mathbf{h}$ | $h$ | PSF | $K$ voxels, one for the whole stack |
| $\psi$ | $\psi$ | one Richardson–Lucy update | map from $\mathbb{R}^Q$ to $\mathbb{R}^Q$ |

The paper's symbols clash with optix: its $\mathbf{y}_k$ is the prediction, not
the data, and its $\mathbf{h}_k$ a step, not the PSF.

$\alpha_k$ is the least-squares projection of the latest change on the one
before (first-order extrapolation). The safeguards of the paper are kept:
$\alpha_k < 0$ is set to 0, which suspends the extrapolation until successive
changes agree again; $\alpha_k \le 1$, so that the step cannot grow
exponentially; and negative pixels of the prediction are set to 0 (the paper
requires non-negativity but leaves the correction open). The equivalent number
of unaccelerated iterations, $\sum_k \eta_k$ of Eqs. 11–13, is recorded in
`history.extras["effective_iterations"]`.

Statistically, $\psi$ is one EM step for the Poisson maximum-likelihood
estimate (see [Richardson–Lucy](richardson-lucy.md#inference-derivation)), so
the scheme is an accelerated EM: a first-order vector extrapolation of the EM
fixed-point map, close to the momentum (heavy-ball) acceleration of gradient
methods and to the squared extrapolation methods for EM. It leaves
the fixed point, the maximum-likelihood estimate, unchanged but gives up the
monotone increase of the likelihood that plain EM guarantees.

## Algorithm

**Input:** an image $\mathbf{y} \in \mathbb{R}^P$ in counts, the PSF
$\mathbf{h}$; hyper-parameters `alpha` $= 0.05$ (χ² level), `max_iter`
$= 100$, `n_iter` $=$ `None`, `alpha_scope` $=$ `"global"`. \
**Output:** the deconvolved image $\hat{\mathbf{x}}$, cropped to the observed
region.

1. **Initialise** $\mathbf{x}_0$ uniform, at the mean intensity of
   $\mathbf{y} - \mathbf{b}$.
2. **First update:** plain Richardson–Lucy,
   $\mathbf{x}_1 = \psi(\mathbf{x}_0)$, which gives
   $\mathbf{A}^\top\mathbf{y} / \mathbf{A}^\top\mathbf{1}$ (the paper's
   starting image).
3. **Repeat** for $k = 1, 2, \ldots$ until `n_iter` (if given) or `max_iter`
   iterations:
    1. **Test** (only when `n_iter` is `None`): stop if the χ² test of
       Richardson–Lucy on $\mathbf{x}_k$ is no longer significant;
    2. **Predict:** if two changes $\mathbf{g}_{k-1}, \mathbf{g}_{k-2}$ are
       known, compute $\alpha_k$ by Eq. 10, clip it to $[0, 1]$ and set
       $\mathbf{v}_k = \max(\mathbf{x}_k + \alpha_k(\mathbf{x}_k - \mathbf{x}_{k-1}), 0)$;
       otherwise $\mathbf{v}_k = \mathbf{x}_k$;
    3. **Update** $\mathbf{x}_{k+1} = \psi(\mathbf{v}_k)$.
4. **Crop** $\mathbf{x}$ to the observed region and return it.

The first prediction is made at the fourth update; up to three iterations the
result is plain Richardson–Lucy. The stopping rule is not in the paper, which
compares restorations at fixed iteration counts; the default is
[Lucy's (1974)](https://doi.org/10.1086/111605) χ² rule, evaluated on the
iterates $\mathbf{x}_k$.

## Faithful mode and extensions

The keywords of [Richardson–Lucy](richardson-lucy.md#faithful-mode-and-extensions)
(`n_iter`, `alpha`, `max_iter`, `background`, `poisson_gaussian`, `damping`,
`halo`, `block`) apply unchanged, with the same faithful values. `alpha` is the
level of the χ² test, not the acceleration parameter $\alpha_k$. One keyword is
added:

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `alpha_scope` | pixels over which the sums of $\alpha_k$ run: `"global"` the whole image, `"block"` each block, `"auto"` global when each image fits in one block | `"global"` | `"auto"` |

The scope used for each image is recorded in `history.extras["alpha_scope"]`;
after `apply`, `is_faithful` is False if the block-wise $\alpha_k$ was used.

## Memory and streaming

With `alpha_scope="global"`, an image larger than one block is iterated in
lockstep: every iteration is one pass over its blocks, which read their data and
the current iterate around them, make one step, write back their core and add
their share of the float64 sums of $\alpha_k$ and χ². The halo covers the PSF, so
the result is the whole-image iteration up to FFT round-off. The last three
iterates, on the image extended by the PSF margin, are kept in memory when they
fit the memory budget of [`config`](../../reference/configuration.md), else in
temporary files, and the data are read once per iteration.

With `alpha_scope="block"`, each block runs all its iterations on its own, with
its own $\alpha_k$, in a single pass over the data, as for Richardson–Lucy; the
result then differs from the whole-image one, by up to about 10 % of the peak on
noisy data.

## Usage

```python
import torch
from optix.deconv import BiggsAndrews, RichardsonLucy
from optix.models import gaussian_psf, simulate_blur

torch.manual_seed(0)

# 1. A Gaussian PSF and a blurred, Poisson-noisy test pattern.
psf = gaussian_psf((15, 15), sigma=2.0)
x = torch.full((96, 96), 5.0)
x[20:76:8, 20:76] = 200.0
y = torch.poisson(simulate_blur(x, psf))

# 2. Accelerated and plain deconvolution, both with the χ² stopping rule.
ba = BiggsAndrews(psf).fit(y)
x_ba = ba.apply(y)
rl = RichardsonLucy(psf).fit(y)
x_rl = rl.apply(y)

# 3. Iterations used, and their unaccelerated equivalent.
print(
    ba.history.extras["iterations"], ba.history.extras["effective_iterations"]
)
print(rl.history.extras["iterations"])
print(float((x_ba - x).abs().mean()), float((x_rl - x).abs().mean()))
```

Common variations: `n_iter=` for a fixed number of iterations;
`BiggsAndrews(psf, faithful=False)` for the block-wise $\alpha_k$ on large images;
`fit(y, init=sensor)` for raw camera data. The function
`biggs_andrews(data, psf, ...)` is a shortcut. See the
[API reference](../../reference/deconv.md).

## References

!!! references "References"

    1. Biggs DSC, Andrews M. *Acceleration of iterative image restoration
       algorithms.* Appl Opt **36**(8):1766-1775, 1997.
       [doi:10.1364/AO.36.001766](https://doi.org/10.1364/AO.36.001766)
    2. Lucy LB. *An iterative technique for the rectification of observed
       distributions.* Astron J **79**(6):745-754, 1974.
       [doi:10.1086/111605](https://doi.org/10.1086/111605)

    ??? quote "BibTeX"

        ```bibtex
        @article{biggs1997acceleration,
          author  = {Biggs, David S. C. and Andrews, Mark},
          title   = {Acceleration of iterative image restoration algorithms},
          journal = {Applied Optics},
          volume  = {36},
          number  = {8},
          pages   = {1766--1775},
          year    = {1997},
          doi     = {10.1364/AO.36.001766},
        }
        @article{lucy1974iterative,
          author  = {Lucy, L. B.},
          title   = {An iterative technique for the rectification of observed distributions},
          journal = {The Astronomical Journal},
          volume  = {79},
          number  = {6},
          pages   = {745--754},
          year    = {1974},
          doi     = {10.1086/111605},
        }
        ```
