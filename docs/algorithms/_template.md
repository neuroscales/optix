# Algorithm name

<!--
Page template for an algorithm. Copy this file, rename it after the algorithm,
and replace each section. Keep the nine headings and their order. Let every
section be as long as its topic needs, no longer. See the Algorithms page for
what each section holds.
-->

!!! info "At a glance"
    - **Estimates:** the result type, for instance `ShadingEstimate` (fields it fills)
    - **Noise model:** none, Gaussian, Poisson or Poisson-Gaussian
    - **Streaming:** pointwise, per-frame, halo, per-pixel-stats or whole-stack
    - **Needs many frames:** yes or no
    - **Reference:** Author, Journal, Year, DOI

## Overview

*What the algorithm estimates and what for, in plain words. When to use it, when
another algorithm is a better choice. What it returns.*

## Image formation (optics)

*The forward model: the physical process behind the data that the algorithm
assumes, more concisely than the paper but precisely. Refer to the shared
models in [Image formation and noise](../concepts/image-formation.md) and state
only what is specific here. Give the equations in tabs, in optix notation and
in the paper's, followed by a table mapping one set of symbols onto the other.*

=== "optix"

    $$
    \mathbf{y}_i = \ldots
    $$

=== "Author et al. 2000"

    $$
    I_i = \ldots
    $$

| optix | Author et al. 2000 | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}_i$ | $I_i$ | recorded frame $i$ | $P$, one per frame |
| $\mathbf{f}$ | $S$ | flat field | $P$, shared by all frames of a group |

*Every table of symbols says in words what each symbol means, gives the shape
of vectors and matrices ($\mathbf{y}_i \in \mathbb{R}^P$,
$\boldsymbol{\Phi} \in \mathbb{R}^{P \times K}$), and says what each
parameter is shared across: one value per pixel, per frame, per group, or one
for the whole stack. Annotate terms of an equation with `\underbrace`, never
with text such as `mode=""` inside the equation.*

## Probabilistic model

*The assumptions, stated explicitly, and the generative model or objective
they lead to: observation model, noise, priors or regularisers. Define every
symbol when it first appears, in the [notation](../concepts/notation.md); use
the same tabs as above wherever the paper has its own notation.*

$$
y = \ldots
$$

## Inference / derivation

*The estimation strategy, clearly explained but without every derivation.
Name its statistical counterpart whenever there is one: maximum likelihood,
MAP, EM, variational Bayes, iteratively reweighted least squares, method of
moments (for instance, Richardson–Lucy is EM for a Poisson likelihood).
what is optimised or inferred, by what kind of scheme (alternation, convex
splitting, EM, fixed point, ...), which terms are fixed or updated in turn, and
how any ambiguity is resolved (normalisation of a flat field, for instance).
Full update equations are only needed where the strategy cannot be understood
without them.*

## Algorithm

*The steps of the algorithm as a numbered list, with the maths rendered:
inputs, initialisation, iterations, stopping rule, outputs, and the default
hyper-parameters. No raw-text pseudocode block.*

**Input:** frames $\mathbf{y}_1, \ldots, \mathbf{y}_N \in \mathbb{R}^P$;
hyper-parameters $\lambda$, ... \
**Output:** the estimate, for instance $\hat{\mathbf{f}}$.

1. **Initialise** $\mathbf{f}^{(0)} = \mathbf{1}$, ...
2. **Repeat** until the relative change of the objective is below `tol`:
    1. update ... with ...;
    2. update ... with ....
3. **Normalise** ... and return the estimate.

## Faithful mode and extensions

*What `faithful=True` reproduces, and where and why it departs from the paper.*

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `option` | what it does | the paper's | what `faithful=False` uses |

## Memory and streaming

*Which chunk iterator it uses, what it keeps in memory, how cost scales with the
size of the stack, and what to tune.*

## Usage

```python
import optix
from optix.models import (
    random_sensor,
    random_shading,
    simulate_sensor,
    simulate_shading,
)

# 1. A synthetic stack with known truth.
# 2. est = Algorithm().fit(Y)
# 3. corrected = est.apply(Y)
# 4. Compare est.estimates[...] to the truth.
```

*Common variations: layouts with groups, initialisation from another estimate,
a large `zarr` array, the extensions.*

## References

!!! references "References"

    1. Author A, Author B. *Title.* Journal **vol**(issue):pages, year.
       [doi:10.xxxx/xxxxx](https://doi.org/10.xxxx/xxxxx)

    ??? quote "BibTeX"

        ```bibtex
        @article{author2000key,
          author  = {Author, A. and Author, B.},
          title   = {Title},
          journal = {Journal},
          volume  = {1},
          number  = {1},
          pages   = {1--10},
          year    = {2000},
          doi     = {10.xxxx/xxxxx},
        }
        ```
