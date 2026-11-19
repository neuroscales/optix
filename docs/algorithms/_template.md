# Algorithm name

<!--
Page template for an algorithm. Copy this file, rename it after the algorithm,
and replace each section. Keep the nine headings and their order. Let every
section be as long as its topic needs, no longer. See the Algorithms page for
what each section holds.
-->

!!! info "At a glance"
    **Estimates:** the result type, for instance `ShadingEstimate` (fields it fills) \
    **Noise model:** none, Gaussian, Poisson or Poisson-Gaussian \
    **Streaming:** pointwise, per-frame, halo, per-pixel-stats or whole-stack \
    **Needs many frames:** yes or no \
    **Reference:** Author, Journal, Year, DOI

## Overview

*What the algorithm estimates and what for, in plain words. When to use it, when
another algorithm is a better choice. What it returns.*

## Image formation (optics)

*The physical process behind the data that the algorithm assumes. Refer to the
shared models in [Image formation and noise](../concepts/image-formation.md)
and state only what is specific here.*

## Probabilistic model

*The generative model and its assumptions as equations: observation model,
noise, priors or regularisers. Define every symbol when it first appears, in the
notation of [Concepts](../concepts/index.md#notation).*

$$
y = \ldots
$$

## Inference / derivation

*From the model to the update equations, one step at a time. State how any
ambiguity is resolved (normalisation of a flat field, for instance).*

## Algorithm

*Pseudocode: inputs, initialisation, iterations, stopping rule, outputs, and
the default hyper-parameters.*

```text
Input: data Y, hyper-parameters ...
Initialise ...
Repeat until the relative change of the objective is below tol:
    1. Update ...
    2. Update ...
Return the estimate
```

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
