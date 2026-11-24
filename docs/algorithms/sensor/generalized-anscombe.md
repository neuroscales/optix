# Generalised Anscombe transform

!!! info "At a glance"
    - **Estimates:** none: a transform; reads a `SensorEstimate` (`offset`, `variance`, `gain`)
    - **Noise model:** Poisson-Gaussian
    - **Streaming:** pointwise
    - **Needs many frames:** no
    - **Reference:** Mäkitalo, Foi, IEEE Trans Image Process, 2013, [doi:10.1109/TIP.2012.2202675](https://doi.org/10.1109/TIP.2012.2202675)

## Overview

`GeneralizedAnscombe` is a variance-stabilising transform: it maps
Poisson-Gaussian camera data to values whose noise is approximately standard
normal, whatever the signal. Gaussian denoisers, or any method that assumes
constant noise, can then be applied, and the result is mapped back with an
inverse. The exact unbiased inverse of
[Mäkitalo & Foi 2011](https://doi.org/10.1109/TIP.2010.2056693) and
[Mäkitalo & Foi 2013](https://doi.org/10.1109/TIP.2012.2202675) removes the bias
that the plain algebraic inverse introduces at low counts.

It estimates nothing from data: the camera parameters come from a
[`SensorEstimate`](../../reference/sensor.md) (for instance from the
[photon-transfer curve](photon-transfer-curve.md)) via `fit(None, init=sensor)`,
or from the keywords `alpha`, `sigma`, `mu`. `apply` transforms,
`inverse` maps back to an estimate of $\mathbb{E}[y]$ in ADU.

## Image formation (optics)

The camera model of [Image formation and noise](../../concepts/image-formation.md),
without clipping. Nothing about the optics is assumed.

=== "optix"

    $$
    y = g\,e + o + \varepsilon, \qquad e \sim \operatorname{Poisson}(\lambda),\quad
    \varepsilon \sim \mathcal{N}(0, \sigma^2)
    $$

=== "Mäkitalo & Foi 2013"

    $$
    z = \alpha\,p + n, \qquad p \sim \mathcal{P}(y/\alpha),\quad
    n \sim \mathcal{N}(\mu, \sigma^2)
    $$

## Probabilistic model

The transform and the definition of the exact unbiased inverse, in normalised
units ($\alpha = 1$, $\mu = 0$, Gaussian standard deviation $\sigma/\alpha$):

=== "optix"

    $$
    f(y) = \frac{2}{g}\sqrt{\max\Bigl(0,\; g\,y + \tfrac{3}{8}g^2 + \sigma^2 - g\,o\Bigr)}
    \qquad
    \hat{\mathbb{E}}[y] = g\,\mathcal{I}_{\sigma/g}(d) + o
    $$

=== "Mäkitalo & Foi 2013"

    $$
    f(z) = \frac{2}{\alpha}\sqrt{\max\Bigl(0,\; \alpha z + \tfrac{3}{8}\alpha^2 + \sigma^2 - \alpha\mu\Bigr)},
    \qquad
    \mathcal{I}_\sigma : \mathbb{E}\{f(z) \mid y, \sigma\} \longmapsto y
    $$

| optix | [Mäkitalo & Foi 2013](https://doi.org/10.1109/TIP.2012.2202675) | Meaning | Shape |
|---|---|---|---|
| $y$ | $z$ | recorded value (ADU) | scalar, per pixel and per frame |
| $g$ | $\alpha$ | gain (ADU per electron) | scalar, one per group (per pixel with `per_pixel=True`) |
| $o$ | $\mu$ | offset, mean of the Gaussian part | scalar, one per group (per pixel with `per_pixel=True`) |
| $\sigma^2$ | $\sigma^2$ | read-noise variance (ADU²) | scalar, one per group (per pixel with `per_pixel=True`) |
| $g\,\lambda$ | $y$ | expected signal above the offset | scalar, per pixel |
| $f(y)$ | $f(z)$ | stabilised value, noise variance about 1 | scalar, per pixel and per frame |
| $d$ | $D$ | denoised stabilised value, input of the inverse | scalar, per pixel |
| $\mathcal{I}_{\sigma/g}$ | $\mathcal{I}_\sigma$ | exact unbiased inverse at normalised noise level | function, one per value of $\sigma/g$ |

Assumptions: the parameters are known and, by default, the same for every pixel
(the means of the maps of the estimate); no clipping. The stabilisation is
approximate: at very low counts the variance of $f(y)$ is below 1.

## Inference / derivation

Nothing is fitted. The forward transform is the generalised Anscombe formula.
The strategy of the inverse is the point of the papers: a denoiser applied to
$f(y)$ estimates $\mathbb{E}\{f(y)\}$, not $f(\mathbb{E}[y])$, and because $f$
is nonlinear the algebraic inverse is biased. The exact unbiased inverse
$\mathcal{I}_\sigma$ inverts the map $y \mapsto \mathbb{E}\{f(z) \mid y, \sigma\}$
instead. optix computes that expectation numerically (Poisson sum times a
Gaussian quadrature, Eq. (10)) on a grid of $y$ and $\sigma^2$, inverts it by
interpolation, and switches to the closed-form approximation of
[Mäkitalo & Foi 2011](https://doi.org/10.1109/TIP.2011.2121085) above the table.
The closed-form, asymptotic and algebraic inverses are available too.

Statistically, $f$ is a delta-method variance-stabilising transform, and the
exact unbiased inverse is a method-of-moments estimator: it returns the
parameter $y$ whose model expectation $\mathbb{E}\{f(z) \mid y\}$ equals the
denoised value $d$. Under a Gaussian posterior for $f(z)$ it is the inverse of
the posterior-mean map, not a full Bayesian estimate of $y$.

## Algorithm

**Input:** frames $\mathbf{y}_1, \ldots, \mathbf{y}_N \in \mathbb{R}^{\lvert\Omega\rvert}$ ($\lvert\Omega\rvert$
pixels per frame); camera parameters $(g, o, \sigma^2)$ from a `SensorEstimate`
or from `alpha`, `sigma`, `mu`; `inverse="exact"`. \
**Output:** `apply`: stabilised frames $f(\mathbf{y}_n) \in \mathbb{R}^{\lvert\Omega\rvert}$;
`inverse`: an estimate of $\mathbb{E}[\mathbf{y}] \in \mathbb{R}^{\lvert\Omega\rvert}$ in ADU.

1. **Forward** (`apply`), pointwise: $d = f(y)$.
2. **Inverse** (`inverse`), pointwise, with $s^2 = (\sigma/g)^2$:
    1. `"exact"`: interpolate the inverse of the table of
       $\mathbb{E}\{f \mid y, s^2\}$ (built once in float64 on the CPU), with
       the closed form above the table;
    2. `"closed-form"`: Eq. (21);
    3. `"asymptotic"`: $\hat y = d^2/4 - 1/8 - s^2$;
    4. `"algebraic"`: $\hat y = d^2/4 - 3/8 - s^2$ (biased).
3. **Return** $g\,\hat y + o$.

## Faithful mode and extensions

`faithful=True` and `faithful=False` are identical.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `alpha`, `sigma`, `mu` | parameters given directly instead of `fit(init=...)` | `None` | `None` |
| `inverse` | `"exact"`, `"closed-form"`, `"asymptotic"`, `"algebraic"` | `"exact"` | `"exact"` |
| `per_pixel` | use the per-pixel maps of the estimate (extension) | `False` | `False` |

## Memory and streaming

Pointwise: data are processed chunk by chunk with no state but the three
parameters. The exact-inverse table (33 values of $\sigma^2$ × 4096 values of
$D$ for per-pixel maps, one row otherwise) is computed once and reused.

## Usage

The example images an intensity [ramp](../../reference/models.md#phantoms)
through a simulated camera with known gain and read noise.

```python
from optix.models import phantoms, random_sensor, simulate_sensor
from optix.sensor import GeneralizedAnscombe

# 1. Four frames of an intensity ramp through a known camera.
truth = random_sensor(
    (128, 128), offset=100.0, offset_sd=0.0, read_sd=2.0, gain=0.5, seed=0
)
x = phantoms.ramp((4, 128, 128), low=20.0, high=3020.0)
y = simulate_sensor(x, truth, seed=1)

# 2. Build the transform from the known noise parameters.
vst = GeneralizedAnscombe().fit(None, init=truth)

# 3. Stabilise, and map a mean back to ADU.
d = vst.apply(y)
mean = vst.inverse(d.mean(0, keepdim=True))

# 4. The noise is now close to 1 everywhere along the ramp.
print(float(d.std(0).mean()))  # about 0.92
```

## References

!!! references "References"

    1. Mäkitalo M, Foi A. *Optimal inversion of the Anscombe transformation in
       low-count Poisson image denoising.* IEEE Trans Image Process
       **20**(1):99-109, 2011.
       [doi:10.1109/TIP.2010.2056693](https://doi.org/10.1109/TIP.2010.2056693)
    2. Mäkitalo M, Foi A. *A closed-form approximation of the exact unbiased
       inverse of the Anscombe variance-stabilizing transformation.* IEEE Trans
       Image Process **20**(9):2697-2698, 2011.
       [doi:10.1109/TIP.2011.2121085](https://doi.org/10.1109/TIP.2011.2121085)
    3. Mäkitalo M, Foi A. *Optimal inversion of the generalized Anscombe
       transformation for Poisson-Gaussian noise.* IEEE Trans Image Process
       **22**(1):91-103, 2013.
       [doi:10.1109/TIP.2012.2202675](https://doi.org/10.1109/TIP.2012.2202675)

    ??? quote "BibTeX"

        ```bibtex
        @article{makitalo2011optimal,
          author  = {M{\"a}kitalo, Markku and Foi, Alessandro},
          title   = {Optimal inversion of the {A}nscombe transformation in low-count {P}oisson image denoising},
          journal = {IEEE Transactions on Image Processing},
          volume  = {20},
          number  = {1},
          pages   = {99--109},
          year    = {2011},
          doi     = {10.1109/TIP.2010.2056693},
        }
        @article{makitalo2011closed,
          author  = {M{\"a}kitalo, Markku and Foi, Alessandro},
          title   = {A closed-form approximation of the exact unbiased inverse of the {A}nscombe variance-stabilizing transformation},
          journal = {IEEE Transactions on Image Processing},
          volume  = {20},
          number  = {9},
          pages   = {2697--2698},
          year    = {2011},
          doi     = {10.1109/TIP.2011.2121085},
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
