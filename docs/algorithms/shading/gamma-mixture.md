# Gamma mixture

!!! info "At a glance"
    - **Estimates:** `ShadingEstimate` (`flat`, `dark`), `SensorEstimate` (`offset`, `variance`, nominal `gain`), `Classification` (`concentration`, `weights`, `parameters`)
    - **Noise model:** Poisson-Gaussian (Gaussian limit)
    - **Streaming:** per-frame
    - **Needs many frames:** yes
    - **Reference:** none: an original optix model, not published (the Dirichlet update follows [Minka 2000](https://www.microsoft.com/en-us/research/publication/estimating-dirichlet-distribution/))

## Overview

`GammaMixture` is a joint generative model of the camera, the shading and the
specimen. Each pixel of each frame is either *background* (no signal) or
belongs to one of $K$ *tissue classes* whose signal follows a Gamma
distribution; the signal is modulated by a smooth flat field and recorded
with signal-dependent noise. Fitting the model gives, at once, the flat field,
the per-pixel offset and read variance of empty pixels, the parameters and
proportions of the classes, and, for any frame, the posterior probability that
a pixel is foreground.

Use it on sparse fluorescence data in which every pixel is empty in a good
fraction of the frames (tiles of a slide, sparse cells), when a calibrated
noise model or a foreground map is wanted alongside the flat field. For dense
specimens, or when only a flat field is needed quickly, prefer
[BaSiC](basic.md) or the [illumination function](illumination-function.md).

## Image formation (optics)

The shading model of [Image formation and noise](../../concepts/image-formation.md)
with the camera model of the same page: frame $i$ records

$$
\mathbf{y}_i = \mathbf{o} + g\,\mathrm{Poisson}\bigl(\mathbf{f} \odot (\mathbf{s}_i + \mathbf{b}) + \mathbf{d}_e\bigr) + \boldsymbol{\varepsilon}_i ,
\qquad \boldsymbol{\varepsilon}_i \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\sigma}^2),
$$

with $\mathbf{s}_i$ the specimen signal in electrons, $\mathbf{b}$ a
background and $\mathbf{d}_e$ a dark signal, both constant over frames. In the
Gaussian limit, and writing the signal in ADU, $\mathbf{u}_i = g\,\mathbf{s}_i$,
each pixel $p$ is

$$
y_{ip} \sim \mathcal{N}\bigl(c_p + f_p\,u_{ip},\; v_p + g\,f_p\,u_{ip}\bigr),
\qquad u_{ip} \ge 0,
$$

with the offset $\mathbf{c} = \mathbf{o} + g(\mathbf{f}\odot\mathbf{b} + \mathbf{d}_e)$
and the variance of an empty pixel $\mathbf{v} = \boldsymbol{\sigma}^2 + g^2(\mathbf{f}\odot\mathbf{b} + \mathbf{d}_e)$.
In optix terms the dark field is the offset, $\hat{\mathbf{d}} = \mathbf{c}$
(`estimates["sensor"].offset` = `estimates["shading"].dark`), the flat field
is $\mathbf{f}$ (`estimates["shading"].flat`, geometric mean 1), $\mathbf{v}$
is `estimates["sensor"].variance`, and the baseline is absorbed in
$\mathbf{c}$; the gain $g$ is held at its nominal value. With
`noise="constant"` the variance is $v_p$ (no shot noise). The class
docstring uses the same notation.

## Probabilistic model

The signal is a mixture of a background class (index $0$ below; stored last
in code) and $K$ Gamma tissue classes, with class proportions drawn per frame;
the log flat field is smooth, expanded in a DCT basis:

$$
\begin{aligned}
\boldsymbol{\pi}_i &\sim \mathrm{Dirichlet}(\boldsymbol{\alpha}), &
z_{ip} &\sim \mathrm{Categorical}(\boldsymbol{\pi}_i), \\
u_{ip} \mid z_{ip} = 0 &\sim \delta(0), &
u_{ip} \mid z_{ip} = k &\sim \mathrm{Gamma}(\nu_k,\ \nu_k/\mu_k), \\
\log \mathbf{f} &= \boldsymbol{\Phi}\boldsymbol{\theta}, &
\boldsymbol{\theta} &\sim \mathcal{N}\bigl(\mathbf{0},\ (\lambda\,\operatorname{diag}\boldsymbol{\kappa})^{-1}\bigr),
\end{aligned}
$$

with $\boldsymbol{\Phi}$ the separable DCT-II basis without its constant term
(so the geometric mean of $\mathbf{f}$ is 1), $\boldsymbol{\kappa}$ the
bending energies of its functions, $\mu_k$, $\nu_k$ the mean and shape of
class $k$ (Gamma in the $(\mu, \nu)$ form of the
[notation](../../concepts/notation.md)), and a weak inverse-gamma-like prior
on each $v_p$, $\log p(v_p) = -\tfrac12 \nu_v (\log v_p + v_0/v_p)$
(`v_dof` $=\nu_v$, `v_prior` $=v_0$).

Assumptions: the frames are independent, every pixel is empty in a fair share
of them (so $\mathbf{c}$ and $\mathbf{v}$ are identified), the flat field is
smooth, and the gain is known: in this model $g$ is identified only through
the shape of the signal distribution, so it is held at its nominal value.
Data are quantised: each value stands for a uniform bin of width $\Delta$,
which adds $\Delta^2/12$ to every squared residual and keeps the bound finite
for pixels that repeat the same value.

After fitting, `concentration_` holds the concentrations
$\boldsymbol{\alpha}_i$ of $q(\boldsymbol{\pi}_i)$ for every training frame,
and `apply(..., output="mean")` returns the posterior mean
$\mathbb{E}[u \mid y]$ of the signal of each pixel (0 in the background).

## Inference / derivation

Variational EM. The posterior is approximated by

$$
q = \prod_i q(\boldsymbol{\pi}_i)\ \prod_{i,p} q(z_{ip})\, q(u_{ip} \mid z_{ip})
\ \cdot\ \delta(\mathbf{c}, \mathbf{v}, \boldsymbol{\theta}, \boldsymbol{\mu}, \boldsymbol{\nu}, \boldsymbol{\alpha}),
$$

and the evidence lower bound is raised by alternating:

- **local step**: $q(z, u)$ is exact given the rest; the integral over $u$
  is a Gauss-Hermite quadrature in $\log u$, centred on the mode of the
  integrand. $q(\boldsymbol{\pi}_i)$ is Dirichlet (conjugate update).
- **class step**: $\mu_k$, $\nu_k$ are refitted to the aggregated posterior
  of $u$ in each class, $\nu_k$ kept in `nu_bounds`; $\boldsymbol{\alpha}$ by
  [Minka's fixed-point Newton method](https://www.microsoft.com/en-us/research/publication/estimating-dirichlet-distribution/).
- **field step**: $\mathbf{c}$, $\mathbf{v}$, $\boldsymbol{\theta}$ take one
  joint Fisher-scoring step; the per-pixel unknowns are eliminated exactly,
  leaving a small dense system in $\boldsymbol{\theta}$. A step that lowers
  the bound is halved.

The scale ambiguity between $\mathbf{f}$ and $\mathbf{u}$ is resolved by the
basis (geometric mean of $\mathbf{f}$ equal to 1, `meta["flat_norm"] ==
"geomean"`); the ambiguity between background and offset is resolved by
putting all of it in $\mathbf{c}$. Initialisation is data-driven: low
per-pixel order statistics give $\mathbf{c}$ and the background spread,
values above `threshold` spreads are labelled tissue, which gives a first
flat field and a k-means of the classes on $\log y$.

## Algorithm

```text
Input: frames Y (per group), gain g, options (nclass, order, nodes, ...)
Initialise c, v from per-pixel order statistics; f, classes from the tissue pixels
Repeat until the bound improves by less than tol per data point, or max_iterations:
    1. Local step: q(z, u) by quadrature, q(π) Dirichlet        (one pass over frames)
    2. Update μ, ν of each class; α (Minka)
    3. One Fisher-scoring step in (c, v, θ); halve it if the bound drops
Return sensor (c, v, g), shading (f, d = c), classification (α, μ, ν)
apply: x̂ = (y − c) / f, or foreground / class probabilities / E[u | y]
```

Defaults: `nclass=1`, `order=(8, 8)`, `nodes=24`, `max_iterations=100`,
`tol=1e-6`, `lam=1`. Each group of a layout gets its own fit.

## Faithful mode and extensions

There is no paper to be faithful to: `paper` and `recommended` are the same
reference preset, so `faithful=True` and `faithful=False` behave identically.
The main options of that preset:

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `nclass` | number $K$ of tissue classes | `1` | `1` |
| `noise` | variance $v + g f u$ (`"full"`) or $v$ (`"constant"`) | `"full"` | `"full"` |
| `order` | DCT functions of $\log\mathbf{f}$ per axis | `(8, 8)` | `(8, 8)` |
| `gain` | nominal $g$; default the gain of an initial `SensorEstimate`, else 1 | `None` | `None` |
| `nodes` | Gauss-Hermite nodes of the integral over $u$ | `24` | `24` |
| `nu_bounds` | bounds on the shapes $\nu_k$ | `(1.0, 1e3)` | `(1.0, 1e3)` |
| `quantization` | bin width $\Delta$ (`0` continuous, `"auto"` detect) | `1.0` | `1.0` |
| `learn_modulation`, `learn_alpha` | estimate $\mathbf{f}$, $\boldsymbol{\alpha}$ | `True` | `True` |
| `lr`, `max_halvings`, `max_log_step` | damping, step halvings, largest change of $\log\mathbf{f}$ | `1.0`, `4`, `0.5` | `1.0`, `4`, `0.5` |
| `bg_fraction`, `threshold`, `spread_floor` | initialisation | `0.5`, `3.0`, `False` | `0.5`, `3.0`, `False` |
| `concentration` | initial $\boldsymbol{\alpha}$ | `1.0` | `1.0` |
| `v_prior`, `v_dof` | prior $v_0$ (default: median initial $v$) and its strength | `None`, `5.0` | `None`, `5.0` |
| `lam` | precision $\lambda$ of the bending prior | `1.0` | `1.0` |
| `max_iterations`, `tol` | stopping rule | `100`, `1e-6` | `100`, `1e-6` |

## Memory and streaming

The fit reads the data in batches of frames, once per iteration (plus one pass
per halved step), and twice more for the initialisation: once by strips of
rows holding every frame, for the per-pixel order statistics, and once by
frame batches. In memory it keeps per-pixel statistics and the fields; the
cost of a pass grows with frames × pixels × (`nclass`) × `nodes`, so `nodes`
is the main knob. `apply` and `predict` work frame by frame.

## Usage

```python
import optix
from optix.models import (
    random_sensor,
    random_shading,
    simulate_gamma_mixture,
    simulate_sensor,
)
from optix.shading import GammaMixture

# 1. A synthetic stack with known truth: one Gamma class, a flat field, a camera.
shading = random_shading((32, 32), strength=0.2, seed=0)
draw = simulate_gamma_mixture((40, 32, 32), [200.0], [3.0], seed=1)
sensor = random_sensor((32, 32), gain=2.0, offset_sd=0.0, seed=2)
Y = simulate_sensor(draw.signal * shading.flat / 2.0, sensor, seed=3).round()

# 2. Fit.
mix = GammaMixture(gain=2.0, max_iterations=20).fit(Y)

# 3. Correct, and map the foreground.
corrected = mix.apply(Y)
foreground = mix.apply(Y, output="foreground")

# 4. Compare to the truth (flat fields with geometric mean 1).
flat = mix.estimates["shading"].flat
truth = shading.flat / shading.flat.log().mean().exp()
print(float((flat / truth - 1).abs().max()))
print(mix.estimates["classification"].parameters)  # about [[200, 3]]
```

Common variations: `nclass=2` for two tissue types; an initial
`SensorEstimate` (`fit(Y, init=...)`) to supply the gain; `quantization="auto"`
for non-integer data; `mix.predict(Y)` for class probabilities of new frames.
See the [API reference](../../reference/shading.md).

## References

!!! references "References"

    1. Minka TP. *Estimating a Dirichlet distribution.* Technical report,
       Microsoft Research, 2000.
       [microsoft.com/en-us/research](https://www.microsoft.com/en-us/research/publication/estimating-dirichlet-distribution/)

    ??? quote "BibTeX"

        ```bibtex
        @techreport{minka2000dirichlet,
          author      = {Minka, Thomas P.},
          title       = {Estimating a Dirichlet distribution},
          institution = {Microsoft Research},
          year        = {2000},
          url         = {https://www.microsoft.com/en-us/research/publication/estimating-dirichlet-distribution/},
        }
        ```
