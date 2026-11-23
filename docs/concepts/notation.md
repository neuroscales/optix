# Notation

One notation is used across every page of these docs and every docstring of
optix, whatever the algorithm and whatever the paper it comes from. This page
states the rules, lists the symbols, and says how the quantities that the
algorithms estimate relate to the physics of the microscope. Each algorithm
page shows its equations in this notation and, in a second tab, in the notation
of its paper.

## Typography

| Kind | Style | Examples |
|---|---|---|
| scalar | italic, lowercase | $g$, $o$, $\sigma^2$, $\lambda$, $b_i$ |
| vector: an image flattened over its pixels, or a set of per-frame values | bold lowercase | $\mathbf{y}_i$, $\mathbf{x}_i$, $\mathbf{f}$, $\mathbf{d}$, $\mathbf{b}$ |
| matrix or linear operator | bold uppercase | $\mathbf{A}$, $\boldsymbol{\Phi}$, $\mathbf{W}$, $\mathbf{Y}$ |
| count, size | plain uppercase | $N$, $K$, $P$, $H$, $W$ |
| Greek vector or matrix | bold Greek | $\boldsymbol{\alpha}$, $\boldsymbol{\pi}$, $\boldsymbol{\eta}$ |
| set, distribution family | calligraphic or upright | $\mathcal{N}$, $\mathrm{Gamma}$, $\mathcal{D}$ |

- Images are vectors. A picture of $H \times W$ pixels is a vector of
  $P = HW$ values, and a stack of $N$ frames is the matrix
  $\mathbf{Y} = [\mathbf{y}_1, \ldots, \mathbf{y}_N] \in \mathbb{R}^{P \times N}$,
  one frame per column.
- Products of images are pixel-wise and written $\odot$; division of images
  is pixel-wise too. A scalar added to an image is broadcast:
  $\mathbf{x}_i + b_i$ means $\mathbf{x}_i + b_i \mathbf{1}$.
- The position of a pixel is rarely needed. When it is, it is written
  $\mathbf{r}$, and a pixel of an image $\mathbf{f}(\mathbf{r})$ or $f_p$.
- Hats mark estimates ($\hat{\mathbf{f}}$), bars means ($\bar{x}$).
- Probability distributions follow Bishop's [*Pattern Recognition and Machine
  Learning*](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) (2006). Some letters therefore have a second meaning there (the rate $b$
  of a Gamma is not a baseline); the context always tells which is meant.

### In docstrings

Docstrings write maths in UTF-8, not LaTeX, so that they read well in
`help()`, in an editor and in the API reference alike (see the
[style guide](../contributing/style.md)). Bold is not available there, so the
case carries the meaning: `yᵢ = f ⊙ (xᵢ + bᵢ) + d`, `A x`, `Φ θ`. The
symbols are the same as on this page.

Indices and exponents use Unicode subscripts and superscripts wherever they
exist: `xᵢ`, `bᵢ`, `σ²`, `Φᵀ`, `τ⁻¹`, `x̂ᵢ`, `rₙₖ`, `α₀`, `10⁻⁶`. Superscripts
cover the digits, `⁺ ⁻ ⁼ ⁽ ⁾`, almost every lowercase letter and `ᵀ`.
Subscripts cover only the digits, `₊ ₋ ₌ ₍ ₎` and the letters
`ₐ ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ`; for any other index (`y_{ig}`,
`x_{max}`, `b_g`), fall back to `_` with braces when it spans several
characters. Do not mix the two styles within one symbol: `r_{nk}` or `rₙₖ`,
never `rₙ_k`.

## Image formation

| Symbol | Meaning | Unit | In code |
|---|---|---|---|
| $\mathbf{y}_i$, $\mathbf{Y}$ | recorded frame $i$; stack of frames | ADU | the data `Y` |
| $\mathbf{x}_i$ | latent image: frame $i$ without shading | ADU | result of `apply` |
| $\mathbf{f}$ | flat field (multiplicative) | dimensionless | `ShadingEstimate.flat` |
| $\mathbf{d}$ | dark field (additive, camera offset included) | ADU | `ShadingEstimate.dark` |
| $b_i$, $\mathbf{b}$ | baseline of frame $i$; all baselines | ADU | `ShadingEstimate.baseline` |
| $\lambda$ | expected number of photoelectrons | e⁻ | `signal` of `simulate_sensor` |
| $n$ | actual number of photoelectrons (Poisson) | e⁻ | |
| $g$ | gain | ADU / e⁻ | `SensorEstimate.gain` |
| $o$ | camera offset | ADU | `SensorEstimate.offset` |
| $\sigma^2$ | read-noise variance | ADU² | `SensorEstimate.variance` |
| $\varepsilon$ | read noise, $\mathcal{N}(0, \sigma^2)$ | ADU | |
| $\Delta$ | quantisation step (1 ADU) | ADU | |

A camera model with one value per pixel, as for an sCMOS sensor, makes the
camera parameters vectors: $\mathbf{g}$, $\mathbf{o}$, $\boldsymbol{\sigma}^2$.

The two models are

$$
y = g\,n + o + \varepsilon, \quad n \sim \mathrm{Poisson}(\lambda)
\qquad\text{and}\qquad
\mathbf{y}_i = \mathbf{f} \odot (\mathbf{x}_i + b_i) + \mathbf{d},
$$

and correcting a frame means

$$
\hat{\mathbf{x}}_i = \frac{\mathbf{y}_i - \mathbf{d}}{\mathbf{f}} - b_i .
$$

The baseline sits *inside* the flat field: it is a uniform background of the
specimen (autofluorescence, fluorescence of the medium), lit by the same
uneven illumination as the signal.

### What the fields mean physically

The flat field, the dark field and the baseline are defined statistically:
the flat field is whatever multiplies every frame alike, the dark field
whatever is added to every frame alike, the baseline whatever is constant over
one frame. Written out, the expected recorded value of a pixel $p$ is

$$
\mathbb{E}[y_{ip}] =
g_p \Bigl(
\underbrace{e_p\, c_p\, \eta_p}_{\text{multiplicative}}\,(u_{ip} + a_{ip})
+ \underbrace{\ell_p + j_p\, t}_{\text{additive}}
\Bigr) + o_p ,
$$

and each physical quantity lands in one of the estimated terms:

| Physical quantity | Symbol | Lands in |
|---|---|---|
| illumination profile | $e_p$ | flat $\mathbf{f}$ |
| collection efficiency, vignetting | $c_p$ | flat $\mathbf{f}$ |
| quantum efficiency and gain of each pixel (photo-response non-uniformity) | $\eta_p$, $g_p$ | flat $\mathbf{f}$ |
| specimen signal | $u_{ip}$ | latent $\mathbf{x}_i$ |
| autofluorescence, medium, out-of-focus haze | $a_{ip}$ | baseline $b_i$ where uniform, latent $\mathbf{x}_i$ otherwise |
| stray or ambient light, outside the illumination path | $\ell_p$ | dark $\mathbf{d}$ |
| dark current, over exposure time $t$ | $j_p\, t$ | dark $\mathbf{d}$ |
| camera offset, including its fixed pattern | $o_p$ | dark $\mathbf{d}$ |
| overall scale: lamp power, mean efficiency, mean gain | | shared by $\mathbf{f}$ and $\mathbf{x}_i$; fixed by `meta["flat_norm"]` |

So the flat field is not the illumination: it is illumination × collection ×
pixel response, which is why it overlaps with the gain map of an sCMOS camera.
And the dark field is not the camera offset: it is offset + dark current +
stray light. This is the reason why corrections never subtract both
`ShadingEstimate.dark` and `SensorEstimate.offset`.

## Bases and regularisation

| Symbol | Meaning | In code |
|---|---|---|
| $P$ | number of samples along an axis | |
| $\boldsymbol{\Phi}$ | DCT basis matrix, one function per column | `dct_basis` |
| $\boldsymbol{\theta}$, $\boldsymbol{\Theta}$ | coefficients in the basis (1D, 2D) | `project`, `reconstruct` |
| $\kappa_j$ | eigenvalue of the $j$-th basis function for the Laplacian | `dct_bending` |
| $h$, $\ell$ | grid spacing; order of the energy $\mathbf{f}^\top(-\boldsymbol{\Delta}_h)^\ell \mathbf{f}$ | `dct_bending(order=)` |
| $\mathbf{A}$ | forward operator (blur) | |
| $\lambda_{\mathrm{TV}}$ | weight of the total-variation penalty $\sum_p \lVert\nabla\mathbf{x}\rVert_p$ | `lambda_tv` |
| $\nabla$, $\operatorname{div}$ | gradient and divergence of an image (forward and backward differences) | `tv_curvature` |
| $\mathbf{W}$ | pixel weights | |

## Distributions

These follow [Bishop (2006)](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/).

| Symbol | Meaning | In code |
|---|---|---|
| $\mathcal{N}(m, \tau^{-1})$ | Normal with mean $m$ and precision $\tau = 1/v$ | `Normal` |
| $\mathrm{Gamma}(a, b)$ | Gamma with shape $a$ and rate $b$ | `Gamma` |
| $\mu$, $\nu$ | Gamma mean $a/b$ and shape, in the form $(\mu, \nu)$ | `Gamma(mean=, shape=)` |
| $\boldsymbol{\alpha}$, $\alpha_0$ | Dirichlet concentration, and $\alpha_0 = \sum_k \alpha_k$ | `Dirichlet` |
| $\boldsymbol{\pi}$ | class probabilities | `Categorical.probs` |
| $z_n$, $r_{nk}$ | class of observation $n$; responsibility of class $k$ for it | |
| $T$, $\boldsymbol{\eta}$, $A$ | sufficient statistics, natural parameters, log-normaliser | `expected_stats`, `natural_params`, `log_normalizer` |
| $\bar{x}$, $M$ | weighted mean, and scatter about it | `NormalStats.mean`, `.scatter` |
| $\chi(x)$ | base measure of an exponential family | |
| $\psi$, $\Gamma$ | digamma and gamma functions | |
| $\rho(a)$, $\delta(a)$, $\varphi(u)$ | Stirling residual, $\log a - \psi(a)$, $u - 1 - \log u$ | `stirling_residual`, `log_minus_digamma`, `ratio_minus_log` |
| $\theta$ | difference of two means, in the pairwise merge | |
| $\mathbb{E}[\cdot]$, $\mathrm{KL}(q \Vert p)$ | expectation; Kullback-Leibler divergence | `kl_divergence(q, p)` |

## Indices and layouts

| Symbol | Ranges over |
|---|---|
| $i$ | frames (samples of a stack), $1 \ldots N$ |
| $p$ | pixels, $1 \ldots P$ |
| $n$ | observations fed to a distribution |
| $k$ | classes, $1 \ldots K$ |
| $j$ | basis functions |
| $t$ | iterations; $L_t$ is the objective at iteration $t$ (`History.objective`) |
| $G_1, \ldots$; $X_1, \ldots$ | sizes of the group and spatial axes of an array |

In prose, *sample* means one picture of a stack, as in the
[layout](axes-and-layouts.md) roles, and *observation* means one value fed to
a distribution. Shapes are written in code font, as in `(*group, *spatial)`.

## Native notations

Every algorithm page writes its model twice, in tabs: in optix notation, and
in the notation of its paper, followed by a table that maps one onto the
other. For BaSiC, for instance:

=== "optix"

    $$
    \mathbf{y}_i = \mathbf{f} \odot (\mathbf{x}_i + b_i) + \mathbf{d}
    $$

=== "Peng et al. 2017"

    $$
    I_i^{\text{meas}}(x) = S^F(x)\,\bigl(B_i + I_i^R(x)\bigr) + S^D(x)
    $$

| optix | [Peng et al. 2017](https://doi.org/10.1038/ncomms14836) | CIDRE ([Smith et al. 2015](https://doi.org/10.1038/nmeth.3323)) |
|---|---|---|
| $\mathbf{y}_i$ | $I_i^{\text{meas}}$ | $I_o$ |
| $\mathbf{x}_i$ | $I_i^R$ | $I$ |
| $\mathbf{f}$ | $S^F$ | $v$ |
| $\mathbf{d}$ | $S^D$ | $z$ |
| $b_i$ | $B_i$ | — |

!!! references "References"

    1. Bishop CM. *Pattern Recognition and Machine Learning.* Springer, 2006.
       [Microsoft Research](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
    2. Peng T, Thorn K, Schroeder T, Wang L, Theis FJ, Marr C, Navab N.
       *A BaSiC tool for background and shading correction of optical
       microscopy images.* Nat Commun **8**:14836, 2017.
       [doi:10.1038/ncomms14836](https://doi.org/10.1038/ncomms14836)
    3. Smith K, Li Y, Piccinini F, Csucs G, Balazs C, Bevilacqua A, Horvath P.
       *CIDRE: an illumination-correction method for optical microscopy.*
       Nat Methods **12**(5):404-406, 2015.
       [doi:10.1038/nmeth.3323](https://doi.org/10.1038/nmeth.3323)

    ??? quote "BibTeX"

        ```bibtex
        @book{bishop2006pattern,
          author    = {Bishop, Christopher M.},
          title     = {Pattern Recognition and Machine Learning},
          publisher = {Springer},
          address   = {New York},
          year      = {2006},
          isbn      = {978-0-387-31073-2},
          url       = {https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/},
        }
        @article{peng2017basic,
          author  = {Peng, Tingying and Thorn, Kurt and Schroeder, Timm and Wang, Lichao and Theis, Fabian J. and Marr, Carsten and Navab, Nassir},
          title   = {A {BaSiC} tool for background and shading correction of optical microscopy images},
          journal = {Nature Communications},
          volume  = {8},
          pages   = {14836},
          year    = {2017},
          doi     = {10.1038/ncomms14836},
        }
        @article{smith2015cidre,
          author  = {Smith, Kevin and Li, Yunpeng and Piccinini, Filippo and Csucs, Gabor and Balazs, Csaba and Bevilacqua, Alessandro and Horvath, Peter},
          title   = {{CIDRE}: an illumination-correction method for optical microscopy},
          journal = {Nature Methods},
          volume  = {12},
          number  = {5},
          pages   = {404--406},
          year    = {2015},
          doi     = {10.1038/nmeth.3323},
        }
        ```
