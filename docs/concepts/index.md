# Concepts

These pages explain the ideas behind optix, without going through every
function. They are meant to be read once, in order, and then used as a reference
when an algorithm page assumes something. Each page runs on the same few
objects, and each example can be pasted into a Python session.

- [Axes and layouts](axes-and-layouts.md) explains how an array is read as a
  pile of pictures, with axes that are *spatial*, *sample* or *group*. It
  covers OME-Zarr stacks, light-sheet volumes and the streaming of arrays that
  do not fit in memory.
- [Image formation and noise](image-formation.md) says what a pixel value
  means: the camera model, photon transfer, quantisation, and the shading model
  $I = S\,I^0 + D$. It defines the conventions that results follow.
- [Estimators](estimators.md) describes the `fit` / `apply` contract, faithful
  mode and extensions, starting one estimator from another, and the choice of
  device, precision and memory.
- [Probability distributions](distributions.md) presents the Gamma, Normal,
  Dirichlet and Categorical families: their shapes, sufficient statistics and
  conjugate updates, and what keeps them numerically safe.

## Notation

One notation is used across all pages. Symbols are defined where they first
appear and recalled here.

| Symbol | Meaning | In code |
|---|---|---|
| $H$, $W$ | height and width of a picture | `Y.shape` |
| $N$ | number of samples (frames), or total weight of a set of observations | `chunk.n`, `stats.n` |
| $K$ | number of classes, or categories | |
| $y$ | recorded value, in ADU | the data `Y` |
| $\lambda$ | expected number of photoelectrons | `signal` of `simulate_sensor` |
| $n$ | actual number of photoelectrons (Poisson) | |
| $g$ | gain, in ADU per electron | `SensorEstimate.gain` |
| $o$ | camera offset, in ADU | `SensorEstimate.offset` |
| $\sigma^2$ | read-noise variance, in ADU$^2$ | `SensorEstimate.variance` |
| $\varepsilon$ | read noise, $\mathcal{N}(0, \sigma^2)$ | |
| $\Delta$ | quantisation step (1 ADU) | |
| $I^0$ | image without shading, in ADU | |
| $S$ | flat field (multiplicative shading) | `ShadingEstimate.flat` |
| $D$ | dark image (additive term, camera offset included) | `ShadingEstimate.dark` |
| $\beta_i$ | baseline of frame $i$ | `ShadingEstimate.baseline` |
| $P$, $B$, $\kappa_j$ | number of samples along an axis; DCT basis matrix; eigenvalue of its $j$-th function | `dct_basis`, `dct_bending` |
| $h$, $\ell$ | grid spacing; order of the derivative energy $f^\top(-\Delta_h)^\ell f$ | `dct_bending(order=)` |
| $\mathcal{N}(m, \tau^{-1})$ | Normal with mean $m$ and precision $\tau = 1/v$ | `Normal` |
| $\mathrm{Gamma}(a, b)$ | Gamma with shape $a$ and rate $b$ | `Gamma` |
| $\mu$ | Gamma mean, $\mu = a/b$ | `Gamma.mean` |
| $\nu$ | Gamma shape in the mean-and-shape form $(\mu, \nu)$ | `Gamma(mean=, shape=)` |
| $\alpha_k$, $\alpha_0$ | Dirichlet concentration, and $\alpha_0 = \sum_k \alpha_k$ | `Dirichlet` |
| $\pi_k$ | class probability (Dirichlet variable, Categorical parameter) | `Categorical.probs` |
| $r_{nk}$ | responsibility: probability that sample $n$ is in class $k$ | |
| $T$, $\eta$, $A$ | sufficient statistics, natural parameters, log-normaliser | `expected_stats`, `natural_params`, `log_normalizer` |
| $\bar{x}$, $M$ | weighted mean, and sum of squared deviations from it (scatter) | `NormalStats.mean`, `.scatter` |
| $\chi(x)$ | base measure of an exponential family | |
| $\psi$, $\Gamma$ | digamma function and gamma function | |
| $\rho(a)$, $\delta(a)$, $\varphi(u)$ | Stirling residual, $\log a - \psi(a)$ and $u - 1 - \log u$, in their series forms | `stirling_residual`, `log_minus_digamma`, `ratio_minus_log` |
| $G_i$, $X_i$ | group axes and spatial axes of an array | |
| $\theta$ | difference of two means, in the pairwise merge | |
| $\mathbb{E}[\cdot]$ | expectation | |
| $\mathrm{KL}(q \Vert p)$ | Kullback-Leibler divergence | `kl_divergence(q, p)` |
| $L_t$ | objective at iteration $t$ | `History.objective` |

In prose, *sample* means one picture of a stack, as in the
[layout](axes-and-layouts.md) roles, and *observation* means one value fed to a
distribution. Shapes are written in code font, as in `(*group, *spatial)`.
