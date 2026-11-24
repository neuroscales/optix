# Optimisation tools

Regularised estimators minimise a sum of simple terms: a data-fidelity term, a
penalty on differences of the image, a constraint. The module `optix.optim`
holds the pieces they are assembled from, so that an estimator is a few lines
that choose terms, and the numerical care is written once. It has three layers.

| Layer | Names | What it is |
|---|---|---|
| Proximity operators and functions | `prox_box`, `prox_group_l2`, `project_group_l2`, `prox_squared_l2`, `prox_shifted_poisson`, `prox_schatten`, `schatten_norm`, `prox_conjugate`; `Box`, `GroupL2`, `SquaredL2`, `ShiftedPoisson`, `Schatten` | the building blocks of splitting algorithms |
| Smooth functions | `Smooth`, `PoissonGaussian` | a differentiable term with a Lipschitz gradient, for the primal–dual solver |
| Linear operators | `LinearOperator`, `Circulant`; `Identity`, `Convolution`, `as_circulant`; `diff`, `diff_adjoint`, `Gradient`, `Hessian` | the operator protocols, a transfer function as an operator, and finite differences with their Fourier symbols |
| Solvers | `Term`, `admm_fft`, `AdmmResult`; `primal_dual`, `primal_dual_step`, `PrimalDualResult` | ADMM for circulant operators, solved in the Fourier domain; primal–dual splitting with a smooth term, which needs no inversion |

The module is semi-public: it is documented in the
[API reference](../reference/optim.md) and may change without notice. It does not
depend on `optix.deconv`; [Ikoma](../algorithms/deconv/ikoma.md) is its first user,
[HSPIRAL](../algorithms/deconv/hspiral.md) uses it too, and
[Chouzenoux](../algorithms/deconv/chouzenoux.md) uses the primal–dual solver.

```pycon
>>> import torch
>>> from optix.optim import (
...     Box,
...     Gradient,
...     GroupL2,
...     Hessian,
...     PoissonGaussian,
...     SquaredL2,
...     Term,
...     admm_fft,
...     primal_dual,
...     primal_dual_step,
...     project_group_l2,
...     prox_group_l2,
... )
>>> _ = torch.manual_seed(0)
```

## Proximity operators

The proximity operator of a convex function $f$ with a step $t > 0$ is

$$
\mathrm{prox}_{tf}(\mathbf{v}) = \arg\min_{\mathbf{z}} \; f(\mathbf{z}) + \frac{1}{2t} \lVert \mathbf{z} - \mathbf{v} \rVert^2 ,
$$

the point that trades off staying near $\mathbf{v}$ against lowering $f$. Splitting
algorithms such as ADMM only need this operator for each non-smooth term, and for
the terms of imaging it has a closed form. The module provides the following.

| Function | Operator | Name |
|---|---|---|
| indicator of a box $[l, u]$ | projection, $\min(\max(\mathbf{v}, l), u)$; any $t$ | `prox_box`, `Box` |
| $w\sum_j \lVert \mathbf{z}_j \rVert_2$ (mixed $\ell_{2,1}$) | block soft threshold, $\max(1 - tw / \lVert\mathbf{v}_j\rVert_2, 0)\,\mathbf{v}_j$ | `prox_group_l2`, `GroupL2` |
| indicator of the $\ell_2$ ball of radius $r$ | $\mathbf{v}_j \min(1, r / \lVert\mathbf{v}_j\rVert_2)$ | `project_group_l2` |
| $\tfrac{w}{2}\lVert \mathbf{z} - \mathbf{y} \rVert^2$ | $(\mathbf{v} + tw\mathbf{y}) / (1 + tw)$ | `prox_squared_l2`, `SquaredL2` |
| $\sum_j z_j - (y_j + \tilde\sigma^2)\log(z_j + \tilde\sigma^2)$ | root of a quadratic, see below | `prox_shifted_poisson`, `ShiftedPoisson` |
| $w\sum_j \lVert \mathbf{X}_j \rVert_{S_p}$ (Schatten norm of a field of symmetric matrices), $p \in \{1, 2, \infty\}$ | a shrinkage of the singular values, see below | `prox_schatten`, `Schatten` |

In the group norm, $\mathbf{z}_j$ collects the components of $\mathbf{z}$ along one
axis (`dim`) at voxel $j$: the gradient for isotropic total variation, the packed
Hessian for its Frobenius norm. The group soft threshold and the projection
onto the ball are linked by the Moreau identity, for any $\mathbf{v}$ and $r$,

$$
\mathbf{v} = \mathrm{prox}_{tf}(\mathbf{v}) + t\, \mathrm{prox}_{f^*/t}(\mathbf{v}/t),
\qquad\text{so that}\qquad
\mathbf{v} = \mathrm{prox}_{r\lVert\cdot\rVert_{2,1}}(\mathbf{v}) + \mathrm{proj}_{r\text{-ball}}(\mathbf{v}),
$$

and `prox_conjugate(prox, v, t)` applies the identity to turn the prox of any
$f$ into that of its convex conjugate $f^*$. On a box, `prox_box(v, lower, upper)`
defaults to the non-negative orthant, $[0, +\infty)$.

```pycon
>>> v = torch.tensor(
...     [[3.0, 0.0], [4.0, 0.5]]
... )  # two voxels, components along dim 0
>>> prox_group_l2(v, 1.0)
tensor([[2.4000, 0.0000],
        [3.2000, 0.0000]])
>>> project_group_l2(v, 1.0)
tensor([[0.6000, 0.0000],
        [0.8000, 0.5000]])
>>> prox_group_l2(v, 1.0) + project_group_l2(v, 1.0) - v
tensor([[0., 0.],
        [0., 0.]])
```

### The shifted-Poisson prox

The shifted-Poisson fidelity of [Ikoma](../algorithms/deconv/ikoma.md),
$f(\mathbf{z}) = \sum_j z_j - c_j \log(z_j + \tilde\sigma^2)$ with
$c = \tilde{y} + \tilde\sigma^2 \ge 0$, is the negative log-likelihood of
$\tilde{y} + \tilde\sigma^2 \sim \mathrm{Poisson}(z + \tilde\sigma^2)$; with
$\tilde\sigma^2 = 0$ it is the Poisson one. Its stationarity condition is a
quadratic in $w = z + \tilde\sigma^2$, with $\rho = 1/t$ and $s = v + \tilde\sigma^2$,

$$
\rho w^2 + (1 - \rho s)\, w - c = 0 ,
$$

whose non-negative root is computed without cancellation: with $q = \rho s - 1$ and
$\Delta = \sqrt{q^2 + 4\rho c}$,

$$
w = \frac{q + \Delta}{2\rho} \;\; (q \ge 0),
\qquad
w = \frac{2c}{\Delta - q} \;\; (q < 0),
\qquad
z = w - \tilde\sigma^2 .
$$

The two branches are equal in exact arithmetic, and each adds numbers of the
same sign. `ShiftedPoisson(data, variance)` also provides `grad` and the
Lipschitz constant `lipschitz` of the gradient, $\max_j (y_j + \sigma_j^2) / \sigma_j^4$ (infinite for
the Poisson case), which first-order methods need to choose a step.

### Schatten norms

The Schatten norm of order $p$ of a matrix is the $\ell_p$ norm of its singular
values $\boldsymbol\sigma$, $\lVert\mathbf{X}\rVert_{S_p} = \lVert\boldsymbol\sigma\rVert_p$:
the nuclear norm for $p = 1$, the Frobenius norm for $p = 2$ and the spectral norm for
$p = \infty$. Applied to the Hessian matrix $\mathbf{X}_j$ at every pixel, as in
[HSPIRAL](../algorithms/deconv/hspiral.md), it penalises the principal curvatures of
the image. The functions take $\mathbf{X}_j$ packed as the output of `Hessian`
(diagonal entries, then $\sqrt{2}$ times the entries above the diagonal, $C = d(d+1)/2$
components along `dim`), so that $\lVert\mathbf{v}_j\rVert_2$ is the Frobenius norm;
`schatten_norm(v, p, dim=0)` returns $\lVert\mathbf{X}_j\rVert_{S_p}$ per voxel and
`Schatten(weight, p)` is the function $w\sum_j \lVert\mathbf{X}_j\rVert_{S_p}$, with
`value` and `prox`.

The proximity operator of a Schatten norm acts on the singular values only
([Lefkimmiatis and Unser 2013](https://doi.org/10.1109/TIP.2013.2271852),
Proposition 1): if $\mathbf{Y} = \mathbf{U}\,\mathrm{diag}(\boldsymbol\sigma)\,\mathbf{V}^H$,

$$
\mathrm{prox}_{t\lVert\cdot\rVert_{S_p}}(\mathbf{Y})
= \mathbf{U}\,\mathrm{diag}\bigl(\mathrm{prox}_{t\lVert\cdot\rVert_p}(\boldsymbol\sigma)\bigr)\,\mathbf{V}^H ,
$$

with the $\ell_p$ proxes of the paper (Eqs. 43–47):

| $p$ | $\mathrm{prox}_{t\lVert\cdot\rVert_p}(\boldsymbol\sigma)$ |
|---|---|
| 1 | soft threshold, $\hat\sigma_i = \max(\sigma_i - t, 0)$ |
| 2 | block soft threshold of the whole matrix, $\max(1 - t/\lVert\mathbf{Y}\rVert_F, 0)\,\mathbf{Y}$: `prox_group_l2` on the packed vector |
| $\infty$ | $\hat\sigma_i = \min(\sigma_i, c)$, with $c = 0$ if $\sigma_1 + \sigma_2 \le t$, $c = (\sigma_1 + \sigma_2 - t)/2$ if $\lvert\sigma_1 - \sigma_2\rvert < t < \sigma_1 + \sigma_2$, and $c = \sigma_1 - t$ if $\sigma_1 - \sigma_2 \ge t$ ($\sigma_1 \ge \sigma_2$) |

The paper states the case $p = \infty$ for a threshold of 1; the form above is the
same with the threshold written out ($c = t\,\gamma(\boldsymbol\sigma/t)$).

For a symmetric $2 \times 2$ matrix, which is the Hessian of a 2-D image, no
decomposition is computed. Its eigenvalues are $\mu_\pm = m \pm r$, with $m$ the
mean of the diagonal and $r \ge 0$ half the gap between them, and their absolute values
are the singular values. The $\ell_p$ prox is applied to $\mu_\pm$ keeping their signs, to
give $\hat\mu_\pm$, and

$$
\hat{\mathbf{X}} = \hat{m}\,\mathbf{I} + \frac{\hat{r}}{r}\,(\mathbf{X} - m\,\mathbf{I}),
\qquad
\hat{m} = \frac{\hat\mu_+ + \hat\mu_-}{2},
\quad
\hat{r} = \frac{\hat\mu_+ - \hat\mu_-}{2},
$$

which keeps the eigenvectors of $\mathbf{X}$ ($\hat{\mathbf{X}} = \hat{m}\mathbf{I}$ where
$r = 0$). The packing is an isometry for the Frobenius inner product, so this is
exactly the matrix prox. The norms $p = 1$ and $p = \infty$ are available for
$2 \times 2$ matrices (and $1 \times 1$ ones) only; $p = 2$ for any size, and a
`NotImplementedError` says so otherwise.

```pycon
>>> from optix.optim import Schatten, prox_schatten, schatten_norm
>>> v = torch.tensor([[3.0], [-1.0], [0.0]])  # one voxel, diag(3, -1)
>>> [round(float(schatten_norm(v, p)), 4) for p in (1, 2, float("inf"))]
[4.0, 3.1623, 3.0]
>>> prox_schatten(v, 0.5, 1).T  # eigenvalues 3 and -1 become 2.5 and -0.5
tensor([[ 2.5000, -0.5000,  0.0000]])
>>> prox_schatten(v, 0.5, float("inf")).T  # the larger one is capped at 2.5
tensor([[ 2.5000, -1.0000,  0.0000]])
>>> float(Schatten(0.5, 1).value(v))
2.0
```

## Linear operators

A solver needs the same few things from every operator, so they are written as
two protocols. A `LinearOperator` maps an image `(..., *N)` to `(..., C, *N)`, the $C$
components stacked along dim $-(d+1)$, and has `forward`, `adjoint` and `norm2`, an
upper bound on $\lVert\mathbf{A}\rVert^2$ (the largest eigenvalue of
$\mathbf{A}^\top\mathbf{A}$), for step sizes. A `Circulant` operator is a
`LinearOperator` that is diagonal in the Fourier domain: each of its components is a
periodic convolution, given by its transfer function on the half spectrum of
`torch.fft.rfftn`. It adds `symbols(grid)`, the list of the $C$ transfer functions, and
`gram(grid)`, $\sum_c |\hat{A}_c|^2$, the transfer function of $\mathbf{A}^\top\mathbf{A}$.
Both protocols are checkable at run time.

Three small pieces make every operand of `admm_fft` a `Circulant`:

- `Identity(ndim)` is the identity, with one component and `norm2` equal to 1.
- `Convolution(transfer, grid)` wraps a transfer function, for instance the `otf` of
  `PaddedConvolution`, as a one-component operator with `norm2` equal to
  $\max|\hat{A}|^2$.
- `as_circulant(operator, grid)` returns the operator that an argument means:
  `None` is the identity, a tensor is a convolution, and a `Circulant` is returned as
  it is. Anything else raises a `TypeError`.

`Gradient` and `Hessian`, below, are `Circulant` operators on a periodic grid.

```pycon
>>> from optix.optim import as_circulant
>>> grid = (8, 8)
>>> otf = torch.fft.rfftn(torch.ones(8, 8) / 64)  # a uniform blur
>>> A = as_circulant(otf, grid)
>>> type(A).__name__, type(as_circulant(None, grid)).__name__
('Convolution', 'Identity')
>>> x = torch.randn(8, 8)
>>> A.forward(x).shape, round(A.norm2, 3)
(torch.Size([1, 8, 8]), 1.0)
```

## Difference operators

`diff(x, dim)` is the forward difference $(\mathbf{D}x)_i = x_{i+1} - x_i$ along an
axis and `diff_adjoint` its adjoint $(\mathbf{D}^\top y)_i = y_{i-1} - y_i$. The
boundary is `"periodic"` (the image wraps around) or `"neumann"` (the image is
extended by replication, the last difference is 0).

`Gradient(ndim)` stacks the $d$ first differences of an image of $d$ trailing
dimensions along a new axis, $\nabla\mathbf{x} = (\mathbf{D}_1\mathbf{x}, \dots,
\mathbf{D}_d\mathbf{x})$, and `Hessian(ndim)` the $d(d+1)/2$ distinct second
differences $\mathbf{D}_{ab} = \mathbf{D}_a\mathbf{D}_b$, in the packing in which
the Frobenius norm of the Hessian matrix is a plain Euclidean norm:

$$
\mathbf{H}\mathbf{x} =
\bigl(\mathbf{D}_{11}\mathbf{x}, \dots, \mathbf{D}_{dd}\mathbf{x},\;
\sqrt{2}\,\mathbf{D}_{12}\mathbf{x}, \dots\bigr),
\qquad
\lVert (\mathbf{H}\mathbf{x})_j \rVert_2 = \lVert \text{Hessian matrix at } j \rVert_F .
$$

So `GroupL2(w)` on the output of `Gradient` is isotropic total variation, and
`GroupL2(w)` on that of `Hessian`, the Hessian–Frobenius penalty. Both take a
weight `weights=` per axis (the difference along axis $a$ is replaced by
$w_a\mathbf{D}_a$), for anisotropic sampling. Components are stacked along dim
$-(d+1)$, so that an image `(..., *N)` maps to `(..., C, *N)` with $C = d$ or $d(d+1)/2$.

Each operator has `forward`, `adjoint`, `norm2` (an upper bound on its squared
operator norm, for step sizes) and, on a periodic grid, **`symbols`**: its
transfer function, one complex tensor per component. A forward difference is
diagonal in the Fourier domain, $\hat{D}_a(\boldsymbol\xi) = e^{2\pi i \xi_a} - 1$ at
frequency $\xi_a = k_a / N_a$, so the operator is never materialised: the
solver multiplies spectra by the symbols. `gram(grid)` returns the symbol of
$\mathbf{A}^\top\mathbf{A}$, $\sum_c |\hat{A}_c|^2$.

```pycon
>>> D, H = Gradient(2), Hessian(2)
>>> a, b = torch.randn(64, 64), torch.randn(2, 64, 64)
>>> D.forward(a).shape, H.forward(a).shape
(torch.Size([2, 64, 64]), torch.Size([3, 64, 64]))
>>> abs(
...     float((D.forward(a) * b).sum() - (a * D.adjoint(b)).sum())
... ) < 1e-3  # adjoint test
True
>>> D.norm2, H.norm2
(8.0, 64.0)
```

## ADMM for circulant operators

`admm_fft` minimises a sum of terms $f_k(\mathbf{A}_k\mathbf{x})$ in which every
$\mathbf{A}_k$ is circulant on a periodic grid, so that $\mathbf{A}_k$ is diagonal in
the Fourier domain:

$$
\min_{\mathbf{x}} \; \sum_k f_k(\mathbf{A}_k\mathbf{x}) .
$$

Each term is a `Term(operator, function, region)`:

- **`operator`** is a [`Circulant`](#linear-operators) operator, such as `Gradient`,
  `Hessian`, `Identity` or `Convolution`; or one of two shorthands, normalised by
  `as_circulant`: `None` (the identity) and a complex tensor (the transfer function of a
  convolution, such as the `otf` of `PaddedConvolution`).
- **`function`** is any object with `value(z)` and `prox(v, t)` (the protocol
  `Function`): `Box`, `GroupL2`, `SquaredL2`, `ShiftedPoisson` or your own.
- **`region`** is a tuple of slices restricting $f_k$ to part of the grid; the term is
  dropped elsewhere. Ikoma's fidelity is applied to the observed region of the
  extended grid this way.

Every term is split, $\mathbf{z}_k = \mathbf{A}_k\mathbf{x}$, and the
iteration alternates, with one penalty $\rho$ and the scaled dual variables $\mathbf{u}_k$
([Boyd et al. 2011, §3.1.1](https://doi.org/10.1561/2200000016)):

$$
\begin{aligned}
\mathbf{x} &\leftarrow \Bigl(\sum_k \mathbf{A}_k^\top\mathbf{A}_k\Bigr)^{-1} \sum_k \mathbf{A}_k^\top(\mathbf{z}_k - \mathbf{u}_k), \\
\mathbf{z}_k &\leftarrow \mathrm{prox}_{f_k/\rho}(\mathbf{A}_k\mathbf{x} + \mathbf{u}_k), \\
\mathbf{u}_k &\leftarrow \mathbf{u}_k + \mathbf{A}_k\mathbf{x} - \mathbf{z}_k .
\end{aligned}
$$

The $\mathbf{x}$-update is exact: a division of spectra, by
$\sum_k |\hat{A}_k|^2$, which must be positive at every frequency (an identity
term guarantees it; the solver checks). Outside a term's region, $\mathbf{z}_k = \mathbf{A}_k\mathbf{x} + \mathbf{u}_k$.

| Argument | Meaning |
|---|---|
| `rho` | penalty $\rho > 0$ |
| `order` | `"x-first"`: $\mathbf{x}$, then $\mathbf{z}$, then $\mathbf{u}$, from $\mathbf{z} = \mathbf{u} = 0$ (as in [Ikoma et al. 2018](https://doi.org/10.1038/s41598-018-29768-x)); `"z-first"`: $\mathbf{z}$ from $(\mathbf{x}, \mathbf{u})$, then $\mathbf{x}$, then $\mathbf{u}$, from `x0` and $\mathbf{u} = 0$ (as in [Lefkimmiatis and Unser 2013](https://doi.org/10.1109/TIP.2013.2271852)) |
| `tol`, `max_iter` | stop when $\lVert\mathbf{x}^{k+1} - \mathbf{x}^k\rVert \le \mathrm{tol}\,\lVert\mathbf{x}^k\rVert$, or after `max_iter` iterations; `tol=None` runs them all |
| `trace` | a dict filled during the run (a callback can read it) with per-iteration lists: `"change"`, `"residual"` (the primal residual), `"dual_residual"` and `"objective"`, $\sum_k f_k(\mathbf{z}_k)$ in float64 (the objective at $\mathbf{x}$ once the primal residual is 0). The last two cost extra FFTs and are computed only with a trace |
| `callback` | `callback(k, x)` after each iteration |

The result is an `AdmmResult` with the last `x`, the splits and duals, the number of
iterations `n_iter`, the last relative `change`, and the last primal and dual
residuals, `primal_residual` and `dual_residual`
([Boyd et al. 2011, §3.3](https://doi.org/10.1561/2200000016)):

$$
\lVert\mathbf{r}\rVert = \Bigl(\sum_k \lVert\mathbf{A}_k\mathbf{x} - \mathbf{z}_k\rVert^2\Bigr)^{1/2},
\qquad
\lVert\mathbf{s}\rVert = \rho \Bigl\lVert \sum_k \mathbf{A}_k^\top(\mathbf{z}_k - \mathbf{z}_k^{\text{prev}}) \Bigr\rVert
$$

for `order="x-first"`; for `"z-first"` the dual residual is
$\rho\lVert\mathbf{A}(\mathbf{x} - \mathbf{x}^{\text{prev}})\rVert$.

!!! warning "A small change is not convergence"
    The stop tests the change of $\mathbf{x}$ only. A solve can stop with a large primal
    or dual residual, and its result is then far from the minimiser (the estimator
    [Ikoma](../algorithms/deconv/ikoma.md) is such a case, by design). Read the
    residuals when the minimiser is what you want.

**Memory.** The persistent state is $\mathbf{x}$ plus one split and one dual per
component of every term, all on the grid. The peak is about twice that, since an
iteration also holds the spectra, the new split, the prox temporaries and a second
$\mathbf{x}$: measured on the whole solver, about $21 + 2.5C$ volumes of the grid in
single precision, with $C$ the components of the penalty (36 for the 3-D Hessian, 28.5
for the 2-D one, 21 without a penalty), a little less in double precision. An iteration
costs about two FFTs of the grid per component of every term, plus a few for
$\mathbf{x}$.

The example removes noise from a square by total variation,
$\min_\mathbf{x} \tfrac12\lVert\mathbf{x} - \mathbf{y}\rVert^2 + 1.5\sum_j\lVert(\nabla\mathbf{x})_j\rVert_2$:
an identity term with a quadratic, and a gradient term with a group norm.

```pycon
>>> x = torch.zeros(64, 64)
>>> x[16:48, 16:48] = 10.0
>>> y = x + 2.0 * torch.randn(64, 64)
>>> terms = [Term(None, SquaredL2(1.0, y)), Term(Gradient(2), GroupL2(1.5))]
>>> trace = {}
>>> res = admm_fft(terms, (64, 64), rho=1.0, max_iter=200, tol=1e-4, trace=trace)
>>> res.n_iter, res.x.shape
(43, torch.Size([64, 64]))
>>> [round(float((e - x).abs().mean()), 2) for e in (y, res.x)]
[1.6, 0.25]
>>> sorted(trace)
['change', 'dual_residual', 'objective', 'residual']
>>> [s.shape for s in res.splits]
[torch.Size([1, 64, 64]), torch.Size([2, 64, 64])]
```

For a Hessian penalty replace `Gradient(2)` with `Hessian(2)`; a non-negativity
constraint is the term `Term(None, Box(0.0))`; a blur is the term `Term(otf, ...)` with
the transfer function of the PSF on the grid.

## Primal–dual splitting

ADMM needs the data term to have a proximity operator, and a linear solve of the
whole problem at every iteration. Some terms have neither: the exact
Poisson–Gaussian likelihood, for instance, is smooth but has no closed-form prox.
`primal_dual` minimises

$$
\min_{\mathbf{x}} \; h(\mathbf{x}) + \psi_0(\mathbf{x}) + \sum_r \psi_r(\mathbf{V}_r\mathbf{x}),
$$

with $h$ convex and differentiable with a $\mu$-Lipschitz gradient, $\psi_0$ and the
$\psi_r$ convex with proximity operators, and any linear operators $\mathbf{V}_r$, which
need not be circulant. It is the forward–backward–forward scheme of
[Combettes and Pesquet (2012)](https://doi.org/10.1007/s11228-011-0191-y), written as
Algorithm 1 of [Chouzenoux et al. (2015)](https://doi.org/10.1137/15M1014395), the
iteration displayed on the [Chouzenoux](../algorithms/deconv/chouzenoux.md#inference-derivation)
page, with one dual variable $\mathbf{v}_r$ per penalty and no error terms. It calls
$\nabla h$ twice, and each $\mathbf{V}_r$ and its adjoint twice, per iteration, and
inverts nothing.

| Argument | Meaning |
|---|---|
| `grad` | a callable, $\nabla h$ |
| `terms` | the pairs $(\mathbf{V}_r, \psi_r)$: a `LinearOperator` and a `Function` (`GroupL2` for TV or the Hessian norm); the dual step uses `prox_conjugate` on the prox of $\psi_r$ |
| `x0`, `duals` | starting point (sets dtype and device) and starting dual variables, zeros by default |
| `step` | $\gamma > 0$, from `primal_dual_step` |
| `constraint` | $\psi_0$, a `Function`, for example `Box(0.0)`; none by default |
| `max_iter`, `tol` | stop when $\lVert\mathbf{x}_{t+1} - \mathbf{x}_t\rVert \le \mathrm{tol}\,\lVert\mathbf{x}_t\rVert$, or after `max_iter` iterations (5000 and $10^{-5}$ by default; these defaults are not in the paper); `tol=None` runs them all |
| `value`, `trace`, `callback` | $h$ for the `"objective"` of the trace; a dict filled with the per-iteration lists `"change"`, `"residual"` and `"objective"` (empty without `value`); `callback(k, p1)` after each iteration |

`primal_dual_step(lipschitz, norms2)` returns the largest admissible step of
[Chouzenoux et al. (2015)](https://doi.org/10.1137/15M1014395), Theorem 3.2 (iii):
with $\delta = \mu + \bigl(\sum_r \lVert\mathbf{V}_r\rVert^2\bigr)^{1/2}$, convergence
holds for $\gamma \in [\epsilon, (1 - \epsilon)/\delta]$, $\epsilon \in (0, 1/(\delta + 1))$,
and the function returns $\gamma = (1 - \epsilon)/\delta$. The paper takes "the maximum
possible value" without giving $\epsilon$; the default here is
$\epsilon = \min(10^{-3}, \tfrac{1}{2(\delta + 1)})$ (not in the paper), and `eps=` sets it.
`norms2` holds upper bounds on $\lVert\mathbf{V}_r\rVert^2$, for which `Gradient.norm2` and
`Hessian.norm2` are given. The function returns 0 for an infinite $\mu$.

The result is a `PrimalDualResult`: `x`, the primal estimate $\mathbf{p}_1$ of the last
iteration, which lies in the domain of $\psi_0$ and has the same limit as the iterates
$\mathbf{x}_t$; the `duals`; `n_iter`; the last relative `change`; and the last `residual`, the norm of the full forward–backward–forward residual
$\bigl(\lVert\mathbf{x}_t - \mathbf{p}_{1,t}\rVert^2 + \sum_r \lVert\mathbf{v}_{r,t} - \mathbf{p}_{2,r,t}\rVert^2\bigr)^{1/2}/\gamma$,
which tends to 0 at a solution. The relative change is about $\gamma$ times a residual, so
the stopping test depends on the step; the residual does not, and is the diagnostic to read
when the minimiser is wanted.

The smooth term is any object with `value`, `grad` and `lipschitz` (the protocol
`Smooth`). `PoissonGaussian(data, variance)` is the exact Poisson–Gaussian negative
log-likelihood of the [Chouzenoux](../algorithms/deconv/chouzenoux.md#probabilistic-model)
page, continued for $z < 0$ by the quadratic of Eq. 2.28 so that it is convex and twice
differentiable on the whole space; its `lipschitz` is $\max_i \eta_i(0)$, which may be
infinite, and `log_lipschitz` is its logarithm, finite when the constant overflows. The
derivative uses `optix.special.poisson_normal_ratio`.

!!! warning "Slow at high counts"
    The Lipschitz constant of `PoissonGaussian` grows like $e^{2\max y/\sigma^2}$, so that
    the step collapses at high counts with a low read noise. The solver then needs a
    number of iterations that grows with $(\max y + \sigma^2)/\gamma$.

The example is the total-variation denoising of the ADMM example above, with the
Gaussian fidelity $h(\mathbf{x}) = \tfrac12\lVert\mathbf{x} - \mathbf{y}\rVert^2$ (so that $\mu = 1$), a
non-negativity constraint and the gradient term. It reaches the minimiser of `admm_fft`
for the same problem, with a box term, to within 0.1.

```pycon
>>> terms = [(Gradient(2), GroupL2(1.5))]
>>> step = primal_dual_step(1.0, [Gradient(2).norm2])  # μ = 1, ‖∇‖² ≤ 8
>>> round(step, 4)
0.2609
>>> trace = {}
>>> res = primal_dual(
...     lambda z: z - y,
...     terms,
...     torch.full_like(y, float(y.mean())),
...     step=step,
...     constraint=Box(0.0),
...     max_iter=2000,
...     tol=1e-5,
...     trace=trace,
... )
>>> res.n_iter, res.x.shape, [d.shape for d in res.duals]
(256, torch.Size([64, 64]), [torch.Size([2, 64, 64])])
>>> [round(float((e - x).abs().mean()), 2) for e in (y, res.x)]
[1.6, 0.2]
>>> float(res.x.min()) >= 0, sorted(trace)
(True, ['change', 'objective', 'residual'])
>>> admm_terms = [
...     Term(None, SquaredL2(1.0, y)),
...     Term(Gradient(2), GroupL2(1.5)),
...     Term(None, Box(0.0)),
... ]
>>> ref = admm_fft(admm_terms, (64, 64), rho=1.0, max_iter=2000, tol=1e-7)
>>> float((ref.x - res.x).abs().max()) < 0.1
True
```

The exact likelihood of a small image, and its Lipschitz constant:

```pycon
>>> fidelity = PoissonGaussian(torch.tensor([3.0, 0.0, 8.0]), 9.0)
>>> round(fidelity.lipschitz, 3)
0.557
>>> z = torch.tensor([2.0, 1.0, 6.0])
>>> fidelity.grad(z).shape, bool(fidelity.value(z) > 0)
(torch.Size([3]), True)
```

## References

!!! references "References"

    1. Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. *Distributed
       optimization and statistical learning via the alternating direction
       method of multipliers.* Found Trends Mach Learn **3**(1):1-122, 2011.
       [doi:10.1561/2200000016](https://doi.org/10.1561/2200000016)
    2. Chouzenoux E, Jezierska A, Pesquet J-C, Talbot H. *A convex approach for image
       restoration with exact Poisson–Gaussian likelihood.* SIAM J Imaging Sci
       **8**(4):2662-2682, 2015.
       [doi:10.1137/15M1014395](https://doi.org/10.1137/15M1014395)
    3. Combettes PL, Pesquet J-C. *Primal-dual splitting algorithm for solving
       inclusions with mixtures of composite, Lipschitzian, and parallel-sum type
       monotone operators.* Set-Valued Var Anal **20**(2):307-330, 2012.
       [doi:10.1007/s11228-011-0191-y](https://doi.org/10.1007/s11228-011-0191-y)
    4. Ikoma H, Broxton M, Kudo T, Wetzstein G. *A convex 3D deconvolution
       algorithm for low photon count fluorescence imaging.* Sci Rep
       **8**:11489, 2018.
       [doi:10.1038/s41598-018-29768-x](https://doi.org/10.1038/s41598-018-29768-x)
    5. Lefkimmiatis S, Unser M. *Poisson image reconstruction with Hessian
       Schatten-norm regularization.* IEEE Trans Image Process
       **22**(11):4314-4327, 2013.
       [doi:10.1109/TIP.2013.2271852](https://doi.org/10.1109/TIP.2013.2271852)

    ??? quote "BibTeX"

        ```bibtex
        @article{boyd2011distributed,
          author  = {Boyd, Stephen and Parikh, Neal and Chu, Eric and Peleato, Borja and Eckstein, Jonathan},
          title   = {Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers},
          journal = {Foundations and Trends in Machine Learning},
          volume  = {3},
          number  = {1},
          pages   = {1--122},
          year    = {2011},
          doi     = {10.1561/2200000016},
        }
        @article{chouzenoux2015convex,
          author  = {Chouzenoux, Emilie and Jezierska, Anna and Pesquet, Jean-Christophe and Talbot, Hugues},
          title   = {A Convex Approach for Image Restoration with Exact {Poisson}--{Gaussian} Likelihood},
          journal = {SIAM Journal on Imaging Sciences},
          volume  = {8},
          number  = {4},
          pages   = {2662--2682},
          year    = {2015},
          doi     = {10.1137/15M1014395},
        }
        @article{combettes2012primal,
          author  = {Combettes, Patrick L. and Pesquet, Jean-Christophe},
          title   = {Primal-Dual Splitting Algorithm for Solving Inclusions with Mixtures of Composite, {Lipschitzian}, and Parallel-Sum Type Monotone Operators},
          journal = {Set-Valued and Variational Analysis},
          volume  = {20},
          number  = {2},
          pages   = {307--330},
          year    = {2012},
          doi     = {10.1007/s11228-011-0191-y},
        }
        @article{ikoma2018convex,
          author  = {Ikoma, Hayato and Broxton, Michael and Kudo, Takamasa and Wetzstein, Gordon},
          title   = {A convex {3D} deconvolution algorithm for low photon count fluorescence imaging},
          journal = {Scientific Reports},
          volume  = {8},
          pages   = {11489},
          year    = {2018},
          doi     = {10.1038/s41598-018-29768-x},
        }
        @article{lefkimmiatis2013poisson,
          author  = {Lefkimmiatis, Stamatios and Unser, Michael},
          title   = {Poisson Image Reconstruction With {Hessian} {Schatten}-Norm Regularization},
          journal = {IEEE Transactions on Image Processing},
          volume  = {22},
          number  = {11},
          pages   = {4314--4327},
          year    = {2013},
          doi     = {10.1109/TIP.2013.2271852},
        }
        ```
