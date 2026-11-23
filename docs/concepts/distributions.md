# Probability distributions

Several algorithms in optix are probabilistic models fitted by variational
Bayes or by expectation-maximisation. Their inner loop is the same few
operations: compute the expected value of some function under a distribution,
accumulate statistics over the data, and update a distribution with them. The
`optix.distributions` module provides these operations once, with the
conventions that make them safe to run on millions of pixels at a time.

It has two layers.

- **`optix.distributions.functional`** holds pure functions on tensors: means,
  expectations, entropies, log-densities and KL divergences, with one
  canonical signature per family. They validate nothing, never touch the
  configuration and never wait for the device, so they can be inlined in
  a compiled kernel.
- **The classes** (`Gamma`, `Dirichlet`, `Categorical`, `Normal`) wrap the
  functions with the conveniences of an object: several parameterisations,
  shapes and broadcasting, `kl_divergence` and `posterior`.

| Family | Variable | Parameterisations | Notes |
|---|---|---|---|
| `Gamma` | $x > 0$, one at a time | shape $a$ and rate $b$; or shape and scale $1/b$; or mean $\mu$ and shape $\nu$ | for noise precisions, intensities, gains |
| `Normal` | $x \in \mathbb{R}$, one at a time | mean $m$ with variance $v$, standard deviation $\sqrt{v}$, or precision $\tau = 1/v$ | diagonal multivariate normals are a batch of these |
| `Dirichlet` | $\pi$ on the simplex of $K$ classes | concentration $\alpha_1, \ldots, \alpha_K$ | priors on class proportions |
| `Categorical` | label $z \in \{0, \ldots, K-1\}$ | probabilities $\pi_k$, or logits | class assignments |

A Gamma density is $p(x) = b^a x^{a-1} e^{-bx} / \Gamma(a)$, a Normal one
$p(x) = \sqrt{\tau / 2\pi}\; e^{-\tau (x - m)^2 / 2}$, and a Dirichlet one
$p(\pi) = \Gamma(\alpha_0) \prod_k \pi_k^{\alpha_k - 1} / \Gamma(\alpha_k)$ with
$\alpha_0 = \sum_k \alpha_k$. Here $\Gamma$ without a subscript is the gamma
function, and $\psi$ will stand for its logarithmic derivative, the digamma
function.

```python
import torch

import optix
from optix.distributions import (
    Categorical,
    CategoricalStats,
    Dirichlet,
    Gamma,
    GammaStats,
    Normal,
    NormalStats,
    kl_divergence,
    posterior,
)
from optix.distributions import functional as F

torch.manual_seed(0)  # the examples below draw random numbers
```

## Parameterisations: stored as given, derived on demand

The same distribution is written in different ways depending on what you know
about it. A Gamma is natural in shape and rate when updating a posterior,
in shape and scale when sampling, and in mean and shape when it describes an
intensity whose average level you care about. A Normal may come with a variance, a
standard deviation or a precision.

Each class accepts all of these. It **stores exactly the parameters it
was given** and computes the others when you ask for them:

```pycon
>>> q = Gamma(shape=4.0, scale=0.5)
>>> q.form  # what is stored
('shape', 'scale')
>>> q.rate, q.mean  # derived on access
(tensor(2.), tensor(2.))
>>> Gamma.from_mean_shape(mean=2.0, shape=4.0).form
('mean', 'shape')
>>> Normal(0.0, std=2.0).precision
tensor(0.2500)
```

Nothing is converted at construction, so no precision is lost by a round trip
through another form, and a parameter that the algorithm updates in place (a
rate that is nudged at every iteration) stays one tensor. The
mean-and-shape form matters numerically too: the expected logarithm of a Gamma,
$\mathbb{E}[\log x] = \psi(a) - \log b$, loses its digits for large $a$ when
computed from shape and rate, while in terms of $\mu$ and $\nu$ it is
$\log\mu - (\log\nu - \psi(\nu))$, which does not (see
[Numerical stability](#numerical-stability)).

`params()` returns the *canonical* parameterisation of a family, the one the
`functional` kernels take: `(shape, rate)` for a Gamma, `(mean, precision)` for
a Normal, the concentration for a Dirichlet, and normalised logits for a
Categorical. `stored()` returns what is stored.

## Shapes: batch, event, and no implicit sums

A distribution object usually stands for many distributions at once, one per
pixel, per class or per pixel and class. Two shapes keep it organised, as in
`torch.distributions`:

- the **event shape** is the shape of *one draw*: `()` for a Gamma, a Normal and
  a Categorical, `(K,)` for a Dirichlet;
- the **batch shape** is everything else: the independent distributions.

```pycon
>>> posterior_per_pixel = Gamma(torch.full((4, 5), 5.0), torch.rand(4, 5) + 1)
>>> posterior_per_pixel.batch_shape, posterior_per_pixel.event_shape
(torch.Size([4, 5]), torch.Size([]))
>>> proportions = Dirichlet(torch.ones(4, 5, 3))  # K = 3 classes
>>> proportions.batch_shape, proportions.event_shape
(torch.Size([4, 5]), torch.Size([3]))
>>> labels = Categorical(logits=torch.randn(4, 5, 3))
>>> labels.batch_shape, labels.event_shape, labels.probs.shape
(torch.Size([4, 5]), torch.Size([]), torch.Size([4, 5, 3]))
```

The `K` classes of a Categorical are a trailing dimension of its *parameters*
but not an event of the distribution: a draw is a single label.

**Parameters broadcast, and are not expanded.** The parameters of one object
may have different batch shapes, which are broadcast against each other.
Each is *stored* with its own shape: a prior with a single scalar shape and a
per-pixel rate holds one number and an image, not two images. The same holds
for the results of arithmetic between objects. Public attributes and method
results are `expand` views of the full shape (so no memory is allocated), which
is why **returned tensors must not be modified in place**: `clone()` first.

**There are no implicit sums.** A quantity that is defined per distribution,
like a KL divergence or an entropy, is returned *per element of the batch
shape*, with only the event dimensions reduced. Summing over pixels, classes
or anything else is up to you:

```pycon
>>> prior = Gamma(2.0, 1.0)  # one prior, for every pixel
>>> kl = kl_divergence(posterior_per_pixel, prior)
>>> kl.shape  # one divergence per pixel
torch.Size([4, 5])
>>> total = kl.sum()  # the KL term of an objective, with the sum explicit
>>> kl_divergence(proportions, Dirichlet(torch.ones(3))).shape  # event reduced
torch.Size([4, 5])
```

This is deliberate. A multivariate Normal with independent components is just a
`Normal` with a trailing batch dimension, and whether you want the KL summed
over it, averaged, or masked first is a modelling decision that should be
visible in the code.

Broadcasting is **right-aligned**, as everywhere in torch. A prior with one
value per class, batch shape `(K,)`, lines up with a posterior of batch shape
`(*pixels, K)`. If you keep the class axis first, as in `(K, H, W)`, the prior
has to be given the trailing axes it lacks, by indexing; indexing a batch
(basic indices only: integers, slices, `None`, `...`) is supported and
returns a distribution:

```pycon
>>> class_prior = Gamma(torch.tensor([1.0, 2.0, 3.0]), torch.ones(3))
>>> per_class = Gamma(torch.full((3, 4, 5), 5.0), torch.ones(3, 4, 5))
>>> kl_divergence(per_class, class_prior)
Traceback (most recent call last):
    ...
ValueError: kl_divergence: batch shapes are not broadcastable: q (3, 4, 5), p (3,)
>>> kl_divergence(per_class, class_prior[:, None, None]).shape
torch.Size([3, 4, 5])
```

`dist.expand(batch_shape)` returns the same distribution with a larger batch
shape, without copying. `log_prob(value)` takes values shaped
`(*sample, *batch, *event)`, and checks that the event dimensions match.

## Expected statistics and exponential families

Gamma, Normal, Dirichlet and Categorical belong to *exponential families*,
whose densities can be written

$$
p(x \mid \eta) = \chi(x)\,\exp\bigl(\eta \cdot T(x) - A(\eta)\bigr),
$$

with *sufficient statistics* $T(x)$, *natural parameters* $\eta$ and a
*log-normaliser* $A(\eta)$ that makes the density integrate to one. In this
form most of what variational Bayes needs is simple.

| Family | $T(x)$ | $\eta$ | $A(\eta)$ |
|---|---|---|---|
| Gamma$(a, b)$ | $(\log x,\; x)$ | $(a - 1,\; -b)$ | $\log\Gamma(a) - a \log b$ |
| $\mathcal{N}(m, \tau^{-1})$ | $(x,\; x^2)$ | $(\tau m,\; -\tau/2)$ | $\tau m^2/2 - \tfrac12\log\tau$ |
| Dirichlet$(\alpha)$ | $\log \pi$ | $\alpha - 1$ | $\sum_k \log\Gamma(\alpha_k) - \log\Gamma(\alpha_0)$ |
| Categorical$(\pi)$ | one-hot$(z)$ | $\log \pi$ | $\log\sum_k e^{\eta_k}$ |

The **expected sufficient statistics** $\mathbb{E}[T(x)]$ are what other parts of
a model need from a distribution: a Gamma over a precision contributes
$\mathbb{E}[\log \tau]$ and $\mathbb{E}[\tau]$ to the likelihood of a Normal
that uses it, a Dirichlet over class proportions contributes
$\mathbb{E}[\log\pi_k]$, and so on. They are the *gradient of the
log-normaliser*, $\mathbb{E}[T(x)] = \nabla_\eta A(\eta)$, which gives a simple
check:

```pycon
>>> q = Gamma(4.0, 2.0)
>>> eta = tuple(t.clone().requires_grad_() for t in q.natural_params())
>>> A = Gamma.from_natural(*eta).log_normalizer()
>>> torch.autograd.grad(A, eta)  # gradient of A ...
(tensor(0.5630), tensor(2.))
>>> q.expected_stats()  # ... equals E[T(x)]
GammaExpectations(log_x=tensor(0.5630), x=tensor(2.))
```

Every class has `natural_params()`, `expected_stats()` (a named tuple, in the
order of the natural parameters), `log_normalizer()`, `entropy()`, `mean`,
`variance`, `log_prob`, `from_natural` and, through `to_torch()`, `sample`.

## Sufficient statistics and conjugate updates

An exponential family has a *conjugate prior*: a prior of a form such that the
posterior, after seeing data, is in the same family, with parameters that are
the prior's plus the data's sufficient statistics. Learning from $N$ observations
means adding up $N$ values of $T$. The data enter through nothing else, which is
why a few numbers per pixel replace the whole stack.

optix keeps those numbers in *sufficient-statistics containers*, one per
likelihood: they hold the total weight $N = \sum_n w_n$ and the weighted sum of $T$.

| Container | Holds | For a likelihood that is |
|---|---|---|
| `GammaStats` | $N$, $\sum_n w_n x_n$, $\sum_n w_n \log x_n$ | Gamma |
| `NormalStats` | $N$, $\bar{x} = \sum_n w_n x_n / N$, $M = \sum_n w_n (x_n - \bar{x})^2$ | Normal |
| `CategoricalStats` | counts $\sum_n w_n\,\text{one-hot}(z_n)$ | Categorical |
| `DirichletStats` | $N$, $\sum_n w_n \log\pi_n$ | Dirichlet |

(`NormalStats` stores the mean and the scatter $M$ rather than $\sum_n w_n x_n$ and
$\sum_n w_n x_n^2$, for the numerical reasons given [below](#numerical-stability).)
They are built from observed samples with `from_samples(x, weight, dim=...)`, or
from the *expected* statistics of latent variables with `from_expectations`.
All weights must be non-negative; a boolean mask is a valid weight.

Containers merge: `a + b` adds two, `a.add_(b)` accumulates in place into a
buffer made with `zeros()`, and `a.sum(dim)` merges along batch dimensions, so
statistics can be built batch by batch, or on several devices, and combined.

`posterior(prior, stats, **known)` applies the update. The parameters of the
likelihood that are *not* being inferred are passed as keywords. The four
updates available are:

| Prior | Statistics | Known | Posterior |
|---|---|---|---|
| Dirichlet$(\alpha^{(0)})$ | `CategoricalStats` | | Dirichlet$\bigl(\alpha^{(0)} + \sum_n w_n\,\text{one-hot}(z_n)\bigr)$ |
| Gamma$(a_0, b_0)$ over the rate of a Gamma | `GammaStats` | `shape` $=\nu$ | Gamma$(a_0 + \nu N,\; b_0 + \sum_n w_n x_n)$ |
| Gamma$(a_0, b_0)$ over the precision of a Normal | `NormalStats` | `mean` $=m$ | Gamma$(a_0 + N/2,\; b_0 + \tfrac12 (M + N(\bar{x} - m)^2))$ |
| $\mathcal{N}(m_0, \tau_0^{-1})$ over the mean of a Normal | `NormalStats` | `precision` $=\tau$ | $\mathcal{N}\bigl(m_N, (\tau_0 + N\tau)^{-1}\bigr)$, $m_N = \dfrac{\tau_0 m_0 + N\tau\bar{x}}{\tau_0 + N\tau}$ |

The third row is worth seeing once. For observations $x_n \sim \mathcal{N}(m, \tau^{-1})$ with
a Gamma$(a_0, b_0)$ prior on $\tau$, the posterior is proportional to

$$
\tau^{a_0 - 1} e^{-b_0 \tau} \cdot \tau^{N/2}
\exp\Bigl(-\frac{\tau}{2} \sum_n (x_n - m)^2\Bigr),
$$

which is a Gamma with shape $a_0 + N/2$ and rate $b_0 + \tfrac12\sum_n (x_n - m)^2$;
and $\sum_n (x_n - m)^2 = M + N(\bar{x} - m)^2$, so the three numbers
$(N, \bar{x}, M)$ are all the data have to give. The result is stored in the
parameterisation in which the update is additive, here shape and rate.

As an example, take the dark frames of a camera and infer, pixel by pixel, the
precision $\tau = 1/\sigma^2$ of the read noise. With each pixel's mean taken as known, the result should be near $1/1.5^2 \approx 0.44$.
Plugging in the sample mean costs one degree of freedom, so the estimate is high
by a factor of about $N/(N-1)$, on top of the prior's pull:

```pycon
>>> from optix.models import random_sensor, simulate_sensor
>>> sensor = random_sensor(
...     (4, 5), offset=100.0, offset_sd=0.0, read_sd=1.5, seed=0
... )
>>> frames = simulate_sensor(torch.zeros(50, 4, 5), sensor, seed=1)
>>> stats = NormalStats.from_samples(frames, dim=0)  # reduce over the frames
>>> stats.n, stats.mean.shape
(tensor(50), torch.Size([4, 5]))
>>> precision = posterior(Gamma(1.0, 1.0), stats, mean=stats.mean)
>>> precision
Gamma(shape, rate; batch_shape=(4, 5), event_shape=(), dtype=float32, device=cpu)
>>> round(precision.mean.mean().item(), 2)
0.46
```

A scalar prior and per-pixel statistics give a per-pixel posterior, as the
shape rules above promise. Weak priors move the estimate towards the prior's
mean by an amount that fades as $N$ grows.

For a stack that does not fit in memory, build the statistics in one pass over
frame batches. The reducers of [`optix.stats`](../reference/stats.md) already
keep exactly this state, and `NormalStats.from_moments` converts without a
copy:

```pycon
>>> from optix.stats import Moments, reduce
>>> (moments,) = reduce(frames, Moments(2), batch=10)  # five batches of ten
>>> streamed = NormalStats.from_moments(moments)
>>> again = posterior(Gamma(1.0, 1.0), streamed, mean=streamed.mean)
>>> torch.allclose(again.rate, precision.rate)
True
```

### Latent variables

In variational Bayes, the unknowns also include latent variables, such as the
class of each pixel. Their statistics are not observed but *expected* under
the current approximation $q$. This is what `from_expectations` is for, and it
takes the output of the matching `expected_stats()`.

Consider a mixture with $K$ classes, class proportions $\pi$ with a Dirichlet
prior, and $N$ pixels whose class $z_n$ is unknown. The best update of the
distribution of $z_n$ weighs the log-proportion of each class against how well
the class explains the pixel:

$$
r_{nk} \propto \exp\bigl(\mathbb{E}[\log\pi_k] + \mathbb{E}[\log p(x_n \mid z_n = k)]\bigr),
$$

which is *by definition* a Categorical with logits
$\mathbb{E}[\log\pi_k] + \mathbb{E}[\log p(x_n \mid z_n = k)]$. The expected
counts $\sum_n r_{nk}$ then update the Dirichlet:

```pycon
>>> prior = Dirichlet(torch.ones(3))
>>> expected_log_lik = torch.randn(1000, 3)  # E[log p(x_n | k)], from the model
>>> responsibilities = Categorical(logits=prior.mean_log + expected_log_lik)
>>> responsibilities.batch_shape, responsibilities.probs.shape
(torch.Size([1000]), torch.Size([1000, 3]))
>>> counts = CategoricalStats.from_expectations(responsibilities.probs, dim=0)
>>> round(counts.n.item())  # every pixel is spread over the classes: total N
1000
>>> updated = posterior(prior, counts)
>>> round(updated.concentration.sum().item())  # 3 (prior) + 1000 (data)
1003
```

Two further tools complete the picture. `stats.scale(c)` multiplies the total
weight by a factor: `posterior(prior, stats.scale(c))` is a *tempered* update, and
$c = 1/f$ rescales the statistics of a random fraction $f$ of the data to the
size of the whole, as stochastic variational inference does. And when a latent-variable
model needs a distribution *fitted* rather than updated,
`Gamma.fit(stats)` and `Dirichlet.fit(stats, init)` return maximum-likelihood
estimates. The Gamma shape $\nu$ solves

$$
\log\nu - \psi(\nu) = \log\bar{x} - \overline{\log x},
$$

where $\bar{x}$ and $\overline{\log x}$ are the means of $x$ and $\log x$ in the
statistics; the equation is solved by a fixed number of Newton steps on $1/\nu$, which keeps
it batched and free of host synchronisation. With `mean=` given (a fixed $\mu$, so only the
shape is fitted), the equation becomes

$$
\log\nu - \psi(\nu) = \log\bar{x} - \overline{\log x} + (q - 1 - \log q),
\qquad q = \bar{x} / \mu.
$$

```pycon
>>> x = Gamma(5.0, 0.5).sample((20000,))
>>> fitted = Gamma.fit(GammaStats.from_samples(x, dim=0, dtype=torch.float64))
>>> fitted.form
('mean', 'shape')
>>> round(fitted.mean.item()), round(fitted.shape.item())  # true: 10 and 5
(10, 5)
```

## Categorical: labels, one-hot and soft assignments

A Categorical describes one discrete choice among $K$ classes. Yet the thing
you evaluate it at comes in different forms: a *label* (an integer), a *one-hot*
vector, or a *soft assignment* (a vector of non-negative weights that sum to 1,
like the responsibilities above). For a label, the log-probability is
$\log \pi_z$. For an assignment $r$ it is the expected log-likelihood
$\sum_k r_k \log \pi_k$, which is what a variational update needs, and for a
one-hot vector the two agree.

`log_prob` tells them apart **by dtype**:

| `value` | Read as | Shape |
|---|---|---|
| integer | labels in $[0, K)$ | `(*sample, *batch)` |
| floating point or boolean | one-hot or soft assignments | `(*sample, *batch, K)` |

```pycon
>>> cat = Categorical(probs=torch.tensor([0.5, 0.3, 0.2]))
>>> cat.log_prob(torch.tensor(1))  # a label
tensor(-1.2040)
>>> cat.log_prob(torch.tensor([0.0, 1.0, 0.0]))  # the same, one-hot
tensor(-1.2040)
>>> cat.log_prob(torch.tensor([0.2, 0.5, 0.3]))  # soft: sum_k r_k log(pi_k)
tensor(-1.2234)
```

The consequence to remember is that **integers always mean labels**, including
the `int64` output of `torch.nn.functional.one_hot`. Passing it directly does not
fail; it is read as a batch of three labels. Cast it first:

```pycon
>>> one_hot = torch.nn.functional.one_hot(torch.tensor(1), 3)  # int64!
>>> cat.log_prob(one_hot)  # three labels: 0, 1, 0
tensor([-0.6931, -1.2040, -0.6931])
>>> cat.log_prob(one_hot.float())  # a one-hot vector
tensor(-1.2040)
```

Labels are not range-checked by default, because the check would stall the
device. On the CPU an out-of-range label raises an error anyway, while on CUDA
it triggers a device-side assertion that leaves the CUDA context unusable. If
the labels come from a source you do not trust, build the distribution with
`validate=True`:

```pycon
>>> strict = Categorical(probs=torch.tensor([0.5, 0.3, 0.2]), validate=True)
>>> strict.log_prob(torch.tensor(5))
Traceback (most recent call last):
    ...
ValueError: Categorical.log_prob: labels must be in [0, 3); integer values are labels (cast one-hot vectors to float or bool)
```

The other operations do not depend on the representation: the mean is
the probabilities $\pi$, and the entropy and KL divergence are those of the
distribution. `probs` and `logits` are stored *as given*, unnormalised, and the
normalised versions are derived, so that
`Categorical(logits=expected_log_pi + expected_log_lik)` is a valid update with
no normalisation step. `sample()` returns `int64` labels, or `float`
one-hot vectors with `one_hot=True`. There is no reparameterised sampling.

## Caching

Deriving a parameter, such as the rate of a Gamma given by its scale, costs a
tensor operation. In a loop that reads it many times, it is worth doing once.
Caching is **off by default** and switched on per object, with `cache=True` at
construction or `dist.cache = True` later.

```pycon
>>> scale = torch.tensor([0.5, 1.0])
>>> q = Gamma(shape=torch.tensor([4.0, 4.0]), scale=scale, cache=True)
>>> q.rate
tensor([2., 1.])
>>> scale.mul_(2)  # an in-place change of a stored tensor ...
tensor([1., 2.])
>>> q.rate  # ... invalidates what was derived from it
tensor([1.0000, 0.5000])
```

A cached value is *never* stale, because it is keyed on the version counters
that torch keeps for each tensor and increments at every in-place operation.
When any stored parameter changes version, the cache is dropped. The
cache never changes a result, only whether it is recomputed. It is bypassed
altogether where reusing a value would be wrong or impossible:

- while compiling, because traced functions have no side effects and never see it;
- when a stored parameter requires gradients, since a cached graph would break a
  second backward pass;
- for inference tensors, which have no version counter.

A derived parameter is cached with the broadcast shape of the stored tensors it
is computed from: the rate of a Gamma given by a scalar shape and a per-pixel mean is
per-pixel.

Distribution objects are *immutable by convention*: update a distribution by
building a new one (which is what `posterior` and `fit` do), not by assigning
to its fields.

## Numerical stability

Variational algorithms combine $\log\Gamma$, $\psi$ and logarithms of very
large and very small numbers, in single precision, on millions of pixels. The
textbook formulas fail there in a specific, avoidable way: they subtract two
huge numbers that are nearly equal, and what is left is rounding noise. The
KL divergence between two Gamma distributions of shape one million is an
example. Here $q$ and $p$ are almost the same distribution, and the true
divergence is $0.18$:

```pycon
>>> q, p = Gamma(1_000_500.0, 1_000_000.0), Gamma(1_000_000.0, 1_000_100.0)
>>> exact = kl_divergence(q.to(dtype=torch.float64), p.to(dtype=torch.float64))
>>> round(exact.item(), 4)
0.18
>>> round(kl_divergence(q, p).item(), 4)  # float32
0.18
>>> (a_q, b_q), (a_p, b_p) = q.params(), p.params()
>>> textbook = (
...     (a_q - a_p) * torch.digamma(a_q)
...     - torch.lgamma(a_q)
...     + torch.lgamma(a_p)
...     + a_p * torch.log(b_q / b_p)
...     + a_q * (b_p - b_q) / b_q
... )
>>> round(textbook.item(), 4)  # the same formula in float32
1.0284
>>> F.gamma_kl(a_q, b_q, a_p, b_p).item() == kl_divergence(q, p).item()
True
```

(`kl_divergence` is a thin layer over the pure function `F.gamma_kl`, which takes
the canonical parameters of both distributions.)

The remedy is algebraic rather than numerical. `optix.special` provides the
building blocks. For large $a$, Stirling's formula gives

$$
\log\Gamma(a) = \bigl(a - \tfrac12\bigr)\log a - a + \tfrac12\log 2\pi + \rho(a),
\qquad
\rho(a) \approx \frac{1}{12a} - \frac{1}{360a^3} + \cdots,
$$

and the digamma function is $\psi(a) = \log a - \delta(a)$ with

$$
\delta(a) \approx \frac{1}{2a} + \frac{1}{12a^2} - \frac{1}{120a^4} + \frac{1}{252a^6}.
$$

Substituting these in the KL divergence cancels the large terms *on paper*,
and what remains is a sum of small quantities that are computed accurately:
the residuals $\rho(a)$ (`stirling_residual`), $\delta(a)$ (`log_minus_digamma`),
and $\varphi(u) = u - 1 - \log u$ for a ratio $u$ near 1 (`ratio_minus_log`), which behaves like $\epsilon^2/2$ for
$u = 1 + \epsilon$ and is evaluated from a series so that it does not collapse to zero. The
same trick is used for the entropy, the expected logarithms and the Dirichlet
divergence, and is why `Gamma.mean_log` has a dedicated path in the mean-and-shape
form. Even $\delta(a)$ alone shows the effect:

```pycon
>>> from optix.special import log_minus_digamma
>>> a = torch.tensor(1e5)
>>> f"{log_minus_digamma(a).item():.6e}"  # series
'5.000008e-06'
>>> f"{(a.log() - torch.digamma(a)).item():.6e}"  # difference of two numbers near 11.5
'4.768372e-06'
```

The correct value is $5.0000083 \times 10^{-6}$. The direct difference is wrong
by 5 %, because it is a small remainder of two numbers near 11.5, which a
`float32` knows to about $10^{-6}$.

Other habits follow from the same concern.

- **Accumulate in high precision.** Sums over many pixels or frames drift in
  `float32`. Reducers accumulate in `float64`, counts are kept as `int64` or `float64`
  (a `float32` count stops growing at $2^{24} \approx 1.7 \times 10^7$), and
  `SuffStats.zeros(..., dtype=torch.float64)` makes a double-precision
  accumulator for batches of single-precision statistics.
- **Never form raw power sums.** `NormalStats` and `Moments` hold the mean and the
  scatter $M = \sum (x - \bar{x})^2$ and merge them with
  [Pébay's pairwise formulas](https://doi.org/10.1007/s00180-015-0637-z), instead of holding $\sum x$ and $\sum x^2$. For a camera with an
  offset of 100 ADU and a noise of 1.5 ADU, $\sum x^2$ is thousands of times larger
  than the scatter it is meant to produce, and `float32` keeps about seven digits.
- **No silent failure at the edges.** $0 \log 0 = 0$ in entropies and in the
  expected log-likelihood of a one-hot vector, with finite gradients; statistics
  and fits with no data ($N = 0$) are legal and return what you gave as `init`
  (or `nan`), and a flat Normal (`precision=0`) has infinite entropy and KL but
  valid conjugate updates.
- **KL divergences are not clamped at zero.** In `float32`, nearly equal
  distributions may give a very slightly negative value, which is returned as
  computed.
- **Values are not checked, unless you ask.** Invalid parameters give `nan` or
  infinities, as in any tensor expression, because checking would force the device to
  synchronise with the host. Pass `validate=True` (or call `.validate()`) while
  developing; it raises a `ValueError` that names the parameter:

```pycon
>>> Gamma(-1.0, 1.0).mean  # no error: garbage in, garbage out
tensor(-1.)
>>> Gamma(-1.0, 1.0, validate=True)
Traceback (most recent call last):
    ...
ValueError: Gamma: shape must be positive and finite
```

The iterative solvers of `optix.special` (`fit_gamma_shape`, `inv_digamma`,
`fit_dirichlet`) run a fixed number of iterations with no data-dependent
control flow. They are batched and compile-friendly, and never synchronise with
the host. The price is that convergence is local: `fit_dirichlet` needs a warm
start such as the previous estimate, which is why `Dirichlet.fit` requires
`init`. And `gauss_hermite` supplies quadrature nodes and log-weights for
Gaussian expectations of functions that have no closed form,

$$
\mathbb{E}_{\mathcal{N}(m, s^2)}[f] \approx
\frac{1}{\sqrt{\pi}} \sum_i w_i\, f\bigl(m + \sqrt{2}\, s\, t_i\bigr),
$$

with $s = \sqrt{v}$, nodes $t_i$ and weights $w_i$ for the weight function $e^{-t^2}$.

!!! references "References"

    1. Pébay P, Terriberry TB, Kolla H, Bennett J. *Numerically stable, scalable
       formulas for parallel and online computation of higher-order multivariate
       central moments with arbitrary weights.* Comput Stat **31**(4):1305-1325,
       2016. [doi:10.1007/s00180-015-0637-z](https://doi.org/10.1007/s00180-015-0637-z)

    ??? quote "BibTeX"

        ```bibtex
        @article{pebay2016numerically,
          author  = {P{\'e}bay, Philippe and Terriberry, Timothy B. and Kolla, Hemanth and Bennett, Janine},
          title   = {Numerically stable, scalable formulas for parallel and online computation of higher-order multivariate central moments with arbitrary weights},
          journal = {Computational Statistics},
          volume  = {31},
          number  = {4},
          pages   = {1305--1325},
          year    = {2016},
          doi     = {10.1007/s00180-015-0637-z},
        }
        ```
