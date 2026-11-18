# Estimators

An *estimator* learns something about an imaging system from data (the flat
field of a microscope, the gain of a camera, the stripes of a light sheet) and
can then use what it learned to correct the data. Every algorithm in optix is
an estimator, and they all behave the same way, so that learning one teaches
you all of them, and so that one can be swapped for another or fed with its
result.

The examples on this page use two small estimators, `MeanDark` and
`MeanFlat`, that stand in for real algorithms. They are written with the
same building blocks as a real one, and the [last section](#writing-an-estimator)
goes through them.

```{.python title="meandark.py"}
--8<-- "snippets/meandark.py"
```

```{.python title="meanflat.py"}
--8<-- "snippets/meanflat.py"
```

To follow along, here is a flat-field stack and a dark stack from the
camera and shading models of [Image formation and noise](image-formation.md):

```python
import numpy as np
import torch

import optix
from optix.estimates import SensorEstimate, ShadingEstimate
from optix.models import (
    random_sensor,
    random_shading,
    simulate_sensor,
    simulate_shading,
)

H, W, N = 64, 80, 100
sensor = random_sensor((H, W), seed=0)
shading = random_shading((H, W), seed=1)
generator = torch.Generator().manual_seed(2)
light = 400 * torch.rand(N, H, W, generator=generator)

flat_frames = simulate_sensor(
    simulate_shading(light, shading), sensor, seed=3, quantize=torch.uint16
).numpy()
dark_frames = simulate_sensor(
    torch.zeros(50, H, W), sensor, seed=4, quantize=torch.uint16
).numpy()
```

## Fit and apply

Using an estimator takes three steps: construct it with its options, `fit` it
to data, and `apply` it.

```pycon
>>> dark = MeanDark().fit(dark_frames)
>>> dark.estimates["shading"].dark.shape
torch.Size([64, 80])
>>> corrected = dark.apply(flat_frames)
>>> corrected.shape, corrected.dtype
(torch.Size([100, 64, 80]), torch.float32)
```

`fit(Y, *, layout=None, init=None, callback=None, verbose=False)` returns the
estimator itself, so calls can be chained. It reads `Y` in batches (see
[Axes and layouts](axes-and-layouts.md#streaming-over-large-arrays)) and stores
what it learns in `estimator.estimates`, a dictionary from the name of a
*target* to an estimate: `"shading"` for a flat field and a dark image,
`"sensor"` for an offset, a gain and a read noise. Calling `fit` again starts
afresh. For an iterative algorithm the progress of the optimisation is kept in
`estimator.history`; the layout that was used is kept in `estimator.layout`.

`apply(Y, out=None)` uses the fitted model, for example to correct `Y`, and
returns `out`. Without an `out` it allocates a floating-point result: on the
device of `Y` if it is a tensor, and on the CPU otherwise. A result the size of
a memmap input must not land on a GPU by surprise. With an `out` it writes
into any array that accepts slices (tensor, `numpy`, `zarr`, `memmap`), batch by batch,
so the corrected stack never has to exist in memory. `out` may even be `Y`
itself, provided the algorithm needs no halo around the pixels.

An `apply` does not have to be a correction. What it computes depends on the
algorithm: a class probability for each pixel, a denoised image, a destriped
one. Its result is always data-sized and goes to `out`, which is why estimates
themselves never hold it.

### Estimates

Results are small dataclasses with tensor fields, one per quantity (`flat`,
`dark`, `baseline` for a `ShadingEstimate`; `offset`, `variance`, `gain`,
`defective` for a `SensorEstimate`). A field is `None` when it was not
estimated. They share a few conveniences:

```pycon
>>> estimate = dark.estimates["shading"]
>>> estimate.flat is None, estimate.dark.shape
(True, torch.Size([64, 80]))
>>> copy = estimate.clone()  # deep copy
>>> copy.allclose(estimate)
True
>>> sorted(estimate.meta)
['estimator', 'faithful', 'options', 'version']
>>> estimate.meta["estimator"], estimate.meta["options"], estimate.meta["faithful"]
('MeanDark', {}, True)
```

`meta` records where the result came from: the estimator, the version of optix,
the resolved options and whether the run was faithful. `replace(**fields)`
copies with some fields replaced, `to(device, dtype)` returns a copy on another
device or in another floating-point type (masks and integer fields keep their
type), and `to_dict()` / `from_dict()` convert to and from a plain dictionary
for storage in the format of your choice. The conventions that fill these
fields are in [Image formation and noise](image-formation.md#result-conventions).

## Faithful mode and extensions

Published algorithms are rarely used as published. A BaSiC or a CIDRE flat
field comes with a preprocessing recipe and a set of default hyper-parameters,
and the community has since added options, faster solvers and better defaults.
Mixing the two up makes results impossible to compare and, worse, impossible to
attribute: if an estimate differs from the paper's, is it because of the
method or because of an option that was quietly changed?

optix keeps them apart with one switch. Every estimator has two sets of default
options:

- **`paper`**, used with `faithful=True` (the default), is the published
  algorithm: its preprocessing and its default hyper-parameters.
- **`recommended`**, used with `faithful=False`, is what the maintainers
  suggest for real data, extensions included.

Every extension is an explicit keyword argument that is off by default, and is
described in the "Faithful mode and extensions" section of the algorithm's
page. An option passed explicitly always wins over the preset.

```pycon
>>> faithful = MeanFlat()
>>> faithful.options, faithful.is_faithful
({'smooth': 0}, True)
>>> recommended = MeanFlat(faithful=False)
>>> recommended.options, recommended.is_faithful
({'smooth': 4}, False)
>>> MeanFlat(smooth=2).is_faithful  # an explicit extension
False
>>> MeanFlat(faithful=False, smooth=0).is_faithful  # back to the paper's value
True
```

`is_faithful` does not record how the object was *built*, but what it *does*: it
is true exactly when every resolved option equals the value of the `paper`
preset. A run that opts out of the recommended defaults one by one, until it
matches the paper again, is faithful.

## Starting from another estimate

Algorithms often depend on each other. A flat-field estimator may need a dark
image. An iterative method converges much faster from the result of a cheap
one. A probabilistic model may start from a calibrated camera. All of these use
the same argument, `init`, which accepts

- an estimate (`ShadingEstimate`, `SensorEstimate`, ...);
- a sequence of estimates;
- or a fitted estimator, which stands for all its `estimates`.

```pycon
>>> flat = MeanFlat().fit(flat_frames, init=dark)
>>> fitted = flat.estimates["shading"]
>>> (fitted.dark - dark.estimates["shading"].dark).abs().max().item()
0.0
```

Here `MeanFlat` found the dark image of `MeanDark` in `init` and subtracted it.
Three rules make this safe:

1. **Nothing is modified.** The estimator works on a clone of what it takes from
   `init`, moved to its own device and type. The estimates you pass in are not
   touched, and are not shared with the result.
2. **Fields can be partial.** An estimate whose field is `None` leaves it to
   the estimator's own data-driven start, never to zero.
3. **Shapes are checked.** The fields must broadcast to the shape of a
   parameter map, `(*group, *spatial)`. A scalar always does, and so does a
   map shared between the groups, but a map for the wrong picture size does
   not, and fails with a clear message rather than deep in an optimisation.

```pycon
>>> bad = ShadingEstimate(dark=torch.zeros(10, 10))
>>> MeanFlat().fit(flat_frames, init=bad)
Traceback (most recent call last):
    ...
ValueError: init ShadingEstimate.dark has shape (10, 10), which does not broadcast to (64, 80)
```

An estimate the estimator has no use for is not an error, but it is not
silent either. The estimator warns, in case you have passed the wrong thing:

```pycon
>>> import warnings
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     _ = MeanFlat().fit(flat_frames, init=sensor)
>>> str(caught[0].message)
'MeanFlat ignores init of type SensorEstimate'
```

Estimates are also the way algorithms are compared: two estimators that produce
a `ShadingEstimate` can be scored against the same ground truth, as the model
simulators of `optix.models` return the truth as an estimate of the same type.

## Devices, dtypes and compilation

Three questions come up when a computation moves from a laptop to a cluster:
where does it run, in which precision, and how much memory may it use. optix
answers them with one small set of options, held by a configuration object.

| Option | Default | Meaning |
|---|---|---|
| `device` | that of a tensor input, else the torch default device | where chunks are processed |
| `dtype` | `torch.get_default_dtype()` (`float32`) | floating-point type of the computation |
| `memory` | 256 MiB | size of **one chunk** on the working device |
| `batch` | none | frames per chunk; overrides `memory` |
| `prefetch` | 0 | chunks read ahead in a background thread |
| `compile` | `True` | whether kernels go through `torch.compile` |

```pycon
>>> optix.get_config()
Config(dtype=None, device=None, memory=268435456, batch=None, compile=True, prefetch=0)
```

Values are validated when set and accept friendly spellings: `"float64"` or
`torch.float64`, `"cuda:1"`, and a size like `"2GiB"` or `"512MB"` (decimal
units are powers of 1000, binary units powers of 1024).

There are three ways to change them. Each is more local than the one before,
and the more local one wins:

```python
optix.set_config(memory="1GiB")  # the process-wide defaults

with optix.config(dtype="float64", device="cuda"):  # a block of code
    est = MeanFlat().fit(flat_frames)

est = MeanFlat(dtype="float64", device="cuda")  # one estimator
```

The last form fixes the working backend of that object, whatever the
configuration says. `optix.config` is also a decorator, which is convenient for
a function that should always run in double precision. A `with` block applies
to the current thread or asyncio task only; threads started inside it see the
process-wide defaults. The initial process-wide defaults come from the
`OPTIX_*` [environment variables](../getting-started/installation.md#defaults-from-the-environment).

```pycon
>>> with optix.config(dtype="float64"):
...     fitted = MeanFlat().fit(flat_frames)
>>> fitted.estimates["shading"].flat.dtype
torch.float64
```

A few details are worth knowing.

- **The working type never follows the data.** Integer data such as `uint16`
  are converted to the working floating-point type, `float32` by default. Data
  that are already `float64` are not processed in double precision unless you
  ask for it. Parts of a computation where precision is essential are the
  exception: running sums, such as those of `Moments`, are accumulated in
  `float64`, and counts are exact integers.
- **The device follows a tensor, but not an array.** A tensor input
  keeps the computation on its device when none is set. A `numpy` or `zarr`
  array has no device, so the torch default applies. Float64 is refused on
  Apple's MPS devices.
- **Configuration is read once.** `fit`, `apply` and the iterators take their
  settings at the start; changing the configuration while an iteration is in
  progress does not affect it. The sizes of chunks are fixed at construction,
  so iterating twice gives identical chunks.
- **`Estimator.to(device, dtype)`** moves a fitted estimator, its estimates
  included, in place.

### Compilation

The inner loops of optix are small tensor kernels, applied to one chunk after
another. `torch.compile` can fuse the elementwise operations of a kernel,
which saves memory traffic and, on a GPU, a good deal of time. optix uses it
when it can, and never depends on it:

- Kernels are compiled lazily, on their first call, and the compiled version is
  reused afterwards. Changing the shape of the input (the last, shorter batch
  of frames, or the clipped blocks at the edge of a picture) triggers a
  recompilation, after which torch treats that dimension as dynamic.
- If compiling fails, or running the compiled kernel does, the call is
  repeated eagerly and a single `RuntimeWarning` explains why. The kernel
  then stays eager for the rest of the process. If the eager run fails too, the error is
  genuine and is raised.
- Where `torch.compile` is not available (Windows, or some combinations of Python and
  torch versions) it is skipped without a warning.

Compilation costs seconds on the first call. For a quick experiment on small
data, or while debugging, switch it off with `optix.config(compile=False)` or
`OPTIX_COMPILE=0`. The flag is read each time a kernel is called.

## History and convergence

Iterative estimators maximise an objective, typically a log-likelihood or a
variational bound. They record it in `estimator.history`, a `History` object, and
stop when it no longer moves. At iteration $t$ the relative change is

$$
\frac{L_t - L_{t-1}}{\lvert L_t \rvert},
$$

and `history.check(tol)` declares convergence when its absolute value falls
below `tol`, recording the reason. Variational and EM objectives increase at
every step, so `check(tol, monotone=True)` additionally warns if the objective
*decreases*, which almost always flags a bug in the update equations.

```pycon
>>> history = optix.History()
>>> for objective in [-100.0, -50.0, -49.9, -49.899]:
...     history.record(objective)
>>> history.check(1e-4), history.reason
(True, 'relative change 2e-05 < tol 0.0001')
```

Extra per-iteration diagnostics go in `history.extras`, with
`history.record(objective, step=...)`. `fit(callback=...)` calls a function with the
estimator after every iteration, and `verbose=True` is up to the estimator to
interpret.

## Saving and loading

Fitted estimators and estimates are saved with `torch.save`, using only
tensors and plain Python containers. No class is pickled, and the files are
loaded with `weights_only=True`, so that opening a file from somebody else does
not run their code and a file stays readable when the code moves on.

```pycon
>>> flat.save("flat.pt")
>>> again = optix.load("flat.pt")
>>> again.options, again.is_faithful
({'smooth': 0}, True)
>>> again.estimates["shading"].allclose(flat.estimates["shading"])
True
```

`optix.load` works out from the file whether it holds an estimator or an
estimate, and rebuilds the right class. For an estimator, the class must be
registered (imported) in the loading process; otherwise a `TypeError` names
the missing class. `estimator.state_dict()` and `load_state_dict()` give the
same content in memory.

## Writing an estimator

The two toy classes above are the template for a real algorithm. This
section is for contributors, and goes through them. The things to implement
are few, and the rest is inherited.

**Tags.** `tags` describes the estimator without running it: what it
estimates (`targets`), how it accesses the data (`streaming`: `"pointwise"`,
`"per-frame"`, `"halo"`, `"per-pixel-stats"` or `"whole-stack"`), which
`modality` and `noise` model it assumes, whether it needs `many_frames`, and a
`reference`. Defining `tags` registers the class, which is how
`optix.estimators()` finds it and how `optix.load` rebuilds it. The tags are
also what the documentation uses to compare algorithms by their assumptions
(see [Algorithms](../algorithms/index.md)).

```pycon
>>> "MeanFlat" in optix.estimators()
True
>>> MeanFlat.tags.streaming, MeanFlat.tags.noise
('per-pixel-stats', 'poisson-gaussian')
```

**Options and presets.** `paper` and `recommended` map each option to its
default in the two modes. The constructor lists its options as keyword-only
parameters that default to `None`, passes the standard ones (`dtype`,
`device`, `layout`) to the base class, and ends with
`self._resolve(faithful, locals())`. That call replaces every `None` by the
preset's value, keeps explicit values, stores the result in `self.options`
and computes `is_faithful`.

**`_fit`.** It receives the data (a labelled array whenever a layout is
known), `init`, a `callback` and `verbose`, and sets `self.estimates`. The
helpers of the base class take care of the plumbing:

- `self._map_shape(Y)` gives the shape `(*group, *spatial)` of the parameter maps;
- `self._backend(Y)` resolves the working dtype and device;
- `self._pick(init, ShadingEstimate, shape=..., like=Y)` extracts, clones and
  checks what the estimator needs from `init`, and `self._warn_unused(init, ...)` warns
  about the rest;
- `reduce_groups` from [`optix.stats`](../reference/stats.md) makes one pass over
  the frames and keeps one summary per group.

**`_apply`.** `self._apply_frames(Y, out, fn)` calls `fn(chunk, batch)` on each
batch of frames and writes the result with `chunk.store`; `chunk.group`
selects the parameters of the group the batch belongs to. For algorithms
that need neighbouring pixels, `self._apply_blocks(Y, out, fn, halo=...)` calls
`fn(block)` on blocks with a halo and writes the cores (see
[Axes and layouts](axes-and-layouts.md#blocks-and-halos)); it refuses to
run in place. Both preserve the layout, allocate `out` when needed, and flush it.

Everything else is done by `fit` and `apply`: they reset the history, reconcile
the layout of a labelled `Y` with the one given, run the private methods
without gradient tracking, and stamp every estimate with its provenance.
