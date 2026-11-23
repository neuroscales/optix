# Axes and layouts

A microscopy array is a pile of pictures, and almost everything optix does
depends on knowing how the pile is organised. Which axes make up one picture?
Which pictures were taken under the same conditions, so that they can share
what is learned from them? Which belong to a different experiment altogether?
A shape such as `(5, 3, 7, 64, 64)` does not say.

optix therefore asks you to name the *role* of each axis. There are only three,
and every axis has exactly one.

| Role | Meaning | Typical axes |
|---|---|---|
| **spatial** | the axes that span one picture (a frame) | `y`, `x`; for a 3D algorithm `z`, `y`, `x` |
| **sample** | different pictures that share the same parameters | frames, time points, `z` planes, tiles |
| **group** | independent problems, each with its own parameters | channels, cameras, illumination arms |

The distinction between *sample* and *group* is the one that matters. Samples
are pooled: the flat field of a channel is estimated from all of its frames at
once. Groups are kept apart: each channel gets its own flat field. Fitted
parameter maps are shaped

$$
(\underbrace{G_1, \ldots}_{\text{group axes}},\;
\underbrace{X_1, \ldots}_{\text{spatial axes}}),
$$

one map of the spatial shape for each combination of group indices. The
sample axes disappear: they are what was averaged over.

## The default layout

When you tell optix nothing, it assumes `(*samples, H, W)`: any number of
leading sample axes and two trailing spatial axes. A stack of frames
`(N, H, W)` and a time series of stacks `(T, Z, H, W)` are both handled
without any ceremony, and every function that takes data accepts an `ndim`
argument if the pictures are 3D rather than 2D.

```pycon
>>> from optix.io import Layout
>>> Layout.default(4, 2)
Layout(axes=None, spatial=(-2, -1), sample=None, group=())
```

The default layout uses negative positions, so it fits arrays of any number of
dimensions.

## Describing a layout

A `Layout` lists the labels of the axes in storage order, then says which
labels play which role. Axes that are not mentioned are samples. Here is an
OME-NGFF stack, stored as `(t, c, z, y, x)` and with one flat field per
channel, shared across time points and planes:

```pycon
>>> layout = Layout("tczyx", spatial="yx", group="c")
>>> layout.shape_of((5, 3, 7, 64, 64))
((3,), (5, 7), (64, 64))
>>> layout.map_shape((5, 3, 7, 64, 64))
(3, 64, 64)
```

`shape_of` returns the shapes of the group, sample and spatial axes. There
are 3 groups (the channels), $5 \times 7 = 35$ samples per group (the time
points and the planes, `t` and `z`), and every picture is $64 \times 64$.
`map_shape` then gives the shape of a fitted map: one $64 \times 64$ flat
field for each of the 3 channels.

Spatial axes are listed in the order *you* want pictures and maps to have, which
need not be the storage order. That is not a curiosity: it is what makes the
next example possible.

### A light-sheet stack

A light-sheet microscope lights up one thin plane of the specimen, and the
camera records that plane. Everything that spoils the picture (uneven
illumination, shadows, stripes, noise) lives in the plane the camera sees.
Suppose the specimen is translated along `x` through the sheet, and the
volume is saved as `(z, y, x)`. The pictures are then the `(y, z)` planes, not
the `(z, y)` slices you would meet first when indexing the array, and each
value of `x` is a different picture:

```pycon
>>> sheet = Layout("zyx", spatial="yz", sample="x")
>>> sheet.shape_of((40, 30, 500))
((), (500,), (30, 40))
```

There are no groups, 500 samples (the positions along `x`), and each picture
is $30 \times 40$: `y` first, then `z`. Maps will have that shape too.

The reordering is applied for you when data are loaded. Take a tiny array to
see it: the picture at `x = 1` arrives as the transposed `(y, z)` plane,
whatever the storage order.

```pycon
>>> import numpy as np
>>> from optix.io import frame_batches
>>> Y = np.random.default_rng(0).normal(size=(4, 5, 6)).astype("float32")
>>> chunk = next(iter(frame_batches(Y, layout=sheet)))
>>> chunk.data.shape  # (samples, y, z)
torch.Size([6, 5, 4])
>>> bool((chunk.data[1] == Y[:, :, 1].T).all())
True
```

Anything an estimator computes afterwards, such as a destriping filter along
`z` or a flat field in the `(y, z)` plane, sees pictures the way the camera
did.

### Time, channels and tiles

The same three roles cover most acquisitions. A few worked patterns:

| Data | Layout | Parameter maps |
|---|---|---|
| one stack of frames `(n, y, x)` | default | `(y, x)` |
| `(t, c, z, y, x)`, a flat per channel | `Layout("tczyx", spatial="yx", group="c")` | `(c, y, x)` |
| `(t, c, z, y, x)`, a flat for the whole experiment | `Layout("tczyx", spatial="yx")` | `(y, x)` |
| `(tile, z, y, x)`, 3D pictures, one per tile | `Layout("tzyx", spatial="zyx", sample="t")` | `(z, y, x)` |
| `(camera, n, y, x)`, two cameras | `Layout("cnyx", spatial="yx", group="c")` | `(c, y, x)` |

An axis can also be given by position instead of by name, which is how the
default layout works (`spatial=(-2, -1)`), and `Layout` checks the roles
against the data as soon as it can: an unknown label, an axis with two
roles or a position out of range raises an error immediately rather than
deep inside a fit.

## Attaching a layout to data

Pass a layout to `fit` with `layout=`, or attach it to the array once with
`labeled`. The array is *not* read or copied; the result is a thin wrapper
that behaves like the original (`shape`, `dtype`, `chunks`, indexing) and
carries its layout around:

```pycon
>>> from optix.io import labeled
>>> Z = labeled(
...     np.zeros((5, 3, 7, 16, 16), "float32"),
...     axes="tczyx",
...     spatial="yx",
...     group="c",
... )
>>> Z
Labeled(ndarray, shape=(5, 3, 7, 16, 16), Layout(axes=('t', 'c', 'z', 'y', 'x'), spatial=('y', 'x'), sample=None, group=('c',)))
>>> Z.shape, Z.layout.map_shape(Z)
((5, 3, 7, 16, 16), (3, 16, 16))
```

`labeled(Y, layout)` accepts a ready-made layout instead of the keywords. Every
function that takes data (iterators, reducers, estimators) accepts a labelled
array, and refuses to guess when you pass a labelled array *and* a different
layout.

## Walking through the data

Three iterators cut an array into chunks, each for a different kind of
computation. They all return a re-iterable object whose length is exact and
whose chunks are loaded lazily, one at a time.

| Iterator | A chunk holds | Use it for |
|---|---|---|
| `frame_batches` | `n` whole pictures of one group, `(n, *spatial)` | anything that works frame by frame, or accumulates per-pixel sums over frames |
| `row_tiles` | all the samples of one group, but a strip of the picture, `(*sample, rows, ...)` | per-pixel statistics that need every sample at once, such as medians |
| `blocks` | a block of a picture plus a halo, `(*spatial)` | filters and deconvolution, which need the neighbours of a pixel |

Each chunk is a tuple, `(index, data)` (or `(index, data, inner)` for blocks),
with the loaded tensor in `data`. It also knows where it came from:

```pycon
>>> layout = Layout("tczyx", spatial="yx", group="c")
>>> Y = np.zeros((2, 3, 4, 5, 6), "float32")  # t, c, z, y, x
>>> chunks = frame_batches(Y, layout=layout, batch=4)
>>> len(chunks)
6
>>> chunk = next(iter(chunks))
>>> chunk.group, chunk.samples, chunk.data.shape
((0,), (slice(0, 1, None), slice(0, 4, None)), torch.Size([4, 5, 6]))
```

This chunk belongs to channel 0. It holds the 4 planes `z = 0..3` of the first
time point, flattened into one leading axis and ready for a tensor operation.
Groups are visited one at a time, so a chunk never mixes channels.

`frame_batches` fills a batch from the inner sample axes outwards: whole
`z`-stacks if they fit, otherwise a part of one. The size of a batch is
decided by a memory budget, covered [below](#streaming-over-large-arrays).

### Writing results back: `chunk.store`

A result computed on a chunk belongs at a particular place in a particular
output, and that place can be far from obvious once axes have been flattened
and permuted. `chunk.store(out, result)` does the bookkeeping: it undoes the
flattening and the reordering and writes into `out` in storage order. Here is
a toy computation, doubling the `(y, z)` pictures of the light-sheet stack:

```pycon
>>> out = np.zeros_like(Y := np.random.default_rng(0).normal(size=(4, 5, 6)))
>>> for chunk in frame_batches(Y, layout=sheet, batch=3):
...     chunk.store(out, 2 * chunk.data)
>>> bool(np.allclose(out, 2 * Y))
True
```

`out` has the shape and axis order of `Y`, even though the chunks were
`(y, z)` pictures. The tempting alternative, `out[chunk.index] = ...`, is only
valid with the default layout.

`store` has three modes, selected by `into=`, for the three shapes a result
can take:

| `into=` | `out` is shaped like | `result` is |
|---|---|---|
| `"source"` (default) | the data, in storage order | `(n, *spatial)`, one picture per sample |
| `"map"` | a parameter map, `(*group, *spatial)` | `(*spatial)`, already reduced over the samples |
| `"samples"` | a per-sample state, `(*group, *sample, ...)` | `(n, ...)`, one entry per sample |

The `"map"` mode is for results of `row_tiles`, which hold every sample of a
group: the mean over samples of each strip is a strip of the map. The
`"samples"` mode stores something that exists once per picture, such as a
per-frame baseline or a per-frame quality score:

```pycon
>>> Y = np.random.default_rng(1).normal(size=(2, 3, 4, 5, 6)).astype("float32")
>>> layout = Layout("tczyx", spatial="yx", group="c")
>>> from optix.io import row_tiles
>>> mean = np.zeros(layout.map_shape(Y), "float32")  # (c, y, x)
>>> for chunk in row_tiles(Y, layout=layout, batch=1):
...     chunk.store(mean, chunk.data.mean((0, 1)), into="map")
>>> bool(np.allclose(mean, Y.mean((0, 2)), atol=1e-6))
True
>>> frame_mean = np.zeros((3, 2, 4), "float32")  # (c, t, z)
>>> for chunk in frame_batches(Y, layout=layout):
...     chunk.store(frame_mean, chunk.data.mean((-2, -1)), into="samples")
>>> bool(np.allclose(frame_mean, Y.mean((3, 4)).transpose(1, 0, 2), atol=1e-6))
True
```

Results are written with the care a raw cast would not take. Floating
values written into an integer array are rounded and clamped to its range
(a plain cast would wrap around, turning a saturated pixel into a dark one),
`NaN` becomes 0, and floats written into a boolean array become `value != 0`.
The rounding happens on the device where the result was computed, before it is
copied to the host. `optix.io.flush` calls `.flush()` on outputs that have one, which is
what a `numpy.memmap` needs at the end.

### Blocks and halos

Filters and deconvolution need the neighbours of a pixel. `blocks` loads a
block of the picture extended by a *halo*, and gives back the part you
should write: `data[inner]` is the core, `index` says where the core lives.
With `pad=None` the halo is clipped at the array boundary, so edge blocks are
smaller and the algorithm applies its own boundary condition, as it would
on the whole picture. With `pad="reflect"`, `"replicate"` or `"zeros"`, every
block has exactly the shape `block + 2 * halo`, which suits kernels compiled for
a fixed size. Because a block's halo overlaps its neighbours, results must not
be written back into the input, and the estimator helpers refuse to try.

## Streaming over large arrays

optix never asks for a whole array. Everything it knows about your data is a
`.shape` and the ability to slice it, with basic indexing only: a full-length
tuple of integers and step-1 slices. So the same code runs on a `torch.Tensor`,
a `numpy` array or memmap, a `zarr` array, an `h5py` dataset or a `dask` array, and optix imports none of
them. Each slice is converted by `numpy.asarray`, which is the moment a
chunked array does its decompression.

For a stack that lives on disk, open it lazily and use it as you would any
array:

```python
import tempfile
from pathlib import Path

import numpy as np
import zarr

import optix
from optix.io import chunks_of, frame_batches

path = Path(tempfile.mkdtemp()) / "stack.zarr"
Y = zarr.open_array(
    path,
    mode="w",
    shape=(120, 256, 256),
    chunks=(10, 256, 256),
    dtype="uint16",
)
rng = np.random.default_rng(0)
for start in range(0, 120, 10):
    Y[start : start + 10] = rng.integers(0, 4000, (10, 256, 256))
```

This array is small, so that the example runs quickly, but nothing below
depends on its size. A real light-sheet volume is read the same way, with
`zarr.open_array("volume.zarr", mode="r")` or, for one level of an OME-Zarr
pyramid, `zarr.open_group("volume.ome.zarr")["0"]`.

### The memory budget

How many frames are loaded at a time? The answer is a budget, not a number:
`memory` is the size in bytes of **one chunk on the working device**. It is not
the total memory of the process, which is larger by the temporaries that
the computation itself holds (iterators take an `overhead` argument to
account for them). The default is 256 MiB, and it can be given in plain
bytes or as `"512MiB"`, `"2GB"` and the like.

```pycon
>>> chunks = frame_batches(Y, memory="4MiB")
>>> len(chunks)
12
>>> index, batch = next(iter(chunks))
>>> batch.shape, batch.dtype
(torch.Size([10, 256, 256]), torch.float32)
```

A 4 MiB budget holds 16 single-precision frames of this size. The batch is
instead 10 frames, the storage chunk length: batches are rounded down to a
multiple of the chunk of the array along the axis that is split, so that no
chunk is decompressed twice. If the output is itself chunked, pass
`align=chunks_of(out)` to line up with it too, which estimators do for you. You can
set the budget for one call (`memory=`), for a block of code, or for the process:

```pycon
>>> with optix.config(memory="4MiB"):
...     len(frame_batches(Y))
12
```

`batch=` fixes the number of frames instead, and wins over `memory`.
`prefetch=` reads chunks ahead in a background thread, to overlap disk and
computation. Both are described with the other options in
[Estimators](estimators.md#devices-dtypes-and-compilation).

Data cross to the working device in their stored type and are converted to the
working floating-point type there, so `uint16` frames travel the link at half
the size of `float32`. Every chunk is owned by the consumer: you may modify it
in place without touching the source. A chunk is a permuted, strided view of the
loaded block, in layout order, rather than a dense copy; pass `contiguous=True` to
the iterator if a kernel needs dense memory (this copies whenever the layout
permutes the stored axes).

### Reading in the right order

The cheapest read is one that matches how the data are stored. Two things go
wrong when they do not, and optix warns about both.

- A chunked store that is cut into pieces which do not line up with its
  chunks decompresses some chunks several times. The warning names the factor
  (it appears above a factor of 4), and the fix is a larger budget or a store
  rechunked along the axis you iterate.
- `row_tiles` needs every sample of a pixel at once, so on a store chunked along
  the samples, every tile touches every chunk. That is the price of exact
  medians and quantiles; if an approximation is enough, a
  [`Histogram`](../reference/stats.md) reads frame batches only.

### Streaming statistics

Most estimators begin by summarising the data per pixel. The reducers in
[`optix.stats`](../reference/stats.md) do that in one pass over frame
batches. Each is *mergeable*: summarising two halves of the data separately
and merging the summaries gives the same result as one pass over the whole,
up to round-off. Computing the per-pixel mean and variance of the stack
above takes a single call:

```pycon
>>> from optix.stats import Moments, reduce
>>> with optix.config(memory="4MiB"):
...     (m,) = reduce(Y, Moments(2))
>>> m.mean.shape, m.var().shape
(torch.Size([256, 256]), torch.Size([256, 256]))
>>> m.count
tensor(120., dtype=torch.float64)
```

Merging is what lets you spread the work. If two partial summaries have
total weights $N_a$ and $N_b$ (the number of frames, when unweighted), means
$\bar{x}_a$ and $\bar{x}_b$ and sums of squared deviations $M_a$ and $M_b$,
the summary of their union is

$$
\begin{aligned}
N &= N_a + N_b, \\
\theta &= \bar{x}_b - \bar{x}_a, \\
\bar{x} &= \bar{x}_a + \theta\,\frac{N_b}{N}, \\
M &= M_a + M_b + \theta^2\,\frac{N_a N_b}{N},
\end{aligned}
$$

and the variance is $M / (N - 1)$ (`var(ddof=1)`; for non-integer weights such as
responsibilities use $M / N$, `var(ddof=0)`). These are [Pébay's pairwise formulas](https://doi.org/10.1007/s00180-015-0637-z).
`Moments` never accumulates raw power sums, which lose all their digits when
the mean is large compared with the spread, and that is exactly the situation
of a camera whose offset is 100 ADU and whose noise is a few. It also keeps
its running state in `float64`, whatever the type of the data.

```pycon
>>> import torch
>>> first = Moments(2).update(torch.as_tensor(Y[:60]).float())
>>> second = Moments(2).update(torch.as_tensor(Y[60:]).float())
>>> bool(torch.allclose(first.merge(second).var(), m.var(), rtol=1e-5))
True
```

Besides `Moments`, the module has `MinMax`, a per-pixel `Histogram` from which
quantiles are read with an error of at most one bin width, a uniform
`Reservoir` sample, and the exact per-pixel `order_statistics` and
`median_mad`. For a layout with groups, use `reduce_groups`, which keeps one
copy of the reducers per group.

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
