# optix

Processing algorithms for microscopy, in [PyTorch](https://pytorch.org).

Microscopy images are measurements, and most of what stands between a raw image
and a quantity you can trust is a correction: for the uneven illumination of
the field of view, for the offset and the noise of the camera, for stripes
and shadows, for blur. optix collects those corrections as
*estimators* that share one interface and are written to run on volumes
far larger than memory. Light-sheet microscopy comes first.

```python
import optix

est = Algorithm().fit(Y, layout=layout)  # learn from a stack, batch by batch
corrected = est.apply(Y, out=out)  # correct it, into any array
```

(`Algorithm` stands for any estimator. The [quick tour](getting-started/quick-tour.md)
runs the real thing.)

## Three ideas

**Estimators.** Every algorithm has the same shape: construct it with its
options, `fit` it to data, `apply` it. What it learns is stored in small,
serialisable *estimates* (a flat field, a camera model) with shared conventions
that let one algorithm start from another's result, or be compared with it. See
[Estimators](concepts/estimators.md).

**Faithful by default.** An estimator with default options reproduces the
published method, preprocessing and hyper-parameters included. Every extension
is an explicit option, switched off unless you ask, and described on its own
section of the algorithm's page. When you publish a result, you know which
algorithm you ran.

**Streaming.** Data are anything with a shape that can be sliced: a tensor, a
memmap, a `zarr` array. They are read a few frames at a time under a memory
budget you set, and results are written the same way. You tell optix which axes
are the picture, which are repeated measurements and which are independent
experiments, once, and everything follows. See
[Axes and layouts](concepts/axes-and-layouts.md).

## Where to start

- New here? [Install](getting-started/installation.md) the package and take the
  [quick tour](getting-started/quick-tour.md).
- Want the ideas behind the API? Read the [Concepts](concepts/index.md): axes
  and layouts, image formation and noise, estimators, probability
  distributions.
- Looking for an algorithm? The [Algorithms](algorithms/index.md) section
  explains how they are organised: by *what they estimate*, with the assumptions
  they make as tags.
- Need a signature? The [API reference](reference/index.md) lists every public
  name.

## Status

The foundations are in place: configuration, chunked access to arrays,
streaming statistics, smooth bases, special functions, probability
distributions, and simulators of the camera and shading models. The algorithms
that build on them are added by estimation target, and each one comes with a page
that follows the [template](algorithms/index.md#the-page-template).
