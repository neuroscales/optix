# Quick tour

This page walks through one complete round trip: simulate a stack of camera
frames with a known flat field, label its axes, fit an estimator and apply it.
It should take about five minutes, and every snippet runs as written.

optix is organised around two ideas. Data are *array-likes*, which are read a
few frames at a time and never loaded whole. Algorithms are *estimators*,
which learn something from data with `fit` and use it with `apply`. The first
fills in the rest of this page; the details live in
[Axes and layouts](../concepts/axes-and-layouts.md) and
[Estimators](../concepts/estimators.md).

## A synthetic stack

Real calibration data come with an unknown truth, which makes them poor for
learning an API. The [`optix.models`](../reference/models.md) module simulates
the two things that spoil a microscopy image, with the ground truth kept at
hand: a smooth multiplicative *shading* of the field of view, and the noise
of a *camera*.

```python
import numpy as np
import torch

import optix
from optix.models import (
    random_sensor,
    random_shading,
    simulate_sensor,
    simulate_shading,
)

H, W, N = 64, 80, 100  # image height and width, number of frames

sensor = random_sensor((H, W), offset=100.0, gain=0.5, read_sd=1.5, seed=0)
shading = random_shading((H, W), strength=0.3, seed=1)
```

`sensor` and `shading` are *estimates*: small dataclasses of tensors that hold
a camera model (offset, read-noise variance, gain) and a shading model (flat
field and dark image). Here they are the truth we will try to recover. The
flat field `shading.flat` is a smooth map with a mean of 1 and a spread of
about 30 % across the field of view.

Now we image a specimen that is, on average, the same everywhere. The light
is multiplied by the flat field, then the camera turns it into integers:

```python
generator = torch.Generator().manual_seed(2)
light = 400 * torch.rand(N, H, W, generator=generator)  # electrons

signal = simulate_shading(light, shading)  # flat * light
y = simulate_sensor(signal, sensor, seed=3, quantize=torch.uint16)
Y = y.numpy()
```

```pycon
>>> Y.shape, Y.dtype
((100, 64, 80), dtype('uint16'))
```

`Y` is what a file reader would hand you: 100 frames of 64 by 80 pixels, as
unsigned 16-bit integers. The pixel values follow the model
$y = g\,\mathrm{Poisson}(n) + o + \varepsilon$ described in
[Image formation and noise](../concepts/image-formation.md).

## An estimator

optix will ship its algorithms as estimators. Until the first ones land, here
is a small one written with the same building blocks, so that the tour has
something to run. It estimates a flat field as the mean image of many frames,
after subtracting a dark image that it is given.

```{.python title="meanflat.py"}
--8<-- "snippets/meanflat.py"
```

Do not worry about the details yet. Three things matter. The `tags` describe
what the estimator is (what it estimates, how it streams the data, which
noise it assumes). `paper` and `recommended` hold two sets of default options,
which is what `faithful` switches between. And `_fit` and `_apply` are the two
methods an algorithm has to provide: everything else, including the public
`fit` and `apply`, comes from the base class. The
[Estimators](../concepts/estimators.md#writing-an-estimator) page goes through
each of them.

## Fit and apply

The dark image is not something `MeanFlat` can estimate, so we pass it
through `init`. In practice it would come from a stack of frames recorded
without light. To keep the tour short we use the true camera offset instead.

```python
from optix.estimates import ShadingEstimate

dark = ShadingEstimate(dark=sensor.offset)

est = MeanFlat().fit(Y, init=dark)
flat = est.estimates["shading"].flat
```

```pycon
>>> flat.shape
torch.Size([64, 80])
>>> rms = ((flat - shading.flat) ** 2).mean().sqrt()
>>> round(rms.item(), 3)  # error of the estimated flat field
0.06
```

The estimate has a flat-field value for each pixel, with an error of about
6 % of the signal. `apply` uses it to correct the stack, subtracting the dark
image and dividing by the flat field:

```pycon
>>> corrected = est.apply(Y)
>>> corrected.shape, corrected.dtype
(torch.Size([100, 64, 80]), torch.float32)
```

`Y` was never loaded whole: `fit` and `apply` walked through it in batches of
frames, whose size follows a memory budget (256 MiB per chunk by default). A
result that does not fit in memory can be written to any array that accepts
slices, such as a `numpy.memmap` or a `zarr` array, with `est.apply(Y, out=out)`.

## Faithful or not

An estimator that reproduces a published method is more useful if you can
trust that it does. By default every estimator is *faithful*: its options
are those of the original publication. Anything beyond that is an *extension*,
and is switched on explicitly.

Here the extension keeps only the lowest spatial frequencies of the mean
image, on the grounds that a flat field is smooth while the noise is not:

```pycon
>>> est.is_faithful, est.options
(True, {'smooth': 0})
>>> better = MeanFlat(faithful=False).fit(Y, init=dark)
>>> better.is_faithful, better.options
(False, {'smooth': 4})
>>> rms = ((better.estimates["shading"].flat - shading.flat) ** 2).mean().sqrt()
>>> round(rms.item(), 3)
0.039
```

Smoothing lowers the error by about a third. Because the options are recorded
in the estimate's `meta` and in `est.options`, a result always says how it was
obtained.

## Axes with roles

So far the stack was `(frames, height, width)`, which is what optix assumes
when you tell it nothing. Real data are rarely so tidy. Suppose we imaged two
channels, each with its own shading, and stacked them as `(c, z, y, x)`:

```python
other = random_shading((H, W), strength=0.3, seed=5)
y2 = simulate_sensor(
    simulate_shading(light, other), sensor, seed=4, quantize=torch.uint16
)
Y2 = np.stack([Y, y2.numpy()])
```

We want one flat field *per channel*, estimated from all the planes of that
channel. We say so by attaching a layout to the array, without loading
anything:

```python
Z = optix.io.labeled(Y2, axes="czyx", spatial="yx", group="c")

est2 = MeanFlat().fit(Z, init=dark)
```

```pycon
>>> Z
Labeled(ndarray, shape=(2, 100, 64, 80), Layout(axes=('c', 'z', 'y', 'x'), spatial=('y', 'x'), sample=None, group=('c',)))
>>> est2.estimates["shading"].flat.shape
torch.Size([2, 64, 80])
```

Axes `y` and `x` are *spatial*, because together they make one picture. Axis
`c` is a *group*, because each channel has its own parameters. Axis `z` was
not named, so it is a *sample*: planes that share their parameters. The fitted
maps are shaped `(*group, *spatial)`, here `(2, 64, 80)`.
[Axes and layouts](../concepts/axes-and-layouts.md) explains the three roles,
and how the same idea handles OME-Zarr stacks and light-sheet volumes.

## Saving and loading

Estimators and estimates are saved with `torch.save` and no pickled classes,
so a file stays loadable when the code moves on:

```python
est.save("flat.pt")
again = optix.load("flat.pt")
```

```pycon
>>> again.estimates["shading"].allclose(est.estimates["shading"])
True
```

## Where to go next

- [Concepts](../concepts/index.md) explains the ideas behind the API, in
  the order you are likely to meet them.
- [Algorithms](../algorithms/index.md) describes how algorithm pages will be
  organised.
- The [API reference](../reference/index.md) lists every public function and
  class.
