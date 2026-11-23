# Image formation and noise

Before correcting an image, it helps to know what the numbers in it mean. A
pixel value is the end of a short chain: light reaches the sensor, is
converted into a handful of electrons, is amplified, and is finally rounded to
an integer. Each link adds its own imperfection. optix models two families of
them, which are treated separately because they are estimated separately:

- the **camera** turns light into numbers, and does so noisily (the *sensor
  model*);
- the **optics and the specimen illumination** weight the light unevenly
  across the field of view, and add a background (the *shading model*).

This page gives both models, the conventions that results follow, and the
smooth bases used to describe illumination.

## The camera model

Take one pixel, and let $\lambda$ be the number of photoelectrons it
expects to collect during the exposure. Photons arrive at random, so the number
actually collected, $e$, varies from exposure to exposure even if nothing in
the scene changes. It follows a Poisson distribution with mean $\lambda$.
The electronics then multiply the count by a gain $g$, add an offset $o$ so
that the output never goes negative, and add a little noise $\varepsilon$ of
their own (the *read noise*). The recorded value is

$$
y = g\,e + o + \varepsilon,
\qquad e \sim \mathrm{Poisson}(\lambda),
\qquad \varepsilon \sim \mathcal{N}(0, \sigma^2).
$$

| Symbol | Meaning | Unit |
|---|---|---|
| $y$ | recorded pixel value | ADU (analog-to-digital units, the integers in the file) |
| $\lambda$ | expected number of photoelectrons | electrons |
| $e$ | actual number of photoelectrons | electrons |
| $g$ | gain | **ADU per electron** |
| $o$ | camera offset | ADU |
| $\sigma^2$ | read-noise variance | ADU$^2$ |

!!! note "Gain is ADU per electron"
    Datasheets of scientific cameras often quote the *conversion gain* in
    electrons per ADU, which is the inverse. optix always stores $g$ in ADU
    per electron, as in the [EMVA 1288 standard](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf) (where it is called $K$), so
    that $y$ grows with $g$. Invert the datasheet value before using it.

Each parameter can differ from pixel to pixel; modern sCMOS sensors in
particular have an offset, a gain and a read noise of their own at every
pixel. That is why the fields of a `SensorEstimate` are scalars or `(H, W)`
maps.

### Mean and variance

The mean and the variance of $y$ follow directly. The Poisson distribution
has variance equal to its mean, so $\mathrm{Var}[g\,e] = g^2 \lambda$, and the
three noise sources are independent:

$$
\mathbb{E}[y] = g\lambda + o,
\qquad
\mathrm{Var}[y] = g^2\lambda + \sigma^2.
$$

Eliminating $\lambda$ expresses the variance as a function of the mean, the
quantity an image actually offers:

$$
\mathrm{Var}[y] = g\,\bigl(\mathbb{E}[y] - o\bigr) + \sigma^2.
$$

This is a straight line. Its slope is the gain, and where the signal is zero
(a mean of $o$) it reaches the read-noise variance. Writing it as $a\,\mathbb{E}[y] + b$ gives the
Poisson-Gaussian parametrisation of [Foi et al.](https://doi.org/10.1109/TIP.2008.2001399), with $a = g$ and
$b = \sigma^2 - g\,o$ (the symbols $a$ and $b$ are local to this paragraph).
optix stores $(o, \sigma^2, g)$, and $(a, b)$ follow from these as above.
`SensorEstimate.noise_variance(mean)` evaluates the line:

```pycon
>>> import torch
>>> from optix.estimates import SensorEstimate
>>> sensor = SensorEstimate(offset=100.0, variance=2.25, gain=0.5)
>>> sensor.noise_variance(torch.tensor([100.0, 300.0]))
tensor([  2.2500, 102.2500])
```

The two noise sources are equal when $g^2\lambda = \sigma^2$, that is at
$\lambda = \sigma^2 / g^2$ electrons. With the numbers above this is 9 electrons:
darker than that, the image is limited by the camera, brighter and it is
limited by the light itself. Algorithms that treat noise properly have to know
which side of that threshold they are on.

### The photon-transfer curve

The straight line is also how a camera is calibrated. Record frames of a
steady, uniform light at several brightness levels, measure the mean and the
variance of each pixel over the frames, and fit a line. This is called a
*photon-transfer curve*. Here it is on simulated data, using
`optix.models` to produce the frames:

```pycon
>>> from optix.models import random_sensor, simulate_sensor
>>> from optix.stats import Moments
>>> truth = random_sensor(
...     (64, 64), offset=100.0, offset_sd=0.0, read_sd=1.5, gain=0.5, seed=0
... )
>>> mean, var = [], []
>>> for i, electrons in enumerate([0, 20, 50, 100, 200, 400, 800, 1600]):
...     frames = simulate_sensor(
...         torch.full((200, 64, 64), float(electrons)), truth, seed=i
...     )
...     m = Moments(2).update(frames)
...     mean.append(m.mean.mean())
...     var.append(m.var().mean())
>>> mean, var = torch.stack(mean).double(), torch.stack(var).double()
>>> design = torch.stack([mean - 100.0, torch.ones_like(mean)], dim=1)
>>> gain, read_var = torch.linalg.lstsq(design, var[:, None]).solution.flatten()
>>> round(gain.item(), 2), round(read_var.item(), 2)
(0.5, 2.25)
```

The slope recovers the gain of $0.5$ ADU per electron and the intercept the
read-noise variance of $1.5^2 = 2.25$ ADU$^2$. (This fit uses the true
offset to place the intercept; a real calibration estimates it from frames
recorded in the dark. The [EMVA 1288 standard](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf) prescribes the procedure
in full.)

### Quantisation

The analog value is finally rounded to an integer. If the step between
consecutive integers is $\Delta$ (one ADU), the rounding error is
approximately uniform on $[-\Delta/2, \Delta/2]$, and its variance is

$$
\int_{-\Delta/2}^{\Delta/2} u^2\,\frac{\mathrm{d}u}{\Delta} = \frac{\Delta^2}{12}.
$$

For $\Delta = 1$ that is $1/12 \approx 0.083$ ADU$^2$, which adds to the read
noise: $\sigma^2_{\text{eff}} \approx \sigma^2 + \Delta^2/12$. It is
negligible next to a typical read noise of a few ADU, but it sets a floor
for very quiet sensors or data that were binned or down-scaled. The
approximation is good when the noise is wider than a step. With much less
noise the rounding error is no longer uniform, and it is not independent of
the signal.

The ADC also saturates: values below zero or above the largest integer are
clipped. `simulate_sensor(..., quantize=torch.uint16)` rounds and saturates in
the same way, and the effect on the variance is measurable:

```pycon
>>> quiet = random_sensor(
...     (1000, 1000), offset=100.0, offset_sd=0.0, read_sd=2.0, gain=0.5, seed=0
... )
>>> dark = torch.zeros(4, 1000, 1000)  # no light at all
>>> analog = simulate_sensor(dark, quiet, seed=1)
>>> digital = simulate_sensor(dark, quiet, seed=1, quantize=torch.uint16)
>>> round(analog.var().item(), 2)
4.0
>>> round(digital.float().var().item(), 2)  # about 4 + 1/12
4.08
```

## The shading model

The second family of imperfections concerns *where* in the picture the light
goes. A microscope does not illuminate and image the field of view evenly:
a lens vignettes, a light sheet is brighter at its waist, a LED has a profile.
These effects multiply the signal by a smooth map. Other effects add a
background that does not depend on the specimen: the camera offset, dark
current, stray light. Together they give the **shading model**

$$
\mathbf{y} = \mathbf{f} \odot \mathbf{x} + \mathbf{d},
$$

where $\mathbf{y}$ is the recorded image, $\mathbf{x}$ the **latent image** we
would record under uniform illumination and without additive background,
$\mathbf{f}$ the **flat field** (multiplicative, dimensionless) and
$\mathbf{d}$ the **dark field** (additive, in ADU). All four are images, and
$\odot$ is the pixel-wise product (see [Notation](notation.md)). When frames
are recorded in a sequence and the overall level drifts, for example because
of a fluctuating light source or a background in the specimen that changes
from frame to frame, each frame $n$ may carry an additional constant $b_n$,
the **baseline**:

$$
\mathbf{y}_n = \mathbf{f} \odot (\mathbf{x}_n + b_n) + \mathbf{d} .
$$

The baseline sits inside the flat field: it is light from the specimen
(autofluorescence, fluorescence of the medium, out-of-focus haze), lit by the
same uneven illumination as the signal. [Notation](notation.md#what-the-fields-mean-physically)
lists which physical quantity ends up in which term.

In code the three terms are the fields `flat`, `dark` and `baseline` of a
`ShadingEstimate`; `baseline` has shape `(*batch, frames)`.

Correcting an image means inverting the model:

$$
\hat{\mathbf{x}}_n = \frac{\mathbf{y}_n - \mathbf{d}}{\mathbf{f}} - b_n .
$$

An estimate may instead declare `meta["baseline_units"] = "raw"`, for a
baseline that is added after the flat field,
$\mathbf{y}_n = \mathbf{f} \odot \mathbf{x}_n + \mathbf{d} + b_n$: a drifting
camera offset, say. The default is `"signal"`, the model above.

### How the two models fit together

The flat field acts on the light, before it reaches the camera. If the
specimen would produce $\lambda^0$ electrons under uniform illumination, the
pixel receives $f\lambda^0$, and the whole chain is

$$
y = g\,\mathrm{Poisson}(f\,\lambda^0) + o + \varepsilon,
\qquad
\mathbb{E}[y] = f\,(g\lambda^0) + o .
$$

(one pixel, so all scalars). This is a shading model with $x = g\lambda^0$
and $d = o$: the dark field here is the offset of the camera. Note also what
the flat field does to the noise. The variance of $y$ is
$g^2 f\lambda^0 + \sigma^2$, so a dim corner is not only darker but also
*relatively* noisier, and after dividing by $f$ the variance of the corrected
image is that of the recorded one divided by $f^2$. A
correction removes the bias of shading and amplifies the noise of the
pixels it brightens. This is why the noise model matters for flat-field
estimation as well.

`optix.models` simulates exactly this chain, in two steps:

```pycon
>>> from optix.models import random_shading, simulate_shading
>>> shading = random_shading((64, 64), strength=0.3, seed=1)
>>> light = 100 * torch.ones(10, 64, 64)  # electrons, uniform specimen
>>> y = simulate_sensor(simulate_shading(light, shading), truth, seed=0)
>>> y.shape
torch.Size([10, 64, 64])
```

### Result conventions

Estimators from different algorithms have to produce results that can be
compared, chained and applied by the same code. The result types in
`optix.estimates` therefore follow a handful of conventions. They are the
ones that most often cause silent mistakes, so it is worth knowing them.

- **Flat fields are stored as produced.** The model has a scale ambiguity:
  $(\mathbf{f}, \mathbf{x}_n, b_n)$ and $(c\mathbf{f}, \mathbf{x}_n/c, b_n/c)$
  describe the same images for any $c > 0$. Each
  algorithm resolves it its own way, and optix does not rescale behind your
  back. The convention used is recorded in `meta["flat_norm"]`, which is
  `"mean"` (the mean of the flat is 1) or `"geomean"` (the mean of its
  logarithm is 0). `ShadingEstimate.normalize(mode)` returns a copy
  rescaled to the convention you want, with `meta` updated. The scale moves
  into the latent image, so `dark` is untouched and a baseline inside the flat
  field is rescaled with the signal (one added after it is untouched). A
  similar ambiguity exists between $\mathbf{d}$ and the $b_n$: replacing
  $\mathbf{d}$ by $\mathbf{d} + c\,\mathbf{f}$ and every $b_n$ by $b_n - c$
  changes nothing, so a constant can move from one to the other.

- **`dark` includes the camera offset.** `ShadingEstimate.dark` is the
  whole additive image of the correction
  $(\mathbf{y}_n - \mathbf{d})/\mathbf{f} - b_n$, in ADU.
  `SensorEstimate.offset` is the camera offset alone. A model that has only
  one additive term returns it in both fields. Because of that overlap,
  *corrections never subtract both*: pick the `dark` of a `ShadingEstimate`, or
  the `offset` of a `SensorEstimate` when there is no dark field.

- **Gain is in ADU per electron**, as above.

- **`None` means "not estimated"**, never zero. A consumer that needs the
  field falls back to its own default (a flat field of 1, say), or raises an
  error that names the field if it has none. Calling `noise_variance` on a
  `SensorEstimate` without a gain is such an error.

- **Estimates are model-sized.** Every field broadcasts against the spatial
  shape: a scalar, or a map of the spatial shape (with the group axes in front when
  the layout has groups). Results the size of the data, such as corrected
  stacks or class probabilities, are never stored in an estimate: they come
  from `Estimator.apply(Y, out=)`.

```pycon
>>> shading.meta["flat_norm"]
'mean'
>>> geo = shading.normalize("geomean")
>>> geo.meta["flat_norm"], round(geo.flat.log().mean().exp().item(), 6)
('geomean', 1.0)
>>> torch.equal(geo.dark, shading.dark)
True
```

## Smooth illumination: DCT bases

Flat fields vary slowly across the field of view, which is also what makes
them identifiable: structure that is smooth and shared by every frame can only
be the instrument. Describing a flat field with a handful of smooth basis
functions rather than with one value per pixel makes it cheaper to estimate and
much less sensitive to noise.

optix uses the orthonormal discrete cosine transform (DCT-II). For $L$ samples, the
$j$-th basis function evaluated at sample $m$ is

$$
\Phi_{mj} = c_j \cos\!\Bigl(\frac{\pi}{L}\,\bigl(m + \tfrac12\bigr)\,j\Bigr),
\qquad c_0 = \sqrt{\tfrac1L},\quad c_{j>0} = \sqrt{\tfrac2L},
$$

for $m = 0, \ldots, L-1$. The columns of $\boldsymbol{\Phi}$ are orthonormal,
$\boldsymbol{\Phi}^\top \boldsymbol{\Phi} = \mathbf{I}$, and
function $j$ completes $j/2$ oscillations over the axis. Keeping only the $k$
lowest-frequency columns gives a smooth subspace; with $k = 1$ the picture is
constant. In 2D the basis is separable: for an $H \times W$ picture held as
a matrix $\mathbf{F}$, the coefficients are
$\boldsymbol{\Theta} = \boldsymbol{\Phi}_1^\top \mathbf{F} \boldsymbol{\Phi}_2$
and the reconstruction is
$\mathbf{F} = \boldsymbol{\Phi}_1 \boldsymbol{\Theta} \boldsymbol{\Phi}_2^\top$. `project` and `reconstruct` do this for the
trailing dimensions of a tensor.

```pycon
>>> from optix.basis import dct_basis, dct_bending, project, reconstruct
>>> by, bx = dct_basis(64, 4), dct_basis(80, 4)  # 4 functions per axis
>>> by.shape, torch.allclose(by.T @ by, torch.eye(4), atol=1e-6)
(torch.Size([64, 4]), True)
>>> flat = random_shading((64, 80), order=3, strength=0.3, seed=1).flat
>>> noisy = flat * (
...     1 + 0.1 * torch.randn(64, 80, generator=torch.Generator().manual_seed(0))
... )
>>> coefficients = project(noisy, [by, bx])
>>> coefficients.shape
torch.Size([4, 4])
>>> smooth = reconstruct(coefficients, [by, bx])
>>> def rms(a, b):
...     return ((a - b) ** 2).mean().sqrt().item()
>>> round(rms(noisy, flat), 3), round(rms(smooth, flat), 3)
(0.102, 0.039)
```

The DCT has a second virtue: it diagonalises the *negative* discrete Laplacian
with reflecting (Neumann) boundaries, which are the natural boundary conditions of an
image, $-\boldsymbol{\Delta}_h \boldsymbol{\phi}_j = \kappa_j \boldsymbol{\phi}_j$ for the $j$-th column $\boldsymbol{\phi}_j$ of $\boldsymbol{\Phi}$. The eigenvalue of function $j$ on an axis of $L$ samples spaced $h$ apart
is

$$
\kappa_j = \frac{2 - 2\cos(\pi j / L)}{h^2},
$$

and for the separable function with indices $(j_1, \ldots, j_d)$ the weight
$\bigl(\kappa_{j_1} + \cdots + \kappa_{j_d}\bigr)^\ell$ is the discrete energy
$\mathbf{f}^\top(-\boldsymbol{\Delta}_h)^\ell \mathbf{f}$: the gradient (membrane) energy for $\ell = 1$ and
$\lVert\boldsymbol{\Delta}_h \mathbf{f}\rVert^2$ for $\ell = 2$, which approximates the bending energy.
Either can serve as a penalty on the coefficients that grows with frequency and vanishes for the
constant function, which leaves the overall level free. `dct_bending(shape, k, order=...)` returns these
weights, one per coefficient:

```pycon
>>> weights = dct_bending((64, 80), (4, 4), order=2)
>>> weights.shape, weights[0, 0].item()
(torch.Size([4, 4]), 0.0)
>>> bool(weights[0, 1] < weights[0, 2] < weights[0, 3])
True
```

`random_shading` is built from the same pieces: it draws random coefficients on
the lowest `order` DCT functions of each axis (leaving out the constant),
exponentiates the result, and normalises it to a mean of 1.

!!! references "References"

    1. Foi A, Trimeche M, Katkovnik V, Egiazarian K. *Practical
       Poissonian-Gaussian noise modeling and fitting for single-image raw-data.*
       IEEE Trans Image Process **17**(10):1737-1754, 2008.
       [doi:10.1109/TIP.2008.2001399](https://doi.org/10.1109/TIP.2008.2001399)

    ??? quote "BibTeX"

        ```bibtex
        @article{foi2008practical,
          author  = {Foi, Alessandro and Trimeche, Mejdi and Katkovnik, Vladimir and Egiazarian, Karen},
          title   = {Practical {P}oissonian-{G}aussian noise modeling and fitting for single-image raw-data},
          journal = {IEEE Transactions on Image Processing},
          volume  = {17},
          number  = {10},
          pages   = {1737--1754},
          year    = {2008},
          doi     = {10.1109/TIP.2008.2001399},
        }
        ```

    2. European Machine Vision Association. *EMVA Standard 1288: Standard for
       Characterization of Image Sensors and Cameras, Release 4.0 Linear.*
       16 June 2021. [emva.org](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf)

    ??? quote "BibTeX"

        ```bibtex
        @techreport{emva1288linear40,
          author      = {{European Machine Vision Association}},
          title       = {{EMVA} Standard 1288: Standard for Characterization of Image Sensors and Cameras, Release 4.0 Linear},
          institution = {EMVA}, year = {2021}, month = jun,
          url = {https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf}
        }
        ```
