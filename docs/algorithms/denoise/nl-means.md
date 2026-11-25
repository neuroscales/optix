# Non-local means

!!! info "At a glance"
    - **Estimates:** denoised image (or volume), same shape as the data
    - **Noise model:** Gaussian (white, known $\sigma$); Poisson–Gaussian through the generalised Anscombe transform
    - **Streaming:** halo
    - **Needs many frames:** no
    - **Reference:** Buades, Coll & Morel, CVPR, 2005, [doi:10.1109/CVPR.2005.38](https://doi.org/10.1109/CVPR.2005.38)

## Overview

`NLMeans` removes noise from an image by replacing every pixel with a weighted
mean of the pixels around it, as in [Buades et al. (2005)](https://doi.org/10.1109/CVPR.2005.38).
The weights do not depend on how close two pixels are, but on how much their
*neighbourhoods* look alike: a small patch around the pixel is compared with the
patches around the other pixels of a search window, and the pixels whose
patches are similar get large weights. A pixel on a filament is averaged with
the other pixels of the filament, however far away, and not with the dark
background beside it, so edges and thin structures are kept. The comparison of
whole patches is also what makes the weights robust to noise, where comparing
single grey levels (as the neighbourhood filters do) is not.

Use it when the noise is white and Gaussian with a known standard deviation, or
when the data are Poisson–Gaussian (a camera) and a [`SensorEstimate`](../../reference/estimates.md)
is available: the data are then stabilised by the
[generalised Anscombe transform](../sensor/generalized-anscombe.md), denoised
with $\sigma = 1$ and transformed back by its exact unbiased inverse. It has a
single filtering parameter $h$, set to $10\sigma$ by default, and no training
or transform: it is the simplest of the non-local denoisers. For the same
non-local idea with a transform-domain filter, see [BM3D and BM4D](bm3d.md). It is not a
deconvolution: it keeps blur as it finds it (see
[Richardson–Lucy](../deconv/richardson-lucy.md) and [Wiener](../deconv/wiener.md)),
and the noise must be uncorrelated between pixels: stripes are matched as if
they were signal (see [VSNR](../destripe/vsnr.md)). It works best where the
image is redundant (textures, periodic structures, many similar objects) and
the cost grows with the size of the search window.

The estimator is [`NLMeans`][optix.denoise.NLMeans], with the function
[`nl_means`][optix.denoise.nl_means] as a shortcut for Gaussian noise (see the
[API reference](../../reference/denoise.md)). `fit` fixes the noise model,
`apply` denoises, and large arrays are streamed in blocks with a halo,
giving the result of the whole-image computation.

## Image formation (optics)

Only the noise is modelled; the optics are not (see
[Image formation and noise](../../concepts/image-formation.md)). The recorded
image is the latent image plus white Gaussian noise:

=== "optix"

    $$
    \mathbf{y} = \mathbf{x} + \boldsymbol{\varepsilon},
    \qquad
    \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
    $$

=== "Buades et al. 2005"

    $$
    v(i) = u(i) + n(i), \quad i \in I,
    \qquad n(i)\ \text{white noise of standard deviation}\ \sigma
    $$

| optix | Buades et al. 2005 | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}$ | $v$ | noisy image or volume | $\mathbb{R}^{\lvert\Omega\rvert}$, per image or volume |
| $\mathbf{x}$ | $u$ | latent (noise-free) image or volume | $\mathbb{R}^{\lvert\Omega\rvert}$, per image or volume |
| $\boldsymbol{\varepsilon}$ | $n$ | white Gaussian noise | $\mathbb{R}^{\lvert\Omega\rvert}$, per image or volume |
| $\sigma$ | $\sigma$ | noise standard deviation, in the units of the data | scalar, one per call (one per fit) |
| $\Omega$ | $I$ | pixel (voxel) grid, 2D or 3D | $\lvert\Omega\rvert = HW$ or $DHW$ |

The paper calls the noisy image $v$ and the noise-free one $u$: the reverse of
the letters of optix, where $\mathbf{y}$ is what the camera recorded. Images of a
stack, or of different groups, are denoised one by one, each with the same
$\sigma$.

Poisson–Gaussian data, the camera model $y = g\,e + o + \varepsilon$ with
$e \sim \mathrm{Poisson}(\lambda)$, are brought to this model by the generalised
Anscombe transform $f$ of
[Mäkitalo & Foi (2013)](https://doi.org/10.1109/TIP.2012.2202675), as for
[BM3D and BM4D](bm3d.md): see
[Poisson–Gaussian data](#poissongaussian-data) below. The paper does not treat that case.

## Probabilistic model

Non-local means is not derived from a probabilistic model: it is a filter, built
on two assumptions. First, the noise is white and Gaussian with a known
$\sigma$ (above). Second, *redundancy*: for most pixels there are other pixels,
somewhere in the image, whose neighbourhoods are the same up to noise, and the
latent values at these pixels are the same too. The mean of the recorded values
at such pixels then has a noise variance divided by their number, without the
blur of a mean over a fixed neighbourhood.

The paper makes the second assumption precise in its Section 4: if the image is
a realisation of a stationary, mixing random field, then as the image grows
NL-means converges almost surely to the conditional expectation
$\mathrm{E}[\,y_i \mid \text{the rest of the patch of } i\,]$ of the
pixel given its neighbourhood (Theorem 4 of [Buades et al. 2005](https://doi.org/10.1109/CVPR.2005.38)), which is the
estimator of least mean square error. In that section the patch excludes the
pixel itself; the algorithm of Section 3, which the method implements, does
not.

Notation, with the symbols of the [notation page](../../concepts/notation.md#non-local-means)
and the paper's own:

- A *patch* ($N \times N$ pixels in 2D, $N \times N \times N$ voxels in 3D)
  is a vector $\mathbf{p}_i = \mathbf{P}_i\,\mathbf{y} \in \mathbb{R}^{K}$, $K = N^d$,
  cut from the image *centred* on the pixel $i$ by the operator $\mathbf{P}_i$.
  Unlike the blocks of BM3D, which are indexed by their first corner, patches are
  indexed by their centre, so that $N$ is odd.
- The *search window* $W_i$ of $i$ is the square (cube) of side $N_S$ centred
  on $i$, clipped to the image.
- The patches are compared with a *Gaussian window* $\boldsymbol{\omega} \in \mathbb{R}^{K}$ of
  standard deviation $a$ pixels, which gives more weight to the pixels near the
  centre of the patch:
  $\omega_{\mathbf{u}} \propto \exp\bigl(-\lVert\mathbf{u}\rVert^2 / (2a^2)\bigr)$ for the offsets
  $\mathbf{u}$ in the patch. It is separable, the outer product of $d$ copies of a
  1-D window of $N$ values.

=== "optix"

    $$
    \hat{x}_i = \sum_{j \in W_i} w_{ij}\, y_j,
    \qquad
    w_{ij} = \frac{1}{Z_i}\exp\!\Bigl(-\frac{d_{ij}}{h^2}\Bigr),
    \qquad
    Z_i = \sum_{j \in W_i} \exp\!\Bigl(-\frac{d_{ij}}{h^2}\Bigr)
    $$

    $$
    d_{ij} = \lVert \mathbf{p}_i - \mathbf{p}_j \rVert_{\boldsymbol{\omega}}^2
    = \sum_{\mathbf{u}} \omega_{\mathbf{u}}\,\bigl(y_{i+\mathbf{u}} - y_{j+\mathbf{u}}\bigr)^2
    $$

=== "Buades et al. 2005"

    $$
    NL[v](i) = \sum_{j} w(i, j)\, v(j),
    \qquad
    w(i, j) = \frac{1}{Z(i)}\, e^{-\frac{\lVert v(N_i) - v(N_j)\rVert_{2,a}^2}{h^2}},
    \qquad
    Z(i) = \sum_{j} e^{-\frac{\lVert v(N_i) - v(N_j)\rVert_{2,a}^2}{h^2}}
    $$

    $$
    \lVert v(N_i) - v(N_j)\rVert_{2,a}^2
    = \bigl(G_a * \lvert v(i + \cdot) - v(j + \cdot)\rvert^2\bigr)(0)
    $$

| optix | Buades et al. 2005 | Meaning | Shape |
|---|---|---|---|
| $\hat{x}_i$ | $NL[v](i)$ | estimate at pixel $i$ | scalar, per pixel |
| $w_{ij}$ | $w(i, j)$ | weight of pixel $j$ in the estimate of $i$; $0 \le w_{ij} \le 1$, $\sum_j w_{ij} = 1$ | scalar, per pair in a window |
| $Z_i$ | $Z(i)$ | normalising constant | scalar, per pixel |
| $d_{ij}$ | $\lVert v(N_i) - v(N_j)\rVert_{2,a}^2$ | Gaussian-weighted squared distance between the patches at $i$ and $j$ | scalar, per pair in a window |
| $\mathbf{p}_i$ | $v(N_i)$ | patch centred on $i$ | $\mathbb{R}^{K}$, $K = N^d$ |
| $N$ | 7 (the paper's value) | side of the patch | scalar |
| $W_i$, $N_S$ | the $S \times S$ search window ($S = 21$); in the formula, all of $I$ | search window of $i$ and its side | scalar $N_S$ |
| $\boldsymbol{\omega}$, $a$ | $G_a$, $a$ | Gaussian window on the patch and its standard deviation | $\mathbb{R}^{K}$, scalar |
| $h$ | $h$ | filtering parameter, in the units of the data | scalar, one per call |

The paper's formula sums over all the pixels of the image and, "for
computational purposes", restricts the sum to a search window of $21 \times 21$
pixels with patches of $7 \times 7$ and $h = 10\sigma$ in all its experiments
(Section 5). The estimator implements the windowed form.

## Inference / derivation

There is nothing to fit: the estimate is a closed-form function of the data
and of the four settings $\sigma$ (through $h$), $N$, $N_S$ and the window
$\boldsymbol{\omega}$. In statistical terms it is a kernel-regression
(Nadaraya–Watson) estimator, in which the "predictor" of pixel $i$ is its patch,
the kernel is Gaussian, and its bandwidth is $h$. Three properties follow
from the formula.

- The pixel always takes part in its own estimate with the largest unnormalised
  weight, $\exp(0) = 1$, so $Z_i \ge 1$: the normalisation never divides by a
  small number.
- As $h \to 0$ only patches identical to that of $i$ keep a weight, and the
  estimate tends to the data; as $h \to \infty$ all the weights are equal and it
  tends to the mean of the search window. `h=float("inf")` gives that mean.
- Large distances give weights that are numerically zero, "an automatic
  threshold" in the words of the paper, so that only the similar patches
  contribute.

### Scale of the distance and of $h$

Since $\hat{x}$ depends on $d_{ij}/h^2$, the scale of $\boldsymbol{\omega}$ and
that of $h$ are one and the same choice, and the paper fixes neither of the two
completely. For two patches of the *noisy* image, the paper states

$$
\mathrm{E}\,\lVert v(N_i) - v(N_j)\rVert_{2,a}^2 = \lVert u(N_i) - u(N_j)\rVert_{2,a}^2 + 2\sigma^2 ,
$$

which holds when the Gaussian window integrates to 1. For a general window the
noise adds $2\sigma^2 \sum_{\mathbf{u}} \omega_{\mathbf{u}}$, since each pixel difference of noise has
variance $2\sigma^2$:

$$
\mathrm{E}\,d_{ij} = d_{ij}^{\,\text{clean}} + 2\sigma^2 \sum_{\mathbf{u}} \omega_{\mathbf{u}} .
$$

Two scales of $\boldsymbol{\omega}$ are possible, selected by `kernel_norm`:

| `kernel_norm` | $\sum_{\mathbf{u}} \omega_{\mathbf{u}}$ | distance $d_{ij}$ | noise-only $\mathrm{E}\,d_{ij}$, $K = 49$ |
|---|---|---|---|
| `"mean"` (default) | $K$ (mean 1 over the patch) | weighted *sum* of squared differences | $2\sigma^2 K = 98\sigma^2$ |
| `"sum"` | 1 (sum 1) | weighted *mean* of squared differences | $2\sigma^2$ |

The two distances differ by the factor $K$, and the same estimate is obtained
from one with $h$ and from the other with $h/\sqrt{K}$: for $7 \times 7$
patches, a factor 7 in $h$.

The identity of the paper is written for a window of sum 1, which suggests
`"sum"`. With it and $h = 10\sigma$, two patches that differ by noise alone have
$d \approx 2\sigma^2$ and a weight $\exp(-2\sigma^2/100\sigma^2) \approx 0.98$: nearly every patch of the
window gets a weight close to 1, and the result is close to the mean of the
window, which blurs the image (see the [results](#results-on-synthetic-data)).
With `"mean"` the same two patches have $d \approx 98\sigma^2$ and a weight
$\exp(-0.98) \approx 0.37$, patches that differ by $\delta$ grey levels per
pixel have a weight $\exp(-49\delta^2/100\sigma^2)$, which is small for
$\delta \gtrsim 1.5\sigma$, and $h = 10\sigma$ works as the paper describes it.
The default is therefore `"mean"`; the setting is
open to the user, and `kernel_norm="sum"` with $h = 10\sigma/7$ is the same
estimator.

## Algorithm

**Input:** noisy images $\mathbf{y} \in \mathbb{R}^{\lvert\Omega\rvert}$ (volumes with `ndim=3`);
noise standard deviation $\sigma$ (or a `SensorEstimate`); $h = 10\sigma$, $N = 7$, $N_S = 21$,
$a = (N - 1)/4$, `kernel_norm="mean"` \
**Output:** denoised images $\hat{\mathbf{x}} \in \mathbb{R}^{\lvert\Omega\rvert}$.

1. **Variance stabilisation** (with a `SensorEstimate` only): replace $\mathbf{y}$ by
   $f(\mathbf{y})$ and $\sigma$ by $1$.
2. **Window.** Build the 1-D window $\omega(u) \propto \exp(-u^2/2a^2)$ for
   $u = -\lfloor N/2\rfloor, \ldots, \lfloor N/2\rfloor$, normalised to sum $N$
   (`"mean"`) or 1 (`"sum"`), and $\boldsymbol\omega$ as its outer product.
3. **Pad** $\mathbf{y}$ by reflection by $\lfloor N/2\rfloor$ pixels, and by zeros by
   $\lfloor N_S/2\rfloor$ more.
4. **Accumulate.** Set $\mathbf{n} = \mathbf{0}$ and $\mathbf{z} = \mathbf{0}$ (numerator and
   denominator, images). For every offset $\mathbf{t} \in \{-\lfloor N_S/2\rfloor, \ldots, \lfloor N_S/2\rfloor\}^d$,
   in raster order, with $j = i + \mathbf{t}$:
    1. **distance**: $d_{\mathbf{t}}(i) = \sum_{\mathbf{u}} \omega_{\mathbf{u}} (y_{i+\mathbf{u}} - y_{i+\mathbf{t}+\mathbf{u}})^2$,
       the weighted sum over the patch of the squared difference between $\mathbf{y}$ and
       its shift by $\mathbf{t}$, computed one axis at a time;
    2. **weight**: $w_{\mathbf{t}}(i) = \exp(-d_{\mathbf{t}}(i)/h^2)$, and 0 where $i + \mathbf{t}$ is outside the
       image;
    3. **add**: $\mathbf{n} \leftarrow \mathbf{n} + w_{\mathbf{t}} \odot \mathbf{y}_{\mathbf{t}}$ and $\mathbf{z} \leftarrow \mathbf{z} + w_{\mathbf{t}}$, with $\mathbf{y}_{\mathbf{t}}$ the shifted image.
5. **Divide**: $\hat{\mathbf{x}} = \mathbf{n} / \mathbf{z}$.
6. **Invert the transform** (with a `SensorEstimate` only): $\hat{\mathbf{x}} \leftarrow g\,\mathcal{I}_{\sigma/g}(\hat{\mathbf{x}}) + o$.

Offsets are taken in batches, and the sums over offsets run one by one in the
same order, so the result does not depend on the batch size. The cost is
$N_S^d (2N + 4)$ operations per pixel, since the sum over the patch is
separable, against $N^d N_S^d$ for the paper's direct computation.

## Faithful mode and extensions

`faithful=True` (the default) is the setting of Section 5 of the paper: images
(2-D), patches of $7 \times 7$, a search window of $21 \times 21$ and
$h = 10\sigma$. The paper leaves a number of choices open, and the code makes
them as follows.

### Choices not in the paper

1. **The window $\boldsymbol\omega$.** The paper says that $G_a$ is a Gaussian
   kernel of standard deviation $a$, and gives neither $a$ nor its sampling. It
   is the Gaussian sampled at the integer offsets of the patch, with the default
   $a = (N - 1)/4$: 1.5 pixels for $7 \times 7$ patches, so that the patch spans
   $\pm 2a$. `a=float("inf")` gives uniform weights.
2. **`kernel_norm`.** The window is normalised to mean 1 over the patch
   (`"mean"`, default) rather than to sum 1 (`"sum"`), for the reasons given in
   [Scale of the distance and of h](#scale-of-the-distance-and-of-h). The two differ by a
   factor 7 in $h$ for $7 \times 7$ patches: the paper's $h = 10\sigma$ denoises with
   `"mean"` and over-smooths with `"sum"`. The paper's identity of the expected
   distance holds for `"sum"` (it is then $2\sigma^2$); for `"mean"` the same
   expectation is $2\sigma^2 K$.
3. **Window of the search.** The sum of the paper runs over the whole image; in
   its experiments it is restricted to a $21 \times 21$ window. The window is
   centred on the pixel (offsets $-10$ to $10$) and clipped at the border of
   the image: only pixels inside the image take part.
4. **Borders.** The patches that cross the border are completed by reflection
   about the edge pixel (replication along an axis too short to reflect). The
   paper does not treat the border.
5. **$h$ and $\sigma$.** The paper sets $h = 10\sigma$ "when a noise of standard
   deviation $\sigma$ is added": $\sigma$ is the known standard deviation of the
   noise, in the units of the data (1 after stabilisation).
6. **The pixel itself.** The pixel is in its own window, with the weight
   $\exp(0) = 1$, as the formula has it, without special treatment.
7. **Stacks.** Each image (or volume) of a stack is denoised separately.

### Extensions (opt-in, not faithful)

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `ndim=3` | volumes: cubic patches and search windows of the same sides, with the separable 3-D window | `2` | `2` |
| `bias_correction` | replaces $d_{ij}$ by $\max(d_{ij} - 2\sigma^2 \sum_{\mathbf{u}}\omega_{\mathbf{u}},\, 0)$ | `False` | `False` |

**3-D.** The paper treats images. For a volume, `ndim=3` takes the cubic
$N^3$ patch, the $N_S^3$ search window and the outer product of three 1-D
windows. The cost per voxel grows from $N_S^2$ to $N_S^3$ offsets, so a
smaller search window than 21 is used in practice (see the results). A volume
can also be denoised plane by plane with `ndim=2` (the default), which is the
faithful method applied to each plane.

**Bias correction.** The paper notes that the noisy distance is biased by
$2\sigma^2$ (the identity of the previous section) but uses it as it is. The
correction subtracts the expected distance of two patches of pure noise,
$2\sigma^2 \sum_{\mathbf{u}}\omega_{\mathbf{u}}$, and clamps the result at 0, so that two patches
whose difference is within the noise get the weight 1. It lets $h$ be
smaller for the same smoothing of the flat regions. This is not in the paper.

The settings of the paper can also be changed; each sets `is_faithful` to false:

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `h` | filtering parameter, in data units (stabilised units with a `SensorEstimate`) | `None` ($10\sigma$) | `None` |
| `patch` | odd side $N$ of the patch | 7 | 7 |
| `search` | odd side $N_S$ of the search window | 21 | 21 |

`a` and `kernel_norm` are the open choices 1 and 2 above and do not change
`is_faithful`. The other keywords do not change the method:

| Keyword | Changes | Faithful value |
|---|---|---|
| `sigma` | noise standard deviation $\sigma$; omit it to give a `SensorEstimate` to `fit` | `None` |
| `inverse` | inverse of the Anscombe transform, for Poisson–Gaussian data | `"exact"` |
| `block` | core block size per spatial axis of the streaming; the result does not depend on it | `None` (memory budget) |

### Poisson–Gaussian data

With `fit(init=sensor)`, the three parameters $g$, $\sigma$, $o$ of the
[`SensorEstimate`](../../reference/estimates.md) define the generalised
Anscombe transform $f$, pixel by pixel when they are maps (such as the offset
and variance maps of an sCMOS camera), and the estimate is

$$
\hat{\mathbf{x}} = g\,\mathcal{I}_{\sigma/g}\bigl(\operatorname{NL}(f(\mathbf{y}),\ \sigma = 1)\bigr) + o,
$$

with $\mathcal{I}_{\sigma/g}$ the exact unbiased inverse of
[Mäkitalo & Foi (2013)](https://doi.org/10.1109/TIP.2012.2202675), the
default; `inverse` selects the closed-form, asymptotic or algebraic inverse
instead. The stabilised noise has unit variance, so $h$ defaults to 10 in
stabilised units, and `h` and `sigma` are not needed. This chaining is an
application of the transform, not part of the paper, and it is the same as for
[BM3D](bm3d.md).

## Memory and streaming

The tag is `halo`. The estimate at a pixel depends on the data within
$\lfloor N_S/2 \rfloor + \lfloor N/2 \rfloor$ pixels: the pixels of its search
window and their patches. [`nl_means_halo`][optix.denoise.nl_means_halo] computes
it, and `NLMeans.halo()` does for a fitted estimator:

| `patch`, `search` | `ndim` | halo |
|---|---|---|
| 7, 21 (paper) | 2 | 13 |
| 5, 9 | 2 | 6 |
| 3, 7 | 3 | 4 |

`apply` reads the data with the [`blocks`](../../reference/io.md) iterator: a core
of `block` pixels per spatial axis (by default sized from the memory budget),
with this halo around it. Each block is denoised, the core is kept and written to
the output, and the result equals the whole-image computation, up to floating-point
rounding, for any `block`. The data may be a zarr array or a memory-mapped
file larger than memory; the output cannot be the input (an in-place call is
refused, as a halo needs the unchanged neighbours). `fit` does not read the
data (NL-means needs no intensity range), and `apply` makes one pass. Images
of a stack, and groups, are independent. Maps of a `SensorEstimate` are cropped
to each block.

Within a block of $P$ values (padded), the computation holds the padded image,
the numerator and the denominator, and a few temporary images, about six
buffers of $P$ values, and a *batch of offsets*: each offset of the batch
needs about five more buffers (the shifted image, the squared differences, the
two passes of the separable sum, the weights). The batch is sized to half the
memory budget of [`optix.config`](../../reference/configuration.md), and the
default `block` leaves the other half for the six buffers of the block. A
batch is at least one offset. The cost per pixel is about $N_S^d (2N + 4)$
operations, linear in the size of the image; the search window $N_S$ dominates
and is the one to lower first (extension) to go faster. The computation is vectorised and runs on any
torch device. The offsets are added one by one in a fixed order, so the result
does not depend on the size of the batches. A $128^2$ image
takes about half a second on a CPU with the default parameters.

## Usage

The first example denoises a synthetic image of filaments with Gaussian noise
of standard deviation 25 on a range of 255. It also shows what the choice of
the kernel normalisation (`kernel_norm`) does at the paper's $h = 10\sigma$.

```pycon
>>> import torch
>>> from optix.denoise import NLMeans, nl_means, nl_means_halo
>>> from optix.models import phantoms
>>> _ = torch.manual_seed(0)
>>> x = phantoms.filaments((64, 64), intensity=200.0, seed=0)
>>> x = 255 * (x - x.min()) / (x.max() - x.min())
>>> y = x + 25 * torch.randn_like(x)
>>> def psnr(a):
...     return round(float(10 * torch.log10(255**2 / (a - x).square().mean())), 1)
>>> psnr(y)
20.2
>>> psnr(nl_means(y, sigma=25))
27.5
>>> psnr(nl_means(y, sigma=25, kernel_norm="sum"))
15.4
```

With a kernel of sum 1 the paper's $h = 10\sigma$ is too large and the result
is close to the mean of the window. The same estimator is obtained with
`kernel_norm="sum"` and $h$ seven times smaller:

```pycon
>>> a = nl_means(y, sigma=25)
>>> b = nl_means(y, sigma=25, kernel_norm="sum", h=250 / 7)
>>> bool((a - b).abs().max() < 1e-2)
True
```

The estimator form streams the image in blocks with a halo, and gives the
whole-image result: to rounding in float32, where the order in which the
offsets are summed can move a value by one ulp. Here a $60 \times 70$ image is
cut into blocks of $17 \times 23$ pixels; `patch=5` and `search=9` only shorten
the example, and reduce the halo from 13 to 6 pixels.

```pycon
>>> big = 255 * phantoms.filaments((60, 70), intensity=1.0, seed=1)
>>> noisy_big = big + 20 * torch.randn_like(big)
>>> est = NLMeans(sigma=20, patch=5, search=9, block=(17, 23)).fit()
>>> est.halo(), nl_means_halo()
(6, 13)
>>> out = est.apply(noisy_big)
>>> ref = nl_means(noisy_big, sigma=20, patch=5, search=9)
>>> bool((out - ref).abs().max() < 1e-3)
True
>>> est.is_faithful
False
```

Poisson–Gaussian data: a camera with gain 2, offset 100 and read noise 3 ADU,
looking at 1 to 26 electrons. The estimator is given the
`SensorEstimate`, and denoises after the Anscombe transform.

```pycon
>>> from optix.estimates import SensorEstimate
>>> from optix.models import simulate_sensor
>>> sensor = SensorEstimate(offset=100.0, variance=9.0, gain=2.0)
>>> rate = 1 + 0.1 * x
>>> z = simulate_sensor(rate, sensor, seed=0)
>>> den = NLMeans().fit(z, init=sensor).apply(z)
>>> truth = 2.0 * rate + 100.0
>>> round(float((z - truth).square().mean().sqrt()), 1)
5.0
>>> round(float((den - truth).square().mean().sqrt()), 1)
3.2
```

Volumes use the extension `ndim=3`, here with $3 \times 3 \times 3$ patches in a
$7 \times 7 \times 7$ window, which is cheaper than the defaults, on a ball in
noise.

```pycon
>>> axes = torch.meshgrid(
...     *[torch.linspace(0, 1, n) for n in (12, 24, 24)], indexing="ij"
... )
>>> ball = 100 * ((sum((a - 0.5) ** 2 for a in axes)) < 0.1).float()
>>> noisy = ball + 15 * torch.randn_like(ball)
>>> den = nl_means(noisy, sigma=15, ndim=3, patch=3, search=7)
>>> bool((den - ball).square().mean() < (noisy - ball).square().mean() / 10)
True
```

Common variations: `layout=` to say which axes are spatial when the array has
frames or channels; `inverse=` to choose another inverse of the Anscombe
transform; a large zarr array in place of the tensor, with `out=` a separate
array; `h=`, `patch=` and `search=` (settings of the paper, not faithful) to
trade smoothing and speed; `bias_correction=True` (extension). A fitted
estimator can be saved and loaded like any other.

### Results on synthetic data

PSNR of the estimates of a synthetic scene of the test suite (cells and
filaments on a smooth background, $128^2$ pixels, range $[0, 255]$, one
realisation of the noise each), in dB, with the default settings. The last
column is the extension `bias_correction`.

| $\sigma$ | Noisy | `"mean"` (default) | `"sum"`, $h = 10\sigma$ | `"mean"`, `bias_correction` |
|---|---|---|---|---|
| 10 | 28.11 | 36.79 | 23.81 | 37.03 |
| 20 | 22.09 | 32.84 | 21.21 | 32.76 |
| 35 | 17.23 | 28.26 | 20.50 | 28.20 |

The effect of $h$ at $\sigma = 20$ (same scene), for the two normalisations;
the noisy image is at 22.09 dB and the mean of the $21 \times 21$ window
(`h=inf`) at 20.13 dB:

| $h / \sigma$ | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| `"mean"` | 22.09 | 27.40 | 32.84 | 27.71 | 23.12 |
| `"sum"` | 30.70 | 23.83 | 21.21 | 20.41 | 20.20 |

The two curves are shifted by the factor of 7 in $h$ explained above: `"sum"`
needs an $h$ near $1.4\sigma$ for what `"mean"` does with $10\sigma$. The
default $10\sigma$ is the best of this grid for `"mean"`. A volume
of $24 \times 48^2$ voxels with $\sigma = 20$ (range $[0, 255]$, $5 \times 5$
patches in an $11 \times 11$ window, so not the paper's settings) gives 22.07 dB
noisy, 32.75 dB denoised plane by plane, 33.91 dB with `ndim=3` (extension) and
36.22 dB with `ndim=3` and `bias_correction=True`.

These numbers describe the synthetic scene only, not natural images. In
`float32` and in `float64` the results agree to $3 \times 10^{-4}$ grey levels
on the same data (a PSNR to $10^{-7}$ dB).

## References

!!! references "References"

    1. Buades A, Coll B, Morel J-M. *A non-local algorithm for image
       denoising.* IEEE Computer Society Conference on Computer Vision and
       Pattern Recognition (CVPR'05) **2**, 2005.
       [doi:10.1109/CVPR.2005.38](https://doi.org/10.1109/CVPR.2005.38)
    2. Mäkitalo M, Foi A. *Optimal inversion of the generalized Anscombe
       transformation for Poisson-Gaussian noise.* IEEE Trans Image Process
       **22**(1):91-103, 2013.
       [doi:10.1109/TIP.2012.2202675](https://doi.org/10.1109/TIP.2012.2202675)

    ??? quote "BibTeX"

        ```bibtex
        @inproceedings{buades2005nonlocal,
          author    = {Buades, Antoni and Coll, Bartomeu and Morel, Jean-Michel},
          title     = {A non-local algorithm for image denoising},
          booktitle = {IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)},
          volume    = {2},
          year      = {2005},
          doi       = {10.1109/CVPR.2005.38},
        }
        @article{makitalo2013optimal,
          author  = {M{\"a}kitalo, Markku and Foi, Alessandro},
          title   = {Optimal inversion of the generalized {A}nscombe transformation for {P}oisson-{G}aussian noise},
          journal = {IEEE Transactions on Image Processing},
          volume  = {22},
          number  = {1},
          pages   = {91--103},
          year    = {2013},
          doi     = {10.1109/TIP.2012.2202675},
        }
        ```
