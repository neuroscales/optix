# Rolling ball

!!! info "At a glance"
    **Estimates:** the background of an image or volume, and the image minus it \
    **Noise model:** none \
    **Streaming:** halo \
    **Needs many frames:** no \
    **Reference:** Sternberg, Computer, 1983, [doi:10.1109/MC.1983.1654163](https://doi.org/10.1109/MC.1983.1654163)

## Overview

`RollingBall` estimates a slowly varying background and subtracts it, with the
background normalisation of
[Sternberg (1983)](https://doi.org/10.1109/MC.1983.1654163): the image is read
as a landscape, and the background is the surface traced by the top of a ball
rolled underneath it. The ball follows the smooth background but cannot enter
peaks narrower than itself, so the peaks are kept in the difference. `TopHat`
does the same with a flat disk or square instead of a ball.

Use it when the specimen is made of features smaller than the scale of the
background, such as spots on an uneven or drifting background, and no stack of
frames is available to fit a shading model. The two sizes are the only
parameters: the radius must exceed the half-width of the largest feature to
keep, and stay below the scale of the background variations. When many frames
share one illumination pattern, [BaSiC](../shading/basic.md) or
[CIDRE](../shading/cidre.md) estimate it from the data instead.

There is nothing to fit: `fit` only records the layout, and `apply` returns the
corrected frames, or the background with `output="background"`. The result is
an image of the shape of the input, not a result type of
[`optix.estimates`](../../reference/estimates.md). See
[`optix.background`](../../reference/background.md) for the API.

## Image formation (optics)

The model is the additive background of
[Image formation and noise](../../concepts/image-formation.md), with no
assumption on the camera or the illumination beyond their smoothness:

=== "optix"

    $$
    \mathbf{y} = \mathbf{x} + \boldsymbol{\beta}
    $$

=== "Sternberg 1983"

    $$
    A(x, y) = \text{spots}(x, y) + \text{background}(x, y)
    $$

The image $\mathbf{y}$ is the sum of the features $\mathbf{x} \ge 0$ and a
background $\boldsymbol{\beta}$ that is smooth with respect to the ball while
the features are not ([Sternberg 1983, p. 29](https://doi.org/10.1109/MC.1983.1654163)).
The paper does not write this sum as an equation; it describes the gel image as
spots on a background that "is not uniform over a gel and varies between gels".
The ball is built in units where one pixel and one grey level have the same
length, so the radius below is both a distance in the image and a height in
intensity.

## Probabilistic model

The method is a morphological operator, not a statistical model. Its objective
is the **opening** of the image by a structuring element, which Sternberg
describes through the *umbra* of the image: the solid under the surface
$(\mathbf{r}, \mathbf{y}(\mathbf{r}))$, a pile of cubic "boxels" at every pixel.
The opening is the union of all translations of the element that fit entirely
inside the umbra, and its upper surface is the background.

=== "optix"

    $$
    \begin{aligned}
    (\mathbf{y} \ominus B)(\mathbf{r}) &= \min_{\mathbf{u}}
        \bigl[\, y(\mathbf{r} + \mathbf{u}) - B(\mathbf{u}) \,\bigr], \\
    (\mathbf{z} \oplus B)(\mathbf{r}) &= \max_{\mathbf{u}}
        \bigl[\, z(\mathbf{r} - \mathbf{u}) + B(\mathbf{u}) \,\bigr], \\
    \boldsymbol{\beta} &= (\mathbf{y} \ominus B) \oplus B,
    \qquad \hat{\mathbf{x}} = \mathbf{y} - \boldsymbol{\beta}.
    \end{aligned}
    $$

=== "Sternberg 1983"

    $$
    \begin{aligned}
    C(x, y) &= \min_{i, j}\,\bigl[\, A(x - i,\, y - j) - B_k(-i, -j) \,\bigr], \\
    D(x, y) &= \max_{i, j}\,\bigl[\, C(x - i,\, y - j) + B_k(i, j) \,\bigr],
    \end{aligned}
    $$

    with $C = A \ominus B_k$ and $D = (A \ominus B) \oplus B = A_B$ (p. 30);
    the background image is $D$ and the result is $A - D$ (Fig. 9).

| optix | [Sternberg 1983](https://doi.org/10.1109/MC.1983.1654163) |
|---|---|
| $\mathbf{y}$, $y(\mathbf{r})$ | $A$, $A(x, y)$ |
| $\mathbf{r}$ | $(x, y)$ |
| $\mathbf{u}$ | $(-i, -j)$ in the erosion, $(i, j)$ in the dilation |
| $B(\mathbf{u})$ | $B_k(i, j)$, the grey level of the element |
| $\mathbf{y} \ominus B$ | $C = A \ominus B_k$ |
| $\boldsymbol{\beta}$ | $D = A_B$, the "background image" |
| $\hat{\mathbf{x}}$ | background-normalised image |
| $\rho$ | not named: half the "sphere diameter" of the paper (p. 29), as $\rho$ is a radius |
| $B$ | $B = (B_1 \oplus \cdots \oplus B_K)$, the composite element of the sequence $B_1, \ldots, B_K$ |

The paper builds its element as a sequence of small elements $B_1, \ldots, B_K$,
applied one after the other, and relies on the identities

$$
(A \ominus B_1) \ominus B_2 = A \ominus (B_1 \oplus B_2),
\qquad
B = [(B_1 \oplus B_2) \oplus B_3] \oplus \cdots
$$

so that the sequence acts as one erosion, then one dilation, by the composite
element $B$ (p. 27 and 30). Here $B$ is given directly, and the sequence is not
used. The substitution $\mathbf{u} \to -\mathbf{u}$ turns the paper's erosion
into the form above, which adds the offset of the element to the image. Both are
a minimum (or maximum) over the same set of offsets only because the support of
the ball is symmetric ($\mathbf{u}$ and $-\mathbf{u}$ are both in it, with the
same height), so that reflecting the window changes nothing.

**The ball.** With radius $\rho$, the umbra of the ball is the element

$$
B(\mathbf{u}) = \sqrt{\rho^2 - \lVert \mathbf{u} \rVert^2} - \rho,
\qquad \lVert \mathbf{u} \rVert \le \rho,
$$

on the integer offsets $\mathbf{u}$, and $-\infty$ elsewhere (those offsets are left out). The shift by $-\rho$ makes $B(\mathbf{0}) = 0$; an opening does not
depend on it. For an image ($n = 2$) the ball has three dimensions, the two of
the image plane and the grey level; for a volume ($n = 3$) it has four.

**The flat element.** For a flat element $S$ ($B = 0$ on $S$), the same formulas
reduce to a minimum then a maximum filter over $S$, which is the top-hat of
mathematical morphology:

$$
\boldsymbol{\beta} = (\mathbf{y} \ominus S) \oplus S,
\qquad \hat{\mathbf{x}} = \mathbf{y} - \boldsymbol{\beta}.
$$

!!! note "What the paper says about flat elements"
    The paper illustrates opening with flat elements on binary images (Figs. 6
    and 7) and defines the grey-scale version with any element $B_k$, so the flat
    case is the same equations with $B = 0$. It does not name the top-hat, and
    it uses a ball for the background.

## Inference / derivation

There is no estimation: the background follows from the image by the two
filters above. The reasoning of the paper (Fig. 10) is geometric:

- The erosion $\mathbf{y} \ominus B$ is the height of the apex of the ball (the top
  of its umbra, where $B = 0$) as it is pushed up against the underside of the
  surface at every position.
- The dilation by the same element sweeps the ball over those apex positions and takes
  the top of the ball. The result is the surface touched by the ball's top while
  it rolls under the image.
- The opening is **anti-extensive**: $\boldsymbol{\beta} \le \mathbf{y}$, so
  $\hat{\mathbf{x}} \ge 0$. Peaks narrower than the ball are cut off by the
  opening, and survive in $\hat{\mathbf{x}}$; slow variations are followed by the
  ball and removed.

The radius is a trade-off that the paper leaves to the user: the ball must be
"considerably wider than any of the peaks", and "small enough to follow the
smooth contours of the changing background".

The result at a pixel depends only on the image within $2\lfloor\rho\rfloor$
pixels, since the erosion and the dilation each reach $\lfloor\rho\rfloor$ (one more
with `smooth=True`). This
is the halo that makes block-wise processing exact (see
[Memory and streaming](#memory-and-streaming)).

## Algorithm

```text
Input: image or volume y, radius ρ > 0 (ball) or half-width k (flat element)
Build the element B on the integer offsets u, ‖u‖ ≤ ρ  (or the disk / square of k)
1. c(r) = min over u of [ y(r + u) − B(u) ]        (erosion)
2. β(r) = max over u of [ c(r − u) + B(u) ]        (dilation)
3. β = min(β, y)
Return y − β  (or β with output="background")
```

Offsets that fall outside the image are left out of the minimum and the maximum.
Step 3 changes nothing in exact arithmetic, as the opening never exceeds $\mathbf{y}$;
it removes the rounding error of $(\mathbf{y} - B) + B$ in floating point, so that
the result is never negative.

## Faithful mode and extensions

`faithful=True` (the default) is the opening of the paper. The radius and the
output are free parameters of the method, so any value of them is faithful.
`TopHat` has no extension: it is always faithful.

Choices that the code makes where the paper is silent:

- **Edges.** The paper does not say what happens at the border. Out-of-image
  offsets are ignored (equivalent to $+\infty$ for the erosion and $-\infty$
  for the dilation), so the ball can overhang the edge.
- **Exact ball.** The paper builds its ball as the dilation of one point by 26
  small elements of a $3 \times 3 \times 3$ window (Fig. 11), a hardware
  approximation. Here $B$ is sampled from the analytic formula above and the
  opening is exact for it; the 26-element construction is not reproduced.
- **Units.** A pixel and a grey level are the same length (the boxels are
  cubes), so $\rho$ is in pixels and in intensity units. Rescale the intensities
  if the ball should be flatter or taller.
- **Dimension.** The number of spatial axes of the layout (up to three) sets the
  dimension of the ball; the paper treats only images.

Extensions, off by default, make the estimate faster or less sensitive to noise
and are **not** in the paper. They follow the public descriptions of the
background subtraction of ImageJ and do not reproduce its output bit for bit:

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `shrink` | $s \ge 1$: take the minimum over cells of $s^n$ pixels, open that coarse image with the ellipsoid of lateral radius $\rho / s$ and height $\rho$, then interpolate the background linearly from the cell centres back to the pixel grid. The cost drops by about $s^{2n}$ | 1 | 1 |
| `smooth` | open the $3^n$ mean of the image (edges replicated) instead of the image; the result is still the image minus the background | `False` | `False` |

With either, the background approximates the opening and is capped at the image.
Sternberg's streak removal, by openings with a horizontal and a vertical bar, is
not implemented.

!!! note "scikit-image"
    `skimage.restoration.rolling_ball` returns the erosion $\mathbf{y} \ominus B$,
    the height of the apex of the ball, not the opening. It matches
    `optix.background.utils.grey_erosion` to $10^{-12}$, and the background here is
    the dilation of its output.

## Memory and streaming

`RollingBall` and `TopHat` are tagged `halo`. `apply` reads blocks of the spatial
axes with an overlap of $2\lfloor\rho\rfloor$ pixels, filters each block and
stores its core, so zarr and memmap inputs that exceed memory give the same
result as a whole-array computation, and several frames or channels are
independent. The halo is available as
[`rolling_ball_halo`](../../reference/background.md). In-place output is
refused, since a block reads pixels that other blocks have written.

With `shrink=s`, blocks start on multiples of $s$ so that the cells lie on the
global grid, and the halo grows to $s\,(2\lfloor\rho/s\rfloor + 2 + [\text{smooth}])$, a
multiple of $s$ that covers the coarse opening, the cells and the interpolation
($[\text{smooth}]$ is 1 with `smooth=True` and 0 otherwise).

The cost of the exact ball is $O(N \rho^n)$ for $N$ pixels, one operation per
offset of the ball. On one CPU thread, a $1024^2$ image takes about 0.4 s with
$\rho = 10$ and 11 s with $\rho = 50$; `shrink=4` brings the latter to 0.03 s. A
$128^3$ volume takes about 12 s with $\rho = 10$. The flat disk and square are
computed with running minima and maxima over one axis at a time, and are much
cheaper.

## Usage

```pycon
>>> import torch
>>> from optix.background import RollingBall, rolling_ball
>>> r, c = torch.meshgrid(torch.arange(128.0), torch.arange(128.0), indexing="ij")
>>> background = 20 + 10 * torch.sin(c / 40) + 5 * torch.cos(r / 30)
>>> spots = sum(
...     60 * torch.exp(-((r - cy) ** 2 + (c - cx) ** 2) / 18)
...     for cy, cx in [(30, 40), (80, 90), (100, 25)]
... )
>>> data = background + spots

>>> corrected = rolling_ball(data, radius=15)
>>> estimate = rolling_ball(data, radius=15, output="background")
>>> bool((corrected >= 0).all())
True
>>> bool((estimate - background).abs().mean() < 0.5)
True
```

For a stack, the estimator filters every frame and returns the background with
`output="background"`:

```pycon
>>> stack = data[None] + torch.zeros(4, 1, 1)  # (frames, H, W)
>>> ball = RollingBall(radius=15, output="background").fit(stack)
>>> ball.is_faithful
True
>>> bool(torch.equal(ball.apply(stack)[0], estimate))
True
```

Common variations:

```python
from optix.background import TopHat

# A flat square element, and the fast approximation of the ball.
corrected = TopHat(radius=15, shape="square").fit(stack).apply(stack)
fast = RollingBall(radius=15, shrink=4).fit(stack).apply(stack)  # not faithful

# A volume: a 4-D ball over the last three axes.
volume = rolling_ball(torch.rand(32, 64, 64), radius=5, ndim=3)
```

For a large zarr array, pass it to `apply` with `out=` set to a zarr array; the
filter reads and writes in blocks. See the
[API reference](../../reference/background.md) for every option.

Validation: the morphology is bit-identical to a brute-force transcription of the
formulas above in 2-D and 3-D, in float32 and float64; spots on a smooth
background in synthetic scenes are recovered; block-wise and whole-image results
are identical (to $10^{-10}$ with `shrink`), and zarr input gives the same
result as in-memory input.

## References

!!! references "References"

    1. Sternberg SR. *Biomedical image processing.* Computer
       **16**(1):22-34, 1983.
       [doi:10.1109/MC.1983.1654163](https://doi.org/10.1109/MC.1983.1654163)

    ??? quote "BibTeX"

        ```bibtex
        @article{sternberg1983biomedical,
          author  = {Sternberg, Stanley R.},
          title   = {Biomedical image processing},
          journal = {Computer},
          volume  = {16},
          number  = {1},
          pages   = {22--34},
          year    = {1983},
          doi     = {10.1109/MC.1983.1654163},
        }
        ```
