# Depth-resolved attenuation

!!! info "At a glance"
    **Estimates:** the attenuation coefficient $\mu$ of every pixel along the propagation axis, and the data compensated for it (arrays of the shape of the input, not a result type of `optix.estimates`) \
    **Noise model:** none \
    **Streaming:** halo (blocks that hold whole profiles) \
    **Needs many frames:** no \
    **Reference:** Vermeer et al., Biomedical Optics Express, 2014, [doi:10.1364/BOE.5.000322](https://doi.org/10.1364/BOE.5.000322)

## Overview

`DepthResolved` estimates how strongly the medium attenuates the light at every
pixel of a profile along the propagation axis of the illumination, and undoes
the attenuation that has accumulated up to that pixel. The estimate is the
depth-resolved method of
[Vermeer et al. (2014)](https://doi.org/10.1364/BOE.5.000322), which was
introduced for optical coherence tomography (OCT): $\mu$ at a pixel is the ratio
of the intensity of that pixel to the sum of all the deeper intensities. No
segmentation, reference layer or fit over a depth range is needed, and every
profile is processed on its own.

Use it on light-sheet data in which the excitation light decays along its path
through the specimen, so that the deeper side of every frame is darker. It
assumes that almost all the light is absorbed within the profile and that the
signal at a depth is proportional to the local attenuation: both are
approximations for fluorescence (see
[Image formation](#image-formation-optics)). When the medium is uniform, fitting
an exponential to the mean intensity of every depth is simpler and less
sensitive to noise, as in
[Frantz et al. (2024)](https://doi.org/10.1364/BOE.531210), which is not what
this page implements. The shading of the illumination across the frame is a
different problem: see [BaSiC](../shading/basic.md) and
[CIDRE](../shading/cidre.md).

There is nothing to fit: `fit` only records the layout. `apply` writes the
compensated data and `attenuation` writes the map of $\mu$. Both are in
[`optix.attenuation`](../../reference/attenuation.md), together with the
functions `depth_resolved` (the map of $\mu$) and `compensate` (the compensated
data). The extension `method="smith"` adds the lateral denoising of
[Smith et al. (2015)](https://doi.org/10.1109/TMI.2015.2450197).

## Image formation (optics)

The excitation light crosses the specimen along an axis $z$ and its irradiance
$L$ decays by the Beer–Lambert law, $\mathrm{d}L = -\mu(z)\,L(z)\,\mathrm{d}z$,
where $\mu(z) \ge 0$ is the attenuation coefficient, in the inverse of the
length unit of $z$. A fixed fraction of the attenuated light is turned into the
recorded signal, so that the signal at depth $z$ is proportional to
$\mu(z)\,L(z)$, and the light that is recorded has itself been attenuated on
its way to the detector. For OCT, the light travels to the depth and back along
the same path, which is the origin of a factor 2 in the exponent. The model of
the paper, with $s$ the number of times the light crosses the medium
(`passes`):

=== "optix"

    $$
    y(z) = \xi\,\mu(z)\,\exp\!\Bigl(-s \int_0^z \mu(u)\,\mathrm{d}u\Bigr),
    \qquad s = 2 \text{ in the paper}
    $$

=== "Vermeer et al. 2014"

    $$
    I(z) = \alpha\,\beta\,L_0\,\mu(z)\,\exp\!\Bigl(-2 \int_0^z \mu(u)\,\mathrm{d}u\Bigr)
    \qquad \text{(Eq. 8)}
    $$

| optix | [Vermeer et al. 2014](https://doi.org/10.1364/BOE.5.000322) | Meaning |
|---|---|---|
| $y(z)$, $y_j$ | $I(z)$, $I[i]$ | signal at depth $z$; its value at pixel $j$ |
| $z$ | $z$ | depth along the propagation axis |
| $\mu(z)$, $\mu_j$ | $\mu(z)$, $\mu[i]$ | attenuation coefficient |
| $s$ | 2 | number of crossings of the medium (`passes`) |
| $\xi$ | $\alpha\beta L_0$ | constant: backscattered fraction, conversion factor, incident irradiance |
| $\Delta$ | $\Delta$ | pixel size along the axis (`pixel_size`) |
| $T_j$ | $A$ of pixel $j-1$ ($\sum_{k>j-1} I[k]$) | sum of the intensities from pixel $j$ to the end of the profile |

Eqs. (4) to (8) of the paper derive this model for OCT: the light is
attenuated on the way in (Eq. 5), a fraction $\alpha$ of the attenuated light is
backscattered (Eq. 6), the backscattered light is attenuated again on the way
out (Eq. 7), and $\beta$ converts it to the digital signal (Eq. 8).

Applied to a light sheet, the model reads differently, and the page states
what this changes:

- The constant backscattered fraction becomes a locally constant fluorophore
  emission: the fluorescence of a voxel is a fixed fraction of the excitation
  light attenuated there, so that $y \propto \mu\,L$. A specimen whose
  fluorophore concentration varies is not described by this assumption.
- The excitation crosses the specimen once, which corresponds to $s = 1$. The
  second factor of Eq. (7), the attenuation of the emitted light on its way to
  the detector, is not modelled.
- The pixel index $j$ increases with depth: the light enters at pixel 1.

A pixel holds the integral of the signal over its width, and the signal is
taken as constant over a pixel (Section 2.2.3 of the paper).

## Probabilistic model

The method is not statistical: no noise model is assumed and nothing is
optimised. It rests on four assumptions, the first three from the paper:

1. **Single scattering.** The light is attenuated according to the model above;
   multiply-scattered light is not taken into account.
2. **A fixed fraction is recorded.** The constant $\xi$ does not depend on depth.
3. **All the light is attenuated within the profile.** The signal beyond the last
   pixel is zero, so that $\int_z^\infty y\,\mathrm{d}u$ is the sum of the
   pixels deeper than $z$ (Eq. 13).
4. **Non-negative signal.** The intensities are used as they are, so the
   additive offset of the camera is subtracted first (the option `background`).

The intensities $y_j \ge 0$ of a profile and the pixel size $\Delta$ are the
inputs, and the attenuation coefficients $\mu_j$ the outputs.

The extension `method="smith"` adds a model for the measurement and a prior
for $\boldsymbol{\mu}$. The signal is recorded with additive noise of average
power $\sigma^2$, and $\mu$ is taken piecewise constant across the lateral
axis, which is expressed as a total-variation prior on every row of
the lateral-depth image of $\mu$ (below).

## Inference / derivation

### Estimate of the attenuation coefficient

Differentiating the model gives
$\frac{\mathrm{d}}{\mathrm{d}z}\exp(-s\int_0^z \mu) = -s\,\mu(z)\exp(-s\int_0^z \mu)$,
so $y(z) = -\frac{\xi}{s}\frac{\mathrm{d}}{\mathrm{d}z}\exp(-s\int_0^z \mu)$ and,
since the exponential vanishes at infinity,

$$
\int_z^\infty y(u)\,\mathrm{d}u = \frac{\xi}{s}\exp\!\Bigl(-s\int_0^z \mu\Bigr)
= \frac{y(z)}{s\,\mu(z)},
\qquad
\mu(z) = \frac{y(z)}{s \int_z^\infty y(u)\,\mathrm{d}u}
\approx \frac{y(z)}{s \int_z^D y(u)\,\mathrm{d}u}
$$

(Eqs. 9 to 13 for $s = 2$; $D$ is the depth range of the profile). The
attenuation of a pixel $j$ is the average of $\mu$ over its width, and with $y$
constant over the pixel the integral is exact (Eqs. 14 to 17):

=== "optix"

    $$
    \mu_j = \frac{1}{s\,\Delta}\,\log\!\Bigl(1 + \frac{y_j}{\sum_{k > j} y_k}\Bigr)
    $$

=== "Vermeer et al. 2014"

    $$
    \mu[i] = \frac{1}{2\Delta}\,\log\!\Bigl(1 + \frac{I[i]}{\sum_{k=i+1}^{\infty} I[k]}\Bigr)
    \qquad \text{(Eq. 17)}
    $$

The paper states that this equation was used in all its experiments. It also
gives the first-order linearisation $\mu[i] \approx I[i] / (2\Delta \sum_{k>i} I[k])$
(Eq. 18), which is not used here. The sum runs to the last pixel of the profile,
so that at the last pixel it is empty and $\mu = \infty$; where the intensity
and the deeper sum both vanish, $\mu$ is undefined (NaN). Before the end, $\mu$
is overestimated whenever light remains in the profile (Section 3.1 and Fig. 1
of the paper).

### Compensation

The paper estimates $\mu$ and gives no compensated intensity. The correction
below is derived from its model and is not part of it. The attenuation that
has accumulated in front of pixel $j$ multiplies the signal by
$\exp(-s\Delta\sum_{i<j}\mu_i)$, so the compensated signal is

$$
y^{\mathrm{c}}_j = y_j \,\gamma_j,
\qquad
\gamma_j = \exp\!\Bigl(s\,\Delta \sum_{i<j} \mu_i\Bigr).
$$

With $T_j = \sum_{k \ge j} y_k$, Eq. (17) says $\exp(s\Delta\mu_i) = T_i / T_{i+1}$.
The product over $i < j$ telescopes, and the gain is

$$
\gamma_j = \frac{T_1}{T_j},
\qquad
y^{\mathrm{c}}_j = y_j\,\frac{T_1}{T_j},
$$

which is what `DepthResolved.apply` computes: no exponential and no cumulated
logarithm, and neither $s$ nor $\Delta$ appears. Where $T_j \le 0$ (no signal
from pixel $j$ on) or $T_1 \le 0$, the gain is 0. The compensated signal is proportional to
$\xi\,\mu_j$, the local emission under the locally constant emission
assumption, so a medium with a constant $\mu$ gives a flat profile.

### Extension: lateral denoising (Smith et al. 2015)

[Smith et al. (2015)](https://doi.org/10.1109/TMI.2015.2450197) build on the
model of Vermeer et al. and add a restoration filter for the noise, the
estimate of Eq. (10) and a lateral denoising of $\mu$. In optix notation, with
the confocal and sensitivity fall-off functions $f$ and $h$ of the paper set to 1
(the factor $f h$ in Eq. 9):

=== "optix"

    $$
    y^{\mathrm{r}}_j = y_j\,\frac{y_j^2}{y_j^2 + \sigma^2},
    \qquad
    \hat{\mu}_j = \frac{y^{\mathrm{r}}_j}{s\,\Delta \sum_{k \ge j} y^{\mathrm{r}}_k}
    $$

    $$
    \tilde{\boldsymbol{\mu}}_j^\star =
    \operatorname*{arg\,min}_{\tilde{\boldsymbol{\mu}} \,\ge\, 0}\;
    \tfrac{1}{2}\,\lVert \tilde{\boldsymbol{\mu}} - \hat{\boldsymbol{\mu}}_j \rVert_2^2
    + \frac{2}{s}\,\frac{\eta}{Q\,\Delta}\,
      \sum_{q=1}^{Q-1} w_{q}\,\bigl|\tilde{\mu}_{q+1} - \tilde{\mu}_{q}\bigr|,
    \qquad
    w_q = \frac{1}{y_{j,q} + \epsilon}
    $$

=== "Smith et al. 2015"

    $$
    \hat{\boldsymbol{I}}[m] = \boldsymbol{I}_{\text{meas}}[m]\,\boldsymbol{H}_{\text{res}}[m],
    \qquad
    \boldsymbol{H}_{\text{res}}[m] =
      \frac{1}{\hat{\boldsymbol{f}}[m]\,\hat{\boldsymbol{h}}[m]}\,
      \frac{|\boldsymbol{I}_{\text{meas}}[m]|^2}{|\boldsymbol{I}_{\text{meas}}[m]|^2 + |\boldsymbol{c}[m]|^2}
    \quad \text{(Eqs. 8, 9)}
    $$

    $$
    \hat{\boldsymbol{\mu}}[m] \approx
      \frac{\hat{\boldsymbol{I}}[m]}{2\Delta z \sum_{r=m}^{M} \hat{\boldsymbol{I}}[r]}
    \quad \text{(Eq. 10)}
    $$

    $$
    \underset{\check{\boldsymbol{\mu}}_m \succeq 0}{\text{minimise}}\;
    \tfrac{1}{2}\lVert \check{\boldsymbol{\mu}}_m - \hat{\boldsymbol{\mu}}_m \rVert_2^2
    + \eta\,\mathrm{Reg}(\check{\boldsymbol{\mu}}_m),
    \quad
    \mathrm{Reg} = \frac{1}{N_x \Delta z}\,
      \bigl\lVert \boldsymbol{D}\check{\boldsymbol{\mu}}_m^T \bigr\rVert_{\boldsymbol{X}_m \boldsymbol{W}_m,1},
    \quad
    \boldsymbol{W}_m[n] = \frac{1}{\boldsymbol{I}_{\text{meas}}[m,n] + \epsilon}
    \quad \text{(Eqs. 11, 14, 15)}
    $$

| optix | [Smith et al. 2015](https://doi.org/10.1109/TMI.2015.2450197) | Meaning |
|---|---|---|
| $y_j$ | $\boldsymbol{I}_{\text{meas}}[m]$ | measured intensity at depth index $j$ ($m$) |
| $y^{\mathrm{r}}_j$ | $\hat{\boldsymbol{I}}[m]$ | intensity after the restoration filter |
| $y^{\mathrm{c}}_j$ | none | compensated intensity (derived above, not in the paper) |
| $\sigma^2$ | $\lvert\boldsymbol{c}[m]\rvert^2$ | average noise power (`noise_power`) |
| $s$ | 2 | crossings of the medium (`passes`) |
| $\Delta$ | $\Delta z$ | pixel size along the propagation axis |
| $\hat{\mu}_j$, $\tilde{\mu}_q$ | $\hat{\boldsymbol{\mu}}[m]$, $\check{\boldsymbol{\mu}}_m[n]$ | estimate; denoised estimate |
| $q$, $Q$ | $n$, $N$ | lateral position and their number |
| $\eta$ | $\eta$ | regularisation weight (`eta`) |
| $\epsilon$ | $\epsilon$ | floor of the weights (`epsilon`) |
| $w_q$ | $\boldsymbol{W}_m[n]$ | weight of the total variation |

The denoising acts on every row of constant depth $j$ of the lateral-depth image
of $\hat{\mu}$, along a lateral axis (`tv_axis`). The weight is small where the
intensity is high, so the total variation smooths the deep, noisy regions more
than the bright ones. The problem is convex and is solved exactly, line by
line, by the direct dynamic-programming solver of
[Johnson (2013)](https://doi.org/10.1080/10618600.2012.681238) for the
weighted total variation, at a cost $O(Q)$ per line, followed by clamping at 0:
in one dimension, $\operatorname{prox}_{\mathrm{TV} + \iota_{\ge 0}} =
\max(\operatorname{prox}_{\mathrm{TV}}, 0)$, so the clamped solution is exact
for the constrained problem. Smith et al. use CVX.

The weight of the total variation is $\frac{2}{s}\frac{\eta}{Q\Delta}$ rather
than the paper's $\frac{\eta}{Q\Delta}$. The two are equal at $s = 2$. Since
$\hat{\mu}$ is proportional to $1/s$, the factor $2/s$ makes the denoised
$\tilde{\mu}$, and hence the compensation, independent of $s$.
$\eta$ and $\epsilon$ are tied to the intensity units of Smith et al.: the
weight $w_q$ scales as the inverse of the intensity, so that multiplying the data
by a factor $c$ divides the strength of the smoothing by about $c$. Values chosen
for intensities of another magnitude must be rescaled.

Choices that the paper leaves open, or where the code departs from it:

- The factor $f h$ of Eq. (9) is 1, and the noise power $\sigma^2$ is 0 unless
  given: the restoration filter is then the identity.
- There is no tissue mask: $\boldsymbol{X}_m$ is the identity, so that
  $N_x = Q$ (Eq. 15).
- The paper's $\boldsymbol{W}_m$ has $N$ entries and $\boldsymbol{D}$ has $N-1$
  rows. Here $w_q$ weights the difference between positions $q+1$ and $q$.
- Negative intensities count as 0 in the weights.
- After a background subtraction, noise can make the tail sum of $y^{\mathrm{r}}$
  small or negative near the end of a profile. Then $\hat{\mu} = 0$ where the
  sum is $\le 0$; the gain is 0 where the tail sum at $j$ or at the first pixel is
  $\le 0$ (as for Vermeer); and the compensated data are kept finite, within the
  range of the output dtype.
- In three dimensions the total variation runs along one lateral axis only,
  chosen with `tv_axis`.

## Algorithm

```text
Input: data y with the propagation axis z, pixel size Δ, passes s,
       optional background, method, max_gain
Subtract the background, if given
for every profile along z:
    Vermeer (default):
        T_j = sum of y_k for k ≥ j              (float64)
        μ_j = log(1 + y_j / (T_j − y_j)) / (s Δ)
        γ_j = T_1 / T_j                         (0 where T_j ≤ 0 or T_1 ≤ 0)
    Smith (method="smith"):
        y^r = y · y² / (y² + σ²)
        μ̂_j = y^r_j / (s Δ sum of y^r_k for k ≥ j)  (0 where the sum is ≤ 0)
        for every depth j, over the lateral axis:
            μ̃_j = argmin ½‖μ̃ − μ̂_j‖² + (2/s) η/(QΔ) sum_q w_q |μ̃_{q+1} − μ̃_q|, μ̃ ≥ 0
        γ_j = exp(s Δ sum of μ̃_i for i < j)    (0 where the tail sum of y^r
                                                  at j or at 1 is ≤ 0)
    γ_j = min(γ_j, max_gain), if given
Return  μ  (attenuation)   or   y^c_j = γ_j · y_j  (apply), kept finite
```

Defaults: `passes=2`, `pixel_size=1`, `eta=6`, `epsilon=1e-3`, `noise_power=0`.

## Faithful mode and extensions

`faithful=True` (the default) is the estimate of Eq. (17) with $s = 2$, and
`is_faithful` is true for it. The axis, the pixel size and `tv_axis` are free
parameters of the data, and any value of them is faithful;
`eta`, `epsilon` and `noise_power` only matter with `method="smith"`.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `passes` | $s$, the number of crossings in the exponent; $\mu$ is proportional to $1/s$ | 2 | 1 (light sheet) |
| `method` | `"smith"`: restoration filter and lateral TV denoising of Smith et al. (2015); extension | `"vermeer"` | `"vermeer"` |
| `background` | an additive term subtracted first: a number, a tensor, the `offset` of a `SensorEstimate` or the `dark` of a `ShadingEstimate`; extension | `None` | `None` |
| `max_gain` | $\ge 1$, caps the compensation gain, to limit the amplification of the noise deep in the profile; extension | `None` | `None` |

`faithful=False` differs from `faithful=True` by `passes` only: it uses the single-crossing value of a light sheet, and gives the Vermeer estimate otherwise.

Choices that the paper leaves open:

- **The compensated intensity** is derived here from the model (above); the paper
  gives only $\mu$.
- **The end of the profile** is as the equation is written: the sum is empty at
  the last pixel and $\mu = \infty$ there. Cropping the profile where the signal
  has vanished avoids the overestimate.
- **The intensities** must be non-negative. The paper subtracts the noise floor
  and divides by the sensitivity roll-off of the system before the estimate
  (Eq. 19); the subtraction is available with `background`, which is off by
  default, and the roll-off is not modelled.
- **Precision.** The sums are accumulated in float64 (float32 on MPS)
  whatever the dtype of the data.

!!! note "The default number of passes"
    The paper sets $s = 2$ because the light of an OCT system crosses the tissue
    twice. In a light sheet, the excitation crosses the specimen once, which
    corresponds to `passes=1`. Two choices for the default are possible:
    `passes=2` is the value of the paper and keeps the default faithful, and
    `passes=1` is the single-pass value of the light-sheet model above. The
    default is 2; `faithful=False` selects 1.

    The choice changes the values of $\mu$ and nothing else for the default
    method: with `passes=2`, $\mu$ is half the single-pass coefficient, and with
    `passes=1` it is the coefficient of a single crossing. The compensated data
    is the same for both, with either method: the gain $T_1 / T_j$ does not
    contain $s$, and with `method="smith"` the weight $2/s$ of the
    total-variation term cancels the factor $1/s$ of $\hat{\mu}$. The compensation
    does not depend on $\Delta$ either. A value other than 2 makes `is_faithful`
    false, even though it is the physical value for a light sheet.

## Memory and streaming

`DepthResolved` is tagged `halo`. `apply` and `attenuation` read blocks that
contain the whole propagation axis (and, with `method="smith"`, the whole of
`tv_axis`) and tile the other axes: every profile is independent, so there is no
overlap between blocks and the result is the same as for a whole-array
computation. Each block is read once and its result written to `out`, so
arrays larger than memory (zarr, memmap) are processed in one pass. The block
size follows the memory budget of `optix.config`: the free axes are halved until
a block fits, and a warning is issued when a single profile (or plane, for
Smith) exceeds the budget.

The cost of the Vermeer estimate is linear in the number of pixels: a reversed
cumulative sum and a logarithm. The Smith extension adds the exact
total-variation solver, linear in the length of the lateral axis for every row,
and `eta=0` skips it. On one CPU thread, a $5 \times 64 \times 300$ B-scan takes
about 0.13 s, and the solver alone takes about 8 s on a $2048^2$ plane. Smith
blocks hold more intermediate arrays than Vermeer blocks, so they are smaller for
the same budget.

## Usage

A synthetic stack in which the light of a single crossing decays through a
uniform medium, with Poisson noise, laid out as frames, rows and depth:

```pycon
>>> import torch
>>> from optix.attenuation import DepthResolved, compensate, depth_resolved
>>> from optix.io import Layout
>>> _ = torch.manual_seed(0)
>>> depth = torch.arange(300.0)
>>> light = torch.exp(-0.02 * depth)
>>> data = torch.poisson(2000 * light * torch.ones(5, 64, 1))  # (frames, y, z)
>>> layout = Layout("fyz", spatial="yz", sample="f")
```

`passes=1` corresponds to a single crossing, and the estimate recovers the
coefficient $0.02$ used to build the data, away from the end of the profile:

```pycon
>>> est = DepthResolved(axis="z", passes=1, layout=layout).fit()
>>> mu = est.attenuation(data)
>>> mu.shape
torch.Size([5, 64, 300])
>>> round(float(mu[..., :150].mean()), 2)
0.02
>>> est.is_faithful
False
```

The compensated frames have a flat profile:

```pycon
>>> corrected = est.apply(data)
>>> top = corrected[..., 10:40].mean()
>>> middle = corrected[..., 110:140].mean()
>>> bool((middle / top - 1).abs() < 0.1)
True
```

Common variations:

```python
# The paper's number of passes (the default), and the functions.
mu = depth_resolved(data[0, 0], pixel_size=0.004)  # one profile, in 1/mm
flat = compensate(data, axis=-1)

# Subtract the camera offset first, and cap the gain deep in the profile.
est = DepthResolved(axis="z", background=100.0, max_gain=20.0, layout=layout)

# Lateral denoising of Smith et al. (2015), with the lateral axis named.
est = DepthResolved(axis="z", method="smith", tv_axis="y", layout=layout)
```

For a large zarr array, pass it to `apply` with `out=` set to a zarr array. A
`SensorEstimate` or a `ShadingEstimate` given to `background` is placed with the
layout of the estimator, so that its map follows the spatial axes of the data.
Saving an estimator whose `background` is an estimate is not supported.

Validation: on profiles simulated from the model, constant, layered and ramped
$\mu$ are recovered to $10^{-9}$ (when the truncation at the end of the profile
is negligible); the end of the profile follows Eq. (17) as written; the
compensation flattens a uniform emitter; float32 agrees with float64. The
Smith denoising reduces the lateral spread of $\mu$ by more than 40 % on noisy
simulated B-scans, and the TV solution is checked for optimality.

## References

!!! references "References"

    1. Vermeer KA, Mo J, Weda JJA, Lemij HG, de Boer JF. *Depth-resolved
       model-based reconstruction of attenuation coefficients in optical
       coherence tomography.* Biomed Opt Express **5**(1):322-337, 2014.
       [doi:10.1364/BOE.5.000322](https://doi.org/10.1364/BOE.5.000322)
    2. Smith GT, Dwork N, O'Connor D, Sikora U, Lurie KL, Pauly JM, Ellerbee AK.
       *Automated, depth-resolved estimation of the attenuation coefficient from
       optical coherence tomography data.* IEEE Trans Med Imaging
       **34**(12):2592-2602, 2015.
       [doi:10.1109/TMI.2015.2450197](https://doi.org/10.1109/TMI.2015.2450197)
    3. Johnson NA. *A dynamic programming algorithm for the fused lasso and
       L0-segmentation.* J Comput Graph Stat **22**(2):246-260, 2013.
       [doi:10.1080/10618600.2012.681238](https://doi.org/10.1080/10618600.2012.681238)
    4. Frantz D, Wright CJ, Schaser AJ, Kirik D, Kristensson E, Berrocal E.
       *Image post-processing for SILMAS: structured illumination light sheet
       microscopy with axial sweeping.* Biomed Opt Express
       **15**(8):4943-4962, 2024.
       [doi:10.1364/BOE.531210](https://doi.org/10.1364/BOE.531210)

    ??? quote "BibTeX"

        ```bibtex
        @article{vermeer2014depth,
          author  = {Vermeer, Koenraad A. and Mo, Jelena and Weda, Jelmer J. A. and Lemij, Hans G. and de Boer, Johannes F.},
          title   = {Depth-resolved model-based reconstruction of attenuation coefficients in optical coherence tomography},
          journal = {Biomedical Optics Express},
          volume  = {5},
          number  = {1},
          pages   = {322--337},
          year    = {2014},
          doi     = {10.1364/BOE.5.000322},
        }
        @article{smith2015automated,
          author  = {Smith, Gennifer T. and Dwork, Nicholas and O'Connor, Daniel and Sikora, Uzair and Lurie, Kristen L. and Pauly, John M. and Ellerbee, Audrey K.},
          title   = {Automated, depth-resolved estimation of the attenuation coefficient from optical coherence tomography data},
          journal = {IEEE Transactions on Medical Imaging},
          volume  = {34},
          number  = {12},
          pages   = {2592--2602},
          year    = {2015},
          doi     = {10.1109/TMI.2015.2450197},
        }
        @article{johnson2013dynamic,
          author  = {Johnson, Nicholas A.},
          title   = {A dynamic programming algorithm for the fused lasso and {L0}-segmentation},
          journal = {Journal of Computational and Graphical Statistics},
          volume  = {22},
          number  = {2},
          pages   = {246--260},
          year    = {2013},
          doi     = {10.1080/10618600.2012.681238},
        }
        @article{frantz2024image,
          author  = {Frantz, David and Wright, Courtney J. and Schaser, Allison J. and Kirik, Deniz and Kristensson, Elias and Berrocal, Edouard},
          title   = {Image post-processing for {SILMAS}: structured illumination light sheet microscopy with axial sweeping},
          journal = {Biomedical Optics Express},
          volume  = {15},
          number  = {8},
          pages   = {4943--4962},
          year    = {2024},
          doi     = {10.1364/BOE.531210},
        }
        ```
