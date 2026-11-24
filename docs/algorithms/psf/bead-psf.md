# Bead PSF

!!! info "At a glance"
    - **Estimates:** `PSFEstimate` (`psf`, `fwhm`, `count`, `centers`)
    - **Noise model:** Poisson-Gaussian (only through robust thresholds; no likelihood)
    - **Streaming:** halo
    - **Needs many frames:** no (many beads, in one or several images)
    - **Reference:** none; a standard procedure without a single reference publication

## Overview

`BeadPSF` measures the point-spread function from images of sub-resolution
fluorescent beads. Each bead is a near-point source, so its image is a noisy,
shifted copy of the PSF; averaging many isolated beads, each aligned to
sub-voxel precision, gives a low-noise empirical PSF. Use it when a bead
acquisition with the same optics as the specimen is available; the result, in
`estimates["psf"].psf`, can be given directly to the
[deconvolution](../../reference/deconv.md) estimators. There is no `apply`.
It works in 1, 2 or 3 spatial dimensions and gives one PSF per group (for
instance per channel); sample axes, such as several bead acquisitions, are
pooled.

## Image formation (optics)

A bead $n$ much smaller than the resolution is a point of brightness $a_n$ at a
sub-voxel position $\mathbf{c}_n$. Its image, cropped to a window of half-size
`radius` around its peak, is the PSF shifted to $\mathbf{c}_n$, plus a locally
constant background and noise (see
[Image formation and noise](../../concepts/image-formation.md)):

=== "optix"

    $$
    \mathbf{y}_n = a_n\, \mathbf{S}_{\mathbf{c}_n} \mathbf{h} + b_n + \boldsymbol{\varepsilon}_n ,
    $$

    with $\mathbf{y}_n, \mathbf{h} \in \mathbb{R}^{K}$ crops of $K$ voxels
    (shape `2·radius + 1` per spatial axis, $d$ axes), $\mathbf{c}_n \in \mathbb{R}^d$,
    and $\mathbf{S}_{\mathbf{c}} \in \mathbb{R}^{K\times K}$ the shift that
    moves the centre of $\mathbf{h}$ (index `K // 2`) to $\mathbf{c}$.

=== "optix docstring"

    $$
    \mathbf{h} = N\Bigl(\sum_n w_n\, T_n(\mathbf{y}_n - b_n)\Bigr)
    $$

    with $T_n$ the sub-voxel shift that moves the centre $c_n$ to index
    `K // 2` and $N$ the post-processing and normalisation.

| optix | optix docstring | Meaning | Shape |
|---|---|---|---|
| $\mathbf{y}_n$ | $y_n$ | crop around bead $n$ | $\mathbb{R}^K$, per bead |
| $\mathbf{h}$ | $h$ | PSF, unit sum | $\mathbb{R}^K$, one per group |
| $a_n$ | | bead brightness | scalar, per bead |
| $b_n$ | $b_n$ | local background | scalar, per bead |
| $\mathbf{c}_n$ | $c_n$ | sub-voxel centre | $\mathbb{R}^d$, per bead |
| $\boldsymbol{\varepsilon}_n$ | | noise | $\mathbb{R}^K$, per bead |
| $\mathbf{S}_{\mathbf{c}_n}^{-1}$ | $T_n$ | alignment shift | $\mathbb{R}^{K\times K}$, per bead |
| $w_n$ | $w_n$ | averaging weight | scalar, per bead |
| | $N$ | post-processing and normalisation | one per group |

There is no paper, so the second tab gives the notation of the class
docstring. The background $b_n$ is a scalar per bead, broadcast over its crop.

## Probabilistic model

Assumptions: the beads are much smaller than the resolution (their size is
neglected; the PSF is not deconvolved by the bead shape); they are isolated,
so that no other bead lies within a crop; the PSF is the same for all beads
in a group (shift-invariant over the field and depth sampled); the background
is constant over a crop; no bead is saturated. Noise is not modelled
explicitly: it enters only through the detection threshold, set from a robust
estimate of its level, and is reduced by averaging. The estimate is the
(weighted) mean of the aligned, background-subtracted crops,

$$
\hat{\mathbf{h}} \propto \sum_n w_n\, \mathbf{S}_{\mathbf{c}_n}^{-1}(\mathbf{y}_n - b_n),
\qquad
w_n = \begin{cases}
1 / \mathbf{1}^\top(\mathbf{y}_n - b_n) & \text{equal weights,}\\
1 & \text{brightness weights,}
\end{cases}
$$

which is unbiased for $\mathbf{h}$ when the centres and backgrounds are exact.
With brightness weights it is the least-squares (Gaussian maximum-likelihood)
estimate of $\mathbf{h}$ given the aligned crops, up to normalisation; the
centres are themselves least-squares (Gaussian-fit) or method-of-moments
(centroid) estimates, and the background median and MAD threshold are robust
M-estimates. The pipeline is a plug-in estimator: each nuisance parameter is
fixed at its estimate rather than marginalised.

## Inference / derivation

The strategy is detect, reject, register, average:

1. **Detection.** The data are smoothed with a Gaussian (`sigma`), and the
   noise level of the smoothed data is estimated robustly, per group, from a
   random sample of $10^5$ values: median $\mu$ and
   $\sigma = 1.4826\,\mathrm{MAD}$. Candidates are local maxima within a
   window of half-width `min_distance` that exceed $\mu + t\sigma$
   (`threshold` $t$).
2. **Rejection** of candidates whose crop leaves the data, contains another
   candidate, reaches the `saturation` level, whose fitted centre lies more
   than 2 voxels from the detected peak, or whose background-subtracted sum
   is not positive.
3. **Background** $b_n$: the median of the voxels on the faces of the crop
   (`"border"`), or a given constant.
4. **Registration.** The sub-voxel centre $\mathbf{c}_n$ is estimated by a
   least-squares fit of an axis-aligned Gaussian plus a constant
   (`"gaussian"`) or by the centroid of the part of the crop above a quarter
   of its peak (`"centroid"`); the crop is then shifted so that
   $\mathbf{c}_n$ falls on the central voxel, by a Fourier phase ramp
   (`"fourier"`: exact for band-limited data, circular on the crop) or by
   separable linear interpolation (`"linear"`: slightly smoothing).
5. **Averaging and post-processing.** The aligned crops are summed, each
   normalised to unit sum first (`"equal"`) or as they are
   (`"brightness"`); the sum is optionally mirror-symmetrised along chosen
   axes and tapered by a Tukey window, clipped at 0 and normalised to unit
   sum. The FWHM along each axis is reported as a quality check.

## Algorithm

**Input:** bead images; `radius = 8`; `sigma = 1`; `threshold` $t = 5$;
`min_distance = 2` \
**Output:** `PSFEstimate(psf, fwhm, count, centers)`, one PSF
$\hat{\mathbf{h}} \in \mathbb{R}^K$ per group.

1. **Noise level**, per group: median $\mu$ and $\sigma = 1.4826\,\mathrm{MAD}$
   of a sample of $10^5$ values of the smoothed data $G_{\texttt{sigma}} * \mathbf{y}$.
2. **For each block** (with a halo covering a crop and its neighbours):
    1. **detect** the local maxima of $G_{\texttt{sigma}} * \mathbf{y}$ in
       windows of half-width `min_distance` above $\mu + t\sigma$;
    2. **keep** the peaks in the block core whose crop fits, holds no other
       peak and is not saturated;
    3. **for each bead** $n$:
        1. background $b_n$: median of the crop faces;
        2. centre $\mathbf{c}_n$: Gaussian fit (or centroid) of
           $\mathbf{y}_n - b_n$; reject if more than 2 voxels from the
           central voxel;
        3. align: $\mathbf{z}_n = \mathbf{S}_{\mathbf{c}_n}^{-1}(\mathbf{y}_n - b_n)$;
           reject if $\mathbf{1}^\top\mathbf{z}_n \le 0$;
        4. add $\mathbf{z}_n / \mathbf{1}^\top\mathbf{z}_n$ (equal) or
           $\mathbf{z}_n$ (brightness) to the group sum;
    4. stop collecting a group once `max_beads` beads are kept.
3. **Post-process**, per group: symmetrise, taper, clip at 0, normalise to
   unit sum; measure the FWHM.

Fails with an error when a group has no accepted bead.

## Faithful mode and extensions

There is no reference publication, so `faithful` has no effect: both presets
are the defaults below, each a standard choice.

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `radius` | crop half-size per spatial axis; PSF shape `2·radius + 1` | `8` | `8` |
| `sigma` | std (voxels) of the detection smoothing | `1.0` | `1.0` |
| `threshold` | detection threshold, in noise std above the median | `5.0` | `5.0` |
| `min_distance` | half-width of the local-maximum window | `2` | `2` |
| `saturation` | rejection level | `None` (dtype max for integers, no test for floats) | `None` |
| `background` | `"border"` or a constant | `"border"` | `"border"` |
| `center` | `"gaussian"` or `"centroid"` | `"gaussian"` | `"gaussian"` |
| `shift` | `"fourier"` or `"linear"` | `"fourier"` | `"fourier"` |
| `weights` | `"equal"` or `"brightness"` | `"equal"` | `"equal"` |
| `symmetrize` | axes of mirror symmetrisation | `()` | `()` |
| `taper` | Tukey taper fraction, 0 for none | `0.0` | `0.0` |
| `max_beads` | beads kept per group | `None` (all) | `None` |
| `block` | core block size; does not change the result | `None` (memory budget) | `None` |

## Memory and streaming

The data are read block by block, twice: once to sample the noise level (a
reservoir of $10^5$ values per group), once to detect and align beads. The
halo is wide enough to see every crop and its neighbours, and a bead belongs
to the block whose core holds its peak, so the result does not depend on
`block`. Only one running sum of shape `2·radius + 1` per group, and the bead
centres, are kept in memory. Cost scales with the size of the data plus the
number of beads times the cost of one fit and one shift.

## Usage

The example simulates a [phantom](../../reference/models.md#phantoms) of beads imaged with a Gaussian PSF and
averages the isolated ones.

```python
import torch
from optix.models import phantoms
from optix.psf import BeadPSF

torch.manual_seed(0)

# 1. 40 beads imaged with a Gaussian PSF (sigma 1.5 pixels), on a background.
x, _ = phantoms.beads((256, 256), sigma=1.5, brightness=7000.0, seed=1)
y = torch.poisson(x + 20)

# 2. Fit: average the isolated beads.
psf = BeadPSF(radius=6).fit(y).estimates["psf"]

# 3. Compare to the truth: FWHM 3.56 and 3.53 pixels (expected 2.355 · 1.5 ≈ 3.53).
print(int(psf.count), psf.psf.shape, psf.fwhm.tolist())
```

Common variations: a 3D bead stack (the trailing three dims are spatial);
`symmetrize=(1, 2)` for a laterally symmetric PSF; `weights="brightness"`;
the shortcut `bead_psf(y, radius=6)`, which returns the PSF tensor. See the
[API reference](../../reference/psf.md).

## References

No reference publication: bead averaging is a standard procedure assembled
from common choices, each of which is an option above.
