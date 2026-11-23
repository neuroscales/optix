# Wiener–Butterworth

!!! info "At a glance"
    **Estimates:** deconvolved image (the latent image $\mathbf{x}$, same shape as the data) \
    **Noise model:** Poisson \
    **Streaming:** halo \
    **Needs many frames:** no \
    **Reference:** Guo et al., Nat Biotechnol, 2020, [doi:10.1038/s41587-020-0560-x](https://doi.org/10.1038/s41587-020-0560-x)

## Overview

`WienerButterworth` is [Richardson–Lucy](richardson-lucy.md) deconvolution with
an *unmatched back projector* ([Guo et al. 2020](https://doi.org/10.1038/s41587-020-0560-x)):
the transposed PSF of the update is replaced by a filter that inverts the
optical transfer function over its pass band and cuts off sharply beyond it.
One or a few iterations then do the work of tens of Richardson–Lucy
iterations. Use it when speed matters (large 3D stacks, many views) and a
small number of iterations is acceptable; use plain Richardson–Lucy, or
[Biggs–Andrews](biggs-andrews.md), when a maximum-likelihood estimate with
guaranteed convergence is wanted.

## Image formation (optics)

As for [Richardson–Lucy](richardson-lucy.md#image-formation-optics): a linear,
shift-invariant blur $\mathbf{A}$ by a known PSF $\mathbf{h}$ of unit sum, with
the latent image estimated on a domain extended by the kernel support (the
paper uses a periodic model; see
[Image formation and noise](../../concepts/image-formation.md)).

=== "optix"

    $$
    \mathbb{E}[\mathbf{y}] = \mathbf{A}\mathbf{x} = \mathbf{h} * \mathbf{x}
    $$

=== "Guo et al. 2020"

    $$
    \mathbb{E}[y] = H e = f * e
    $$

| optix | [Guo et al. 2020](https://doi.org/10.1038/s41587-020-0560-x) |
|---|---|
| $\mathbf{y}$ (recorded image) | $y$ |
| $\mathbf{x}$ (latent image) | $e$ |
| $\mathbf{h}$, $\mathbf{A}$ (PSF, blur) | $f$ (forward projector), $H$ |
| $\hat{\mathbf{h}} = \mathrm{DFT}(\mathbf{h})$ (OTF) | $F$ |

The paper's $f$ is the PSF, not the flat field of optix.

## Probabilistic model

The model is that of Richardson–Lucy, $\mathbf{y} \sim
\mathrm{Poisson}(\mathbf{A}\mathbf{x})$ with $\mathbf{x} \ge 0$, but it only
motivates the iteration: with an unmatched back projector the scheme no longer
maximises this likelihood. Assumptions: a known, shift-invariant PSF whose
resolution (FWHM per axis) sets the cutoff of the back projector, and data in
a regime where a few iterations suffice, so that the noise amplification that
the sharper back projector brings is kept in check by stopping early rather
than by a prior.

## Inference / derivation

The strategy is to keep the multiplicative form of the Richardson–Lucy update
and change only the back projection, after
[Zeng and Gullberg (2000)](https://doi.org/10.1109/42.870265). The product of
the forward and back transfer functions governs how fast each spatial
frequency is restored: with the matched back projector it is $|F|^2$, small at
high frequencies, which is why Richardson–Lucy is slow. A back projector whose
transfer function makes the product close to 1 over the pass band, and close
to 0 beyond it, restores all passed frequencies in a few steps without
boosting the noise outside the band.

=== "optix"

    $$
    \mathbf{x}_{k+1} = \frac{\mathbf{x}_k}{\mathbf{B}\mathbf{1}} \odot
    \mathbf{B}\,\frac{\mathbf{y}}{\mathbf{A}\mathbf{x}_k},
    \qquad
    \mathrm{DFT}(\mathbf{h}_b) \propto
    \frac{\hat{\mathbf{h}}^*}{|\hat{\mathbf{h}}|^2 + \alpha}
    \cdot
    \frac{1}{\sqrt{1 + \epsilon^2 \bigl[\sum_a (k_a / k_{c,a})^2\bigr]^{n}}},
    $$

    with $\mathbf{B}$ the convolution with the back-projector kernel
    $\mathbf{h}_b$ of unit sum, $k_{c,a} = 1/\mathrm{FWHM}_a$ of the PSF,
    and $\epsilon^2 = 1/\beta^2 - 1$.

=== "Guo et al. 2020"

    $$
    e^{(k+1)} = e^{(k)} \cdot \frac{B\bigl[y / (H e^{(k)})\bigr]}{B 1},
    \qquad
    B_{\mathrm{WB}} = \frac{F^*}{F^2 + \alpha}\cdot
    \frac{1}{\sqrt{1 + \epsilon^2 (k/k_c)^{2n}}}
    \quad \text{(Suppl. Note 2, Eq. 25)}.
    $$

| optix | [Guo et al. 2020](https://doi.org/10.1038/s41587-020-0560-x) |
|---|---|
| $\mathbf{x}_k$ | $e^{(k)}$ |
| $\mathbf{h}_b$, $\mathbf{B}$ (back projector) | $b$, $B$ |
| $\hat{\mathbf{h}}$ | $F$ |
| $\alpha$ (Wiener parameter) | $\alpha$ |
| $\beta$ (gain at the cutoff) | $\beta$ |
| $n$ (Butterworth order) | $n$ |
| $k_{c,a}$ | $k_c$ |

The first factor is a Wiener inverse of the OTF; the second a Butterworth
low-pass filter whose gain at the cutoff is $\beta$. optix uses $|F|^2$ where
the paper writes $F^2$ (the same for a real, symmetric OTF). Without the
Poisson-likelihood structure, the iteration is not EM: the likelihood need not
increase, flux need not be conserved and convergence is not guaranteed. The
kernel has negative lobes, so negative values of the update are set to 0 at
each iteration, as in the paper. `back_projector=` also gives the paper's
Butterworth and Gaussian back projectors, and `"traditional"` (the flipped
PSF), which is plain Richardson–Lucy.

## Algorithm

```text
Input: data Y, PSF h, back_projector = "wiener-butterworth",
       α = 0.001, β = 0.001, n = 8, n_iter = 1, start = "data"
k_c = 1 / FWHM(h) per axis (or 1 / fwhm)
h_b = DFT⁻¹ of the WB filter, sampled on a 2× grid, cropped to 12 FWHM per axis, unit sum
Initialise x(0) = y with values < 0.001 set to 0.001;
           outside the image (margin of the object domain): mean of y
Repeat n_iter times:
    x_{k+1} = x_k / (B 1) ⊙ B[y / (A x_k)],  negative values set to 0
Return x, cropped to the observed region
```

The defaults are those of Supplementary Table S2.1 (iSIM and confocal data);
the paper uses $\alpha, \beta \in [0.001, 0.05]$, $n \in [4, 10]$ and 1–5
iterations, and recommends a single iteration as a rule of thumb. On the
margin of the object domain, outside the image, the matched back projection
$\mathbf{A}^\top$ is used (not in the paper).

## Faithful mode and extensions

| Keyword | Changes | Faithful value | Recommended value |
|---|---|---|---|
| `back_projector` | `"wiener-butterworth"`, `"butterworth"`, `"gaussian"` or `"traditional"` | `"wiener-butterworth"` | `"wiener-butterworth"` |
| `wiener` | Wiener parameter $\alpha$ | `0.001` | `0.001` |
| `cutoff_gain` | Butterworth gain $\beta$ at the cutoff | `0.001` | `0.001` |
| `order` | Butterworth order $n$ | `8` | `8` |
| `fwhm` | resolution per axis; cutoff $1/\mathrm{FWHM}$ | `None` (measured on the PSF) | `None` |
| `start` | `"data"` (Methods) or `"uniform"`: the first iteration is then the linear filter $\mathbf{y} * \mathbf{h}_b$ | `"data"` | `"data"` |
| `n_iter` | number of iterations | `1` | `1` |
| `chi2_stop` | stop with Lucy's χ² test instead of after `n_iter` | `False` | `False` |
| `alpha` | level of the χ² test (with `chi2_stop`) | `0.05` | `0.05` |
| `max_iter` | iteration cap of the χ² rule | `100` | `100` |
| `damping` | damping of the update, as in Richardson–Lucy | `1.0` | `1.0` |
| `halo` | halo per axis, at least the kernel support | `None` (twice the support) | `None` |
| `block` | core block size per spatial axis | `None` (memory budget) | `None` |

`alpha` is the χ² level, not the Wiener parameter (`wiener`). The cutoff
$k_c = 1/\mathrm{FWHM}$ follows the paper's text; its Supplementary
Fig. S2.3 corresponds to a cutoff about 1.3 times higher, obtained by passing
a smaller `fwhm`. The camera model of Richardson–Lucy (`fit(y, init=sensor)`)
also applies.

## Memory and streaming

As for [Richardson–Lucy](richardson-lucy.md#memory-and-streaming): each block,
with a halo of at least the kernel support, runs its iterations on its own in
one pass over the data. The back-projector kernel rings over many PSF widths
and is kept on 12 FWHM per axis; the PSF is zero-padded to that support, so
the FFT grid of each block, and the halo, are set by this support rather than
by the PSF. With the default single iteration the cost is about that of one
Richardson–Lucy iteration.

## Usage

```python
import torch
from optix.deconv import RichardsonLucy, WienerButterworth
from optix.deconv.utils import PaddedConvolution

torch.manual_seed(0)

# 1. A Gaussian PSF and a blurred, Poisson-noisy test pattern.
r = torch.arange(-7, 8, dtype=torch.float32)
psf = torch.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * 2.0**2))
psf = psf / psf.sum()
x = torch.full((96, 96), 5.0)
x[20:76:8, 20:76] = 200.0
op = PaddedConvolution(psf, (96, 96))
y = torch.poisson(op.forward(op.pad(x)))

# 2. One WB iteration, against 20 Richardson–Lucy iterations.
wb = WienerButterworth(psf).fit(y)
x_wb = wb.apply(y)
x_rl = RichardsonLucy(psf, n_iter=20).fit(y).apply(y)

# 3. The back-projector kernel, and the errors against the truth.
print(tuple(wb.kernel().shape))
print(float((x_wb - x).abs().mean()), float((x_rl - x).abs().mean()))
```

Common variations: `n_iter=2` or more for a sharper result; a smaller `fwhm`
for a higher cutoff; `back_projector="gaussian"` or `"butterworth"` for the
paper's other filters. See the [API reference](../../reference/deconv.md).

## References

!!! references "References"

    1. Guo M, Li Y, Su Y, Lambert T, Dalle Nogare D, Moyle MW, Duncan LH,
       Ikegami R, Santella A, Rey-Suarez I, Green D, Beiriger A, Chen J,
       Vishwasrao H, Ganesan S, Prince V, Waters JC, Annunziata CM, Hafner M,
       Mohler WA, Chitnis AB, Upadhyaya A, Usdin TB, Bao Z, Colón-Ramos D,
       La Riviere P, Liu H, Wu Y, Shroff H. *Rapid image deconvolution and
       multiview fusion for optical microscopy.* Nat Biotechnol
       **38**(11):1337-1346, 2020.
       [doi:10.1038/s41587-020-0560-x](https://doi.org/10.1038/s41587-020-0560-x)
    2. Zeng GL, Gullberg GT. *Unmatched projector/backprojector pairs in an
       iterative reconstruction algorithm.* IEEE Trans Med Imaging
       **19**(5):548-555, 2000.
       [doi:10.1109/42.870265](https://doi.org/10.1109/42.870265)

    ??? quote "BibTeX"

        ```bibtex
        @article{guo2020rapid,
          author  = {Guo, Min and Li, Yue and Su, Yijun and Lambert, Talley and
                     Dalle Nogare, Damian and Moyle, Mark W. and Duncan, Leighton H.
                     and Ikegami, Richard and Santella, Anthony and Rey-Suarez, Ivan
                     and Green, Daniel and Beiriger, Anastasia and Chen, Jiji and
                     Vishwasrao, Harshad and Ganesan, Sundar and Prince, Victoria
                     and Waters, Jennifer C. and Annunziata, Christina M. and
                     Hafner, Markus and Mohler, William A. and Chitnis, Ajay B. and
                     Upadhyaya, Arpita and Usdin, Ted B. and Bao, Zhirong and
                     Col{\'o}n-Ramos, Daniel and La Riviere, Patrick and Liu, Huafeng
                     and Wu, Yicong and Shroff, Hari},
          title   = {Rapid image deconvolution and multiview fusion for optical microscopy},
          journal = {Nature Biotechnology},
          volume  = {38},
          number  = {11},
          pages   = {1337--1346},
          year    = {2020},
          doi     = {10.1038/s41587-020-0560-x},
        }
        @article{zeng2000unmatched,
          author  = {Zeng, Gengsheng L. and Gullberg, Grant T.},
          title   = {Unmatched projector/backprojector pairs in an iterative reconstruction algorithm},
          journal = {IEEE Transactions on Medical Imaging},
          volume  = {19},
          number  = {5},
          pages   = {548--555},
          year    = {2000},
          doi     = {10.1109/42.870265},
        }
        ```
