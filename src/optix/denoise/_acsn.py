"""ACsN: automatic correction of sCMOS-related noise (Mandracchia et al. 2020).

Written from the paper and its Supplementary Information. Details the
paper leaves open (the fit of the noise level, the intensity mapping and
the optional hot-pixel removal) follow the authors' MATLAB code,
github.com/ShuJiaLab/ACsN, released under the BSD 3-Clause licence:

    Copyright (c) 2019, Biagio Mandracchia. All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met: (1) redistributions of source code must retain the above
    copyright notice, this list of conditions and the following
    disclaimer; (2) redistributions in binary form must reproduce the
    above copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided with
    the distribution; (3) neither the name of the copyright holder nor
    the names of its contributors may be used to endorse or promote
    products derived from this software without specific prior written
    permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
    CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE
    DISCLAIMED; IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
    BE LIABLE FOR ANY DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE.

The sparse filter of that code is a compiled third-party BM3D and was
not used: the collaborative filtering runs on the BM3D engine of
`optix.denoise`.
"""

import math
from collections.abc import Callable, Sequence
from typing import Literal, Self

import numpy as np
import torch
from torch import Tensor

from .._config import resolve_backend
from .._estimator import Estimator, Tags
from ..estimates import SensorEstimate
from ..io import (
    ArrayLike,
    Labeled,
    Layout,
    WritableArrayLike,
    blocks,
    prepare_out,
)
from ._bm3d import _check_int, _check_positive, _select
from ._engine import Stage, block_overhead, run_stage, stage_halo

__all__ = [
    "ACsN",
    "acsn",
    "acsn_noise",
    "acsn_profile",
    "highpass",
    "noise_ratio",
    "otf_cutoff",
    "quality_score",
    "sigma_from_highpass",
    "temporal_psnr",
]

Ratio = Literal["1d", "area"]
SigmaFit = Literal["histogram", "std"]


# -- noise estimation ---------------------------------------------------------
def otf_cutoff(na: float, wavelength: float) -> float:
    """Cutoff frequency f_c = 2·NA/λ of the incoherent OTF.

    The radius of the optical transfer function of a fluorescence
    microscope (Supplementary Note 2.3 of Mandracchia et al. 2020:
    "calculate the radius of the OTF" from NA and the emission
    wavelength), in cycles per unit of `wavelength`.

    Parameters
    ----------
    na : float
        Numerical aperture.
    wavelength : float
        Emission wavelength λ (the unit of length of `pixel_size`).

    Returns
    -------
    float
    """
    return (
        2.0
        * _check_positive("na", na)
        / _check_positive("wavelength", wavelength)
    )


def _radial_frequency(
    shape: Sequence[int], pixel_size: float, like: Tensor
) -> Tensor:
    """|f| of the DFT frequencies of an (H, W) frame, cycles per unit."""
    fy = torch.fft.fftfreq(
        shape[0], d=pixel_size, dtype=like.dtype, device=like.device
    )
    fx = torch.fft.fftfreq(
        shape[1], d=pixel_size, dtype=like.dtype, device=like.device
    )
    return torch.sqrt(fy[:, None].square() + fx.square())


def highpass(x: Tensor, pixel_size: float, cutoff: float) -> Tensor:
    """High-pass filter that keeps the frequencies outside the OTF.

    h = F⁻¹[F(x)·𝟙(|f| > f_c)] on the DFT frequencies of each frame
    (Supplementary Note 2.3 and Supplementary Fig. 12b: "the threshold
    of the filter at the OTF boundary"). The ideal (step) filter is that
    of the ACsN code; the paper does not specify the filter's shape.

    Parameters
    ----------
    x : (..., H, W) Tensor
    pixel_size : float
        Pixel size in the image plane (camera pixel / magnification).
    cutoff : float
        f_c, in cycles per unit of `pixel_size`.

    Returns
    -------
    (..., H, W) Tensor
        Real high-pass image.
    """
    mask = _radial_frequency(x.shape[-2:], pixel_size, x) > cutoff
    return torch.fft.ifft2(torch.fft.fft2(x) * mask).real


def noise_ratio(
    shape: Sequence[int],
    pixel_size: float,
    cutoff: float,
    *,
    ratio: Ratio = "1d",
    camera_cutoff: float = 1.0,
) -> float:
    """Ratio σ_N/σ_c of the noise to the noise of the high-pass image.

    `"1d"` (the paper): white noise of power N₀ per unit bandwidth,
    recorded up to the camera cutoff f̄, has σ_N² = ∫₀^f̄ N₀ df (Eqs.
    (6)-(8) of Supplementary Note 2.3); the part above f_c has
    σ_c² = ∫_{f_c}^f̄ N₀ df (Eq. (9)), hence (Eqs. (10)-(11))

        σ_N = √(f̄ / (f̄ − f_c)) · σ_c.

    f̄ is the Nyquist frequency 1/(2Δ) times `camera_cutoff` (the paper
    calls f̄ the cutoff of the camera; the ACsN code uses 1.1/(2Δ)).

    `"area"` (extension): the same argument over the 2-D DFT
    frequencies of the frame, where white noise has the same power at
    every frequency: σ_N² = σ_c²·n/n_c, with n the number of
    frequencies and n_c the number above f_c. This is exact for white
    noise; the 1-D ratio is larger (√5 against 1.42 for f_c = 0.8 f̄).

    Parameters
    ----------
    shape : (H, W)
        Frame shape.
    pixel_size, cutoff : float
        As in [`highpass`][optix.denoise.utils.highpass].
    ratio : {"1d", "area"}
    camera_cutoff : float
        f̄ in units of the Nyquist frequency (`"1d"` only).

    Returns
    -------
    float
    """
    if ratio == "1d":
        fbar = _check_positive("camera_cutoff", camera_cutoff) / (
            2.0 * pixel_size
        )
        if cutoff >= fbar:
            raise ValueError(
                f"The OTF cutoff {cutoff:.4g} is not below the camera "
                f"cutoff {fbar:.4g}: the image is not sampled finely "
                "enough to estimate the noise outside the OTF"
            )
        return math.sqrt(fbar / (fbar - cutoff))
    if ratio == "area":
        f = _radial_frequency(shape, pixel_size, torch.empty(0))
        outside = int((f > cutoff).sum())
        if outside == 0:
            raise ValueError("No frequency of the frame lies outside the OTF")
        return math.sqrt(f.numel() / outside)
    raise ValueError(f"ratio must be '1d' or 'area', got {ratio!r}")


def _matlab_hist(x: Tensor, bins: int = 10) -> tuple[Tensor, Tensor]:
    """Counts and centres of `bins` equal bins between min and max."""
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    centres = lo + width * (torch.arange(bins, dtype=torch.float64) + 0.5)
    counts = torch.histc(x.double().cpu(), bins=bins, min=lo, max=hi)
    return counts, centres


def _gauss_fit(
    x: Tensor, y: Tensor, a0: float, a1: float, iterations: int = 200
) -> float:
    """Least-squares fit of y ≈ a0·exp(−x²/(2a1²)) (Levenberg-Marquardt)."""
    p = torch.tensor([a0, a1], dtype=torch.float64)

    def residual(p: Tensor) -> Tensor:
        return p[0] * torch.exp(-0.5 * (x / p[1]).square()) - y

    damping = 1e-3
    cost = float(residual(p).square().sum())
    for _ in range(iterations):
        e = torch.exp(-0.5 * (x / p[1]).square())
        jac = torch.stack([e, p[0] * e * x.square() / p[1] ** 3], dim=1)
        r = residual(p)
        jtj = jac.T @ jac
        grad = jac.T @ r
        while True:
            step = torch.linalg.solve(
                jtj + damping * torch.diag(jtj.diagonal()), -grad
            )
            trial = p + step
            new = float(residual(trial).square().sum())
            if math.isfinite(new) and new <= cost:
                damping = max(damping / 10, 1e-12)
                break
            damping *= 10
            if damping > 1e12:
                return abs(float(p[1]))
        converged = abs(cost - new) <= 1e-12 * max(cost, 1e-300)
        p, cost = trial, new
        if converged:
            break
    return abs(float(p[1]))


def sigma_from_highpass(h: Tensor, method: SigmaFit = "histogram") -> float:
    """Noise standard deviation σ_c of a high-pass image.

    `"histogram"`: the ACsN code's reading of "the intensity
    distribution of the pixel values in a high-pass filtered image"
    (Supplementary Fig. 12b): the magnitudes |h| are binned in 10 equal
    bins between their minimum and maximum, and a0·exp(−x²/(2σ_c²)) is
    fitted to the counts at the bin centres by nonlinear least squares,
    starting from a0 = the largest count and σ_c = the centre of bin
    round(k/2), k the (1-based) bin of the smallest count. `"std"`
    (extension): the root mean square of h (its mean is 0).

    Parameters
    ----------
    h : Tensor
        High-pass image (zero mean).
    method : {"histogram", "std"}

    Returns
    -------
    float
    """
    if method == "std":
        return float(h.double().square().mean().sqrt())
    if method != "histogram":
        raise ValueError(
            f"sigma_fit must be 'histogram' or 'std', got {method!r}"
        )
    counts, centres = _matlab_hist(h.abs().reshape(-1))
    first_min = int(torch.argmin(counts)) + 1  # 1-based, first minimum
    start = float(centres[max(math.floor(first_min / 2 + 0.5), 1) - 1])
    return _gauss_fit(centres, counts, float(counts.max()), start)


def _median3(x: Tensor) -> Tensor:
    """3 × 3 median with replicated borders (MATLAB's medfilt2 on a
    replicate-padded image, then cropped)."""
    p = torch.nn.functional.pad(x[None, None], (1, 1, 1, 1), mode="replicate")
    win = p[0, 0].unfold(0, 3, 1).unfold(1, 3, 1).reshape(*x.shape, 9)
    return win.median(-1).values


def _remove_hotspots(m: Tensor, pixel_size: float, cutoff: float) -> Tensor:
    """Hot-pixel removal of the ACsN code (not in the paper).

    Pixels whose high-pass magnitude |h| exceeds mean(|h|) + 3·sd(|h|)
    are replaced by the 3 × 3 median of the frame.
    """
    a = highpass(m, pixel_size, cutoff).abs()
    hot = a > (a.mean() + 3 * a.std()).abs()
    return torch.where(hot, _median3(m), m)


def acsn_noise(
    m: Tensor,
    pixel_size: float,
    cutoff: float,
    *,
    ratio: Ratio = "1d",
    sigma_fit: SigmaFit = "histogram",
    camera_cutoff: float = 1.0,
) -> float:
    """Noise level σ_N of one frame, from its frequencies outside the OTF.

    σ_N = ρ·σ_c, σ_c the noise of the high-pass image
    ([`highpass`][optix.denoise.utils.highpass],
    [`sigma_from_highpass`][optix.denoise.utils.sigma_from_highpass]) and
    ρ the ratio of [`noise_ratio`][optix.denoise.utils.noise_ratio]
    (Supplementary Eq. (11)). The noise, readout plus photon shot noise
    approximated as Gaussian (Supplementary Eq. (5)), is assumed white,
    and the signal confined within the OTF. For noise whose variance
    varies over the frame, σ_N² is the mean variance.

    Parameters
    ----------
    m : (H, W) Tensor
        Frame corrected for offset and gain (Supplementary Eq. (4)).
    pixel_size, cutoff : float
        As in [`highpass`][optix.denoise.utils.highpass].
    ratio, camera_cutoff
        As in [`noise_ratio`][optix.denoise.utils.noise_ratio].
    sigma_fit : {"histogram", "std"}
        As in [`sigma_from_highpass`][optix.denoise.utils.sigma_from_highpass].

    Returns
    -------
    float
        σ_N, in the units of `m`.
    """
    rho = noise_ratio(
        m.shape[-2:],
        pixel_size,
        cutoff,
        ratio=ratio,
        camera_cutoff=camera_cutoff,
    )
    return rho * sigma_from_highpass(
        highpass(m, pixel_size, cutoff), sigma_fit
    )


# -- sparse filtering ---------------------------------------------------------
def acsn_profile() -> tuple[Stage, Stage]:
    """Stages of the ACsN sparse filter, in 8-bit units.

    From Supplementary Notes 2.4 and 3.1 and Supplementary Fig. 12c:
    k × k = 8 × 8 patches, an m × m = 19 × 19 search window, groups of
    at most 8 patches, reference patches every 3 pixels, the pixel
    distance d(P, Q) = ‖P − Q‖₂²/k² (Eq. (12)) with τ_match = 2500, hard
    threshold λ_γ·σ_N with λ_γ = 4.0 (Eq. (14)) and aggregation weights
    1/N_P (Eq. (15)); then the same grouping on the intermediate image
    and empirical Wiener shrinkage (Eqs. (16)-(17)).

    Not specified in the paper (taken from BM3D's normal profile, which
    the paper cites): the transforms (bior1.5 then DCT for the patches,
    Haar along the group), τ_match = 400 and the weights
    σ⁻²‖W‖₂⁻² of the Wiener stage. The paper's aggregation has no
    Kaiser window, and none is used.

    Returns
    -------
    (Stage, Stage)
        Hard-thresholding and Wiener stages.
    """
    ht = Stage(
        "bior1.5",
        8,
        8,
        3,
        19,
        2500.0,
        4.0,
        pixel_distance=True,
        sigma_weight=False,
    )
    wie = Stage("dct", 8, 8, 3, 19, 400.0, None, strict=True)
    return ht, wie


def _halo() -> int:
    return sum(stage_halo(s) for s in acsn_profile())


def _sparse_filter(
    m: Tensor,
    sigma: float,
    block: Sequence[int | None] | None = None,
) -> Tensor:
    """Both stages on one frame, in blocks with a halo.

    The frame is mapped to [0, 255] by its minimum and maximum (as in
    the ACsN code), with σ scaled alike.
    """
    lo, hi = float(m.min()), float(m.max())
    if not hi > lo or sigma <= 0:
        return m.clone()
    scale = 255.0 / (hi - lo)
    x = (m - lo) * scale
    s = sigma * scale
    ht, wie = acsn_profile()
    if any(n < ht.size for n in x.shape):
        raise ValueError(f"Frames of {tuple(x.shape)} are smaller than 8 × 8")
    out = torch.empty_like(x)
    chunks = blocks(
        x,
        block,
        _halo(),
        overhead=block_overhead(ht.size**2),
        contiguous=True,
        dtype=x.dtype,
        device=x.device,
    )
    for chunk in chunks:
        origin = tuple(
            core.start - inner.start
            for core, inner in zip(chunk.spatial, chunk.inner, strict=True)
        )
        y = chunk.data
        basic = run_stage(y, s, ht, origin=origin)
        final = run_stage(y, s, wie, pilot=basic, origin=origin)
        chunk.store(out, final[chunk.inner])
    return out / scale + lo


def _frame(
    z: Tensor,
    offset: Tensor | float,
    gain: Tensor | float,
    pixel_size: float,
    cutoff: float,
    o: dict,
) -> tuple[Tensor, float, Tensor]:
    """ACsN on one frame in ADU: (denoised M, σ_N, M)."""
    m = (z - offset) / gain  # Supplementary Eq. (4)
    if o["hotspot"]:
        m = _remove_hotspots(m, pixel_size, cutoff)
    sigma = o["sigma"]
    if sigma is None:
        sigma = acsn_noise(
            m,
            pixel_size,
            cutoff,
            ratio=o["ratio"],
            sigma_fit=o["sigma_fit"],
            camera_cutoff=o["camera_cutoff"],
        )
    return _sparse_filter(m, sigma, o["block"]), sigma, m


def _check_options(o: dict) -> None:
    if o["ratio"] not in ("1d", "area"):
        raise ValueError(f"ratio must be '1d' or 'area', got {o['ratio']!r}")
    if o["sigma_fit"] not in ("histogram", "std"):
        raise ValueError(
            f"sigma_fit must be 'histogram' or 'std', got {o['sigma_fit']!r}"
        )
    o["camera_cutoff"] = _check_positive("camera_cutoff", o["camera_cutoff"])
    if o["sigma"] is not None:
        o["sigma"] = _check_positive("sigma", o["sigma"])
    if o["block"] is not None:
        o["block"] = [_check_int("block", b) for b in o["block"]]


def acsn(
    data: Tensor,
    na: float,
    wavelength: float,
    pixel_size: float,
    *,
    offset: Tensor | float = 0.0,
    gain: Tensor | float = 1.0,
    sigma: float | None = None,
    ratio: Ratio = "1d",
    sigma_fit: SigmaFit = "histogram",
    camera_cutoff: float = 1.0,
    hotspot: bool = False,
    block: Sequence[int | None] | None = None,
) -> Tensor:
    """ACsN denoising of sCMOS frames (Mandracchia et al. 2020).

    Each frame is processed independently, in three steps
    (Supplementary Note 2, Table 5):

    1. **Fixed-pattern noise.** With the camera model
       Z_p = γ_p·{Pois(S_p) + 𝒩(0, σ_R)} + β_p (Eq. (1); Supplementary
       Eq. (3)), the offset map β and the gain map γ (ADU per electron),
       from a one-time calibration, give
       M_p = (Z_p − β_p)/γ_p (Supplementary Eq. (4)), in electrons.
    2. **Noise estimation.** The readout and shot noise of M are
       modelled as one white Gaussian noise of variance
       σ_N² = σ_R² + σ_G² (Supplementary Eq. (5)). The signal lies
       within the OTF, of radius f_c = 2·NA/λ, so the high-pass image
       h = F⁻¹[F(M)·𝟙(|f| > f_c)] holds noise only. Its standard
       deviation σ_c gives σ_N = √(f̄/(f̄ − f_c))·σ_c, f̄ the camera
       cutoff (Supplementary Eqs. (6)-(11);
       [`acsn_noise`][optix.denoise.utils.acsn_noise]).
    3. **Sparse filtering.** Collaborative filtering of BM3D type with
       the parameters of [`acsn_profile`][optix.denoise.utils.acsn_profile]
       (Supplementary Note 2.4): 8 × 8 patches grouped by the distance
       ‖P − Q‖₂²/k² ≤ 2500 (Eq. (12)) in a 19 × 19 window, at most 8 per
       group, hard thresholding at 4·σ_N of the 3-D spectrum (Eqs.
       (13)-(14)), aggregation with weights 1/N_P (Eq. (15)); then
       grouping on that intermediate image and empirical Wiener
       filtering, c_P = |τ(𝒫)|²/(|τ(𝒫)|² + σ_N²) (Eqs. (16)-(17)).

    The result is the denoised M, in electrons (ADU/gain, offset
    removed).

    !!! note "Differences from BM3D (Dabov et al. 2007)"
        The engine is that of [`bm3d`][optix.denoise.bm3d], with the
        ACsN parameters: a 19 × 19 search window (BM3D: 39), at most 8
        patches per group (16 and 32), λ = 4.0 (2.7), the hard-threshold
        stage matched on pixels rather than on bior1.5 coefficients,
        weights 1/N_P (1/(σ²·N_P), which differs only in the weight of
        a group with no coefficient left), the same τ_match = 2500 for
        every σ (BM3D switches profile for σ > 40) and no Kaiser window.

    !!! note "Choices not specified in the paper"
        * From the ACsN code (BSD-3), not in the paper: σ_c is fitted
          to the histogram of |h| (see
          [`sigma_from_highpass`][optix.denoise.utils.sigma_from_highpass]);
          the filter outside the OTF is an ideal step; each frame is
          mapped to [0, 255] by its minimum and maximum, and σ_N
          with it, before the sparse filter (the paper's τ_match and λ
          are in 8-bit units).
        * f̄ is the Nyquist frequency 1/(2Δ) (`camera_cutoff=1`; the
          code uses 1.1/(2Δ)).
        * The transforms and the Wiener-stage τ_match = 400 and weights
          are those of BM3D's normal profile.
        * The code clamps the gain map to ≥ 1 and M to > 0; neither is
          done here (in optix the gain is in ADU per electron and may
          be below 1).

    !!! note "Extensions"
        * `ratio="area"` replaces the 1-D ratio of Supplementary Eq.
          (11) by its exact value for white noise over the 2-D
          frequencies of the frame (see
          [`noise_ratio`][optix.denoise.utils.noise_ratio]); the 1-D
          formula overestimates σ_N (by 58 % for f_c = 0.8·f̄).
        * `sigma_fit="std"`: σ_c is the RMS of h.
        * `sigma`: a known σ_N (electrons) replaces the estimate.
        * `hotspot=True`: the hot-pixel removal of the ACsN code (BSD-3;
          on by default there, not in the paper): pixels whose |h|
          exceeds mean(|h|) + 3·sd(|h|) are replaced by their 3 × 3
          median before the noise estimate.
        * The video filtering of Supplementary Note 2.5 (a further
          spatio-temporal grouping along motion trajectories, run when
          [`temporal_psnr`][optix.denoise.utils.temporal_psnr] is below
          35 dB) is not implemented.

    !!! note "Cost and memory"
        The noise estimate needs the FFT of the whole frame. The sparse
        filter runs on blocks of the frame with a halo of 50 pixels,
        sized from the memory budget (or `block`); the result does not
        depend on the blocks.

    !!! references "References"
        1. Mandracchia B, Hua X, Guo C, Son J, Urner T, Jia S. *Fast and
           accurate sCMOS noise correction for fluorescence microscopy.*
           Nat Commun **11**:94, 2020.
           [doi:10.1038/s41467-019-13841-8](https://doi.org/10.1038/s41467-019-13841-8)
        2. Dabov K, Foi A, Katkovnik V, Egiazarian K. *Image denoising by
           sparse 3-D transform-domain collaborative filtering.* IEEE Trans
           Image Process **16**(8):2080–2095, 2007.
           [doi:10.1109/TIP.2007.901238](https://doi.org/10.1109/TIP.2007.901238)

        ??? quote "BibTeX"
            ```bibtex
            @article{mandracchia2020fast,
              author  = {Mandracchia, Biagio and Hua, Xuanwen and
                         Guo, Changliang and Son, Jeonghwan and
                         Urner, Tara and Jia, Shu},
              title   = {Fast and accurate {sCMOS} noise correction for
                         fluorescence microscopy},
              journal = {Nature Communications},
              volume  = {11}, pages = {94},
              year    = {2020}, doi = {10.1038/s41467-019-13841-8}
            }
            @article{dabov2007image,
              author  = {Dabov, Kostadin and Foi, Alessandro and
                         Katkovnik, Vladimir and Egiazarian, Karen},
              title   = {Image denoising by sparse {3-D} transform-domain
                         collaborative filtering},
              journal = {IEEE Transactions on Image Processing},
              volume  = {16}, number = {8}, pages = {2080--2095},
              year    = {2007}, doi = {10.1109/TIP.2007.901238}
            }
            ```

    Parameters
    ----------
    data : Tensor
        `(..., H, W)` frames in ADU.
    na : float
        Numerical aperture.
    wavelength : float
        Emission wavelength, in the unit of `pixel_size`.
    pixel_size : float
        Pixel size in the image plane (camera pixel / magnification).
    offset : Tensor or float
        Offset β in ADU: scalar or `(H, W)` map.
    gain : Tensor or float
        Gain γ in ADU per electron: scalar or `(H, W)` map.
    sigma : float, optional
        Extension: known σ_N in electrons (no estimate).
    ratio : {"1d", "area"}
        σ_N/σ_c of Supplementary Eq. (11) (`"1d"`) or its 2-D value
        (`"area"`, extension).
    sigma_fit : {"histogram", "std"}
        Estimate of σ_c.
    camera_cutoff : float
        f̄ in units of the Nyquist frequency.
    hotspot : bool
        Extension: remove hot pixels first.
    block : sequence of int, optional
        Core block size of the sparse filter (default: memory budget).

    Returns
    -------
    Tensor
        Denoised frames, in electrons, of the shape of `data`.
    """
    o = dict(
        sigma=sigma,
        ratio=ratio,
        sigma_fit=sigma_fit,
        camera_cutoff=camera_cutoff,
        hotspot=hotspot,
        block=block,
    )
    _check_options(o)
    pixel_size = _check_positive("pixel_size", pixel_size)
    cutoff = otf_cutoff(na, wavelength)
    if data.ndim < 2:
        raise ValueError(f"data needs at least 2 dims, got {data.ndim}")
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    offset = torch.as_tensor(offset, dtype=data.dtype, device=data.device)
    gain = torch.as_tensor(gain, dtype=data.dtype, device=data.device)
    flat = data.reshape(-1, *data.shape[-2:])
    out = [_frame(z, offset, gain, pixel_size, cutoff, o)[0] for z in flat]
    return torch.stack(out).reshape(data.shape)


# -- quality ------------------------------------------------------------------
def _gauss_window(dtype: torch.dtype, device: torch.device) -> Tensor:
    r = torch.arange(11, dtype=dtype, device=device) - 5
    g = torch.exp(-0.5 * (r / 1.5).square())
    g = g / g.sum()
    return (g[:, None] * g)[None, None]


def _local(x: Tensor, w: Tensor) -> Tensor:
    return torch.nn.functional.conv2d(x[None, None], w)[0, 0]


def _structure(a: Tensor, b: Tensor, w: Tensor, c: float) -> Tensor:
    """ρ(A, B) = (σ_AB + c)/(σ_A·σ_B + c), local (Supplementary Eq. (18))."""
    ma, mb = _local(a, w), _local(b, w)
    va = (_local(a * a, w) - ma * ma).clamp(min=0)
    vb = (_local(b * b, w) - mb * mb).clamp(min=0)
    cov = _local(a * b, w) - ma * mb
    return (cov + c) / (va.sqrt() * vb.sqrt() + c)


def quality_score(noisy: Tensor, denoised: Tensor) -> float:
    """No-reference quality score of a denoised frame (Q-score).

    Supplementary Note 7 (after Kong et al.): with the method noise
    image M = I − Î (Eq. (17)), the local structure similarity maps
    N = ρ(I, M) and P = ρ(I, Î), ρ(A, B) = (σ_AB + c)/(σ_A·σ_B + c)
    (Eq. (18)), and

        Q = −corr(N, P)

    (Eq. (19)), the Pearson correlation over the maps; Q ∈ [−1, 1],
    higher is better.

    !!! note "Choices not specified in the paper"
        The local statistics use the window of SSIM (Wang et al. 2004,
        [doi:10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861)):
        11 × 11 Gaussian of σ = 1.5, valid positions only, and
        c = (0.03·L)²/2 with L the range of I.

    Parameters
    ----------
    noisy, denoised : (H, W) Tensor
        I (offset- and gain-corrected) and Î.

    Returns
    -------
    float
    """
    i = noisy.double()
    d = denoised.double()
    if i.shape != d.shape or i.ndim != 2:
        raise ValueError("noisy and denoised must be two (H, W) frames")
    if min(i.shape) < 11:
        raise ValueError("Frames must be at least 11 × 11")
    w = _gauss_window(i.dtype, i.device)
    c = (0.03 * float(i.max() - i.min())) ** 2 / 2 + 1e-30
    n = _structure(i, i - d, w, c).reshape(-1)
    p = _structure(i, d, w, c).reshape(-1)
    n, p = n - n.mean(), p - p.mean()
    tiny = 1e-12 * math.sqrt(n.numel())
    if float(n.norm()) <= tiny or float(p.norm()) <= tiny:
        return 0.0  # a constant map: no correlation
    return -float(n @ p) / float(n.norm() * p.norm())


def temporal_psnr(frames: Tensor, neighbours: int = 5) -> Tensor:
    """PSNR of each frame against its average with the next frames.

    The trigger of the video filtering of Supplementary Note 2.5: frame
    t is compared with the mean of frames t, …, t + `neighbours`, with
    peak the maximum of frame t,

        PSNR_t = 10·log₁₀(max(x_t)² / mean((x_t − x̄_t)²)),

    and the video filtering runs when the average PSNR is below 35 dB.
    The frames "and its 5 nearest neighbours" of the paper are the 5
    following ones in the ACsN code.

    Parameters
    ----------
    frames : (T, H, W) Tensor
        Denoised frames, T > `neighbours`.
    neighbours : int

    Returns
    -------
    (T − neighbours,) Tensor
        PSNR in dB.
    """
    t = frames.shape[0]
    if frames.ndim != 3 or t <= neighbours:
        raise ValueError(
            f"Need (T, H, W) frames with T > {neighbours}, got "
            f"{tuple(frames.shape)}"
        )
    x = frames.double()
    windows = x.unfold(0, neighbours + 1, 1)  # (T − n, H, W, n + 1)
    ref = windows.mean(-1)
    cur = x[: t - neighbours]
    mse = (cur - ref).square().mean((-2, -1))
    peak = cur.amax((-2, -1))
    return 10 * torch.log10(peak.square() / mse)


# -- estimator ----------------------------------------------------------------
class ACsN(Estimator):
    """ACsN: automatic correction of sCMOS-related noise.

    Applies [`acsn`][optix.denoise.acsn] to each frame (the two spatial
    axes of the layout; other axes are samples or groups) with the
    offset and gain maps of a
    [`SensorEstimate`][optix.estimates.SensorEstimate] given to
    [`fit`][optix.Estimator.fit] (for instance that of
    [`SCMOSCalibration`][optix.sensor.SCMOSCalibration], the calibration
    of the paper's Methods, Eq. (2)). The noise level σ_N is estimated
    from each frame, which makes the processing frame-independent.

    [`apply`][optix.Estimator.apply] streams batches of frames; each
    frame is loaded whole (the noise estimate needs its FFT) and its
    sparse filter runs in blocks with a halo. The estimated σ_N of every
    frame is kept in `sigma_` (shape `(*group, *sample)`, electrons),
    and its no-reference quality score (the "image quality check" of
    Supplementary Table 5;
    [`quality_score`][optix.denoise.utils.quality_score]) in `quality_`.
    The output is in electrons.

    !!! note
        Faithful mode is the paper (`ratio="1d"`, `sigma_fit="histogram"`,
        `camera_cutoff=1`, `hotspot=False`, `sigma=None`).
        `faithful=False` selects `ratio="area"`, the exact white-noise
        ratio. `block` does not change the result. The variance map of
        the estimate is not used: σ_N comes from the frame.

    !!! references "References"
        1. Mandracchia B, Hua X, Guo C, Son J, Urner T, Jia S. *Fast and
           accurate sCMOS noise correction for fluorescence microscopy.*
           Nat Commun **11**:94, 2020.
           [doi:10.1038/s41467-019-13841-8](https://doi.org/10.1038/s41467-019-13841-8)
        2. Dabov K, Foi A, Katkovnik V, Egiazarian K. *Image denoising by
           sparse 3-D transform-domain collaborative filtering.* IEEE Trans
           Image Process **16**(8):2080–2095, 2007.
           [doi:10.1109/TIP.2007.901238](https://doi.org/10.1109/TIP.2007.901238)

        ??? quote "BibTeX"
            ```bibtex
            @article{mandracchia2020fast,
              author  = {Mandracchia, Biagio and Hua, Xuanwen and
                         Guo, Changliang and Son, Jeonghwan and
                         Urner, Tara and Jia, Shu},
              title   = {Fast and accurate {sCMOS} noise correction for
                         fluorescence microscopy},
              journal = {Nature Communications},
              volume  = {11}, pages = {94},
              year    = {2020}, doi = {10.1038/s41467-019-13841-8}
            }
            @article{dabov2007image,
              author  = {Dabov, Kostadin and Foi, Alessandro and
                         Katkovnik, Vladimir and Egiazarian, Karen},
              title   = {Image denoising by sparse {3-D} transform-domain
                         collaborative filtering},
              journal = {IEEE Transactions on Image Processing},
              volume  = {16}, number = {8}, pages = {2080--2095},
              year    = {2007}, doi = {10.1109/TIP.2007.901238}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.denoise import ACsN
    >>> from optix.estimates import SensorEstimate
    >>> y = torch.arange(48.0).expand(48, 48)
    >>> sensor = SensorEstimate(
    ...     offset=torch.tensor(100.0),
    ...     variance=torch.tensor(4.0),
    ...     gain=torch.tensor(2.0),
    ... )
    >>> raw = 100 + 2 * torch.poisson(y) + 2 * torch.randn(48, 48)
    >>> den = ACsN(na=1.4, wavelength=0.7, pixel_size=0.1)
    >>> den = den.fit(init=sensor).apply(raw)
    >>> den.shape
    torch.Size([48, 48])
    """

    tags = Tags(
        targets=("denoised",),
        streaming="per-frame",
        modality=("widefield", "light-sheet"),
        noise="poisson-gaussian",
        reference="doi:10.1038/s41467-019-13841-8",
    )
    paper = {
        "na": None,
        "wavelength": None,
        "pixel_size": None,
        "ratio": "1d",
        "sigma_fit": "histogram",
        "camera_cutoff": 1.0,
        "hotspot": False,
        "sigma": None,
        "block": None,
    }
    recommended = {**paper, "ratio": "area"}

    def __init__(
        self,
        *,
        na: float | None = None,
        wavelength: float | None = None,
        pixel_size: float | None = None,
        ratio: Ratio | None = None,
        sigma_fit: SigmaFit | None = None,
        camera_cutoff: float | None = None,
        hotspot: bool | None = None,
        sigma: float | None = None,
        block: Sequence[int | None] | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the denoiser.

        Parameters
        ----------
        na : float
            Numerical aperture of the detection objective.
        wavelength : float
            Emission wavelength, in the unit of `pixel_size` (e.g. µm).
        pixel_size : float
            Pixel size in the image plane (camera pixel divided by the
            magnification).
        ratio : {"1d", "area"}
            σ_N/σ_c: Supplementary Eq. (11) (`"1d"`, the paper) or its
            exact 2-D value for white noise (`"area"`, extension; the
            default with `faithful=False`).
        sigma_fit : {"histogram", "std"}
            σ_c from a Gaussian fit to the histogram of |h| (the ACsN
            code, default) or the RMS of h (extension).
        camera_cutoff : float
            Camera cutoff f̄ in units of the Nyquist frequency (default
            1; the ACsN code uses 1.1).
        hotspot : bool
            Extension: hot-pixel removal of the ACsN code. Default False.
        sigma : float, optional
            Extension: known σ_N in electrons, for every frame.
        block : sequence of int, optional
            Core block size of the sparse filter (default: from the
            memory budget). The result does not depend on it.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        self._setup()
        self.sigma_: Tensor | None = None
        self.quality_: Tensor | None = None
        self._fitted = False

    def _setup(self) -> None:
        o = self.options
        for name in ("na", "wavelength", "pixel_size"):
            if o[name] is None:
                raise ValueError(f"ACsN needs {name}=")
            o[name] = _check_positive(name, o[name])
        _check_options(o)
        self.is_faithful = all(
            o[k] == self.paper[k]
            for k in (
                "ratio",
                "sigma_fit",
                "camera_cutoff",
                "hotspot",
                "sigma",
            )
        )

    def fit(
        self,
        data: ArrayLike | None = None,
        *,
        layout: Layout | None = None,
        init: object = None,
        **kwargs: object,
    ) -> Self:
        """Set the camera calibration.

        Parameters
        ----------
        data : array-like, optional
            Only its layout is recorded: the noise is estimated frame by
            frame in [`apply`][optix.Estimator.apply].
        layout : Layout, optional
        init : SensorEstimate or Estimator, optional
            Offset β (ADU) and gain γ (ADU per electron), scalars or
            `(*group, *spatial)` maps. Without it, β = 0 and γ = 1 (data
            already in electrons).
        **kwargs
            Ignored (`callback`, `verbose`).

        Returns
        -------
        Estimator
            self
        """
        if data is None:
            self._set_layout(layout)
            self._fit(None, init=init, callback=None, verbose=False)
            return self
        return super().fit(data, layout=layout, init=init)

    def _fit(
        self,
        data: ArrayLike | None,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        self.estimates = {}
        if init is not None:
            sensor = self._pick(init, SensorEstimate)
            self._warn_unused(init, SensorEstimate)
            if sensor is not None:
                for name in ("offset", "gain"):
                    if getattr(sensor, name) is None:
                        raise ValueError(f"init SensorEstimate lacks {name}")
                if bool((sensor.gain <= 0).any()):
                    raise ValueError("The gain must be > 0")
                self.estimates = {"sensor": sensor}
        self._fitted = True

    def _maps(self, group: tuple[int, ...], like: Tensor) -> tuple:
        sensor = self.estimates.get("sensor")
        if sensor is None:
            return 0.0, 1.0
        return tuple(
            _select(getattr(sensor, k), group, 2).to(like)
            for k in ("offset", "gain")
        )

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if not self._fitted:
            raise RuntimeError("Call fit before apply")
        layout = self._layout_of(data, 2)
        group_shape, sample_shape, spatial = layout.shape_of(data)
        if len(spatial) != 2:
            raise ValueError(f"ACsN needs 2 spatial axes, got {len(spatial)}")
        sigmas = torch.zeros(group_shape + sample_shape, dtype=torch.float64)
        quality = torch.full_like(sigmas, math.nan)
        o = self.options
        cutoff = otf_cutoff(o["na"], o["wavelength"])

        def run(chunk: object, batch: Tensor) -> Tensor:
            if not batch.is_floating_point():
                batch = batch.to(self._backend(batch)["dtype"])
            offset, gain = self._maps(chunk.group, batch)
            res, sig, qs = [], [], []
            for z in batch:
                d, s, m = _frame(z, offset, gain, o["pixel_size"], cutoff, o)
                res.append(d)
                sig.append(s)
                qs.append(
                    quality_score(m, d) if min(m.shape) >= 11 else math.nan
                )
            index = chunk.group + chunk.samples
            shape = chunk.sample_shape
            sigmas[index] = torch.tensor(sig, dtype=torch.float64).reshape(
                shape
            )
            quality[index] = torch.tensor(qs, dtype=torch.float64).reshape(
                shape
            )
            return torch.stack(res)

        if not sample_shape and not group_shape:
            # a single frame: process it with a frame axis
            raw = data.array if isinstance(data, Labeled) else data
            x = (
                raw
                if isinstance(raw, Tensor)
                else torch.as_tensor(np.asarray(raw))
            )
            frame = x.permute(layout.permutation(2))[None]
            sigmas, quality = sigmas.reshape(1), quality.reshape(1)
            result = self._apply_frames(
                frame, None, run, overhead=8.0, contiguous=True
            )[0].permute(layout.inverse_permutation(2))
            out = prepare_out(
                out, tuple(data.shape), like=raw, dtype=self.dtype
            )
            if isinstance(out, Tensor):
                out.copy_(result)
            else:
                out[...] = result.cpu().numpy()
            self.sigma_, self.quality_ = (
                sigmas.reshape(()),
                quality.reshape(()),
            )
            return out
        out = self._apply_frames(data, out, run, overhead=8.0, contiguous=True)
        self.sigma_, self.quality_ = sigmas, quality
        return out

    def _extra_state(self) -> dict[str, object]:
        return {"fitted": torch.tensor(self._fitted)}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self._setup()
        self._fitted = bool(state.get("fitted", False))
