"""Combined wavelet–Fourier stripe filter of Münch et al. (2009).

After [Münch et al. (2009)](https://doi.org/10.1364/OE.17.008567), with
opt-in extensions from pystripe
([Swaney et al. 2019](https://doi.org/10.1101/576595)).

Parts of this module are adapted from pystripe
(https://github.com/chunglabmit/pystripe), under the MIT licence:

    MIT License

    Copyright (c) 2018 Chung Lab

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
from collections.abc import Callable, Sequence
from typing import Self

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .._config import resolve_backend
from .._estimator import Estimator, Tags, _same
from ..estimates import ShadingEstimate
from ..io import ArrayLike, Labeled, Layout, WritableArrayLike
from ..stats import otsu_threshold
from ._dwt import Wavelet, get_wavelet, max_level, wavedec2, waverec2

__all__ = ["WaveletFFT", "stripe_damping", "wavelet_fft"]


def stripe_damping(
    n: int,
    sigma: float,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Gaussian band-stop of Münch et al. (2009), Eq. (9).

    See [Münch et al. (2009), Eq. (9)](https://doi.org/10.1364/OE.17.008567):

        g(ŷ) = 1 − exp(−ŷ²/(2σ²)),

    on the non-negative frequency indices ŷ = 0, …, ⌊n/2⌋ of a real FFT
    of length n (the damping is even in ŷ). σ = 0 is the limit σ → 0:
    g(0) = 0 and g(ŷ) = 1 elsewhere.

    Parameters
    ----------
    n : int
        Length of the transformed signal.
    sigma : float
        Width σ ≥ 0, in frequency samples.
    dtype, device : optional

    Returns
    -------
    Tensor
        `(n // 2 + 1,)`.
    """
    k = torch.arange(n // 2 + 1, dtype=dtype, device=device)
    if sigma == 0:
        return (k > 0).to(k.dtype)
    return -torch.expm1(-k.square() / (2 * sigma * sigma))


def _sigmas(
    sigma: object, level: int | None, name: str = "sigma"
) -> list[float]:
    if isinstance(sigma, bool):
        raise ValueError(f"{name} must be a number, got {sigma!r}")
    if isinstance(sigma, int | float):
        sigmas = [float(sigma)] * (1 if level is None else level)
    else:
        sigmas = [float(s) for s in sigma]
        if level is not None and len(sigmas) != level:
            raise ValueError(
                f"{name} has {len(sigmas)} values for {level} levels"
            )
    if any(not math.isfinite(s) or s < 0 for s in sigmas):
        raise ValueError(f"{name} must be finite and >= 0, got {sigma!r}")
    return sigmas


def _check(
    wavelet: object,
    level: object,
    sigma: object,
    levels: object,
    n: int | None = None,
) -> tuple[Wavelet, int | None, list[float], set[int] | None]:
    """Validate the filter; `level="max"` is resolved for frames of min size n.

    Without `n`, a `"max"` level is left unresolved (None) and only the
    values of `sigma` are checked.
    """
    if wavelet is None or level is None or sigma is None:
        raise ValueError("wavelet, level and sigma must not be None")
    w = get_wavelet(wavelet)
    if isinstance(level, str):
        if level != "max":
            raise ValueError(
                f"level must be an integer or 'max', got {level!r}"
            )
        if n is None:
            return w, None, _sigmas(sigma, None), None
        level = max_level(n, w)
        if level < 1:
            raise ValueError(
                f"frames of size {n} are too small for a level of {w.name}"
            )
    elif isinstance(level, bool) or int(level) != level or level < 1:
        raise ValueError(f"level must be an integer >= 1, got {level!r}")
    level = int(level)
    sigmas = _sigmas(sigma, level)
    if levels is None:
        chosen = set(range(1, level + 1))
    else:
        chosen = {int(v) for v in levels}
        if not chosen <= set(range(1, level + 1)):
            raise ValueError(
                f"levels must lie in 1..{level}, got {sorted(chosen)}"
            )
    return w, level, sigmas, chosen


_BANDS = ("single", "dual", "foreground", "background")
_UNITS = ("band", "pystripe")


def _check_extensions(
    bands: object,
    sigma_background: object,
    threshold: object,
    crossover: object,
    smoothing: object,
    sigma_units: object,
) -> None:
    if bands not in _BANDS:
        raise ValueError(f"bands must be one of {_BANDS}, got {bands!r}")
    if sigma_units not in _UNITS:
        raise ValueError(
            f"sigma_units must be one of {_UNITS}, got {sigma_units!r}"
        )
    if bands in ("dual", "background"):
        if sigma_background is None:
            raise ValueError(f"bands={bands!r} needs sigma_background")
        _sigmas(sigma_background, None, "sigma_background")
    if threshold is not None and not math.isfinite(float(threshold)):
        raise ValueError(f"threshold must be finite, got {threshold!r}")
    if not (math.isfinite(float(crossover)) and crossover > 0):
        raise ValueError(f"crossover must be > 0, got {crossover!r}")
    if not (math.isfinite(float(smoothing)) and smoothing >= 0):
        raise ValueError(f"smoothing must be >= 0, got {smoothing!r}")


def wavelet_fft(
    data: Tensor,
    *,
    wavelet: str | Wavelet = "db10",
    level: int | str = 4,
    sigma: float | Sequence[float] = 1.5,
    axis: int = -2,
    levels: Sequence[int] | None = None,
    bands: str = "single",
    sigma_background: float | Sequence[float] | None = None,
    threshold: float | None = None,
    crossover: float = 10.0,
    smoothing: float = 1.0,
    log_transform: bool = False,
    sigma_units: str = "band",
    pad_even: bool = False,
    flat: Tensor | None = None,
    dark: Tensor | float | None = None,
) -> Tensor:
    """Remove stripes with the combined wavelet–FFT filter.

    Each frame f(x, y), with stripes along y (constant offsets over the
    whole frame, Eq. (8)), is processed as in
    [Münch et al. (2009), Fig. 5](https://doi.org/10.1364/OE.17.008567):

    1. multilevel 2-D DWT down to level L (Eq. (6)), giving the vertical
       detail bands c_{v,l} (low pass along y, high pass along x), l = 1, …, L;
    2. 1-D FFT of each c_{v,l} along y, ĉ_{v,l}(x, ŷ);
    3. damping by the Gaussian band-stop of Eq. (9),
       ĉ′_v,l = ĉ_{v,l} · (1 − exp(−ŷ²/(2σ²))), which removes the stripe
       information condensed on the line ŷ = 0;
    4. inverse FFT, then inverse DWT.

    The DWT uses symmetric (mirror) boundary extension (Section 2.3) and
    the Daubechies filters dbN of the paper; see
    [`wavedec2`][optix.destripe.utils.wavedec2]. The FFT runs along the
    stripes over the whole band: the paper does not specify padding or
    windowing, and none is applied.

    The default arguments give this filter. The other arguments are
    opt-in extensions from pystripe
    ([Swaney et al. 2019](https://doi.org/10.1101/576595)), described
    below; their equations are from the pystripe code, not in the paper,
    which only names the dual-band mode, the crossover and the flat-field
    correction.

    !!! note "Extensions from pystripe"
        Write W(y; σ) for the filter above with bandwidth σ (one value,
        or one per level). From the pystripe code, not in the paper:

        - **Log domain** (`log_transform=True`): the filter runs on
          log(1 + y), W(y; σ) = exp(W₀(log(1 + y); σ)) − 1, with W₀ the
          linear filter.
        - **Dual-band filtering** (`bands`): with a threshold t (Otsu's
          threshold of the frame,
          [`otsu_threshold`][optix.stats.otsu_threshold], unless given)
          and the foreground fraction

              φ = G_s ∗ sigmoid((y − t)/c),

          with c the `crossover` and G_s a Gaussian of width s =
          `smoothing` pixels, the frame is split into a background
          min(y, t) and a foreground max(y, t), filtered with their own
          bandwidths σ_bg (`sigma_background`) and σ_fg (`sigma`), and
          blended:

              x = φ · W(max(y, t); σ_fg) + (1 − φ) · W(min(y, t); σ_bg).

          `bands="foreground"` replaces the background term by y itself
          (only the foreground is filtered), `bands="background"` the
          foreground term. With `bands="dual"` and σ_fg = σ_bg the frame
          is not split: the filter is the single-band one.
        - **Bandwidth mapping** (`sigma_units="pystripe"`): σ is given in
          pixels of the frame, and level l uses σ_l = σ · n_l / n, with
          n_l and n the sizes of the band and of the frame across the
          stripes. The damping is the Gaussian of Eq. (9) evaluated on
          the index j = 0, …, m − 1 of the real FFT of length m stored in
          the packed (real, imaginary) order of `scipy.fftpack.rfft`:
          the real part of frequency k is damped by g(2k − 1), its
          imaginary part by g(2k).
        - **Even padding** (`pad_even=True`): frames of odd size are
          padded to even size by repeating their last row or column, and
          cropped back after filtering. The threshold is computed before
          padding.
        - **Flat-field and dark-offset correction** (`flat`, `dark`):
          after destriping, x ← (x − d)/f. pystripe normalises its flat
          field to a maximum of 1 before use; `flat` is used as given.

        pystripe's other defaults are db3, `level="max"`, `crossover=10`,
        and Otsu's threshold. Its output is clipped to [0, 2¹⁶ − 1] and
        cast to `uint16`; here it stays floating point.

    !!! references "References"
        1. Münch B, Trtik P, Marone F, Stampanoni M. *Stripe and ring
           artifact removal with combined wavelet — Fourier filtering.*
           Opt Express **17**(10):8567–8591, 2009.
           [doi:10.1364/OE.17.008567](https://doi.org/10.1364/OE.17.008567)
        2. Swaney J, Kamentsky L, Evans NB, Xie K, Park Y-G, Drummond G,
           Yun DH, Chung K. *Scalable image processing techniques for
           quantitative analysis of volumetric biological images from
           light-sheet microscopy.* bioRxiv 576595, 2019.
           [doi:10.1101/576595](https://doi.org/10.1101/576595)
           Code: [pystripe](https://github.com/chunglabmit/pystripe)
           (MIT licence, Copyright (c) 2018 Chung Lab).

        ??? quote "BibTeX"
            ```bibtex
            @article{munch2009stripe,
              author  = {Münch, Beat and Trtik, Pavel and Marone,
                         Federica and Stampanoni, Marco},
              title   = {Stripe and ring artifact removal with combined
                         wavelet --- {Fourier} filtering},
              journal = {Optics Express},
              volume  = {17}, number = {10}, pages = {8567--8591},
              year    = {2009}, doi = {10.1364/OE.17.008567}
            }
            @article{swaney2019scalable,
              author  = {Swaney, Justin and Kamentsky, Lee and Evans,
                         Nicholas B. and Xie, Katherine and Park,
                         Young-Gyun and Drummond, Gabrielle and Yun,
                         Dae Hee and Chung, Kwanghun},
              title   = {Scalable image processing techniques for
                         quantitative analysis of volumetric biological
                         images from light-sheet microscopy},
              journal = {bioRxiv},
              pages   = {576595},
              year    = {2019}, doi = {10.1101/576595}
            }
            ```

    !!! note
        The defaults (db10, L = 4, σ = 1.5) are the paper's suggestions for
        typical data (Section 3.8: σ = 1.5 and L = 4 remove the visible
        stripes of the Lena image and of the tomography data of Section 4.2,
        and db10–db15 is the recommended wavelet size). Heavier or wider
        stripes need a larger L or σ (σ = 7 and L = 8 for the FIB-nt image of
        Section 4.1).

        The filter is linear and stateless, and costs O(N·F) for N pixels
        and a filter of length F. The log domain and the dual-band mode
        make it non-linear; the dual-band mode filters two images.

    Parameters
    ----------
    data : Tensor
        Frames `(..., H, W)` (floating point; other dtypes are converted
        to the configured dtype).
    wavelet : str or Wavelet, default `"db10"`
        `"dbN"` (N = 1, …, 45) or `"haar"`. The paper recommends
        Daubechies wavelets with more than three vanishing moments,
        db10–db15 being a good compromise (Section 3.8).
    level : int or `"max"`, default 4
        Highest decomposition level L ≥ 1 (`decNum` in Fig. 5). It sets
        the widest stripes removed (Section 3.6). `"max"` is the largest
        level at which the bands are at least one filter long
        ([`max_level`][optix.destripe.utils.max_level] of the smaller
        frame size; pystripe's default).
    sigma : float or sequence of float, default 1.5
        Damping width σ ≥ 0 of Eq. (9), in frequency samples of each band;
        0 is the limit σ → 0 (ideal stripes). One value for all levels
        (as in the paper), or one per level, finest first (Section 2.3
        mentions adapting σ to each level). In the dual-band modes, the
        foreground bandwidth (pystripe's `sigma1`).
    axis : {-2, -1}
        Frame axis along which the stripes run: -2 for stripes along the
        rows axis (vertical stripes in an `(H, W)` image, the case of the
        paper), -1 for horizontal stripes (Section 2.1: "handled
        analogously"; pystripe's case).
    levels : sequence of int, optional
        Detail levels to filter (1 is the finest); default: all of
        1, …, L, as in Fig. 5. Section 3.6 discusses filtering a subset
        of the levels.
    bands : {"single", "dual", "foreground", "background"}, default "single"
        Which intensity bands are filtered (see the note above):
        `"single"` filters the whole frame (the paper).
    sigma_background : float or sequence of float, optional
        Background bandwidth σ_bg (pystripe's `sigma2`), required by
        `bands="dual"` and `"background"`.
    threshold : float, optional
        Intensity t separating background and foreground. Default:
        Otsu's threshold of each frame (256 bins, or one bin per integer
        value for integer `data`, as scikit-image's `threshold_otsu`).
    crossover : float, default 10
        Intensity range c of the transition between the bands.
    smoothing : float, default 1
        Width, in pixels, of the Gaussian smoothing the foreground
        fraction (fixed to 1 in pystripe); 0 for none.
    log_transform : bool, default False
        Filter log(1 + data) (pystripe).
    sigma_units : {"band", "pystripe"}, default "band"
        `"band"`: σ in frequency samples of each band (the paper);
        `"pystripe"`: pystripe's bandwidth mapping (see the note above).
    pad_even : bool, default False
        Pad frames of odd size to even size before filtering (pystripe).
    flat : Tensor, optional
        Flat field f, broadcasting against the frames (`(H, W)`).
    dark : Tensor or float, optional
        Dark offset d, subtracted before dividing by `flat`.

    Returns
    -------
    Tensor
        Destriped frames, of the shape of `data`.
    """
    return _wavelet_fft(
        data,
        wavelet=wavelet,
        level=level,
        sigma=sigma,
        axis=axis,
        levels=levels,
        bands=bands,
        sigma_background=sigma_background,
        threshold=threshold,
        crossover=crossover,
        smoothing=smoothing,
        log_transform=log_transform,
        sigma_units=sigma_units,
        pad_even=pad_even,
        flat=flat,
        dark=dark,
        integer=not data.is_floating_point(),
    )


def _wavelet_fft(
    data: Tensor,
    *,
    wavelet: str | Wavelet,
    level: int | str,
    sigma: float | Sequence[float],
    axis: int,
    levels: Sequence[int] | None,
    bands: str,
    sigma_background: float | Sequence[float] | None,
    threshold: float | None,
    crossover: float,
    smoothing: float,
    log_transform: bool,
    sigma_units: str,
    pad_even: bool,
    flat: Tensor | None,
    dark: Tensor | float | None,
    integer: bool,
) -> Tensor:
    """`wavelet_fft`; `integer` selects the Otsu binning."""
    _check(wavelet, level, sigma, levels)
    _check_extensions(
        bands, sigma_background, threshold, crossover, smoothing, sigma_units
    )
    if axis not in (-2, -1, data.ndim - 2, data.ndim - 1) or data.ndim < 2:
        raise ValueError(f"axis must be -2 or -1, got {axis}")
    axis = axis - data.ndim if axis >= 0 else axis
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    shape = tuple(data.shape[-2:])

    t = None
    if bands != "single":
        if threshold is None:
            t = otsu_threshold(data, bins=None if integer else 256)
            t = t.to(data.dtype)[..., None, None]
        else:
            t = torch.as_tensor(
                threshold, dtype=data.dtype, device=data.device
            )
    x = data
    if pad_even:
        for dim in (-2, -1):
            if x.shape[dim] % 2:
                x = torch.cat([x, x.narrow(dim, x.shape[dim] - 1, 1)], dim)

    def band(y: Tensor, s: float | Sequence[float]) -> Tensor:
        return _subband(
            y, wavelet, level, s, axis, levels, log_transform, sigma_units
        )

    if bands == "dual" and _same(
        _sigmas(sigma, None), _sigmas(sigma_background, None)
    ):
        bands = "single"  # pystripe: equal bandwidths, no split
    if bands == "single":
        out = band(x, sigma)
    else:
        phi = _smooth(torch.sigmoid((x - t) / crossover), smoothing)
        if bands == "background":
            fg = x
        else:
            fg = band(torch.maximum(x, t), sigma)
        if bands == "foreground":
            bg = x
        else:
            bg = band(torch.minimum(x, t), sigma_background)
        out = phi * fg + (1 - phi) * bg
    if out.shape[-2:] != shape:
        out = out[..., : shape[0], : shape[1]]
    if dark is not None:
        out = out - torch.as_tensor(dark).to(out)
    if flat is not None:
        out = out / torch.as_tensor(flat).to(out)
    return out


def _subband(
    x: Tensor,
    wavelet: str | Wavelet,
    level: int | str,
    sigma: float | Sequence[float],
    axis: int,
    levels: Sequence[int] | None,
    log_transform: bool,
    sigma_units: str,
) -> Tensor:
    """The wavelet–FFT filter W(x; σ) of one frame batch."""
    shape = tuple(x.shape[-2:])
    w, level, sigmas, chosen = _check(
        wavelet, level, sigma, levels, min(shape)
    )
    if log_transform:
        x = torch.log1p(x)
    coeffs = wavedec2(x, w, level)
    # vertical details (index 1) hold stripes along -2, horizontal (0)
    # stripes along -1
    band = 1 if axis == -2 else 0
    across = -1 if axis == -2 else -2  # the axis across the stripes
    for i in range(1, level + 1):
        lvl = level + 1 - i  # coeffs[1] is the coarsest level L
        if lvl not in chosen:
            continue
        details = list(coeffs[i])
        if sigma_units == "band":
            details[band] = _damp(details[band], sigmas[lvl - 1], axis)
        else:
            d = details[band]
            s = d.shape[across] * (sigmas[lvl - 1] / x.shape[across])
            details[band] = _damp_packed(d, s, axis)
        coeffs[i] = tuple(details)
    out = waverec2(coeffs, w, shape=shape)
    return torch.expm1(out) if log_transform else out


def _damp(band: Tensor, sigma: float, axis: int) -> Tensor:
    """Eq. (9) applied along `axis` (FFT, damping, inverse FFT)."""
    n = band.shape[axis]
    spectrum = torch.fft.rfft(band, dim=axis)
    g = stripe_damping(n, sigma, dtype=band.dtype, device=band.device)
    g = g.view(-1, *([1] * (-axis - 1)))
    return torch.fft.irfft(spectrum * g, n=n, dim=axis)


def _damp_packed(band: Tensor, sigma: float, axis: int) -> Tensor:
    """pystripe's damping, on the packed index of `scipy.fftpack.rfft`.

    Adapted from pystripe (`notch`, `filter_subband`), MIT licence,
    Copyright (c) 2018 Chung Lab, https://github.com/chunglabmit/pystripe.
    """
    n = band.shape[axis]
    j = torch.arange(n, dtype=band.dtype, device=band.device)
    if sigma == 0:
        g = (j > 0).to(j.dtype)
    else:
        g = -torch.expm1(-j.square() / (2 * sigma * sigma))
    k = torch.arange(n // 2 + 1, device=band.device)
    g_re = g[(2 * k - 1).clamp(0, n - 1)]
    g_im = g[(2 * k).clamp(0, n - 1)]
    view = (-1, *([1] * (-axis - 1)))
    spectrum = torch.fft.rfft(band, dim=axis)
    spectrum = torch.complex(
        spectrum.real * g_re.view(view), spectrum.imag * g_im.view(view)
    )
    return torch.fft.irfft(spectrum, n=n, dim=axis)


def _smooth(x: Tensor, sigma: float) -> Tensor:
    """Gaussian smoothing of the last two dims, mirror boundaries.

    scipy.ndimage's `gaussian_filter` (mode `"reflect"`, kernel truncated
    at ⌊4σ + 0.5⌋ samples), as in pystripe's `foreground_fraction`.
    """
    if sigma == 0:
        return x
    r = int(4 * sigma + 0.5)
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma).square())
    k = k / k.sum()
    for dim in (-2, -1):
        n = x.shape[dim]
        # half-sample symmetric extension, periodic with period 2n
        i = torch.arange(-r, n + r, device=x.device) % (2 * n)
        i = torch.where(i < n, i, 2 * n - 1 - i)
        y = x.index_select(dim, i).movedim(dim, -1)
        lead = y.shape[:-1]
        y = F.conv1d(y.reshape(-1, 1, y.shape[-1]), k.view(1, 1, -1))
        x = y.reshape(*lead, n).movedim(-1, dim)
    return x


class WaveletFFT(Estimator):
    """Stripe removal by combined wavelet–Fourier filtering.

    The filter of [Münch et al. (2009)](https://doi.org/10.1364/OE.17.008567),
    applied to every frame independently: see
    [`wavelet_fft`][optix.destripe.wavelet_fft] for the method. It has no
    parameters to fit: [`fit`][optix.destripe.WaveletFFT.fit] only records the
    layout, and [`apply`][optix.Estimator.apply] streams the frames in batches,
    so arrays larger than memory (zarr, memmap) can be processed.

    !!! note
        The paper leaves its three parameters (wavelet, highest level L,
        damping width σ) to be tuned to the data (Section 4.4). The
        defaults, db10, L = 4 and σ = 1.5, are the paper's suggestions for
        typical data (Section 3.8); any value is faithful to the paper.

    !!! note "Extensions from pystripe"
        The options `bands`, `sigma_background`, `threshold`, `crossover`,
        `smoothing`, `log_transform`, `sigma_units` and `pad_even`, and a
        flat-field and dark-offset correction given to
        [`fit`][optix.destripe.WaveletFFT.fit] as a
        [`ShadingEstimate`][optix.estimates.ShadingEstimate], add the
        filtering options of pystripe
        ([Swaney et al. 2019](https://doi.org/10.1101/576595)); see
        [`wavelet_fft`][optix.destripe.wavelet_fft]. They are off by
        default; their equations are from the pystripe code, not in the
        paper. With pystripe's settings (db3, `level="max"`,
        `log_transform=True`, `sigma_units="pystripe"`, `pad_even=True`,
        `axis=-1`), the filter matches pystripe's `filter_streaks` up to
        its final cast to `uint16`.

    !!! references "References"
        1. Münch B, Trtik P, Marone F, Stampanoni M. *Stripe and ring
           artifact removal with combined wavelet — Fourier filtering.*
           Opt Express **17**(10):8567–8591, 2009.
           [doi:10.1364/OE.17.008567](https://doi.org/10.1364/OE.17.008567)

        ??? quote "BibTeX"
            ```bibtex
            @article{munch2009stripe,
              author  = {Münch, Beat and Trtik, Pavel and Marone,
                         Federica and Stampanoni, Marco},
              title   = {Stripe and ring artifact removal with combined
                         wavelet --- {Fourier} filtering},
              journal = {Optics Express},
              volume  = {17}, number = {10}, pages = {8567--8591},
              year    = {2009}, doi = {10.1364/OE.17.008567}
            }
            ```
    """

    tags = Tags(
        targets=("stripes",),
        streaming="per-frame",
        reference="doi:10.1364/OE.17.008567",
    )
    paper = {
        # suggestions of Section 3.8 for typical data
        "wavelet": "db10",
        "level": 4,
        "sigma": 1.5,
        "axis": -2,
        "levels": None,
        # pystripe extensions, off
        "bands": "single",
        "sigma_background": None,
        "threshold": None,
        "crossover": 10.0,
        "smoothing": 1.0,
        "log_transform": False,
        "sigma_units": "band",
        "pad_even": False,
    }
    recommended = dict(paper)
    # free parameters of the published method: any value is faithful
    _free = ("wavelet", "level", "sigma", "axis", "levels")
    # used only by the dual-band modes
    _split = ("sigma_background", "threshold", "crossover", "smoothing")

    def __init__(
        self,
        *,
        wavelet: str | None = None,
        level: int | str | None = None,
        sigma: float | Sequence[float] | None = None,
        axis: int | str | None = None,
        levels: Sequence[int] | None = None,
        bands: str | None = None,
        sigma_background: float | Sequence[float] | None = None,
        threshold: float | None = None,
        crossover: float | None = None,
        smoothing: float | None = None,
        log_transform: bool | None = None,
        sigma_units: str | None = None,
        pad_even: bool | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the filter.

        Parameters
        ----------
        wavelet : str, default `"db10"`
            `"dbN"` (N = 1, …, 45) or `"haar"`; see
            [`wavelet_fft`][optix.destripe.wavelet_fft].
        level : int or `"max"`, default 4
            Highest decomposition level L ≥ 1, or `"max"` for the largest
            useful level of the frames.
        sigma : float or sequence of float, default 1.5
            Damping width σ ≥ 0 of Eq. (9) (one value, or one per level,
            finest first); the foreground bandwidth in the dual-band
            modes.

            The three defaults are the paper's suggestions for typical
            data (Section 3.8).
        axis : int or str
            Spatial axis along which the stripes run: a position among the
            two spatial axes of the layout (0 or -2: the first, 1 or -1:
            the second), or an axis label of the layout (e.g. `"y"`).
            Paper: the first spatial axis (vertical stripes in `(y, x)`
            frames). pystripe filters stripes along the second.
        levels : sequence of int, optional
            Detail levels to filter (1 is the finest); default: all.
        bands : {"single", "dual", "foreground", "background"}
            Intensity bands filtered (pystripe extension; `"single"` is
            the paper's filter).
        sigma_background : float or sequence of float, optional
            Background bandwidth (pystripe extension), required by
            `bands="dual"` and `"background"`.
        threshold : float, optional
            Background/foreground threshold (pystripe extension; default:
            Otsu's threshold of each frame).
        crossover : float, default 10
            Intensity range of the transition between the bands
            (pystripe extension).
        smoothing : float, default 1
            Width, in pixels, of the Gaussian smoothing of the foreground
            fraction (pystripe extension).
        log_transform : bool, default False
            Filter log(1 + data) (pystripe extension).
        sigma_units : {"band", "pystripe"}, default "band"
            Units of σ (pystripe extension: `"pystripe"`).
        pad_even : bool, default False
            Pad odd frame sizes to even (pystripe extension).
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator]. The options of the
            dual-band modes do not change the filter when
            `bands="single"`, and do not count against `is_faithful`
            then.
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        for name in ("sigma", "sigma_background"):
            if o[name] is not None and not isinstance(o[name], int | float):
                o[name] = [float(s) for s in o[name]]
        if o["levels"] is not None:
            o["levels"] = [int(v) for v in o["levels"]]
        self._update_faithful()
        _check(o["wavelet"], o["level"], o["sigma"], o["levels"])
        _check_extensions(*(o[k] for k in self._extension_args))

    _extension_args = (
        "bands",
        "sigma_background",
        "threshold",
        "crossover",
        "smoothing",
        "sigma_units",
    )

    def _update_faithful(self) -> None:
        ignore = set(self._free)
        if self.options["bands"] == "single":
            ignore |= set(self._split)
        self.is_faithful = "shading" not in self.estimates and all(
            _same(value, self.paper.get(name))
            for name, value in self.options.items()
            if name not in ignore
        )

    def fit(
        self,
        data: ArrayLike | None = None,
        *,
        layout: Layout | None = None,
        init: object = None,
        **kwargs: object,
    ) -> Self:
        """Record the layout, and the flat field and dark offset if given.

        The filter has no parameters to estimate. A
        [`ShadingEstimate`][optix.estimates.ShadingEstimate] in `init`
        adds pystripe's illumination correction after destriping,
        x ← (x − d)/f, with its `flat` f and `dark` d (either may be
        None; its per-frame `baseline` is not used). Its maps are
        `(*group, *spatial)` or broadcast to the frames.

        Parameters
        ----------
        data : array-like, optional
            Unused (its layout, if labelled, is recorded).
        layout : Layout, optional
            Recorded for [`apply`][optix.Estimator.apply].
        init : ShadingEstimate, optional
            Flat field and dark offset (pystripe extension).
        **kwargs
            Ignored (`callback`, `verbose`).

        Returns
        -------
        WaveletFFT
            self
        """
        self.estimates = {}
        shading = self._pick(init, ShadingEstimate)
        self._warn_unused(init, ShadingEstimate)
        if shading is not None:
            self.estimates["shading"] = ShadingEstimate(
                flat=shading.flat, dark=shading.dark, meta=dict(shading.meta)
            )
        self._update_faithful()
        if data is None:
            self._set_layout(layout)
            return self
        return super().fit(data, layout=layout)

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self._update_faithful()

    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        pass

    def _frame_axis(self, layout: Layout, ndim: int) -> int:
        """The stripe axis as a frame dim, -2 or -1."""
        _, _, spatial = layout.positions(ndim)
        if len(spatial) != 2:
            raise ValueError(
                f"WaveletFFT needs 2 spatial axes, the layout has "
                f"{len(spatial)}"
            )
        axis = self.options["axis"]
        if isinstance(axis, str):
            if layout.axes is None or axis not in layout.axes:
                raise ValueError(
                    f"axis {axis!r} is not a label of the layout {layout!r}"
                )
            pos = layout.axes.index(axis)
            if pos not in spatial:
                raise ValueError(f"axis {axis!r} is not a spatial axis")
            return spatial.index(pos) - 2
        if axis not in (-2, -1, 0, 1):
            raise ValueError(f"axis must be 0, 1, -2 or -1, got {axis!r}")
        return axis - 2 if axis >= 0 else axis

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        o = self.options
        _check(o["wavelet"], o["level"], o["sigma"], o["levels"])
        layout = self._layout_of(data)
        axis = self._frame_axis(layout, len(data.shape))
        raw = data.array if isinstance(data, Labeled) else data
        integer = _is_integer(raw.dtype)
        shading = self.estimates.get("shading")

        def pick(
            m: Tensor | None, group: tuple, like: Tensor
        ) -> Tensor | None:
            if m is None:
                return None
            if group and m.ndim > 2:
                m = m[group]
            return m.to(like)

        def fn(chunk: object, batch: Tensor) -> Tensor:
            flat = dark = None
            if shading is not None:
                flat = pick(shading.flat, chunk.group, batch)
                dark = pick(shading.dark, chunk.group, batch)
            return _wavelet_fft(
                batch,
                wavelet=o["wavelet"],
                level=o["level"],
                sigma=o["sigma"],
                axis=axis,
                levels=o["levels"],
                bands=o["bands"],
                sigma_background=o["sigma_background"],
                threshold=o["threshold"],
                crossover=o["crossover"],
                smoothing=o["smoothing"],
                log_transform=o["log_transform"],
                sigma_units=o["sigma_units"],
                pad_even=o["pad_even"],
                flat=flat,
                dark=dark,
                integer=integer,
            )

        overhead = 4.0 if o["bands"] == "single" else 10.0
        return self._apply_frames(
            data, out, fn, overhead=overhead, contiguous=True
        )


def _is_integer(dtype: object) -> bool:
    """Whether an array dtype (torch, numpy, zarr) is an integer type."""
    if isinstance(dtype, torch.dtype):
        return not (dtype.is_floating_point or dtype.is_complex)
    try:
        return bool(np.issubdtype(np.dtype(dtype), np.integer))
    except TypeError:
        return False
