"""Combined wavelet–Fourier stripe filter of Münch et al. (2009)."""

import math
from collections.abc import Callable, Sequence
from typing import Self

import torch
from torch import Tensor

from .._config import resolve_backend
from .._estimator import Estimator, Tags, _same
from ..io import ArrayLike, Layout, WritableArrayLike
from ._dwt import Wavelet, get_wavelet, wavedec2, waverec2

__all__ = ["WaveletFFT", "stripe_damping", "wavelet_fft"]


def stripe_damping(
    n: int,
    sigma: float,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Gaussian band-stop of Münch et al. (2009), Eq. (9).

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


def _check(
    wavelet: object, level: object, sigma: object, levels: object
) -> tuple[Wavelet, int, list[float], set[int]]:
    if wavelet is None or level is None or sigma is None:
        raise ValueError("wavelet, level and sigma must not be None")
    w = get_wavelet(wavelet)
    if isinstance(level, bool) or int(level) != level or level < 1:
        raise ValueError(f"level must be an integer >= 1, got {level!r}")
    level = int(level)
    if isinstance(sigma, int | float):
        sigmas = [float(sigma)] * level
    else:
        sigmas = [float(s) for s in sigma]
        if len(sigmas) != level:
            raise ValueError(
                f"sigma has {len(sigmas)} values for {level} levels"
            )
    if any(not math.isfinite(s) or s < 0 for s in sigmas):
        raise ValueError(f"sigma must be finite and >= 0, got {sigma!r}")
    if levels is None:
        chosen = set(range(1, level + 1))
    else:
        chosen = {int(v) for v in levels}
        if not chosen <= set(range(1, level + 1)):
            raise ValueError(
                f"levels must lie in 1..{level}, got {sorted(chosen)}"
            )
    return w, level, sigmas, chosen


def wavelet_fft(
    data: Tensor,
    *,
    wavelet: str | Wavelet = "db10",
    level: int = 4,
    sigma: float | Sequence[float] = 1.5,
    axis: int = -2,
    levels: Sequence[int] | None = None,
) -> Tensor:
    """Remove stripes with the combined wavelet–FFT filter.

    Each frame f(x, y), with stripes along y (constant offsets over the
    whole frame, Eq. (8)), is processed as in Münch et al. (2009), Fig. 5:

    1. multilevel 2-D DWT down to level L (Eq. (6)), giving the vertical
       detail bands c_v,l (low pass along y, high pass along x), l = 1, …, L;
    2. 1-D FFT of each c_v,l along y, ĉ_v,l(x, ŷ);
    3. damping by the Gaussian band-stop of Eq. (9),
       ĉ′_v,l = ĉ_v,l · (1 − exp(−ŷ²/(2σ²))), which removes the stripe
       information condensed on the line ŷ = 0;
    4. inverse FFT, then inverse DWT.

    The DWT uses symmetric (mirror) boundary extension (Section 2.3) and
    the Daubechies filters dbN of the paper; see
    [`wavedec2`][optix.destripe.utils.wavedec2]. The FFT runs along the
    stripes over the whole band: the paper does not specify padding or
    windowing, and none is applied.

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

    Parameters
    ----------
    data : Tensor
        Frames `(..., H, W)` (floating point; other dtypes are converted
        to the configured dtype).
    wavelet : str or Wavelet, default `"db10"`
        `"dbN"` (N = 1, …, 45) or `"haar"`. The paper recommends
        Daubechies wavelets with more than three vanishing moments,
        db10–db15 being a good compromise (Section 3.8).
    level : int, default 4
        Highest decomposition level L ≥ 1 (`decNum` in Fig. 5). It sets
        the widest stripes removed (Section 3.6).
    sigma : float or sequence of float, default 1.5
        Damping width σ ≥ 0 of Eq. (9), in frequency samples of each band;
        0 is the limit σ → 0 (ideal stripes). One value for all levels
        (as in the paper), or one per level, finest first (Section 2.3
        mentions adapting σ to each level).
    axis : {-2, -1}
        Frame axis along which the stripes run: -2 for stripes along the
        rows axis (vertical stripes in an `(H, W)` image, the case of the
        paper), -1 for horizontal stripes (Section 2.1: "handled
        analogously").
    levels : sequence of int, optional
        Detail levels to filter (1 is the finest); default: all of
        1, …, L, as in Fig. 5. Section 3.6 discusses filtering a subset
        of the levels.

    Returns
    -------
    Tensor
        Destriped frames, of the shape of `data`.

    Notes
    -----
    The defaults (db10, L = 4, σ = 1.5) are the paper's suggestions for
    typical data (Section 3.8: σ = 1.5 and L = 4 remove the visible
    stripes of the Lena image and of the tomography data of Section 4.2,
    and db10–db15 is the recommended wavelet size). Heavier or wider
    stripes need a larger L or σ (σ = 7 and L = 8 for the FIB-nt image of
    Section 4.1).

    The filter is linear and stateless, and costs O(N·F) for N pixels
    and a filter of length F.
    """
    w, level, sigmas, chosen = _check(wavelet, level, sigma, levels)
    if axis not in (-2, -1, data.ndim - 2, data.ndim - 1) or data.ndim < 2:
        raise ValueError(f"axis must be -2 or -1, got {axis}")
    axis = axis - data.ndim if axis >= 0 else axis
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    shape = tuple(data.shape[-2:])
    coeffs = wavedec2(data, w, level)
    # vertical details (index 1) hold stripes along -2, horizontal (0)
    # stripes along -1
    band = 1 if axis == -2 else 0
    for i in range(1, level + 1):
        lvl = level + 1 - i  # coeffs[1] is the coarsest level L
        if lvl not in chosen:
            continue
        details = list(coeffs[i])
        details[band] = _damp(details[band], sigmas[lvl - 1], axis)
        coeffs[i] = tuple(details)
    return waverec2(coeffs, w, shape=shape)


def _damp(band: Tensor, sigma: float, axis: int) -> Tensor:
    """Eq. (9) applied along `axis` (FFT, damping, inverse FFT)."""
    n = band.shape[axis]
    spectrum = torch.fft.rfft(band, dim=axis)
    g = stripe_damping(n, sigma, dtype=band.dtype, device=band.device)
    g = g.view(-1, *([1] * (-axis - 1)))
    return torch.fft.irfft(spectrum * g, n=n, dim=axis)


class WaveletFFT(Estimator):
    """Stripe removal by combined wavelet–Fourier filtering.

    The filter of Münch et al. (2009), applied to every frame
    independently: see [`wavelet_fft`][optix.destripe.wavelet_fft] for
    the method. It has no parameters to fit:
    [`fit`][optix.destripe.WaveletFFT.fit] only records the layout, and
    [`apply`][optix.Estimator.apply] streams the frames in batches, so
    arrays larger than memory (zarr, memmap) can be processed.

    !!! note
        The paper leaves its three parameters (wavelet, highest level L,
        damping width σ) to be tuned to the data (Section 4.4). The
        defaults, db10, L = 4 and σ = 1.5, are the paper's suggestions for
        typical data (Section 3.8); any value is faithful to the paper.

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
    }
    recommended = dict(paper)
    # free parameters of the published method: any value is faithful
    _free = ("wavelet", "level", "sigma", "axis", "levels")

    def __init__(
        self,
        *,
        wavelet: str | None = None,
        level: int | None = None,
        sigma: float | Sequence[float] | None = None,
        axis: int | str | None = None,
        levels: Sequence[int] | None = None,
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
        level : int, default 4
            Highest decomposition level L ≥ 1.
        sigma : float or sequence of float, default 1.5
            Damping width σ ≥ 0 of Eq. (9) (one value, or one per level,
            finest first).

            The three defaults are the paper's suggestions for typical
            data (Section 3.8).
        axis : int or str
            Spatial axis along which the stripes run: a position among the
            two spatial axes of the layout (0 or -2: the first, 1 or -1:
            the second), or an axis label of the layout (e.g. `"y"`).
            Paper: the first spatial axis (vertical stripes in `(y, x)`
            frames).
        levels : sequence of int, optional
            Detail levels to filter (1 is the finest); default: all.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        if not isinstance(self.options["sigma"], int | float):
            self.options["sigma"] = [float(s) for s in self.options["sigma"]]
        if self.options["levels"] is not None:
            self.options["levels"] = [int(v) for v in self.options["levels"]]
        self.is_faithful = all(
            _same(value, self.paper.get(name))
            for name, value in self.options.items()
            if name not in self._free
        )
        o = self.options
        _check(o["wavelet"], o["level"], o["sigma"], o["levels"])

    def fit(
        self,
        data: ArrayLike | None = None,
        *,
        layout: Layout | None = None,
        **kwargs: object,
    ) -> Self:
        """No-op: the filter has no parameters to estimate.

        Parameters
        ----------
        data : array-like, optional
            Unused (its layout, if labelled, is recorded).
        layout : Layout, optional
            Recorded for [`apply`][optix.Estimator.apply].
        **kwargs
            Ignored (`init`, `callback`, `verbose`).

        Returns
        -------
        WaveletFFT
            self
        """
        if data is None:
            self._set_layout(layout)
            return self
        return super().fit(data, layout=layout)

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

        def fn(chunk: object, batch: Tensor) -> Tensor:
            return wavelet_fft(
                batch,
                wavelet=o["wavelet"],
                level=o["level"],
                sigma=o["sigma"],
                axis=axis,
                levels=o["levels"],
            )

        return self._apply_frames(data, out, fn, overhead=4.0, contiguous=True)
