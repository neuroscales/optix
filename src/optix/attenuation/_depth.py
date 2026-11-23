"""Depth-resolved attenuation estimation and compensation."""

import math
import warnings
from collections.abc import Callable
from typing import Self

import torch
from torch import Tensor

from .._config import get_config, resolve_backend
from .._estimator import Estimator, Tags
from ..estimates import Estimate, SensorEstimate, ShadingEstimate
from ..io import (
    ArrayLike,
    Labeled,
    Layout,
    WritableArrayLike,
    blocks,
    chunks_of,
    flush,
    prepare_out,
)
from ._kernels import (
    compensation_gain,
    smith_attenuation,
    vermeer_attenuation,
    vermeer_gain,
    weighted_tv,
)

__all__ = ["DepthResolved", "compensate", "depth_resolved"]

_METHODS = ("vermeer", "smith")
Background = float | Tensor | SensorEstimate | ShadingEstimate | None


def _background_field(background: Background) -> float | Tensor | None:
    """The additive term held by `background` (number, tensor or estimate)."""
    if isinstance(background, ShadingEstimate):
        if background.dark is None:
            raise ValueError("ShadingEstimate.dark is None (not estimated)")
        return background.dark
    if isinstance(background, SensorEstimate):
        if background.offset is None:
            raise ValueError("SensorEstimate.offset is None (not estimated)")
        return background.offset
    if isinstance(background, Estimate):
        raise TypeError(
            f"background must be a number, a Tensor, a SensorEstimate or a "
            f"ShadingEstimate, got {type(background).__name__}"
        )
    return background


def _subtract(data: Tensor, background: float | Tensor | None) -> Tensor:
    if background is None:
        return data
    if isinstance(background, Tensor):
        background = background.to(dtype=data.dtype, device=data.device)
    return data - background


def _other_axis(axis: int, ndim: int, tv_axis: int | None) -> int:
    """Lateral axis of the TV term (default: the other of the last two)."""
    if tv_axis is None:
        last = (ndim - 2, ndim - 1)
        if ndim < 2 or axis not in last:
            raise ValueError(
                "tv_axis must be given unless the propagation axis is one "
                "of the last two dims"
            )
        return last[0] if axis == last[1] else last[1]
    tv = tv_axis % ndim
    if tv == axis:
        raise ValueError("tv_axis must differ from the propagation axis")
    return tv


def _mu(
    data: Tensor,
    axis: int,
    tv: int | None,
    *,
    pixel_size: float,
    passes: int,
    method: str,
    eta: float,
    epsilon: float,
    noise_power: float,
    max_iter: int,
    tol: float,
) -> Tensor:
    """μ of an in-memory block holding whole profiles (after background)."""
    if method == "vermeer":
        return vermeer_attenuation(
            data, axis, pixel_size=pixel_size, passes=passes
        )
    mu = smith_attenuation(
        data,
        axis,
        pixel_size=pixel_size,
        passes=passes,
        noise_power=noise_power,
    )
    if eta == 0:
        return mu
    weights = 1 / (data.clamp_min(0) + epsilon)
    lam = eta / (data.shape[tv] * pixel_size)
    mu, _, _ = weighted_tv(
        mu, weights, lam, tv, nonneg=True, max_iter=max_iter, tol=tol
    )
    return mu


def _corrected(
    data: Tensor,
    axis: int,
    tv: int | None,
    *,
    max_gain: float | None,
    **options: object,
) -> Tensor:
    """Compensated block: (I − background)·G, G capped at `max_gain`."""
    if options["method"] == "vermeer":
        gain = vermeer_gain(data, axis)
    else:
        mu = _mu(data, axis, tv, **options)
        gain = compensation_gain(
            mu,
            axis,
            pixel_size=options["pixel_size"],
            passes=options["passes"],
        )
    if max_gain is not None:
        gain = gain.clamp_max(max_gain)
    return (data.to(gain.dtype) * gain).to(data.dtype)


def _check_options(
    method: str,
    max_gain: float | None,
    eta: float,
    epsilon: float,
    max_iter: int,
    tol: float,
) -> None:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if max_gain is not None and not max_gain >= 1:
        raise ValueError(f"max_gain must be >= 1, got {max_gain!r}")
    if not eta >= 0:
        raise ValueError(f"eta must be >= 0, got {eta!r}")
    if not epsilon > 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon!r}")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter must be a positive int, got {max_iter!r}")
    if not tol > 0:
        raise ValueError(f"tol must be > 0, got {tol!r}")


def _prepare(data: Tensor, axis: int) -> tuple[Tensor, int]:
    if not isinstance(data, Tensor):
        raise TypeError(
            "data must be a Tensor; use DepthResolved to stream arrays"
        )
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    if data.ndim < 1:
        raise ValueError("data must have at least one dim")
    if not -data.ndim <= axis < data.ndim:
        raise ValueError(f"axis {axis} is out of range for {data.ndim}D data")
    return data, axis % data.ndim


def depth_resolved(
    data: Tensor,
    *,
    axis: int = -1,
    pixel_size: float = 1.0,
    passes: int = 2,
    method: str = "vermeer",
    background: Background = None,
    tv_axis: int | None = None,
    eta: float = 6.0,
    epsilon: float = 1e-3,
    noise_power: float = 0.0,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> Tensor:
    """Per-pixel attenuation coefficient along the propagation axis.

    The depth-resolved estimator of Vermeer et al. (2014): every pixel of
    a profile I[i] along the propagation axis gets

        μ[i] = 1/(p·Δ) · log(1 + I[i] / ∑_{k > i} I[k]),            (17)

    the ratio of the local intensity to the sum of all deeper
    intensities, with Δ the pixel size and p = 2 (the light crosses the
    tissue twice in OCT, Eq. (8)). It follows from the single-scattering
    model I(z) = α·β·L₀·μ(z)·exp(−2∫₀^z μ(u) du) (Eq. (8)) and the
    assumption that almost all the light is attenuated within the
    profile, which truncates ∫_z^∞ to the end of the profile (Eq. (13)).

    End of the profile: the sum runs to the last pixel, where it is
    empty, so μ = ∞ at the last pixel (log(1 + I/0)), and μ is
    overestimated towards the end of the profile when light remains
    there (Vermeer et al., Section 3.1, Fig. 1). Where both I[i] and the
    deeper sum are zero, μ is undefined and returned as NaN.

    !!! note "Application to fluorescence"

        The estimator comes from OCT, where the signal is backscatter. It
        is applied here to fluorescence along the illumination
        propagation axis of a light sheet. The OCT model assumes that a
        fixed fraction α of the attenuated light is backscattered; the
        fluorescence analogue is a locally constant fluorophore emission:
        the detected fluorescence is a fixed fraction of the excitation
        light attenuated at each depth, so I ∝ μ·L. The excitation
        crosses the tissue once, which corresponds to `passes=1`; with
        the default `passes=2` (the paper's Eq. (17)), μ is half the
        single-pass coefficient. The compensated intensity
        ([`compensate`][optix.attenuation.compensate]) does not depend on
        `passes`. Attenuation of the emitted light on its way to the
        detector is not modelled.

    Extensions (off by default):

    - `background`: an additive term (number, tensor, or the `offset` of
      a [`SensorEstimate`][optix.estimates.SensorEstimate] or the `dark`
      of a [`ShadingEstimate`][optix.estimates.ShadingEstimate]) is
      subtracted first, since the model has no background (Vermeer et
      al. subtract the noise floor, Eq. (19)).
    - `method="smith"`: the estimate of Smith et al. (2015),
      Eqs. (8)–(10),

          Î[m] = I[m] · I[m]² / (I[m]² + |c|²),
          μ̂[m] = Î[m] / (p·Δ·∑_{r ≥ m} Î[r]),

      with the confocal and sensitivity fall-off functions of Eq. (9)
      set to 1 and |c|² = `noise_power`, followed by the intensity
      weighted horizontal TV denoising (iwhTV) of Eqs. (11)–(15): for
      every depth m, the line μ̂_m across the lateral axis `tv_axis` is
      replaced by the solution of

          minimise ½‖μ̃ − μ̂_m‖² + η/(N·Δ) ∑_n W[n]·|μ̃[n + 1] − μ̃[n]|
          subject to μ̃ ≥ 0,   W[n] = 1/(I[m, n] + ε),

      with N the number of lateral positions. No tissue mask is used
      (X = identity in Eq. (15), N_x = N). The weight W[n] goes with the
      difference μ̃[n + 1] − μ̃[n] (the paper indexes the weights by
      pixel and the differences by pair). Negative intensities count as
      0 in W. The problem is solved through its dual (see
      [`weighted_tv`][optix.attenuation.utils.weighted_tv]); Smith et
      al. use CVX. `eta=0` skips the denoising.

    !!! references "References"

        1. Vermeer KA, Mo J, Weda JJA, Lemij HG, de Boer JF.
           *Depth-resolved model-based reconstruction of attenuation
           coefficients in optical coherence tomography.* Biomed Opt
           Express **5**(1):322–337, 2014.
           [doi:10.1364/BOE.5.000322](https://doi.org/10.1364/BOE.5.000322)
        2. Smith GT, Dwork N, O'Connor D, Sikora U, Lurie KL, Pauly JM,
           Ellerbee AK. *Automated, depth-resolved estimation of the
           attenuation coefficient from optical coherence tomography
           data.* IEEE Trans Med Imaging **34**(12):2592–2602, 2015.
           [doi:10.1109/TMI.2015.2450197](https://doi.org/10.1109/TMI.2015.2450197)
        3. Frantz D, Wright CJ, Schaser AJ, Kirik D, Kristensson E,
           Berrocal E. *Image post-processing for SILMAS: structured
           illumination light sheet microscopy with axial sweeping.*
           Biomed Opt Express **15**(8):4943–4962, 2024.
           [doi:10.1364/BOE.531210](https://doi.org/10.1364/BOE.531210)
           (attenuation compensation along the light-sheet propagation
           axis)

        ??? quote "BibTeX"

            ```bibtex
            @article{vermeer2014depth,
              author  = {Vermeer, Koenraad A. and Mo, Jelena and Weda,
                         Jelmer J. A. and Lemij, Hans G. and
                         de Boer, Johannes F.},
              title   = {Depth-resolved model-based reconstruction of
                         attenuation coefficients in optical coherence
                         tomography},
              journal = {Biomedical Optics Express},
              volume  = {5}, number = {1}, pages = {322--337},
              year    = {2014}, doi = {10.1364/BOE.5.000322}
            }
            @article{smith2015automated,
              author  = {Smith, Gennifer T. and Dwork, Nicholas and
                         O'Connor, Daniel and Sikora, Uzair and
                         Lurie, Kristen L. and Pauly, John M. and
                         Ellerbee, Audrey K.},
              title   = {Automated, depth-resolved estimation of the
                         attenuation coefficient from optical coherence
                         tomography data},
              journal = {IEEE Transactions on Medical Imaging},
              volume  = {34}, number = {12}, pages = {2592--2602},
              year    = {2015}, doi = {10.1109/TMI.2015.2450197}
            }
            @article{frantz2024image,
              author  = {Frantz, David and Wright, Courtney J. and
                         Schaser, Allison J. and Kirik, Deniz and
                         Kristensson, Elias and Berrocal, Edouard},
              title   = {Image post-processing for {SILMAS}: structured
                         illumination light sheet microscopy with axial
                         sweeping},
              journal = {Biomedical Optics Express},
              volume  = {15}, number = {8}, pages = {4943--4962},
              year    = {2024}, doi = {10.1364/BOE.531210}
            }
            ```

    Parameters
    ----------
    data : Tensor
        Intensities, any shape; non-integer dtypes are kept, integers are
        converted to the configured dtype.
    axis : int, default -1
        Propagation axis (depth z, pixel index i increasing with depth).
    pixel_size : float, default 1.0
        Δ, in the length unit of μ⁻¹ (e.g. mm for μ in mm⁻¹).
    passes : {1, 2}, default 2
        p: 2 in the papers (OCT round trip); 1 for single-pass
        illumination (see the note above).
    method : {"vermeer", "smith"}, default "vermeer"
        Estimator: Vermeer et al. (2014), Eq. (17), or the extension of
        Smith et al. (2015), Eqs. (8)–(15).
    background : float, Tensor, SensorEstimate or ShadingEstimate, optional
        Additive term subtracted first (extension). A tensor or an
        estimate's map broadcasts against `data`.
    tv_axis : int, optional
        `method="smith"`: lateral axis of the TV term (the B-scan rows of
        the paper). Default: the other of the last two dims when `axis`
        is one of them.
    eta : float, default 6.0
        `method="smith"`: η ≥ 0 of Eq. (11) (6 in Smith et al.,
        Section II-E).
    epsilon : float, default 1e-3
        `method="smith"`: ε of Eq. (14) (0.001 in Smith et al., chosen
        for the magnitude of their intensities).
    noise_power : float, default 0.0
        `method="smith"`: |c|² of Eq. (9) (user-specified in the paper);
        0 disables the restoration filter.
    max_iter : int, default 2000
        `method="smith"`: iterations of the TV solver (not specified in
        the paper).
    tol : float, default 1e-6
        `method="smith"`: relative duality gap at which the TV solver
        stops (not specified in the paper).

    Returns
    -------
    Tensor
        μ, of the shape of `data`, in units of 1/`pixel_size`.
    """
    data, axis = _prepare(data, axis)
    _check_options(method, None, eta, epsilon, max_iter, tol)
    tv = _other_axis(axis, data.ndim, tv_axis) if method == "smith" else None
    data = _subtract(data, _background_field(background))
    return _mu(
        data,
        axis,
        tv,
        pixel_size=pixel_size,
        passes=passes,
        method=method,
        eta=eta,
        epsilon=epsilon,
        noise_power=noise_power,
        max_iter=max_iter,
        tol=tol,
    )


def compensate(
    data: Tensor,
    *,
    axis: int = -1,
    pixel_size: float = 1.0,
    passes: int = 2,
    method: str = "vermeer",
    background: Background = None,
    max_gain: float | None = None,
    tv_axis: int | None = None,
    eta: float = 6.0,
    epsilon: float = 1e-3,
    noise_power: float = 0.0,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> Tensor:
    """Compensate the attenuation along the propagation axis.

    Under the model of [`depth_resolved`][optix.attenuation.depth_resolved],
    I[i] ∝ μ[i]·exp(−p·∫₀^{z_i} μ), the attenuation up to pixel i is
    undone by

        I_c[i] = I[i] · exp(p·Δ·∑_{j < i} μ[j]).

    Vermeer et al. (2014) and Smith et al. (2015) estimate μ but give no
    compensated intensity; this formula is derived from their model and
    is not part of the papers. With Vermeer's μ (Eq. (17)),
    exp(p·Δ·μ[j]) = T[j]/T[j + 1] with T[i] = ∑_{k ≥ i} I[k], and the
    product telescopes to the exact closed form

        I_c[i] = I[i] · T[0] / T[i],

    which is what is computed (it depends neither on p nor on Δ). Where
    T[i] ≤ 0 (no signal from pixel i on), I_c[i] = 0. With
    `method="smith"`, the cumulated sum of the denoised μ̃ is used. The
    compensated intensity is proportional to μ·L₀ (for fluorescence:
    the local emission under the uniform-emission assumption), so a
    uniform medium gives a flat profile. See
    [`depth_resolved`][optix.attenuation.depth_resolved] for the model,
    its adaptation to fluorescence, the options and the references.

    Extension: `max_gain` caps the gain T[0]/T[i] (or the exponential) to
    limit the amplification of the noise deep in the profile.

    Parameters
    ----------
    data : Tensor
        Intensities.
    axis, pixel_size, passes, method, background
        See [`depth_resolved`][optix.attenuation.depth_resolved].
    max_gain : float, optional
        Upper bound ≥ 1 of the gain (extension; default: none).
    tv_axis, eta, epsilon, noise_power, max_iter, tol
        `method="smith"` options; see
        [`depth_resolved`][optix.attenuation.depth_resolved].

    Returns
    -------
    Tensor
        I_c, of the shape and dtype of `data` (after background
        subtraction).
    """
    data, axis = _prepare(data, axis)
    _check_options(method, max_gain, eta, epsilon, max_iter, tol)
    tv = _other_axis(axis, data.ndim, tv_axis) if method == "smith" else None
    data = _subtract(data, _background_field(background))
    return _corrected(
        data,
        axis,
        tv,
        max_gain=max_gain,
        pixel_size=pixel_size,
        passes=passes,
        method=method,
        eta=eta,
        epsilon=epsilon,
        noise_power=noise_power,
        max_iter=max_iter,
        tol=tol,
    )


def _place(value: Tensor, layout: Layout, ndim: int) -> Tensor:
    """A `(*group, *spatial)` map as a tensor broadcastable to storage."""
    group, _, spatial = layout.positions(ndim)
    axes = group + spatial
    if value.ndim > len(axes):
        raise ValueError(
            f"background map has {value.ndim} dims, the layout has "
            f"{len(axes)} group and spatial axes"
        )
    value = value.reshape((1,) * (len(axes) - value.ndim) + value.shape)
    order = sorted(range(len(axes)), key=lambda k: axes[k])
    value = value.permute(order)
    shape = [1] * ndim
    for k in order:
        shape[axes[k]] = value.shape[order.index(k)]
    return value.reshape(shape)


class DepthResolved(Estimator):
    """Depth-resolved attenuation along the light-sheet propagation axis.

    The per-pixel attenuation coefficient of Vermeer et al. (2014),
    applied to the fluorescence profiles along the illumination
    propagation axis, and the compensation that follows from it; see
    [`depth_resolved`][optix.attenuation.depth_resolved] for the model,
    its adaptation to fluorescence and the extensions, and
    [`compensate`][optix.attenuation.compensate] for the compensation.

    There are no global parameters to fit:
    [`fit`][optix.attenuation.DepthResolved.fit] only records the
    layout. [`apply`][optix.Estimator.apply] writes the compensated data
    and [`attenuation`][optix.attenuation.DepthResolved.attenuation] the
    μ map. Both stream the data in blocks that hold whole profiles (the
    full extent of the propagation axis, and of `tv_axis` with
    `method="smith"`) and are tiled over the other axes, so arrays larger
    than memory (zarr, memmap) can be processed in one pass.

    !!! references "References"

        1. Vermeer KA, Mo J, Weda JJA, Lemij HG, de Boer JF.
           *Depth-resolved model-based reconstruction of attenuation
           coefficients in optical coherence tomography.* Biomed Opt
           Express **5**(1):322–337, 2014.
           [doi:10.1364/BOE.5.000322](https://doi.org/10.1364/BOE.5.000322)
        2. Smith GT, Dwork N, O'Connor D, Sikora U, Lurie KL, Pauly JM,
           Ellerbee AK. *Automated, depth-resolved estimation of the
           attenuation coefficient from optical coherence tomography
           data.* IEEE Trans Med Imaging **34**(12):2592–2602, 2015.
           [doi:10.1109/TMI.2015.2450197](https://doi.org/10.1109/TMI.2015.2450197)
        3. Frantz D, Wright CJ, Schaser AJ, Kirik D, Kristensson E,
           Berrocal E. *Image post-processing for SILMAS: structured
           illumination light sheet microscopy with axial sweeping.*
           Biomed Opt Express **15**(8):4943–4962, 2024.
           [doi:10.1364/BOE.531210](https://doi.org/10.1364/BOE.531210)

        ??? quote "BibTeX"

            ```bibtex
            @article{vermeer2014depth,
              author  = {Vermeer, Koenraad A. and Mo, Jelena and Weda,
                         Jelmer J. A. and Lemij, Hans G. and
                         de Boer, Johannes F.},
              title   = {Depth-resolved model-based reconstruction of
                         attenuation coefficients in optical coherence
                         tomography},
              journal = {Biomedical Optics Express},
              volume  = {5}, number = {1}, pages = {322--337},
              year    = {2014}, doi = {10.1364/BOE.5.000322}
            }
            @article{smith2015automated,
              author  = {Smith, Gennifer T. and Dwork, Nicholas and
                         O'Connor, Daniel and Sikora, Uzair and
                         Lurie, Kristen L. and Pauly, John M. and
                         Ellerbee, Audrey K.},
              title   = {Automated, depth-resolved estimation of the
                         attenuation coefficient from optical coherence
                         tomography data},
              journal = {IEEE Transactions on Medical Imaging},
              volume  = {34}, number = {12}, pages = {2592--2602},
              year    = {2015}, doi = {10.1109/TMI.2015.2450197}
            }
            @article{frantz2024image,
              author  = {Frantz, David and Wright, Courtney J. and
                         Schaser, Allison J. and Kirik, Deniz and
                         Kristensson, Elias and Berrocal, Edouard},
              title   = {Image post-processing for {SILMAS}: structured
                         illumination light sheet microscopy with axial
                         sweeping},
              journal = {Biomedical Optics Express},
              volume  = {15}, number = {8}, pages = {4943--4962},
              year    = {2024}, doi = {10.1364/BOE.531210}
            }
            ```
    """

    tags = Tags(
        targets=("attenuation",),
        streaming="halo",
        modality=("lightsheet",),
        reference="doi:10.1364/BOE.5.000322",
    )
    paper = {
        "axis": -1,
        "pixel_size": 1.0,
        "passes": 2,
        "method": "vermeer",
        "background": None,
        "max_gain": None,
        "tv_axis": None,
        "eta": 6.0,
        "epsilon": 1e-3,
        "noise_power": 0.0,
        "max_iter": 2000,
        "tol": 1e-6,
    }
    recommended = dict(paper)
    # data-dependent settings: any value is faithful
    _free = ("axis", "pixel_size", "tv_axis", "max_iter", "tol")

    def __init__(
        self,
        *,
        axis: int | str | None = None,
        pixel_size: float | None = None,
        passes: int | None = None,
        method: str | None = None,
        background: Background = None,
        max_gain: float | None = None,
        tv_axis: int | str | None = None,
        eta: float | None = None,
        epsilon: float | None = None,
        noise_power: float | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the estimator.

        Parameters
        ----------
        axis : int or str, default -1
            Propagation axis: a storage position of the data, or an axis
            label of the layout (e.g. `"x"`).
        pixel_size : float, default 1.0
            Pixel size Δ along `axis`, in the length unit of μ⁻¹.
        passes : {1, 2}, default 2
            p of Eq. (17): 2 in the paper, 1 for single-pass
            illumination; see
            [`depth_resolved`][optix.attenuation.depth_resolved].
        method : {"vermeer", "smith"}, default "vermeer"
            `"smith"` selects the extension of Smith et al. (2015).
        background : float, Tensor, SensorEstimate or ShadingEstimate, optional
            Additive term subtracted first (extension). The `offset` /
            `dark` map of an estimate is `(*group, *spatial)` in the
            layout; a plain tensor broadcasts against the data in storage
            order.
        max_gain : float, optional
            Cap ≥ 1 of the compensation gain (extension).
        tv_axis : int or str, optional
            `method="smith"`: lateral axis of the TV term (storage
            position or label). Default: the other spatial axis when the
            layout has two spatial axes, one of which is `axis`.
        eta, epsilon, noise_power, max_iter, tol
            `method="smith"` options; see
            [`depth_resolved`][optix.attenuation.depth_resolved].
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        if o["method"] != "smith":
            free = self._free + ("eta", "epsilon", "noise_power")
        else:
            free = self._free
        self.is_faithful = o["method"] == "vermeer" and all(
            o[name] is None
            if self.paper[name] is None
            else o[name] == self.paper[name]
            for name in o
            if name not in free
        )
        _check_options(
            o["method"], o["max_gain"], o["eta"], o["epsilon"],
            o["max_iter"], o["tol"],
        )  # fmt: skip
        if isinstance(o["passes"], bool) or o["passes"] not in (1, 2):
            raise ValueError(f"passes must be 1 or 2, got {o['passes']!r}")
        if not o["pixel_size"] > 0:
            raise ValueError(f"pixel_size must be > 0, got {o['pixel_size']}")

    def fit(
        self,
        data: ArrayLike | None = None,
        *,
        layout: Layout | None = None,
        **kwargs: object,
    ) -> Self:
        """No-op: there are no global parameters to estimate.

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
        DepthResolved
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

    def _position(self, axis: int | str, layout: Layout, ndim: int) -> int:
        if isinstance(axis, str):
            if layout.axes is None or axis not in layout.axes:
                raise ValueError(
                    f"axis {axis!r} is not a label of the layout {layout!r}"
                )
            return layout.axes.index(axis)
        if isinstance(axis, bool) or not -ndim <= int(axis) < ndim:
            raise ValueError(f"axis {axis!r} is out of range for {ndim}D data")
        return int(axis) % ndim

    def _axes(self, layout: Layout, ndim: int) -> tuple[int, int | None]:
        o = self.options
        axis = self._position(o["axis"], layout, ndim)
        if o["method"] != "smith":
            return axis, None
        if o["tv_axis"] is not None:
            tv = self._position(o["tv_axis"], layout, ndim)
            if tv == axis:
                raise ValueError("tv_axis must differ from the axis")
            return axis, tv
        _, _, spatial = layout.positions(ndim)
        if len(spatial) != 2 or axis not in spatial:
            raise ValueError(
                "tv_axis must be given unless the layout has two spatial "
                "axes, one of which is the propagation axis"
            )
        return axis, spatial[0] if spatial[1] == axis else spatial[1]

    def _background(self, layout: Layout, ndim: int) -> Tensor | float | None:
        bg = self.options["background"]
        value = _background_field(bg)
        if isinstance(bg, Estimate):
            value = _place(value, layout, ndim)
        elif isinstance(value, Tensor) and value.ndim > ndim:
            raise ValueError(
                f"background has {value.ndim} dims, the data have {ndim}"
            )
        if isinstance(value, Tensor):
            value = value.reshape((1,) * (ndim - value.ndim) + value.shape)
        return value

    def _stream(
        self,
        data: ArrayLike,
        out: WritableArrayLike | None,
        compute: Callable[[Tensor, int, int | None], Tensor],
        overhead: float,
    ) -> WritableArrayLike:
        raw = data.array if isinstance(data, Labeled) else data
        shape = tuple(int(s) for s in raw.shape)
        ndim = len(shape)
        layout = self._layout_of(data)
        axis, tv = self._axes(layout, ndim)
        background = self._background(layout, ndim)
        out = prepare_out(out, shape, like=raw, dtype=self.dtype)
        backend = resolve_backend(self.dtype, self.device, like=raw)
        whole = {axis} if tv is None else {axis, tv}
        block = _block_shape(
            shape,
            whole,
            get_config().memory / (backend["dtype"].itemsize * overhead),
            chunks_of(out),
        )
        chunks = blocks(
            raw,
            block,
            0,
            layout=Layout(spatial=tuple(range(ndim))),
            contiguous=True,
            align=chunks_of(out),
            **backend,
        )
        for chunk in chunks:
            x = chunk.data
            if isinstance(background, Tensor):
                index = tuple(
                    slice(None) if n == 1 else i
                    for i, n in zip(chunk.index, background.shape, strict=True)
                )
                x = _subtract(x, background[index])
            else:
                x = _subtract(x, background)
            chunk.store(out, compute(x, axis, tv))
        flush(out)
        return out

    def _kernel_options(self) -> dict[str, object]:
        o = self.options
        return {
            k: o[k]
            for k in (
                "pixel_size", "passes", "method", "eta", "epsilon",
                "noise_power", "max_iter", "tol",
            )
        }  # fmt: skip

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        options = self._kernel_options()
        max_gain = self.options["max_gain"]

        def compute(x: Tensor, axis: int, tv: int | None) -> Tensor:
            return _corrected(x, axis, tv, max_gain=max_gain, **options)

        return self._stream(data, out, compute, overhead=8.0)

    def attenuation(
        self, data: ArrayLike, out: WritableArrayLike | None = None
    ) -> WritableArrayLike:
        """Per-pixel attenuation coefficient μ of the data.

        Parameters
        ----------
        data : array-like or Labeled
            Data, in the layout of the fit (unless labelled otherwise).
        out : writable array-like, optional
            Output of the shape of `data` (default: a new tensor).

        Returns
        -------
        writable array-like
            μ in units of 1/`pixel_size`, written to `out`.
        """
        if not isinstance(data, Labeled):
            data = self._labeled(data, self.layout)
        options = self._kernel_options()

        def compute(x: Tensor, axis: int, tv: int | None) -> Tensor:
            return _mu(x, axis, tv, **options)

        with torch.no_grad():
            return self._stream(data, out, compute, overhead=8.0)


def _block_shape(
    shape: tuple[int, ...],
    whole: set[int],
    budget: float,
    align: tuple[int, ...] | None,
) -> list[int]:
    """Block covering `whole` axes entirely, the others halved to fit."""
    block = [max(n, 1) for n in shape]
    free = [a for a in range(len(shape)) if a not in whole]
    while math.prod(block) > budget and any(block[a] > 1 for a in free):
        a = max(free, key=lambda k: block[k])
        block[a] = -(-block[a] // 2)
    if math.prod(block) > budget:
        warnings.warn(
            "Whole profiles exceed the memory budget; processing one "
            "profile (or plane) at a time.",
            RuntimeWarning,
            stacklevel=4,
        )
    if align is not None:
        for a in free:
            c = align[a]
            if c and block[a] >= c:
                block[a] = (block[a] // c) * c
    return block
