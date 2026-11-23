"""Linear deconvolution filters (Wiener, Tikhonov), block-wise with halos."""

import math
import warnings
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from .._estimator import Estimator, Tags
from ..estimates import SensorEstimate
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
from ._fft import next_fast_len, transfer_function
from ._rl import RichardsonLucy

__all__ = [
    "Tikhonov",
    "Wiener",
    "frequencies",
    "laplacian_power",
    "linear_filter",
    "smooth_periodic_extension",
    "tikhonov",
    "wiener",
]

_OVERHEAD = 10.0  # working buffers per loaded voxel (extended grid, spectra)
_BOUNDARY = ("taper", "periodic")
_OPERATOR = ("identity", "laplacian")
#: λ values scanned by generalised cross-validation.
_GCV_GRID = torch.logspace(-10, 2, 49, dtype=torch.float64)


# -- building blocks ----------------------------------------------------------
def smooth_periodic_extension(
    data: Tensor, grid: Sequence[int], mirror: int | Sequence[int] = 0
) -> Tensor:
    """Extend a block to a larger grid so that it is smooth when wrapped.

    Along each of the trailing `len(grid)` dims, in turn, the block
    x₀ … x_{L−1} is continued past both of its ends by mirror reflection
    over m samples (x_{L−1+t} = x_{L−1−t} and x₋ₜ = xₜ, t = 1 … m), and
    the two reflections are joined across the remaining g = N − L − 2m
    samples by a raised-cosine blend from a = x_{L−1−m} to b = xₘ,

        a + w(t) (b − a),   w(t) = (1 − cos(π t / (g + 1))) / 2,
        t = 1 … g.

    The left reflection is stored at the end of the grid (index −t), so
    the periodic continuation of the result is continuous everywhere and
    the samples within m of the block edges see a mirror image of the
    data rather than a jump: a filter applied by FFT does not ring at the
    edges (apodisation of the boundary).

    Parameters
    ----------
    data : Tensor
        Block `(..., *L)`.
    grid : sequence of int
        Extended shape N ≥ L + 2m of the trailing dims.
    mirror : int or sequence of int
        Reflection width m per dim, reduced to L − 1 for short blocks.

    Returns
    -------
    Tensor
        `(..., *N)`, equal to `data` on its first L entries along each dim.
    """
    grid = tuple(int(n) for n in grid)
    ndim = len(grid)
    if isinstance(mirror, int):
        mirror = (mirror,) * ndim
    x = data
    for i, n in enumerate(grid):
        dim = x.ndim - ndim + i
        length = x.shape[dim]
        m = max(min(int(mirror[i]), length - 1), 0)
        gap = n - length - 2 * m
        if gap < 0:
            raise ValueError(f"grid {grid} is smaller than the extended block")
        right = x.narrow(dim, length - 1 - m, m).flip(dim)
        left = x.narrow(dim, 1, m).flip(dim)
        parts = [x, right]
        if gap > 0:
            t = torch.arange(1, gap + 1, dtype=x.dtype, device=x.device)
            w = (1 - torch.cos(math.pi * t / (gap + 1))) / 2
            w = w.reshape((gap,) + (1,) * (x.ndim - dim - 1))
            a = x.narrow(dim, length - 1 - m, 1)
            b = x.narrow(dim, m, 1)
            parts.append(a + w * (b - a))
        x = torch.cat(parts + [left], dim=dim)
    return x


def frequencies(
    grid: Sequence[int],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[Tensor, ...]:
    """Frequencies ν (cycles per pixel) of the half spectrum of a grid.

    Parameters
    ----------
    grid : sequence of int
        Grid shape N.
    dtype, device : optional
        Backend of the result.

    Returns
    -------
    tuple of Tensor
        One tensor per dim, broadcastable to the shape of
        `torch.fft.rfftn` on the grid, `(*N[:-1], N[-1] // 2 + 1)`, with
        values in [−1/2, 1/2).
    """
    grid = tuple(int(n) for n in grid)
    out = []
    for i, n in enumerate(grid):
        last = i == len(grid) - 1
        f = torch.fft.rfftfreq(n) if last else torch.fft.fftfreq(n)
        shape = [1] * len(grid)
        shape[i] = -1
        out.append(f.to(dtype=dtype, device=device).reshape(shape))
    return tuple(out)


def laplacian_power(freqs: Sequence[Tensor]) -> Tensor:
    """|L(ν)|² of the discrete Laplacian (2d + 1 point stencil).

        L(ν) = −∑ₐ 4 sin²(π νₐ),   |L(ν)|² = (∑ₐ 4 sin²(π νₐ))²,

    the transfer function of the stencil with −2d at the centre and 1 at
    each of the 2d nearest neighbours (unit pixel spacing).

    Parameters
    ----------
    freqs : sequence of Tensor
        Frequencies per dim, as returned by
        [`frequencies`][optix.deconv.utils.frequencies].

    Returns
    -------
    Tensor
        |L(ν)|², broadcast over the frequency grid.
    """
    total = sum(4 * torch.sin(math.pi * f).square() for f in freqs)
    return torch.as_tensor(total).square()


def linear_filter(otf: Tensor, penalty: Tensor | float) -> Tensor:
    """Spectrum of the regularised inverse filter.

        R(ν) = H*(ν) / (|H(ν)|² + P(ν)),

    with R = 0 where the denominator vanishes (frequencies at which
    neither the PSF nor the penalty carries information).

    Parameters
    ----------
    otf : Tensor
        Transfer function H (complex).
    penalty : Tensor or float
        P(ν) ≥ 0, broadcastable to `otf`: K(ν) for the Wiener filter,
        λ |L(ν)|² for Tikhonov regularisation.

    Returns
    -------
    Tensor
        R, complex, shaped like `otf`.
    """
    power = otf.real.square() + otf.imag.square()
    den = power + penalty
    tiny = torch.finfo(power.dtype).tiny
    ok = den > tiny
    return torch.where(ok, otf.conj() / den.clamp(min=tiny), 0)


def _rfft_weight(grid: tuple[int, ...], like: Tensor) -> Tensor:
    """Multiplicity of each half-spectrum bin in the full spectrum."""
    n = grid[-1]
    w = torch.ones(n // 2 + 1, dtype=like.dtype, device=like.device)
    w[1 : (n - 1) // 2 + 1] = 2
    return w


# -- estimators ---------------------------------------------------------------
class _LinearDeconvolution(Estimator):
    """Shared machinery of the linear deconvolution filters."""

    _strength_name = ""

    # Reused from RichardsonLucy: PSF and halo checks, layouts, the camera
    # model (offset and gain, data converted to electrons).
    _halo = RichardsonLucy._halo
    _labeled_for = RichardsonLucy._labeled_for
    _maps = RichardsonLucy._maps
    _region = staticmethod(RichardsonLucy._region)
    _prepare = RichardsonLucy._prepare
    _init_from = RichardsonLucy._init_from

    def _setup(self, psf: Tensor | None, faithful: bool, params: dict) -> None:
        self._resolve(faithful, params)
        self._faithful_options = self.is_faithful
        self.psf = psf
        self.sensor: SensorEstimate | None = None
        self.strength: float | Callable | None = None
        opts = self.options
        if psf is not None and psf.ndim not in (2, 3):
            raise ValueError(f"psf must be 2-D or 3-D, got {psf.ndim} dims")
        if opts["boundary"] not in _BOUNDARY:
            raise ValueError(f"boundary must be one of {_BOUNDARY}")

    def _psf(self, backend: dict) -> Tensor:
        if self.psf is None:
            raise ValueError(f"{type(self).__name__} needs a psf")
        psf = torch.as_tensor(self.psf).to(**backend)
        total = psf.sum()
        if not total > 0:
            raise ValueError("psf must have a positive sum")
        return psf / total

    def _grid(self, shape: tuple[int, ...], psf: Tensor) -> tuple[int, ...]:
        if self.options["boundary"] == "periodic":
            return shape
        mirror = self._mirror(psf)
        return tuple(
            next_fast_len(n + 2 * m + k)
            for n, m, k in zip(shape, mirror, psf.shape, strict=True)
        )

    def _mirror(self, psf: Tensor) -> tuple[int, ...]:
        if self.options["boundary"] == "periodic":
            return (0,) * psf.ndim
        return self._halo(psf)

    def _penalty(
        self, freqs: tuple[Tensor, ...], strength: object
    ) -> Tensor | float:
        """P(ν) of the filter for a resolved strength."""
        raise NotImplementedError

    def _filter(
        self, shape: tuple[int, ...], psf: Tensor
    ) -> tuple[tuple[int, ...], Tensor]:
        grid = self._grid(shape, psf)
        if any(n < k for n, k in zip(grid, psf.shape, strict=True)):
            raise ValueError(
                f"block {shape} is smaller than the PSF {tuple(psf.shape)}"
            )
        otf = transfer_function(psf, grid)
        freqs = frequencies(grid, dtype=psf.dtype, device=psf.device)
        return grid, linear_filter(otf, self._penalty(freqs, self.strength))

    def _resolve_strength(self, data: Labeled, backend: dict) -> None:
        self.strength = self.options[self._strength_name]

    # -- estimator protocol ----------------------------------------------
    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        data = self._labeled_for(data)
        self.layout = data.layout
        backend = self._backend(data)
        shape = data.layout.map_shape(data.array)
        self._init_from(init, shape, data)
        psf = self._psf(backend)
        if psf.ndim != len(data.layout.spatial):
            raise ValueError(
                f"The PSF has {psf.ndim} dims but the layout has "
                f"{len(data.layout.spatial)} spatial axes"
            )
        self._resolve_strength(data, backend)

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if self.strength is None:
            raise RuntimeError("Call fit(data) before apply(data)")
        data = self._labeled_for(data)
        backend = self._backend(data)
        psf = self._psf(backend)
        halo = self._halo(psf)
        raw = data.array
        target = out.array if isinstance(out, Labeled) else out
        if out is not None and target is raw:
            raise ValueError("In-place deconvolution is not possible")
        out = prepare_out(out, tuple(data.shape), like=raw, dtype=self.dtype)
        maps = self._maps(data, backend)
        filters: dict[tuple[int, ...], tuple] = {}
        chunks = blocks(
            data,
            self.options["block"],
            halo,
            overhead=_OVERHEAD,
            contiguous=True,
            align=chunks_of(out),
            **backend,
        )
        for chunk in chunks:
            y = self._prepare(chunk, maps).data
            key = tuple(y.shape)
            if key not in filters:
                filters[key] = self._filter(key, psf)
            grid, filt = filters[key]
            dims = tuple(range(-len(grid), 0))
            spec = torch.fft.rfftn(
                smooth_periodic_extension(y, grid, self._mirror(psf)), dim=dims
            )
            f = torch.fft.irfftn(spec * filt, s=grid, dim=dims)
            f = f[tuple(slice(0, n) for n in key)]
            chunk.store(out, f[chunk.inner])
        flush(out)
        return out

    def _extra_state(self) -> dict[str, object]:
        sensor = None if self.sensor is None else self.sensor._payload()
        return {"psf": self.psf, "strength": self.strength, "sensor": sensor}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.psf = state.get("psf")
        self.strength = state.get("strength")
        sensor = state.get("sensor")
        if sensor is not None:
            self.sensor = SensorEstimate._from_payload(sensor)


class Wiener(_LinearDeconvolution):
    """Wiener deconvolution with a known PSF, block by block.

    Model: the image is the blurred object plus noise, y = H f + n, with
    (H f)ᵢ = ∑ⱼ S(i − j) fⱼ and S the point spread function (normalised
    to unit sum, so that |H(ν)| ≤ 1 and H(0) = 1). The Wiener filter is
    the linear estimate of minimum mean squared error for a stationary
    object and noise; in the frequency domain,

        F̂(ν) = H*(ν) Y(ν) / (|H(ν)|² + K(ν)),   K(ν) = Sₙ(ν) / S_f(ν),

    with Sₙ and S_f the power spectra of the noise and of the object.
    When they are unknown, K is a constant noise-to-signal ratio, tuned
    by hand. K → 0 gives the inverse filter Y / H; a larger K suppresses
    the frequencies at which the PSF transmits little, trading
    resolution for noise. With a constant K, the filter equals the
    zero-order Tikhonov filter with λ = K (see
    [`Tikhonov`][optix.deconv.Tikhonov]).

    The result is linear in the data: it may be negative and has noise
    correlated over the PSF scale. It is a fast preview, or a starting
    point for iterative methods such as
    [`RichardsonLucy`][optix.deconv.RichardsonLucy].

    **Boundaries and blocks.** The filter is applied by FFT, which treats
    the image as periodic. With `boundary="taper"` (the default), each
    block (core plus halo) is extended past each edge by its mirror
    image over the halo width, and the two reflections are joined by a
    raised-cosine blend over at least the PSF size (see
    [`smooth_periodic_extension`][optix.deconv.utils.smooth_periodic_extension]),
    so its periodic continuation has no jump and the edges do not ring;
    only the core of each block is kept. The halo is at least the PSF
    size (default: twice). The filter is linear and its impulse response
    decays away from the origin, so the block-wise result approaches the
    whole-image one (with the same halo, which also sets the mirror
    width at the image edges) geometrically as the halo grows; the
    residual difference is larger for weaker regularisation, whose
    impulse response is longer.
    With `boundary="periodic"`, each block is filtered as a periodic
    image of its own, without extension: on a single block (the whole
    image), this exactly inverts a circular blur.

    Images are independent: sample and group axes of the layout are
    iterated over, the spatial axes (as many as the PSF has dims, in
    layout order) are deconvolved, in one pass over the data.

    !!! note "Extensions (off by default)"
        - **Camera model:** `fit(data, init=sensor)` with a
          [`SensorEstimate`][optix.estimates.SensorEstimate] converts the
          data to electrons, (y − offset)/gain, before deconvolution; the
          result is then in electrons.

    !!! references "References"

        1. Gonzalez RC, Woods RE. *Digital Image Processing*, chapter
           "Image Restoration and Reconstruction". Pearson.
        2. Bertero M, Boccacci P. *Introduction to Inverse Problems in
           Imaging.* IOP Publishing, 1998.

        ??? quote "BibTeX"

            ```bibtex
            @book{gonzalez2018digital,
              author    = {Gonzalez, Rafael C. and Woods, Richard E.},
              title     = {Digital Image Processing},
              edition   = {4},
              publisher = {Pearson},
              year      = {2018}
            }
            @book{bertero1998introduction,
              author    = {Bertero, Mario and Boccacci, Patrizia},
              title     = {Introduction to Inverse Problems in Imaging},
              publisher = {IOP Publishing},
              year      = {1998}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import Wiener
    >>> psf = torch.ones(5, 5)
    >>> data = torch.rand(64, 64)
    >>> Wiener(psf, nsr=1e-2).fit(data).apply(data).shape
    torch.Size([64, 64])
    """

    tags = Tags(targets=("deconvolution",), streaming="halo", noise="gaussian")
    paper = {"nsr": 1e-2, "boundary": "taper", "halo": None, "block": None}
    recommended = dict(paper)
    _strength_name = "nsr"

    def __init__(
        self,
        psf: Tensor | None = None,
        *,
        nsr: float | Callable[..., Tensor] | None = None,
        boundary: str | None = None,
        halo: int | Sequence[int] | None = None,
        block: Sequence[int | None] | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the filter.

        Parameters
        ----------
        psf : Tensor
            Point spread function, `(*K)` with 2 or 3 dims, in the order
            of the spatial axes of the layout, centred at index `K // 2`.
            Normalised to unit sum.
        nsr : float or callable
            Noise-to-signal ratio K ≥ 0. A callable is K(ν): it receives
            the frequencies of the block's spectrum (one tensor per
            spatial axis, in cycles per pixel, as returned by
            [`frequencies`][optix.deconv.utils.frequencies]) and returns
            Sₙ(ν)/S_f(ν), broadcastable to them. An estimator with a
            callable `nsr` cannot be saved. Default: 0.01 (not specified
            in the references, where K is tuned by hand).
        boundary : {"taper", "periodic"}
            Extension of each block before the FFT (see above). Default:
            "taper".
        halo : int or sequence of int, optional
            Halo width per spatial axis, at least the PSF size along that
            axis. Default: twice the PSF size.
        block : sequence of int or None, optional
            Core block size per spatial axis (None entries: whole
            extent). Default: chosen from the memory budget.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        params = {k: v for k, v in locals().items() if k != "psf"}
        self._setup(psf, faithful, params)
        nsr = self.options["nsr"]
        if not callable(nsr) and not float(nsr) >= 0:
            raise ValueError("nsr must be non-negative or a callable")

    def _penalty(
        self, freqs: tuple[Tensor, ...], strength: object
    ) -> Tensor | float:
        if not callable(strength):
            return float(strength)
        k = torch.as_tensor(strength(freqs)).to(freqs[0])
        if (k < 0).any():
            raise ValueError("nsr(ν) must be non-negative")
        return k


class Tikhonov(_LinearDeconvolution):
    """Tikhonov-regularised deconvolution with a known PSF, block by block.

    Model: y = H f + n, with (H f)ᵢ = ∑ⱼ S(i − j) fⱼ and S the point
    spread function (normalised to unit sum). The estimate minimises

        ‖y − H f‖² + λ ‖L f‖²,

    with L the identity (zero-order Tikhonov regularisation, a penalty
    on the energy of f) or the discrete Laplacian (a penalty on its
    roughness, as in constrained least-squares filtering). Both H and L
    are convolutions, so the minimiser is, in the frequency domain,

        F̂(ν) = H*(ν) Y(ν) / (|H(ν)|² + λ |L(ν)|²),

    with |L(ν)|² = 1 for the identity and (∑ₐ 4 sin²(π νₐ))² for the
    Laplacian (see [`laplacian_power`][optix.deconv.utils.laplacian_power]).
    λ → 0 gives the inverse filter; a larger λ suppresses noise at the
    cost of resolution. With the Laplacian, the mean of the image
    (ν = 0) is not penalised and is preserved; with the identity, it is
    scaled by 1/(1 + λ).

    The result is linear in the data and may be negative.

    **Boundaries and blocks.** The filter is applied by FFT, which treats
    the image as periodic. With `boundary="taper"` (the default), each
    block (core plus halo) is extended past each edge by its mirror
    image over the halo width, and the two reflections are joined by a
    raised-cosine blend over at least the PSF size (see
    [`smooth_periodic_extension`][optix.deconv.utils.smooth_periodic_extension]),
    so its periodic continuation has no jump and the edges do not ring;
    only the core of each block is kept. The halo is at least the PSF
    size (default: twice). The filter is linear and its impulse response
    decays away from the origin, so the block-wise result approaches the
    whole-image one (with the same halo, which also sets the mirror
    width at the image edges) geometrically as the halo grows; the
    residual difference is larger for weaker regularisation, whose
    impulse response is longer.
    With `boundary="periodic"`, each block is filtered as a periodic
    image of its own, without extension: on a single block (the whole
    image), this exactly inverts a circular blur.

    Images are independent: sample and group axes of the layout are
    iterated over, the spatial axes (as many as the PSF has dims, in
    layout order) are deconvolved, in one pass over the data.

    !!! note "Extensions (off by default)"
        - **Camera model:** `fit(data, init=sensor)` with a
          [`SensorEstimate`][optix.estimates.SensorEstimate] converts the
          data to electrons, (y − offset)/gain, before deconvolution; the
          result is then in electrons.
        - **Automatic λ:** `lam="gcv"` picks λ by generalised
          cross-validation (Golub, Heath & Wahba 1979) at `fit`: the
          influence matrix A has eigenvalues
          r(ν) = |H(ν)|² / (|H(ν)|² + λ |L(ν)|²) on each block's FFT grid,
          and

              V(λ) = n ∑ |(1 − r) Y|² / (∑ (1 − r))²,

          summed over the frequencies of all blocks (the blocks without
          halo, each extended as above) and all images, is minimised
          over λ ∈ [10⁻¹⁰, 10²] on a grid of four values per decade.
          This selection rule is not part of the references above; it
          is a generic choice. The chosen value is `strength`; a value
          at either end of the grid is flagged with a warning.

    !!! references "References"

        1. Gonzalez RC, Woods RE. *Digital Image Processing*, chapter
           "Image Restoration and Reconstruction". Pearson.
        2. Bertero M, Boccacci P. *Introduction to Inverse Problems in
           Imaging.* IOP Publishing, 1998.

        3. Golub GH, Heath M, Wahba G. *Generalized cross-validation
           as a method for choosing a good ridge parameter.*
           Technometrics **21**(2):215-223, 1979.
           [doi:10.1080/00401706.1979.10489751](https://doi.org/10.1080/00401706.1979.10489751)

        ??? quote "BibTeX"

            ```bibtex
            @book{gonzalez2018digital,
              author    = {Gonzalez, Rafael C. and Woods, Richard E.},
              title     = {Digital Image Processing},
              edition   = {4},
              publisher = {Pearson},
              year      = {2018}
            }
            @book{bertero1998introduction,
              author    = {Bertero, Mario and Boccacci, Patrizia},
              title     = {Introduction to Inverse Problems in Imaging},
              publisher = {IOP Publishing},
              year      = {1998}
            }
            @article{golub1979generalized,
              author  = {Golub, Gene H. and Heath, Michael and
                         Wahba, Grace},
              title   = {Generalized Cross-Validation as a Method for
                         Choosing a Good Ridge Parameter},
              journal = {Technometrics},
              volume  = {21},
              number  = {2},
              pages   = {215--223},
              year    = {1979},
              doi     = {10.1080/00401706.1979.10489751}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import Tikhonov
    >>> psf = torch.ones(5, 5)
    >>> data = torch.rand(64, 64)
    >>> Tikhonov(psf, lam=1e-2, operator="laplacian").fit(data).apply(
    ...     data
    ... ).shape
    torch.Size([64, 64])
    """

    tags = Tags(targets=("deconvolution",), streaming="halo", noise="gaussian")
    paper = {
        "lam": 1e-2,
        "operator": "identity",
        "boundary": "taper",
        "halo": None,
        "block": None,
    }
    recommended = dict(paper)
    _strength_name = "lam"

    def __init__(
        self,
        psf: Tensor | None = None,
        *,
        lam: float | str | None = None,
        operator: str | None = None,
        boundary: str | None = None,
        halo: int | Sequence[int] | None = None,
        block: Sequence[int | None] | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the filter.

        Parameters
        ----------
        psf : Tensor
            Point spread function, `(*K)` with 2 or 3 dims, in the order
            of the spatial axes of the layout, centred at index `K // 2`.
            Normalised to unit sum.
        lam : float or "gcv"
            Regularisation weight λ ≥ 0, or `"gcv"` (extension) to choose
            it by generalised cross-validation. Default: 0.01 (not
            specified in the references, where λ is tuned by hand).
        operator : {"identity", "laplacian"}
            Regularisation operator L. Default: "identity".
        boundary : {"taper", "periodic"}
            Extension of each block before the FFT (see above). Default:
            "taper".
        halo : int or sequence of int, optional
            Halo width per spatial axis, at least the PSF size along that
            axis. Default: twice the PSF size.
        block : sequence of int or None, optional
            Core block size per spatial axis (None entries: whole
            extent). Default: chosen from the memory budget.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        params = {k: v for k, v in locals().items() if k != "psf"}
        self._setup(psf, faithful, params)
        if self.options["operator"] not in _OPERATOR:
            raise ValueError(f"operator must be one of {_OPERATOR}")
        lam = self.options["lam"]
        if lam != "gcv" and (isinstance(lam, str) or not float(lam) >= 0):
            raise ValueError('lam must be non-negative or "gcv"')

    def _power(self, freqs: tuple[Tensor, ...]) -> Tensor | float:
        """|L(ν)|²."""
        if self.options["operator"] == "laplacian":
            return laplacian_power(freqs)
        return 1.0

    def _penalty(
        self, freqs: tuple[Tensor, ...], strength: object
    ) -> Tensor | float:
        return float(strength) * self._power(freqs)

    def _resolve_strength(self, data: Labeled, backend: dict) -> None:
        lam = self.options["lam"]
        if lam != "gcv":
            self.strength = float(lam)
            return
        self.strength = self._gcv(data, backend)
        self.history.extras["lam"] = [self.strength]

    def _gcv(self, data: Labeled, backend: dict) -> float:
        """λ minimising the generalised cross-validation score."""
        psf = self._psf(backend)
        maps = self._maps(data, backend)
        lams = _GCV_GRID
        num = torch.zeros_like(lams)
        trace = torch.zeros_like(lams)
        count = 0
        cache: dict[tuple[int, ...], tuple] = {}
        chunks = blocks(
            data, self.options["block"], 0, overhead=_OVERHEAD, **backend
        )
        for chunk in chunks:
            y = self._prepare(chunk, maps).data
            key = tuple(y.shape)
            if key not in cache:
                grid = self._grid(key, psf)
                otf = transfer_function(psf, grid)
                freqs = frequencies(grid, dtype=psf.dtype, device=psf.device)
                h2 = (otf.real.square() + otf.imag.square()).double()
                power = torch.as_tensor(self._power(freqs)).double()
                weight = _rfft_weight(grid, h2)
                cache[key] = (grid, h2, power.to(h2.device), weight)
            grid, h2, power, weight = cache[key]
            dims = tuple(range(-len(grid), 0))
            spec = torch.fft.rfftn(
                smooth_periodic_extension(y, grid, self._mirror(psf)), dim=dims
            )
            y2 = (spec.real.square() + spec.imag.square()).double()
            n = math.prod(grid)
            for i, lam in enumerate(lams.tolist()):
                den = h2 + lam * power
                tiny = torch.finfo(den.dtype).tiny
                r = torch.where(den > tiny, h2 / den.clamp(min=tiny), 0)
                rest = (1 - r) * weight
                num[i] += float((rest * (1 - r) * y2).sum()) / n
                trace[i] += float(rest.sum())
            count += n
        score = count * num / trace.square().clamp(min=1e-300)
        best = int(torch.argmin(score))
        if best in (0, len(lams) - 1):
            warnings.warn(
                "GCV chose λ at the end of its search range "
                f"({float(lams[best]):.3g})",
                RuntimeWarning,
                stacklevel=4,
            )
        return float(lams[best])


def wiener(
    data: ArrayLike,
    psf: Tensor,
    *,
    nsr: float | Callable[..., Tensor] | None = None,
    out: WritableArrayLike | None = None,
    sensor: SensorEstimate | None = None,
    layout: Layout | None = None,
    **options: object,
) -> WritableArrayLike:
    """Wiener deconvolution of `data` with a known PSF.

    Shortcut for `Wiener(psf, nsr=nsr, **options).fit(data,
    init=sensor).apply(data, out)`; see [`Wiener`][optix.deconv.Wiener]
    for the filter, F̂ = H* Y / (|H|² + K), and its options.

    Parameters
    ----------
    data : array-like or Labeled
        Images; `(*samples, *spatial)` without a layout, with as many
        spatial dims as the PSF.
    psf : Tensor
        Point spread function, centred at index `K // 2`.
    nsr : float or callable, optional
        Noise-to-signal ratio K, constant or K(ν).
    out : writable array-like, optional
        Output (tensor, numpy, zarr, ...). Default: a new tensor.
    sensor : SensorEstimate, optional
        Camera model; the data are converted to electrons.
    layout : Layout, optional
        Axis roles.
    **options
        Options of [`Wiener`][optix.deconv.Wiener] (`boundary`, `halo`,
        `block`, ...).

    Returns
    -------
    writable array-like
        `out`, the deconvolved images.
    """
    est = Wiener(psf, nsr=nsr, layout=layout, **options)
    est.fit(data, init=sensor)
    return est.apply(data, out)


def tikhonov(
    data: ArrayLike,
    psf: Tensor,
    *,
    lam: float | str | None = None,
    operator: str | None = None,
    out: WritableArrayLike | None = None,
    sensor: SensorEstimate | None = None,
    layout: Layout | None = None,
    **options: object,
) -> WritableArrayLike:
    """Tikhonov-regularised deconvolution of `data` with a known PSF.

    Shortcut for `Tikhonov(psf, lam=lam, operator=operator,
    **options).fit(data, init=sensor).apply(data, out)`; see
    [`Tikhonov`][optix.deconv.Tikhonov] for the filter,
    F̂ = H* Y / (|H|² + λ |L|²), and its options.

    Parameters
    ----------
    data : array-like or Labeled
        Images; `(*samples, *spatial)` without a layout, with as many
        spatial dims as the PSF.
    psf : Tensor
        Point spread function, centred at index `K // 2`.
    lam : float or "gcv", optional
        Regularisation weight λ.
    operator : {"identity", "laplacian"}, optional
        Regularisation operator L.
    out : writable array-like, optional
        Output (tensor, numpy, zarr, ...). Default: a new tensor.
    sensor : SensorEstimate, optional
        Camera model; the data are converted to electrons.
    layout : Layout, optional
        Axis roles.
    **options
        Options of [`Tikhonov`][optix.deconv.Tikhonov] (`boundary`,
        `halo`, `block`, ...).

    Returns
    -------
    writable array-like
        `out`, the deconvolved images.
    """
    est = Tikhonov(psf, lam=lam, operator=operator, layout=layout, **options)
    est.fit(data, init=sensor)
    return est.apply(data, out)
