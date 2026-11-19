"""Richardson–Lucy deconvolution, block-wise with halos."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .._estimator import Estimator, Tags
from ..estimates import SensorEstimate
from ..io import (
    ArrayLike,
    Chunk,
    Labeled,
    Layout,
    WritableArrayLike,
    blocks,
    chunks_of,
    flush,
    prepare_out,
)
from ._fft import PaddedConvolution

__all__ = [
    "RichardsonLucy",
    "chi2_probability",
    "richardson_lucy",
    "rl_block",
    "rl_step",
]

_OVERHEAD = 12.0  # working buffers per loaded voxel (FFT grid, ratios, ...)


def rl_step(
    f: Tensor,
    data: Tensor,
    op: PaddedConvolution,
    *,
    inv_norm: Tensor,
    background: Tensor | float = 0.0,
    damping: float = 1.0,
    mu: Tensor | None = None,
) -> Tensor:
    """One Richardson–Lucy update on a block.

        f ← f · (1 − ε + ε c),   c = Hᵀ[y / (H f + b)] / Hᵀ1,

    with ε = `damping` (ε = 1: the plain update f ← f · c).

    Parameters
    ----------
    f : Tensor
        Current estimate on the object grid `op.grid`, non-negative.
    data : Tensor
        Observed block y, `op.shape`.
    op : PaddedConvolution
        Blur operator H.
    inv_norm : Tensor
        1 / Hᵀ1 on the object grid, 0 where Hᵀ1 vanishes (object pixels
        that do not reach the block).
    background : Tensor or float
        Background b added to H f (broadcasts to the block).
    damping : float
        Fraction ε ∈ (0, 1] of the multiplicative correction applied.
    mu : Tensor, optional
        H f + b if already computed (e.g. for a stopping test).

    Returns
    -------
    Tensor
        Updated estimate, non-negative.
    """
    if mu is None:
        mu = _expected(f, op, background)
    c = op.adjoint(data / mu) * inv_norm
    if damping != 1.0:
        c = 1.0 - damping + damping * c
    return (f * c).clamp_(min=0)


def _expected(
    f: Tensor, op: PaddedConvolution, background: Tensor | float
) -> Tensor:
    """H f + b, floored at a tiny positive value (FFT round-off)."""
    mu = op.forward(f) + background
    eps = torch.finfo(mu.dtype)
    floor = max(float(mu.max()) * eps.eps, eps.tiny)
    return mu.clamp_(min=floor)


def chi2_probability(data: Tensor, mu: Tensor) -> float:
    """P(χ² > χ²_obs) of Pearson's statistic for counts.

        χ²_obs = ∑ᵢ (yᵢ − μᵢ)² / μᵢ,   ν = n − 1 degrees of freedom,

    with n the number of pixels (Lucy 1974, §IV).

    Parameters
    ----------
    data : Tensor
        Observed counts y.
    mu : Tensor
        Expected counts μ > 0, same shape.

    Returns
    -------
    float
        Upper tail probability of the χ²(ν) distribution.
    """
    chi2 = ((data - mu).square() / mu).sum().double()
    nu = torch.tensor(max(data.numel() - 1, 1), dtype=torch.float64)
    return float(torch.special.gammaincc(nu / 2, chi2.cpu() / 2))


def rl_block(
    data: Tensor,
    op: PaddedConvolution,
    start: float | Tensor,
    *,
    n_iter: int | None = None,
    alpha: float = 0.05,
    max_iter: int = 100,
    background: Tensor | float = 0.0,
    damping: float = 1.0,
) -> tuple[Tensor, int]:
    """Richardson–Lucy iterations on one block.

    Parameters
    ----------
    data : Tensor
        Observed block y (counts), `op.shape`.
    op : PaddedConvolution
        Blur operator H.
    start : float or Tensor
        Starting estimate f⁽⁰⁾ > 0: a constant or a tensor on `op.grid`.
    n_iter : int, optional
        Fixed number of iterations. None: stop at the first r for which
        P(χ² > χ²(f⁽ʳ⁾)) ≥ `alpha` (see
        [`chi2_probability`][optix.deconv.utils.chi2_probability]), or at
        `max_iter`.
    alpha : float
        Significance level of the χ² stopping test.
    max_iter : int
        Iteration cap of the χ² stopping rule.
    background : Tensor or float
        Background b of the model y ~ Poisson(H f + b).
    damping : float
        Fraction ε ∈ (0, 1] of the correction applied (see
        [`rl_step`][optix.deconv.utils.rl_step]).

    Returns
    -------
    f : Tensor
        Estimate on the object grid `op.grid`; `op.crop(f)` is the
        block.
    n : int
        Number of updates applied.
    """
    norm = op.norm
    valid = norm > norm.max() * torch.finfo(norm.dtype).eps * 16
    tiny = torch.finfo(norm.dtype).tiny
    inv_norm = torch.where(valid, 1 / norm.clamp(min=tiny), 0)
    f = torch.zeros_like(norm)
    f = torch.where(valid, f + start, f)
    limit = max_iter if n_iter is None else n_iter
    n = 0
    while n < limit:
        mu = _expected(f, op, background)
        if n_iter is None and chi2_probability(data, mu) >= alpha:
            break
        f = rl_step(
            f,
            data,
            op,
            inv_norm=inv_norm,
            background=background,
            damping=damping,
            mu=mu,
        )
        n += 1
    return f, n


@dataclass
class _Block:
    """Per-block inputs: data in counts and background."""

    data: Tensor
    background: Tensor | float


class RichardsonLucy(Estimator):
    """Richardson–Lucy deconvolution with a known PSF, block by block.

    Model: the image y is a Poisson sample of the blurred object,

        y ~ Poisson(H f),   (H f)ᵢ = ∑ⱼ S(i − j) fⱼ,

    with S the point spread function (normalised to unit sum). The
    iteration of Richardson (1972) and Lucy (1974) is

        f⁽ʳ⁺¹⁾ = f⁽ʳ⁾ · Hᵀ[y / (H f⁽ʳ⁾)] / Hᵀ1.

    It keeps f ≥ 0, increases the Poisson likelihood of y at every
    step, and conserves the flux: ∑ᵢ (H f⁽ʳ⁾)ᵢ = ∑ᵢ yᵢ for r ≥ 1.

    **Starting estimate.** The uniform image of the same integrated
    intensity as the data (Richardson's f⁽⁰⁾ = W/I; Lucy: a smooth,
    non-negative image with the data's integrated intensity), one level
    per image.

    **Stopping rule.** Lucy (1974, §IV): stop as soon as the χ² test of
    the fit, P(χ² > χ²_obs) with χ²_obs = ∑ᵢ (yᵢ − μᵢ)²/μᵢ, μ = H f⁽ʳ⁾,
    and ν = n − 1 degrees of freedom, is not small (≥ `alpha` = 0.05).
    The test assumes y in counts (photons or electrons). A fixed number
    of iterations (Richardson shows 10, 20 and 30) is selected with
    `n_iter`.

    **Boundaries and blocks.** The object is estimated on a domain that
    extends past the observed region by the PSF support, so light from
    outside the field of view is modelled rather than piled up at the
    edges. H crops the blurred object to the observed region and the
    normaliser Hᵀ1, which falls below 1 near the edges of that region, is
    computed on the padded grid. Large inputs are processed in blocks
    with a halo; each block is deconvolved as an image of its own (core
    plus halo), and only its core is kept. The halo is at least the PSF
    size (default: twice), and the seams fade with the halo width; along
    an image edge, the block-wise and whole-image results coincide.

    Images are independent: sample and group axes of the layout are
    iterated over, the spatial axes (as many as the PSF has dims, in
    layout order) are deconvolved.

    !!! note "Extensions (off by default)"
        - **Camera model:** `fit(data, init=sensor)` with a
          [`SensorEstimate`][optix.estimates.SensorEstimate] converts the
          data to electrons, (y − offset)/gain, before deconvolution; the
          result is then in electrons.
        - **Background:** `background=b` uses y ~ Poisson(H f + b), and
          the update Hᵀ[y / (H f + b)] / Hᵀ1.
        - **Poisson–Gaussian noise:** `poisson_gaussian=True` uses the
          large-read-noise approximation of Snyder et al. (1993, §3.C):
          the Gaussian read noise, of mean m and variance σ², is replaced
          by a Poisson variable of mean σ², so that the data, in electrons
          and shifted to m = σ², follow
          y − m + σ² ~ Poisson(H f + b + σ²), and the update is their
          Eq. (15) with the background μ₀ = b replaced by b + σ²:
          f ← f · Hᵀ[(y − m + σ²) / (H f + b + σ²)] / Hᵀ1. Here m is the
          sensor offset, y − m is converted to electrons, (y − m)/gain,
          and σ² is the read-noise variance in electrons² (variance/gain²).
          y − m + σ² is clipped at 0 (not specified in the paper). The flat
          field β(j) of Snyder et al. (Eqs. 6 and 15) is taken as uniform,
          β = 1; data divided by a flat field beforehand are no longer
          Poisson counts.
        - **Damping:** `damping=ε` applies a fraction ε ∈ (0, 1] of the
          correction, f ← f + ε(f·c − f), as in Lucy (1974, §III), where
          it is used to prove that the likelihood increases; Lucy
          recommends ε = 1.

    !!! references "References"

        1. Richardson WH. *Bayesian-based iterative method of image
           restoration.* J Opt Soc Am **62**(1):55-59, 1972.
           [doi:10.1364/JOSA.62.000055](https://doi.org/10.1364/JOSA.62.000055)
        2. Lucy LB. *An iterative technique for the rectification of
           observed distributions.* Astron J **79**(6):745-754, 1974.
           [doi:10.1086/111605](https://doi.org/10.1086/111605)
        3. Snyder DL, Hammoud AM, White RL. *Image recovery from data
           acquired with a charge-coupled-device camera.* J Opt Soc Am A
           **10**(5):1014-1023, 1993.
           [doi:10.1364/JOSAA.10.001014](https://doi.org/10.1364/JOSAA.10.001014)

        ??? quote "BibTeX"

            ```bibtex
            @article{richardson1972bayesian,
              author  = {Richardson, William Hadley},
              title   = {Bayesian-Based Iterative Method of Image
                         Restoration},
              journal = {Journal of the Optical Society of America},
              volume  = {62},
              number  = {1},
              pages   = {55--59},
              year    = {1972},
              doi     = {10.1364/JOSA.62.000055}
            }
            @article{lucy1974iterative,
              author  = {Lucy, L. B.},
              title   = {An iterative technique for the rectification of
                         observed distributions},
              journal = {The Astronomical Journal},
              volume  = {79},
              number  = {6},
              pages   = {745--754},
              year    = {1974},
              doi     = {10.1086/111605}
            }
            @article{snyder1993image,
              author  = {Snyder, Donald L. and Hammoud, Abed M. and
                         White, Richard L.},
              title   = {Image recovery from data acquired with a
                         charge-coupled-device camera},
              journal = {Journal of the Optical Society of America A},
              volume  = {10},
              number  = {5},
              pages   = {1014--1023},
              year    = {1993},
              doi     = {10.1364/JOSAA.10.001014}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import RichardsonLucy
    >>> psf = torch.ones(5, 5)
    >>> data = torch.poisson(torch.full((64, 64), 10.0))
    >>> rl = RichardsonLucy(psf, n_iter=10).fit(data)
    >>> rl.apply(data).shape
    torch.Size([64, 64])
    """

    tags = Tags(
        targets=("deconvolution",),
        streaming="halo",
        noise="poisson",
        reference="doi:10.1364/JOSA.62.000055, doi:10.1086/111605",
    )
    paper = {
        "n_iter": None,
        "alpha": 0.05,
        "max_iter": 100,
        "background": None,
        "poisson_gaussian": False,
        "damping": 1.0,
        "halo": None,
        "block": None,
    }
    recommended = dict(paper)

    def __init__(
        self,
        psf: Tensor | None = None,
        *,
        n_iter: int | None = None,
        alpha: float | None = None,
        max_iter: int | None = None,
        background: float | Tensor | None = None,
        poisson_gaussian: bool | None = None,
        damping: float | None = None,
        halo: int | Sequence[int] | None = None,
        block: Sequence[int | None] | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the deconvolution.

        Parameters
        ----------
        psf : Tensor
            Point spread function, `(*K)` with 2 or 3 dims, in the order
            of the spatial axes of the layout, centred at index `K // 2`.
            Normalised to unit sum.
        n_iter : int, optional
            Fixed number of iterations. Default (Lucy 1974): None, the χ²
            stopping rule.
        alpha : float
            Level of the χ² stopping test: stop once P(χ² > χ²_obs) ≥
            `alpha`. Default (Lucy 1974): 0.05.
        max_iter : int
            Iteration cap of the χ² stopping rule. Default: 100 (not
            specified in the papers).
        background : float or Tensor, optional
            Extension. Background b, in data units (offset excluded);
            a scalar or a map that broadcasts to `(*group, *spatial)`.
            Default: none.
        poisson_gaussian : bool
            Extension. Shifted-Poisson approximation of the Gaussian read
            noise (Snyder et al. 1993, §3.C and Eq. 15), with the offset
            and read-noise variance of the sensor estimate given to
            [`fit`][optix.Estimator.fit]. Default: False.
        damping : float
            Extension. Fraction ε ∈ (0, 1] of the correction applied.
            Default (Lucy 1974): 1.
        halo : int or sequence of int, optional
            Halo width per spatial axis, at least the PSF size along that
            axis. Default: twice the PSF size (not specified in the papers).
        block : sequence of int or None, optional
            Core block size per spatial axis (None entries: whole
            extent). Default: chosen from the memory budget.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        params = {k: v for k, v in locals().items() if k != "psf"}
        self._resolve(faithful, params)
        self._faithful_options = self.is_faithful
        self.psf = psf
        self.sensor: SensorEstimate | None = None
        self.level: Tensor | None = None
        opts = self.options
        if psf is not None and psf.ndim not in (2, 3):
            raise ValueError(f"psf must be 2-D or 3-D, got {psf.ndim} dims")
        if opts["n_iter"] is not None and int(opts["n_iter"]) < 0:
            raise ValueError("n_iter must be non-negative")
        if not 0.0 < float(opts["damping"]) <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        if not 0.0 < float(opts["alpha"]) < 1.0:
            raise ValueError("alpha must be in (0, 1)")

    # -- helpers ---------------------------------------------------------
    def _psf(self, backend: dict) -> Tensor:
        if self.psf is None:
            raise ValueError("RichardsonLucy needs a psf")
        psf = torch.as_tensor(self.psf).to(**backend)
        if (psf < 0).any():
            raise ValueError("psf must be non-negative")
        total = psf.sum()
        if not total > 0:
            raise ValueError("psf must have a positive sum")
        return psf / total

    def _halo(self, psf: Tensor) -> tuple[int, ...]:
        halo = self.options["halo"]
        need = tuple(psf.shape)
        if halo is None:
            return tuple(2 * k for k in need)
        halo = (
            (int(halo),) * psf.ndim
            if isinstance(halo, int)
            else tuple(int(h) for h in halo)
        )
        if len(halo) != psf.ndim:
            raise ValueError("halo must have one entry per PSF dim")
        if any(h < k for h, k in zip(halo, need, strict=True)):
            raise ValueError(
                f"halo {halo} must be at least the PSF size {need}"
            )
        return halo

    def _labeled_for(self, data: ArrayLike) -> Labeled:
        if isinstance(data, Labeled):
            return data
        psf = self.psf
        ndim = 2 if psf is None else psf.ndim
        return self._labeled(data, Layout.default(len(data.shape), ndim))

    def _maps(self, data: Labeled, backend: dict) -> dict[str, Tensor | None]:
        """Sensor and background maps, broadcast to `(*group, *spatial)`."""
        shape = data.layout.map_shape(data.array)
        out: dict[str, Tensor | None] = {
            "offset": None,
            "gain": None,
            "variance": None,
            "background": None,
        }
        sensor = self.sensor
        if sensor is not None:
            for name in ("offset", "gain"):
                value = getattr(sensor, name)
                if value is not None:
                    out[name] = torch.broadcast_to(value.to(**backend), shape)
            if self.options["poisson_gaussian"]:
                out["variance"] = torch.broadcast_to(
                    sensor.variance.to(**backend), shape
                )
        b = self.options["background"]
        if b is not None:
            out["background"] = torch.broadcast_to(
                torch.as_tensor(b).to(**backend), shape
            )
        return out

    @staticmethod
    def _region(chunk: Chunk) -> tuple[slice, ...]:
        """Loaded region (core + halo) along the spatial axes."""
        region = []
        for core, inner, n in zip(
            chunk.spatial, chunk.inner, chunk.data.shape, strict=True
        ):
            start = core.start - inner.start
            region.append(slice(start, start + n))
        return tuple(region)

    def _prepare(self, chunk: Chunk, maps: dict) -> _Block:
        """Counts y (electrons with a sensor), background, read noise."""
        index = chunk.group + self._region(chunk)
        y = chunk.data
        if maps["offset"] is not None:
            y = y - maps["offset"][index]
        gain = None if maps["gain"] is None else maps["gain"][index]
        if gain is not None:
            y = y / gain
        b: Tensor | float = 0.0
        if maps["background"] is not None:
            b = maps["background"][index]
            if gain is not None:
                b = b / gain
        if maps["variance"] is not None:
            var = maps["variance"][index]
            if gain is not None:
                var = var / gain.square()
            y = (y + var).clamp(min=0)
            b = b + var
        return _Block(y, b)

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
        self.sensor = self._pick(init, SensorEstimate, shape=shape, like=data)
        self._warn_unused(init, SensorEstimate)
        if self.sensor is not None and self.sensor.gain is None:
            if self.sensor.offset is None:
                raise ValueError("The sensor estimate has no offset or gain")
        if self.options["poisson_gaussian"] and (
            self.sensor is None or self.sensor.variance is None
        ):
            raise ValueError(
                "poisson_gaussian=True needs a SensorEstimate with a "
                "variance, given as fit(data, init=sensor)"
            )
        self.is_faithful = self._faithful_options and self.sensor is None
        psf = self._psf(backend)
        if psf.ndim != len(data.layout.spatial):
            raise ValueError(
                f"The PSF has {psf.ndim} dims but the layout has "
                f"{len(data.layout.spatial)} spatial axes"
            )
        # one pass: per-image mean of y − b (in counts), the start level
        group, sample, spatial = data.layout.shape_of(data.array)
        total = torch.zeros(group + sample, dtype=torch.float64)
        maps = self._maps(data, backend)
        for chunk in blocks(data, None, 0, **backend):
            blk = self._prepare(chunk, maps)
            value = (blk.data - blk.background).sum().double().cpu()
            total[chunk.group + chunk.samples] += value
        level = total / math.prod(spatial)
        self.level = level.clamp(min=torch.finfo(backend["dtype"]).tiny)
        self.history.extras["iterations"] = []

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if self.level is None:
            raise RuntimeError("Call fit(data) before apply(data)")
        data = self._labeled_for(data)
        group, sample, _ = data.layout.shape_of(data.array)
        if tuple(self.level.shape) != group + sample:
            raise ValueError(
                "apply(data) needs the data given to fit (images per "
                f"group/sample {tuple(self.level.shape)}, got "
                f"{group + sample})"
            )
        backend = self._backend(data)
        psf = self._psf(backend)
        halo = self._halo(psf)
        raw = data.array
        target = out.array if isinstance(out, Labeled) else out
        if out is not None and target is raw:
            raise ValueError("In-place deconvolution is not possible")
        out = prepare_out(out, tuple(data.shape), like=raw, dtype=self.dtype)
        maps = self._maps(data, backend)
        ops: dict[tuple[int, ...], PaddedConvolution] = {}
        n_iter = self.options["n_iter"]
        iterations = self.history.extras.setdefault("iterations", [])
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
            blk = self._prepare(chunk, maps)
            key = tuple(blk.data.shape)
            if key not in ops:
                ops[key] = PaddedConvolution(psf, key)
            op = ops[key]
            start = float(self.level[chunk.group + chunk.samples].squeeze())
            f, n = rl_block(
                blk.data,
                op,
                start,
                n_iter=None if n_iter is None else int(n_iter),
                alpha=float(self.options["alpha"]),
                max_iter=int(self.options["max_iter"]),
                background=blk.background,
                damping=float(self.options["damping"]),
            )
            iterations.append(float(n))
            chunk.store(out, op.crop(f)[chunk.inner])
        flush(out)
        return out

    def _extra_state(self) -> dict[str, object]:
        return {"psf": self.psf, "level": self.level}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.psf = state.get("psf")
        self.level = state.get("level")


def richardson_lucy(
    data: ArrayLike,
    psf: Tensor,
    *,
    out: WritableArrayLike | None = None,
    sensor: SensorEstimate | None = None,
    layout: Layout | None = None,
    **options: object,
) -> WritableArrayLike:
    """Richardson–Lucy deconvolution of `data` with a known PSF.

    Shortcut for `RichardsonLucy(psf, **options).fit(data,
    init=sensor).apply(data, out)`; see
    [`RichardsonLucy`][optix.deconv.RichardsonLucy] for the model, the
    defaults and the extensions.

    Parameters
    ----------
    data : array-like or Labeled
        Images; `(*samples, *spatial)` without a layout, with as many
        spatial dims as the PSF.
    psf : Tensor
        Point spread function, centred at index `K // 2`.
    out : writable array-like, optional
        Output (tensor, numpy, zarr, ...). Default: a new tensor.
    sensor : SensorEstimate, optional
        Camera model; the data are converted to electrons.
    layout : Layout, optional
        Axis roles.
    **options
        Options of [`RichardsonLucy`][optix.deconv.RichardsonLucy]
        (`n_iter`, `background`, `damping`, `halo`, `faithful`, ...).

    Returns
    -------
    writable array-like
        `out`, the deconvolved images.
    """
    est = RichardsonLucy(psf, layout=layout, **options)
    est.fit(data, init=sensor)
    return est.apply(data, out)
