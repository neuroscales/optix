"""Snyder–Hammoud–White deconvolution of CCD data."""

import warnings
from collections.abc import Sequence

import torch
from torch import Tensor

from .._estimator import Estimator, Tags, _same
from ..estimates import SensorEstimate, ShadingEstimate
from ..io import Chunk, Labeled, Layout
from ..special import poisson_normal_mean
from ._fft import PaddedConvolution
from ._rl import RichardsonLucy, _Block, rl_block

__all__ = ["Snyder"]

_NOISE = ("exact", "shifted-poisson")
#: Options from the paper that do not make an estimate unfaithful.
_PAPER_CONTENT = ("noise", "flat", "background", "sieve")


class Snyder(RichardsonLucy):
    """Deconvolution of CCD camera data (Snyder, Hammoud & White 1993).

    **Data model** (Eqs. 5–11). The value read from pixel j, in
    electrons, is

        r(j) = n_obj(j) + n₀(j) + g(j),

    - n_obj(j) ~ Poisson(β(j) ∑ᵢ p(j|i) λ(i)): photo-electrons from the
      object λ, blurred by the PSF p and detected with efficiency β(j),
      the flat field (Eqs. 6 and 11);
    - n₀(j) ~ Poisson(μ₀(j)): background counts, from external and
      internal background radiation, dark current and the Poisson
      ("fat zero") part of the bias (Eqs. 7–8);
    - g(j) ~ N(m, σ²): read noise of the output amplifier, whose mean m
      is the deterministic bias.

    The estimate maximises the likelihood of r (Eq. 10) under λ ≥ 0, by
    expectation–maximisation. With μₖ(j) = β(j) ∑ᵢ p(j|i) λₖ(i) + μ₀(j)
    (Eq. 20) and β̄(i) = ∑ⱼ β(j) p(j|i), the iteration is (Eq. 18)

        λₖ₊₁(i) = λₖ(i)/β̄(i) · ∑ⱼ β(j) p(j|i) F[r(j), μₖ(j), σ] / μₖ(j),

    where F[r, μ, σ] = E[n_obj + n₀ | r] is the conditional mean of the
    Poisson counts given the data (Eqs. 19 and 21),

        F[r, μ, σ] = ∑ₙ n μⁿ/n! e^{−(r−n−m)²/2σ²}
                     / ∑ₙ μⁿ/n! e^{−(r−n−m)²/2σ²}

    (see [`poisson_normal_mean`][optix.special.poisson_normal_mean]).
    Every iterate is non-negative and the likelihood never decreases.
    Without read noise (σ → 0), F = r − m and Eq. (18) is Eq. (15) on the
    bias-corrected data; with β = 1, μ₀ = 0 and σ = 0 it is the
    Richardson–Lucy iteration.

    **Read noise.** `noise="exact"` (the default) evaluates F at every
    iteration and pixel, which holds for any read-noise level; it is the
    iteration that the paper builds on (regularisation, Section 4, and
    spectral response, Section 3.D). `noise="shifted-poisson"` is the
    large-variance approximation of Section 3.C: with m = σ², g is close
    in distribution to a Poisson variable of mean σ², so the data,
    shifted so that m = σ², are Poisson with mean μₖ + σ², and Eq. (15)
    is used with r + σ² − m in place of r and μ₀ + σ² in place of μ₀.
    The paper's Hubble Space Telescope example uses this approximation.
    The shifted data are clipped at 0 (not specified in the paper).

    **Camera calibration.** The model terms are taken from the estimates
    given to [`fit`][optix.Estimator.fit] (`init=sensor`,
    `init=shading` or `init=[sensor, shading]`) and from the options:

    - [`SensorEstimate`][optix.estimates.SensorEstimate]: the data are
      converted to electrons, r = (y − offset)/gain. `offset` is the
      deterministic bias m (it is subtracted, so m = 0 in electrons),
      `variance`/gain² is σ². Without a sensor estimate the data are
      taken in electrons with m = 0 and σ = 0.
    - [`ShadingEstimate`][optix.estimates.ShadingEstimate]: `flat` is β,
      rescaled to unit mean (so λ is in electrons at a pixel of average
      efficiency); `dark`, which includes the camera offset, gives the
      dark counts (dark − offset)/gain, clipped at 0, added to μ₀ as
      Poisson counts, never subtracted from the data. A per-frame
      `baseline` is not part of the model and is ignored.
    - `flat=β`: the flat field, used as given (for example a quantum
      efficiency map in (0, 1]); it takes precedence over the shading
      estimate's.
    - `background=b`: further background counts, in data units with the
      offset excluded, b/gain added to μ₀.

    The flat field enters the model rather than being divided out of the
    data, which would make them no longer Poisson (Section 1).

    **Regularisation** (Section 4, Eq. 30). `sieve=s` blurs the final
    iterate with the kernel s (normalised to unit sum): the method of
    sieves with equal sieve and resolution kernels, which estimates the
    blurred object d = s * λ. Default: none, as in the paper's example.

    **Starting estimate and stopping rule.** Not specified in the paper.
    As in [`RichardsonLucy`][optix.deconv.RichardsonLucy], the start is
    uniform, ∑ⱼ (r(j) − μ₀(j)) / ∑ⱼ β(j) per image, and the iteration
    stops at Lucy's χ² test, with the variance of the data,
    χ²_obs = ∑ⱼ (r(j) − m − μₖ(j))² / (μₖ(j) + σ²), or after `n_iter`
    iterations (the paper's example uses 600). Boundaries, blocks and
    layouts are handled as in
    [`RichardsonLucy`][optix.deconv.RichardsonLucy].

    !!! references "References"

        1. Snyder DL, Hammoud AM, White RL. *Image recovery from data
           acquired with a charge-coupled-device camera.* J Opt Soc Am A
           **10**(5):1014-1023, 1993.
           [doi:10.1364/JOSAA.10.001014](https://doi.org/10.1364/JOSAA.10.001014)
        2. Lucy LB. *An iterative technique for the rectification of
           observed distributions.* Astron J **79**(6):745-754, 1974.
           [doi:10.1086/111605](https://doi.org/10.1086/111605)

        ??? quote "BibTeX"

            ```bibtex
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
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import Snyder
    >>> from optix.estimates import SensorEstimate
    >>> psf = torch.ones(5, 5)
    >>> data = 100 + torch.poisson(torch.full((64, 64), 10.0))
    >>> sensor = SensorEstimate(offset=100.0, gain=1.0, variance=4.0)
    >>> est = Snyder(psf, n_iter=10).fit(data, init=sensor)
    >>> est.apply(data).shape
    torch.Size([64, 64])
    """

    tags = Tags(
        targets=("deconvolution",),
        streaming="halo",
        noise="poisson-gaussian",
        reference="doi:10.1364/JOSAA.10.001014",
    )
    paper = {
        "noise": "exact",
        "flat": None,
        "background": None,
        "sieve": None,
        "n_iter": None,
        "alpha": 0.05,
        "max_iter": 100,
        "halo": None,
        "block": None,
    }
    recommended = dict(paper)

    def __init__(
        self,
        psf: Tensor | None = None,
        *,
        noise: str | None = None,
        flat: Tensor | None = None,
        background: float | Tensor | None = None,
        sieve: Tensor | None = None,
        n_iter: int | None = None,
        alpha: float | None = None,
        max_iter: int | None = None,
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
            Point spread function p, `(*K)` with 2 or 3 dims, in the
            order of the spatial axes of the layout, centred at index
            `K // 2`. Normalised to unit sum.
        noise : {"exact", "shifted-poisson"}
            Read-noise treatment: the exact conditional mean F of
            Eq. (21) (default), or the shifted-Poisson approximation of
            Section 3.C.
        flat : Tensor, optional
            Flat field β ≥ 0 (detection efficiency), a map that broadcasts
            to `(*group, *spatial)`, used as given. Default: the shading
            estimate's flat, rescaled to unit mean, else β = 1.
        background : float or Tensor, optional
            Background counts in data units (offset excluded), a scalar
            or a map that broadcasts to `(*group, *spatial)`; added to
            μ₀. Default: none.
        sieve : Tensor, optional
            Sieve kernel s (Section 4), no larger than the PSF along any
            axis, centred at `K // 2`; normalised to unit sum. The final
            iterate is blurred with it. Default: none.
        n_iter : int, optional
            Fixed number of iterations. Default: None, the χ² stopping
            rule (not specified in the paper).
        alpha : float
            Level of the χ² stopping test. Default: 0.05 (Lucy 1974).
        max_iter : int
            Iteration cap of the χ² stopping rule. Default: 100 (not
            specified in the paper).
        halo : int or sequence of int, optional
            Halo width per spatial axis, at least the PSF size along that
            axis. Default: twice the PSF size (not specified in the paper).
        block : sequence of int or None, optional
            Core block size per spatial axis (None entries: whole
            extent). Default: chosen from the memory budget.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        Estimator.__init__(self, dtype=dtype, device=device, layout=layout)
        params = {k: v for k, v in locals().items() if k != "psf"}
        self._resolve(faithful, params)
        self._faithful_options = self.is_faithful
        self.psf = psf
        self.sensor: SensorEstimate | None = None
        self.shading: ShadingEstimate | None = None
        self.level: Tensor | None = None
        self._check_options()
        if self.options["noise"] not in _NOISE:
            raise ValueError(
                f"noise must be one of {_NOISE}, got {self.options['noise']!r}"
            )
        s = self.options["sieve"]
        if s is not None and psf is not None:
            s = torch.as_tensor(s)
            if s.ndim != psf.ndim or any(
                a > b for a, b in zip(s.shape, psf.shape, strict=True)
            ):
                raise ValueError(
                    f"sieve {tuple(s.shape)} must have the PSF's dims and "
                    f"be no larger than the PSF {tuple(psf.shape)}"
                )

    def _resolve(self, faithful: bool, params: object) -> None:
        super()._resolve(faithful, params)
        # the camera terms and the paper's own variants keep it faithful
        self.is_faithful = all(
            _same(value, self.paper.get(name))
            for name, value in self.options.items()
            if name not in _PAPER_CONTENT
        )

    # -- hooks of RichardsonLucy -----------------------------------------
    def _init_from(self, init: object, shape: tuple, like: object) -> None:
        self.sensor = self._pick(init, SensorEstimate, shape=shape, like=like)
        self.shading = self._pick(
            init, ShadingEstimate, shape=shape, like=like
        )
        self._warn_unused(init, SensorEstimate, ShadingEstimate)
        if self.shading is not None and self.shading.baseline is not None:
            warnings.warn(
                "Snyder ignores the per-frame baseline of the shading "
                "estimate",
                RuntimeWarning,
                stacklevel=4,
            )
        self.is_faithful = self._faithful_options

    def _maps(self, data: Labeled, backend: dict) -> dict[str, Tensor | None]:
        """Camera maps (data units), broadcast to `(*group, *spatial)`."""
        shape = data.layout.map_shape(data.array)

        def full(value: object) -> Tensor | None:
            if value is None:
                return None
            return torch.broadcast_to(
                torch.as_tensor(value).to(**backend), shape
            )

        sensor, shading = self.sensor, self.shading
        out = {
            name: full(None if sensor is None else getattr(sensor, name))
            for name in ("offset", "gain", "variance")
        }
        out["background"] = full(self.options["background"])
        flat = self.options["flat"]
        if flat is None and shading is not None and shading.flat is not None:
            flat = shading.normalize("mean").flat
        out["flat"] = full(flat)
        if out["flat"] is not None and (
            (out["flat"] < 0).any() or not out["flat"].isfinite().all()
        ):
            raise ValueError("The flat field must be finite and non-negative")
        out["dark"] = full(None if shading is None else shading.dark)
        return out

    def _prepare(self, chunk: Chunk, maps: dict) -> _Block:
        """Electrons r, background μ₀, flat β and read noise σ²."""
        index = chunk.group + self._region(chunk)

        def at(name: str) -> Tensor | None:
            value = maps[name]
            return None if value is None else value[index]

        offset, gain = at("offset"), at("gain")
        r = chunk.data
        if offset is not None:
            r = r - offset
        if gain is not None:
            r = r / gain
        mu0: Tensor | float = 0.0
        b = at("background")
        if b is not None:
            mu0 = mu0 + (b if gain is None else b / gain)
        dark = at("dark")
        if dark is not None:
            if offset is not None:
                dark = dark - offset
            if gain is not None:
                dark = dark / gain
            mu0 = mu0 + dark.clamp(min=0)
        var: Tensor | float = 0.0
        v = at("variance")
        if v is not None:
            var = v if gain is None else v / gain.square()
        if self.options["noise"] == "shifted-poisson":
            # g ≈ Poisson(σ²) once m = σ²: r + σ² ~ Poisson(μ + σ²)
            return _Block((r + var).clamp(min=0), mu0 + var, at("flat"))
        return _Block(r, mu0, at("flat"), var)

    def _solve(
        self, blk: _Block, op: PaddedConvolution, start: float
    ) -> tuple[Tensor, int]:
        var = blk.variance
        expect = None
        if isinstance(var, Tensor) and bool((var > 0).any()):

            def expect(mu: Tensor) -> Tensor:
                return poisson_normal_mean(blk.data, mu, var)

        n_iter = self.options["n_iter"]
        f, n = rl_block(
            blk.data,
            op,
            start,
            n_iter=None if n_iter is None else int(n_iter),
            alpha=float(self.options["alpha"]),
            max_iter=int(self.options["max_iter"]),
            background=blk.background,
            flat=blk.flat,
            expect=expect,
            variance=var,
        )
        sieve = self.options["sieve"]
        if sieve is not None:
            f = _blur(f, torch.as_tensor(sieve).to(f))
        return f, n

    def _extra_state(self) -> dict[str, object]:
        state = super()._extra_state()
        shading = self.shading
        state["shading"] = None if shading is None else shading._payload()
        return state

    def _load_extra_state(self, state: dict[str, object]) -> None:
        super()._load_extra_state(state)
        shading = state.get("shading")
        if shading is not None:
            self.shading = ShadingEstimate._from_payload(shading)


def _blur(f: Tensor, kernel: Tensor) -> Tensor:
    """Circular convolution of `f` on its grid with `kernel` (unit sum).

    The kernel is centred at `K // 2`. On the padded object grid of
    [`PaddedConvolution`][optix.deconv.utils.PaddedConvolution], a kernel
    no larger than the PSF does not wrap into the block.
    """
    if (kernel < 0).any() or not kernel.sum() > 0:
        raise ValueError("sieve must be non-negative with a positive sum")
    kernel = kernel / kernel.sum()
    dims = tuple(range(-kernel.ndim, 0))
    grid = f.shape[-kernel.ndim :]
    buf = f.new_zeros(grid)
    buf[tuple(slice(0, k) for k in kernel.shape)] = kernel
    buf = torch.roll(buf, tuple(-(k // 2) for k in kernel.shape), dims)
    spec = torch.fft.rfftn(f, dim=dims) * torch.fft.rfftn(buf, dim=dims)
    return torch.fft.irfftn(spec, s=grid, dim=dims).clamp_(min=0)
