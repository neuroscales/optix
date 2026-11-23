"""Biggs–Andrews accelerated Richardson–Lucy deconvolution."""

from collections.abc import Sequence

from torch import Tensor

from .._config import get_config
from .._estimator import Tags, _same
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
from ._lockstep import lockstep
from ._rl import _OVERHEAD_ACCEL, RichardsonLucy

__all__ = ["BiggsAndrews", "biggs_andrews"]

_SCOPES = ("global", "block", "auto")


class BiggsAndrews(RichardsonLucy):
    """Richardson–Lucy deconvolution accelerated by vector extrapolation.

    The iteration ψ of [`RichardsonLucy`][optix.deconv.RichardsonLucy]
    (same model, start, stopping rule, blocks and extensions) is applied
    to a point predicted from the last two iterates (Biggs and Andrews
    1997, "technique II"):

        yₖ = xₖ + αₖ hₖ,        hₖ = xₖ − xₖ₋₁,
        xₖ₊₁ = yₖ + gₖ,         gₖ = ψ(yₖ) − yₖ,

    with the acceleration parameter

        αₖ = ∑ gₖ₋₁ · gₖ₋₂ / ∑ gₖ₋₂ · gₖ₋₂,   0 ≤ αₖ ≤ 1,

    the least-squares projection of the latest change on the one before
    (first-order extrapolation, Eq. 10). The safeguards of the paper:
    αₖ < 0 is set to 0, which suspends the extrapolation until successive
    changes agree again; αₖ is capped at 1 so that the predicted step
    cannot grow exponentially; and the prediction is kept non-negative,
    negative pixels of yₖ being set to 0 (the paper requires the check
    but leaves the correction open). No extrapolation is made until two
    changes are known.

    **Start.** From the uniform start, the first Richardson–Lucy update
    gives Hᵀy/Hᵀ1, the PSF-correlated data h ⋆ g that the paper uses as
    its starting image; the extrapolation history starts there, which
    also gives the paper's "few unaccelerated iterations" from a constant
    start. The first prediction is therefore made at the fourth update,
    and up to three iterations the result is plain Richardson–Lucy.

    **Acceleration factor.** The estimated number of unaccelerated
    iterations equivalent to the accelerated ones, ∑ₖ ηₖ with

        ηₖ = 1 + ∑ⱼ₌₁ᵏ ∏ₘ₌ₖ₋ⱼ₊₁ᵏ γₘ,   γₖ = (αₖ βₖ)²,
        βₖ = ∑ gₖ₋₁ · gₖ₋₂ / ∑ gₖ₋₁ · gₖ₋₁,

    (Eqs. 11–13) is recorded in `history.extras["effective_iterations"]`
    (per image with a global α, per block with a block-wise α).

    **Blocks and the scope of α.** The sums of Eq. 10 run over the whole
    image, as in the paper. An image larger than one block is then
    processed in lockstep (`alpha_scope="global"`): every iteration is
    one pass over its blocks, which read the data of their core and
    halo and the current iterate around it from a full-size buffer,
    make one step, write back their core and add their share of the
    float64 sums; α and the χ² stopping test are evaluated once per
    image and iteration. The halo covers the PSF, so the result is that
    of the whole-image iteration. The iterates (three of them, on the
    image extended by the PSF margin) are held in memory when they fit
    the memory budget of [`config`][optix.config], else in temporary
    files; the data are read once per iteration.

    With `alpha_scope="block"`, each block instead runs all its
    iterations as an image of its own, with its own extrapolation
    history and its own αₖ, in a single pass over the data (as for
    [`RichardsonLucy`][optix.deconv.RichardsonLucy]); the results then
    differ from the whole-image ones by up to about 10 % of the peak on
    noisy data. `alpha_scope="auto"` uses the global α when each image
    fits in one block and the block-wise α otherwise. The scope used for
    each image is recorded in `history.extras["alpha_scope"]`.

    **Stopping rule.** Not specified in the paper, which compares
    restorations at fixed iteration counts; the default is Lucy's χ²
    rule of [`RichardsonLucy`][optix.deconv.RichardsonLucy], evaluated on
    the iterates xₖ. Accelerated iterations amplify noise sooner, so the
    stopping rule or `n_iter` matters more than for the plain iteration.

    !!! references "References"

        1. Biggs DSC, Andrews M. *Acceleration of iterative image
           restoration algorithms.* Appl Opt **36**(8):1766-1775, 1997.
           [doi:10.1364/AO.36.001766](https://doi.org/10.1364/AO.36.001766)
        2. Richardson WH. *Bayesian-based iterative method of image
           restoration.* J Opt Soc Am **62**(1):55-59, 1972.
           [doi:10.1364/JOSA.62.000055](https://doi.org/10.1364/JOSA.62.000055)
        3. Lucy LB. *An iterative technique for the rectification of
           observed distributions.* Astron J **79**(6):745-754, 1974.
           [doi:10.1086/111605](https://doi.org/10.1086/111605)

        ??? quote "BibTeX"

            ```bibtex
            @article{biggs1997acceleration,
              author  = {Biggs, David S. C. and Andrews, Mark},
              title   = {Acceleration of iterative image restoration
                         algorithms},
              journal = {Applied Optics},
              volume  = {36},
              number  = {8},
              pages   = {1766--1775},
              year    = {1997},
              doi     = {10.1364/AO.36.001766}
            }
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
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import BiggsAndrews
    >>> psf = torch.ones(5, 5)
    >>> data = torch.poisson(torch.full((64, 64), 10.0))
    >>> est = BiggsAndrews(psf, n_iter=10).fit(data)
    >>> est.apply(data).shape
    torch.Size([64, 64])
    """

    tags = Tags(
        targets=("deconvolution",),
        streaming="halo",
        noise="poisson",
        reference=(
            "doi:10.1364/AO.36.001766, doi:10.1364/JOSA.62.000055, "
            "doi:10.1086/111605"
        ),
    )
    paper = dict(RichardsonLucy.paper, alpha_scope="global")
    recommended = dict(paper, alpha_scope="auto")
    _accelerate = True

    def __init__(
        self,
        psf: Tensor | None = None,
        *,
        n_iter: int | None = None,
        alpha: float | None = None,
        max_iter: int | None = None,
        damping: float | None = None,
        halo: int | Sequence[int] | None = None,
        block: Sequence[int | None] | None = None,
        alpha_scope: str | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the accelerated deconvolution.

        Parameters
        ----------
        psf, n_iter, alpha, max_iter
            See [`RichardsonLucy`][optix.deconv.RichardsonLucy]. `alpha`
            is the level of the χ² stopping test, not the acceleration
            parameter αₖ.
        damping, halo, block
            See [`RichardsonLucy`][optix.deconv.RichardsonLucy].
        alpha_scope : {"global", "block", "auto"}
            Pixels over which the sums of the acceleration parameter αₖ
            (Eq. 10) run. `"global"`: the whole image, as in the paper;
            images larger than one block are iterated in lockstep, with
            one pass over the data per iteration. `"block"`: each block
            on its own, in a single pass over the data. `"auto"`:
            `"global"` when each image fits in one block, else
            `"block"`. Default (Biggs and Andrews 1997): `"global"`;
            `"auto"` with `faithful=False`.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator]. After
            [`apply`][optix.Estimator.apply], `is_faithful` is False if
            the block-wise α was used.
        """
        super().__init__(
            psf,
            n_iter=n_iter,
            alpha=alpha,
            max_iter=max_iter,
            damping=damping,
            halo=halo,
            block=block,
            faithful=faithful,
            dtype=dtype,
            device=device,
            layout=layout,
        )
        params = {k: v for k, v in locals().items() if k != "psf"}
        self._resolve(faithful, params)
        if self.options["alpha_scope"] not in _SCOPES:
            raise ValueError(
                f"alpha_scope must be one of {_SCOPES}, got "
                f"{self.options['alpha_scope']!r}"
            )
        self._faithful_options = all(
            _same(value, self.paper.get(name))
            for name, value in self.options.items()
            if name != "alpha_scope"
        )
        self._scope_used: str | None = None
        self.is_faithful = self._faithful_options and self._scope_faithful()

    def _scope_faithful(self) -> bool:
        if self._scope_used is not None:
            return self._scope_used == "global"
        return self.options["alpha_scope"] == "global"

    def _fit(self, data: ArrayLike, **kwargs: object) -> None:
        self._scope_used = None
        super()._fit(data, **kwargs)
        self.history.extras["alpha_scope"] = []
        self.is_faithful = self.is_faithful and self._scope_faithful()

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if self.level is None:
            raise RuntimeError("Call fit(data) before apply(data)")
        data = self._labeled_for(data)
        backend = self._backend(data)
        psf = self._psf(backend)
        raw = data.array
        target = out.array if isinstance(out, Labeled) else out
        if out is not None and target is raw:
            raise ValueError("In-place deconvolution is not possible")
        out = prepare_out(out, tuple(data.shape), like=raw, dtype=self.dtype)
        chunks = blocks(
            data,
            self.options["block"],
            self._halo(psf),
            overhead=_OVERHEAD_ACCEL,
            contiguous=True,
            align=chunks_of(out),
            **backend,
        )
        images = max(self.level.numel(), 1)
        single = len(chunks) <= images
        scope = self.options["alpha_scope"]
        if scope == "auto":
            scope = "global" if single else "block"
        if scope == "block" or single:
            out = super()._apply(data, out)
        else:
            group, sample, _ = data.layout.shape_of(raw)
            if tuple(self.level.shape) != group + sample:
                raise ValueError(
                    "apply(data) needs the data given to fit (images per "
                    f"group/sample {tuple(self.level.shape)}, got "
                    f"{group + sample})"
                )
            maps = self._maps(data, backend)
            n_iter = self.options["n_iter"]
            reports = lockstep(
                chunks,
                out,
                prepare=lambda chunk: self._prepare(chunk, maps),
                psf=psf,
                level=self.level,
                n_iter=None if n_iter is None else int(n_iter),
                alpha=float(self.options["alpha"]),
                max_iter=int(self.options["max_iter"]),
                damping=float(self.options["damping"]),
                memory=get_config().memory,
                backend=backend,
            )
            extras = self.history.extras
            for report in reports:
                extras.setdefault("iterations", []).append(
                    float(report["iterations"])
                )
                extras.setdefault("effective_iterations", []).append(
                    float(report["effective"])
                )
            flush(out)
        self.history.extras.setdefault("alpha_scope", []).extend(
            [scope] * images
        )
        self._scope_used = scope
        self.is_faithful = (
            self._faithful_options
            and self.sensor is None
            and self._scope_faithful()
        )
        return out


def biggs_andrews(
    data: ArrayLike,
    psf: Tensor,
    *,
    out: WritableArrayLike | None = None,
    sensor: SensorEstimate | None = None,
    layout: Layout | None = None,
    **options: object,
) -> WritableArrayLike:
    """Accelerated Richardson–Lucy deconvolution of `data`.

    Shortcut for `BiggsAndrews(psf, **options).fit(data,
    init=sensor).apply(data, out)`; see
    [`BiggsAndrews`][optix.deconv.BiggsAndrews] for the acceleration and
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
        Options of [`BiggsAndrews`][optix.deconv.BiggsAndrews]
        (`n_iter`, `alpha_scope`, `damping`, `halo`, `faithful`, ...).

    Returns
    -------
    writable array-like
        `out`, the deconvolved images.
    """
    est = BiggsAndrews(psf, layout=layout, **options)
    est.fit(data, init=sensor)
    return est.apply(data, out)
