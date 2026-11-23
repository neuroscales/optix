"""Richardson–Lucy deconvolution with total-variation regularisation."""

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from .._estimator import Estimator, Tags
from ..estimates import SensorEstimate
from ..io import ArrayLike, Layout, WritableArrayLike
from ._fft import PaddedConvolution
from ._rl import RichardsonLucy, _Block, rl_block

__all__ = ["RichardsonLucyTV", "richardson_lucy_tv", "tv_curvature"]

_STARTS = ("data", "uniform")


def tv_curvature(
    f: Tensor, ndim: int, *, epsilon: float | None = None
) -> Tensor:
    """Curvature term div(∇f / |∇f|) of the total variation.

    Discretised with forward differences for the gradient and backward
    differences for the divergence (its negative adjoint), periodic on
    the grid:

        (∇f)ₖ(x) = f(x + eₖ) − f(x),
        nₖ = (∇f)ₖ / max(|∇f|, ε),
        div n (x) = ∑ₖ nₖ(x) − nₖ(x − eₖ),

    with |∇f| the Euclidean norm over the `ndim` trailing axes. Where
    |∇f| ≤ ε the unit normal n is taken as (∇f)/ε, i.e. it vanishes
    with the gradient. Since |nₖ| ≤ 1, |div n| ≤ d + √d (d = `ndim`).

    Parameters
    ----------
    f : Tensor
        Image, `(..., *N)`.
    ndim : int
        Number of trailing (spatial) axes.
    epsilon : float, optional
        Floor ε of |∇f|. Default: the round-off level of f,
        ε = eps · max|f| (eps the machine epsilon of its dtype).

    Returns
    -------
    Tensor
        div(∇f / |∇f|), same shape as `f`.
    """
    dims = tuple(range(-ndim, 0))
    grad = [torch.roll(f, -1, d) - f for d in dims]
    norm = torch.zeros_like(f)
    for g in grad:
        norm.addcmul_(g, g)
    norm.sqrt_()
    if epsilon is None:
        info = torch.finfo(f.dtype)
        epsilon = max(float(f.abs().max()) * info.eps, info.tiny)
    norm.clamp_(min=float(epsilon))
    div = torch.zeros_like(f)
    for d, g in zip(dims, grad, strict=True):
        n = g / norm
        div += n - torch.roll(n, 1, d)
    return div


def _max_lambda(ndim: int) -> float:
    """Bound on λ_TV below which 1 − λ_TV · div n > 0 everywhere."""
    return 1.0 / (ndim + math.sqrt(ndim))


class RichardsonLucyTV(RichardsonLucy):
    """Richardson–Lucy deconvolution with total-variation regularisation.

    Model (Dey et al. 2006): the image i is a Poisson sample of the
    blurred object, i ~ Poisson(o ∗ h), and the object minimises the
    negative log-likelihood plus its total variation,

        J(o) = ∑ₛ [(o ∗ h)(s) − i(s) log (o ∗ h)(s)] + λ_TV ∑ₛ |∇o(s)|.

    The minimisation uses the explicit, one-step-late multiplicative
    scheme of Green (1990), Eq. 11 of the paper:

        oₖ₊₁ = oₖ · [(i / (oₖ ∗ h)) ⋆ h] / (1 − λ_TV div(∇oₖ / |∇oₖ|)),

    i.e. the Richardson–Lucy correction of
    [`RichardsonLucy`][optix.deconv.RichardsonLucy] divided by the
    derivative of the regulariser. TV smooths along the level lines of
    the image and not across them, so edges are kept while the noise
    amplification of plain Richardson–Lucy is suppressed. With
    λ_TV = 0 the iteration is plain Richardson–Lucy.

    **Regularisation weight.** λ_TV = 0.002, the value used by the paper
    for all synthetic data (0.005 for one real data set).

    **Discretisation of the TV term.** Not specified in the paper:
    forward differences for ∇, backward differences for div, unit voxel
    spacing along every axis, and the unit normal ∇o/|∇o| set to 0
    where the gradient is at the round-off level (see
    [`tv_curvature`][optix.deconv.utils.tv_curvature]). The paper gives
    no ε for |∇o|; one can be set with `epsilon`.

    **Positivity.** The paper requires 1 − λ_TV div(∇o/|∇o|) > 0 and
    obtains it by keeping λ_TV small. With this discretisation
    |div(∇o/|∇o|)| ≤ d + √d in d dims, so λ_TV < 1 / (d + √d) (0.29 in
    2-D, 0.21 in 3-D) guarantees a positive denominator; larger values
    are refused.

    **Starting estimate.** The measured image (negative values set to 0),
    as in the paper; its scale does not matter, since both the
    Richardson–Lucy correction and the TV term are invariant to the
    scale of o. Pixels at 0 in the start stay at 0 (multiplicative
    update). Outside the observed region (see Boundaries), the start is
    the mean of the image.

    **Stopping rule.** The paper stops "at convergence, defined when the
    relative difference between two images is less than a threshold"
    of 10⁻⁵. The relative difference is ‖oₖ₊₁ − oₖ‖₂ / ‖oₖ‖₂ on the
    observed block (the norm is not specified in the paper), and the
    iteration is capped at `max_iter` = 1000 (not specified in the
    paper). A fixed number of iterations is selected with `n_iter`.

    **Boundaries and blocks.** As in
    [`RichardsonLucy`][optix.deconv.RichardsonLucy]: the object is
    estimated on a domain extended by the PSF support, and the
    normalised correction (Hᵀ1 ≠ 1 near the edges) is divided by the TV
    denominator; Eq. 11 is written for Hᵀ1 = 1. The TV stencil reaches
    one voxel beyond the PSF, so the halo of each block is at least the
    PSF size plus one (default: twice that). The stopping rule is
    applied per block.

    !!! note "Extensions (off by default)"
        - **Camera model:** `fit(data, init=sensor)` with a
          [`SensorEstimate`][optix.estimates.SensorEstimate] converts the
          data to electrons, as in
          [`RichardsonLucy`][optix.deconv.RichardsonLucy].
        - **Uniform start:** `start="uniform"` starts from the uniform
          image of [`RichardsonLucy`][optix.deconv.RichardsonLucy]
          instead of the measured image.

    !!! references "References"

        1. Dey N, Blanc-Féraud L, Zimmer C, Roux P, Kam Z, Olivo-Marin
           JC, Zerubia J. *Richardson–Lucy algorithm with total
           variation regularization for 3D confocal microscope
           deconvolution.* Microsc Res Tech **69**(4):260-266, 2006.
           [doi:10.1002/jemt.20294](https://doi.org/10.1002/jemt.20294)
        2. Green PJ. *On use of the EM algorithm for penalized likelihood
           estimation.* J R Stat Soc B **52**(3):443-452, 1990.
           [doi:10.1111/j.2517-6161.1990.tb01798.x](https://doi.org/10.1111/j.2517-6161.1990.tb01798.x)

        ??? quote "BibTeX"

            ```bibtex
            @article{dey2006richardson,
              author  = {Dey, Nicolas and Blanc-Féraud, Laure and
                         Zimmer, Christophe and Roux, Pascal and
                         Kam, Zvi and Olivo-Marin, Jean-Christophe and
                         Zerubia, Josiane},
              title   = {Richardson--Lucy algorithm with total variation
                         regularization for {3D} confocal microscope
                         deconvolution},
              journal = {Microscopy Research and Technique},
              volume  = {69},
              number  = {4},
              pages   = {260--266},
              year    = {2006},
              doi     = {10.1002/jemt.20294}
            }
            @article{green1990use,
              author  = {Green, Peter J.},
              title   = {On use of the {EM} algorithm for penalized
                         likelihood estimation},
              journal = {Journal of the Royal Statistical Society:
                         Series B (Methodological)},
              volume  = {52},
              number  = {3},
              pages   = {443--452},
              year    = {1990},
              doi     = {10.1111/j.2517-6161.1990.tb01798.x}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import RichardsonLucyTV
    >>> psf = torch.ones(5, 5)
    >>> data = torch.poisson(torch.full((64, 64), 10.0))
    >>> est = RichardsonLucyTV(psf, n_iter=10).fit(data)
    >>> est.apply(data).shape
    torch.Size([64, 64])
    """

    tags = Tags(
        targets=("deconvolution",),
        streaming="halo",
        noise="poisson",
        reference="doi:10.1002/jemt.20294",
    )
    paper = {
        "lambda_tv": 0.002,
        "n_iter": None,
        "tol": 1e-5,
        "max_iter": 1000,
        "epsilon": None,
        "start": "data",
        "halo": None,
        "block": None,
    }
    recommended = dict(paper)
    _extra_overhead = 9.0  # gradient, its norm, normals, divergence

    def __init__(
        self,
        psf: Tensor | None = None,
        *,
        lambda_tv: float | None = None,
        n_iter: int | None = None,
        tol: float | None = None,
        max_iter: int | None = None,
        epsilon: float | None = None,
        start: str | None = None,
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
        lambda_tv : float
            Weight λ_TV of the total variation, 0 ≤ λ_TV < 1 / (d + √d).
            Default (Dey et al. 2006): 0.002.
        n_iter : int, optional
            Fixed number of iterations. Default: None, the relative-change
            stopping rule.
        tol : float
            Threshold of the relative change between two iterates.
            Default (Dey et al. 2006): 10⁻⁵.
        max_iter : int
            Iteration cap of the stopping rule. Default: 1000 (not
            specified in the paper).
        epsilon : float, optional
            Floor ε of |∇f| in the TV term. Default: None, the round-off
            level of the estimate (not specified in the paper).
        start : {"data", "uniform"}
            Starting estimate: the measured image, or (extension) the
            uniform image of the same integrated intensity. Default (Dey
            et al. 2006): `"data"`.
        halo : int or sequence of int, optional
            Halo width per spatial axis, at least the PSF size plus one
            along that axis. Default: twice the PSF size plus one (not
            specified in the paper).
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
        self.level: Tensor | None = None
        self._check_options()

    def _check_options(self) -> None:
        opts = self.options
        psf = self.psf
        if psf is not None and psf.ndim not in (2, 3):
            raise ValueError(f"psf must be 2-D or 3-D, got {psf.ndim} dims")
        if opts["n_iter"] is not None and int(opts["n_iter"]) < 0:
            raise ValueError("n_iter must be non-negative")
        if not float(opts["tol"]) > 0:
            raise ValueError("tol must be positive")
        if opts["start"] not in _STARTS:
            raise ValueError(
                f"start must be one of {_STARTS}, got {opts['start']!r}"
            )
        if opts["epsilon"] is not None and not float(opts["epsilon"]) > 0:
            raise ValueError("epsilon must be positive")
        lam = float(opts["lambda_tv"])
        if lam < 0:
            raise ValueError("lambda_tv must be non-negative")
        if psf is not None and lam >= _max_lambda(psf.ndim):
            raise ValueError(
                f"lambda_tv must be below {_max_lambda(psf.ndim):.4g} in "
                f"{psf.ndim}-D, so that the denominator stays positive"
            )

    def _halo(self, psf: Tensor) -> tuple[int, ...]:
        halo = self.options["halo"]
        need = tuple(k + 1 for k in psf.shape)
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
                f"halo {halo} must be at least the PSF size plus one {need}"
            )
        return halo

    def _penalty(self, ndim: int) -> Callable[[Tensor], Tensor] | None:
        lam = float(self.options["lambda_tv"])
        if lam >= _max_lambda(ndim):
            raise ValueError(
                f"lambda_tv must be below {_max_lambda(ndim):.4g} in {ndim}-D"
            )
        if lam == 0:
            return None
        eps = self.options["epsilon"]
        eps = None if eps is None else float(eps)

        def penalty(f: Tensor) -> Tensor:
            return 1.0 - lam * tv_curvature(f, ndim, epsilon=eps)

        return penalty

    def _solve(
        self,
        blk: _Block,
        op: PaddedConvolution,
        start: float,
        trace: dict | None = None,
    ) -> tuple[Tensor, int]:
        """Deconvolve one prepared block: estimate on `op.grid`, updates."""
        first: float | Tensor = start
        if self.options["start"] == "data":
            inside = op.pad(torch.ones_like(blk.data)) > 0
            first = torch.where(inside, op.pad(blk.data.clamp(min=0)), start)
        n_iter = self.options["n_iter"]
        return rl_block(
            blk.data,
            op,
            first,
            n_iter=None if n_iter is None else int(n_iter),
            max_iter=int(self.options["max_iter"]),
            tol=float(self.options["tol"]),
            penalty=self._penalty(op.ndim),
            trace=trace,
        )


def richardson_lucy_tv(
    data: ArrayLike,
    psf: Tensor,
    *,
    out: WritableArrayLike | None = None,
    sensor: SensorEstimate | None = None,
    layout: Layout | None = None,
    **options: object,
) -> WritableArrayLike:
    """Richardson–Lucy deconvolution with total-variation regularisation.

    Shortcut for `RichardsonLucyTV(psf, **options).fit(data,
    init=sensor).apply(data, out)`; see
    [`RichardsonLucyTV`][optix.deconv.RichardsonLucyTV] for the model,
    the defaults and the extensions.

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
        Options of [`RichardsonLucyTV`][optix.deconv.RichardsonLucyTV]
        (`lambda_tv`, `n_iter`, `tol`, `halo`, `faithful`, ...).

    Returns
    -------
    writable array-like
        `out`, the deconvolved images.
    """
    est = RichardsonLucyTV(psf, layout=layout, **options)
    est.fit(data, init=sensor)
    return est.apply(data, out)
