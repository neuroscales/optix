"""Richardson–Lucy with an unmatched back projector (Guo et al. 2020)."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .._estimator import Tags
from ..estimates import SensorEstimate
from ..io import ArrayLike, Layout, WritableArrayLike
from ._fft import PaddedConvolution
from ._rl import RichardsonLucy, _Block, rl_block

__all__ = [
    "BackProjection",
    "WienerButterworth",
    "back_projector",
    "psf_fwhm",
    "wiener_butterworth",
]

_KINDS = ("wiener-butterworth", "butterworth", "gaussian", "traditional")
_STARTS = ("data", "uniform")
_START_FLOOR = 0.001  # value replacing zeros of the starting image (Guo 2020)


def psf_fwhm(psf: Tensor) -> tuple[float, ...]:
    """Full width at half maximum of a PSF along each axis, in pixels.

    The width is measured on the line profile through the maximum of the
    PSF, between the two half-maximum crossings, located by linear
    interpolation.

    Parameters
    ----------
    psf : Tensor
        Point spread function, `(*K)`.

    Returns
    -------
    tuple of float
        FWHM along each axis.
    """
    psf = torch.as_tensor(psf).double()
    peak = torch.unravel_index(torch.argmax(psf), psf.shape)
    peak = tuple(int(p) for p in peak)
    half = float(psf[peak]) / 2
    if not half > 0:
        raise ValueError("The PSF must have a positive maximum")
    out = []
    for axis in range(psf.ndim):
        index = list(peak)
        index[axis] = slice(None)
        line = psf[tuple(index)].tolist()
        c = peak[axis]
        right = _crossing(line[c:], half)
        left = _crossing(line[c::-1], half)
        if right is None or left is None:
            raise ValueError(
                f"The PSF does not fall below half its maximum along axis "
                f"{axis}; pass fwhm explicitly"
            )
        out.append(left + right)
    return tuple(out)


def _crossing(line: list[float], half: float) -> float | None:
    """Distance from line[0] to the first half-maximum crossing."""
    for i in range(1, len(line)):
        if line[i] <= half:
            a, b = line[i - 1], line[i]
            return i - 1 + (a - half) / (a - b)
    return None


def _freqs(
    shape: Sequence[int], cutoff: Sequence[float], like: Tensor
) -> Tensor:
    """(k/k_c)² summed over axes, on the full DFT grid of `shape`."""
    grids = torch.meshgrid(
        *(
            torch.fft.fftfreq(n, dtype=like.dtype, device=like.device) / kc
            for n, kc in zip(shape, cutoff, strict=True)
        ),
        indexing="ij",
    )
    return sum(g.square() for g in grids)


def back_projector(
    psf: Tensor,
    kind: str = "wiener-butterworth",
    *,
    wiener: float = 0.001,
    cutoff_gain: float = 0.001,
    order: float = 8,
    fwhm: float | Sequence[float] | None = None,
) -> Tensor:
    """Back-projector kernel b of Guo et al. (2020, Supplementary Note 2).

    With f the PSF (normalised to unit sum), F = DFT(f) its OTF, k the
    spatial frequency per axis (cycles per pixel), wₐ the FWHM of the PSF
    along axis a and k_cₐ = 1/wₐ the cutoff frequency, the transfer
    functions are

        traditional:   B = conj(F)                                  (13)
        Gaussian:      b = exp(−4 ln2 ∑ₐ (xₐ / wₐ)²)                (18)
        Butterworth:   B = 1 / √(1 + ε² [∑ₐ (kₐ/k_cₐ)²]ⁿ)           (21)
        Wiener–Butterworth:
                       B = conj(F) / (|F|² + α) · B_Butterworth     (25)

    with ε² = 1/β² − 1, so that the Butterworth gain at the cutoff is β
    (Eq. 23). The kernel is b = DFT⁻¹(B) (Eqs. 22 and 26), normalised to
    unit sum (B(0) = 1), as the iteration requires.

    The filters are sampled on the DFT grid of the PSF array, and the
    kernel has the shape of the PSF, centred at index `K // 2`; a larger
    (zero-padded) PSF array samples them more finely. The FWHM of the
    PSF is measured with [`psf_fwhm`][optix.deconv.utils.psf_fwhm]
    unless given. |F|² is the squared modulus of the OTF (the paper
    writes [DFT(f)]², for a real, symmetric OTF).

    Parameters
    ----------
    psf : Tensor
        Point spread function, `(*K)`, centred at index `K // 2`.
    kind : str
        `"wiener-butterworth"`, `"butterworth"`, `"gaussian"` or
        `"traditional"` (the transposed PSF).
    wiener : float
        Wiener parameter α > 0.
    cutoff_gain : float
        Butterworth gain β ∈ (0, 1) at the cutoff frequency.
    order : float
        Butterworth order n > 0.
    fwhm : float or sequence of float, optional
        Resolution (FWHM of the PSF) per axis, in pixels. Default:
        measured on `psf`.

    Returns
    -------
    Tensor
        Kernel b, `(*K)`, of unit sum.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    psf = torch.as_tensor(psf)
    if not psf.is_floating_point():
        psf = psf.double()
    psf = psf / psf.sum()
    shape = tuple(psf.shape)
    centre = tuple(k // 2 for k in shape)
    dims = tuple(range(psf.ndim))
    if kind != "traditional":
        if fwhm is None:
            fwhm = psf_fwhm(psf)
        elif isinstance(fwhm, int | float):
            fwhm = (float(fwhm),) * psf.ndim
        fwhm = tuple(float(w) for w in fwhm)
        if len(fwhm) != psf.ndim or min(fwhm) <= 0:
            raise ValueError("fwhm must be positive, one value per PSF dim")
    if kind == "gaussian":
        grids = torch.meshgrid(
            *(
                torch.arange(k, dtype=psf.dtype, device=psf.device) - c
                for k, c in zip(shape, centre, strict=True)
            ),
            indexing="ij",
        )
        r2 = sum(g.square() / w**2 for g, w in zip(grids, fwhm, strict=True))
        b = torch.exp(-4 * math.log(2) * r2)
        return b / b.sum()
    otf = torch.fft.fftn(torch.roll(psf, [-c for c in centre], dims))
    if kind == "traditional":
        spec = otf.conj()
    else:
        if not 0.0 < cutoff_gain < 1.0:
            raise ValueError("cutoff_gain must be in (0, 1)")
        if not order > 0:
            raise ValueError("order must be positive")
        eps2 = 1.0 / cutoff_gain**2 - 1.0
        k2 = _freqs(shape, [1.0 / w for w in fwhm], psf)
        spec = 1.0 / torch.sqrt(1.0 + eps2 * k2**order)
        if kind == "wiener-butterworth":
            if not wiener > 0:
                raise ValueError("wiener must be positive")
            spec = spec * otf.conj() / (otf.abs().square() + wiener)
        spec = spec / spec.flatten()[0]
    b = torch.fft.ifftn(spec).real
    return torch.roll(b, list(centre), dims)


class BackProjection:
    """Unmatched back projector B of a block: convolution with a kernel b.

    For a residual r on the block (shape L), B r is the linear convolution
    of r with b, on the padded object grid N of a
    [`PaddedConvolution`][optix.deconv.utils.PaddedConvolution]:

        (B r)ⱼ = ∑ᵢ b(j − i) rᵢ.

    With b the flipped PSF, B = Hᵀ. A kernel no larger than the PSF of
    the operator fits on its grid without wrap-around.

    On the margin of the object domain (the object pixels outside the
    block), B is replaced by Hᵀ: the data barely constrain these pixels,
    and a kernel with negative lobes makes the correction B r / B1 a
    weighted mean of r with weights of both signs, which does not pull
    them back. Hᵀ is also used inside the block where the negative
    weights nearly cancel the positive ones, B1 < ½ ∑ᵢ|b(j − i)| / ∑|b|
    (near its edges, for strongly oscillating kernels).

    Attributes
    ----------
    usable : Tensor
        Object pixels, on the grid N, where B is used (elsewhere Hᵀ).
    """

    def __init__(self, kernel: Tensor, op: PaddedConvolution) -> None:
        """Precompute the transfer function of b on the grid of `op`.

        Parameters
        ----------
        kernel : Tensor
            Kernel b, `(*K)` with K at most the PSF shape of `op` along
            each dim, centred at index `K // 2`, of unit sum.
        op : PaddedConvolution
            Blur operator H of the block, which sets the grids.
        """
        if kernel.ndim != op.ndim:
            raise ValueError("kernel and operator dims differ")
        self.op = op
        self._dims = tuple(range(-op.ndim, 0))
        kernel = kernel.to(dtype=op.norm.dtype, device=op.norm.device)
        self.transfer = self._transfer(kernel)
        ones = op.norm.new_ones(op.shape)
        b1 = self._apply(ones, self.transfer)
        abs1 = self._apply(ones, self._transfer(kernel.abs()))
        total = kernel.abs().sum() / kernel.sum()
        tol = abs1.max() * torch.finfo(abs1.dtype).eps * 16
        inside = op.pad(ones) > 0
        self.usable = inside & (b1 * total >= 0.5 * abs1 - tol)

    def _transfer(self, kernel: Tensor) -> Tensor:
        op = self.op
        buf = kernel.new_zeros(op.grid)
        buf[tuple(slice(0, k) for k in kernel.shape)] = kernel
        buf = torch.roll(
            buf,
            shifts=tuple(-(k // 2) for k in kernel.shape),
            dims=tuple(range(op.ndim)),
        )
        return torch.fft.rfftn(buf, dim=self._dims)

    def _apply(self, r: Tensor, transfer: Tensor) -> Tensor:
        spec = torch.fft.rfftn(self.op.pad(r), dim=self._dims) * transfer
        return torch.fft.irfftn(spec, s=self.op.grid, dim=self._dims)

    def __call__(self, r: Tensor) -> Tensor:
        """B r: block `(..., *L)` to object `(..., *N)`."""
        out = self._apply(r, self.transfer)
        if bool(self.usable.all()):
            return out
        return torch.where(self.usable, out, self.op.adjoint(r))


class WienerButterworth(RichardsonLucy):
    r"""Richardson–Lucy deconvolution with an unmatched back projector.

    Guo et al. (2020) replace the transposed PSF of the Richardson–Lucy
    update by a back projector b chosen so that the product of the
    transfer functions, DFT(b)·DFT(f), is as flat as possible up to the
    resolution limit (Supplementary Note 1, after Zeng & Gullberg 2000):

        e⁽ᵏ⁺¹⁾ = e⁽ᵏ⁾ · B[y / (H e⁽ᵏ⁾)] / B1,

    with H the blur by the PSF f and B the convolution with b, of unit
    sum (B1 = 1 away from the image edges). The default b is the
    Wiener–Butterworth (WB) filter, a Wiener inverse of the OTF times a
    Butterworth low-pass filter (Supplementary Note 2, Eq. 25):

        DFT(b) ∝ conj(F) / (|F|² + α) · 1 / √(1 + ε² [∑ₐ (kₐ/k_cₐ)²]ⁿ),

    with F the OTF, k_c = 1/FWHM of the PSF per axis, and ε² = 1/β² − 1
    (β the Butterworth gain at the cutoff). See
    [`back_projector`][optix.deconv.utils.back_projector] for the
    Gaussian and Butterworth back projectors the paper compares, and
    the traditional one (b the flipped PSF: plain Richardson–Lucy).

    **Defaults.** α = β = 0.001, n = 8 and a single iteration: the
    values of Supplementary Table S2.1 for the iSIM and confocal data
    (the paper uses α, β ∈ [0.001, 0.05], n ∈ [4, 10] and 1-5
    iterations, and recommends a single iteration as a rule of thumb).
    The starting estimate is the measured image with its zeros replaced
    by 0.001 (Methods). On the margin of the object domain, outside the
    image, it is the mean of the image (not specified in the paper).

    !!! warning "Convergence"
        With an unmatched back projector the iteration is not the
        expectation-maximisation algorithm: the Poisson likelihood need
        not increase, the flux need not be conserved, and convergence is
        not guaranteed (Zeng & Gullberg 2000 give conditions on b, which
        the paper notes the truncation below invalidates). The WB
        iteration reaches a resolution-limited estimate in about one
        iteration; more iterations can introduce artefacts (main text,
        Discussion). The WB back projector has negative lobes, so the
        update can produce negative values: as in the paper, they are
        set to 0 at each iteration, and a pixel set to 0 stays at 0.

    Boundaries, blocks, layouts and the extensions (camera model,
    background, Poisson–Gaussian noise, damping) are as in
    [`RichardsonLucy`][optix.deconv.RichardsonLucy]. The paper uses a
    periodic (circulant) model; here the object extends past the image,
    and on the margin of that domain (outside the image) the matched
    update Hᵀ is used (see
    [`BackProjection`][optix.deconv.utils.BackProjection]; not in the
    paper).

    !!! note "Extensions (off by default)"
        - **χ² stopping rule:** `chi2_stop=True` stops with Lucy's χ²
          test (`alpha`, `max_iter`) instead of after `n_iter`
          iterations.
        - **Uniform start:** `start="uniform"` starts from the mean of
          the image; the first iteration is then the linear filter
          e⁽¹⁾ = y * b (Supplementary Note 3).
        - The extensions of
          [`RichardsonLucy`][optix.deconv.RichardsonLucy].
    !!! references "References"

        1. Guo M, Li Y, Su Y, Lambert T, Dalle Nogare D, Moyle MW, Duncan
           LH, Ikegami R, Santella A, Rey-Suarez I, Green D, Beiriger A,
           Chen J, Vishwasrao H, Ganesan S, Prince V, Waters JC, Annunziata
           CM, Hafner M, Mohler WA, Chitnis AB, Upadhyaya A, Usdin TB, Bao
           Z, Colón-Ramos D, La Riviere P, Liu H, Wu Y, Shroff H. *Rapid
           image deconvolution and multiview fusion for optical
           microscopy.* Nat Biotechnol **38**(11):1337-1346, 2020.
           [doi:10.1038/s41587-020-0560-x](https://doi.org/10.1038/s41587-020-0560-x)
        2. Zeng GL, Gullberg GT. *Unmatched projector/backprojector pairs
           in an iterative reconstruction algorithm.* IEEE Trans Med
           Imaging **19**(5):548-555, 2000.
           [doi:10.1109/42.870265](https://doi.org/10.1109/42.870265)

        ??? quote "BibTeX"

            ```bibtex
            @article{guo2020rapid,
              author  = {Guo, Min and Li, Yue and Su, Yijun and Lambert,
                         Talley and Dalle Nogare, Damian and Moyle, Mark W.
                         and Duncan, Leighton H. and Ikegami, Richard and
                         Santella, Anthony and Rey-Suarez, Ivan and Green,
                         Daniel and Beiriger, Anastasia and Chen, Jiji and
                         Vishwasrao, Harshad and Ganesan, Sundar and
                         Prince, Victoria and Waters, Jennifer C. and
                         Annunziata, Christina M. and Hafner, Markus and
                         Mohler, William A. and Chitnis, Ajay B. and
                         Upadhyaya, Arpita and Usdin, Ted B. and Bao,
                         Zhirong and Col{\'o}n-Ramos, Daniel and
                         La Riviere, Patrick and Liu, Huafeng and Wu,
                         Yicong and Shroff, Hari},
              title   = {Rapid image deconvolution and multiview fusion
                         for optical microscopy},
              journal = {Nature Biotechnology},
              volume  = {38},
              number  = {11},
              pages   = {1337--1346},
              year    = {2020},
              doi     = {10.1038/s41587-020-0560-x}
            }
            @article{zeng2000unmatched,
              author  = {Zeng, Gengsheng L. and Gullberg, Grant T.},
              title   = {Unmatched projector/backprojector pairs in an
                         iterative reconstruction algorithm},
              journal = {IEEE Transactions on Medical Imaging},
              volume  = {19},
              number  = {5},
              pages   = {548--555},
              year    = {2000},
              doi     = {10.1109/42.870265}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.deconv import WienerButterworth
    >>> x = torch.arange(15.0) - 7
    >>> g = torch.exp(-(x**2) / 8)
    >>> psf = g[:, None] * g
    >>> data = torch.poisson(torch.full((64, 64), 10.0))
    >>> wb = WienerButterworth(psf).fit(data)
    >>> wb.apply(data).shape
    torch.Size([64, 64])
    """

    tags = Tags(
        targets=("deconvolution",),
        streaming="halo",
        noise="poisson",
        reference="doi:10.1038/s41587-020-0560-x",
    )
    paper = {
        "back_projector": "wiener-butterworth",
        "wiener": 0.001,
        "cutoff_gain": 0.001,
        "order": 8,
        "fwhm": None,
        "start": "data",
        "n_iter": 1,
        "chi2_stop": False,
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
        back_projector: str | None = None,
        wiener: float | None = None,
        cutoff_gain: float | None = None,
        order: float | None = None,
        fwhm: float | Sequence[float] | None = None,
        start: str | None = None,
        n_iter: int | None = None,
        chi2_stop: bool | None = None,
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
            Point spread function (forward projector f), `(*K)` with 2 or
            3 dims, in the order of the spatial axes of the layout,
            centred at index `K // 2`. Normalised to unit sum.
        back_projector : str
            `"wiener-butterworth"` (default), `"butterworth"`,
            `"gaussian"` or `"traditional"` (plain Richardson–Lucy); see
            [`back_projector`][optix.deconv.utils.back_projector].
        wiener : float
            Wiener parameter α. Default (Table S2.1, iSIM): 0.001.
        cutoff_gain : float
            Butterworth gain β at the cutoff frequency. Default (Table
            S2.1, iSIM): 0.001.
        order : float
            Butterworth order n. Default (Table S2.1, iSIM): 8.
        fwhm : float or sequence of float, optional
            Resolution per spatial axis, in pixels: the cutoff frequency
            is 1/FWHM. Default: the FWHM of `psf`, measured with
            [`psf_fwhm`][optix.deconv.utils.psf_fwhm].
        start : str
            Starting estimate: `"data"` (default, Methods), the measured
            image (background removed) with values below 0.001 replaced
            by 0.001; or `"uniform"` (extension), its mean.
        n_iter : int
            Number of iterations. Default (Table S2.1, iSIM): 1.
        chi2_stop : bool
            Extension. Stop with the χ² test of
            [`RichardsonLucy`][optix.deconv.RichardsonLucy] instead of
            after `n_iter` iterations. Default: False.
        alpha : float
            Level of the χ² stopping test (with `chi2_stop`). Default:
            0.05.
        max_iter : int
            Iteration cap of the χ² stopping rule. Default: 100.
        background, poisson_gaussian, damping, halo, block
            Extensions and blocking, as in
            [`RichardsonLucy`][optix.deconv.RichardsonLucy].
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super(RichardsonLucy, self).__init__(
            dtype=dtype, device=device, layout=layout
        )
        params = {k: v for k, v in locals().items() if k != "psf"}
        self._resolve(faithful, params)
        self._faithful_options = self.is_faithful
        self.psf = psf
        self.sensor: SensorEstimate | None = None
        self.level: Tensor | None = None
        self._backs: dict[tuple, BackProjection] = {}
        opts = self.options
        if psf is not None and psf.ndim not in (2, 3):
            raise ValueError(f"psf must be 2-D or 3-D, got {psf.ndim} dims")
        if opts["back_projector"] not in _KINDS:
            raise ValueError(f"back_projector must be one of {_KINDS}")
        if opts["start"] not in _STARTS:
            raise ValueError(f"start must be one of {_STARTS}")
        if int(opts["n_iter"]) < 0:
            raise ValueError("n_iter must be non-negative")
        if not 0.0 < float(opts["damping"]) <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        if not 0.0 < float(opts["alpha"]) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 < float(opts["cutoff_gain"]) < 1.0:
            raise ValueError("cutoff_gain must be in (0, 1)")
        if not float(opts["wiener"]) > 0:
            raise ValueError("wiener must be positive")
        if not float(opts["order"]) > 0:
            raise ValueError("order must be positive")

    def kernel(self, **backend: object) -> Tensor:
        """The back-projector kernel b of the PSF, `(*K)`.

        Parameters
        ----------
        **backend
            `dtype` and `device` of the result (default: those of the
            estimator's backend).

        Returns
        -------
        Tensor
            Kernel b, of unit sum.
        """
        if not backend:
            backend = self._backend(None)
        opts = self.options
        return back_projector(
            self._psf(backend),
            opts["back_projector"],
            wiener=float(opts["wiener"]),
            cutoff_gain=float(opts["cutoff_gain"]),
            order=float(opts["order"]),
            fwhm=opts["fwhm"],
        )

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        self._backs = {}
        return super()._apply(data, out)

    def _back(self, op: PaddedConvolution) -> BackProjection | None:
        """Back projector of a block grid (None: the transpose Hᵀ)."""
        if self.options["back_projector"] == "traditional":
            return None
        norm = op.norm
        key = (op.shape, norm.dtype, norm.device)
        if key not in self._backs:
            b = self.kernel(dtype=norm.dtype, device=norm.device)
            self._backs[key] = BackProjection(b, op)
        return self._backs[key]

    def _solve(
        self, blk: _Block, op: PaddedConvolution, level: float
    ) -> tuple[Tensor, int]:
        opts = self.options
        start: float | Tensor = level
        if opts["start"] == "data":
            obj = (blk.data - blk.background).clamp(min=_START_FLOOR)
            start = op.pad(obj - level) + level
        chi2 = bool(opts["chi2_stop"])
        return rl_block(
            blk.data,
            op,
            start,
            n_iter=None if chi2 else int(opts["n_iter"]),
            alpha=float(opts["alpha"]),
            max_iter=int(opts["max_iter"]),
            background=blk.background,
            damping=float(opts["damping"]),
            back=self._back(op),
        )


def wiener_butterworth(
    data: ArrayLike,
    psf: Tensor,
    *,
    out: WritableArrayLike | None = None,
    sensor: SensorEstimate | None = None,
    layout: Layout | None = None,
    **options: object,
) -> WritableArrayLike:
    """Richardson–Lucy deconvolution with a Wiener–Butterworth back projector.

    Shortcut for `WienerButterworth(psf, **options).fit(data,
    init=sensor).apply(data, out)`; see
    [`WienerButterworth`][optix.deconv.WienerButterworth] for the model,
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
        Options of [`WienerButterworth`][optix.deconv.WienerButterworth]
        (`back_projector`, `wiener`, `cutoff_gain`, `order`, `n_iter`,
        ...).

    Returns
    -------
    writable array-like
        `out`, the deconvolved images.
    """
    est = WienerButterworth(psf, layout=layout, **options)
    est.fit(data, init=sensor)
    return est.apply(data, out)
