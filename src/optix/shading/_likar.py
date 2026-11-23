"""Retrospective shading correction by entropy minimisation (Likar 2000)."""

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .._estimator import Estimator, Tags
from ..estimates import ShadingEstimate
from ..io import ArrayLike, Labeled, Layout, WritableArrayLike, frame_batches
from ..stats import MinMax

__all__ = ["Likar"]

_COMPONENTS = ("both", "additive", "multiplicative")
_NBASIS = 5  # x, y, xy, x² − W²/12, y² − H²/12

_PRESET: dict[str, object] = {
    "levels": 256,
    "data_range": None,
    "window": 2,
    "components": "both",
    "sampling": 1,
    "tol": 1e-6,
    "max_iterations": 200,
}


# -- model -------------------------------------------------------------------
def _basis(
    shape: Sequence[int], step: int = 1, *, device: object = None
) -> Tensor:
    """Neutralised quadratic terms (Eqs. 16-17) in scaled coordinates.

    Returns `(5, h, w)` float64 maps of x̃, ỹ, x̃ỹ, x̃² − 1/3, ỹ² − 1/3,
    with x̃ = 2x/W and ỹ = 2y/H, the origin at the image centre, sampled
    every `step` pixels. In pixel units, x̃² − 1/3 = (4/W²)(x² − W²/12).
    """
    H, W = (int(s) for s in shape)
    kw = {"dtype": torch.float64, "device": device}
    y = (torch.arange(0, H, step, **kw) - (H - 1) / 2) * (2 / H)
    x = (torch.arange(0, W, step, **kw) - (W - 1) / 2) * (2 / W)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([X, Y, X * Y, X * X - 1 / 3, Y * Y - 1 / 3])


def _paper_coefficients(
    theta: Sequence[float], shape: Sequence[int]
) -> list[float]:
    """Scaled-coordinate coefficients → the paper's (a₁ … a₅) or (m₁ … m₅).

    The paper's polynomials use pixel coordinates x, y with the origin
    at the image centre.
    """
    H, W = (float(s) for s in shape)
    scale = (2 / W, 2 / H, 4 / (W * H), 4 / W**2, 4 / H**2)
    return [float(t) * s for t, s in zip(theta, scale, strict=True)]


# -- entropy -----------------------------------------------------------------
def _blurred_entropy(
    position: Tensor, weight: Tensor, levels: int, window: int
) -> Tensor:
    """Entropy of the partial-volume histogram of grey `position`s.

    Eqs. 19-20: each value g′ updates bins k = ⌊g′⌋ and k + 1 by
    k + 1 − g′ and g′ − k (times its `weight`), on the grey range
    [−G, 2G]; the histogram is blurred by the triangular window
    t + 1 − |i|, |i| ≤ t, normalised, and H = −∑ p log p (natural log).
    Values outside [−G, 2G] are clamped to it.
    """
    G = levels
    nbins = 3 * G + 1
    p = position.clamp(-G, 2 * G)
    k = p.floor().clamp(max=2 * G - 1)
    frac = p - k
    idx = (k + G).long()
    h = torch.bincount(idx, weight * (1 - frac), minlength=nbins)
    h = h + torch.bincount(idx + 1, weight * frac, minlength=nbins)
    t = int(window)
    if t > 0:
        i = torch.arange(-t, t + 1, dtype=h.dtype, device=h.device)
        kernel = (t + 1 - i.abs()).view(1, 1, -1)
        h = F.conv1d(h.view(1, 1, -1), kernel, padding=t).view(-1)
    q = h / h.sum()
    q = q[q > 0]
    return -(q * q.log()).sum()


# -- optimiser (Numerical Recipes, Press et al. 1992, §10.2, §10.5) -----------
_GOLD = 1.618034
_GLIMIT = 100.0
_TINY = 1e-20
_CGOLD = 0.3819660
_ZEPS = 1e-10


def _bracket(
    f: Callable[[float], float], ax: float, bx: float, fa: float
) -> tuple[float, float, float, float]:
    """`mnbrak`: a, b, c with f(b) <= f(a), f(c). Returns (a, b, c, f(b))."""
    fb = f(bx)
    if fb > fa:
        ax, bx, fa, fb = bx, ax, fb, fa
    cx = bx + _GOLD * (bx - ax)
    fc = f(cx)
    for _ in range(1000):
        if fb <= fc:
            break
        r = (bx - ax) * (fb - fc)
        q = (bx - cx) * (fb - fa)
        den = 2 * math.copysign(max(abs(q - r), _TINY), q - r)
        u = bx - ((bx - cx) * q - (bx - ax) * r) / den
        ulim = bx + _GLIMIT * (cx - bx)
        if (bx - u) * (u - cx) > 0:
            fu = f(u)
            if fu < fc:
                return bx, u, cx, fu
            if fu > fb:
                return ax, bx, u, fb
            u = cx + _GOLD * (cx - bx)
            fu = f(u)
        elif (cx - u) * (u - ulim) > 0:
            fu = f(u)
            if fu < fc:
                bx, cx, u = cx, u, u + _GOLD * (u - cx)
                fb, fc, fu = fc, fu, f(u)
        elif (u - ulim) * (ulim - cx) >= 0:
            u = ulim
            fu = f(u)
        else:
            u = cx + _GOLD * (cx - bx)
            fu = f(u)
        ax, bx, cx = bx, cx, u
        fa, fb, fc = fb, fc, fu
    return ax, bx, cx, fb


def _brent(
    f: Callable[[float], float],
    ax: float,
    bx: float,
    cx: float,
    fbx: float,
    tol: float = 2e-4,
    itmax: int = 100,
) -> tuple[float, float]:
    """`brent`: minimum of f bracketed by (ax, bx, cx). Returns (x, f(x))."""
    a, b = min(ax, cx), max(ax, cx)
    x = w = v = bx
    fx = fw = fv = fbx
    d = e = 0.0
    for _ in range(itmax):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + _ZEPS
        tol2 = 2 * tol1
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            break
        golden = True
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            etemp, e = e, d
            if not (
                abs(p) >= abs(0.5 * q * etemp)
                or p <= q * (a - x)
                or p >= q * (b - x)
            ):
                golden = False
                d = p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = math.copysign(tol1, xm - x)
        if golden:
            e = (a - x) if x >= xm else (b - x)
            d = _CGOLD * e
        u = x + d if abs(d) >= tol1 else x + math.copysign(tol1, d)
        fu = f(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x, fx


def _linmin(
    f: Callable[[np.ndarray], float], p: np.ndarray, xi: np.ndarray, fp: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """`linmin`: minimise f along `xi` from `p` (Brent, tol 2·10⁻⁴)."""

    def line(t: float) -> float:
        return f(p + t * xi)

    ax, bx, cx, fb = _bracket(line, 0.0, 1.0, fp)
    xmin, fret = _brent(line, ax, bx, cx, fb)
    xi = xi * xmin
    return p + xi, xi, fret


def _powell(
    f: Callable[[np.ndarray], float],
    p: np.ndarray,
    *,
    ftol: float,
    itmax: int,
    on_iteration: Callable[[np.ndarray, float], None] | None = None,
) -> tuple[np.ndarray, float, bool]:
    """`powell`: direction-set minimisation from the unit directions.

    Returns (p, f(p), converged).
    """
    n = p.size
    xi = np.eye(n)
    p = p.astype(np.float64).copy()
    fret = f(p)
    pt = p.copy()
    for _ in range(itmax):
        fp = fret
        ibig, delta = 0, 0.0
        for i in range(n):
            fptt = fret
            p, xi[:, i], fret = _linmin(f, p, xi[:, i].copy(), fret)
            if fptt - fret > delta:
                delta, ibig = fptt - fret, i
        if on_iteration is not None:
            on_iteration(p, fret)
        if 2 * (fp - fret) <= ftol * (abs(fp) + abs(fret)) + 1e-25:
            return p, fret, True
        ptt = 2 * p - pt
        xit = p - pt
        pt = p.copy()
        fptt = f(ptt)
        if fptt < fp:
            t = 2 * (fp - 2 * fret + fptt) * (fp - fret - delta) ** 2
            t -= delta * (fp - fptt) ** 2
            if t < 0:
                p, xit, fret = _linmin(f, p, xit, fret)
                xi[:, ibig] = xi[:, n - 1]
                xi[:, n - 1] = xit
    return p, fret, False


# -- estimator ---------------------------------------------------------------
class _GreyCounts:
    """Counts of (pixel, grey level) pairs: a mergeable sparse histogram."""

    def __init__(self, levels: int) -> None:
        self.levels = levels
        self.keys: Tensor | None = None
        self.counts: Tensor | None = None

    def update(self, pixel: Tensor, grey: Tensor) -> None:
        keys = pixel.reshape(-1) * self.levels + (grey.reshape(-1) - 1)
        counts = torch.ones_like(keys)
        if self.keys is not None:
            keys = torch.cat([self.keys, keys])
            counts = torch.cat([self.counts, counts])
        self.keys, inverse = torch.unique(keys, return_inverse=True)
        self.counts = torch.zeros_like(self.keys).scatter_add_(
            0, inverse, counts
        )

    def pairs(self) -> tuple[Tensor, Tensor, Tensor]:
        """Pixel index, grey level (1 … G) and count of each pair."""
        return (
            self.keys // self.levels,
            self.keys % self.levels + 1,
            self.counts,
        )


class Likar(Estimator):
    """Retrospective shading correction by entropy minimisation.

    Likar et al. (2000) model the acquired image N as the true image U
    with a multiplicative and an additive shading component,

        N(x, y) = U(x, y) · S_M(x, y) + S_A(x, y),            (Eq. 3)

    and correct it by inverting the model,
    Û = (N − Ŝ_A) / Ŝ_M (Eq. 5). Both components are second-order
    polynomials, with the origin at the image centre, neutralised so
    that ∫∫ Ŝ_A = 0 and (1/WH) ∫∫ Ŝ_M = 1 over the W × H image
    (Eqs. 14-17):

        Ŝ_A = a₁x + a₂y + a₃xy + a₄(x² − W²/12) + a₅(y² − H²/12)
        Ŝ_M = 1 + m₁x + m₂y + m₃xy + m₄(x² − W²/12) + m₅(y² − H²/12)

    The ten parameters minimise the Shannon entropy of the corrected
    image (Eq. 18),

        {a, m} = argmin H((N − Ŝ_A) / Ŝ_M),   H = −∑ₙ p(n) log p(n),

    found with Powell's direction-set method and Brent's line
    minimisation (Press et al., 1992). The histogram is built with
    partial intensity interpolation (a grey value g transformed to
    g′ ∈ [k, k + 1] adds k + 1 − g′ to bin k and g′ − k to bin k + 1),
    on the range [−G, 2G] of G grey levels, and blurred by a triangular
    window before normalisation (Eq. 20):

        *h(n) = ∑_{i=−t}^{t} h(n + i) (t + 1 − |i|),   t = 2.

    Pixels at the marginal grey values 1 and G of the input image are
    ignored (they may be saturated).

    Grey levels are defined on the data range [lo, hi] (by default the
    observed minimum and maximum of each group): a value v has grey level
    g = round((v − lo)/δ) + 1 with δ = (hi − lo)/(G − 1), so that 8-bit
    data with `data_range=(0, 255)` and `levels=256` use its integer
    values. The model and the correction are in data units.

    !!! note "Several images"
        The paper corrects one image at a time. Given several frames
        (the sample axes of the layout), one shading per group is fitted
        to minimise the entropy of the histogram of all corrected frames
        together; fit each frame separately for per-image shading.

    !!! note "Streaming"
        The fit reads the data once (twice without `data_range`): the
        statistic is the count of each (pixel, grey level) pair, which
        holds at most min(frames, G) entries per pixel. The objective is
        then evaluated on these counts. `apply` streams frame batches.

    Estimates: `"shading"`, a
    [`ShadingEstimate`][optix.estimates.ShadingEstimate] with
    `flat` = Ŝ_M and `dark` = Ŝ_A, maps `(*group, H, W)` in data units,
    and `meta["flat_norm"] = "mean"` (Eq. 15); `apply` returns
    (N − dark) / flat (Eq. 5). `meta["coefficients"]` holds the paper's
    (a₁ … a₅) and (m₁ … m₅) per group, in pixel coordinates and data
    units, and `meta["entropy"]` the final entropies (nats).

    !!! references "References"
        1. Likar B, Maintz JBA, Viergever MA, Pernuš F. *Retrospective
           shading correction based on entropy minimization.* J Microsc
           **197**(3):285–295, 2000.
           [doi:10.1046/j.1365-2818.2000.00669.x](https://doi.org/10.1046/j.1365-2818.2000.00669.x)
        2. Press WH, Teukolsky SA, Vetterling WT, Flannery BP. *Numerical
           Recipes in C.* 2nd edn. Cambridge University Press, 1992.

        ??? quote "BibTeX"
            ```bibtex
            @article{likar2000retrospective,
              author  = {Likar, B. and Maintz, J. B. A. and
                         Viergever, M. A. and Pernu{\\v{s}}, F.},
              title   = {Retrospective shading correction based on
                         entropy minimization},
              journal = {Journal of Microscopy},
              volume  = {197}, number = {3}, pages = {285--295},
              year    = {2000},
              doi     = {10.1046/j.1365-2818.2000.00669.x}
            }
            @book{press1992numerical,
              author    = {Press, W. H. and Teukolsky, S. A. and
                           Vetterling, W. T. and Flannery, B. P.},
              title     = {Numerical Recipes in {C}},
              edition   = {2nd},
              publisher = {Cambridge University Press},
              year      = {1992}
            }
            ```

    Examples
    --------
    >>> est = Likar().fit(image)
    >>> corrected = est.apply(image)
    """

    tags = Tags(
        targets=("shading",),
        streaming="per-pixel-stats",
        modality=("brightfield", "reflectance", "fluorescence"),
        noise="none",
        many_frames=False,
        reference="doi:10.1046/j.1365-2818.2000.00669.x",
    )
    paper = dict(_PRESET)
    recommended = dict(_PRESET)

    def __init__(
        self,
        *,
        levels: int | None = None,
        data_range: tuple[float, float] | None = None,
        window: int | None = None,
        components: str | None = None,
        sampling: int | None = None,
        tol: float | None = None,
        max_iterations: int | None = None,
        faithful: bool = True,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        layout: Layout | None = None,
    ) -> None:
        """Set the options; unset ones come from the preset.

        Parameters
        ----------
        levels : int
            Number G of grey levels. Default 256 (the paper's 8-bit
            images).
        data_range : (float, float), optional
            Data values [lo, hi] of grey levels 1 and G. Default: the
            observed minimum and maximum of each group. Not specified in
            the paper, whose images are 8-bit integers. It describes the
            data and does not affect `is_faithful`.
        window : int
            Half-width t of the triangular histogram blur (Eq. 20).
            Default 2 (window of 2t + 1 = 5 bins); 0 disables it.
        components : {"both", "additive", "multiplicative"}
            Shading components to fit. Extension: the paper fits both;
            fitting a single one is one of the modifications its
            discussion suggests.
        sampling : int
            Extension: use every `sampling`-th pixel along each spatial
            axis in the fit (the paper's discussion suggests sparse
            sampling to speed it up). Default 1 (all pixels).
        tol : float
            Fractional tolerance `ftol` of Powell's method on the entropy.
            Default 10⁻⁶. The line minimisations use Brent's method with
            tolerance 2·10⁻⁴ (Numerical Recipes). Not specified in the
            paper.
        max_iterations : int
            Maximum number of Powell iterations. Default 200. Not
            specified in the paper.
        faithful : bool
            Selects the preset; both presets are the paper's method.
        dtype, device, layout : optional
            See [`Estimator`][optix.Estimator]. The layout must have two
            spatial axes, the first (in layout order) being y, of size H,
            and the second x, of size W; each group (e.g. channel)
            gets its own shading.

        Notes
        -----
        Not specified in the paper, and chosen here: the optimisation
        starts from no shading (all parameters 0); Powell's initial
        directions are the unit vectors of parameters rescaled to
        coordinates x̃ = 2x/W, ỹ = 2y/H ∈ [−1, 1] (additive ones in
        units of hi − lo), which changes the scale of the search but
        not the model; pixel coordinates are pixel centres; corrected
        values outside [−G, 2G] are clamped to the end bins; parameters
        that make Ŝ_M ≤ 0 at a pixel are rejected (their objective
        exceeds the maximum entropy).
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        opt = self.options
        if opt["components"] not in _COMPONENTS:
            raise ValueError(
                f"components must be one of {_COMPONENTS}, "
                f"got {opt['components']!r}"
            )
        if int(opt["levels"]) < 3:
            raise ValueError(f"levels must be >= 3: {opt['levels']}")
        if int(opt["window"]) < 0:
            raise ValueError(f"window must be >= 0: {opt['window']}")
        if int(opt["sampling"]) < 1:
            raise ValueError(f"sampling must be >= 1: {opt['sampling']}")
        rng = opt["data_range"]
        if rng is not None and not float(rng[1]) > float(rng[0]):
            raise ValueError(f"data_range must be (lo, hi), lo < hi: {rng}")

    def _resolve(self, faithful: bool, params: dict) -> None:
        # `data_range` describes the data, not the method
        super()._resolve(faithful, params)
        self.is_faithful = all(
            value == self.paper.get(name)
            for name, value in self.options.items()
            if name != "data_range"
        )

    # -- data reduction ------------------------------------------------------
    def _counts(
        self, data: ArrayLike, spatial: tuple[int, int]
    ) -> tuple[dict, dict]:
        """Per-group (pixel, grey) counts and data ranges."""
        opt = self.options
        G = int(opt["levels"])
        s = int(opt["sampling"])
        ndim = None if isinstance(data, Labeled) else 2
        kw = {"ndim": ndim, "dtype": torch.float64, "overhead": 3.0}
        if self.device is not None:
            kw["device"] = self.device
        ranges: dict[tuple[int, ...], tuple[float, float]] = {}
        if opt["data_range"] is None:
            extrema: dict[tuple[int, ...], MinMax] = {}
            for chunk in frame_batches(data, **kw):
                sub = chunk.data[:, ::s, ::s]
                extrema.setdefault(chunk.group, MinMax(dim=None)).update(sub)
            for group, mm in extrema.items():
                lo, hi = float(mm.min), float(mm.max)
                if not hi > lo:
                    raise ValueError(f"Group {group} is constant: no shading")
                ranges[group] = (lo, hi)
        counts: dict[tuple[int, ...], _GreyCounts] = {}
        h = len(range(0, spatial[0], s))
        w = len(range(0, spatial[1], s))
        for chunk in frame_batches(data, **kw):
            if opt["data_range"] is None:
                lo, hi = ranges[chunk.group]
            else:
                lo, hi = (float(v) for v in opt["data_range"])
                ranges[chunk.group] = (lo, hi)
            sub = chunk.data[:, ::s, ::s]
            delta = (hi - lo) / (G - 1)
            grey = ((sub - lo) / delta).round().clamp(0, G - 1).long() + 1
            pixel = torch.arange(h * w, device=sub.device).view(1, h, w)
            pixel = pixel.expand_as(grey)
            keep = (grey > 1) & (grey < G)  # marginal grey values ignored
            counts.setdefault(chunk.group, _GreyCounts(G)).update(
                pixel[keep], grey[keep]
            )
        return counts, ranges

    # -- objective -----------------------------------------------------------
    def _objective(
        self,
        counts: _GreyCounts,
        lo: float,
        hi: float,
        basis: Tensor,
    ) -> Callable[[np.ndarray], float]:
        """f(θ) = H of the corrected grey levels; θ = (ã₁…ã₅, m̃₁…m̃₅)."""
        opt = self.options
        G = int(opt["levels"])
        t = int(opt["window"])
        delta = (hi - lo) / (G - 1)
        if counts.keys is None or counts.keys.numel() == 0:
            raise ValueError("No pixel inside the grey range to fit")
        pixel, grey, count = counts.pairs()
        device = pixel.device
        value = lo + (grey.to(torch.float64) - 1) * delta
        weight = count.to(torch.float64)
        B = basis.reshape(_NBASIS, -1).to(device)
        bad = math.log(3 * G + 1) + 1.0

        def f(theta: np.ndarray) -> float:
            th = torch.as_tensor(theta, dtype=torch.float64, device=device)
            sm = 1 + th[_NBASIS:] @ B
            smin = float(sm.min())
            if smin <= 0:
                return bad - smin
            sa = (hi - lo) * (th[:_NBASIS] @ B)
            u = (value - sa[pixel]) / sm[pixel]
            position = (u - lo) / delta + 1
            return float(_blurred_entropy(position, weight, G, t))

        return f

    # -- fit / apply ---------------------------------------------------------
    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        self._warn_unused(init)
        layout = self._layout_of(data)
        raw = data.array if isinstance(data, Labeled) else data
        group_shape, sample_shape, spatial = layout.shape_of(raw)
        if len(spatial) != 2:
            raise ValueError(
                f"Likar needs two spatial axes, the layout has {len(spatial)}"
            )
        if math.prod(sample_shape) < 1:
            raise ValueError("Likar needs at least one image per group")
        spatial = (int(spatial[0]), int(spatial[1]))
        opt = self.options
        backend = self._backend(data)
        counts, ranges = self._counts(data, spatial)
        components = opt["components"]
        free = np.zeros(2 * _NBASIS, dtype=bool)
        if components in ("both", "additive"):
            free[:_NBASIS] = True
        if components in ("both", "multiplicative"):
            free[_NBASIS:] = True
        coarse = _basis(spatial, int(opt["sampling"]))
        full = _basis(spatial)
        flats, darks, coefs, entropies, converged = [], [], [], [], []
        for group in itertools.product(*(range(g) for g in group_shape)):
            lo, hi = ranges[group]
            f_all = self._objective(counts[group], lo, hi, coarse)
            theta = np.zeros(2 * _NBASIS)

            def f(sub: np.ndarray, f_all: Callable = f_all) -> float:
                th = np.zeros(2 * _NBASIS)
                th[free] = sub
                return f_all(th)

            def step(p: np.ndarray, fret: float) -> None:
                self.history.record(-fret, entropy=fret)
                if callback is not None:
                    callback(self)

            self.history.record(-f(theta[free]), entropy=f(theta[free]))
            sub, fret, ok = _powell(
                f,
                theta[free],
                ftol=float(opt["tol"]),
                itmax=int(opt["max_iterations"]),
                on_iteration=step,
            )
            theta[free] = sub
            converged.append(ok)
            entropies.append(float(fret))
            th = torch.as_tensor(theta, dtype=torch.float64)
            dark = (hi - lo) * torch.einsum("k,khw->hw", th[:_NBASIS], full)
            flat = 1 + torch.einsum("k,khw->hw", th[_NBASIS:], full)
            darks.append(dark)
            flats.append(flat)
            coefs.append(
                {
                    "additive": [
                        c * (hi - lo)
                        for c in _paper_coefficients(theta[:_NBASIS], spatial)
                    ],
                    "multiplicative": _paper_coefficients(
                        theta[_NBASIS:], spatial
                    ),
                }
            )
            if verbose:
                print(
                    f"Likar group {group}: entropy {fret:.6g} nats, "
                    f"converged={ok}"
                )
        self.history.converged = all(converged)
        self.history.reason = (
            f"relative entropy change below tol={opt['tol']:g}"
            if all(converged)
            else f"max_iterations={opt['max_iterations']} reached"
        )
        flat = torch.stack(flats).reshape(*group_shape, *spatial)
        dark = torch.stack(darks).reshape(*group_shape, *spatial)
        self.estimates = {
            "shading": ShadingEstimate(
                flat=None if components == "additive" else flat.to(**backend),
                dark=(
                    None
                    if components == "multiplicative"
                    else dark.to(**backend)
                ),
                meta={
                    "flat_norm": "mean",
                    "model": "y = flat * x + dark",
                    "coefficients": coefs,
                    "entropy": entropies,
                    "data_range": [list(ranges[g]) for g in sorted(ranges)],
                },
            )
        }

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if "shading" not in self.estimates:
            raise RuntimeError("Likar is not fitted: call fit first")
        est = self.estimates["shading"]

        def pick(m: Tensor | None, group: tuple, like: Tensor) -> Tensor:
            if m is None:
                return None
            return (m[group] if group else m).to(like)

        def fn(chunk: object, batch: Tensor) -> Tensor:
            model = ShadingEstimate(
                flat=pick(est.flat, chunk.group, batch),
                dark=pick(est.dark, chunk.group, batch),
            )
            return model.correct(batch)

        return self._apply_frames(data, out, fn, overhead=3.0)
