"""CIDRE illumination correction (Smith et al., 2015)."""

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .._config import get_config
from .._estimator import Estimator, Tags
from ..estimates import ShadingEstimate
from ..io import ArrayLike, Labeled, Layout, WritableArrayLike, frame_batches
from ..stats import Histogram, MinMax, order_statistics
from ._basic_solvers import resize

__all__ = ["CIDRE"]

_MODES = ("dynamic_range", "zero_light", "direct")
_QUANTILE_METHODS = ("auto", "exact", "histogram")
_INITS = ("pivot", "least_squares")

_PRESET: dict[str, object] = {
    "working_size": 128,
    "quantiles": 100,
    "quantile_method": "auto",
    "bins": 1024,
    "lambda_r": None,
    "lambda_z": 0.5,
    "lambda_b": 1.0,
    "cauchy_width": 0.05,
    "barrier_width": 1e-3,
    "band": 0.25,
    "scales": (1.0, 2.0, 4.0, 8.0),
    "zero_light_limits": None,
    "max_iterations": 500,
    "history_size": 20,
    "tol": 1e-9,
    "init_method": "pivot",
    "normalize_fit": False,
}


def lambda_r_schedule(n: int) -> float:
    """Default λ_r for `n` images (Supplementary Note 3).

    Linear from 9.5 at N = 1 to 6 at N = 200, and 6 beyond.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n >= 200:
        return 6.0
    return 9.5 - 3.5 * (n - 1) / 199


def _working_shape(
    size: int | Sequence[int] | None, spatial: Sequence[int]
) -> tuple[int, int] | None:
    """Working grid: long side `size` (int), an explicit pair, or None."""
    H, W = (int(s) for s in spatial)
    if size is None:
        return None
    if isinstance(size, Sequence):
        pair = tuple(int(s) for s in size)
        if len(pair) != 2 or min(pair) < 1:
            raise ValueError(f"working_size must be an int or a pair: {size}")
        return None if pair == (H, W) else pair
    size = int(size)
    if size < 1:
        raise ValueError(f"working_size must be >= 1: {size}")
    long = max(H, W)
    if long <= size:
        return None
    return (max(1, round(H * size / long)), max(1, round(W * size / long)))


def _log_filters(sigma: float, dtype: torch.dtype, device: object) -> tuple:
    """1-D Gaussian and second-derivative kernels at scale `sigma`.

    The sampled second derivative is corrected to sum to 0 (a constant
    has no curvature) and to have second moment 2 (x² has curvature 2).
    """
    r = max(1, math.ceil(4 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float64)
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    g2 = g * (x**2 / sigma**4 - 1 / sigma**2)
    g2 = g2 - g * g2.sum()
    g2 = g2 * (2 / (x**2 * g2).sum())
    backend = {"dtype": dtype, "device": device}
    return r, g.to(**backend), g2.to(**backend)


def _odd_pad(x: Tensor, r: int, dim: int) -> Tensor:
    """Pad `x` by `r` along `dim` by point reflection, f(−k) = 2 f(0) − f(k).

    Linear trends continue across the border; pads longer than the axis
    are built by repeated reflection.
    """
    while r > 0:
        n = x.shape[dim]
        if n < 2:
            shape = list(x.shape)
            shape[dim] = r
            edge = x.narrow(dim, 0, 1).expand(shape)
            return torch.cat([edge, x, edge], dim)
        k = min(r, n - 1)
        first, last = x.narrow(dim, 0, 1), x.narrow(dim, n - 1, 1)
        left = 2 * first - x.narrow(dim, 1, k).flip(dim)
        right = 2 * last - x.narrow(dim, n - 1 - k, k).flip(dim)
        x = torch.cat([left, x, right], dim)
        r -= k
    return x


def normalized_log(v: Tensor, sigma: float) -> Tensor:
    """Scale-normalised Laplacian of Gaussian σ² ∇²G_σ ∗ v of a 2-D map.

    Separable, the borders extended by point reflection: constant and
    linear maps give 0.
    """
    r, g, g2 = _log_filters(sigma, v.dtype, v.device)
    k = g.numel()
    x = v[None, None]

    def conv(t: Tensor, kernel: Tensor, dim: int) -> Tensor:
        shape = (1, 1, k, 1) if dim == 2 else (1, 1, 1, k)
        return F.conv2d(_odd_pad(t, r, dim), kernel.view(shape))

    gy, g2y = conv(x, g, 2), conv(x, g2, 2)
    out = conv(g2y, g, 3) + conv(gy, g2, 3)
    return sigma**2 * out[0, 0]


def _barrier(x: Tensor, lo: float, hi: float, width: float) -> Tensor:
    """Eq. 13: zero inside [lo, hi], quadratic outside."""
    below = torch.clamp(lo - x, min=0) / width
    above = torch.clamp(x - hi, min=0) / width
    return below**2 + above**2


@dataclass
class _Problem:
    """Scaled data of one group, in pivot space (Eqs. 19-20)."""

    Qp: Tensor  # (K,) Q − median(Q)
    qp: Tensor  # (K, h, w) q − median(q)
    Q_med: Tensor  # ()
    q_med: Tensor  # (h, w)
    alpha: tuple[float, float]


def cidre_energy(
    v: Tensor,
    bp: Tensor,
    ZQ: Tensor,
    Zq: Tensor,
    problem: _Problem,
    *,
    lambda_r: float,
    lambda_z: float,
    lambda_b: float,
    cauchy_width: float,
    barrier_width: float,
    scales: Sequence[float],
    normalize_fit: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Energy of Eq. 9 and its terms (Eqs. 10-14).

    With `normalize_fit`, E_f is divided by the number K of quantiles.
    """
    w = cauchy_width
    r = problem.Qp[:, None, None] * v + bp - problem.qp
    Ef = (0.5 * w**2 * torch.log1p((r / w) ** 2)).sum()
    if normalize_fit:
        Ef = Ef / problem.Qp.numel()
    nT = len(scales)
    Er = v.new_zeros(())
    for t in scales:
        Er = Er + ((normalized_log(v, t) / nT) ** 2).sum()
    b = bp - v * problem.Q_med + problem.q_med
    Ez = ((v * ZQ + b - Zq) ** 2).sum()
    lo, hi = problem.alpha
    Eb = _barrier(ZQ, lo, hi, barrier_width) + _barrier(
        Zq, lo, hi, barrier_width
    )
    E = Ef + lambda_r * Er + lambda_z * Ez + lambda_b * Eb
    return E, {"Ef": Ef, "Er": Er, "Ez": Ez, "Eb": Eb}


class CIDRE(Estimator):
    """CIDRE: illumination gain and zero-light term by energy minimisation.

    CIDRE (corrected intensity distributions using regularised energy
    minimisation) estimates, from many images acquired under the same
    conditions, the illumination gain v and the additive zero-light
    term z of

        I_o(x) = I(x) v(x) + z(x)                            (1)

    It assumes that objects appear anywhere with equal probability, so
    that the sorted intensities q(x) at every location are a linear
    transform of a distribution common to all locations:

        q(x) = v(x) Q + b(x),   b(x) = z(x) − v(x) Z_Q       (7, 8)

    with Z_Q the zero-light point.

    **Target distribution** (Eqs. 5-6). Q is the robust mean of the
    q(x) whose mean q̄(x) lies in the central band
    [ρ_{½−ω}, ρ_{½+ω}] of {q̄(x)}, ρ_u being the u-th quantile over the
    locations.

    **Energy** (Eq. 9). v, b and the zero-light point (Z_Q, Z_q)
    minimise

        E = E_f + λ_r E_r + λ_z E_z + λ_b E_b

    with

    - E_f = ∑_x ∑_n (w²/2) log(1 + ((Q_n v(x) + b(x) − q_n(x)) / w)²),
      a Cauchy fit of the K sorted values (Eq. 10);
    - E_r = ∑_{t∈T} ∑_{i,j} ((1/|T|) t² ∇²G_t ∗ v(i, j))², the
      scale-normalised Laplacian of Gaussian of v (Eq. 11), t being the
      standard deviation of the Gaussian;
    - E_z = ∑_x (v(x) Z_Q + b(x) − Z_q)², which pulls every regression
      line through the zero-light point (Eq. 12);
    - E_b = β(Z_Q) + β(Z_q), with β zero on [α₁, α₂] and
      ((x − α)/w_b)² outside (Eqs. 13-14). By default α₁ = 0 and
      α₂ = min q(x), the smallest value of the data.

    The default λ_z is 0.5 and λ_r goes linearly from 9.5 for one image
    to 6 for 200 images, and is 6 beyond. E is minimised by L-BFGS in
    the pivot space (Eqs. 19-20), where Q and q(x) are shifted by their
    medians; then z = b + v Z_Q (Eq. 7).

    **Correction.** [`apply`][optix.shading.CIDRE.apply] reverses Eq. 1
    and, by default, rescales the result by the mean gain v̄ ("dynamic
    range corrected"):

        I = (I_o − z) / v · v̄        (+ z̄ for "zero-light preserved")

    !!! note
        The fit makes one pass over the data, plus a second one when the
        per-pixel distributions are summarised by histograms. Each frame
        is first resampled to the working grid (bilinear, with
        anti-aliasing), then the K quantiles are computed at every location
        of the grid. The fitted v and z are resampled back to the image size
        (bilinear). [`apply`][optix.shading.CIDRE.apply] works on frame
        batches.

    !!! note "Implementation choices"
        - The intensities are divided by their maximum (over the
          working grid) before the fit, so that the default weights and
          widths apply whatever the range of the data. Not specified in
          the paper.
        - Each location is summarised by K = 100 quantiles at levels
          (k − ½)/K (all values when there are fewer images), so E_f has
          at most 100 terms per location whatever the number of images.
          They are interpolated linearly between exact order statistics
          (rank N u + ½ for level u) when the resampled stack fits the
          memory budget (`quantile_method="auto"`), else read from
          per-pixel histograms of `bins` bins over the per-pixel range.
          Not specified in the paper.
        - The pivot space reparametrises b only: the medians shift Q
          and q, and b is recovered exactly, so the energy is unchanged.
        - The gradients are those of Eqs. 10-14 as written, obtained by
          automatic differentiation; the optimisation runs in float64.
        - Q is the mean over the central band (Eq. 6 prints a sum; the
          text calls it a robust mean).
        - The scale-space pyramid that the authors use when few images
          are available is not part of the method here: the paper does
          not give the function from entropy to the number of octaves.
        - The Laplacian of Gaussian is normalised by t² with t the
          standard deviation (Supplementary Note 3 calls t the variance
          but multiplies by t², which is scale-invariant only for the
          standard deviation).
        - The optimiser works on rescaled variables (a diagonal
          preconditioning that leaves the energy unchanged).
        - Not specified in the paper: the working grid (long side 128),
          the Cauchy width w = 0.05 and the barrier width w_b = 10⁻³
          (in units of the maximum intensity), λ_b = 1, ω = 0.25, the
          scales T = {1, 2, 4, 8} (pixels of a grid whose long side is
          128, rescaled with the working grid), the borders of the
          filters (point reflection, so that linear trends are not
          penalised), the L-BFGS settings and the initialisation (v = 1
          and each line through the pivot; Z_Q = Z_q = (α₁ + α₂)/2).

    !!! references "References"
        1. Smith K, Li Y, Piccinini F, Csucs G, Balazs C, Bevilacqua A,
           Horvath P. *CIDRE: an illumination-correction method for
           optical microscopy.* Nat Methods **12**(5):404–406, 2015.
           [doi:10.1038/nmeth.3323](https://doi.org/10.1038/nmeth.3323)

        ??? quote "BibTeX"
            ```bibtex
            @article{smith2015cidre,
              author  = {Smith, Kevin and Li, Yunpeng and Piccinini,
                         Filippo and Csucs, Gabor and Balazs, Csaba and
                         Bevilacqua, Alessandro and Horvath, Peter},
              title   = {{CIDRE}: an illumination-correction method for
                         optical microscopy},
              journal = {Nature Methods},
              volume  = {12},
              number  = {5},
              pages   = {404--406},
              year    = {2015},
              doi     = {10.1038/nmeth.3323},
            }
            ```

    !!! note "Extension (off by default)"
        `normalize_fit=True` divides E_f by the number K of quantiles (the
        λ then weigh 1/K as much data).

    Attributes
    ----------
    estimates : dict
        `{"shading": ShadingEstimate}` with `flat` = v and `dark` = z,
        shaped `(*group, *spatial)`. v is relative to the target
        distribution Q and is stored as produced
        (`meta["flat_norm"] == "none"`); z includes the camera offset.
    zero_light_ : (*group, 2) Tensor
        Zero-light point (Z_Q, Z_q) of each group, in data units.
    lambda_r_ : float
        λ_r used by the last fit.
    target_ : (*group, K) Tensor
        Target distribution Q of each group, in data units.

    Examples
    --------
    >>> import torch
    >>> from optix.models import random_shading, simulate_shading
    >>> truth = random_shading((64, 64), dark=20.0, seed=0)
    >>> x = 100 + 400 * torch.rand(50, 64, 64)
    >>> data = simulate_shading(x, truth)
    >>> cidre = CIDRE().fit(data)
    >>> flat = cidre.estimates["shading"].flat
    >>> corrected = cidre.apply(data)
    """

    tags = Tags(
        targets=("shading",),
        streaming="per-pixel-stats",
        modality=("fluorescence", "brightfield"),
        noise="none",
        many_frames=True,
        reference="doi:10.1038/nmeth.3323",
    )
    paper = dict(_PRESET)
    recommended = dict(_PRESET)

    def __init__(
        self,
        *,
        working_size: int | tuple[int, int] | None = None,
        quantiles: int | None = None,
        quantile_method: str | None = None,
        bins: int | None = None,
        lambda_r: float | None = None,
        lambda_z: float | None = None,
        lambda_b: float | None = None,
        cauchy_width: float | None = None,
        barrier_width: float | None = None,
        band: float | None = None,
        scales: Sequence[float] | None = None,
        zero_light_limits: tuple[float, float] | None = None,
        max_iterations: int | None = None,
        history_size: int | None = None,
        tol: float | None = None,
        init_method: str | None = None,
        normalize_fit: bool | None = None,
        faithful: bool = True,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
        layout: Layout | None = None,
    ) -> None:
        """Set the options; unset ones come from the preset.

        Parameters
        ----------
        working_size : int or (int, int) or None
            Grid of the fit: an int is the length of the long side
            (images that are smaller are not resampled), a pair the
            exact shape, None the image size. Default 128. Not specified
            in the paper.
        quantiles : int
            Number K of quantiles per location (all values when there are
            fewer images). Default 100. Not specified in the paper.
        quantile_method : {"auto", "exact", "histogram"}
            Exact order statistics of the resampled stack, held in
            memory; quantiles of per-pixel histograms (second pass over
            the data); or "auto": exact when the resampled stack fits the
            configured memory budget. Not specified in the paper.
        bins : int
            Histogram bins per pixel. Default 1024. Not specified in the
            paper.
        lambda_r : float, optional
            Weight of the smoothness energy E_r. Default: 9.5 for one
            image, decreasing linearly to 6 at 200 images, 6 beyond.
        lambda_z : float
            Weight of the zero-light energy E_z. Default 0.5.
        lambda_b : float
            Weight of the barrier energy E_b. Default 1. Not specified
            in the paper.
        cauchy_width : float
            Width w of the Cauchy function of E_f, as a fraction of the
            maximum intensity. Default 0.05. Not specified in the paper.
        barrier_width : float
            Width w_b of the barrier, as a fraction of the maximum
            intensity. Default 10⁻³. Not specified in the paper.
        band : float
            Half-width ω of the central band of Eq. 5. Default 0.25. Not
            specified in the paper.
        scales : sequence of float
            Scales T of E_r: standard deviations of the Gaussians, in
            pixels of a grid whose long side is 128 (they are rescaled
            with the working grid). Default (1, 2, 4, 8). Not specified
            in the paper.
        zero_light_limits : (float, float), optional
            Limits [α₁, α₂] of the zero-light point, in data units.
            Default: [0, min q(x)].
        max_iterations, history_size, tol : int, int, float
            L-BFGS (strong Wolfe line search): at most `max_iterations`
            iterations with `history_size` correction pairs; stops when
            the energy changes by less than `tol` · max(|E|, 1) in an
            iteration. Defaults 500, 20 and 10⁻⁹. Not specified in the
            paper.
        init_method : {"pivot", "least_squares"}
            Initial lines: slope 1 through the pivot (the medians of Q
            and q(x)), or the per-pixel least-squares fit. The
            zero-light point starts at the middle of [α₁, α₂]. Default
            "pivot". Not specified in the paper.
        normalize_fit : bool
            Divide E_f by the number of quantiles (extension). Default
            False.
        faithful : bool
            Selects the preset; both presets are the paper's method.
        dtype, device, layout : optional
            See [`Estimator`][optix.Estimator]. The layout must have two
            spatial axes; each group (e.g. channel) gets its own maps.
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        opt = self.options
        for name, allowed in (
            ("quantile_method", _QUANTILE_METHODS),
            ("init_method", _INITS),
        ):
            if opt[name] not in allowed:
                raise ValueError(
                    f"{name} must be one of {allowed}, got {opt[name]!r}"
                )
        if int(opt["quantiles"]) < 1:
            raise ValueError(f"quantiles must be >= 1: {opt['quantiles']}")
        if not 0 < float(opt["band"]) <= 0.5:
            raise ValueError(f"band must be in (0, 0.5]: {opt['band']}")
        for name in ("cauchy_width", "barrier_width"):
            if float(opt[name]) <= 0:
                raise ValueError(f"{name} must be > 0: {opt[name]}")
        if not opt["scales"] or min(float(t) for t in opt["scales"]) <= 0:
            raise ValueError(f"scales must be positive: {opt['scales']}")
        self.zero_light_: Tensor | None = None
        self.lambda_r_: float | None = None
        self.target_: Tensor | None = None

    # -- data reduction ----------------------------------------------------
    def _distributions(
        self, data: ArrayLike, size: tuple[int, int] | None
    ) -> tuple[dict, float, float, dict]:
        """Per-group quantiles `(K, h, w)` and the global extrema.

        Returns `({group: quantiles}, global min, global max,
        {group: MinMax})` in data units, at the working grid.
        """
        opt = self.options
        layout = self._layout_of(data)
        group_shape, sample_shape, spatial = layout.shape_of(data)
        n = math.prod(sample_shape)
        h, w = tuple(spatial) if size is None else size
        backend = self._backend(data)
        dtype, device = backend["dtype"], backend["device"]
        method = opt["quantile_method"]
        if method == "auto":
            ngroups = max(1, math.prod(group_shape))
            nbytes = n * h * w * dtype.itemsize * ngroups
            method = "exact" if nbytes <= get_config().memory else "histogram"
        exact = method == "exact"
        stack = None
        if exact:
            stack = torch.empty(
                *group_shape, *sample_shape, h, w, dtype=dtype, device=device
            )
        ndim = None if isinstance(data, Labeled) else 2
        extrema: dict[tuple[int, ...], MinMax] = {}
        for chunk in frame_batches(
            data, ndim=ndim, dtype=dtype, device=device, overhead=3.0
        ):
            small = resize(chunk.data, size)
            extrema.setdefault(chunk.group, MinMax(dim=0)).update(small)
            if exact:
                chunk.store(stack, small, into="samples")
        K = min(int(opt["quantiles"]), n)
        levels = torch.tensor([(j - 0.5) / K for j in range(1, K + 1)])
        # level u sits at the (1-based) rank n u + ½, between two order
        # statistics (exactly on rank j when K = n)
        pos = [n * (j - 0.5) / K + 0.5 for j in range(1, K + 1)]
        below = [min(max(math.floor(p), 1), n) for p in pos]
        above = [min(r + 1, n) for r in below]
        ranks = sorted(set(below) | set(above))
        where = {r: i for i, r in enumerate(ranks)}
        frac = torch.tensor(
            [p - r for p, r in zip(pos, below, strict=True)], dtype=dtype
        ).clamp(0, 1)[:, None, None]
        out: dict[tuple[int, ...], Tensor] = {}
        if exact:
            for group in extrema:
                frames = stack[group].reshape(n, h, w)
                stats = order_statistics(frames, ranks).reshape(-1, h, w)
                lo_ = stats[[where[r] for r in below]]
                hi_ = stats[[where[r] for r in above]]
                out[group] = lo_ + frac.to(lo_) * (hi_ - lo_)
        else:
            hists: dict[tuple[int, ...], Histogram] = {}
            for chunk in frame_batches(
                data, ndim=ndim, dtype=dtype, device=device, overhead=3.0
            ):
                mm = extrema[chunk.group]
                hist = hists.setdefault(
                    chunk.group,
                    Histogram(mm.min, mm.max, int(opt["bins"]), dim=0),
                )
                hist.update(resize(chunk.data, size))
            for group, hist in hists.items():
                out[group] = hist.quantile(levels).to(dtype)
        lo = min(float(m.min.min()) for m in extrema.values())
        hi = max(float(m.max.max()) for m in extrema.values())
        return out, lo, hi, extrema

    @staticmethod
    def _target(q: Tensor, band: float) -> Tensor:
        """Eqs. 5-6: mean of the q(x) whose mean is in the central band."""
        qbar = q.mean(0).reshape(-1)
        levels = torch.tensor([0.5 - band, 0.5 + band], dtype=qbar.dtype)
        lo, hi = torch.quantile(qbar, levels.to(qbar.device))
        mask = (qbar >= lo) & (qbar <= hi)
        return q.reshape(q.shape[0], -1)[:, mask].mean(-1)

    # -- optimisation ------------------------------------------------------
    def _solve(
        self, q: Tensor, alpha: tuple[float, float], lambda_r: float
    ) -> dict[str, object]:
        """Minimise Eq. 9 for scaled quantiles `q` (K, h, w), in float64."""
        opt = self.options
        q = q.to(torch.float64)
        Q = self._target(q, float(opt["band"]))
        Q_med = Q.median()
        q_med = q.median(0).values
        problem = _Problem(
            Qp=Q - Q_med, qp=q - q_med, Q_med=Q_med, q_med=q_med, alpha=alpha
        )
        h, w = q.shape[1:]
        if opt["init_method"] == "least_squares" and q.shape[0] > 1:
            Qc = problem.Qp - problem.Qp.mean()
            var = (Qc**2).sum()
            qm = problem.qp.mean(0)
            if var > 0:
                v0 = (Qc[:, None, None] * (problem.qp - qm)).sum(0) / var
            else:
                v0 = torch.ones_like(qm)
            b0 = qm - v0 * problem.Qp.mean()
        else:
            v0 = torch.ones(h, w, dtype=q.dtype, device=q.device)
            b0 = torch.zeros_like(v0)
        mid = 0.5 * (alpha[0] + alpha[1])
        # diagonal preconditioning: the optimiser sees x / scale, with
        # scales that equalise the curvatures of the data terms
        spread = float(problem.Qp.square().mean().sqrt())
        sv = 1 / spread if spread > 0 else 1.0
        sz = 1 / math.sqrt(h * w)
        u = (v0 / sv).requires_grad_(True)
        c = b0.clone().requires_grad_(True)
        zQ = torch.tensor(mid / sz, dtype=q.dtype, device=q.device)
        zq = zQ.clone()
        zQ.requires_grad_(True)
        zq.requires_grad_(True)
        params = [u, c, zQ, zq]
        kw = {
            "lambda_r": lambda_r,
            "lambda_z": float(opt["lambda_z"]),
            "lambda_b": float(opt["lambda_b"]),
            "cauchy_width": float(opt["cauchy_width"]),
            "barrier_width": float(opt["barrier_width"]),
            # scales are given for a long side of 128 pixels
            "scales": [float(t) * max(h, w) / 128 for t in opt["scales"]],
            "normalize_fit": bool(opt["normalize_fit"]),
        }

        def energy() -> tuple[Tensor, dict[str, Tensor]]:
            return cidre_energy(u * sv, c, zQ * sz, zq * sz, problem, **kw)

        lbfgs = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=1,
            max_eval=26,  # one closure call plus up to 25 line-search steps
            history_size=int(opt["history_size"]),
            tolerance_grad=0.0,
            tolerance_change=0.0,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            lbfgs.zero_grad()
            E, _ = energy()
            E.backward()
            return E

        trace: list[tuple[float, dict[str, float]]] = []

        def evaluate() -> float:
            with torch.no_grad():
                E, parts = energy()
            trace.append(
                (
                    float(E),
                    {
                        **{k: float(p) for k, p in parts.items()},
                        "Z_Q": float(zQ.detach()) * sz,
                        "Z_q": float(zq.detach()) * sz,
                    },
                )
            )
            return float(E)

        tol = float(opt["tol"])
        last = evaluate()
        converged = False
        with torch.enable_grad():
            for _ in range(int(opt["max_iterations"])):
                lbfgs.step(closure)
                current = evaluate()
                if abs(last - current) <= tol * max(abs(current), 1.0):
                    converged = True
                    break
                last = current
        with torch.no_grad():
            v = u.detach() * sv
            ZQ, Zq = zQ.detach() * sz, zq.detach() * sz
            z = c.detach() - v * Q_med + q_med + v * ZQ  # Eqs. 7, 19-20
        return {
            "v": v,
            "z": z,
            "Z": torch.stack([ZQ, Zq]),
            "Q": Q,
            "trace": trace,
            "converged": converged,
        }

    # -- fit / apply -------------------------------------------------------
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
                f"CIDRE needs two spatial axes, the layout has {len(spatial)}"
            )
        n = math.prod(sample_shape)
        if n < 1:
            raise ValueError("CIDRE needs at least one image per group")
        opt = self.options
        backend = self._backend(data)
        size = _working_shape(opt["working_size"], spatial)
        dists, _, hi, extrema = self._distributions(data, size)
        scale = hi if hi > 0 else 1.0
        lambda_r = opt["lambda_r"]
        lambda_r = lambda_r_schedule(n) if lambda_r is None else lambda_r
        self.lambda_r_ = float(lambda_r)
        flats, darks, zeros, targets, converged = [], [], [], [], []
        for group in itertools.product(*(range(g) for g in group_shape)):
            q = dists[group] / scale
            limits = opt["zero_light_limits"]
            if limits is None:
                alpha = (0.0, float(extrema[group].min.min()) / scale)
            else:
                alpha = (float(limits[0]) / scale, float(limits[1]) / scale)
            result = self._solve(q, alpha, float(lambda_r))
            for E, parts in result["trace"]:
                self.history.record(-E, **parts)
            converged.append(result["converged"])
            v = result["v"].to(**backend)
            z = (result["z"] * scale).to(**backend)
            flats.append(resize(v[None], tuple(spatial))[0])
            darks.append(resize(z[None], tuple(spatial))[0])
            zeros.append(result["Z"] * scale)
            targets.append(result["Q"] * scale)
            if verbose:
                print(
                    f"CIDRE group {group}: {len(result['trace']) - 1} "
                    f"iterations, energy {result['trace'][-1][0]:.6g}, "
                    f"converged={result['converged']}"
                )
            if callback is not None:
                callback(self)
        self.history.converged = all(converged)
        self.history.reason = (
            "energy change below tol"
            if all(converged)
            else f"max_iterations={opt['max_iterations']} reached"
        )
        self.zero_light_ = torch.stack(zeros).reshape(*group_shape, 2)
        self.target_ = torch.stack(targets).reshape(*group_shape, -1)
        self.estimates = {
            "shading": ShadingEstimate(
                flat=torch.stack(flats).reshape(*group_shape, *spatial),
                dark=torch.stack(darks).reshape(*group_shape, *spatial),
                meta={
                    "flat_norm": "none",
                    "model": "y = flat * x + dark",
                    "lambda_r": float(lambda_r),
                },
            )
        }

    def apply(
        self,
        data: ArrayLike,
        out: WritableArrayLike | None = None,
        *,
        mode: str = "dynamic_range",
    ) -> WritableArrayLike:
        """Correct images with the fitted v and z.

        Parameters
        ----------
        data : array-like or Labeled
            Images, in the layout of the fit (unless labelled otherwise).
        out : writable array-like, optional
            Output (see [`Estimator.apply`][optix.Estimator.apply]).
        mode : {"dynamic_range", "zero_light", "direct"}
            "dynamic_range" (dynamic range corrected, the default):
            I = (I_o − z) / v · v̄. "zero_light" (zero-light preserved):
            I = (I_o − z) / v · v̄ + z̄. "direct": Eq. 1,
            I = (I_o − z) / v. v̄ and z̄ are the means of the maps of the
            group.

        Returns
        -------
        writable array-like
            `out`.
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self._mode = mode
        try:
            return super().apply(data, out)
        finally:
            del self._mode

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if "shading" not in self.estimates:
            raise RuntimeError("CIDRE is not fitted: call fit first")
        mode = getattr(self, "_mode", "dynamic_range")
        est = self.estimates["shading"]
        layout = self._layout_of(data)
        group_shape, _, spatial = layout.shape_of(data)
        flat, dark = est.flat, est.dark
        if tuple(flat.shape) != tuple(group_shape) + tuple(spatial):
            raise ValueError(
                f"Data have maps of shape {group_shape + spatial}, the fit "
                f"has {tuple(flat.shape)}"
            )
        dims = (-2, -1)
        vbar = flat.mean(dims, keepdim=True)
        zbar = dark.mean(dims)
        if mode == "direct":
            gain = flat
        else:
            gain = flat / vbar

        def fn(chunk: object, batch: Tensor) -> Tensor:
            model = ShadingEstimate(
                flat=gain[chunk.group].to(batch),
                dark=dark[chunk.group].to(batch),
            )
            x = model.correct(batch)
            if mode == "zero_light":
                x = x + zbar[chunk.group].to(batch)
            return x

        return self._apply_frames(data, out, fn, overhead=3.0)

    # -- state -------------------------------------------------------------
    def _extra_state(self) -> dict[str, object]:
        return {
            "zero_light": self.zero_light_,
            "lambda_r": self.lambda_r_,
            "target": self.target_,
        }

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.zero_light_ = state.get("zero_light")
        self.lambda_r_ = state.get("lambda_r")
        self.target_ = state.get("target")
