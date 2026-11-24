"""Separate background and foreground flat fields (Kask et al., 2016)."""

import heapq
import itertools
import math
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .._estimator import Estimator, Tags
from ..estimates import ShadingEstimate
from ..io import ArrayLike, Labeled, WritableArrayLike, frame_batches

__all__ = ["KaskFlatField"]

# Author choices for steps the paper describes without numbers.
_CURVATURES = tuple(0.05 * i for i in range(13))  # 1 − c·ρ², c ∈ [0, 0.6]
_WARMUP = 10  # background images before the first provisional profile
_MODE_BINS = 128
_MODE_SAMPLES = 1 << 20
_MAX_ITER = 200
_TOL = 1e-9


# -- binning grid -------------------------------------------------------------
class _Grid:
    """Partition of an (H, W) image into (h, w) blocks and back.

    Row i belongs to block ⌊i·h/H⌋ (likewise for columns). Block values
    are brought back to pixels by separable linear interpolation between
    block centres (linear extrapolation beyond the outer centres).
    """

    def __init__(
        self, shape: tuple[int, int], bins: tuple[int, int], device: object
    ) -> None:
        (H, W), (h, w) = shape, bins
        if h > H or w > W:
            raise ValueError(f"bins={bins} exceed the image shape {shape}")
        self.shape, self.bins = (H, W), (h, w)
        rows = torch.arange(H, device=device) * h // H
        cols = torch.arange(W, device=device) * w // W
        self.index = (rows[:, None] * w + cols[None, :]).reshape(-1)
        f64 = torch.float64
        self.centres = []
        for n, k, which in ((H, h, rows), (W, w, cols)):
            pos = torch.arange(n, dtype=f64, device=device)
            s = torch.zeros(k, dtype=f64, device=device).index_add_(
                0, which, pos
            )
            c = torch.zeros(k, dtype=f64, device=device).index_add_(
                0, which, torch.ones_like(pos)
            )
            self.centres.append(s / c)

    def sum(self, x: Tensor) -> Tensor:
        """Per-block sums of `x` (H, W) → (h, w)."""
        h, w = self.bins
        out = torch.zeros(h * w, dtype=x.dtype, device=x.device)
        return out.index_add_(0, self.index, x.reshape(-1)).reshape(h, w)

    def expand(self, t: Tensor) -> Tensor:
        """Block value of every pixel (nearest block), (h, w) → (H, W)."""
        return t.reshape(-1)[self.index].reshape(self.shape)

    def upsample(self, t: Tensor) -> Tensor:
        """Linear interpolation of block values at pixels."""
        out = t
        for dim, (n, c) in enumerate(
            zip(self.shape, self.centres, strict=True)
        ):
            pos = torch.arange(n, dtype=c.dtype, device=c.device)
            i = (torch.searchsorted(c, pos) - 1).clamp(0, c.numel() - 2)
            frac = ((pos - c[i]) / (c[i + 1] - c[i])).to(out.dtype)
            lo = out.index_select(dim, i)
            hi = out.index_select(dim, i + 1)
            shape = [1, 1]
            shape[dim] = n
            frac = frac.reshape(shape)
            out = lo + frac * (hi - lo)
        return out

    def coords(self, full: bool) -> tuple[Tensor, Tensor]:
        """Coordinates scaled to [-1, 1] (pixels or block centres)."""
        out = []
        for n, c in zip(self.shape, self.centres, strict=True):
            p = torch.arange(n, dtype=c.dtype, device=c.device) if full else c
            out.append((2 * p + 1) / n - 1)
        return out[0][:, None], out[1][None, :]


# -- profile models (Eq. 5 and the fourth-order polynomial) ------------------
def _second_differences(h: int, w: int, device: object) -> Tensor:
    """Stacked operators (Lxx; Lyy) of interior second differences."""
    f64 = torch.float64

    def d2(n: int) -> Tensor:
        d = torch.zeros(n - 2, n, dtype=f64, device=device)
        i = torch.arange(n - 2, device=device)
        d[i, i], d[i, i + 1], d[i, i + 2] = 1.0, -2.0, 1.0
        return d

    eye_h = torch.eye(h, dtype=f64, device=device)
    eye_w = torch.eye(w, dtype=f64, device=device)
    return torch.cat([torch.kron(eye_h, d2(w)), torch.kron(d2(h), eye_w)])


def _monomials(u: Tensor, v: Tensor, degree: int = 4) -> list[Tensor]:
    return [
        (u**i * v**j).expand(u.shape[0], v.shape[1])
        for i in range(degree + 1)
        for j in range(degree + 1 - i)
    ]


class _Model:
    """Smooth fit of a weighted (h, w) image: Eq. 5 or a polynomial."""

    def __init__(self, grid: _Grid, kind: str, strength: float) -> None:
        self.grid, self.kind, self.strength = grid, kind, float(strength)
        h, w = grid.bins
        dev = grid.centres[0].device
        if kind == "regularization":
            L = _second_differences(h, w, dev)
            self.penalty = L.T @ L
        else:
            u, v = grid.coords(full=False)
            self.basis = torch.stack(
                [m.reshape(-1) for m in _monomials(u, v)], 1
            )

    def smooth(self, data: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
        """Minimiser T of ∑ w (D − T)² + R ∑ (Lxx² + Lyy²), or the
        weighted least-squares polynomial. Returns (T, parameters)."""
        e = torch.nan_to_num(data.reshape(-1))
        wt = weight.reshape(-1)
        if self.kind == "regularization":
            n = wt.numel()
            M = torch.diag(wt) + self.strength * self.penalty
            M = M + 1e-12 * M.diagonal().mean() * torch.eye(
                n, dtype=M.dtype, device=M.device
            )
            t = torch.linalg.solve(M, wt * e)
            return t.reshape(data.shape), t
        sw = wt.sqrt()[:, None]
        coef = torch.linalg.lstsq(self.basis * sw, sw[:, 0:1] * e[:, None])
        coef = coef.solution[:, 0]
        return (self.basis @ coef).reshape(data.shape), coef

    def scaled(self, factor: float) -> "_Model":
        """The same model with R multiplied by `factor`."""
        out = object.__new__(_Model)
        out.__dict__.update(self.__dict__)
        out.strength = self.strength * factor
        return out

    def full(self, t: Tensor, params: Tensor, raw: bool = False) -> Tensor:
        """Profile at full resolution, normalised to mean 1 (or not)."""
        if self.kind == "regularization":
            out = self.grid.upsample(t)
        else:
            u, v = self.grid.coords(full=True)
            out = sum(
                c * m for c, m in zip(params, _monomials(u, v), strict=True)
            )
        return out if raw else out / out.mean()


def _global_fit(E: Tensor, model: _Model) -> tuple[Tensor, Tensor, Tensor]:
    """Fit E_k ≈ a_k · T over images k (NaN = undefined), mean(T) = 1.

    Alternates two steps until T is stable: the levels a_k by least
    squares, a_k = argmin ∑_ij (E_kij − a_k T_ij)²; then the profile, the
    smooth fit (Eq. 5) of the mean normalised image ē = mean_k(E_k / a_k),
    weighted by the number W_ij of images defining each pixel,
    T = argmin ∑_ij W_ij (ē_ij − T_ij)² + R ∑_ij (Lxx² + Lyy²),
    renormalised to mean 1. This is not coordinate descent on a single
    objective; its fixed point is the profile of the set. Returns
    (T, a, parameters of T).
    """
    valid = torch.isfinite(E)
    E0 = torch.where(valid, E, 0.0)
    V = valid.to(E.dtype)
    T = torch.ones(E.shape[1:], dtype=E.dtype, device=E.device)
    params = T.reshape(-1)
    for _ in range(_MAX_ITER):
        a = (E0 * T).sum((1, 2)) / (V * T.square()).sum((1, 2))
        W = V.sum(0)
        D = (E0 / a[:, None, None]).sum(0) / W.clamp_min(1)
        new, params = model.smooth(D, W)
        mean = new.mean()
        new, params = new / mean, params / mean
        delta = (new - T).abs().max()
        T = new
        if delta < _TOL:
            break
    a = (E0 * T).sum((1, 2)) / (V * T.square()).sum((1, 2))
    return T, a, params


def _deviation(E: Tensor, T: Tensor, a: Tensor) -> Tensor:
    """Mean square deviation of each image from the fit, relative units."""
    valid = torch.isfinite(E)
    r = torch.where(valid, E / a[:, None, None] - T, 0.0)
    return r.square().sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)


def _outliers(E: Tensor, model: _Model, factor: float) -> Tensor:
    """Images kept by the iterative outlier filter (bool, (n,))."""
    keep = torch.ones(E.shape[0], dtype=torch.bool, device=E.device)
    while int(keep.sum()) >= 3:
        idx = keep.nonzero()[:, 0]
        T, a, _ = _global_fit(E[idx], model)
        msd = _deviation(E[idx], T, a)
        bad = msd > factor * msd.median()
        if not bool(bad.any()):
            break
        keep[idx[bad]] = False
    return keep


def _random_error(E: Tensor, model: _Model, subsets: int) -> float:
    """Expected range (max − min) of the random error of the profile.

    The images are split into S contiguous subsets fitted separately; with
    T̄ the mean of their profiles, each deviation T_s − T̄ has (S − 1)
    times the variance of the error of the full estimate, so the error
    range is mean_s range(T_s − T̄) / √(S − 1). The deviation term of a
    subset fit sums over n_s ≈ n/S images, so its R is scaled by n_s/n
    to smooth as much as the full fit.
    """
    n = E.shape[0]
    if subsets < 2 or n < subsets:
        return math.inf
    parts = torch.arange(n, device=E.device).tensor_split(subsets)
    profiles = torch.stack(
        [_global_fit(E[p], model.scaled(len(p) / n))[0] for p in parts]
    )
    dev = profiles - profiles.mean(0)
    span = dev.flatten(1).amax(1) - dev.flatten(1).amin(1)
    return float(span.mean() / math.sqrt(subsets - 1))


# -- segmentation -------------------------------------------------------------
def _box_mean(x: Tensor, size: int) -> Tensor:
    return F.avg_pool2d(
        x[None, None], size, 1, size // 2, count_include_pad=False
    )[0, 0]


def _noise_image(x: Tensor, profile: Tensor, size: int) -> Tensor:
    """Local variance over local mean of x/P, times P (Fig. 4).

    Dividing by the background profile P removes the contribution of its
    slope to the local variance; multiplying back by P gives, for Poisson
    data, var(x)/⟨x⟩ = κ, independent of the position.
    """
    z = x / profile
    shift = z.median()
    zc = z - shift
    m = _box_mean(zc, size)
    var = (_box_mean(zc.square(), size) - m.square()).clamp_min(0)
    mean = m + shift
    out = profile * var / mean
    return torch.where(mean > 0, out, torch.inf)


def _mode(x: Tensor) -> Tensor:
    """Position of the highest peak of the histogram of log(x)."""
    v = x[torch.isfinite(x) & (x > 0)]
    if v.numel() == 0:
        return torch.tensor(torch.inf, dtype=x.dtype, device=x.device)
    step = max(1, v.numel() // _MODE_SAMPLES)
    v = v[::step].log()
    lo, hi = torch.quantile(v, torch.tensor([0.01, 0.99]).to(v))
    if not bool(hi > lo):
        return v[0].exp()
    hist = torch.histc(v, _MODE_BINS, float(lo), float(hi))
    hist = F.avg_pool1d(hist[None, None], 3, 1, 1, count_include_pad=False)
    k = int(hist.reshape(-1).argmax())
    return (lo + (k + 0.5) * (hi - lo) / _MODE_BINS).exp()


def _radial(shape: tuple[int, int], like: Tensor) -> Tensor:
    """ρ² from the image centre, 1 at the corners."""
    u, v = (
        torch.linspace(-1, 1, n, dtype=like.dtype, device=like.device)
        for n in shape
    )
    return 0.5 * (u[:, None].square() + v[None, :].square())


def _initial_profile(x: Tensor, size: int) -> Tensor:
    """Centred profile 1 − c·ρ² whose curvature minimises the noise level."""
    rho2 = _radial(tuple(x.shape), x)
    best, best_level = None, math.inf
    for c in _CURVATURES:
        p = 1 - c * rho2
        level = float(_mode(_noise_image(x, p, size)))
        if level < best_level:
            best, best_level = p, level
    return best


def _background_mask(
    x: Tensor,
    profile: Tensor,
    opts: dict,
    kappa: Callable[[Tensor], Tensor] | Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Noise-based background segmentation. Returns (mask, κ).

    κ is the mode of the frame's noise image when `kappa` is None, a
    function of that mode when callable (e.g. a running estimate over
    the series), or the given constant.
    """
    noise = _noise_image(x, profile, opts["window"])
    if kappa is None or callable(kappa):
        mode = _mode(noise)
        kappa = mode if kappa is None else kappa(mode)
    kappa = torch.as_tensor(kappa, dtype=x.dtype, device=x.device)
    mask = noise < opts["noise_threshold"] * kappa
    level = _box_mean(x / profile, opts["window"])
    if bool(mask.any()):
        ref = level[mask].median()
        mask &= level < opts["intensity_threshold"] * ref
    return mask, kappa


def _foreground_mask(
    x: Tensor, kappa: Tensor, grid: _Grid, k: float
) -> Tensor:
    """Per-block threshold: block mean + k · √(κ · block mean)."""
    counts = grid.sum(torch.ones_like(x))
    mean = grid.sum(x) / counts
    thr = mean + k * (kappa * mean.clamp_min(0)).sqrt()
    return x > grid.expand(thr)


def _level(x: Tensor, profile: Tensor, mask: Tensor) -> Tensor:
    """Background level B̂ = ∑ x / ∑ P over the background pixels."""
    den = profile[mask].sum()
    return x[mask].sum() / den if bool(den > 0) else x.new_tensor(math.nan)


# -- estimator ----------------------------------------------------------------
class _RunningMedian:
    """Median of a stream of scalars (two heaps)."""

    def __init__(self) -> None:
        self.low: list[float] = []  # max-heap (negated)
        self.high: list[float] = []

    def __len__(self) -> int:
        return len(self.low) + len(self.high)

    def add(self, value: float) -> None:
        heapq.heappush(self.low, -value)
        heapq.heappush(self.high, -heapq.heappop(self.low))
        if len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def median(self) -> float:
        if not self:
            return math.nan
        if len(self.low) > len(self.high):
            return -self.low[0]
        return 0.5 * (self.high[0] - self.low[0])


class _GroupState:
    """Binned images of one group, accumulated frame by frame."""

    def __init__(self) -> None:
        self.bg_sum, self.bg_count, self.accepted = [], [], []
        self.fg_sum, self.fg_count = [], []
        self.profile: Tensor | None = None
        self.kappa = _RunningMedian()
        self.n_bg = 0
        self.next_refit = _WARMUP


class KaskFlatField(Estimator):
    """Separate background and foreground flat fields (Kask et al., 2016).

    Fluorescence images are modelled with an additive background shaded by
    a background profile S_b and a signal shaded by a different
    foreground profile S_f, both normalised to mean 1 (Eqs. 1–3):

        N = S_f · U + B · S_b,
        Û = (N − B̂ · S_b) / S_f + B̂,

    with B̂ the mean background intensity of the image being corrected.
    Both profiles are estimated retrospectively from many sample images
    (Fig. 1), in one pass over the data:

    1. **Background segmentation.** The noise image is the local variance
       over the local mean in a `window`² window, after removing the slope
       of the background profile (division by the current profile
       estimate; before one exists, by a centred profile 1 − c·ρ² whose
       curvature c minimises the noise level). It is ≈ κ, a constant of
       the detection channel estimated from the series (`kappa`), in the
       background (Eq. 4) and much higher on objects. Background pixels
       lie below an upper threshold on the noise image; an intensity
       threshold then removes flat bright regions inside objects.
    2. **Foreground segmentation.** The image is cut into blocks; in each,
       pixels above the block mean plus `foreground_sigma` expected
       standard deviations √(κ · mean) are foreground.
    3. **Binning.** Both segmented images are averaged in blocks into
       `bins` (17 × 16 pixels in the paper, from its blocks of 80 × 64
       pixels); blocks without a segmented pixel are undefined.
    4. **Background profile.** Images with less than
       `min_background_fraction` background are discarded. The others are
       fitted globally (E_k ≈ a_k · S_b, step 6), images with an
       abnormally high mean square deviation from the fit are discarded,
       and the fit and the filter are repeated until no outlier remains.
    5. **Foreground profile.** Consecutive sets of `group_size` binned
       foreground images are outlier-filtered and averaged; the background
       B̄ · S_b is subtracted from each average (B̄ the mean background
       level of its images), and the averages are fitted globally.
    6. **Global fit and smoothing (Eq. 5).** A global fit alternates the
       levels a_k, by least squares (min ∑_ij (E_kij − a_k T_ij)²), and
       the profile T, the smooth fit of ē = mean_k(E_k / a_k) weighted by
       the number W_ij of images defining each bin,
       min ∑_ij W_ij (ē_ij − T_ij)² + R ∑_ij (Lxx_ij² + Lyy_ij²),
       renormalised to mean 1, until T is stable. R is given by
       `regularization` (0.2 for the background and 20 for the
       foreground in the paper); `model="polynomial"` fits a
       fourth-order polynomial instead. Lxx and Lyy are second
       differences between neighbouring bins, at interior bins only.
       Not specified in the paper: Eq. 5 is written for a single image,
       and the paper does not give the objective of the global fit.
       With W_ij = K, the deviation term is ∑_k ‖E_k / a_k − T‖² up to
       a constant, so R does not depend on the number of images K.
    7. **Quality control.** The data are split into `subsets` fitted
       separately (with R scaled by their share of the images, so that
       they smooth as much as the full fit); the spread of their
       profiles gives the range of the random error of the profile. A
       profile whose error exceeds `max_error` is not applied (replaced
       by 1, with a warning); the foreground profile is only estimated
       when the background profile passes.

    !!! note "Estimates"
        `"shading"`: `flat` is the foreground profile S_f (mean 1,
        `meta["flat_norm"] == "mean"`), and `dark` the background
        profile scaled to the mean background level of the images used
        in the background fit, B̄ · S_b, in ADU (it includes the camera
        offset); the levels of discarded images are not averaged. The
        background of each image is rescaled: `apply` estimates B̂ per
        image (`meta["dark_scaling"] == "per-frame"`), whereas
        [`ShadingEstimate.correct`][optix.estimates.ShadingEstimate.correct]
        subtracts `dark` as is. `meta` also holds the random errors,
        the numbers of images used, B̄ and κ (`noise_constant`). The
        background level of every image is in `background_level_`
        (`(*group, n)`).

    !!! note "Streaming"
        Frames are segmented one at a time, in storage order, and only
        their binned images are kept (4 · h · w values per frame: the
        sums and counts of the background and foreground pixels of each
        bin). κ is a running median of the frames' modes. The
        provisional background profile used to segment later frames is
        refitted when the number of background images reaches 10, 20,
        40, … (not specified in the paper), so the result depends on
        the frame order but not on the batch size.

    !!! note "Settings not specified in the paper"
        κ is the median over the frames of the mode of each frame's noise
        image (the position of the highest peak of the histogram of its
        logarithm); while streaming, each frame uses the median of the
        modes of the frames seen so far. The paper fixes the block size
        (80 × 64 or 64 × 64 pixels), the estimator the number of bins:
        `bins=(16, 17)` gives the paper's blocks on 1360 × 1024 frames
        only; on other sizes the blocks, and so the pixels behind each
        bin and the foreground threshold, differ from the paper, while R
        keeps its per-bin meaning. The initial curvatures are
        c ∈ {0, 0.05, …, 0.6}, with ρ² = 1 at the corners. Profiles are
        brought from bins to pixels by linear interpolation between bin
        centres (the polynomial is evaluated at the pixels). The random
        error uses contiguous subsets. B̂ is ∑ N / ∑ S_b over the
        background pixels of the frame, and the fit's mean level when a
        frame has no background.

    !!! references "References"
        1. Kask P, Palo K, Hinnah C, Pommerencke T. *Flat field correction
           for high-throughput imaging of fluorescent samples.* J Microsc
           **263**(3):328–340, 2016.
           [doi:10.1111/jmi.12404](https://doi.org/10.1111/jmi.12404)

        ??? quote "BibTeX"
            ```bibtex
            @article{kask2016flat,
              author  = {Kask, P. and Palo, K. and Hinnah, C. and
                         Pommerencke, T.},
              title   = {Flat field correction for high-throughput imaging
                         of fluorescent samples},
              journal = {Journal of Microscopy},
              volume  = {263}, number = {3}, pages = {328--340},
              year    = {2016}, doi = {10.1111/jmi.12404}
            }
            ```
    """

    tags = Tags(
        targets=("shading",),
        streaming="per-frame",
        modality=("widefield", "confocal", "high-content"),
        noise="poisson",
        many_frames=True,
        reference="doi:10.1111/jmi.12404",
    )
    paper = {
        "model": "regularization",
        "regularization": (0.2, 20.0),
        "bins": (16, 17),
        "window": 9,
        "noise_threshold": 1.5,
        "intensity_threshold": 1.25,
        "min_background_fraction": 0.15,
        "foreground_sigma": 5.0,
        "group_size": 16,
        "outlier_factor": 3.0,
        "subsets": 4,
        "max_error": (0.02, 0.04),
        "kappa": "series",
    }
    recommended = dict(paper)

    def __init__(
        self,
        *,
        model: str | None = None,
        regularization: tuple[float, float] | None = None,
        bins: tuple[int, int] | None = None,
        window: int | None = None,
        noise_threshold: float | None = None,
        intensity_threshold: float | None = None,
        min_background_fraction: float | None = None,
        foreground_sigma: float | None = None,
        group_size: int | None = None,
        outlier_factor: float | None = None,
        subsets: int | None = None,
        max_error: tuple[float | None, float | None] | None = None,
        kappa: str | float | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: object = None,
    ) -> None:
        """Configure the estimator.

        Parameters
        ----------
        model : {"regularization", "polynomial"}
            Profile model: image regularisation (Eq. 5, the paper's
            default) or a fourth-order polynomial of the coordinates,
            which the paper also uses.
        regularization : (float, float)
            Regularisation coefficients R of Eq. 5 for the background and
            the foreground profiles, on the binned grid. Paper: (0.2, 20).
        bins : (int, int)
            Shape (rows, columns) of the binned images, which is also the
            grid of foreground-segmentation blocks. Paper: 17 × 16 pixels
            (x × y), i.e. (16, 17), from blocks of 80 × 64 pixels of
            1360 × 1024 frames. The paper fixes the block size; the
            number of bins is fixed here, whatever the image size.
        window : int
            Odd side of the window of the local mean and variance of the
            noise image. Paper: 9.
        noise_threshold : float
            Upper threshold on the noise image, in units of κ (see
            `kappa`). Not specified in the paper.
        intensity_threshold : float
            Pixels whose local mean (after division by the profile)
            exceeds this factor times the median over the noise-selected
            pixels are not background. Not specified in the paper.
        min_background_fraction : float
            Images with a smaller background fraction are not used for the
            background profile. Paper: 0.15.
        foreground_sigma : float
            Foreground threshold, in expected standard deviations above
            the block mean. Paper: 5.
        group_size : int
            Number of foreground images averaged together. Paper: 16. A
            trailing incomplete set is not used.
        outlier_factor : float
            An image is an outlier when its mean square deviation from
            the global fit exceeds this factor times the median over the
            images. Not specified in the paper.
        subsets : int
            Number of subsets fitted separately to estimate the random
            error. Not specified in the paper.
        max_error : (float or None, float or None)
            Quality-control limits on the range of the random error of the
            background and foreground profiles; None disables the check.
            The paper gives 2 % as an example for the background; the
            foreground limit is not specified in the paper.
        kappa : {"series", "per-frame"} or float
            Noise constant κ (var/mean of the background), a constant of
            the detection channel in the paper. `"series"` (paper)
            estimates one κ per group from the series: the median of the
            modes of the frames' noise images, a running median while
            streaming (each frame is segmented with the median of the
            modes so far) and the final median in `apply`. A float
            gives κ directly. `"per-frame"` (extension) uses the mode of
            each frame's own noise image, which follows frame-to-frame
            changes of var/mean, e.g. from a camera offset.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        o = self.options
        if o["model"] not in ("regularization", "polynomial"):
            raise ValueError(f"Unknown model {o['model']!r}")
        if int(o["window"]) < 3 or int(o["window"]) % 2 == 0:
            raise ValueError("window must be odd and at least 3")
        if min(o["bins"]) < 3:
            raise ValueError("bins must be at least 3 along each axis")
        if int(o["group_size"]) < 1:
            raise ValueError("group_size must be positive")
        if isinstance(o["kappa"], str):
            if o["kappa"] not in ("series", "per-frame"):
                raise ValueError(f"Unknown kappa {o['kappa']!r}")
        elif not float(o["kappa"]) > 0:
            raise ValueError("kappa must be positive")

    # -- per-frame accumulation -------------------------------------------
    def _kappa(self, state: _GroupState) -> object:
        """κ rule for `_background_mask` while fitting (see `kappa`)."""
        k = self.options["kappa"]
        if k == "per-frame":
            return None
        if k != "series":
            return float(k)

        def running(mode: Tensor) -> float:
            if bool(torch.isfinite(mode)):
                state.kappa.add(float(mode))
            value = state.kappa.median()
            return float(mode) if math.isnan(value) else value

        return running

    def _accumulate(self, x: Tensor, state: _GroupState, grid: _Grid) -> None:
        o = self.options
        if state.profile is None:
            profile = _initial_profile(x, o["window"])
        else:
            profile = state.profile
        bg, kappa = _background_mask(x, profile, o, self._kappa(state))
        fg = _foreground_mask(x, kappa, grid, o["foreground_sigma"])
        f64 = torch.float64
        for mask, s, c in (
            (bg, state.bg_sum, state.bg_count),
            (fg, state.fg_sum, state.fg_count),
        ):
            s.append(grid.sum(torch.where(mask, x, 0)).to(f64))
            c.append(grid.sum(mask.to(f64)))
        ok = float(bg.to(f64).mean()) >= o["min_background_fraction"]
        state.accepted.append(ok)
        state.n_bg += ok
        if ok and state.n_bg >= state.next_refit:
            state.next_refit *= 2
            E = self._binned(state.bg_sum, state.bg_count, state.accepted)
            model = self._model(grid, 0)
            keep = _outliers(E, model, o["outlier_factor"])
            T, _, params = _global_fit(E[keep], model)
            state.profile = model.full(T, params).to(x.dtype)

    @staticmethod
    def _binned(
        sums: list, counts: list, select: list | None = None
    ) -> Tensor:
        s, c = torch.stack(sums), torch.stack(counts)
        if select is not None:
            idx = torch.tensor(select, dtype=torch.bool, device=s.device)
            s, c = s[idx], c[idx]
        return torch.where(c > 0, s / c.clamp_min(1), math.nan)

    def _model(self, grid: _Grid, which: int) -> _Model:
        return _Model(
            grid, self.options["model"], self.options["regularization"][which]
        )

    def _check(
        self, error: float, which: int, name: str, group: tuple
    ) -> bool:
        limit = self.options["max_error"][which]
        if limit is None or error <= limit:
            return True
        warnings.warn(
            f"KaskFlatField group {group}: random error of the {name} "
            f"profile {error:.3g} exceeds {limit:g}; the {name} profile is "
            "not applied",
            RuntimeWarning,
            stacklevel=4,
        )
        return False

    # -- per-group profiles -------------------------------------------------
    def _profiles(self, state: _GroupState, grid: _Grid, group: tuple) -> dict:
        o = self.options
        H, W = grid.shape
        dev = grid.centres[0].device
        ones = torch.ones(H, W, dtype=torch.float64, device=dev)
        out = {
            "background": ones,
            "foreground": ones,
            "background_error": math.inf,
            "foreground_error": math.nan,
            "background_images": 0,
            "foreground_groups": 0,
        }
        n = len(state.accepted)
        bg_model, fg_model = self._model(grid, 0), self._model(grid, 1)
        # background profile
        Tb = torch.ones(grid.bins, dtype=torch.float64, device=dev)
        accepted = torch.tensor(state.accepted, device=dev)
        fitted = accepted.clone()
        if state.n_bg > 0:
            E = self._binned(state.bg_sum, state.bg_count, state.accepted)
            keep = _outliers(E, bg_model, o["outlier_factor"])
            fitted[accepted.nonzero()[:, 0][~keep]] = False
            Tb, _, params = _global_fit(E[keep], bg_model)
            raw = bg_model.full(Tb, params, raw=True)
            out["background"] = raw / raw.mean()
            # bins in the units of the pixel profile (mean 1 over pixels)
            Tb = Tb / raw.mean()
            out["background_error"] = _random_error(
                E[keep], bg_model, o["subsets"]
            )
            out["background_images"] = int(keep.sum())
        bg_ok = self._check(out["background_error"], 0, "background", group)
        if not bg_ok:
            out["background"] = ones
            Tb = torch.ones_like(Tb)
        # background level of every image, from its binned background
        s, c = torch.stack(state.bg_sum), torch.stack(state.bg_count)
        den = (c * Tb).sum((1, 2))
        levels = torch.where(den > 0, s.sum((1, 2)) / den, math.nan)
        out["levels"] = levels
        out["fitted"] = (
            fitted if bool(fitted.any()) else torch.ones_like(fitted)
        )
        if not bg_ok:
            return out
        # foreground profile
        Ef = self._binned(state.fg_sum, state.fg_count)
        size = int(o["group_size"])
        averages = []
        for start in range(0, n - size + 1, size):
            sl = slice(start, start + size)
            Eg, Bg = Ef[sl], levels[sl]
            ok = torch.isfinite(Bg) & torch.isfinite(Eg).flatten(1).any(1)
            if not bool(ok.any()):
                continue
            Eg, Bg = Eg[ok], Bg[ok]
            keep = _outliers(Eg, fg_model, o["outlier_factor"])
            avg = Eg[keep].nanmean(0) - Bg[keep].mean() * Tb
            averages.append(avg)
        out["foreground_groups"] = len(averages)
        if averages:
            A = torch.stack(averages)
            Tf, _, params = _global_fit(A, fg_model)
            out["foreground"] = fg_model.full(Tf, params)
            out["foreground_error"] = _random_error(A, fg_model, o["subsets"])
        else:
            out["foreground_error"] = math.inf
        if not self._check(out["foreground_error"], 1, "foreground", group):
            out["foreground"] = ones
        return out

    # -- fit / apply ------------------------------------------------------
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
        group_shape, _, spatial = layout.shape_of(raw)
        if len(spatial) != 2:
            raise ValueError(
                "KaskFlatField needs two spatial axes, the layout has "
                f"{len(spatial)}"
            )
        backend = self._backend(data)
        grid = _Grid(
            tuple(spatial), tuple(self.options["bins"]), backend["device"]
        )
        states: dict[tuple, _GroupState] = {}
        for chunk in frame_batches(
            data,
            ndim=None if isinstance(data, Labeled) else 2,
            overhead=8.0,
            **backend,
        ):
            state = states.setdefault(chunk.group, _GroupState())
            for x in chunk.data.reshape(-1, *spatial):
                self._accumulate(x, state, grid)
        flats, darks, levels, meta = [], [], [], {}
        for group in itertools.product(*(range(g) for g in group_shape)):
            state = states.get(group)
            if state is None or not state.accepted:
                raise ValueError("KaskFlatField needs at least one image")
            p = self._profiles(state, grid, group)
            level = p["levels"][p["fitted"]].nanmean()
            level = torch.nan_to_num(level, nan=0.0)
            flats.append(p["foreground"])
            darks.append(level * p["background"])
            levels.append(p["levels"])
            for key in (
                "background_error",
                "foreground_error",
                "background_images",
                "foreground_groups",
            ):
                meta.setdefault(key, []).append(p[key])
            meta.setdefault("background_level", []).append(float(level))
            meta.setdefault("noise_constant", []).append(
                self._final_kappa(state)
            )
            if verbose:
                print(
                    f"KaskFlatField group {group}: background error "
                    f"{p['background_error']:.3g} "
                    f"({p['background_images']} images), foreground error "
                    f"{p['foreground_error']:.3g} "
                    f"({p['foreground_groups']} averages)"
                )
            if callback is not None:
                callback(self)
        shape = (*group_shape, *spatial)
        self.background_level_ = (
            torch.stack(levels).reshape(*group_shape, -1).to(**backend)
        )
        self.estimates = {
            "shading": ShadingEstimate(
                flat=torch.stack(flats).reshape(shape).to(**backend),
                dark=torch.stack(darks).reshape(shape).to(**backend),
                meta={
                    "flat_norm": "mean",
                    "model": "y = flat * (x - B) + B * dark / mean(dark)",
                    "dark_scaling": "per-frame",
                    **{k: _json_list(v, group_shape) for k, v in meta.items()},
                },
            )
        }

    def _final_kappa(self, state: _GroupState) -> float:
        k = self.options["kappa"]
        if k == "per-frame":
            return math.nan
        if k == "series":
            return state.kappa.median()
        return float(k)

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        if "shading" not in self.estimates:
            raise RuntimeError("KaskFlatField is not fitted: call fit first")
        est = self.estimates["shading"]
        layout = self._layout_of(data)
        group_shape, _, spatial = layout.shape_of(data)
        if tuple(est.flat.shape) != tuple(group_shape) + tuple(spatial):
            raise ValueError(
                f"Data have maps of shape {group_shape + spatial}, the fit "
                f"has {tuple(est.flat.shape)}"
            )
        dark = est.dark
        mean = dark.mean((-2, -1), keepdim=True)
        profile = torch.where(mean > 0, dark / mean, torch.ones_like(dark))
        fallback = mean[..., 0, 0]
        opts = self.options
        kappas = est.meta.get("noise_constant")
        if opts["kappa"] == "per-frame" or kappas is None:
            kappas = torch.full(group_shape, math.nan, dtype=torch.float64)
        else:
            kappas = torch.tensor(kappas, dtype=torch.float64).reshape(
                group_shape
            )

        def fn(chunk: object, batch: Tensor) -> Tensor:
            p = profile[chunk.group].to(batch)
            f = est.flat[chunk.group].to(batch)
            frames = batch.reshape(-1, *spatial)
            res = torch.empty_like(frames)
            for i, x in enumerate(frames):
                k = float(kappas[chunk.group])
                k = None if math.isnan(k) else k
                mask, _ = _background_mask(x, p, opts, k)
                b = _level(x, p, mask)
                if not bool(torch.isfinite(b)):
                    b = fallback[chunk.group].to(batch)
                res[i] = (x - b * p) / f + b
            return res.reshape(batch.shape)

        return self._apply_frames(data, out, fn, overhead=8.0)

    def _extra_state(self) -> dict[str, object]:
        return {"background_level": getattr(self, "background_level_", None)}

    def _load_extra_state(self, state: dict[str, object]) -> None:
        self.background_level_ = state.get("background_level")


def _json_list(values: list, group_shape: tuple[int, ...]) -> object:
    """Per-group values as a nested list of shape `group_shape` (JSON)."""
    clean = [
        v if not isinstance(v, float) or math.isfinite(v) else None
        for v in values
    ]
    if not group_shape:
        return clean[0]
    out: object = clean
    for n in reversed(group_shape[1:]):
        out = [out[i : i + n] for i in range(0, len(out), n)]
    return out
