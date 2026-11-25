"""Non-local means (Buades, Coll & Morel 2005).

Written from the CVPR 2005 paper.
"""

import itertools
import math
from collections.abc import Sequence
from typing import Literal, Self

import torch
import torch.nn.functional as F
from torch import Tensor

from .._config import get_config, resolve_backend
from .._estimator import Tags
from ..io import ArrayLike, Layout
from ._bm3d import _BlockMatching, _check_int, _check_positive

__all__ = ["NLMeans", "nl_means", "nl_means_halo"]

KernelNorm = Literal["mean", "sum"]


# Share of the memory budget held by a batch of offsets; the streamed
# block (padded data, numerator, denominator, temporaries) takes the rest.
OFFSET_SHARE = 0.5
# Image-sized buffers per offset of a batch: shifted data, squared
# differences, the two passes of the separable sum, weights.
OFFSET_COPIES = 5
# Image-sized buffers of a streamed block, outside the offset batch.
BLOCK_COPIES = 6


# -- building blocks ---------------------------------------------------------
def patch_kernel(
    patch: int,
    a: float | None = None,
    norm: KernelNorm = "mean",
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """1-D factor of the Gaussian kernel of the patch distance.

    g(u) ∝ exp(−u²/(2a²)) for u = −r … r (r = ⌊patch/2⌋). The
    d-dimensional kernel G_a is the outer product of d copies. With
    `norm="mean"` each factor sums to `patch`, so G_a has mean 1 over the
    patch (it sums to patchᵈ) and the distance is a weighted *sum* of
    squared differences; with `norm="sum"` each factor sums to 1, so G_a
    sums to 1 and the distance is a weighted *mean*.

    !!! note "Choices not specified in the paper"
        * The paper gives neither a nor the normalisation of the
          discrete kernel. The default a = (patch − 1)/4 is 1.5 for 7 × 7
          patches (the patch spans ±2a). `a = inf` gives uniform weights.
        * The two normalisations differ by the factor patchᵈ (49 for
          7 × 7 patches), that is a factor 7 on h. The paper's identity
          E‖y(N_i) − y(N_j)‖²_{2,a} = ‖x(N_i) − x(N_j)‖²_{2,a} + 2σ² holds
          for `"sum"`; its setting h = 10σ denoises only with `"mean"`
          (with `"sum"` nearly every patch of the window gets a weight
          close to 1, the result being close to the mean of the window).
          The default is `"mean"`.

    Parameters
    ----------
    patch : int
        Odd side of the patch.
    a : float, optional
        Standard deviation of the Gaussian, in pixels (> 0, may be inf).
    norm : {"mean", "sum"}
        Normalisation of G_a: mean 1 or sum 1.
    dtype, device
        Of the result.

    Returns
    -------
    Tensor
        `(patch,)`.
    """
    r = _radius("patch", patch)
    a = (patch - 1) / 4 if a is None else float(a)
    if not a > 0:
        raise ValueError(f"a must be > 0, got {a!r}")
    if norm not in ("mean", "sum"):
        raise ValueError(f"norm must be 'mean' or 'sum', got {norm!r}")
    u = torch.arange(-r, r + 1, dtype=torch.float64)
    if math.isinf(a) or r == 0:
        g = torch.ones_like(u)
    else:
        g = torch.exp(-(u * u) / (2 * a * a))
    g = g / g.sum() * (patch if norm == "mean" else 1)
    backend = resolve_backend(dtype, device)
    return g.to(dtype=backend["dtype"], device=backend["device"])


def search_offsets(search: int, ndim: int) -> list[tuple[int, ...]]:
    """Offsets t of the search window, in raster order.

    The window is the cube of side `search` (odd) centred on the pixel:
    t ∈ {−s … s}ᵈ with s = ⌊search/2⌋.

    Parameters
    ----------
    search : int
        Odd side of the search window.
    ndim : int
        Number of spatial dimensions d.

    Returns
    -------
    list of tuple of int
    """
    s = _radius("search", search)
    return list(itertools.product(range(-s, s + 1), repeat=ndim))


def offset_batch(pixels: int, itemsize: int, memory: int | None = None) -> int:
    """Offsets per batch of the vectorised NL-means.

    A batch of b offsets holds about b·`OFFSET_COPIES` buffers of the
    size of the padded block (P values of e bytes); it is sized to
    `OFFSET_SHARE` of the budget.

    Parameters
    ----------
    pixels : int
        Values P of the padded block.
    itemsize : int
        Bytes per value e.
    memory : int, optional
        Budget in bytes (default: `get_config().memory`).

    Returns
    -------
    int
    """
    memory = get_config().memory if memory is None else memory
    per = OFFSET_COPIES * pixels * itemsize
    return max(1, int(OFFSET_SHARE * memory) // per)


def patch_distances(
    data: Tensor,
    offsets: Sequence[Sequence[int]],
    kernel: Tensor,
    search: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Patch distances to the pixels at a set of offsets.

    For each offset t and pixel i, with j = i + t,

        d_t(i) = ‖y(N_i) − y(N_j)‖²_{2,a} = ∑_u G_a(u)·(y(i+u) − y(j+u))²

    (Section 3 of the paper), computed from shifted copies of the image:
    the squared difference of y and its shift by t, then a separable
    weighted sum over the patch. The differences are taken pixel by
    pixel (no expansion of the square), so d_t ≥ 0 and d_t(i) = d_{−t}(j)
    to rounding.

    !!! note "Choices not specified in the paper"
        Patches that cross the border of the image are completed by
        reflection about the edge pixels (`torch.nn.functional.pad`,
        mode `"reflect"`; mode `"replicate"` along axes too short to
        reflect). Offsets whose pixel j falls outside the image are
        invalid, which clips the search window at the border.

    Parameters
    ----------
    data : Tensor
        `(*spatial)`, one image or volume (d = `data.ndim`).
    offsets : sequence of tuple of int
        K offsets, each within the search window.
    kernel : Tensor
        1-D kernel of [`patch_kernel`][optix.denoise.utils.patch_kernel].
    search : int
        Side of the search window (sets the padding).

    Returns
    -------
    distances : Tensor
        `(K, *spatial)`; 0 where invalid.
    values : Tensor
        `(K, *spatial)`, y(i + t); 0 where invalid.
    valid : Tensor
        `(K, *spatial)` bool, whether i + t is in the image.
    """
    padded = _pad(data, kernel.numel() // 2, search // 2)
    return _distances(padded, data.shape, offsets, kernel, search // 2)


def _radius(name: str, side: object) -> int:
    side = _check_int(name, side)
    if side is None or side % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= 1, got {side!r}")
    return side // 2


def _pad(x: Tensor, r: int, s: int) -> Tensor:
    """Reflect by r (the patches), then zeros by s (invalid offsets)."""
    d = x.ndim
    y = x[None, None]
    if r > 0:
        mode = "reflect" if min(x.shape) > r else "replicate"
        y = F.pad(y, (r,) * (2 * d), mode=mode)
    if s > 0:
        y = F.pad(y, (s,) * (2 * d))
    return y[0, 0]


def _patch_sum(sq: Tensor, kernel: Tensor, ndim: int) -> Tensor:
    """Separable weighted sum over the patch, axis by axis (shrinks 2r)."""
    p = kernel.numel()
    for axis in range(sq.ndim - ndim, sq.ndim):
        n = sq.shape[axis] - p + 1
        acc = kernel[0] * sq.narrow(axis, 0, n)
        for u in range(1, p):
            acc = acc + kernel[u] * sq.narrow(axis, u, n)
        sq = acc
    return sq


def _distances(
    xq: Tensor,
    shape: Sequence[int],
    offsets: Sequence[Sequence[int]],
    kernel: Tensor,
    s: int,
) -> tuple[Tensor, Tensor, Tensor]:
    d = len(shape)
    r = kernel.numel() // 2
    p = r + s  # position of the image in the padded array
    ref = xq[tuple(slice(s, s + n + 2 * r) for n in shape)]
    shifted = torch.stack(
        [
            xq[
                tuple(
                    slice(s + t, s + t + n + 2 * r)
                    for t, n in zip(o, shape, strict=True)
                )
            ]
            for o in offsets
        ]
    )
    values = torch.stack(
        [
            xq[
                tuple(
                    slice(p + t, p + t + n)
                    for t, n in zip(o, shape, strict=True)
                )
            ]
            for o in offsets
        ]
    )
    dist = _patch_sum((ref - shifted).square(), kernel, d)
    valid = torch.ones(dist.shape, dtype=torch.bool, device=xq.device)
    for k, o in enumerate(offsets):
        for axis, (t, n) in enumerate(zip(o, shape, strict=True)):
            if t > 0:
                valid[(k,) + (slice(None),) * axis + (slice(n - t, None),)] = (
                    False
                )
            elif t < 0:
                valid[(k,) + (slice(None),) * axis + (slice(None, -t),)] = (
                    False
                )
    zero = torch.zeros((), dtype=dist.dtype, device=dist.device)
    return dist.where(valid, zero), values.where(valid, zero), valid


def _nlm_image(
    x: Tensor,
    sigma: float,
    h: float,
    patch: int,
    search: int,
    a: float | None,
    norm: KernelNorm,
    bias_correction: bool,
) -> Tensor:
    """NL-means of one image or volume `(*spatial)`."""
    kernel = patch_kernel(patch, a, norm, dtype=x.dtype, device=x.device)
    s = search // 2
    xq = _pad(x, patch // 2, s)
    offsets = search_offsets(search, x.ndim)
    padded = math.prod(n + 2 * (patch // 2) for n in x.shape)
    batch = offset_batch(padded, x.element_size())
    num = torch.zeros_like(x)
    den = torch.zeros_like(x)
    inv_h2 = 0.0 if math.isinf(h) else 1.0 / (h * h)
    # E of the distance between two patches of noise: 2σ²·∑G_a
    mass = float(kernel.double().sum()) ** x.ndim
    shift = 2.0 * sigma * sigma * mass if bias_correction else 0.0
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    for start in range(0, len(offsets), batch):
        dist, values, valid = _distances(
            xq, x.shape, offsets[start : start + batch], kernel, s
        )
        if shift:
            dist = (dist - shift).clamp(min=0)
        w = torch.exp(-dist * inv_h2).where(valid, zero)
        # sequential over offsets: the sum does not depend on the batches
        for k in range(w.shape[0]):
            num = num + w[k] * values[k]
            den = den + w[k]
    return num / den


def _check_h(h: object) -> float | None:
    if h is None:
        return None
    if isinstance(h, bool):
        raise ValueError(f"h must be a number > 0, got {h!r}")
    v = float(h)
    if not v > 0 or math.isnan(v):
        raise ValueError(f"h must be > 0 (may be inf), got {h!r}")
    return v


def _check_ndim(ndim: object) -> int:
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3, got {ndim!r}")
    return int(ndim)


# -- functional API ----------------------------------------------------------
def nl_means(
    data: Tensor,
    sigma: float,
    *,
    h: float | None = None,
    patch: int = 7,
    search: int = 21,
    a: float | None = None,
    kernel_norm: KernelNorm = "mean",
    ndim: int = 2,
    bias_correction: bool = False,
) -> Tensor:
    """Non-local means denoising of images with white Gaussian noise.

    Model: y(i) = x(i) + ε(i), ε ~ 𝒩(0, σ²) i.i.d. (Eq. (1) of the
    paper, which writes v for the data y and u for the clean image x).
    Each pixel is replaced by a weighted mean of the pixels j of its
    search window W_i, weighted by the similarity of their patches
    (Section 3):

        NL[y](i) = ∑_{j ∈ W_i} w(i, j)·y(j),
        w(i, j) = exp(−‖y(N_i) − y(N_j)‖²_{2,a} / h²) / Z(i),
        Z(i) = ∑_{j ∈ W_i} exp(−‖y(N_i) − y(N_j)‖²_{2,a} / h²),

    where N_i is the patch centred on i and ‖·‖²_{2,a} the squared
    Euclidean distance weighted by a Gaussian kernel G_a of standard
    deviation a (see
    [`patch_distances`][optix.denoise.utils.patch_distances]). The
    weights satisfy 0 ≤ w(i, j) ≤ 1 and ∑_j w(i, j) = 1; the pixel itself
    has the largest unnormalised weight, exp(0) = 1. Section 5 sets the
    parameters used here by default: a 21 × 21 search window, 7 × 7
    patches and h = 10σ.

    As h → 0 the result tends to the data (only the pixel's own patch
    has a non-vanishing weight, unless another patch is identical); as
    h → ∞ it tends to the mean of the search window. `h = inf` is
    accepted and gives that mean.

    The paper notes that E‖y(N_i) − y(N_j)‖²_{2,a} =
    ‖x(N_i) − x(N_j)‖²_{2,a} + 2σ² (Section 3, for a kernel of sum 1)
    but uses the distance as is. `bias_correction=True` uses
    max(‖·‖²_{2,a} − 2σ²·∑G_a, 0) instead, the expected distance between
    two patches of noise being 2σ²·∑G_a (not in the paper).

    !!! note "Choices not specified in the paper"
        * The kernel G_a: a = (patch − 1)/4, with mean 1 over the patch
          (`kernel_norm="mean"`), so that the distance is a weighted sum
          of squared differences. The paper's identity above suggests a
          kernel of sum 1 (`kernel_norm="sum"`), but with it h = 10σ
          averages nearly the whole search window; see
          [`patch_kernel`][optix.denoise.utils.patch_kernel].
        * Borders: patches are completed by reflection, and the search
          window is clipped to the image (see
          [`patch_distances`][optix.denoise.utils.patch_distances]).
        * Volumes (`ndim=3`): cubic patches and search windows of the
          same sides, with the separable 3-D kernel.

    !!! note "Cost and memory"
        The cost is about searchᵈ·(2·patch + 4) operations per pixel
        (the patch sum is separable). The computation is vectorised over
        batches of offsets of the search window: each batch holds a few
        shifted copies of the image, sized from the memory budget of
        [`optix.config`][optix.config]. It runs on any torch device and,
        the offsets being accumulated one by one in a fixed order, does
        not depend on the batch size (to rounding). The result at a
        pixel depends on the data within
        [`nl_means_halo`][optix.denoise.nl_means_halo] pixels, which
        [`NLMeans`][optix.denoise.NLMeans] uses to stream large arrays
        in blocks.

    !!! references "References"
        1. Buades A, Coll B, Morel J-M. *A non-local algorithm for image
           denoising.* IEEE Computer Society Conference on Computer
           Vision and Pattern Recognition (CVPR'05), vol. 2, 2005.
           [doi:10.1109/CVPR.2005.38](https://doi.org/10.1109/CVPR.2005.38)

        ??? quote "BibTeX"
            ```bibtex
            @inproceedings{buades2005nonlocal,
              author    = {Buades, Antoni and Coll, Bartomeu and
                           Morel, Jean-Michel},
              title     = {A non-local algorithm for image denoising},
              booktitle = {IEEE Computer Society Conference on Computer
                           Vision and Pattern Recognition (CVPR'05)},
              volume    = {2},
              year      = {2005},
              doi       = {10.1109/CVPR.2005.38}
            }
            ```

    Parameters
    ----------
    data : Tensor
        `(..., H, W)` (or `(..., D, H, W)` with `ndim=3`); each image is
        denoised separately (floating point; other dtypes are converted
        to the configured dtype).
    sigma : float
        Noise standard deviation σ > 0, in the units of `data`.
    h : float, optional
        Filtering parameter h > 0 (may be inf), in the units of `data`.
        Default: 10σ (Section 5).
    patch : int
        Odd side of the patches (paper: 7).
    search : int
        Odd side of the search window (paper: 21).
    a : float, optional
        Standard deviation of the kernel G_a, in pixels. Default:
        (patch − 1)/4.
    kernel_norm : {"mean", "sum"}
        Normalisation of G_a: mean 1 over the patch (default) or sum 1
        (see [`patch_kernel`][optix.denoise.utils.patch_kernel]).
    ndim : {2, 3}
        Number of spatial dimensions (3 is not in the paper).
    bias_correction : bool
        Subtract 2σ² from the patch distances (not in the paper).

    Returns
    -------
    Tensor
        Of the shape of `data`.
    """
    sigma = _check_positive("sigma", sigma)
    ndim = _check_ndim(ndim)
    h = _check_h(h)
    h = 10.0 * sigma if h is None else h
    patch_kernel(patch, a, kernel_norm)  # validates
    _radius("search", search)
    if data.ndim < ndim:
        raise ValueError(f"data needs at least {ndim} dims, got {data.ndim}")
    if not data.is_floating_point():
        data = data.to(resolve_backend(like=data)["dtype"])
    shape = data.shape
    flat = data.reshape(-1, *shape[-ndim:])
    out = torch.stack(
        [
            _nlm_image(
                x, sigma, h, patch, search, a, kernel_norm, bias_correction
            )
            for x in flat
        ]
    )
    return out.reshape(shape)


def nl_means_halo(*, patch: int = 7, search: int = 21) -> int:
    """Halo, in pixels, that makes block-wise NL-means exact.

    ⌊search/2⌋ + ⌊patch/2⌋ (13 for the paper's parameters): the pixels
    of the search window and their patches.

    Parameters
    ----------
    patch, search : int
        As in [`nl_means`][optix.denoise.nl_means].

    Returns
    -------
    int
    """
    return _radius("search", search) + _radius("patch", patch)


# -- estimator ---------------------------------------------------------------
class NLMeans(_BlockMatching):
    """Non-local means denoising (Buades, Coll & Morel 2005).

    Applies [`nl_means`][optix.denoise.nl_means] to each image (the
    spatial axes of the layout; other axes are samples or groups). The
    noise is white Gaussian with a known σ, or Poisson-Gaussian with the
    parameters of a [`SensorEstimate`][optix.estimates.SensorEstimate],
    as in [`BM3D`][optix.denoise.BM3D]: the image is denoised after the
    generalised Anscombe transform with σ = 1 (so h defaults to 10 in
    stabilised units), and transformed back by its exact unbiased
    inverse; maps of the estimate apply pixel by pixel. The variance
    stabilisation is not in the paper.

    [`fit`][optix.Estimator.fit] only sets the noise model (NL-means
    needs no intensity range). [`apply`][optix.Estimator.apply] streams
    blocks with the halo of [`nl_means_halo`][optix.denoise.nl_means_halo],
    so arrays larger than memory give the whole-image result.

    !!! note
        Faithful mode is the setting of Section 5 of the paper (2-D,
        7 × 7 patches, 21 × 21 search window, h = 10σ). `h`, `patch`,
        `search`, `ndim=3` and `bias_correction` are extensions; `a`
        and `kernel_norm` are choices the paper leaves open, and
        `inverse` and `block` do not change the method.

    !!! references "References"
        1. Buades A, Coll B, Morel J-M. *A non-local algorithm for image
           denoising.* IEEE Computer Society Conference on Computer
           Vision and Pattern Recognition (CVPR'05), vol. 2, 2005.
           [doi:10.1109/CVPR.2005.38](https://doi.org/10.1109/CVPR.2005.38)

        ??? quote "BibTeX"
            ```bibtex
            @inproceedings{buades2005nonlocal,
              author    = {Buades, Antoni and Coll, Bartomeu and
                           Morel, Jean-Michel},
              title     = {A non-local algorithm for image denoising},
              booktitle = {IEEE Computer Society Conference on Computer
                           Vision and Pattern Recognition (CVPR'05)},
              volume    = {2},
              year      = {2005},
              doi       = {10.1109/CVPR.2005.38}
            }
            ```

    Examples
    --------
    >>> import torch
    >>> from optix.denoise import NLMeans
    >>> clean = torch.linspace(0, 255, 48).expand(48, 48)
    >>> noisy = clean + 20 * torch.randn(48, 48)
    >>> den = NLMeans(sigma=20).fit().apply(noisy)
    >>> bool((den - clean).square().mean() < (noisy - clean).square().mean())
    True
    """

    tags = Tags(
        targets=("denoised",),
        streaming="halo",
        noise="gaussian",
        reference="doi:10.1109/CVPR.2005.38",
    )
    paper = {
        "sigma": None,
        "h": None,
        "patch": 7,
        "search": 21,
        "a": None,
        "kernel_norm": "mean",
        "ndim": 2,
        "bias_correction": False,
        "inverse": "exact",
        "block": None,
    }
    recommended = dict(paper)

    def __init__(
        self,
        *,
        sigma: float | None = None,
        h: float | None = None,
        patch: int | None = None,
        search: int | None = None,
        a: float | None = None,
        kernel_norm: KernelNorm | None = None,
        ndim: int | None = None,
        bias_correction: bool | None = None,
        inverse: str | None = None,
        block: Sequence[int | None] | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: Layout | None = None,
    ) -> None:
        """Configure the denoiser.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation σ of white Gaussian noise, in data units.
            Omit it to give a `SensorEstimate` to `fit` instead.
        h : float, optional
            Extension: filtering parameter h (data units; stabilised
            units with a `SensorEstimate`). Default: 10σ (Section 5).
        patch : int, optional
            Extension: odd side of the patches (paper: 7).
        search : int, optional
            Extension: odd side of the search window (paper: 21).
        a : float, optional
            Standard deviation of the patch kernel G_a, in pixels (not
            specified in the paper). Default: (patch − 1)/4.
        kernel_norm : {"mean", "sum"}, optional
            Normalisation of G_a (not specified in the paper): mean 1
            over the patch (default) or sum 1; see
            [`patch_kernel`][optix.denoise.utils.patch_kernel].
        ndim : {2, 3}, optional
            Spatial dimensions of the layout; 3 (volumes) is an
            extension. Default: 2.
        bias_correction : bool, optional
            Extension: subtract 2σ² from the patch distances (not in the
            paper). Default: `False`.
        inverse : {"exact", "closed-form", "asymptotic", "algebraic"}
            Inverse of the Anscombe transform (see
            [`inverse_anscombe`][optix.sensor.inverse_anscombe]).
            Default: `"exact"`.
        block : sequence of int or None, optional
            Core block size per spatial axis (default: from the memory
            budget). The result does not depend on it.
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        self._setup()

    def _setup(self) -> None:
        o = self.options
        if o["sigma"] is not None:
            o["sigma"] = _check_positive("sigma", o["sigma"])
        o["h"] = _check_h(o["h"])
        _radius("patch", o["patch"])
        _radius("search", o["search"])
        patch_kernel(o["patch"], o["a"], o["kernel_norm"])  # validates
        o["ndim"] = self._ndim = _check_ndim(o["ndim"])
        o["bias_correction"] = bool(o["bias_correction"])
        if o["inverse"] not in (
            "exact",
            "closed-form",
            "asymptotic",
            "algebraic",
        ):
            raise ValueError(f"Unknown inverse {o['inverse']!r}")
        self.is_faithful = all(
            o[k] == self.paper[k]
            for k in ("h", "patch", "search", "ndim", "bias_correction")
        )
        self._range = None
        self._vst = None
        self._maps = None
        self._table = None

    def fit(
        self,
        data: ArrayLike | None = None,
        *,
        layout: Layout | None = None,
        init: object = None,
        **kwargs: object,
    ) -> Self:
        """Set the noise model.

        Gaussian noise: `sigma` given at construction. Poisson-Gaussian
        noise: `init` is a [`SensorEstimate`][optix.estimates.SensorEstimate]
        (or an estimator holding one): the data are stabilised by the
        generalised Anscombe transform, denoised with σ = 1 and
        transformed back by the inverse selected by `inverse`. Maps of
        the estimate (of shape `(*group, *spatial)` or broadcastable to
        it) are applied pixel by pixel, cropped to each streamed block.

        Parameters
        ----------
        data : array-like, optional
            Not read; only its layout is recorded.
        layout : Layout, optional
        init : SensorEstimate or Estimator, optional
            Noise parameters (gain α, offset μ, read variance σ²; scalars
            or maps).
        **kwargs
            Ignored (`callback`, `verbose`).

        Returns
        -------
        Estimator
            self
        """
        return super().fit(data, layout=layout, init=init, **kwargs)

    def _fit_range(self, data: ArrayLike | None) -> None:
        self._range = None

    def _check_fitted(self) -> None:
        if self._vst is None and self.options["sigma"] is None:
            raise RuntimeError("Call fit before apply")

    def _overhead(self) -> float:
        return BLOCK_COPIES / (1 - OFFSET_SHARE)

    def _h(self) -> float:
        h = self.options["h"]
        return 10.0 * self._sigma() if h is None else h

    def _core(self, x: Tensor, origin: tuple[int, ...]) -> Tensor:
        o = self.options
        return _nlm_image(
            x,
            self._sigma(),
            self._h(),
            o["patch"],
            o["search"],
            o["a"],
            o["kernel_norm"],
            o["bias_correction"],
        )

    def halo(self) -> int:
        """Halo of the block-wise processing.

        Returns
        -------
        int
        """
        o = self.options
        return nl_means_halo(patch=o["patch"], search=o["search"])
