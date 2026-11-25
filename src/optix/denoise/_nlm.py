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
# Image-sized buffers per offset of a batch, at the peak: the squared
# differences, and the accumulator and product of one pass of the
# separable patch sum (the weights are then computed in place).
OFFSET_COPIES = 3
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

    !!! note "Choice not specified in the paper"
        The paper does not give a. The default a = (patch − 1)/4 is 1.5
        for 7 × 7 patches (the patch spans ±2a). `a = inf` gives uniform
        weights; a has no effect for `patch=1`.

    !!! warning "Departure from the paper: the default normalisation"
        The paper implies a kernel of sum 1 (`"sum"`): its identity
        E‖y(N_i) − y(N_j)‖²_{2,a} = ‖x(N_i) − x(N_j)‖²_{2,a} + 2σ²
        (Section 3) holds only then, and Sections 1 and 2.1 use a
        normalised Gaussian. Its setting h = 10σ (Section 5) is not
        consistent with that kernel: the distance is then at most R² for
        data of range R, so every weight is at least exp(−R²/(100σ²)) of
        the pixel's own (0.20 at σ = 20 on 8-bit data), whereas Section 5
        and Fig. 2 describe weights that fall to nearly zero; the result
        is close to the mean of the search window. The default `"mean"`
        is a repair inferred from the paper's numbers, not stated in it:
        with a kernel of mean 1 the expected distance between two 7 × 7
        patches of noise is 2·49·σ² ≈ h² = 100σ², so h = 10σ is the
        natural scale and the weights behave as Section 5 describes. The
        two normalisations differ by the factor patchᵈ (49 for 7 × 7
        patches), a factor 7 on h.

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
    if r > 0 and not a > 0:
        raise ValueError(f"a must be > 0, got {a!r}")
    if norm not in ("mean", "sum"):
        raise ValueError(f"norm must be 'mean' or 'sum', got {norm!r}")
    u = torch.arange(-r, r + 1, dtype=torch.float64)
    if r == 0 or math.isinf(a):
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
        mode `"reflect"`), axis by axis; an axis of at most ⌊patch/2⌋
        pixels is completed by replication (mode `"replicate"`) instead.
        Offsets whose pixel j falls outside the image are invalid, which
        clips the search window at the border.

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
    r, s = kernel.numel() // 2, search // 2
    padded = _pad(data, r, s)
    dist = _distances(padded, data.shape, offsets, kernel, s)
    valid = torch.ones(dist.shape, dtype=torch.bool, device=data.device)
    _invalidate(valid, offsets, data.shape, False)
    values = torch.stack(
        [_shift(padded, r + s, o, data.shape) for o in offsets]
    )
    return dist.masked_fill_(~valid, 0), values.masked_fill_(~valid, 0), valid


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
        for axis, n in enumerate(x.shape):
            pad = [0] * (2 * d)
            # F.pad lists the last axis first
            pad[2 * (d - 1 - axis)] = pad[2 * (d - 1 - axis) + 1] = r
            y = F.pad(y, pad, mode="reflect" if n > r else "replicate")
    if s > 0:
        y = F.pad(y, (s,) * (2 * d))
    return y[0, 0]


def _shift(
    xq: Tensor, start: int, offset: Sequence[int], size: Sequence[int]
) -> Tensor:
    """View of `xq` of shape `size`, from `start + offset` on each axis."""
    return xq[
        tuple(
            slice(start + t, start + t + n)
            for t, n in zip(offset, size, strict=True)
        )
    ]


def _invalidate(
    t: Tensor,
    offsets: Sequence[Sequence[int]],
    shape: Sequence[int],
    value: object,
) -> None:
    """Set `t[k]` to `value` (in place) where i + offsets[k] is outside."""
    for k, o in enumerate(offsets):
        for axis, (u, n) in enumerate(zip(o, shape, strict=True)):
            if u > 0:
                t[(k,) + (slice(None),) * axis + (slice(n - u, None),)] = value
            elif u < 0:
                t[(k,) + (slice(None),) * axis + (slice(None, -u),)] = value


def _distances(
    xq: Tensor,
    shape: Sequence[int],
    offsets: Sequence[Sequence[int]],
    kernel: Tensor,
    s: int,
) -> Tensor:
    """Patch distances `(K, *shape)`; arbitrary (finite) where invalid."""
    d = len(shape)
    r = kernel.numel() // 2
    size = [n + 2 * r for n in shape]
    ref = _shift(xq, s, (0,) * d, size)
    sq = torch.empty((len(offsets), *size), dtype=xq.dtype, device=xq.device)
    for k, o in enumerate(offsets):
        torch.sub(ref, _shift(xq, s, o, size), out=sq[k])
    sq.square_()
    # separable weighted sum over the patch, axis by axis (shrinks 2r); at
    # most three buffers of the batch are alive at once: `sq`, the
    # accumulator and the current term
    p = kernel.numel()
    for axis in range(1, d + 1):
        n = sq.shape[axis] - p + 1
        acc = torch.mul(sq.narrow(axis, 0, n), kernel[0])
        if p > 1:
            term = torch.empty_like(acc)
            for u in range(1, p):
                torch.mul(sq.narrow(axis, u, n), kernel[u], out=term)
                acc.add_(term)
            del term
        del sq
        sq = acc
    return sq


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
    # exp(−(d/h)/h): no overflow of 1/h², and d = 0 gives 1 for any h
    # (h below the smallest normal number of the dtype acts as that number)
    h = max(h, torch.finfo(x.dtype).tiny)
    # E of the distance between two patches of noise: 2σ²·∑G_a
    mass = float(kernel.double().sum()) ** x.ndim
    shift = 2.0 * sigma * sigma * mass if bias_correction else 0.0
    p = patch // 2 + s  # position of the image in the padded array
    for start in range(0, len(offsets), batch):
        batch_offsets = offsets[start : start + batch]
        w = _distances(xq, x.shape, batch_offsets, kernel, s)
        if shift:
            w.sub_(shift).clamp_(min=0)
        w.div_(h).div_(h).neg_().exp_()
        _invalidate(w, batch_offsets, x.shape, 0.0)
        # sequential over offsets: the sum does not depend on the batches
        for k, o in enumerate(batch_offsets):
            den.add_(w[k])
            num.add_(w[k].mul_(_shift(xq, p, o, x.shape)))
        del w  # before the next batch is allocated
    return num.div_(den)


def default_h(sigma: float, patch: int = 7, ndim: int = 2) -> float:
    """Default filtering parameter h = 10σ·√(K/49), K = patchᵈ.

    Section 5 of the paper sets h = 10σ for 7 × 7 patches (K = 49). With
    the default kernel (mean 1 over the patch), the expected distance
    between two patches of noise is 2σ²·K, so h is scaled by √(K/49) to
    keep the ratio of that distance to h² at its value in the paper,
    2·49/100 = 0.98. It is h = 10σ at the paper's settings.

    Parameters
    ----------
    sigma : float
        Noise standard deviation σ.
    patch : int
        Odd side of the patches.
    ndim : int
        Number of spatial dimensions d.

    Returns
    -------
    float
    """
    return 10.0 * sigma * math.sqrt(patch**ndim / 49)


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
    patches and h = 10σ. For other patches the default h is
    10σ·√(K/49) with K = patchᵈ (see
    [`default_h`][optix.denoise.utils.default_h]), which keeps h = 10σ
    at the paper's settings.

    This is the algorithm of Section 3, where the patch N_i contains the
    pixel i. The consistency results of Section 4 (Theorems 4 and 5) are
    stated for the patch without its centre, N_i∖{i}; that variant is
    not implemented. The paper prints ‖v(N_i) − v(N_i)‖ in Z(i); N_j is
    meant, as written above.

    As h → 0 the result tends to the data (only the pixel's own patch
    has a non-vanishing weight, unless another patch is identical); as
    h → ∞ it tends to the mean of the search window. `h = inf` is
    accepted and gives that mean.

    The paper notes that E‖y(N_i) − y(N_j)‖²_{2,a} =
    ‖x(N_i) − x(N_j)‖²_{2,a} + 2σ² (Section 3, for a kernel of sum 1)
    but uses the distance as is. `bias_correction=True` uses
    max(‖·‖²_{2,a} − 2σ²·∑G_a, 0) instead, the expected distance between
    two patches of noise being 2σ²·∑G_a (not in the paper).

    !!! warning "Departure from the paper: the kernel normalisation"
        The paper implies a kernel G_a of sum 1 (the identity above holds
        only then), but with it h = 10σ averages nearly the whole search
        window, contrary to Section 5 and Fig. 2; Sections 3 and 5 are
        inconsistent. The default `kernel_norm="mean"` (mean 1 over the
        patch, so the distance is a weighted sum of squared differences)
        is a repair inferred from h² = 100σ² ≈ 2·49·σ², the expected
        distance between two 7 × 7 patches of noise; see
        [`patch_kernel`][optix.denoise.utils.patch_kernel].
        `kernel_norm="sum"` with h = 10σ/7 gives the same result.

    !!! note "Choices not specified in the paper"
        * The width of G_a: a = (patch − 1)/4.
        * Borders: patches are completed by reflection, and the search
          window is clipped to the image (see
          [`patch_distances`][optix.denoise.utils.patch_distances]).
        * Volumes (`ndim=3`): cubic patches and search windows of the
          same sides, with the separable 3-D kernel.

    !!! note "Cost and memory"
        The cost is about searchᵈ·(d·patch + 4) operations per pixel
        (the patch sum is separable, one pass per axis). The computation
        is vectorised over batches of offsets of the search window: each
        offset of a batch holds at most three image-sized buffers, the
        batch being sized from the memory budget of
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
        Default: 10σ·√(K/49), K = patchᵈ, that is 10σ for 7 × 7 patches
        (Section 5).
    patch : int
        Odd side of the patches (paper: 7).
    search : int
        Odd side of the search window (paper: 21).
    a : float, optional
        Standard deviation of the kernel G_a, in pixels. Default:
        (patch − 1)/4.
    kernel_norm : {"mean", "sum"}
        Normalisation of G_a: mean 1 over the patch (default, a
        departure from the paper) or sum 1 (implied by the paper); see
        [`patch_kernel`][optix.denoise.utils.patch_kernel].
    ndim : {2, 3}
        Number of spatial dimensions (3 is not in the paper).
    bias_correction : bool
        Subtract 2σ²·∑G_a, the expected distance between two patches of
        noise, from the patch distances, clamped at 0 (not in the paper).

    Returns
    -------
    Tensor
        Of the shape of `data`.
    """
    sigma = _check_positive("sigma", sigma)
    ndim = _check_ndim(ndim)
    h = _check_h(h)
    patch_kernel(patch, a, kernel_norm)  # validates
    h = default_h(sigma, patch, ndim) if h is None else h
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
        7 × 7 patches, 21 × 21 search window, h = 10σ), with the kernel
        normalisation `kernel_norm="mean"`, a departure from the paper
        that repairs an inconsistency between its Sections 3 and 5 (see
        [`nl_means`][optix.denoise.nl_means]). `h`, `patch`, `search`,
        `ndim=3`, `bias_correction` and `kernel_norm="sum"` (the
        normalisation the paper implies) make `is_faithful` false; `a`
        is a choice the paper leaves open, and `inverse` and `block` do
        not change the method. The default h is 10σ·√(K/49), K =
        patchᵈ (see [`default_h`][optix.denoise.utils.default_h]).

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
            units with a `SensorEstimate`). Default: 10σ·√(K/49), K =
            patchᵈ, that is 10σ for 7 × 7 patches (Section 5).
        patch : int, optional
            Extension: odd side of the patches (paper: 7).
        search : int, optional
            Extension: odd side of the search window (paper: 21).
        a : float, optional
            Standard deviation of the patch kernel G_a, in pixels (not
            specified in the paper). Default: (patch − 1)/4.
        kernel_norm : {"mean", "sum"}, optional
            Normalisation of G_a: mean 1 over the patch (default, a
            departure from the paper) or sum 1 (implied by the paper,
            not faithful here); see
            [`patch_kernel`][optix.denoise.utils.patch_kernel].
        ndim : {2, 3}, optional
            Spatial dimensions of the layout; 3 (volumes) is an
            extension. Default: 2.
        bias_correction : bool, optional
            Extension: subtract 2σ²·∑G_a, the expected distance between
            two patches of noise, from the patch distances, clamped at 0
            (not in the paper). Default: `False`.
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
            for k in (
                "h",
                "patch",
                "search",
                "kernel_norm",
                "ndim",
                "bias_correction",
            )
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
        if h is None:
            return default_h(self._sigma(), self.options["patch"], self._ndim)
        return h

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
