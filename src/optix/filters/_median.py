"""Exact median filters (square/rectangular windows and disks)."""

import itertools
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["median_disk", "median_filter"]

# Windows of at most this many pixels use the direct (sorting) method in
# `method="auto"`, whose cost grows with the window area.
_SORT_AREA = 225


def _pair(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        size = (size, size)
    sh, sw = (int(s) for s in size)
    if sh < 1 or sw < 1:
        raise ValueError(f"window size must be >= 1: {size}")
    return sh, sw


def _ranks(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Sorted values, the rank of every pixel and the valid (non-NaN) mask.

    Ranks are unique (a stable sort breaks ties by position) and NaN
    ranks last, so the k-th smallest rank of a set of pixels points to its
    k-th smallest value.
    """
    flat = x.reshape(-1)
    vals, perm = torch.sort(flat, stable=True)
    rank = torch.empty_like(perm)
    rank[perm] = torch.arange(perm.numel(), device=x.device)
    if x.is_floating_point() or x.is_complex():
        valid = ~torch.isnan(x)
    else:
        valid = torch.ones_like(x, dtype=torch.bool)
    return vals, rank.reshape(x.shape), valid


def _lift(fn: Callable[..., Tensor], data: Tensor, *args: object) -> Tensor:
    if data.ndim < 2:
        raise ValueError("median filtering needs 2 spatial dims")
    lead = data.shape[:-2]
    flat = data.reshape(-1, *data.shape[-2:])
    out = torch.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = fn(flat[i], *args)
    return out.reshape(*lead, *data.shape[-2:])


def median_filter(
    data: Tensor,
    size: int | tuple[int, int],
    *,
    method: str = "auto",
    budget: int = 1 << 24,
) -> Tensor:
    """Exact median filter over a rectangular window.

    For every pixel (i, j) of the last two dims of `data`,

        out[i, j] = lower median { data[i', j'] : i' ∈ Wᵢ, j' ∈ Vⱼ },
        Wᵢ = [i − ⌊sₕ/2⌋, i − ⌊sₕ/2⌋ + sₕ − 1] ∩ [0, H),

    (likewise Vⱼ with s_w), where the lower median of n values is the
    value of rank ⌊(n − 1)/2⌋ (0-based) in increasing order, as returned
    by `torch.median` and `torch.nanmedian`. Pixels outside the image are
    ignored, so border windows are shrunk rather than padded. NaN pixels
    are ignored as well; a window holding only NaN gives NaN. The result
    equals `torch.nanmedian` over each window exactly (it is one of the
    window's values), for floating and integer dtypes.

    Leading dims are batch dims, filtered independently.

    Parameters
    ----------
    data : (..., H, W) Tensor
        Image(s) to filter.
    size : int or (int, int)
        Window side sₕ = s_w, or (sₕ, s_w). An even side puts one more pixel
        before the centre than after it. A side larger than 2·n − 1 along a
        dim of length n is equivalent to 2·n − 1 there (every window then
        spans the whole dim).
    method : {"auto", "histogram", "sort"}
        `"histogram"`: rank histograms, whose cost per pixel does not
        depend on the window size;
        `"sort"`: sorts every window, costing O(sₕ·s_w·log(sₕ·s_w)) per
        pixel. `"auto"` sorts small windows (at most 225 pixels) and uses
        histograms otherwise. Both give the same result.
    budget : int
        `"sort"` only: number of window values held in memory at once.

    Returns
    -------
    (..., H, W) Tensor
        The filtered image, with the dtype and device of `data`.

    !!! note "Method"
        The image is first replaced by the ranks of its pixels (a stable
        sort, NaN last), so that ties are broken by position and every
        rank is unique; the median rank of a window maps back to the
        median value through the sorted values, whatever the dtype. Ranks
        are grouped into blocks of F ≈ √(HW)/4 consecutive ranks, and
        blocks into S ≈ G superblocks of G blocks. Output rows are visited
        in order; for every column, histograms of the block and superblock
        indices of the sₕ window rows are kept, and updated by adding the
        entering row and removing the leaving one. Prefix sums of the
        column histograms along the columns give the window histogram of
        every pixel; a prefix sum over the superblocks locates the
        superblock that holds the median rank, then one over its G blocks
        locates the block. The median rank is finally selected among the F
        ranks of that block by testing which of their pixels lie inside
        the window. The cost per pixel is O(S + G + F) ≈ O((HW)^½)
        operations whatever the window size, plus the block prefix sums of
        the superblocks met on the current row (few when the median varies
        smoothly). Memory is about 4·√(HW)·W integers. Column histograms with a
        sliding update are the idea of Perreault & Hébert (2007); the
        search over rank blocks is a choice of this implementation.

    !!! references "References"
        1. Perreault S, Hébert P. *Median filtering in constant time.*
           IEEE Trans Image Process **16**(9):2389–2394, 2007.
           [doi:10.1109/TIP.2007.902329](https://doi.org/10.1109/TIP.2007.902329)

        ??? quote "BibTeX"
            ```bibtex
            @article{perreault2007median,
              author  = {Perreault, S. and H{\\'e}bert, P.},
              title   = {Median filtering in constant time},
              journal = {IEEE Transactions on Image Processing},
              volume  = {16}, number = {9}, pages = {2389--2394},
              year    = {2007}, doi = {10.1109/TIP.2007.902329}
            }
            ```
    """
    sh, sw = _pair(size)
    if method == "auto":
        h, w = data.shape[-2:] if data.ndim >= 2 else (1, 1)
        area = min(sh, 2 * h - 1) * min(sw, 2 * w - 1)
        method = "sort" if area <= _SORT_AREA else "histogram"
    if method == "histogram":
        return _lift(_median_hist, data, sh, sw)
    if method == "sort":
        return _lift(_median_sort, data, sh, sw, budget)
    raise ValueError(f"Unknown method {method!r}")


def _clip(size: int, n: int) -> tuple[int, int]:
    s = min(size, 2 * n - 1)
    return s, s // 2


def _median_sort(x: Tensor, sh: int, sw: int, budget: int) -> Tensor:
    h, w = x.shape
    vals, rank, valid = _ranks(x)
    # ranks are exact in float64 (and in float32 below 2²⁴ pixels)
    ft = torch.float32 if rank.numel() <= 1 << 24 else torch.float64
    r = torch.where(valid, rank.to(ft), math.nan)
    sh, th = _clip(sh, h)
    sw, tw = _clip(sw, w)
    pad = F.pad(
        r[None, None], (tw, sw - tw - 1, th, sh - th - 1), value=math.nan
    )
    pad = pad[0, 0]
    med = torch.empty_like(r)
    per = max(1, budget // (sh * sw))
    cols = min(w, per)
    rows = max(1, per // cols)
    for i in range(0, h, rows):
        ie = min(h, i + rows)
        for j in range(0, w, cols):
            je = min(w, j + cols)
            win = pad[i : ie + sh - 1, j : je + sw - 1].unfold(0, sh, 1)
            win = win.unfold(1, sw, 1)  # (rows, cols, sh, sw)
            med[i:ie, j:je] = (
                win.reshape(ie - i, je - j, -1).nanmedian(-1).values
            )
    return _from_rank(vals, med)


def _from_rank(vals: Tensor, med: Tensor) -> Tensor:
    ok = ~torch.isnan(med)
    out = vals[torch.where(ok, med, 0).long()]
    if ok.all():
        return out
    return torch.where(ok, out, math.nan)


def _box_cols(pre: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """pre[..., hi] − pre[..., lo]: column-window sums from prefix sums."""
    return pre.index_select(-1, hi) - pre.index_select(-1, lo)


def _block_sizes(n: int) -> tuple[int, int, int]:
    """Ranks per block F, blocks per superblock G and superblocks S."""
    fw = max(1, math.ceil(math.sqrt(n) / 4))
    nb = math.ceil(n / fw)
    g = max(1, math.ceil(math.sqrt(nb)))
    return fw, g, math.ceil(nb / g)


def _median_hist(x: Tensor, sh: int, sw: int) -> Tensor:
    h, w = x.shape
    n = h * w
    dev = x.device
    i32 = torch.int32
    vals, rank, valid = _ranks(x)
    sh, th = _clip(sh, h)
    sw, tw = _clip(sw, w)

    # ranks are grouped into blocks of F, blocks into superblocks of G
    fw, g, ns = _block_sizes(n)
    nb = g * ns
    blk = (rank // fw).to(i32)  # (h, w) block of every pixel
    sup = (blk // g).to(i32)  # (h, w) superblock of every pixel
    # column of the pixel of every rank while its row is inside the row
    # window, else −1 (never inside a column window)
    cdt = torch.int16 if w < (1 << 15) else i32
    ccw = torch.full((nb * fw,), -1, dtype=cdt, device=dev)
    jj = torch.arange(w, device=dev)
    j_col = jj.to(cdt)

    # column window [c0, c1] of every output column, and its prefix indices
    c0 = (jj - tw).clamp_min(0)
    c1 = (jj - tw + sw - 1).clamp_max(w - 1)
    lo, hi = c0, c1 + 1
    c0, c1 = c0.to(cdt)[:, None], c1.to(cdt)[:, None]
    sv = torch.zeros(h + 1, w + 1, dtype=torch.long, device=dev)
    sv[1:, 1:] = valid.long().cumsum(0).cumsum(1)

    # column histograms of the row window, over blocks and superblocks
    hist_blk = torch.zeros(nb, w, dtype=i32, device=dev)
    hist_sup = torch.zeros(ns, w, dtype=i32, device=dev)
    pres = torch.zeros(ns, w + 1, dtype=i32, device=dev)
    one = torch.ones(w, dtype=i32, device=dev)
    zero = torch.zeros(1, w, dtype=i32, device=dev)
    g_ids = torch.arange(g, device=dev)
    g_off = g_ids[:, None] * (w + 1)

    def update(row: int, sign: int) -> None:
        if 0 <= row < h:
            s = one if sign > 0 else -one
            hist_blk.view(-1).index_add_(0, blk[row].long() * w + jj, s)
            hist_sup.view(-1).index_add_(0, sup[row].long() * w + jj, s)
            ccw[rank[row]] = j_col if sign > 0 else -1

    for row in range(0, min(h, sh - th)):
        update(row, 1)
    out = torch.empty_like(x)
    has_nan = not bool(valid.all())
    for i in range(h):
        if i > 0:
            update(i - th - 1, -1)
            update(i - th + sh - 1, 1)
        r0, r1 = max(0, i - th), min(h - 1, i - th + sh - 1)
        nv = sv[r1 + 1, hi] - sv[r0, hi] - sv[r1 + 1, lo] + sv[r0, lo]
        k = ((nv - 1).clamp_min(0) // 2).to(i32)  # target rank (0-based)
        # superblock: the first whose cumulative window count exceeds k
        torch.cumsum(hist_sup, 1, dtype=i32, out=pres[:, 1:])
        cum = _box_cols(pres, lo, hi).cumsum(0, dtype=i32)  # (S, w)
        s = (cum <= k).sum(0, dtype=torch.long)  # (w,)
        k = k - torch.cat([zero, cum]).gather(0, s[None])[0]
        # block within the superblock, from the block prefix sums of the
        # superblocks in use
        u = torch.unique(s)
        sel = (u[:, None] * g + g_ids).reshape(-1)
        pre = torch.zeros(sel.numel(), w + 1, dtype=i32, device=dev)
        torch.cumsum(
            hist_blk.index_select(0, sel), 1, dtype=i32, out=pre[:, 1:]
        )
        loc = torch.searchsorted(u, s) * g
        base = loc[None] * (w + 1) + g_off  # (G, w) flat row offsets
        pf = pre.view(-1)
        cum = (pf[base + hi] - pf[base + lo]).cumsum(0, dtype=i32)  # (G, w)
        bb = (cum <= k).sum(0, dtype=torch.long)
        k = k - torch.cat([zero, cum]).gather(0, bb[None])[0]
        b = s * g + bb
        # rank within the block: the (k + 1)-th of its pixels in the window
        cc = ccw.view(nb, fw)[b]  # (w, F)
        inside = (cc >= c0) & (cc <= c1)
        pos = torch.searchsorted(
            inside.cumsum(1, dtype=i32), (k + 1)[:, None]
        )[:, 0]
        res = vals[(b * fw + pos).clamp_max(n - 1)]
        if has_nan:
            res = torch.where(nv > 0, res, math.nan)
        out[i] = res
    return out


def median_disk(data: Tensor, radius: int, *, rows: int = 64) -> Tensor:
    """Exact median filter over a disk of `radius`.

    For every pixel (i, j) of the last two dims of `data`, the lower median
    (rank ⌊(n − 1)/2⌋ of the n values, as `torch.nanmedian`) of the pixels
    (i + di, j + dj) with di² + dj² ≤ radius². Pixels outside the image and
    NaN pixels are ignored; a window holding only NaN gives NaN. Every
    window is sorted, so the cost per pixel grows with the disk area; it
    is computed in strips of `rows` rows. Leading dims are batch dims.

    Parameters
    ----------
    data : (..., H, W) Tensor
        Image(s) to filter (floating or integer).
    radius : int
        Disk radius in pixels.
    rows : int
        Number of rows filtered at once.

    Returns
    -------
    (..., H, W) Tensor
        The filtered image, with the dtype and device of `data`.
    """
    return _lift(_median_disk, data, int(radius), rows)


def _median_disk(x: Tensor, r: int, rows: int) -> Tensor:
    vals, rank, valid = _ranks(x)
    ft = torch.float32 if rank.numel() <= 1 << 24 else torch.float64
    rk = torch.where(valid, rank.to(ft), math.nan)
    offs = [
        (i, j)
        for i, j in itertools.product(range(-r, r + 1), repeat=2)
        if i * i + j * j <= r * r
    ]
    h, w = x.shape
    pad = F.pad(rk[None, None], (r, r, r, r), value=math.nan)[0, 0]
    med = torch.empty_like(rk)
    for s in range(0, h, rows):
        e = min(h, s + rows)
        stack = torch.stack(
            [pad[s + r + i : e + r + i, r + j : r + j + w] for i, j in offs]
        )
        med[s:e] = stack.nanmedian(0).values
    return _from_rank(vals, med)
