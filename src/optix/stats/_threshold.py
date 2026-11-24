"""Otsu's global threshold, frame by frame."""

import torch
from torch import Tensor

__all__ = ["otsu_threshold"]


def otsu_threshold(
    data: Tensor, *, bins: int | None = 256, ndim: int = 2
) -> Tensor:
    """Otsu's threshold of each frame.

    The intensities of a frame are binned into a histogram with counts
    nₖ at bin centres cₖ; the threshold is the centre cₖ that maximises
    the between-class variance

        w₀(k) · w₁(k) · (μ₀(k) − μ₁(k))²,

    with w₀(k) = ∑_{j ≤ k} nⱼ, w₁(k) = ∑_{j > k} nⱼ and μ₀, μ₁ the mean
    centres of the two classes
    ([Otsu 1979](https://doi.org/10.1109/TSMC.1979.4310076)). The first
    maximiser is taken, and a constant frame returns its value.

    The binning follows scikit-image's `threshold_otsu`: `bins` equal
    bins over [min, max] (numpy's `histogram` edges), or, with
    `bins=None`, one bin per integer value from min to max (what
    scikit-image does for integer images).

    !!! references "References"
        1. Otsu N. *A threshold selection method from gray-level
           histograms.* IEEE Trans Syst Man Cybern **9**(1):62–66, 1979.
           [doi:10.1109/TSMC.1979.4310076](https://doi.org/10.1109/TSMC.1979.4310076)

        ??? quote "BibTeX"
            ```bibtex
            @article{otsu1979threshold,
              author  = {Otsu, Nobuyuki},
              title   = {A Threshold Selection Method from Gray-Level
                         Histograms},
              journal = {IEEE Transactions on Systems, Man, and
                         Cybernetics},
              volume  = {9}, number = {1}, pages = {62--66},
              year    = {1979}, doi = {10.1109/TSMC.1979.4310076}
            }
            ```

    Parameters
    ----------
    data : Tensor
        Frames `(..., *spatial)`.
    bins : int or None, default 256
        Number of equal bins, or None for one bin per integer value
        (`data` must then hold integers).
    ndim : int, default 2
        Number of trailing (spatial) dims forming a frame.

    Returns
    -------
    Tensor
        `data.shape[:-ndim]` thresholds (floating point).
    """
    if bins is not None and (isinstance(bins, bool) or int(bins) < 1):
        raise ValueError(
            f"bins must be a positive integer or None, got {bins}"
        )
    lead = data.shape[: data.ndim - ndim]
    flat = data.reshape(-1, *data.shape[data.ndim - ndim :]).flatten(1)
    if not flat.is_floating_point():
        flat = flat.to(torch.float64)
    out = torch.empty(flat.shape[0], dtype=flat.dtype, device=flat.device)
    for i, x in enumerate(flat):
        out[i] = _otsu(x, bins)
    return out.reshape(lead)


def _otsu(x: Tensor, bins: int | None) -> Tensor:
    lo, hi = x.min(), x.max()
    if lo == hi:
        return lo
    if bins is None:
        if not torch.equal(x, x.round()):
            raise ValueError("bins=None needs integer-valued data")
        lo_i = int(lo)
        counts = torch.bincount((x - lo).long()).to(x.dtype)
        centres = torch.arange(counts.numel(), dtype=x.dtype, device=x.device)
        centres = centres + lo_i
    else:
        # numpy.histogram: edges lo + i·(hi − lo)/bins, last edge = hi
        step = (hi - lo) / bins
        edges = torch.arange(bins + 1, dtype=x.dtype, device=x.device) * step
        edges = edges + lo
        edges[-1] = hi
        idx = ((x - lo) * (bins / (hi - lo))).long().clamp_(0, bins - 1)
        idx = idx - (x < edges[idx]).long()
        idx = idx + ((x >= edges[idx + 1]) & (idx != bins - 1)).long()
        counts = torch.bincount(idx, minlength=bins).to(x.dtype)
        centres = (edges[:-1] + edges[1:]) / 2
    w0 = counts.cumsum(0)
    w1 = counts.flip(0).cumsum(0).flip(0)
    m0 = (counts * centres).cumsum(0) / w0
    m1 = (counts * centres).flip(0).cumsum(0).flip(0) / w1
    between = w0[:-1] * w1[1:] * (m0[:-1] - m1[1:]).square()
    return centres[torch.argmax(between)]
