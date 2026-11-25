"""Block matching and collaborative filtering (BM3D / BM4D engine).

Written from Dabov et al. (2007), doi:10.1109/TIP.2007.901238, and
Maggioni et al. (2013), doi:10.1109/TIP.2012.2210725. The authors' code
(non-commercial licence) was not used.

The engine works on one image (2-D) or volume (3-D) `x` in the units of
the paper, with blocks (cubes) of `n` samples per axis flattened to
`K = nᵈ` values in row-major order. "Positions" are the top-left
corners of the blocks: a grid of `P = shape − n + 1` per axis.
"""

import itertools
import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .._config import get_config
from ._transforms import haar_matrix, kaiser_window, transform_matrix

__all__ = [
    "Stage",
    "aggregate",
    "block_match",
    "block_overhead",
    "bm3d_profile",
    "bm4d_profile",
    "filter_batch",
    "hard_threshold",
    "match_batch",
    "patch_spectra",
    "reference_grid",
    "run_stage",
    "stage_halo",
    "wiener",
]


@dataclass(frozen=True)
class Stage:
    """Parameters of one stage of BM3D or BM4D.

    Attributes
    ----------
    transform : {"bior1.5", "dct"}
        Block transform T₂D (T₃D for cubes), separable.
    size : int
        Block side N₁ (cube side L).
    group : int
        Maximum group size N₂ (M).
    step : int
        Step N_step between reference blocks, per axis.
    search : int
        Side N_S of the search window (cube), centred on the reference.
    tau : float
        Matching threshold τ_match on the normalised distance
        d = ‖·‖²/K (paper units).
    lam : float or None
        Hard threshold λ₃D (λ₄D), in units of σ; None for the Wiener stage.
    lam2d : float
        Pre-filter threshold λ₂D of the matching distance (0: none).
    beta : float or None
        Kaiser window parameter β of the aggregation (None: no window).
    keep_dc : bool
        Never threshold the DC coefficient of the group spectrum.
    strict : bool
        Matching with d < τ (Wiener stage) instead of d ≤ τ.
    pixel_distance : bool
        Hard-thresholding stage: match on the blocks' samples,
        d = ‖Z_{x_R} − Z_x‖²/K, instead of their spectra (ACsN,
        Mandracchia et al. 2020, Supplementary Eq. (12)). Same as the
        spectra for an orthonormal transform.
    sigma_weight : bool
        Hard-thresholding weight 1/(σ²·N_har) (BM3D, Eq. (10)) or
        1/N_har (ACsN, Supplementary Eq. (15)); both are 1 when
        N_har = 0.
    """

    transform: str
    size: int
    group: int
    step: int
    search: int
    tau: float
    lam: float | None
    lam2d: float = 0.0
    beta: float | None = None
    keep_dc: bool = False
    strict: bool = False
    pixel_distance: bool = False
    sigma_weight: bool = True


def bm3d_profile(sigma: float, profile: str = "normal") -> tuple[Stage, Stage]:
    """Stages of the BM3D "normal" profile (Dabov et al. 2007, Table I).

    Parameters
    ----------
    sigma : float
        Noise standard deviation in 8-bit units ([0, 255]); σ > 40
        selects the high-noise column.
    profile : {"normal"}

    Returns
    -------
    (Stage, Stage)
        Hard-thresholding and Wiener stages.
    """
    if profile != "normal":
        raise ValueError(f"BM3D profile must be 'normal', got {profile!r}")
    if sigma <= 40:
        ht = Stage("bior1.5", 8, 16, 3, 39, 2500.0, 2.7, 0.0, 2.0)
        wie = Stage("dct", 8, 32, 3, 39, 400.0, None, 0.0, 2.0, strict=True)
    else:
        ht = Stage("dct", 12, 16, 4, 39, 5000.0, 2.8, 2.0, 2.0)
        wie = Stage("dct", 11, 32, 6, 39, 3500.0, None, 0.0, 2.0, strict=True)
    return ht, wie


def bm4d_profile(profile: str = "normal") -> tuple[Stage, Stage]:
    """Stages of BM4D (Maggioni et al. 2013, Table I), data in [0, 1].

    The table gives τ_match as the sum of squared differences of two
    cubes, ‖C₁ − C₂‖²; it is stored here divided by L³, the normalised
    distance d of Eq. (2).

    Parameters
    ----------
    profile : {"normal", "modified"}

    Returns
    -------
    (Stage, Stage)
        Hard-thresholding and Wiener stages.
    """
    if profile == "normal":
        ht = Stage("bior1.5", 4, 16, 3, 11, 2.9 / 64, 2.7, keep_dc=True)
        wie = Stage("dct", 4, 32, 3, 11, 0.4 / 64, None, strict=True)
    elif profile == "modified":
        ht = Stage("bior1.5", 4, 32, 3, 11, 24.6 / 64, 2.8, keep_dc=True)
        wie = Stage("dct", 5, 32, 3, 11, 6.7 / 125, None, strict=True)
    else:
        raise ValueError(
            f"BM4D profile must be 'normal' or 'modified', got {profile!r}"
        )
    return ht, wie


def override(stage: Stage, **changes: object) -> Stage:
    """Copy of `stage` with the non-None `changes`."""
    return replace(
        stage, **{k: v for k, v in changes.items() if v is not None}
    )


# -- memory ------------------------------------------------------------------
# Share of the memory budget (`get_config().memory`) of each phase of a
# stage: the spectra of all blocks (sized by the streaming block, see
# `SPECTRA_COPIES`), then, one after the other, a batch of block matching
# and a batch of collaborative filtering.
SPECTRA_SHARE = 0.5
BATCH_SHARE = 0.25
# Values per pixel held while the spectra are computed: the spectra of
# the noisy data and of the pilot (or pre-filtered) data, plus one
# transform temporary.
SPECTRA_COPIES = 3


def block_overhead(k: int) -> float:
    """Values per sample of a streamed block, for the memory budget.

    The spectra of a stage take `SPECTRA_COPIES`·K values per sample,
    plus about eight image-sized buffers (data, estimates, aggregation
    buffers); the block is sized so that they fit in `SPECTRA_SHARE` of
    the budget, which leaves `BATCH_SHARE` to each batch.

    Parameters
    ----------
    k : int
        Values per block K.

    Returns
    -------
    float
        `overhead` of [`blocks`][optix.io.blocks].
    """
    return (SPECTRA_COPIES * k + 8) / SPECTRA_SHARE


def match_batch(
    candidates: int, k: int, itemsize: int, memory: int | None = None
) -> int:
    """References per batch of the block matching of a stage.

    A batch of b references holds about b·N_c·(2K·e + 40) bytes: the
    gathered candidate spectra and their differences (K values of e bytes
    each per candidate), plus the indices, masks and distances of the N_c
    candidates. It is sized to `BATCH_SHARE` of the budget.

    Parameters
    ----------
    candidates : int
        Candidates per reference N_c (N_Sᵈ).
    k : int
        Values per block K.
    itemsize : int
        Bytes per value e.
    memory : int, optional
        Budget in bytes (default: `get_config().memory`).

    Returns
    -------
    int
    """
    memory = get_config().memory if memory is None else memory
    per_ref = candidates * (2 * k * itemsize + 40)
    return max(1, int(BATCH_SHARE * memory) // per_ref)


def filter_batch(
    group: int, k: int, itemsize: int, memory: int | None = None
) -> int:
    """Groups per batch of the collaborative filtering of a stage.

    A batch of b groups of m blocks holds about b·m·K·(8e + 8) bytes: up
    to eight temporaries of the size of the group (gathered spectra,
    transforms, shrinkage, block estimates, weights) and the int64
    aggregation indices. It is sized to `BATCH_SHARE` of the budget.

    Parameters
    ----------
    group : int
        Blocks per group m.
    k : int
        Values per block K.
    itemsize : int
        Bytes per value e.
    memory : int, optional
        Budget in bytes (default: `get_config().memory`).

    Returns
    -------
    int
    """
    memory = get_config().memory if memory is None else memory
    per_group = group * k * (8 * itemsize + 8)
    return max(1, int(BATCH_SHARE * memory) // per_group)


def stage_halo(stage: Stage) -> int:
    """Reach of a stage: output samples depend on inputs within this.

    A sample is covered by blocks at positions m within n − 1 of it,
    whose groups belong to references within ⌊N_S/2⌋ of m and read
    blocks within ⌊N_S/2⌋ of the reference: 2·⌊N_S/2⌋ + n − 1.
    """
    return 2 * (stage.search // 2) + stage.size - 1


# -- transforms --------------------------------------------------------------
def _sep(x: Tensor, mat: Tensor, ndim: int) -> Tensor:
    """Apply `mat` along each of the last `ndim` axes of `x`."""
    if ndim == 2:
        return torch.einsum("...ij,ai,bj->...ab", x, mat, mat)
    if ndim == 3:
        return torch.einsum("...ijk,ai,bj,ck->...abc", x, mat, mat, mat)
    raise ValueError(f"ndim must be 2 or 3, got {ndim}")


def patch_spectra(x: Tensor, mat: Tensor) -> Tensor:
    """Transform of every block of `x` (all positions).

    Parameters
    ----------
    x : (*shape) Tensor
        Image or volume (2 or 3 dims).
    mat : (n, n) Tensor
        1-D transform, applied along every axis of the blocks.

    Returns
    -------
    (∏P, nᵈ) Tensor
        Spectra in row-major order of the positions P = shape − n + 1.
    """
    n = mat.shape[0]
    p = x
    for axis in range(x.ndim):
        p = p.unfold(axis, n, 1)
    return _sep(p, mat, x.ndim).reshape(-1, n**x.ndim)


def reference_grid(
    extent: int, size: int, step: int, origin: int = 0
) -> Tensor:
    """Reference positions along one axis.

    Positions p in [0, extent − size] with (p + origin) a multiple of
    `step`, plus the last position extent − size, so that every sample
    is covered.

    Parameters
    ----------
    extent, size, step : int
    origin : int
        Global coordinate of sample 0 (blocks of a larger image).

    Returns
    -------
    Tensor of int64
    """
    last = extent - size
    first = (-origin) % step
    grid = list(range(first, last + 1, step))
    if not grid or grid[-1] != last:
        grid.append(last)
    return torch.tensor(grid, dtype=torch.long)


# -- grouping ----------------------------------------------------------------
def _offsets(search: int, ndim: int, device: torch.device) -> Tensor:
    lo = -((search - 1) // 2)
    hi = search // 2
    rng = range(lo, hi + 1)
    return torch.tensor(
        list(itertools.product(rng, repeat=ndim)),
        dtype=torch.long,
        device=device,
    )


def block_match(
    spectra: Tensor,
    positions: tuple[int, ...],
    refs: Tensor,
    stage: Stage,
    *,
    batch: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Group the blocks most similar to each reference.

    For each reference x_R, the candidates are the positions x of its
    N_Sᵈ search window (clipped to the image), with the distance
    d(x_R, x) = ‖S(x_R) − S(x)‖²/K between their spectra (Dabov et al.
    2007, Eqs. (4), (7); Maggioni et al. 2013, Eqs. (2), (9)). The group
    is made of the matched blocks (d ≤ τ, or d < τ for the Wiener
    stage) sorted by distance, the reference first; their number is
    capped at N₂ and rounded down to a power of two.

    Parameters
    ----------
    spectra : (∏P, K) Tensor
        Spectra used for the distance (pre-filtered if λ₂D > 0).
    positions : tuple of int
        Grid of positions P.
    refs : (R, d) Tensor of int64
        Reference positions.
    stage : Stage
    batch : int, optional
        References per batch (default:
        [`match_batch`][optix.denoise.utils.match_batch]).

    Returns
    -------
    index : (R, N₂) Tensor of int64
        Flat positions of the group members, best first (entries past
        the group size are padding).
    size : (R,) Tensor of int64
        Group sizes (powers of two).
    """
    device = spectra.device
    ndim = len(positions)
    shape = torch.tensor(positions, device=device)
    stride = torch.tensor(
        [math.prod(positions[i + 1 :]) for i in range(ndim)], device=device
    )
    delta = _offsets(stage.search, ndim, device)
    centre = int(((delta == 0).all(-1)).nonzero()[0, 0])
    nc, k = delta.shape[0], spectra.shape[1]
    if batch is None:
        batch = match_batch(nc, k, spectra.element_size())
    keep = min(stage.group, nc)
    refs = refs.to(device)
    index = torch.empty(len(refs), keep, dtype=torch.long, device=device)
    size = torch.empty(len(refs), dtype=torch.long, device=device)
    for start in range(0, len(refs), batch):
        r = refs[start : start + batch]
        c = r[:, None, :] + delta
        valid = ((c >= 0) & (c < shape)).all(-1)
        flat = (c.clamp(min=0).minimum(shape - 1) * stride).sum(-1)
        rflat = (r * stride).sum(-1)
        dist = (spectra[flat] - spectra[rflat][:, None]).square_().sum(-1)
        dist /= k
        dist.masked_fill_(~valid, math.inf)
        matched = dist < stage.tau if stage.strict else dist <= stage.tau
        matched[:, centre] = True
        dist[:, centre] = -1.0  # the reference comes first
        order = torch.sort(dist, dim=1, stable=True).indices[:, :keep]
        count = (matched & valid).sum(1).clamp(max=stage.group)
        index[start : start + len(r)] = flat.gather(1, order)
        size[start : start + len(r)] = (
            2 ** torch.floor(torch.log2(count.double())).long()
        )
    return index, size


# -- collaborative filtering -------------------------------------------------
def hard_threshold(
    spectrum: Tensor,
    threshold: float,
    sigma: float,
    keep_dc: bool,
    sigma_weight: bool = True,
) -> tuple[Tensor, Tensor]:
    """Collaborative hard thresholding of group spectra.

    Υ(c) = c·[|c| > λσ] (Dabov et al. 2007, Eq. (6)), with weight
    w = 1/(σ²·N_har) for N_har ≥ 1 retained coefficients, else 1
    (Eq. (10)); w = 1/N_har without `sigma_weight` (Mandracchia et al.
    2020, Supplementary Eq. (15)).

    !!! note
        The weight 1 of a group with N_har = 0 does not scale with the
        data, unlike 1/(σ²·N_har): in 8-bit units and σ = 20 it is at
        least 400 times any other weight, so such a group (all of its
        coefficients, the DC included, below λσ) dominates the samples
        it covers, where its estimate is 0. It occurs only without
        `keep_dc`, in groups whose mean is close to 0.

    Parameters
    ----------
    spectrum : (B, m, K) Tensor
        3-D (4-D) spectra of B groups of m blocks, DC at [0, 0].
    threshold : float
        λ₃D (λ₄D).
    sigma : float
    keep_dc : bool
        Never threshold the DC (BM4D, Maggioni et al. 2013, Section
        II-B).
    sigma_weight : bool
        Include σ² in the weight.

    Returns
    -------
    spectrum : (B, m, K) Tensor
    weight : (B,) Tensor
    """
    keep = spectrum.abs() > threshold * sigma
    if keep_dc:
        keep[:, 0, 0] = True
    nhar = keep.sum((1, 2))
    scale = sigma * sigma if sigma_weight else 1.0
    weight = torch.where(
        nhar >= 1,
        1.0 / (scale * nhar.clamp(min=1).to(spectrum.dtype)),
        torch.ones((), dtype=spectrum.dtype, device=spectrum.device),
    )
    return spectrum * keep, weight


def wiener(
    noisy: Tensor, pilot: Tensor, sigma: float
) -> tuple[Tensor, Tensor]:
    """Collaborative empirical Wiener filtering of group spectra.

    W = |T(Ŷ)|²/(|T(Ŷ)|² + σ²) from the pilot (basic) estimate, applied
    to the noisy spectrum, with weight w = σ⁻²·‖W‖₂⁻² (Dabov et al.
    2007, Eqs. (8), (9), (11)).

    Parameters
    ----------
    noisy, pilot : (B, m, K) Tensor
    sigma : float

    Returns
    -------
    spectrum : (B, m, K) Tensor
    weight : (B,) Tensor
    """
    p2 = pilot.square()
    w = p2 / (p2 + sigma * sigma)
    norm = w.square().sum((1, 2)).clamp(min=torch.finfo(w.dtype).tiny)
    return w * noisy, 1.0 / (sigma * sigma * norm)


def aggregate(
    num: Tensor,
    den: Tensor,
    blocks: Tensor,
    corners: Tensor,
    offsets: Tensor,
    weight: Tensor,
    window: Tensor | None,
) -> None:
    """Add weighted block estimates to the aggregation buffers.

    num(x) += Σ w·κ·Ŷ(x), den(x) += Σ w·κ·χ(x) (Dabov et al. 2007,
    Eq. (12)), κ the Kaiser window (or 1).

    !!! note
        `index_add_` is deterministic on the CPU; on CUDA its atomic
        additions make the sums order-dependent in the last bits.

    Parameters
    ----------
    num, den : (N,) Tensor
        Flattened buffers, updated in place.
    blocks : (B, m, K) Tensor
        Block estimates in the spatial domain.
    corners : (B, m) Tensor of int64
        Flat sample index of each block's first sample.
    offsets : (K,) Tensor of int64
        Flat sample offsets of the block's samples.
    weight : (B,) Tensor
    window : (K,) Tensor or None
    """
    w = weight[:, None, None].expand(blocks.shape)
    if window is not None:
        w = w * window
    idx = (corners[..., None] + offsets).reshape(-1)
    num.index_add_(0, idx, (w * blocks).reshape(-1))
    den.index_add_(0, idx, w.reshape(-1))


def run_stage(
    noisy: Tensor,
    sigma: float,
    stage: Stage,
    *,
    pilot: Tensor | None = None,
    origin: tuple[int, ...] | None = None,
) -> Tensor:
    """One stage: grouping, collaborative filtering and aggregation.

    Without `pilot`, the hard-thresholding stage (step 1) on `noisy`;
    with it, the Wiener stage (step 2) grouping in and filtering with
    the pilot (basic) estimate.

    Parameters
    ----------
    noisy : (*shape) Tensor
        Image or volume, paper units.
    sigma : float
        Noise standard deviation, paper units.
    stage : Stage
    pilot : (*shape) Tensor, optional
        Basic estimate (Wiener stage).
    origin : tuple of int, optional
        Global coordinates of the first sample (for the reference grid
        of blocks of a larger image).

    Returns
    -------
    (*shape) Tensor
        Samples covered by no block keep their input value (`noisy`, or
        `pilot` in the Wiener stage).
    """
    ndim = noisy.ndim
    n = stage.size
    shape = tuple(noisy.shape)
    if any(s < n for s in shape):
        raise ValueError(f"Blocks of size {n} do not fit in {shape}")
    origin = (0,) * ndim if origin is None else origin
    dtype, device = noisy.dtype, noisy.device
    positions = tuple(s - n + 1 for s in shape)
    mat = transform_matrix(stage.transform, n, dtype=torch.float64)
    inv = torch.linalg.inv(mat).to(dtype=dtype, device=device)
    mat = mat.to(dtype=dtype, device=device)

    spectra = patch_spectra(noisy, mat)
    if pilot is None:
        dist = spectra
        if stage.pixel_distance:
            eye = torch.eye(n, dtype=dtype, device=device)
            dist = patch_spectra(noisy, eye)
        elif stage.lam2d > 0:
            dist = spectra * (spectra.abs() > stage.lam2d * sigma)
    else:
        dist = patch_spectra(pilot, mat)

    axes = [
        reference_grid(s, n, stage.step, o)
        for s, o in zip(shape, origin, strict=True)
    ]
    refs = torch.cartesian_prod(*axes).reshape(-1, ndim)
    index, size = block_match(dist, positions, refs, stage)

    # position grid → sample index of the block's first sample
    pstride = [math.prod(positions[i + 1 :]) for i in range(ndim)]
    sstride = [math.prod(shape[i + 1 :]) for i in range(ndim)]

    def corners(flat: Tensor) -> Tensor:
        out = torch.zeros_like(flat)
        for ps, ss in zip(pstride, sstride, strict=True):
            out += (flat // ps) * ss
            flat = flat % ps
        return out

    offsets = torch.zeros((), dtype=torch.long, device=device)
    for ss in sstride:
        offsets = offsets[..., None] + torch.arange(n, device=device) * ss
    offsets = offsets.reshape(-1)
    window = (
        None
        if stage.beta is None
        else kaiser_window(n, stage.beta, ndim, dtype=dtype, device=device)
    )

    num = torch.zeros(noisy.numel(), dtype=dtype, device=device)
    den = torch.zeros_like(num)
    block = (n,) * ndim
    for m in torch.unique(size).tolist():
        sel = (size == m).nonzero()[:, 0]
        haar = haar_matrix(m, dtype=dtype, device=device)
        bmax = filter_batch(m, n**ndim, spectra.element_size())
        for start in range(0, len(sel), bmax):
            members = index[sel[start : start + bmax], :m]
            group = torch.einsum("ij,bjk->bik", haar, spectra[members])
            if pilot is None:
                group, weight = hard_threshold(
                    group, stage.lam, sigma, stage.keep_dc, stage.sigma_weight
                )
            else:
                ref = torch.einsum("ij,bjk->bik", haar, dist[members])
                group, weight = wiener(group, ref, sigma)
            group = torch.einsum("ji,bjk->bik", haar, group)
            est = _sep(group.reshape(*group.shape[:2], *block), inv, ndim)
            aggregate(
                num,
                den,
                est.reshape(group.shape),
                corners(members),
                offsets,
                weight,
                window,
            )
    fallback = noisy if pilot is None else pilot
    out = torch.where(den > 0, num / den.clamp(min=torch.finfo(dtype).tiny), 0)
    out = torch.where(den > 0, out, fallback.reshape(-1))
    return out.reshape(shape)
