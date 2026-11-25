"""Building blocks of `optix.denoise` (no stability promise).

The stages, transforms, block matching and collaborative filters of
BM3D and BM4D, the patch distances of non-local means, the noise estimate,
profile and quality score of ACsN, the default intensity range and the memory
budget; the transform, thresholding functions and PURE system of PURE-LET;
the OTF mask, cost, solver and segmentation of NCS.
"""

from ._acsn import (
    acsn_noise,
    acsn_profile,
    highpass,
    noise_ratio,
    otf_cutoff,
    quality_score,
    sigma_from_highpass,
    temporal_psnr,
)
from ._bm3d import clean_range
from ._engine import (
    Stage,
    aggregate,
    block_match,
    block_overhead,
    bm3d_profile,
    bm4d_profile,
    filter_batch,
    hard_threshold,
    match_batch,
    patch_spectra,
    reference_grid,
    run_stage,
    stage_halo,
    wiener,
)
from ._ncs import (
    NcsResult,
    ncs_gradient,
    ncs_mask,
    ncs_objective,
    ncs_precorrect,
    ncs_solve,
    raised_cosine,
    segment_bounds,
)
from ._nlm import (
    default_h,
    offset_batch,
    patch_distances,
    patch_kernel,
    search_offsets,
)
from ._purelet import PureSums, haar_uwt, let_basis, let_threshold
from ._transforms import (
    bior15_matrix,
    dct_matrix,
    haar_matrix,
    kaiser_window,
    transform_matrix,
)

__all__ = [
    "NcsResult",
    "PureSums",
    "Stage",
    "acsn_noise",
    "acsn_profile",
    "aggregate",
    "bior15_matrix",
    "block_match",
    "block_overhead",
    "bm3d_profile",
    "bm4d_profile",
    "clean_range",
    "dct_matrix",
    "default_h",
    "filter_batch",
    "haar_matrix",
    "haar_uwt",
    "hard_threshold",
    "highpass",
    "kaiser_window",
    "let_basis",
    "let_threshold",
    "match_batch",
    "ncs_gradient",
    "ncs_mask",
    "ncs_objective",
    "ncs_precorrect",
    "ncs_solve",
    "noise_ratio",
    "offset_batch",
    "otf_cutoff",
    "patch_distances",
    "patch_kernel",
    "patch_spectra",
    "quality_score",
    "raised_cosine",
    "reference_grid",
    "run_stage",
    "search_offsets",
    "segment_bounds",
    "sigma_from_highpass",
    "stage_halo",
    "temporal_psnr",
    "transform_matrix",
    "wiener",
]
