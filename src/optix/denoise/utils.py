"""Building blocks of `optix.denoise` (no stability promise).

The stages, transforms, block matching and collaborative filters of
BM3D and BM4D, the noise estimate, profile and quality score of
ACsN, the default intensity range and the memory budget.
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
from ._transforms import (
    bior15_matrix,
    dct_matrix,
    haar_matrix,
    kaiser_window,
    transform_matrix,
)

__all__ = [
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
    "filter_batch",
    "haar_matrix",
    "hard_threshold",
    "highpass",
    "kaiser_window",
    "match_batch",
    "noise_ratio",
    "otf_cutoff",
    "patch_spectra",
    "quality_score",
    "reference_grid",
    "run_stage",
    "sigma_from_highpass",
    "stage_halo",
    "temporal_psnr",
    "transform_matrix",
    "wiener",
]
