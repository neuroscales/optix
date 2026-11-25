"""Building blocks of `optix.denoise` (no stability promise).

The stages, transforms, block matching and collaborative filters of
BM3D and BM4D, the default intensity range and the memory budget; the
total-variation solver of ROF.
"""

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
from ._tv import TVResult, tv_denoise

__all__ = [
    "Stage",
    "TVResult",
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
    "kaiser_window",
    "match_batch",
    "patch_spectra",
    "reference_grid",
    "run_stage",
    "stage_halo",
    "transform_matrix",
    "tv_denoise",
    "wiener",
]
