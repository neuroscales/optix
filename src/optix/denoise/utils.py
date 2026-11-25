"""Building blocks of `optix.denoise` (no stability promise).

The stages, transforms, block matching and collaborative filters of
BM3D and BM4D, the patch distances of non-local means, the default
intensity range and the memory budget.
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
from ._nlm import offset_batch, patch_distances, patch_kernel, search_offsets
from ._transforms import (
    bior15_matrix,
    dct_matrix,
    haar_matrix,
    kaiser_window,
    transform_matrix,
)

__all__ = [
    "Stage",
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
    "offset_batch",
    "patch_distances",
    "patch_kernel",
    "patch_spectra",
    "reference_grid",
    "run_stage",
    "search_offsets",
    "stage_halo",
    "transform_matrix",
    "wiener",
]
