"""Building blocks of `optix.denoise` (no stability promise).

The stages, transforms, block matching and collaborative filters of
BM3D and BM4D.
"""

from ._engine import (
    Stage,
    aggregate,
    block_match,
    bm3d_profile,
    bm4d_profile,
    hard_threshold,
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
    "aggregate",
    "bior15_matrix",
    "block_match",
    "bm3d_profile",
    "bm4d_profile",
    "dct_matrix",
    "haar_matrix",
    "hard_threshold",
    "kaiser_window",
    "patch_spectra",
    "reference_grid",
    "run_stage",
    "stage_halo",
    "transform_matrix",
    "wiener",
]
