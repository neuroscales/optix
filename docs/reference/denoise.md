# Denoising (`optix.denoise`)

## Estimators

::: optix.denoise
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - BM3D
        - BM4D
        - NCS

## Transforms

::: optix.denoise
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - bm3d
        - bm4d
        - bm3d_halo
        - bm4d_halo
        - ncs
        - ncs_halo

## Utilities

!!! warning "No stability promise"
    `optix.denoise.utils` holds building blocks of the denoisers above; they may change without notice.

::: optix.denoise.utils
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - Stage
        - bm3d_profile
        - bm4d_profile
        - stage_halo
        - run_stage
        - patch_spectra
        - reference_grid
        - block_match
        - hard_threshold
        - wiener
        - aggregate
        - clean_range
        - block_overhead
        - match_batch
        - filter_batch
        - transform_matrix
        - bior15_matrix
        - haar_matrix
        - dct_matrix
        - kaiser_window
        - ncs_mask
        - raised_cosine
        - ncs_precorrect
        - ncs_objective
        - ncs_gradient
        - ncs_solve
        - NcsResult
        - segment_bounds
