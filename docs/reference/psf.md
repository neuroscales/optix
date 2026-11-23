# Point spread functions (`optix.psf`)

## Estimators

::: optix.psf
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - BeadPSF
        - PhaseRetrieval

## Transforms

::: optix.psf
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - bead_psf
        - lightsheet_psf
        - gaussian_sheet
        - phase_retrieval
        - pupil_psf
        - zernike_psf
        - index_mismatch_aberration
        - gibson_lanni

## Utilities

!!! warning "No stability promise"
    `optix.psf.utils` holds building blocks of the functions above; they may change without notice.

::: optix.psf.utils
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - gaussian_filter
        - local_maxima
        - fit_gaussian
        - threshold_centroid
        - fourier_shift
        - linear_shift
        - tukey_window
        - fwhm
        - rayleigh_range
        - beam_radius
        - sheet_intensity
        - zernike
        - noll_index
        - zernike_fit
        - unwrap_phase
        - pupil_coordinates
        - zernike_pupil
        - vectorial_factors
        - bead_otf
        - fit_otf_rescaling
        - gibson_lanni_opd
        - gibson_lanni_profile
