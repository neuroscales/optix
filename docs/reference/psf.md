# Point spread functions (`optix.psf`)

## Estimators

::: optix.psf
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - BeadPSF

## Transforms

::: optix.psf
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - bead_psf

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
