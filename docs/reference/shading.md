# Shading correction (`optix.shading`)

Estimators of the flat field, dark field and per-frame baseline, and the correction that applies them.

## Estimators

::: optix.shading
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - BaSiC
        - CIDRE
        - DarkFlatCalibration
        - GammaMixture
        - IlluminationFunction

## Transforms

::: optix.shading
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - correct

## Utilities

!!! warning "No stability promise"
    `optix.shading.utils` holds building blocks of the estimators above; they may change without notice.

::: optix.shading.utils
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - fit_polynomial
        - gaussian_smooth
        - median_smooth
        - median_window
        - robust_minimum
