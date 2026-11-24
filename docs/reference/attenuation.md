# Attenuation (`optix.attenuation`)

## Estimators

::: optix.attenuation
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - DepthResolved

## Transforms

::: optix.attenuation
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - depth_resolved
        - compensate

## Utilities

!!! warning "No stability promise"
    `optix.attenuation.utils` holds building blocks of the estimators above; they may change without notice.

::: optix.attenuation.utils
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - tail_sums
        - vermeer_attenuation
        - vermeer_gain
        - compensation_gain
        - smith_attenuation
        - weighted_tv
