# Background subtraction (`optix.background`)

## Estimators

::: optix.background
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - RollingBall
        - TopHat

## Transforms

::: optix.background
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - rolling_ball
        - top_hat
        - rolling_ball_halo

## Utilities

!!! warning "No stability promise"
    `optix.background.utils` holds building blocks of the filters above; they may change without notice.

::: optix.background.utils
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - ball
        - footprint
        - grey_erosion
        - grey_dilation
        - grey_opening
