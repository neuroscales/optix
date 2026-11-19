# Sensor calibration (`optix.sensor`)

Estimators of the camera noise model and defective pixels, and the variance-stabilising transforms.

## Estimators

::: optix.sensor
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - PoissonGaussian
        - SCMOSCalibration
        - PhotonTransferCurve
        - DefectivePixels
        - GeneralizedAnscombe

[`DarkFlatCalibration`][optix.shading.DarkFlatCalibration] is also exported here; it is documented under [Shading correction](shading.md).

## Transforms

::: optix.sensor
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - anscombe
        - inverse_anscombe
        - replace_defective

## Utilities

!!! warning "No stability promise"
    `optix.sensor.utils` holds building blocks of the estimators above; they may change without notice.

::: optix.sensor.utils
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - gain_regression
        - pair_stats
        - bspline_slopes
        - emva_highpass
        - linearity_error
        - bimodality
        - robust_z
        - closed_form_inverse
        - exact_inverse_table
        - SegmentStats
        - to_normalized
        - from_normalized
        - sensor_estimate
