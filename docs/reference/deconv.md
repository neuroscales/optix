# Deconvolution (`optix.deconv`)

Deconvolution with a known point spread function.

## Estimators

::: optix.deconv
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - RichardsonLucy
        - Snyder
        - WienerButterworth
        - BiggsAndrews

## Transforms

::: optix.deconv
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - richardson_lucy
        - wiener_butterworth
        - biggs_andrews

## Utilities

!!! warning "No stability promise"
    `optix.deconv.utils` holds building blocks of the estimators above; they may change without notice.

::: optix.deconv.utils
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - PaddedConvolution
        - next_fast_len
        - rl_step
        - rl_block
        - biggs_andrews_alpha
        - chi2_probability
        - back_projector
        - kernel_shape
        - BackProjection
        - psf_fwhm
