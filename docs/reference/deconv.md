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
        - BiggsAndrews
        - Wiener
        - Tikhonov

## Transforms

::: optix.deconv
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - richardson_lucy
        - biggs_andrews
        - wiener
        - tikhonov

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
        - transfer_function
        - smooth_periodic_extension
        - frequencies
        - laplacian_power
        - linear_filter
        - rl_step
        - rl_block
        - biggs_andrews_alpha
        - chi2_probability
