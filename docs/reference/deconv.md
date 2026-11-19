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

## Transforms

::: optix.deconv
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - richardson_lucy

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
        - chi2_probability
