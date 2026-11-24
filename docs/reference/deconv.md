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
        - RichardsonLucyTV
        - Snyder
        - WienerButterworth
        - BiggsAndrews
        - NagyOLeary
        - MultiviewRL
        - Wiener
        - Tikhonov
        - Ikoma
        - HSPIRAL

## Shift-variant PSF models

::: optix.deconv
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - NagyOLearyPSF
        - DenisPSF

## Transforms

::: optix.deconv
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - richardson_lucy
        - richardson_lucy_tv
        - wiener_butterworth
        - biggs_andrews
        - nagy_oleary
        - multiview_rl
        - wiener
        - tikhonov
        - ikoma
        - hspiral

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
        - ShiftVariantConvolution
        - ShiftVariantPSF
        - interpolation_weights
        - object_positions
        - optimal_circulant
        - RegionPreconditioner
        - cgls
        - next_fast_len
        - transfer_function
        - smooth_periodic_extension
        - frequencies
        - laplacian_power
        - linear_filter
        - rl_step
        - rl_block
        - tv_curvature
        - biggs_andrews_alpha
        - chi2_probability
        - back_projector
        - kernel_shape
        - BackProjection
        - psf_fwhm
        - Multiview
        - compound_kernels
