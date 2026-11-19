# Stripe removal (`optix.destripe`)

## Estimators

::: optix.destripe
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - WaveletFFT

## Transforms

::: optix.destripe
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - wavelet_fft

## Utilities

!!! warning "No stability promise"
    `optix.destripe.utils` holds building blocks of the filters above; they may change without notice.

::: optix.destripe.utils
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - get_wavelet
        - Wavelet
        - dwt
        - idwt
        - dwt2
        - idwt2
        - wavedec2
        - waverec2
        - max_level
        - stripe_damping
