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

### Overlapping tiles

Integer tile positions for [`GammaMixture`][optix.shading.GammaMixture]
(`overlap=`): voxels seen by two overlapping tiles constrain the flat field.
With `TileOverlap(fit_gain=True)` each tile also gets a gain (reported as
`GammaMixture.tile_gain_`), so that a uniform per-tile fade is not mistaken
for a ramp of the flat field.

::: optix.shading.mix
    options:
      heading_level: 4
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - TileOverlap
        - tile_pairs
        - overlap_slices

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
