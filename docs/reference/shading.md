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
        - Likar
        - KaskFlatField

### Overlapping tiles

Integer tile positions for [`GammaMixture`][optix.shading.GammaMixture]
(`overlap=`). A scene voxel seen by several overlapping tiles has one class
and one signal shared by all of them; its contribution to the bound
integrates the product of the tile likelihoods over that signal, so the
same structure seen at different places of the field of view constrains
the flat field. With `TileOverlap(fit_gain=True)` each tile t also gets a
gain g_t (flat field g_t·f, reported as `GammaMixture.tile_gain_`), so
that a uniform per-tile fade is not mistaken for a ramp of the flat field;
`apply(..., tile_gain=True)` divides tile t by it, and
`classify(..., joint=True)` gives every observation of a shared voxel its
joint posterior.

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
