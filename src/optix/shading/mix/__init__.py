"""Mixture models of the sensor, the shading and the tissue classes.

[`GammaMixture`][optix.shading.GammaMixture] models the signal of
each tissue class with a Gamma distribution.
"""

from ._mixture import GammaMixture
from ._overlap import TileOverlap, overlap_slices, tile_pairs

__all__ = ["GammaMixture", "TileOverlap", "overlap_slices", "tile_pairs"]
