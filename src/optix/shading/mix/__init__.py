"""Mixture models of the sensor, the shading and the tissue classes.

[`GammaMixture`][optix.shading.GammaMixture] models the signal of
each tissue class with a Gamma distribution.
"""

from ._mixture import GammaMixture

__all__ = ["GammaMixture"]
