"""Shading correction: flat field, dark field and per-frame baseline.

Building blocks of these estimators are in `optix.shading.utils`.
"""

from . import utils
from ._basic import BaSiC
from ._darkflat import DarkFlatCalibration, correct
from ._illumination import IlluminationFunction
from .mix import GammaMixture

__all__ = [
    "BaSiC",
    "DarkFlatCalibration",
    "GammaMixture",
    "IlluminationFunction",
    "correct",
    "utils",
]
