"""Shading correction: flat field, dark field and per-frame baseline.

Building blocks of these estimators are in `optix.shading.utils`.
"""

from . import utils
from ._basic import BaSiC
from ._cidre import CIDRE
from ._darkflat import DarkFlatCalibration, correct
from ._illumination import IlluminationFunction
from ._kask import KaskFlatField
from .mix import GammaMixture

__all__ = [
    "BaSiC",
    "CIDRE",
    "DarkFlatCalibration",
    "GammaMixture",
    "IlluminationFunction",
    "KaskFlatField",
    "correct",
    "utils",
]
