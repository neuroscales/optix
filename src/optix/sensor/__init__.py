"""Camera sensor calibration: offset, gain, read noise, noise-model fits,
defective pixels, and variance stabilisation.

Building blocks of these estimators are in `optix.sensor.utils`.
"""

from ..shading._darkflat import DarkFlatCalibration
from . import utils
from ._anscombe import GeneralizedAnscombe, anscombe, inverse_anscombe
from ._defects import DefectivePixels, replace_defective
from ._poisson_gaussian import PoissonGaussian
from ._ptc import PhotonTransferCurve
from ._scmos import SCMOSCalibration

__all__ = [
    "DarkFlatCalibration",
    "DefectivePixels",
    "GeneralizedAnscombe",
    "PhotonTransferCurve",
    "PoissonGaussian",
    "SCMOSCalibration",
    "anscombe",
    "inverse_anscombe",
    "replace_defective",
    "utils",
]
