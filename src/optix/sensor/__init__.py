"""Camera sensor calibration: offset, gain, read noise, defective pixels,
and variance stabilisation."""

from ..shading._darkflat import DarkFlatCalibration
from ._anscombe import (
    GeneralizedAnscombe,
    anscombe,
    exact_inverse_table,
    inverse_anscombe,
)
from ._base import PixelStats, pixel_stats
from ._defects import DefectivePixels, bimodality, replace_defective, robust_z
from ._ptc import PhotonTransferCurve
from ._scmos import SCMOSCalibration, gain_regression

__all__ = [
    "DarkFlatCalibration",
    "DefectivePixels",
    "GeneralizedAnscombe",
    "PhotonTransferCurve",
    "PixelStats",
    "SCMOSCalibration",
    "anscombe",
    "bimodality",
    "exact_inverse_table",
    "gain_regression",
    "inverse_anscombe",
    "pixel_stats",
    "replace_defective",
    "robust_z",
]
