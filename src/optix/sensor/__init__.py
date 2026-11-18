"""Camera sensor calibration: offset, gain, read noise, defective pixels,
and variance stabilisation."""

from ..shading._darkflat import DarkFlatCalibration
from ._anscombe import (
    GeneralizedAnscombe,
    anscombe,
    closed_form_inverse,
    exact_inverse_table,
    inverse_anscombe,
)
from ._base import PixelStats, pixel_stats
from ._defects import DefectivePixels, bimodality, replace_defective, robust_z
from ._ptc import (
    PhotonTransferCurve,
    bspline_slopes,
    emva_highpass,
    linearity_error,
    pair_stats,
)
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
    "bspline_slopes",
    "closed_form_inverse",
    "emva_highpass",
    "exact_inverse_table",
    "gain_regression",
    "inverse_anscombe",
    "linearity_error",
    "pair_stats",
    "pixel_stats",
    "replace_defective",
    "robust_z",
]
