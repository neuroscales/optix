"""Camera sensor calibration: offset, gain, read noise, noise-model fits,
defective pixels, and variance stabilisation."""

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
from ._foi import (
    SegmentStats,
    clipped_normal_moments,
    from_normalized,
    group_frames,
    integer_range,
    sensor_estimate,
    to_normalized,
)
from ._poisson_gaussian import PoissonGaussian
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
    "PoissonGaussian",
    "SCMOSCalibration",
    "SegmentStats",
    "anscombe",
    "bimodality",
    "bspline_slopes",
    "clipped_normal_moments",
    "closed_form_inverse",
    "emva_highpass",
    "exact_inverse_table",
    "from_normalized",
    "gain_regression",
    "group_frames",
    "integer_range",
    "inverse_anscombe",
    "linearity_error",
    "pair_stats",
    "pixel_stats",
    "replace_defective",
    "robust_z",
    "sensor_estimate",
    "to_normalized",
]
