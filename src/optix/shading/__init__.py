"""Shading: flat field, dark field and per-frame baseline."""

from ._darkflat import DarkFlatCalibration, correct
from ._illumination import (
    IlluminationFunction,
    fit_polynomial,
    gaussian_smooth,
    median_smooth,
    median_window,
    robust_minimum,
)

__all__ = [
    "DarkFlatCalibration",
    "IlluminationFunction",
    "correct",
    "fit_polynomial",
    "gaussian_smooth",
    "median_smooth",
    "median_window",
    "robust_minimum",
]
