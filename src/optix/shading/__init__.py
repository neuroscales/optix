"""Shading correction: flat field, dark field and per-frame baseline."""

from ._basic import BaSiC
from ._darkflat import DarkFlatCalibration, correct
from ._illumination import (
    IlluminationFunction,
    fit_polynomial,
    gaussian_smooth,
    median_smooth,
    median_window,
    robust_minimum,
)
from .mix import GammaMixture

__all__ = [
    "BaSiC",
    "DarkFlatCalibration",
    "GammaMixture",
    "IlluminationFunction",
    "correct",
    "fit_polynomial",
    "gaussian_smooth",
    "median_smooth",
    "median_window",
    "robust_minimum",
]
