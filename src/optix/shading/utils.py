"""Building blocks of the shading estimators.

Semi-public: these may change without notice.
"""

from ._illumination import (
    fit_polynomial,
    gaussian_smooth,
    median_smooth,
    median_window,
    robust_minimum,
)

__all__ = [
    "fit_polynomial",
    "gaussian_smooth",
    "median_smooth",
    "median_window",
    "robust_minimum",
]
