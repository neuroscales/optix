"""Building blocks of the attenuation estimators.

Semi-public: these may change without notice.
"""

from ._kernels import (
    compensation_gain,
    smith_attenuation,
    tail_sums,
    vermeer_attenuation,
    vermeer_gain,
    weighted_tv,
)

__all__ = [
    "compensation_gain",
    "smith_attenuation",
    "tail_sums",
    "vermeer_attenuation",
    "vermeer_gain",
    "weighted_tv",
]
