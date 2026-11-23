"""Building blocks of `optix.background` (no stability promise).

Structuring elements and grey-scale erosion, dilation and opening
(Sternberg 1983).
"""

from ._morphology import (
    ball,
    footprint,
    grey_dilation,
    grey_erosion,
    grey_opening,
)

__all__ = [
    "ball",
    "footprint",
    "grey_dilation",
    "grey_erosion",
    "grey_opening",
]
