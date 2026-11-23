"""Background subtraction by grey-scale morphology.

Rolling ball (opening by a ball) and top-hat (opening by a flat disk or
square), after Sternberg (1983). Building blocks (structuring elements,
erosion, dilation, opening) are in `optix.background.utils`.
"""

from . import utils
from ._rolling_ball import (
    RollingBall,
    TopHat,
    rolling_ball,
    rolling_ball_halo,
    top_hat,
)

__all__ = [
    "RollingBall",
    "TopHat",
    "rolling_ball",
    "rolling_ball_halo",
    "top_hat",
    "utils",
]
