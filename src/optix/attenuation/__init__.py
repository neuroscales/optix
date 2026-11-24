"""Attenuation along the light-sheet propagation axis.

Building blocks (tail sums, the per-method kernels and the weighted TV
solver) are in `optix.attenuation.utils`.
"""

from . import utils
from ._depth import DepthResolved, compensate, depth_resolved

__all__ = ["DepthResolved", "compensate", "depth_resolved", "utils"]
