"""Streaming, mergeable statistics and per-pixel order statistics."""

from ._base import Reducer, reduce, reduce_groups
from ._histogram import Histogram
from ._minmax import MinMax
from ._moments import Moments
from ._order import median_mad, order_statistics, tile_reduce
from ._pixel import PixelStats, pixel_stats
from ._reservoir import Reservoir

__all__ = [
    "Histogram",
    "MinMax",
    "PixelStats",
    "Moments",
    "Reducer",
    "Reservoir",
    "median_mad",
    "order_statistics",
    "pixel_stats",
    "reduce",
    "reduce_groups",
    "tile_reduce",
]
