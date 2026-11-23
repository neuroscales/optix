"""Spatial filters on the last two (spatial) dims of a tensor."""

from ._median import median_disk, median_filter

__all__ = ["median_disk", "median_filter"]
