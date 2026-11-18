"""Forward models and simulators returning ground-truth estimates.

Samples depend on the seed **and** on how the caller chunks the calls.
"""

from .sensor import random_sensor, simulate_sensor
from .shading import random_shading, simulate_shading

__all__ = [
    "random_sensor",
    "random_shading",
    "simulate_sensor",
    "simulate_shading",
]
