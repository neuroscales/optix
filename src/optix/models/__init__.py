"""Forward models and simulators returning ground-truth estimates.

Samples depend on the seed **and** on how the caller chunks the calls.
"""

from .mixture import GammaMixtureSample, simulate_gamma_mixture
from .sensor import random_sensor, simulate_sensor
from .shading import random_shading, simulate_shading
from .tiles import simulate_tiles

__all__ = [
    "GammaMixtureSample",
    "random_sensor",
    "random_shading",
    "simulate_gamma_mixture",
    "simulate_sensor",
    "simulate_shading",
    "simulate_tiles",
]
