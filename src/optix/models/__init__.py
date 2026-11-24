"""Forward models and simulators returning ground-truth estimates.

Samples depend on the seed **and** on how the caller chunks the calls.
"""

from . import phantoms
from .blur import gaussian_psf, simulate_blur
from .mixture import GammaMixtureSample, simulate_gamma_mixture
from .sensor import random_sensor, simulate_sensor
from .shading import random_shading, simulate_shading
from .tiles import simulate_tiles

__all__ = [
    "GammaMixtureSample",
    "gaussian_psf",
    "phantoms",
    "random_sensor",
    "random_shading",
    "simulate_blur",
    "simulate_gamma_mixture",
    "simulate_sensor",
    "simulate_shading",
    "simulate_tiles",
]
