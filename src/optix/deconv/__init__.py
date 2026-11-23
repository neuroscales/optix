"""Deconvolution with a known point spread function.

Richardson–Lucy for Poisson counts, and its extension by Snyder et al.
(1993) to CCD data (flat field, background counts, read noise).

Building blocks (FFT convolution with padding, the Richardson–Lucy update)
are in `optix.deconv.utils`.
"""

from . import utils
from ._rl import RichardsonLucy, richardson_lucy
from ._snyder import Snyder

__all__ = ["RichardsonLucy", "Snyder", "richardson_lucy", "utils"]
