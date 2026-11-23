"""Deconvolution with a known point spread function.

Richardson–Lucy for Poisson counts, its acceleration by Biggs & Andrews
(1997), and its extension by Snyder et al. (1993) to CCD data (flat field,
background counts, read noise); the linear Wiener and Tikhonov filters.
Spatially variant blur: the PSF models of Nagy & O'Leary (1998) and Denis
et al. (2015), usable as the PSF of the Richardson–Lucy family, and the
preconditioned conjugate-gradient restoration of Nagy & O'Leary.

Building blocks (FFT convolution with padding, the shift-variant
operator, the Richardson–Lucy update, the Biggs–Andrews acceleration
parameter, the back projectors) are in `optix.deconv.utils`.
"""

from . import utils
from ._ba import BiggsAndrews, biggs_andrews
from ._linear import Tikhonov, Wiener, tikhonov, wiener
from ._nagy import NagyOLeary, nagy_oleary
from ._rl import RichardsonLucy, richardson_lucy
from ._shiftvariant import DenisPSF, NagyOLearyPSF
from ._snyder import Snyder
from ._wb import WienerButterworth, wiener_butterworth

__all__ = [
    "BiggsAndrews",
    "DenisPSF",
    "NagyOLeary",
    "NagyOLearyPSF",
    "RichardsonLucy",
    "Snyder",
    "Tikhonov",
    "Wiener",
    "WienerButterworth",
    "biggs_andrews",
    "nagy_oleary",
    "richardson_lucy",
    "tikhonov",
    "utils",
    "wiener",
    "wiener_butterworth",
]
