"""Deconvolution with a known point spread function.

Richardson–Lucy for Poisson counts, its acceleration by Biggs & Andrews
(1997), its extension by Snyder et al. (1993) to CCD data (flat field,
background counts, read noise) and its total-variation regularisation by
Dey et al. (2006); the linear Wiener and Tikhonov filters.

Building blocks (FFT convolution with padding, the Richardson–Lucy update,
the Biggs–Andrews acceleration parameter, the back projectors) are in
`optix.deconv.utils`.
"""

from . import utils
from ._ba import BiggsAndrews, biggs_andrews
from ._linear import Tikhonov, Wiener, tikhonov, wiener
from ._rl import RichardsonLucy, richardson_lucy
from ._rltv import RichardsonLucyTV, richardson_lucy_tv
from ._snyder import Snyder
from ._wb import WienerButterworth, wiener_butterworth

__all__ = [
    "BiggsAndrews",
    "RichardsonLucy",
    "RichardsonLucyTV",
    "Snyder",
    "Tikhonov",
    "Wiener",
    "WienerButterworth",
    "biggs_andrews",
    "richardson_lucy",
    "richardson_lucy_tv",
    "tikhonov",
    "utils",
    "wiener",
    "wiener_butterworth",
]
