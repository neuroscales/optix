"""Deconvolution with a known point spread function.

Richardson–Lucy for Poisson counts, its acceleration by Biggs & Andrews
(1997), its extension by Snyder et al. (1993) to CCD data (flat field,
background counts, read noise) and its total-variation regularisation by
Dey et al. (2006); multiview deconvolution of registered views by
Preibisch et al. (2014); the linear Wiener and Tikhonov filters; the
regularised shifted-Poisson deconvolution of Ikoma et al. (2018), solved by
ADMM. Spatially variant blur: the PSF models of Nagy & O'Leary (1998) and
Denis et al. (2015), usable as the PSF of the Richardson–Lucy family, and the
preconditioned conjugate-gradient restoration of Nagy & O'Leary.

Building blocks (FFT convolution with padding, the shift-variant
operator, the Richardson–Lucy update, the Biggs–Andrews acceleration
parameter, the back projectors) are in `optix.deconv.utils`.
"""

from . import utils
from ._ba import BiggsAndrews, biggs_andrews
from ._ikoma import Ikoma, ikoma
from ._linear import Tikhonov, Wiener, tikhonov, wiener
from ._multiview import MultiviewRL, multiview_rl
from ._nagy import NagyOLeary, nagy_oleary
from ._rl import RichardsonLucy, richardson_lucy
from ._rltv import RichardsonLucyTV, richardson_lucy_tv
from ._shiftvariant import DenisPSF, NagyOLearyPSF
from ._snyder import Snyder
from ._wb import WienerButterworth, wiener_butterworth

__all__ = [
    "BiggsAndrews",
    "Ikoma",
    "DenisPSF",
    "NagyOLeary",
    "NagyOLearyPSF",
    "MultiviewRL",
    "RichardsonLucy",
    "RichardsonLucyTV",
    "Snyder",
    "Tikhonov",
    "Wiener",
    "WienerButterworth",
    "biggs_andrews",
    "ikoma",
    "nagy_oleary",
    "multiview_rl",
    "richardson_lucy",
    "richardson_lucy_tv",
    "tikhonov",
    "utils",
    "wiener",
    "wiener_butterworth",
]
