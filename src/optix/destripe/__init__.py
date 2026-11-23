"""Stripe artifact removal.

Building blocks (wavelet transforms, damping, difference operators) are
in `optix.destripe.utils`.
"""

from . import utils
from ._munch import WaveletFFT, wavelet_fft
from ._rottmayer import GeneralStripeRemover, general_stripe_remover

__all__ = [
    "GeneralStripeRemover",
    "WaveletFFT",
    "general_stripe_remover",
    "utils",
    "wavelet_fft",
]
