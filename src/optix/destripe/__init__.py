"""Stripe artifact removal.

Building blocks (wavelet transforms, damping) are in
`optix.destripe.utils`.
"""

from . import utils
from ._munch import WaveletFFT, wavelet_fft

__all__ = ["WaveletFFT", "utils", "wavelet_fft"]
