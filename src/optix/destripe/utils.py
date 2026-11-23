"""Building blocks of the stripe filters.

Semi-public: these may change without notice.
"""

from ._dwt import (
    Wavelet,
    dwt,
    dwt2,
    get_wavelet,
    idwt,
    idwt2,
    max_level,
    wavedec2,
    waverec2,
)
from ._munch import stripe_damping
from ._rottmayer import (
    general_stripe_objective,
    lattice_diff,
    lattice_diff_adjoint,
    stripe_offset,
)

__all__ = [
    "Wavelet",
    "dwt",
    "dwt2",
    "general_stripe_objective",
    "get_wavelet",
    "idwt",
    "idwt2",
    "lattice_diff",
    "lattice_diff_adjoint",
    "max_level",
    "stripe_damping",
    "stripe_offset",
    "wavedec2",
    "waverec2",
]
