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
from ._vsnr import (
    VSNRSolution,
    dirac,
    gaussian_stripe,
    line,
    operator_norm,
    vsnr_solve,
)

__all__ = [
    "VSNRSolution",
    "Wavelet",
    "dirac",
    "dwt",
    "dwt2",
    "gaussian_stripe",
    "general_stripe_objective",
    "get_wavelet",
    "idwt",
    "idwt2",
    "lattice_diff",
    "lattice_diff_adjoint",
    "line",
    "max_level",
    "operator_norm",
    "stripe_damping",
    "stripe_offset",
    "vsnr_solve",
    "wavedec2",
    "waverec2",
]
