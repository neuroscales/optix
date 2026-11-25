"""Denoising.

Block matching and collaborative filtering: BM3D for images (Dabov et
al. 2007) and BM4D for volumes (Maggioni et al. 2013), for white
Gaussian noise or, after the generalised Anscombe transform,
Poisson-Gaussian noise; total-variation denoising (Rudin, Osher &
Fatemi 1992). Building blocks (transforms, block matching, the two
filtering stages, the total-variation solver) are in
`optix.denoise.utils`.
"""

from . import utils
from ._bm3d import BM3D, BM4D, bm3d, bm3d_halo, bm4d, bm4d_halo
from ._tv import ROF, rof

__all__ = [
    "BM3D",
    "BM4D",
    "ROF",
    "bm3d",
    "bm3d_halo",
    "bm4d",
    "bm4d_halo",
    "rof",
    "utils",
]
