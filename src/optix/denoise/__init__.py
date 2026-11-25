"""Denoising.

Block matching and collaborative filtering: BM3D for images (Dabov et
al. 2007) and BM4D for volumes (Maggioni et al. 2013), for white
Gaussian noise or, after the generalised Anscombe transform,
Poisson-Gaussian noise. Building blocks (transforms, block matching,
the two filtering stages) are in `optix.denoise.utils`.
"""

from . import utils
from ._bm3d import BM3D, BM4D, bm3d, bm3d_halo, bm4d, bm4d_halo

__all__ = ["BM3D", "BM4D", "bm3d", "bm3d_halo", "bm4d", "bm4d_halo", "utils"]
