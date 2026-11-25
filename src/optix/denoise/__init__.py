"""Denoising.

Block matching and collaborative filtering: BM3D for images (Dabov et
al. 2007) and BM4D for volumes (Maggioni et al. 2013), for white
Gaussian noise or, after the generalised Anscombe transform,
Poisson-Gaussian noise; non-local means (Buades, Coll & Morel 2005); ACsN
for sCMOS frames (Mandracchia et al. 2020). Building blocks (transforms,
block matching, the two filtering stages) are in `optix.denoise.utils`.
"""

from . import utils
from ._acsn import ACsN, acsn, acsn_halo
from ._bm3d import BM3D, BM4D, bm3d, bm3d_halo, bm4d, bm4d_halo
from ._nlm import NLMeans, nl_means, nl_means_halo

__all__ = [
    "ACsN",
    "BM3D",
    "BM4D",
    "NLMeans",
    "acsn",
    "acsn_halo",
    "bm3d",
    "bm3d_halo",
    "bm4d",
    "bm4d_halo",
    "nl_means",
    "nl_means_halo",
    "utils",
]
