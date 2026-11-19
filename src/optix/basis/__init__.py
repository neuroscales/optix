"""Smooth bases (orthonormal DCT-II) and their regularisers."""

from ._dct import dct_basis, dct_bending, gram, project, reconstruct

__all__ = ["dct_basis", "dct_bending", "gram", "project", "reconstruct"]
