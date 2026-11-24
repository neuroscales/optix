"""Point spread functions: measured from beads, or modelled.

- [`BeadPSF`][optix.psf.BeadPSF] / [`bead_psf`][optix.psf.bead_psf]:
  empirical PSF averaged over images of isolated beads.
- [`lightsheet_psf`][optix.psf.lightsheet_psf]: effective PSF of
  light-sheet imaging, a detection PSF times the sheet profile
  ([`gaussian_sheet`][optix.psf.gaussian_sheet]).
- [`PhaseRetrieval`][optix.psf.PhaseRetrieval] /
  [`phase_retrieval`][optix.psf.phase_retrieval]: pupil function
  retrieved from a bead z-stack, with its Zernike expansion (Hanser et
  al. 2004); [`pupil_psf`][optix.psf.pupil_psf] and
  [`zernike_psf`][optix.psf.zernike_psf] render PSFs from it, and
  [`index_mismatch_aberration`][optix.psf.index_mismatch_aberration]
  modifies it for a refractive-index mismatch.
- [`gibson_lanni`][optix.psf.gibson_lanni]: scalar PSF model of an
  objective used outside its design conditions (refractive-index and
  thickness mismatch of the specimen, coverslip and immersion layers).

Building blocks (smoothing, peak detection, sub-voxel centres and
shifts, FWHM, Gaussian-beam optics) are in `optix.psf.utils`.
"""

from . import utils
from ._beads import BeadPSF, bead_psf
from ._gibson_lanni import gibson_lanni
from ._lightsheet import gaussian_sheet, lightsheet_psf
from ._pupil import (
    PhaseRetrieval,
    index_mismatch_aberration,
    phase_retrieval,
    pupil_psf,
    zernike_psf,
)

__all__ = [
    # estimators
    "BeadPSF",
    "PhaseRetrieval",
    # transforms
    "bead_psf",
    "gaussian_sheet",
    "gibson_lanni",
    "lightsheet_psf",
    "phase_retrieval",
    "pupil_psf",
    "zernike_psf",
    "index_mismatch_aberration",
    # building blocks
    "utils",
]
