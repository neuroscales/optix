"""Point spread functions: measured from beads, or modelled.

- [`BeadPSF`][optix.psf.BeadPSF] / [`bead_psf`][optix.psf.bead_psf]:
  empirical PSF averaged over images of isolated beads.
- [`lightsheet_psf`][optix.psf.lightsheet_psf]: effective PSF of
  light-sheet imaging, a detection PSF times the sheet profile
  ([`gaussian_sheet`][optix.psf.gaussian_sheet]).
- [`gibson_lanni`][optix.psf.gibson_lanni]: scalar PSF model of an
  objective used outside its design conditions (refractive-index and
  thickness mismatch of the specimen, coverslip and immersion layers).
- [`richards_wolf`][optix.psf.richards_wolf]: vectorial PSF of an
  aplanatic system of high aperture (Richards & Wolf, 1959).

Building blocks (smoothing, peak detection, sub-voxel centres and
shifts, FWHM, Gaussian-beam optics) are in `optix.psf.utils`.
"""

from . import utils
from ._beads import BeadPSF, bead_psf
from ._gibson_lanni import gibson_lanni
from ._lightsheet import gaussian_sheet, lightsheet_psf
from ._richards_wolf import richards_wolf

__all__ = [
    # estimators
    "BeadPSF",
    # transforms
    "bead_psf",
    "gaussian_sheet",
    "gibson_lanni",
    "lightsheet_psf",
    "richards_wolf",
    # building blocks
    "utils",
]
