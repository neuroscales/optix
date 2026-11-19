"""Generate `skimage_rl_fixtures.npz` (run in a separate venv).

Reference: scikit-image 0.26.0 (BSD-3-Clause licence),
`skimage.restoration.richardson_lucy`, float64:

```text
uv venv -p 3.12 skimage-venv
VIRTUAL_ENV=skimage-venv uv pip install scikit-image==0.26.0
skimage-venv/bin/python tests/data/make_skimage_rl_fixtures.py
```

scikit-image models the image on the image grid with zero boundaries
and does not divide by Hᵀ1. On an image whose objects lie at least the
PSF size away from the border (zero counts there), both models give the
same iterates, so the comparison isolates the update. Settings:
`num_iter` 10 and 30, `clip=False`, `filter_epsilon=None`.

Stored arrays
-------------
image : (64, 64) float64
    Poisson counts of the blurred objects (zero within 16 px of the
    border).
psf : (7, 7) float64
    Gaussian PSF (σ = 1.2 px), unit sum.
rl10, rl30 : (64, 64) float64
    scikit-image results after 10 and 30 iterations.
"""

import os

import numpy as np
from scipy.signal import convolve
from skimage.restoration import richardson_lucy

rng = np.random.default_rng(0)
x = np.arange(7) - 3
g = np.exp(-(x**2) / (2 * 1.2**2))
psf = np.outer(g, g)
psf /= psf.sum()
truth = np.zeros((64, 64))
truth[20:44, 20:44] = rng.uniform(0, 1, (24, 24)) < 0.15
truth[20:44, 20:44] *= 400.0
truth[24:40, 30:34] += 50.0
blurred = convolve(truth, psf, mode="same", method="direct")
image = rng.poisson(blurred).astype(np.float64)
assert not image[:16].any() and not image[-16:].any()
assert not image[:, :16].any() and not image[:, -16:].any()
out = {"image": image, "psf": psf}
for n in (10, 30):
    out[f"rl{n}"] = richardson_lucy(
        image, psf, num_iter=n, clip=False, filter_epsilon=None
    )
path = os.path.join(os.path.dirname(__file__), "skimage_rl_fixtures.npz")
np.savez_compressed(path, **out)
