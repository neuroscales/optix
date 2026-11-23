"""Generate `skimage_rolling_ball_fixtures.npz` (run in a separate venv).

Reference: scikit-image 0.26.0 (BSD-3-Clause licence),
`skimage.restoration.rolling_ball`, float64:

```text
uv venv -p 3.12 skimage-venv
VIRTUAL_ENV=skimage-venv uv pip install scikit-image==0.26.0
skimage-venv/bin/python tests/data/make_skimage_rolling_ball_fixtures.py
```

scikit-image returns, at each pixel, the height of the apex of the ball
pushed up under the image: the grey-scale erosion by the ball umbra
(shifted so that the centre height is 0), not the opening. The tests
compare it with `optix.background.utils.grey_erosion`, and its dilation
with the opening of `optix.background.rolling_ball`.

Stored arrays
-------------
image2d : (48, 56) float64
    Gaussian spots on a tilted, curved background, plus uniform noise.
ball2d_5, ball2d_7_5 : (48, 56) float64
    scikit-image results for radius 5 and 7.5.
image3d : (14, 16, 18) float64
ball3d_3 : (14, 16, 18) float64
    scikit-image result for radius 3.
"""

import os

import numpy as np
from skimage.restoration import rolling_ball

rng = np.random.default_rng(0)
y, x = np.mgrid[:48, :56]
image2d = 20 + 0.3 * x + 0.01 * (y - 20) ** 2 + 2 * rng.random((48, 56))
for cy, cx in [(10, 12), (30, 40), (24, 20), (40, 8)]:
    image2d += 30 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 4.0)
z, y, x = np.mgrid[:14, :16, :18]
image3d = 5 + 0.2 * x + 0.1 * z + rng.random((14, 16, 18))
image3d += 10 * np.exp(-((z - 7) ** 2 + (y - 8) ** 2 + (x - 9) ** 2) / 2.0)

out = dict(
    image2d=image2d,
    ball2d_5=rolling_ball(image2d, radius=5),
    ball2d_7_5=rolling_ball(image2d, radius=7.5),
    image3d=image3d,
    ball3d_3=rolling_ball(image3d, radius=3),
)
here = os.path.dirname(os.path.abspath(__file__))
np.savez_compressed(
    os.path.join(here, "skimage_rolling_ball_fixtures.npz"), **out
)
