"""Generate `microscpsf_fixtures.npz` (run in a separate venv).

Reference: MicroscPSF-Py 0.2 (MIT licence; Douglass & Babcock's Python
port of the code of Li, Xue & Blu 2017), `gLZRFocalScan`, which
evaluates the Gibson–Lanni model with the Bessel series of Li et al.
(100 basis functions, 1000 pupil samples):

```text
uv venv -p 3.12 microscpsf-venv
VIRTUAL_ENV=microscpsf-venv uv pip install MicroscPSF-Py==0.2 scipy numpy
microscpsf-venv/bin/python tests/data/make_microscpsf_fixtures.py
```

Settings of Li et al. Sec. 3A: NA 1.4, λ = 0.61 µm, n_s = 1.33,
n_i = n_i* = n_g = n_g* = 1.5, t_i* = 150 µm, t_g = t_g* = 170 µm; the
magnification is set to 10⁸ so that the lateral argument is
k NA r exactly. MicroscPSF scans the immersion thickness
t_i = t_i* + zv; with matched coverslip and immersion this is
Δz = zv + n_i t_s / n_s (Gibson & Lanni Eq. 3). Its intensities are
|∫₀^a …|², a quarter of `gibson_lanni_profile`.

Stored arrays
-------------
radius : (R,) float64
    Radii (µm).
defocus : (Z,) float64
    Δz (µm).
depths : (D,) float64
    Source depths t_s (µm).
intensity : (D, Z, R) float64
    MicroscPSF-Py intensities, not normalised.
"""

import os

import numpy as np
from microscPSF import microscPSF as ms

radius = np.arange(0.0, 3.0, 0.05)
defocus = np.arange(-4.0, 4.0001, 0.1)
depths = np.array([0.4, 2.0, 10.0])
params = dict(ms.m_params)
params.update(
    M=1e8,
    NA=1.4,
    ng0=1.5,
    ng=1.5,
    ni0=1.5,
    ni=1.5,
    ns=1.33,
    ti0=150.0,
    tg=170.0,
    tg0=170.0,
)
out = []
for pz in depths:
    zv = defocus - params["ni"] * pz / params["ns"]
    out.append(
        ms.gLZRFocalScan(params, radius, zv, normalize=False, pz=pz, wvl=0.61)
    )
np.savez_compressed(
    os.path.join(os.path.dirname(__file__), "microscpsf_fixtures.npz"),
    radius=radius,
    defocus=defocus,
    depths=depths,
    intensity=np.stack(out),
)
