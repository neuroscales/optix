"""Generate `pystripe_fixtures.npz` (run in a separate venv).

Reference: pystripe (https://github.com/chunglabmit/pystripe, commit
540e045, MIT licence, Copyright (c) 2018 Chung Lab), with numpy 1.23
(pystripe uses `np.float`), scipy, scikit-image 0.21 and PyWavelets
1.8.0, float64. pystripe is not on PyPI under that name (the `pystripe`
package there is a fork), so its `pystripe/` folder is put on the path:

```text
python3.11 -m venv ref-venv
ref-venv/bin/pip install "numpy<1.24" scipy "scikit-image<0.22" \\
    pywavelets==1.8.0 tifffile tqdm pathlib2 dcimg pillow
# pystripe/{__init__,core,raw,lightsheet_correct}.py from the repository
PYTHONPATH=<dir holding pystripe/> ref-venv/bin/python \\
    tests/data/make_pystripe_fixtures.py
```

`filter_streaks` returns `uint16`; the floating-point result before its
final clip and cast is also recorded, by wrapping the `np.clip` call it
makes.

Stored arrays
-------------
image, image_odd : (48, 40) and (45, 38) float64
    Background near 100 with bright blobs and horizontal stripes
    (numpy only, no optix import).
image_u16 : (48, 40) uint16
    `image` rounded.
flat : (48, 40) float64
    Smooth flat field, maximum 1.
subband_<case> : float64
    `filter_subband(image, sigma, level, wavelet)` (the log-domain
    single-band filter) for `a` = (8, 0, db3), `b` = (4, 3, db4).
streaks_<case>, streaks_<case>_u16
    `filter_streaks` before the clip (float64) and its `uint16` output,
    with the arguments in `args_<case>` (JSON).
otsu_float, otsu_u16
    `skimage.filters.threshold_otsu(image)`, `(image_u16)`.
"""

import json
import os

import numpy as np
import pystripe.core as core
from skimage.filters import threshold_otsu

HERE = os.path.dirname(os.path.abspath(__file__))


def make_image(shape, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: shape[0], : shape[1]].astype(float)
    img = 100 + 20 * np.sin(x / 7 + y / 11)
    for _ in range(4):
        cy, cx = rng.uniform(0, shape[0]), rng.uniform(0, shape[1])
        r2 = (y - cy) ** 2 + (x - cx) ** 2
        img += rng.uniform(300, 1500) * np.exp(-r2 / rng.uniform(10, 40))
    stripes = rng.normal(0, 15, shape[0])[:, None]  # along x (axis -1)
    img += stripes * (1 + img / 500)
    return img + rng.normal(0, 2, shape)


class _RecordClip:
    """Proxy of numpy recording the input of `np.clip`."""

    def __init__(self):
        self.seen = None

    def clip(self, a, *args, **kwargs):
        self.seen = np.array(a, copy=True)
        return np.clip(a, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(np, name)


def streaks(img, **kwargs):
    proxy = _RecordClip()
    core.np = proxy
    try:
        out = core.filter_streaks(img, **kwargs)
    finally:
        core.np = np
    return proxy.seen, out


def main():
    image = make_image((48, 40))
    image_odd = make_image((45, 38), seed=1)
    image_u16 = np.round(image).astype(np.uint16)
    y, x = np.mgrid[:48, :40] / 40.0
    flat = 1 - 0.3 * ((x - 0.5) ** 2 + (y - 0.6) ** 2)
    flat = flat / flat.max()
    out = {
        "image": image,
        "image_odd": image_odd,
        "image_u16": image_u16,
        "flat": flat,
        "otsu_float": np.float64(threshold_otsu(image)),
        "otsu_u16": np.float64(threshold_otsu(image_u16)),
    }
    for case, (sigma, level, wavelet) in {
        "a": (8.0, 0, "db3"),
        "b": (4.0, 3, "db4"),
    }.items():
        out[f"subband_{case}"] = core.filter_subband(
            image, sigma, level, wavelet
        )
    cases = {
        "single": ("image", dict(sigma=[8.0, 8.0])),
        "dual": ("image", dict(sigma=[4.0, 16.0], level=4, threshold=300)),
        "dual_otsu": ("image", dict(sigma=[16.0, 4.0], crossover=50)),
        "foreground": ("image", dict(sigma=[8.0, 0.0], threshold=300)),
        "background": ("image", dict(sigma=[0.0, 8.0], threshold=300)),
        "flat": ("image", dict(sigma=[8.0, 8.0], dark=20.0)),
        "odd": ("image_odd", dict(sigma=[4.0, 12.0], wavelet="db4")),
        "u16": ("image_u16", dict(sigma=[4.0, 16.0], crossover=20)),
    }
    for case, (name, kwargs) in cases.items():
        kwargs = dict(kwargs)
        out[f"args_{case}"] = np.array(json.dumps({"image": name, **kwargs}))
        if case == "flat":
            kwargs["flat"] = flat
        seen, u16 = streaks(out[name], **kwargs)
        out[f"streaks_{case}"] = seen
        out[f"streaks_{case}_u16"] = u16
    np.savez_compressed(os.path.join(HERE, "pystripe_fixtures.npz"), **out)


if __name__ == "__main__":
    main()
