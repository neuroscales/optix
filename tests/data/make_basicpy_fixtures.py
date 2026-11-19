"""Generate `basicpy_fixtures.npz` (run in a separate venv with BaSiCPy).

Reference: BaSiCPy 2.0.0 (MIT licence), PyTorch 2.15.0, CPU, float32:

```text
uv venv -p 3.12 basicpy-venv
VIRTUAL_ENV=basicpy-venv uv pip install basicpy==2.0.0
basicpy-venv/bin/python tests/data/make_basicpy_fixtures.py
```

The synthetic stack is built with numpy only (no optix import), at the
working size (64 x 64), so that BaSiCPy's resizing is the identity and the
comparison isolates the solvers. Settings: BaSiCPy defaults
(`epsilon=0.1`, `optimization_tol=1e-3`, `reweighting_tol=1e-2`,
`max_iterations=500`, `max_reweight_iterations=10`, automatic
`smoothness_flatfield` / `smoothness_darkfield`), with
`working_size=64`, `device="cpu"` and, per case, `fitting_mode` and
`get_darkfield`.

Stored arrays
-------------
stack : (40, 64, 64) uint16
    The stack (Poisson counts).
true_flat, true_dark : (64, 64) float32
    Ground truth (flat with mean 1).
true_level : (40,) float32
    Per-frame background level.
<case>_flat, <case>_dark : (64, 64) float32
    BaSiCPy `flatfield` / `darkfield`.
<case>_baseline : (40,) float32
    BaSiCPy `baseline` (from the fit).
<case>_fit_only_baseline : (40,) float32
    BaSiCPy `fit_only_baseline(stack, None, flatfield, darkfield)`.

Cases: `approx` (approximate, no dark field) and `approx_dark`
(approximate, dark field). BaSiCPy 2.0.0's `fitting_mode="ladmap"`
fails on 2-D stacks (shape error in `LadmapFit.calc_weights`), so it is
not used.
"""

import os
import warnings

import numpy as np


def make_stack(seed: int = 0) -> dict[str, np.ndarray]:
    """Shaded, sparse foreground on a varying background, Poisson noise."""
    rng = np.random.default_rng(seed)
    n, h, w = 40, 64, 64
    yy, xx = np.meshgrid(
        (np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij"
    )
    log_flat = 0.35 * np.cos(np.pi * yy) + 0.25 * np.cos(np.pi * xx)
    log_flat += 0.15 * np.cos(np.pi * yy) * np.cos(np.pi * xx)
    flat = np.exp(log_flat)
    flat /= flat.mean()
    dark = 40 + 15 * np.cos(np.pi * xx) + 10 * np.cos(np.pi * yy)
    level = 200 * (1 + 0.3 * np.sin(np.arange(n) / 5))
    signal = np.empty((n, h, w))
    for i in range(n):
        objects = np.zeros((h, w))
        for _ in range(12):
            cy, cx = rng.uniform(0, 1, size=2)
            r = rng.uniform(0.02, 0.06)
            amp = rng.uniform(300, 1500)
            objects += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / r**2)
        signal[i] = level[i] + objects
    mean = flat * signal + dark
    stack = rng.poisson(mean).astype(np.uint16)
    return {
        "stack": stack,
        "true_flat": flat.astype(np.float32),
        "true_dark": dark.astype(np.float32),
        "true_level": level.astype(np.float32),
    }


def run_basicpy(stack: np.ndarray, mode: str, dark: bool) -> dict:
    from basicpy import BaSiC

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BaSiC(
            fitting_mode=mode,
            get_darkfield=dark,
            working_size=64,
            device="cpu",
        )
        images = stack.astype(np.float32)
        model.fit(images)
        fob = model.fit_only_baseline(
            images, None, model.flatfield, model.darkfield
        )
    return {
        "flat": np.asarray(model.flatfield, np.float32),
        "dark": np.asarray(model.darkfield, np.float32),
        "baseline": np.asarray(model.baseline, np.float32),
        "fit_only_baseline": np.asarray(fob.cpu(), np.float32),
    }


def main() -> None:
    from importlib.metadata import version

    import torch

    data = make_stack()
    out = dict(data)
    cases = {
        "approx": ("approximate", False),
        "approx_dark": ("approximate", True),
    }
    for name, (mode, dark) in cases.items():
        for key, value in run_basicpy(data["stack"], mode, dark).items():
            out[f"{name}_{key}"] = value
    out["basicpy_version"] = np.array(version("basicpy"))
    out["torch_version"] = np.array(torch.__version__)
    path = os.path.join(os.path.dirname(__file__), "basicpy_fixtures.npz")
    np.savez_compressed(path, **out)
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
