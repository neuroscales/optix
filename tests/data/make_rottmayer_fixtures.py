"""Generate `rottmayer_fixtures.npz` (run in a separate venv).

Reference: the Python code of Rottmayer et al. (2025),
[General-Stripe-Removal](https://github.com/NiklasRottmayer/General-Stripe-Removal)
(MIT licence), file `Python-Stripe-Removal/GeneralStripeRemover.py` at commit
f1972aa47fc3ecf6580984f72ce7ce630b1ea683 (the file was last changed in
7838aa99cc4ea1c6d44e0bd5256303f1a59c79d0, "Improved GSR", 29 July 2025),
PyTorch 2.15.0, CPU, float64:

```text
SHA=f1972aa47fc3ecf6580984f72ce7ce630b1ea683
curl -O https://raw.githubusercontent.com/NiklasRottmayer/General-Stripe-Removal/$SHA/Python-Stripe-Removal/GeneralStripeRemover.py
uv venv -p 3.12 gsr-venv
VIRTUAL_ENV=gsr-venv uv pip install torch==2.15.0 numpy
gsr-venv/bin/python -I tests/data/make_rottmayer_fixtures.py \\
    GeneralStripeRemover.py
```

The authors' function runs with `torch.set_default_dtype(torch.float64)`
(its dual buffers use the default dtype), `GPU=False` and
`verbose=False`. The inputs are built with numpy only (no optix import).
The authors' arrays are `(nx, ny)` or `(nx, ny, nz)`, z last, with the
stripes along `nx` for `direction=[1, 0, 0]`; optix's frames are `(H, W)`
with the stripes along `H` (axis -2) and its volumes `(Z, H, W)`, so the
volumes are stored in optix order, `(Z, H, W)`.

Stored arrays
-------------
frame : (40, 48) float64
    Piecewise-smooth content plus vertical stripes, in [0, 1].
oblique : (40, 48) float64
    The same content plus stripes at about 26.6° (offset (2, 1)), in [0, 1].
volume : (6, 24, 20) float64
    `(Z, H, W)`, smooth 3-D content plus stripes along H, in [0, 1].
<case>_input : str
    Name of the input array of the case.
<case>_u : float64
    The authors' result `u`, in optix order.
<case>_mu1, <case>_mu2, <case>_iterations, <case>_proj, <case>_resz,
<case>_normalize, <case>_direction, <case>_scale, <case>_offset
    Settings of the case. The authors' function gets
    `scale * input + offset`.

Cases: see `CASES`.
"""

import importlib.util
import os
import sys

import numpy as np

SHA = "f1972aa47fc3ecf6580984f72ce7ce630b1ea683"

# name: (input, mu, iterations, proj, resz, normalize, direction,
#        scale, offset)
CASES = {
    # optix's faithful defaults: μ = (1/3, 1/300), 25000 iterations
    "default": ("frame", (1 / 3, 1 / 300), 25000, True, 0, False,
                (1, 0, 0), 1.0, 0.0),
    # the authors' default iterations, and their normalisation
    "recommended": ("frame", (1 / 3, 1 / 300), 1000, True, 0, True,
                    (1, 0, 0), 300.0, 20.0),
    "noproj": ("frame", (0.17, 0.003), 1000, False, 0, False,
               (1, 0, 0), 1.0, 0.0),
    "oblique": ("oblique", (0.5, 0.017), 1000, True, 0, False,
                (2, 1, 0), 1.0, 0.0),
    "diag_neg": ("frame", (0.33, 0.003), 1000, True, 0, False,
                 (1, -1, 0), 1.0, 0.0),
    "steep": ("frame", (0.4, 0.007), 1000, True, 0, False,
              (1, 2, 0), 1.0, 0.0),
    # stack processed slice by slice (resz = 0) and as a volume (resz = 1)
    "stack": ("volume", (1 / 3, 1 / 300), 1000, True, 0, False,
              (1, 0, 0), 1.0, 0.0),
    "volume": ("volume", (1 / 3, 1 / 300), 1000, True, 1, False,
               (1, 0, 0), 1.0, 0.0),
}  # fmt: skip


def unit(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())


def make_inputs(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = 40, 48
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    disc = (y - 14.0) ** 2 + (x - 20.0) ** 2 < 8.0**2
    box = (x > 30) & (x < 42) & (y > 22) & (y < 34)
    content = 0.25 + 0.01 * y + 0.4 * disc + 0.25 * box
    # vertical stripes of random length and amplitude
    stripes = np.zeros((h, w))
    for j in rng.choice(w, 10, replace=False):
        start = rng.integers(0, h // 2)
        stripes[start : start + rng.integers(h // 3, h), j] = rng.uniform(
            -0.3, 0.3
        )
    noise = 0.01 * rng.standard_normal((h, w))
    frame = unit(content + stripes + noise)
    # stripes along the lattice offset (2, 1)
    oblique = np.zeros((h, w))
    for j0 in rng.choice(np.arange(-h // 2, w), 10, replace=False):
        amp = rng.uniform(-0.3, 0.3)
        for i in range(h):
            j = j0 + i // 2
            if 0 <= j < w:
                oblique[i, j] += amp
    oblique = unit(content + oblique + noise)
    # (Z, H, W) volume
    nz, nh, nw = 6, 24, 20
    z, y, x = np.meshgrid(
        np.arange(nz), np.arange(nh), np.arange(nw), indexing="ij"
    )
    ball = (z - 2.5) ** 2 + (y - 11.0) ** 2 + (x - 9.0) ** 2 < 6.0**2
    vol = 0.2 + 0.02 * x + 0.5 * ball
    vs = np.zeros((nz, nh, nw))
    for _ in range(12):
        k, j = rng.integers(0, nz), rng.integers(0, nw)
        vs[k, rng.integers(0, nh // 2) :, j] = rng.uniform(-0.3, 0.3)
    volume = unit(vol + vs + 0.01 * rng.standard_normal((nz, nh, nw)))
    return {"frame": frame, "oblique": oblique, "volume": volume}


def load(path: str):
    spec = importlib.util.spec_from_file_location("gsr_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GeneralStripeRemover


def main(path: str) -> None:
    from importlib.metadata import version

    import torch

    torch.set_default_dtype(torch.float64)
    gsr = load(path)
    inputs = make_inputs()
    out: dict[str, np.ndarray] = dict(inputs)
    for name, case in CASES.items():
        src, mu, iters, proj, resz, norm, direction, scale, offset = case
        data = scale * inputs[src] + offset
        if data.ndim == 3:  # (Z, H, W) -> (nx, ny, nz)
            data = data.transpose(1, 2, 0)
        direction = list(direction[: data.ndim]) if resz == 0 else direction
        u = gsr(
            torch.as_tensor(np.ascontiguousarray(data)),
            iterations=iters,
            mu=list(mu),
            proj=proj,
            resz=resz,
            normalize=norm,
            direction=np.array(direction, dtype=float),
            GPU=False,
            verbose=False,
        ).numpy()
        if u.ndim == 3:
            u = u.transpose(2, 0, 1)
        out[f"{name}_input"] = np.array(src)
        out[f"{name}_u"] = u
        out[f"{name}_mu1"] = np.array(mu[0])
        out[f"{name}_mu2"] = np.array(mu[1])
        out[f"{name}_iterations"] = np.array(iters)
        out[f"{name}_proj"] = np.array(proj)
        out[f"{name}_resz"] = np.array(float(resz))
        out[f"{name}_normalize"] = np.array(norm)
        out[f"{name}_direction"] = np.array(direction, dtype=float)
        out[f"{name}_scale"] = np.array(scale)
        out[f"{name}_offset"] = np.array(offset)
        print(name, "done")
    out["gsr_commit"] = np.array(SHA)
    out["torch_version"] = np.array(torch.__version__)
    out["numpy_version"] = np.array(version("numpy"))
    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "rottmayer_fixtures.npz")
    np.savez_compressed(target, **out)
    print(f"wrote {target} ({os.path.getsize(target)} bytes)")


if __name__ == "__main__":
    main(sys.argv[1])
