# Installation

optix is a pure-Python package built on [PyTorch](https://pytorch.org). It
runs on Python 3.11 or later and needs `numpy` (1.24 or later) and `torch`
(2.3 or later). Nothing else is required: arrays from `zarr`, `h5py`, `dask`
or `numpy.memmap` are read through their ordinary slicing interface, so those
libraries are only needed if *you* use them.

## Install from source

```bash
pip install git+https://github.com/neuroscales/optix
```

To work on the library itself, clone the repository and install it in
editable mode, with the test dependencies:

```bash
git clone https://github.com/neuroscales/optix
cd optix
pip install -e ".[test]"
pytest
```

The `test` extra brings in `pytest`, `pytest-cov` and `zarr`; the tests that
read chunked storage are skipped when `zarr` is missing.

## A GPU, if you have one

optix follows PyTorch: install the build of `torch` that matches your
hardware (see the [PyTorch installation guide](https://pytorch.org/get-started/locally/)),
then ask for the device you want, either per call or for a whole block of code:

```python
import optix

with optix.config(device="cuda"):
    ...  # everything inside runs on the GPU
```

The [Estimators](../concepts/estimators.md#devices-dtypes-and-compilation)
page explains how the device, the floating-point precision and the memory
budget are chosen.

## Check the installation

```python
import optix

print(optix.__version__)
print(optix.get_config())
```

## Defaults from the environment

The process-wide defaults are read once, at import, from `OPTIX_*`
environment variables. They are handy on a cluster, where editing the script
is less convenient than editing the job file:

| Variable | Meaning | Example |
|---|---|---|
| `OPTIX_DTYPE` | working floating-point type | `float64` |
| `OPTIX_DEVICE` | working device | `cuda:1` |
| `OPTIX_MEMORY` | size of one chunk loaded on the device | `2GiB` |
| `OPTIX_BATCH` | frames per chunk (overrides the memory budget) | `64` |
| `OPTIX_COMPILE` | whether kernels go through `torch.compile` | `0` |

An invalid value is ignored with a warning rather than stopping the import.

## Build these docs

```bash
pip install -e ".[doc]"
zensical serve
```

`zensical build` writes the static site to `site/`.
