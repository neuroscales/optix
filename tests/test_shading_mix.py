"""Tests of the Gamma-mixture model (optix.shading.mix)."""

import importlib.util
import pathlib
import warnings

import numpy as np
import pytest
import torch

import optix
from optix.estimates import SensorEstimate
from optix.io import Layout, labeled
from optix.models import (
    random_sensor,
    random_shading,
    simulate_gamma_mixture,
    simulate_sensor,
)
from optix.shading import BaSiC
from optix.shading.mix import GammaMixture

_REFERENCE = pathlib.Path(__file__).parent / "reference"


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    # The tensors are small: thread start-up dominates otherwise.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _prototype():
    path = _REFERENCE / "gamma_mixture_prototype.py"
    spec = importlib.util.spec_from_file_location("_gmm_prototype", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.COMPILE = False
    return module


def _working_dtype():
    return optix.get_config().dtype or torch.get_default_dtype()


def _simulate(
    shape=(30, 24, 24),
    means=(30.0, 250.0),
    shapes=(4.0, 2.0),
    gain=2.0,
    read_sd=2.0,
    seed=0,
    dtype=None,
):
    """Integer data from the sensor, shading and mixture simulators."""
    dtype = dtype or _working_dtype()
    spatial = shape[-2:]
    sensor = random_sensor(
        spatial,
        offset=100.0,
        offset_sd=2.0,
        read_sd=read_sd,
        gain=gain,
        seed=seed,
        dtype=torch.float64,
    )
    shading = random_shading(
        spatial, order=3, strength=0.2, seed=seed, dtype=torch.float64
    )
    draw = simulate_gamma_mixture(
        shape,
        means,
        shapes,
        concentration=(2.0,) * len(means) + (3.0,),
        seed=seed,
        dtype=torch.float64,
    )
    electrons = shading.flat * draw.signal / gain
    data = simulate_sensor(electrons, sensor, seed=seed + 1)
    data = data.round().to(dtype)
    geo = shading.flat.log().mean().exp()
    truth = {
        "offset": sensor.offset,
        "variance": sensor.variance,
        "flat": shading.flat / geo,
        "means": torch.tensor(means, dtype=torch.float64) * geo,
        "shapes": torch.tensor(shapes, dtype=torch.float64),
        "labels": draw.labels,
    }
    return data, truth


def _rel(a, b):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return ((a - b).abs().max() / b.abs().max()).item()


# -- parity with the reference prototype -----------------------------------
# float64: every building block is equivalent to the prototype's local
# copy up to round-off (measured ≲ 1e-13 relative). float32: the
# differences come from summation order (float64 accumulators of the
# per-pixel statistics and of the bound, a different contraction order of
# the DCT normal matrix) and the more accurate Gauss-Hermite weights;
# measured ≈ 1e-7 on the bound and ≈ 1e-6 on v after 30 iterations.
_TOL = {torch.float64: 1e-9, torch.float32: 2e-4}


@pytest.mark.parametrize("noise", ["full", "constant"])
def test_parity_with_prototype(noise):
    gp = _prototype()
    dtype = _working_dtype()
    tol = _TOL[dtype]
    data, _ = _simulate(dtype=dtype)
    niter = 12
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        ref = gp.GammaMixtureModel(data, nclass=2, gain=2.0, noise=noise)
        ref.fit(niter=niter)
    finally:
        torch.set_default_dtype(old)
    with optix.config(compile=noise == "full" and optix.get_config().compile):
        est = GammaMixture(
            nclass=2, gain=2.0, noise=noise, max_iterations=niter
        ).fit(data)
    history = torch.tensor(est.history.objective, dtype=torch.float64)
    expected = torch.tensor(ref.history, dtype=torch.float64)
    assert history.shape == expected.shape
    assert _rel(history, expected) < tol
    sensor = est.estimates["sensor"]
    shading = est.estimates["shading"]
    classes = est.estimates["classification"]
    assert _rel(sensor.offset, ref.param.C) < tol
    assert _rel(sensor.variance, ref.param.V) < 10 * tol
    assert _rel(shading.flat, ref.param.M) < tol
    assert torch.equal(shading.dark, sensor.offset)
    assert (est.theta_ - ref.param.theta).abs().max() < 10 * tol
    assert _rel(classes.parameters[:2, 0], ref.prior.mu) < 10 * tol
    assert _rel(classes.parameters[:2, 1], ref.prior.nu) < 10 * tol
    assert _rel(classes.concentration, ref.prior.alpha) < 10 * tol
    assert _rel(est.concentration_, ref.post.alpha) < 10 * tol
    z, _ = est.classify(data)
    assert (z - ref.Z).abs().max() < 10 * tol


# -- recovery of the ground truth ------------------------------------------
def test_recovery():
    data, truth = _simulate(shape=(48, 24, 24), seed=3)
    # eager: the compiled kernel is covered by the parity test, and a
    # recompilation for this shape would dominate the runtime
    with warnings.catch_warnings(), optix.config(compile=False):
        warnings.simplefilter("error")
        est = GammaMixture(nclass=2, gain=2.0, max_iterations=60).fit(data)
    sensor = est.estimates["sensor"]
    shading = est.estimates["shading"]
    classes = est.estimates["classification"]
    # objective is (almost) monotone: steps that lower it are halved
    objective = torch.tensor(est.history.objective, dtype=torch.float64)
    assert (objective.diff() > -1e-6 * objective.abs().max()).all()
    # offset and variance of an empty pixel: about 30 empty frames each
    assert (sensor.offset - truth["offset"]).abs().mean() < 0.5
    assert abs(sensor.variance.mean() - truth["variance"].mean()) < 0.6
    assert float(sensor.gain) == 2.0
    assert shading.meta["flat_norm"] == "geomean"
    assert abs(shading.flat.log().mean()) < 1e-5
    # 63 coefficients from ~24 tissue values per pixel: the statistical
    # error of log m is about 0.04 per pixel
    error = (shading.flat.log() - truth["flat"].log()).abs()
    assert error.mean() < 0.06 and error.max() < 0.2
    assert _rel(classes.parameters[:2, 0], truth["means"]) < 0.1
    assert (classes.parameters[:2, 1] - truth["shapes"]).abs().max() < 0.6
    assert torch.isnan(classes.parameters[2, 1])
    assert abs(classes.weights.sum().item() - 1) < 1e-6
    z, _ = est.classify(data, per_class=True)
    labels = truth["labels"]
    fg = est.apply(data)
    assert ((fg > 0.5) == (labels < 2)).float().mean() > 0.95
    # bright class: well separated
    bright = labels == 1
    assert (z[1][bright] > 0.5).float().mean() > 0.9


# -- streaming and layouts -----------------------------------------------
class _Lazy:
    """Array-like that records the size of every read."""

    def __init__(self, array):
        self.array = np.asarray(array)
        self.shape = self.array.shape
        self.dtype = self.array.dtype
        self.shapes = []

    def __getitem__(self, index):
        out = self.array[index]
        self.shapes.append(out.shape)
        return out


@pytest.fixture(scope="module")
def small():
    data, _ = _simulate(shape=(16, 24, 24), seed=1)
    return data


def _fit(data, **kw):
    kw.setdefault("max_iterations", 4)
    layout = kw.pop("layout", None)
    return GammaMixture(nclass=2, gain=2.0, **kw).fit(data, layout=layout)


def test_streaming_matches_in_memory(small):
    ref = _fit(small)
    try:
        import zarr

        lazy = zarr.create_array(
            store={},
            shape=small.shape,
            chunks=(4, 12, 12),
            dtype=np.float32,
        )
        lazy[...] = small.cpu().numpy()
    except ImportError:
        lazy = _Lazy(small.cpu().numpy())
    with optix.config(batch=40 * 4):  # four frames per batch
        est = _fit(lazy)
    tol = dict(rtol=1e-5, atol=1e-5)
    for name in ("sensor", "shading", "classification"):
        assert est.estimates[name].allclose(
            ref.estimates[name], equal_nan=True, **tol
        )
    assert np.allclose(est.history.objective, ref.history.objective, 1e-6)
    out = np.zeros(small.shape, dtype=np.float32)
    est.classify(lazy, out=out)
    z, _ = ref.classify(small)
    assert np.allclose(out, z.cpu().numpy(), atol=1e-5)
    # frame batches of four frames; one pass by row tiles (order
    # statistics of the initialisation), here a single tile
    probe = _Lazy(small.cpu().numpy())
    with optix.config(batch=40 * 4):
        _fit(probe, max_iterations=1)
    big = [s for s in probe.shapes if s[0] > 4]
    assert len(big) == 1 and len(probe.shapes) > 4


def test_layout_frames_along_x(small):
    # (z, y, x) storage, pictures of the (y, z) plane indexed by x:
    # `small` is (frames, Y, Z) in the default layout.
    stored = small.permute(2, 1, 0).contiguous()  # (Z, Y, X)
    layout = Layout("zyx", spatial="yz", sample="x")
    est = _fit(stored, layout=layout)
    ref = _fit(small)
    assert est.estimates["shading"].flat.shape == (24, 24)
    assert est.estimates["sensor"].allclose(
        ref.estimates["sensor"], rtol=1e-5, atol=1e-4
    )
    z = est.apply(labeled(stored, layout))
    assert z.shape == stored.shape
    zr = ref.apply(small)
    assert torch.allclose(z, zr.permute(2, 1, 0), atol=1e-4)


def test_groups_are_independent_fits(small):
    other, _ = _simulate(shape=(16, 24, 24), seed=2)
    both = torch.stack([small, other], 1)  # (frames, channel, H, W)
    layout = Layout("tcyx", spatial="yx", group="c")
    est = _fit(both, layout=layout)
    flat = est.estimates["shading"].flat
    assert flat.shape == (2, 24, 24)
    assert est.concentration_.shape == (2, 16, 3)
    assert est.estimates["classification"].parameters.shape == (2, 3, 2)
    for c, data in enumerate((small, other)):
        ref = _fit(data)
        assert torch.allclose(
            flat[c], ref.estimates["shading"].flat, rtol=1e-5
        )
    z, mean = est.classify(both, per_class=True, mean=True)
    assert z.shape == (2, *both.shape) and mean.shape == both.shape
    assert (z.sum(0) <= 1 + 1e-5).all()


# -- initialisation from other estimates ---------------------------------
def test_init_from_estimates(small):
    ref = _fit(small, max_iterations=6)
    sensor = SensorEstimate(
        offset=ref.estimates["sensor"].offset,
        variance=ref.estimates["sensor"].variance,
        gain=torch.tensor(2.0),
    )
    warm = GammaMixture(nclass=2, max_iterations=1).fit(
        small, init=[sensor, ref.estimates["shading"]]
    )
    assert float(warm.estimates["sensor"].gain) == 2.0
    assert torch.allclose(
        warm.estimates["shading"].flat,
        ref.estimates["shading"].flat,
        rtol=0.05,
    )
    # a whole fitted estimator, classes included
    again = GammaMixture(nclass=2, max_iterations=1).fit(small, init=ref)
    params = again.estimates["classification"].parameters[:2]
    expected = ref.estimates["classification"].parameters[:2]
    assert torch.allclose(params, expected, rtol=0.05)
    with pytest.raises(ValueError, match="parameters of shape"):
        GammaMixture(nclass=3, max_iterations=1).fit(small, init=ref)
    # BaSiC gives a mean-normalised flat field; a calibration result
    basic = BaSiC(working_size=None, max_reweight=2).fit(small)
    est = GammaMixture(nclass=2, gain=2.0, max_iterations=2).fit(
        small, init=[basic, SensorEstimate(offset=torch.tensor(100.0))]
    )
    assert abs(est.estimates["shading"].flat.log().mean()) < 1e-5


# -- prediction, outputs, state ------------------------------------------
def test_predict_and_outputs(small):
    est = _fit(small)
    z, _ = est.classify(small)
    zp, up, alpha = est.predict(small, mean=True)
    assert alpha.shape == (16, 3)
    assert torch.allclose(zp, z, atol=0.05)
    assert (up >= 0).all()
    out = np.zeros((2, 5, 24, 24), dtype=np.float32)
    est.predict(small[:5], per_class=True, out=out)
    zc, _, _ = est.predict(small[:5], per_class=True)
    assert np.allclose(out, zc.numpy(), atol=1e-6)
    with pytest.raises(ValueError, match="training data"):
        est.classify(small[:5])
    with pytest.raises(ValueError, match="maps of shape"):
        est.apply(small[:, :8])


def test_save_and_load(small, tmp_path):
    est = _fit(small, max_iterations=2)
    path = tmp_path / "mixture.pt"
    est.save(path)
    new = optix.load(path)
    assert isinstance(new, GammaMixture)
    assert torch.equal(new.concentration_, est.concentration_)
    assert torch.allclose(new.apply(small), est.apply(small))


def test_options():
    with pytest.raises(ValueError, match="noise"):
        GammaMixture(noise="poisson")
    assert GammaMixture(faithful=True).is_faithful
    assert not GammaMixture(lam=2.0).is_faithful
    assert GammaMixture.tags.targets == ("shading", "sensor", "classification")
