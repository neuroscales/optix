"""Tests of the Gamma-mixture model (optix.shading.mix)."""

import importlib.util
import math
import pathlib
import warnings

import numpy as np
import pytest
import torch

import optix
from optix.estimates import SensorEstimate, ShadingEstimate
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
    assert classes.parameters.shape == (2, 2)
    assert abs(classes.weights.sum().item() - 1) < 1e-6
    z, _ = est.classify(data, per_class=True)
    labels = truth["labels"]
    fg = est.apply(data, output="foreground")
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
    init = kw.pop("init", None)
    return GammaMixture(nclass=2, gain=2.0, **kw).fit(
        data, layout=layout, init=init
    )


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
    assert est.estimates["classification"].parameters.shape == (2, 2, 2)
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


# -- more parity cases: halving, early stop, one class, continuous data,
# degenerate spread
def _parity_case(data, *, niter, compile=False, **kw):
    """Fit the port and the prototype with the same options; compare."""
    gp = _prototype()
    dtype = _working_dtype()
    ctor = {"nclass", "gain", "noise", "quantization"}
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        ref = gp.GammaMixtureModel(
            data, **{k: v for k, v in kw.items() if k in ctor}
        )
        ref.fit(
            niter=niter,
            classify=False,
            **{k: v for k, v in kw.items() if k not in ctor},
        )
    finally:
        torch.set_default_dtype(old)
    with optix.config(compile=compile and optix.get_config().compile):
        est = GammaMixture(max_iterations=niter, **kw).fit(data)
    tol = _TOL[dtype]
    history = torch.tensor(est.history.objective, dtype=torch.float64)
    expected = torch.tensor(ref.history, dtype=torch.float64)
    assert history.shape == expected.shape
    assert _rel(history, expected) < tol
    sensor = est.estimates["sensor"]
    classes = est.estimates["classification"]
    nclass = kw.get("nclass", 1)
    assert classes.parameters.shape == (nclass, 2)
    assert _rel(sensor.offset, ref.param.C) < tol
    assert _rel(sensor.variance, ref.param.V) < 10 * tol
    assert _rel(est.estimates["shading"].flat, ref.param.M) < tol
    assert _rel(classes.parameters[:, 0], ref.prior.mu) < 10 * tol
    assert _rel(classes.parameters[:, 1], ref.prior.nu) < 10 * tol
    assert _rel(classes.concentration, ref.prior.alpha) < 10 * tol
    return est


def test_parity_step_halving():
    # an overshooting step (lr = 2.5) forces the halving branch
    data, _ = _simulate(shape=(16, 24, 24))
    est = _parity_case(data, niter=8, nclass=2, gain=2.0, lr=2.5)
    assert sum(est.history.extras["halvings"]) > 0


def test_parity_early_stop():
    data, _ = _simulate(shape=(16, 24, 24))
    est = _parity_case(data, niter=40, nclass=2, gain=2.0, tol=1e-3)
    assert est.history.converged
    assert len(est.history.objective) < 40


def test_parity_one_class():
    data, _ = _simulate(shape=(16, 24, 24), means=(80.0,), shapes=(2.0,))
    _parity_case(data, niter=6, nclass=1, gain=2.0)


def test_parity_continuous_data():
    data, _ = _simulate(shape=(16, 24, 24))
    gen = torch.Generator().manual_seed(0)
    data = data + torch.rand(data.shape, generator=gen, dtype=data.dtype)
    _parity_case(data, niter=6, nclass=2, gain=2.0, quantization=0.0)


def test_parity_degenerate_spread():
    # Every pixel holds the same values in a different order: the lower
    # order statistics are all equal (zero initial spread), and the
    # fallback scale (median absolute deviation) is the same whether it
    # is taken per pixel or over tiles.
    gen = torch.Generator().manual_seed(4)
    nframes, npix = 30, 24 * 24
    with torch.random.fork_rng():
        torch.manual_seed(4)
        tissue = torch.distributions.Gamma(2.0, 2.0 / 80.0).sample((18,))
    values = torch.cat([torch.zeros(12), tissue.round().clamp_min(1)])
    order = torch.rand(npix, nframes, generator=gen).argsort(-1)
    data = (100 + values[order]).T.reshape(nframes, 24, 24)
    data = data.to(_working_dtype())
    _parity_case(data, niter=6, nclass=2, gain=2.0)


# -- porting fixes -------------------------------------------------------
def _peak_bytes(fn):
    """Peak of the bytes allocated by `fn()` (CPU allocator, exact)."""
    from torch.profiler import ProfilerActivity, profile
    from torch.profiler._memory_profiler import Action

    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        fn()
    current = peak = 0
    for _, action, _, size in prof._memory_profile().timeline:
        if action == Action.CREATE:
            current += size
        elif action == Action.DESTROY:
            current -= size
        peak = max(peak, current)
    return peak


@pytest.mark.parametrize("nclass", [1, 3])
def test_local_step_memory_within_budget(nclass, monkeypatch):
    # Frame batches are sized from the declared number of chunk-sized
    # temporaries: the local step must not hold more (no float64 copy of
    # the statistics, a budget that grows with the number of classes).
    from optix.shading.mix import _model
    from optix.shading.mix._gamma import GammaComponent

    seen = {}
    frame_batches = _model.frame_batches

    def spy(*args, **kwargs):
        seen["overhead"] = kwargs["overhead"]
        return frame_batches(*args, **kwargs)

    monkeypatch.setattr(_model, "frame_batches", spy)
    n, height, width = 4, 64, 64
    gen = torch.Generator().manual_seed(0)
    data = (100 + 3 * torch.randn(n, height, width, generator=gen)).round()
    backend = dict(dtype=torch.float32, device=torch.device("cpu"))
    with optix.config(compile=False), torch.no_grad():
        fit = _model.MixtureFit(
            data,
            Layout.default(3, 2),
            component=GammaComponent(24, (1.0, 1e3)),
            nclass=nclass,
            noise="full",
            order=(8, 8),
            gain=2.0,
            quantization=1.0,
            alpha=torch.ones(nclass + 1),
            params=torch.ones(nclass, 2),
            v_prior=None,
            v_dof=5.0,
            lam=1.0,
            gain_sd=0.1,
            backend=backend,
        )
        fit.param.C.fill_(100.0)
        fit.param.V.fill_(9.0)
        fit.prior.params = torch.tensor(
            [[50.0 * (k + 1), 2.0] for k in range(nclass)]
        )
        fit.frames()
        elogpi = fit._elogpi(torch.ones(n, nclass + 1))
        peak = _peak_bytes(lambda: fit._latent_chunk(data, elogpi))
    assert peak / (data.numel() * 4) <= seen["overhead"]


def test_init_from_invalid_estimates(small):
    ref = _fit(small, max_iterations=2)
    sensor = ref.estimates["sensor"]
    variance = sensor.variance.clone()
    variance[0, 0] = 0
    variance[1, 1] = math.nan
    bad = SensorEstimate(offset=sensor.offset, variance=variance)
    with pytest.warns(RuntimeWarning, match="variance"):
        est = _fit(small, max_iterations=2, init=bad)
    assert np.isfinite(est.history.objective).all()
    assert est.estimates["sensor"].variance.isfinite().all()
    assert est.estimates["shading"].flat.isfinite().all()
    flat = torch.ones(24, 24, dtype=small.dtype)
    flat[3, 3] = 0
    with pytest.raises(ValueError, match="flat"):
        _fit(small, max_iterations=1, init=ShadingEstimate(flat=flat))


def test_non_finite_bound_stops(small):
    data = small.clone()
    data[3, 5, 5] = math.nan
    with pytest.warns(RuntimeWarning, match="not finite"):
        est = _fit(data, max_iterations=5)
    assert not est.history.converged
    assert "not finite" in est.history.reason
    assert len(est.history.objective) < 5


def test_fixed_flat_is_used_exactly(small):
    # A sharp vignette that the DCT basis cannot represent, held fixed
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 24, dtype=torch.float64),
        torch.linspace(-1, 1, 24, dtype=torch.float64),
        indexing="ij",
    )
    flat = (-0.5 * (yy.square() + xx.square()) / 0.3**2).exp() + 0.2
    flat[5, 7] = 3.0
    expected = flat / flat.log().mean().exp()
    shading = ShadingEstimate(flat=(2 * flat).to(small.dtype))
    # lr = 2.5 forces step halvings
    est = _fit(
        small,
        max_iterations=6,
        learn_modulation=False,
        lr=2.5,
        init=shading,
    )
    assert sum(est.history.extras["halvings"]) > 0
    out = est.estimates["shading"].flat.to(torch.float64)
    assert torch.allclose(out, expected, rtol=1e-5)


def test_accumulators_follow_state_dtype(small, monkeypatch):
    # Devices without float64 (MPS) accumulate in float32, as optix.stats
    # does: emulate one on the CPU and check that no float64 tensor is
    # created by the fit.
    from torch.utils._python_dispatch import (
        TorchDispatchMode,
        _disable_current_modes,
    )

    from optix.shading.mix import _model

    if _working_dtype() != torch.float32:
        pytest.skip("needs a float32 working dtype")
    from optix.distributions import _stats

    for module in (_model, _stats):
        monkeypatch.setattr(
            module, "state_dtype", lambda device: torch.float32, raising=False
        )
    quantile = torch.quantile

    def outside(*args, **kwargs):  # torch's own float64 internals
        with _disable_current_modes():
            return quantile(*args, **kwargs)

    monkeypatch.setattr(torch, "quantile", outside)

    class NoFloat64(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in torch.utils._pytree.tree_leaves(out):
                if isinstance(t, torch.Tensor) and t.dtype == torch.float64:
                    raise AssertionError(f"float64 result of {func}")
            return out

    est = GammaMixture(nclass=2, gain=2.0, max_iterations=3)
    with optix.config(compile=False):
        layout = Layout.default(3, 2)
        model = est._model(
            small,
            layout,
            dict(dtype=torch.float32, device=torch.device("cpu")),
            alpha=torch.ones(3),
            gain=2.0,
        )
        with NoFloat64():
            model.init()
            model.run(
                niter=3,
                tol=1e-6,
                learn_modulation=True,
                learn_alpha=True,
                lr=1.0,
                max_halvings=4,
                max_log_step=0.5,
            )
    assert np.isfinite(model.history).all()


# -- maintainer decisions ------------------------------------------------
def test_apply_outputs(small):
    est = _fit(small)
    sensor = est.estimates["sensor"]
    flat = est.estimates["shading"].flat
    corrected = est.apply(small)
    assert torch.allclose(
        corrected, (small - sensor.offset) / flat, rtol=1e-5, atol=1e-4
    )
    zp, up, _ = est.predict(small, mean=True)
    zc, _, _ = est.predict(small, per_class=True)
    fg = est.apply(small, output="foreground")
    assert torch.allclose(fg, zp, atol=1e-6)
    classes = est.apply(small, output="classes")
    assert classes.shape == (2, *small.shape)
    assert torch.allclose(classes, zc, atol=1e-6)
    mean = est.apply(small, output="mean")
    assert torch.allclose(mean, up, rtol=1e-5, atol=1e-4)
    out = np.zeros(small.shape, dtype=np.float32)
    est.apply(small, out, output="foreground")
    assert np.allclose(out, zp.numpy(), atol=1e-6)
    with pytest.raises(ValueError, match="output"):
        est.apply(small, output="probabilities")


def test_parameters_hold_tissue_classes_only(small):
    est = _fit(small, max_iterations=1)
    classes = est.estimates["classification"]
    assert classes.parameters.shape == (2, 2)
    assert classes.parameters.isfinite().all()
    assert classes.weights.shape == classes.concentration.shape == (3,)
    with pytest.raises(ValueError, match="parameters of shape"):
        GammaMixture(nclass=3, max_iterations=1).fit(small, init=est)


def test_detect_quantization():
    from optix.shading.mix._model import detect_quantization

    layout = Layout.default(3, 2)
    gen = torch.Generator().manual_seed(0)
    counts = torch.poisson(torch.full((20, 24, 24), 50.0), generator=gen)
    # integer dtype: one ADU
    assert detect_quantization(counts.to(torch.int16), layout) == 1.0
    assert detect_quantization(counts.numpy().astype(np.uint16), layout) == 1
    # integer values stored as floats, with an offset and a scale
    for scale in (1.0, 0.5, 0.1, 3.0):
        data = (counts * scale + 7.25).to(torch.float32)
        delta = detect_quantization(data, layout)
        assert abs(delta - scale) < 1e-4 * scale, (scale, delta)
    # continuous data
    smooth = counts + torch.rand(counts.shape, generator=gen)
    assert detect_quantization(smooth, layout) == 0.0
    jitter = torch.rand(counts.shape, generator=gen, dtype=torch.float64)
    assert detect_quantization(counts.double() + jitter, layout) == 0.0
    # a few frame batches are read, not the whole array
    probe = _Lazy((counts * 0.5).numpy())
    with optix.config(batch=2):
        assert detect_quantization(probe, layout) == 0.5
    assert sum(s[0] for s in probe.shapes) <= 8


def test_quantization_auto_and_warning(small):
    half = small * 0.5
    auto = _fit(half, quantization="auto", max_iterations=2)
    fixed = _fit(half, quantization=0.5, max_iterations=2)
    assert np.allclose(auto.history.objective, fixed.history.objective)
    assert not auto.is_faithful
    assert GammaMixture().options["quantization"] == 1.0
    noisy = small + 0.3
    noisy[::2] += 0.2
    with pytest.warns(UserWarning, match="integer"):
        _fit(noisy, max_iterations=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _fit(small, max_iterations=1)
        _fit(small.to(torch.int32), max_iterations=1)
        _fit(noisy, max_iterations=1, quantization=0.0)


def _quiet(seed=0):
    gen = torch.Generator().manual_seed(seed)
    labels = torch.rand(16, 24, 24, generator=gen) > 0.7
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        signal = torch.distributions.Gamma(2.0, 0.025).sample((16, 24, 24))
    noise = 0.2 * torch.randn(16, 24, 24, generator=gen)
    data = (100 + noise + labels * signal).round()
    return data.to(_working_dtype()), labels


def test_quiet_data_warning_and_spread_floor():
    # Read noise well below one ADU: most empty pixels repeat one value,
    # and the initial spread of the background is zero.
    data, labels = _quiet()
    with pytest.warns(RuntimeWarning, match="quantisation variance"):
        GammaMixture(gain=1.0, max_iterations=2).fit(data)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        floored = GammaMixture(
            gain=1.0, max_iterations=10, spread_floor=True
        ).fit(data)
    assert not GammaMixture(spread_floor=True).is_faithful
    fg = floored.apply(data, output="foreground") > 0.5
    assert abs(fg.float().mean() - labels.float().mean()) < 0.05


# -- overlapping tiles -------------------------------------------------------
from optix.shading.mix import (  # noqa: E402
    TileOverlap,
    overlap_slices,
    tile_pairs,
)
from optix.shading.mix._overlap import Overlap  # noqa: E402


@pytest.mark.parametrize(
    "shift",
    [
        (0, 0),
        (3, 0),
        (-3, 2),
        (0, -7),
        (5, 5),
        (8, 0),
        (-8, 1),
        (9, -9),
        (7, -7),
    ],
)
def test_overlap_slices_brute_force(shift):
    shape = (8, 7)
    got = overlap_slices(shift, shape)
    pairs = [
        (p, tuple(a - s for a, s in zip(p, shift, strict=True)))
        for p in np.ndindex(*shape)
    ]
    pairs = [
        (p, q)
        for p, q in pairs
        if all(0 <= v < n for v, n in zip(q, shape, strict=True))
    ]
    if not pairs:
        assert got is None
        return
    si, sj = got
    ii = {p for p, _ in pairs}
    a = np.zeros(shape, int)
    a[si] = 1
    assert {tuple(x) for x in np.argwhere(a)} == ii
    # slices match voxel by voxel
    idx = np.arange(np.prod(shape)).reshape(shape)
    mapped = idx[si]
    for p, q in pairs:
        loc = tuple(v - s.start for v, s in zip(p, si, strict=True))
        assert idx[sj][loc] == idx[q]
    assert mapped.shape == idx[sj].shape


def test_tile_positions_shortcuts_agree():
    pos = TileOverlap(positions=[[0, 0, 0], [1, 5, 0], [2, 10, 0]])
    steps = TileOverlap(steps=[[1, 5, 0], [1, 5, 0]])
    step = TileOverlap(step=[1, 5, 0])
    for o in (pos, steps, step):
        assert np.array_equal(o.poses(3, 3), pos.positions)
    with pytest.raises(ValueError, match="exactly one"):
        TileOverlap(step=[0, 1, 0], steps=[[0, 1, 0]])
    with pytest.raises(ValueError, match="integers"):
        TileOverlap(step=[0, 1.5, 0])
    with pytest.raises(ValueError, match="steps"):
        steps.poses(4, 3)
    # same fit from the three descriptions
    data, _ = _tiles(seed=0)
    fits = [
        _tile_fit(data, o, it=3)
        for o in (
            TileOverlap(positions=[[0, 16 * t, 0] for t in range(4)]),
            TileOverlap(steps=[[0, 16, 0]] * 3),
            TileOverlap(step=(0, 16, 0)),
        )
    ]
    for f in fits[1:]:
        assert f.estimates["shading"].allclose(fits[0].estimates["shading"])
        assert f.history.objective == fits[0].history.objective


def test_tile_pairs_grid_and_snake():
    # 2x2 grid of 10x10 tiles stepping 8: the diagonal pair (0, 3)
    # overlaps by a 2x2 corner, (1, 2) too; (0, 1) and (2, 3) are not
    # consecutive-only.
    pos = np.array([[0, 0], [0, 8], [8, 0], [8, 8]])
    pairs = tile_pairs(pos, (10, 10))
    assert [(i, j) for i, j, _ in pairs] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    assert dict(((i, j), s) for i, j, s in pairs)[(0, 3)] == (8, 8)
    si, sj = overlap_slices((8, 8), (10, 10))
    assert si == (slice(8, 10), slice(8, 10)) and sj == (slice(0, 2),) * 2
    # far tiles are skipped
    assert tile_pairs(np.array([[0, 0], [0, 10]]), (10, 10)) == []


def test_overlap_stats_on_a_grid_match_brute_force():
    # (tile, frame, y, x) with 4 tiles on a 2x2 grid, noise-free
    gen = torch.Generator().manual_seed(0)
    H = W = 6
    scene = 50 + 100 * torch.rand(
        (3, 12, 12), generator=gen, dtype=torch.float64
    )
    yy = torch.linspace(-1, 1, H, dtype=torch.float64)
    logm = 0.3 * yy[:, None] - 0.2 * yy[None]
    pos = np.array([[0, 0, 0], [0, 0, 4], [0, 4, 0], [0, 4, 4]])
    tiles = torch.stack(
        [logm.exp() * scene[:, p[1] : p[1] + H, p[2] : p[2] + W] for p in pos]
    )
    C = torch.zeros(H, W, dtype=torch.float64)
    V = torch.ones(H, W, dtype=torch.float64)
    layout = Layout(spatial=(-2, -1))
    ov = Overlap.build(TileOverlap(positions=pos), tiles.shape, layout, 0.0)
    assert len(ov.pairs) == 6  # (1, 2) and (0, 3) are diagonal
    sub = Layout(spatial=(-2, -1))
    st = ov.stats(tiles, sub, C, V, 1.0, False, dict(dtype=torch.float64))
    # noise-free: the true flat field fits every log-ratio exactly
    assert abs(st.bound(logm)) < 1e-8
    assert st.bound(torch.zeros_like(logm)) < -1
    # curvature and score of a quadratic in θ, against finite differences
    from optix.basis import dct_basis, reconstruct

    by = dct_basis(H, 3, dtype=torch.float64)
    bx = dct_basis(W, 3, dtype=torch.float64)
    theta = torch.randn(3, 3, generator=gen, dtype=torch.float64) * 0.1
    Hs, g = st.system(reconstruct(theta, [by, bx]), [by, bx])
    f = lambda t: st.bound(reconstruct(t, [by, bx]))  # noqa: E731
    # finite differences (bound uses .item())
    eps = 1e-5
    num = torch.zeros(9, dtype=torch.float64)
    for k in range(9):
        d = torch.zeros(9, dtype=torch.float64)
        d[k] = eps
        d = d.reshape(3, 3)
        num[k] = (f(theta + d) - f(theta - d)) / (2 * eps)
    assert torch.allclose(num, g, rtol=1e-5, atol=1e-6)
    # exact quadratic: f(θ + δ) = f(θ) + gᵀδ − ½ δᵀHδ
    d = 0.05 * torch.randn(9, generator=gen, dtype=torch.float64)
    pred = f(theta) + g @ d - 0.5 * d @ Hs @ d
    assert abs(f(theta + d.reshape(3, 3)) - pred) < 1e-8 * abs(pred) + 1e-8


def _brute_overlap(data, axis, pos, C, V, gain, full, threshold, weight, logm):
    """Reference sums of the overlap term, voxel by voxel.

    `data` is group-free with the tile axis `axis` and the spatial axes
    last two. Returns (count, total weight, bound at `logm`).
    """
    x = np.moveaxis(np.asarray(data, dtype=np.float64), axis, 0)
    C, V, logm = (np.asarray(a, dtype=np.float64) for a in (C, V, logm))
    shape = x.shape[1:]
    count, total, bound = 0, 0.0, 0.0
    for i in range(len(pos)):
        for j in range(len(pos)):
            if j <= i:
                continue
            s = np.asarray(pos[j]) - np.asarray(pos[i])
            for p in np.ndindex(*shape):
                q = tuple(int(v) for v in np.asarray(p) - s)
                if not all(0 <= v < n for v, n in zip(q, shape, strict=True)):
                    continue
                a, b = x[(i, *p)], x[(j, *q)]
                ri, rj = a - C[p[-2:]], b - C[q[-2:]]
                vi, vj = V[p[-2:]], V[q[-2:]]
                if not (ri > threshold * vi**0.5 and rj > threshold * vj**0.5):
                    continue
                if full:
                    vi, vj = vi + gain * ri, vj + gain * rj
                w = weight / (vi / ri**2 + vj / rj**2)
                d = math.log(ri) - math.log(rj)
                r = d - logm[p[-2:]] + logm[q[-2:]]
                count += 1
                total += w
                bound -= 0.5 * w * r**2
    return count, total, bound


@pytest.mark.parametrize("full", [False, True])
def test_overlap_stats_match_voxel_brute_force(full):
    # (x, y, t, z) data: the tile axis is not first, the tiles also move
    # along the sample axis x (frame shifts), with negative shifts, and
    # the spatial grid (y, z) is not square
    gen = torch.Generator().manual_seed(3)
    X, H, W, T = 4, 7, 5, 4
    pos = np.array([[0, 0, 0], [1, 3, -2], [-1, -2, 3], [2, 1, 1]])
    data = 100 + 40 * torch.rand((X, H, T, W), generator=gen).double()
    data = data + 30 * torch.randn((X, H, T, W), generator=gen).double()
    C = 100 + 3 * torch.rand((H, W), generator=gen).double()
    V = 4 + torch.rand((H, W), generator=gen).double()
    logm = 0.1 * torch.randn((H, W), generator=gen).double()
    layout = Layout("xytz", spatial="yz")
    spec = TileOverlap(positions=pos, axis="t")
    ov = Overlap.build(spec, tuple(data.shape), layout, 1.5)
    st = ov.stats(data, layout, C, V, 2.0, full, dict(dtype=torch.float64))
    count, total, bound = _brute_overlap(
        data.permute(2, 0, 1, 3),  # (t, x, y, z)
        0,
        pos,
        C,
        V,
        2.0,
        full,
        1.5,
        1.0,
        logm,
    )
    assert 0 < st.count == count
    assert math.isclose(
        sum(float(m[0].sum()) for _, _, m, *_ in st.terms.values()),
        total,
        rel_tol=1e-10,
    )
    assert math.isclose(st.bound(logm), bound, rel_tol=1e-10)
    # score and curvature against finite differences, non-square basis
    from optix.basis import dct_basis, reconstruct

    by = dct_basis(H, 3, dtype=torch.float64)
    bx = dct_basis(W, 2, dtype=torch.float64)
    theta = 0.1 * torch.randn(3, 2, generator=gen, dtype=torch.float64)
    Hs, g = st.system(reconstruct(theta, [by, bx]), [by, bx])

    def f(t):
        return _brute_overlap(
            data.permute(2, 0, 1, 3),
            0,
            pos,
            C,
            V,
            2.0,
            full,
            1.5,
            1.0,
            reconstruct(t, [by, bx]),
        )[2]

    eps = 1e-5
    num = torch.zeros(6, dtype=torch.float64)
    for k in range(6):
        e = torch.zeros(6, dtype=torch.float64)
        e[k] = eps
        e = e.reshape(3, 2)
        num[k] = (f(theta + e) - f(theta - e)) / (2 * eps)
    assert torch.allclose(num, g, rtol=1e-5, atol=1e-6)
    d = 0.05 * torch.randn(6, generator=gen, dtype=torch.float64)
    pred = f(theta) + g @ d - 0.5 * d @ Hs @ d
    assert abs(f(theta + d.reshape(3, 2)) - pred) < 1e-8 * abs(pred) + 1e-8


def test_tile_pairs_scale_and_errors():
    # a 40x40 grid of 1600 tiles: bucketed, it scales (the brute-force
    # O(N²) loop took seconds), and agrees with the definition
    import time

    gy, gx = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
    pos = np.stack([gy.ravel() * 9, gx.ravel() * 9], -1)
    t0 = time.perf_counter()
    pairs = tile_pairs(pos, (10, 10))
    assert time.perf_counter() - t0 < 1.0
    # each interior tile touches its 8 neighbours
    assert len(pairs) == 2 * 39 * 40 + 2 * 39 * 39
    rng = np.random.default_rng(0)
    pos = rng.integers(-30, 30, size=(60, 3))
    ref = [
        (i, j, tuple(int(v) for v in pos[j] - pos[i]))
        for i in range(60)
        for j in range(i + 1, 60)
        if overlap_slices(pos[j] - pos[i], (7, 11, 5)) is not None
    ]
    assert tile_pairs(pos, (7, 11, 5)) == ref
    assert tile_pairs(np.zeros((0, 2), np.int64), (4, 4)) == []
    with pytest.raises(ValueError, match="at least one tile"):
        Overlap.build(
            TileOverlap(step=(0, 4)),
            (0, 3, 8),
            Layout(spatial=(-2, -1)),
            3.0,
        )


def _tiles(
    ntiles=4,
    X=12,
    H=32,
    W=24,
    step=16,
    seed=0,
    gain=2.0,
    bright=1.0,
    fade=0.0,
    amp=200.0,
    xstep=0,
):
    """(tile, x, y, z) tiles of one scene whose brightness varies along y.

    Tile t is dimmed by exp(−fade·t) (a uniform per-tile fade), and
    starts `xstep`·t frames into the scene along x.
    """
    nx = X + xstep * (ntiles - 1)
    g = torch.Generator().manual_seed(seed)
    total = step * (ntiles - 1) + H
    yy = torch.arange(total, dtype=torch.float64)
    bright = torch.exp(bright * torch.sin(3 * math.pi * yy / total))[:, None]
    gam = (
        torch._standard_gamma(
            torch.full((nx, total, W), 2.0, dtype=torch.float64), generator=g
        )
        / 2.0
    )
    fg = torch.rand((nx, total, W), generator=g, dtype=torch.float64) < 0.6
    u = amp * bright * gam * fg
    y = torch.linspace(-1, 1, H, dtype=torch.float64)[:, None]
    x = torch.linspace(-1, 1, W, dtype=torch.float64)[None]
    logm = 0.4 * y - 0.3 * y**2 + 0.2 * x * y
    logm = logm - logm.mean()
    out = []
    for t in range(ntiles):
        f = u[t * xstep : t * xstep + X, t * step : t * step + H]
        e = math.exp(-fade * t) * logm.exp() * f
        e = e / gain
        y_ = 100 + gain * torch.poisson(e, generator=g)
        y_ = y_ + 2.0 * torch.randn(e.shape, generator=g, dtype=torch.float64)
        out.append(y_.round())
    return torch.stack(out).to(_working_dtype()), logm


def _tile_fit(data, overlap, it=40, **kw):
    with warnings.catch_warnings(), optix.config(compile=False):
        warnings.simplefilter("ignore")
        est = GammaMixture(
            nclass=1,
            gain=2.0,
            order=(4, 4),
            max_iterations=it,
            overlap=overlap,
            **kw,
        )
        return est.fit(data, layout=Layout("txyz", spatial="yz"))


def test_overlap_default_and_inactive_are_unchanged():
    data, _ = _tiles(seed=1)
    ref = _tile_fit(data, None, it=5)
    assert GammaMixture().options["overlap"] is None
    # no tile touches another: the same fit
    est = _tile_fit(data, TileOverlap(step=(0, 32, 0)), it=5)
    assert est.history.objective == ref.history.objective
    assert est.estimates["shading"].allclose(ref.estimates["shading"])
    with pytest.raises(ValueError, match="sample axis"):
        GammaMixture(overlap=TileOverlap(step=(0, 1), axis=2)).fit(data[0])


@pytest.mark.parametrize("seed", [0, 1])
def test_overlap_improves_flat_field(seed):
    data, logm = _tiles(seed=seed)
    errors, drops, exhausted = [], [], []
    for ov in (None, TileOverlap(step=(0, 16, 0), axis="t")):
        est = _tile_fit(data, ov)
        flat = est.estimates["shading"].flat.double()
        errors.append((flat.log() - logm).abs().mean().item())
        objective = torch.tensor(est.history.objective, dtype=torch.float64)
        drops.append((-objective.diff()).max().item() / objective.abs().max())
        exhausted.append(sum(h == 4 for h in est.history.extras["halvings"]))
    # the scene is brighter in some tiles: without overlap its gradient
    # leaks into m (measured ≈ 0.10 on seeds 0-3); with it, ≈ 0.026-0.028
    assert errors[1] < 0.5 * errors[0]
    assert errors[1] < 0.05
    # this scene breaks the one-class model (its mean depends on the
    # position): the fit is not quite monotone with or without overlap
    # (the class step is not halved). Worst decrease measured on seeds
    # 0-3: 2e-4 to 1.2e-3 of the bound without overlap, 3e-4 to 8e-4
    # with it, recovered at the next iteration. The overlap term must not
    # make this materially worse: bound it relative to the fit without
    # overlap, with a floor well above the measured values.
    assert drops[1] < max(3 * drops[0], 3e-3)
    # halvings exhausted (the same class-step issue): 5-10 of 40
    # iterations without overlap, 6-15 with it; bound it loosely
    assert exhausted[1] <= exhausted[0] + 8


def test_overlap_with_frame_shifts_in_a_fit():
    # tiles that also move by 2 frames along x per step
    data, logm = _tiles(seed=0, xstep=2)
    errors = []
    for ov in (None, TileOverlap(step=(2, 16, 0), axis="t")):
        flat = _tile_fit(data, ov).estimates["shading"].flat.double()
        errors.append((flat.log() - logm).abs().mean().item())
    assert errors[1] < 0.5 * errors[0]
    assert errors[1] < 0.05
    # ignoring the frame shift pairs unrelated frames: worse
    wrong = TileOverlap(step=(0, 16, 0), axis="t")
    flat = _tile_fit(data, wrong).estimates["shading"].flat.double()
    assert (flat.log() - logm).abs().mean().item() > errors[1] + 0.01


def test_overlap_with_groups_fits_each_group():
    # two channels (groups) of tiles: a grouped fit equals separate fits
    a, _ = _tiles(seed=0)
    b, _ = _tiles(seed=1)
    data = torch.stack([a, b], 1)  # (t, c, x, y, z)
    ov = TileOverlap(step=(0, 16, 0), axis="t")
    with warnings.catch_warnings(), optix.config(compile=False):
        warnings.simplefilter("ignore")
        est = GammaMixture(
            nclass=1, gain=2.0, order=(4, 4), max_iterations=5, overlap=ov
        ).fit(data, layout=Layout("tcxyz", spatial="yz", group="c"))
    flat = est.estimates["shading"].flat
    for k, part in enumerate((a, b)):
        ref = _tile_fit(part, ov, it=5).estimates["shading"].flat
        assert torch.allclose(flat[k], ref, rtol=1e-4, atol=1e-6)


def _profiled_grid(fit_gain=True):
    """Noise-free 2x2 grid of tiles with per-tile gains, and its stats."""
    gen = torch.Generator().manual_seed(0)
    H = W = 6
    scene = 50 + 100 * torch.rand(
        (3, 12, 12), generator=gen, dtype=torch.float64
    )
    yy = torch.linspace(-1, 1, H, dtype=torch.float64)
    logm = 0.3 * yy[:, None] - 0.2 * yy[None] + 0.4 * yy[:, None] ** 2
    gains = torch.tensor([1.0, 0.9, 1.2, 0.8], dtype=torch.float64)
    pos = np.array([[0, 0, 0], [0, 0, 4], [0, 4, 0], [0, 4, 4]])
    tiles = torch.stack(
        [
            g * logm.exp() * scene[:, p[1] : p[1] + H, p[2] : p[2] + W]
            for g, p in zip(gains, pos, strict=True)
        ]
    )
    C = torch.zeros(H, W, dtype=torch.float64)
    V = torch.ones(H, W, dtype=torch.float64)
    layout = Layout(spatial=(-2, -1))
    spec = TileOverlap(positions=pos, fit_gain=fit_gain)
    ov = Overlap.build(spec, tiles.shape, layout, 0.0)
    st = ov.stats(tiles, layout, C, V, 1.0, False, dict(dtype=torch.float64))
    return st, logm, gains


def test_tile_gain_flag_and_repr():
    assert TileOverlap(step=(0, 1)).fit_gain is False
    assert TileOverlap(step=(0, 1)) != TileOverlap(step=(0, 1), fit_gain=True)
    assert "fit_gain=True" in repr(TileOverlap(step=(0, 1), fit_gain=True))
    assert "fit_gain" not in repr(TileOverlap(step=(0, 1)))


def test_overlap_accumulators_follow_state_dtype(monkeypatch):
    # on MPS (no float64) the accumulators are float32, like the model's
    import optix.shading.mix._overlap as mod

    monkeypatch.setattr(mod, "state_dtype", lambda device: torch.float32)
    for fit_gain in (False, True):
        st, logm, _ = _profiled_grid(fit_gain=fit_gain)
        assert all(
            m.dtype == torch.float32 for _, _, m, *_ in st.terms.values()
        )
        from optix.basis import dct_basis

        bases = [dct_basis(6, 3, dtype=torch.float32)] * 2
        H, g = st.system(logm.float(), bases)
        assert H.dtype == g.dtype == torch.float32
        assert st.log_gain(logm.float()).dtype == torch.float32
    # with the gains profiled, the true flat field fits (float32 sums
    # cancel in the noise-free bound: a relative check)
    zero = torch.zeros_like(logm, dtype=torch.float32)
    assert abs(st.bound(logm.float())) < 1e-4 * abs(st.bound(zero))


def test_tile_gain_profiled_term_is_exact():
    st, logm, gains = _profiled_grid()
    # noise-free: the true flat field and the profiled gains fit exactly
    assert abs(st.bound(logm)) < 1e-8
    a = st.log_gain(logm)
    rel = (a - a.mean()).exp()
    assert torch.allclose(rel, gains / gains.log().mean().exp(), atol=1e-8)
    assert abs(a.sum().item()) < 1e-10
    # without gains the same data do not fit
    plain, _, _ = _profiled_grid(fit_gain=False)
    assert plain.bound(logm) < -1
    # the profiled term is an exact quadratic in θ with the Schur system
    from optix.basis import dct_basis, reconstruct

    gen = torch.Generator().manual_seed(1)
    by = dct_basis(6, 3, dtype=torch.float64)
    bx = dct_basis(6, 3, dtype=torch.float64)
    theta = torch.randn(3, 3, generator=gen, dtype=torch.float64) * 0.1
    Hs, g = st.system(reconstruct(theta, [by, bx]), [by, bx])
    f = lambda t: st.bound(reconstruct(t, [by, bx]))  # noqa: E731
    eps = 1e-5
    num = torch.zeros(9, dtype=torch.float64)
    for k in range(9):
        d = torch.zeros(9, dtype=torch.float64)
        d[k] = eps
        d = d.reshape(3, 3)
        num[k] = (f(theta + d) - f(theta - d)) / (2 * eps)
    assert torch.allclose(num, g, rtol=1e-5, atol=1e-6)
    d = 0.05 * torch.randn(9, generator=gen, dtype=torch.float64)
    pred = f(theta) + g @ d - 0.5 * d @ Hs @ d
    assert abs(f(theta + d.reshape(3, 3)) - pred) < 1e-8 * abs(pred) + 1e-8
    # the profiled curvature is below the plain one (gains absorb some)
    H0, _ = plain.system(reconstruct(theta, [by, bx]), [by, bx])
    assert torch.linalg.eigvalsh(H0 - Hs).min() > -1e-8


def test_tile_gain_normalised_per_component():
    # tiles 0-1 and 2-3-4 overlap in two separate groups; tile 5 is alone
    gen = torch.Generator().manual_seed(2)
    H = W = 6
    scene = 50 + 100 * torch.rand(
        (2, 40, 6), generator=gen, dtype=torch.float64
    )
    gains = torch.tensor([1.3, 0.7, 2.0, 1.0, 0.5, 3.0], dtype=torch.float64)
    pos = np.array([[0, 0, 0], [0, 3, 0], [0, 12, 0], [0, 15, 0],
                    [0, 18, 0], [0, 30, 0]])  # fmt: skip
    tiles = torch.stack(
        [
            g * scene[:, p[1] : p[1] + H, p[2] : p[2] + W]
            for g, p in zip(gains, pos, strict=True)
        ]
    )
    layout = Layout(spatial=(-2, -1))
    ov = Overlap.build(
        TileOverlap(positions=pos, fit_gain=True), tiles.shape, layout, 0.0
    )
    assert [(i, j) for i, j, _ in ov.pairs] == [(0, 1), (2, 3), (3, 4)]
    C = torch.zeros(H, W, dtype=torch.float64)
    V = torch.ones(H, W, dtype=torch.float64)
    st = ov.stats(tiles, layout, C, V, 1.0, False, dict(dtype=torch.float64))
    a = st.log_gain(torch.zeros(H, W, dtype=torch.float64))
    for group in ([0, 1], [2, 3, 4]):
        assert abs(a[group].sum().item()) < 1e-10
        truth = gains[group].log()
        assert torch.allclose(a[group], truth - truth.mean(), atol=1e-8)
    assert a[5].item() == 0


def test_tile_gain_off_is_unchanged():
    data, _ = _tiles(seed=1)
    ref = _tile_fit(data, TileOverlap(step=(0, 16, 0)), it=5)
    off = _tile_fit(data, TileOverlap(step=(0, 16, 0), fit_gain=False), it=5)
    assert off.history.objective == ref.history.objective
    assert off.estimates["shading"].allclose(ref.estimates["shading"])
    assert ref.tile_gain_ is None and off.tile_gain_ is None


@pytest.mark.parametrize("fade", [0.05, 0.1])
def test_tile_gain_removes_fade_ramp(fade):
    # a scene of uniform mean brightness, dimmed by a factor per tile
    clean, logm = _tiles(seed=0, X=24, bright=0.0)
    data, _ = _tiles(seed=0, X=24, bright=0.0, fade=fade)

    def error(est):
        flat = est.estimates["shading"].flat.double()
        return (flat.log() - logm).abs().mean().item()

    base = error(_tile_fit(clean, TileOverlap(step=(0, 16, 0))))
    nogain = _tile_fit(data, TileOverlap(step=(0, 16, 0)))
    plain = error(nogain)
    est = _tile_fit(data, TileOverlap(step=(0, 16, 0), fit_gain=True))
    # without gains the fade becomes a ramp of m (measured ≈ 0.036 and
    # 0.052 against ≈ 0.026 without fade); with gains, ≈ 0.027 and 0.026;
    # gain ratios within 1.2 % and 1.5 % at the stop (Newton decrement)
    assert plain > base + 0.006
    assert error(est) < base + 0.003
    g = est.tile_gain_.double()
    assert g.shape == (4,)
    assert abs(g.log().sum().item()) < 1e-6
    truth = torch.exp(-fade * torch.arange(4, dtype=torch.float64))
    assert torch.allclose(g / g[0], truth, rtol=0.025)
    # the fit is not quite monotone (the pre-existing class-step issue):
    # worst decrease 2e-4 to 2e-3 of the bound run alone, up to ≈ 7e-3
    # within the suite (in the working dtype), with or without gains. The
    # gains must not make it materially worse than the fit without them.
    drops = []
    for fit in (nogain, est):
        objective = torch.tensor(fit.history.objective, dtype=torch.float64)
        drops.append((-objective.diff()).max() / objective.abs().max())
    assert drops[1] < max(3 * drops[0], 1e-2)


def test_tile_gain_without_fade_does_not_degrade():
    data, logm = _tiles(seed=0)
    errors = []
    for gain in (False, True):
        est = _tile_fit(data, TileOverlap(step=(0, 16, 0), fit_gain=gain))
        flat = est.estimates["shading"].flat.double()
        errors.append((flat.log() - logm).abs().mean().item())
    # measured ≈ 0.026 either way (the scene gradient still does not leak)
    assert errors[1] < errors[0] + 0.005
    assert errors[1] < 0.05
    assert torch.allclose(est.tile_gain_, torch.ones(4).to(est.tile_gain_),
                          atol=0.03)  # fmt: skip
