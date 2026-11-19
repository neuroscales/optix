import os

import numpy as np
import pytest
import torch

import optix
from optix import Estimator, Tags
from optix.estimates import ShadingEstimate
from optix.io import Layout, labeled
from optix.models import random_shading, simulate_shading
from optix.shading import BaSiC
from optix.shading._basic_solvers import (
    Dct2,
    _weights,
    baseline_basicpy,
    baseline_l1,
    default_lambdas,
    resize,
    resize_matrix,
    shrink,
)

FIXTURES = os.path.join(
    os.path.dirname(__file__), "data", "basicpy_fixtures.npz"
)


def make_stack(
    n=40,
    shape=(64, 64),
    *,
    seed=0,
    dark=0.0,
    strength=0.25,
    drift=0.3,
    dtype=torch.float64,
):
    """Sparse bright objects on a background that drifts, shaded."""
    truth = random_shading(
        shape, seed=seed, strength=strength, dark=dark, dtype=dtype
    )
    g = torch.Generator().manual_seed(seed + 1)
    level = 200 * (1 + drift * torch.sin(torch.arange(n, dtype=dtype) / 5))
    spots = torch.rand(n, *shape, generator=g, dtype=dtype) > 0.97
    x = level[:, None, None] + 800 * spots
    return simulate_shading(x, truth), truth, level


# -- solver building blocks ------------------------------------------------
def test_dct_round_trip_and_orthonormal(dtype):
    dct = Dct2(6, 9, dtype=dtype, device=torch.device("cpu"))
    x = torch.randn(3, 6, 9, dtype=dtype)
    c = dct.forward(x)
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(dct.inverse(c), x, atol=tol, rtol=0)
    torch.testing.assert_close(c.norm(), x.norm(), atol=tol * 10, rtol=0)


def test_shrink():
    x = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
    torch.testing.assert_close(
        shrink(x, 1.0), torch.tensor([-2.0, 0.0, 0.0, 0.0, 2.0])
    )


def test_resize_identity_and_shape():
    x = torch.rand(2, 10, 12)
    assert resize(x, (10, 12)) is x and resize(x, None) is x
    assert resize(x, (5, 6)).shape == (2, 5, 6)


@pytest.mark.parametrize(
    "shape,size", [((96, 80), (48, 48)), ((37, 53), (64, 20))]
)
def test_resize_matrix_is_resize(shape, size):
    x = torch.rand(3, *shape, dtype=torch.float64)
    rh = resize_matrix(shape[0], size[0], dtype=x.dtype, block=16)
    rw = resize_matrix(shape[1], size[1], dtype=x.dtype)
    torch.testing.assert_close(rh @ x @ rw.T, resize(x, size))


def test_default_lambdas_note4():
    # Eq. (15): λ_S = ‖𝓕(Ī)‖₁ / 800, λ_D = ‖𝓕(Ī)‖₁ / 2000
    D = torch.rand(5, 8, 8, dtype=torch.float64) + 1
    dct = Dct2(8, 8, dtype=D.dtype, device=D.device)
    mean = D.mean(0) / D.mean()
    s = float(dct.forward(mean).abs().sum())
    lam_s, lam_d = default_lambdas(D, dct)
    assert lam_s == pytest.approx(s / 800) and lam_d == pytest.approx(s / 2000)


def test_weights_eq9():
    # Eq. (9): W = 1 / (|I_R / I_B| + ε), elementwise, not normalised
    g = torch.Generator().manual_seed(0)
    I_B = torch.rand(3, 4, 5, generator=g, dtype=torch.float64) + 0.5
    E = torch.randn(3, 4, 5, generator=g, dtype=torch.float64)
    W = _weights(I_B, E, 0.1, "paper")
    torch.testing.assert_close(W, 1 / ((E / I_B).abs() + 0.1))


def test_baseline_l1_exact():
    data, truth, level = make_stack()
    zero = torch.zeros_like(truth.flat)
    b = baseline_l1(data, truth.flat, zero)
    torch.testing.assert_close(b, level, rtol=1e-10, atol=1e-8)


def test_baseline_l1_reweighting_removes_bias():
    # dense foreground with varied intensities: the plain L1 fit (median)
    # is pulled up, the reweighted fit of Eq. (14) is not
    truth = random_shading((64, 64), seed=0, dtype=torch.float64)
    g = torch.Generator().manual_seed(5)
    n = 20
    level = torch.linspace(100, 300, n, dtype=torch.float64)
    u = torch.rand(n, 64, 64, generator=g, dtype=torch.float64)
    fg = (u < 0.4) * (200 + 1500 * u)
    noise = 5 * torch.randn(n, 64, 64, generator=g, dtype=torch.float64)
    data = truth.flat * (level[:, None, None] + fg + noise)
    zero = torch.zeros_like(truth.flat)
    plain = baseline_l1(data, truth.flat, zero, max_reweight=1)
    b = baseline_l1(data, truth.flat, zero)
    assert ((plain - level) / level).abs().max() > 0.02
    assert ((b - level) / level).abs().max() < 0.005
    # frames are independent of the batching
    torch.testing.assert_close(
        baseline_l1(data[:7], truth.flat, zero), b[:7], rtol=0, atol=0
    )


# -- synthetic ground truth ------------------------------------------------
def make_cells(n=30, shape=(160, 144), *, dark=50.0, frac=0.3, seed=0):
    """Static images: foreground of varied brightness on a fixed background."""
    truth = random_shading(
        shape, seed=seed, strength=0.25, dark=dark, dtype=torch.float64
    )
    g = torch.Generator().manual_seed(seed + 1)
    u = torch.rand(n, *shape, generator=g, dtype=torch.float64)
    x = 100 + (u < frac) * (100 + 2000 * u)
    return simulate_shading(x, truth), truth


def test_faithful_recovers_flat_and_dark():
    # the setting of the paper: static images, sorted, working size 128
    data, truth = make_cells()
    b = BaSiC(darkfield=True).fit(data)
    assert b.options["sort_intensity"] and b.options["working_size"] == 128
    est = b.estimates["shading"]
    assert est.flat.shape == (160, 144)
    assert (est.flat - truth.flat).abs().max() < 0.015
    assert (est.dark - truth.dark).abs().max() < 1.0
    assert b.history.converged
    assert est.meta["estimator"] == "BaSiC"
    plain = BaSiC().fit(data)
    assert plain.is_faithful and plain.estimates["shading"].meta["faithful"]


def test_recovers_dark_with_drift():
    data, truth, _ = make_stack(dark=50.0)
    est = BaSiC(working_size=64, darkfield=True).fit(data)
    est = est.estimates["shading"]
    assert (est.flat - truth.flat).abs().max() < 0.06
    assert (est.dark - truth.dark).abs().max() < 2.0
    assert abs(float(est.dark.mean()) - 50.0) < 1.0
    unsorted = BaSiC(working_size=64, darkfield=True, faithful=False)
    est = unsorted.fit(data).estimates["shading"]
    assert (est.flat - truth.flat).abs().max() < 0.01
    assert (est.dark - truth.dark).abs().max() < 0.5


def test_sorts_before_resizing():
    data, truth, _ = make_stack(n=30, shape=(96, 80))
    b = BaSiC(working_size=48)
    layout = Layout.default(3, 2)
    backend = b._backend(data)
    frames = torch.arange(30)
    D = b._stack(data, layout, (), frames, backend, (96, 80), (48, 48))
    ref = resize(data.sort(0).values, (48, 48))
    torch.testing.assert_close(D.double(), ref, atol=1e-3, rtol=1e-5)
    est = b.fit(data).estimates["shading"]
    assert (est.flat - truth.flat).abs().max() < 0.04
    unsorted = BaSiC(working_size=48, faithful=False).fit(data)
    flat = unsorted.estimates["shading"].flat
    assert (flat - truth.flat).abs().max() < 0.01


def test_sorted_stack_in_strips():
    data, _, _ = make_stack(n=12, shape=(40, 36))
    b = BaSiC(working_size=(24, 20))
    layout = Layout.default(3, 2)
    args = (data, layout, (), torch.arange(12), b._backend(data), (40, 36))
    whole = b._stack(*args, (24, 20))
    with optix.config(memory=12 * 36 * 8 * 3 * 5):  # strips of 5 rows
        strips = b._stack(*args, (24, 20))
        same = b._stack(*args, None)
    torch.testing.assert_close(strips, whole)
    torch.testing.assert_close(same, data.sort(0).values.to(same))


def test_float32_matches_float64():
    data, _, _ = make_stack()
    f64 = BaSiC(working_size=64, dtype=torch.float64).fit(data)
    f32 = BaSiC(working_size=64, dtype=torch.float32).fit(data.float())
    flat32 = f32.estimates["shading"].flat
    assert flat32.dtype == torch.float32
    torch.testing.assert_close(
        flat32.double(), f64.estimates["shading"].flat, atol=1e-3, rtol=0
    )


def test_timelapse_baseline_and_apply():
    data, truth, level = make_stack()
    b = BaSiC(working_size=64, timelapse=True).fit(data)
    est = b.estimates["shading"]
    assert est.baseline.shape == (40,)
    assert est.meta["baseline_units"] == "signal"
    assert ((est.baseline - level) / level).abs().max() < 0.03
    assert b.fit_baseline_ is None  # sorted fit: B_i are not baselines
    unsorted = BaSiC(working_size=64, faithful=False).fit(data)
    torch.testing.assert_close(
        unsorted.fit_baseline_[unsorted.fit_frames_].double(),
        level,
        rtol=2e-3,
        atol=0,
    )
    x = b.apply(data)
    # background pixels become ~0 once the baseline is removed
    spots = (data / truth.flat - level[:, None, None]) > 100
    assert x[~spots].abs().max() < 0.08 * level.min()
    # other data (other sample shape): baseline computed on the fly
    x2 = b.apply(data[:7])
    assert x2.shape == (7, 64, 64)
    torch.testing.assert_close(x2, x[:7])


def test_apply_without_timelapse():
    data, _, _ = make_stack()
    b = BaSiC(working_size=64).fit(data)
    x = b.apply(data)
    flat = b.estimates["shading"].flat
    torch.testing.assert_close(x, data.to(x.dtype) / flat)


# -- streaming and layouts -------------------------------------------------
@pytest.mark.parametrize("sampling", ["stride", "reservoir"])
def test_frame_subsample(sampling):
    data, truth, _ = make_stack(n=50)
    b = BaSiC(
        working_size=64,
        max_frames=20,
        sampling=sampling,
        seed=3,
        faithful=False,
    )
    b.fit(data.numpy())
    frames = b.fit_frames_
    assert len(frames) == 20 and len(set(frames.tolist())) == 20
    assert bool((frames[1:] > frames[:-1]).all())
    assert b.fit_baseline_.shape == (20,)
    assert (b.estimates["shading"].flat - truth.flat).abs().max() < 0.02
    if sampling == "stride":
        assert frames.tolist() == [i * 50 // 20 for i in range(20)]


@pytest.mark.parametrize("sort", [False, True])
def test_layout_groups_and_spatial_order(sort):
    zarr = pytest.importorskip("zarr")
    # (t, c, y, x) stored, one flat per channel, maps in (c, x, y) order
    stacks, flats = [], []
    for c in range(2):
        data, truth, _ = make_stack(n=24, shape=(32, 40), seed=c, drift=0.1)
        stacks.append(data)
        flats.append(truth.flat)
    y = torch.stack(stacks, 1).numpy().astype("f4")  # (t, c, y, x)
    Z = zarr.create_array(
        store={}, shape=y.shape, chunks=(4, 1, 32, 40), dtype="f4"
    )
    Z[:] = y
    layout = Layout("tcyx", spatial="xy", group="c")
    b = BaSiC(working_size=32, timelapse=True, sort_intensity=sort)
    b.fit(labeled(Z, layout))
    est = b.estimates["shading"]
    assert est.flat.shape == (2, 40, 32)
    assert est.baseline.shape == (2, 24)
    for c in range(2):
        err = (est.flat[c].T.double() - flats[c]).abs().max()
        assert err < 0.03
    out = zarr.create_array(
        store={}, shape=y.shape, chunks=(5, 2, 32, 40), dtype="f4"
    )
    assert b.apply(Z, out=out) is out
    ref = (y / est.flat.numpy().transpose(0, 2, 1)[None]) - est.baseline.T[
        :, :, None, None
    ].numpy()
    np.testing.assert_allclose(out[:], ref, rtol=1e-4, atol=1e-2)


def test_needs_two_spatial_axes():
    with pytest.raises(ValueError, match="two spatial"):
        BaSiC().fit(np.zeros((4, 3, 8, 8)), layout=Layout.default(4, 3))


def test_options():
    assert BaSiC().is_faithful
    assert BaSiC(faithful=False, sort_intensity=True).is_faithful
    assert not BaSiC(faithful=False).is_faithful
    opt = BaSiC().options
    assert opt["sort_intensity"] and opt["variant"] == "paper"
    assert opt["reweight_tol"] == 1e-3 and opt["max_reweight"] == 10
    assert opt["epsilon"] == 0.1 and opt["working_size"] == 128
    for name in ("baseline_norm", "baseline_method", "sampling", "variant"):
        with pytest.raises(ValueError, match=name):
            BaSiC(**{name: "nope"})
    with pytest.raises(RuntimeError, match="not fitted"):
        BaSiC().apply(np.zeros((2, 8, 8)))


def test_baseline_norm():
    data, truth, level = make_stack()
    b = BaSiC(working_size=64, timelapse=True, baseline_norm="mean").fit(data)
    x = b.apply(data)
    spots = (data / truth.flat - level[:, None, None]) > 100
    base = b.estimates["shading"].baseline.mean()
    assert (x[~spots] - base).abs().max() < 0.08 * level.min()
    b2 = BaSiC(working_size=64, timelapse=True, baseline_norm=10.0).fit(data)
    assert (b2.apply(data)[~spots] - 10).abs().max() < 0.08 * level.min()


# -- interop ---------------------------------------------------------------
class UsesShading(Estimator):
    tags = Tags(targets=("test",), streaming="pointwise")

    def _fit(self, data, *, init, callback, verbose):
        start = self._pick(init, ShadingEstimate, shape=(64, 64), like=data)
        self.estimates = {"shading": start}


def test_estimate_initialises_other_estimators(tmp_path):
    data, _, _ = make_stack()
    b = BaSiC(working_size=64, darkfield=True, timelapse=True).fit(data)
    # Estimator._pick checks every field against the map shape, which the
    # per-frame baseline never matches: pass the profiles only
    profiles = b.estimates["shading"].replace(baseline=None)
    other = UsesShading().fit(data, init=profiles)
    assert other.estimates["shading"].allclose(profiles)
    b.save(tmp_path / "b.pt")
    loaded = optix.load(tmp_path / "b.pt")
    assert loaded.estimates["shading"].allclose(b.estimates["shading"])
    torch.testing.assert_close(loaded.fit_frames_, b.fit_frames_)
    torch.testing.assert_close(loaded.apply(data), b.apply(data))
    geo = b.estimates["shading"].normalize("geomean")
    assert geo.meta["flat_norm"] == "geomean"


# -- reference: BaSiCPy 2.0.0 ----------------------------------------------
@pytest.fixture(scope="module")
def ref():
    return np.load(FIXTURES)


def test_fixture_provenance(ref):
    assert str(ref["basicpy_version"]) == "2.0.0"


@pytest.mark.parametrize(
    "case,dark", [("approx", False), ("approx_dark", True)]
)
def test_basicpy_variant_matches_basicpy(ref, case, dark):
    # BaSiCPy defaults: unsorted, reweighting tolerance 1 %
    b = BaSiC(
        working_size=64,
        darkfield=dark,
        variant="basicpy",
        sort_intensity=False,
        reweight_tol=1e-2,
    )
    b.fit(ref["stack"])
    est = b.estimates["shading"]
    flat = est.flat.double().numpy()
    np.testing.assert_allclose(flat, ref[f"{case}_flat"], rtol=1e-4)
    np.testing.assert_allclose(
        b.fit_baseline_.double().numpy(), ref[f"{case}_baseline"], rtol=1e-4
    )
    if dark:
        rd = ref[f"{case}_dark"]
        span = rd.max() - rd.min()
        np.testing.assert_allclose(
            est.dark.double().numpy(), rd, atol=1e-3 * span
        )


@pytest.mark.parametrize("case", ["approx", "approx_dark"])
def test_baseline_matches_basicpy(ref, case):
    D = torch.as_tensor(ref["stack"].astype("f8"))
    b = baseline_basicpy(
        D,
        torch.as_tensor(ref[f"{case}_flat"]).double(),
        torch.as_tensor(ref[f"{case}_dark"]).double(),
    )
    np.testing.assert_allclose(
        b.numpy(), ref[f"{case}_fit_only_baseline"], rtol=1e-4
    )
