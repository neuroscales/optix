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
    baseline_basicpy,
    baseline_median,
    resize,
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
    dtype=torch.float64,
):
    """Sparse bright objects on a time-varying background, shaded."""
    truth = random_shading(
        shape, seed=seed, strength=strength, dark=dark, dtype=dtype
    )
    g = torch.Generator().manual_seed(seed + 1)
    level = 200 * (1 + 0.3 * torch.sin(torch.arange(n, dtype=dtype) / 5))
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


def test_baseline_median_exact():
    Y, truth, level = make_stack()
    zero = torch.zeros_like(truth.flat)
    b = baseline_median(Y, truth.flat, zero)
    torch.testing.assert_close(b, level, rtol=1e-10, atol=1e-8)


# -- synthetic ground truth ------------------------------------------------
def test_recovers_flat():
    Y, truth, _ = make_stack()
    b = BaSiC(working_size=64).fit(Y)
    est = b.estimates["shading"]
    assert est.flat.shape == (64, 64) and est.dark is None
    assert (est.flat - truth.flat).abs().max() < 0.01
    assert est.meta["flat_norm"] == "mean"
    assert est.meta["estimator"] == "BaSiC"
    assert est.meta["faithful"] is False  # working_size differs
    assert b.history.converged


def test_recovers_flat_with_resizing():
    Y, truth, _ = make_stack(n=30, shape=(96, 80))
    est = BaSiC(working_size=48, faithful=False).fit(Y).estimates["shading"]
    assert est.flat.shape == (96, 80)
    assert (est.flat - truth.flat).abs().max() < 0.02


def test_recovers_dark():
    Y, truth, _ = make_stack(dark=50.0)
    opts = dict(working_size=64, faithful=False)
    plain = BaSiC(**opts).fit(Y).estimates["shading"]
    est = BaSiC(darkfield=True, **opts).fit(Y).estimates["shading"]
    err = (est.flat - truth.flat).abs().max()
    assert err < 0.06 < (plain.flat - truth.flat).abs().max()
    assert abs(float(est.dark.mean()) - 50.0) < 1.0


def test_float32_matches_float64():
    Y, _, _ = make_stack()
    f64 = BaSiC(working_size=64, dtype=torch.float64).fit(Y)
    f32 = BaSiC(working_size=64, dtype=torch.float32).fit(Y.float())
    flat32 = f32.estimates["shading"].flat
    assert flat32.dtype == torch.float32
    torch.testing.assert_close(
        flat32.double(), f64.estimates["shading"].flat, atol=1e-4, rtol=0
    )


def test_timelapse_baseline_and_apply():
    Y, truth, level = make_stack()
    b = BaSiC(working_size=64, timelapse=True).fit(Y)
    est = b.estimates["shading"]
    assert est.baseline.shape == (40,)
    assert est.meta["baseline_units"] == "signal"
    assert ((est.baseline - level) / level).abs().max() < 0.01
    assert b.fit_baseline_ is None  # sorted fit: B_i are not baselines
    unsorted = BaSiC(working_size=64, faithful=False).fit(Y)
    torch.testing.assert_close(
        unsorted.fit_baseline_[unsorted.fit_frames_].double(),
        level,
        rtol=1e-3,
        atol=0,
    )
    x = b.apply(Y)
    # background pixels become ~0 once the baseline is removed
    spots = (Y / truth.flat - level[:, None, None]) > 100
    assert x[~spots].abs().max() < 0.05 * level.min()
    # other data (other sample shape): cheap per-frame baseline
    x2 = b.apply(Y[:7])
    assert x2.shape == (7, 64, 64)
    assert x2[~spots[:7]].abs().max() < 0.05 * level.min()


def test_apply_without_timelapse():
    Y, truth, level = make_stack()
    b = BaSiC(working_size=64).fit(Y)
    x = b.apply(Y)
    flat = b.estimates["shading"].flat
    torch.testing.assert_close(x, Y.to(x.dtype) / flat)


# -- streaming and layouts -------------------------------------------------
@pytest.mark.parametrize("sampling", ["stride", "reservoir"])
def test_frame_subsample(sampling):
    Y, truth, _ = make_stack(n=50)
    b = BaSiC(
        working_size=64,
        max_frames=20,
        sampling=sampling,
        seed=3,
        faithful=False,
    )
    b.fit(Y.numpy())
    frames = b.fit_frames_
    assert len(frames) == 20 and len(set(frames.tolist())) == 20
    assert bool((frames[1:] > frames[:-1]).all())
    assert b.fit_baseline_.shape == (20,)
    assert (b.estimates["shading"].flat - truth.flat).abs().max() < 0.02
    if sampling == "stride":
        assert frames.tolist() == [i * 50 // 20 for i in range(20)]


def test_layout_groups_and_spatial_order():
    zarr = pytest.importorskip("zarr")
    # (t, c, y, x) stored, one flat per channel, maps in (c, x, y) order
    stacks, flats = [], []
    for c in range(2):
        Y, truth, _ = make_stack(n=24, shape=(32, 40), seed=c)
        stacks.append(Y)
        flats.append(truth.flat)
    y = torch.stack(stacks, 1).numpy().astype("f4")  # (t, c, y, x)
    Z = zarr.create_array(
        store={}, shape=y.shape, chunks=(4, 1, 32, 40), dtype="f4"
    )
    Z[:] = y
    layout = Layout("tcyx", spatial="xy", group="c")
    b = BaSiC(working_size=32, timelapse=True, faithful=False)
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
    assert BaSiC().options["sort_intensity"]
    with pytest.raises(ValueError, match="baseline_norm"):
        BaSiC(baseline_norm="median")
    with pytest.raises(ValueError, match="baseline_method"):
        BaSiC(baseline_method="nope")
    with pytest.raises(ValueError, match="sampling"):
        BaSiC(sampling="nope")
    with pytest.raises(RuntimeError, match="not fitted"):
        BaSiC().apply(np.zeros((2, 8, 8)))


def test_unsorted_extension():
    Y, truth, _ = make_stack()
    est = BaSiC(working_size=64, sort_intensity=False).fit(Y)
    flat = est.estimates["shading"].flat
    assert (flat - truth.flat).abs().max() < 0.01


def test_baseline_norm():
    Y, truth, level = make_stack()
    b = BaSiC(working_size=64, timelapse=True, baseline_norm="mean").fit(Y)
    x = b.apply(Y)
    spots = (Y / truth.flat - level[:, None, None]) > 100
    base = b.estimates["shading"].baseline.mean()
    assert (x[~spots] - base).abs().max() < 0.05 * level.min()
    b2 = BaSiC(working_size=64, timelapse=True, baseline_norm=10.0).fit(Y)
    assert (b2.apply(Y)[~spots] - 10).abs().max() < 0.05 * level.min()


# -- interop ---------------------------------------------------------------
class UsesShading(Estimator):
    tags = Tags(targets=("test",), streaming="pointwise")

    def _fit(self, Y, *, init, callback, verbose):
        start = self._pick(init, ShadingEstimate, shape=(64, 64), like=Y)
        self.estimates = {"shading": start}


def test_estimate_initialises_other_estimators(tmp_path):
    Y, _, _ = make_stack()
    b = BaSiC(working_size=64, darkfield=True, timelapse=True).fit(Y)
    # Estimator._pick checks every field against the map shape, which the
    # per-frame baseline never matches: pass the profiles only
    profiles = b.estimates["shading"].replace(baseline=None)
    other = UsesShading().fit(Y, init=profiles)
    assert other.estimates["shading"].allclose(profiles)
    b.save(tmp_path / "b.pt")
    loaded = optix.load(tmp_path / "b.pt")
    assert loaded.estimates["shading"].allclose(b.estimates["shading"])
    torch.testing.assert_close(loaded.fit_frames_, b.fit_frames_)
    torch.testing.assert_close(loaded.apply(Y), b.apply(Y))
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
def test_matches_basicpy(ref, case, dark):
    # BaSiCPy defaults: unsorted (and S starts at ones, which makes no
    # measurable difference to the zero start of the paper)
    b = BaSiC(working_size=64, darkfield=dark, sort_intensity=False)
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
