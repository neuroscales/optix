import numpy as np
import pytest
import torch

import optix
from optix import Estimator, History, Tags
from optix.estimates import SensorEstimate, ShadingEstimate
from optix.io import Layout, labeled
from optix.stats import Moments, reduce, reduce_groups


class MeanDark(Estimator):
    """Toy estimator: the per-pixel mean (plus a bias) is the dark image."""

    tags = Tags(targets=("shading",), streaming="per-pixel-stats")
    paper = {"bias": 0.0, "scale": 1.0}
    recommended = {"bias": 0.0, "scale": 2.0}

    def __init__(
        self,
        *,
        bias=None,
        scale=None,
        faithful=True,
        dtype=None,
        device=None,
    ):
        super().__init__(dtype=dtype, device=device)
        self._resolve(faithful, locals())

    def _fit(self, data, *, init, callback, verbose):
        self._warn_unused(init, ShadingEstimate)
        start = self._pick(
            init, ShadingEstimate, shape=data.shape[-2:], like=data
        )
        (m,) = reduce(data, Moments(1), dtype=self.dtype, device=self.device)
        dark = m.mean.to(self._backend(data)["dtype"]) * self.options["scale"]
        dark = dark + self.options["bias"]
        if start is not None and start.dark is not None:
            dark = (dark + start.dark) / 2
        self.estimates = {"shading": ShadingEstimate(dark=dark)}
        self.history.record(1.0)
        self.history.record(1.0)
        self.history.check(1e-6)
        if callback is not None:
            callback(self)

    def _apply(self, data, out):
        dark = self.estimates["shading"].dark

        def fn(index, data):
            return data - dark.to(data)

        return self._apply_frames(data, out, fn)


class Blur(Estimator):
    tags = Tags(targets=("test",), streaming="halo")

    def _apply(self, data, out):
        return self._apply_blocks(data, out, lambda d: d, halo=1)


@pytest.fixture
def volume():
    g = np.random.default_rng(0)
    return (100 + g.normal(size=(2, 6, 5, 4))).astype("f4")


def test_presets_and_is_faithful():
    e = MeanDark()
    assert e.options == {"bias": 0.0, "scale": 1.0} and e.is_faithful
    r = MeanDark(faithful=False)
    assert r.options["scale"] == 2.0 and not r.is_faithful
    x = MeanDark(bias=1.0)
    assert x.options == {"bias": 1.0, "scale": 1.0} and not x.is_faithful
    assert MeanDark(faithful=False, scale=1.0).is_faithful


@pytest.mark.parametrize("kind", ["numpy", "tensor", "zarr"])
def test_fit_apply_round_trip(volume, kind, tmp_path):
    if kind == "zarr":
        zarr = pytest.importorskip("zarr")
        data = zarr.create_array(
            store={}, shape=volume.shape, chunks=(1, 2, 5, 4), dtype="f4"
        )
        data[:] = volume
        out = zarr.create_array(
            store={}, shape=volume.shape, chunks=(1, 3, 5, 4), dtype="f4"
        )
    elif kind == "tensor":
        data = torch.from_numpy(volume)
        out = torch.empty(volume.shape)
    else:
        data = volume
        out = np.empty_like(volume)
    e = MeanDark().fit(data)
    dark = e.estimates["shading"].dark
    np.testing.assert_allclose(dark.numpy(), volume.mean((0, 1)), rtol=1e-6)
    assert e.estimates["shading"].meta["estimator"] == "MeanDark"
    assert e.estimates["shading"].meta["faithful"] is True
    assert e.history.converged
    res = e.apply(data, out=out)
    assert res is out
    np.testing.assert_allclose(
        np.asarray(res[:]), volume - volume.mean((0, 1)), atol=1e-4
    )
    # default output: in memory, on the CPU for non-tensor inputs
    default = e.apply(data)
    assert isinstance(default, torch.Tensor) and default.device.type == "cpu"


def test_history_reset_on_each_fit(volume):
    e = MeanDark().fit(volume)
    first = e.history
    assert first.objective == [1.0, 1.0] and first.converged
    e.fit(volume)
    assert e.history.objective == [1.0, 1.0] and e.history.converged
    assert e.history is not first
    # a fit that fails does not keep the previous trace either
    e.history.record(5.0)
    with pytest.raises(ValueError, match="fewer than"):
        e.fit(np.zeros((2,), dtype="f4"))
    assert 5.0 not in e.history.objective


def test_apply_frames_in_place(volume):
    y = volume.copy()
    e = MeanDark().fit(y)
    e.apply(y, out=y)
    np.testing.assert_allclose(y, volume - volume.mean((0, 1)), atol=1e-4)


def test_apply_blocks_refuses_in_place():
    y = torch.randn(4, 4)
    with pytest.raises(ValueError, match="In-place"):
        Blur().apply(y, out=y)
    with pytest.raises(ValueError, match="In-place"):
        Blur().apply(y, out=y[:])
    out = Blur().apply(y, out=torch.empty(4, 4))
    torch.testing.assert_close(out, y)


def test_init_is_cloned_and_unusable_warns(volume):
    start = ShadingEstimate(dark=torch.zeros(5, 4))
    with pytest.warns(RuntimeWarning, match="SensorEstimate"):
        e = MeanDark().fit(volume, init=[start, SensorEstimate(offset=1.0)])
    assert start.dark.eq(0).all()
    np.testing.assert_allclose(
        e.estimates["shading"].dark.numpy(), volume.mean((0, 1)) / 2, rtol=1e-6
    )
    # a fitted estimator can be the init
    MeanDark().fit(volume, init=e)
    with pytest.raises(ValueError, match="broadcast"):
        MeanDark().fit(volume, init=ShadingEstimate(dark=torch.zeros(3, 3)))


def test_callback_and_no_grad(volume):
    seen = []
    y = torch.from_numpy(volume).requires_grad_()
    e = MeanDark().fit(y, callback=seen.append)
    assert seen == [e]
    assert not e.estimates["shading"].dark.requires_grad


def test_to(volume):
    e = MeanDark().fit(volume).to(dtype=torch.float64)
    assert e.estimates["shading"].dark.dtype == torch.float64
    assert e.dtype == torch.float64
    e.to("meta")
    assert e.estimates["shading"].dark.device.type == "meta"


def test_save_load(volume, tmp_path):
    e = MeanDark(bias=2.0).fit(volume)
    path = tmp_path / "e.pt"
    e.save(path)
    loaded = optix.load(path)
    assert isinstance(loaded, MeanDark)
    assert loaded.options == e.options and not loaded.is_faithful
    assert loaded.estimates["shading"].allclose(e.estimates["shading"])
    assert loaded.history.objective == e.history.objective


def test_registry():
    reg = optix.estimators()
    assert reg["MeanDark"] is MeanDark and reg["Blur"] is Blur
    assert "Estimator" not in reg


def test_history_convergence_and_monotone():
    h = History()
    assert h.gain() == float("inf")
    assert not h.check(1e-3)
    h.record(-100.0, kl=1.0)
    h.record(-50.0, kl=0.5)
    assert h.gain() == pytest.approx(1.0)
    assert h.extras == {"kl": [1.0, 0.5]}
    assert not h.check(1e-3, monotone=True)
    h.record(-60.0)
    with pytest.warns(RuntimeWarning, match="decreased"):
        h.check(1e-3, monotone=True)
    h.record(-60.0 + 1e-9)
    assert h.check(1e-6)
    assert h.converged and "tol" in h.reason


# --- layouts ----------------------------------------------------------------


class GroupDark(Estimator):
    """Toy estimator: per-group mean image, shared over the samples."""

    tags = Tags(targets=("shading",), streaming="per-pixel-stats")

    def __init__(self, *, faithful=True, dtype=None, device=None, layout=None):
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())

    def _fit(self, data, *, init, callback, verbose):
        shape = self._map_shape(data)
        dark = torch.zeros(shape, **self._backend(data))
        groups = reduce_groups(
            data, Moments(1), batch=3, dtype=self.dtype, device=self.device
        )
        for key, (m,) in groups.items():
            dark[key] = m.mean.to(dark)
        self.estimates = {"shading": ShadingEstimate(dark=dark)}

    def _apply(self, data, out):
        dark = self.estimates["shading"].dark

        def fn(chunk, data):
            return data - dark[chunk.group].to(data)

        return self._apply_frames(data, out, fn)


def test_estimator_with_layout_end_to_end(tmp_path):
    zarr = pytest.importorskip("zarr")
    g = np.random.default_rng(1)
    y = (100 + g.normal(size=(2, 3, 4, 5, 6))).astype("f4")  # t c z y x
    data = zarr.create_array(
        store={}, shape=y.shape, chunks=(1, 1, 2, 5, 6), dtype="f4"
    )
    data[:] = y
    layout = Layout("tczyx", spatial="xy", group="c")
    ref = y.mean((0, 2)).transpose(0, 2, 1)  # (C, X, Y)
    for e in (
        GroupDark(layout=layout).fit(data),
        GroupDark().fit(data, layout=layout),
        GroupDark().fit(labeled(data, layout)),
    ):
        assert e.layout == layout
        dark = e.estimates["shading"].dark
        assert dark.shape == (3, 6, 5)
        np.testing.assert_allclose(dark.numpy(), ref, rtol=1e-5)
    # apply uses the fitted layout for a plain array; zarr out round trip
    out = zarr.create_array(
        store={}, shape=y.shape, chunks=(2, 1, 4, 5, 3), dtype="f4"
    )
    assert e.apply(data, out=out) is out
    expected = y - y.mean((0, 2), keepdims=True)
    np.testing.assert_allclose(out[:], expected, atol=1e-4)
    # the layout survives save / load
    e.save(tmp_path / "e.pt")
    loaded = optix.load(tmp_path / "e.pt")
    assert loaded.layout == layout
    res = loaded.apply(y)
    np.testing.assert_allclose(res.numpy(), expected, atol=1e-4)
    with pytest.raises(ValueError, match="conflicts"):
        GroupDark().fit(labeled(data, layout), layout=Layout.default(5, 2))
    with pytest.raises(TypeError):
        GroupDark(layout="tczyx")


def test_estimator_light_sheet_layout():
    """(z, y, x) stack whose pictures are the (y, z) planes, one per x."""
    g = np.random.default_rng(2)
    y = (10 + g.normal(size=(6, 5, 20))).astype("f4")
    layout = Layout("zyx", spatial="yz", sample="x")
    e = GroupDark(layout=layout).fit(y)
    dark = e.estimates["shading"].dark
    assert dark.shape == (5, 6)
    np.testing.assert_allclose(dark.numpy(), y.mean(2).T, rtol=1e-5)
    res = e.apply(torch.from_numpy(y))
    np.testing.assert_allclose(
        res.numpy(), y - y.mean(2, keepdims=True), atol=1e-4
    )
    # without a layout, nothing changes
    d = MeanDark().fit(y[None])
    assert d.layout is None
    assert d.estimates["shading"].dark.shape == (5, 20)


def test_labeled_tensor_keeps_device_and_blocks_layout():
    y = torch.randn(4, 3, 5)
    e = GroupDark(layout=Layout("zyx", spatial="yx")).fit(y)
    assert e._backend(labeled(y, e.layout))["device"] == y.device
    data = labeled(y, axes="zyx", spatial="xz")
    with pytest.raises(ValueError, match="In-place"):
        Blur().apply(data, out=data)
    torch.testing.assert_close(Blur().apply(data, out=torch.empty(4, 3, 5)), y)


class DenseBlur(Estimator):
    tags = Tags(targets=("test",), streaming="halo")

    def _apply(self, data, out):
        def fn(d):
            assert d.is_contiguous()
            return d * 1

        return self._apply_blocks(data, out, fn, halo=1, contiguous=True)


def test_apply_blocks_contiguous_option():
    y = np.random.default_rng(2).normal(size=(4, 5, 6)).astype("f4")
    data = labeled(y, axes="zyx", spatial="yz")
    out = DenseBlur().apply(data, out=np.zeros_like(y))
    np.testing.assert_allclose(out, y)


# --- issue #68: layout routing, per-frame baseline in init ------------------


class ForgetsLayout(Estimator):
    """Takes `layout=` but does not pass it on to `Estimator.__init__`."""

    tags = Tags(targets=("test",), streaming="pointwise")

    def __init__(self, *, faithful=True, dtype=None, device=None, layout=None):
        super().__init__(dtype=dtype, device=device)
        self._resolve(faithful, locals())


def test_resolve_routes_layout():
    layout = Layout("zyx", spatial="yx")
    e = ForgetsLayout(layout=layout)
    assert e.layout == layout
    assert "layout" not in e.options
    assert ForgetsLayout().layout is None
    with pytest.raises(TypeError, match="Layout"):
        ForgetsLayout(layout="zyx")


class PicksShading(Estimator):
    tags = Tags(targets=("test",), streaming="pointwise")

    def _fit(self, data, *, init, callback, verbose):
        shape = self._map_shape(data)
        self.estimates = {
            "shading": self._pick(
                init, ShadingEstimate, shape=shape, like=data
            )
        }


@pytest.mark.parametrize("units", ["raw", "signal"])
def test_pick_accepts_per_frame_baseline(units):
    data = torch.zeros(7, 5, 4)
    start = ShadingEstimate(
        flat=torch.ones(5, 4),
        dark=torch.zeros(5, 4),
        baseline=torch.arange(7.0),
        meta={"baseline_units": units},
    )
    got = PicksShading().fit(data, init=start).estimates["shading"]
    assert got.allclose(start) and got is not start
    assert got.baseline_units == units
    # the spatial maps are still checked
    bad = start.replace(flat=torch.ones(3, 3))
    with pytest.raises(ValueError, match="flat"):
        PicksShading().fit(data, init=bad)


def test_basic_estimate_initialises_other_estimator():
    from optix.models import random_shading, simulate_shading
    from optix.shading import BaSiC

    truth = random_shading((32, 32), seed=0, strength=0.25)
    g = torch.Generator().manual_seed(1)
    level = 200 * (1 + 0.3 * torch.sin(torch.arange(12.0) / 3))
    x = level[:, None, None] + 800 * (
        torch.rand(12, 32, 32, generator=g) > 0.97
    )
    data = simulate_shading(x, truth)
    b = BaSiC(working_size=32, darkfield=True, timelapse=True).fit(data)
    shading = b.estimates["shading"]
    assert shading.baseline is not None and shading.baseline.shape == (12,)
    got = PicksShading().fit(data, init=b).estimates["shading"]
    assert got.allclose(shading)
    assert got.baseline_units == shading.baseline_units == "signal"
