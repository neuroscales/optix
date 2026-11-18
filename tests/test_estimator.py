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

    def _fit(self, Y, *, init, callback, verbose):
        self._warn_unused(init, ShadingEstimate)
        start = self._pick(init, ShadingEstimate, shape=Y.shape[-2:], like=Y)
        (m,) = reduce(Y, Moments(1), dtype=self.dtype, device=self.device)
        dark = m.mean.to(self._backend(Y)["dtype"]) * self.options["scale"]
        dark = dark + self.options["bias"]
        if start is not None and start.dark is not None:
            dark = (dark + start.dark) / 2
        self.estimates = {"shading": ShadingEstimate(dark=dark)}
        self.history.record(1.0)
        self.history.record(1.0)
        self.history.check(1e-6)
        if callback is not None:
            callback(self)

    def _apply(self, Y, out):
        dark = self.estimates["shading"].dark

        def fn(index, data):
            return data - dark.to(data)

        return self._apply_frames(Y, out, fn)


class Blur(Estimator):
    tags = Tags(targets=("test",), streaming="halo")

    def _apply(self, Y, out):
        return self._apply_blocks(Y, out, lambda d: d, halo=1)


@pytest.fixture
def data():
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
def test_fit_apply_round_trip(data, kind, tmp_path):
    if kind == "zarr":
        zarr = pytest.importorskip("zarr")
        Y = zarr.create_array(
            store={}, shape=data.shape, chunks=(1, 2, 5, 4), dtype="f4"
        )
        Y[:] = data
        out = zarr.create_array(
            store={}, shape=data.shape, chunks=(1, 3, 5, 4), dtype="f4"
        )
    elif kind == "tensor":
        Y = torch.from_numpy(data)
        out = torch.empty(data.shape)
    else:
        Y = data
        out = np.empty_like(data)
    e = MeanDark().fit(Y)
    dark = e.estimates["shading"].dark
    np.testing.assert_allclose(dark.numpy(), data.mean((0, 1)), rtol=1e-6)
    assert e.estimates["shading"].meta["estimator"] == "MeanDark"
    assert e.estimates["shading"].meta["faithful"] is True
    assert e.history.converged
    res = e.apply(Y, out=out)
    assert res is out
    np.testing.assert_allclose(
        np.asarray(res[:]), data - data.mean((0, 1)), atol=1e-4
    )
    # default output: in memory, on the CPU for non-tensor inputs
    default = e.apply(Y)
    assert isinstance(default, torch.Tensor) and default.device.type == "cpu"


def test_history_reset_on_each_fit(data):
    e = MeanDark().fit(data)
    first = e.history
    assert first.objective == [1.0, 1.0] and first.converged
    e.fit(data)
    assert e.history.objective == [1.0, 1.0] and e.history.converged
    assert e.history is not first
    # a fit that fails does not keep the previous trace either
    e.history.record(5.0)
    with pytest.raises(ValueError, match="fewer than"):
        e.fit(np.zeros((2,), dtype="f4"))
    assert 5.0 not in e.history.objective


def test_apply_frames_in_place(data):
    y = data.copy()
    e = MeanDark().fit(y)
    e.apply(y, out=y)
    np.testing.assert_allclose(y, data - data.mean((0, 1)), atol=1e-4)


def test_apply_blocks_refuses_in_place():
    y = torch.randn(4, 4)
    with pytest.raises(ValueError, match="In-place"):
        Blur().apply(y, out=y)
    with pytest.raises(ValueError, match="In-place"):
        Blur().apply(y, out=y[:])
    out = Blur().apply(y, out=torch.empty(4, 4))
    torch.testing.assert_close(out, y)


def test_init_is_cloned_and_unusable_warns(data):
    start = ShadingEstimate(dark=torch.zeros(5, 4))
    with pytest.warns(RuntimeWarning, match="SensorEstimate"):
        e = MeanDark().fit(data, init=[start, SensorEstimate(offset=1.0)])
    assert start.dark.eq(0).all()
    np.testing.assert_allclose(
        e.estimates["shading"].dark.numpy(), data.mean((0, 1)) / 2, rtol=1e-6
    )
    # a fitted estimator can be the init
    MeanDark().fit(data, init=e)
    with pytest.raises(ValueError, match="broadcast"):
        MeanDark().fit(data, init=ShadingEstimate(dark=torch.zeros(3, 3)))


def test_callback_and_no_grad(data):
    seen = []
    y = torch.from_numpy(data).requires_grad_()
    e = MeanDark().fit(y, callback=seen.append)
    assert seen == [e]
    assert not e.estimates["shading"].dark.requires_grad


def test_to(data):
    e = MeanDark().fit(data).to(dtype=torch.float64)
    assert e.estimates["shading"].dark.dtype == torch.float64
    assert e.dtype == torch.float64
    e.to("meta")
    assert e.estimates["shading"].dark.device.type == "meta"


def test_save_load(data, tmp_path):
    e = MeanDark(bias=2.0).fit(data)
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

    def _fit(self, Y, *, init, callback, verbose):
        shape = self._map_shape(Y)
        dark = torch.zeros(shape, **self._backend(Y))
        groups = reduce_groups(
            Y, Moments(1), batch=3, dtype=self.dtype, device=self.device
        )
        for key, (m,) in groups.items():
            dark[key] = m.mean.to(dark)
        self.estimates = {"shading": ShadingEstimate(dark=dark)}

    def _apply(self, Y, out):
        dark = self.estimates["shading"].dark

        def fn(chunk, data):
            return data - dark[chunk.group].to(data)

        return self._apply_frames(Y, out, fn)


def test_estimator_with_layout_end_to_end(tmp_path):
    zarr = pytest.importorskip("zarr")
    g = np.random.default_rng(1)
    y = (100 + g.normal(size=(2, 3, 4, 5, 6))).astype("f4")  # t c z y x
    Y = zarr.create_array(
        store={}, shape=y.shape, chunks=(1, 1, 2, 5, 6), dtype="f4"
    )
    Y[:] = y
    layout = Layout("tczyx", spatial="xy", group="c")
    ref = y.mean((0, 2)).transpose(0, 2, 1)  # (C, X, Y)
    for e in (
        GroupDark(layout=layout).fit(Y),
        GroupDark().fit(Y, layout=layout),
        GroupDark().fit(labeled(Y, layout)),
    ):
        assert e.layout == layout
        dark = e.estimates["shading"].dark
        assert dark.shape == (3, 6, 5)
        np.testing.assert_allclose(dark.numpy(), ref, rtol=1e-5)
    # apply uses the fitted layout for a plain array; zarr out round trip
    out = zarr.create_array(
        store={}, shape=y.shape, chunks=(2, 1, 4, 5, 3), dtype="f4"
    )
    assert e.apply(Y, out=out) is out
    expected = y - y.mean((0, 2), keepdims=True)
    np.testing.assert_allclose(out[:], expected, atol=1e-4)
    # the layout survives save / load
    e.save(tmp_path / "e.pt")
    loaded = optix.load(tmp_path / "e.pt")
    assert loaded.layout == layout
    res = loaded.apply(y)
    np.testing.assert_allclose(res.numpy(), expected, atol=1e-4)
    with pytest.raises(ValueError, match="conflicts"):
        GroupDark().fit(labeled(Y, layout), layout=Layout.default(5, 2))
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
    Y = labeled(y, axes="zyx", spatial="xz")
    with pytest.raises(ValueError, match="In-place"):
        Blur().apply(Y, out=Y)
    torch.testing.assert_close(Blur().apply(Y, out=torch.empty(4, 3, 5)), y)


class DenseBlur(Estimator):
    tags = Tags(targets=("test",), streaming="halo")

    def _apply(self, Y, out):
        def fn(d):
            assert d.is_contiguous()
            return d * 1

        return self._apply_blocks(Y, out, fn, halo=1, contiguous=True)


def test_apply_blocks_contiguous_option():
    y = np.random.default_rng(2).normal(size=(4, 5, 6)).astype("f4")
    Y = labeled(y, axes="zyx", spatial="yz")
    out = DenseBlur().apply(Y, out=np.zeros_like(y))
    np.testing.assert_allclose(out, y)
