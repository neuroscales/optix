import numpy as np
import pytest
import torch

import optix
from optix import Estimator, History, Tags
from optix.estimates import SensorEstimate, ShadingEstimate
from optix.stats import Moments, reduce


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
