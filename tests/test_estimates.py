import pytest
import torch

import optix
from optix.estimates import (
    Classification,
    Estimate,
    SensorEstimate,
    ShadingEstimate,
)


def _sensor():
    return SensorEstimate(
        offset=torch.full((3, 4), 100.0),
        variance=2.0,
        gain=torch.full((3, 4), 0.5),
        defective=torch.zeros(3, 4, dtype=torch.bool),
        meta={"units": "ADU"},
    )


def test_numbers_become_tensors_and_eq_is_identity():
    s = _sensor()
    assert isinstance(s.variance, torch.Tensor) and s.variance.ndim == 0
    assert s == s and s != _sensor()  # eq=False: no tensor comparison


def test_to_casts_floating_only():
    s = _sensor().to(dtype=torch.float64)
    assert s.offset.dtype == torch.float64
    assert s.defective.dtype == torch.bool
    m = s.to("meta")
    assert m.offset.device.type == "meta" and s.offset.device.type == "cpu"


def test_clone_replace_allclose():
    s = _sensor()
    c = s.clone()
    c.offset += 1
    c.meta["units"] = "e"
    assert s.offset[0, 0] == 100 and s.meta["units"] == "ADU"
    assert not s.allclose(c)
    assert s.allclose(c.replace(offset=s.offset.clone()))
    assert not s.allclose(ShadingEstimate())
    assert not s.allclose(s.replace(gain=None))
    assert s.allclose(s.replace(offset=s.offset + 1e-3), atol=1e-2)


def test_dict_round_trip():
    s = _sensor()
    d = s.to_dict()
    assert set(d) == {"offset", "variance", "gain", "defective", "meta"}
    assert SensorEstimate.from_dict(d).allclose(s)


def test_save_load(tmp_path):
    s = _sensor()
    path = tmp_path / "s.pt"
    s.save(path)
    loaded = SensorEstimate.load(path)
    assert isinstance(loaded, SensorEstimate) and loaded.allclose(s)
    assert loaded.meta == {"units": "ADU"}
    assert isinstance(Estimate.load(path), SensorEstimate)
    assert optix.load(path).allclose(s)
    with pytest.raises(TypeError):
        ShadingEstimate.load(path)


def test_save_rejects_non_json_meta(tmp_path):
    s = _sensor()
    s.meta["bad"] = object()
    with pytest.raises(TypeError, match="JSON"):
        s.save(tmp_path / "x.pt")


def test_noise_variance():
    s = _sensor()
    mean = torch.full((3, 4), 300.0)
    torch.testing.assert_close(
        s.noise_variance(mean), torch.full((3, 4), 102.0)
    )
    with pytest.raises(ValueError, match="gain"):
        s.replace(gain=None).noise_variance(mean)


def test_shading_normalize():
    flat = torch.tensor([[1.0, 4.0]])
    sh = ShadingEstimate(flat=flat, dark=torch.zeros(1, 2), meta={"a": 1})
    m = sh.normalize("mean")
    assert m.flat.mean().item() == pytest.approx(1)
    assert m.meta == {"a": 1, "flat_norm": "mean"}
    g = sh.normalize("geomean")
    assert g.flat.log().mean().item() == pytest.approx(0, abs=1e-7)
    assert torch.equal(sh.flat, flat)
    with pytest.raises(ValueError):
        sh.normalize("median")


@pytest.mark.parametrize("units", ["raw", "signal"])
@pytest.mark.parametrize("mode", ["mean", "geomean"])
def test_shading_normalize_keeps_model_consistent(units, mode):
    g = torch.Generator().manual_seed(0)
    flat = 0.5 + torch.rand(3, 4, generator=g, dtype=torch.float64)
    dark = torch.rand(3, 4, generator=g, dtype=torch.float64)
    baseline = torch.tensor([0.3, -0.7, 1.2], dtype=torch.float64)
    x = torch.rand(3, 3, 4, generator=g, dtype=torch.float64)
    sh = ShadingEstimate(
        flat=flat, dark=dark, baseline=baseline,
        meta={"baseline_units": units},
    )
    c = flat.mean() if mode == "mean" else flat.log().mean().exp()
    y = sh.shade(x)
    n = sh.normalize(mode)
    torch.testing.assert_close(n.correct(y), c * x)
    torch.testing.assert_close(n.shade(c * x), y)
    if units == "raw":
        assert torch.equal(n.baseline, baseline)
    else:
        torch.testing.assert_close(n.baseline, c * baseline)
    assert ShadingEstimate(flat=flat).normalize(mode).baseline is None


def test_classification_minimal():
    c = Classification(weights=torch.tensor([0.3, 0.7]))
    assert c.parameters is None
    assert c.clone().allclose(c)
