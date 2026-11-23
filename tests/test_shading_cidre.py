import numpy as np
import pytest
import torch

import optix
from optix.io import Layout, labeled
from optix.models import random_shading, simulate_shading
from optix.shading import CIDRE
from optix.shading._cidre import (
    _odd_pad,
    _working_shape,
    lambda_r_schedule,
    normalized_log,
)


def make_stack(n=500, shape=(48, 48), *, dark=40.0, seed=0, dtype=None):
    """Uniform random intensities in [100, 1000], shaded, with a dark."""
    dtype = torch.float64 if dtype is None else dtype
    truth = random_shading(
        shape, seed=seed, strength=0.3, dark=dark, dtype=torch.float64
    )
    g = torch.Generator().manual_seed(seed + 1)
    x = 100 + 900 * torch.rand(n, *shape, generator=g, dtype=torch.float64)
    return simulate_shading(x, truth).to(dtype), truth


@pytest.fixture(scope="module")
def fitted():
    data, truth = make_stack(n=1000, seed=2)
    return CIDRE(dtype=torch.float64).fit(data), data, truth


# -- building blocks -------------------------------------------------------
def test_lambda_r_schedule():
    assert lambda_r_schedule(1) == 9.5
    assert lambda_r_schedule(200) == 6.0
    assert lambda_r_schedule(5000) == 6.0
    assert lambda_r_schedule(100) == pytest.approx(9.5 - 3.5 * 99 / 199)
    values = [lambda_r_schedule(n) for n in range(1, 202)]
    assert all(a > b for a, b in zip(values[:199], values[1:200], strict=True))
    with pytest.raises(ValueError):
        lambda_r_schedule(0)
    c = CIDRE(max_iterations=2).fit(make_stack(n=50, shape=(16, 16))[0])
    assert c.lambda_r_ == pytest.approx(lambda_r_schedule(50))
    assert CIDRE(lambda_r=3.0, max_iterations=2).fit(
        make_stack(n=50, shape=(16, 16))[0]
    ).lambda_r_ == pytest.approx(3.0)


def test_working_shape():
    assert _working_shape(128, (256, 512)) == (64, 128)
    assert _working_shape(128, (100, 120)) is None
    assert _working_shape((32, 40), (64, 80)) == (32, 40)
    assert _working_shape(None, (64, 80)) is None


def test_normalized_log():
    y, x = torch.meshgrid(
        torch.arange(20.0, dtype=torch.float64),
        torch.arange(24.0, dtype=torch.float64),
        indexing="ij",
    )
    for sigma in (0.5, 1.0, 3.0, 30.0):
        # constant and linear maps are not penalised, borders included
        for f in (torch.ones_like(x), 0.3 * x - 0.2 * y + 1):
            assert normalized_log(f, sigma).abs().max() < 1e-10
    # interior response to a quadratic: σ² ∇²(x² + y²) = 4 σ²
    out = normalized_log(x**2 + y**2, 2.0)
    torch.testing.assert_close(
        out[8:12, 8:12], torch.full((4, 4), 16.0, dtype=torch.float64)
    )
    padded = _odd_pad(torch.arange(3.0)[None], 7, 1)
    torch.testing.assert_close(padded[0], torch.arange(-7.0, 10.0))


# -- fit -------------------------------------------------------------------
def test_recovers_flat_and_dark(fitted):
    c, _, truth = fitted
    assert c.is_faithful and c.history.converged
    est = c.estimates["shading"]
    assert est.flat.shape == (48, 48) and est.dark.shape == (48, 48)
    assert est.meta["estimator"] == "CIDRE"
    assert est.meta["flat_norm"] == "none"
    flat = est.flat / est.flat.mean()
    err = (flat - truth.flat).abs()
    assert err.mean() < 0.012 and err.max() < 0.08
    # the zero-light point and the mean dark field (truth 40)
    ZQ, Zq = c.zero_light_.tolist()
    assert 30 < ZQ < 45 and 30 < Zq < 45
    assert abs(float(est.dark.mean()) - 40) < 6
    assert c.target_.shape == (100,)


def test_energy_decreases(fitted):
    c, _, _ = fitted
    energy = -torch.tensor(c.history.objective)
    assert len(energy) > 5
    assert (energy[1:] <= energy[:-1] + 1e-12).all()
    assert energy[-1] < 0.5 * energy[0]
    terms = c.history.extras
    for name in ("Ef", "Er", "Ez", "Eb", "Z_Q", "Z_q"):
        assert len(terms[name]) == len(energy)


def test_correction_modes(fitted):
    c, data, truth = fitted
    est = c.estimates["shading"]
    x = (data - truth.dark) / truth.flat
    vbar, zbar = est.flat.mean(), est.dark.mean()
    direct = c.apply(data, mode="direct")
    torch.testing.assert_close(direct, (data - est.dark) / est.flat)
    dynamic = c.apply(data)
    torch.testing.assert_close(dynamic, direct * vbar)
    zero = c.apply(data, mode="zero_light")
    torch.testing.assert_close(zero, direct * vbar + zbar)
    # the corrected images are proportional to the unshaded ones
    ratio = dynamic.mean(0) / x.mean(0)
    assert ratio.std() / ratio.mean() < 0.02
    raw = data.mean(0) / x.mean(0)
    assert raw.std() / raw.mean() > 0.2
    with pytest.raises(ValueError, match="mode"):
        c.apply(data, mode="nope")


@pytest.mark.filterwarnings("ignore:Histogram state")
@pytest.mark.parametrize("method", ["exact", "histogram"])
def test_streaming_parity(method):
    data, _ = make_stack(n=120, shape=(40, 36))
    kw = {"working_size": 24, "quantile_method": method}
    whole = CIDRE(**kw).fit(data)
    with optix.config(memory=40 * 36 * 8 * 3 * 7):  # batches of a few frames
        chunked = CIDRE(**kw).fit(data.numpy())
    a, b = whole.estimates["shading"], chunked.estimates["shading"]
    torch.testing.assert_close(a.flat, b.flat.to(a.flat))
    torch.testing.assert_close(a.dark, b.dark.to(a.dark))


def test_histogram_sketch_close_to_exact():
    data, _ = make_stack(n=120, shape=(40, 36))
    exact = CIDRE(working_size=24, quantile_method="exact").fit(data)
    sketch = CIDRE(working_size=24, quantile_method="histogram").fit(data)
    a, b = exact.estimates["shading"], sketch.estimates["shading"]
    assert (a.flat - b.flat).abs().max() < 0.03
    # the two quantile definitions differ by up to half a rank
    rel = (exact.target_ - sketch.target_).abs() / exact.target_
    assert rel.median() < 0.005 and rel.max() < 0.05


def test_float32_matches_float64():
    data, _ = make_stack(n=200, shape=(32, 32))
    f64 = CIDRE(dtype=torch.float64).fit(data)
    f32 = CIDRE(dtype=torch.float32).fit(data.float())
    flat32 = f32.estimates["shading"].flat
    assert flat32.dtype == torch.float32
    torch.testing.assert_close(
        flat32.double(), f64.estimates["shading"].flat, atol=2e-3, rtol=0
    )


def test_layout_groups_and_spatial_order():
    zarr = pytest.importorskip("zarr")
    stacks, flats = [], []
    for c in range(2):
        data, truth = make_stack(n=300, shape=(32, 40), seed=c)
        stacks.append(data)
        flats.append(truth.flat)
    y = torch.stack(stacks, 1).numpy().astype("f4")  # (t, c, y, x)
    Z = zarr.create_array(
        store={}, shape=y.shape, chunks=(16, 1, 32, 40), dtype="f4"
    )
    Z[:] = y
    layout = Layout("tcyx", spatial="xy", group="c")
    c = CIDRE().fit(labeled(Z, layout))
    est = c.estimates["shading"]
    assert est.flat.shape == (2, 40, 32) and c.zero_light_.shape == (2, 2)
    for g in range(2):
        flat = est.flat[g].T.double()
        err = (flat / flat.mean() - flats[g]).abs().mean()
        assert err < 0.03
    out = c.apply(Z, out=np.zeros(y.shape, "f4"), mode="direct")
    ref = (
        y - est.dark.numpy().transpose(0, 2, 1)[None]
    ) / est.flat.numpy().transpose(0, 2, 1)[None]
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-3)


def test_options_and_errors():
    c = CIDRE()
    assert c.is_faithful
    assert c.options["lambda_z"] == 0.5 and c.options["lambda_r"] is None
    assert c.options["quantiles"] == 100 and c.options["working_size"] == 128
    assert not CIDRE(normalize_fit=True).is_faithful
    for name in ("quantile_method", "init_method"):
        with pytest.raises(ValueError, match=name):
            CIDRE(**{name: "nope"})
    with pytest.raises(ValueError, match="band"):
        CIDRE(band=0.7)
    with pytest.raises(RuntimeError, match="not fitted"):
        CIDRE().apply(np.zeros((2, 8, 8)))
    with pytest.raises(ValueError, match="two spatial"):
        CIDRE().fit(np.zeros((4, 3, 8, 8)), layout=Layout.default(4, 3))


def test_init_and_limits():
    data, truth = make_stack(n=200, shape=(24, 24))
    a = CIDRE().fit(data).estimates["shading"]
    b = CIDRE(init_method="least_squares").fit(data).estimates["shading"]
    assert (a.flat - b.flat).abs().max() < 1e-3
    c = CIDRE(zero_light_limits=(0.0, 5.0)).fit(data)
    assert c.zero_light_.max() < 5.5
