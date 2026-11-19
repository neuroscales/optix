"""Tests of the Biggs–Andrews accelerated Richardson–Lucy."""

import pytest
import torch
from test_deconv import blur, gaussian_psf, poisson_image

from optix.deconv import (
    BiggsAndrews,
    RichardsonLucy,
    biggs_andrews,
    richardson_lucy,
)
from optix.deconv.utils import (
    PaddedConvolution,
    biggs_andrews_alpha,
    chi2_probability,
    rl_block,
)

F64 = torch.float64


def noiseless_scene():
    psf = gaussian_psf(9, 2.0)
    truth = torch.full((64, 64), 20.0, dtype=F64)
    truth[12:52, 12:52] = 100.0
    truth[20:28, 16:48] = 600.0
    truth[36:40, 24:44] = 0.0
    return truth, blur(truth, psf), psf


def error(f, truth):
    return float((f - truth).norm())


# -- acceleration parameter (Eq. 10) and its safeguards ----------------------
def test_alpha_is_projection_clipped_to_unit_interval():
    gen = torch.Generator().manual_seed(0)
    g2 = torch.randn(50, generator=gen, dtype=F64)
    g1 = 0.6 * g2 + 0.01 * torch.randn(50, generator=gen, dtype=F64)
    expected = float((g1 * g2).sum() / (g2 * g2).sum())
    assert biggs_andrews_alpha(g1, g2) == pytest.approx(expected)
    assert 0.5 < expected < 0.7
    # α < 0 is set to 0, α > 1 to 1, and no change gives no acceleration
    assert biggs_andrews_alpha(-g1, g2) == 0.0
    assert biggs_andrews_alpha(3 * g2, g2) == 1.0
    assert biggs_andrews_alpha(g1, torch.zeros_like(g2)) == 0.0
    # float64 accumulation in a float32 call
    assert biggs_andrews_alpha(g1.float(), g2.float()) == pytest.approx(
        expected, rel=1e-6
    )


def test_trace_safeguards_and_first_prediction():
    truth, data, psf = noiseless_scene()
    op = PaddedConvolution(psf, data.shape)
    trace = {}
    f, n = rl_block(
        data, op, float(data.mean()), n_iter=50, accelerate=True, trace=trace
    )
    alpha = trace["alpha"]
    assert n == len(alpha) == 50
    # no extrapolation before two changes after h ⋆ g are known
    assert alpha[:3] == [0.0, 0.0, 0.0]
    assert all(0.0 < a <= 1.0 for a in alpha[3:])
    # the zero-valued region drives predictions negative: they are clipped
    assert trace["clipped"] > 0
    assert (f >= 0).all()
    # Eq. 12: more effective than actual iterations
    assert trace["effective"] > 2 * n


def test_negative_alpha_suspends_acceleration(monkeypatch):
    """α < 0 is set to 0 (Eq. 10), which suspends the extrapolation.

    Negative raw α values are forced deterministically by flipping the sign
    of the latest change on alternate calls, instead of relying on them to
    occur by chance on noisy data.
    """
    from optix.deconv import _rl

    real = _rl.biggs_andrews_alpha
    truth, data, psf = noiseless_scene()
    op = PaddedConvolution(psf, data.shape)

    calls = []

    def alternating(g, g_prev):
        calls.append(None)
        return real(-g if len(calls) % 2 else g, g_prev)

    monkeypatch.setattr(_rl, "biggs_andrews_alpha", alternating)
    trace = {}
    f, _ = rl_block(
        data, op, float(data.mean()), n_iter=30, accelerate=True, trace=trace
    )
    alpha = trace["alpha"][3:]
    assert any(a == 0.0 for a in alpha)
    assert any(a > 0.0 for a in alpha)
    assert (f >= 0).all()

    # α ≡ 0 at every step: exactly plain Richardson–Lucy
    monkeypatch.setattr(_rl, "biggs_andrews_alpha", lambda g, g_prev: 0.0)
    f0, _ = rl_block(data, op, float(data.mean()), n_iter=30, accelerate=True)
    f1, _ = rl_block(data, op, float(data.mean()), n_iter=30)
    torch.testing.assert_close(f0, f1)


# -- reduction to plain Richardson–Lucy -------------------------------------
@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_first_iterations_are_plain_rl(n):
    _, data = poisson_image((40, 36), gaussian_psf(7, 1.2), seed=6)
    psf = gaussian_psf(7, 1.2)
    torch.testing.assert_close(
        biggs_andrews(data, psf, n_iter=n),
        richardson_lucy(data, psf, n_iter=n),
    )


def test_acceleration_off_is_plain_rl():
    psf = gaussian_psf(7, 1.2)
    _, data = poisson_image((40, 36), psf, seed=6)
    op = PaddedConvolution(psf, data.shape)
    trace = {}
    f, _ = rl_block(data, op, float(data.mean()), n_iter=12, trace=trace)
    assert trace["alpha"] == [0.0] * 12
    assert trace["effective"] == 12
    torch.testing.assert_close(
        op.crop(f), richardson_lucy(data, psf, n_iter=12, dtype=F64)
    )
    assert not torch.allclose(
        biggs_andrews(data, psf, n_iter=12, dtype=F64), op.crop(f)
    )


# -- acceleration -----------------------------------------------------------
def test_closer_to_truth_and_fewer_iterations():
    truth, data, psf = noiseless_scene()
    rl = {
        n: error(richardson_lucy(data, psf, n_iter=n), truth)
        for n in (10, 30, 50, 150)
    }
    ba = {
        n: error(biggs_andrews(data, psf, n_iter=n), truth)
        for n in (10, 30, 50)
    }
    for n in (10, 30, 50):
        assert ba[n] < rl[n]
    # 50 accelerated iterations beat 150 plain ones
    assert ba[50] < rl[150]


def test_effective_iterations_estimate():
    """Eq. 12: RL at the estimated effective count has a similar error."""
    truth, data, psf = noiseless_scene()
    est = BiggsAndrews(psf, n_iter=30, dtype=F64).fit(data)
    out = est.apply(data)
    eff = est.history.extras["effective_iterations"][0]
    assert eff > 60
    ref = error(richardson_lucy(data, psf, n_iter=round(eff)), truth)
    assert error(out, truth) == pytest.approx(ref, rel=0.25)


def test_chi2_stop_is_reached_sooner():
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((96, 80), psf, seed=1)
    plain = RichardsonLucy(psf, dtype=F64).fit(data)
    plain.apply(data)
    fast = BiggsAndrews(psf, dtype=F64).fit(data)
    fast.apply(data)
    n = fast.history.extras["iterations"][0]
    assert n < plain.history.extras["iterations"][0]
    op = PaddedConvolution(psf, data.shape)
    start = float(data.mean())
    f, _ = rl_block(data, op, start, n_iter=int(n), accelerate=True)
    assert chi2_probability(data, op.forward(f)) >= 0.05
    f, _ = rl_block(data, op, start, n_iter=int(n) - 1, accelerate=True)
    assert chi2_probability(data, op.forward(f)) < 0.05


def test_nonnegative_on_poisson_data():
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((64, 64), psf, seed=3)
    out = biggs_andrews(data, psf, n_iter=60, block=(32, 32))
    assert (out >= 0).all()


# -- blocks -----------------------------------------------------------------
@pytest.mark.parametrize("n_iter", [5, 10])
def test_blocks_match_whole_image_2d(n_iter):
    """Each block has its own α: agreement within a few percent."""
    psf = gaussian_psf(9, 1.5)
    truth, data = poisson_image((96, 80), psf, seed=1)
    whole = biggs_andrews(data, psf, n_iter=n_iter, block=(96, 80), dtype=F64)
    parts = biggs_andrews(data, psf, n_iter=n_iter, block=(32, 24), dtype=F64)
    diff = (parts - whole).abs()
    assert float(diff.max() / whole.max()) < 0.1
    assert float(diff.mean() / whole.mean()) < 0.05
    assert error(parts, truth) == pytest.approx(error(whole, truth), rel=0.01)


def test_blocks_match_whole_image_3d():
    psf = gaussian_psf(5, 1.0, ndim=3)
    truth, data = poisson_image((20, 24, 22), psf, seed=2)
    kw = {"n_iter": 10, "dtype": F64}
    whole = biggs_andrews(data, psf, block=(20, 24, 22), **kw)
    parts = biggs_andrews(data, psf, block=(10, 12, 11), **kw)
    diff = (parts - whole).abs()
    assert float(diff.max() / whole.max()) < 0.05
    assert float(diff.mean() / whole.mean()) < 0.05
    assert error(parts, truth) == pytest.approx(error(whole, truth), rel=0.01)


def test_float32_matches_float64():
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((48, 48), psf, seed=5)
    a = biggs_andrews(data, psf, n_iter=20, dtype=F64)
    b = biggs_andrews(data.float(), psf.float(), n_iter=20, dtype="float32")
    assert float((a - b).abs().max() / a.max()) < 1e-3


# -- estimator --------------------------------------------------------------
def test_faithful_flags_and_extensions():
    psf = gaussian_psf(5, 1.0)
    assert BiggsAndrews(psf).is_faithful
    assert RichardsonLucy(psf).is_faithful
    assert BiggsAndrews(psf, n_iter=3).is_faithful is False
    assert BiggsAndrews(psf, damping=0.5).is_faithful is False
    assert "doi:10.1364/AO.36.001766" in BiggsAndrews.tags.reference
    assert "effective_iterations" not in (
        RichardsonLucy(psf, n_iter=2).fit(torch.ones(16, 16)).history.extras
    )
    # extensions compose with the acceleration
    psf = gaussian_psf(9, 1.5)
    truth, _ = poisson_image((64, 64), psf, seed=8)
    data = torch.poisson(
        blur(truth, psf) + 30.0, generator=torch.Generator().manual_seed(8)
    )
    plain = biggs_andrews(data, psf, n_iter=20)
    with_b = biggs_andrews(data, psf, n_iter=20, background=30.0)
    assert error(with_b, truth) < error(plain, truth)


def test_save_load(tmp_path):
    from optix import load

    psf = gaussian_psf(5, 1.0)
    _, data = poisson_image((24, 24), psf, seed=11)
    est = BiggsAndrews(psf, n_iter=8).fit(data)
    est.save(tmp_path / "ba.pt")
    new = load(tmp_path / "ba.pt")
    assert isinstance(new, BiggsAndrews)
    torch.testing.assert_close(new.apply(data), est.apply(data))
