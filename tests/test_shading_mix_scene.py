"""Tests of the shared-signal model of overlapping tiles."""

import json
import pathlib
import platform

import pytest
import torch
from _scene_helpers import (
    digest,
    oracle_loglik,
    oracle_moments,
    reference_fit,
)

_REFERENCE = pathlib.Path(__file__).parent / "reference"


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# -- stage 0: references -----------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2])
def test_no_overlap_is_bit_identical_to_reference(seed):
    """overlap=None: exactly the results of e7d4cf7 (before the scene model).

    The hashes were recorded on CPU with one thread; on another platform or
    torch version the low bits may differ, and only closeness is checked.
    """
    ref = json.loads((_REFERENCE / "overlap_none_e7d4cf7.json").read_text())
    fit = ref["fits"][str(seed)]
    est = reference_fit(seed=seed)
    same_platform = (
        ref["torch"] == torch.__version__
        and ref["machine"] == platform.machine()
    )
    if not same_platform:
        assert est.history.objective == pytest.approx(
            fit["objective"], rel=1e-5
        )
        flat = est.estimates["shading"].flat.double().sum().item()
        assert flat == pytest.approx(fit["flat_sum"], rel=1e-5)
        return
    assert est.history.objective == fit["objective"]
    sh, se = est.estimates["shading"], est.estimates["sensor"]
    assert digest(sh.flat) == fit["flat"]
    assert digest(se.offset) == fit["offset"]
    assert digest(se.variance) == fit["variance"]
    assert digest(est.theta_) == fit["theta"]
    assert digest(est.concentration_) == fit["concentration"]
    params = est.estimates["classification"].parameters
    assert digest(params) == fit["parameters"]


@pytest.mark.parametrize("full", [True, False])
def test_oracle_matches_quadrature_single_tile(full):
    from optix.shading.mix._gamma import GammaComponent, quadrature

    comp = GammaComponent(nodes=16, shape_bounds=(0.1, 100.0))
    f64 = dict(dtype=torch.float64)
    Y = torch.tensor([[[100.0, 105.0, 160.0, 400.0, 2000.0]]], **f64)
    C = torch.full((1, 5), 100.0, **f64)
    V = torch.full((1, 5), 9.0, **f64)
    M = torch.tensor([[1.0, 0.8, 1.2, 1.0, 0.9]], **f64)
    gain, mu, nu = 2.0, 150.0, 2.0
    logp, EU, ElogU, _ = quadrature(
        Y,
        C,
        V,
        M,
        torch.tensor(gain, **f64),
        torch.tensor(mu, **f64),
        torch.tensor(nu, **f64),
        dq=1 / 12,
        nodes=comp.nodes,
        logws=comp.logws,
        full=full,
    )
    for i in range(5):
        lp, eu, elu = oracle_moments(
            [Y[0, 0, i].item()],
            [100.0],
            [9.0],
            [M[0, i].item()],
            gain,
            mu,
            nu,
            dq=1 / 12,
            full=full,
        )
        assert logp[0, 0, i].item() == pytest.approx(lp, abs=2e-3)
        assert EU[0, 0, i].item() == pytest.approx(eu, rel=2e-3, abs=1e-3)
        assert ElogU[0, 0, i].item() == pytest.approx(elu, abs=5e-3)
    assert oracle_loglik([150.0], [100.0], [9.0], [1.0], gain, mu, nu) < 0
