"""Tests of the shared-signal model of overlapping tiles."""

import json
import os
import pathlib
import platform
import warnings

import pytest
import torch
from _scene_helpers import (
    digest,
    oracle_loglik,
    oracle_moments,
    reference_fit,
)

import optix

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


# -- stage 1: the shared-signal kernel ---------------------------------------
_F64 = dict(dtype=torch.float64)


def _kernel_args(S, n, seed, amp=1.0, read=3.0, disagree=False, full=True):
    gen = torch.Generator().manual_seed(seed)
    mu, nu, gam = 150.0, 2.0, 2.0
    u = torch._standard_gamma(torch.full((n,), nu, **_F64), generator=gen)
    u = u * mu / nu * amp
    b = 0.7 + 0.6 * torch.rand((S, n), generator=gen, **_F64)
    c = 100 + torch.randn((S, n), generator=gen, **_F64)
    v = read**2 * (0.5 + torch.rand((S, n), generator=gen, **_F64))
    var = v + (gam * b * u if full else 0)
    y = c + b * u + var.sqrt() * torch.randn((S, n), generator=gen, **_F64)
    if disagree:  # the first tile sees background, the others tissue
        y[0] = c[0] + v[0].sqrt() * torch.randn(n, generator=gen, **_F64)
    return y, c, v, b, gam, mu, nu


def _shared(y, c, v, b, gam, mu, nu, full, nodes=24, compile=False):
    from optix.shading.mix._gamma import GammaComponent, quadrature_shared

    comp = GammaComponent(nodes=nodes, shape_bounds=(0.1, 100.0))

    def sh(t):
        return t[:, None, None, :]

    with optix.config(compile=compile):
        return quadrature_shared(
            sh(y),
            sh(c),
            sh(v),
            sh(b),
            torch.tensor(gam, **_F64),
            torch.tensor(mu, **_F64),
            torch.tensor(nu, **_F64),
            dq=1 / 12,
            nodes=comp.nodes,
            logws=comp.logws,
            full=full,
        )


@pytest.mark.parametrize("full", [True, False])
def test_shared_kernel_one_tile_is_the_single_tile_kernel(full):
    from optix.shading.mix._gamma import GammaComponent, quadrature

    y, c, v, b, gam, mu, nu = _kernel_args(1, 64, seed=3, full=full)
    L, EU, ElogU, Emom = _shared(y, c, v, b, gam, mu, nu, full)
    comp = GammaComponent(nodes=24, shape_bounds=(0.1, 100.0))
    with optix.config(compile=False):
        L1, EU1, ElogU1, Emom1 = quadrature(
            y[0][None, None],
            c[0][None],
            v[0][None],
            b[0][None],
            torch.tensor(gam, **_F64),
            torch.tensor(mu, **_F64),
            torch.tensor(nu, **_F64),
            dq=1 / 12,
            nodes=comp.nodes,
            logws=comp.logws,
            full=full,
        )
    # the same rule, centred on a mode found by another route
    assert torch.allclose(L[0], L1[0], atol=1e-5, rtol=0)
    assert torch.allclose(EU[0], EU1[0], rtol=1e-5)
    assert torch.allclose(ElogU[0], ElogU1[0], atol=1e-5)
    assert torch.allclose(Emom[:, 0, 0], Emom1[:, 0], rtol=1e-4, atol=1e-9)


@pytest.mark.parametrize("full", [True, False])
@pytest.mark.parametrize("S", [2, 3])
@pytest.mark.parametrize(
    "amp, read, disagree",
    [
        (1.0, 3.0, False),
        (0.05, 3.0, False),
        (0.02, 15.0, False),
        (1.0, 3.0, True),
    ],  # fmt: skip
)
def test_shared_kernel_matches_oracle(full, S, amp, read, disagree):
    y, c, v, b, gam, mu, nu = _kernel_args(
        S, 40, seed=S, amp=amp, read=read, disagree=disagree, full=full
    )
    L, EU, ElogU, _ = _shared(y, c, v, b, gam, mu, nu, full)
    for i in range(y.shape[1]):
        lp, eu, elu = oracle_moments(
            y[:, i].tolist(),
            c[:, i].tolist(),
            v[:, i].tolist(),
            b[:, i].tolist(),
            gam,
            mu,
            nu,
            dq=1 / 12,
            full=full,
            n=8001,
        )
        # Gauss-Hermite in log u: within ~1e-3 for ν = 2 (as one tile)
        assert L[0, 0, i].item() == pytest.approx(lp, abs=2e-3)
        assert EU[0, 0, i].item() == pytest.approx(eu, rel=2e-3)
        assert ElogU[0, 0, i].item() == pytest.approx(elu, abs=5e-3)


@pytest.mark.parametrize("full", [True, False])
def test_shared_kernel_statistics_are_fisher_scores(full):
    from optix.shading.mix._model import G1, G2, GC, GV

    y, c, v, b, gam, mu, nu = _kernel_args(3, 6, seed=5, amp=0.3, full=full)
    _, _, _, Emom = _shared(y, c, v, b, gam, mu, nu, full)
    eps = 1e-4
    for i in range(y.shape[1]):
        args = [y[:, i].tolist(), c[:, i].tolist(), v[:, i].tolist()]
        args.append(b[:, i].tolist())

        def f(j, which, h, args=args):
            a = [list(x) for x in args]
            a[which][j] += h
            return oracle_loglik(*a, gam, mu, nu, dq=1 / 12, full=full)

        for t in range(3):
            scale = [1.0, 1.0, v[t, i].item(), b[t, i].item()]
            num = [
                (f(t, w, eps * scale[w]) - f(t, w, -eps * scale[w]))
                / (2 * eps * scale[w])
                for w in (1, 2, 3)
            ]
            dc = Emom[GC, t, 0, 0, i].item()
            dv = Emom[GV, t, 0, 0, i].item()
            db = Emom[G1, t, 0, 0, i].item()
            if full:
                db += gam * Emom[G2, t, 0, 0, i].item()
            # ∂/∂c log p = E[r/σ²], ∂/∂v = E[q], ∂/∂b = E[u·ψ₁ + γu·ψ₂]
            assert dc == pytest.approx(num[0], rel=5e-3, abs=1e-4)
            assert dv == pytest.approx(num[1], rel=5e-3, abs=1e-5)
            assert db == pytest.approx(num[2], rel=5e-3, abs=1e-4)


@pytest.mark.skipif(
    not os.environ.get("OPTIX_TEST_COMPILE"),
    reason="slow (minutes); set OPTIX_TEST_COMPILE=1",
)
def test_shared_kernel_compiled_matches_eager():
    y, c, v, b, gam, mu, nu = _kernel_args(2, 32, seed=7)
    eager = _shared(y, c, v, b, gam, mu, nu, True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no silent fallback to eager
        compiled = _shared(y, c, v, b, gam, mu, nu, True, compile=True)
    for a, e in zip(compiled, eager, strict=True):
        assert torch.allclose(a, e, rtol=1e-9, atol=1e-12)
