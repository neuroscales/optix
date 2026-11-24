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
    sample_scene,
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


# -- stage 2: scene geometry -------------------------------------------------
def _brute_cover(pos, shape):
    """Scene array of the sorted tuples of tiles covering each voxel."""
    import numpy as np

    pos = np.asarray(pos)
    lo = pos.min(0)
    ext = pos.max(0) - lo + np.asarray(shape)
    cover = np.empty(tuple(ext.tolist()), dtype=object)
    for idx in np.ndindex(*cover.shape):
        cover[idx] = ()
    for t, p in enumerate(pos):
        box = tuple(
            slice(int(a - b), int(a - b + n))
            for a, b, n in zip(p, lo, shape, strict=True)
        )
        sub = cover[box]
        for idx in np.ndindex(*sub.shape):
            sub[idx] = sub[idx] + (t,)
    return cover, lo


_POSES = {
    "frame shifts": ([[0, 0, 0], [2, 5, 0], [3, 9, 1], [5, 14, 0]], (6, 8, 5)),
    "grid corners": (
        [[0, 0, 0], [0, 0, 5], [0, 6, 0], [0, 6, 5], [0, 3, 2]],
        (3, 8, 7),
    ),
    "components": ([[0, 0, 0], [0, 4, 0], [0, 30, 0], [0, 33, 1]], (2, 6, 4)),
    "isolated": ([[0, 0, 0], [0, 50, 0]], (2, 6, 4)),
}


@pytest.mark.parametrize("name", sorted(_POSES))
def test_scene_cells_match_voxel_brute_force(name):
    import numpy as np

    from optix.shading.mix._scene import SceneIndex

    pos, shape = _POSES[name]
    pos = np.asarray(pos)
    idx = SceneIndex(pos, shape)
    cover, lo = _brute_cover(pos, shape)
    seen = np.zeros(cover.shape, dtype=np.int64)
    for c in idx.cells:
        assert len(c.tiles) >= 2 and list(c.tiles) == sorted(c.tiles)
        box = tuple(
            slice(a - b, h - b) for a, h, b in zip(c.lo, c.hi, lo, strict=True)
        )
        assert all(s == c.tiles for s in cover[box].ravel())
        seen[box] += 1
        for t in c.tiles:  # the local slices lie in the tile
            loc = c.local(pos[t])
            assert all(
                0 <= s.start < s.stop <= n
                for s, n in zip(loc, shape, strict=True)
            )
    multi = np.vectorize(len)(cover) >= 2
    # every multiply-seen voxel in exactly one cell, nothing else
    assert np.array_equal(seen, multi.astype(np.int64))
    for t, p in enumerate(pos):
        box = tuple(
            slice(int(a - b), int(a - b + n))
            for a, b, n in zip(p, lo, shape, strict=True)
        )
        count = np.vectorize(len)(cover[box])
        assert np.array_equal(idx.cover_count(t), count)
        full = tuple(slice(0, n) for n in shape)
        assert np.array_equal(idx.single_mask(t, full), count == 1)
        # a sub-box of the mask is the mask of the sub-box
        part = (slice(1, shape[0]), slice(2, shape[1] - 1), slice(0, 3))
        assert np.array_equal(idx.single_mask(t, part), (count == 1)[part])
        assert idx.covered(t) == bool((count > 1).any())
        assert {c for c in idx.cells if t in c.tiles} == set(idx.cells_of(t))


# -- stage 3: the scene local step -------------------------------------------
_TINY_POS = [[0, 0, 0], [0, 2, 1], [1, 3, 0]]  # a 3-fold corner, a frame shift


def _tiny_fit(nclass=2, nodes=48, full=True, positions=_TINY_POS, gain=False):
    """A MixtureFit on a tiny float64 mosaic, after its initialisation."""
    import numpy as np

    from optix.io import Layout
    from optix.shading.mix import GammaMixture, TileOverlap
    from optix.shading.mix._overlap import Overlap

    pos = np.asarray(positions)
    tiles, truth = sample_scene(
        pos,
        tile=(3, 5, 4),
        seed=11,
        mu=(60.0, 300.0)[:nclass],
        nu=(3.0, 2.0)[:nclass],
        frac=(0.4, 0.3)[:nclass],
        tile_gain=[1.0, 0.8, 1.15][: len(pos)] if gain else None,
    )
    layout = Layout("txyz", spatial="yz")
    est = GammaMixture(
        nclass=nclass,
        gain=2.0,
        order=(3, 3),
        nodes=nodes,
        noise="full" if full else "constant",
        v_dof=0.0,
    )
    est.options["dtype"] = torch.float64
    spec = TileOverlap(positions=pos, axis="t", fit_gain=gain)
    sub = Layout("txyz", spatial="yz")
    ov = Overlap.build(spec, tuple(tiles.shape), sub, threshold=3.0)
    backend = dict(dtype=torch.float64, device=torch.device("cpu"))
    with optix.config(compile=False):
        model = est._model(
            tiles,
            sub,
            backend,
            alpha=est._prior_alpha(backend),
            gain=2.0,
            quantization=1.0,
            overlap=ov,
        )
        model.init()
    return model, tiles, pos, layout


def _scene_voxels(pos, shape):
    import numpy as np

    lo = pos.min(0)
    for p in pos:
        for q in np.ndindex(*shape):
            yield p - lo + np.asarray(q)


def _oracle_bound(model, tiles, pos):
    """The bound of the scene model, voxel by voxel, by dense trapezoid."""
    import math

    import numpy as np

    from optix.distributions import Dirichlet, kl_divergence

    prm, prior, post = model.param, model.prior, model.post
    elogpi = Dirichlet(post.alpha).mean_log  # (T, X, K+1)
    shape = tiles.shape[1:]
    lo = pos.min(0)
    ext = pos.max(0) - lo + np.asarray(shape)
    loga = model._log_tile_gain()
    obs = {}
    for t, p in enumerate(pos):
        for q in np.ndindex(*shape):
            z = tuple((p - lo + np.asarray(q)).tolist())
            obs.setdefault(z, []).append((t, q))
    assert len(obs) <= math.prod(ext)
    dq = model.dq
    total = 0.0
    nk = model.nclass
    for items in obs.values():
        Y = [tiles[(t, *q)].item() for t, q in items]
        C = [prm.C[q[1:]].item() for _, q in items]
        V = [prm.V[q[1:]].item() for _, q in items]
        B = [
            (loga[t].exp() * prm.M[q[1:]]).item() for t, q in items
        ]  # b = g·m
        ep = torch.stack([elogpi[t, q[0]] for t, q in items])
        logw = (torch.logsumexp(ep, 0) - math.log(len(items))).tolist()
        terms = [
            logw[nk]
            + sum(
                -0.5 * ((y - c) ** 2 + dq) / v
                - 0.5 * math.log(2 * math.pi * v)
                for y, c, v in zip(Y, C, V, strict=True)
            )
        ]
        for k in range(nk):
            mu, nu = prior.params[k].tolist()
            terms.append(
                logw[k]
                + oracle_loglik(
                    Y,
                    C,
                    V,
                    B,
                    prm.gain.item(),
                    mu,
                    nu,
                    dq=dq,
                    full=model.full,
                    n=8001,
                )  # fmt: skip
            )
        total += torch.logsumexp(torch.tensor(terms), 0).item()
    total += model.numel * math.log(model.quantization)
    kl = kl_divergence(Dirichlet(post.alpha), Dirichlet(prior.alpha))
    total -= kl.sum().item()
    total -= (
        0.5 * prior.lam * (model.bending * prm.theta.square()).sum().item()
    )
    return total


@pytest.mark.parametrize("full", [True, False])
def test_scene_bound_matches_oracle(full):
    model, tiles, pos, _ = _tiny_fit(full=full)
    with optix.config(compile=False):
        bound = model.update_latent()
    ref = _oracle_bound(model, tiles, pos)
    # Gauss-Hermite error per voxel well below 1e-3 with 48 nodes
    assert bound == pytest.approx(ref, abs=0.02)
    # every scene voxel has a total count of one
    nvox = len({tuple(z) for z in _scene_voxels(pos, tiles.shape[1:])})
    assert nvox < tiles.numel()
    assert model._counts.sum().item() == pytest.approx(nvox, abs=1e-6)


def test_scene_scores_are_finite_differences_of_the_bound():
    from optix.basis import project

    model, tiles, pos, _ = _tiny_fit(nclass=1)
    prm = model.param
    with optix.config(compile=False):
        model.update_latent()
        st = model._stats.clone()

        def bound():
            prm.M = model.modulation()
            b = model.update_latent()
            model.history.pop()
            return b

        eps = 1e-4
        for p in [(0, 0), (2, 1), (4, 3)]:  # in 1, 2 and 3 tiles
            for name, idx in (("C", 1), ("V", 6)):
                field = getattr(prm, name)
                h = eps * (1 if name == "C" else field[p].item())
                field[p] += h
                up = bound()
                field[p] -= 2 * h
                down = bound()
                field[p] += h
                num = (up - down) / (2 * h)
                assert st[idx][p].item() == pytest.approx(num, rel=2e-3)
        M = prm.M
        Sm = M * st[3] + prm.gain * M * st[8]
        g = project(Sm, [model.by, model.bx]).flatten()
        lam = model.prior.lam * model.bending.flatten()
        g = g - lam * prm.theta.flatten()
        for k in (1, 3, 4):
            d = torch.zeros_like(prm.theta).flatten()
            d[k] = 1e-5
            d = d.reshape(prm.theta.shape)
            prm.theta += d
            up = bound()
            prm.theta -= 2 * d
            down = bound()
            prm.theta += d
            num = (up - down) / 2e-5
            assert g[k].item() == pytest.approx(num, rel=2e-3, abs=1e-3)


# -- stage 4: tile gains -----------------------------------------------------
def test_gain_score_is_the_derivative_of_the_bound():
    model, *_ = _tiny_fit(nclass=1, gain=True)
    prm = model.param
    assert prm.A is not None and bool(model._fitted_tiles.all())
    with optix.config(compile=False):
        model.update_latent()
        s = model._gains["s"].clone()
        for t in range(3):
            h = 1e-5
            prm.A[t] += h
            up = model.update_latent()
            prm.A[t] -= 2 * h
            down = model.update_latent()
            prm.A[t] += h
            num = (up - down) / (2 * h)
            assert s[t].item() == pytest.approx(num, rel=2e-3, abs=1e-3)


def test_recentring_leaves_the_bound_unchanged():
    model, *_ = _tiny_fit(nclass=2, gain=True)
    prm = model.param
    prm.A = torch.tensor([0.3, -0.1, 0.25], dtype=torch.float64)
    with optix.config(compile=False):
        before = model.update_latent()
        shift = model._recentre()
        after = model.update_latent()
    assert shift == pytest.approx(0.15)
    assert abs(prm.A.sum().item()) < 1e-12
    assert after == pytest.approx(before, rel=1e-11, abs=1e-8)


def test_isolated_tiles_keep_unit_gain():
    pos = [[0, 0, 0], [0, 2, 1], [0, 40, 0]]  # tile 2 overlaps nothing
    model, *_ = _tiny_fit(nclass=1, gain=True, positions=pos)
    assert model._fitted_tiles.tolist() == [True, True, False]
    with optix.config(compile=False):
        for _ in range(3):
            model.update_latent()
            model.update_post()
            model.update_prior()
            model.update_parameters()
    a = model.param.A
    assert a[2].item() == 0
    assert abs(a[:2].sum().item()) < 1e-10
    assert a[0].item() != 0


@pytest.mark.parametrize("fade", [0.05, 0.1])
def test_gain_step_reaches_the_exact_fixed_point(fade, monkeypatch):
    """The default step (no gain–c, v coupling) against the exact oracle."""
    from test_shading_mix import _tile_fit, _tiles

    from optix.shading.mix import TileOverlap, _model

    data, logm = _tiles(seed=0, X=24, bright=0.0, fade=fade)
    fits = []
    for exact in (False, True):
        orig = _model.MixtureFit.__init__

        def init(self, *a, orig=orig, exact=exact, **k):
            orig(self, *a, **k)
            self.exact_gain = exact

        monkeypatch.setattr(_model.MixtureFit, "__init__", init)
        ov = TileOverlap(step=(0, 16, 0), fit_gain=True)
        fits.append(_tile_fit(data, ov, it=60))
        monkeypatch.undo()
    a, b = fits
    # measured: same iterations, bound within 0.2, gains within 0.05 %
    assert "decrement" in a.history.reason and "decrement" in b.history.reason
    assert abs(a.history.objective[-1] - b.history.objective[-1]) < 1.0
    assert torch.allclose(a.tile_gain_, b.tile_gain_, rtol=2e-3)
    fa, fb = (f.estimates["shading"].flat.double().log() for f in fits)
    assert (fa - fb).abs().max() < 5e-3


def test_coarse_start_option():
    from test_shading_mix import _tile_fit, _tiles

    from optix.shading.mix import TileOverlap

    spec = TileOverlap(step=(0, 16, 0), fit_gain=True, coarse=4)
    assert "coarse=4" in repr(spec)
    assert spec != TileOverlap(step=(0, 16, 0), fit_gain=True)
    with pytest.raises(ValueError, match="coarse"):
        TileOverlap(step=(0, 1), coarse=1)
    data, logm = _tiles(seed=0, X=12, bright=0.0, fade=0.1)
    est = _tile_fit(data, spec, it=5)
    assert est.tile_gain_.shape == (4,)
    assert torch.isfinite(torch.tensor(est.history.objective)).all()


# -- stage 5: options --------------------------------------------------------
@pytest.fixture(scope="module")
def _gain_fit():
    from test_shading_mix import _tile_fit, _tiles

    from optix.shading.mix import TileOverlap

    data, _ = _tiles(seed=0, X=6, bright=0.0, fade=0.1)
    est = _tile_fit(data, TileOverlap(step=(0, 16, 0), fit_gain=True), it=8)
    return est, data


def test_apply_divides_by_tile_gain(_gain_fit):
    est, data = _gain_fit
    plain = est.apply(data)
    eq = est.apply(data, tile_gain=True)
    g = est.tile_gain_
    for t in range(4):
        assert torch.allclose(eq[t], plain[t] / g[t], rtol=1e-5, atol=1e-4)
    # without fitted gains
    est2 = type(est)(**{**est.options, "overlap": None})
    est2.estimates = est.estimates
    est2.tile_gain_ = None
    with pytest.raises(ValueError, match="fit_gain"):
        est2.apply(data, tile_gain=True)


def test_joint_classify_shares_the_posterior(_gain_fit):
    est, data = _gain_fit
    p0, _ = est.classify(data)
    pj, uj = est.classify(data, joint=True, mean=True)
    assert pj.shape == data.shape and uj.shape == data.shape
    assert ((pj >= 0) & (pj <= 1)).all()
    # tile t's rows 16: are tile t+1's rows :16 (step 16 along y)
    for t in range(3):
        assert torch.equal(pj[t, :, 16:], pj[t + 1, :, :16])
        assert torch.allclose(uj[t, :, 16:], uj[t + 1, :, :16])
    # not the per-picture posterior there
    assert not torch.equal(pj[0, :, 16:], p0[0, :, 16:])
    per, _ = est.classify(data, joint=True, per_class=True)
    assert per.shape == (1, *data.shape)
    assert torch.allclose(per.sum(0), pj, atol=1e-5)


# -- stage 6: regression from the experiments ------------------------------
def test_low_snr_overlap_does_not_bias_the_flat_field():
    """Stationary scene, signal ≈ 8 ADU (excess ≈ 1-2 sd of the noise).

    Measured (seeds 0-1): no overlap 0.028, log-ratio term (e7d4cf7)
    0.046 with a y-ramp of −0.07, shared-signal model 0.027, ramp −0.008.
    """
    import math

    from test_shading_mix import _tile_fit, _tiles

    from optix.shading.mix import TileOverlap

    data, logm = _tiles(seed=0, X=24, bright=0.0, amp=8.0)
    errs, ramps = [], []
    for ov in (None, TileOverlap(step=(0, 16, 0), axis="t")):
        d = _tile_fit(data, ov, it=40).estimates["shading"].flat.double()
        d = d.log() - logm
        y = torch.linspace(-1, 1, d.shape[0], dtype=torch.float64)
        ramps.append(((d.mean(1) - d.mean()) * y).sum().item() / (y**2).sum())
        errs.append(d.abs().mean().item())
    assert errs[1] < errs[0] + 0.005
    assert abs(ramps[1]) < 0.03
    assert math.isfinite(errs[1])
