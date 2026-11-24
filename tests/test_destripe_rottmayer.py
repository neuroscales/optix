"""Variational stripe removal (Rottmayer, Redenbach and Fahrbach 2025)."""

import math

import numpy as np
import pytest
import torch

import optix
from optix.destripe import (
    GeneralStripeRemover,
    general_stripe_remover,
    utils,
    wavelet_fft,
)
from optix.io import Layout, labeled
from optix.models import random_shading

F64 = torch.float64


def make_frames(n=2, shape=(48, 40), *, seed=0, dtype=F64):
    """Smooth shading plus sharp discs, in [0.1, 0.9]: content only."""
    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(
        torch.arange(shape[0], dtype=dtype),
        torch.arange(shape[1], dtype=dtype),
        indexing="ij",
    )
    frames = []
    for i in range(n):
        flat = random_shading(shape, seed=seed + i, dtype=dtype).flat
        frame = 0.2 + 0.2 * (flat - flat.min()) / (flat.max() - flat.min())
        for _ in range(3):
            c = torch.rand(2, generator=g, dtype=F64).to(dtype)
            disc = (
                (y - c[0] * shape[0]).square() + (x - c[1] * shape[1]).square()
            ) < 30
            frame = frame + 0.3 * disc.to(dtype)
        frames.append(frame)
    return torch.stack(frames)


def make_stripes(n, shape, *, seed=1, angle=0, dtype=F64):
    """Ideal stripes along dim -2 (angle 0) or along the 45° diagonal."""
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(n, *shape, dtype=dtype)
    y, x = torch.meshgrid(
        torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij"
    )
    across = x if angle == 0 else x - y
    for i in range(n):
        for _ in range(6):
            x0 = int(
                torch.randint(-shape[0] // 2, shape[1] - 4, (1,), generator=g)
            )
            w = int(torch.randint(1, 4, (1,), generator=g))
            a = 0.1 * float(torch.randn(1, generator=g, dtype=F64))
            out[i] += a * ((across >= x0) & (across < x0 + w)).to(dtype)
    return out


def rmse(a, b):
    return float((a - b).square().mean().sqrt())


def objective(u, data, **kw):
    """Eq. (1) in the normalised units of the solver."""
    dims = tuple(range(1, data.ndim))
    lo = data.amin(dims, keepdim=True)
    scale = data.amax(dims, keepdim=True) - lo
    return utils.general_stripe_objective(
        (u - lo) / scale, (data - lo) / scale, **kw
    )


# -- operators ----------------------------------------------------------------
@pytest.mark.parametrize(
    "offset",
    [(1, 0), (0, 1), (2, 1), (1, -1), (-2, 1), (1, 2), (1, 0, 0), (1, 1, 1)],
)
def test_lattice_diff_adjoint(offset):
    g = torch.Generator().manual_seed(0)
    shape = (2, 5, 7, 6)[-len(offset) - 1 :]
    x = torch.randn(shape, generator=g, dtype=F64)
    p = torch.randn(shape, generator=g, dtype=F64)
    lhs = (utils.lattice_diff(x, offset) * p).sum()
    rhs = (x * utils.lattice_diff_adjoint(p, offset)).sum()
    assert float(lhs) == pytest.approx(float(rhs), abs=1e-12)
    # D_v x[i] = x[i + v] − x[i] in the grid, 0 outside
    d = utils.lattice_diff(x, offset)
    if offset == (2, 1):
        torch.testing.assert_close(d[:, 1, 3], x[:, 3, 4] - x[:, 1, 3])
        assert (d[:, -2:] == 0).all() and (d[:, :, -1] == 0).all()
    # ‖D_v‖² ≤ 4, used for the step size
    assert float(d.square().sum()) <= 4 * float(x.square().sum())


def test_stripe_offset():
    assert utils.stripe_offset(0) == (1, 0)
    assert utils.stripe_offset(180) == (1, 0)
    assert utils.stripe_offset(26.6) == (2, 1)
    assert utils.stripe_offset(20) == (2, 1)
    assert utils.stripe_offset(45) == (1, 1)
    assert utils.stripe_offset(-45) == (-1, 1)  # same line as (1, -1)
    assert utils.stripe_offset(60) == (1, 2)
    assert utils.stripe_offset(90) == (0, 1)
    assert utils.stripe_offset(-10) == (1, 0)
    with pytest.raises(ValueError, match="finite"):
        utils.stripe_offset(math.nan)


# -- the model ----------------------------------------------------------------
def test_removes_stripes_and_keeps_content():
    content = make_frames(2)
    stripes = make_stripes(2, content.shape[-2:])
    # the default μ₁ = 1/3 also smooths the shading of these frames a
    # little; the printed μ₁ = 1/30 suits their weak, ideal stripes
    default = general_stripe_remover(content + stripes, max_iter=1500)
    assert rmse(default, content) < 0.4 * rmse(content + stripes, content)
    out = general_stripe_remover(content + stripes, mu1=1 / 30, max_iter=1500)
    assert rmse(out, content) < 0.25 * rmse(content + stripes, content)
    # no stripes: the content passes nearly unchanged
    same = general_stripe_remover(content, mu1=1 / 30, max_iter=1500)
    assert rmse(same, content) < 0.1 * float(content.std())
    # the discs keep their contrast
    disc = content > 0.45
    assert float(out[disc].mean() - out[~disc].mean()) == pytest.approx(
        float(content[disc].mean() - content[~disc].mean()), rel=0.1
    )


def test_objective_decreases():
    data = make_frames(2) + make_stripes(2, (48, 40))
    kw = dict(mu1=1 / 30, mu2=1 / 300)
    # PDHGMp is not a descent method: after a transient over the first
    # tens of iterations, the objective decreases towards the minimum
    steps = (0, 100, 300, 1000, 3000)
    energies = [
        objective(general_stripe_remover(data, max_iter=n, **kw), data, **kw)
        for n in steps
    ]
    for a, b in zip(energies[1:], energies[2:], strict=False):
        assert (b < a).all()
    assert (energies[-2] < 0.7 * energies[0]).all()
    # the last steps change it little: close to convergence
    assert ((energies[-2] - energies[-1]) < 0.05 * energies[-1]).all()
    # n = 0 returns the input (u = u₀, s = 0)
    torch.testing.assert_close(general_stripe_remover(data, max_iter=0), data)


def test_minimiser_is_optimal():
    """Feasible perturbations of the solution do not lower Eq. (1)."""
    data = make_frames(1, (24, 20)) + make_stripes(1, (24, 20))
    kw = dict(mu1=0.1, mu2=0.01)
    u = general_stripe_remover(data, max_iter=8000, **kw)
    best = objective(u, data, **kw)
    lo, hi = data.min(), data.max()
    g = torch.Generator().manual_seed(0)
    for scale in (1e-2, 1e-3):
        for _ in range(20):
            v = (
                u + scale * torch.randn(u.shape, generator=g, dtype=F64)
            ).clamp(lo, hi)
            assert float(objective(v, data, **kw)) >= float(best) - 1e-6


def test_box_constraint():
    data = make_frames(2) + make_stripes(2, (48, 40))
    out = general_stripe_remover(data, mu1=0.3, max_iter=300)
    dims = (1, 2)
    lo, hi = data.amin(dims, keepdim=True), data.amax(dims, keepdim=True)
    assert (out >= lo - 1e-12).all() and (out <= hi + 1e-12).all()
    # without ι_[0,1]ᴺ (extension) the relaxed problem reaches a lower
    # or equal objective
    kw = dict(mu1=0.3, mu2=1 / 300)
    free = general_stripe_remover(data, box=False, max_iter=3000, **kw)
    boxed = general_stripe_remover(data, max_iter=3000, **kw)
    assert (
        objective(free, data, **kw) <= objective(boxed, data, **kw) + 1e-3
    ).all()
    # data already in [0, 1]: normalize=False solves the same problem
    unit = (data - lo) / (hi - lo)
    torch.testing.assert_close(
        general_stripe_remover(unit, normalize=False, max_iter=200),
        (general_stripe_remover(data, max_iter=200) - lo) / (hi - lo),
    )


def test_affine_equivariance():
    data = make_frames(2) + make_stripes(2, (48, 40))
    out = general_stripe_remover(data, max_iter=200)
    torch.testing.assert_close(
        general_stripe_remover(250 * data + 40, max_iter=200), 250 * out + 40
    )
    # constant frames pass unchanged
    flat = torch.full((1, 8, 9), 3.0, dtype=F64)
    torch.testing.assert_close(general_stripe_remover(flat, max_iter=20), flat)


def test_parameters():
    """μ₁ is the strength of removal, μ₂ the precaution (Section 2.1.1)."""
    data = make_frames(1) + make_stripes(1, (48, 40))

    def removed(**kw):
        out = general_stripe_remover(data, max_iter=800, **kw)
        return float((data - out).abs().sum())

    strength = [removed(mu1=m) for m in (0.03, 0.1, 0.3)]
    assert strength[0] < strength[1] < strength[2]
    caution = [removed(mu1=0.1, mu2=m) for m in (1 / 300, 1 / 100, 1 / 30)]
    assert caution[0] > caution[1] > caution[2]


def test_oblique_and_multidirectional():
    content = make_frames(1, (48, 48))
    diag = make_stripes(1, (48, 48), angle=45, seed=2)
    vert = make_stripes(1, (48, 48), seed=3)
    kw = dict(mu1=1 / 30, max_iter=1500)
    err = {
        a: rmse(general_stripe_remover(content + diag, angle=a, **kw), content)
        for a in (0, 45, -45)
    }
    assert err[45] < 0.3 * err[0] and err[45] < 0.3 * err[-45]
    assert err[45] < 0.3 * rmse(content + diag, content)
    # several directions: Eq. (1) with ∑ᵢ ‖∇_θᵢ s‖₁ on the same s. With
    # μ₁ large enough (of order 1), 0° + 45° removes both families of
    # stripes better than either direction alone, and the solver
    # minimises the joint objective
    both = content + diag + vert
    kw = dict(mu1=0.7, mu2=1 / 300, max_iter=3000)
    offsets = [(1, 0), (1, 1)]
    multi = general_stripe_remover(both, angle=[0, 45], **kw)
    best = objective(multi, both, offsets=offsets, mu1=0.7, mu2=1 / 300)
    assert rmse(multi, content) < 0.75 * rmse(both, content)
    for a in (0, 45):
        single = general_stripe_remover(both, angle=a, **kw)
        assert rmse(multi, content) < 0.8 * rmse(single, content)
        e = objective(single, both, offsets=offsets, mu1=0.7, mu2=1 / 300)
        assert (best < e).all()


def test_orthogonal_directions_keep_input():
    """0° + 90° with μ₁ ≤ 1: u = u₀ is the unique minimiser (and warns)."""
    data = make_frames(1, (32, 32)) + make_stripes(1, (32, 32))
    with pytest.warns(UserWarning, match="is the input itself"):
        out = general_stripe_remover(
            data, angle=[0, 90], mu1=0.9, max_iter=3000
        )
    # the iterates converge to u₀, which no iterate improves on
    assert rmse(out, data) < 1e-3 * rmse(data, data.mean())
    kw = dict(offsets=[(1, 0), (0, 1)], mu1=0.9, mu2=1 / 300)
    assert (objective(out, data, **kw) >= objective(data, data, **kw)).all()


def test_several_directions_warning():
    import warnings

    data = make_frames(1, (16, 16))
    with pytest.warns(UserWarning, match="little or no stripes"):
        general_stripe_remover(data, angle=[0, 45], max_iter=2)
    with pytest.warns(UserWarning, match="several stripe directions"):
        GeneralStripeRemover(angle=[0, 90], mu1=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        general_stripe_remover(data, angle=[0, 45], mu1=0.7, max_iter=2)
        general_stripe_remover(data, angle=[0, 90], mu1=1.1, max_iter=2)
        general_stripe_remover(data, angle=[0, 180], max_iter=2)
        GeneralStripeRemover(angle=[0, 45], mu1=0.7).apply(data)


@pytest.mark.parametrize("volume", [False, True])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_step_bound(volume, m):
    """τσ‖K‖² < 1, with ‖K‖² from power iteration on KᵀK."""
    from optix.destripe._rottmayer import (
        _grad,
        _grad_offsets,
        _step,
    )

    offsets = [(1, 0), (1, 1), (2, -1)][:m]
    rho_z = 0.7 if volume else 0.0
    shape = (1, 5, 12, 11) if volume else (1, 12, 11)

    def ktk(u, s):
        g = _grad(u, rho_z, volume)
        a = torch.zeros_like(u)
        for (v, w), p in zip(_grad_offsets(rho_z, volume), g, strict=True):
            a += w * utils.lattice_diff_adjoint(p, v)
        c = s.clone()
        for v in offsets:
            c += utils.lattice_diff_adjoint(utils.lattice_diff(s, v), v)
        return a, c

    g = torch.Generator().manual_seed(0)
    u = torch.randn(shape, generator=g, dtype=F64)
    s = torch.randn(shape, generator=g, dtype=F64)
    for _ in range(300):
        u, s = ktk(u, s)
        n = float((u.square().sum() + s.square().sum()).sqrt())
        u, s = u / n, s / n
    a, c = ktk(u, s)
    norm = float((a * u).sum() + (c * s).sum())
    # the bound ‖K‖² ≤ max(4(2 + ρ_z²), 1 + 4m) used for the steps
    assert norm <= max(4 * (2 + rho_z**2), 1 + 4 * m) + 1e-9
    step = _step(volume, rho_z, offsets)
    assert step**2 * norm < 1


def test_volumes_rho_z():
    # (z, y, x): content coherent along z, stripes differing per slice
    content = make_frames(1, (24, 20)).expand(4, 24, 20)
    data = content + make_stripes(4, (24, 20))
    # ρ_z = 0: slices decouple, the 3-D solve equals the 2-D one
    kw = dict(normalize=False, max_iter=300)
    unit = (data - data.min()) / (data.max() - data.min())
    vol = general_stripe_remover(unit[None], ndim=3, **kw)[0]
    torch.testing.assert_close(vol, general_stripe_remover(unit, **kw))
    # ρ_z = 1 couples the slices: it minimises the 3-D objective
    kw = dict(mu1=0.1, mu2=1 / 300)
    u3 = general_stripe_remover(
        data[None], ndim=3, rho_z=1.0, max_iter=3000, **kw
    )
    u2 = general_stripe_remover(data, max_iter=3000, **kw)
    lo, hi = data.min(), data.max()
    e = [
        utils.general_stripe_objective(
            (u[None] - lo) / (hi - lo),
            (data[None] - lo) / (hi - lo),
            rho_z=1.0,
            volume=True,
            **kw,
        )
        for u in (u3[0], u2)
    ]
    assert float(e[0]) < float(e[1])
    assert rmse(u3[0], content) < 0.5 * rmse(data, content)
    with pytest.raises(ValueError, match="rho_z must be 0"):
        general_stripe_remover(data, rho_z=0.5)


def test_tol_extension():
    """`tol` bounds the relative duality gap, hence the excess objective."""
    data = make_frames(3) + make_stripes(3, (48, 40))
    kw = dict(mu1=1 / 3, mu2=1 / 300)
    full = general_stripe_remover(data, max_iter=6000, **kw)
    early = general_stripe_remover(data, max_iter=6000, tol=1e-2, **kw)
    assert rmse(early, full) > 0  # stopped before max_iter
    # E(uₖ) − E* ≤ Δ(uₖ, pₖ) ≤ tol · Δ(u₀, p₀) = tol · μ₁‖∇u₀‖₂,₁
    gap0 = objective(data, data, **kw)
    excess = objective(early, data, **kw) - objective(full, data, **kw)
    assert (excess <= 1e-2 * gap0).all()
    assert rmse(early, full) < 0.2 * rmse(data, full)
    # frames stop one by one: the batch equals the frames alone
    for i in range(3):
        torch.testing.assert_close(
            early[i : i + 1],
            general_stripe_remover(
                data[i : i + 1], max_iter=6000, tol=1e-2, **kw
            ),
        )
    with pytest.raises(ValueError, match="box=True"):
        general_stripe_remover(data, tol=1e-3, box=False)


def test_float32_matches_float64():
    data = make_frames(2) + make_stripes(2, (48, 40))
    u64 = general_stripe_remover(data, max_iter=500)
    u32 = general_stripe_remover(data.float(), max_iter=500)
    assert u32.dtype == torch.float32
    assert float((u32.double() - u64).abs().max()) < 1e-4


def test_beats_wavelet_fft_on_trails():
    """LSFM-like trails of finite length and varying width (cf. Fig. 2)."""
    shape = (64, 64)
    y, x = torch.meshgrid(
        torch.arange(shape[0], dtype=F64),
        torch.arange(shape[1], dtype=F64),
        indexing="ij",
    )
    g = torch.Generator().manual_seed(3)
    clean = 0.2 + 0.1 * torch.exp(
        -((x - 32) ** 2 / 700 + (y - 32) ** 2 / 1100)
    )
    trails = torch.zeros(shape, dtype=F64)
    for k in range(10):
        cy, cx = (torch.rand(2, generator=g, dtype=F64) * 50 + 5).tolist()
        w = 0.7 + 2.0 * float(torch.rand(1, generator=g, dtype=F64))
        clean = clean + 0.4 * torch.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 8)
        trail = torch.exp(-((x - cx) ** 2) / (2 * w * w)) * (y > cy)
        trails = trails + (0.25 if k % 2 else -0.12) * trail * torch.exp(
            -(y - cy).clamp(min=0) / 40
        )
    data = (clean + trails)[None]

    def psnr(u):
        return 10 * math.log10(
            float(
                (clean.max() - clean.min()) ** 2
                / (u[0] - clean).square().mean()
            )
        )

    ours = psnr(
        general_stripe_remover(data, mu1=1 / 3, mu2=1 / 300, max_iter=1500)
    )
    wfft = max(
        psnr(wavelet_fft(data, level=lv, sigma=s))
        for lv, s in ((3, 1.5), (4, 1.5), (4, 3.0), (5, 3.0))
    )
    assert ours > psnr(data) + 3
    assert ours > wfft + 1


# -- orientation, estimator and streaming -------------------------------------
def test_axis_and_layout():
    content = make_frames(2)
    data = content + make_stripes(2, (48, 40))
    kw = dict(mu1=0.1, max_iter=200)
    expected = general_stripe_remover(data, **kw)
    turned = general_stripe_remover(
        data.transpose(-2, -1).contiguous(), axis=-1, **kw
    )
    torch.testing.assert_close(turned.transpose(-2, -1), expected)
    # oblique angles turn with the axis: θ from y towards x is θ from y
    # (now the 2nd axis) towards x
    exp45 = general_stripe_remover(data, angle=45, **kw)
    t45 = general_stripe_remover(
        data.transpose(-2, -1).contiguous(), axis=-1, angle=45, **kw
    )
    torch.testing.assert_close(t45.transpose(-2, -1), exp45)

    layout = Layout("zyx", spatial="yx")
    for axis in ("y", 0, -2):
        out = GeneralStripeRemover(
            axis=axis, layout=layout, dtype=F64, **kw
        ).apply(data)
        torch.testing.assert_close(out, expected)
    swapped = Layout("zyx", spatial="xy")
    out = GeneralStripeRemover(
        axis="y", layout=swapped, dtype=F64, **kw
    ).apply(data)
    torch.testing.assert_close(out, expected)
    out = GeneralStripeRemover(axis=1, dtype=F64, **kw).apply(
        labeled(data, swapped)
    )
    torch.testing.assert_close(out, expected)
    with pytest.raises(ValueError, match="not a spatial axis"):
        GeneralStripeRemover(axis="z", layout=layout, **kw).apply(data)
    with pytest.raises(ValueError, match="2 or 3 spatial axes"):
        GeneralStripeRemover(layout=Layout("zyx", spatial="x"), **kw).apply(
            data
        )


def test_estimator_volumes():
    data = make_frames(4, (24, 20)) + make_stripes(4, (24, 20))
    stack = torch.stack([data, data.flip(0)])  # (t, z, y, x)
    kw = dict(mu1=0.1, rho_z=0.5, max_iter=200)
    layout = Layout("tzyx", spatial="zyx")
    expected = general_stripe_remover(stack, ndim=3, **kw)
    out = GeneralStripeRemover(layout=layout, dtype=F64, **kw).apply(stack)
    torch.testing.assert_close(out, expected)
    with optix.config(batch=1):
        out = GeneralStripeRemover(
            axis="y", layout=layout, dtype=F64, **kw
        ).apply(stack.numpy())
    torch.testing.assert_close(torch.as_tensor(out), expected)
    with pytest.raises(ValueError, match="last two spatial axes"):
        GeneralStripeRemover(axis="z", layout=layout, **kw).apply(stack)
    with pytest.raises(ValueError, match="two spatial axes"):
        GeneralStripeRemover(rho_z=0.5).apply(data)


def test_stream_matches_batch():
    data = make_frames(5, (32, 28)) + make_stripes(5, (32, 28))
    kw = dict(mu1=0.1, max_iter=150)
    expected = general_stripe_remover(data, **kw)
    est = GeneralStripeRemover(**kw, dtype=F64)
    with optix.config(batch=2):
        out = est.apply(data.numpy())
    torch.testing.assert_close(torch.as_tensor(out), expected)
    out = np.zeros(data.shape)
    with optix.config(batch=1):
        est.apply(data.numpy(), out=out)
    torch.testing.assert_close(torch.as_tensor(out), expected)
    stack = data.reshape(1, 5, 32, 28).expand(2, 5, 32, 28).contiguous()
    lay = Layout("ctyx", spatial="yx", group="c")
    with optix.config(batch=3):
        out = GeneralStripeRemover(layout=lay, **kw, dtype=F64).apply(stack)
    torch.testing.assert_close(out, expected.expand(2, 5, 32, 28))


def test_stream_zarr():
    zarr = pytest.importorskip("zarr")
    data = make_frames(3, (24, 20)) + make_stripes(3, (24, 20))
    kw = dict(max_iter=100)
    src = zarr.zeros(data.shape, chunks=(1, 24, 20), dtype="f8")
    src[:] = data.numpy()
    dst = zarr.zeros(data.shape, chunks=(1, 24, 20), dtype="f8")
    with optix.config(batch=1):
        GeneralStripeRemover(**kw, dtype=F64).apply(src, out=dst)
    torch.testing.assert_close(
        torch.as_tensor(dst[:]), general_stripe_remover(data, **kw)
    )


def test_defaults_and_faithful():
    est = GeneralStripeRemover()
    o = est.options
    assert o["mu1"] == pytest.approx(1 / 3) and o["mu2"] == pytest.approx(
        1 / 300
    )
    assert o["max_iter"] == 25000 and o["tol"] is None and o["box"] is True
    assert o["rho_z"] == 0 and o["angle"] == 0 and o["axis"] == -2
    assert est.is_faithful
    assert GeneralStripeRemover.recommended == dict(
        GeneralStripeRemover.paper, max_iter=1000
    )
    rec = GeneralStripeRemover(faithful=False)
    assert rec.options["max_iter"] == 1000 and rec.is_faithful
    # tuned parameters stay faithful, extensions do not
    assert GeneralStripeRemover(
        mu1=0.7, mu2=1 / 150, angle=[0, 45], max_iter=10
    ).is_faithful
    assert not GeneralStripeRemover(tol=1e-5).is_faithful
    assert not GeneralStripeRemover(box=False).is_faithful
    assert est.fit() is est and est.estimates == {}
    assert est.fit(make_frames(1)) is est and est.estimates == {}
    assert "GeneralStripeRemover" in optix.estimators()
    assert est.tags.streaming == "per-frame"
    data = make_frames(2)
    torch.testing.assert_close(
        GeneralStripeRemover(max_iter=50, dtype=F64).apply(data),
        general_stripe_remover(data, max_iter=50),
    )


def test_save_load(tmp_path):
    est = GeneralStripeRemover(mu1=0.7, angle=[0, 45], max_iter=40, axis=-1)
    est.save(tmp_path / "g.pt")
    back = optix.load(tmp_path / "g.pt")
    assert (
        isinstance(back, GeneralStripeRemover) and back.options == est.options
    )
    data = make_frames(2)
    torch.testing.assert_close(back.apply(data), est.apply(data))


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        (dict(mu1=0), "mu1"),
        (dict(mu2=-1), "mu2"),
        (dict(mu1=math.inf), "mu1"),
        (dict(angle=[]), "at least one"),
        (dict(rho_z=2.0, ndim=3), "rho_z"),
        (dict(max_iter=-1), "max_iter"),
        (dict(max_iter=1.5), "max_iter"),
        (dict(tol=0), "tol"),
        (dict(axis=0), "axis"),
        (dict(ndim=4), "ndim"),
    ],
)
def test_invalid(kw, match):
    with pytest.raises(ValueError, match=match):
        general_stripe_remover(torch.zeros(1, 2, 4, 4), **kw)


def test_integer_input():
    data = (
        (1000 * (make_frames(1) + make_stripes(1, (48, 40))))
        .round()
        .to(torch.int32)
    )
    out = general_stripe_remover(data, max_iter=20)
    assert out.is_floating_point() and out.shape == data.shape
