"""Tests of the Biggs–Andrews accelerated Richardson–Lucy."""

import pytest
import torch
from test_deconv import blur, gaussian_psf, poisson_image

from optix import config
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
from optix.io import Layout

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


# -- blocks: block-wise α ---------------------------------------------------
@pytest.mark.parametrize("n_iter", [5, 10])
def test_blocks_match_whole_image_2d(n_iter):
    """Each block has its own α: agreement within a few percent."""
    psf = gaussian_psf(9, 1.5)
    truth, data = poisson_image((96, 80), psf, seed=1)
    kw = {"n_iter": n_iter, "dtype": F64, "alpha_scope": "block"}
    whole = biggs_andrews(data, psf, block=(96, 80), **kw)
    parts = biggs_andrews(data, psf, block=(32, 24), **kw)
    diff = (parts - whole).abs()
    assert float(diff.max() / whole.max()) < 0.1
    assert float(diff.mean() / whole.mean()) < 0.05
    assert error(parts, truth) == pytest.approx(error(whole, truth), rel=0.01)


def test_blocks_match_whole_image_3d():
    psf = gaussian_psf(5, 1.0, ndim=3)
    truth, data = poisson_image((20, 24, 22), psf, seed=2)
    kw = {"n_iter": 10, "dtype": F64, "alpha_scope": "block"}
    whole = biggs_andrews(data, psf, block=(20, 24, 22), **kw)
    parts = biggs_andrews(data, psf, block=(10, 12, 11), **kw)
    diff = (parts - whole).abs()
    assert float(diff.max() / whole.max()) < 0.05
    assert float(diff.mean() / whole.mean()) < 0.05
    assert error(parts, truth) == pytest.approx(error(whole, truth), rel=0.01)


def test_block_scope_is_per_block_rl_block():
    """alpha_scope="block": each block is an image of its own (one pass)."""
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((96, 80), psf, seed=1)
    out = biggs_andrews(
        data, psf, n_iter=12, block=(48, 80), alpha_scope="block", dtype=F64
    )
    # default halo: twice the PSF size, 18
    for core, region in (
        (slice(0, 48), slice(0, 66)),
        (slice(48, 96), slice(30, 96)),
    ):
        y = data[region]
        op = PaddedConvolution(psf, y.shape)
        f, _ = rl_block(y, op, float(data.mean()), n_iter=12, accelerate=True)
        inner = slice(core.start - region.start, core.stop - region.start)
        torch.testing.assert_close(out[core], op.crop(f)[inner])


# -- blocks: global α (lockstep) ---------------------------------------------
@pytest.mark.parametrize("n_iter", [3, 10, 30])
def test_global_alpha_matches_whole_image_2d(n_iter):
    """One α per image and iteration: block-wise = whole-image BA."""
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((96, 80), psf, seed=1)
    kw = {"n_iter": n_iter, "dtype": F64}
    whole = biggs_andrews(data, psf, block=(96, 80), **kw)
    parts = BiggsAndrews(psf, block=(32, 24), alpha_scope="global", **kw)
    out = parts.fit(data).apply(data)
    assert float((out - whole).abs().max() / whole.max()) < 1e-10
    assert parts.history.extras["alpha_scope"] == ["global"]
    assert parts.history.extras["iterations"] == [float(n_iter)]
    op = PaddedConvolution(psf, data.shape)
    trace = {}
    rl_block(
        data,
        op,
        float(data.mean()),
        n_iter=n_iter,
        accelerate=True,
        trace=trace,
    )
    assert parts.history.extras["effective_iterations"][0] == pytest.approx(
        trace["effective"], rel=1e-9
    )
    if n_iter >= 10:  # the block-wise α is visibly different
        block = biggs_andrews(
            data, psf, block=(32, 24), alpha_scope="block", **kw
        )
        assert float((block - whole).abs().max() / whole.max()) > 1e-2


def test_global_alpha_matches_whole_image_3d_and_edge_blocks():
    """3-D, an even PSF size and short edge blocks."""
    psf = gaussian_psf(4, 0.8, ndim=3)
    _, data = poisson_image((20, 23, 22), psf, seed=2)
    kw = {"n_iter": 15, "dtype": F64}
    whole = biggs_andrews(data, psf, block=(20, 23, 22), **kw)
    parts = biggs_andrews(data, psf, block=(7, 9, 22), **kw)
    assert float((parts - whole).abs().max() / whole.max()) < 1e-10


def test_global_alpha_stack_and_extensions():
    """Images of a stack are iterated separately, extensions apply."""
    psf = gaussian_psf(7, 1.2)
    imgs = [poisson_image((40, 36), psf, seed=s)[1] for s in range(3)]
    stack = torch.stack(imgs, dim=-1) + 10.0  # (y, x, t)
    layout = Layout("yxt", spatial="yx")
    kw = {"n_iter": 12, "dtype": F64, "background": 10.0, "damping": 0.8}
    est = BiggsAndrews(psf, block=(16, 12), layout=layout, **kw)
    out = est.fit(stack).apply(stack)
    assert est.history.extras["alpha_scope"] == ["global"] * 3
    for i in range(3):
        ref = biggs_andrews(stack[..., i], psf, **kw)
        torch.testing.assert_close(out[..., i], ref)


def test_global_alpha_chi2_stop_on_whole_image():
    """The χ² test is evaluated once per iteration, over the whole image."""
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((96, 80), psf, seed=1)
    whole = BiggsAndrews(psf, dtype=F64).fit(data)
    ref = whole.apply(data)
    est = BiggsAndrews(psf, block=(32, 24), dtype=F64).fit(data)
    out = est.apply(data)
    n = est.history.extras["iterations"]
    assert n == whole.history.extras["iterations"]
    assert len(n) == 1
    assert float((out - ref).abs().max() / ref.max()) < 1e-10
    op = PaddedConvolution(psf, data.shape)
    start = float(data.mean())
    f, _ = rl_block(data, op, start, n_iter=int(n[0]), accelerate=True)
    assert chi2_probability(data, op.forward(f)) >= 0.05
    f, _ = rl_block(data, op, start, n_iter=int(n[0]) - 1, accelerate=True)
    assert chi2_probability(data, op.forward(f)) < 0.05
    # per block, the test stops at other iterations
    block = BiggsAndrews(psf, block=(32, 24), alpha_scope="block", dtype=F64)
    block.fit(data).apply(data)
    counts = block.history.extras["iterations"]
    assert len(counts) == 12
    assert set(counts) != {n[0]}


def test_global_alpha_n_iter_zero_and_one():
    psf = gaussian_psf(7, 1.2)
    _, data = poisson_image((40, 36), psf, seed=6)
    for n in (0, 1):
        torch.testing.assert_close(
            biggs_andrews(data, psf, n_iter=n, block=(16, 16), dtype=F64),
            richardson_lucy(data, psf, n_iter=n, dtype=F64),
        )


def test_auto_scope():
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((64, 48), psf, seed=3)
    kw = {"n_iter": 10, "dtype": F64, "alpha_scope": "auto"}
    one = BiggsAndrews(psf, **kw)
    assert one.is_faithful is False  # not known before apply
    whole = one.fit(data).apply(data)
    assert one.history.extras["alpha_scope"] == ["global"]
    many = BiggsAndrews(psf, block=(32, 24), **kw).fit(data)
    out = many.apply(data)
    assert many.history.extras["alpha_scope"] == ["block"]
    torch.testing.assert_close(
        out,
        biggs_andrews(
            data,
            psf,
            block=(32, 24),
            alpha_scope="block",
            **{k: v for k, v in kw.items() if k != "alpha_scope"},
        ),
    )
    assert not torch.allclose(out, whole)
    # the memory budget decides whether an image fits in one block
    with config(memory="600KiB"):
        small = BiggsAndrews(psf, **kw).fit(data)
        small.apply(data)
    assert small.history.extras["alpha_scope"] == ["block"]
    # faithful=False defaults to "auto"
    assert BiggsAndrews(psf, faithful=False).options["alpha_scope"] == "auto"


def test_faithful_scope():
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((64, 48), psf, seed=3)
    est = BiggsAndrews(psf, dtype=F64)
    assert est.options["alpha_scope"] == "global"
    assert est.is_faithful
    with config(memory="600KiB"):  # several blocks: still global α
        est.fit(data).apply(data)
    assert est.history.extras["alpha_scope"] == ["global"]
    assert est.is_faithful
    auto = BiggsAndrews(psf, alpha_scope="auto", dtype=F64).fit(data)
    auto.apply(data)
    assert auto.is_faithful  # one block: the global α was used
    with config(memory="600KiB"):
        auto.apply(data)
    assert auto.is_faithful is False
    assert BiggsAndrews(psf, alpha_scope="block").is_faithful is False
    with pytest.raises(ValueError, match="alpha_scope"):
        BiggsAndrews(psf, alpha_scope="image")


def _zarr(path, data, chunks):
    zarr = pytest.importorskip("zarr")
    z = zarr.open_array(
        str(path), mode="w", shape=data.shape, chunks=chunks, dtype="f4"
    )
    z[:] = data.numpy().astype("f4")
    return z


def test_global_alpha_out_of_core(tmp_path, monkeypatch):
    """zarr in, zarr out, iterates on disk: same as in memory."""
    from optix.deconv import _lockstep

    psf = gaussian_psf(7, 1.2)
    data = torch.stack(
        [poisson_image((64, 48), psf, seed=s)[1] for s in (4, 5)]
    )
    src = _zarr(tmp_path / "in.zarr", data, (1, 16, 16))
    dst = _zarr(tmp_path / "out.zarr", torch.zeros_like(data), (1, 16, 16))
    kinds = []
    real = _lockstep._State

    class Spy(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            kinds.append(self.kind)

    monkeypatch.setattr(_lockstep, "_State", Spy)
    kw = {"n_iter": 12, "block": (32, 16), "dtype": torch.float32}
    with config(memory="16KiB"):  # 3 × 70 × 54 float32 iterates do not fit
        est = BiggsAndrews(psf, **kw).fit(src)
        est.apply(src, dst)
    assert kinds == ["disk"]
    assert est.history.extras["alpha_scope"] == ["global", "global"]
    mem = BiggsAndrews(psf, **kw).fit(data.float())
    ref = mem.apply(data.float())
    assert kinds == ["disk", "memory"]
    torch.testing.assert_close(torch.from_numpy(dst[:]), ref)
    # float32: block-wise and whole-image sums round differently, and the
    # extrapolation amplifies it over 12 iterations (float64 agrees to
    # 1e-10, see test_global_alpha_matches_whole_image_2d); same bound as the
    # float32-vs-float64 check
    for i in range(2):
        whole = biggs_andrews(data[i].float(), psf, n_iter=12)
        assert float((ref[i] - whole).abs().max() / whole.max()) < 1e-3


class _Recorder:
    """Array-like that records the regions read from it."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.reads = []

    def __getitem__(self, index):
        value = self.array[index]
        self.reads.append(value.size)
        return value


def test_global_alpha_memory_and_io_bounded(monkeypatch):
    """Block-sized reads only, one read of each block per iteration."""
    from optix.deconv import _lockstep

    psf = gaussian_psf(7, 1.2)
    _, data = poisson_image((96, 80), psf, seed=7)
    src = _Recorder(data.numpy())
    grids = []
    real = _lockstep.PaddedConvolution

    def spy(psf, shape, **kwargs):
        op = real(psf, shape, **kwargs)
        grids.append(op.grid)
        return op

    monkeypatch.setattr(_lockstep, "PaddedConvolution", spy)
    n_iter, block, halo = 9, (32, 40), 14
    with config(memory="32KiB"):
        est = BiggsAndrews(psf, n_iter=n_iter, block=block, dtype=F64)
        est.fit(src)  # the start level: one pass, in budget-sized blocks
        fit_reads = len(src.reads)
        out = est.apply(src)
    count = 3 * 2
    assert len(src.reads) == fit_reads + n_iter * count
    largest = (block[0] + 2 * halo) * (block[1] + 2 * halo)
    assert max(src.reads[fit_reads:]) <= largest
    assert all(g[0] * g[1] <= 2 * largest for g in grids)
    whole = biggs_andrews(data, psf, n_iter=n_iter, dtype=F64)
    assert float((out - whole).abs().max() / whole.max()) < 1e-10


def test_float32_matches_float64():
    psf = gaussian_psf(9, 1.5)
    _, data = poisson_image((48, 48), psf, seed=5)
    a = biggs_andrews(data, psf, n_iter=20, dtype=F64)
    b = biggs_andrews(data.float(), psf.float(), n_iter=20, dtype="float32")
    assert float((a - b).abs().max() / a.max()) < 1e-3
    c = biggs_andrews(
        data.float(), psf.float(), n_iter=20, block=(16, 24), dtype="float32"
    )
    assert float((a - c).abs().max() / a.max()) < 1e-3


# -- estimator --------------------------------------------------------------
def test_faithful_flags_and_extensions():
    psf = gaussian_psf(5, 1.0)
    assert BiggsAndrews(psf).is_faithful
    assert RichardsonLucy(psf).is_faithful
    assert BiggsAndrews(psf, n_iter=3).is_faithful is False
    assert BiggsAndrews(psf, damping=0.5).is_faithful is False
    assert BiggsAndrews(psf).options["alpha_scope"] == "global"
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
    assert new.options["alpha_scope"] == "global"
    torch.testing.assert_close(new.apply(data), est.apply(data))
