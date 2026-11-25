"""BM3D (Dabov et al. 2007) and BM4D (Maggioni et al. 2013)."""

import math

import numpy as np
import pytest
import torch

import optix
from optix.denoise import BM3D, BM4D, bm3d, bm3d_halo, bm4d, bm4d_halo
from optix.denoise import utils as U
from optix.estimates import SensorEstimate
from optix.io import Layout
from optix.models import phantoms

F64 = dict(dtype=torch.float64)


def scene(shape=(64, 64), seed=0) -> torch.Tensor:
    """Cells and filaments on a smooth background, in [0, 255]."""
    ndim = len(shape)
    x = phantoms.cells(shape, 6, radius=7.0, ndim=ndim, seed=seed, **F64)
    x = x + phantoms.filaments(
        shape, 4, width=1.5, ndim=ndim, seed=seed, **F64
    )
    grid = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float64) for n in shape), indexing="ij"
    )
    x = x + 30 * torch.sin(grid[-1] / 9) + 40
    x = x - x.min()
    return x * (255 / x.max())


def noisy(clean, sigma, seed=1):
    g = torch.Generator().manual_seed(seed)
    return clean + sigma * torch.randn(
        clean.shape, generator=g, dtype=clean.dtype
    )


def psnr(x, ref, peak=255.0):
    return 10 * math.log10(peak**2 / float((x - ref).square().mean()))


# -- transforms --------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_bior15_matrix(n):
    t = U.bior15_matrix(n, dtype=torch.float64)
    torch.testing.assert_close(
        t.norm(dim=1), torch.ones(n, dtype=torch.float64)
    )
    torch.testing.assert_close(
        t[0], torch.full((n,), n**-0.5, dtype=torch.float64)
    )
    assert torch.linalg.cond(t) < 2
    # wavelets (all rows but the DC) have one vanishing moment
    torch.testing.assert_close(
        t[1:].sum(1), torch.zeros(n - 1, dtype=torch.float64)
    )


def test_bior15_is_haar_for_four():
    """Maggioni et al. (2013): for L = 4, bior1.5 is the Haar transform."""
    t = U.bior15_matrix(4, dtype=torch.float64)
    h = U.haar_matrix(4, dtype=torch.float64)
    torch.testing.assert_close(t @ t.T, torch.eye(4, dtype=torch.float64))
    torch.testing.assert_close(t.abs(), h.abs())


def test_bior15_filters():
    t = U.bior15_matrix(8, dtype=torch.float64)
    # finest details: Haar differences (x₂ₖ₊₁ − x₂ₖ)/√2
    r = 2**-0.5
    torch.testing.assert_close(
        t[4, :2], torch.tensor([-r, r], dtype=torch.float64)
    )
    # coarser rows are not orthogonal: bior1.5 is biorthogonal
    assert (t @ t.T - torch.eye(8, dtype=torch.float64)).abs().max() > 0.05
    with pytest.raises(ValueError):
        U.bior15_matrix(6)


@pytest.mark.parametrize("n", [1, 2, 8, 32])
def test_haar_matrix(n):
    h = U.haar_matrix(n, dtype=torch.float64)
    torch.testing.assert_close(h @ h.T, torch.eye(n, dtype=torch.float64))
    torch.testing.assert_close(
        h[0], torch.full((n,), n**-0.5, dtype=torch.float64)
    )


def test_dct_and_kaiser():
    d = U.transform_matrix("dct", 11, dtype=torch.float64)
    torch.testing.assert_close(d @ d.T, torch.eye(11, dtype=torch.float64))
    w = U.kaiser_window(8, 2.0, 2, dtype=torch.float64)
    w1 = torch.kaiser_window(8, periodic=False, beta=2.0, dtype=torch.float64)
    torch.testing.assert_close(w, torch.outer(w1, w1).reshape(-1))
    assert U.kaiser_window(4, 2.0, 3).shape == (64,)
    with pytest.raises(ValueError):
        U.transform_matrix("db4", 8)


# -- profiles ----------------------------------------------------------------
def test_profiles():
    ht, wie = U.bm3d_profile(25.0)
    assert (ht.transform, ht.size, ht.group, ht.step, ht.search) == (
        "bior1.5",
        8,
        16,
        3,
        39,
    )
    assert (ht.tau, ht.lam, ht.lam2d, ht.beta) == (2500, 2.7, 0, 2.0)
    assert (wie.transform, wie.size, wie.group, wie.tau) == ("dct", 8, 32, 400)
    ht, wie = U.bm3d_profile(41.0)
    assert (ht.transform, ht.size, ht.step, ht.lam2d, ht.lam, ht.tau) == (
        "dct",
        12,
        4,
        2.0,
        2.8,
        5000,
    )
    assert (wie.size, wie.step, wie.tau) == (11, 6, 3500)
    ht, wie = U.bm4d_profile("normal")
    assert (ht.size, ht.group, ht.search, ht.lam, ht.beta) == (
        4,
        16,
        11,
        2.7,
        None,
    )
    assert ht.keep_dc and ht.tau * 64 == pytest.approx(2.9)
    assert wie.tau * 64 == pytest.approx(0.4)
    ht, wie = U.bm4d_profile("modified")
    assert (ht.group, ht.lam, wie.size) == (32, 2.8, 5)
    assert wie.tau * 125 == pytest.approx(6.7)
    with pytest.raises(ValueError):
        U.bm3d_profile(10.0, "fast")
    with pytest.raises(ValueError):
        U.bm4d_profile("fast")


def test_halo():
    assert bm3d_halo(25) == 90
    assert bm3d_halo(25, output="basic") == 45
    assert bm3d_halo(50) == 49 + 48
    # σ is compared with 40 in 8-bit units
    assert bm3d_halo(50, data_range=1000) == 90
    assert bm4d_halo() == 26
    assert bm4d_halo(profile="modified") == 27
    assert bm3d_halo(25, search=9, step=2) == 2 * (8 + 7)


# -- grouping ----------------------------------------------------------------
def test_reference_grid():
    assert U.reference_grid(20, 8, 3).tolist() == [0, 3, 6, 9, 12]
    assert U.reference_grid(21, 8, 3).tolist() == [0, 3, 6, 9, 12, 13]
    # blocks of a larger image keep the global grid
    assert U.reference_grid(21, 8, 3, origin=4).tolist() == [2, 5, 8, 11, 13]


def _brute_distances(x, ref, n, search):
    h = (search - 1) // 2
    out = {}
    for dy in range(-h, search // 2 + 1):
        for dx in range(-h, search // 2 + 1):
            y0, x0 = ref[0] + dy, ref[1] + dx
            if 0 <= y0 <= x.shape[0] - n and 0 <= x0 <= x.shape[1] - n:
                a = x[ref[0] : ref[0] + n, ref[1] : ref[1] + n]
                b = x[y0 : y0 + n, x0 : x0 + n]
                out[(y0, x0)] = float((a - b).square().mean())
    return out


@pytest.mark.parametrize("ref", [(0, 0), (10, 7), (15, 15)])
def test_block_match_brute_force(ref):
    g = torch.Generator().manual_seed(3)
    x = torch.rand(20, 20, generator=g, dtype=torch.float64)
    n, search = 4, 9
    stage = U.Stage("dct", n, 8, 1, search, 0.17, 2.7)
    spectra = U.patch_spectra(x, U.dct_matrix(n, dtype=torch.float64))
    index, size = U.block_match(spectra, (17, 17), torch.tensor([ref]), stage)
    dist = _brute_distances(x, ref, n, search)
    matched = sorted((d, p) for p, d in dist.items() if d <= stage.tau)
    count = min(len(matched), stage.group)
    assert size[0] == 2 ** int(math.log2(count))
    got = [divmod(int(i), 17) for i in index[0, : int(size[0])]]
    assert got[0] == ref
    assert set(got) == {p for _, p in matched[: int(size[0])]}
    assert all(dist[p] <= stage.tau for p in got)


def test_block_match_finds_copies():
    x = torch.rand(
        40, 40, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    patch = x[5:13, 5:13].clone()
    x[20:28, 22:30] = patch
    x[2:10, 25:33] = patch
    stage = U.Stage("dct", 8, 16, 3, 39, 1e-12, 2.7)
    spectra = U.patch_spectra(x, U.dct_matrix(8, dtype=torch.float64))
    index, size = U.block_match(
        spectra, (33, 33), torch.tensor([[5, 5]]), stage
    )
    assert int(size[0]) == 2  # 3 exact matches, rounded down to 2
    assert int(index[0, 0]) == 5 * 33 + 5
    assert int(index[0, 1]) in (20 * 33 + 22, 2 * 33 + 25)
    # the Wiener stage matches with a strict inequality
    stage = U.Stage("dct", 8, 16, 3, 39, 0.0, None, strict=True)
    _, size = U.block_match(spectra, (33, 33), torch.tensor([[5, 5]]), stage)
    assert int(size[0]) == 1


# -- collaborative filtering -------------------------------------------------
def test_hard_threshold():
    s = torch.tensor([[[0.5, 3.0], [-4.0, 1.0]]], dtype=torch.float64)
    out, w = U.hard_threshold(s, 2.0, 1.0, keep_dc=False)
    torch.testing.assert_close(
        out, torch.tensor([[[0.0, 3.0], [-4.0, 0.0]]], dtype=torch.float64)
    )
    torch.testing.assert_close(w, torch.tensor([0.5], dtype=torch.float64))
    out, w = U.hard_threshold(s, 2.0, 2.0, keep_dc=True)
    assert out[0, 0, 0] == 0.5 and w[0] == pytest.approx(1 / 4)  # DC only
    out, w = U.hard_threshold(s, 10.0, 1.0, keep_dc=False)
    assert (out == 0).all() and w[0] == 1  # Eq. (10), N_har = 0


def test_wiener():
    z = torch.tensor([[[2.0, -1.0]]], dtype=torch.float64)
    p = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    out, w = U.wiener(z, p, 1.0)
    torch.testing.assert_close(
        out, torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    )
    torch.testing.assert_close(w, torch.tensor([4.0], dtype=torch.float64))
    _, w = U.wiener(z, torch.zeros_like(p), 1.0)
    assert torch.isfinite(w).all()


def test_aggregate():
    num, den = (
        torch.zeros(16, dtype=torch.float64),
        torch.zeros(16, dtype=torch.float64),
    )
    blocks = torch.full((2, 1, 4), 3.0, dtype=torch.float64)
    corners = torch.tensor([[0], [5]])
    offsets = torch.tensor([0, 1, 4, 5])
    window = torch.tensor([1.0, 2.0, 2.0, 4.0], dtype=torch.float64)
    U.aggregate(
        num,
        den,
        blocks,
        corners,
        offsets,
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        window,
    )
    assert den[5] == 4 + 2 * 1
    covered = den > 0
    torch.testing.assert_close(
        num[covered] / den[covered],
        torch.full((int(covered.sum()),), 3.0, dtype=torch.float64),
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_stage_identity_without_noise(ndim):
    """With σ → 0 nothing is shrunk: each stage returns its input."""
    shape = (20, 21) if ndim == 2 else (9, 10, 11)
    x = torch.rand(
        shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1)
    )
    ht, wie = U.bm3d_profile(10) if ndim == 2 else U.bm4d_profile()
    ht = U.Stage(**{**ht.__dict__, "search": 5})
    wie = U.Stage(**{**wie.__dict__, "search": 5})
    basic = U.run_stage(x, 1e-9, ht)
    torch.testing.assert_close(basic, x)
    torch.testing.assert_close(U.run_stage(x, 1e-9, wie, pilot=x), x)


def test_stage_constant_and_gaps():
    x = torch.full((30, 30), 7.0, dtype=torch.float64)
    ht, _ = U.bm3d_profile(10)
    torch.testing.assert_close(U.run_stage(x, 5.0, ht), x)
    # a step larger than the block leaves gaps, filled with the input
    g = torch.Generator().manual_seed(2)
    y = torch.rand(30, 30, generator=g, dtype=torch.float64)
    sparse = U.Stage("dct", 4, 1, 12, 1, 1.0, 2.7)
    out = U.run_stage(y, 0.1, sparse)
    assert torch.isfinite(out).all()
    assert torch.equal(out[6, 6], y[6, 6])
    with pytest.raises(ValueError):
        U.run_stage(torch.zeros(5, 5), 1.0, ht)


# -- denoising ---------------------------------------------------------------
@pytest.mark.parametrize("sigma, gain", [(25.0, 8.0), (50.0, 10.0)])
def test_bm3d_psnr_gain(sigma, gain):
    clean = scene()
    z = noisy(clean, sigma)
    basic = bm3d(z, sigma, data_range=255, output="basic")
    final = bm3d(z, sigma, data_range=255)
    p0, p1, p2 = psnr(z, clean), psnr(basic, clean), psnr(final, clean)
    assert p1 > p0 + gain
    assert p2 > p1  # the Wiener step improves on the basic estimate


def test_bm4d_psnr_gain():
    clean = scene((12, 24, 24)) / 255
    for profile in ("normal", "modified"):
        z = noisy(clean, 0.1)
        basic = bm4d(z, 0.1, data_range=1, profile=profile, output="basic")
        final = bm4d(z, 0.1, data_range=1, profile=profile)
        p0, p1, p2 = (psnr(v, clean, 1.0) for v in (z, basic, final))
        assert p1 > p0 + 8
        assert p2 > p1


def test_bm4d_uses_the_third_dimension():
    """A volume of identical slices denoises better than each slice."""
    clean = scene((32, 32))[None].repeat(8, 1, 1) / 255
    z = noisy(clean, 0.1)
    vol = bm4d(z, 0.1, data_range=1, search=7)
    # BM3D on each slice, with its own (8-bit) scaling of the same data
    slices = bm3d(z, 0.1, data_range=1, search=7)
    assert psnr(vol, clean, 1.0) > psnr(slices, clean, 1.0)


def test_deterministic():
    z = noisy(scene((48, 48)), 20)
    a = bm3d(z, 20, data_range=255, search=15)
    b = bm3d(z.clone(), 20, data_range=255, search=15)
    assert torch.equal(a, b)


def test_float32_matches_float64():
    clean = scene((48, 48))
    z = noisy(clean, 20)
    ref = bm3d(z, 20, data_range=255, search=15)
    out = bm3d(z.float(), 20, data_range=255, search=15)
    assert out.dtype == torch.float32
    assert (out.double() - ref).abs().mean() < 1e-3
    assert abs(psnr(out.double(), clean) - psnr(ref, clean)) < 0.01


def test_range_invariance():
    """The result scales with the data (units only set τ and σ).

    It is not translation invariant: BM3D also thresholds the DC.
    """
    z = noisy(scene((40, 40)), 15)
    ref = bm3d(z, 15, search=11)
    out = bm3d(z / 64, 15 / 64, search=11)
    torch.testing.assert_close(out * 64, ref, rtol=0, atol=1e-8)


def test_bad_arguments():
    z = torch.zeros(16, 16)
    with pytest.raises(ValueError):
        bm3d(z + torch.rand(16, 16), 0.0)
    with pytest.raises(ValueError):
        bm3d(z, 1.0)  # zero range
    with pytest.raises(ValueError):
        bm3d(z, 1.0, data_range=1, output="wiener")
    with pytest.raises(ValueError):
        bm3d(z, 1.0, data_range=1, step=0)
    with pytest.raises(ValueError):
        bm4d(torch.rand(8, 8), 1.0)


# -- estimators and streaming ------------------------------------------------
def test_estimator_defaults():
    est = BM3D(sigma=10)
    assert est.is_faithful and est.options["profile"] == "normal"
    assert not BM3D(sigma=10, search=11).is_faithful
    assert not BM3D(sigma=10, keep_dc=True).is_faithful
    assert BM3D(sigma=10, keep_dc=False).is_faithful
    assert "keep_dc" not in BM4D(sigma=1).options
    assert BM4D(sigma=1, profile="modified").is_faithful
    assert {"BM3D", "BM4D"} <= set(optix.estimators())
    with pytest.raises(ValueError):
        BM3D(sigma=-1)
    with pytest.raises(ValueError):
        BM3D(profile="fast")
    with pytest.raises(ValueError):
        BM3D().fit(torch.rand(16, 16))  # neither sigma nor a sensor
    with pytest.raises(ValueError):
        BM3D(sigma=1).fit()  # no range
    with pytest.raises(RuntimeError):
        BM3D(sigma=1).apply(torch.rand(16, 16))


def test_estimator_matches_function():
    z = noisy(scene((2, 40, 40)), 20)
    est = BM3D(sigma=20, search=11, **F64).fit(z)
    expected = bm3d(z, 20, search=11)  # the same default range
    torch.testing.assert_close(est.apply(z), expected, rtol=0, atol=1e-9)


def test_blocks_match_whole_2d():
    z = noisy(scene((70, 83)), 20)
    data = torch.stack([z, z.flip(-1)])
    expected = bm3d(data, 20, data_range=255, search=9)
    est = BM3D(sigma=20, data_range=255, search=9, **F64)
    shapes = []
    filt = est._filter

    def spy(x, origin, vst):
        shapes.append(tuple(x.shape))
        return filt(x, origin, vst)

    est._filter = spy
    halo = est.halo()
    overhead = U.block_overhead(8 * 8)
    budget = int((40 + 2 * halo) ** 2 * overhead * 8)
    with optix.config(memory=budget):
        out = est.apply(data.numpy())
    # memory scaling: blocks are bounded by the budget, not by the image
    assert len(shapes) > 2
    assert max(math.prod(s) for s in shapes) * overhead * 8 <= budget
    np.testing.assert_allclose(
        np.asarray(out), expected.numpy(), rtol=0, atol=1e-9
    )


def test_blocks_match_whole_3d_slabs():
    v = noisy(scene((20, 22, 24)) / 255, 0.1)
    expected = bm4d(v, 0.1, data_range=1, search=5)
    est = BM4D(sigma=0.1, data_range=1, search=5, block=(4, None, None), **F64)
    out = est.fit().apply(v)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-12)
    # permuted spatial axes, from a layout
    lay = Layout("yxz", spatial="zyx")
    perm = v.permute(1, 2, 0).contiguous()
    out = (
        BM4D(
            sigma=0.1,
            data_range=1,
            search=5,
            layout=lay,
            block=(6, None, None),
            **F64,
        )
        .fit()
        .apply(perm)
    )
    torch.testing.assert_close(
        out, expected.permute(1, 2, 0), rtol=0, atol=1e-12
    )


def test_lazy_zarr():
    zarr = pytest.importorskip("zarr")
    data = noisy(scene((48, 52)), 20)
    src = zarr.zeros(data.shape, chunks=(24, 26), dtype="f8")
    src[:] = data.numpy()
    dst = zarr.zeros(data.shape, chunks=(24, 26), dtype="f8")
    est = BM3D(sigma=20, search=7, block=(24, 26), **F64).fit(src)
    est.apply(src, out=dst)
    expected = bm3d(data, 20, search=7)
    np.testing.assert_allclose(dst[:], expected.numpy(), rtol=0, atol=1e-9)


def test_in_place_refused():
    data = np.random.default_rng(0).random((20, 20))
    with pytest.raises(ValueError):
        BM3D(sigma=0.1, data_range=1, **F64).apply(data, out=data)


def test_poisson_gaussian_vst():
    """Generalised Anscombe → BM3D with σ = 1 → exact unbiased inverse."""
    g = torch.Generator().manual_seed(4)
    lam = scene((48, 48)) / 10 + 1  # 1 to 26 electrons
    alpha, s, mu = 2.0, 3.0, 100.0
    truth = alpha * lam + mu
    z = alpha * torch.poisson(lam, generator=g) + mu
    z = z + s * torch.randn(z.shape, generator=g, dtype=z.dtype)
    sensor = SensorEstimate(offset=mu, variance=s * s, gain=alpha)
    est = BM3D(search=15, **F64).fit(z, init=sensor)
    assert est.estimates["sensor"].gain == alpha
    out = est.apply(z)
    mse0 = float((z - truth).square().mean())
    mse1 = float((out - truth).square().mean())
    assert mse1 < mse0 / 5
    assert abs(float((out - truth).mean())) < 0.5  # unbiased inverse


def test_save_load(tmp_path):
    z = noisy(scene((32, 32)), 20)
    est = BM3D(sigma=20, search=7, **F64).fit(z)
    path = tmp_path / "bm3d.pt"
    est.save(path)
    back = optix.load(path)
    assert isinstance(back, BM3D) and back.options == est.options
    assert back.halo() == est.halo()
    torch.testing.assert_close(
        back.apply(z).double(), est.apply(z), rtol=0, atol=1e-2
    )


# -- default range -----------------------------------------------------------
def test_clean_range():
    n, sigma = 10_000, 3.0
    extent = 2 * math.sqrt(2 * math.log(n)) * sigma
    assert U.clean_range(0.0, 100.0, n, sigma) == pytest.approx(100 - extent)
    # floored at half the range of the data
    assert U.clean_range(-5.0, 15.0, n, sigma) == pytest.approx(10.0)
    assert U.clean_range(0.0, 1.0, 1, 0.0) == 1.0


def test_default_range_reaches_high_noise_profile():
    """σ > 40 (8-bit units) selects the high-noise column of Table I."""
    clean = scene((96, 96))
    z = noisy(clean, 60)
    est = BM3D(sigma=60, **F64).fit(z)
    assert est.halo() == bm3d_halo(60) == 97  # σ > 40 profile
    # the noisy range would give σ̃ ≈ 24, the σ ≤ 40 profile
    assert bm3d_halo(60, data_range=float(z.max() - z.min())) == 90
    lo, hi = float(z.min()), float(z.max())
    rng = U.clean_range(lo, hi, z.numel(), 60.0)
    assert 255 * 60 / rng > 40
    expected = bm3d(z, 60, data_range=rng, search=9)
    torch.testing.assert_close(bm3d(z, 60, search=9), expected)
    # σ = 20 stays in the σ ≤ 40 profile
    assert BM3D(sigma=20).fit(noisy(clean, 20)).halo() == 90


def test_default_range_sensor_low_counts():
    """Stabilised low-count data (σ = 1) select the high-noise profile."""
    g = torch.Generator().manual_seed(5)
    lam = scene((64, 64)) / 50 + 0.5  # 0.5 to 5.6 electrons
    z = 2.0 * torch.poisson(lam, generator=g) + 100.0
    sensor = SensorEstimate(offset=100.0, variance=1.0, gain=2.0)
    est = BM3D(**F64).fit(z, init=sensor)
    assert 255 / est._range > 40
    assert est.halo() == 97


# -- offsets -----------------------------------------------------------------
def _offset_scene():
    x = phantoms.filaments((64, 64), intensity=200.0, seed=0, **F64)
    x = 255 * (x - x.min()) / (x.max() - x.min())  # background 0
    return noisy(x, 20, seed=0)


def test_bm3d_depends_on_offset():
    """BM3D thresholds the DC: an offset changes the result (Eq. (6))."""
    z = _offset_scene()
    kw = dict(data_range=255, search=15, output="basic")
    diff = bm3d(z + 100, 20, **kw) - 100 - bm3d(z, 20, **kw)
    assert diff.abs().max() > 0.5


@pytest.mark.parametrize("output", ["basic", "final"])
def test_keep_dc_offset_invariance(output):
    z = _offset_scene()
    kw = dict(data_range=255, search=15, output=output, keep_dc=True)
    a = bm3d(z, 20, **kw)
    b = bm3d(z + 100, 20, **kw) - 100
    if output == "basic":
        torch.testing.assert_close(a, b, rtol=0, atol=1e-9)
    else:  # the Wiener gain of the DC depends on the mean
        ref = bm3d(z + 100, 20, **{**kw, "keep_dc": False}) - 100
        ref = (ref - bm3d(z, 20, **{**kw, "keep_dc": False})).abs().max()
        assert (a - b).abs().max() < ref
    est = BM3D(sigma=20, data_range=255, search=15, keep_dc=True, **F64)
    torch.testing.assert_close(
        est.fit().apply(z),
        bm3d(z, 20, data_range=255, search=15, keep_dc=True),
    )


def test_bm4d_offset_invariance():
    v = phantoms.filaments((12, 20, 20), intensity=1.0, ndim=3, seed=0, **F64)
    z = noisy((v - v.min()) / (v.max() - v.min()), 0.1)
    kw = dict(data_range=1, search=5, output="basic")
    torch.testing.assert_close(
        bm4d(z + 0.5, 0.1, **kw) - 0.5, bm4d(z, 0.1, **kw), rtol=0, atol=1e-12
    )


# -- memory ------------------------------------------------------------------
def test_memory_budget_formula():
    nc, k, e = 39 * 39, 64, 4
    for budget in (2**24, 2**26, 2**28):
        b = U.match_batch(nc, k, e, budget)
        assert b * nc * (2 * k * e + 40) <= budget / 4
        assert (b + 1) * nc * (2 * k * e + 40) > budget / 4
        f = U.filter_batch(16, k, e, budget)
        assert f * 16 * k * (8 * e + 8) <= budget / 4
    # linear in the budget, and from the configuration by default
    big, small = (U.filter_batch(16, k, e, b) for b in (2**26, 2**24))
    assert abs(big - 4 * small) < 4
    with optix.config(memory=2**24):
        assert U.match_batch(nc, k, e) == U.match_batch(nc, k, e, 2**24)
    assert U.match_batch(nc, k, e, 1) == U.filter_batch(32, k, e, 1) == 1
    # the spectra (three copies of K values per sample) get half the budget
    assert U.block_overhead(k) == 2 * (3 * k + 8)


def test_batches_follow_budget(monkeypatch):
    """`run_stage` sizes its batches from the configured budget."""
    from optix.denoise import _engine

    seen = {}
    real = _engine.filter_batch

    def spy(m, k, e, memory=None):
        out = real(m, k, e, memory)
        seen.setdefault(optix.get_config().memory, set()).add(out)
        return out

    monkeypatch.setattr(_engine, "filter_batch", spy)
    z = noisy(scene((40, 40)), 20)
    outs = []
    for budget in (2**16, 2**26):
        with optix.config(memory=budget):
            outs.append(bm3d(z, 20, data_range=255, search=9))
    assert max(seen[2**16]) < min(seen[2**26])
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=1e-9)


# -- streaming ---------------------------------------------------------------
def _spy_blocks(est):
    shapes = []
    filt = est._filter

    def spy(x, origin, vst):
        shapes.append(tuple(x.shape))
        return filt(x, origin, vst)

    est._filter = spy
    return shapes


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_blocks_interior_edges(dtype):
    """Blocks of 31 × 37 (dividing neither the image nor N_step = 3)."""
    z = noisy(scene((150, 170)), 20).to(dtype)
    est = BM3D(sigma=20, data_range=255, search=9, block=(31, 37), dtype=dtype)
    halo = est.halo()
    assert halo == 30 and 150 >= 2 * halo + 2 * 31 and 170 >= 2 * halo + 74
    shapes = _spy_blocks(est)
    out = est.apply(z)
    assert len(shapes) == 25
    assert min(s[0] for s in shapes) < 150 and min(s[1] for s in shapes) < 170
    expected = bm3d(z, 20, data_range=255, search=9)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-9)


def test_halo_is_tight():
    z = noisy(scene((150, 170)), 20)
    est = BM3D(sigma=20, data_range=255, search=9, block=(31, 37), **F64)
    halo = est.halo()
    est.halo = lambda: halo - 1
    out = est.apply(z)
    expected = bm3d(z, 20, data_range=255, search=9)
    assert (out - expected).abs().max() > 1e-6


def test_blocks_default_halo():
    """The default profile (N_S = 39, halo 90), blocked along one axis."""
    z = noisy(scene((250, 40)), 20)
    est = BM3D(sigma=20, data_range=255, block=(50, None), **F64)
    assert est.halo() == 90
    shapes = _spy_blocks(est)
    out = est.apply(z)
    assert len(shapes) == 5 and min(s[0] for s in shapes) < 250
    torch.testing.assert_close(
        out, bm3d(z, 20, data_range=255), rtol=0, atol=1e-9
    )


def test_blocks_bm4d_two_axes():
    v = noisy(scene((44, 46, 10)) / 255, 0.1)
    est = BM4D(sigma=0.1, data_range=1, search=5, block=(8, 9, None), **F64)
    halo = est.halo()
    assert 44 >= 2 * halo + 16 and 46 >= 2 * halo + 18
    shapes = _spy_blocks(est)
    out = est.apply(v)
    assert len(shapes) > 4
    expected = bm4d(v, 0.1, data_range=1, search=5)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-12)


def _camera(shape=(48, 48), seed=4):
    g = torch.Generator().manual_seed(seed)
    lam = scene(shape) / 10 + 1  # 1 to 26 electrons
    alpha, s, mu = 2.0, 3.0, 100.0
    z = alpha * torch.poisson(lam, generator=g) + mu
    z = z + s * torch.randn(z.shape, generator=g, dtype=z.dtype)
    return (
        z,
        alpha * lam + mu,
        SensorEstimate(offset=mu, variance=s * s, gain=alpha),
    )


def test_sensor_blocks_match_whole():
    z, _, sensor = _camera((96, 100))
    whole = BM3D(search=9, **F64).fit(z, init=sensor)
    blocked = BM3D(search=9, block=(20, 23), **F64).fit(z, init=sensor)
    torch.testing.assert_close(
        blocked.apply(z), whole.apply(z), rtol=0, atol=1e-9
    )


@pytest.mark.parametrize("inverse", ["closed-form", "asymptotic", "algebraic"])
def test_sensor_inverses(inverse):
    from optix.sensor import anscombe, inverse_anscombe

    z, truth, sensor = _camera()
    est = BM3D(search=9, inverse=inverse, **F64).fit(z, init=sensor)
    out = est.apply(z)
    d = bm3d(
        anscombe(z, 2.0, 3.0, 100.0), 1.0, data_range=est._range, search=9
    )
    expected = inverse_anscombe(d, 2.0, 3.0, 100.0, method=inverse)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-9)
    assert float((out - truth).square().mean()) < float(
        (z - truth).square().mean()
    )


def test_sensor_maps_per_pixel():
    """Per-pixel offset and variance maps are applied pixel by pixel."""
    from optix.sensor import anscombe, inverse_anscombe
    from optix.sensor._anscombe import build_table

    z, truth, _ = _camera((60, 64))
    g = torch.Generator().manual_seed(7)
    offset = 100 + 10 * torch.rand(60, 64, generator=g, dtype=torch.float64)
    variance = 9 + 4 * torch.rand(60, 64, generator=g, dtype=torch.float64)
    z = z - 100 + offset
    sensor = SensorEstimate(offset=offset, variance=variance, gain=2.0)
    est = BM3D(search=9, block=(17, 19), **F64).fit(z, init=sensor)
    s = variance.sqrt()
    f = anscombe(z, 2.0, s, offset)
    rng = U.clean_range(float(f.min()), float(f.max()), z.numel(), 1.0)
    assert est._range == pytest.approx(rng)
    d = bm3d(f, 1.0, data_range=rng, search=9)
    table = build_table((s / 2.0).square().reshape(-1))
    expected = inverse_anscombe(d, 2.0, s, offset, table=table)
    torch.testing.assert_close(est.apply(z), expected, rtol=0, atol=1e-9)
    # constant maps are the scalar case
    flat = SensorEstimate(
        offset=torch.full((60, 64), 100.0), variance=9.0, gain=2.0
    )
    a = BM3D(search=9, **F64).fit(z, init=flat)
    b = BM3D(search=9, **F64).fit(
        z, init=SensorEstimate(offset=100.0, variance=9.0, gain=2.0)
    )
    assert a._maps is None
    torch.testing.assert_close(a.apply(z), b.apply(z), rtol=0, atol=0)
