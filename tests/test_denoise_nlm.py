"""Non-local means (Buades, Coll & Morel 2005)."""

import math
import weakref

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

import optix
from optix.denoise import NLMeans, nl_means, nl_means_halo
from optix.denoise import utils as U
from optix.denoise._nlm import OFFSET_COPIES, OFFSET_SHARE
from optix.estimates import SensorEstimate
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


def pad_patches(y, r):
    """Reflect about the edge pixels, or replicate along too short axes."""
    out = y.numpy()
    for axis, n in enumerate(y.shape):
        width = [(0, 0)] * y.ndim
        width[axis] = (r, r)
        out = np.pad(out, width, mode="reflect" if n > r else "edge")
    return torch.from_numpy(out)


def brute_force(y, h, patch, search, a=None, norm="mean"):
    """Pixel-by-pixel NL-means, written from Section 3 of the paper."""
    r, s = patch // 2, search // 2
    g1 = U.patch_kernel(patch, a, norm, **F64)
    g = torch.outer(g1, g1)
    yp = pad_patches(y, r)
    n0, n1 = y.shape
    out = torch.empty_like(y)
    for i0 in range(n0):
        for i1 in range(n1):
            pi = yp[i0 : i0 + patch, i1 : i1 + patch]
            num = den = 0.0
            for j0 in range(max(0, i0 - s), min(n0, i0 + s + 1)):
                for j1 in range(max(0, i1 - s), min(n1, i1 + s + 1)):
                    pj = yp[j0 : j0 + patch, j1 : j1 + patch]
                    d = float((g * (pi - pj).square()).sum())
                    w = math.exp(-d / h**2)
                    num += w * float(y[j0, j1])
                    den += w
            out[i0, i1] = num / den
    return out


class LiveBytes(TorchDispatchMode):
    """Bytes of the tensor storages allocated under the mode, and the peak.

    A storage is counted from the op that creates it until the last
    tensor that refers to it is collected; storages of the inputs are not
    counted.
    """

    def __init__(self):
        super().__init__()
        self.live = {}
        self.current = self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        leaves = pytree.tree_leaves((args, kwargs))
        inputs = {
            t.untyped_storage().data_ptr()
            for t in leaves
            if isinstance(t, torch.Tensor)
        }
        out = func(*args, **kwargs)
        for t in pytree.tree_leaves(out):
            if not isinstance(t, torch.Tensor):
                continue
            key = t.untyped_storage().data_ptr()
            if key in self.live:
                self.live[key][0] += 1
            elif key in inputs or key == 0:
                continue
            else:
                self.live[key] = [1, t.untyped_storage().nbytes()]
                self.current += self.live[key][1]
                self.peak = max(self.peak, self.current)
            weakref.finalize(t, self._release, key)
        return out

    def _release(self, key):
        entry = self.live[key]
        entry[0] -= 1
        if entry[0] == 0:
            self.current -= entry[1]
            del self.live[key]


# -- building blocks ---------------------------------------------------------
@pytest.mark.parametrize("norm,total", [("mean", 49.0), ("sum", 1.0)])
def test_patch_kernel(norm, total):
    g = U.patch_kernel(7, norm=norm, **F64)
    assert float(torch.outer(g, g).sum()) == pytest.approx(total)
    torch.testing.assert_close(g, g.flip(0))
    assert int(g.argmax()) == 3
    flat = U.patch_kernel(5, math.inf, "sum", **F64)
    torch.testing.assert_close(flat, torch.full((5,), 0.2, **F64))
    with pytest.raises(ValueError):
        U.patch_kernel(4)
    with pytest.raises(ValueError):
        U.patch_kernel(5, -1.0)
    with pytest.raises(ValueError):
        U.patch_kernel(5, norm="max")


def test_search_offsets():
    off = U.search_offsets(5, 2)
    assert len(off) == 25 and off[0] == (-2, -2) and off[12] == (0, 0)
    assert len(U.search_offsets(3, 3)) == 27


def test_distances_symmetric_and_brute_force():
    """d_t(i) = d_{−t}(i + t), and the shifted-image distances are the
    patch sums of the definition."""
    y = noisy(scene((14, 17)), 20)
    g1 = U.patch_kernel(5, **F64)
    offsets = U.search_offsets(7, 2)
    dist, values, valid = U.patch_distances(y, offsets, g1, 7)
    assert (dist >= 0).all()
    index = {o: k for k, o in enumerate(offsets)}
    for k, (t0, t1) in enumerate(offsets):
        m = index[(-t0, -t1)]
        for i0 in range(14):
            for i1 in range(17):
                j0, j1 = i0 + t0, i1 + t1
                inside = 0 <= j0 < 14 and 0 <= j1 < 17
                assert bool(valid[k, i0, i1]) == inside
                if inside:
                    assert float(dist[k, i0, i1]) == pytest.approx(
                        float(dist[m, j0, j1]), rel=1e-12, abs=1e-9
                    )
                    assert values[k, i0, i1] == y[j0, j1]
    # one distance from the definition
    yp = F.pad(y[None, None], (2,) * 4, mode="reflect")[0, 0]
    g = torch.outer(g1, g1)
    k = index[(2, -3)]
    i0, i1 = 1, 5
    ref = (
        g
        * (
            yp[i0 : i0 + 5, i1 : i1 + 5] - yp[i0 + 2 : i0 + 7, i1 - 3 : i1 + 2]
        ).square()
    ).sum()
    assert float(dist[k, i0, i1]) == pytest.approx(float(ref), rel=1e-12)


def test_weights_normalised():
    """w(i, j) ∈ [0, 1], ∑_j w(i, j) = 1, and NL[y] = ∑_j w(i, j)·y(j)."""
    y = noisy(scene((20, 22)), 15)
    h = 150.0
    g1 = U.patch_kernel(7, **F64)
    dist, values, valid = U.patch_distances(y, U.search_offsets(9, 2), g1, 9)
    w = torch.exp(-dist / h**2) * valid
    w = w / w.sum(0)
    assert (w >= 0).all() and (w <= 1).all()
    torch.testing.assert_close(w.sum(0), torch.ones_like(y))
    out = nl_means(y, 15, h=h, search=9)
    torch.testing.assert_close(out, (w * values).sum(0), rtol=0, atol=1e-10)


def test_matches_brute_force():
    y = noisy(scene((11, 13)), 20)
    out = nl_means(y, 20, h=60.0, patch=3, search=5)
    torch.testing.assert_close(
        out, brute_force(y, 60.0, 3, 5), rtol=0, atol=1e-10
    )
    out = nl_means(y, 20, h=10.0, patch=5, search=7, a=1.0, kernel_norm="sum")
    torch.testing.assert_close(
        out, brute_force(y, 10.0, 5, 7, 1.0, "sum"), rtol=0, atol=1e-10
    )


def test_patch_one():
    """patch=1 is the neighbourhood filter on single pixels, for any a."""
    assert float(U.patch_kernel(1)) == 1.0
    y = noisy(scene((11, 13)), 20)
    out = nl_means(y, 20, h=40.0, patch=1, search=5)
    torch.testing.assert_close(
        out, nl_means(y, 20, h=40.0, patch=1, search=5, a=1.0)
    )
    torch.testing.assert_close(
        out, brute_force(y, 40.0, 1, 5), rtol=0, atol=1e-10
    )
    assert NLMeans(sigma=20, patch=1, search=5, **F64).fit().apply(y).shape


def test_short_axis_replicates_only_that_axis():
    y = noisy(scene((3, 12)), 20)
    g1 = U.patch_kernel(7, **F64)
    dist, _, _ = U.patch_distances(y, [(0, 1)], g1, 3)
    yp = pad_patches(y, 3)
    g = torch.outer(g1, g1)
    expected = (g * (yp[:7, :7] - yp[:7, 1:8]).square()).sum()
    torch.testing.assert_close(dist[0, 0, 0], expected)
    out = nl_means(y, 20, search=5)
    torch.testing.assert_close(
        out, brute_force(y, 10.0 * 20, 7, 5), rtol=0, atol=1e-10
    )


def test_kernel_normalisations_are_equivalent():
    """ "mean" with h equals "sum" with h/√K, K = patchᵈ."""
    z = noisy(scene((40, 44)), 20)
    a = nl_means(z, 20)
    b = nl_means(z, 20, kernel_norm="sum", h=10 * 20 / 7)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-12)


@pytest.mark.parametrize("patch", [3, 5, 9])
def test_default_h_scales_with_patch(patch):
    z = noisy(scene((40, 44)), 20)
    h = 10 * 20 * math.sqrt(patch**2 / 49)
    assert U.default_h(20, patch) == pytest.approx(h)
    a = nl_means(z, 20, patch=patch, search=11)
    torch.testing.assert_close(
        a, nl_means(z, 20, patch=patch, search=11, h=h), rtol=0, atol=0
    )
    b = nl_means(z, 20, patch=patch, search=11, kernel_norm="sum", h=200 / 7)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-12)


def test_default_h_volume_default_patch():
    clean = scene((12, 20, 22))
    v = noisy(clean, 20)
    h = 10 * 20 * math.sqrt(7**3 / 49)
    assert U.default_h(20, 7, 3) == pytest.approx(h)
    out = nl_means(v, 20, ndim=3, search=5)
    torch.testing.assert_close(
        out, nl_means(v, 20, ndim=3, search=5, h=h), rtol=0, atol=0
    )
    assert psnr(out, clean) > psnr(v, clean) + 6
    est = NLMeans(sigma=20, ndim=3, search=5, **F64).fit()
    torch.testing.assert_close(est.apply(v), out, rtol=0, atol=1e-12)


# -- limits and quality ------------------------------------------------------
def test_h_to_zero_is_identity():
    y = noisy(scene((24, 26)), 20)
    torch.testing.assert_close(nl_means(y, 20, h=1e-4), y, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype,h", [("float64", 1e-170), ("float64", 1e-320), ("float32", 1e-25)]
)
def test_very_small_h_is_identity(dtype, h):
    y = noisy(scene((20, 22)), 20).to(getattr(torch, dtype))
    out = nl_means(y, 20, h=h, search=7)
    torch.testing.assert_close(out, y, rtol=0, atol=0)


def test_h_to_infinity_is_window_mean():
    y = noisy(scene((24, 26)), 20)
    k = torch.ones(1, 1, 9, 9, **F64)
    total = F.conv2d(y[None, None], k, padding=4)[0, 0]
    count = F.conv2d(torch.ones_like(y)[None, None], k, padding=4)[0, 0]
    mean = total / count
    torch.testing.assert_close(
        nl_means(y, 20, h=math.inf, search=9), mean, rtol=0, atol=1e-10
    )
    torch.testing.assert_close(
        nl_means(y, 20, h=1e9, search=9), mean, rtol=0, atol=1e-8
    )


@pytest.mark.parametrize("sigma,gain", [(10, 5.0), (20, 7.0), (35, 7.0)])
def test_psnr_gain(sigma, gain):
    clean = scene((64, 64))
    z = noisy(clean, sigma)
    out = nl_means(z, sigma)
    assert psnr(out, clean) > psnr(z, clean) + gain


def test_kernel_sum_with_h_10_sigma_oversmooths():
    """With a kernel of sum 1, h = 10σ is close to the window mean."""
    clean = scene((48, 48))
    z = noisy(clean, 20)
    summed = nl_means(z, 20, kernel_norm="sum")
    mean = nl_means(z, 20, h=math.inf)
    default = nl_means(z, 20)
    assert psnr(summed, clean) < psnr(default, clean) - 5
    gap = float((summed - mean).abs().mean())
    assert gap < float((default - mean).abs().mean()) / 2


def test_bias_correction():
    """E of the distance between noise patches is 2σ²·∑G_a."""
    g = torch.Generator().manual_seed(3)
    n = 3.0 * torch.randn(80, 80, generator=g, **F64)
    g1 = U.patch_kernel(7, **F64)
    dist, _, valid = U.patch_distances(n, [(0, 9), (5, -7)], g1, 21)
    assert float(dist[valid].mean()) == pytest.approx(2 * 9 * 49, rel=0.05)
    clean = scene((48, 48))
    z = noisy(clean, 20)
    out = nl_means(z, 20, bias_correction=True)
    assert psnr(out, clean) > psnr(z, clean) + 5
    assert not torch.equal(out, nl_means(z, 20))
    # with h → 0 only exact matches remain, as without the correction
    a = nl_means(z, 20, h=1e-3, bias_correction=True)
    assert psnr(a, clean) > psnr(z, clean)


def test_volume():
    clean = scene((16, 18, 20))
    z = noisy(clean, 20)
    out = nl_means(z, 20, ndim=3, patch=3, search=5)
    assert psnr(out, clean) > psnr(z, clean) + 4
    with pytest.raises(ValueError):
        nl_means(z, 20, ndim=4)


def test_float32_matches_float64():
    z = noisy(scene((40, 40)), 20)
    a = nl_means(z, 20, search=11)
    b = nl_means(z.float(), 20, search=11)
    assert b.dtype == torch.float32
    torch.testing.assert_close(b.double(), a, rtol=0, atol=1e-3)


def test_offset_batches_do_not_change_result():
    z = noisy(scene((30, 31)), 20)
    a = nl_means(z, 20, search=11)
    with optix.config(memory=2**14):
        assert U.offset_batch(36 * 37, 8) == 1
        b = nl_means(z, 20, search=11)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-12)


def test_offset_batch_formula():
    per = OFFSET_COPIES * 1000 * 8
    assert U.offset_batch(1000, 8, memory=int(5 * per / OFFSET_SHARE)) == 5
    assert U.offset_batch(10**9, 8, memory=1) == 1


def test_measured_offset_buffers():
    """Each offset of a batch holds at most `OFFSET_COPIES` padded images."""
    z = noisy(scene((40, 40)), 20)
    pixels = 46 * 46  # padded by ⌊patch/2⌋ = 3
    peaks = {}
    for batch in (4, 16):
        memory = int(batch * OFFSET_COPIES * pixels * 8 / OFFSET_SHARE) + 1
        with optix.config(memory=memory):
            assert U.offset_batch(pixels, 8) == batch
            with LiveBytes() as live:
                nl_means(z, 20, search=11)
        peaks[batch] = live.peak
    per_offset = (peaks[16] - peaks[4]) / 12 / (pixels * 8)
    assert OFFSET_COPIES - 1 < per_offset <= OFFSET_COPIES


@pytest.mark.parametrize("budget", [2_000_000, 5_000_000])
def test_apply_peak_within_budget(budget):
    z = noisy(scene((150, 170)), 20).numpy()
    est = NLMeans(sigma=20, **F64).fit()
    with optix.config(memory=budget), LiveBytes() as live:
        est.apply(z)
    assert live.peak <= budget


def test_bad_arguments():
    z = torch.rand(16, 16)
    for kwargs in (
        dict(h=0),
        dict(h=-1),
        dict(patch=4),
        dict(search=0),
        dict(kernel_norm="max"),
    ):
        with pytest.raises(ValueError):
            nl_means(z, 0.1, **kwargs)
    with pytest.raises(ValueError):
        nl_means(z, 0)
    with pytest.raises(ValueError):
        nl_means(torch.rand(16), 0.1)


def test_halo():
    assert nl_means_halo() == 13
    assert nl_means_halo(patch=3, search=5) == 3


# -- estimator ---------------------------------------------------------------
def test_estimator_defaults():
    est = NLMeans(sigma=10)
    assert est.is_faithful and est.options["patch"] == 7
    assert est.options["search"] == 21 and est.halo() == 13
    assert NLMeans(sigma=10, a=2.0).is_faithful
    for kwargs in (
        dict(kernel_norm="sum"),
        dict(h=50.0),
        dict(patch=5),
        dict(search=11),
        dict(ndim=3),
        dict(bias_correction=True),
    ):
        assert not NLMeans(sigma=10, **kwargs).is_faithful
    assert "NLMeans" in set(optix.estimators())
    with pytest.raises(ValueError):
        NLMeans(sigma=-1)
    with pytest.raises(ValueError):
        NLMeans(patch=6)
    with pytest.raises(ValueError):
        NLMeans().fit(torch.rand(16, 16))  # neither sigma nor a sensor
    with pytest.raises(RuntimeError):
        NLMeans().apply(torch.rand(16, 16))
    # nothing to fit with Gaussian noise
    z = torch.rand(16, 16, **F64)
    torch.testing.assert_close(
        NLMeans(sigma=0.1, search=5, **F64).apply(z),
        nl_means(z, 0.1, search=5),
    )


def test_blocks_match_whole_2d():
    z = noisy(scene((70, 83)), 20)
    data = torch.stack([z, z.flip(-1)])
    expected = nl_means(data, 20, search=11)
    est = NLMeans(sigma=20, search=11, **F64).fit()
    shapes = []
    core = est._core

    def spy(x, origin):
        shapes.append(tuple(x.shape))
        return core(x, origin)

    est._core = spy
    budget = int((30 + 2 * est.halo()) ** 2 * est._overhead() * 8)
    with optix.config(memory=budget):
        out = est.apply(data.numpy())
    assert len(shapes) > 2
    assert max(math.prod(s) for s in shapes) * est._overhead() * 8 <= budget
    np.testing.assert_allclose(
        np.asarray(out), expected.numpy(), rtol=0, atol=1e-12
    )


def test_blocks_match_whole_default_halo():
    """Blocks smaller than the halo 13, not dividing the image: bitwise."""
    z = noisy(scene((97, 113)), 20)
    expected = nl_means(z, 20)
    for block in ((23, 31), (5, 97), (11, None), (40, 7)):
        out = NLMeans(sigma=20, block=block, **F64).fit().apply(z)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    stack = torch.stack([z, z.flip(0), z.flip(1)])
    expected = nl_means(stack, 20)
    out = NLMeans(sigma=20, block=(23, 31), **F64).fit().apply(stack)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_blocks_interior_edges_float32():
    z = noisy(scene((60, 64)), 20).float()
    expected = nl_means(z, 20, search=9)
    out = (
        NLMeans(sigma=20, search=9, block=(17, 19), dtype=torch.float32)
        .fit()
        .apply(z)
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=2e-4)


def test_blocks_match_whole_3d():
    v = noisy(scene((14, 16, 18)), 20)
    expected = nl_means(v, 20, ndim=3, patch=3, search=5)
    est = NLMeans(
        sigma=20, ndim=3, patch=3, search=5, block=(4, 7, None), **F64
    )
    out = est.fit().apply(v)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-12)


def test_lazy_zarr():
    zarr = pytest.importorskip("zarr")
    data = noisy(scene((48, 52)), 20)
    src = zarr.zeros(data.shape, chunks=(24, 26), dtype="f8")
    src[:] = data.numpy()
    dst = zarr.zeros(data.shape, chunks=(24, 26), dtype="f8")
    NLMeans(sigma=20, search=7, block=(24, 26), **F64).fit(src).apply(
        src, out=dst
    )
    expected = nl_means(data, 20, search=7)
    np.testing.assert_allclose(dst[:], expected.numpy(), rtol=0, atol=1e-12)


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


def test_poisson_gaussian_vst():
    """Generalised Anscombe → NL-means with σ = 1 → exact inverse."""
    from optix.sensor import anscombe, inverse_anscombe
    from optix.sensor._anscombe import build_table

    z, truth, sensor = _camera()
    est = NLMeans(search=11, **F64).fit(z, init=sensor)
    out = est.apply(z)
    d = nl_means(anscombe(z, 2.0, 3.0, 100.0), 1.0, search=11)
    table = build_table(torch.tensor([(3.0 / 2.0) ** 2], **F64))
    expected = inverse_anscombe(d, 2.0, 3.0, 100.0, table=table)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-9)
    mse0 = float((z - truth).square().mean())
    mse1 = float((out - truth).square().mean())
    assert mse1 < mse0 / 3
    assert abs(float((out - truth).mean())) < 0.5


def test_sensor_maps_blocks_match_whole():
    z, _, _ = _camera((50, 54))
    g = torch.Generator().manual_seed(7)
    offset = 100 + 10 * torch.rand(50, 54, generator=g, **F64)
    variance = 9 + 4 * torch.rand(50, 54, generator=g, **F64)
    z = z - 100 + offset
    sensor = SensorEstimate(offset=offset, variance=variance, gain=2.0)
    whole = NLMeans(search=7, **F64).fit(z, init=sensor).apply(z)
    blocked = NLMeans(search=7, block=(13, 17), **F64).fit(z, init=sensor)
    torch.testing.assert_close(blocked.apply(z), whole, rtol=0, atol=1e-9)


def test_save_load(tmp_path):
    z = noisy(scene((32, 32)), 20)
    est = NLMeans(sigma=20, search=7, bias_correction=True, **F64).fit(z)
    path = tmp_path / "nlm.pt"
    est.save(path)
    back = optix.load(path)
    assert isinstance(back, NLMeans) and back.options == est.options
    torch.testing.assert_close(
        back.apply(z).double(), est.apply(z), rtol=0, atol=1e-2
    )
