import time

import pytest
import torch

from optix.filters import median_disk, median_filter
from optix.shading import IlluminationFunction
from optix.shading.utils import median_smooth, median_window

F64 = torch.float64


def window_median(x, i, j, sh, sw):
    """Lower median of the (shrunk) window of pixel (i, j), NaN ignored."""
    a, b = i - sh // 2, j - sw // 2
    win = x[max(0, a) : a + sh, max(0, b) : b + sw].reshape(-1)
    if win.is_floating_point():
        return win.nanmedian()
    return win.median()


def brute(x, sh, sw):
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = window_median(x, i, j, sh, sw)
    return out


def same(a, b):
    """Bit-identical, NaN matching NaN."""
    if a.dtype != b.dtype:
        return False
    if not a.is_floating_point():
        return torch.equal(a, b)
    return torch.equal(a.isnan(), b.isnan()) and torch.equal(
        a.nan_to_num(0.0), b.nan_to_num(0.0)
    )


def images(g):
    h, w = 13, 17
    yield torch.randint(0, 4, (h, w), generator=g)  # int64, many ties
    yield torch.randint(-3, 3, (h, w), generator=g).to(torch.int16)
    yield torch.randint(0, 3, (h, w), generator=g).float()  # float ties
    yield torch.randn(h, w, generator=g, dtype=F64)
    x = torch.randn(h, w, generator=g)
    x[torch.rand(h, w, generator=g) < 0.3] = float("nan")
    x[:5, :6] = float("nan")  # windows holding only NaN
    yield x


@pytest.mark.parametrize("method", ["histogram", "sort", "auto"])
@pytest.mark.parametrize("size", [1, 2, 3, 4, 7, (2, 5), (6, 3), 40])
def test_median_filter_exact(method, size):
    g = torch.Generator().manual_seed(0)
    sh, sw = (size, size) if isinstance(size, int) else size
    for x in images(g):
        y = median_filter(x, size, method=method)
        assert same(y, brute(x, sh, sw)), (x.dtype, size, method)


def test_median_filter_random_shapes():
    g = torch.Generator().manual_seed(1)
    for _ in range(20):
        h, w = torch.randint(1, 16, (2,), generator=g).tolist()
        sh, sw = torch.randint(1, 24, (2,), generator=g).tolist()
        x = torch.randint(0, 6, (h, w), generator=g).to(F64)
        x[torch.rand(h, w, generator=g) < 0.2] = float("nan")
        ref = brute(x, sh, sw)
        for method in ("histogram", "sort"):
            assert same(median_filter(x, (sh, sw), method=method), ref)


def test_median_filter_batch_and_errors():
    g = torch.Generator().manual_seed(2)
    x = torch.rand(2, 3, 9, 8, generator=g)
    y = median_filter(x, 5, method="histogram")
    for i in range(2):
        for j in range(3):
            assert same(y[i, j], brute(x[i, j], 5, 5))
    with pytest.raises(ValueError):
        median_filter(torch.rand(5), 3)
    with pytest.raises(ValueError):
        median_filter(x, 0)
    with pytest.raises(ValueError):
        median_filter(x, 3, method="nope")


def test_median_filter_large_window_samples():
    """Large window: histogram result at sampled pixels vs nanmedian."""
    g = torch.Generator().manual_seed(3)
    x = torch.rand(120, 150, generator=g)
    x[torch.rand(120, 150, generator=g) < 0.05] = float("nan")
    x[x > 0.9] = 0.5  # ties
    y = median_filter(x, (61, 80))
    for _ in range(60):
        i = int(torch.randint(0, 120, (), generator=g))
        j = int(torch.randint(0, 150, (), generator=g))
        assert y[i, j] == window_median(x, i, j, 61, 80)


def test_median_disk():
    g = torch.Generator().manual_seed(4)
    x = torch.randint(0, 5, (11, 12), generator=g)
    xf = x.to(F64)
    xf[0, :4] = float("nan")
    for r in (1, 3):
        for img in (x, xf):
            y = median_disk(img, r)
            ref = torch.empty_like(img)
            for i in range(11):
                for j in range(12):
                    vals = [
                        img[i + a, j + b]
                        for a in range(-r, r + 1)
                        for b in range(-r, r + 1)
                        if a * a + b * b <= r * r
                        and 0 <= i + a < 11
                        and 0 <= j + b < 12
                    ]
                    v = torch.stack(vals)
                    ref[i, j] = (
                        v.nanmedian() if v.is_floating_point() else v.median()
                    )
            assert same(y, ref)
            assert same(median_smooth(img, r), y)


def test_singh_preset_end_to_end():
    """Mean over the batch, then the exact square median of side 500."""
    g = torch.Generator().manual_seed(5)
    imgs = 1 + torch.rand(4, 40, 37, generator=g, dtype=F64)
    for size in (500, 21):
        est = IlluminationFunction(size=size, dtype=F64)
        f = est.fit(imgs).estimates["shading"].flat
        ref = brute(imgs.mean(0), int(size), int(size))
        assert same(f, ref)
        assert same(f, median_window(imgs.mean(0), int(size)))
    big = IlluminationFunction(dtype=F64).fit(imgs).estimates["shading"].flat
    m = imgs.mean(0).reshape(-1).median()
    assert torch.equal(big, torch.full_like(big, m))


def test_median_filter_timing_smoke():
    """A large window on a mid-sized image stays fast (generous bound)."""
    x = torch.rand(256, 320, generator=torch.Generator().manual_seed(6))
    t = time.perf_counter()
    y = median_filter(x, 125)
    assert time.perf_counter() - t < 120
    assert y[100, 100] == window_median(x, 100, 100, 125, 125)
