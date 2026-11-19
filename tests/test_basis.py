import math

import pytest
import torch

from optix.basis import dct_basis, dct_bending, gram, project, reconstruct


def test_orthonormal(device, dtype):
    b = dct_basis(17, 6, dtype=dtype, device=device)
    assert b.shape == (17, 6) and b.dtype == dtype
    eye = torch.eye(6, dtype=dtype, device=device)
    torch.testing.assert_close(b.T @ b, eye)


def test_explicit_formula():
    n = 9
    b = dct_basis(n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            c = math.sqrt((1 if j == 0 else 2) / n)
            ref = c * math.cos(math.pi * (i + 0.5) * j / n)
            assert b[i, j].item() == pytest.approx(ref, abs=1e-12)


def test_project_reconstruct_round_trip():
    x = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    bases = [dct_basis(7, dtype=torch.float64), dct_basis(5, dtype=x.dtype)]
    coef = project(x, bases)
    assert coef.shape == x.shape
    torch.testing.assert_close(reconstruct(coef, bases), x)


def test_project_truncated_shape():
    bases = [dct_basis(10, 3), dct_basis(8, 2)]
    x = torch.randn(4, 10, 8, dtype=bases[0].dtype)
    assert project(x, bases).shape == (4, 3, 2)
    assert reconstruct(project(x, bases), bases).shape == x.shape


def _fd_energy(f, order, h):
    """Sum of squared forward differences (order 1)."""
    energy = 0.0
    for d in range(f.ndim):
        energy += (f.diff(dim=d) / h[d]).square().sum()
    return energy


def _neumann_laplacian(n, h):
    diff = torch.diff(torch.eye(n, dtype=torch.float64), dim=0) / h
    return diff.T @ diff


def test_bending_dc_zero_and_increasing():
    w = dct_bending([16, 12], [5, 4], dtype=torch.float64)
    assert w.shape == (5, 4)
    assert w[0, 0] == 0
    assert (w.diff(dim=0) > 0).all() and (w.diff(dim=1) > 0).all()


def test_membrane_matches_finite_differences_exactly():
    shape, k, h = [12, 9], [4, 3], [1.0, 0.5]
    w = dct_bending(shape, k, order=1, spacing=h, dtype=torch.float64)
    bases = [
        dct_basis(n, kk, dtype=torch.float64)
        for n, kk in zip(shape, k, strict=True)
    ]
    for i in range(k[0]):
        for j in range(k[1]):
            coef = torch.zeros(k, dtype=torch.float64)
            coef[i, j] = 1
            f = reconstruct(coef, bases)
            fd = _fd_energy(f, 1, h)
            assert w[i, j].item() == pytest.approx(float(fd), rel=1e-10)


def test_bending_matches_finite_differences():
    shape, k, h = [10, 7], [4, 3], [1.0, 2.0]
    w = dct_bending(shape, k, order=2, spacing=h, dtype=torch.float64)
    bases = [
        dct_basis(n, kk, dtype=torch.float64)
        for n, kk in zip(shape, k, strict=True)
    ]
    lap = [_neumann_laplacian(n, hh) for n, hh in zip(shape, h, strict=True)]
    for i in range(k[0]):
        for j in range(k[1]):
            coef = torch.zeros(k, dtype=torch.float64)
            coef[i, j] = 1
            f = reconstruct(coef, bases)
            lf = lap[0] @ f + f @ lap[1].T
            assert w[i, j].item() == pytest.approx(
                lf.square().sum().item(), rel=1e-10, abs=1e-14
            )


def test_bending_close_to_continuous_at_low_frequency():
    w = dct_bending([64], [4], order=2, dtype=torch.float64)
    for j in range(1, 4):
        assert w[j].item() == pytest.approx((math.pi * j / 64) ** 4, rel=0.01)


def test_bending_2d_matches_prototype_formula():
    # weight = (ly[:, None] + lx[None, :]) ** 2: the square of the sum,
    # not ly**2 + lx**2.
    ny, nx = 12, 9
    ky = torch.arange(ny, dtype=torch.float64)
    kx = torch.arange(nx, dtype=torch.float64)
    ly = 2 - 2 * torch.cos(math.pi * ky / ny)
    lx = 2 - 2 * torch.cos(math.pi * kx / nx)
    ref = (ly[:, None] + lx[None, :]) ** 2
    w = dct_bending([ny, nx], [ny, nx], dtype=torch.float64)
    torch.testing.assert_close(w, ref, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(w, ly[:, None] ** 2 + lx[None, :] ** 2)


def test_gram_matches_dense():
    torch.manual_seed(0)
    bases = [
        dct_basis(7, 3, dtype=torch.float64),
        dct_basis(5, 4, dtype=torch.float64),
    ]
    w = torch.rand(2, 7, 5, dtype=torch.float64)
    dense = torch.kron(bases[0], bases[1])  # (35, 12), row-major pixels
    expected = dense.T @ (w.reshape(2, -1, 1) * dense)
    torch.testing.assert_close(gram(w, bases), expected)
    # normal equations of a weighted fit
    x = torch.randn(7, 5, dtype=torch.float64)
    coef = torch.linalg.solve(
        gram(w[0], bases), project(w[0] * x, bases).flatten()
    )
    ref = torch.linalg.lstsq(
        w[0].reshape(-1, 1).sqrt() * dense,
        w[0].reshape(-1).sqrt() * x.reshape(-1),
    ).solution
    torch.testing.assert_close(coef, ref)
