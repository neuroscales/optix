"""Tests of the shift-variant blur models and their deconvolution."""

import itertools

import pytest
import torch

from optix import config
from optix.deconv import (
    BiggsAndrews,
    DenisPSF,
    NagyOLeary,
    NagyOLearyPSF,
    RichardsonLucy,
    Snyder,
    WienerButterworth,
    nagy_oleary,
)
from optix.deconv.utils import (
    PaddedConvolution,
    RegionPreconditioner,
    ShiftVariantConvolution,
    ShiftVariantPSF,
    cgls,
    interpolation_weights,
    object_positions,
    optimal_circulant,
)

F64 = torch.float64


def gaussian(size, sigmas, dtype=F64):
    """Separable Gaussian `(size,)*d` with one σ per axis, unit sum."""
    x = torch.arange(size, dtype=dtype) - size // 2
    out = None
    for s in sigmas:
        g = torch.exp(-(x**2) / (2 * s**2))
        out = g if out is None else out[..., None] * g
    return out / out.sum()


def psf_grid(ndim, size=7, nodes=3, dtype=F64):
    """A grid of Gaussian PSFs whose width grows along every axis."""
    psfs = []
    for idx in itertools.product(range(nodes), repeat=ndim):
        sig = [0.7 + 0.6 * i + 0.2 * a for a, i in enumerate(idx)]
        psfs.append(gaussian(size, sig, dtype))
    shape = (nodes,) * ndim + (size,) * ndim
    return torch.stack(psfs).reshape(shape)


def centres_for(shape, nodes=3):
    return [torch.linspace(4.0, n - 5.0, nodes) for n in shape]


def checkerboard(shape, size=4, value=100.0):
    grids = torch.meshgrid(*[torch.arange(n) for n in shape], indexing="ij")
    return (sum(g // size for g in grids) % 2).to(F64) * value


def sharp_grid(ndim, size=15):
    """HST-like PSFs: a sharp core on a halo that widens across the field."""
    psfs = []
    for idx in itertools.product(range(3), repeat=ndim):
        sig = [1.5 + 0.8 * i for i in idx]
        core = torch.zeros((size,) * ndim, dtype=F64)
        core[(size // 2,) * ndim] = 1
        psfs.append(0.3 * core + 0.7 * gaussian(size, sig))
    return torch.stack(psfs).reshape((3,) * ndim + (size,) * ndim)


def models(psf, centres):
    return [
        NagyOLearyPSF(psf, centres),
        NagyOLearyPSF(psf, centres, interpolation="linear"),
        DenisPSF(psf, centres),
        DenisPSF(psf, centres, model="modes", rank=3),
    ]


def embed(x, op):
    """A block-sized tensor placed on the operator's object grid."""
    out = x.new_zeros(op.grid)
    out[tuple(slice(0, n) for n in x.shape)] = x
    return out


# -- interpolation weights ----------------------------------------------
@pytest.mark.parametrize("kind", ["nearest", "simplex", "multilinear"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_weights_partition_of_unity(kind, ndim):
    centres = [torch.tensor([2.0, 7.0, 15.0])] * ndim
    coords = [torch.linspace(-3, 19, 12, dtype=F64)] * ndim
    nodes, w = interpolation_weights(centres, coords, kind)
    assert nodes.shape[1] == ndim and w.shape[0] == nodes.shape[0]
    assert (w >= 0).all()
    torch.testing.assert_close(w.sum(0), torch.ones_like(w[0]))


@pytest.mark.parametrize("kind", ["simplex", "multilinear"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_weights_reproduce_affine_functions(kind, ndim):
    """Linear interpolation is exact for affine functions in the hull."""
    centres = [torch.tensor([0.0, 3.0, 10.0], dtype=F64)] * ndim
    coords = [torch.linspace(0, 10, 9, dtype=F64)] * ndim
    nodes, w = interpolation_weights(centres, coords, kind)
    coef = torch.arange(1, ndim + 1, dtype=F64)
    values = (
        torch.stack([centres[a][nodes[:, a]] for a in range(ndim)], -1) @ coef
    )
    got = torch.tensordot(values + 2.0, w, dims=1)
    grids = torch.meshgrid(*coords, indexing="ij")
    want = sum(c * g for c, g in zip(coef, grids, strict=True)) + 2.0
    torch.testing.assert_close(got, want)


def test_simplex_is_triangulation():
    """At most d + 1 non-zero weights per pixel (one simplex)."""
    centres = [torch.tensor([0.0, 8.0])] * 3
    coords = [torch.linspace(0, 8, 7, dtype=F64)] * 3
    _, w = interpolation_weights(centres, coords, "simplex")
    assert int((w > 0).sum(0).max()) <= 4
    _, w = interpolation_weights(centres, coords, "multilinear")
    assert int((w > 0).sum(0).max()) == 8


# -- operator -------------------------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
def test_constant_grid_is_padded_convolution(ndim):
    shape = (26, 30) if ndim == 2 else (14, 16, 18)
    k = gaussian(5, [1.2] * ndim)
    psf = k.expand((3,) * ndim + k.shape).clone()
    ref = PaddedConvolution(k, shape)
    gen = torch.Generator().manual_seed(0)
    f = torch.rand(ref.grid, generator=gen, dtype=F64)
    r = torch.rand(shape, generator=gen, dtype=F64)
    for model in models(psf, centres_for(shape)):
        op = model.operator((0,) * ndim, shape)
        assert op.grid == ref.grid
        torch.testing.assert_close(op.forward(f), ref.forward(f))
        torch.testing.assert_close(op.adjoint(r), ref.adjoint(r))
        torch.testing.assert_close(op.norm, ref.norm)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, F64])
def test_adjoint_dot_product(ndim, dtype):
    """⟨H f, r⟩ = ⟨f, Hᵀ r⟩ for every model."""
    shape = (28, 33) if ndim == 2 else (14, 15, 17)
    psf = psf_grid(ndim, size=5).to(dtype)
    gen = torch.Generator().manual_seed(1)
    for model in models(psf, centres_for(shape)):
        op = model.operator((3,) * ndim, shape)
        f = torch.rand(op.grid, generator=gen, dtype=F64).to(dtype)
        r = torch.rand(shape, generator=gen, dtype=F64).to(dtype)
        lhs = float((op.forward(f) * r).sum(dtype=F64))
        rhs = float((f * op.adjoint(r)).sum(dtype=F64))
        tol = 1e-4 if dtype == torch.float32 else 1e-11
        assert abs(lhs - rhs) <= tol * abs(lhs)


def test_operator_explicit_weights():
    """Weights given as a tensor or a callable; orders are transposes."""
    gen = torch.Generator().manual_seed(2)
    k = torch.rand(2, 3, 3, generator=gen, dtype=F64)
    shape = (8, 9)
    grid = ShiftVariantConvolution(
        k, lambda c: torch.ones(2, *[x.numel() for x in c]), shape
    )
    w_in = torch.rand((2,) + grid.grid, generator=gen, dtype=F64)
    op = ShiftVariantConvolution(k, w_in, shape)
    f = torch.rand(op.grid, generator=gen, dtype=F64)
    r = torch.rand(shape, generator=gen, dtype=F64)
    assert (
        abs(float((op.forward(f) * r).sum() - (f * op.adjoint(r)).sum()))
        < 1e-11
    )
    w_out = torch.rand((2,) + shape, generator=gen, dtype=F64)
    op = ShiftVariantConvolution(k, w_out, shape, order="output")
    assert (
        abs(float((op.forward(f) * r).sum() - (f * op.adjoint(r)).sum()))
        < 1e-11
    )
    with pytest.raises(ValueError, match="weights must have shape"):
        ShiftVariantConvolution(k, w_out, shape)
    with pytest.raises(ValueError, match="order"):
        ShiftVariantConvolution(k, w_out, shape, order="both")


@pytest.mark.parametrize("ndim", [2, 3])
def test_block_whole_parity(ndim):
    """A block's operator is the whole-image operator on its core."""
    shape = (40, 44) if ndim == 2 else (18, 20, 22)
    k = 5
    psf = psf_grid(ndim, size=k)
    gen = torch.Generator().manual_seed(3)
    obj = torch.rand(shape, generator=gen, dtype=F64)
    origin = (9,) * ndim
    sub = tuple(n - 15 for n in shape)
    for model in models(psf, centres_for(shape)):
        whole = model.operator((0,) * ndim, shape)
        full = whole.forward(embed(obj, whole))
        op = model.operator(origin, sub)
        # object on the block's grid, from global positions
        pos = object_positions(sub, (k,) * ndim)
        idx = [p.long() + o for p, o in zip(pos, origin, strict=True)]
        inside = [(i >= 0) & (i < n) for i, n in zip(idx, shape, strict=True)]
        f = obj[
            torch.meshgrid(
                *[i.clamp(0, n - 1) for i, n in zip(idx, shape, strict=True)],
                indexing="ij",
            )
        ]
        for a, m in enumerate(inside):
            view = [1] * ndim
            view[a] = -1
            f = f * m.reshape(view)
        got = op.forward(f)
        core = tuple(slice(k, s - k) for s in sub)
        want = full[
            tuple(
                slice(o + k, o + s - k)
                for o, s in zip(origin, sub, strict=True)
            )
        ]
        torch.testing.assert_close(got[core], want)


# -- PSF models -------------------------------------------------------------
def test_nagy_regions_cover_block():
    psf = psf_grid(2, size=5)
    model = NagyOLearyPSF(psf, [[10.0, 30.0, 50.0]] * 2)
    coords = tuple(torch.arange(60, dtype=F64) for _ in range(2))
    regions = model.regions(coords)
    assert len(regions) == 9
    mask = torch.zeros(60, 60)
    for _, box in regions:
        mask[box] += 1
    assert (mask == 1).all()
    _, w = interpolation_weights(model.centres, coords, "nearest")
    assert regions[0][1] == (slice(0, 20), slice(0, 20))
    assert w.shape[0] == 9


def test_modes_error_falls_with_rank():
    """Eckart–Young: the PSF error is the tail of the singular values."""
    ndim = 2
    psf = psf_grid(ndim, size=9)
    centres = centres_for((40, 40))
    full = DenisPSF(psf, centres)
    coords = tuple(torch.arange(40, dtype=F64) for _ in range(ndim))

    def equivalent(model):
        k, w = model.terms(coords)
        return torch.einsum(
            "tl,tn->nl", k.reshape(k.shape[0], -1), w.reshape(k.shape[0], -1)
        )

    ref = equivalent(full)
    samples = psf.reshape(9, -1)
    errors, sample_errors = [], []
    for rank in range(1, 10):
        model = DenisPSF(psf, centres, model="modes", rank=rank)
        errors.append(float((equivalent(model) - ref).norm()))
        approx = model.kernels.reshape(rank, -1).T @ model._coeff
        err = float((approx.T - samples).norm())
        tail = float(model.singular_values[rank:].square().sum().sqrt())
        assert err == pytest.approx(tail, abs=1e-10)
        sample_errors.append(err)
    assert all(b < a for a, b in itertools.pairwise(sample_errors))
    assert all(b < a for a, b in itertools.pairwise(errors))
    assert errors[-1] < 1e-10


def test_optimal_beats_interpolation():
    """The optimal local approximation lowers the PSF error (§2.5)."""
    H, W, K = 32, 36, 9

    def psf_model(pos):
        x = torch.arange(K, dtype=F64) - K // 2
        sy = 0.8 + 2.0 * (pos[:, 0] / H) ** 2
        sx = 0.8 + 1.5 * pos[:, 1] / W
        gy = torch.exp(-(x[None] ** 2) / (2 * sy[:, None] ** 2))
        gx = torch.exp(-(x[None] ** 2) / (2 * sx[:, None] ** 2))
        k = gy[:, :, None] * gx[:, None, :]
        return k / k.sum((1, 2), keepdim=True)

    centres = [torch.linspace(0, H - 1, 3), torch.linspace(0, W - 1, 3)]
    psf = psf_model(torch.cartesian_prod(*centres)).reshape(3, 3, K, K)
    pos = torch.cartesian_prod(
        torch.arange(H, dtype=F64), torch.arange(W, dtype=F64)
    )
    truth = psf_model(pos).reshape(H * W, -1)
    coords = (torch.arange(H, dtype=F64), torch.arange(W, dtype=F64))

    def error(model):
        k, w = model.terms(coords)
        eq = torch.einsum(
            "tl,tn->nl", k.reshape(k.shape[0], -1), w.reshape(k.shape[0], -1)
        )
        return float((eq - truth).norm() / truth.norm())

    interp = DenisPSF(psf, centres)
    optimal = DenisPSF(
        psf, centres, model="optimal", psf_model=psf_model, extent=(H, W)
    )
    assert error(optimal) < 0.5 * error(interp)
    # its operator is a sum of convolutions with an exact adjoint
    op = optimal.operator((4, 5), (20, 22))
    gen = torch.Generator().manual_seed(4)
    f = torch.rand(op.grid, generator=gen, dtype=F64)
    r = torch.rand(op.shape, generator=gen, dtype=F64)
    lhs = float((op.forward(f) * r).sum())
    assert lhs == pytest.approx(float((f * op.adjoint(r)).sum()), rel=1e-11)
    # ω* is stored once (§2.5.2): the model saves without psf_model
    assert tuple(optimal.omega.shape) == (H, W, 4)
    back = ShiftVariantPSF._from_payload(optimal._payload())
    assert isinstance(back, DenisPSF) and back.psf_model is None
    assert error(back) == pytest.approx(error(optimal), rel=1e-12)
    op1 = optimal.operator((-3, 5), (20, 40))
    f = torch.rand(op1.grid, generator=gen, dtype=F64)
    torch.testing.assert_close(
        back.operator((-3, 5), (20, 40)).forward(f), op1.forward(f)
    )
    with pytest.raises(ValueError, match="integer"):
        optimal.terms((torch.tensor([0.5]), torch.tensor([1.0])))
    # the fit batch follows the memory budget
    with config(memory=8 * 81 * 4 * 50):
        small = DenisPSF(
            psf, centres, model="optimal", psf_model=psf_model, extent=(H, W)
        )
        assert small._batch() == 50
    torch.testing.assert_close(small.omega, optimal.omega)


def test_model_validation():
    psf = psf_grid(2, size=5)
    with pytest.raises(ValueError, match="centres"):
        NagyOLearyPSF(psf, [[1.0, 2.0], [1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="increasing"):
        DenisPSF(psf, [[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="interpolation"):
        NagyOLearyPSF(psf, [[1.0, 2.0, 3.0]] * 2, interpolation="cubic")
    with pytest.raises(ValueError, match="psf_model"):
        DenisPSF(psf, [[1.0, 2.0, 3.0]] * 2, model="optimal")
    with pytest.raises(ValueError, match="rank"):
        DenisPSF(psf, [[1.0, 2.0, 3.0]] * 2, model="modes", rank=10)


# -- deconvolution ----------------------------------------------------------
def sv_data(shape, model, seed=0, n=None, peak=300.0):
    gen = torch.Generator().manual_seed(seed)
    numel = int(torch.tensor(shape).prod())
    truth = torch.full(shape, 4.0, dtype=F64)
    idx = torch.randint(
        0, numel, (numel // 60 if n is None else n,), generator=gen
    )
    truth.view(-1)[idx] += peak
    op = model.operator((0,) * len(shape), shape)
    blurred = op.forward(embed(truth, op)).clamp(min=0)
    return truth, torch.poisson(blurred, generator=gen), blurred


def strong_grid(ndim, size=11):
    """PSFs from sharp (σ = 0.6) to wide (σ = 2.6) across the field."""
    psfs = []
    for idx in itertools.product(range(3), repeat=ndim):
        s = 0.6 + 1.0 * sum(idx) / ndim
        psfs.append(gaussian(size, [s] * ndim))
    return torch.stack(psfs).reshape((3,) * ndim + (size,) * ndim)


@pytest.mark.parametrize("ndim", [2, 3])
def test_rl_shift_variant_beats_single_psf(ndim):
    shape = (64, 64) if ndim == 2 else (24, 24, 24)
    psf = strong_grid(ndim, size=11 if ndim == 2 else 7)
    model = DenisPSF(psf, centres_for(shape))
    truth = checkerboard(shape, size=4 if ndim == 2 else 6, value=200.0) + 10
    op = model.operator((0,) * ndim, shape)
    gen = torch.Generator().manual_seed(0)
    data = torch.poisson(
        op.forward(embed(truth, op)).clamp(min=0), generator=gen
    )
    n_iter = 40 if ndim == 2 else 20
    central = psf[(1,) * ndim]
    single = RichardsonLucy(central, n_iter=n_iter, dtype=F64).fit(data)
    sv = RichardsonLucy(model, n_iter=n_iter, dtype=F64).fit(data)
    nagy = RichardsonLucy(
        NagyOLearyPSF(psf, centres_for(shape), interpolation="linear"),
        n_iter=n_iter,
        dtype=F64,
    ).fit(data)

    def err(x):
        return float((x - truth).norm() / truth.norm())

    e_single = err(single.apply(data))
    e_sv = err(sv.apply(data))
    assert e_sv < 0.9 * e_single
    assert err(nagy.apply(data)) < e_single
    assert RichardsonLucy(model).is_faithful


def test_rl_shift_variant_chi2_and_blocks():
    shape = (56, 60)
    model = NagyOLearyPSF(strong_grid(2, 9), centres_for(shape))
    _, data, _ = sv_data(shape, model, seed=1)
    est = RichardsonLucy(model, dtype=F64, max_iter=30).fit(data)
    out = est.apply(data)
    blocked = RichardsonLucy(
        model, dtype=F64, max_iter=30, block=(28, 30)
    ).fit(data)
    out_b = blocked.apply(data)
    assert torch.isfinite(out).all() and (out >= 0).all()
    assert est.history.extras["iterations"][0] >= 1
    # seams fade with the halo: blocks agree with the whole image closely
    rel = float((out - out_b).norm() / out.norm())
    assert rel < 0.05


@pytest.mark.parametrize("ndim", [2, 3])
def test_biggs_andrews_lockstep_matches_whole(ndim):
    """Global-α Biggs–Andrews over blocks reproduces the whole image."""
    shape = (48, 52) if ndim == 2 else (20, 22, 24)
    model = DenisPSF(psf_grid(ndim, size=5), centres_for(shape))
    _, data, _ = sv_data(shape, model, seed=2)
    whole = BiggsAndrews(model, n_iter=8, dtype=F64).fit(data).apply(data)
    est = BiggsAndrews(
        model, n_iter=8, dtype=F64, block=tuple(n // 2 for n in shape)
    ).fit(data)
    blocked = est.apply(data)
    assert est.history.extras["alpha_scope"] == ["global"]
    torch.testing.assert_close(blocked, whole, rtol=1e-8, atol=1e-8)


def test_snyder_shift_variant():
    shape = (40, 44)
    model = DenisPSF(strong_grid(2, 9), centres_for(shape))
    truth, data, _ = sv_data(shape, model, seed=3)
    est = Snyder(model, n_iter=20, dtype=F64).fit(data)
    out = est.apply(data)
    single = Snyder(model.kernels[4], n_iter=20, dtype=F64).fit(data)
    e = float((out - truth).norm())
    assert e < float((single.apply(data) - truth).norm())


def test_rl_save_load_shift_variant(tmp_path):
    from optix import load

    shape = (32, 32)
    model = DenisPSF(
        psf_grid(2, size=5), centres_for(shape), model="modes", rank=4
    )
    _, data, _ = sv_data(shape, model, seed=4)
    est = RichardsonLucy(model, n_iter=5, dtype=F64).fit(data)
    path = tmp_path / "rl.pt"
    est.save(path)
    back = load(path)
    assert isinstance(back.psf, DenisPSF) and back.psf.rank == 4
    torch.testing.assert_close(
        back.apply(data).double(), est.apply(data), rtol=1e-4, atol=1e-3
    )


def test_rl_save_load_optimal(tmp_path):
    """An "optimal" model saves its kernels and ω*, not the callable."""
    from optix import load

    shape = (24, 26)
    psf = psf_grid(2, size=5)
    centres = centres_for(shape)

    def psf_model(pos):
        return psf[1, 1].expand(pos.shape[0], 5, 5) * (1 + pos[:, :1, None])

    model = DenisPSF(
        psf, centres, model="optimal", psf_model=psf_model, extent=shape
    )
    _, data, _ = sv_data(shape, model, seed=5)
    est = BiggsAndrews(model, n_iter=4, dtype=F64).fit(data)
    path = tmp_path / "ba.pt"
    est.save(path)
    back = load(path)
    assert back.psf.model == "optimal" and back.psf.psf_model is None
    torch.testing.assert_close(
        back.apply(data).double(), est.apply(data), rtol=1e-4, atol=1e-3
    )


def test_n_terms_bounds_block_terms():
    """The memory estimate counts the terms meeting the image."""
    shape = (40, 44)
    psf = psf_grid(2, size=5, nodes=3)
    for model in models(psf, centres_for(shape)):
        bound = model.n_terms(shape)
        kernels, _ = model.terms(
            tuple(torch.arange(-5, n + 5, dtype=F64) for n in shape)
        )
        assert kernels.shape[0] <= bound <= 9
    # PSFs whose nodes all lie in one half of the axis: fewer terms
    model = NagyOLearyPSF(psf, [[0.0, 10.0, 200.0], [0.0, 10.0, 200.0]])
    assert model.n_terms((12, 12)) == 4
    assert (
        DenisPSF(psf, centres_for(shape), model="modes", rank=2).n_terms(shape)
        == 2
    )


def test_wiener_butterworth_rejects_shift_variant():
    shape = (32, 32)
    model = DenisPSF(psf_grid(2, size=5), centres_for(shape))
    with pytest.raises(TypeError, match="single PSF"):
        WienerButterworth(model)
    est = WienerButterworth(psf_grid(2, size=5)[1, 1])
    est.psf = model
    with pytest.raises(TypeError, match="single PSF"):
        est.fit(torch.rand(shape, dtype=F64) + 1).apply(
            torch.rand(shape, dtype=F64) + 1
        )


# -- Nagy & O'Leary preconditioned CG ---------------------------------------
@pytest.mark.parametrize("shape", [(7, 6), (5, 5, 4)])
def test_optimal_circulant_is_frobenius_projection(shape):
    """Eigenvalues match the projection of the Toeplitz matrix (Chan)."""
    gen = torch.Generator().manual_seed(5)
    d = len(shape)
    k = torch.rand((3,) * (d - 1) + (4,), generator=gen, dtype=F64)
    n = int(torch.tensor(shape).prod())
    pts = list(itertools.product(*[range(s) for s in shape]))
    first = torch.zeros(shape, dtype=F64)
    for j in pts:
        total = 0.0
        for b in pts:
            a = tuple(
                (bb + jj) % s for bb, jj, s in zip(b, j, shape, strict=True)
            )
            off = tuple(
                aa - bb + kk // 2
                for aa, bb, kk in zip(a, b, k.shape, strict=True)
            )
            if all(0 <= o < kk for o, kk in zip(off, k.shape, strict=True)):
                total += float(k[off])
        first[j] = total / n
    torch.testing.assert_close(
        optimal_circulant(k, shape), torch.fft.rfftn(first)
    )


@pytest.mark.parametrize("interpolation", ["constant", "linear"])
def test_pcg_accelerates_cg(interpolation):
    """§5: with τ = 0, PCG reaches a given error in fewer iterations."""
    shape = (64, 64)
    psf = sharp_grid(2)
    centres = centres_for(shape)
    model = NagyOLearyPSF(psf, centres, interpolation=interpolation)
    op = model.operator((0, 0), shape)
    truth = checkerboard(shape)
    data = op.forward(embed(truth, op))
    est = NagyOLeary(psf, centres, dtype=F64)
    pc = est._preconditioner(model, (0, 0), shape, F64)

    def error(f):
        return float((op.crop(f) - truth).norm() / truth.norm())

    for n in (3, 5):
        plain, _ = cgls(data, op, n)
        fast, _ = cgls(data, op, n, precond=pc)
        assert error(fast) < 0.7 * error(plain)


@pytest.mark.parametrize("ndim", [2, 3])
def test_nagy_oleary_restores(ndim):
    shape = (48, 48) if ndim == 2 else (20, 20, 20)
    psf = sharp_grid(ndim, size=9 if ndim == 2 else 7)
    centres = centres_for(shape)
    model = NagyOLearyPSF(psf, centres)
    op = model.operator((0,) * ndim, shape)
    truth = checkerboard(shape, size=6)
    data = op.forward(embed(truth, op))
    gen = torch.Generator().manual_seed(6)
    data = data + 0.01 * torch.randn(data.shape, generator=gen, dtype=F64)
    err0 = float((data - truth).norm() / truth.norm())
    for precondition, tau in [(False, 0.0), (True, 0.0), (True, 0.05)]:
        est = NagyOLeary(
            psf, centres, precondition=precondition, tau=tau, dtype=F64
        ).fit(data)
        out = est.apply(data)
        assert float((out - truth).norm() / truth.norm()) < 0.7 * err0
        assert est.history.extras["iterations"] == [14.0]
    # the paper's own variants keep the estimator faithful
    assert NagyOLeary(psf, centres).is_faithful
    for kw in (
        {"tau": 0.05},
        {"precondition": False},
        {"interpolation": "linear"},
    ):
        assert NagyOLeary(psf, centres, **kw).is_faithful
    assert not NagyOLeary(psf, centres, boundary="extended").is_faithful
    with pytest.raises(ValueError, match="boundary"):
        NagyOLeary(psf, centres, boundary="periodic")
    # the extended object (extension) restores as well
    est = NagyOLeary(psf, centres, boundary="extended", dtype=F64).fit(data)
    out = est.apply(data)
    assert float((out - truth).norm() / truth.norm()) < 0.7 * err0


def test_nagy_oleary_zero_boundary_is_restriction():
    """boundary="zero": unknowns on the image only, as the paper's K."""
    shape = (30, 34)
    psf = psf_grid(2, size=5)
    centres = centres_for(shape)
    data = torch.rand(shape, dtype=F64) * 10
    est = NagyOLeary(psf, centres, n_iter=3, dtype=F64).fit(data)
    model = est.model
    op = model.operator((0, 0), shape)
    mask = est._inside(model, (0, 0), shape, data)
    assert float(mask.sum()) == data.numel()
    torch.testing.assert_close(op.crop(mask), torch.ones_like(data))
    # an inner block keeps the image pixels around it as unknowns
    inner = est._inside(model, (10, 10), (40, 40), torch.zeros(8, 8))
    assert bool((inner == 1).all())


def test_nagy_oleary_blocks_and_shortcut():
    shape = (40, 44)
    psf = psf_grid(2, size=5)
    centres = centres_for(shape)
    data = torch.rand(2, *shape, dtype=F64) * 10
    out = nagy_oleary(data, psf, centres, n_iter=4, tau=0.05, dtype=F64)
    blocked = nagy_oleary(
        data, psf, centres, n_iter=4, tau=0.05, dtype=F64, block=(20, 22)
    )
    assert out.shape == data.shape and torch.isfinite(blocked).all()
    with pytest.raises(ValueError, match="centres"):
        NagyOLeary(psf)
    with pytest.raises(RuntimeError, match="fit"):
        NagyOLeary(psf, centres).apply(data)


def test_nagy_oleary_blocked_close_to_whole():
    """Per-block CG differs from the whole-image CG, within ≈ 15 %.

    Each block runs its own CGLS (own step sizes, hence its own
    regularisation), so the block-wise restoration is not the global CG
    iterate. With 24 × 24 cores of a 48 × 48 image (all pixels are block
    cores), the relative L2 difference is ≈ 14 %; it is asserted below
    25 %, and both are about as close to the truth.
    """
    shape = (48, 48)
    psf = psf_grid(2, size=5)
    centres = centres_for(shape)
    model = NagyOLearyPSF(psf, centres)
    op = model.operator((0, 0), shape)
    truth = checkerboard(shape, size=4)
    gen = torch.Generator().manual_seed(0)
    data = op.forward(embed(truth, op))
    data = data + torch.randn(shape, generator=gen, dtype=F64)
    whole = nagy_oleary(data, psf, centres, tau=0.1, dtype=F64)
    blocked = nagy_oleary(
        data, psf, centres, tau=0.1, dtype=F64, block=(24, 24)
    )
    rel = float((blocked - whole).norm() / whole.norm())
    assert rel < 0.25

    def err(x):
        return float((x - truth).norm() / truth.norm())

    assert err(blocked) < 1.1 * err(whole)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("precondition", [False, True])
def test_nagy_oleary_float32_matches_float64(ndim, precondition):
    shape = (32, 32) if ndim == 2 else (14, 14, 14)
    psf = sharp_grid(ndim, size=9 if ndim == 2 else 5)
    centres = centres_for(shape)
    op = NagyOLearyPSF(psf, centres).operator((0,) * ndim, shape)
    truth = checkerboard(shape, size=4)
    gen = torch.Generator().manual_seed(7)
    data = op.forward(embed(truth, op))
    data = data + 0.01 * torch.randn(shape, generator=gen, dtype=F64)
    out = {}
    for dtype in (torch.float32, F64):
        est = NagyOLeary(psf, centres, precondition=precondition, dtype=dtype)
        out[dtype] = est.fit(data.to(dtype)).apply(data.to(dtype))
    assert out[torch.float32].dtype == torch.float32
    rel = float(
        (out[torch.float32].double() - out[F64]).norm() / out[F64].norm()
    )
    # CG amplifies float32 rounding over the iterations: ~3e-7 locally,
    # 1.5e-4 on CI (3-D plain CG); 1e-3 still catches dtype bugs
    assert rel < 1e-3
    # cgls itself, on the operator in both precisions
    f32, _ = cgls(
        data.float(),
        NagyOLearyPSF(psf.float(), centres).operator((0,) * ndim, shape),
        5,
    )
    f64, _ = cgls(data, op, 5)
    assert float((f32.double() - f64).norm() / f64.norm()) < 1e-4


def test_pcg_with_tau_does_not_diverge_on_gaussian_psfs():
    """Smooth PSFs: τ = 0 diverges, τ = 0.2 stays close to plain CG."""
    shape = (64, 64)
    psf = psf_grid(2, size=9)  # Gaussian, σ 0.7–2.1
    centres = centres_for(shape)
    model = NagyOLearyPSF(psf, centres)
    op = model.operator((0, 0), shape)
    truth = checkerboard(shape, size=4)
    gen = torch.Generator().manual_seed(8)
    data = op.forward(embed(truth, op))
    data = data + torch.randn(shape, generator=gen, dtype=F64)
    err0 = float((data - truth).norm() / truth.norm())
    regions = model.regions(tuple(torch.arange(n, dtype=F64) for n in shape))
    boxes = [(model.kernels[i], box) for i, box in regions]

    def errors(precond):
        out = []
        cgls(
            data,
            op,
            14,
            precond=precond,
            callback=lambda k, f: out.append(
                float((op.crop(f) - truth).norm() / truth.norm())
            ),
        )
        return out

    plain = errors(None)
    pcg = errors(RegionPreconditioner(boxes, tau=0.2))
    assert max(pcg) < 1.25 * max(plain)  # no blow-up at any iteration
    assert pcg[-1] < 1.2 * plain[-1]  # about as good as CG, not better
    assert errors(RegionPreconditioner(boxes, tau=0.0))[-1] > err0
