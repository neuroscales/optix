import pytest
import torch

from optix import stats as ostats
from optix.distributions import (
    CategoricalStats,
    Dirichlet,
    DirichletStats,
    Gamma,
    GammaStats,
    Normal,
    NormalStats,
    posterior,
)

D64 = torch.float64


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _data(kind, n=30, seed=0, dtype=D64):
    g = _gen(seed)
    if kind == "gamma":
        return 0.1 + torch.rand(n, 4, generator=g, dtype=dtype) * 3
    if kind == "dirichlet":
        return torch.softmax(
            torch.randn(n, 4, 3, generator=g, dtype=dtype), -1
        )
    if kind == "categorical":
        return torch.softmax(
            torch.randn(n, 4, 3, generator=g, dtype=dtype), -1
        )
    return torch.randn(n, 4, generator=g, dtype=dtype) * 2 + 1


CLASSES = {
    "gamma": GammaStats,
    "dirichlet": DirichletStats,
    "categorical": CategoricalStats,
    "normal": NormalStats,
}


_COUNT_DTYPES = (torch.int64, D64)


def _close_stats(a, b, **kw):
    assert type(a) is type(b)
    c = a._count_index
    pairs = zip(a.astuple(), b.astuple(), strict=True)
    for i, (x, y) in enumerate(pairs):
        shape = torch.broadcast_shapes(x.shape, y.shape)
        x, y = x.expand(shape), y.expand(shape)
        if i == c:  # exact int64 or float64 counts, compared by value
            assert x.dtype in _COUNT_DTYPES and y.dtype in _COUNT_DTYPES
            x, y = x.to(D64), y.to(D64)
        torch.testing.assert_close(x, y, **kw)


@pytest.mark.parametrize("kind", list(CLASSES))
@pytest.mark.parametrize("split", [1, 7, 30])
def test_merge_law(kind, split):
    cls = CLASSES[kind]
    x = _data(kind)
    full = cls.from_samples(x, dim=0)
    a = cls.from_samples(x[:split], dim=0)
    b = cls.from_samples(x[split:], dim=0)
    _close_stats(a + b, full)
    acc = cls.zeros((4,), event_shape=full.event_shape, dtype=D64)
    acc.add_(a).merge(b)
    _close_stats(acc, full)
    _close_stats(sum([a, b]), full)
    _close_stats(0 + full, full)


@pytest.mark.parametrize("kind", list(CLASSES))
def test_weights_equal_repeats_and_masks(kind):
    cls = CLASSES[kind]
    x = _data(kind, n=5)
    w = torch.tensor([1.0, 2.0, 0.0, 3.0, 1.0], dtype=D64)
    w = w.reshape(5, 1)
    rep = torch.repeat_interleave(x, w.flatten().long(), 0)
    _close_stats(cls.from_samples(x, w, dim=0), cls.from_samples(rep, dim=0))
    mask = w > 0
    _close_stats(
        cls.from_samples(x, mask, dim=0),
        cls.from_samples(x[mask.flatten()], dim=0),
    )


@pytest.mark.parametrize("kind", list(CLASSES))
def test_zero_weight_nan_ignored(kind):
    cls = CLASSES[kind]
    x = _data(kind, n=4).clone()
    x[0] = torch.nan
    w = torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=D64).reshape(4, 1)
    out = cls.from_samples(x, w, dim=0)
    _close_stats(out, cls.from_samples(x[1:], dim=0))


@pytest.mark.parametrize("kind", list(CLASSES))
def test_sum_scale_state_dict_to(kind):
    cls = CLASSES[kind]
    x = _data(kind)
    per = cls.from_samples(x, dim=())
    assert per.batch_shape == x.shape[: x.ndim - len(per.event_shape)]
    _close_stats(per.sum(0), cls.from_samples(x, dim=0))
    _close_stats(per.sum(), cls.from_samples(x, dim=None))
    _close_stats(per.sum((-2, -1)), cls.from_samples(x, dim=None))
    s = cls.from_samples(x, dim=0)
    double = s.scale(2.0)
    torch.testing.assert_close(double.n, 2 * s.n.to(D64))
    _close_stats(cls.from_state_dict(s.state_dict()), s)
    f32 = s.to(dtype=torch.float32)
    assert s.dtype == D64
    # counts keep their dtype; CategoricalStats has nothing else
    assert f32.dtype == (D64 if kind == "categorical" else torch.float32)
    assert f32.astuple()[s._count_index].dtype in _COUNT_DTYPES


def test_scale_equals_duplicated_data():
    x = _data("normal")
    s = NormalStats.from_samples(x, dim=0)
    _close_stats(
        s.scale(2.0), NormalStats.from_samples(torch.cat([x, x]), dim=0)
    )
    g = GammaStats.from_samples(_data("gamma"), dim=0)
    _close_stats(g.scale(3.0), g + g + g)


def test_add_float32_into_float64_accumulator():
    x = _data("gamma", dtype=torch.float32)
    acc = GammaStats.zeros((4,), dtype=D64)
    for chunk in x.split(7):
        acc.add_(GammaStats.from_samples(chunk, dim=0))
    assert acc.dtype == D64
    ref = GammaStats.from_samples(x.double(), dim=0)
    _close_stats(acc, ref, rtol=1e-6, atol=1e-6)


def test_native_reduction_with_state_dtype():
    x = _data("gamma", dtype=torch.float32)
    s = GammaStats.from_samples(x, dim=0, dtype=D64)
    assert s.dtype == D64
    assert GammaStats.from_samples(x, dim=0).dtype == torch.float32


def test_add_errors():
    acc = GammaStats.zeros((3,), dtype=D64)
    with pytest.raises(
        ValueError, match=r"cannot accumulate batch shape \(4, 3\)"
    ):
        acc.add_(GammaStats.zeros((4, 3), dtype=D64))
    shared = GammaStats(torch.tensor(1.0), torch.ones(3), torch.ones(3))
    with pytest.raises(ValueError, match="does not own"):
        shared.add_(shared)
    shared.copy().add_(shared)  # copies own full buffers
    with pytest.raises(TypeError):
        acc.add_(NormalStats.zeros((3,), dtype=D64))


def test_normal_stats_large_offset_float32():
    g = _gen(0)
    x = 1e4 + torch.randn(10_000, generator=g)
    acc = None
    for chunk in x.split(1000):
        s = NormalStats.from_samples(chunk, dim=0)
        acc = s.copy() if acc is None else acc.add_(s)
    ref = ((x.double() - x.double().mean()) ** 2).sum()
    torch.testing.assert_close(acc.scatter.double(), ref, rtol=1e-3, atol=0)


def test_normal_stats_from_moments():
    x = _data("normal")
    m = ostats.Moments(order=2, dim=0).update(x)
    s = NormalStats.from_moments(m)
    _close_stats(s, NormalStats.from_samples(x, dim=0))
    with pytest.raises(ValueError):
        NormalStats.from_moments(ostats.Moments(order=2))


def test_normal_stats_raw_sums():
    x = _data("normal")
    s = NormalStats.from_samples(x, dim=0)
    torch.testing.assert_close(s.sum_x, x.sum(0))
    torch.testing.assert_close(s.sum_x_sq, (x**2).sum(0))


def test_from_expectations():
    x = _data("normal")
    _close_stats(
        NormalStats.from_expectations(x, x**2, dim=0),
        NormalStats.from_samples(x, dim=0),
    )
    # latent variances add to the scatter
    out = NormalStats.from_expectations(x, x**2 + 0.5, dim=0)
    torch.testing.assert_close(
        out.scatter, NormalStats.from_samples(x, dim=0).scatter + 15
    )
    g = _data("gamma")
    _close_stats(
        GammaStats.from_expectations(g.log(), g, dim=0),
        GammaStats.from_samples(g, dim=0),
    )
    d = _data("dirichlet")
    _close_stats(
        DirichletStats.from_expectations(d.log(), dim=0),
        DirichletStats.from_samples(d, dim=0),
    )
    _close_stats(
        CategoricalStats.from_expectations(d, dim=0),
        CategoricalStats.from_samples(d, dim=0),
    )


def test_from_labels():
    labels = torch.tensor([[0, 2], [1, 2], [2, 2]])
    s = CategoricalStats.from_labels(labels, 3, dim=0, dtype=D64)
    ref = torch.tensor([[1, 1, 1], [0, 0, 3]])
    torch.testing.assert_close(s.counts, ref)  # exact int64 counts
    torch.testing.assert_close(s.n, torch.tensor([3, 3]))


# -- posterior ------------------------------------------------------------


def test_dirichlet_posterior():
    prior = Dirichlet(torch.tensor([1.0, 2.0, 3.0], dtype=D64))
    counts = torch.rand(5, 6, 3, dtype=D64, generator=_gen(1))
    post = posterior(prior, CategoricalStats(counts))
    assert post.batch_shape == (5, 6)
    torch.testing.assert_close(
        post.concentration, prior.concentration + counts
    )
    # the shared prior is not expanded in the stored result's computation
    assert prior.stored()[0].shape == (3,)
    assert prior.posterior(CategoricalStats(counts)).batch_shape == (5, 6)


def _loglik_gamma_rate(x, shape, rate):
    return Gamma(shape, rate).log_prob(x).sum(0)


@pytest.mark.parametrize(
    "case", ["gamma_rate", "gamma_precision", "normal_mean"]
)
def test_posterior_exact_conjugacy(case):
    g = _gen(3)
    if case == "gamma_rate":
        prior = Gamma(
            torch.tensor(2.0, dtype=D64), torch.tensor(1.5, dtype=D64)
        )
        x = 0.2 + torch.rand(20, generator=g, dtype=D64)
        stats, known = GammaStats.from_samples(x, dim=0), {"shape": 3.0}

        def loglik(t):
            return Gamma(3.0, t, dtype=D64).log_prob(x).sum()

        def at(q, t):
            return q.log_prob(t)

    elif case == "gamma_precision":
        prior = Gamma(
            torch.tensor(2.0, dtype=D64), torch.tensor(1.5, dtype=D64)
        )
        x = torch.randn(20, generator=g, dtype=D64) + 0.3
        stats, known = NormalStats.from_samples(x, dim=0), {"mean": 0.5}

        def loglik(t):
            return Normal(0.5, precision=t, dtype=D64).log_prob(x).sum()

        def at(q, t):
            return q.log_prob(t)

    else:
        prior = Normal(torch.tensor(0.5, dtype=D64), 2.0)
        x = torch.randn(20, generator=g, dtype=D64) + 0.3
        stats, known = NormalStats.from_samples(x, dim=0), {"precision": 4.0}

        def loglik(t):
            return Normal(t, precision=4.0, dtype=D64).log_prob(x).sum()

        def at(q, t):
            return q.log_prob(t)

    post = posterior(prior, stats, **known)
    ts = torch.tensor([0.3, 0.8, 1.7, 2.5], dtype=D64)
    consts = [at(post, t) - at(prior, t) - loglik(t) for t in ts]
    for c in consts[1:]:
        torch.testing.assert_close(c, consts[0])


def test_posterior_hand_formulas_and_sequential():
    g = _gen(4)
    x = torch.randn(40, 3, generator=g, dtype=D64)
    s1 = NormalStats.from_samples(x[:15], dim=0)
    s2 = NormalStats.from_samples(x[15:], dim=0)
    s = s1 + s2
    p = Gamma(torch.tensor(2.0, dtype=D64), 1.0)
    post = posterior(p, s, mean=0.2)
    torch.testing.assert_close(post.shape, 2 + s.n / 2 + 0 * post.rate)
    torch.testing.assert_close(post.rate, 1 + ((x - 0.2) ** 2).sum(0) / 2)
    seq = posterior(posterior(p, s1, mean=0.2), s2, mean=0.2)
    torch.testing.assert_close(seq.rate, post.rate)
    torch.testing.assert_close(seq.shape, post.shape)
    m = Normal(torch.tensor(0.0, dtype=D64), precision=2.0)
    pm = posterior(m, s, precision=3.0)
    lam = 2 + 40 * 3.0
    torch.testing.assert_close(pm.precision, torch.full((3,), lam, dtype=D64))
    torch.testing.assert_close(pm.mean, 3.0 * x.sum(0) / lam)
    seqm = posterior(posterior(m, s1, precision=3.0), s2, precision=3.0)
    torch.testing.assert_close(seqm.mean, pm.mean)
    gs = GammaStats.from_samples(x.abs() + 0.1, dim=0)
    pg = posterior(p, gs, shape=2.5)
    torch.testing.assert_close(pg.shape, 2 + 2.5 * gs.n + 0 * pg.rate)
    torch.testing.assert_close(pg.rate, 1 + (x.abs() + 0.1).sum(0))


def test_posterior_empty_returns_prior():
    p = Gamma(torch.tensor([2.0, 3.0], dtype=D64), torch.tensor([1.0, 4.0]))
    empty = NormalStats.zeros((2,), dtype=D64)
    post = posterior(p, empty, mean=1.0)
    torch.testing.assert_close(post.shape, p.shape)
    torch.testing.assert_close(post.rate, p.rate.double())
    n = Normal(
        torch.tensor([1.0, 2.0], dtype=D64), precision=torch.tensor([0.0, 1])
    )
    pn = posterior(n, empty, precision=2.0)
    torch.testing.assert_close(pn.mean, n.mean)
    torch.testing.assert_close(pn.precision, n.precision.double())
    d = Dirichlet(torch.zeros(3, dtype=D64))  # improper, floored
    pd = posterior(d, CategoricalStats.zeros(event_shape=(3,), dtype=D64))
    assert (pd.concentration > 0).all()


def test_posterior_shared_prior_per_pixel():
    p = Gamma(torch.tensor(2.0, dtype=D64), 1.0)
    x = torch.rand(10, 8, 8, dtype=D64, generator=_gen(5)) + 0.1
    post = posterior(p, GammaStats.from_samples(x, dim=0), shape=1.0)
    assert post.batch_shape == (8, 8)


def test_posterior_errors():
    p = Gamma(torch.tensor(2.0, dtype=D64), 1.0)
    s = NormalStats.zeros((2,), dtype=D64)
    with pytest.raises(TypeError, match="mean"):
        posterior(p, s)
    with pytest.raises(
        NotImplementedError, match="no conjugate update of a Gamma"
    ):
        posterior(p, DirichletStats.zeros(event_shape=(3,), dtype=D64))
    with pytest.raises(ValueError, match="event shapes differ"):
        posterior(
            Dirichlet(torch.ones(4)),
            CategoricalStats.zeros(event_shape=(3,), dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="not broadcastable"):
        posterior(Gamma(torch.ones(3), 1.0), NormalStats.zeros((4,)), mean=0.0)


# -- fits -----------------------------------------------------------------


def test_gamma_fit_exact_expectations():
    mu = torch.tensor([0.5, 2.0, 10.0], dtype=D64)
    nu = torch.tensor([0.7, 5.0, 300.0], dtype=D64)
    q = Gamma.from_mean_shape(mu, nu)
    s = GammaStats(1.0, q.mean, q.mean_log)
    fit = Gamma.fit(s)
    assert fit.form == ("mean", "shape")
    torch.testing.assert_close(fit.mean, mu)
    torch.testing.assert_close(fit.shape, nu, rtol=1e-8, atol=0)


def test_gamma_fit_samples():
    torch.manual_seed(0)
    x = torch.distributions.Gamma(
        torch.tensor(3.0, dtype=D64), torch.tensor(2.0, dtype=D64)
    ).sample((100_000,))
    fit = Gamma.fit(GammaStats.from_samples(x, dim=0))
    assert fit.shape.item() == pytest.approx(3.0, rel=0.03)
    assert fit.rate.item() == pytest.approx(2.0, rel=0.03)


def test_gamma_fit_fixed_mean_stationarity():
    x = 0.1 + torch.rand(500, 2, dtype=D64, generator=_gen(6)) * 4
    s = GammaStats.from_samples(x, dim=0)
    mean = torch.tensor([1.5, 2.5], dtype=D64)
    fit = Gamma.fit(s, mean=mean)
    torch.testing.assert_close(fit.mean, mean)
    nu = fit.shape
    xbar, lbar = s.sum_x / s.n, s.sum_log_x / s.n
    lhs = nu.log() - torch.digamma(nu)
    rhs = xbar / mean - 1 + mean.log() - lbar
    torch.testing.assert_close(lhs, rhs)


def test_gamma_fit_empty():
    s = GammaStats(
        torch.tensor([0.0, 10.0], dtype=D64),
        torch.tensor([0.0, 20.0]),
        torch.tensor([0.0, 5.0]),
    )
    nan = Gamma.fit(s)
    assert torch.isnan(nan.shape[0]) and torch.isfinite(nan.shape[1])
    init = Gamma(torch.tensor(4.0, dtype=D64), 2.0)
    fit = Gamma.fit(s, init=init)
    assert fit.shape[0].item() == 4.0 and fit.mean[0].item() == 2.0
    # gradients stay finite through the empty entry
    sx = torch.tensor([0.0, 20.0], dtype=D64, requires_grad=True)
    out = Gamma.fit(GammaStats(s.n, sx, s.sum_log_x), init=init)
    (g,) = torch.autograd.grad(out.shape.sum() + out.mean.sum(), sx)
    assert torch.isfinite(g).all()


def test_dirichlet_fit():
    alpha = torch.tensor([[0.5, 2.0, 7.0], [3.0, 3.0, 1.0]], dtype=D64)
    q = Dirichlet(alpha)
    s = DirichletStats(torch.tensor([3.0, 0.0], dtype=D64), 3 * q.mean_log)
    init = Dirichlet(torch.tensor([1.0, 2.0, 5.0], dtype=D64))
    fit = Dirichlet.fit(s, init, niter=50)
    torch.testing.assert_close(fit.concentration[0], alpha[0])
    torch.testing.assert_close(fit.concentration[1], init.concentration)
    fit2 = Dirichlet.fit(s, torch.ones(3, dtype=D64), niter=50)
    torch.testing.assert_close(fit2.concentration[0], alpha[0])


# -- counts and labels (regressions) --------------------------------------

_BIG = 2**24 + 3  # not representable in float32


def test_unweighted_counts_are_exact_int64():
    x = torch.ones((), dtype=torch.float32).expand(_BIG)  # no memory
    for cls in (GammaStats, NormalStats):
        s = cls.from_samples(x + 1, dim=0)
        assert s.n.dtype == torch.int64 and int(s.n) == _BIG
        assert s.dtype == torch.float32  # the other fields keep theirs
    d = DirichletStats.from_samples(torch.full((7, 3), 1 / 3), dim=0)
    assert d.n.dtype == torch.int64 and int(d.n) == 7


def test_weighted_counts_are_float64():
    x = _data("gamma", dtype=torch.float32)
    w = torch.rand(x.shape, generator=_gen(2))
    s = GammaStats.from_samples(x, w, dim=0)
    assert s.n.dtype == D64 and s.dtype == torch.float32
    torch.testing.assert_close(s.n, w.sum(0).double())
    c = CategoricalStats.from_samples(_data("categorical").float(), dim=0)
    assert c.counts.dtype == D64


def test_merge_int64_and_float64_counts_past_2_24():
    x = torch.full((_BIG,), 2.0).expand(_BIG)
    big = NormalStats.from_samples(x, dim=0)  # int64 count
    one = NormalStats.from_samples(torch.tensor([2.0]), torch.ones(1), dim=0)
    assert big.n.dtype == torch.int64 and one.n.dtype == D64
    out = big + one
    assert out.n.dtype == D64 and out.n.item() == _BIG + 1
    assert (big + big).n.dtype == torch.int64
    assert (big + big).n.item() == 2 * _BIG
    torch.testing.assert_close(out.mean, torch.tensor(2.0))
    # in place: an int64 accumulator takes a weighted batch
    acc = GammaStats.zeros(dtype=torch.float32)
    assert acc.n.dtype == torch.int64
    acc.add_(GammaStats.from_samples(x, dim=0))
    acc.add_(GammaStats.from_samples(x, dim=0))
    assert acc.n.dtype == torch.int64 and acc.n.item() == 2 * _BIG
    acc.add_(
        GammaStats.from_samples(torch.tensor([2.0]), torch.ones(1), dim=0)
    )
    assert acc.n.dtype == D64 and acc.n.item() == 2 * _BIG + 1
    assert acc.dtype == torch.float32
    # scaling a count gives a float64 count
    assert big.scale(0.5).n.dtype == D64
    assert big.scale(0.5).n.item() == _BIG / 2


def test_from_labels_matches_one_hot_without_building_it(monkeypatch):
    g = _gen(4)
    labels = torch.randint(0, 5, (6, 7, 8), generator=g)
    w = torch.rand(6, 7, 8, generator=g, dtype=torch.float32)
    oh = torch.nn.functional.one_hot(labels, 5).double()

    def no_one_hot(*args, **kwargs):
        raise AssertionError("one_hot must not be called")

    monkeypatch.setattr(torch.nn.functional, "one_hot", no_one_hot)
    for dim in (None, 0, (0, 2), -1, ()):
        s = CategoricalStats.from_labels(labels, 5, dim=dim)
        ref = CategoricalStats.from_samples(oh, dim=dim)
        assert s.counts.dtype == torch.int64
        torch.testing.assert_close(s.counts.double(), ref.counts)
        sw = CategoricalStats.from_labels(labels, 5, w, dim=dim)
        ref_w = CategoricalStats.from_samples(oh, w.double(), dim=dim)
        assert sw.counts.dtype == D64  # summed in float32, stored in f64
        torch.testing.assert_close(sw.counts, ref_w.counts, rtol=1e-6, atol=0)
    s = CategoricalStats.from_labels(torch.tensor(3), 5, dim=None)
    torch.testing.assert_close(s.counts, torch.tensor([0, 0, 0, 1, 0]))


def test_posterior_ignores_mean_of_empty_stats():
    nan = torch.tensor([float("nan"), 1.0], dtype=D64)
    stats = NormalStats(
        torch.tensor([0.0, 2.0], dtype=D64), nan, torch.zeros(2, dtype=D64)
    )
    prior = Normal(torch.tensor(0.5, dtype=D64), precision=2.0)
    post = posterior(prior, stats, precision=torch.tensor(3.0, dtype=D64))
    assert torch.isfinite(post.mean).all()
    torch.testing.assert_close(post.mean[0], torch.tensor(0.5, dtype=D64))
    g = posterior(
        Gamma(torch.tensor(2.0, dtype=D64), 1.0),
        stats,
        mean=torch.tensor(0.0, dtype=D64),
    )
    assert torch.isfinite(g.rate).all()
    torch.testing.assert_close(g.rate[0], torch.tensor(1.0, dtype=D64))


def test_posterior_keeps_prior_dtype_with_float64_counts():
    x = _data("gamma", dtype=torch.float32)
    stats = GammaStats.from_samples(x, torch.ones_like(x), dim=0)
    assert stats.n.dtype == D64
    prior = Gamma(torch.tensor(1.0), 1.0)
    post = posterior(prior, stats, shape=torch.tensor(2.0))
    assert post.dtype == torch.float32
    d = posterior(
        Dirichlet(torch.ones(3)),
        CategoricalStats.from_labels(torch.tensor([0, 2, 2]), 3, dim=0),
    )
    assert d.dtype == torch.float32
    torch.testing.assert_close(d.concentration, torch.tensor([2.0, 1.0, 3.0]))
    assert Gamma.fit(stats).dtype == torch.float32
