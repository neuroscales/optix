import warnings

import pytest
import torch
import torch.distributions as TD

import optix
from optix import _compile
from optix._compile import maybe_compile
from optix.distributions import (
    Categorical,
    Dirichlet,
    Distribution,
    Gamma,
    Normal,
    functional,
    kl_divergence,
)

D64 = torch.float64


def _rand(*shape, lo=0.1, hi=5.0, seed=0, dtype=D64):
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(shape, generator=g, dtype=dtype)


def _close(a, b, **kw):
    torch.testing.assert_close(a, b, **kw)


# -- builders: (ours, oracle) pairs on the same parameters ---------------


def _gamma(shape=(4, 3), seed=0):
    a, b = _rand(*shape, seed=seed), _rand(*shape, seed=seed + 1)
    return Gamma(a, b), TD.Gamma(a, b)


def _dirichlet(shape=(4, 3), seed=0):
    a = _rand(*shape, seed=seed)
    return Dirichlet(a), TD.Dirichlet(a)


def _normal(shape=(4, 3), seed=0):
    m = _rand(*shape, seed=seed) - 2
    lam = _rand(*shape, seed=seed + 1)
    return Normal(m, precision=lam), TD.Normal(m, lam.rsqrt())


def _onehot(shape=(4, 3), seed=0):
    logits = torch.randn(
        tuple(shape), dtype=D64, generator=torch.Generator().manual_seed(seed)
    )
    return Categorical(logits=logits), TD.OneHotCategorical(logits=logits)


def _categorical(shape=(4, 3), seed=0):
    logits = torch.randn(
        tuple(shape), dtype=D64, generator=torch.Generator().manual_seed(seed)
    )
    return Categorical(logits=logits), TD.Categorical(logits=logits)


BUILDERS = {
    "gamma": _gamma,
    "dirichlet": _dirichlet,
    "normal": _normal,
    "onehot": _onehot,
    "categorical": _categorical,
}


# -- oracle comparisons ---------------------------------------------------


@pytest.mark.parametrize("family", ["gamma", "dirichlet", "normal", "onehot"])
def test_oracle_mean_variance(family):
    q, ref = BUILDERS[family]()
    _close(q.mean, ref.mean)
    _close(q.variance, ref.variance)


@pytest.mark.parametrize("family", list(BUILDERS))
def test_oracle_entropy_log_prob_kl(family):
    q, ref = BUILDERS[family](seed=0)
    p, ref_p = BUILDERS[family](seed=10)
    _close(q.entropy(), ref.entropy())
    x = ref.sample((5,))
    _close(q.log_prob(x), ref.log_prob(x))
    _close(kl_divergence(q, p), TD.kl_divergence(ref, ref_p))
    _close(q.kl(p), kl_divergence(q, p))
    assert q.entropy().shape == q.batch_shape
    assert q.event_shape == (() if family == "onehot" else ref.event_shape)
    assert q.batch_shape == ref.batch_shape


@pytest.mark.parametrize("family", list(BUILDERS))
def test_to_torch_and_sampling(family):
    q, ref = BUILDERS[family]()
    t = q.to_torch()
    _close(t.entropy(), ref.entropy())
    s = q.sample((2, 5))
    assert s.shape == (2, 5) + q.batch_shape + q.event_shape
    if family in ("onehot", "categorical"):
        with pytest.raises(NotImplementedError):
            q.rsample()
        assert type(t) is TD.Categorical
    else:
        assert q.rsample().shape == q.batch_shape + q.event_shape


def test_rsample_has_gradient():
    a = torch.tensor([2.0, 3.0], dtype=D64, requires_grad=True)
    q = Gamma(a, 1.0)
    q.rsample((10,)).sum().backward()
    assert torch.isfinite(a.grad).all() and (a.grad != 0).all()


def test_categorical_inputs_from_dtype():
    q, ref_oh = _onehot()
    ref = TD.Categorical(logits=q.logits)
    labels = torch.tensor([0, 2, 1, 1])
    x = torch.nn.functional.one_hot(labels, 3)
    assert q.event_shape == ()
    _close(q.log_prob(labels), ref.log_prob(labels))
    _close(q.log_prob(x.double()), ref_oh.log_prob(x.double()))
    _close(q.log_prob(x.bool()), ref.log_prob(labels))
    # soft assignments: expected log-likelihood sum_k r_k log p_k
    r = torch.softmax(torch.randn(6, 4, 3, dtype=D64), -1)
    _close(q.log_prob(r), (r * q.logits).sum(-1))
    # int64 one-hot vectors are labels: wrong range, clear error when
    # validating (the check synchronises, so it is opt-in)
    bad = torch.tensor([0, 3, 1, 1])
    qv = Categorical(q.probs, validate=True)
    with pytest.raises(ValueError, match=r"labels must be in \[0, 3\)"):
        qv.log_prob(bad)
    with pytest.raises(ValueError, match=r"labels must be in \[0, 3\)"):
        qv.expand((2, 4))[1].log_prob(bad)  # the flag is propagated
    with pytest.raises(RuntimeError, match="out of bounds"):
        q.log_prob(bad)  # not validated: torch's own error (CPU)
    with pytest.raises(ValueError, match="must end with K = 3"):
        q.log_prob(torch.ones(4))
    with pytest.raises(ValueError, match="not broadcastable"):
        q.log_prob(torch.zeros(5, dtype=torch.long))
    # samples: labels by default, float one-hot on request
    s = q.sample((7,))
    assert s.shape == (7, 4) and s.dtype == torch.int64
    h = q.sample((7,), one_hot=True)
    assert h.shape == (7, 4, 3) and h.dtype == D64
    assert (h.sum(-1) == 1).all()
    # expectations, entropy and KL do not depend on the representation
    _close(q.mean, q.probs)
    _close(q.expected_stats().x, q.probs)
    _close(q.entropy(), ref_oh.entropy())


# -- parameterisations ----------------------------------------------------


def test_gamma_forms_round_trip():
    a, b = _rand(5, seed=1), _rand(5, seed=2)
    forms = [
        Gamma(a, b),
        Gamma(a, scale=1 / b),
        Gamma(a, mean=a / b),
        Gamma.from_mean_shape(a / b, a),
    ]
    assert [q.form for q in forms] == [
        ("shape", "rate"),
        ("shape", "scale"),
        ("mean", "shape"),
        ("mean", "shape"),
    ]
    for q in forms:
        _close(q.shape, a)
        _close(q.rate, b)
        _close(q.scale, 1 / b)
        _close(q.mean, a / b)
        _close(q.variance, a / b**2)
        _close(q.mean_log, torch.digamma(a) - b.log())
        _close(q.entropy(), forms[0].entropy())
        _close(kl_divergence(q, forms[0]), torch.zeros(5, dtype=D64))


def test_stored_tensors_are_the_given_ones():
    a, mu = _rand(5, seed=1), _rand(5, seed=2)
    q = Gamma(a, mean=mu)
    assert q.stored()[0] is mu and q.stored()[1] is a
    v = _rand(5, seed=3)
    n = Normal(0.0, v)
    assert n.form == ("mean", "var") and n.stored()[1] is v


def test_normal_forms_round_trip():
    m, v = _rand(6, seed=1) - 2, _rand(6, seed=2)
    forms = [
        Normal(m, v),
        Normal(m, var=v),
        Normal(m, std=v.sqrt()),
        Normal(m, precision=1 / v),
    ]
    for q in forms:
        _close(q.var, v)
        _close(q.variance, v)
        _close(q.std, v.sqrt())
        _close(q.precision, 1 / v)
        _close(q.mean_sq, m**2 + v)
        _close(q.log_prob(m + 1), forms[0].log_prob(m + 1))
        _close(q.entropy(), forms[0].entropy())
    assert forms[0].form == ("mean", "var")
    assert forms[3].form == ("mean", "precision")


def test_categorical_forms_round_trip():
    p = torch.tensor([[1.0, 3.0, 0.0], [2.0, 2.0, 4.0]], dtype=D64)
    a = Categorical(p)
    b = Categorical(logits=p.log())
    for q in (a, b):
        _close(q.probs, p / p.sum(-1, keepdim=True))
        _close(q.logits.exp(), q.probs)
        _close(q.entropy(), a.to_torch().entropy())
    assert a.stored()[0] is p
    assert a.num_categories == 3


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda: Normal(0.0), "exactly one of var, std or precision"),
        (lambda: Normal(0.0, 1.0, std=1.0), "exactly one"),
        (lambda: Gamma(1.0), r"give \(shape, rate\) or"),
        (lambda: Gamma(1.0, 1.0, mean=1.0), "got mean, rate, shape"),
        (lambda: Categorical(), "exactly one of probs or logits"),
        (
            lambda: Categorical(torch.ones(2), logits=torch.ones(2)),
            "exactly one",
        ),
    ],
)
def test_parameterisation_errors(call, match):
    with pytest.raises(ValueError, match=match):
        call()


# -- exponential-family identities ---------------------------------------


@pytest.mark.parametrize("family", ["gamma", "dirichlet", "normal", "onehot"])
def test_exponential_family_identities(family):
    q, _ = BUILDERS[family](seed=0)
    p, _ = BUILDERS[family](seed=10)
    e = 0 if family in ("gamma", "normal") else 1
    log_h = -0.5 * torch.log(torch.tensor(2 * torch.pi, dtype=D64))
    elog_h = log_h if family == "normal" else 0.0

    def inner(eta, stats):
        return sum(
            (a * b).sum(-1) if e else a * b
            for a, b in zip(eta, stats, strict=True)
        )

    eta_q, eta_p = q.natural_params(), p.natural_params()
    t_q = q.expected_stats()
    _close(q.entropy(), q.log_normalizer() - inner(eta_q, t_q) - elog_h)
    kl = (
        inner([a - b for a, b in zip(eta_q, eta_p, strict=True)], t_q)
        - q.log_normalizer()
        + p.log_normalizer()
    )
    _close(kl_divergence(q, p), kl)
    # round trip and dA / d eta = E[T]
    back = type(q).from_natural(*eta_q)
    _close(back.entropy(), q.entropy())
    eta = [t.detach().clone().requires_grad_(True) for t in eta_q]
    grads = torch.autograd.grad(
        type(q).from_natural(*eta).log_normalizer().sum(), eta
    )
    for g, t in zip(grads, t_q, strict=True):
        _close(g, t)


def test_categorical_natural_params_from_probs():
    p = torch.tensor([1.0, 3.0, 0.0], dtype=D64)
    q = Categorical(p)
    (eta,) = q.natural_params()
    _close(q.log_normalizer(), torch.tensor(4.0, dtype=D64).log())
    back = Categorical.from_natural(eta)
    _close(back.probs, q.probs)


# -- Monte Carlo ----------------------------------------------------------


def _mc(samples, expected):
    mean = samples.mean(0)
    se = samples.std(0) / samples.shape[0] ** 0.5
    assert ((mean - expected).abs() <= 5 * se + 1e-12).all()


def test_monte_carlo_expectations():
    torch.manual_seed(0)
    n = 200_000
    a, b = torch.tensor([2.5, 4.0], dtype=D64), torch.tensor([0.5, 3.0])
    q = Gamma(a, b.double())
    x = TD.Gamma(a, b.double()).sample((n,))
    _mc(x.log(), q.mean_log)
    _mc(1 / x, q.mean_inv)
    _mc(-q.log_prob(x), q.entropy())
    alpha = torch.tensor([[0.5, 2.0, 3.0]], dtype=D64)
    d = Dirichlet(alpha)
    pi = TD.Dirichlet(alpha).sample((n,))
    _mc(pi.log(), d.mean_log)
    _mc(-d.log_prob(pi), d.entropy())
    g = Normal(
        torch.tensor([1.0, -2.0], dtype=D64), var=torch.tensor([0.5, 2])
    )
    y = g.sample((n,))
    _mc(y**2, g.mean_sq)
    _mc(-g.log_prob(y), g.entropy())


# -- broadcasting ---------------------------------------------------------


def test_batch_shape_broadcast():
    q = Gamma(_rand(3), _rand(4, 1))
    assert q.batch_shape == (4, 3)
    assert q.shape.shape == (4, 3) and q.rate.shape == (4, 3)
    assert q.entropy().shape == (4, 3)


@pytest.mark.parametrize("pshape", [(), (4,), (1, 4), (5, 1), (3, 5, 4)])
@pytest.mark.parametrize("family", list(BUILDERS))
def test_kl_broadcasting_matches_expanded(family, pshape):
    event = (3,) if family in ("dirichlet", "onehot", "categorical") else ()
    q, _ = BUILDERS[family]((5, 4) + event, seed=0)
    p, _ = BUILDERS[family](pshape + event, seed=10)
    batch = torch.broadcast_shapes(q.batch_shape, p.batch_shape)
    out = kl_divergence(q, p)
    assert out.shape == batch
    qe = type(q)._new(
        *(t.expand(batch + event).contiguous() for t in q.stored()),
        form=q.form,
    )
    pe = type(p)._new(
        *(t.expand(batch + event).contiguous() for t in p.stored()),
        form=p.form,
    )
    _close(out, kl_divergence(qe, pe))


def test_dirichlet_kl_shared_prior():
    q = Dirichlet(_rand(2, 7, 4))
    p = Dirichlet(_rand(4, seed=1))
    assert kl_divergence(q, p).shape == (2, 7)


def test_kl_errors():
    with pytest.raises(ValueError, match="event shapes differ"):
        kl_divergence(Dirichlet(torch.ones(4)), Dirichlet(torch.ones(3)))
    with pytest.raises(ValueError, match="not broadcastable: q"):
        kl_divergence(Gamma(_rand(8, 5), 1.0), Gamma(_rand(4), 1.0))
    with pytest.raises(NotImplementedError, match=r"KL\(Gamma \|\| Normal\)"):
        kl_divergence(Gamma(1.0, 1.0), Normal(0.0, 1.0))
    with pytest.raises(ValueError, match="event shapes differ"):
        kl_divergence(Categorical(torch.ones(3)), Categorical(torch.ones(4)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_kl_device_mismatch():
    with pytest.raises(ValueError, match="q is on"):
        kl_divergence(Gamma(1.0, 1.0, device="cuda"), Gamma(1.0, 1.0))


def test_shape_errors():
    with pytest.raises(ValueError, match="not broadcastable: shape"):
        Gamma(torch.ones(3), torch.ones(4))
    with pytest.raises(ValueError, match="needs at least 1 dim"):
        Dirichlet(torch.tensor(1.0))
    with pytest.raises(ValueError, match="does not end with the event"):
        Dirichlet(torch.ones(3)).log_prob(torch.ones(4) / 4)


def test_log_prob_sample_dims():
    q, ref = _dirichlet((4, 3))
    x = ref.sample((2, 6))
    assert q.log_prob(x).shape == (2, 6, 4)
    g = Gamma(_rand(4), 1.0)
    assert g.log_prob(torch.ones(7, 1, 4)).shape == (7, 1, 4)
    assert g.log_prob(torch.tensor(1.0)).shape == (4,)
    c, _ = _categorical((4, 3))
    assert c.log_prob(torch.tensor([[1], [2]])).shape == (2, 4)


def test_expand_and_getitem():
    q, _ = _dirichlet((4, 3))
    big = q.expand((2, 4))
    assert big.batch_shape == (2, 4) and big.event_shape == (3,)
    assert big.stored()[0] is q.stored()[0]
    with pytest.raises(ValueError, match="cannot expand to"):
        q.expand((3,))
    g = Gamma(_rand(3), _rand(4, 1, seed=1))
    for index in [
        1,
        slice(1, 3),
        (None, 0),
        (Ellipsis, 1),
        (0, Ellipsis),
        (slice(None), None, 2),
    ]:
        sub = g[index]
        _close(sub.mean, g.mean[index])
        _close(sub.entropy(), g.entropy()[index])
    for index in [1, (Ellipsis, 2), (None, slice(0, 2))]:
        sub = big[index]
        _close(
            sub.mean,
            big.mean.expand(2, 4, 3)[
                index + (slice(None),)
                if isinstance(index, tuple) and Ellipsis in index
                else index
            ],
        )
        assert sub.event_shape == (3,)
    with pytest.raises(IndexError):
        g[0, 0, 0]
    with pytest.raises(TypeError):
        g[torch.tensor([0, 1])]


# -- no materialisation ---------------------------------------------------


def test_shared_parameters_not_materialised():
    a = _rand(3)
    b = _rand(64, 64, 3, seed=1)
    q = Gamma(a, b)
    assert q.params()[0].shape == (3,)
    assert q.shape.shape == (64, 64, 3)
    assert q.shape.data_ptr() == a.data_ptr()
    assert q.params()[1] is b


def test_kl_runs_kernels_on_own_shapes(monkeypatch):
    seen = []
    lgamma, digamma = torch.lgamma, torch.digamma

    def rec(fn):
        def wrapper(x, *args, **kwargs):
            seen.append(tuple(x.shape))
            return fn(x, *args, **kwargs)

        return wrapper

    q = Gamma(_rand(64, 64, 3), _rand(64, 64, 3, seed=1))
    p = Gamma(_rand(3, seed=2), _rand(3, seed=3))
    monkeypatch.setattr(torch, "lgamma", rec(lgamma))
    monkeypatch.setattr(torch, "digamma", rec(digamma))
    kl_divergence(q, p)
    assert (3,) in seen  # lgamma(p.shape)
    assert seen.count((64, 64, 3)) == 2  # digamma(q) and lgamma(q)


def test_construction_does_not_copy():
    a, b = _rand(3), _rand(3, seed=1)
    q = Gamma(a, b)
    assert q.params()[0] is a and q.params()[1] is b
    alpha = _rand(2, 3)
    assert Dirichlet(alpha).params()[0] is alpha


# -- caching --------------------------------------------------------------


def test_cache_opt_in(monkeypatch):
    v = _rand(5)
    q = Normal(0.0, v)
    assert not q.cache
    assert q.precision.data_ptr() != q.precision.data_ptr()
    q.cache = True
    first = q.precision
    assert first.data_ptr() == q.precision.data_ptr()
    q.cache = False
    assert not q.cache
    r = Normal(0.0, v, cache=True)
    assert r.cache and r._get("precision") is r._get("precision")
    # derived objects keep the flag but not the contents
    assert r.expand((2, 5)).cache
    # bypassed while compiling
    from optix.distributions import _base

    monkeypatch.setattr(_base, "_is_compiling", lambda: True)
    s = Normal(0.0, v, cache=True)
    assert s._get("precision") is not s._get("precision")


# -- gradients ------------------------------------------------------------


@pytest.mark.parametrize("family", ["gamma", "dirichlet", "normal", "onehot"])
def test_kl_gradient_zero_at_equality(family):
    q, _ = BUILDERS[family]()
    raw = [t.clone().requires_grad_(True) for t in q.stored()]
    q2 = type(q)._new(*raw, form=q.form)
    grads = torch.autograd.grad(kl_divergence(q2, q).sum(), raw)
    for g in grads:
        _close(g, torch.zeros_like(g), atol=1e-10, rtol=0)


def test_categorical_zero_probs_finite_gradients():
    p = torch.tensor([[0.0, 0.3, 0.7], [0.5, 0.5, 0.0]], dtype=D64)
    p.requires_grad_(True)
    q = Categorical(p)
    r = Categorical(logits=torch.zeros(3, dtype=D64))
    value = q.entropy().sum() + kl_divergence(q, r).sum()
    (g,) = torch.autograd.grad(value, p)
    assert torch.isfinite(value) and torch.isfinite(g).all()


# -- dtype / device / configuration --------------------------------------


def test_dtype_device_follow_tensors(device, dtype):
    a = torch.ones(3, device=device, dtype=dtype)
    q = Gamma(a, 2.0)
    assert q.dtype == dtype and q.device.type == device.type
    assert q.rate.dtype == dtype and q.rate.device.type == device.type
    p = Gamma(a, 3.0)
    assert kl_divergence(q, p).device.type == device.type


def test_numbers_follow_config():
    with optix.config(dtype=torch.float64):
        assert Gamma(1.0, 2.0).dtype == torch.float64
    with optix.config(dtype=torch.float32):
        assert Normal(0.0, 1.0).dtype == torch.float32


def test_mixed_dtypes_promote():
    q = Gamma(torch.ones(3, dtype=torch.float32), torch.ones(3, dtype=D64))
    assert q.dtype == D64
    assert Gamma(1.0, 1.0, dtype=torch.float16).dtype == torch.float16


def test_to_returns_new_object():
    q = Gamma(torch.ones(3, dtype=D64), 2.0)
    r = q.to(dtype=torch.float32)
    assert r is not q and r.dtype == torch.float32 and q.dtype == D64
    assert q.detach().form == q.form


def test_validate():
    Gamma(1.0, -1.0)  # off by default
    with pytest.raises(ValueError, match="rate must be positive"):
        Gamma(1.0, -1.0, validate=True)
    with pytest.raises(ValueError, match="scale must be positive"):
        Gamma(1.0, scale=torch.tensor([1.0, 0.0])).validate()
    with pytest.raises(ValueError, match="probs must be non-negative"):
        Categorical(torch.tensor([1.0, -1.0])).validate()
    with pytest.raises(ValueError, match="mean must be finite"):
        Normal(torch.tensor(torch.nan), 1.0).validate()
    q = Normal(0.0, 1.0)
    assert q.validate() is q


def test_repr_and_base():
    q = Gamma(torch.ones(2, 3, dtype=D64), 1.0)
    assert repr(q).startswith("Gamma(shape, rate; batch_shape=(2, 3)")
    with pytest.raises(NotImplementedError):
        Distribution.entropy(Gamma(1.0, 1.0))
    assert issubclass(Gamma, Distribution)


# -- compile --------------------------------------------------------------


@maybe_compile
def _entropy_of(q):
    return q.entropy() + q.mean_log


@maybe_compile
def _vb_kernel(x, shape, rate, m, lam, alpha):
    elog = functional.gamma_mean_log(shape, rate)
    ll = functional.normal_log_prob(x, m, lam)
    return ll + elog + functional.dirichlet_mean_log(alpha).sum(-1)


@pytest.mark.skipif(
    not _compile._compile_supported(), reason="torch.compile unsupported"
)
def test_compile_with_objects_and_params():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="torch.compile failed", category=RuntimeWarning
        )
        with optix.config(compile=True):
            for seed in (0, 1):
                q = Gamma(
                    _rand(4, seed=seed),
                    mean=_rand(4, seed=seed + 1),
                    cache=True,
                )
                with optix.config(compile=False):
                    ref = _entropy_of(q)
                _close(_entropy_of(q), ref)
            g, n, d = (
                Gamma(_rand(4), 1.0),
                Normal(0.0, _rand(4)),
                Dirichlet(_rand(4, 3)),
            )
            x = _rand(4, seed=5)
            args = (x, *g.params(), *n.params(), *d.params())
            out = _vb_kernel(*args)
    with optix.config(compile=False):
        _close(out, _vb_kernel(*args))


@maybe_compile
def _properties_of(q):
    moments = (q.mean + q.variance).reshape(*q.batch_shape, -1).sum(-1)
    return moments + q.entropy()


@maybe_compile
def _stats_fields(s):
    return s.n + s.mean + s.sum_x + s.sum_x_sq


@pytest.mark.skipif(
    not _compile._compile_supported(), reason="torch.compile unsupported"
)
def test_compile_reads_properties_of_fresh_objects():
    # Regression (torch 2.3): dynamo evaluated the properties of slotted
    # objects eagerly and guarded on the short-lived results, which failed
    # with InternalTorchDynamoError ("<weakref ... to 'Tensor'>").
    from optix.distributions import NormalStats

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="torch.compile failed", category=RuntimeWarning
        )
        with optix.config(compile=True):
            for seed in (0, 1):
                objs = [
                    Gamma(_rand(4, seed=seed), mean=_rand(4, seed=seed + 1)),
                    Normal(_rand(4, seed=seed), std=_rand(4), cache=True),
                    Categorical(_rand(4, 3, seed=seed)),
                ]
                for q in objs:
                    out = _properties_of(q)
                    # traced code never fills (or reads) the cache
                    assert not q._cache
                    with optix.config(compile=False):
                        _close(out, _properties_of(q))
                s = NormalStats.from_samples(_rand(5, 4, seed=seed), dim=0)
                out = _stats_fields(s)
                with optix.config(compile=False):
                    _close(out, _stats_fields(s))


# -- cache invalidation (regressions) -------------------------------------


def test_cache_invalidated_by_inplace_update():
    s = _rand(5)
    q = Normal(0.0, std=s, cache=True)
    _close(q.precision, 1 / s.square())
    s.mul_(2)  # the stored tensor is s itself (same dtype and device)
    _close(q.precision, 1 / s.square())
    # views share the version counter of their base
    r = Normal(0.0, std=_rand(3, 5)[1], cache=True)
    first = r.precision.clone()
    r.stored()[1].add_(1)
    assert not torch.equal(r.precision, first)


def test_cache_cleared_by_assignment():
    q = Normal(0.0, var=_rand(5), cache=True)
    q.precision  # noqa: B018 (fills the cache)
    assert q._cache
    new = _rand(5, seed=3)
    q._raw = (q.stored()[0], new)  # the path every replacement takes
    assert q.cache and not q._cache
    _close(q.precision, 1 / new)


def test_cache_bypassed_with_gradients():
    v = _rand(5).requires_grad_(True)
    q = Normal(0.0, var=v, cache=True)
    for _ in range(2):  # a cached graph would fail the second backward
        (g,) = torch.autograd.grad(q.precision.sum(), v)
        _close(g, -1 / v.detach().square())
    assert not q._cache


def test_cache_bypassed_while_compiling(monkeypatch):
    from optix.distributions import _base

    q = Normal(0.0, var=_rand(5), cache=True)
    monkeypatch.setattr(_base, "_is_compiling", lambda: True)
    q.precision  # noqa: B018
    assert q._cache == {}


def test_cache_inference_tensors():
    with torch.inference_mode():
        q = Normal(0.0, var=_rand(5), cache=True)
        _close(q.precision, 1 / q.var)
    assert not q._cache  # no version counter: not cached


# -- review regressions ---------------------------------------------------


def test_categorical_soft_log_prob_finite_gradient():
    p = torch.tensor([[0.0, 0.4, 0.6]], dtype=D64, requires_grad=True)
    r = torch.tensor([[0.0, 0.5, 0.5]], dtype=D64, requires_grad=True)
    value = Categorical(p).log_prob(r).sum()
    gp, gr = torch.autograd.grad(value, (p, r))
    assert torch.isfinite(value)
    assert torch.isfinite(gp).all() and torch.isfinite(gr).all()
    assert gr[0, 0] == 0  # r = 0 where p = 0: log p is masked
    _close(gr[0, 1:], p.detach()[0, 1:].log())


def test_categorical_label_check_needs_validate():
    q = Categorical(_rand(4, 3))
    good = torch.tensor([0, 2, 1, 1])
    # no check (it would synchronise) by default
    assert not q._validate
    _close(
        q.log_prob(good), Categorical(q.probs, validate=True).log_prob(good)
    )
    qv = Categorical(q.probs, validate=True)
    for derived in (qv.detach(), qv.to(dtype=torch.float32), qv[1:]):
        assert derived._validate


def test_indexing_collapses_broadcast_dims():
    q = Gamma(torch.tensor(2.0, dtype=D64), mean=torch.tensor(3.0, dtype=D64))
    big = q.expand((50, 60))
    for sub in (big[1:3], big[..., 5:], big[:, None, 2:4], big[7]):
        assert all(t.numel() == 1 for t in sub.stored())
        assert sub._get("rate").numel() == 1
        assert sub.rate.shape == sub.batch_shape
    sub = big[1:3]
    assert sub.batch_shape == (2, 60)
    _close(sub.rate, torch.full((2, 60), 2 / 3, dtype=D64))
    # genuinely varying dims are kept
    r = Gamma(_rand(4, 1), mean=_rand(1, 5)).expand((3, 4, 5))[1:]
    shapes = dict(zip(r.form, (t.shape for t in r.stored()), strict=True))
    assert shapes == {"shape": (1, 4, 1), "mean": (1, 1, 5)}


def test_new_promotes_dtypes():
    q = Gamma._new(torch.ones(3), torch.ones(3, dtype=D64))
    assert all(t.dtype == D64 for t in q.stored()) and q.dtype == D64
    labels = torch.ones(3, dtype=torch.long)  # non-floating are left alone
    assert Gamma._new(labels.double(), torch.ones(3)).dtype == D64


def test_kl_categorical_k_mismatch_message():
    q, p = Categorical(_rand(4, 3)), Categorical(_rand(4, 5))
    with pytest.raises(ValueError, match="K = 3 categories.*K = 5"):
        kl_divergence(q, p)


def test_derived_parameter_has_broadcast_shape_of_inputs():
    q = Gamma(torch.tensor(2.0, dtype=D64), mean=_rand(4, 5), cache=True)
    assert q._get("rate").shape == (4, 5)
    q = Gamma(torch.tensor(2.0, dtype=D64), mean=_rand(5), cache=True)
    assert q.expand((4, 5))._get("rate").shape == (5,)


# -- float32 numerics at large parameters (regressions) -------------------


def _lg(x):
    return torch.lgamma(x)


def _dg(x):
    return torch.digamma(x)


_NAIVE = {
    "gamma_entropy": lambda a, b: a - b.log() + _lg(a) + (1 - a) * _dg(a),
    "gamma_mean_log": lambda a, b: _dg(a) - b.log(),
    "gamma_kl": lambda aq, bq, ap, bp: (
        (aq - ap) * _dg(aq)
        - _lg(aq)
        + _lg(ap)
        + ap * (bq.log() - bp.log())
        + aq * (bp - bq) / bq
    ),
    "gamma_log_prob": lambda x, a, b: (
        a * b.log() - _lg(a) + torch.xlogy(a - 1, x) - b * x
    ),
    "normal_kl": lambda mq, lq, mp, lp: (
        0.5 * (lq.log() - lp.log() + lp / lq + lp * (mq - mp).square() - 1)
    ),
}


def _ref64(name, *args):
    """float64 reference on the float32-rounded inputs."""
    return _NAIVE[name](*(a.double() for a in args))


@pytest.mark.parametrize("nu", [1e2, 1e4, 1e6])
def test_gamma_float32_large_shape(nu):
    f = torch.float32
    a = torch.tensor([nu, 2 * nu], dtype=f)
    b = a / torch.tensor([2.0, 0.5])
    a2 = (a.double() * 1.001).float()
    b2 = (b.double() * 0.999).float()
    x = (a / b).double() * torch.tensor([0.999, 1.002], dtype=D64)
    x = x.float()
    tol = {"rtol": 1e-5, "atol": 1e-5}
    for name, args in [
        ("gamma_entropy", (a, b)),
        ("gamma_mean_log", (a, b)),
        ("gamma_kl", (a, b, a2, b2)),
        ("gamma_log_prob", (x, a, b)),
    ]:
        out = getattr(functional, name)(*args)
        assert out.dtype == f
        # log p(x) has condition number |a - 1 - b x| (~4e3 here) in x:
        # the rounding of b x alone costs ~eps * 4e3
        kw = {"rtol": 1e-5, "atol": 1e-3} if name == "gamma_log_prob" else tol
        _close(out.double(), _ref64(name, *args), **kw)
    # through the object API too
    q = Gamma(a, b)
    _close(q.entropy().double(), _ref64("gamma_entropy", a, b), **tol)


@pytest.mark.parametrize("scale", [1e2, 1e4, 1e6])
def test_dirichlet_float32_large_concentration(scale):
    f = torch.float32
    al = torch.tensor([[1.0, 2.0, 3.0], [0.5, 4.0, 7.0]]) * scale
    be = (al.double() * torch.tensor([1.001, 0.999, 1.0005])).float()
    x = torch.tensor([[0.17, 0.33, 0.5], [0.05, 0.35, 0.6]], dtype=f)
    a64, b64, x64 = al.double(), be.double(), x.double()

    def ent(a):
        a0 = a.sum(-1)
        k = a.shape[-1]
        return (
            _lg(a).sum(-1)
            - _lg(a0)
            + (a0 - k) * _dg(a0)
            - ((a - 1) * _dg(a)).sum(-1)
        )

    def kl(a, b):
        a0, b0 = a.sum(-1), b.sum(-1)
        el = _dg(a) - _dg(a0)[..., None]
        return (
            _lg(a0)
            - _lg(b0)
            - (_lg(a) - _lg(b)).sum(-1)
            + ((a - b) * el).sum(-1)
        )

    def lp(x, a):
        return _lg(a.sum(-1)) - _lg(a).sum(-1) + torch.xlogy(a - 1, x).sum(-1)

    elog = _dg(a64) - _dg(a64.sum(-1, keepdim=True))
    tol = {"rtol": 1e-5, "atol": 1e-5}
    _close(functional.dirichlet_entropy(al).double(), ent(a64), **tol)
    _close(functional.dirichlet_kl(al, be).double(), kl(a64, b64), **tol)
    _close(functional.dirichlet_mean_log(al).double(), elog, **tol)
    # the rounding of the mean m = alpha / a0 costs ~eps * a0 (0.1 at
    # a0 ~ 1e7); the naive formula is off by ~2 there
    _close(
        functional.dirichlet_log_prob(x, al).double(),
        lp(x64, a64),
        rtol=1e-5,
        atol=max(1e-3, 1e-7 * scale),
    )
    _close(
        functional.dirichlet_log_normalizer(al).double(),
        _lg(a64).sum(-1) - _lg(a64.sum(-1)),
        rtol=1e-6,
        atol=1e-5,
    )


def test_normal_kl_float32_nearly_equal():
    f = torch.float32
    lq = torch.tensor([1e4, 1.0, 1e-4], dtype=f)
    lp = (lq.double() * 1.0003).float()
    mq = torch.tensor([0.0, 1.0, 2.0], dtype=f)
    mp = mq + torch.tensor([1e-4, 1e-3, 1.0], dtype=f)
    out = functional.normal_kl(mq, lq, mp, lp)
    _close(
        out.double(), _ref64("normal_kl", mq, lq, mp, lp), rtol=1e-5, atol=1e-9
    )
