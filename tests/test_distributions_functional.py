import warnings

import pytest
import torch

import optix
from optix import _compile, special
from optix._compile import maybe_compile
from optix.distributions import functional as F

D64 = torch.float64


def _rand(*shape, lo=0.1, hi=5.0, seed=0, dtype=D64):
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=dtype)


def test_aliases_are_the_special_functions():
    assert F.gamma_kl is special.kl_gamma
    assert F.dirichlet_kl is special.kl_dirichlet
    assert F.dirichlet_mean_log is special.dirichlet_elog


def test_log_minus_digamma_public():
    a = torch.logspace(-3, 6, 50, dtype=D64)
    ref = a.log() - torch.digamma(a)
    small = a <= 10
    torch.testing.assert_close(special.log_minus_digamma(a)[small], ref[small])
    # asymptotic branch: compare with the series to high order
    big = a[~small]
    torch.testing.assert_close(
        special.log_minus_digamma(big), ref[~small], rtol=1e-8, atol=0
    )


def test_normal_kl_matches_special_variance_form():
    mq, mp = _rand(10, seed=1) - 2, _rand(10, seed=2) - 2
    vq, vp = _rand(10, seed=3), _rand(10, seed=4)
    torch.testing.assert_close(
        F.normal_kl(mq, 1 / vq, mp, 1 / vp),
        special.kl_normal(mq, vq, mp, vp),
    )


def test_normal_log_prob_forms_agree():
    x, m, v = _rand(7, seed=1), _rand(7, seed=2), _rand(7, seed=3)
    torch.testing.assert_close(
        F.normal_log_prob(x, m, 1 / v), F.normal_log_prob_var(x, m, v)
    )


def test_gamma_mean_log_ms_matches_and_is_accurate():
    a, b = _rand(20, seed=1), _rand(20, seed=2)
    torch.testing.assert_close(
        F.gamma_mean_log_ms(a / b, a), F.gamma_mean_log(a, b)
    )
    # large shape in float32: no cancellation between digamma and log
    mu = torch.tensor([2.0, 0.5])
    nu = torch.tensor([1e6, 1e6])
    ref = F.gamma_mean_log_ms(mu.double(), nu.double())
    out = F.gamma_mean_log_ms(mu, nu)
    assert (out.double() - ref).abs().max() < 2e-7


def test_gamma_mean_inv_inf_and_finite_gradient():
    a = torch.tensor([0.5, 1.0, 3.0], dtype=D64, requires_grad=True)
    b = torch.tensor(2.0, dtype=D64)
    out = F.gamma_mean_inv(a, b)
    assert torch.isinf(out[:2]).all()
    assert out[2].item() == pytest.approx(1.0)
    (g,) = torch.autograd.grad(out[2], a)
    assert torch.isfinite(g).all()


def test_categorical_logits_from_probs():
    p = torch.tensor([0.0, 1.0, 3.0], dtype=D64)
    out = F.categorical_logits_from_probs(p)
    torch.testing.assert_close(out.exp(), p / 4)
    assert out[0] == -torch.inf


def test_categorical_zero_probabilities_have_finite_gradients():
    p = torch.tensor([[0.0, 0.3, 0.7], [0.5, 0.5, 0.0]], dtype=D64)
    p.requires_grad_(True)
    lq = F.categorical_logits_from_probs(p)
    lp = F.categorical_normalize(torch.randn(2, 3, dtype=D64))
    value = F.categorical_entropy(lq).sum() + F.categorical_kl(lq, lp).sum()
    (g,) = torch.autograd.grad(value, p)
    assert torch.isfinite(value) and torch.isfinite(g).all()
    # single-where version would give NaN here; check the value too
    ref = -(p[0, 1:] * p[0, 1:].log()).sum()
    torch.testing.assert_close(F.categorical_entropy(lq)[0], ref)


def test_categorical_kl_inf_when_p_zero():
    lq = F.categorical_normalize(torch.zeros(3, dtype=D64))
    lp = F.categorical_logits_from_probs(torch.tensor([0.0, 1, 1], dtype=D64))
    assert F.categorical_kl(lq, lp) == torch.inf


def test_categorical_log_prob_broadcasts():
    logits = F.categorical_normalize(torch.randn(4, 3, dtype=D64))
    index = torch.tensor([[0], [2]])  # (2, 1) against batch (4,)
    out = F.categorical_log_prob(index, logits)
    assert out.shape == (2, 4)
    torch.testing.assert_close(out[1], logits[:, 2])


_GRADCHECK = [
    (F.gamma_mean, 2, 0),
    (F.gamma_variance, 2, 0),
    (F.gamma_mean_log, 2, 0),
    (F.gamma_mean_log_ms, 2, 0),
    (F.gamma_entropy, 2, 0),
    (F.gamma_log_prob, 3, 0),
    (F.gamma_log_normalizer, 2, 0),
    (F.gamma_kl, 4, 0),
    (F.dirichlet_mean, 1, 1),
    (F.dirichlet_variance, 1, 1),
    (F.dirichlet_mean_log, 1, 1),
    (F.dirichlet_entropy, 1, 1),
    (F.dirichlet_log_normalizer, 1, 1),
    (F.dirichlet_kl, 2, 1),
    (F.categorical_normalize, 1, 1),
    (F.categorical_logits_from_probs, 1, 1),
    (F.categorical_entropy, 1, 2),
    (F.categorical_kl, 2, 2),
    (F.categorical_log_normalizer, 1, 1),
    (F.normal_variance, 1, 0),
    (F.normal_mean_sq, 2, 0),
    (F.normal_entropy, 1, 0),
    (F.normal_log_prob, 3, 0),
    (F.normal_log_prob_var, 3, 0),
    (F.normal_log_normalizer, 2, 0),
    (F.normal_kl, 4, 0),
]


@pytest.mark.parametrize(
    "fn,nargs,kind", _GRADCHECK, ids=[f.__name__ for f, *_ in _GRADCHECK]
)
def test_gradcheck(fn, nargs, kind):
    shape = (3,) if kind == 0 else (2, 3)
    args = []
    for i in range(nargs):
        x = _rand(*shape, lo=0.5, hi=3.0, seed=i)
        if kind == 2:  # normalised logits
            x = torch.log_softmax(x, -1)
        args.append(x.requires_grad_(True))
    assert torch.autograd.gradcheck(fn, args)


def test_dirichlet_log_prob_gradcheck():
    x = torch.softmax(_rand(2, 3, seed=1), -1).requires_grad_(True)
    a = _rand(2, 3, seed=2, lo=0.5).requires_grad_(True)
    assert torch.autograd.gradcheck(F.dirichlet_log_prob, (x, a))


# -- compile smoke ------------------------------------------------------


@maybe_compile
def _toy_kernel(x, a, b, m, v, alpha, beta, logits):
    elog = F.gamma_mean_log(a, b)
    ll = F.normal_log_prob_var(x, m, v)
    pi = F.dirichlet_mean_log(alpha)
    kl = F.gamma_kl(a, b, b, a) + F.dirichlet_kl(alpha, beta)
    lg = F.categorical_normalize(logits)
    h = F.categorical_entropy(lg) + F.categorical_kl(lg, lg.flip(-1))
    lms = F.gamma_mean_log_ms(a / b, a)
    return elog + ll + pi.sum(-1) + kl + h + lms + F.normal_kl(m, v, m, v)


@maybe_compile
def _toy_fit(c, elog, alpha):
    return special.fit_gamma_shape(c, 4), special.fit_dirichlet(elog, alpha, 4)


@pytest.mark.skipif(
    not _compile._compile_supported(), reason="torch.compile unsupported"
)
def test_compile_smoke():
    args = [_rand(5, seed=i) for i in range(5)]
    args += [_rand(5, 3, seed=6), _rand(5, 3, seed=7), _rand(5, 3, seed=8)]
    with optix.config(compile=False):
        ref = _toy_kernel(*args)
        ref_fit = _toy_fit(args[0], -args[5], args[6])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="torch.compile failed", category=RuntimeWarning
        )
        with optix.config(compile=True):
            out = _toy_kernel(*args)
            out_fit = _toy_fit(args[0], -args[5], args[6])
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(out_fit, ref_fit)
