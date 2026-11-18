import pytest
import torch

import optix
from optix import _compile
from optix._compile import maybe_compile


def _kernel(x: torch.Tensor) -> torch.Tensor:
    return (x * 2).exp().sum(-1)


@pytest.fixture
def calls(monkeypatch):
    """Count torch.compile invocations without compiling anything."""
    record = []

    def fake_compile(fn, **kwargs):
        record.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(_compile, "_compile_supported", lambda: True)
    return record


def test_eager_when_off(calls):
    fn = maybe_compile(_kernel)
    x = torch.randn(3, 4)
    with optix.config(compile=False):
        assert torch.equal(fn(x), _kernel(x))
    assert calls == []


def test_compile_kwargs_and_decorator_forms(calls):
    @maybe_compile(mode="reduce-overhead")
    def f(x):
        return x + 1

    with optix.config(compile=True):
        assert f(torch.ones(1)).item() == 2
    assert calls == [{"mode": "reduce-overhead"}]
    assert f.__name__ == "f"


def test_toggle_read_at_call_time(calls):
    fn = maybe_compile(_kernel)
    x = torch.randn(2)
    with optix.config(compile=False):
        fn(x)
    with optix.config(compile=True):
        fn(x)
    assert len(calls) == 1


def test_failure_falls_back_with_one_warning(monkeypatch):
    monkeypatch.setattr(_compile, "_compile_supported", lambda: True)

    def broken_compile(fn, **kwargs):
        def raiser(*args, **kw):
            raise RuntimeError("no compiler here")

        return raiser

    monkeypatch.setattr(torch, "compile", broken_compile)
    fn = maybe_compile(_kernel)
    x = torch.randn(3)
    with optix.config(compile=True):
        with pytest.warns(RuntimeWarning, match="no compiler here") as rec:
            out = fn(x)
            out2 = fn(x)
    assert len(rec) == 1
    assert torch.equal(out, _kernel(x))
    assert torch.equal(out2, _kernel(x))


def test_genuine_errors_propagate(calls):
    @maybe_compile
    def bad(x):
        raise ValueError("genuine")

    for flag in (False, True):
        with optix.config(compile=flag):
            with pytest.raises(ValueError, match="genuine"):
                bad(torch.ones(1))


def test_unsupported_is_eager(monkeypatch):
    monkeypatch.setattr(_compile, "_compile_supported", lambda: False)

    def explode(*args, **kwargs):
        raise AssertionError("torch.compile must not be called")

    monkeypatch.setattr(torch, "compile", explode)
    fn = maybe_compile(_kernel)
    with optix.config(compile=True):
        fn(torch.ones(2))


@pytest.mark.skipif(
    not _compile._compile_supported(), reason="torch.compile unsupported"
)
def test_compiled_equals_eager():
    fn = maybe_compile(_kernel)
    x = torch.randn(5, 7)
    with optix.config(compile=True):
        compiled = fn(x)
    with optix.config(compile=False):
        eager = fn(x)
    torch.testing.assert_close(compiled, eager)
