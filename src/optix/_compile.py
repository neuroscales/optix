"""Optional ``torch.compile`` with an eager fallback."""

import functools
import sys
import warnings
from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar, overload

import torch

from ._config import get_config

__all__ = ["maybe_compile"]

P = ParamSpec("P")
R = TypeVar("R")

_UNTRIED, _COMPILED, _BROKEN = "untried", "compiled", "broken"


def _torch_version() -> tuple[int, int]:
    major, minor = torch.__version__.split(".")[:2]
    return int(major), int("".join(c for c in minor if c.isdigit()) or 0)


def _compile_supported() -> bool:
    """Whether torch.compile can be expected to work here at all."""
    if not hasattr(torch, "compile") or sys.platform.startswith("win"):
        return False
    python = sys.version_info[:2]
    version = _torch_version()
    # minimum torch version supporting torch.compile on a Python version.
    # The Python 3.14 threshold is unverified; a wrong value only costs a
    # fallback warning (the first compile fails and the wrapper goes eager).
    minimum = {(3, 12): (2, 4), (3, 13): (2, 6), (3, 14): (2, 9)}
    for py, required in minimum.items():
        if python >= py and version < required:
            return False
    return True


class _MaybeCompiled(Generic[P, R]):
    """Callable wrapper holding an eager and a lazily compiled version."""

    def __init__(self, fn: Callable[P, R], compile_kwargs: dict) -> None:
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._kwargs = compile_kwargs
        self._compiled: Callable[P, R] | None = None
        self._state = _UNTRIED if _compile_supported() else _BROKEN

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if self._state == _BROKEN or not get_config().compile:
            return self._fn(*args, **kwargs)
        try:
            if self._compiled is None:
                self._compiled = torch.compile(self._fn, **self._kwargs)
            result = self._compiled(*args, **kwargs)
        except Exception as error:
            # Re-run eagerly; if eager also fails, it is a genuine error.
            result = self._fn(*args, **kwargs)
            self._state = _BROKEN
            self._compiled = None
            warnings.warn(
                f"torch.compile failed for {self._fn.__qualname__}; "
                f"falling back to eager mode. Original error: "
                f"{type(error).__name__}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return result
        self._state = _COMPILED
        return result


@overload
def maybe_compile(
    fn: Callable[P, R], **compile_kwargs: object
) -> Callable[P, R]: ...


@overload
def maybe_compile(
    fn: None = None, **compile_kwargs: object
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def maybe_compile(
    fn: Callable[P, R] | None = None, **compile_kwargs: object
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Run a kernel through ``torch.compile`` when ``config.compile`` is on.

    Usable as ``@maybe_compile`` or ``@maybe_compile(mode=...)``.

    Parameters
    ----------
    fn : callable, optional
        The kernel to wrap.
    **compile_kwargs
        Keyword arguments forwarded to ``torch.compile``.

    Returns
    -------
    callable
        A wrapper that reads ``get_config().compile`` at call time. When it
        is on, the compiled version is used; if compiling or running it
        raises, the call is re-run eagerly. If the eager call also raises,
        that error propagates (it is a genuine bug). Otherwise one
        ``RuntimeWarning`` is emitted and the wrapper stays eager from then
        on. Compilation is skipped silently where it is not supported
        (Windows, Python/torch combinations without ``torch.compile``).

    Notes
    -----
    Decorated kernels must be **pure functions of tensors**: no in-place
    mutation of their arguments and no Python side effects. Otherwise the
    eager re-run after a failure would give wrong results.

    Shape changes (last short batch, clipped edge blocks) trigger
    recompilations; ``dynamic`` is left to torch's default, which marks
    dims as dynamic after the first recompilation.
    """
    if fn is None:
        return functools.partial(maybe_compile, **compile_kwargs)
    return _MaybeCompiled(fn, compile_kwargs)
