"""Process-wide and context-local configuration.

The active configuration is the innermost `with optix.config(...)` block
of the current context, else the process-wide defaults set with
[`set_config`][optix.set_config] (seeded at import from `OPTIX_*` environment
variables).

Public entry points (iterator construction, `fit`, `apply`) read the
configuration once and pass explicit values down; private kernels never
call [`get_config`][optix.get_config].
"""

import contextlib
import dataclasses
import math
import os
import re
import threading
import warnings
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TypedDict

import torch

__all__ = [
    "Backend",
    "Config",
    "config",
    "get_config",
    "parse_bytes",
    "resolve_backend",
    "set_config",
]


@dataclass(frozen=True)
class Config:
    """Immutable set of options.

    Attributes
    ----------
    dtype : torch.dtype or None
        Working floating-point dtype. `None` means
        `torch.get_default_dtype()`.
    device : torch.device or None
        Working device. `None` means the device of a Tensor input, else
        the torch default device.
    memory : int
        Budget, in bytes, of one loaded chunk on the working device.
    batch : int or None
        Explicit number of frames per chunk. Overrides `memory`.
    compile : bool
        Whether kernels decorated with `maybe_compile` go through
        `torch.compile`.
    prefetch : int
        Number of chunks read ahead in a background thread (0 = off).
    """

    dtype: torch.dtype | None = None
    device: torch.device | None = None
    memory: int = 256 * 2**20
    batch: int | None = None
    compile: bool = True
    prefetch: int = 0


class Backend(TypedDict):
    """Resolved working dtype and device, usable as `x.to(**backend)`."""

    dtype: torch.dtype
    device: torch.device


_FIELDS = tuple(f.name for f in dataclasses.fields(Config))

_UNITS = {
    "": 1,
    "b": 1,
    "k": 10**3,
    "kb": 10**3,
    "m": 10**6,
    "mb": 10**6,
    "g": 10**9,
    "gb": 10**9,
    "t": 10**12,
    "tb": 10**12,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
}


def parse_bytes(value: int | float | str) -> int:
    """Convert a memory size to a number of bytes.

    Parameters
    ----------
    value : int or float or str
        A number of bytes (`1e9`) or a string with an optional unit
        (`"512MiB"`, `"2GB"`, `"1.5 GiB"`). Decimal units (kB, MB,
        GB, TB) are powers of 1000, binary units (KiB, MiB, GiB, TiB) are
        powers of 1024. Units are case-insensitive.

    Returns
    -------
    int
        Number of bytes (strictly positive).

    Raises
    ------
    ValueError
        If the value cannot be parsed or is not positive.
    """
    if isinstance(value, bool):
        raise ValueError(f"Invalid memory size: {value!r}")
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([0-9.eE+]+)\s*([a-zA-Z]*)\s*", value)
        unit = match and match.group(2).lower()
        if not match or unit not in _UNITS:
            raise ValueError(f"Invalid memory size: {value!r}")
        try:
            number = float(match.group(1))
        except ValueError:
            raise ValueError(f"Invalid memory size: {value!r}") from None
        value = number * _UNITS[unit]
    if not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"Invalid memory size: {value!r}")
    nbytes = int(value)
    if nbytes <= 0:
        raise ValueError(f"Memory size must be positive, got {value!r}")
    return nbytes


def _parse_dtype(value: object) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, str):
        name = value.removeprefix("torch.")
        dtype = getattr(torch, name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unknown dtype: {value!r}")
        value = dtype
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Invalid dtype: {value!r}")
    if not value.is_floating_point:
        raise ValueError(f"Working dtype must be floating, got {value}")
    return value


def _parse_device(value: object) -> torch.device | None:
    if value is None:
        return None
    if isinstance(value, str | int):
        try:
            value = torch.device(value)
        except RuntimeError as e:
            raise ValueError(f"Invalid device: {value!r}") from e
    if not isinstance(value, torch.device):
        raise ValueError(f"Invalid device: {value!r}")
    return value


def _parse_count(name: str, value: object, minimum: int) -> int:
    if isinstance(value, str):
        value = value.strip()
        if not re.fullmatch(r"[+-]?[0-9]+", value):
            raise ValueError(f"Invalid {name}: {value!r}")
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid {name}: {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _parse_bool(value: object) -> bool:
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off", ""):
            return False
        raise ValueError(f"Invalid boolean: {value!r}")
    if not isinstance(value, bool | int):
        raise ValueError(f"Invalid boolean: {value!r}")
    return bool(value)


def _normalize(options: dict[str, object]) -> dict[str, object]:
    """Validate and normalise option values (raises on unknown keys)."""
    unknown = set(options) - set(_FIELDS)
    if unknown:
        raise TypeError(
            f"Unknown option(s): {sorted(unknown)}. Valid: {list(_FIELDS)}"
        )
    out: dict[str, object] = {}
    for key, value in options.items():
        if key == "dtype":
            out[key] = _parse_dtype(value)
        elif key == "device":
            out[key] = _parse_device(value)
        elif key == "memory":
            out[key] = parse_bytes(value)
        elif key == "batch":
            out[key] = None if value is None else _parse_count(key, value, 1)
        elif key == "compile":
            out[key] = _parse_bool(value)
        elif key == "prefetch":
            out[key] = _parse_count(key, value, 0)
    return out


def _from_environ() -> Config:
    """Build the process-wide defaults from `OPTIX_*` variables."""
    cfg = Config()
    for key in ("dtype", "device", "memory", "batch", "compile"):
        var = f"OPTIX_{key.upper()}"
        value = os.environ.get(var)
        if value is None or value.strip() == "":
            continue
        try:
            cfg = dataclasses.replace(cfg, **_normalize({key: value}))
        except ValueError as e:
            warnings.warn(
                f"Ignoring invalid {var}={value!r}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
    return cfg


_lock = threading.Lock()
_global: Config = _from_environ()
_local: ContextVar[Config | None] = ContextVar("optix_config", default=None)


def get_config() -> Config:
    """Return the active configuration.

    Returns
    -------
    Config
        The innermost `with config(...)` override of the current
        context, else the process-wide defaults.
    """
    local = _local.get()
    return _global if local is None else local


def set_config(**options: object) -> None:
    """Set process-wide defaults.

    Parameters
    ----------
    **options
        Any field of `Config`. Values are validated and normalised
        (`"float64"` → `torch.float64`, `"1GiB"` → bytes, ...).

    Raises
    ------
    TypeError
        If an option name is unknown.
    ValueError
        If a value is invalid.

    Notes
    -----
    An active `with config(...)` block keeps shadowing the options it
    overrides.
    """
    global _global
    values = _normalize(options)
    with _lock:
        _global = dataclasses.replace(_global, **values)


class config(contextlib.ContextDecorator):
    """Temporarily override options.

    Usable as a context manager (`with optix.config(device="cuda"):`) or
    as a decorator. Overrides apply to the current thread / asyncio task
    only: threads started inside the block see the process-wide defaults.

    Notes
    -----
    Nesting works, and an instance may be re-entered (but not used
    concurrently from several threads).
    """

    def __init__(self, **options: object) -> None:
        """Validate the overrides.

        Parameters
        ----------
        **options
            Any field of `Config`. Values are validated immediately.
        """
        self._options = _normalize(options)
        self._tokens: list[Token] = []

    def __enter__(self) -> Config:
        new = dataclasses.replace(get_config(), **self._options)
        self._tokens.append(_local.set(new))
        return new

    def __exit__(self, *exc: object) -> None:
        _local.reset(self._tokens.pop())


def _default_device() -> torch.device:
    return torch.get_default_device()


def resolve_backend(
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
    like: object = None,
) -> Backend:
    """Resolve the working dtype and device.

    Parameters
    ----------
    dtype : torch.dtype or str, optional
        Explicit dtype. Default: configuration, else
        `torch.get_default_dtype()`. Never follows `like`.
    device : torch.device or str, optional
        Explicit device. Default: configuration, else the device of
        `like` if it is a Tensor, else the torch default device.
    like : object, optional
        Input data whose device is followed when no device is set.

    Returns
    -------
    Backend
        `{"dtype": ..., "device": ...}`, usable as `x.to(**backend)`.

    Raises
    ------
    ValueError
        If float64 is requested on an MPS device, or a value is invalid.
    """
    cfg = get_config()
    dtype = _parse_dtype(dtype)
    device = _parse_device(device)
    if dtype is None:
        dtype = cfg.dtype or torch.get_default_dtype()
    if device is None:
        device = cfg.device
    if device is None:
        if isinstance(like, torch.Tensor):
            device = like.device
        else:
            device = _default_device()
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError(
            "float64 is not supported on MPS devices; use float32 or "
            "another device."
        )
    return Backend(dtype=dtype, device=device)
