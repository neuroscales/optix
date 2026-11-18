import asyncio
import importlib
import threading

import pytest
import torch

import optix
from optix import _config
from optix._config import Config, parse_bytes, resolve_backend


@pytest.fixture(autouse=True)
def _restore_global():
    saved = _config._global
    yield
    _config._global = saved


def test_defaults():
    cfg = Config()
    assert cfg.dtype is None
    assert cfg.device is None
    assert cfg.memory == 256 * 2**20
    assert cfg.batch is None
    assert cfg.compile is True
    assert cfg.prefetch == 0


def test_set_get_round_trip():
    optix.set_config(dtype="float64", memory="1GiB", batch=4, prefetch=2)
    cfg = optix.get_config()
    assert cfg.dtype is torch.float64
    assert cfg.memory == 2**30
    assert cfg.batch == 4
    assert cfg.prefetch == 2


def test_setting_changes_resolve_backend():
    # regression: setters and readers must share one storage location
    optix.set_config(dtype=torch.float64, device="cpu")
    assert resolve_backend() == {
        "dtype": torch.float64,
        "device": torch.device("cpu"),
    }
    with optix.config(dtype="float16"):
        assert resolve_backend()["dtype"] is torch.float16


def test_nesting_and_exception_restore():
    base = optix.get_config()
    with optix.config(batch=3) as outer:
        assert outer.batch == 3
        with optix.config(memory="1MB"):
            assert optix.get_config().batch == 3
            assert optix.get_config().memory == 10**6
        assert optix.get_config().memory == base.memory
        with pytest.raises(KeyError):
            with optix.config(batch=5):
                raise KeyError
        assert optix.get_config().batch == 3
    assert optix.get_config() == base


def test_reusable_instance():
    ctx = optix.config(batch=7)
    with ctx:
        with ctx:
            assert optix.get_config().batch == 7
        assert optix.get_config().batch == 7
    assert optix.get_config().batch is None


def test_decorator_form():
    @optix.config(batch=2)
    def inner():
        return optix.get_config().batch

    assert inner() == 2
    assert optix.get_config().batch is None


def test_thread_isolation():
    seen = {}

    def worker():
        seen["batch"] = optix.get_config().batch

    with optix.config(batch=9):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert seen["batch"] is None


def test_asyncio_tasks_inherit():
    async def read():
        return optix.get_config().batch

    async def main():
        with optix.config(batch=5):
            return await asyncio.create_task(read())

    assert asyncio.run(main()) == 5


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("dtype", "float64", torch.float64),
        ("dtype", "torch.float32", torch.float32),
        ("device", "cpu", torch.device("cpu")),
        ("memory", "512MiB", 512 * 2**20),
        ("memory", 1e9, 10**9),
        ("batch", "3", 3),
        ("compile", "false", False),
        ("compile", 1, True),
    ],
)
def test_normalisation(key, value, expected):
    optix.set_config(**{key: value})
    assert getattr(optix.get_config(), key) == expected


@pytest.mark.parametrize(
    "key,value",
    [
        ("dtype", "int32"),
        ("dtype", torch.int64),
        ("dtype", "nonsense"),
        ("device", "nonsense"),
        ("memory", -1),
        ("memory", "12 parsecs"),
        ("batch", 0),
        ("batch", 1.5),
        ("prefetch", -1),
        ("compile", "maybe"),
    ],
)
def test_invalid_values(key, value):
    with pytest.raises(ValueError):
        optix.set_config(**{key: value})
    with pytest.raises(ValueError):
        optix.config(**{key: value})


def test_unknown_key():
    with pytest.raises(TypeError):
        optix.set_config(colour="blue")
    with pytest.raises(TypeError):
        optix.config(colour="blue")


def test_parse_bytes():
    assert parse_bytes(10) == 10
    assert parse_bytes("2GB") == 2 * 10**9
    assert parse_bytes("1.5 kib") == 1536
    assert parse_bytes("100") == 100
    with pytest.raises(ValueError):
        parse_bytes(True)


def test_env_seeding(monkeypatch):
    monkeypatch.setenv("OPTIX_DTYPE", "float64")
    monkeypatch.setenv("OPTIX_DEVICE", "cpu")
    monkeypatch.setenv("OPTIX_MEMORY", "1MiB")
    monkeypatch.setenv("OPTIX_BATCH", "4")
    monkeypatch.setenv("OPTIX_COMPILE", "0")
    try:
        module = importlib.reload(_config)
        cfg = module.get_config()
        assert cfg.dtype is torch.float64
        assert cfg.device == torch.device("cpu")
        assert cfg.memory == 2**20
        assert cfg.batch == 4
        assert cfg.compile is False
        monkeypatch.setenv("OPTIX_BATCH", "zero")
        with pytest.warns(RuntimeWarning, match="OPTIX_BATCH"):
            module = importlib.reload(_config)
        assert module.get_config().batch is None
    finally:
        monkeypatch.delenv("OPTIX_DTYPE")
        monkeypatch.delenv("OPTIX_DEVICE")
        monkeypatch.delenv("OPTIX_MEMORY")
        monkeypatch.delenv("OPTIX_BATCH")
        monkeypatch.delenv("OPTIX_COMPILE")
        importlib.reload(_config)
        importlib.reload(optix)


def test_resolve_backend_precedence():
    cpu = torch.device("cpu")
    meta = torch.device("meta")
    like = torch.empty(0, device=meta)
    default = optix.get_config().dtype or torch.get_default_dtype()
    # torch default
    assert resolve_backend() == {"dtype": default, "device": cpu}
    # like (device only, never dtype)
    like64 = like.double()
    assert resolve_backend(like=like64) == {"dtype": default, "device": meta}
    # config beats like
    with optix.config(device="cpu", dtype="float64"):
        assert resolve_backend(like=like) == {
            "dtype": torch.float64,
            "device": cpu,
        }
        # explicit beats config
        assert resolve_backend("float16", "meta") == {
            "dtype": torch.float16,
            "device": meta,
        }


def test_mps_float64_rejected():
    with pytest.raises(ValueError, match="MPS"):
        resolve_backend(torch.float64, "mps")
