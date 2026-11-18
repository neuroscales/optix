import pytest
import torch

_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")
if torch.backends.mps.is_available():
    _DEVICES.append("mps")


@pytest.fixture(params=_DEVICES)
def device(request) -> torch.device:
    """Every available device (cpu, plus cuda / mps when present)."""
    return torch.device(request.param)


@pytest.fixture(params=[torch.float32, torch.float64], ids=["f32", "f64"])
def dtype(request, device) -> torch.dtype:
    """float32 and float64 (float64 skipped on MPS)."""
    if device.type == "mps" and request.param == torch.float64:
        pytest.skip("float64 is not supported on MPS")
    return request.param
