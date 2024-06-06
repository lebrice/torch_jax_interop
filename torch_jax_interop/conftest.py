import jax
import jax.numpy as jnp
import pytest
import torch

from torch_jax_interop.to_torch import jax_to_torch_device


@pytest.fixture(scope="session", params=[123], ids="seed={}".format)
def seed(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(
    scope="session", params=["cpu", "cuda", "rocm", "tpu"], ids="backend={}".format
)
def jax_device(request: pytest.FixtureRequest) -> jax.Device:
    backend_str = request.param
    try:
        devices = jax.devices(backend=request.param)
    except RuntimeError:
        devices = None
    if not devices:
        pytest.skip(f"No devices found for backend {backend_str}.")
    return devices[0]


@pytest.fixture(scope="session")
def torch_device(
    request: pytest.FixtureRequest, jax_device: jax.Device
) -> torch.device:
    param = getattr(request, "param", None)
    # in case of an indirect parametrization, use the specified device:
    if param is not None:
        assert isinstance(param, str | torch.device)
        return torch.device(param) if isinstance(param, str) else param
    return jax_to_torch_device(jax_device)


@pytest.fixture(
    scope="session",
    params=[
        pytest.param((torch.float32, jnp.float32), id="float32"),
        pytest.param((torch.float64, jnp.float32), id="float64"),  # important!
        pytest.param((torch.int32, jnp.int32), id="int32"),
        pytest.param((torch.int64, jnp.int32), id="int64"),  # important!
    ],
)
def torch_jax_dtypes(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(scope="session")
def torch_dtype(torch_jax_dtypes: tuple[torch.dtype, jnp.dtype]) -> torch.dtype:
    return torch_jax_dtypes[0]


@pytest.fixture(scope="session")
def jax_dtype(torch_jax_dtypes: tuple[torch.dtype, jnp.dtype]) -> jnp.dtype:
    return torch_jax_dtypes[1]
