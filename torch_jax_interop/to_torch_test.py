import jax
import numpy
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from torch_jax_interop import jax_to_torch, torch_to_jax
import jax.numpy as jnp
import jaxlib.xla_extension


@pytest.fixture(scope="session", params=[123], ids="seed={}".format)
def seed(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(
    scope="session", params=["cpu", "cuda", "rocm", "tpu"], ids="backend={}".format
)
def device(request: pytest.FixtureRequest) -> jax.Device:
    backend_str = request.param
    try:
        devices = jax.devices(backend=request.param)
    except RuntimeError:
        devices = None
    if not devices:
        pytest.skip(f"No devices found for backend {backend_str}.")
    return devices[0]


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (10, 10),
        (10, 10, 10),
        pytest.param(
            tuple(range(1, 6)),
            marks=pytest.mark.xfail(
                raises=jaxlib.xla_extension.XlaRuntimeError,
                reason="TODO: Getting a UNIMPLEMENTED due to non-default layout?",
            ),
        ),
    ],
    ids="shape={}".format,
)
def test_jax_to_torch_tensor(
    shape: tuple[int, ...],
    device: jax.Device,
    torch_dtype: torch.dtype,
    jax_dtype: jnp.dtype,
    seed: int,
    benchmark: BenchmarkFixture,
):
    if numpy.prod(shape) >= 1_000_000 and device.platform == "cpu":
        pytest.skip("Skipping test with large tensor on CPU.")

    key = jax.random.key(seed)
    # todo: don't know what the equivalent is on a np/jax dtype for checking if the dtype is
    # floating-point.
    if torch_dtype.is_floating_point:
        jax_value = jax.random.uniform(key=key, shape=shape, dtype=jax_dtype)
    else:
        jax_value = jax.random.randint(
            key=key, shape=shape, minval=0, maxval=100, dtype=jax_dtype
        )
    jax_value = jax.device_put(jax_value, device=device)

    torch_expected_device = jax_to_torch(device)
    assert isinstance(torch_expected_device, torch.device)

    torch_value = benchmark(jax_to_torch, jax_value)
    assert isinstance(torch_value, torch.Tensor)
    assert torch_value.device == torch_expected_device

    # Convert the torch Tensor to a numpy array so we can compare the contents.
    torch_numpy_value = torch_value.cpu().numpy()
    numpy.testing.assert_allclose(jax_value, torch_numpy_value)

    # round-trip:
    jax_round_trip = torch_to_jax(torch_value)
    numpy.testing.assert_allclose(jax_round_trip, jax_value)
