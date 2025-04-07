import random

import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from tensor_regression.stats import get_simple_attributes

from torch_jax_interop import torch_to_jax
from torch_jax_interop.to_torch import jax_to_torch, jax_to_torch_device
from torch_jax_interop.utils import to_channels_last


# Add support for Jax arrays in the tensor regression fixture.
@get_simple_attributes.register(jax.Array)
def jax_array_simple_attributes(array: jnp.ndarray, precision: int | None) -> dict:
    return get_simple_attributes(jax_to_torch(array), precision=precision)


DEFAULT_SEED = 123


@pytest.fixture(autouse=True)
def seed(request: pytest.FixtureRequest):
    """Fixture that seeds everything for reproducibility and yields the random seed used."""
    random_seed = getattr(request, "param", DEFAULT_SEED)
    assert isinstance(random_seed, int) or random_seed is None

    random_state = random.getstate()
    np_random_state = np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        yield random_seed

    random.setstate(random_state)
    np.random.set_state(np_random_state)


@pytest.fixture(scope="session", params=["cpu", "cuda", "rocm", "tpu"], ids="backend={}".format)
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
def torch_device(request: pytest.FixtureRequest, jax_device: jax.Device) -> torch.device:
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


@pytest.fixture
def jax_input(torch_input: torch.Tensor):
    return torch_to_jax(torch_input)


@pytest.fixture
def torch_input(torch_device: torch.device, seed: int):
    input_shape: tuple[int, ...] = (1, 3, 32, 32)
    torch_input = torch.randn(
        input_shape,
        generator=torch.Generator(device=torch_device).manual_seed(seed),
        device=torch_device,
    )
    return torch_input


class JaxCNN(flax.linen.Module):
    """A simple CNN model.

    Taken from
    https://flax.readthedocs.io/en/latest/quick_start.html#define-network
    """

    num_classes: int = 10

    @flax.linen.compact
    def __call__(self, x: jax.Array):
        x = to_channels_last(x)
        x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = flax.linen.Conv(features=64, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = flatten(x)
        x = flax.linen.Dense(features=256)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=self.num_classes)(x)
        return x


class TorchCNN(torch.nn.Sequential):
    def __init__(self, num_classes: int = 10):
        super().__init__(
            torch.nn.LazyConv2d(out_channels=32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=num_classes),
        )


def flatten(x: jax.Array) -> jax.Array:
    return x.reshape((x.shape[0], -1))


class JaxFcNet(flax.linen.Module):
    num_classes: int = 10

    @flax.linen.compact
    def __call__(self, x: jax.Array):
        x = flatten(x)
        x = flax.linen.Dense(features=256)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=self.num_classes)(x)
        return x


class TorchFcNet(torch.nn.Sequential):
    def __init__(self, num_classes: int = 10):
        super().__init__(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=num_classes),
        )


@pytest.fixture
def num_classes():
    return 10


@pytest.fixture(autouse=True, scope="session")
def make_torch_deterministic():
    mode = torch.get_deterministic_debug_mode()

    torch.set_deterministic_debug_mode("error")
    yield
    torch.set_deterministic_debug_mode(mode)


@pytest.fixture(params=[TorchCNN, TorchFcNet])
def torch_network(
    request: pytest.FixtureRequest,
    seed: int,
    torch_input: torch.Tensor,
    num_classes: int,
    torch_device: torch.device,
):
    torch_network_type: type[torch.nn.Module] = request.param
    with (
        torch_device,
        torch.random.fork_rng([torch_device] if torch_device.type == "cuda" else []),
    ):
        torch_network = torch_network_type(num_classes=num_classes)
        # initialize any un-initialized parameters in the network by doing a forward pass
        # with a dummy input.
        torch_network(torch_input)
    return torch_network


@pytest.fixture(params=[JaxCNN, JaxFcNet])
def jax_network_and_params(
    request: pytest.FixtureRequest,
    seed: int,
    jax_input: jax.Array,
    num_classes: int,
    jax_device: jax.Device,
):
    jax_network_type: type[flax.linen.Module]
    jax_network_type = request.param
    with jax.default_device(jax_device):
        # todo: fix channels_last vs channels_first issues automatically in torch_to_jax?
        jax_network = jax_network_type(num_classes=num_classes)
        jax_params = jax_network.init(jax.random.key(seed), jax_input)
    return jax_network, jax_params
