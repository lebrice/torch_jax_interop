import flax.linen
import jax
import jax.test_util
import pytest
import torch
from flax.typing import VariableDict
from tensor_regression import TensorRegressionFixture

from torch_jax_interop import jax_to_torch, torch_to_jax
from torch_jax_interop.to_torch import JaxModule
from torch_jax_interop.utils import to_channels_last


class JaxCNN(flax.linen.Module):
    """A simple CNN model.

    Taken from https://flax.readthedocs.io/en/latest/quick_start.html#define-network
    """

    num_classes: int = 10

    @flax.linen.compact
    def __call__(self, x: jax.Array):
        x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = flax.linen.Conv(features=64, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # flatten
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


class JaxFcNet(flax.linen.Module):
    num_classes: int = 10

    @flax.linen.compact
    def __call__(self, x: jax.Array):
        x = x.reshape((x.shape[0], -1))  # flatten
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
def torch_input(torch_device: torch.device, seed: int):
    input_shape: tuple[int, ...] = (1, 3, 32, 32)
    torch_input = torch.randn(
        input_shape,
        generator=torch.Generator(device=torch_device).manual_seed(seed),
        device=torch_device,
    )
    return torch_input


@pytest.fixture
def jax_input(torch_input: torch.Tensor):
    return to_channels_last(torch_to_jax(torch_input))


@pytest.fixture
def num_classes():
    return 10


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


@pytest.fixture(params=[TorchCNN, TorchFcNet])
def torch_network(
    request: pytest.FixtureRequest,
    seed: int,
    torch_input: torch.Tensor,
    num_classes: int,
    torch_device: torch.device,
):
    torch_network_type: type[torch.nn.Module] = request.param
    with torch_device:
        with torch.random.fork_rng([torch_device]):
            torch_network = torch_network_type(num_classes=num_classes)
            # initialize any un-initialized parameters in the network by doing a forward pass
            # with a dummy input.
            torch_network(torch_input)
    return torch_network


@pytest.fixture
def torch_and_jax_networks_with_same_params(
    torch_network: torch.nn.Module,
    jax_network_and_params: tuple[flax.linen.Module, VariableDict],
):
    jax_network, jax_params = jax_network_and_params
    if isinstance(torch_network, TorchCNN) or isinstance(jax_network, JaxCNN):
        pytest.skip(reason="Params dont even lign up, its too hard to do atm.")

    flattened_jax_params, jax_params_treedef = jax.tree.flatten(jax_params)
    torch_params = list(torch_network.parameters())
    assert len(flattened_jax_params) == len(torch_params)

    flattened_jax_params = sorted(flattened_jax_params, key=lambda p: tuple(p.shape))
    torch_params = sorted(torch_params, key=lambda p: tuple(p.shape))

    jax_param_shapes = [p.shape for p in flattened_jax_params]
    torch_param_shapes = [p.shape for p in torch_params]
    assert jax_param_shapes == torch_param_shapes

    # todo: find the equivalence between params, the ordering doesn't appear to be the same.
    with torch.no_grad():
        for jax_param, torch_param in zip(flattened_jax_params, torch_params):
            assert jax_param.shape == torch_param.shape
            # initialize both networks with the same parameters.
            torch_param.data[:] = jax_to_torch(jax_param)[:]

    return jax_network, jax_params, torch_network


@pytest.mark.xfail(reason="Params dont even lign up, its too hard to do atm.")
def test_jax_and_torch_modules_have_same_forward_pass(
    torch_and_jax_networks_with_same_params: tuple[
        flax.linen.Module, VariableDict, torch.nn.Module
    ],
    torch_input: torch.Tensor,
    jax_input: jax.Array,
):
    jax_network, jax_params, torch_network = torch_and_jax_networks_with_same_params

    jax_output = jax_network.apply(jax_params, jax_input)
    torch_output = torch_network(torch_input)

    torch.testing.assert_close(jax_output, torch_output)


def test_use_jax_module_in_torch_graph(
    jax_network_and_params: tuple[flax.linen.Module, VariableDict],
    torch_input: torch.Tensor,
    tensor_regression: TensorRegressionFixture,
    num_classes: int,
    seed: int,
):
    jax_network, jax_params = jax_network_and_params

    batch_size = torch_input.shape[0]

    input = to_channels_last(torch_input).clone().detach().requires_grad_(True)
    labels = torch.randint(
        0,
        num_classes,
        (batch_size,),
        device=input.device,
        generator=torch.Generator(device=input.device).manual_seed(seed),
    )
    wrapped_jax_module = JaxModule(jax_network, jax_params, jit=True)

    assert len(list(wrapped_jax_module.parameters())) == len(
        jax.tree.leaves(jax_params)
    )
    assert all(p.requires_grad for p in wrapped_jax_module.parameters())

    output = wrapped_jax_module(input)
    assert isinstance(output, torch.Tensor) and output.requires_grad

    loss = torch.nn.functional.cross_entropy(output, labels, reduction="mean")
    loss.backward()

    assert all(
        p.requires_grad and p.grad is not None for p in wrapped_jax_module.parameters()
    )

    assert input.grad is not None
    tensor_regression.check(
        {
            "input": input,
            "output": output,
            "loss": loss,
            "input_grad": input.grad,
        }
        | {name: p for name, p in wrapped_jax_module.named_parameters()},
        include_gpu_name_in_stats=False,
    )
