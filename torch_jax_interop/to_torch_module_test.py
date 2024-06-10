import flax.linen
import jax
import jax.test_util
import optax
import pytest
import torch
from flax.typing import VariableDict
from tensor_regression import TensorRegressionFixture

from torch_jax_interop import jax_to_torch
from torch_jax_interop.conftest import JaxCNN, TorchCNN
from torch_jax_interop.to_torch_module import JaxModule
from torch_jax_interop.utils import to_channels_last


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

    wrapped_jax_module = JaxModule(jax.jit(jax_network.apply), jax_params)
    output = wrapped_jax_module(input)

    loss = torch.nn.functional.cross_entropy(output, labels, reduction="mean")
    loss.backward()

    assert len(list(wrapped_jax_module.parameters())) == len(
        jax.tree.leaves(jax_params)
    )
    assert all(p.requires_grad for p in wrapped_jax_module.parameters())
    assert isinstance(output, torch.Tensor) and output.requires_grad
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


def test_use_jax_scalar_function_in_torch_graph(
    jax_network_and_params: tuple[flax.linen.Module, VariableDict],
    torch_input: torch.Tensor,
    tensor_regression: TensorRegressionFixture,
    num_classes: int,
    seed: int,
):
    """Same idea, but now its the entire loss function that is in jax, not just the
    module."""
    jax_network, jax_params = jax_network_and_params

    batch_size = torch_input.shape[0]

    @jax.jit
    def loss_fn(
        params: VariableDict, x: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        logits = jax_network.apply(params, x)
        assert isinstance(logits, jax.Array)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss, logits

    # todo: add a test case where the input is floating point and requires gradients.
    input = (
        torch_input  # note: the input can't require grad because it's an int tensor.
    )
    labels = torch.randint(
        0,
        num_classes,
        (batch_size,),
        device=input.device,
        generator=torch.Generator(device=input.device).manual_seed(seed),
    )

    # Only get the parameter gradients.
    value_and_grad_function = jax.jit(
        jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    )
    wrapped_jax_module = JaxModule(
        loss_fn, jax_params, jax_value_and_grad_function=value_and_grad_function
    )

    assert len(list(wrapped_jax_module.parameters())) == len(
        jax.tree.leaves(jax_params)
    )
    assert all(p.requires_grad for p in wrapped_jax_module.parameters())
    assert not input.requires_grad
    assert not labels.requires_grad
    loss, logits = wrapped_jax_module(input, labels)
    assert isinstance(loss, torch.Tensor) and loss.requires_grad
    assert isinstance(logits, torch.Tensor) and logits.requires_grad
    loss.backward()

    assert all(
        p.requires_grad and p.grad is not None for p in wrapped_jax_module.parameters()
    )

    tensor_regression.check(
        {
            "input": input,
            "output": logits,
            "loss": loss,
            "input_grad": input.grad,
        }
        | {name: p for name, p in wrapped_jax_module.named_parameters()},
        include_gpu_name_in_stats=False,
    )


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
