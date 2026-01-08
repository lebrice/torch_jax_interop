from typing import Callable

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
from torch_jax_interop.to_torch_module import (
    JaxPyTree,
    WrappedJaxFunction,
    WrappedJaxScalarFunction,
)
from torch_jax_interop.types import jit

# TODO: The regression check in this test occasionally fails? Unable to precisely
# replicate it yet.
# This test case seems to fail occasionally:
# - `input_grad` tensor differs in this case: [backend=cuda-JaxFcNet-input_requires_grad=True-aux=True-jit=False-clone_params=False]


@pytest.mark.parametrize("clone_params", [False, True], ids="clone_params={}".format)
@pytest.mark.parametrize("use_jit", [False, True], ids="jit={}".format)
@pytest.mark.parametrize("has_aux", [False, True], ids="aux={}".format)
@pytest.mark.parametrize("input_requires_grad", [False, True], ids="input_requires_grad={}".format)
@pytest.mark.parametrize(
    "do_regression_check",
    [
        False,
        pytest.param(True, marks=pytest.mark.xfail(reason="Regression tests don't work on CPU?")),
    ],
)
def test_use_jax_module_in_torch_graph(
    jax_network_and_params: tuple[flax.linen.Module, VariableDict],
    torch_input: torch.Tensor,
    tensor_regression: TensorRegressionFixture,
    num_classes: int,
    seed: int,
    has_aux: bool,
    use_jit: bool,
    clone_params: bool,
    input_requires_grad: bool,
    torch_device: torch.device,
    do_regression_check: bool,
):
    jax_network, jax_params = jax_network_and_params

    batch_size = torch_input.shape[0]

    input = torch_input.clone().detach().requires_grad_(input_requires_grad)
    labels = torch.randint(
        0,
        num_classes,
        (batch_size,),
        device=input.device,
        generator=torch.Generator(device=input.device).manual_seed(seed),
    )

    if not has_aux:
        jax_function: Callable[[JaxPyTree, *tuple[jax.Array, ...]], jax.Array] = jax_network.apply  # type: ignore

        if use_jit:
            jax_function = jit(jax_function)

        wrapped_jax_module = WrappedJaxFunction(
            jax_function, jax_params, has_aux=has_aux, clone_params=clone_params
        )

        logits = wrapped_jax_module(input)

        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
        loss.backward()

    else:

        def jax_function_with_aux(
            params: JaxPyTree, *inputs: jax.Array
        ) -> tuple[jax.Array, JaxPyTree]:
            out = jax_network.apply(params, *inputs)
            assert isinstance(out, jax.Array)
            aux = {"mean": out.mean(), "max": out.max()}
            return out, aux

        if use_jit:
            jax_function_with_aux = jit(jax_function_with_aux)

        wrapped_jax_module = WrappedJaxFunction(
            jax_function_with_aux,
            jax_params,
            has_aux=has_aux,
            clone_params=clone_params,
        )

        logits, stats_dict = wrapped_jax_module(input)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
        loss.backward()

        # Check that the stats dict has the same structure but contains pytorch tensors
        # instead of jax arrays.

        assert isinstance(stats_dict, dict)
        mean = stats_dict["mean"]
        assert isinstance(mean, torch.Tensor)
        assert mean.device == torch_device
        torch.testing.assert_close(mean, logits.mean())
        assert not mean.requires_grad
        max = stats_dict["max"]
        assert isinstance(max, torch.Tensor)
        assert max.device == torch_device
        torch.testing.assert_close(max, logits.max())
        assert not max.requires_grad

    assert len(list(wrapped_jax_module.parameters())) == len(jax.tree.leaves(jax_params))
    assert all(p.requires_grad for p in wrapped_jax_module.parameters())
    assert isinstance(logits, torch.Tensor) and logits.requires_grad
    assert all(p.requires_grad and p.grad is not None for p in wrapped_jax_module.parameters())
    if input_requires_grad:
        assert input.grad is not None
    else:
        assert input.grad is None

    if do_regression_check:
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


@pytest.mark.parametrize("input_requires_grad", [False, True])
# todo: seems like regression checks fail on CPU!
@pytest.mark.parametrize(
    "do_regression_check",
    [
        False,
        pytest.param(True, marks=pytest.mark.xfail(reason="Regression tests don't work on CPU?")),
    ],
)
def test_use_jax_scalar_function_in_torch_graph(
    jax_network_and_params: tuple[flax.linen.Module, VariableDict],
    torch_input: torch.Tensor,
    tensor_regression: TensorRegressionFixture,
    num_classes: int,
    seed: int,
    input_requires_grad: bool,
    do_regression_check: bool,
):
    """Same idea, but now its the entire loss function that is in jax, not just the module."""
    jax_network, jax_params = jax_network_and_params

    batch_size = torch_input.shape[0]

    @jit
    def loss_fn(params: VariableDict, x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
        logits = jax_network.apply(params, x)
        assert isinstance(logits, jax.Array)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss, logits

    # todo: add a test case where the input is floating point and requires gradients.
    if not input_requires_grad:
        # note: the input can't require grad because it's an int tensor.
        input = torch_input
    else:
        input = torch_input.float().clone().detach().requires_grad_(True)

    labels = torch.randint(
        0,
        num_classes,
        (batch_size,),
        device=input.device,
        generator=torch.Generator(device=input.device).manual_seed(seed),
    )

    wrapped_jax_module = WrappedJaxScalarFunction(loss_fn, jax_params)

    assert len(list(wrapped_jax_module.parameters())) == len(jax.tree.leaves(jax_params))
    assert all(p.requires_grad for p in wrapped_jax_module.parameters())
    if not input_requires_grad:
        assert not input.requires_grad
    else:
        assert input.requires_grad
    assert not labels.requires_grad
    loss, logits = wrapped_jax_module(input, labels)
    assert isinstance(loss, torch.Tensor) and loss.requires_grad
    assert isinstance(logits, torch.Tensor) and logits.requires_grad
    loss.backward()

    assert all(p.requires_grad and p.grad is not None for p in wrapped_jax_module.parameters())
    if input_requires_grad:
        assert input.grad is not None
    else:
        assert input.grad is None

    if do_regression_check:
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
