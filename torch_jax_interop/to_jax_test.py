import logging
import operator
from typing import Any

import jax
import jax.numpy as jnp
import numpy
import optax
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from tensor_regression import TensorRegressionFixture

from torch_jax_interop import jax_to_torch, torch_to_jax
from torch_jax_interop.to_jax import torch_to_jax_nn_module
from torch_jax_interop.to_torch import jax_to_torch_device
from torch_jax_interop.types import jit, value_and_grad
from torch_jax_interop.utils import log_once, to_channels_first


@pytest.mark.parametrize(
    ("shape", "might_warn"),
    [
        ((1,), False),
        ((10, 10), False),
        ((100, 100, 100), False),
        (tuple(range(1, 6)), True),
        ((1, 3, 32, 32), True),
    ],
    ids="shape={}".format,
)
def test_torch_to_jax_tensor(
    torch_device: torch.device,
    shape: tuple[int, ...],
    might_warn: bool,
    torch_dtype: torch.dtype,
    jax_dtype: jax.numpy.dtype,
    seed: int,
    benchmark: BenchmarkFixture,
    caplog: pytest.LogCaptureFixture,
):
    if numpy.prod(shape) >= 1_000_000 and torch_device.type == "cpu":
        pytest.skip("Skipping test with large tensor on CPU.")

    gen = torch.Generator(device=torch_device).manual_seed(seed)
    if torch_dtype.is_floating_point:
        torch_value = torch.rand(
            shape, device=torch_device, generator=gen, dtype=torch_dtype
        )
    else:
        torch_value = torch.randint(
            low=0,
            high=100,
            size=shape,
            device=torch_device,
            generator=gen,
            dtype=torch_dtype,
        )
    assert torch_value.shape == shape

    log_once.cache_clear()
    with caplog.at_level(logging.WARNING):
        jax_value = benchmark(torch_to_jax, torch_value)
    if not might_warn:
        assert not caplog.records
    assert isinstance(jax_value, jax.Array)
    assert jax_value.dtype == jax_dtype

    jax_expected_device = torch_to_jax(torch_value.device)
    assert jax_value.devices() == {jax_expected_device}

    torch_numpy_value = torch_value.cpu().numpy()
    jax_numpy_value = numpy.asarray(jax_value)
    numpy.testing.assert_allclose(torch_numpy_value, jax_numpy_value)

    # round-trip:
    torch_round_trip = jax_to_torch(jax_value)
    assert isinstance(torch_round_trip, torch.Tensor)

    if torch_dtype == torch.float64:
        assert jax_dtype == jnp.float32
        assert torch_round_trip.dtype == torch.float32
        torch.testing.assert_close(
            torch_round_trip, torch_value.to(torch_round_trip.dtype)
        )
    elif torch_dtype == torch.int64:
        assert jax_dtype == jnp.int32
        assert torch_round_trip.dtype == torch.int32
        torch.testing.assert_close(
            torch_round_trip, torch_value.to(torch_round_trip.dtype)
        )
    else:
        torch.testing.assert_close(torch_value, torch_round_trip)


def some_torch_function(x: torch.Tensor) -> torch.Tensor:
    return x + torch.ones_like(x)


def test_torch_to_jax_function(
    torch_device: torch.device,
    benchmark: BenchmarkFixture,
):
    torch_input = torch.arange(5, dtype=torch.int32, device=torch_device)
    torch_function = some_torch_function
    expected_torch_output = torch_function(torch_input)

    jax_input = torch_to_jax(torch_input)
    jax_function = torch_to_jax(torch_function)
    jax_output = benchmark(jax_function, jax_input)

    torch_output = jax_to_torch(jax_output)
    # todo: dtypes might be mismatched for int64 and float64
    torch.testing.assert_close(torch_output, expected_torch_output)

    # todo: Should it return torch Tensor when given a torch Tensor?
    # ? = jax_function(torch_input)


def test_torch_to_jax_nn_module(torch_device: torch.device):
    with torch_device:
        torch_net = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
        torch_params = dict(torch_net.named_parameters())
        torch_input = torch.randn(1, 10, requires_grad=True)
    expected_torch_output = torch_net(torch_input)
    assert isinstance(expected_torch_output, torch.Tensor)
    expected_torch_output.backward(gradient=torch.ones_like(expected_torch_output))
    expected_input_grad = torch_input.grad

    jax_net_fn, jax_net_params = torch_to_jax_nn_module(torch_net)

    for jax_param, torch_param in zip(jax_net_params, torch_params.values()):
        torch.testing.assert_close(jax_to_torch(jax_param), torch_param)

    jax_input = torch_to_jax(torch_input)
    jax_output = jax_net_fn(jax_net_params, jax_input)

    torch_output = jax_to_torch(jax_output)
    torch.testing.assert_close(torch_output, expected_torch_output)

    def loss_fn(params, input):
        return jax_net_fn(params, input).sum()

    grad_fn = jax.grad(loss_fn, argnums=1)

    input_grad = grad_fn(jax_net_params, jax_input)
    torch_input_grad = jax_to_torch(input_grad)

    torch.testing.assert_close(torch_input_grad, expected_input_grad)


class FooBar:
    pass


@pytest.mark.parametrize("unsupported_value", [FooBar()])
def test_log_once_on_unsupported_value(
    unsupported_value: Any, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.DEBUG):
        assert torch_to_jax(unsupported_value) is unsupported_value
    assert len(caplog.records) == 1
    assert "No registered handler for values of type" in caplog.records[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        assert torch_to_jax(unsupported_value) is unsupported_value
    assert len(caplog.records) == 0


@pytest.mark.parametrize("with_jit", [False, True])
def test_use_torch_module_in_jax_graph(
    torch_network: torch.nn.Module,
    jax_input: jax.Array,
    tensor_regression: TensorRegressionFixture,
    num_classes: int,
    seed: int,
    with_jit: bool,
    torch_device: torch.device,
):
    torch_parameters = {name: p for name, p in torch_network.named_parameters()}
    # todo: check that only trainable parameters have a gradient?
    _is_trainable = {name: p.requires_grad for name, p in torch_parameters.items()}
    _num_parameters = len(torch_parameters)
    _total_num_parameters = sum(
        map(
            operator.methodcaller("numel"),
            filter(operator.attrgetter("requires_grad"), torch_parameters.values()),
        )
    )
    flat_torch_params, params_treedef = jax.tree.flatten(torch_parameters)

    # Pass the example output so the fn can be jitted!
    example_out = torch_network(jax_to_torch(jax_input))

    wrapped_torch_network_fn, jax_params = torch_to_jax_nn_module(
        torch_network, example_output=example_out
    )

    assert callable(wrapped_torch_network_fn)
    assert isinstance(jax_params, tuple) and all(
        isinstance(p, jax.Array) for p in jax_params
    )
    assert len(jax_params) == len(flat_torch_params)
    # TODO: Why would the ordering change?!
    jax_param_shapes = sorted([p.shape for p in jax_params])
    torch_param_shapes = sorted([p.shape for p in flat_torch_params])
    assert jax_param_shapes == torch_param_shapes

    # BUG: values are different? Is it only due to the dtype?
    # assert all(
    #     numpy.testing.assert_allclose(jax_p, torch_to_jax(torch_p))
    #     for jax_p, torch_p in zip(
    #         sorted(jax_params, key=operator.attrgetter("shape")),
    #         sorted(flat_torch_params, key=operator.attrgetter("shape")),
    #     )
    # )

    batch_size = jax_input.shape[0]
    labels = jax.random.randint(
        key=jax.random.key(seed),
        minval=0,
        maxval=num_classes,
        shape=(batch_size,),
    )

    def loss_fn(
        params: tuple[jax.Array, ...], x: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        x = to_channels_first(x)
        logits = wrapped_torch_network_fn(params, x)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    # TODO: Unfortunately can't use `.jit` here.. Perhaps there's a way to make only that part stay un-jitted somehow?
    grad_fn = value_and_grad(loss_fn, has_aux=True)

    if with_jit:
        grad_fn = jit(grad_fn)

    (loss, logits), param_grads = grad_fn(jax_params, jax_input, labels)
    assert len(param_grads) == len(jax_params)

    def _get_device(v: jax.Array) -> torch.device:
        assert len(v.devices()) == 1
        jax_device = v.devices().pop()
        return jax_to_torch_device(jax_device)

    assert _get_device(loss) == torch_device
    assert _get_device(logits) == torch_device
    assert len(param_grads) == len(jax_params)
    for param, grad in zip(jax_params, param_grads):
        assert param.shape == grad.shape
        assert param.dtype == grad.dtype
        assert _get_device(param) == torch_device
        assert _get_device(grad) == torch_device

    grads_dict = jax.tree.unflatten(params_treedef, param_grads)
    tensor_regression.check(
        {
            "input": jax_input,
            "output": logits,
            "loss": loss,
        }
        | {name: p for name, p in grads_dict.items()},
        include_gpu_name_in_stats=False,
    )
