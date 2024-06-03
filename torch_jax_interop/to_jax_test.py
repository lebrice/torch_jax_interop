import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from torch_jax_interop import jax_to_torch, torch_to_jax
from torch_jax_interop.to_jax import torch_to_jax_nn_module


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (10, 10),
        (100, 100, 100),
        pytest.param(
            tuple(range(1, 6)),
            marks=pytest.mark.xfail(
                reason="TODO: Getting a UNIMPLEMENTED due to non-default layout?"
            ),
        ),
    ],
    ids="shape={}".format,
)
def test_torch_to_jax_tensor(
    torch_device: torch.device,
    shape: tuple[int, ...],
    torch_dtype: torch.dtype,
    jax_dtype: jax.numpy.dtype,
    seed: int,
    benchmark: BenchmarkFixture,
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

    jax_value = benchmark(torch_to_jax, torch_value)
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
