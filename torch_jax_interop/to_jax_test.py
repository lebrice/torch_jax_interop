import logging
import shutil
from typing import Any

import jax
import jax.numpy as jnp
import numpy
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from tensor_regression import TensorRegressionFixture

from torch_jax_interop import jax_to_torch, torch_to_jax
from torch_jax_interop.to_jax_module import torch_module_to_jax
from torch_jax_interop.utils import log_once


def test_jax_can_use_the_GPU():
    """Test that Jax can use the GPU if it we have one."""
    # NOTE: Super interesting: Seems like running just an
    # `import jax.numpy; print(jax.numpy.zeros(1).devices())` in a new terminal FAILS, but if you
    # do `import torch` before that, then it works!
    import jax.numpy

    device = jax.numpy.zeros(1).devices().pop()
    if shutil.which("nvidia-smi"):
        assert str(device) == "cuda:0"
    else:
        assert "cpu" in str(device).lower()


def test_torch_can_use_the_GPU():
    """Test that torch can use the GPU if it we have one."""

    assert torch.cuda.is_available() == bool(shutil.which("nvidia-smi"))


@pytest.mark.parametrize(
    ("shape", "might_warn"),
    [
        ((1,), False),
        ((10, 10), False),
        ((100, 100, 100), False),
        (tuple(range(1, 6)), True),
        ((1, 3, 32, 32), True),
    ],
    ids=str,
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
        torch_value = torch.rand(shape, device=torch_device, generator=gen, dtype=torch_dtype)
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
        torch.testing.assert_close(torch_round_trip, torch_value.to(torch_round_trip.dtype))
    elif torch_dtype == torch.int64:
        assert jax_dtype == jnp.int32
        assert torch_round_trip.dtype == torch.int32
        torch.testing.assert_close(torch_round_trip, torch_value.to(torch_round_trip.dtype))
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


class FooBar:
    pass


@pytest.mark.parametrize("unsupported_value", [FooBar()])
def test_log_once_on_unsupported_value(unsupported_value: Any, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.DEBUG):
        assert torch_to_jax(unsupported_value) is unsupported_value
    assert len(caplog.records) == 1
    assert "No registered handler for values of type" in caplog.records[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        assert torch_to_jax(unsupported_value) is unsupported_value
    assert len(caplog.records) == 0


def test_torch_params_dont_change(
    torch_network: torch.nn.Module, tensor_regression: TensorRegressionFixture
):
    tensor_regression.check(
        dict(torch_network.named_parameters()),
        include_gpu_name_in_stats=False,
    )


def test_benchmark_forward_pass(
    torch_network: torch.nn.Module,
    torch_input: torch.Tensor,
    benchmark: BenchmarkFixture,
    tensor_regression: TensorRegressionFixture,
):
    output = torch_network(torch_input)

    jax_fn, params = torch_module_to_jax(torch_network, example_output=output)
    output = benchmark(jax_fn, params, torch_to_jax(torch_input))
    tensor_regression.check(
        {"output": output},
        include_gpu_name_in_stats=False,
    )
