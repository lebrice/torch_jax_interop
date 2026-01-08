import logging
from typing import Any

import jax
import jax.numpy as jnp
import jax.test_util
import numpy
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from torch_jax_interop import jax_to_torch, torch_to_jax


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (10, 10),
        (10, 10, 10),
        pytest.param(
            tuple(range(1, 6)),
        ),
    ],
    ids=repr,
)
def test_jax_to_torch_tensor(
    shape: tuple[int, ...],
    jax_device: jax.Device,
    torch_dtype: torch.dtype,
    jax_dtype: jnp.dtype,
    seed: int,
    benchmark: BenchmarkFixture,
):
    if numpy.prod(shape) >= 1_000_000 and jax_device.platform == "cpu":
        pytest.skip("Skipping test with large tensor on CPU.")

    key = jax.random.key(seed)
    # todo: don't know what the equivalent is on a np/jax dtype for checking if the dtype is
    # floating-point.
    if torch_dtype.is_floating_point:
        jax_value = jax.random.uniform(key=key, shape=shape, dtype=jax_dtype)
    else:
        jax_value = jax.random.randint(key=key, shape=shape, minval=0, maxval=100, dtype=jax_dtype)
    jax_value = jax.device_put(jax_value, device=jax_device)

    torch_expected_device = jax_to_torch(jax_device)
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


def some_jax_function(x: jnp.ndarray) -> jnp.ndarray:
    return x + jnp.ones_like(x)


def test_jax_to_torch_function(jax_device: torch.device, benchmark: BenchmarkFixture):
    jax_input: jax.Array = jax.device_put(jnp.arange(5), device=jax_device)
    jax_function = some_jax_function
    expected_jax_output = jax_function(jax_input)

    torch_input = jax_to_torch(jax_input)
    torch_function = jax_to_torch(jax_function)
    torch_output = benchmark(torch_function, torch_input)

    jax_output = torch_to_jax(torch_output)
    # todo: dtypes might be mismatched for int64 and float64
    numpy.testing.assert_allclose(jax_output, expected_jax_output)

    # todo: Should it return a jax.Array when given a jax.Array as input?
    # ? = torch_function(jax_input)


class FooBar:
    pass


@pytest.mark.parametrize("unsupported_value", [FooBar()])
def test_log_once_on_unsupported_value(unsupported_value: Any, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.DEBUG):
        assert jax_to_torch(unsupported_value) is unsupported_value
    assert len(caplog.records) == 1
    assert "No registered handler for values of type" in caplog.records[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        assert jax_to_torch(unsupported_value) is unsupported_value
    assert len(caplog.records) == 0
