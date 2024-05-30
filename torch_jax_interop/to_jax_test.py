import jax
import jax.numpy as jnp
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
