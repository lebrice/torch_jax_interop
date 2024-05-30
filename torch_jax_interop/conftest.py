import torch
import pytest
import jax.numpy as jnp


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
