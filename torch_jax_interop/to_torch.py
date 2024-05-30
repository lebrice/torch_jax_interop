from __future__ import annotations

import collections.abc
import dataclasses
import functools
from typing import Any, Callable

import jax
import torch
from jax.dlpack import to_dlpack as jax_to_dlpack  # type: ignore (not exported there?)
from torch.utils import dlpack as torch_dlpack

from .types import Dataclass, DataclassType, NestedMapping


@functools.singledispatch
def jax_to_torch(value: Any, /) -> Any:
    """Converts JAX arrays to PyTorch Tensors.

    Converts the tensors "in-place", without the need for copies or moving data to the CPU.

    Args:
      value: jax array

    Returns:
      a PyTorch tensor
    """
    raise NotImplementedError(
        f"No registered handler for converting value of type {type(value)} to jax! (value={value})"
    )


# Keep `None`s the same.
@jax_to_torch.register(type(None))
def no_op(v: Any) -> Any:
    return v


def jax_to_torch_tensor(value: jax.Array, /) -> torch.Tensor:
    """Converts a Jax array into a torch.Tensor."""
    dpack = jax_to_dlpack(value)
    return torch_dlpack.from_dlpack(dpack)


# Register it like this so the type hints are preserved on the functions (which are also called
# directly in some places).
jax_to_torch.register(jax.Array, jax_to_torch_tensor)


@jax_to_torch.register(collections.abc.Mapping)
def jax_to_torch_mapping(
    value: NestedMapping[str, jax.Array | Any],
) -> NestedMapping[str, torch.Tensor | Any]:
    """Converts a dict of Jax arrays into a dict of PyTorch tensors ."""
    return type(value)(**{k: jax_to_torch(v) for k, v in value.items()})  # type: ignore


@jax_to_torch.register(Dataclass)
def jax_to_torch_dataclass(value: DataclassType) -> DataclassType:
    """Converts any jax.Arrays in the dataclass fields to torch Tensors."""
    return type(value)(**jax_to_torch(dataclasses.asdict(value)))


@jax_to_torch.register(jax.Device)
def jax_to_torch_device(jax_device: jax.Device) -> torch.device:
    jax_device_str = str(jax_device)
    if jax_device_str.startswith("cuda"):
        device_type, _, index = jax_device_str.partition(":")
        assert index.isdigit()
        return torch.device(device_type, int(index))
    return torch.device("cpu")


@jax_to_torch.register(collections.abc.Callable)
def jax_to_torch_callable(jax_callable: Callable) -> Callable:
    """Wraps a jax function so that it can be used from pytorch.

    NOTE: You shouldn't the backward pass through this jax function to work (at least for now).

    TODO: Create a custom autograd Function that computes the gradient using jax.grad.
    """
    from .to_jax import torch_to_jax

    @functools.wraps(jax_callable)
    def _wrapped(*torch_args, **torch_kwargs):
        jax_args = [torch_to_jax(arg) for arg in torch_args]
        jax_kwargs = {k: torch_to_jax(v) for k, v in torch_kwargs.items()}
        jax_outputs = jax_callable(*jax_args, **jax_kwargs)
        return jax_to_torch(jax_outputs)

    return _wrapped
