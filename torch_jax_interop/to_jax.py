from __future__ import annotations

import collections.abc
import dataclasses
import functools
from typing import Any, Callable

import functorch
import jax
import pytorch2jax
import torch
from jax.dlpack import from_dlpack as jax_from_dlpack  # type: ignore
from torch.utils.dlpack import to_dlpack as torch_to_dlpack  # type: ignore

from .types import Dataclass, DataclassType, K, NestedMapping, is_sequence_of


@functools.singledispatch
def torch_to_jax(value: Any, /) -> Any:
    """Converts PyTorch tensors to JAX arrays.

    Converts the tensors "in-place", without the need for copies or moving data to the CPU.

    Args:
      value: torch tensor

    Returns:
      a JAX array
    """
    return value
    # raise NotImplementedError(
    #     f"No registered handler for converting value of type {type(value)} to jax! (value={value})"
    # )


@torch_to_jax.register(type(None))
def no_op(v: Any) -> Any:
    return v


def torch_to_jax_tensor(value: torch.Tensor) -> jax.Array:
    """Converts a PyTorch Tensor into a jax.Array.

    NOTE: seems like torch.float64 tensors are implicitly converted to jax.float32 tensors?
    """
    if not value.is_contiguous():
        value = value.contiguous()
    dlpack = torch_to_dlpack(value)
    array = jax_from_dlpack(dlpack, copy=False)
    return array


# Register it like this so the type hints are preserved on the functions (which are also called
# directly in some places).
torch_to_jax.register(torch.Tensor, torch_to_jax_tensor)


@torch_to_jax.register(collections.abc.Mapping)
def torch_to_jax_dict(
    value: NestedMapping[K, torch.Tensor],
) -> NestedMapping[K, jax.Array]:
    """Converts a dict of PyTorch tensors into a dict of jax.Arrays."""
    return type(value)(**{k: torch_to_jax(v) for k, v in value.items()})  # type: ignore


@torch_to_jax.register(Dataclass)
def torch_to_jax_dataclass(value: DataclassType) -> DataclassType:
    """Converts any torch Tensors in the dataclass fields to jax arrays."""
    return type(value)(**torch_to_jax(dataclasses.asdict(value)))


@torch_to_jax.register(torch.device)
def torch_to_jax_device(torch_device: torch.device) -> jax.Device:
    if torch_device.type == "cuda":
        backend = "gpu"
    elif jax.default_backend() == "tpu":
        backend = "tpu"
    else:
        backend = "cpu"
    devices = jax.devices(backend=backend)
    if torch_device.type == "cuda":
        return devices[torch_device.index]
    else:
        torch_device.index
        return devices[0]


@torch_to_jax.register(collections.abc.Callable)
def torch_to_jax_callable(torch_callable: Callable) -> Callable:
    """Wraps a torch function so that it can be used from jax.

    NOTE: You shouldn't expect jax.jit or jax.grad to work through this torch function (at least
    for now).
    """
    from .to_torch import jax_to_torch

    @functools.wraps(torch_callable)
    def _wrapped(*jax_args, **jax_kwargs):
        torch_args = [jax_to_torch(arg) for arg in jax_args]
        torch_kwargs = {k: jax_to_torch(v) for k, v in jax_kwargs.items()}
        torch_outputs = torch_callable(*torch_args, **torch_kwargs)
        return torch_to_jax(torch_outputs)

    return _wrapped


def torch_to_jax_nn_module(
    model: torch.nn.Module,
) -> tuple[jax.custom_vjp[Any], tuple[jax.Array, ...]]:
    # Convert a Pytorch model to a jax function and parameters
    jax_fn, jax_params = pytorch2jax.convert_pytnn_to_jax(model)
    assert is_sequence_of(jax_params, jax.Array)
    assert isinstance(jax_params, tuple)
    return jax_fn, jax_params


torch_to_jax.register(torch.nn.Module, torch_to_jax_nn_module)


def _convert_pytorch_module_to_jax(model: torch.nn.Module):
    # Copied and adapted from https://github.com/subho406/pytorch2jax/blob/main/pytorch2jax/pytorch2jax.py#L32
    from .to_torch import jax_to_torch

    # Convert the PyTorch model to a functional representation and extract the model function and parameters
    model_fn, model_params = functorch.make_functional(model)

    # Convert the model parameters from PyTorch to JAX representations
    jax_model_params: tuple[jax.Array, ...] = jax.tree.map(torch_to_jax, model_params)

    # Define the apply function using a custom VJP
    @jax.custom_vjp
    def apply(params, *args, **kwargs):
        # Convert the input data from PyTorch to JAX representations
        params = jax.tree_map(jax_to_torch, params)
        args = jax.tree_map(jax_to_torch, args)
        kwargs = jax.tree_map(jax_to_torch, kwargs)
        # Apply the model function to the input data
        out = model_fn(params, *args, **kwargs)
        # Convert the output data from JAX to PyTorch representations
        out = jax.tree_map(torch_to_jax, out)
        return out

    # Define the forward and backward passes for the VJP
    def apply_fwd(params, *args, **kwargs):
        return apply(params, *args, **kwargs), (params, args, kwargs)

    def apply_bwd(res, grads):
        params, args, kwargs = res
        # Convert the input data and gradients from PyTorch to JAX representations
        params = jax.tree_map(jax_to_torch, params)
        args = jax.tree_map(jax_to_torch, args)
        kwargs = jax.tree_map(jax_to_torch, kwargs)
        grads = jax.tree_map(jax_to_torch, grads)
        # Compute the gradients using the model function and convert them from JAX to PyTorch representations
        grads = functorch.vjp(model_fn, params, *args, **kwargs)[1](grads)
        grads = jax.tree_map(torch_to_jax, grads)
        return grads

    apply.defvjp(apply_fwd, apply_bwd)

    # Return the apply function and the converted model parameters
    return apply, jax_model_params
