from __future__ import annotations

import collections.abc
import dataclasses
import functools
import logging
from logging import getLogger as get_logger
from typing import Any, Callable, overload

import jax
import jax.core
import jaxlib
import jaxlib.xla_extension
import torch
import torch.func
import torch.utils._pytree
from jax.dlpack import from_dlpack as jax_from_dlpack  # type: ignore
from torch.utils.dlpack import to_dlpack as torch_to_dlpack

from .types import (
    Dataclass,
    DataclassType,
    K,
    NestedDict,
    NestedMapping,
)
from .utils import (
    log_once,
)

logger = get_logger(__name__)


@overload
def torch_to_jax(value: torch.Tensor, /) -> jax.Array:
    ...


@overload
def torch_to_jax(value: torch.device, /) -> jax.Device:
    ...


@overload
def torch_to_jax(value: tuple[torch.Tensor, ...], /) -> tuple[jax.Array, ...]:
    ...


@overload
def torch_to_jax(value: list[torch.Tensor], /) -> list[jax.Array]:
    ...


@overload
def torch_to_jax(value: NestedDict[K, torch.Tensor], /) -> NestedDict[K, jax.Array]:
    ...


@overload
def torch_to_jax(value: Any, /) -> Any:
    ...


def torch_to_jax(value: Any, /) -> Any:
    """Converts PyTorch tensors to JAX arrays.

    Converts the tensors "in-place", without the need for copies or moving data to the CPU.

    Args:
      value: torch tensor

    Returns:
      a JAX array
    """
    log_once(
        logger,
        message=f"No registered handler for values of type {type(value)}, returning it as-is.",
        level=logging.DEBUG,
    )
    return value


torch_to_jax = functools.singledispatch(torch_to_jax)  # type: ignore


@torch_to_jax.register(type(None))
@torch_to_jax.register(int)
@torch_to_jax.register(float)
@torch_to_jax.register(str)
@torch_to_jax.register(bool)
@torch_to_jax.register(bytes)
def no_op(v: Any) -> Any:
    return v


def torch_to_jax_tensor(value: torch.Tensor) -> jax.Array:
    """Converts a PyTorch Tensor into a jax.Array.

    NOTE: seems like torch.float64 tensors are implicitly converted to jax.float32 tensors?

    TODOs:
    - [ ] Try to fix some of the issues related to the dimension layout (channels_first vs channels_last?)
    """
    value = value.detach()
    try:
        dlpack = torch_to_dlpack(value)
        return jax_from_dlpack(dlpack, copy=False)
    except jaxlib.xla_extension.XlaRuntimeError as err:
        log_once(
            logger,
            message=(
                f"Unable to view tensor of shape {tuple(value.shape)} as a jax.Array in-place:\n"
                f"'{err}'\n"
                f"Tensors of this shape will be flattened and unflattened (which may or "
                f"may not involve making a copy of the tensor's data)."
            ),
            level=logging.WARNING,
        )
    # NOTE: This may or may not involve making a copy of the tensor.
    # See https://pytorch.org/docs/stable/generated/torch.flatten.html#torch.flatten
    flattened_value = value.flatten()
    dlpack = torch_to_dlpack(flattened_value)
    array: jax.Array = jax_from_dlpack(dlpack, copy=False)
    array = array.reshape(value.shape)
    return array


# Register it like this so the type hints are preserved on the functions (which are also called
# directly in some places).
torch_to_jax.register(torch.Tensor, torch_to_jax_tensor)


@torch_to_jax.register(tuple)
def torch_to_jax_tuple(value: tuple) -> tuple:
    return type(value)(*[torch_to_jax(v) for v in value])  # type: ignore


@torch_to_jax.register(list)
def torch_to_jax_list(value: list) -> list:
    return list(torch_to_jax(v) for v in value)


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
