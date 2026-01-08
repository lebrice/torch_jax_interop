from __future__ import annotations

import collections.abc
import contextlib
import dataclasses
import functools
import logging
import warnings
from logging import getLogger as get_logger
from typing import Any, Callable, overload

import jax
import jax.core
import torch
import torch.func
import torch.utils._pytree
from jax.dlpack import from_dlpack as jax_from_dlpack  # type: ignore

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
def torch_to_jax(value: torch.Tensor, /) -> jax.Array: ...


@overload
def torch_to_jax(value: torch.device, /) -> jax.Device: ...


@overload
def torch_to_jax(value: tuple[torch.Tensor, ...], /) -> tuple[jax.Array, ...]: ...


@overload
def torch_to_jax(value: list[torch.Tensor], /) -> list[jax.Array]: ...


@overload
def torch_to_jax(value: NestedDict[K, torch.Tensor], /) -> NestedDict[K, jax.Array]: ...


@overload
def torch_to_jax(value: Any, /) -> Any: ...


def torch_to_jax(value: Any, /) -> Any:
    """Converts PyTorch tensors to JAX arrays.

    Converts the tensors "in-place", without the need for copies or moving data to the CPU.

    Args:
      value: a torch tensor

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


def _direct_conversion(v: torch.Tensor) -> jax.Array:
    return jax_from_dlpack(v, copy=False)


def _to_from_dlpack(v: torch.Tensor, ignore_deprecation_warning: bool = True) -> jax.Array:
    with warnings.catch_warnings() if ignore_deprecation_warning else contextlib.nullcontext():
        # Only way to get this to work for CPU seems to be with to/from dlpack... so we have to use this deprecated
        # conversion method for now.
        # todo: Should we let it though though?
        if ignore_deprecation_warning:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
        return jax_from_dlpack(v, copy=False)


def torch_to_jax_tensor(value: torch.Tensor) -> jax.Array:
    """Converts a PyTorch Tensor into a jax.Array.

    NOTE: seems like torch.float64 tensors are implicitly converted to jax.float32 tensors?
    TODO:
    - If the tensor is on the GPU, then we can use the direct conversion with jax from_dlpack.
      Otherwise we might have to convert to/from dlpack, which is apparently being deprecated.
        - ALSO: this seems to happen when jitted code is calling a pure callback. Not sure if it happens in other cases too
        (e.g. just calling this with a jax tensor in non-jit mode).
    """
    value = value.detach()

    if value.device.type == "cpu":
        try:
            # todo: Calling jax_from_dlpack with a cpu tensor causes issues in jax pure callbacks **later**,
            # when they are run by jax somehow. This causes issues when using a nn.Module in jax graph.
            # return _direct_conversion(value)
            return _to_from_dlpack(value, ignore_deprecation_warning=True)

        except RuntimeError as err:
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
            return _direct_conversion(value.flatten()).reshape(value.shape)

    try:
        return _direct_conversion(value)
    except RuntimeError as err:
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
        return _direct_conversion(value.flatten()).reshape(value.shape)

    # NOTE: This may or may not involve making a copy of the tensor.
    # See https://pytorch.org/docs/stable/generated/torch.flatten.html#torch.flatten
    return torch_to_jax_tensor(value.flatten()).reshape(value.shape)


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
