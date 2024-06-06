from __future__ import annotations

import collections.abc
import dataclasses
import functools
import logging
import operator
import typing
from logging import getLogger as get_logger
from typing import Any, Callable, Generic, TypeVar, overload

import jax
import torch
from chex import PyTreeDef
from jax.dlpack import to_dlpack as jax_to_dlpack  # type: ignore (not exported there?)
from torch.utils import dlpack as torch_dlpack

from .types import Dataclass, DataclassType, K, NestedDict, NestedMapping
from .utils import log_once

logger = get_logger(__name__)


@overload
def jax_to_torch(value: jax.Array, /) -> torch.Tensor:
    ...


@overload
def jax_to_torch(value: jax.Device, /) -> torch.device:
    ...


@overload
def jax_to_torch(value: tuple[jax.Array, ...], /) -> tuple[torch.Tensor, ...]:
    ...


@overload
def jax_to_torch(value: list[jax.Array], /) -> list[torch.Tensor]:
    ...


@overload
def jax_to_torch(value: NestedDict[K, jax.Array], /) -> NestedDict[K, torch.Tensor]:
    ...


@overload
def jax_to_torch(value: Any, /) -> Any:
    ...


def jax_to_torch(value: Any, /) -> Any:
    """Converts JAX arrays to PyTorch Tensors.

    Converts the tensors "in-place", without the need for copies or moving data to the CPU.

    Args:
      value: jax array

    Returns:
      a PyTorch tensor
    """
    log_once(
        logger,
        message=f"No registered handler for values of type {type(value)}, returning it as-is.",
        level=logging.DEBUG,
    )
    return value


# Make it a singledispatch here instead of above, so the overloads are presented as
# options for code completion.
jax_to_torch = functools.singledispatch(jax_to_torch)  # type: ignore


# Keep `None`s the same.
@jax_to_torch.register(None | int | float | str | bool | bytes)
def no_op(v: Any) -> Any:
    return v


def jax_to_torch_tensor(value: jax.Array, /) -> torch.Tensor:
    """Converts a Jax array into a torch.Tensor."""
    dpack = jax_to_dlpack(value)
    return torch_dlpack.from_dlpack(dpack)


# Register it like this so the type hints are preserved on the functions (which are also called
# directly in some places).
jax_to_torch.register(jax.Array, jax_to_torch_tensor)


@jax_to_torch.register(tuple)
def jax_to_torch_tuple(value: tuple) -> tuple:
    return type(value)(*[jax_to_torch(v) for v in value])


@jax_to_torch.register(list)
def jax_to_torch_list(value: list) -> list:
    return list(jax_to_torch(v) for v in value)


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


Params = TypeVar("Params")


class JaxModule(torch.nn.Module, Generic[Params]):
    def __init__(
        self,
        jax_function: Callable[[Params, jax.Array], jax.Array],
        jax_params: Params,
    ):
        super().__init__()
        self.jax_function = jax.jit(jax_function)
        params_list, self.params_treedef = jax.tree.flatten(jax_params)
        # Register the parameters.
        # Need to call .clone() when doing distributed training, otherwise we get a RuntimeError:
        # Invalid device pointer when trying to share the CUDA memory.
        # TODO: Only do this cloning when necessary (when in distributed training and on
        # a per-tensor basis) instead of every time.
        self.params = torch.nn.ParameterList(
            map(operator.methodcaller("clone"), map(jax_to_torch, params_list))
        )

    def forward(self, input: torch.Tensor, /) -> torch.Tensor:
        output = JaxFunction.apply(
            input,
            self.params_treedef,
            self.jax_function,
            *self.params,
        )
        assert isinstance(output, tuple) and len(output) == 2
        out, _jvp_function = output
        return out

    if typing.TYPE_CHECKING:
        __call__ = forward


class JaxFunction(torch.autograd.Function, Generic[Params]):
    """Wrapper for a jax function, making it usable in PyTorch's autograd system."""

    @staticmethod
    def forward(
        input: torch.Tensor,
        params_treedef: PyTreeDef,
        jax_function: Callable[[Params, jax.Array], jax.Array],
        *flat_params: torch.Tensor,  # need to flatten the params for autograd to understand that they need a gradient.
    ):
        from .to_jax import torch_to_jax

        jax_input = torch_to_jax(input)
        jax_params = jax.tree.unflatten(params_treedef, map(torch_to_jax, flat_params))
        output, jvp_function = jax.vjp(jax_function, jax_params, jax_input)
        output = jax_to_torch(output)
        return output, jvp_function

    # setup_context is responsible for calling methods and/or assigning to
    # the ctx object. Please do not do additional compute (e.g. add
    # Tensors together) in setup_context.
    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.BackwardCFunction, inputs: tuple, output: tuple
    ):
        input, params_treedef, jax_function, *params = inputs
        output, jvp_function = output
        ctx.jvp_function = jvp_function  # type: ignore   # saving for backward.

    @staticmethod
    def backward(
        ctx: torch.autograd.function.NestedIOFunction,
        grad_output: torch.Tensor,
        _jvp_function_grad: None,
    ):
        from .to_jax import torch_to_jax

        input_need_grad, _, _, *params_needs_grad = ctx.needs_input_grad
        # todo: broaden this a bit in case we need the grad of the input.
        # todo: Figure out how to do jax.grad for a function that outputs a matrix or vector.
        assert not input_need_grad

        grad_input = None
        if input_need_grad or any(params_needs_grad):
            assert all(params_needs_grad)  # assuming every parameter needs a gradient.
            jvp_function = ctx.jvp_function  # type: ignore
            jax_grad_output = torch_to_jax(grad_output)
            jax_grad_params, jax_input_grad = jvp_function(jax_grad_output)
            params_grads = jax.tree.map(jax_to_torch, jax.tree.leaves(jax_grad_params))

            if input_need_grad:
                grad_input = jax_to_torch(jax_input_grad)
        else:
            assert not any(params_needs_grad)
            params_grads = tuple(None for _ in params_needs_grad)
        return grad_input, None, None, *params_grads

    @staticmethod
    def jvp(
        ctx,
        input_grad: torch.Tensor,
        params_treedef: PyTreeDef,
        jax_function: Callable[[Params, jax.Array], jax.Array],
        *params_grads: torch.Tensor,  # need to flatten the params for autograd to understand that they need a gradient.
    ):
        # todo: debug and test this further.
        # https://pytorch.org/docs/stable/notes/extending.html#forward-mode-ad
        # Called after `forward`
        # Should return as many tensors as there were outputs.
        from .to_jax import torch_to_jax

        jax_params = jax.tree.unflatten(params_treedef, map(torch_to_jax, params_grads))
        primals_out, tangents_out = jax.jvp(jax_function, jax_params, input_grad)
        output_grads = jax.tree.map(jax_to_torch, tangents_out)
        return output_grads

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[int | None, ...],
        input: torch.Tensor,
        params_treedef: PyTreeDef,
        jax_function: Callable[[Params, jax.Array], jax.Array],
        *params: torch.Tensor,
    ):
        # todo: debug and test this further.
        input_vmap_dim, _, _, *params_vmap_dims = in_dims

        params_vmap_dims_dict = jax.tree.unflatten(params_treedef, params_vmap_dims)
        # todo: use something like functools.cache so we can jit this?
        vmapped_jax_function = jax.vmap(
            jax_function, in_axes=(params_vmap_dims_dict, input_vmap_dim)
        )
        from .to_jax import torch_to_jax

        jax_params = jax.tree.unflatten(params_treedef, map(torch_to_jax, params))
        jax_input = torch_to_jax(input)
        vmapped_result = vmapped_jax_function(jax_params, jax_input)
        return vmapped_result
