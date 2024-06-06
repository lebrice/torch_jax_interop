from __future__ import annotations

import collections.abc
import dataclasses
import functools
import logging
import operator
import typing
from logging import getLogger as get_logger
from typing import Any, Callable, Generic, TypeVar, overload

import flax.linen
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
    """nn.Module that wraps a jax function (including `flax.linen.Module`s).

    This should be passed a function that accepts two arguments: parameters and an
    input.
    """

    def __init__(
        self,
        jax_function: flax.linen.Module | Callable[[Params, jax.Array], jax.Array],
        jax_params: Params,
        jit: bool = True,
    ):
        super().__init__()
        if isinstance(jax_function, flax.linen.Module):
            jax_function = jax_function.apply

        self.jax_function = jax_function
        if jit:
            self.jax_function = jax.jit(self.jax_function)
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
        # todo: check if we could somehow pass an arbitrary number of input arguments instead of just one.
        output = JaxFunction.apply(
            self.params_treedef,
            self.jax_function,
            input,
            *self.params,
        )
        assert isinstance(output, torch.Tensor)
        assert output.requires_grad == input.requires_grad
        return output

    if typing.TYPE_CHECKING:
        __call__ = forward


class JaxFunction(torch.autograd.Function, Generic[Params]):
    """Wrapper for a jax function, making it usable in PyTorch's autograd system."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,
        params_treedef: PyTreeDef,
        jax_function: Callable[[Params, jax.Array], jax.Array],
        input: torch.Tensor,
        *flat_params: torch.Tensor,  # need to flatten the params for autograd to understand that they need a gradient.
    ):
        from .to_jax import torch_to_jax

        jax_input = torch_to_jax(input)
        jax_params = jax.tree.unflatten(params_treedef, map(torch_to_jax, flat_params))
        output, jvp_function = jax.vjp(jax_function, jax_params, jax_input)
        # Save the function to use to compute the backward pass.
        ctx.jvp_function = jvp_function  # type: ignore
        output = jax_to_torch(output)
        return output

    # # setup_context is responsible for calling methods and/or assigning to
    # # the ctx object. Please do not do additional compute (e.g. add
    # # Tensors together) in setup_context.
    # @staticmethod
    # def setup_context(
    #     ctx: torch.autograd.function.BackwardCFunction, inputs: tuple, output: tuple
    # ):
    #     input, params_treedef, jax_function, *params = inputs
    #     output, jvp_function = output
    #     ctx.jvp_function = jvp_function  # type: ignore   # saving for backward.

    @torch.autograd.function.once_differentiable
    @staticmethod
    def backward(
        ctx: torch.autograd.function.NestedIOFunction,
        grad_output: torch.Tensor,
    ):
        from .to_jax import torch_to_jax

        _, _, input_needs_grad, *params_need_grad = ctx.needs_input_grad

        input_grad = None
        flat_param_grads = tuple(None for _ in params_need_grad)
        if input_needs_grad or any(params_need_grad):
            logger.debug(f"{input_needs_grad=}, {params_need_grad=}")
            assert all(params_need_grad) and input_needs_grad  # FIXME: debugging.
            jvp_function = ctx.jvp_function  # type: ignore
            _jax_grad_output = torch_to_jax(grad_output)
            _jax_grad_params, _jax_input_grad = jvp_function(_jax_grad_output)
            flat_param_grads = jax.tree.leaves(
                jax.tree.map(jax_to_torch, _jax_grad_params)
            )

            # Only give out the gradients if they were requested.
            flat_param_grads = tuple(
                flat_param_grad if params_need_grad[i] else None
                for i, flat_param_grad in enumerate(flat_param_grads)
            )
            input_grad = jax_to_torch(_jax_input_grad) if input_needs_grad else None

        # Check that we created all the gradients we needed to.
        assert (input_grad is not None) == input_needs_grad
        for i, param_needs_grad in enumerate(params_need_grad):
            assert (flat_param_grads[i] is not None) == param_needs_grad
        return None, None, input_grad, *flat_param_grads

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

        log_once(
            logger,
            message="This is untested! Use at your own risk!",
            level=logging.WARNING,
        )
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
        log_once(
            logger,
            message="This is untested! Use at your own risk!",
            level=logging.WARNING,
        )
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
