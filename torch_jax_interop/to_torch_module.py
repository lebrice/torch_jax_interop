import logging
import operator
import typing
from logging import getLogger as get_logger
from typing import Callable, Generic, TypeVar

import flax.linen
import jax
import torch
from chex import PyTreeDef

from torch_jax_interop.types import is_sequence_of

from .to_torch import jax_to_torch
from .utils import log_once

Params = TypeVar("Params")
logger = get_logger(__name__)


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
        clone_params: bool = False,
    ):
        super().__init__()
        if isinstance(jax_function, flax.linen.Module):
            jax_function = jax_function.apply  # type: ignore

        self.jax_function = jax_function
        if jit:
            self.jax_function = jax.jit(self.jax_function)

        # Flatten the jax parameters so we can store them in a nn.ParameterList.
        flat_params, self.params_treedef = jax.tree.flatten(jax_params)
        # Register the parameters.
        # Need to call .clone() when doing distributed training, otherwise we get a RuntimeError:
        # Invalid device pointer when trying to share the CUDA memory.
        # TODO: Only do this cloning when necessary (when in distributed training and on
        # a per-tensor basis) instead of every time.
        flat_params = map(jax_to_torch, flat_params)
        if clone_params:
            flat_params = map(operator.methodcaller("clone"), flat_params)
        self.params = torch.nn.ParameterList(flat_params)

    def forward(self, input: torch.Tensor, /) -> torch.Tensor:
        # todo: check if we could somehow pass an arbitrary number of input arguments instead of just one.
        outputs = JaxFunction.apply(
            self.jax_function,
            self.params_treedef,
            input,
            *self.params,
        )
        assert isinstance(outputs, tuple) and len(outputs) == 2
        output, _jvp_fn = outputs
        assert isinstance(output, torch.Tensor)
        return output

    if typing.TYPE_CHECKING:
        __call__ = forward


class JaxFunction(torch.autograd.Function, Generic[Params]):
    """Wrapper for a jax function, making it usable in PyTorch's autograd system."""

    @staticmethod
    def forward(
        jax_function: Callable[[Params, jax.Array], jax.Array],
        params_treedef: PyTreeDef,
        input: torch.Tensor,
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
        jax_function, params_treedef, input, *params = inputs
        output, jvp_function = output
        # Save the function to use to compute the backward pass.
        ctx.jvp_function = jvp_function  # type: ignore

    @torch.autograd.function.once_differentiable
    @staticmethod
    def backward(
        ctx: torch.autograd.function.NestedIOFunction,
        grad_output: torch.Tensor,
        _not_used: None,
    ):
        from .to_jax import torch_to_jax

        needs_input_grad = tuple(ctx.needs_input_grad)
        assert (
            is_sequence_of(needs_input_grad, bool)
            and isinstance(needs_input_grad, tuple)
            and len(needs_input_grad) >= 4
        )
        _, _, input_needs_grad, *params_need_grad = needs_input_grad

        jvp_function = ctx.jvp_function  # type: ignore

        # todo: remove after debugging is finished.
        logger.debug(f"{input_needs_grad=}, {params_need_grad=}")

        input_grad = None
        flat_param_grads = tuple(None for _ in params_need_grad)
        if input_needs_grad or any(params_need_grad):
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
        jax_function: Callable[[Params, jax.Array], jax.Array],
        params_treedef: PyTreeDef,
        input_grad: torch.Tensor,
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
        return output_grads, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[int | None, ...],
        jax_function: Callable[[Params, jax.Array], jax.Array],
        params_treedef: PyTreeDef,
        input: torch.Tensor,
        *params: torch.Tensor,
    ):
        log_once(
            logger,
            message="This is untested! Use at your own risk!",
            level=logging.WARNING,
        )
        # todo: debug and test this further.
        _, _, input_vmap_dim, *params_vmap_dims = in_dims

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
