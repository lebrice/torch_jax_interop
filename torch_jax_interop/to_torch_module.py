import logging
import operator
import typing
from logging import getLogger as get_logger
from typing import Any, Callable, Generic, TypeVar

import chex
import flax.linen
import jax
import torch
from chex import PyTreeDef

from torch_jax_interop.types import is_sequence_of

from .to_torch import jax_to_torch
from .utils import log_once

logger = get_logger(__name__)

Params = TypeVar("Params")
Out = TypeVar("Out")
Aux = TypeVar("Aux")


class JaxModule(torch.nn.Module, Generic[Params, Out, Aux]):
    """nn.Module that wraps a jax function (including `flax.linen.Module`s).

    This should be passed a function that accepts two arguments: parameters and an
    input.
    """

    def __init__(
        self,
        jax_function: Callable[[Params, *tuple[jax.Array, ...]], Out],
        jax_params: Params,
        jax_value_and_grad_function: Callable[
            [Params, *tuple[jax.Array, ...]],
            tuple[tuple[Out, Aux], Params],  # grad w.r.t.  params only
        ]
        | Callable[
            [Params, *tuple[jax.Array, ...]],
            # grad w.r.t.  params only
            tuple[tuple[Out, Aux], tuple[Params, *tuple[jax.Array, ...]]],
        ]
        | None = None,
        clone_params: bool = False,
    ):
        super().__init__()
        if isinstance(jax_function, flax.linen.Module):
            jax_function = jax_function.apply  # type: ignore

        self.jax_function = jax_function
        self.jax_value_and_grad_function = jax_value_and_grad_function

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

    def forward(self, *inputs: torch.Tensor) -> Out | tuple[Out, Aux]:
        # todo: check if we could somehow pass an arbitrary number of input arguments instead of just one.

        if self.jax_value_and_grad_function is not None:
            inputs_treedef = jax.tree.structure(inputs)
            output = JaxScalarFunction.apply(
                self.jax_function,
                self.jax_value_and_grad_function,
                inputs_treedef,
                self.params_treedef,
                *inputs,
                *self.params,
            )
        else:
            outputs = JaxFunction.apply(
                self.jax_function,
                self.params_treedef,
                *inputs,
                *self.params,
            )
            assert isinstance(outputs, tuple) and len(outputs) == 2
            output, _jvp_fn = outputs
        return output

    if typing.TYPE_CHECKING:
        __call__ = forward


class JaxFunction(torch.autograd.Function, Generic[Params]):
    """Wrapper for a jax function, making it usable in PyTorch's autograd system.

    TODOs: make this more flexible in terms of input/output signature:
    - [ ] Currently assumes that has_aux is False.
    - [ ] Currently assumes that the function returns a single array.
    - [ ] Currently assumes that the function accepts only params and one input...
    """

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


class JaxScalarFunction(torch.autograd.Function, Generic[Params, Aux]):
    """Wrapper for a jax scalar-valued function, making it usable in PyTorch's autograd
    system.

    This has potentially an advantage compared to `JaxFunction` (which is more general):
    It gets to use (and jit) the `jax.value_and_grad` of the function.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,
        jax_function: Callable[
            [Params, *tuple[jax.Array, ...]], tuple[chex.Scalar, Aux]
        ],
        jax_value_and_grad_function: Callable[
            [Params, *tuple[jax.Array, ...]],
            tuple[tuple[chex.Scalar, Aux], tuple[Params, *tuple[jax.Array, ...]]],
        ],
        inputs_treedef: PyTreeDef,
        params_treedef: PyTreeDef,
        # need to flatten the inputs and params for autograd to understand that they need a gradient.
        *flatened_inputs_and_params: torch.Tensor,
    ):
        from .to_jax import torch_to_jax

        flat_inputs, flat_params = (
            flatened_inputs_and_params[: inputs_treedef.num_leaves],
            flatened_inputs_and_params[inputs_treedef.num_leaves :],
        )
        jax_inputs = jax.tree.unflatten(inputs_treedef, map(torch_to_jax, flat_inputs))
        jax_params = jax.tree.unflatten(params_treedef, map(torch_to_jax, flat_params))

        _, _, _, _, *inputs_and_params_need_grad = ctx.needs_input_grad  # type: ignore

        inputs_need_grad, params_need_grad = (
            inputs_and_params_need_grad[: inputs_treedef.num_leaves],
            inputs_and_params_need_grad[inputs_treedef.num_leaves :],
        )
        # Save these for the backward pass.
        ctx.inputs_treedef = inputs_treedef  # type: ignore
        ctx.params_treedef = params_treedef  # type: ignore

        if any(inputs_need_grad) or any(params_need_grad):
            # todo: a bit hard to tell if the grads are for the params and the input or
            # just the params.
            # It's also hard to tell how to pass the inputs to the function.
            # Should we fix the number of input arguments, for example?
            # todo: only calculate the gradients we care about by changing the argnums
            # passed to value_and_grad, possibly using something like functools.cache to
            # save the results.
            (
                (jax_output, jax_aux),
                jax_grads,
            ) = jax_value_and_grad_function(jax_params, *jax_inputs)

            flat_jax_grads = jax.tree.leaves(jax_grads)
            if len(flat_jax_grads) == params_treedef.num_leaves:
                # The `value_and_grad_function` calculated the gradients w.r.t. only the parameters.
                if any(inputs_need_grad):
                    raise RuntimeError(
                        f"The {jax_value_and_grad_function=} only calculated the gradients "
                        f"for the params (arg 0), but Torch also wants the gradients for the inputs!"
                    )
                param_grads = map(jax_to_torch, flat_jax_grads)
                input_grads = [None] * inputs_treedef.num_leaves
                ctx.save_for_backward(*param_grads)
            else:
                # The `value_and_grad_function` calculated the gradients w.r.t. the parameters and inputs.
                assert isinstance(jax_grads, tuple) and len(jax_grads) >= 2
                jax_param_grads = jax_grads[0]
                param_grads = jax.tree.leaves(
                    jax.tree.map(jax_to_torch, jax_param_grads)
                )
                jax_input_grads = jax.tree.unflatten(inputs_treedef, jax_grads[1:])
                input_grads = jax.tree.leaves(
                    jax.tree.map(jax_to_torch, jax_input_grads)
                )
                ctx.save_for_backward(*input_grads, *param_grads)

            output = jax_to_torch(jax_output)
            aux = jax.tree.map(jax_to_torch, jax_aux)
        else:
            # just a forward pass.
            assert not any(inputs_need_grad) and not any(params_need_grad)
            jax_output, jax_aux = jax_function(jax_params, *jax_inputs)
            output = jax_to_torch(jax_output)
            aux = jax.tree.map(jax_to_torch, jax_aux)
        return output, aux

    @torch.autograd.function.once_differentiable
    @staticmethod
    def backward(
        ctx: torch.autograd.function.NestedIOFunction,
        grad_output: torch.Tensor,
        _grad_aux: Any,
    ):
        assert not grad_output.shape
        assert (grad_output == torch.ones_like(grad_output)).all()
        inputs_treedef: PyTreeDef = ctx.inputs_treedef  # type: ignore
        params_treedef: PyTreeDef = ctx.params_treedef  # type: ignore
        _, _, _, _, *flat_inputs_and_params_need_grad = ctx.needs_input_grad  # type: ignore
        if len(ctx.saved_tensors) == len(flat_inputs_and_params_need_grad):
            inputs_need_frad, params_need_grad = (
                flat_inputs_and_params_need_grad[: inputs_treedef.num_leaves],
                flat_inputs_and_params_need_grad[inputs_treedef.num_leaves :],
            )
            saved_tensors = ctx.saved_tensors
            input_grads, param_grads = (
                saved_tensors[: inputs_treedef.num_leaves],
                saved_tensors[inputs_treedef.num_leaves :],
            )
            input_grads = [
                input_grad if needed_grad else None
                for input_grad, needed_grad in zip(input_grads, inputs_need_frad)
            ]
            param_grads = [
                param_grad if needed_grad else None
                for param_grad, needed_grad in zip(param_grads, params_need_grad)
            ]
            return None, None, None, None, *input_grads, *param_grads
        else:
            # We saved the gradients of the parameters (but not of the inputs).
            inputs_need_grad, params_need_grad = (
                flat_inputs_and_params_need_grad[: inputs_treedef.num_leaves],
                flat_inputs_and_params_need_grad[inputs_treedef.num_leaves :],
            )
            assert not any(inputs_need_grad)
            assert (
                len(ctx.saved_tensors)
                == params_treedef.num_leaves
                == len(params_need_grad)
            )

            input_grads = [None] * inputs_treedef.num_leaves
            param_grads = ctx.saved_tensors

            param_grads = [
                param_grad if needed_grad else None
                for param_grad, needed_grad in zip(param_grads, params_need_grad)
            ]
            return None, None, None, None, *input_grads, *param_grads
