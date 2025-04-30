from __future__ import annotations

import logging
import operator
import typing
from logging import getLogger as get_logger
from typing import Any, Callable, Generic, Literal, overload

import chex
import jax
import torch
from chex import PyTreeDef
from typing_extensions import Unpack

from torch_jax_interop.types import (
    In,
    is_sequence_of,
    jit,
    value_and_grad,
)

from .to_torch import jax_to_torch
from .types import Aux, JaxPyTree, Params, TorchPyTree
from .utils import log_once

logger = get_logger(__name__)
# todo: make it possible to have different out type than just a single tensor/array.
Out = jax.Array


class WrappedJaxFunction(torch.nn.Module):
    """Wraps a jax function that returns vectors or matrices into a `torch.nn.Module`.

    This function should accept parameters as a first argument, followed by some inputs
    (jax.Arrays) and should return a single output (jax.Array).

    TODOs:

    - [ ] Test and add support for different combinations of .requires_grad in inputs.
    - [ ] Add support for multiple outputs instead of a single tensor.
    - [ ] Somehow support pytrees as inputs instead of just jax Arrays, maybe with a
          classmethod that flattens / unflattens stuff?

    ## Examples

    Suppose we have some jax function we'd like to use in a PyTorch model:

    ```python
    import jax
    import jax.numpy as jnp
    def some_jax_function(params: jax.Array, x: jax.Array):
        '''Some toy function that takes in some parameters and an input vector.'''
        return jnp.dot(x, params)
    ```

    By importing this:

    ```python
    from torch_jax_interop import WrappedJaxFunction
    ```

    We can then wrap this jax function into a torch.nn.Module with learnable parameters:

    ```python
    import torch
    import torch.nn
    module = WrappedJaxFunction(some_jax_function, jax.random.normal(jax.random.key(0), (2, 1)))
    module = module.to("cpu")  # jax arrays are on GPU by default, moving them to CPU for this example.
    ```

    The parameters are now learnable parameters of the module parameters:

    ```python
    dict(module.state_dict())
    {'params.0': tensor([[-0.7848],
            [ 0.8564]])}
    ```

    You can use this just like any other torch.nn.Module:

    ```python
    x, y = torch.randn(2), torch.rand(1)
    output = module(x)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()

    model = torch.nn.Sequential(
        torch.nn.Linear(123, 2),
        module,
    )
    ```
    """

    @overload
    def __init__(
        self,
        jax_function: Callable[[Params, *tuple[jax.Array, ...]], jax.Array],
        jax_params: Params,
        has_aux: Literal[False] = False,
        clone_params: bool = False,
    ):
        ...

    @overload
    def __init__(
        self,
        jax_function: Callable[
            [Params, *tuple[jax.Array, ...]], tuple[jax.Array, JaxPyTree]
        ],
        jax_params: Params,
        has_aux: Literal[True] = True,
        clone_params: bool = False,
    ):
        ...

    @overload
    def __init__(
        self,
        jax_function: Callable[[Params, *tuple[jax.Array, ...]], jax.Array]
        | Callable[[Params, *tuple[jax.Array, ...]], tuple[jax.Array, JaxPyTree]],
        jax_params: Params,
        has_aux: bool = ...,
        clone_params: bool = False,
    ):
        ...

    def __init__(
        self,
        jax_function: Callable[[Params, *tuple[jax.Array, ...]], jax.Array]
        | Callable[[Params, *tuple[jax.Array, ...]], tuple[jax.Array, JaxPyTree]],
        jax_params: Params,
        has_aux: bool = False,
        clone_params: bool = False,
    ):
        """Wraps the given jax function into a torch.nn.Module.

        Parameters
        ----------
        jax_function: Function to wrap.
        jax_params: Initial value for the parameters (PyTree of jax arrays).
        has_aux: Whether the jax function returns an additional output (auxiliary data).
        clone_params: Whether the torch tensors should be copies of the jax parameters \
            instead of sharing the same memory. Set this to `True` when you plan to do \
            distributed training, otherwise you could run into 'invalid device \
            pointer' errors.
        """
        super().__init__()
        self.jax_function = jax_function
        self.has_aux = has_aux

        # Flatten the jax parameters so we can store them in a nn.ParameterList.
        flat_params, self.params_treedef = jax.tree.flatten(jax_params)
        # Register the parameters.
        # Need to call .clone() when doing distributed training, otherwise we get a RuntimeError:
        # Invalid device pointer when trying to share the CUDA memory.
        flat_params = map(jax_to_torch, flat_params)
        if clone_params:
            flat_params = map(operator.methodcaller("clone"), flat_params)
        self.params = torch.nn.ParameterList(flat_params)

    def forward(
        self, *args: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, TorchPyTree]:
        flat_inputs, inputs_treedef = jax.tree.flatten(args)
        # Flatten everything out before passing it to autograd.
        # todo: this should be `flat_outputs` and be unflattened before being returned.
        outputs = _JaxFunction.apply(
            self.jax_function,
            inputs_treedef,
            self.params_treedef,
            self.has_aux,
            *flat_inputs,
            *self.params,
        )
        assert isinstance(outputs, tuple) and len(outputs) == 3
        output, _jvp_fn, aux = outputs
        if self.has_aux:
            return output, aux
        return output

    if typing.TYPE_CHECKING:
        __call__ = forward


Output = jax.Array


class WrappedJaxScalarFunction(WrappedJaxFunction):
    """Wraps a jax function that returns scalars into a `torch.nn.Module`.

    Compared to `WrappedJaxFunction`, this has the advantage of using jax.value_and_grad
    for the combined forward and backward pass.

    This function should accept parameters as a first argument, followed by some inputs
    (jax.Arrays) and should return a tuple with an output and some additional data (aux)
    """

    @overload
    def __init__(
        self,
        jax_function: Callable[
            [Params, *tuple[jax.Array, ...]],
            jax.Array,
        ],
        jax_params: Params,
        has_aux: Literal[False] = False,
        clone_params: bool = False,
    ):
        ...

    @overload
    def __init__(
        self,
        jax_function: Callable[
            [Params, *tuple[jax.Array, ...]],
            tuple[jax.Array, JaxPyTree],
        ],
        jax_params: Params,
        has_aux: Literal[True] = True,
        clone_params: bool = False,
    ):
        ...

    def __init__(
        self,
        jax_function: Callable[
            [Params, *tuple[jax.Array, ...]],
            jax.Array,
        ]
        | Callable[
            [Params, *tuple[jax.Array, ...]],
            tuple[jax.Array, JaxPyTree],
        ],
        jax_params: Params,
        has_aux: bool = True,
        clone_params: bool = False,
    ):
        super().__init__(
            jax_function=jax_function,
            jax_params=jax_params,
            has_aux=has_aux,
            clone_params=clone_params,
        )
        self.jax_function: Callable[
            [Params, *tuple[jax.Array, ...]], tuple[jax.Array, JaxPyTree]
        ]
        self.jax_value_and_grad_function_wrt_only_params = jit(
            value_and_grad(jax_function, argnums=0, has_aux=has_aux)
        )
        self._value_and_grad_fns: dict[
            tuple[bool, ...],  # the `.requires_grad` of the (*inputs, *params).
            Callable[
                [Params, *tuple[jax.Array, ...]],  # same signature as the fn
                tuple[
                    jax.Array | tuple[jax.Array, JaxPyTree],  # returns the output value
                    # and gradients of either just params or params and inputs:
                    Params | tuple[Params, Unpack[tuple[jax.Array, ...]]],
                ],
            ],
        ] = {}

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, TorchPyTree]:
        flat_inputs: list[torch.Tensor]
        flat_inputs, inputs_treedef = jax.tree.flatten(inputs)

        inputs_need_grad = tuple(input.requires_grad for input in flat_inputs)
        params_need_grad = tuple(param.requires_grad for param in self.parameters())
        # IDEA: Reuse or create the `value_and_grad` fn that we need
        # depending on which param / input requires gradients
        n_inputs = inputs_treedef.num_leaves
        n_params = self.params_treedef.num_leaves

        if not self._value_and_grad_fns:
            # When all params need a grad and no input needs one, use the function we
            # already have (argnums=0):
            self._value_and_grad_fns[
                tuple([False] * n_inputs + [True] * n_params)
            ] = self.jax_value_and_grad_function_wrt_only_params
            # When neither the params nor the inputs require a gradient, the
            # value_and_grad function won't be used, so we can just set the same
            # function.
            self._value_and_grad_fns[
                tuple([False] * n_inputs + [False] * n_params)
            ] = self.jax_value_and_grad_function_wrt_only_params

        key = inputs_need_grad + params_need_grad
        # Note: when we don't need the value_and_grad function won't be used anyway.
        if key in self._value_and_grad_fns:
            # We already have the function to be used to compute the value and desired gradients
            value_and_grad_fn = self._value_and_grad_fns[key]
        else:
            logger.info(
                f"Compiling the `value_and_grad` function needed for {inputs_need_grad=} and {params_need_grad=}"
            )
            assert all(params_need_grad)  # assuming all parameters need a grad for now
            # NOTE: Since all parameters are passed via the `param` is the first positional
            # argument, currently either no parameters need a grad, or every parameter needs a grad.
            argnums = (
                0,
                *tuple(
                    i + 1 for i, input in enumerate(flat_inputs) if input.requires_grad
                ),
            )
            logger.debug(f"argnums({argnums=})")
            value_and_grad_fn = jit(
                value_and_grad(
                    self.jax_function,
                    argnums=argnums,
                    has_aux=True,
                )
            )
            self._value_and_grad_fns[key] = value_and_grad_fn

        output = _JaxScalarFunction.apply(
            self.jax_function,
            value_and_grad_fn,
            inputs_treedef,
            self.params_treedef,
            *inputs,
            *self.params,
        )
        assert isinstance(output, tuple) and len(output) == 2
        out, aux = output
        assert isinstance(out, torch.Tensor)
        return out, aux

    if typing.TYPE_CHECKING:
        __call__ = forward


class _JaxFunction(torch.autograd.Function, Generic[Params]):
    """Wrapper for a jax function, making it usable in PyTorch's autograd system.

    TODOs: make this more flexible in terms of input/output signature:
    - [ ] Currently assumes that has_aux is False.
    - [ ] Currently assumes that the function returns a single array.
    - [ ] Currently assumes that the function accepts only params and one input...
    """

    @staticmethod
    def forward(
        jax_function: Callable[[Params, In], Out]
        | Callable[[Params, In], tuple[Out, Aux]],
        inputs_treedef: PyTreeDef,
        params_treedef: PyTreeDef,
        has_aux: bool,
        # need to flatten the params for autograd to understand that they need a gradient.
        *flat_inputs_and_params: torch.Tensor,
    ):
        from .to_jax import torch_to_jax

        n_inputs = inputs_treedef.num_leaves
        flat_inputs, flat_params = (
            flat_inputs_and_params[:n_inputs],
            flat_inputs_and_params[n_inputs:],
        )
        jax_inputs = jax.tree.unflatten(inputs_treedef, map(torch_to_jax, flat_inputs))
        jax_params = jax.tree.unflatten(params_treedef, map(torch_to_jax, flat_params))
        # todo: support multiple outputs and/or `has_aux=True`.
        if has_aux:
            jax_function_with_aux = typing.cast(
                Callable[[Params, In], tuple[Out, Aux]], jax_function
            )
            output, jvp_function, aux = jax.vjp(
                jax_function_with_aux, jax_params, *jax_inputs, has_aux=has_aux
            )
            output = jax.tree.map(jax_to_torch, output)
            aux = jax.tree.map(jax_to_torch, aux)
            return output, jvp_function, aux
        else:
            output, jvp_function = jax.vjp(
                jax_function, jax_params, *jax_inputs, has_aux=has_aux
            )
            output = jax.tree.map(jax_to_torch, output)
            # flat_outputs, = jax.tree.leaves(output)
            return output, jvp_function, None

    if typing.TYPE_CHECKING:
        apply = forward  # type: ignore

    # setup_context is responsible for calling methods and/or assigning to
    # the ctx object. Please do not do additional compute (e.g. add
    # Tensors together) in setup_context.
    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.BackwardCFunction, inputs: tuple, output: tuple
    ):
        (
            jax_function,
            inputs_treedef,
            params_treedef,
            has_aux,
            *inputs_and_params,
        ) = inputs
        output, jvp_function, aux = output
        # Save the function to use to compute the backward pass.
        ctx.jvp_function = jvp_function  # type: ignore
        ctx.inputs_treedef = inputs_treedef  # type: ignore
        ctx.params_treedef = params_treedef  # type: ignore
        ctx.has_aux = has_aux  # type: ignore
        ctx.aux = aux  # type: ignore

    @torch.autograd.function.once_differentiable
    @staticmethod
    def backward(
        ctx: torch.autograd.function.NestedIOFunction,
        *output_grads: Unpack[tuple[torch.Tensor, Unpack[tuple[None, ...]]]],
    ):
        from .to_jax import torch_to_jax

        grad_output, *_unused_output_grads = output_grads
        assert grad_output is not None
        assert all(unused_grad is None for unused_grad in _unused_output_grads)
        needs_input_grad = tuple(ctx.needs_input_grad)

        assert (
            is_sequence_of(needs_input_grad, bool)
            and isinstance(needs_input_grad, tuple)
            and len(needs_input_grad) >= 5
        )
        _, _, _, _, *inputs_and_params_need_grad = needs_input_grad

        jvp_function = ctx.jvp_function  # type: ignore
        inputs_treedef: PyTreeDef = ctx.inputs_treedef  # type: ignore
        params_treedef: PyTreeDef = ctx.params_treedef  # type: ignore
        # fn_had_aux: bool = ctx.has_aux  # type: ignore

        n_inputs = inputs_treedef.num_leaves
        n_params = params_treedef.num_leaves

        inputs_need_grad, params_need_grad = (
            inputs_and_params_need_grad[:n_inputs],
            inputs_and_params_need_grad[n_inputs:],
        )

        _jax_grad_output = torch_to_jax(grad_output)
        _jax_grad_params, *_jax_input_grads = jvp_function(_jax_grad_output)

        flat_param_grads = jax.tree.leaves(jax.tree.map(jax_to_torch, _jax_grad_params))
        flat_input_grads = jax.tree.leaves(jax.tree.map(jax_to_torch, _jax_input_grads))

        # Only give out the gradients if they were requested.
        assert len(flat_param_grads) == n_params
        assert len(flat_input_grads) == n_inputs
        flat_param_grads = tuple(
            flat_param_grad if params_need_grad[i] else None
            for i, flat_param_grad in enumerate(flat_param_grads)
        )
        # We have gradients for inputs that don't require them.
        assert len(flat_input_grads) == len(inputs_need_grad)
        flat_input_grads = tuple(
            flat_input_grad if inputs_need_grad[i] else None
            for i, flat_input_grad in enumerate(flat_input_grads)
        )

        return None, None, None, None, *flat_input_grads, *flat_param_grads

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


class _JaxScalarFunction(torch.autograd.Function, Generic[Params]):
    """Wrapper for a jax scalar-valued function, making it usable in PyTorch's autograd system.

    This has potentially an advantage compared to `JaxFunction` (which is more general):
    It gets to use (and jit) the `jax.value_and_grad` of the function.

    TODO: Assumes that the function "has an aux": that it returns a tuple of (val, aux).
    """

    # todo: If we used setup_context we could maybe have **kwargs in the forward?

    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,
        jax_function: Callable[
            [Params, *tuple[jax.Array, ...]], tuple[chex.Scalar, Aux]
        ],
        jax_value_and_grad_function: Callable[
            [Params, Unpack[tuple[jax.Array, ...]]],
            tuple[
                tuple[chex.Scalar, Aux],  # outputs
                Params  # grads of params only
                | tuple[
                    Params, Unpack[tuple[jax.Array, ...]]
                ],  # grads of params and inputs
            ],
        ],
        inputs_treedef: PyTreeDef,
        params_treedef: PyTreeDef,
        # need to flatten the inputs and params for autograd to understand that they need a gradient.
        *flatened_inputs_and_params: torch.Tensor,
    ):
        # TODO: Keep the aux the same? Or convert to torch?

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

        if not any(inputs_need_grad) and not any(params_need_grad):
            # Do only the forward pass.
            assert not any(inputs_need_grad) and not any(params_need_grad)
            jax_output, jax_aux = jax_function(jax_params, *jax_inputs)
            output = jax.tree.map(jax_to_torch, jax_output)
            # TODO: Keep the aux the same? Or convert to torch?
            aux = jax.tree.map(jax_to_torch, jax_aux)
            return output, aux

        # We only calculate the gradients we care about by changing the argnums
        # passed to value_and_grad.
        (
            (jax_output, jax_aux),
            jax_grads_depending_on_argnums,
        ) = jax_value_and_grad_function(jax_params, *jax_inputs)

        output = jax.tree.map(jax_to_torch, jax_output)
        aux = jax.tree.map(jax_to_torch, jax_aux)

        if any(params_need_grad) and not any(inputs_need_grad):
            # The `value_and_grad_function` is used to calculate the gradients w.r.t.
            # only the parameters.
            flat_jax_grads = jax.tree.leaves(jax_grads_depending_on_argnums)
            param_grads = tuple(map(jax_to_torch, flat_jax_grads))
            ctx.save_for_backward(*param_grads)
            return output, aux

        assert any(inputs_need_grad)
        assert all(params_need_grad)  # assuming that all params need a gradient.

        # The `value_and_grad_function` calculated the gradients w.r.t. the
        # parameters and (some?) inputs.
        assert (
            isinstance(jax_grads_depending_on_argnums, tuple)
            and len(jax_grads_depending_on_argnums) >= 2
        )
        jax_param_grads, *jax_input_grads = jax_grads_depending_on_argnums
        param_grads = jax.tree.leaves(jax.tree.map(jax_to_torch, jax_param_grads))
        # Some inputs might need gradients.
        assert len(jax_input_grads) <= inputs_treedef.num_leaves
        input_grads = jax.tree.leaves(jax.tree.map(jax_to_torch, jax_input_grads))
        ctx.save_for_backward(*input_grads, *param_grads)

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

        # The gradients have already been computed with `value_and_grad`, so here we
        # just return them from ctx.saved_tensors depending on `ctx.needs_input_grad`.

        inputs_treedef: PyTreeDef = ctx.inputs_treedef  # type: ignore
        params_treedef: PyTreeDef = ctx.params_treedef  # type: ignore
        _, _, _, _, *flat_inputs_and_params_need_grad = ctx.needs_input_grad  # type: ignore

        n_inputs = inputs_treedef.num_leaves
        n_params = params_treedef.num_leaves

        inputs_need_grad, params_need_grad = (
            flat_inputs_and_params_need_grad[:n_inputs],
            flat_inputs_and_params_need_grad[n_inputs:],
        )

        if not any(inputs_need_grad) and len(ctx.saved_tensors) == n_params:
            # We saved the gradients of the parameters (but not of the inputs).
            input_grads = [None] * inputs_treedef.num_leaves
            # Remove the grads of parameters that didn't require them.
            param_grads = [
                param_grad if needed_grad else None
                for param_grad, needed_grad in zip(ctx.saved_tensors, params_need_grad)
            ]
            return None, None, None, None, *input_grads, *param_grads

        assert all(params_need_grad)
        assert any(inputs_need_grad)

        saved_tensors: tuple[torch.Tensor, ...] = ctx.saved_tensors  # type: ignore

        # Here we do it slightly differently, because we might not have `n_inputs` input
        # gradients saved: it depends on the `argnums` of the `value_and_grad` function
        # that was used in forward.

        n_inputs_that_needed_grad = sum(inputs_need_grad)
        n_params_that_needed_grad = sum(params_need_grad)
        # We should have as many saved tensors as there are inputs that needed a grad.
        assert (
            len(ctx.saved_tensors)
            == n_inputs_that_needed_grad + n_params_that_needed_grad
        )

        input_grads, param_grads = (
            saved_tensors[:n_inputs_that_needed_grad],
            saved_tensors[n_inputs_that_needed_grad:],
        )
        # Consume the saved input grads, assigning them to the right index:
        saved_input_grads = list(input_grads)
        input_grads = [
            saved_input_grads.pop(0) if needed_grad else None
            for needed_grad in inputs_need_grad
        ]
        # Only give out a grad for parameters that required them.
        # note: un-needed atm, since above we assume all(params_need_grad!)
        param_grads = [
            param_grad if needed_grad else None
            for param_grad, needed_grad in zip(param_grads, params_need_grad)
        ]
        return None, None, None, None, *input_grads, *param_grads
