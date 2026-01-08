"""Utility to wrap a nn.Module into a jax function with differentiation.

TODO: Maybe convert a torch.nn.Module into a flax.linen.Module?
"""

from __future__ import annotations

import copy
import functools
from logging import getLogger as get_logger
from typing import Callable, Concatenate, Iterable

import jax
import jax.core
import torch
import torch.func
import torch.utils._pytree

from torch_jax_interop.to_jax import torch_to_jax
from torch_jax_interop.types import Module

from .types import JaxPyTree, Out_cov, P, TorchPyTree

logger = get_logger(__name__)


def make_functional(
    module_with_state: Module[P, Out_cov], disable_autograd_tracking=False
) -> tuple[Callable[Concatenate[Iterable[torch.Tensor], P], Out_cov], tuple[torch.Tensor, ...]]:
    """Backward compatibility equivalent for `functorch.make_functional` in the new torch.func API.

    Adapted from https://gist.github.com/zou3519/7769506acc899d83ef1464e28f22e6cf as suggested by
    this: https://pytorch.org/docs/master/func.migrating.html#functorch-make-functional
    """
    params_dict = dict(module_with_state.named_parameters())
    param_names = params_dict.keys()
    params_values = tuple(params_dict.values())
    if disable_autograd_tracking:
        params_values = tuple(map(torch.Tensor.detach, params_values))

    stateless_module = copy.deepcopy(module_with_state)
    stateless_module.to(device="meta")

    def fmodel(parameters: Iterable[torch.Tensor], *args: P.args, **kwargs: P.kwargs):
        parameters = tuple(parameters)
        if len(parameters) != len(param_names):
            raise RuntimeError(
                f"The wrapped PyTorch model {stateless_module} expected "
                f"{len(param_names)} parameters in its inputs, but only received "
                f"{len(parameters)}."
            )
        params_dict = dict(zip(param_names, parameters))
        return torch.func.functional_call(stateless_module, params_dict, args, kwargs)  # type: ignore

    return fmodel, params_values


def torch_module_to_jax(
    model: Module[..., torch.Tensor], example_output: torch.Tensor | None = None
) -> tuple[jax.custom_vjp[jax.Array], tuple[jax.Array, ...]]:
    """Wrap a pytorch model to be used in a jax computation.

    Copied and adapted from https://github.com/subho406/pytorch2jax/blob/main/pytorch2jax/pytorch2jax.py#L32

    Example
    -------

    ```python
    import torch
    import jax
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0) # doctest:+ELLIPSIS
    # <torch._C.Generator object at ...>
    model = torch.nn.Linear(3, 2, device=torch_device)
    wrapped_model, params = torch_module_to_jax(model)
    def loss_function(params, x: jax.Array, y: jax.Array) -> jax.Array:
        y_pred = wrapped_model(params, x)
        return jax.numpy.mean((y - y_pred) ** 2)
    x = jax.random.uniform(key=jax.random.key(0), shape=(1, 3))
    y = jax.random.uniform(key=jax.random.key(1), shape=(1, 1))
    loss, grad = jax.value_and_grad(loss_function)(params, x, y)
    loss  # doctest: +SKIP
    # Array(0.5914371, dtype=float32)
    grad  # doctest: +SKIP
    # (Array([[-0.02565618, -0.00836356, -0.01682458],
    #        [ 1.0495702 ,  0.34214562,  0.68827784]], dtype=float32), Array([-0.02657786,  1.0872754 ], dtype=float32))
    ```

    To use `jax.jit` on the model, you need to pass an example of an output so we can
    tell the JIT compiler the output shapes and dtypes to expect:

    ```python
    # here we reuse the same model as before:
    wrapped_model, params = torch_module_to_jax(model, example_output=torch.zeros(1, 2, device=torch_device))
    def loss_function(params, x: jax.Array, y: jax.Array) -> jax.Array:
        y_pred = wrapped_model(params, x)
        return jax.numpy.mean((y - y_pred) ** 2)
    loss, grad = jax.jit(jax.value_and_grad(loss_function))(params, x, y)
    loss  # doctest: +SKIP
    # Array(0.5914371, dtype=float32)
    grad  # doctest: +SKIP
    # (Array([[-0.02565618, -0.00836356, -0.01682458],
    #        [ 1.0495702 ,  0.34214562,  0.68827784]], dtype=float32), Array([-0.02657786,  1.0872754 ], dtype=float32))
    ```


    Parameters
    ----------
    model : torch.nn.Module
        A Torch module.
    example_output : torch.Tensor | None, optional
        Example of an output of the model, used to specify the expected shapes and
        dtypes so that this computation can be jitted.

    Returns
    -------
    the functional model and the model parameters (converted to jax arrays).
    """

    if example_output is not None:
        example_output = jax.tree.map(torch_to_jax, example_output)

    from .to_torch import jax_to_torch

    def j2t(v: JaxPyTree) -> TorchPyTree:
        if any(isinstance(v_i, jax.core.Tracer) for v_i in jax.tree.leaves(v)):
            # running inside JIT.
            return jax.pure_callback(
                functools.partial(jax.tree.map, jax_to_torch),
                v,
                v,
                vmap_method="legacy_vectorized",
            )
        return jax.tree.map(jax_to_torch, v)

    def t2j(v: TorchPyTree) -> JaxPyTree:
        if any(isinstance(v_i, jax.core.Tracer) for v_i in jax.tree.leaves(v)):
            # running inside JIT.
            return jax.pure_callback(
                functools.partial(jax.tree.map, torch_to_jax),
                v,
                v,
                vmap_method="legacy_vectorized",
            )
        return jax.tree.map(torch_to_jax, v)

    # Convert the PyTorch model to a functional representation and extract the model function and parameters
    model_fn, model_params = make_functional(model)  # type: ignore

    # Convert the model parameters from PyTorch to JAX representations
    jax_model_params: tuple[jax.Array, ...] = tuple(map(torch_to_jax, model_params))

    # Define the apply function using a custom VJP
    @jax.custom_vjp
    def apply(params, *args, **kwargs):
        # Convert the input data from PyTorch to JAX representations
        # Apply the model function to the input data.
        if example_output is None:
            if any(
                isinstance(v, jax.core.Tracer) for v in jax.tree.leaves((params, args, kwargs))
            ):
                raise RuntimeError(
                    "You need to pass `example_output` in order to JIT the torch function!"
                )
            params = j2t(params)
            args = j2t(args)
            kwargs = j2t(kwargs)
            out = model_fn(params, *args, **kwargs)
            # Convert the output data from JAX to PyTorch
            out = t2j(out)
            return out

        result_shape_dtypes = t2j(example_output)
        # idea: use `torch.compile` as the equivalent of jax's `.jit`?
        jitted_model_fn = torch.compile(model_fn)

        def pytorch_model_callback(params, *args, **kwargs):
            params = jax.tree.map(jax_to_torch, params)
            args = jax.tree.map(jax_to_torch, args)
            kwargs = jax.tree.map(jax_to_torch, kwargs)
            out = jitted_model_fn(params, *args, **kwargs)
            return jax.tree.map(torch_to_jax, out)

        # Pass the jax params to the model function in this case, because
        # jax.pure_callback tries to extract the dtypes of the args.
        out = jax.pure_callback(
            pytorch_model_callback,
            result_shape_dtypes,
            params,
            *args,
            **kwargs,
            vmap_method="legacy_vectorized",
        )
        # Convert the output data from JAX to PyTorch representations
        out = t2j(out)
        return out

    # Define the forward and backward passes for the VJP
    def apply_fwd(params, *args, **kwargs):
        return apply(params, *args, **kwargs), (params, args, kwargs)

    def apply_bwd(res, grads: jax.Array):
        params, args, kwargs = res
        # Convert the input data and gradients from PyTorch to JAX

        if isinstance(grads, jax.core.Tracer):
            jitted_model_fn = torch.compile(model_fn)

            # Compute the gradients using the model function and convert them from JAX to PyTorch
            def _pytorch_model_backward_callback(params, grads, *args, **kwargs):
                torch_params = jax.tree.map(jax_to_torch, params)
                torch_args = jax.tree.map(jax_to_torch, args)
                torch_kwargs = jax.tree.map(jax_to_torch, kwargs)
                torch_grads = jax.tree.map(jax_to_torch, grads)
                _torch_out, torch_jvp_fn = torch.func.vjp(
                    jitted_model_fn, torch_params, *torch_args, **torch_kwargs
                )
                torch_in_grads = torch_jvp_fn(torch_grads)
                return torch_in_grads

            # todo: this seems to depend on the model_fn used. Need to
            result_shape_dtypes = (params, args[0])
            in_grads = jax.pure_callback(
                _pytorch_model_backward_callback,
                result_shape_dtypes,
                params,
                grads,
                *args,
                **kwargs,
                vmap_method="legacy_vectorized",
            )
            in_grads = t2j(in_grads)
            return in_grads
        # not JITed
        torch_params = jax.tree.map(jax_to_torch, params)
        torch_args = jax.tree.map(jax_to_torch, args)
        torch_kwargs = jax.tree.map(jax_to_torch, kwargs)
        torch_grads = jax.tree.map(jax_to_torch, grads)
        _torch_out, torch_jvp_fn = torch.func.vjp(
            model_fn, torch_params, *torch_args, **torch_kwargs
        )
        torch_in_grads = torch_jvp_fn(torch_grads)
        in_grads = jax.tree.map(torch_to_jax, torch_in_grads)
        return in_grads

    apply.defvjp(apply_fwd, apply_bwd)

    # Return the apply function and the converted model parameters
    return apply, jax_model_params


torch_to_jax.register(torch.nn.Module, torch_module_to_jax)
