# Torch <-> Jax Interop Utilities

Simple utility functions to simplify interoperability between jax and torch

See also: https://github.com/subho406/pytorch2jax which is very similar. We actually use some
of their code to convert nn.Modules to a jax function, although we don't currently have any real tests for this feature at the moment.


This repository contains utilities for converting PyTorch Tensors to JAX arrays and vice versa.
This conversion happens thanks the `dlpack` format, which is a common format for exchanging tensors between different deep learning frameworks. Crucially, this format allows for zero-copy tensor sharing between PyTorch and JAX.

## Installation
```bash
pip install torch-jax-interop
```

## Usage

```python
import torch
import jax.numpy as jnp
from torch_jax_interop import jax_to_torch, torch_to_jax

@torch_to_jax
def some_jax_function(x: jnp.ndarray) -> jnp.ndarray:
    return x + jnp.ones_like(x)

torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
some_torch_tensor = torch.arange(5, device=device)
some_jax_array = jnp.arange(5)

assert (jax_to_torch(some_jax_array) == some_torch_array).all()
assert (torch_to_jax(some_torch_array) == some_jax_array).all()


print(some_jax_function(some_torch_tensor))


@jax_to_torch
def some_torch_function(x: torch.Tensor) -> torch.Tensor:
    return x + torch.ones_like(x)

print(some_torch_function(some_jax_array))
```


## Examples


### Jax to Torch nn.Module

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


### Torch nn.Module to jax function


```python
>>> import torch
>>> import jax
>>> model = torch.nn.Linear(3, 2, device="cuda")
>>> apply, params = torch_module_to_jax(model)
>>> def loss_function(params, x: jax.Array, y: jax.Array) -> jax.Array:
...     y_pred = apply(params, x)
...     return jax.numpy.mean((y - y_pred) ** 2)
>>> x = jax.random.uniform(key=jax.random.key(0), shape=(1, 3))
>>> y = jax.random.uniform(key=jax.random.key(1), shape=(1, 1))
>>> loss, grad = jax.value_and_grad(loss_function)(params, x, y)
>>> loss
Array(0.3944674, dtype=float32)
>>> grad
(Array([[-0.46541408, -0.15171866, -0.30520514],
        [-0.7201077 , -0.23474531, -0.47222584]], dtype=float32), Array([-0.4821338, -0.7459771], dtype=float32))
```

To use `jax.jit` on the model, you need to pass an example of an output so we can
tell the JIT compiler the output shapes and dtypes to expect:

```python
>>> # here we reuse the same model as before:
>>> apply, params = torch_module_to_jax(model, example_output=torch.zeros(1, 2, device="cuda"))
>>> def loss_function(params, x: jax.Array, y: jax.Array) -> jax.Array:
...     y_pred = apply(params, x)
...     return jax.numpy.mean((y - y_pred) ** 2)
>>> loss, grad = jax.jit(jax.value_and_grad(loss_function))(params, x, y)
>>> loss
Array(0.3944674, dtype=float32)
>>> grad
(Array([[-0.46541408, -0.15171866, -0.30520514],
        [-0.7201077 , -0.23474531, -0.47222584]], dtype=float32), Array([-0.4821338, -0.7459771], dtype=float32))
```
