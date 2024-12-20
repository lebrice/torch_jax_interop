# Torch \<-> Jax Interop Utilities

Hey, you there!

- Do you use PyTorch, but are curious about Jax (or vice-versa)? Would you prefer to start adding some (Jax/PyTorch) progressively into your projects rather than to start from scratch?
- Want to avoid the pain of rewriting a model from an existing PyTorch codebase in Jax (or vice-versa)?
- Do you like the performance benefits of Jax, but aren't prepared to sacrifice your nice PyTorch software frameworks (e.g. [Lightning](https://lightning.ai/docs/pytorch/stable/))?

**Well I have some good news for you!**
You can have it all: Sweet, sweet jit-ed functions and automatic differentiation from Jax, as well as mature, widely-used frameworks from the PyTorch software ecosystem.

## What this does

This package contains a few utility functions to simplify interoperability between jax and torch: `torch_to_jax`, `jax_to_torch`, `WrappedJaxFunction`, `torch_module_to_jax`.

This repository contains utilities for converting PyTorch Tensors to JAX arrays and vice versa.
This conversion happens thanks the `dlpack` format, which is a common format for exchanging tensors between different deep learning frameworks. Crucially, this format allows for zero-copy * tensor sharing between PyTorch and JAX.

> \* Note: For some torch tensors with specific memory layouts, for example channels-first image tensors, Jax will refuse to read the array from the dlpack, so we flatten and unflatten the data when converting, which might involve a copy.This is displayed as a warning at the moment on the command-line.

## Installation

We would  **highly** recommend you use [uv](https://docs.astral.sh/uv/) to manage your project dependencies. This greatly helps avoid cuda dependency conflicts between PyTorch and Jax.

```bash
uv add torch-jax-interop
```

Otherwise, if you don't use `uv`:

```bash
pip install torch-jax-interop
```

> This will package only depends on the base (cpu) version of Jax by default.
> If you want to also install the GPU version of jax, use `uv add torch-jax-interop[gpu]` or `uv add jax[cuda12]` directly (or the pip equivalents).

## Comparable projects

- https://github.com/lucidrains/jax2torch: Seems to be the first minimal prototype for something like this. Supports jax2torch for functions, but not the other way around.
- https://github.com/subho406/pytorch2jax: Very similar. The way we convert `torch.nn.Module`s to `jax.custom_vjp` is actually based on their implementation, with some additions (support for jitting, along with more flexible input/output signatures).
- https://github.com/samuela/torch2jax: Takes a different approach: using a `torch.Tensor` subclass and `__torch_fuction__`.
- https://github.com/rdyro/torch2jax: Just found this, seems to have very good support for the torch to jax conversion, but not the other way around. Has additional features like specifying the depth (levels of derivatives).

## Usage

```python
import torch
import jax.numpy as jnp
from torch_jax_interop import jax_to_torch, torch_to_jax
```

Converting `torch.Tensor`s into `jax.Array`s:

```python
import jax
import torch

tensors = {
    "x": torch.randn(5),
    "y": torch.arange(5),
}

jax_arrays = jax.tree.map(torch_to_jax, tensors)
torch_tensors = jax.tree.map(jax_to_torch, jax_arrays)
```

Passing torch.Tensors to a Jax function:

```python
@jax_to_torch
def some_jax_function(x: jnp.ndarray) -> jnp.ndarray:
    return x + jnp.ones_like(x)


torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
some_torch_tensor = torch.arange(5, device=device)

torch_output = some_jax_function(some_torch_tensor)


some_jax_array = jnp.arange(5)


@torch_to_jax
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
    """Some toy function that takes in some parameters and an input vector."""
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
{"params.0": tensor([[-0.7848], [0.8564]])}
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

Same goes for `flax.linen.Module`s, you can now use them in your torch forward / backward pass:

```python
import flax.linen


class Classifier(flax.linen.Module):
    num_classes: int = 10

    @flax.linen.compact
    def __call__(self, x: jax.Array):
        x = x.reshape((x.shape[0], -1))  # flatten
        x = flax.linen.Dense(features=256)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=self.num_classes)(x)
        return x


jax_module = Classifier(num_classes=10)
jax_params = jax_module.init(jax.random.key(0), x)

from torch_jax_interop import WrappedJaxFunction

torch_module = WrappedJaxFunction(jax.jit(jax_module.apply), jax_params)
```

### Torch nn.Module to jax function

```python
>>> import torch
>>> import jax

>>> model = torch.nn.Linear(3, 2, device="cuda")
>>> apply_fn, params = torch_module_to_jax(model)


>>> def loss_function(params, x: jax.Array, y: jax.Array) -> jax.Array:
...     y_pred = apply_fn(params, x)
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
