# Torch <-> Jax Interop Utilities

Simple utility functions to simplify interoperability between jax and torch

See also: https://github.com/subho406/pytorch2jax is very similar. We actually use some
of their code to convert nn.Modules to a jax function, although this feature isn't as
well tested as the rest of the code..


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
