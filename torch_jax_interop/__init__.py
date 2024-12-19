"""Tools to help interoperability between PyTorch and Jax code.

## Examples

### Converting [torch.Tensor][]s into [jax.Array][]s and vice-versa:

```python
import jax
import torch
from torch_jax_interop import torch_to_jax, jax_to_torch
tensors = {
   "x": torch.randn(5),
   "y": torch.arange(5),
}

jax_arrays = jax.tree.map(torch_to_jax, tensors)
print(jax_arrays)
# {'x': Array([-0.11146712,  0.12036294, -0.3696345 , -0.24041797, -1.1969243 ], dtype=float32),
#  'y': Array([0, 1, 2, 3, 4], dtype=int32)}

torch_tensors = jax.tree.map(jax_to_torch, jax_arrays)
print(torch_tensors)
# {'x': tensor([-0.1115,  0.1204, -0.3696, -0.2404, -1.1969]),
#  'y': tensor([0, 1, 2, 3, 4], dtype=torch.int32)}
```

### Using a Jax function from PyTorch:

```python
@jax_to_torch
def some_wrapped_jax_function(x: jax.Array) -> jax.Array:
    return x + jax.numpy.ones_like(x)

torch_input = torch.arange(5)
torch_output = some_wrapped_jax_function(torch_input)
print(torch_output)
# tensor([1, 2, 3, 4, 5], dtype=torch.int32)
```

### Using a Torch function from Jax:

```python
@torch_to_jax
def some_wrapped_torch_function(x: torch.Tensor) -> torch.Tensor:
    return x + torch.ones_like(x)

jax_input = jax.numpy.arange(5)
jax_output = some_wrapped_torch_function(jax_input)
print(jax_output)
# Array([1, 2, 3, 4, 5], dtype=int32)
```

### Differentiating through a Jax function in PyTorch:

```python
def some_jax_function(params: jax.Array, x: jax.Array):
    '''Some toy function that takes in some parameters and an input vector.'''
    return jax.numpy.dot(x, params)
```

By importing this:

```python
from torch_jax_interop import WrappedJaxFunction
```

We can then wrap this jax function into a torch.nn.Module with learnable parameters:

```python
module = WrappedJaxFunction(some_jax_function, jax_params=jax.random.normal(jax.random.key(0), (2, 1)))
module = module.to("cpu")  # jax arrays are on GPU by default, moving them to CPU for this example.
```

The parameters are now learnable parameters of the module parameters:

```python
print(dict(module.state_dict()))
# {'params.0': tensor([[-0.7848],
#         [ 0.8564]])}
```

You can use this just like any other torch.nn.Module:

```python
x, y = torch.randn(2), torch.rand(1)
output = module(x)
loss = torch.nn.functional.mse_loss(output, y)
loss.backward()
```

This also works the same way for `flax.linen.Module`s:

```python
import flax
class JaxModule(flax.linen.Module):
    output_dims: int
    @flax.linen.compact
    def __call__(self, x: jax.Array):
        x = x.reshape((x.shape[0], -1))  # flatten
        x = flax.linen.Dense(features=256)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=self.output_dims)(x)
        return x


x = jax.random.uniform(key=jax.random.key(0), shape=(16, 28, 28, 1))
jax_module = JaxModule(output_dims=10)
jax_params = jax_module.init(jax.random.key(0), x)
```

You can still of course jit your Jax code:

```python
wrapped_jax_module = WrappedJaxFunction(jax.jit(jax_module.apply), jax_params=jax_params)
```

And you can then use this jax module in PyTorch:

```python
x = jax_to_torch(x)
y = torch.randint(0, 10, (16,), device=x.device)
logits = wrapped_jax_module(x)
loss = torch.nn.functional.cross_entropy(logits, y, reduction="mean")
loss.backward()
print({name: p.grad.shape for name, p in wrapped_jax_module.named_parameters()})
# {'params.0': torch.Size([256]), 'params.1': torch.Size([784, 256]), 'params.2': torch.Size([10]), 'params.3': torch.Size([256, 10])}
```
"""

from .to_jax import torch_to_jax
from .to_jax_module import torch_module_to_jax
from .to_torch import jax_to_torch
from .to_torch_module import WrappedJaxFunction, WrappedJaxScalarFunction

__all__ = [
    "jax_to_torch",
    "torch_to_jax",
    "WrappedJaxFunction",
    "WrappedJaxScalarFunction",
    "torch_module_to_jax",
]
