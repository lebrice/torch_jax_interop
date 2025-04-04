import jax
import jax.numpy as jnp
import optax
import pytest
import torch
from tensor_regression import TensorRegressionFixture

from torch_jax_interop import jax_to_torch, torch_to_jax
from torch_jax_interop.to_jax_module import torch_module_to_jax
from torch_jax_interop.to_torch import jax_to_torch_device
from torch_jax_interop.types import jit, value_and_grad
from torch_jax_interop.utils import to_channels_first


def test_torch_to_jax_nn_module(torch_device: torch.device):
    with torch_device:
        torch_net = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
        torch_params = dict(torch_net.named_parameters())
        torch_input = torch.randn(1, 10, requires_grad=True)

    jax_net_fn, jax_net_params = torch_module_to_jax(torch_net)

    for jax_param, torch_param in zip(jax_net_params, torch_params.values()):
        torch.testing.assert_close(jax_to_torch(jax_param), torch_param)

    expected_torch_output = torch_net(torch_input)
    assert isinstance(expected_torch_output, torch.Tensor)
    assert expected_torch_output.requires_grad
    assert expected_torch_output.device == torch_device

    def _loss(output):
        return (output**2).mean()

    loss = _loss(expected_torch_output)
    loss.backward()
    # expected_torch_output.backward(gradient=torch.ones_like(expected_torch_output))
    # Make a copy of the gradients so we can compare them later.
    expected_torch_grads = {
        k: v.grad.detach().clone() for k, v in torch_params.items() if v.grad is not None
    }
    torch_net.zero_grad(set_to_none=True)

    jax_input = torch_to_jax(torch_input)
    jax_output = jax_net_fn(jax_net_params, jax_input)

    torch_output = jax_to_torch(jax_output)
    torch.testing.assert_close(torch_output, expected_torch_output)

    def loss_fn(params, input):
        return _loss(jax_net_fn(params, input))

    grad_fn = jax.grad(loss_fn, argnums=0)
    grads = grad_fn(jax_net_params, jax_input)
    jax_grads = jax.tree.map(jax_to_torch, grads)
    assert isinstance(jax_grads, tuple) and len(jax_grads) == len(jax_net_params)
    assert len(jax_grads) == len(expected_torch_grads)
    for jax_grad, (name, torch_grad) in zip(jax_grads, expected_torch_grads.items()):
        torch.testing.assert_close(jax_grad, torch_grad)


@pytest.mark.parametrize("with_jit", [False, True])
@pytest.mark.parametrize("input_needs_grad", [False, True])
def test_use_torch_module_in_jax_graph(
    torch_network: torch.nn.Module,
    jax_input: jax.Array,
    tensor_regression: TensorRegressionFixture,
    num_classes: int,
    seed: int,
    with_jit: bool,
    torch_device: torch.device,
    input_needs_grad: bool,
):
    torch_parameters = {name: p for name, p in torch_network.named_parameters()}
    # todo: check that only trainable parameters have a gradient?
    # _is_trainable = {name: p.requires_grad for name, p in torch_parameters.items()}
    # _num_parameters = len(torch_parameters)
    # _total_num_parameters = sum(
    #     map(
    #         operator.methodcaller("numel"),
    #         filter(operator.attrgetter("requires_grad"), torch_parameters.values()),
    #     )
    # )

    with torch.random.fork_rng([torch_device] if torch_device.type == "cuda" else []):
        # Pass the example output so the fn can be jitted!
        example_out = torch_network(jax_to_torch(jax_input))

    flat_torch_params, params_treedef = jax.tree.flatten(torch_parameters)
    wrapped_torch_network_fn, jax_params = torch_module_to_jax(
        torch_network, example_output=example_out
    )

    assert callable(wrapped_torch_network_fn)
    assert isinstance(jax_params, tuple) and all(isinstance(p, jax.Array) for p in jax_params)
    assert len(jax_params) == len(flat_torch_params)
    # TODO: Why would the ordering change?!
    jax_param_shapes = sorted([p.shape for p in jax_params])
    torch_param_shapes = sorted([p.shape for p in flat_torch_params])
    assert jax_param_shapes == torch_param_shapes

    # BUG: values are different? Is it only due to the dtype?
    # assert all(
    #     numpy.testing.assert_allclose(jax_p, torch_to_jax(torch_p))
    #     for jax_p, torch_p in zip(
    #         sorted(jax_params, key=operator.attrgetter("shape")),
    #         sorted(flat_torch_params, key=operator.attrgetter("shape")),
    #     )
    # )

    batch_size = jax_input.shape[0]
    labels = jax.random.randint(
        key=jax.random.key(seed),
        minval=0,
        maxval=num_classes,
        shape=(batch_size,),
    )

    def loss_fn(
        params: tuple[jax.Array, ...], x: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        x = to_channels_first(x)
        logits = wrapped_torch_network_fn(params, x)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    if input_needs_grad:
        jax_input = jax_input.astype(jnp.float32)
        grad_fn = value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)
        if with_jit:
            grad_fn = jit(grad_fn)
        (loss, logits), (param_grads, input_grads) = grad_fn(jax_params, jax_input, labels)
        assert len(param_grads) == len(jax_params)
        assert isinstance(input_grads, jax.Array)
        assert input_grads.shape == jax_input.shape
    else:
        grad_fn = value_and_grad(loss_fn, argnums=0, has_aux=True)
        if with_jit:
            grad_fn = jit(grad_fn)
        (loss, logits), param_grads = grad_fn(jax_params, jax_input, labels)
        input_grads = None
        assert len(param_grads) == len(jax_params)

    def _get_device(v: jax.Array) -> torch.device:
        assert len(v.devices()) == 1
        jax_device = v.devices().pop()
        return jax_to_torch_device(jax_device)

    if input_needs_grad:
        assert input_grads is not None
        assert _get_device(input_grads) == torch_device
    assert _get_device(loss) == torch_device
    assert _get_device(logits) == torch_device
    assert len(param_grads) == len(jax_params)
    for param, grad in zip(jax_params, param_grads):
        assert param.shape == grad.shape
        assert param.dtype == grad.dtype
        assert _get_device(param) == torch_device
        assert _get_device(grad) == torch_device

    grads_dict = jax.tree.unflatten(params_treedef, param_grads)

    tensor_regression.check(
        {
            "input": jax_input,
            "output": logits,
            "loss": loss,
        }
        | {name: p for name, p in grads_dict.items()},
        include_gpu_name_in_stats=False,
    )
