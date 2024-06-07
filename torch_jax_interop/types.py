import dataclasses
import typing
from typing import (
    Any,
    ClassVar,
    Mapping,
    ParamSpec,
    Protocol,
    Sequence,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

import jax
import torch

K = TypeVar("K")
V = TypeVar("V")

NestedDict = dict[K, V | "NestedDict[K, V]"]
NestedMapping = Mapping[K, V | "NestedMapping[K, V]"]

T = TypeVar("T", jax.Array, torch.Tensor)


P = ParamSpec("P")
Out_cov = TypeVar("Out_cov", covariant=True)


@runtime_checkable
class Module(Protocol[P, Out_cov]):
    """Procotol for a torch.nn.Module that gives better type hints for the `__call__`
    method."""

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> Out_cov:
        raise NotImplementedError

    if typing.TYPE_CHECKING:
        # note: Only define this for typing purposes so that we don't actually override anything.
        def __call__(self, *args: P.args, **kwagrs: P.kwargs) -> Out_cov:
            ...

        modules = torch.nn.Module.modules
        named_modules = torch.nn.Module.named_modules
        state_dict = torch.nn.Module.state_dict
        zero_grad = torch.nn.Module.zero_grad
        parameters = torch.nn.Module.parameters
        named_parameters = torch.nn.Module.named_parameters
        cuda = torch.nn.Module.cuda
        cpu = torch.nn.Module.cpu
        # note: the overloads on nn.Module.to cause a bug with missing `self`.
        # This shouldn't be a problem.
        to = torch.nn.Module().to


# NOTE: Not using a `runtime_checkable` version of the `Dataclass` protocol here, because it
# doesn't work correctly in the case of `isinstance(SomeDataclassType, Dataclass)`, which returns
# `True` when it should be `False` (since it's a dataclass type, not a dataclass instance), and the
# runtime_checkable decorator doesn't check the type of the attribute (ClassVar vs instance
# attribute).


class _DataclassMeta(type):
    def __subclasscheck__(self, subclass: type) -> bool:
        return dataclasses.is_dataclass(subclass) and not dataclasses.is_dataclass(
            type(subclass)
        )

    def __instancecheck__(self, instance: Any) -> bool:
        return dataclasses.is_dataclass(instance) and dataclasses.is_dataclass(
            type(instance)
        )


class Dataclass(metaclass=_DataclassMeta):
    """A class which is used to check if a given object is a dataclass.

    This plays nicely with @functools.singledispatch, allowing us to register functions
    to be used for dataclass inputs.
    """


class DataclassInstance(Protocol):
    # Copy of the type stub from dataclasses.
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


DataclassType = TypeVar("DataclassType", bound=DataclassInstance)


def is_sequence_of(
    object: Any, item_type: type[V] | tuple[type[V], ...]
) -> TypeGuard[Sequence[V]]:
    """Used to check (and tell the type checker) that `object` is a sequence of items of
    type `V`."""
    try:
        return all(isinstance(value, item_type) for value in object)
    except TypeError:
        return False


# def jit[C: Callable, **P](
#     c: C,
#     _fn: Callable[Concatenate[C, P], Any] = jax.jit,
#     *args: P.args,
#     **kwargs: P.kwargs,
# ) -> C:
#     # Fix `jax.jit` so it preserves the jit-ed function's signature and docstring.
#     return _fn(c, *args, **kwargs)


# def chexify[C: Callable, **P](
#     c: C,
#     _fn: Callable[Concatenate[C, P], Any] = chex.chexify,
#     *args: P.args,
#     **kwargs: P.kwargs,
# ) -> C:
#     # Fix `chex.chexify` so it preserves the jit-ed function's signature and docstring.
#     return _fn(c, *args, **kwargs)
