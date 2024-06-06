import dataclasses
from typing import Any, ClassVar, Mapping, Protocol, Sequence, TypeGuard, TypeVar

import jax
import torch

K = TypeVar("K")
V = TypeVar("V")

NestedDict = dict[K, V | "NestedDict[K, V]"]
NestedMapping = Mapping[K, V | "NestedMapping[K, V]"]

T = TypeVar("T", jax.Array, torch.Tensor)


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
