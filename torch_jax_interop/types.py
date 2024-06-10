import dataclasses
import functools
import typing
from typing import (
    Any,
    Callable,
    ClassVar,
    Concatenate,
    FrozenSet,
    Literal,
    Mapping,
    ParamSpec,
    Protocol,
    Sequence,
    TypeGuard,
    TypeVar,
    overload,
    runtime_checkable,
)

import chex
import jax
import jax.experimental
import jax.experimental.checkify
import torch

K = TypeVar("K")
V = TypeVar("V")
C = TypeVar("C", bound=Callable)
Out = TypeVar("Out")
P = ParamSpec("P")

NestedDict = dict[K, V | "NestedDict[K, V]"]
NestedMapping = Mapping[K, V | "NestedMapping[K, V]"]

T = TypeVar("T")
PyTree = T | tuple["PyTree[T]", ...] | list["PyTree[T]"] | dict[Any, "PyTree[T]"]


def tree_map(f: Callable[[T], Out], tree: PyTree[T]) -> PyTree[Out]:
    """Maps a function over a PyTree."""
    return jax.tree.map(f, tree)


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


def is_list_of(
    object: Any, item_type: type[V] | tuple[type[V], ...]
) -> TypeGuard[list[V]]:
    """Used to check (and tell the type checker) that `object` is a list of items of
    this type."""
    return isinstance(object, list) and is_sequence_of(object, item_type)


def jit(
    fn: Callable[P, Out],
) -> Callable[P, Out]:
    """Small type hint fix for jax's `jit` (preserves the signature of the callable)."""
    return jax.jit(fn)  # type: ignore


In = TypeVar("In")
Aux = TypeVar("Aux")


@overload
def value_and_grad(
    fn: Callable[Concatenate[In, P], tuple[Out, Aux]],
    argnums: Literal[0] = 0,
    has_aux: Literal[True] = True,
) -> Callable[Concatenate[In, P], tuple[tuple[Out, Aux], In]]:
    ...


@overload
def value_and_grad(
    fn: Callable[Concatenate[In, P], tuple[Out, Aux]],
    argnums: Sequence[int],
    has_aux: Literal[True] = True,
) -> Callable[
    Concatenate[In, P], tuple[tuple[Out, Aux], tuple[In, *tuple[jax.Array, ...]]]
]:
    ...


def value_and_grad(
    fn: Callable[Concatenate[In, P], tuple[Out, Aux]],
    argnums: Literal[0] | Sequence[int] = 0,
    has_aux: Literal[True] = True,
) -> Callable[
    Concatenate[In, P], tuple[tuple[Out, Aux], In | tuple[In, *tuple[jax.Array, ...]]]
]:
    """Small type hint fix for jax's `value_and_grad` (preserves the signature of the
    callable)."""
    return jax.value_and_grad(fn, argnums=argnums, has_aux=has_aux)  # type: ignore


def chexify(
    fn: Callable[P, Out],
    async_check: bool = True,
    errors: FrozenSet[
        jax.experimental.checkify.ErrorCategory
    ] = chex.ChexifyChecks.user,
) -> Callable[P, Out]:
    # Fix `chex.chexify` so it preserves the function's signature.
    return functools.wraps(fn)(chex.chexify(fn, async_check=async_check, errors=errors))  # type: ignore
