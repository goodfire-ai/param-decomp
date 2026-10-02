"""Runtime pytree contracts that retain the container's static Python type.

The shared structure name is scoped by the consuming function's `@jaxtyped` context.
"""

from typing import Annotated

from beartype.vale import Is
from jax import Array, ShapeDtypeStruct
from jax.experimental.layout import Format
from jax.sharding import NamedSharding
from jaxtyping import PyTree

type ArrayTree[T] = Annotated[T, Is[lambda tree: isinstance(tree, PyTree[Array, "tree"])]]
type ShapeTree[T] = Annotated[
    T, Is[lambda tree: isinstance(tree, PyTree[ShapeDtypeStruct, "tree"])]
]
type FormatTree = Annotated[object, Is[lambda tree: isinstance(tree, PyTree[Format, "tree"])]]
type ShardingTree = Annotated[
    object, Is[lambda tree: isinstance(tree, PyTree[NamedSharding, "tree"])]
]
