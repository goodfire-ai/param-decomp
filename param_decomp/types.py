from typing import Annotated, Any, Literal

from annotated_types import Ge, Le

Probability = Annotated[float, Ge(0), Le(1)]
LayerwiseCiFnType = Literal["mlp", "vector_mlp", "shared_mlp"]
GlobalCiFnType = Literal["global_shared_mlp", "global_shared_transformer"]


def runtime_cast[T](type_: type[T], obj: Any) -> T:
    """Typecast with a runtime check."""
    if not isinstance(obj, type_):
        raise TypeError(f"Expected {type_}, got {type(obj)}")
    return obj
