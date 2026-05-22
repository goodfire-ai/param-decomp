from typing import Annotated, Literal

from annotated_types import Ge, Le

Probability = Annotated[float, Ge(0), Le(1)]
LayerwiseCiFnType = Literal["mlp", "vector_mlp", "shared_mlp"]
GlobalCiFnType = Literal["global_shared_mlp", "global_shared_transformer"]
