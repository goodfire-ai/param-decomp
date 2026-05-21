"""Config describing which target modules to decompose."""

from pydantic import Field, PositiveInt

from param_decomp.base_config import BaseConfig


class ModulePatternInfoConfig(BaseConfig):
    """Configuration for a module pattern with its number of components."""

    module_pattern: str = Field(..., description="fnmatch-style pattern to match module names")
    C: PositiveInt = Field(
        ..., description="Number of components for modules matching this pattern"
    )
