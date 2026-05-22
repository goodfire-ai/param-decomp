"""Module-pattern targeting: which target-model modules to decompose, and with what C."""

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass

import torch.nn as nn
from pydantic import Field, PositiveInt

from param_decomp.base_config import BaseConfig


class ModulePatternInfoConfig(BaseConfig):
    """Configuration for a module pattern with its number of components."""

    module_pattern: str = Field(..., description="fnmatch-style pattern to match module names")
    C: PositiveInt = Field(
        ..., description="Number of components for modules matching this pattern"
    )


@dataclass
class ModulePathInfo:
    """Path to a module (e.g. "h.1.attn.k_proj") and its associated number of components."""

    module_path: str
    C: int


def expand_module_patterns(
    model: nn.Module, module_info: Sequence[ModulePatternInfoConfig]
) -> list[ModulePathInfo]:
    """Expand module patterns to concrete module paths with their C values."""
    module_to_info: dict[str, tuple[str, int]] = {}  # module_path -> (pattern, C)

    for info in module_info:
        pattern = info.module_pattern
        c = info.C
        matched_any = False

        for name, _ in model.named_modules():
            if fnmatch.fnmatch(name, pattern):
                matched_any = True

                if name in module_to_info:
                    existing_pattern, _ = module_to_info[name]
                    raise ValueError(
                        f"Module '{name}' matches multiple patterns: "
                        f"'{existing_pattern}' and '{pattern}'"
                    )
                module_to_info[name] = (pattern, c)

        if not matched_any:
            raise ValueError(f"Pattern '{pattern}' in module_info did not match any modules")

    return [ModulePathInfo(module_path=name, C=c) for name, (_, c) in module_to_info.items()]
