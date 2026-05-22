"""Decomposition target resolution from fnmatch module patterns."""

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass

import torch.nn as nn
from pydantic import Field, PositiveInt

from param_decomp.base_config import BaseConfig


class DecompositionTargetConfig(BaseConfig):
    """Pattern selecting target modules and the number of components to use for each."""

    module_pattern: str = Field(..., description="fnmatch-style pattern to match module names")
    C: PositiveInt = Field(
        ..., description="Number of components for modules matching this pattern"
    )


@dataclass(frozen=True)
class DecompositionTarget:
    """Resolved module path and the number of components to use for that module."""

    module_path: str
    C: int


def resolve_decomposition_targets(
    model: nn.Module, decomposition_targets: Sequence[DecompositionTargetConfig]
) -> list[DecompositionTarget]:
    """Resolve module patterns to concrete module paths with their C values."""
    module_to_pattern_and_c: dict[str, tuple[str, int]] = {}

    for target in decomposition_targets:
        pattern = target.module_pattern
        c = target.C
        matched_any = False

        for name, _ in model.named_modules():
            if fnmatch.fnmatch(name, pattern):
                matched_any = True

                if name in module_to_pattern_and_c:
                    existing_pattern, _ = module_to_pattern_and_c[name]
                    raise ValueError(
                        f"Module '{name}' matches multiple patterns: "
                        f"'{existing_pattern}' and '{pattern}'"
                    )
                module_to_pattern_and_c[name] = (pattern, c)

        if not matched_any:
            raise ValueError(
                f"Pattern '{pattern}' in decomposition_targets did not match any modules"
            )

    return [
        DecompositionTarget(module_path=name, C=c)
        for name, (_, c) in module_to_pattern_and_c.items()
    ]
