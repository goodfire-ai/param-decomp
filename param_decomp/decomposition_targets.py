"""Decomposition target resolution from fnmatch module patterns, plus identity insertion.

Identity insertion works by attaching an `Identity` layer to a module as a `pre_identity`
attribute, then registering a forward pre-hook that calls it before the module's forward
pass. This lets downstream functionality treat the identity operation as a regular part of
the model, so it can be decomposed.
"""

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, override

import torch
import torch.nn as nn
from pydantic import Field, PositiveInt
from transformers.pytorch_utils import Conv1D as RadfordConv1D

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


class Identity(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _pre_id_hook(
    mod: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[Any, Any],
) -> tuple[tuple[Any, ...], dict[Any, Any]]:
    assert len(args) == 1, f"Expected 1 positional arg, got {len(args)}"
    # assert no kwargs. This may be overkill. can consider passing kwargs through later but this is
    # simple for now.
    assert not kwargs, f"Expected no kwargs, got {kwargs.keys()}"
    assert hasattr(mod, "pre_identity"), f"Module {mod} has no pre_identity attribute"
    assert isinstance(mod.pre_identity, Identity), (
        f"Module {mod} pre_identity is not an Identity layer"
    )
    return (mod.pre_identity(args[0]),), {}


def insert_identity_operations_(
    target_model: nn.Module, identity_decomposition_targets: list[DecompositionTargetConfig]
) -> None:
    """Insert identity layers before specified modules.

    Args:
        target_model: The model to modify
        identity_decomposition_targets: Decomposition targets whose patterns select modules that
            should receive a pre-identity. C is used later when creating components.
    """
    # Extract just the patterns (ignore C values for insertion)
    identity_module_paths: list[str] = []
    matched_patterns: set[str] = set()
    for target in identity_decomposition_targets:
        if target.module_pattern in matched_patterns:
            raise ValueError(
                f"Duplicate pattern '{target.module_pattern}' in identity_decomposition_targets"
            )
        for name, _ in target_model.named_modules():
            if fnmatch.fnmatch(name, target.module_pattern):
                matched_patterns.add(target.module_pattern)
                identity_module_paths.append(name)

    unmatched = {
        target.module_pattern for target in identity_decomposition_targets
    } - matched_patterns
    if unmatched:
        raise ValueError(f"Identity patterns did not match any modules: {sorted(unmatched)}")

    for module_path in identity_module_paths:
        module = target_model.get_submodule(module_path)

        match module:
            case nn.Linear():
                _, d_in = module.weight.shape
            case RadfordConv1D():
                d_in, _ = module.weight.shape
            case nn.Embedding():
                raise ValueError("Embedding modules not supported for identity insertion")
            case _:
                raise ValueError(f"Module {module} not supported. type: {type(module)}")

        module.pre_identity = Identity(d_in)
        module.register_forward_pre_hook(_pre_id_hook, with_kwargs=True)
