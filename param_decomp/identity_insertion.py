"""Insert identity operations into models, before specified modules.

This works by inserting an Identity layer, as a property on the module, then adding a
forward pre-hook to the module that calls it before the forward pass.

This allows downstream functionality to act as if the identity operation is just a regular part of
the model, namely, allowing us to decompose the identity operation.
"""

import fnmatch
from typing import Any

import torch.nn as nn
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from param_decomp.configs import ModulePatternInfoConfig
from param_decomp.models.components import Identity


def pre_id_hook(
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
    target_model: nn.Module, identity_module_info: list[ModulePatternInfoConfig]
) -> None:
    """Insert identity layers before specified modules.

    Args:
        target_model: The model to modify
        identity_module_info: List of ModulePatternInfoConfig. The C values are ignored here
            (used later when creating components), only patterns are used for matching.
    """
    # Extract just the patterns (ignore C values for insertion)
    identity_module_paths: list[str] = []
    matched_patterns: set[str] = set()
    for info in identity_module_info:
        if info.module_pattern in matched_patterns:
            raise ValueError(f"Duplicate pattern '{info.module_pattern}' in identity_module_info")
        for name, _ in target_model.named_modules():
            if fnmatch.fnmatch(name, info.module_pattern):
                matched_patterns.add(info.module_pattern)
                identity_module_paths.append(name)

    unmatched = {info.module_pattern for info in identity_module_info} - matched_patterns
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
        module.register_forward_pre_hook(pre_id_hook, with_kwargs=True)
