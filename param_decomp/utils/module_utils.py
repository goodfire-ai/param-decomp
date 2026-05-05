import fnmatch
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import calculate_gain

from param_decomp.configs import ModulePatternInfoConfig


@dataclass
class ModulePathInfo:
    """Path to a module (e.g. "h.1.attn.k_proj") and its associated number of components."""

    module_path: str
    C: int


# This is equivalent to `torch.nn.init._NonlinearityType`, but for some reason this is not always
# importable. see https://github.com/goodfire-ai/param-decomp/actions/runs/16927877557/job/47967138342
_NonlinearityType = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]


def init_param_(
    param: Tensor,
    fan_val: float,
    mean: float = 0.0,
    nonlinearity: _NonlinearityType = "linear",
    generator: torch.Generator | None = None,
) -> None:
    """Fill in param with values sampled from a Kaiming normal distribution.

    Args:
        param: The parameter to initialize
        fan_val: The squared denominator of the std used for the kaiming normal distribution
        mean: The mean of the normal distribution
        nonlinearity: The nonlinearity of the activation function
        generator: The generator to sample from
    """
    gain: float = calculate_gain(nonlinearity)
    std: float = gain / math.sqrt(fan_val)
    with torch.no_grad():
        param.normal_(mean, std, generator=generator)


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
