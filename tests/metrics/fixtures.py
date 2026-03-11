"""Shared test fixtures for loss function tests."""

from typing import override

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spd.configs import LayerwiseCiConfig
from spd.models.component_model import ComponentModel
from spd.utils.module_utils import ModulePathInfo


class OneLayerLinearModel(nn.Module):
    """One-layer linear model for testing."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, d_out, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class TwoLayerLinearModel(nn.Module):
    """Two-layer linear model for testing."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def make_one_layer_component_model(
    weight: Float[Tensor, "d_out d_in"], C: int = 1
) -> ComponentModel:
    """Create a ComponentModel with a single linear layer for testing.

    Args:
        weight: Weight matrix for the linear layer

    Returns:
        ComponentModel wrapping the OneLayerLinearModel
    """
    d_out, d_in = weight.shape
    target = OneLayerLinearModel(d_in=d_in, d_out=d_out)
    with torch.no_grad():
        target.fc.weight.copy_(weight)
    target.requires_grad_(False)

    comp_model = ComponentModel(
        target_model=target,
        module_path_info=[ModulePathInfo(module_path="fc", C=C)],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    return comp_model


def make_two_layer_component_model(
    weight1: Float[Tensor, " d_hidden d_in"], weight2: Float[Tensor, " d_out d_hidden"]
) -> ComponentModel:
    """Create a ComponentModel with two linear layers for testing.

    Args:
        weight1: Weight matrix for the first linear layer
        weight2: Weight matrix for the second linear layer

    Returns:
        ComponentModel wrapping the TwoLayerLinearModel
    """
    d_hidden, d_in = weight1.shape
    d_out, d_hidden2 = weight2.shape
    assert d_hidden == d_hidden2, "Hidden dimensions must match"

    target = TwoLayerLinearModel(d_in=d_in, d_hidden=d_hidden, d_out=d_out)
    with torch.no_grad():
        target.fc1.weight.copy_(weight1)
        target.fc2.weight.copy_(weight2)
    target.requires_grad_(False)

    comp_model = ComponentModel(
        target_model=target,
        module_path_info=[
            ModulePathInfo(module_path="fc1", C=1),
            ModulePathInfo(module_path="fc2", C=1),
        ],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    return comp_model
