from typing import override

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from param_decomp.interfaces import LoadableModule, RunInfo
from param_decomp.param_decomp_types import ModelPath


class DeepLinearModel(LoadableModule):
    """A deep linear network with L identity D x D layers and a final softmax.

    The target model is a series of L `nn.Linear(D, D, bias=False)` modules, each
    initialised to the identity matrix, followed by `softmax(beta * x)` along the last
    dim. This serves as a sanity check for parameter decomposition: a faithful
    decomposition should learn one component per input dimension per layer.
    """

    def __init__(self, D: int, L: int, beta: float):
        super().__init__()
        self.D = D
        self.L = L
        self.beta = beta
        identity = torch.eye(D)
        self.layers = nn.ModuleList()
        for _ in range(L):
            layer = nn.Linear(D, D, bias=False)
            with torch.no_grad():
                layer.weight.copy_(identity)
            self.layers.append(layer)

    @override
    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        for layer in self.layers:
            x = layer(x)
        return F.softmax(self.beta * x, dim=-1)

    @classmethod
    @override
    def from_pretrained(cls, _path: ModelPath) -> "DeepLinearModel":
        raise NotImplementedError(
            "DeepLinearModel is constructed directly from its DeepLinearTaskConfig; "
            "there is no pretrained checkpoint to load."
        )

    @classmethod
    @override
    def from_run_info(cls, _run_info: RunInfo[object]) -> "DeepLinearModel":
        raise NotImplementedError(
            "DeepLinearModel is constructed directly from its DeepLinearTaskConfig; "
            "there is no run info to load from."
        )
