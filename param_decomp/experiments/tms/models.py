from dataclasses import dataclass
from typing import Any, Self, override

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from param_decomp.experiments.tms.configs import TMSModelConfig, TMSTrainConfig
from param_decomp.interfaces import LoadableModule, RunInfo
from param_decomp.param_decomp_types import ModelPath


@dataclass
class TMSTargetRunInfo(RunInfo[TMSTrainConfig]):
    """Run info from training a TMSModel."""

    config_class = TMSTrainConfig
    config_filename = "tms_train_config.yaml"
    checkpoint_filename = "tms.pth"


class TMSModel(LoadableModule):
    def __init__(self, config: TMSModelConfig):
        super().__init__()
        self.config = config

        self.linear1 = nn.Linear(config.n_features, config.n_hidden, bias=False)
        self.linear2 = nn.Linear(config.n_hidden, config.n_features, bias=True)
        if config.init_bias_to_zero:
            self.linear2.bias.data.zero_()

        self.hidden_layers = None
        if config.n_hidden_layers > 0:
            self.hidden_layers = nn.ModuleList()
            for _ in range(config.n_hidden_layers):
                layer = nn.Linear(config.n_hidden, config.n_hidden, bias=False)
                self.hidden_layers.append(layer)

        if config.tied_weights:
            self.tie_weights_()

    def tie_weights_(self) -> None:
        self.linear2.weight.data = self.linear1.weight.data.T

    @override
    def to(self, *args: Any, **kwargs: Any) -> Self:
        self = super().to(*args, **kwargs)
        # Weights will become untied if moving device
        if self.config.tied_weights:
            self.tie_weights_()
        return self

    @override
    def forward(
        self, x: Float[Tensor, "... n_features"], **_: Any
    ) -> Float[Tensor, "... n_features"]:
        hidden = self.linear1(x)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                hidden = layer(hidden)
        out_pre_relu = self.linear2(hidden)
        out = F.relu(out_pre_relu)
        return out

    @classmethod
    @override
    def from_run_info(cls, run_info: RunInfo[TMSTrainConfig]) -> "TMSModel":
        """Load a pretrained model from a run info object."""
        tms_model = cls(config=run_info.config.tms_model_config)
        tms_model.load_state_dict(
            torch.load(run_info.checkpoint_path, weights_only=True, map_location="cpu")
        )
        tms_model.tie_weights_()
        return tms_model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "TMSModel":
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        run_info = TMSTargetRunInfo.from_path(path)
        return cls.from_run_info(run_info)
