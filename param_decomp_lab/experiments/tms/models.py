from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, override

import torch
from jaxtyping import Float
from pydantic import NonNegativeInt, PositiveInt, model_validator
from torch import Tensor, nn
from torch.nn import functional as F

from param_decomp.base_config import BaseConfig
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_run_files


class TMSModelConfig(BaseConfig):
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt
    tied_weights: bool
    init_bias_to_zero: bool
    device: str


class TMSTrainConfig(BaseConfig):
    wandb_project: str | None = None
    tms_model_config: TMSModelConfig
    feature_probability: float
    batch_size: PositiveInt
    steps: PositiveInt
    seed: int = 0
    lr_schedule: ScheduleConfig
    data_generation_type: Literal["at_least_zero_active", "exactly_one_active"]
    fixed_identity_hidden_layers: bool = False
    fixed_random_hidden_layers: bool = False
    synced_inputs: list[list[int]] | None = None

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.fixed_identity_hidden_layers and self.fixed_random_hidden_layers:
            raise ValueError(
                "Cannot set both fixed_identity_hidden_layers and fixed_random_hidden_layers to True"
            )
        if self.synced_inputs is not None:
            all_indices = [item for sublist in self.synced_inputs for item in sublist]
            if len(all_indices) != len(set(all_indices)):
                raise ValueError("Synced inputs must be non-overlapping")
        return self


TMS_TRAIN_CONFIG_FILENAME = "tms_train_config.yaml"
TMS_CHECKPOINT_FILENAME = "tms.pth"


@dataclass
class TMSTargetRunInfo:
    """Run info from training a TMSModel."""

    checkpoint_path: Path
    config: TMSTrainConfig

    @classmethod
    def from_path(cls, path: ModelPath) -> "TMSTargetRunInfo":
        files = resolve_run_files(
            path,
            config_filename=TMS_TRAIN_CONFIG_FILENAME,
            checkpoint_filename=TMS_CHECKPOINT_FILENAME,
        )
        return cls(
            checkpoint_path=files.checkpoint_path,
            config=TMSTrainConfig.from_file(files.config_path),
        )


class TMSModel(nn.Module):
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
    def from_run_info(cls, run_info: TMSTargetRunInfo) -> "TMSModel":
        """Load a pretrained model from a run info object."""
        tms_model = cls(config=run_info.config.tms_model_config)
        tms_model.load_state_dict(
            torch.load(run_info.checkpoint_path, weights_only=True, map_location="cpu")
        )
        tms_model.tie_weights_()
        return tms_model

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> "TMSModel":
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        run_info = TMSTargetRunInfo.from_path(path)
        return cls.from_run_info(run_info)
