import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import Field, PositiveFloat, PositiveInt, model_validator
from torch import Tensor, nn

from param_decomp.base_config import BaseConfig
from param_decomp.components import init_param_
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_run_files


class ResidMLPModelConfig(BaseConfig):
    n_features: PositiveInt
    d_embed: PositiveInt
    d_mlp: PositiveInt
    n_layers: PositiveInt
    act_fn_name: Literal["gelu", "relu"] = Field(
        description="Defines the activation function in the model. Also used in the labeling "
        "function if label_type is act_plus_resid."
    )
    in_bias: bool
    out_bias: bool


class ResidMLPTrainConfig(BaseConfig):
    wandb_project: str | None = None
    seed: int = 0
    resid_mlp_model_config: ResidMLPModelConfig
    label_fn_seed: int = 0
    label_type: Literal["act_plus_resid", "abs"] = "act_plus_resid"
    loss_type: Literal["readoff", "resid"] = "readoff"
    use_trivial_label_coeffs: bool = False
    feature_probability: PositiveFloat
    synced_inputs: list[list[int]] | None = None
    importance_val: float | None = None
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    lr_schedule: ScheduleConfig
    fixed_random_embedding: bool = False
    fixed_identity_embedding: bool = False
    n_batches_final_losses: PositiveInt = 1

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        assert not (self.fixed_random_embedding and self.fixed_identity_embedding), (
            "Can't have both fixed_random_embedding and fixed_identity_embedding"
        )
        if self.fixed_identity_embedding:
            assert self.resid_mlp_model_config.n_features == self.resid_mlp_model_config.d_embed, (
                "n_features must equal d_embed if we are using an identity embedding matrix"
            )
        if self.synced_inputs is not None:
            all_indices = [item for sublist in self.synced_inputs for item in sublist]
            if len(all_indices) != len(set(all_indices)):
                raise ValueError("Synced inputs must be non-overlapping")
        return self


RESID_MLP_TRAIN_CONFIG_FILENAME = "resid_mlp_train_config.yaml"
RESID_MLP_CHECKPOINT_FILENAME = "resid_mlp.pth"
RESID_MLP_LABEL_COEFFS_FILENAME = "label_coeffs.json"


@dataclass
class ResidMLPTargetRunInfo:
    """Run info from training a ResidualMLPModel."""

    checkpoint_path: Path
    config: ResidMLPTrainConfig
    label_coeffs: Float[Tensor, " n_features"]

    @classmethod
    def from_path(cls, path: ModelPath) -> "ResidMLPTargetRunInfo":
        files = resolve_run_files(
            path,
            config_filename=RESID_MLP_TRAIN_CONFIG_FILENAME,
            checkpoint_filename=RESID_MLP_CHECKPOINT_FILENAME,
            extras_from_config_path=lambda _: [RESID_MLP_LABEL_COEFFS_FILENAME],
        )
        with open(files.extras[RESID_MLP_LABEL_COEFFS_FILENAME]) as f:
            label_coeffs = torch.tensor(json.load(f))
        return cls(
            checkpoint_path=files.checkpoint_path,
            config=ResidMLPTrainConfig.from_file(files.config_path),
            label_coeffs=label_coeffs,
        )


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        act_fn: Callable[[Tensor], Tensor],
        in_bias: bool,
        out_bias: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.act_fn = act_fn

        self.mlp_in = nn.Linear(d_model, d_mlp, bias=in_bias)
        self.mlp_out = nn.Linear(d_mlp, d_model, bias=out_bias)

    @override
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        mid_pre_act_fn = self.mlp_in(x)
        mid = self.act_fn(mid_pre_act_fn)
        out = self.mlp_out(mid)
        return out


class ResidMLP(nn.Module):
    def __init__(self, config: ResidMLPModelConfig):
        super().__init__()
        self.config = config
        self.W_E = nn.Parameter(torch.empty(config.n_features, config.d_embed))
        init_param_(self.W_E, fan_val=config.n_features, nonlinearity="linear")
        self.W_U = nn.Parameter(torch.empty(config.d_embed, config.n_features))
        init_param_(self.W_U, fan_val=config.d_embed, nonlinearity="linear")

        assert config.act_fn_name in ["gelu", "relu"]
        self.act_fn = F.gelu if config.act_fn_name == "gelu" else F.relu
        self.layers = nn.ModuleList(
            [
                MLP(
                    d_model=config.d_embed,
                    d_mlp=config.d_mlp,
                    act_fn=self.act_fn,
                    in_bias=config.in_bias,
                    out_bias=config.out_bias,
                )
                for _ in range(config.n_layers)
            ]
        )

    @override
    def forward(
        self,
        x: Float[Tensor, "... n_features"],
        return_residual: bool = False,
    ) -> Float[Tensor, "... n_features"] | Float[Tensor, "... d_embed"]:
        residual = einops.einsum(x, self.W_E, "... n_features, n_features d_embed -> ... d_embed")
        for layer in self.layers:
            out = layer(residual)
            residual = residual + out
        if return_residual:
            return residual
        out = einops.einsum(
            residual,
            self.W_U,
            "... d_embed, d_embed n_features -> ... n_features",
        )
        return out

    @classmethod
    def from_run_info(cls, run_info: ResidMLPTargetRunInfo) -> "ResidMLP":
        """Load a pretrained model from a run info object."""
        resid_mlp_model = cls(config=run_info.config.resid_mlp_model_config)
        resid_mlp_model.load_state_dict(
            torch.load(run_info.checkpoint_path, weights_only=True, map_location="cpu")
        )
        return resid_mlp_model

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> "ResidMLP":
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        run_info = ResidMLPTargetRunInfo.from_path(path)
        return cls.from_run_info(run_info)
