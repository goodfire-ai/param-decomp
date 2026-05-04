import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from param_decomp.experiments.resid_mlp.configs import (
    ResidMLPModelConfig,
    ResidMLPTrainConfig,
)
from param_decomp.interfaces import LoadableModule, RunInfo
from param_decomp.param_decomp_types import ModelPath
from param_decomp.utils.module_utils import init_param_


@dataclass
class ResidMLPTargetRunInfo(RunInfo[ResidMLPTrainConfig]):
    """Run info from training a ResidualMLPModel."""

    label_coeffs: Float[Tensor, " n_features"]

    config_class = ResidMLPTrainConfig
    config_filename = "resid_mlp_train_config.yaml"
    checkpoint_filename = "resid_mlp.pth"
    extra_files = ["label_coeffs.json"]

    @classmethod
    @override
    def _process_extra_files(cls, file_paths: dict[str, Path], init_kwargs: dict[str, Any]) -> None:
        with open(file_paths["label_coeffs.json"]) as f:
            init_kwargs["label_coeffs"] = torch.tensor(json.load(f))


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


class ResidMLP(LoadableModule):
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
    @override
    def from_run_info(cls, run_info: RunInfo[ResidMLPTrainConfig]) -> "ResidMLP":
        """Load a pretrained model from a run info object."""
        resid_mlp_model = cls(config=run_info.config.resid_mlp_model_config)
        resid_mlp_model.load_state_dict(
            torch.load(run_info.checkpoint_path, weights_only=True, map_location="cpu")
        )
        return resid_mlp_model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "ResidMLP":
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        run_info = ResidMLPTargetRunInfo.from_path(path)
        return cls.from_run_info(run_info)
