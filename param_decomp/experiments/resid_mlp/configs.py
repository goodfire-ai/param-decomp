from typing import Any, Literal, Self

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.configs import ScheduleConfig, migrate_to_lr_schedule_config


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
    wandb_project: str | None = None  # The name of the wandb project (if None, don't log to wandb)
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

    @model_validator(mode="before")
    @classmethod
    def handle_deprecated_config_keys(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        migrate_to_lr_schedule_config(config_dict)
        return config_dict

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
            # Ensure that the synced_inputs are non-overlapping with eachother
            all_indices = [item for sublist in self.synced_inputs for item in sublist]
            if len(all_indices) != len(set(all_indices)):
                raise ValueError("Synced inputs must be non-overlapping")
        return self
