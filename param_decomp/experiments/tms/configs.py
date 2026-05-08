from typing import Any, Literal, Self

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.configs import PDConfig, ScheduleConfig, migrate_to_lr_schedule_config
from param_decomp.param_decomp_types import Probability


class TMSModelConfig(BaseConfig):
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt
    tied_weights: bool
    init_bias_to_zero: bool
    device: str


class TMSTrainConfig(BaseConfig):
    wandb_project: str | None = None  # The name of the wandb project (if None, don't log to wandb)
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

    @model_validator(mode="before")
    @classmethod
    def handle_deprecated_config_keys(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        migrate_to_lr_schedule_config(config_dict)
        return config_dict

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.fixed_identity_hidden_layers and self.fixed_random_hidden_layers:
            raise ValueError(
                "Cannot set both fixed_identity_hidden_layers and fixed_random_hidden_layers to True"
            )
        if self.synced_inputs is not None:
            # Ensure that the synced_inputs are non-overlapping with eachother
            all_indices = [item for sublist in self.synced_inputs for item in sublist]
            if len(all_indices) != len(set(all_indices)):
                raise ValueError("Synced inputs must be non-overlapping")
        return self


class TMSTargetConfig(BaseConfig):
    """Path to the trained TMS target run."""

    run_path: str = Field(
        ...,
        description="Local or wandb path to a TMS pretrain run (used by TMSTargetRunInfo).",
    )


class TMSDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for TMS PD."""

    feature_probability: Probability
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = (
        "at_least_zero_active"
    )


class TMSExperimentConfig(BaseConfig):
    kind: Literal["tms"] = "tms"
    pd: PDConfig
    target: TMSTargetConfig
    data: TMSDataConfig
