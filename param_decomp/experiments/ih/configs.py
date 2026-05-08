from typing import Literal

from pydantic import Field, PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp.configs import PDConfig


class InductionModelConfig(BaseConfig):
    vocab_size: PositiveInt
    seq_len: PositiveInt
    d_model: PositiveInt
    n_heads: PositiveInt
    n_layers: PositiveInt
    ff_fanout: PositiveInt
    use_ff: bool
    use_pos_encoding: bool
    use_layer_norm: bool
    device: str = "cpu"


class InductionHeadsTrainConfig(BaseConfig):
    wandb_project: str | None = None
    ih_model_config: InductionModelConfig
    steps: PositiveInt
    batch_size: PositiveInt
    lr: float
    lr_warmup: int | float
    weight_decay: float
    lr_schedule: Literal["cosine", "constant", "linear"] = "linear"
    seed: int = 0
    attention_maps_n_steps: PositiveInt
    prefix_window: PositiveInt


class IHTargetConfig(BaseConfig):
    """Path to the trained induction head target run."""

    run_path: str = Field(..., description="Local or wandb path to an IH pretrain run.")


class IHDataConfig(BaseConfig):
    """Synthetic induction-pattern dataset settings."""

    prefix_window: PositiveInt | None = Field(
        default=None,
        description=(
            "Number of tokens to use as a prefix window for the induction head. If None, "
            "uses the full sequence length minus 3."
        ),
    )


class IHExperimentConfig(BaseConfig):
    kind: Literal["ih"] = "ih"
    pd: PDConfig
    target: IHTargetConfig
    data: IHDataConfig
