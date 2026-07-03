"""TMS (Toy Model of Superposition) experiment config schema — torch-free.

The JAX trainer reads this directly (`param_decomp.built_run`), the same way it reads
`LMExperimentConfig`. Unlike the LM target there is no HuggingFace/pretrain-cache weight
source: the tiny TMS target is pretrained from scratch, deterministically from
`target.pretrain` (the original Anthropic `mean((|x| - relu_out)^2)` objective on the
synthetic sparse-feature data), so a run is fully reproducible from the config alone.
"""

from typing import Literal

from pydantic import NonNegativeInt, PositiveInt

from param_decomp.base_config import BaseConfig, Probability
from param_decomp_lab.experiments.config import ExperimentConfig

TMSDataGenerationType = Literal[
    "exactly_one_active",
    "exactly_two_active",
    "exactly_three_active",
    "exactly_four_active",
    "exactly_five_active",
    "at_least_zero_active",
]

TMSHiddenLayerInit = Literal["identity", "random"]


class TMSPretrainConfig(BaseConfig):
    """How the frozen TMS target is pretrained from scratch (Anthropic TMS objective)."""

    steps: PositiveInt
    batch_size: PositiveInt
    lr: float
    seed: int = 0


class TMSTargetConfig(BaseConfig):
    """The TMS target architecture + its from-scratch pretraining.

    The target has tied weights (`linear2 = linear1ᵀ`); the *decomposition* is untied
    (`pd.decomposition_targets = [linear1, linear2]` — each site gets its own components).

    `n_hidden_layers > 0` inserts that many FROZEN `n_hidden -> n_hidden` layers between
    `linear1` and `linear2` (`hidden_layer_init` = `identity` for the `-id` variant or
    `random` for frozen-random), each an extra dense decomposition site `hidden_layers.{i}`.
    `init_bias_to_zero` zeroes `linear2`'s bias at init (else `nn.Linear`'s default uniform
    bias)."""

    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt = 0
    hidden_layer_init: TMSHiddenLayerInit = "identity"
    init_bias_to_zero: bool = True
    pretrain: TMSPretrainConfig


class TMSDataConfig(BaseConfig):
    """Synthetic sparse-feature data for the PD decomposition step."""

    feature_probability: Probability
    data_generation_type: TMSDataGenerationType = "at_least_zero_active"


class TMSExperimentConfig(ExperimentConfig[TMSTargetConfig, TMSDataConfig]):
    pass
