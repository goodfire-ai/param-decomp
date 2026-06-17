"""TMS (Toy Model of Superposition) experiment config schema — torch-free.

The JAX trainer reads this directly (`jax_single_pool.config`), the same way it reads
`LMExperimentConfig`. Unlike the LM target there is no HuggingFace/pretrain-cache weight
source: the tiny TMS target is pretrained from scratch, deterministically from
`target.pretrain` (the original Anthropic `mean((|x| - relu_out)^2)` objective on the
synthetic sparse-feature data), so a run is fully reproducible from the config alone.
"""

from typing import Literal

from pydantic import NonNegativeInt, PositiveInt

from param_decomp_config.base import BaseConfig, Probability
from param_decomp_config.experiment import ExperimentConfig

TMSDataGenerationType = Literal["exactly_one_active", "at_least_zero_active"]


class TMSPretrainConfig(BaseConfig):
    """How the frozen TMS target is pretrained from scratch (Anthropic TMS objective)."""

    steps: PositiveInt
    batch_size: PositiveInt
    lr: float
    seed: int = 0


class TMSTargetConfig(BaseConfig):
    """The TMS target architecture + its from-scratch pretraining.

    The target has tied weights (`linear2 = linear1ᵀ`); the *decomposition* is untied
    (`pd.decomposition_targets = [linear1, linear2]`, `pd.tied_weights = null`)."""

    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt = 0
    pretrain: TMSPretrainConfig


class TMSDataConfig(BaseConfig):
    """Synthetic sparse-feature data for the PD decomposition step."""

    feature_probability: Probability
    data_generation_type: TMSDataGenerationType = "at_least_zero_active"


class TMSExperimentConfig(ExperimentConfig[TMSTargetConfig, TMSDataConfig]):
    pass
