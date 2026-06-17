"""ResidualMLP experiment config schema — torch-free.

The JAX trainer reads this directly (`jax_single_pool.config`), the same way it reads
`TMSExperimentConfig`. Like TMS there is no HuggingFace/pretrain-cache weight source:
the toy ResidualMLP target is pretrained from scratch, deterministically from
`target.pretrain` (the `act_fn(coeffs·x) + x` read-off objective on the synthetic
sparse-feature data), so a run is fully reproducible from the config alone.

The target is the SPD/APD residual-stream toy: a fixed input embedding `W_E`
(`n_features → d_embed`), a stack of `n_layers` MLP blocks each reading from and writing
to the `d_embed` residual stream, and a fixed unembed `W_U` (`d_embed → n_features`). The
*decomposition* targets the per-layer MLP matrices (`layers.{i}.mlp_in` /
`layers.{i}.mlp_out`).
"""

from typing import Literal

from pydantic import PositiveInt, model_validator

from param_decomp_config.base import BaseConfig, Probability
from param_decomp_config.experiment import ExperimentConfig

ResidMLPDataGenerationType = Literal["exactly_one_active", "at_least_zero_active"]
ResidMLPActFn = Literal["gelu", "relu"]


class ResidMLPPretrainConfig(BaseConfig):
    """How the frozen ResidualMLP target is pretrained from scratch.

    The objective is the read-off MSE `mean((out − (act_fn(coeffs·x) + x))²)` with the
    embedding held fixed (`fixed_random_embedding`: unit-norm random rows, `W_U = W_Eᵀ`)
    and trivial unit label coeffs — the canonical clean-recovery regime."""

    steps: PositiveInt
    batch_size: PositiveInt
    lr: float
    seed: int = 0


class ResidMLPTargetConfig(BaseConfig):
    """The ResidualMLP target architecture + its from-scratch pretraining.

    `d_mlp ≥ n_features` (no MLP-width superposition) and an identity/random fixed
    embedding give the clean per-feature ground truth the identity-CI metric checks."""

    n_features: PositiveInt
    d_embed: PositiveInt
    d_mlp: PositiveInt
    n_layers: PositiveInt
    act_fn_name: ResidMLPActFn
    in_bias: bool = False
    out_bias: bool = False
    fixed_identity_embedding: bool = False
    """`W_E = I` (requires `n_features == d_embed`); else a fixed random unit-norm
    embedding (torch `fixed_random_embedding`)."""
    pretrain: ResidMLPPretrainConfig

    @model_validator(mode="after")
    def validate_identity_embedding(self) -> "ResidMLPTargetConfig":
        if self.fixed_identity_embedding:
            assert self.n_features == self.d_embed, (
                "n_features must equal d_embed for fixed_identity_embedding"
            )
        return self


class ResidMLPDataConfig(BaseConfig):
    """Synthetic sparse-feature data for the PD decomposition step (values in [-1, 1])."""

    feature_probability: Probability
    data_generation_type: ResidMLPDataGenerationType = "at_least_zero_active"


class ResidMLPExperimentConfig(ExperimentConfig[ResidMLPTargetConfig, ResidMLPDataConfig]):
    pass
