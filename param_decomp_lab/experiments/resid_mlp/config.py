"""ResidualMLP experiment config schema — torch-free.

The JAX trainer reads this directly (`param_decomp.built_run`), the same way it reads
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

from param_decomp.base_config import BaseConfig, Probability
from param_decomp_lab.experiments.config import ExperimentConfig

ResidMLPDataGenerationType = Literal["exactly_one_active", "at_least_zero_active"]
ResidMLPActFn = Literal["gelu", "relu"]
ResidMLPLabelType = Literal["act_plus_resid", "abs"]
ResidMLPLossType = Literal["readoff", "resid"]


class ResidMLPPretrainConfig(BaseConfig):
    """How the frozen ResidualMLP target is pretrained from scratch.

    The objective is the MSE `mean(((pred − labels)²) · feature_importances)`:
    - `label_type` picks the read-off label (`act_plus_resid`: `act_fn(coeffs·x) + x`) or
      the `abs` label (`|coeffs·x|`).
    - `loss_type` picks `pred`: the model OUTPUT (`readoff`) or the pre-unembed RESIDUAL
      (`resid`, compared to the embedded labels).
    - `use_trivial_label_coeffs` ones-coeffs vs `U[1, 2)`.
    - `importance_val` geometrically down-weights feature `i` by `importance_val ** i`
      (`1.0` is uniform). Only valid in feature space, so requires `loss_type=readoff`."""

    steps: PositiveInt
    batch_size: PositiveInt
    lr: float
    seed: int = 0
    label_type: ResidMLPLabelType = "act_plus_resid"
    loss_type: ResidMLPLossType = "readoff"
    use_trivial_label_coeffs: bool = True
    importance_val: float = 1.0

    @model_validator(mode="after")
    def validate_importance_space(self) -> "ResidMLPPretrainConfig":
        assert self.loss_type == "readoff" or self.importance_val == 1.0, (
            "importance_val only applies in feature space; the resid loss compares in d_embed"
        )
        return self


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
    """`W_E = W_U = I` (requires `n_features == d_embed`); else a fixed random unit-norm
    embedding (torch `fixed_random_embedding`).

    TODO(lab): expose the full `embedding_mode` (`fixed_identity`/`fixed_random`/`learned`)
    and the pretrain `label_type`/`loss_type`/`use_trivial_label_coeffs`/`importance_val`
    knobs here once `experiments/resid_mlp/run.py` threads them into
    `ResidMLPTargetConfig` + `pretrain_resid_mlp_target` (both are off-limits in this
    change). The model.py layer already implements all of them with back-compatible
    defaults."""
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
