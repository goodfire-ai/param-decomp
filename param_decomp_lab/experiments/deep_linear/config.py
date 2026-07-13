"""Deep-linear identity experiment config schema — the target is fully determined by
`(n_features, n_layers)` (identity weights, constructed not pretrained) and the data is
uniform one-hot rows (no knobs), so a run is reproducible from the config alone."""

from typing import Literal

from pydantic import PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp_lab.experiments.config import ExperimentConfig


class DeepLinearTargetConfig(BaseConfig):
    """`n_layers` frozen `eye(n_features)` sites named `layers.{i}`. `logit_scale`
    multiplies the final output only (softmax temperature for the KL recon; layers stay
    exact identities); `recon` picks the comparison (`kl` = LM semantics, `mse` = raw
    logits, no softmax saturation)."""

    n_features: PositiveInt
    n_layers: PositiveInt
    logit_scale: float = 1.0
    recon: Literal["kl", "mse"] = "kl"


class DeepLinearDataConfig(BaseConfig):
    """Uniform one-hot rows; nothing to configure."""


class DeepLinearExperimentConfig(ExperimentConfig[DeepLinearTargetConfig, DeepLinearDataConfig]):
    pass
