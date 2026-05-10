"""Pure Residual MLP experiment spec."""

from param_decomp.experiments.driver import ExperimentSpec
from param_decomp.experiments.resid_mlp.configs import ResidMLPDataConfig, ResidMLPTargetConfig


class ResidMLPExperimentConfig(ExperimentSpec):
    kind: str = "resid_mlp"
    target: ResidMLPTargetConfig
    data: ResidMLPDataConfig
