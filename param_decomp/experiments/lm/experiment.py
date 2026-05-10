"""Pure LM experiment spec."""

from param_decomp.experiments.driver import ExperimentSpec
from param_decomp.experiments.lm.configs import LMDataConfig, LMTargetConfig


class LMExperimentConfig(ExperimentSpec):
    kind: str = "lm"
    target: LMTargetConfig
    data: LMDataConfig
