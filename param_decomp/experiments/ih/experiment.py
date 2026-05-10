"""Pure induction-head experiment spec."""

from param_decomp.experiments.driver import ExperimentSpec
from param_decomp.experiments.ih.configs import IHDataConfig, IHTargetConfig


class IHExperimentConfig(ExperimentSpec):
    kind: str = "ih"
    target: IHTargetConfig
    data: IHDataConfig
