"""Pure TMS experiment spec."""

from param_decomp.experiments.driver import ExperimentSpec
from param_decomp.experiments.tms.configs import TMSDataConfig, TMSTargetConfig


class TMSExperimentConfig(ExperimentSpec):
    kind: str = "tms"
    target: TMSTargetConfig
    data: TMSDataConfig
