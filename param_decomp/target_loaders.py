"""Dispatch `PDTarget` construction by saved `ExperimentConfig` variant.

Used by post-processing tooling (app, harvest, autointerp) and by `ComponentModel`'s
auto-load path to reload the target without the user constructing a `PDTarget` themselves.

New users writing custom integrations should construct `PDTarget` directly and call
`load_pd(path, target=...)`.
"""

from param_decomp.experiment_config import ExperimentConfig
from param_decomp.experiments.ih.configs import IHExperimentConfig
from param_decomp.experiments.ih.target import load_ih_target
from param_decomp.experiments.lm.configs import LMExperimentConfig
from param_decomp.experiments.lm.target import load_lm_target
from param_decomp.experiments.resid_mlp.configs import ResidMLPExperimentConfig
from param_decomp.experiments.resid_mlp.target import load_resid_mlp_target
from param_decomp.experiments.tms.configs import TMSExperimentConfig
from param_decomp.experiments.tms.target import load_tms_target
from param_decomp.models.batch_and_loss_fns import PDTarget


def load_target_from_experiment_config(exp: ExperimentConfig) -> PDTarget:
    """Build a `PDTarget` for a saved run, dispatching on the experiment-config variant."""
    match exp:
        case LMExperimentConfig(target=t):
            return load_lm_target(t)
        case TMSExperimentConfig(target=t):
            target, _ = load_tms_target(t)
            return target
        case ResidMLPExperimentConfig(target=t):
            target, _ = load_resid_mlp_target(t)
            return target
        case IHExperimentConfig(target=t):
            target, _ = load_ih_target(t)
            return target
