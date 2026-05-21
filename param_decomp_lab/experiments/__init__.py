"""In-repo experiment registry.

`EXPERIMENTS[name]` returns the `ExperimentSpec` for a registered experiment. Lab-side tools
(`SavedRun`, harvest, autointerp, app) dispatch by name to rebuild the target model and
dataloaders from a saved run's ``run_meta.yaml``.

To register a new experiment, define an `ExperimentSpec` in your package and add it to
`EXPERIMENTS` (here, for in-repo experiments) or pass it directly to the lab tool that
consumes it.
"""

from param_decomp_lab.experiments.lm.run import EXPERIMENT_SPEC as _LM_SPEC
from param_decomp_lab.experiments.resid_mlp.run import EXPERIMENT_SPEC as _RESID_MLP_SPEC
from param_decomp_lab.experiments.spec import ExperimentSpec
from param_decomp_lab.experiments.tms.run import EXPERIMENT_SPEC as _TMS_SPEC

EXPERIMENTS: dict[str, ExperimentSpec] = {
    _TMS_SPEC.name: _TMS_SPEC,
    _RESID_MLP_SPEC.name: _RESID_MLP_SPEC,
    _LM_SPEC.name: _LM_SPEC,
}

__all__ = ["EXPERIMENTS", "ExperimentSpec"]
