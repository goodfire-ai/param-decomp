"""In-repo experiment registry.

`EXPERIMENTS[name]` returns the experiment module — a plain Python module exposing
`TargetConfig`, `DataConfig`, `build_target`, `build_train_loader`, `build_eval_loader`,
and `make_run_batch`. `SavedRun` dispatches by name to rebuild the target model and
dataloaders from a saved run's ``run_meta.yaml``.

To register a new in-repo experiment, define a `run.py` next to a YAML config and
add it here.
"""

from types import ModuleType

from param_decomp_lab.experiments.lm import run as lm
from param_decomp_lab.experiments.resid_mlp import run as resid_mlp
from param_decomp_lab.experiments.tms import run as tms

EXPERIMENTS: dict[str, ModuleType] = {
    "tms": tms,
    "resid_mlp": resid_mlp,
    "lm": lm,
}
