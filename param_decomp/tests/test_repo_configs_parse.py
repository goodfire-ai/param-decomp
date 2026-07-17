"""Every canonical run yaml in `param_decomp/configs/` must parse under the tip schema.

`test_config.py` converts a handful of NAMED configs; this covers the whole seat registry,
which is what a schema change actually risks. A schema PR that reshapes a field has to
migrate these yamls in the same commit (an unmigrated seat is a run nobody can launch, and
a config that no longer parses is a run nobody can re-open — see the muon-branch runs that
are unloadable at tip today).

Deliberately NOT covering `param_decomp_lab/experiments/lm/*.yaml`: the jose / ss-2L /
pile-4L archetype seats there are known-broken against the tip schema (torch-era
`ci_config` shapes, missing `run_name`) and are owned by their own migration task. Pointing
this test at them would make it red for reasons unrelated to whatever change it's guarding.
"""

from pathlib import Path

import pytest
import yaml

from param_decomp_lab.experiments.lm.config import LMExperimentConfig

CONFIGS = Path(__file__).parent.parent / "configs"
CONFIG_FILES = sorted(CONFIGS.glob("*.yaml"))


def test_configs_dir_is_not_empty():
    """A glob that silently matches nothing would make every test below vacuously pass."""
    assert len(CONFIG_FILES) > 20, [p.name for p in CONFIG_FILES]


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.name)
def test_repo_config_parses(config_path: Path):
    LMExperimentConfig.model_validate(yaml.safe_load(config_path.read_text()))
