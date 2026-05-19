"""Auto-discovery of built-in experiments from YAML configs.

An experiment is any `<name>.yaml` file living next to an `experiment.py` under
`param_decomp/experiments/<kind>/`. The experiment name is the YAML filename
stem with a trailing `_config` stripped. The YAML itself declares its
`driver_path` as a top-level field.
"""

from dataclasses import dataclass
from pathlib import Path

from param_decomp.settings import REPO_ROOT

EXPERIMENTS_DIR = REPO_ROOT / "param_decomp" / "experiments"


@dataclass(frozen=True)
class DiscoveredExperiment:
    name: str
    kind: str
    config_path: Path


def discover_experiments() -> dict[str, DiscoveredExperiment]:
    experiments: dict[str, DiscoveredExperiment] = {}
    for kind_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not kind_dir.is_dir() or not (kind_dir / "experiment.py").exists():
            continue
        kind = kind_dir.name
        for yaml_path in sorted(kind_dir.glob("[!_]*.yaml")):
            name = yaml_path.stem.removesuffix("_config")
            assert name not in experiments, (
                f"Duplicate experiment name {name!r}: {experiments[name].config_path} vs {yaml_path}"
            )
            experiments[name] = DiscoveredExperiment(
                name=name,
                kind=kind,
                config_path=yaml_path.relative_to(REPO_ROOT),
            )
    return experiments
