"""Every LM config yaml the repo maintains must parse at tip (CONFIGS.md rule 1).

A schema PR that breaks a config here migrates it in the same PR, with an
executed in-repo migration — never a script attached to a PR comment (#939
attached one; it never ran, and 97/104 stored runs became unopenable).
"""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from param_decomp_lab.experiments.config import assert_canonical_algorithm_config
from param_decomp_lab.experiments.lm.config import (
    LMExperimentConfig,
    PretrainedTarget,
    assert_placement_claims,
)

REPO = Path(__file__).resolve().parents[2]

# Archetype seats predating the modern schema, awaiting migration (see
# CONFIGS.md registry). When you fix one, remove it here — the gate then
# covers it; this list must only ever shrink.
KNOWN_BROKEN = {
    "param_decomp_lab/experiments/lm/jose.yaml",
    "param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L.yaml",
    "param_decomp_lab/experiments/lm/ss_llama_simple_mlp-2L.yaml",
}

CONFIG_PATHS = sorted(
    path
    for pattern in ("param_decomp/configs/**/*.yaml", "param_decomp_lab/experiments/lm/*.yaml")
    for path in REPO.glob(pattern)
)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: str(p.relative_to(REPO)))
def test_config_parses_and_is_canonical(path: Path) -> None:
    rel = str(path.relative_to(REPO))
    if rel in KNOWN_BROKEN:
        with pytest.raises(ValidationError):
            LMExperimentConfig.model_validate(_load(path))
        pytest.skip(f"{rel}: known-broken seat awaiting migration (CONFIGS.md)")
    cfg = LMExperimentConfig.model_validate(_load(path))
    assert_canonical_algorithm_config(cfg)
    # The placement gate resolves the site set from config + arch, so every maintained
    # config's sharding claim is exercised at its pinned dp (the pre-sbatch check). A
    # `kind: pretrained` target is the enumerated gap: its resolution reads the pretrain
    # cache from the cluster FS, which this gate cannot assume.
    if not isinstance(cfg.target.spec, PretrainedTarget):
        assert_placement_claims(cfg)


def test_known_broken_entries_still_exist() -> None:
    missing = [rel for rel in KNOWN_BROKEN if not (REPO / rel).exists()]
    assert not missing, f"KNOWN_BROKEN lists deleted files, prune it: {missing}"
