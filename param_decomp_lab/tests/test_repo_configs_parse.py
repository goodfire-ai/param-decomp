"""Every config yaml the repo maintains must parse at tip (CONFIGS.md rule 1) — the
LM seats against `LMExperimentConfig`, the toy seats against their domain schemas.

A schema PR that breaks a config here migrates it in the same PR, with an
executed in-repo migration — never a script attached to a PR comment (a migration
that lives outside the repo never runs; see CONFIGS.md's migration registry).
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
from param_decomp_lab.experiments.resid_mlp.config import ResidMLPExperimentConfig
from param_decomp_lab.experiments.tms.config import TMSExperimentConfig

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


def test_gate_collects_the_seat_registry() -> None:
    """Anti-vacuity: a broken glob (moved roots, renamed dirs) collects zero files and
    every per-config test silently vanishes green. Pin the collected set to the
    CONFIGS.md seat policy: non-empty, and within the 10-LM-config registry cap."""
    assert CONFIG_PATHS, "config glob collected nothing — did the config roots move?"
    assert len(CONFIG_PATHS) <= 10, (
        f"{len(CONFIG_PATHS)} LM configs exceed the CONFIGS.md registry cap of 10 — "
        "adding a seat requires an eviction (or this is an uncommitted one-off, see rule 2)"
    )


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


TOY_SCHEMA_BY_DIR = {
    "param_decomp_lab/experiments/tms/configs": TMSExperimentConfig,
    "param_decomp_lab/experiments/resid_mlp/configs": ResidMLPExperimentConfig,
}

TOY_CONFIG_PATHS = sorted(
    path for rel_dir in TOY_SCHEMA_BY_DIR for path in REPO.glob(f"{rel_dir}/*.yaml")
)


@pytest.mark.parametrize("path", TOY_CONFIG_PATHS, ids=lambda p: str(p.relative_to(REPO)))
def test_toy_config_parses_and_is_canonical(path: Path) -> None:
    schema = TOY_SCHEMA_BY_DIR[str(path.parent.relative_to(REPO))]
    cfg = schema.model_validate(_load(path))
    assert_canonical_algorithm_config(cfg)


def test_toy_gate_collects_both_domains() -> None:
    """Anti-vacuity: a moved configs dir would silently drop its whole domain from the gate."""
    collected_dirs = {str(p.parent.relative_to(REPO)) for p in TOY_CONFIG_PATHS}
    assert collected_dirs == set(TOY_SCHEMA_BY_DIR), (
        f"toy config glob collected {collected_dirs or 'nothing'} — did a configs dir move?"
    )
