"""Composition-root smoke: run every `*_SMOKE.yaml` end-to-end on CPU against a tmpdir.

Discovery is glob-based (`experiments/*/configs/*_SMOKE.yaml`), so a new experiment family
gets coverage by shipping a SMOKE config — no test edit. Each config runs through its
family's real `run.py::main` (the `pd-<family>` entry minus fire): YAML schema → BuiltRun →
pretrain → faith warmup → engine step loop → checkpoint → in-loop eval. This is the
automated version of the manual `pd-tms <SMOKE>` full-loop validation, so
`run.py` ↔ core-engine signature drift fails here instead of on a hand-launched run.

The SMOKE step counts are tiny, so this asserts wiring (completion, eval keys, finite
metrics, checkpoint on disk) — the recovers-identity convergence gate is the slow
`test_end_to_end_pretrain_decompose_recovers_identity` in `experiments/tms/test_tms.py`.
"""

import importlib
import json
import math
from pathlib import Path

import pytest

import param_decomp_lab.experiments
from param_decomp_lab.experiments import config as experiments_config

EXPERIMENTS_DIR = Path(param_decomp_lab.experiments.__file__).parent
SMOKE_CONFIGS = sorted(EXPERIMENTS_DIR.glob("*/configs/*_SMOKE.yaml"))

# Family -> the eval-key prefix its `eval_fn` must land in metrics.jsonl. A family absent
# here only needs *some* `eval/` key (discovery still covers it); add an entry to pin the
# family's ground-truth metric.
EVAL_KEY_PREFIX_BY_FAMILY = {
    "tms": "eval/identity_ci_error/",
    "resid_mlp": "eval/identity_ci_error/",
}


def test_smoke_configs_discovered():
    assert {p.stem for p in SMOKE_CONFIGS} >= {"tms_5-5_SMOKE", "resid_mlp_1l_SMOKE"}


@pytest.mark.parametrize("smoke_yaml", SMOKE_CONFIGS, ids=lambda p: p.stem)
def test_smoke_config_runs_end_to_end(
    smoke_yaml: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    family = smoke_yaml.parent.parent.name
    run_module = importlib.import_module(f"param_decomp_lab.experiments.{family}.run")
    # `run_instance` resolves the run dir off this module global (set from the env at
    # import time), so patching it redirects the whole run — ckpts, jsonl, launch config.
    monkeypatch.setattr(experiments_config, "PARAM_DECOMP_OUT_DIR", tmp_path)

    run_module.main(str(smoke_yaml))

    (run_dir,) = (tmp_path / "runs").iterdir()
    assert (run_dir / "launch_config.yaml").exists()
    assert any((run_dir / "ckpts").iterdir()), "no checkpoint written"

    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert records, "no metrics logged"
    for record in records:
        for key, value in record.items():
            assert isinstance(value, int | float), f"non-scalar {key} in metrics.jsonl"
            assert math.isfinite(value), f"non-finite {key}={value} at step {record['step']}"

    eval_keys = {k for r in records for k in r if k.startswith("eval/")}
    prefix = EVAL_KEY_PREFIX_BY_FAMILY.get(family, "eval/")
    assert any(k.startswith(prefix) for k in eval_keys), (
        f"{family}: no {prefix}* key in metrics.jsonl (eval keys: {sorted(eval_keys)})"
    )
