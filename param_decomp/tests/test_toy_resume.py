"""The toy CLIs expose the LM trainer's `--run-id` convention: a rerun with the same id
reuses the run dir and resumes from its checkpoints instead of minting a fresh identity.
Mid-training that is the requeue path; at the configured horizon there is nothing left to
run, and the rerun must say so instead of exiting 0 having done nothing.

Runs the real module entry in a child process (as `test_run_inline` does), so each rerun
starts with fresh process state just as a requeued job does.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

_CONFIG = Path(__file__).parents[1] / "experiments" / "tms" / "configs" / "tms_5-2.yaml"
_RUN_ID = "p-0000abcd"


def _tiny_tms_config(path: Path, name: str, seed: int) -> Path:
    raw = yaml.safe_load(_CONFIG.read_text())
    raw["pd"]["seed"] = seed
    raw["pd"]["steps"] = 2
    raw["pd"]["batch_size"] = 8
    raw["pd"]["faithfulness_warmup_steps"] = 0
    raw["target"]["pretrain"]["steps"] = 2
    raw["target"]["pretrain"]["batch_size"] = 8
    # Both steps checkpointed and kept, so the interrupted-mid-training state (latest
    # checkpoint below the horizon) is one directory deletion away.
    raw["cadence"] = {
        "train_log_every": 1,
        "checkpointing": {
            "kind": "periodic",
            "save_every": 1,
            "retention": {"kind": "keep_last", "n": 2},
        },
    }
    raw["eval"] = None
    raw["wandb"] = None
    config = path / name
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    return config


def _invoke(config: Path, data_root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "param_decomp.experiments.tms.run",
            str(config),
            "--run-id",
            _RUN_ID,
            "--data-root",
            str(data_root),
        ],
        capture_output=True,
        text=True,
        env=env,
    )


@dataclass(frozen=True)
class TrainedRun:
    config: Path
    root: Path
    data_root: Path
    env: dict[str, str]
    stdout: str

    def run_dir(self, data_root: Path) -> Path:
        return data_root / "runs" / _RUN_ID

    @property
    def metrics_jsonl(self) -> Path:
        return self.run_dir(self.data_root) / "metrics.jsonl"


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> TrainedRun:
    """One completed 2-step run, shared by the rerun tests (each only reruns, on its own
    copy of the data root where it needs to mutate one)."""
    root = tmp_path_factory.mktemp("tms_resume")
    config = _tiny_tms_config(root, "tiny_tms.yaml", seed=0)
    data_root = root / "data"
    env = os.environ | {
        "JAX_PLATFORMS": "cpu",
        # Shared across the child processes so only the first invocation pays the compile.
        "JAX_COMPILATION_CACHE_DIR": str(root / "jax_cache"),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1",
    }
    result = _invoke(config, data_root, env)
    assert result.returncode == 0, result.stderr
    return TrainedRun(config, root, data_root, env, result.stdout)


def test_tms_first_run_trains_every_step(trained: TrainedRun) -> None:
    assert "resumed from checkpoint" not in trained.stdout
    assert "checkpoint saved @ step 2" in trained.stdout
    assert trained.metrics_jsonl.read_text().count("\n") == 2


def test_tms_rerun_at_the_configured_horizon_refuses(trained: TrainedRun) -> None:
    rerun = _invoke(trained.config, trained.data_root, trained.env)
    assert rerun.returncode != 0
    assert "already trained to step 2 of pd.steps=2" in rerun.stderr
    # The refusal costs nothing: no step ran, so no record was appended.
    assert trained.metrics_jsonl.read_text().count("\n") == 2


def test_tms_rerun_below_the_horizon_resumes(trained: TrainedRun, tmp_path: Path) -> None:
    """The requeue path: a preempted run's latest checkpoint is mid-training, and rerunning
    the same id must train the remaining steps."""
    data_root = tmp_path / "data"
    shutil.copytree(trained.data_root, data_root)
    run_dir = trained.run_dir(data_root)
    shutil.rmtree(run_dir / "ckpts" / "2")
    records_before = (run_dir / "metrics.jsonl").read_text().count("\n")

    rerun = _invoke(trained.config, data_root, trained.env)
    assert rerun.returncode == 0, rerun.stderr
    assert "resumed from checkpoint step 1" in rerun.stdout
    assert "checkpoint saved @ step 2" in rerun.stdout
    assert (run_dir / "metrics.jsonl").read_text().count("\n") == records_before + 1


def test_tms_rerun_refuses_a_changed_config(trained: TrainedRun) -> None:
    tampered = _tiny_tms_config(trained.root, "tampered_tms.yaml", seed=1)
    rerun = _invoke(tampered, trained.data_root, trained.env)
    assert rerun.returncode != 0
    assert "refusing to resume with a changed config" in rerun.stderr
