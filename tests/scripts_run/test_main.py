"""Tests for the pd-run launcher (param_decomp/scripts/run_slurm.py + runner.py)."""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from param_decomp.configs import RuntimeConfig
from param_decomp.experiments.discovery import discover_experiments
from param_decomp.run import RunConfig
from param_decomp.settings import REPO_ROOT
from param_decomp.sweeps.cartesian import cartesian_product


def _builtin(name: str) -> dict[str, Any]:
    discovered = discover_experiments()
    with open(REPO_ROOT / discovered[name].config_path) as f:
        return yaml.safe_load(f)


class TestResolveSource:
    def test_unknown_experiment_rejected(self):
        from param_decomp.experiments.runner import _resolve_source

        fake = "nonexistent_experiment_please_dont_name_your_experiment_this"
        with pytest.raises(AssertionError, match=f"Unknown experiment '{fake}'"):
            _resolve_source(experiment=fake, config_path=None, rerun=None)


class TestLaunchSlurm:
    @patch("param_decomp.scripts.run_slurm.get_wandb_run_url")
    @patch("param_decomp.scripts.run_slurm.submit_slurm_job")
    @patch("param_decomp.scripts.run_slurm._create_slurm_script")
    @patch("param_decomp.scripts.run_slurm.create_git_snapshot")
    def test_sweep_creates_one_task_per_combination(
        self,
        mock_create_git_snapshot,
        mock_create_slurm_script,
        mock_submit_slurm_job,
        mock_get_wandb_run_url,
        tmp_path: Path,
    ):
        from param_decomp.scripts.run_slurm import launch_slurm
        from param_decomp.utils.slurm import SubmitResult

        mock_create_git_snapshot.return_value = ("test-branch", "12345678")
        mock_create_slurm_script.return_value = "#!/bin/bash\necho test"
        mock_submit_slurm_job.return_value = SubmitResult(
            job_id="12345",
            script_path=tmp_path / "test.sh",
            log_pattern="~/slurm_logs/slurm-12345_*.out",
        )
        mock_get_wandb_run_url.return_value = "https://wandb.ai/test/test/runs/test"

        base_config = _builtin("tms_5-2")
        sweep_spec = cartesian_product(
            base_config=base_config,
            grid={"pd.seed": [0, 1, 2], "pd.steps": [10, 20]},
            description="tiny test grid",
            driver_path=base_config["driver_path"],
        )
        launch_slurm(
            launchable=sweep_spec,
            n_agents=2,
            job_suffix=None,
            runtime=RuntimeConfig(),
            partition="cpu",
            project="test",
        )

        mock_create_slurm_script.assert_called_once()
        call_kwargs = mock_create_slurm_script.call_args.kwargs
        runs = call_kwargs["runs"]
        assert len(runs) == 6  # 3 seeds x 2 steps
        assert len({run.run_id for run in runs}) == 6

    @patch("param_decomp.scripts.run_slurm.get_wandb_run_url")
    @patch("param_decomp.scripts.run_slurm.submit_slurm_job")
    @patch("param_decomp.scripts.run_slurm._create_slurm_script")
    @patch("param_decomp.scripts.run_slurm.create_git_snapshot")
    def test_single_run_produces_one_task(
        self,
        mock_create_git_snapshot,
        mock_create_slurm_script,
        mock_submit_slurm_job,
        mock_get_wandb_run_url,
        tmp_path: Path,
    ):
        from param_decomp.scripts.run_slurm import launch_slurm
        from param_decomp.utils.slurm import SubmitResult

        mock_create_git_snapshot.return_value = ("test-branch", "12345678")
        mock_create_slurm_script.return_value = "#!/bin/bash\necho test"
        mock_submit_slurm_job.return_value = SubmitResult(
            job_id="12345",
            script_path=tmp_path / "test.sh",
            log_pattern="~/slurm_logs/slurm-12345.out",
        )
        mock_get_wandb_run_url.return_value = "https://wandb.ai/test/test/runs/test"

        base_config = _builtin("tms_5-2")
        logging_data = {**base_config.get("logging", {}), "wandb_run_name": "tms_5-2"}
        run = RunConfig.from_dict({**base_config, "logging": logging_data})
        launch_slurm(
            launchable=run,
            n_agents=None,
            job_suffix=None,
            runtime=RuntimeConfig(),
            partition="cpu",
            project="test",
        )

        mock_create_slurm_script.assert_called_once()
        call_kwargs = mock_create_slurm_script.call_args.kwargs
        runs = call_kwargs["runs"]
        assert len(runs) == 1
        assert runs[0].run_id == run.run_id
        assert call_kwargs["is_array"] is False

    def test_worker_args_use_run_embedded_run_id(self):
        from param_decomp.scripts.run_slurm import _build_worker_args

        base_config = _builtin("tms_5-2")
        run = RunConfig.from_dict(base_config)

        args = _build_worker_args("launch-test", run, "test")

        assert "--run_id" not in args
        assert run.run_id in args
