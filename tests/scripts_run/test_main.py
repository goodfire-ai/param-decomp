"""Tests for the pd-run launcher (param_decomp/scripts/run_slurm.py + runner.py)."""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from param_decomp.experiments.discovery import discover_experiments
from param_decomp.settings import REPO_ROOT


def _builtin(name: str) -> tuple[str, dict[str, object]]:
    discovered = discover_experiments()
    exp = discovered[name]
    with open(REPO_ROOT / exp.config_path) as f:
        return exp.driver_path, yaml.safe_load(f)


class TestResolveSource:
    def test_unknown_experiment_rejected(self):
        from param_decomp.experiments.runner import _resolve_source

        fake = "nonexistent_experiment_please_dont_name_your_experiment_this"
        with pytest.raises(AssertionError, match=f"Unknown experiment '{fake}'"):
            _resolve_source(experiment=fake, config_path=None, driver=None, rerun=None)


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

        grid_path = tmp_path / "grid.yaml"
        grid_path.write_text(
            yaml.dump(
                {
                    "description": "tiny test grid",
                    "grid": {
                        "pd.seed": [0, 1, 2],
                        "pd.steps": [10, 20],
                    },
                }
            )
        )

        driver_path, base_config = _builtin("tms_5-2")
        launch_slurm(
            name="tms_5-2",
            driver_path=driver_path,
            base_config=base_config,
            sweep=str(grid_path),
            n_agents=2,
            job_suffix=None,
            cpu=False,
            partition="cpu",
            dp=None,
            project="test",
        )

        mock_create_slurm_script.assert_called_once()
        call_kwargs = mock_create_slurm_script.call_args.kwargs
        task_specs = call_kwargs["task_specs"]
        assert len(task_specs) == 6  # 3 seeds x 2 steps

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

        driver_path, base_config = _builtin("tms_5-2")
        launch_slurm(
            name="tms_5-2",
            driver_path=driver_path,
            base_config=base_config,
            sweep=None,
            n_agents=None,
            job_suffix=None,
            cpu=False,
            partition="cpu",
            dp=None,
            project="test",
        )

        mock_create_slurm_script.assert_called_once()
        task_specs = mock_create_slurm_script.call_args.kwargs["task_specs"]
        assert len(task_specs) == 1
