"""Tests for the pd-run launcher (param_decomp/scripts/run_slurm.py + runner.py)."""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from param_decomp.compose import resolve_run
from param_decomp.experiments.discovery import discover_experiments
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
    @patch("param_decomp.scripts.run_slurm._create_array_slurm_script")
    @patch("param_decomp.scripts.run_slurm.create_git_snapshot")
    def test_sweep_creates_one_task_per_combination(
        self,
        mock_create_git_snapshot,
        mock_create_array_slurm_script,
        mock_submit_slurm_job,
        mock_get_wandb_run_url,
        tmp_path: Path,
    ):
        from param_decomp.scripts.run_slurm import launch_sweep_slurm
        from param_decomp.utils.slurm import SubmitResult

        mock_create_git_snapshot.return_value = ("test-branch", "12345678")
        mock_create_array_slurm_script.return_value = "#!/bin/bash\necho test"
        mock_submit_slurm_job.return_value = SubmitResult(
            job_id="12345",
            script_path=tmp_path / "test.sh",
            log_pattern="~/slurm_logs/slurm-12345_*.out",
        )
        mock_get_wandb_run_url.return_value = "https://wandb.ai/test/test/runs/test"

        base_config, _ = resolve_run(_builtin("tms_5-2"))
        sweep_spec = cartesian_product(
            base_config=base_config,
            grid={"pd.seed": [0, 1, 2], "pd.steps": [10, 20]},
            n_agents=2,
            description="tiny test grid",
            driver_path=base_config.driver_path,
        )
        launch_sweep_slurm(
            sweep=sweep_spec,
            job_suffix=None,
            partition="cpu",
            project="test",
        )

        mock_create_array_slurm_script.assert_called_once()
        call_kwargs = mock_create_array_slurm_script.call_args.kwargs
        run_cfgs = call_kwargs["run_cfgs"]
        assert len(run_cfgs) == 6  # 3 seeds x 2 steps
        assert len({r.run_id for r in run_cfgs}) == 6

        submit_kwargs = mock_submit_slurm_job.call_args.kwargs
        assert submit_kwargs["n_array_tasks"] == 6

    @patch("param_decomp.scripts.run_slurm.get_wandb_run_url")
    @patch("param_decomp.scripts.run_slurm.submit_slurm_job")
    @patch("param_decomp.scripts.run_slurm._create_singleton_slurm_script")
    @patch("param_decomp.scripts.run_slurm.create_git_snapshot")
    def test_single_run_produces_one_task(
        self,
        mock_create_git_snapshot,
        mock_create_singleton_slurm_script,
        mock_submit_slurm_job,
        mock_get_wandb_run_url,
        tmp_path: Path,
    ):
        from param_decomp.scripts.run_slurm import launch_run_slurm
        from param_decomp.utils.slurm import SubmitResult

        mock_create_git_snapshot.return_value = ("test-branch", "12345678")
        mock_create_singleton_slurm_script.return_value = "#!/bin/bash\necho test"
        mock_submit_slurm_job.return_value = SubmitResult(
            job_id="12345",
            script_path=tmp_path / "test.sh",
            log_pattern="~/slurm_logs/slurm-12345.out",
        )
        mock_get_wandb_run_url.return_value = "https://wandb.ai/test/test/runs/test"

        run, _ = resolve_run(_builtin("tms_5-2"))
        launch_run_slurm(
            run_cfg=run,
            job_suffix=None,
            partition="cpu",
            project="test",
        )

        mock_create_singleton_slurm_script.assert_called_once()
        call_kwargs = mock_create_singleton_slurm_script.call_args.kwargs
        assert call_kwargs["run_cfg"].run_id == run.run_id

        submit_kwargs = mock_submit_slurm_job.call_args.kwargs
        assert submit_kwargs["n_array_tasks"] is None

    def test_worker_args_use_run_embedded_run_id(self):
        from param_decomp.scripts.run_slurm import _build_worker_args

        run, _ = resolve_run(_builtin("tms_5-2"))

        args = _build_worker_args("launch-test", run, "test")

        assert "--run_id" not in args
        assert run.run_id in args
