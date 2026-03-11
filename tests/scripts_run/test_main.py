"""Tests for the main() function in spd/scripts/run.py.

This file contains tests for spd-run, which always submits jobs to SLURM.
For local execution tests, see tests/scripts_simple/.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false

from unittest.mock import patch

import pytest

from spd.scripts.run import _create_training_jobs, _get_experiments


class TestSPDRun:
    """Test spd-run command execution."""

    def test_invalid_experiment_name(self):
        """Test that invalid experiment names raise an error."""
        fake_exp_name = "nonexistent_experiment_please_dont_name_your_experiment_this"
        with pytest.raises(ValueError, match=f"Invalid experiments.*{fake_exp_name}"):
            _get_experiments(fake_exp_name)

        with pytest.raises(ValueError, match=f"Invalid experiments.*{fake_exp_name}"):
            _get_experiments(f"{fake_exp_name},tms_5-2")

    @patch("spd.scripts.run.submit_slurm_job")
    @patch("spd.scripts.run.create_slurm_script")
    @patch("spd.scripts.run.create_git_snapshot")
    @patch("spd.scripts.run._create_wandb_views_and_report")
    def test_sweep_creates_slurm_array(
        self,
        mock_create_wandb_views_and_report,
        mock_create_git_snapshot,
        mock_create_slurm_script,
        mock_submit_slurm_job,
    ):
        """Test that sweep runs create SLURM array jobs with sweep params."""
        from pathlib import Path

        from spd.scripts.run_cli import main
        from spd.utils.slurm import SubmitResult

        mock_create_git_snapshot.return_value = ("test-branch", "12345678")
        mock_create_slurm_script.return_value = "#!/bin/bash\necho test"
        mock_submit_slurm_job.return_value = SubmitResult(
            job_id="12345",
            script_path=Path("/tmp/test.sh"),
            log_pattern="~/slurm_logs/slurm-12345_*.out",
        )

        main(
            experiments="tms_5-2",
            sweep="sweep_params.yaml.example",
            n_agents=2,
        )

        # Verify SLURM array script was created
        mock_create_slurm_script.assert_called_once()

        # Verify the run has sweep params and multiple jobs
        call_kwargs = mock_create_slurm_script.call_args.kwargs
        training_jobs = call_kwargs["training_jobs"]
        sweep_params = call_kwargs["sweep_params"]
        assert len(training_jobs) > 1  # Sweep should create multiple jobs
        assert sweep_params is not None

    def test_create_training_jobs_sweep(self):
        """when given sweep params, _create_training_jobs should generate the correct number of
        jobs with params swept correctly"""

        sweep_params = {
            "global": {"lr_schedule": {"start_val": {"values": [1, 2]}}},
            "tms_5-2": {
                "steps": {"values": [100, 200]},
                "module_info": {
                    "values": [
                        [
                            {"module_pattern": "linear1", "C": 10},
                            {"module_pattern": "linear2", "C": 10},
                        ],
                        [
                            {"module_pattern": "linear1", "C": 20},
                            {"module_pattern": "linear2", "C": 20},
                        ],
                    ]
                },
            },
        }

        training_jobs = _create_training_jobs(
            experiments=["tms_5-2"],
            project="test",
            sweep_params=sweep_params,
        )

        configs = [j.config for j in training_jobs]

        def there_is_one_with(start_val: int, steps: int, c: int) -> bool:
            matching = [
                cfg
                for cfg in configs
                if cfg.lr_schedule.start_val == start_val
                and cfg.steps == steps
                and c == cfg.module_info[0].C
                and c == cfg.module_info[1].C
            ]
            return len(matching) == 1

        # 2 start_val * 2 steps * 2 module_info = 8 jobs
        assert len(configs) == 8

        assert there_is_one_with(start_val=1, steps=100, c=10)
        assert there_is_one_with(start_val=1, steps=100, c=20)
        assert there_is_one_with(start_val=1, steps=200, c=10)
        assert there_is_one_with(start_val=1, steps=200, c=20)
        assert there_is_one_with(start_val=2, steps=100, c=10)
        assert there_is_one_with(start_val=2, steps=100, c=20)
        assert there_is_one_with(start_val=2, steps=200, c=10)
        assert there_is_one_with(start_val=2, steps=200, c=20)

    def test_create_training_jobs_sweep_multi_experiment(self):
        """when given sweep params, _create_training_jobs should generate the correct number of
        jobs with params swept correctly across multiple experiments"""

        sweep_params = {
            "tms_5-2": {
                "module_info": {
                    "values": [
                        [
                            {"module_pattern": "linear1", "C": 10},
                            {"module_pattern": "linear2", "C": 10},
                        ],
                    ]
                },
            },
            "tms_40-10": {"steps": {"values": [100, 200]}},
        }

        training_jobs = _create_training_jobs(
            experiments=["tms_5-2", "tms_40-10"],
            project="test",
            sweep_params=sweep_params,
        )

        configs = [j.config for j in training_jobs]

        def there_is_one_with(c: int | None = None, steps: int | None = None) -> bool:
            matching = []
            for cfg in configs:
                match = True
                if c is not None and c != cfg.module_info[0].C:
                    match = False
                if steps is not None and cfg.steps != steps:
                    match = False
                if match:
                    matching.append(cfg)
            return len(matching) == 1

        assert len(configs) == 3

        assert there_is_one_with(c=10)
        assert there_is_one_with(steps=100)
        assert there_is_one_with(steps=200)
