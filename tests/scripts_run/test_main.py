"""Tests for param_decomp/scripts/run_slurm.py (the SLURM launcher under pd-run)."""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false

from unittest.mock import patch

import pytest

from param_decomp.experiments.discovery import discover_experiments
from param_decomp.scripts.run_slurm import _create_run_specs


def _builtin(name: str) -> tuple[str, dict[str, object]]:
    discovered = discover_experiments()
    exp = discovered[name]
    import yaml

    from param_decomp.settings import REPO_ROOT

    with open(REPO_ROOT / exp.config_path) as f:
        return exp.driver_path, yaml.safe_load(f)


class TestLaunchSlurm:
    def test_unknown_experiment_rejected(self):
        from param_decomp.experiments.runner import _resolve_source

        fake = "nonexistent_experiment_please_dont_name_your_experiment_this"
        with pytest.raises(AssertionError, match=f"Unknown experiment '{fake}'"):
            _resolve_source(experiment=fake, config_path=None, driver=None, rerun=None)

    @patch("param_decomp.scripts.run_slurm.get_wandb_run_url")
    @patch("param_decomp.scripts.run_slurm.submit_slurm_job")
    @patch("param_decomp.scripts.run_slurm.create_slurm_script")
    @patch("param_decomp.scripts.run_slurm.create_git_snapshot")
    def test_sweep_creates_slurm_array(
        self,
        mock_create_git_snapshot,
        mock_create_slurm_script,
        mock_submit_slurm_job,
        mock_get_wandb_run_url,
    ):
        """Test that sweep runs create SLURM array jobs with sweep params."""
        from pathlib import Path

        from param_decomp.scripts.run_slurm import launch_slurm
        from param_decomp.utils.slurm import SubmitResult

        mock_create_git_snapshot.return_value = ("test-branch", "12345678")
        mock_create_slurm_script.return_value = "#!/bin/bash\necho test"
        mock_submit_slurm_job.return_value = SubmitResult(
            job_id="12345",
            script_path=Path("/tmp/test.sh"),
            log_pattern="~/slurm_logs/slurm-12345_*.out",
        )
        mock_get_wandb_run_url.return_value = "https://wandb.ai/test/test/runs/test"

        driver_path, base_config = _builtin("tms_5-2")
        launch_slurm(
            name="tms_5-2",
            driver_path=driver_path,
            base_config=base_config,
            sweep="sweep_params.yaml.example",
            n_agents=2,
            job_suffix=None,
            cpu=False,
            partition="cpu",
            dp=None,
            project="test",
        )

        mock_create_slurm_script.assert_called_once()
        call_kwargs = mock_create_slurm_script.call_args.kwargs
        run_specs = call_kwargs["run_specs"]
        sweep_params = call_kwargs["sweep_params"]
        assert len(run_specs) > 1
        assert sweep_params is not None

    def test_create_run_specs_sweep(self):
        """With sweep params, _create_run_specs should expand the grid."""
        sweep_params = {
            "global": {"pd": {"lr_schedule": {"start_val": {"values": [1, 2]}}}},
            "tms_5-2": {
                "pd": {
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
            },
        }

        driver_path, base_config = _builtin("tms_5-2")
        run_specs = _create_run_specs(
            name="tms_5-2",
            driver_path=driver_path,
            base_config=base_config,
            project="test",
            sweep_params=sweep_params,
        )

        configs = [j.config_dict["pd"] for j in run_specs]

        def there_is_one_with(start_val: int, steps: int, c: int) -> bool:
            matching = [
                cfg
                for cfg in configs
                if cfg["lr_schedule"]["start_val"] == start_val
                and cfg["steps"] == steps
                and c == cfg["module_info"][0]["C"]
                and c == cfg["module_info"][1]["C"]
            ]
            return len(matching) == 1

        assert len(configs) == 8
        assert there_is_one_with(start_val=1, steps=100, c=10)
        assert there_is_one_with(start_val=1, steps=100, c=20)
        assert there_is_one_with(start_val=1, steps=200, c=10)
        assert there_is_one_with(start_val=1, steps=200, c=20)
        assert there_is_one_with(start_val=2, steps=100, c=10)
        assert there_is_one_with(start_val=2, steps=100, c=20)
        assert there_is_one_with(start_val=2, steps=200, c=10)
        assert there_is_one_with(start_val=2, steps=200, c=20)
