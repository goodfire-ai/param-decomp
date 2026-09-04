from itertools import product
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from param_decomp.core.schedule import ScheduleConfig, get_scheduled_value
from param_decomp.experiments.lm import config as lm_config
from param_decomp.migrations.schedule_knots import (
    contains_retired_schedule,
    migrate_raw,
    migrate_schedule,
    retired_value,
)


def _steps(total_steps: int, warmup_pct: float) -> list[int]:
    if total_steps <= 250:
        return list(range(total_steps))
    warmup = int(total_steps * warmup_pct)
    return sorted(
        {
            0,
            1,
            max(0, warmup - 1),
            min(total_steps - 1, warmup),
            min(total_steps - 1, warmup + 1),
            total_steps // 2,
            total_steps - 1,
        }
    )


NONCONSTANT_CASES = list(
    product(
        (1, 2, 3, 10, 250, 400_000),
        (0.0, 0.025, 0.5, 1.0),
        (0.0, 0.1, 1.0, 2.0),
        ("linear", "cosine"),
    )
)


@pytest.mark.parametrize(
    ("total_steps", "warmup_pct", "final_val_frac", "fn_type"), NONCONSTANT_CASES
)
def test_migration_preserves_every_sampled_trained_step(
    total_steps: int,
    warmup_pct: float,
    final_val_frac: float,
    fn_type: str,
) -> None:
    retired = {
        "start_val": 0.01,
        "warmup_pct": warmup_pct,
        "final_val_frac": final_val_frac,
        "fn_type": fn_type,
    }
    knots = ScheduleConfig.model_validate(migrate_schedule(retired, total_steps))
    for step in _steps(total_steps, warmup_pct):
        assert get_scheduled_value(step, total_steps, knots) == pytest.approx(
            retired_value(step, total_steps, retired), rel=1e-13, abs=1e-15
        )


@pytest.mark.parametrize("total_steps", [1, 2, 3, 10, 250, 400_000])
@pytest.mark.parametrize("warmup_pct", [0.0, 0.025, 0.5, 1.0])
def test_constant_migration_preserves_defaults_and_full_warmup(
    total_steps: int, warmup_pct: float
) -> None:
    retired: dict[str, Any] = {"start_val": "1e-4"}
    if warmup_pct:
        retired["warmup_pct"] = warmup_pct
    knots = ScheduleConfig.model_validate(migrate_schedule(retired, total_steps))
    for step in _steps(total_steps, warmup_pct):
        assert get_scheduled_value(step, total_steps, knots) == pytest.approx(
            retired_value(step, total_steps, retired), rel=1e-13, abs=1e-15
        )


def test_recursive_config_migration_is_complete_and_idempotent() -> None:
    retired: dict[str, Any] = {
        "run_name": "old",
        "pd": {
            "steps": 100,
            "components_optimizer": {
                "lr_schedule": {
                    "start_val": 1e-3,
                    "warmup_pct": 0.1,
                    "final_val_frac": 0.1,
                    "fn_type": "cosine",
                }
            },
            "loss_metrics": [
                {
                    "type": "ImportanceMinimalityLoss",
                    "gamma": {
                        "start_val": 1.0,
                        "warmup_pct": 0.0,
                        "final_val_frac": 0.01,
                        "fn_type": "linear",
                    },
                }
            ],
        },
    }
    assert contains_retired_schedule(retired)
    migrated = migrate_raw(retired)
    assert not contains_retired_schedule(migrated)
    assert migrate_raw(migrated) == migrated
    assert retired["pd"]["components_optimizer"]["lr_schedule"]["start_val"] == 1e-3


def test_current_schema_still_refuses_retired_authored_schedule() -> None:
    with pytest.raises(ValidationError):
        ScheduleConfig.model_validate({"start_val": 1e-3})


def test_lm_stored_config_boundary_migrates_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "launch_config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "pd": {
                    "steps": 10,
                    "components_optimizer": {
                        "lr_schedule": {
                            "start_val": 1e-3,
                            "warmup_pct": 0.0,
                            "final_val_frac": 0.1,
                            "fn_type": "cosine",
                        }
                    },
                }
            }
        )
    )
    observed: list[dict[str, Any]] = []

    def capture(raw: dict[str, Any], run_id: str, data_root: Path) -> tuple[Any, Any]:
        assert run_id == "p-old" and data_root == tmp_path
        observed.append(raw)
        return object(), object()

    monkeypatch.setattr(lm_config, "build_from_schema", capture)
    lm_config.load_config(path, "p-old", tmp_path)
    assert len(observed) == 1 and not contains_retired_schedule(observed[0])
