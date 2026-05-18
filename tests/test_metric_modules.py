"""Tests for `PDConfig.metric_modules` external metric registration."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from param_decomp.configs import (
    LayerwiseCiConfig,
    OptimizerConfig,
    PDConfig,
    ScheduleConfig,
)
from param_decomp.metrics.registry import METRIC_REGISTRY, import_metric_module

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FILE_FIXTURE = FIXTURES_DIR / "custom_metric_file.py"
DOTTED_FIXTURE_MODULE = "custom_metric_dotted"

FILE_FIXTURE_METRIC_NAME = "_test_metric_modules_file_loss"
DOTTED_FIXTURE_METRIC_NAME = "_test_metric_modules_dotted_loss"


def _pd_config_kwargs(
    *,
    metric_modules: list[str],
    loss_metrics: dict[str, dict[str, float]],
) -> dict[str, object]:
    return dict(
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        sigmoid_type="leaky_hard",
        module_info=[],
        components_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        ci_fn_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        steps=1,
        batch_size=4,
        eval_batch_size=4,
        train_log_freq=1,
        eval_freq=1,
        slow_eval_freq=1,
        n_eval_steps=1,
        slow_eval_on_first_step=False,
        ci_alive_threshold=0.0,
        metric_modules=metric_modules,
        loss_metrics=loss_metrics,
    )


def test_metric_module_file_path_registers_metric():
    assert FILE_FIXTURE.exists()
    cfg = PDConfig.model_validate(
        _pd_config_kwargs(
            metric_modules=[str(FILE_FIXTURE.resolve())],
            loss_metrics={FILE_FIXTURE_METRIC_NAME: {"coeff": 1.0}},
        )
    )
    assert FILE_FIXTURE_METRIC_NAME in METRIC_REGISTRY
    assert FILE_FIXTURE_METRIC_NAME in cfg.loss_metrics
    assert cfg.loss_metrics[FILE_FIXTURE_METRIC_NAME].coeff == 1.0


def test_metric_module_file_path_is_idempotent():
    """Validating the same metric_modules entry twice must not trip the duplicate-registration
    assert in `register_metric`."""
    kwargs = _pd_config_kwargs(
        metric_modules=[str(FILE_FIXTURE.resolve())],
        loss_metrics={FILE_FIXTURE_METRIC_NAME: {"coeff": 1.0}},
    )
    PDConfig.model_validate(kwargs)
    PDConfig.model_validate(kwargs)


def test_metric_module_dotted_name_registers_metric():
    sys.path.insert(0, str(FIXTURES_DIR))
    try:
        cfg = PDConfig.model_validate(
            _pd_config_kwargs(
                metric_modules=[DOTTED_FIXTURE_MODULE],
                loss_metrics={DOTTED_FIXTURE_METRIC_NAME: {"coeff": 2.0}},
            )
        )
    finally:
        sys.path.remove(str(FIXTURES_DIR))
    assert DOTTED_FIXTURE_METRIC_NAME in METRIC_REGISTRY
    assert cfg.loss_metrics[DOTTED_FIXTURE_METRIC_NAME].coeff == 2.0


def test_metric_module_relative_path_rejected():
    with pytest.raises(AssertionError, match="must be absolute"):
        import_metric_module("relative/path/to/metric.py")


def test_metric_module_missing_path_rejected():
    with pytest.raises(AssertionError, match="does not exist"):
        import_metric_module("/nonexistent/path/to/metric.py")


def test_unknown_metric_slug_rejected_when_not_imported():
    """Sanity: without a `metric_modules` entry, an unknown slug fails fast."""
    with pytest.raises(ValidationError, match="unknown metric"):
        PDConfig.model_validate(
            _pd_config_kwargs(
                metric_modules=[],
                loss_metrics={"this_metric_does_not_exist": {"coeff": 1.0}},
            )
        )
