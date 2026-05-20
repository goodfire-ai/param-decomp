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
from param_decomp.metrics.registry import METRIC_REGISTRY

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_MODULE = "custom_metric_dotted"
FIXTURE_METRIC_NAME = "DottedFixtureLoss"


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
        metric_modules=metric_modules,
        loss_metrics=loss_metrics,
    )


def test_metric_module_dotted_name_registers_metric():
    sys.path.insert(0, str(FIXTURES_DIR))
    try:
        cfg = PDConfig.model_validate(
            _pd_config_kwargs(
                metric_modules=[FIXTURE_MODULE],
                loss_metrics={FIXTURE_METRIC_NAME: {"coeff": 2.0}},
            )
        )
    finally:
        sys.path.remove(str(FIXTURES_DIR))
    assert FIXTURE_METRIC_NAME in METRIC_REGISTRY
    assert cfg.loss_metrics[FIXTURE_METRIC_NAME].coeff == 2.0


def test_metric_module_dotted_name_is_idempotent():
    """Validating the same metric_modules entry twice must not trip the duplicate-registration
    assert in `register_metric`."""
    kwargs = _pd_config_kwargs(
        metric_modules=[FIXTURE_MODULE],
        loss_metrics={FIXTURE_METRIC_NAME: {"coeff": 1.0}},
    )
    sys.path.insert(0, str(FIXTURES_DIR))
    try:
        PDConfig.model_validate(kwargs)
        PDConfig.model_validate(kwargs)
    finally:
        sys.path.remove(str(FIXTURES_DIR))


def test_unknown_metric_name_rejected_when_not_imported():
    """Sanity: without a `metric_modules` entry, an unknown metric name fails fast."""
    with pytest.raises(ValidationError, match="unknown metric"):
        PDConfig.model_validate(
            _pd_config_kwargs(
                metric_modules=[],
                loss_metrics={"this_metric_does_not_exist": {"coeff": 1.0}},
            )
        )
