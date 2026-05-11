"""Round-trip tests for open-world experiment manifests."""

import pytest
from pydantic import ValidationError

from param_decomp.configs import LayerwiseCiConfig, PDConfig, ScheduleConfig
from param_decomp.experiment_config import parse_driver_spec, parse_experiment_config
from param_decomp.experiments.driver import ExperimentManifest, ExperimentSpec
from param_decomp.experiments.lm.data import LMDataConfig
from param_decomp.experiments.lm.experiment import (
    LMDriver,
    LMExperimentConfig,
    LMTargetConfig,
)
from param_decomp.experiments.resid_mlp.experiment import (
    ResidMLPDataConfig,
    ResidMLPDriver,
    ResidMLPExperimentConfig,
    ResidMLPTargetConfig,
)
from param_decomp.experiments.tms.experiment import (
    TMSDataConfig,
    TMSDriver,
    TMSExperimentConfig,
    TMSTargetConfig,
)


def _pd_config() -> PDConfig:
    """A minimal PDConfig sufficient for round-trip tests."""
    return PDConfig(
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        sigmoid_type="leaky_hard",
        module_info=[],
        loss_metric_configs=[],
        lr_schedule=ScheduleConfig(start_val=1e-3),
        steps=1,
        batch_size=4,
        eval_batch_size=4,
        train_log_freq=1,
        eval_freq=1,
        slow_eval_freq=1,
        n_eval_steps=1,
        slow_eval_on_first_step=False,
        ci_alive_threshold=0.0,
    )


def _round_trip(exp: ExperimentSpec, driver_path: str) -> ExperimentSpec:
    manifest = ExperimentManifest.from_spec(exp, driver=driver_path)
    parsed = parse_experiment_config(manifest.model_dump(mode="json"))
    assert parsed.kind == exp.kind
    assert parsed.driver == driver_path
    return parse_driver_spec(parsed)


def test_lm_experiment_round_trip():
    exp = LMExperimentConfig(
        pd=_pd_config(),
        target=LMTargetConfig(
            model_class="transformers.GPT2LMHeadModel",
            model_name="openai-community/gpt2",
            output_extract="logits",
        ),
        data=LMDataConfig(
            dataset_name="SimpleStories/SimpleStories",
            tokenizer_name="gpt2",
            column_name="story",
            max_seq_len=128,
        ),
    )
    lm_driver = LMDriver()
    parsed = _round_trip(exp, LMDriver.driver_path)
    assert type(parsed) is LMExperimentConfig
    assert parsed == exp
    assert "GPT2LMHeadModel" in lm_driver.display_name(exp)
    assert "SimpleStories" in lm_driver.display_name(exp)


def test_tms_experiment_round_trip():
    exp = TMSExperimentConfig(
        pd=_pd_config(),
        target=TMSTargetConfig(run_path="wandb:foo/bar/runs/abc"),
        data=TMSDataConfig(feature_probability=0.05),
    )
    parsed = _round_trip(exp, TMSDriver.driver_path)
    assert type(parsed) is TMSExperimentConfig
    assert parsed == exp


def test_resid_mlp_experiment_round_trip():
    exp = ResidMLPExperimentConfig(
        pd=_pd_config(),
        target=ResidMLPTargetConfig(run_path="wandb:foo/bar/runs/abc"),
        data=ResidMLPDataConfig(feature_probability=0.05),
    )
    parsed = _round_trip(exp, ResidMLPDriver.driver_path)
    assert type(parsed) is ResidMLPExperimentConfig
    assert parsed == exp


def test_manual_manifest_does_not_need_registered_driver():
    manifest = ExperimentManifest.from_pd_config(_pd_config(), kind="custom")
    parsed = parse_experiment_config(manifest.model_dump(mode="json"))
    spec = parse_driver_spec(parsed)
    assert spec.kind == "custom"
    assert spec.pd == _pd_config()


def test_lm_target_requires_exactly_one_location():
    with pytest.raises(ValidationError):
        LMTargetConfig(
            model_class="transformers.GPT2LMHeadModel",
            model_name=None,
            model_path=None,
        )
