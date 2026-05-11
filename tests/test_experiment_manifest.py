"""Round-trip tests for open-world experiment manifests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from param_decomp.configs import LayerwiseCiConfig, PDConfig, ScheduleConfig
from param_decomp.experiment_manifest import (
    EXPERIMENT_MANIFEST_FILENAME,
    parse_experiment_manifest,
    parse_manifest_experiment_config,
)
from param_decomp.experiments.driver import (
    ExperimentConfig,
    ExperimentManifest,
    load_driver,
)
from param_decomp.experiments.lm.data import LMDataConfig
from param_decomp.experiments.lm.experiment import (
    Driver as LMDriver,
)
from param_decomp.experiments.lm.experiment import (
    LMExperimentConfig,
    LMTargetConfig,
)
from param_decomp.experiments.resid_mlp.experiment import (
    Driver as ResidMLPDriver,
)
from param_decomp.experiments.resid_mlp.experiment import (
    ResidMLPDataConfig,
    ResidMLPExperimentConfig,
    ResidMLPTargetConfig,
)
from param_decomp.experiments.tms.experiment import (
    Driver as TMSDriver,
)
from param_decomp.experiments.tms.experiment import (
    TMSDataConfig,
    TMSExperimentConfig,
    TMSTargetConfig,
)

LM_DRIVER_PATH = "param_decomp.experiments.lm.experiment:Driver"
TMS_DRIVER_PATH = "param_decomp.experiments.tms.experiment:Driver"
RESID_MLP_DRIVER_PATH = "param_decomp.experiments.resid_mlp.experiment:Driver"


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


def _round_trip(experiment_config: ExperimentConfig, driver_path: str) -> ExperimentConfig:
    manifest = ExperimentManifest(
        kind=experiment_config.kind,
        driver=driver_path,
        experiment_config=experiment_config.model_dump(mode="json"),
    )
    parsed = parse_experiment_manifest(manifest.model_dump(mode="json"))
    assert parsed.kind == experiment_config.kind
    assert parsed.driver == driver_path
    return parse_manifest_experiment_config(parsed)


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
    parsed = _round_trip(exp, LM_DRIVER_PATH)
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
    parsed = _round_trip(exp, TMS_DRIVER_PATH)
    assert type(parsed) is TMSExperimentConfig
    assert parsed == exp


def test_resid_mlp_experiment_round_trip():
    exp = ResidMLPExperimentConfig(
        pd=_pd_config(),
        target=ResidMLPTargetConfig(run_path="wandb:foo/bar/runs/abc"),
        data=ResidMLPDataConfig(feature_probability=0.05),
    )
    parsed = _round_trip(exp, RESID_MLP_DRIVER_PATH)
    assert type(parsed) is ResidMLPExperimentConfig
    assert parsed == exp


def test_driver_class_paths_load():
    assert isinstance(load_driver(LM_DRIVER_PATH), LMDriver)
    assert isinstance(load_driver(TMS_DRIVER_PATH), TMSDriver)
    assert isinstance(load_driver(RESID_MLP_DRIVER_PATH), ResidMLPDriver)


def test_manual_manifest_does_not_need_registered_driver():
    manifest = ExperimentManifest.from_pd_config(_pd_config(), kind="custom")
    parsed = parse_experiment_manifest(manifest.model_dump(mode="json"))
    experiment_config = parse_manifest_experiment_config(parsed)
    assert experiment_config.kind == "custom"
    assert experiment_config.pd == _pd_config()


def test_save_pre_run_info_writes_experiment_manifest(tmp_path: Path):
    from param_decomp.utils.general_utils import save_pre_run_info

    manifest = ExperimentManifest.from_pd_config(_pd_config(), kind="custom")
    save_pre_run_info(
        save_to_wandb=False,
        out_dir=tmp_path,
        sweep_params=None,
        manifest=manifest,
        artifacts=(),
    )

    assert (tmp_path / EXPERIMENT_MANIFEST_FILENAME).exists()


def test_lm_target_requires_exactly_one_location():
    with pytest.raises(ValidationError):
        LMTargetConfig(
            model_class="transformers.GPT2LMHeadModel",
            model_name=None,
            model_path=None,
        )
