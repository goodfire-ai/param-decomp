"""Round-trip tests for run metadata and experiment configs."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from param_decomp.configs import LayerwiseCiConfig, OptimizerConfig, PDConfig, ScheduleConfig
from param_decomp.experiments.driver import (
    ExperimentConfig,
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
from param_decomp.run_metadata import RUN_METADATA_FILENAME, RunMetadata

LM_DRIVER_PATH = "param_decomp.experiments.lm.experiment:Driver"
TMS_DRIVER_PATH = "param_decomp.experiments.tms.experiment:Driver"
RESID_MLP_DRIVER_PATH = "param_decomp.experiments.resid_mlp.experiment:Driver"


def _pd_config() -> PDConfig:
    return PDConfig(
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
    )


def _round_trip(experiment_config: ExperimentConfig, driver_path: str) -> ExperimentConfig:
    """Build RunMetadata for `experiment_config` and re-parse it through the driver."""
    driver = load_driver(driver_path)
    metadata = RunMetadata(
        driver=driver_path,
        config=experiment_config.model_dump(mode="json"),
    )
    assert metadata.driver == driver_path
    return driver.config_type.model_validate(metadata.config)


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
    parsed = _round_trip(exp, LM_DRIVER_PATH)
    assert type(parsed) is LMExperimentConfig
    assert parsed == exp


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


def test_save_pre_run_info_writes_run_metadata(tmp_path: Path):
    from param_decomp.utils.general_utils import save_pre_run_info

    metadata = RunMetadata(
        driver=None,
        config={"pd": _pd_config().model_dump(mode="json")},
    )
    save_pre_run_info(
        save_to_wandb=False,
        out_dir=tmp_path,
        metadata=metadata,
        artifacts={},
    )

    assert (tmp_path / RUN_METADATA_FILENAME).exists()


def test_run_metadata_round_trip_via_file(tmp_path: Path):
    metadata = RunMetadata(
        driver="param_decomp.experiments.lm.experiment:Driver",
        config={"pd": {"seed": 42}, "target": {}, "data": {}},
    )
    path = tmp_path / RUN_METADATA_FILENAME
    metadata.write(path)
    loaded = RunMetadata.from_file(path)
    assert loaded.driver == metadata.driver
    assert loaded.config == metadata.config


def test_lm_target_requires_exactly_one_location():
    with pytest.raises(ValidationError):
        LMTargetConfig(
            model_class="transformers.GPT2LMHeadModel",
            model_name=None,
            model_path=None,
        )
