"""Round-trip tests for the `RunConfig` object."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from param_decomp.configs import (
    AttnConfig,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    LayerwiseCiConfig,
    LoggingConfig,
    OptimizerConfig,
    PDConfig,
    RuntimeConfig,
    ScheduleConfig,
)
from param_decomp.experiments.driver import load_driver
from param_decomp.experiments.lm.data import LMDataConfig
from param_decomp.experiments.lm.experiment import (
    Driver as LMDriver,
)
from param_decomp.experiments.lm.experiment import (
    LMTargetConfig,
)
from param_decomp.experiments.resid_mlp.experiment import (
    Driver as ResidMLPDriver,
)
from param_decomp.experiments.resid_mlp.experiment import (
    ResidMLPDataConfig,
    ResidMLPTargetConfig,
)
from param_decomp.experiments.tms.experiment import (
    Driver as TMSDriver,
)
from param_decomp.experiments.tms.experiment import (
    TMSDataConfig,
    TMSTargetConfig,
)
from param_decomp.run import RUN_CONFIG_FILENAME, RunConfig

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
    )


def _logging_config() -> LoggingConfig:
    return LoggingConfig(
        eval_batch_size=4,
        train_log_freq=1,
        eval_freq=1,
        slow_eval_freq=1,
        n_eval_steps=1,
        slow_eval_on_first_step=False,
    )


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(autocast_bf16=False, device="cpu", dp=None)


def _round_trip(run: RunConfig) -> RunConfig:
    """Round-trip ``run`` through ``model_dump`` → ``RunConfig.model_validate``."""
    return RunConfig.model_validate(run.model_dump(mode="json"))


def test_run_generates_run_id_on_instantiation():
    run = RunConfig(
        driver_path=TMS_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
        target={},
        data={},
    )

    assert run.run_id.startswith("p-")


def test_lm_run_round_trip():
    run = RunConfig(
        driver_path=LM_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
        target=LMTargetConfig(
            model_class="transformers.GPT2LMHeadModel",
            model_name="openai-community/gpt2",
            output_extract="logits",
        ).model_dump(mode="json"),
        data=LMDataConfig(
            dataset_name="SimpleStories/SimpleStories",
            tokenizer_name="gpt2",
            column_name="story",
            max_seq_len=128,
        ).model_dump(mode="json"),
    )
    parsed = _round_trip(run)
    assert parsed == run
    LMDriver().validate_config(parsed)


def test_tms_run_round_trip():
    run = RunConfig(
        driver_path=TMS_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
        target=TMSTargetConfig(run_path="wandb:foo/bar/runs/abc").model_dump(mode="json"),
        data=TMSDataConfig(feature_probability=0.05).model_dump(mode="json"),
    )
    parsed = _round_trip(run)
    assert parsed == run
    TMSDriver().validate_config(parsed)


def test_resid_mlp_run_round_trip():
    run = RunConfig(
        driver_path=RESID_MLP_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
        target=ResidMLPTargetConfig(run_path="wandb:foo/bar/runs/abc").model_dump(mode="json"),
        data=ResidMLPDataConfig(feature_probability=0.05).model_dump(mode="json"),
    )
    parsed = _round_trip(run)
    assert parsed == run
    ResidMLPDriver().validate_config(parsed)


def test_run_requires_runtime_config():
    data = {
        "driver_path": TMS_DRIVER_PATH,
        "pd": _pd_config().model_dump(mode="json"),
        "logging": _logging_config().model_dump(mode="json"),
        "target": TMSTargetConfig(run_path="wandb:foo/bar/runs/abc").model_dump(mode="json"),
        "data": TMSDataConfig(feature_probability=0.05).model_dump(mode="json"),
    }

    with pytest.raises(ValidationError, match="runtime"):
        RunConfig.model_validate(data)


def test_driver_class_paths_load():
    assert isinstance(load_driver(LM_DRIVER_PATH), LMDriver)
    assert isinstance(load_driver(TMS_DRIVER_PATH), TMSDriver)
    assert isinstance(load_driver(RESID_MLP_DRIVER_PATH), ResidMLPDriver)


def test_run_round_trip_via_file(tmp_path: Path):
    run = RunConfig(
        driver_path=TMS_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
        target=TMSTargetConfig(run_path="wandb:foo/bar/runs/abc").model_dump(mode="json"),
        data=TMSDataConfig(feature_probability=0.05).model_dump(mode="json"),
    )
    path = tmp_path / RUN_CONFIG_FILENAME
    run.write(path)
    loaded = RunConfig.from_file(path)
    assert loaded.run_id == run.run_id
    assert loaded.driver_path == run.driver_path
    assert loaded == run


def test_run_from_file_preserves_existing_run_id(tmp_path: Path):
    run = RunConfig(
        run_id="p-existing",
        driver_path=TMS_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
        target=TMSTargetConfig(run_path="wandb:foo/bar/runs/abc").model_dump(mode="json"),
        data=TMSDataConfig(feature_probability=0.05).model_dump(mode="json"),
    )
    path = tmp_path / RUN_CONFIG_FILENAME
    path.write_text(yaml.safe_dump(run.model_dump(mode="json")))

    loaded = RunConfig.from_file(path)

    assert loaded.run_id == "p-existing"


def test_lm_target_requires_exactly_one_location():
    with pytest.raises(ValidationError):
        LMTargetConfig(
            model_class="transformers.GPT2LMHeadModel",
            model_name=None,
            model_path=None,
        )


def test_lm_driver_validate_config_rejects_wrong_shape():
    run = RunConfig(
        driver_path=LM_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
        target={"completely": "wrong"},
        data={"also": "wrong"},
    )
    with pytest.raises(ValidationError):
        LMDriver().validate_config(run)


def test_tms_driver_validate_config_rejects_wrong_shape():
    run = RunConfig(
        driver_path=TMS_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
        target={"completely": "wrong"},
        data={},
    )
    with pytest.raises(ValidationError):
        TMSDriver().validate_config(run)


def test_resid_mlp_driver_validate_config_rejects_wrong_shape():
    run = RunConfig(
        driver_path=RESID_MLP_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
        target={"completely": "wrong"},
        data={},
    )
    with pytest.raises(ValidationError):
        ResidMLPDriver().validate_config(run)


def test_layerwise_mlp_ci_hidden_dims_must_be_non_empty():
    with pytest.raises(ValidationError, match="hidden_dims must be non-empty"):
        LayerwiseCiConfig(fn_type="mlp", hidden_dims=[])

    with pytest.raises(ValidationError, match="hidden_dims must be non-empty"):
        LayerwiseCiConfig(fn_type="vector_mlp", hidden_dims=[])

    assert LayerwiseCiConfig(fn_type="shared_mlp", hidden_dims=[]).hidden_dims == []


def test_ci_hidden_dims_must_be_positive():
    with pytest.raises(ValidationError):
        LayerwiseCiConfig(fn_type="shared_mlp", hidden_dims=[0])

    with pytest.raises(ValidationError):
        GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[0])


def test_transformer_ci_mlp_hidden_dim_can_use_default():
    cfg = GlobalSharedTransformerCiConfig(
        d_model=8,
        n_blocks=1,
        attn_config=AttnConfig(n_heads=2),
    )
    assert cfg.mlp_hidden_dim is None


def test_global_transformer_ci_rejects_unused_hidden_dims():
    transformer_cfg = GlobalSharedTransformerCiConfig(
        d_model=8,
        n_blocks=1,
        attn_config=AttnConfig(n_heads=2),
    )
    with pytest.raises(ValidationError, match="hidden_dims is only used"):
        GlobalCiConfig(
            fn_type="global_shared_transformer",
            hidden_dims=[8],
            simple_transformer_ci_cfg=transformer_cfg,
        )
