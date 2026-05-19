"""Round-trip tests for the `Run` config object."""

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
    LMRun,
    LMTargetConfig,
)
from param_decomp.experiments.resid_mlp.experiment import (
    Driver as ResidMLPDriver,
)
from param_decomp.experiments.resid_mlp.experiment import (
    ResidMLPDataConfig,
    ResidMLPRun,
    ResidMLPTargetConfig,
)
from param_decomp.experiments.tms.experiment import (
    Driver as TMSDriver,
)
from param_decomp.experiments.tms.experiment import (
    TMSDataConfig,
    TMSRun,
    TMSTargetConfig,
)
from param_decomp.run import RUN_METADATA_FILENAME, Run

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
        ci_alive_threshold=0.0,
        eval_batch_size=4,
        train_log_freq=1,
        eval_freq=1,
        slow_eval_freq=1,
        n_eval_steps=1,
        slow_eval_on_first_step=False,
    )


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(autocast_bf16=False, device="cpu", dp=None)


def _round_trip(run: Run) -> Run:
    """Round-trip ``run`` through ``model_dump`` → ``Run.model_validate``."""
    return Run.model_validate(run.model_dump(mode="json"))


def test_run_generates_run_id_on_instantiation():
    run = Run(
        driver_path=None,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
    )

    assert run.run_id.startswith("p-")


def test_lm_run_round_trip():
    run = LMRun(
        driver_path=LM_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
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
    parsed = _round_trip(run)
    assert type(parsed) is LMRun
    assert parsed == run


def test_tms_run_round_trip():
    run = TMSRun(
        driver_path=TMS_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
        target=TMSTargetConfig(run_path="wandb:foo/bar/runs/abc"),
        data=TMSDataConfig(feature_probability=0.05),
    )
    parsed = _round_trip(run)
    assert type(parsed) is TMSRun
    assert parsed == run


def test_resid_mlp_run_round_trip():
    run = ResidMLPRun(
        driver_path=RESID_MLP_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
        target=ResidMLPTargetConfig(run_path="wandb:foo/bar/runs/abc"),
        data=ResidMLPDataConfig(feature_probability=0.05),
    )
    parsed = _round_trip(run)
    assert type(parsed) is ResidMLPRun
    assert parsed == run


def test_run_requires_runtime_config():
    data = {
        "driver_path": None,
        "pd": _pd_config().model_dump(mode="json"),
        "logging": _logging_config().model_dump(mode="json"),
    }

    with pytest.raises(ValidationError, match="runtime"):
        Run.model_validate(data)


def test_driver_class_paths_load():
    assert isinstance(load_driver(LM_DRIVER_PATH), LMDriver)
    assert isinstance(load_driver(TMS_DRIVER_PATH), TMSDriver)
    assert isinstance(load_driver(RESID_MLP_DRIVER_PATH), ResidMLPDriver)


def test_save_pre_run_info_writes_run_metadata(tmp_path: Path):
    from param_decomp.utils.general_utils import save_pre_run_info

    run = Run(
        driver_path=None,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
    )
    save_pre_run_info(
        save_to_wandb=False,
        out_dir=tmp_path,
        run=run,
        artifacts={},
    )

    assert (tmp_path / RUN_METADATA_FILENAME).exists()


def test_run_round_trip_via_file(tmp_path: Path):
    run = TMSRun(
        driver_path=TMS_DRIVER_PATH,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
        target=TMSTargetConfig(run_path="wandb:foo/bar/runs/abc"),
        data=TMSDataConfig(feature_probability=0.05),
    )
    path = tmp_path / RUN_METADATA_FILENAME
    run.write(path)
    loaded = Run.from_file(path)
    assert loaded.run_id == run.run_id
    assert loaded.driver_path == run.driver_path
    assert loaded == run


def test_run_from_file_preserves_existing_run_id(tmp_path: Path):
    run = Run(
        run_id="p-existing",
        driver_path=None,
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
    )
    path = tmp_path / RUN_METADATA_FILENAME
    path.write_text(yaml.safe_dump(run.model_dump(mode="json")))

    loaded = Run.from_file(path)

    assert loaded.run_id == "p-existing"


def test_wandb_fields_default_to_none_on_logging_config():
    cfg = _logging_config()
    assert cfg.wandb_run_name is None
    assert cfg.view_meta == {}


def test_lm_target_requires_exactly_one_location():
    with pytest.raises(ValidationError):
        LMTargetConfig(
            model_class="transformers.GPT2LMHeadModel",
            model_name=None,
            model_path=None,
        )


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
