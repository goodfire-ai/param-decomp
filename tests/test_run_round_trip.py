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
from param_decomp.experiments.lm.data import LMDataConfig
from param_decomp.experiments.lm.experiment import (
    LMRecipeConfig,
    LMTargetConfig,
)
from param_decomp.experiments.resid_mlp.experiment import (
    ResidMLPDataConfig,
    ResidMLPRecipeConfig,
    ResidMLPTargetConfig,
)
from param_decomp.experiments.tms.experiment import (
    TMSDataConfig,
    TMSRecipeConfig,
    TMSTargetConfig,
)
from param_decomp.run import RUN_CONFIG_FILENAME, RecipeRef, RunConfig

LM_RECIPE_PATH = "param_decomp.experiments.lm.experiment:Recipe"
TMS_RECIPE_PATH = "param_decomp.experiments.tms.experiment:Recipe"
RESID_MLP_RECIPE_PATH = "param_decomp.experiments.resid_mlp.experiment:Recipe"


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


def _tms_recipe_ref() -> RecipeRef:
    return RecipeRef(
        path=TMS_RECIPE_PATH,
        config=TMSRecipeConfig(
            target=TMSTargetConfig(run_path="wandb:foo/bar/runs/abc"),
            data=TMSDataConfig(feature_probability=0.05),
        ),
    )


def _resid_mlp_recipe_ref() -> RecipeRef:
    return RecipeRef(
        path=RESID_MLP_RECIPE_PATH,
        config=ResidMLPRecipeConfig(
            target=ResidMLPTargetConfig(run_path="wandb:foo/bar/runs/abc"),
            data=ResidMLPDataConfig(feature_probability=0.05),
        ),
    )


def _lm_recipe_ref() -> RecipeRef:
    return RecipeRef(
        path=LM_RECIPE_PATH,
        config=LMRecipeConfig(
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
        ),
    )


def _round_trip(run: RunConfig) -> RunConfig:
    """Round-trip ``run`` through ``model_dump`` → ``RunConfig.from_dict``."""
    return RunConfig.from_dict(run.model_dump(mode="json"))


def test_run_generates_run_id_on_instantiation():
    run = RunConfig(
        recipe=_tms_recipe_ref(),
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
    )

    assert run.run_id.startswith("p-")


def test_lm_recipe_run_round_trip():
    run = RunConfig(
        recipe=_lm_recipe_ref(),
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
    )
    parsed = _round_trip(run)
    assert type(parsed) is RunConfig
    assert isinstance(parsed.recipe.config, LMRecipeConfig)
    assert parsed == run


def test_tms_recipe_run_round_trip():
    run = RunConfig(
        recipe=_tms_recipe_ref(),
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
    )
    parsed = _round_trip(run)
    assert type(parsed) is RunConfig
    assert isinstance(parsed.recipe.config, TMSRecipeConfig)
    assert parsed == run


def test_resid_mlp_recipe_run_round_trip():
    run = RunConfig(
        recipe=_resid_mlp_recipe_ref(),
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
    )
    parsed = _round_trip(run)
    assert type(parsed) is RunConfig
    assert isinstance(parsed.recipe.config, ResidMLPRecipeConfig)
    assert parsed == run


def test_run_requires_runtime_config():
    data = {
        "recipe": _tms_recipe_ref().model_dump(mode="json"),
        "pd": _pd_config().model_dump(mode="json"),
        "logging": _logging_config().model_dump(mode="json"),
    }

    with pytest.raises(ValidationError, match="runtime"):
        RunConfig.from_dict(data)


def test_run_requires_recipe():
    data = {
        "pd": _pd_config().model_dump(mode="json"),
        "logging": _logging_config().model_dump(mode="json"),
        "runtime": _runtime_config().model_dump(mode="json"),
    }

    with pytest.raises(ValidationError, match="recipe"):
        RunConfig.from_dict(data)


def test_run_round_trip_via_file(tmp_path: Path):
    run = RunConfig(
        recipe=_tms_recipe_ref(),
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
    )
    path = tmp_path / RUN_CONFIG_FILENAME
    run.write(path)
    loaded = RunConfig.from_file(path)
    assert loaded.run_id == run.run_id
    assert loaded.recipe == run.recipe
    assert loaded == run


def test_run_from_file_preserves_existing_run_id(tmp_path: Path):
    run = RunConfig(
        run_id="p-existing",
        recipe=_tms_recipe_ref(),
        pd=_pd_config(),
        logging=_logging_config(),
        runtime=_runtime_config(),
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
