from pathlib import Path

import pytest
import torch

from param_decomp.configs import (
    LayerwiseCiConfig,
    ModulePatternInfoConfig,
    OptimizerConfig,
    PDConfig,
    ScheduleConfig,
)
from param_decomp.experiments.resid_mlp.experiment import (
    TARGET_MODEL_FILENAME as RESID_MLP_TARGET_MODEL_FILENAME,
)
from param_decomp.experiments.resid_mlp.experiment import (
    TARGET_TRAIN_CONFIG_FILENAME as RESID_MLP_TARGET_TRAIN_CONFIG_FILENAME,
)
from param_decomp.experiments.resid_mlp.experiment import (
    Driver as ResidMLPDriver,
)
from param_decomp.experiments.resid_mlp.experiment import (
    ResidMLPDataConfig,
    ResidMLPExperimentConfig,
    ResidMLPTargetConfig,
)
from param_decomp.experiments.resid_mlp.models import (
    ResidMLP,
    ResidMLPModelConfig,
    ResidMLPTrainConfig,
)
from param_decomp.experiments.tms.experiment import (
    TARGET_MODEL_FILENAME as TMS_TARGET_MODEL_FILENAME,
)
from param_decomp.experiments.tms.experiment import (
    TARGET_TRAIN_CONFIG_FILENAME as TMS_TARGET_TRAIN_CONFIG_FILENAME,
)
from param_decomp.experiments.tms.experiment import (
    Driver as TMSDriver,
)
from param_decomp.experiments.tms.experiment import (
    TMSDataConfig,
    TMSExperimentConfig,
    TMSTargetConfig,
)
from param_decomp.experiments.tms.models import TMSModel, TMSModelConfig, TMSTrainConfig
from param_decomp.models.batch_and_loss_fns import PDTarget


def _pd_config() -> PDConfig:
    return PDConfig(
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        module_info=[ModulePatternInfoConfig(module_pattern="*", C=2)],
        components_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        ci_fn_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        steps=1,
        batch_size=2,
        train_log_freq=1,
        eval_freq=1,
        eval_batch_size=2,
        slow_eval_freq=1,
        n_eval_steps=1,
    )


def _assert_state_dict_equal(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        assert torch.equal(actual[key], expected_value)


def test_resid_mlp_saved_run_loads_bundled_target(tmp_path: Path) -> None:
    model_config = ResidMLPModelConfig(
        n_features=3,
        d_embed=3,
        d_mlp=4,
        n_layers=1,
        act_fn_name="relu",
        in_bias=True,
        out_bias=True,
    )
    train_config = ResidMLPTrainConfig(
        resid_mlp_model_config=model_config,
        feature_probability=0.1,
        batch_size=2,
        steps=1,
        print_freq=1,
        lr_schedule=ScheduleConfig(start_val=1e-3),
    )
    train_config.to_file(tmp_path / RESID_MLP_TARGET_TRAIN_CONFIG_FILENAME)

    source_model = ResidMLP(model_config)
    expected_state_dict = source_model.state_dict()
    torch.save(expected_state_dict, tmp_path / RESID_MLP_TARGET_MODEL_FILENAME)

    target = ResidMLPDriver().build_target(
        ResidMLPExperimentConfig(
            pd=_pd_config(),
            target=ResidMLPTargetConfig(run_path=str(tmp_path / "missing-source-run")),
            data=ResidMLPDataConfig(feature_probability=0.1),
        ),
        run_dir=tmp_path,
    )

    assert isinstance(target, PDTarget)
    assert isinstance(target.model, ResidMLP)
    _assert_state_dict_equal(target.model.state_dict(), expected_state_dict)


def test_resid_mlp_saved_run_requires_bundled_target_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=RESID_MLP_TARGET_MODEL_FILENAME):
        ResidMLPDriver().build_target(
            ResidMLPExperimentConfig(
                pd=_pd_config(),
                target=ResidMLPTargetConfig(run_path=str(tmp_path / "missing-source-run")),
                data=ResidMLPDataConfig(feature_probability=0.1),
            ),
            run_dir=tmp_path,
        )


def test_tms_saved_run_loads_bundled_target(tmp_path: Path) -> None:
    model_config = TMSModelConfig(
        n_features=3,
        n_hidden=2,
        n_hidden_layers=0,
        tied_weights=False,
        init_bias_to_zero=False,
        device="cpu",
    )
    train_config = TMSTrainConfig(
        tms_model_config=model_config,
        feature_probability=0.1,
        batch_size=2,
        steps=1,
        lr_schedule=ScheduleConfig(start_val=1e-3),
        data_generation_type="at_least_zero_active",
    )
    train_config.to_file(tmp_path / TMS_TARGET_TRAIN_CONFIG_FILENAME)

    source_model = TMSModel(model_config)
    expected_state_dict = source_model.state_dict()
    torch.save(expected_state_dict, tmp_path / TMS_TARGET_MODEL_FILENAME)

    target = TMSDriver().build_target(
        TMSExperimentConfig(
            pd=_pd_config(),
            target=TMSTargetConfig(run_path=str(tmp_path / "missing-source-run")),
            data=TMSDataConfig(feature_probability=0.1),
        ),
        run_dir=tmp_path,
    )

    assert isinstance(target, PDTarget)
    assert isinstance(target.model, TMSModel)
    _assert_state_dict_equal(target.model.state_dict(), expected_state_dict)


def test_tms_saved_run_requires_bundled_target_train_config(tmp_path: Path) -> None:
    model_config = TMSModelConfig(
        n_features=3,
        n_hidden=2,
        n_hidden_layers=0,
        tied_weights=False,
        init_bias_to_zero=False,
        device="cpu",
    )
    torch.save(TMSModel(model_config).state_dict(), tmp_path / TMS_TARGET_MODEL_FILENAME)

    with pytest.raises(FileNotFoundError, match=TMS_TARGET_TRAIN_CONFIG_FILENAME):
        TMSDriver().build_target(
            TMSExperimentConfig(
                pd=_pd_config(),
                target=TMSTargetConfig(run_path=str(tmp_path / "missing-source-run")),
                data=TMSDataConfig(feature_probability=0.1),
            ),
            run_dir=tmp_path,
        )
