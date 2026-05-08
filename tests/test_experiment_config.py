"""Round-trip tests for the `ExperimentConfig` discriminated union."""

import pytest
from pydantic import ValidationError

from param_decomp.configs import LayerwiseCiConfig, PDConfig, ScheduleConfig
from param_decomp.experiment_config import display_name, parse_experiment_config
from param_decomp.experiments.ih.configs import IHDataConfig, IHExperimentConfig, IHTargetConfig
from param_decomp.experiments.lm.configs import (
    LMDataConfig,
    LMExperimentConfig,
    LMTargetConfig,
)
from param_decomp.experiments.resid_mlp.configs import (
    ResidMLPDataConfig,
    ResidMLPExperimentConfig,
    ResidMLPTargetConfig,
)
from param_decomp.experiments.tms.configs import (
    TMSDataConfig,
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


def _round_trip(
    exp: LMExperimentConfig | TMSExperimentConfig | ResidMLPExperimentConfig | IHExperimentConfig,
) -> None:
    """Dump and re-parse via the discriminated union, asserting equality."""
    parsed = parse_experiment_config(exp.model_dump(mode="json"))
    assert type(parsed) is type(exp)
    assert parsed == exp


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
    assert exp.kind == "lm"
    _round_trip(exp)
    assert "GPT2LMHeadModel" in display_name(exp)
    assert "SimpleStories" in display_name(exp)


def test_tms_experiment_round_trip():
    exp = TMSExperimentConfig(
        pd=_pd_config(),
        target=TMSTargetConfig(run_path="wandb:foo/bar/runs/abc"),
        data=TMSDataConfig(feature_probability=0.05),
    )
    assert exp.kind == "tms"
    _round_trip(exp)


def test_resid_mlp_experiment_round_trip():
    exp = ResidMLPExperimentConfig(
        pd=_pd_config(),
        target=ResidMLPTargetConfig(run_path="wandb:foo/bar/runs/abc"),
        data=ResidMLPDataConfig(feature_probability=0.05),
    )
    assert exp.kind == "resid_mlp"
    _round_trip(exp)


def test_ih_experiment_round_trip():
    exp = IHExperimentConfig(
        pd=_pd_config(),
        target=IHTargetConfig(run_path="wandb:foo/bar/runs/abc"),
        data=IHDataConfig(prefix_window=8),
    )
    assert exp.kind == "ih"
    _round_trip(exp)


def test_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        parse_experiment_config({"kind": "bogus", "pd": {}, "target": {}, "data": {}})
