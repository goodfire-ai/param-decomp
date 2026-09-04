from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError

from param_decomp.core.configs import PGDReconLossConfig
from param_decomp.experiments.resid_mlp.config import ResidMLPExperimentConfig
from param_decomp.experiments.resid_mlp.run import build_resid_mlp_built_run
from param_decomp.experiments.toy_eval import make_toy_evaluation_operations
from param_decomp.targets import resid_mlp

CONFIG_PATH = (
    Path(__file__).parents[3] / "experiments" / "resid_mlp" / "configs" / "resid_mlp_1l.yaml"
)


def test_pretrain_intent_survives_config_build() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["target"]["pretrain"] |= {
        "label_type": "abs",
        "loss_type": "readoff",
        "use_trivial_label_coeffs": False,
        "importance_val": 0.7,
    }

    built = build_resid_mlp_built_run(ResidMLPExperimentConfig(**raw), "p-00000000", Path("out"))

    assert isinstance(built.target, resid_mlp.ResidMLPTargetConfig)
    assert built.target.pretrain_label_type == "abs"
    assert built.target.pretrain_loss_type == "readoff"
    assert built.target.pretrain_use_trivial_label_coeffs is False
    assert built.target.pretrain_importance_val == 0.7


def test_duplicate_eval_metric_identities_refuse_instead_of_last_one_winning() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["eval"]["metrics"].append(dict(raw["eval"]["metrics"][0]))

    with pytest.raises(ValidationError, match="sharing a logged identity"):
        ResidMLPExperimentConfig(**raw)


def test_the_same_metric_twice_is_authorable_when_the_instances_are_named() -> None:
    """The case `LossMetricConfig.name` exists for: one probe at 5 PGD steps beside one at
    20. The binders key emitted metrics on `name or type`, so the two stay distinct in the
    log; before this, the type-level ban made the pair unauthorable."""
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    (pgd,) = [m for m in raw["eval"]["metrics"] if m["type"] == "PGDReconLoss"]
    raw["eval"]["metrics"].append(dict(pgd, name="PGDReconLoss_20step", n_steps=20))

    built = ResidMLPExperimentConfig(**raw)
    assert built.eval is not None
    pgd_names = [m.name for m in built.eval.metrics if isinstance(m, PGDReconLossConfig) and m.name]
    assert pgd_names == ["PGDReconLoss_20step"], pgd_names


@pytest.mark.parametrize(
    "runtime",
    [
        {"dp": 1, "sharding": "ddp"},
        {"compiler_options": {"xla_gpu_enable_triton_gemm": False}},
        {"launch_env": {"xla_python_client_mem_fraction": 0.5}},
        {},
    ],
)
def test_toy_config_refuses_a_runtime_section(runtime: dict[str, object]) -> None:
    """A toy has no compute substrate to author, so the SECTION is unrepresentable —
    refused by the type (`extra="forbid"`), not by a validator subtracting fields from
    `RuntimeConfig` one at a time. Nothing to keep honest as that config grows."""
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    assert "runtime" not in raw, "the shipped toy seat must carry no runtime section"
    raw["runtime"] = runtime

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResidMLPExperimentConfig(**raw)


def test_toy_resume_provenance_refuses_instead_of_being_ignored() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["resume_provenance"] = {"parent_run_dir": "/tmp/p", "parent_step": 1}

    with pytest.raises(ValidationError, match="resume_provenance"):
        ResidMLPExperimentConfig(**raw)


def test_lm_metric_on_toy_fails_when_evaluator_is_built() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["eval"]["metrics"] = [{"type": "CEandKLLosses", "rounding_threshold": 0.1}]
    config = ResidMLPExperimentConfig(**raw)

    assert config.eval is not None
    with pytest.raises(AssertionError, match="categorical output distribution"):
        make_toy_evaluation_operations(
            config.eval,
            seed=0,
            compiler_options={},
            model=cast(Any, None),
            ci_capture_keys=cast(Any, None),
            mesh=cast(Any, None),
            sample_eval_batch=cast(Any, None),
            probe_ci=cast(Any, None),
            wandb_configured=False,
        )


def test_authored_toy_figure_requires_transport() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["eval"]["metrics"] = [
        {"type": "UVPlots", "identity_patterns": None, "dense_patterns": None}
    ]
    config = ResidMLPExperimentConfig(**raw)

    assert config.eval is not None
    with pytest.raises(AssertionError, match="requires a configured wandb transport"):
        make_toy_evaluation_operations(
            config.eval,
            seed=0,
            compiler_options={},
            model=cast(Any, None),
            ci_capture_keys=cast(Any, None),
            mesh=cast(Any, None),
            sample_eval_batch=cast(Any, None),
            probe_ci=cast(Any, None),
            wandb_configured=False,
        )
