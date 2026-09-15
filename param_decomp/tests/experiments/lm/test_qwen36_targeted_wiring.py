"""The qwen36_moe TARGETED (tPD) composition wiring: authored targeted config →
built run through `build_targeted_experiment_config`, the pool-tokenizer seam, and the
refusals nothing else exercises for this combination. Resolution-level only — no 35B
weights and no HF snapshot; the placed targeted step is the trace gate's job
(`trace_check`, `fit_check.lowered_targeted_train_step`)."""

from pathlib import Path
from typing import Any

import pytest

from param_decomp.core.components import ExpertBlocked
from param_decomp.experiments.lm.config import (
    QWEN36_MOE_MODEL_CLASS,
    QWEN36_MOE_MODEL_NAME,
    LMExperimentConfig,
    LMTargetedExperimentConfig,
    build_targeted_experiment_config,
    resolve_decomposition,
)
from param_decomp.experiments.lm.resolved import Qwen36MoeTargetConfig
from param_decomp.experiments.lm.training_targeted import (
    HubTokenizer,
    SnapshotTokenizer,
    pool_tokenizer_source,
)

N_LAYER, N_EXPERTS = 40, 256
SITE_KINDS = 6


def _raw_targeted_config() -> dict[str, Any]:
    """The minimum-world qwen36 tPD shape: whole grammar on every layer (the family's
    requirement), expert kinds at c_per_expert 1 (C = 256/site), `sc` sources, one node
    as `mesh {data: 1, tp: 8}` — the maintained config's shape at test-cheap CI width."""
    return {
        "run_name": "qwen36-tpd-wiring",
        "cadence": {
            "train_log_every": 5,
            "checkpointing": {
                "kind": "periodic",
                "save_every": 250,
                "retention": {"kind": "keep_last", "n": 1},
            },
        },
        "data": {
            "train": {"kind": "name", "name": "fineweb_qwen36_tok_512_train1"},
            "eval": {"kind": "name", "name": "fineweb_qwen36_tok_512_eval1"},
        },
        "decomposition": {
            "sites": {
                "kind": "qwen36_moe",
                "layers": {"kind": "all"},
                "cs": {
                    "experts_gate": 256,
                    "experts_up": 256,
                    "experts_down": 256,
                    "shared_gate": 128,
                    "shared_up": 128,
                    "shared_down": 128,
                },
            },
            "ci": {
                "type": "chunkwise_transformer",
                "blocks_per_chunk": 4,
                "d_model": 64,
                "n_blocks": 1,
                "attention": {"kind": "mha", "n_heads": 8},
                "ffn": {"kind": "gelu", "hidden": 128},
            },
        },
        "pd": {
            "seed": 0,
            "batch_size": 8,
            "steps": 500,
            "components_optimizer": {
                "lr_schedule": 2.0e-05,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "grad_clip_norm": 0.01,
            },
            "ci_fn_optimizer": {
                "lr_schedule": 2.0e-05,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "grad_clip_norm": None,
            },
            "ci_scaled_weight_decay": 0.1,
            "loss_metrics": [
                {"type": "ImportanceMinimalityLoss", "coeff": 5.0e-06, "gamma": 1.0},
                {
                    "type": "StochasticReconSubsetLoss",
                    "coeff": 0.5,
                    "routing": {"type": "uniform_k_subset"},
                },
                {
                    "type": "PersistentPGDReconLoss",
                    "coeff": 0.5,
                    "n_warmup_steps": 2,
                    "source_shape": "sc",
                    "optimizer": {
                        "type": "adam",
                        "beta1": 0.01,
                        "beta2": 0.99,
                        "eps": 1.0e-08,
                        "lr_schedule": 0.01,
                    },
                },
            ],
        },
        "prompts": {
            "kind": "arithmetic_grid",
            "operation": "add",
            "a_range": [1, 4],
            "b_range": [1, 5],
        },
        "nontarget": {
            "batch_size": 8,
            "impmin_coeff": 1.0e-05,
            "recon": [
                {
                    "type": "StochasticReconSubsetLoss",
                    "coeff": 1.0,
                    "routing": {"type": "uniform_k_subset"},
                }
            ],
        },
        "target": {
            "spec": {
                "kind": "hf",
                "model_class": QWEN36_MOE_MODEL_CLASS,
                "model_name": QWEN36_MOE_MODEL_NAME,
            },
            "attention_implementation": "xla",
            "weights_dtype": "bfloat16",
            "output_edge": {"kind": "materialized"},
            "experts_execution": "routed",
        },
        "runtime": {
            "compilation_cache_dir": "~/.cache/param-decomp/xla",
            "sharding": "zero1-replicated-resident-moe",
            "mesh": {"data": 1, "tp": 8},
            "remat_recon_forwards": True,
            "remat_ci_fn": True,
            "compiler_options": "tuned-v2-autotune1",
        },
    }


def test_qwen36_targeted_config_builds(tmp_path: Path):
    cfg = LMTargetedExperimentConfig.model_validate(_raw_targeted_config())
    built = build_targeted_experiment_config(cfg, "p-00000000", tmp_path)

    target = built.target
    assert isinstance(target, Qwen36MoeTargetConfig)
    assert target.model_name == QWEN36_MOE_MODEL_NAME
    assert len(target.sites) == N_LAYER * SITE_KINDS

    resolved = resolve_decomposition(cfg.target, cfg.decomposition, tmp_path)
    by_group = {spec.group: spec.factorization for spec in resolved.site_specs}
    assert by_group["experts_gate"] == ExpertBlocked(
        n_experts=N_EXPERTS, d_in=2048, d_out=512, c_per_expert=1
    )


def test_qwen36_targeted_pool_tokenizer_is_the_target_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The pool encodes with the MODEL's vocabulary: the qwen36 target resolves its
    prompt-pool tokenizer to the local HF snapshot the weights load staged — never the
    dataset meta's hub name (that arm is the lab-pretrained target's)."""
    snapshot = tmp_path / "models--Qwen--Qwen3.6-35B-A3B" / "snapshots" / "0000aaaa"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    cfg = LMTargetedExperimentConfig.model_validate(_raw_targeted_config())
    built = build_targeted_experiment_config(cfg, "p-00000000", tmp_path)
    source = pool_tokenizer_source(built.target, "some-dataset-meta-tokenizer")
    assert source == SnapshotTokenizer(path=snapshot)
    assert not isinstance(source, HubTokenizer)


def test_qwen36_targeted_build_refuses_layer_isolation(tmp_path: Path):
    """`layers: {kind: range}` refuses through the TARGETED build route too — whole-grid
    coverage is the qwen36_moe family's requirement, identical on both run shapes."""
    raw = _raw_targeted_config()
    raw["decomposition"]["sites"]["layers"] = {"kind": "range", "start": 0, "end": 8}
    cfg = LMTargetedExperimentConfig.model_validate(raw)
    with pytest.raises(AssertionError, match="whole-grid"):
        build_targeted_experiment_config(cfg, "p-00000000", tmp_path)


def test_qwen36_targeted_shape_cannot_spell_faithfulness():
    """tPD has no faithfulness role: neither the loss entry nor the warmup knobs
    are representable on the targeted shape."""
    raw = _raw_targeted_config()
    raw["pd"]["loss_metrics"].append({"type": "FaithfulnessLoss", "coeff": 1.0e03})
    with pytest.raises(Exception, match="FaithfulnessLoss"):
        LMTargetedExperimentConfig.model_validate(raw)

    raw = _raw_targeted_config()
    raw["pd"]["faithfulness_warmup_steps"] = 100
    with pytest.raises(Exception, match="faithfulness_warmup_steps"):
        LMTargetedExperimentConfig.model_validate(raw)


def test_qwen36_targeted_raw_refuses_the_plain_shape():
    with pytest.raises(Exception, match="prompts|nontarget"):
        LMExperimentConfig.model_validate(_raw_targeted_config())
