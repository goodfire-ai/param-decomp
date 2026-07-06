"""`ConsumerRunConfig` opens stored run configs across trainer-schema drift: strict on the
fields consumers read, indifferent to everything else. The full `LMExperimentConfig`
(`extra="forbid"`) is expected to REFUSE the same drifted config — that contrast is the
contract under test."""

from typing import Any

import pytest
from pydantic import ValidationError

from param_decomp_lab.experiments.lm.config import ConsumerRunConfig, LMExperimentConfig

DRIFTED_LAUNCH_CONFIG: dict[str, Any] = {
    "run_name": "drifted",
    "removed_top_level_knob": True,
    "target": {
        "spec": {
            "kind": "pretrained",
            "model_class": "param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp.LlamaSimpleMLP",
            "run_path": "goodfire/spd/runs/abcd1234",
        },
        "weights_dtype": "bfloat16",
    },
    "pd": {
        "seed": 0,
        "batch_size": 64,
        "steps": 1000,
        "decomposition_targets": [
            {"module_pattern": "h.*.mlp.c_fc", "C": 1200},
            {"module_pattern": "h.*.mlp.down_proj", "C": 1200},
        ],
        "loss_metrics": [{"type": "LossTypeDeletedFromTip", "coeff": 1.0}],
    },
    "data": {
        "dataset_name": "parquet",
        "data_files": "/shards/pile_neox_tok_512/*.parquet",
        "tokenizer_name": "EleutherAI/gpt-neox-20b",
        "max_seq_len": 512,
        "column_name": "input_ids",
        "field_from_a_branch_ahead_of_tip": 4,
    },
    "runtime": {"dp": 64, "field_removed_at_tip": True},
}


def test_consumer_config_parses_across_drift_where_full_schema_refuses():
    cfg = ConsumerRunConfig.model_validate(DRIFTED_LAUNCH_CONFIG)
    assert cfg.target.spec.kind == "pretrained"
    assert [(t.module_pattern, t.C) for t in cfg.pd.decomposition_targets] == [
        ("h.*.mlp.c_fc", 1200),
        ("h.*.mlp.down_proj", 1200),
    ]
    assert cfg.data.tokenizer_name == "EleutherAI/gpt-neox-20b"
    assert cfg.data.max_seq_len == 512

    with pytest.raises(ValidationError):
        LMExperimentConfig.model_validate(DRIFTED_LAUNCH_CONFIG)


def test_consumer_config_is_strict_on_the_fields_it_reads():
    missing_targets = {
        **DRIFTED_LAUNCH_CONFIG,
        "pd": {
            k: v for k, v in DRIFTED_LAUNCH_CONFIG["pd"].items() if k != "decomposition_targets"
        },
    }
    with pytest.raises(ValidationError):
        ConsumerRunConfig.model_validate(missing_targets)

    bad_spec = {
        **DRIFTED_LAUNCH_CONFIG,
        "target": {"spec": {"kind": "no_such_kind"}},
    }
    with pytest.raises(ValidationError):
        ConsumerRunConfig.model_validate(bad_spec)
