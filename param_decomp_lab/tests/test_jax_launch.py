"""The lab-side single-file config validator (`pd-jax-lm`, torch venv). The runtime
loader (`jax_single_pool.config`, jax venv) can't be imported here, so this exercises
only the lab half: structural dispatch + the no-`run_id` precondition + stamping."""

from pathlib import Path

import pytest
import yaml

from param_decomp_lab.experiments.lm.jax_launch import _stamp_config, _validate_config

_MINIMAL_LM = {
    "run_name": "r",
    "pd": {
        "seed": 0,
        "n_mask_samples": 1,
        "ci_config": {
            "mode": "global",
            "fn_type": "global_shared_transformer",
            "hidden_dims": None,
            "simple_transformer_ci_cfg": {
                "d_model": 16,
                "n_blocks": 1,
                "mlp_hidden_dim": [16],
                "attn_config": {"n_heads": 1, "max_len": 8, "rope_base": 10000.0},
            },
        },
        "decomposition_targets": [{"module_pattern": "layers.0.mlp.gate_proj", "C": 4}],
        "components_optimizer": {"lr_schedule": {"start_val": 1e-4, "fn_type": "cosine"}},
        "ci_fn_optimizer": {"lr_schedule": {"start_val": 1e-4, "fn_type": "cosine"}},
        "steps": 10,
        "batch_size": 8,
        "loss_metrics": [{"type": "FaithfulnessLoss", "coeff": 1.0}],
    },
    "runtime": {"device": "cuda:0"},
    "cadence": {"train_log_every": 1},
    "target": {
        "spec": {
            "kind": "hf",
            "model_class": "transformers.LlamaForCausalLM",
            "model_name": "meta-llama/Llama-3.1-8B",
        }
    },
    "data": {"dataset_name": "parquet", "tokenizer_name": "t"},
    "wandb": {"project": "p"},
}


def test_validate_config_returns_run_name(tmp_path: Path):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(_MINIMAL_LM))
    _, run_name = _validate_config(config)
    assert run_name == "r"


def test_validate_config_rejects_pre_stamped_run_id(tmp_path: Path):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(dict(_MINIMAL_LM, run_id="p-12345678")))
    with pytest.raises(AssertionError, match="run_id is minted at submit"):
        _validate_config(config)


def test_stamp_config_writes_run_id_out_dir_and_wandb(tmp_path: Path):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(_MINIMAL_LM))
    _stamp_config(config, "p-abcd1234", group="grp", tags=["a", "b"])
    raw = yaml.safe_load(config.read_text())
    assert raw["run_id"] == "p-abcd1234"
    assert raw["out_dir"].endswith("/runs")
    assert raw["wandb"]["group"] == "grp" and raw["wandb"]["tags"] == ["a", "b"]


def test_stamp_config_keeps_author_out_dir(tmp_path: Path):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(dict(_MINIMAL_LM, out_dir="/custom/runs")))
    _stamp_config(config, "p-abcd1234", group=None, tags=[])
    raw = yaml.safe_load(config.read_text())
    assert raw["out_dir"] == "/custom/runs"
    assert "group" not in raw["wandb"] and "tags" not in raw["wandb"]
