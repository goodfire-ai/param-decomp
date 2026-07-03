"""The lab-side single-file config validator (`pd-lm`, torch venv). The runtime
loader (`param_decomp.built_run`, jax venv) can't be imported here, so this exercises
only the lab half: structural dispatch + the no-`run_id` precondition + stamping."""

from pathlib import Path

import pytest
import yaml

from param_decomp.configs import LaunchEnv, ProfileConfig
from param_decomp_lab.experiments.lm.launch import (
    _rank_command,
    _render_rank_env,
    _stamp_config,
    _validate_config,
)

_MINIMAL_LM = {
    "run_name": "r",
    "pd": {
        "seed": 0,
        "ci_config": {
            "type": "chunkwise_transformer",
            "blocks_per_chunk": 1,
            "d_model": 16,
            "n_blocks": 1,
            "n_heads": 1,
            "mlp_hidden": 16,
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


def test_rank_command_runs_trainer_as_module_with_run_id():
    command = _rank_command(
        Path("param_decomp/configs/x.yaml"), "p-abcd1234", rank_env="export FOO=1"
    )
    assert "exec python -m param_decomp_lab.experiments.lm.run" in command
    assert "--run-id p-abcd1234" in command
    assert "pd-train" not in command


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


def test_stamp_config_writes_wandb_and_omits_run_identity(tmp_path: Path):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(_MINIMAL_LM))
    _stamp_config(config, group="grp", tags=["a", "b"])
    raw = yaml.safe_load(config.read_text())
    assert "run_id" not in raw and "out_dir" not in raw
    assert raw["wandb"]["group"] == "grp" and raw["wandb"]["tags"] == ["a", "b"]


def test_stamp_config_noop_without_wandb_knobs(tmp_path: Path):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(_MINIMAL_LM))
    _stamp_config(config, group=None, tags=[])
    raw = yaml.safe_load(config.read_text())
    assert "run_id" not in raw and "out_dir" not in raw
    assert "group" not in raw["wandb"] and "tags" not in raw["wandb"]


def test_default_launch_env_matches_legacy_hardcoded_block():
    env = LaunchEnv().as_env()
    # XLA compiler flags are NOT here anymore — they go via RuntimeConfig.compiler_options.
    assert env == {
        "NCCL_DEBUG": "WARN",
        "MALLOC_ARENA_MAX": "2",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.92",
        "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": "1024",
    }


def test_launch_env_renders_allocator_and_free_form_overrides():
    env = LaunchEnv(
        xla_python_client_allocator="platform",
        profile=ProfileConfig(trace=True, trace_start=10, trace_steps=5, no_checkpoint=True),
        env={"NCCL_DEBUG": "INFO"},  # free-form block overrides a typed knob (merged last)
    ).as_env()
    assert env["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"
    assert env["NCCL_DEBUG"] == "INFO"
    # neither profiling toggles nor XLA compiler flags leak into the rank env (config-native)
    assert not any(k.startswith("PD_") for k in env)
    assert "XLA_FLAGS" not in env


def test_render_rank_env_renders_knobs_and_appends_ld_library_path():
    block = _render_rank_env(LaunchEnv(xla_python_client_allocator="platform"))
    assert "export XLA_PYTHON_CLIENT_ALLOCATOR=platform" in block
    assert block.splitlines()[-1].startswith("export LD_LIBRARY_PATH=")
