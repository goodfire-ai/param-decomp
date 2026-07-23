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
    "decomposition": {
        "sites": {
            "kind": "glu_transformer",
            "layers": {"kind": "list", "indices": [0]},
            "cs": {"gate": 4},
        },
        "ci": {
            "type": "chunkwise_transformer",
            "blocks_per_chunk": 1,
            "d_model": 16,
            "n_blocks": 1,
            "attention": {"kind": "mha", "n_heads": 1},
            "ffn": {"kind": "gelu", "hidden": 16},
        },
    },
    "pd": {
        "seed": 0,
        "components_optimizer": {"lr_schedule": {"start_val": 1e-4, "fn_type": "cosine"}},
        "ci_fn_optimizer": {"lr_schedule": {"start_val": 1e-4, "fn_type": "cosine"}},
        "steps": 10,
        "batch_size": 8,
        "loss_metrics": [{"type": "FaithfulnessLoss", "coeff": 1.0}],
    },
    "runtime": {"device": "cuda:0", "launch": "inline", "dp": 1, "sharding": "zero1"},
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


def test_rank_command_builds_node_workspace_then_execs_trainer():
    command = _rank_command(
        "p-abcd1234",
        "refs/runs/snapshot/p-abcd1234",
        Path("/home/u/param-decomp/.git"),
        Path("/out/runs/p-abcd1234"),
        rank_env="export FOO=1",
    )
    # the snapshot comes from the shared-FS common git dir (file:// so --depth works —
    # git ignores it on the bare-path local transport); .env (secrets, not in git) comes
    # from the run dir where submit staged it
    fetch = (
        'git fetch --quiet --depth 1 "file:///home/u/param-decomp/.git" '
        '"refs/runs/snapshot/p-abcd1234"'
    )
    assert fetch in command
    assert "git clone" not in command
    assert 'cp "/out/runs/p-abcd1234/.env" .env' in command
    # per-node job-side workspace: snapshot checkout + CUDA venv, then exec (no EXIT trap
    # — bash is replaced, cleanup is the batch script's trap)
    assert "git checkout --quiet FETCH_HEAD" in command
    # the CUDA extra is driver-gated on the node (cuda13 needs >= r580; cuBLAS TMEM fix)
    assert 'if [ "$DRIVER_MAJOR" -ge 580 ]; then CUDA_EXTRA=cuda13; else CUDA_EXTRA=cuda; fi' in (
        command
    )
    assert 'uv sync --all-packages --no-dev --extra "$CUDA_EXTRA"' in command
    assert "trap" not in command
    # venv activation precedes the rank env (LD_LIBRARY_PATH shells out to the venv python)
    assert command.index("source .venv/bin/activate") < command.index("export FOO=1")
    assert command.endswith(
        "exec python -m param_decomp_lab.experiments.lm.run "
        "/out/runs/p-abcd1234/launch_config.yaml --run-id p-abcd1234"
    )


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


def test_validate_config_fires_the_placement_claim_gate_pre_submit(tmp_path: Path):
    """The build-time placement gate runs at submit validation, before any snapshot or
    sbatch: an `owner+zero1` config whose declared topology makes every shape group tile
    (here the single-site smoke at `launch: inline, dp: 1`) refuses on the login node."""
    raw = dict(_MINIMAL_LM, runtime={"launch": "inline", "dp": 1, "sharding": "owner+zero1"})
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(AssertionError, match="single-device smoke cannot exercise"):
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
