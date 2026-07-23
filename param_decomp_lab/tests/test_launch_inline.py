"""End-to-end `launch: inline` coverage through the REAL `pd-lm` path.

`launch.main` validates the config (placement claims at the declared topology), pins it
into a fresh run dir, and runs the trainer as a child process of this allocation — which
must claim exactly the `runtime.dp` local devices (here 4 simulated CPU devices via
`XLA_FLAGS=--xla_force_host_platform_device_count`) and train end-to-end: a tiny
LlamaSimpleMLP target fabricated into the pretrain cache, tokens from tiny parquet
shards, a real train step, and the final-step orbax checkpoint.
"""

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from safetensors.numpy import save_file

import param_decomp_lab.experiments.lm.launch as launch

_VOCAB = 64
_D = 8
_D_INTERMEDIATE = 16
_SEQ = 16
_RUN_PATH = "goodfire/spd/runs/t-00000000"

_ARCH: dict[str, Any] = {
    "model_type": "LlamaSimpleMLP",
    "vocab_size": _VOCAB,
    "n_layer": 1,
    "n_head": 2,
    "n_key_value_heads": 2,
    "n_embd": _D,
    "n_intermediate": _D_INTERMEDIATE,
    "rotary_base": 10000.0,
    "rms_norm_eps": 1e-5,
    "n_ctx": _SEQ,
    "rotary_dim": _D // 2,  # head_dim (the loader insists rotary_dim == head_dim)
    "block_size": _SEQ,
    "use_grouped_query_attention": True,
    "attn_bias": False,
    "mlp_bias": False,
    "rotary_adjacent_pairs": False,
}


def _write_pretrain_cache(out_dir: Path) -> None:
    """A tiny random LlamaSimpleMLP in the layout `pretrain_cache_dir` resolves
    (`model_config.yaml` + one `model_step_*.safetensors`, torch [out, in] weights)."""
    cache_dir = out_dir / "pretrain_cache" / "spd-t-00000000"
    cache_dir.mkdir(parents=True)
    (cache_dir / "model_config.yaml").write_text(yaml.safe_dump(_ARCH))
    rng = np.random.default_rng(0)
    weights = {
        "wte.weight": (_VOCAB, _D),
        "h.0.rms_1.weight": (_D,),
        "h.0.rms_2.weight": (_D,),
        "h.0.attn.q_proj.weight": (_D, _D),
        "h.0.attn.k_proj.weight": (_D, _D),
        "h.0.attn.v_proj.weight": (_D, _D),
        "h.0.attn.o_proj.weight": (_D, _D),
        "h.0.mlp.c_fc.weight": (_D_INTERMEDIATE, _D),
        "h.0.mlp.down_proj.weight": (_D, _D_INTERMEDIATE),
        "ln_f.weight": (_D,),
    }
    save_file(
        {k: (0.1 * rng.standard_normal(shape)).astype(np.float32) for k, shape in weights.items()},
        str(cache_dir / "model_step_0.safetensors"),
    )


def _write_token_shards(shards_dir: Path) -> None:
    shards_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    rows = rng.integers(0, _VOCAB, size=(64, _SEQ), dtype=np.int32)
    pq.write_table(
        pa.table({"input_ids": [row.tolist() for row in rows]}), shards_dir / "shard_00000.parquet"
    )


def _write_run_config(path: Path, shards_dir: Path, dp: int) -> None:
    config = {
        "run_name": "inline-multidevice-smoke",
        "decomposition": {
            "sites": {
                "kind": "simple_mlp",
                "layers": {"kind": "all"},
                "cs": {"c_fc": 4, "down_proj": 4},
            },
            "ci": {
                "type": "chunkwise_transformer",
                "blocks_per_chunk": 1,
                "d_model": _D,
                "n_blocks": 1,
                "attention": {"kind": "mha", "n_heads": 1},
                "ffn": {"kind": "gelu", "hidden": _D},
            },
        },
        "pd": {
            "seed": 0,
            "components_optimizer": {
                "lr_schedule": {"start_val": 1e-4, "fn_type": "cosine", "final_val_frac": 0.1},
                "grad_clip_norm": 1.0,
            },
            "ci_fn_optimizer": {
                "lr_schedule": {"start_val": 1e-4, "fn_type": "cosine", "final_val_frac": 0.1},
            },
            "steps": 2,
            "batch_size": 4,
            "loss_metrics": [
                {"type": "FaithfulnessLoss", "coeff": 1.0},
                {
                    "type": "ImportanceMinimalityLoss",
                    "coeff": 1.0,
                    "pnorm": {"start_val": 2.0, "fn_type": "constant"},
                },
                {"type": "StochasticReconLoss", "coeff": 1.0},
            ],
        },
        "runtime": {"launch": "inline", "dp": dp, "sharding": "zero1"},
        "cadence": {"train_log_every": 1, "save_every": 2, "keep_last_n_checkpoints": 1},
        "target": {
            "weights_dtype": "bfloat16",
            "spec": {
                "kind": "pretrained",
                "model_class": (
                    "param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp.LlamaSimpleMLP"
                ),
                "run_path": _RUN_PATH,
            },
        },
        "data": {
            "dataset_name": "parquet",
            "data_files": str(shards_dir / "*.parquet"),
            "tokenizer_name": "unused",
            "column_name": "input_ids",
            "max_seq_len": _SEQ,
            "is_tokenized": True,
            "streaming": False,
        },
    }
    path.write_text(yaml.safe_dump(config))


@pytest.fixture
def inline_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Out dir + pretrain cache + shards, wired for both the launcher (module constant,
    parent-process placement validation) and the trainer child (inherited env: forced
    4-device CPU topology, out dir)."""
    out_dir = tmp_path / "out"
    _write_pretrain_cache(out_dir)
    _write_token_shards(tmp_path / "shards")
    monkeypatch.setenv("PARAM_DECOMP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    monkeypatch.setattr(launch, "PARAM_DECOMP_OUT_DIR", out_dir)
    return tmp_path


@pytest.mark.multidevice
def test_inline_launch_trains_on_all_local_devices(inline_setup: Path) -> None:
    tmp_path = inline_setup
    config = tmp_path / "config.yaml"
    _write_run_config(config, tmp_path / "shards", dp=4)

    launch.main(str(config))

    (run_dir,) = (tmp_path / "out" / "runs").iterdir()
    assert (run_dir / "launch_config.yaml").exists()
    assert (run_dir / "metrics.jsonl").read_text().strip(), "no train metrics logged"
    # the trainer always checkpoints the final step
    assert (run_dir / "ckpts" / "2" / "decomposition").exists()


@pytest.mark.multidevice
def test_inline_launch_refuses_a_mis_sized_allocation(inline_setup: Path) -> None:
    """`dp: 2` inside a 4-device allocation must die at trainer startup
    (`assert_inline_topology`), never silently shard over the ambient devices."""
    tmp_path = inline_setup
    config = tmp_path / "config.yaml"
    _write_run_config(config, tmp_path / "shards", dp=2)

    with pytest.raises(subprocess.CalledProcessError):
        launch.main(str(config))
