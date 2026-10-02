"""Train through the LM module entry point on one or more GPUs.

The child process loads a tiny cached LlamaSimpleMLP target, reads parquet batches,
and writes the final checkpoint. Multi-device cases require `--runmultidevice`.
"""

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from safetensors.numpy import save_file

from param_decomp.infra.dataset_store import DatasetMeta, write_dataset_meta

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
    """A tiny random LlamaSimpleMLP in the layout `cache_dir_for_run` resolves
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
    write_dataset_meta(
        shards_dir,
        DatasetMeta(
            format_version=2, seq_len=_SEQ, tokenizer_name="unused", preprocessing_name="fixture"
        ),
    )
    rng = np.random.default_rng(1)
    rows = rng.integers(0, _VOCAB, size=(64, _SEQ), dtype=np.int32)
    pq.write_table(
        pa.table(
            {
                "input_ids": pa.array(rows.tolist(), type=pa.list_(pa.int32())),
            }
        ),
        shards_dir / "shard_00000.parquet",
    )


def _write_run_config(path: Path, shards_dir: Path, dp: int, tp: int, weights_dtype: str) -> None:
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
                # 2 heads so the tp=2 case tiles the q/kv head axes: a head count the
                # assignment cannot tile refuses at the config gate (no replication
                # fallback), which the tp=2 variant would otherwise hit.
                "attention": {
                    "mask": "bidirectional",
                    "kind": "mha",
                    "implementation": "xla",
                    "n_heads": 2,
                },
                "ffn": {"kind": "gelu", "hidden": _D},
            },
        },
        "pd": {
            "seed": 0,
            "components_optimizer": {
                "lr_schedule": {
                    "max_val": 1e-4,
                    "points": [
                        {"at": 0.0, "frac": 1.0},
                        {"at": 1.0, "frac": 0.1, "interp": "cosine"},
                    ],
                },
                "grad_clip_norm": 1.0,
            },
            "ci_fn_optimizer": {
                "lr_schedule": {
                    "max_val": 1e-4,
                    "points": [
                        {"at": 0.0, "frac": 1.0},
                        {"at": 1.0, "frac": 0.1, "interp": "cosine"},
                    ],
                },
            },
            "steps": 2,
            "batch_size": 4,
            "loss_metrics": [
                {"type": "FaithfulnessLoss", "coeff": 1.0},
                {
                    "type": "ImportanceMinimalityLoss",
                    "coeff": 1.0,
                    "gamma": 1.0,
                },
                {"type": "StochasticReconLoss", "coeff": 1.0},
            ],
        },
        "runtime": {
            "mesh": {"replicate": 1, "fsdp": dp // tp, "tp": tp},
            "sharding": "zero1",
            "compilation_cache_dir": str(shards_dir.parent / "xla_compilation_cache"),
            # The production preset, resolved by the trainer; its GPU flags are ignored
            # on this suite's CPU backend.
            "compiler_options": "tuned-v2",
        },
        "cadence": {
            "train_log_every": 1,
            "checkpointing": {
                "kind": "periodic",
                "save_every": 2,
                "retention": {"kind": "keep_last", "n": 1},
            },
        },
        "target": {
            "attention_implementation": "xla",
            "weights_dtype": weights_dtype,
            "output_edge": {"kind": "materialized"},
            "spec": {
                "kind": "pretrained",
                "model_class": (
                    "param_decomp.experiments.lm.pretrain.models.llama_simple_mlp.LlamaSimpleMLP"
                ),
                "run_path": _RUN_PATH,
            },
        },
        "data": {
            "train": {"kind": "dir", "dir": str(shards_dir)},
            "eval": {"kind": "dir", "dir": str(shards_dir.parent / "eval_shards")},
        },
    }
    path.write_text(yaml.safe_dump(config))


def _run_module(config: Path, data_root: Path) -> None:
    runtime = yaml.safe_load(config.read_text())["runtime"]
    local_device_count = math.prod(runtime["mesh"].values())
    subprocess.run(
        [
            sys.executable,
            "-m",
            "param_decomp.experiments.lm.run",
            str(config),
            "--data-root",
            str(data_root),
            "--local-device-count",
            str(local_device_count),
        ],
        check=True,
    )


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, devices: int) -> Path:
    if jax.default_backend() != "gpu" or jax.device_count() < devices:
        pytest.skip(f"requires {devices} GPUs")
    _write_pretrain_cache(tmp_path / "out")
    _write_token_shards(tmp_path / "shards")
    _write_token_shards(tmp_path / "eval_shards")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(jax.device_count()))))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(visible.split(",")[:devices]))
    return tmp_path


@pytest.fixture
def inline_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _scaffold(tmp_path, monkeypatch, devices=4)


@pytest.fixture
def single_device_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _scaffold(tmp_path, monkeypatch, devices=1)


def _assert_trained(out_dir: Path, final_step: int) -> None:
    (run_dir,) = (out_dir / "runs").iterdir()
    assert (run_dir / "launch_config.yaml").exists()
    assert (run_dir / "metrics.jsonl").read_text().strip(), "no train metrics logged"
    # the trainer always checkpoints the final step
    assert (run_dir / "ckpts" / str(final_step) / "decomposition").exists()


@pytest.mark.multidevice
def test_inline_launch_trains_on_all_local_devices(inline_setup: Path) -> None:
    tmp_path = inline_setup
    config = tmp_path / "config.yaml"
    _write_run_config(config, tmp_path / "shards", dp=4, tp=1, weights_dtype="bfloat16")

    _run_module(config, tmp_path / "out")

    _assert_trained(tmp_path / "out", final_step=2)


def test_an_fp32_frozen_target_trains(single_device_setup: Path) -> None:
    """`target.weights_dtype: float32` must survive real steps, not just the load. fp32
    frozen weights meet bf16 compute V/U (`prepare_compute_weights`) in the masked
    forward; that seam promotes rather than refusing, and this run is what says so."""
    tmp_path = single_device_setup
    config = tmp_path / "config.yaml"
    _write_run_config(config, tmp_path / "shards", dp=1, tp=1, weights_dtype="float32")

    _run_module(config, tmp_path / "out")

    _assert_trained(tmp_path / "out", final_step=2)


@pytest.mark.multidevice
def test_inline_launch_refuses_a_mis_sized_allocation(inline_setup: Path) -> None:
    """`dp: 2` inside a 4-device allocation must die at trainer startup
    (`initialize_topology` refuses a world the allocation doesn't tile), never silently
    shard over the ambient devices."""
    tmp_path = inline_setup
    config = tmp_path / "config.yaml"
    _write_run_config(config, tmp_path / "shards", dp=2, tp=1, weights_dtype="bfloat16")

    with pytest.raises(subprocess.CalledProcessError):
        _run_module(config, tmp_path / "out")


@pytest.mark.multidevice
def test_inline_launch_trains_with_tensor_parallel_ci(inline_setup: Path) -> None:
    tmp_path = inline_setup
    config = tmp_path / "config.yaml"
    _write_run_config(config, tmp_path / "shards", dp=4, tp=2, weights_dtype="bfloat16")

    _run_module(config, tmp_path / "out")

    _assert_trained(tmp_path / "out", final_step=2)


@pytest.mark.parametrize("n_vocab_chunks", [1, 4])
def test_streamed_pretrained_target_resolves_and_loads(tmp_path: Path, n_vocab_chunks: int) -> None:
    import jax.numpy as jnp

    from param_decomp.experiments.lm.config import LMExperimentConfig, resolve_decomposition
    from param_decomp.experiments.lm.load_run import load_target
    from param_decomp.experiments.lm.resolved import LlamaSimpleMLPTargetConfig
    from param_decomp.lm.batch import LMBatchWithDocuments
    from param_decomp.targets.lm_output import StreamedLinearOutput, StreamedOutputEdge
    from param_decomp.targets.transformer import TransformerDecomposedModel

    _write_pretrain_cache(tmp_path)
    path = tmp_path / "run.yaml"
    _write_run_config(path, tmp_path / "shards", dp=1, tp=1, weights_dtype="float32")
    raw = yaml.safe_load(path.read_text())
    raw["target"]["output_edge"] = {"kind": "streamed", "n_vocab_chunks": n_vocab_chunks}
    config = LMExperimentConfig.model_validate(raw)
    resolved = resolve_decomposition(config.target, config.decomposition, tmp_path)
    assert isinstance(resolved.target, LlamaSimpleMLPTargetConfig)
    assert resolved.target.output_edge == StreamedOutputEdge(n_vocab_chunks=n_vocab_chunks)
    model = load_target(resolved.target, tmp_path)
    assert isinstance(model, TransformerDecomposedModel)
    output = model.clean_forward(
        LMBatchWithDocuments.from_unsegmented_sequences(jnp.zeros((1, _SEQ), dtype=jnp.int32)),
        placement=None,
    ).output
    assert isinstance(output, StreamedLinearOutput)
    assert output.n_chunks == n_vocab_chunks
    assert output.head is model.embed
    assert output.activations.shape == (1, _SEQ, _D)

    raw["target"]["output_edge"]["n_vocab_chunks"] = 3
    invalid = LMExperimentConfig.model_validate(raw)
    with pytest.raises(AssertionError, match="must divide the"):
        resolve_decomposition(invalid.target, invalid.decomposition, tmp_path)
