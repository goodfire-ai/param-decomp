"""Pretrain subtree: arch forwards, cache round-trip into the decomposition loader, and a
short end-to-end training smoke (loss decreases)."""

import tempfile
from pathlib import Path
from typing import Never

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jax.sharding import NamedSharding
from pydantic import ValidationError

import param_decomp.targets.llama_simple_mlp as lsm
from param_decomp.core import placement
from param_decomp.core.precision import cast_floating
from param_decomp.core.sharding import hsdp_mesh, place_target
from param_decomp.core.tools.fit_check import abstract_placed_model
from param_decomp.infra.dataset_store import (
    DatasetMeta,
    NamedDataset,
    dataset_dir,
    write_dataset_meta,
)
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.pretrain.cache import torch_model_config_dict, write_pretrain_cache
from param_decomp.pretrain.config import PretrainConfig
from param_decomp.pretrain.models import (
    GPT2SimpleConfig,
    LlamaSimpleConfig,
    LlamaSimpleMLPConfig,
    ModelConfig,
    init_model,
    model_logits,
)
from param_decomp.pretrain.train import _next_token_ce, train
from param_decomp.sequence import SequenceLayout
from param_decomp.targets.lm_output import MaterializedOutputEdge
from param_decomp.targets.testing import run_clean


def _tiny_mlp_cfg() -> LlamaSimpleMLPConfig:
    return LlamaSimpleMLPConfig(
        model_type="LlamaSimpleMLP",
        block_size=16,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_embd=32,
        n_intermediate=128,
        n_ctx=16,
        n_key_value_heads=2,
        rms_norm_eps=1e-6,
        rotary_base=10000,
    )


def _tiny_configs() -> list[ModelConfig]:
    return [
        GPT2SimpleConfig(
            model_type="GPT2Simple", block_size=16, vocab_size=64, n_layer=2, n_head=4, n_embd=32
        ),
        LlamaSimpleConfig(
            model_type="LlamaSimple",
            block_size=16,
            vocab_size=64,
            n_layer=2,
            n_head=4,
            n_embd=32,
            n_intermediate=80,
            n_ctx=16,
            n_key_value_heads=2,
        ),
        _tiny_mlp_cfg(),
    ]


@pytest.mark.parametrize("cfg", _tiny_configs())
def test_all_archs_forward(cfg: ModelConfig):
    idx = jnp.zeros((2, 16), jnp.int32)
    out = model_logits(
        init_model(cfg, jax.random.PRNGKey(0), "xla"),
        LMBatchWithDocuments.from_unsegmented_sequences(idx),
    )
    assert out.shape == (2, 16, cfg.vocab_size)
    assert bool(jnp.isfinite(out).all())


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="exercises unavailable cuDNN on CPU")
@pytest.mark.parametrize("cfg", _tiny_configs())
def test_pretrain_flash_request_fails_on_cpu(cfg: ModelConfig):
    model = cast_floating(init_model(cfg, jax.random.PRNGKey(0), "flash"), jnp.bfloat16)
    with pytest.raises(AssertionError, match="cuDNN flash attention requires a GPU"):
        model_logits(
            model, LMBatchWithDocuments.from_unsegmented_sequences(jnp.zeros((2, 16), jnp.int32))
        )


def test_cache_round_trip_matches_decomposition_loader(monkeypatch: pytest.MonkeyPatch):
    """The written cache, read back through the decomposition trainer's loader, forwards
    bit-identically to the pretrain model — the cache-compatibility guarantee."""
    mc = _tiny_mlp_cfg()
    model = init_model(mc, jax.random.PRNGKey(1), "xla")
    cfg = PretrainConfig(
        model=mc,
        data=NamedDataset(name="unused"),
        global_batch=2,
        num_iterations=1,
        learning_rate=1e-3,
        warmup_iters=0,
        learning_rate_decay_frac=0.1,
        weight_decay=0.0,
        grad_clip=1.0,
        dtype="float32",
        attention_implementation="xla",
        run_name="t",
    )
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "pretrain_cache" / "proj-t-abc"
        write_pretrain_cache(cache, model, torch_model_config_dict(cfg), step=5)
        loaded_cfg = lsm.load_model_config(cache)
        mesh = hsdp_mesh(1, jax.device_count(), 1)
        rules = placement.from_config("ddp", mesh, ())

        target = lsm.load_target_from_pretrain_cache(
            cache, loaded_cfg, jnp.float32, MaterializedOutputEdge(), "xla"
        )
        loaded_leaves = jax.tree.leaves(target)
        assert loaded_leaves
        assert all(
            isinstance(weight, jax.Array)
            and all(device.platform == "cpu" for device in weight.devices())
            for weight in loaded_leaves
        )

        def refuse_allocation(*_args: object, **_kwargs: object) -> Never:
            raise AssertionError("Abstraction must not allocate device arrays")

        with monkeypatch.context() as no_allocation:
            no_allocation.setattr(jax, "make_array_from_callback", refuse_allocation)
            no_allocation.setattr(jax, "device_put", refuse_allocation)
            abstract = abstract_placed_model(target, rules).model

        for weight, shape in zip(loaded_leaves, jax.tree.leaves(abstract), strict=True):
            assert isinstance(shape, jax.ShapeDtypeStruct)
            assert (shape.shape, shape.dtype) == (weight.shape, weight.dtype)
            assert isinstance(shape.sharding, NamedSharding)
            assert shape.sharding.mesh == mesh

        placed = place_target(target, rules).model
        for array, shape in zip(jax.tree.leaves(placed), jax.tree.leaves(abstract), strict=True):
            assert isinstance(array, jax.Array)
            assert array.sharding == shape.sharding
        for actual, expected in zip(jax.tree.leaves(placed), loaded_leaves, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert target.stacked.attn.implementation == "xla"
        assert all(layer.attn.implementation == "xla" for layer in target.layers)
        idx = jnp.arange(2 * 16, dtype=jnp.int32).reshape(2, 16) % mc.vocab_size
        loaded_logits = run_clean(target, LMBatchWithDocuments.from_unsegmented_sequences(idx))
        assert isinstance(loaded_logits, jax.Array)
        assert jnp.allclose(
            loaded_logits, model(LMBatchWithDocuments.from_unsegmented_sequences(idx)), atol=1e-4
        )


_LEGACY_MODEL_CONFIG_YAML = """\
attn_bias: false
block_size: 512
flash_attention: false
mlp_bias: false
model_type: LlamaSimpleMLP
n_ctx: 512
n_embd: 768
n_head: 6
n_intermediate: 3072
n_key_value_heads: 6
n_layer: 4
rms_norm_eps: 1.0e-06
rotary_adjacent_pairs: false
rotary_base: 10000
rotary_dim: 128
use_grouped_query_attention: true
vocab_size: 50277
"""
"""Verbatim output of the pre-change writer for `pile_llama_simple_mlp-4L-768`, including
`flash_attention`, which is no longer emitted. Cache entries written in this shape are
permanent — they outlive the run — so the loader must keep ignoring the stale keys."""


def test_legacy_cache_model_config_still_loads(tmp_path: Path):
    (tmp_path / "model_config.yaml").write_text(_LEGACY_MODEL_CONFIG_YAML)
    cfg = lsm.load_model_config(tmp_path)
    assert (cfg.n_layer, cfg.n_head, cfg.n_kv_head, cfg.n_embd) == (4, 6, 6, 768)
    assert cfg.head_dim == 128


def _write_token_shards(data_dir: Path, n_shards: int, rows: int, seq_plus1: int, vocab: int):
    """Learnable synthetic data: each row is the `+1 mod vocab` successor sequence from a
    random start, so next-token prediction is a deterministic rule the model can fit (loss
    must drop). Uniform-random tokens have no structure — CE would stay at ln(vocab)."""
    write_dataset_meta(
        data_dir,
        DatasetMeta(
            format_version=2,
            seq_len=seq_plus1,
            tokenizer_name="synthetic",
            preprocessing_name="successor-sequence",
        ),
    )
    rng = np.random.default_rng(0)
    for s in range(n_shards):
        starts = rng.integers(0, vocab, size=(rows, 1), dtype=np.int64)
        toks = ((starts + np.arange(seq_plus1)) % vocab).astype(np.int32)
        table = pa.table(
            {
                "input_ids": pa.array(list(toks), type=pa.list_(pa.int32())),
                "document_ids": pa.array(list(np.zeros_like(toks)), type=pa.list_(pa.int32())),
            }
        )
        pq.write_table(table, data_dir / f"shard_{s:05d}.parquet")


def test_training_smoke_loss_decreases():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_dir = dataset_dir(root, "toy")
        data_dir.mkdir(parents=True)
        mc = _tiny_mlp_cfg()
        _write_token_shards(data_dir, n_shards=2, rows=64, seq_plus1=mc.block_size + 1, vocab=64)
        cfg = PretrainConfig(
            model=mc,
            data=NamedDataset(name="toy"),
            global_batch=8,
            num_iterations=15,
            learning_rate=1e-2,
            warmup_iters=2,
            learning_rate_decay_frac=0.1,
            weight_decay=0.0,
            grad_clip=1.0,
            dtype="float32",
            attention_implementation="xla",
            log_every=1,
            val_every=100,
            val_steps=1,
            save_every=15,
            keep_last=1,
            run_id="t-smoke",
            run_name="smoke",
            data_root=root,
        )
        train(cfg)
        records = (root / "runs" / "t-smoke" / "metrics.jsonl").read_text().splitlines()
        import json

        losses = [json.loads(r)["train_loss"] for r in records if "train_loss" in json.loads(r)]
        assert len(losses) >= 10
        # the last loss is well below the first (random-init CE ~ ln(64) = 4.16)
        assert losses[-1] < losses[0] - 0.3, (losses[0], losses[-1])
        # the produced cache loads into the decomposition trainer
        cache = root / "pretrain_cache" / "pretrain-t-smoke"
        loaded_cfg = lsm.load_model_config(cache)
        target = lsm.load_target_from_pretrain_cache(
            cache, loaded_cfg, jnp.float32, MaterializedOutputEdge(), "xla"
        )
        assert target.head_weight.shape == (mc.vocab_size, mc.n_embd)


def test_pretrain_attention_backend_is_explicit_and_dtype_compatible():
    path = Path(__file__).parents[2] / "pretrain/configs/pile_llama_simple_mlp-2L-128_SMOKE.yaml"
    raw = PretrainConfig.from_file(path).model_dump()
    assert raw.pop("attention_implementation") == "xla"
    with pytest.raises(ValidationError, match="attention_implementation"):
        PretrainConfig.model_validate(raw)
    raw["attention_implementation"] = "auto"
    with pytest.raises(ValidationError, match="attention_implementation"):
        PretrainConfig.model_validate(raw)
    raw["attention_implementation"] = "flash"
    with pytest.raises(ValidationError, match="requires bfloat16 pretraining compute"):
        PretrainConfig.model_validate(raw)
    raw["dtype"] = "bfloat16"
    assert PretrainConfig.model_validate(raw).attention_implementation == "flash"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_next_token_ce_excludes_document_boundaries() -> None:
    tokens = jnp.asarray([[1, 2, 0, 3, 2, 1]], dtype=jnp.int32)
    batch = LMBatchWithDocuments(
        LMBatch(tokens), SequenceLayout(jnp.asarray([[0, 0, 0, 1, 1, 1]], dtype=jnp.int32))
    )
    logits = jax.random.normal(jax.random.PRNGKey(4), (1, 5, 4))
    documents = [
        _next_token_ce(
            logits[:, :2], LMBatchWithDocuments.from_unsegmented_sequences(tokens[:, :3])
        ),
        _next_token_ce(
            logits[:, 3:], LMBatchWithDocuments.from_unsegmented_sequences(tokens[:, 3:])
        ),
    ]
    np.testing.assert_allclose(_next_token_ce(logits, batch), jnp.mean(jnp.stack(documents)))
    gradient = jax.grad(_next_token_ce)(logits, batch)
    np.testing.assert_array_equal(gradient[:, 2], 0)
    assert float(jnp.linalg.norm(gradient[:, :2])) > 0
    assert float(jnp.linalg.norm(gradient[:, 3:])) > 0


def test_next_token_ce_with_no_within_document_pairs_is_zero() -> None:
    batch = LMBatchWithDocuments(
        LMBatch(jnp.asarray([[1, 2, 3]], dtype=jnp.int32)),
        SequenceLayout(jnp.asarray([[0, 1, 2]], dtype=jnp.int32)),
    )
    logits = jax.random.normal(jax.random.PRNGKey(5), (1, 2, 4))
    assert float(_next_token_ce(logits, batch)) == 0
    np.testing.assert_array_equal(jax.grad(_next_token_ce)(logits, batch), 0)
