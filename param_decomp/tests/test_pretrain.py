"""Pretrain subtree: arch forwards, cache round-trip into the decomposition loader, and a
short end-to-end training smoke (loss decreases)."""

import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import param_decomp.targets.llama_simple_mlp as lsm
from param_decomp.donation import buffer_bytes_by_ptr, reused_fraction
from pretrain.cache import torch_model_config_dict, write_pretrain_cache
from pretrain.config import PretrainConfig, PretrainDataConfig
from pretrain.models import (
    GPT2SimpleConfig,
    LlamaSimpleConfig,
    LlamaSimpleMLPConfig,
    init_model,
    model_logits,
)
from pretrain.train import (
    TrainState,
    _make_checkpoint_manager,
    _restore_latest,
    _save,
    make_optimizer,
    make_train_step,
    train,
)


def _tiny_mlp_cfg() -> LlamaSimpleMLPConfig:
    return LlamaSimpleMLPConfig(
        model_type="LlamaSimpleMLP",
        block_size=16,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_embd=32,
        n_intermediate=128,
        rotary_dim=8,
        n_ctx=16,
        n_key_value_heads=2,
        rms_norm_eps=1e-6,
        rotary_base=10000,
    )


def test_all_archs_forward():
    idx = jnp.zeros((2, 16), jnp.int32)
    cfgs = [
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
            rotary_dim=8,
            n_ctx=16,
            n_key_value_heads=2,
        ),
        _tiny_mlp_cfg(),
    ]
    for cfg in cfgs:
        out = model_logits(init_model(cfg, jax.random.PRNGKey(0)), idx)
        assert out.shape == (2, 16, cfg.vocab_size)
        assert bool(jnp.isfinite(out).all())


def test_cache_round_trip_matches_decomposition_loader():
    """The written cache, read back through the decomposition trainer's loader, forwards
    bit-identically to the pretrain model — the cache-compatibility guarantee."""
    mc = _tiny_mlp_cfg()
    model = init_model(mc, jax.random.PRNGKey(1))
    cfg = PretrainConfig(
        model=mc,
        data=PretrainDataConfig(dir=Path("/tmp"), tokenizer_name="x"),
        global_batch=2,
        num_iterations=1,
        learning_rate=1e-3,
        warmup_iters=0,
        learning_rate_decay_frac=0.1,
        weight_decay=0.0,
        grad_clip=1.0,
        dtype="float32",
        run_name="t",
    )
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "pretrain_cache" / "proj-t-abc"
        write_pretrain_cache(cache, model, torch_model_config_dict(cfg), step=5)
        loaded_cfg = lsm.load_model_config(cache)
        target = lsm.load_target_from_pretrain_cache(cache, loaded_cfg, jnp.float32)
        idx = jnp.arange(2 * 16, dtype=jnp.int32).reshape(2, 16) % mc.vocab_size
        loaded_logits = target.clean_output(idx)
        assert jnp.allclose(loaded_logits, model(idx), atol=1e-4)


def _write_token_shards(data_dir: Path, n_shards: int, rows: int, seq_plus1: int, vocab: int):
    """Learnable synthetic data: each row is the `+1 mod vocab` successor sequence from a
    random start, so next-token prediction is a deterministic rule the model can fit (loss
    must drop). Uniform-random tokens have no structure — CE would stay at ln(vocab)."""
    rng = np.random.default_rng(0)
    for s in range(n_shards):
        starts = rng.integers(0, vocab, size=(rows, 1), dtype=np.int64)
        toks = ((starts + np.arange(seq_plus1)) % vocab).astype(np.int32)
        table = pa.table({"input_ids": pa.array(list(toks), type=pa.list_(pa.int32()))})
        pq.write_table(table, data_dir / f"shard_{s:05d}.parquet")


def test_training_smoke_loss_decreases():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_dir = root / "data"
        data_dir.mkdir()
        mc = _tiny_mlp_cfg()
        _write_token_shards(data_dir, n_shards=2, rows=64, seq_plus1=mc.block_size + 1, vocab=64)
        cfg = PretrainConfig(
            model=mc,
            data=PretrainDataConfig(dir=data_dir, tokenizer_name="x"),
            global_batch=8,
            num_iterations=15,
            learning_rate=1e-2,
            warmup_iters=2,
            learning_rate_decay_frac=0.1,
            weight_decay=0.0,
            grad_clip=1.0,
            dtype="float32",
            log_every=1,
            val_every=100,
            val_steps=1,
            save_every=15,
            keep_last=1,
            run_id="t-smoke",
            run_name="smoke",
            out_dir=root / "runs",
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
        target = lsm.load_target_from_pretrain_cache(cache, loaded_cfg, jnp.float32)
        assert target.lm_head.shape == (mc.vocab_size, mc.n_embd)


def _tiny_train_setup():
    mc = _tiny_mlp_cfg()
    cfg = PretrainConfig(
        model=mc,
        data=PretrainDataConfig(dir=Path("/tmp"), tokenizer_name="x"),
        global_batch=4,
        num_iterations=4,
        learning_rate=1e-3,
        warmup_iters=0,
        learning_rate_decay_frac=0.1,
        weight_decay=0.0,
        grad_clip=1.0,
        dtype="float32",
        run_name="t",
    )
    model = init_model(mc, jax.random.PRNGKey(0))
    optimizer = make_optimizer(cfg, model)
    state = TrainState(
        model=model,
        opt_state=optimizer.init(eqx.filter(model, eqx.is_array)),
        step=jnp.zeros((), jnp.int32),
    )
    return mc, make_train_step(cfg, optimizer), state


def _tokens(mc: LlamaSimpleMLPConfig, seed: int) -> jax.Array:
    return jax.random.randint(jax.random.PRNGKey(seed), (4, mc.block_size + 1), 0, mc.vocab_size)


def test_pretrain_step_donates_state():
    """The train step reuses the model + opt-state buffers (donation, pointer-checked)."""
    mc, step_fn, state = _tiny_train_setup()
    state, _ = step_fn(state, _tokens(mc, 1))  # settle: state is now jit outputs
    in_buffers = buffer_bytes_by_ptr(state)
    new_state, _ = step_fn(state, _tokens(mc, 2))
    jax.block_until_ready(new_state)
    fraction = reused_fraction(in_buffers, new_state)
    assert fraction >= 0.95, f"only {fraction:.2%} of state bytes reused"


def test_restored_pretrain_state_is_donatable(tmp_path: Path):
    """Orbax round-trip preserves donatability: `_restore_latest` re-materialises the
    restored tree as jit outputs, so the first resumed step's donation aliases instead of
    silently copying (jax#18617)."""
    mc, step_fn, state = _tiny_train_setup()
    state, _ = step_fn(state, _tokens(mc, 1))
    mgr = _make_checkpoint_manager(tmp_path / "ckpts", keep_last=1)
    _save(mgr, 1, state)
    restored = _restore_latest(mgr, state)
    assert restored is not None
    restored_state, restored_step = restored
    assert restored_step == 1
    in_buffers = buffer_bytes_by_ptr(restored_state)
    new_state, _ = step_fn(restored_state, _tokens(mc, 2))
    jax.block_until_ready(new_state)
    fraction = reused_fraction(in_buffers, new_state)
    assert fraction >= 0.95, f"only {fraction:.2%} of restored state bytes reused"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
