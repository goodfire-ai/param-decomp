"""`python -m pretrain.train <config.yaml>` — JAX next-token-CE pretraining of an in-house target LM.

The composition root and only I/O layer for pretraining; the step stays pure. Reuses the
decomposition trainer's substrate — `param_decomp.data` (offline pre-tokenized parquet,
never streamed), `param_decomp.sharding` (`init_distributed` / `hsdp_mesh`) — but the
trajectory is a plain LM: fp32 master params, AdamW (weight-decay on 2D weights only,
matching the torch `configure_optimizers` grouping), cosine LR + warmup, grad clip,
next-token cross-entropy. Data-parallel only: the model is small and replicated on every
device; `jax.jit`'s mean-over-the-sharded-batch inserts the grad all-reduce.

Orbax sharded checkpoints under `<run_dir>/ckpts/` are the resume substrate (SIGTERM →
save → SLURM requeue → resume from latest). At each save the pretrained weights are ALSO
written to the decomposition trainer's `pretrain_cache/<project>-<run_id>/` layout
(`cache.write_pretrain_cache`) so the target is immediately decomposable.
"""

import math
import os
import signal
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import cast

import equinox as eqx
import fire
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int
from orbax.checkpoint.type_handlers import ArrayHandler, register_type_handler

from param_decomp.data import BatchSchedule, ShardServer, scan_shards
from param_decomp.sharding import hsdp_mesh, init_distributed
from pretrain.cache import (
    cache_dir_for,
    torch_model_config_dict,
    write_pretrain_cache,
)
from pretrain.config import PretrainConfig, load_pretrain_config
from pretrain.models import PretrainModel, init_model, model_logits

register_type_handler(jax.Array, ArrayHandler(use_replica_parallel=False), override=True)

_sigterm_received = False


def _install_sigterm_flag() -> None:
    def handler(_signum: int, _frame: FrameType | None) -> None:
        global _sigterm_received
        _sigterm_received = True

    signal.signal(signal.SIGTERM, handler)


class TrainState(eqx.Module):
    model: PretrainModel
    opt_state: optax.OptState
    step: Int[Array, ""]


def _decay_mask(model: PretrainModel) -> PretrainModel:
    """True on weight-decayed leaves: 2D+ arrays (matmul weights + embeddings); False on
    norms / biases (the torch `dim() >= 2` grouping)."""
    return jax.tree.map(lambda a: eqx.is_array(a) and a.ndim >= 2, model)


def make_optimizer(cfg: PretrainConfig, model: PretrainModel) -> optax.GradientTransformation:
    def lr_schedule(step: ArrayLike) -> Array:
        peak, frac, warm = cfg.learning_rate, cfg.learning_rate_decay_frac, cfg.warmup_iters
        total = cfg.num_iterations
        min_lr = peak * frac
        it = jnp.asarray(step, dtype=jnp.float32)
        warmup_lr = peak * (it + 1) / max(warm, 1)
        decay_ratio = (it - warm) / max(total - warm, 1)
        decay_ratio = jnp.clip(decay_ratio, 0.0, 1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
        decayed = min_lr + cosine * (peak - min_lr)
        return jnp.where(it < warm, warmup_lr, decayed)

    chain: list[optax.GradientTransformation] = []
    if cfg.grad_clip is not None:
        chain.append(optax.clip_by_global_norm(cfg.grad_clip))
    chain.append(
        optax.adamw(
            learning_rate=lr_schedule,
            b1=cfg.adam_beta1,
            b2=cfg.adam_beta2,
            weight_decay=cfg.weight_decay,
            mask=_decay_mask(model),
        )
    )
    return optax.chain(*chain)


def _next_token_ce(
    logits: Float[Array, "b t1 vocab"], tokens: Int[Array, "b tplus1"]
) -> Float[Array, ""]:
    """Mean cross-entropy of position-`i` logits predicting token `i+1`. `tokens` is one
    token wider than the model context (the staged shards are `block_size + 1` wide)."""
    targets = tokens[:, 1:]
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    picked = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return -picked.mean()


TrainStepFn = Callable[[TrainState, Int[Array, "b tplus1"]], tuple[TrainState, Array]]
EvalStepFn = Callable[[PretrainModel, Int[Array, "b tplus1"]], Array]


def make_train_step(cfg: PretrainConfig, optimizer: optax.GradientTransformation) -> TrainStepFn:
    compute_dtype = jnp.bfloat16 if cfg.dtype == "bfloat16" else jnp.float32
    block = cfg.block_size

    @eqx.filter_jit
    def step_fn(state: TrainState, tokens: Int[Array, "b tplus1"]) -> tuple[TrainState, Array]:
        def loss_fn(model: PretrainModel) -> Array:
            cast_model = _cast_arrays(model, compute_dtype)
            logits = model_logits(cast_model, tokens[:, :block])
            return _next_token_ce(logits, tokens)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(state.model)
        params = eqx.filter(state.model, eqx.is_array)
        updates, new_opt = optimizer.update(grads, state.opt_state, params)
        new_model = eqx.apply_updates(state.model, updates)
        return (
            TrainState(model=new_model, opt_state=new_opt, step=state.step + 1),
            loss,
        )

    return step_fn


def make_eval_step(cfg: PretrainConfig) -> EvalStepFn:
    compute_dtype = jnp.bfloat16 if cfg.dtype == "bfloat16" else jnp.float32
    block = cfg.block_size

    @eqx.filter_jit
    def eval_fn(model: PretrainModel, tokens: Int[Array, "b tplus1"]) -> Array:
        logits = model_logits(_cast_arrays(model, compute_dtype), tokens[:, :block])
        return _next_token_ce(logits, tokens)

    return eval_fn


def _cast_arrays(model: PretrainModel, dtype: jnp.dtype) -> PretrainModel:
    """Cast the floating leaves to the compute dtype; integer/static leaves untouched.
    The masters stay fp32 (the model leaf in `TrainState`); this casts a transient copy."""
    return jax.tree.map(
        lambda a: a.astype(dtype)
        if eqx.is_array(a) and jnp.issubdtype(a.dtype, jnp.floating)
        else a,
        model,
    )


def _replicate(tree: PretrainModel, mesh: Mesh) -> PretrainModel:
    repl = NamedSharding(mesh, P())
    return jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, tree)


def _global_token_batch(local: np.ndarray, mesh: Mesh, global_batch: int) -> jax.Array:
    sharding = NamedSharding(mesh, P(("replicate", "fsdp")))
    return jax.make_array_from_process_local_data(sharding, local, (global_batch, local.shape[1]))


def _make_checkpoint_manager(ckpt_dir: Path, keep_last: int) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        ckpt_dir.resolve(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=keep_last, enable_async_checkpointing=False
        ),
    )


def _save(mgr: ocp.CheckpointManager, step: int, state: TrainState) -> None:
    mgr.save(step, args=ocp.args.StandardSave(state))
    mgr.wait_until_finished()


def _restore_latest(
    mgr: ocp.CheckpointManager, reference: TrainState
) -> tuple[TrainState, int] | None:
    step = mgr.latest_step()
    if step is None:
        return None
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, reference)
    restored = mgr.restore(step, args=ocp.args.StandardRestore(abstract))
    return cast(TrainState, restored), step


class MetricsSink:
    def __init__(self, cfg: PretrainConfig, is_main: bool):
        self._is_main = is_main
        self._wandb = None
        self._jsonl = None
        if not is_main:
            return
        self._jsonl = open(cfg.run_dir / "metrics.jsonl", "a")  # noqa: SIM115 — sink lives the whole run
        if cfg.wandb is not None:
            import wandb

            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                id=cfg.run_id,
                name=cfg.run_name,
                group=cfg.wandb.group,
                tags=list(cfg.wandb.tags),
                resume="allow",
                config=cfg.model_dump(mode="json"),
            )
            self._wandb = wandb

    def log(self, step: int, record: dict[str, float]) -> None:
        if not self._is_main:
            return
        import json

        assert self._jsonl is not None
        self._jsonl.write(json.dumps({"step": step, **record}) + "\n")
        self._jsonl.flush()
        if self._wandb is not None:
            self._wandb.log(record, step=step)

    def finish(self) -> None:
        if self._jsonl is not None:
            self._jsonl.close()
        if self._wandb is not None:
            self._wandb.finish()


def train(cfg: PretrainConfig) -> None:
    _install_sigterm_flag()
    is_distributed = init_distributed(cfg.dp)
    mesh = hsdp_mesh()
    n_proc = jax.process_count()
    ndev = mesh.devices.size
    is_main = jax.process_index() == 0
    assert cfg.global_batch % ndev == 0, (cfg.global_batch, ndev)
    assert cfg.global_batch % n_proc == 0, (cfg.global_batch, n_proc)

    run_dir = cfg.run_dir
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"pretrain run {cfg.run_id} -> {run_dir} (devices={ndev}, procs={n_proc})", flush=True
        )

    key = random.PRNGKey(cfg.seed)
    model = _replicate(init_model(cfg.model, key), mesh)
    optimizer = make_optimizer(cfg, model)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    reference = TrainState(model=model, opt_state=opt_state, step=_global_zero(mesh))

    ckpt_dir = run_dir / "ckpts"
    mgr = _make_checkpoint_manager(ckpt_dir, cfg.keep_last)
    resumed = _restore_latest(mgr, reference)
    if resumed is not None:
        state, start_step = resumed
        if is_main:
            print(f"resumed from step {start_step}", flush=True)
    else:
        state, start_step = reference, 0

    # The shards are `block_size + 1` wide (the extra token is the final label); serve the
    # full row and split x/y inside the step.
    seq_plus1 = cfg.block_size + 1
    schedule = BatchSchedule(scan_shards(cfg.data.dir), cfg.global_batch, cfg.seed)
    server = ShardServer(schedule, seq_plus1, jax.process_index(), n_proc)
    eval_schedule = BatchSchedule(scan_shards(cfg.data.dir), cfg.global_batch, cfg.seed + 1)
    eval_server = ShardServer(eval_schedule, seq_plus1, jax.process_index(), n_proc)

    step_fn = make_train_step(cfg, optimizer)
    eval_fn = make_eval_step(cfg)
    sink = MetricsSink(cfg, is_main)
    model_config = torch_model_config_dict(cfg)
    cache_dir = _cache_dir(cfg) if is_main else None

    tokens_per_step = cfg.global_batch * cfg.block_size
    window_t0 = time.time()

    for step in range(start_step, cfg.num_iterations):
        tokens = _global_token_batch(server.local_batch(step), mesh, cfg.global_batch)
        state, loss = step_fn(state, tokens)
        now = step + 1

        if now % cfg.log_every == 0 or now == cfg.num_iterations:
            jax.block_until_ready(loss)
            dt = time.time() - window_t0
            loss_f = float(loss)
            assert math.isfinite(loss_f), f"non-finite loss at step {now}: {loss_f}"
            per_step = dt / cfg.log_every
            sink.log(
                now,
                {
                    "train_loss": loss_f,
                    "lr": float(_lr_at(cfg, now - 1)),
                    "step_time_s": per_step,
                    "tok_per_s": tokens_per_step / per_step if per_step > 0 else 0.0,
                },
            )
            if is_main:
                print(
                    f"step {now}/{cfg.num_iterations} | loss {loss_f:.4f} | {per_step * 1e3:.1f}ms",
                    flush=True,
                )
            window_t0 = time.time()

        if now % cfg.val_every == 0 or now == cfg.num_iterations:
            val = 0.0
            for j in range(cfg.val_steps):
                eval_tokens = _global_token_batch(
                    eval_server.local_batch((now // cfg.val_every) * cfg.val_steps + j),
                    mesh,
                    cfg.global_batch,
                )
                val += float(eval_fn(state.model, eval_tokens))
            sink.log(now, {"val_loss": val / cfg.val_steps})
            if is_main:
                print(f"  val loss {val / cfg.val_steps:.4f}", flush=True)
            window_t0 = time.time()

        if now % cfg.save_every == 0 or now == cfg.num_iterations or _sigterm_received:
            _save(mgr, now, state)
            if is_main:
                assert cache_dir is not None
                write_pretrain_cache(cache_dir, _gather_model(state.model), model_config, now)
                print(f"checkpoint + cache saved @ step {now}", flush=True)
            window_t0 = time.time()

        if _sigterm_received:
            if is_main:
                print("SIGTERM: saved, exiting for requeue", flush=True)
            break

    sink.finish()
    if is_distributed:
        jax.distributed.shutdown()


def _global_zero(mesh: Mesh) -> Int[Array, ""]:
    repl = NamedSharding(mesh, P())
    return jax.jit(lambda: jnp.zeros((), jnp.int32), out_shardings=repl)()


def _gather_model(model: PretrainModel) -> PretrainModel:
    """Pull the replicated model leaves to host for safetensors write."""
    return jax.tree.map(lambda a: np.asarray(a) if eqx.is_array(a) else a, model)


def _lr_at(cfg: PretrainConfig, step: int) -> float:
    peak, frac, warm = cfg.learning_rate, cfg.learning_rate_decay_frac, cfg.warmup_iters
    total = cfg.num_iterations
    min_lr = peak * frac
    if step < warm:
        return peak * (step + 1) / max(warm, 1)
    decay_ratio = min(max((step - warm) / max(total - warm, 1), 0.0), 1.0)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (peak - min_lr)


def _cache_dir(cfg: PretrainConfig) -> Path:
    """`<PARAM_DECOMP_OUT_DIR>/pretrain_cache/<project>-<run_id>` — the exact dir
    `param_decomp.llama_simple_mlp.pretrain_cache_dir` resolves. `out_dir` is the runs
    dir (`<PARAM_DECOMP_OUT_DIR>/runs`), so the cache ROOT is its parent."""
    assert cfg.out_dir is not None and cfg.run_id is not None
    project = cfg.wandb.project if cfg.wandb is not None else "pretrain"
    return cache_dir_for(cfg.out_dir.parent, project, cfg.run_id)


def main(config: Path) -> None:
    cfg = load_pretrain_config(Path(config))
    if cfg.out_dir is None or cfg.run_id is None:
        # Local hand-run without the launcher: mint an ephemeral identity under cwd.
        cfg = _mint_local_identity(cfg)
    _maybe_enable_compilation_cache(cfg)
    train(cfg)


def _mint_local_identity(cfg: PretrainConfig) -> PretrainConfig:
    import secrets

    out_dir = cfg.out_dir or (Path(os.environ.get("PARAM_DECOMP_OUT_DIR", "out")) / "runs")
    run_id = cfg.run_id or f"t-{secrets.token_hex(4)}"
    return cfg.model_copy(update={"out_dir": out_dir, "run_id": run_id})


def _maybe_enable_compilation_cache(cfg: PretrainConfig) -> None:
    if cfg.out_dir is None:
        return
    cache = cfg.out_dir.parent / "xla_compilation_cache"
    jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 60)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
