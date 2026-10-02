"""`python -m param_decomp.pretrain.train <config.yaml>` — JAX next-token-CE pretraining of an in-house target LM.

The composition root and only I/O layer for pretraining; the step stays pure. Reuses the
decomposition trainer's substrate — `param_decomp.lm.batch_data` (offline pre-tokenized parquet,
never streamed), `param_decomp.core.sharding` (`initialize_topology` / `hsdp_mesh`) — but the
trajectory is a plain LM: fp32 master params, AdamW (weight-decay on 2D weights only,
matching the torch `configure_optimizers` grouping), cosine LR + warmup, grad clip,
next-token cross-entropy. Data-parallel only: the model is small and replicated on every
device; `jax.jit`'s mean-over-the-sharded-batch inserts the grad all-reduce.

Orbax sharded checkpoints under `<run_dir>/ckpts/` are the resume substrate (SIGTERM →
save → restart → resume from latest). At each save the pretrained weights are ALSO
written to the decomposition trainer's `pretrain_cache/<project>-<run_id>/` layout
(`cache.write_pretrain_cache`) so the target is immediately decomposable.
"""

import math
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
from orbax.checkpoint.checkpoint_managers import preservation_policy
from orbax.checkpoint.type_handlers import ArrayHandler, register_type_handler

from param_decomp.core.sharding import HSDP_MESH_AXES, initialize_topology
from param_decomp.core.world_size import world_size_from_device_count
from param_decomp.infra.dataset_store import read_dataset_meta, resolve_dataset_ref
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.lm.batch_data import (
    ShardServer,
    global_lm_batch,
    scan_shards,
)
from param_decomp.lm.batch_schedule import BatchSchedule
from param_decomp.metric_schema import MetricNames, validate_resume_schema
from param_decomp.pretrain.cache import (
    cache_dir_for,
    torch_model_config_dict,
    write_pretrain_cache,
)
from param_decomp.pretrain.config import (
    PretrainConfig,
    PretrainRunPaths,
    load_pretrain_config,
)
from param_decomp.pretrain.models import PretrainModel, init_model, model_logits

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
    logits: Float[Array, "b t1 vocab"], batch: LMBatchWithDocuments
) -> Float[Array, ""]:
    """Mean CE over adjacent tokens in the same document; empty comparisons yield zero.

    Train rows carry an extra final label (`block_size + 1`); a validation row may
    be `block_size` wide, leaving its last logit targetless.
    """
    targets = batch.batch.token_ids[:, 1:]
    logp = jax.nn.log_softmax(logits[:, : targets.shape[1]].astype(jnp.float32), axis=-1)
    picked = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    weights = batch.sequence.next_token_mask()
    return -jnp.sum(jnp.where(weights, picked, 0)) / jnp.maximum(jnp.sum(weights), 1)


TrainStepFn = Callable[[TrainState, LMBatchWithDocuments], tuple[TrainState, Array]]
EvalStepFn = Callable[[PretrainModel, LMBatchWithDocuments], Array]


def make_train_step(cfg: PretrainConfig, optimizer: optax.GradientTransformation) -> TrainStepFn:
    compute_dtype = jnp.bfloat16 if cfg.dtype == "bfloat16" else jnp.float32
    block = cfg.block_size

    @eqx.filter_jit
    def step_fn(state: TrainState, batch: LMBatchWithDocuments) -> tuple[TrainState, Array]:
        def loss_fn(model: PretrainModel) -> Array:
            cast_model = _cast_arrays(model, compute_dtype)
            logits = model_logits(cast_model, jax.tree.map(lambda a: a[:, :block], batch))
            return _next_token_ce(logits, batch)

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
    def eval_fn(model: PretrainModel, batch: LMBatchWithDocuments) -> Array:
        logits = model_logits(
            _cast_arrays(model, compute_dtype), jax.tree.map(lambda a: a[:, :block], batch)
        )
        return _next_token_ce(logits, batch)

    return eval_fn


def _cast_arrays(model: PretrainModel, dtype: jnp.dtype) -> PretrainModel:
    """Cast the floating leaves to the compute dtype; integer/static leaves untouched.
    The masters stay fp32 (the model leaf in `TrainState`); this casts a transient copy."""
    return jax.tree.map(
        lambda a: (
            a.astype(dtype) if eqx.is_array(a) and jnp.issubdtype(a.dtype, jnp.floating) else a
        ),
        model,
    )


def _replicate(tree: PretrainModel, mesh: Mesh) -> PretrainModel:
    repl = NamedSharding(mesh, P())
    return jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, tree)


def _make_checkpoint_manager(ckpt_dir: Path, keep_last: int) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        ckpt_dir.resolve(),
        options=ocp.CheckpointManagerOptions(
            preservation_policy=preservation_policy.LatestN(n=keep_last),
            enable_async_checkpointing=False,
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
        self._metric_names = MetricNames(cfg.wandb.metric_schema if cfg.wandb else "legacy")
        if not is_main:
            return
        if cfg.wandb is not None:
            import wandb

            tracker = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                id=cfg.run_id,
                name=cfg.run_name,
                group=cfg.wandb.group,
                tags=list(cfg.wandb.tags),
                resume="allow",
            )
            if tracker.resumed:
                validate_resume_schema(tracker.config.as_dict(), cfg.wandb.metric_schema)
            else:
                tracker.config.update(cfg.model_dump(mode="json"), allow_val_change=False)
            self._wandb = wandb
        self._jsonl = open(cfg.paths.run_dir / "metrics.jsonl", "a")  # noqa: SIM115 — sink lives the whole run

    def log(self, step: int, record: dict[str, float]) -> None:
        if not self._is_main:
            return
        import json

        assert self._jsonl is not None
        self._jsonl.write(json.dumps({"step": step, **record}) + "\n")
        self._jsonl.flush()
        if self._wandb is not None:
            self._wandb.log(self._metric_names.record(record), step=step)

    def finish(self) -> None:
        if self._jsonl is not None:
            self._jsonl.close()
        if self._wandb is not None:
            self._wandb.finish()


def _pretrain_mesh(replicate: int, fsdp: int) -> Mesh:
    """The pretrainer's GSPMD (Auto-mode) data mesh — this trainer still places by
    constraint propagation, unlike the decomposition trainer's Explicit mesh."""
    devices = np.array(jax.devices())
    assert devices.size == replicate * fsdp, (devices.size, replicate, fsdp)
    return Mesh(devices.reshape(replicate, fsdp, 1), axis_names=HSDP_MESH_AXES)


def train(cfg: PretrainConfig) -> None:
    _install_sigterm_flag()
    if cfg.dp is not None:
        initialize_topology(world_size_from_device_count(cfg.dp), cfg.gpus_per_node)
        mesh = _pretrain_mesh(cfg.dp // cfg.gpus_per_node, cfg.gpus_per_node)
    else:
        mesh = _pretrain_mesh(1, jax.device_count())
    is_distributed = cfg.dp is not None
    n_proc = jax.process_count()
    ndev = mesh.devices.size
    is_main = jax.process_index() == 0
    assert cfg.global_batch % ndev == 0, (cfg.global_batch, ndev)
    assert cfg.global_batch % n_proc == 0, (cfg.global_batch, n_proc)

    paths = cfg.paths
    run_dir = paths.run_dir
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"pretrain run {cfg.run_id} -> {run_dir} (devices={ndev}, procs={n_proc})", flush=True
        )

    key = random.PRNGKey(cfg.seed)
    model = _replicate(init_model(cfg.model, key, cfg.attention_implementation), mesh)
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
    train_dir = resolve_dataset_ref(cfg.data, paths.data_root)
    train_meta = read_dataset_meta(train_dir)
    assert train_meta.seq_len == seq_plus1, (
        f"training rows must contain block_size + 1 = {seq_plus1} tokens"
    )
    shards = scan_shards(train_dir)
    schedule = BatchSchedule(shards, cfg.global_batch, cfg.seed)
    server = ShardServer(schedule, seq_plus1, jax.process_index(), n_proc)
    if cfg.val_data is not None:
        eval_dir = resolve_dataset_ref(cfg.val_data, paths.data_root)
        eval_meta = read_dataset_meta(eval_dir)
        assert eval_meta.tokenizer_name == train_meta.tokenizer_name, (
            "training and evaluation datasets must use the same tokenizer"
        )
        assert eval_meta.seq_len in (cfg.block_size, seq_plus1), (
            "evaluation rows must contain block_size or block_size + 1 tokens"
        )
        eval_shards = scan_shards(eval_dir)
        eval_width = eval_meta.seq_len
    else:
        eval_shards, eval_width = shards, seq_plus1
    eval_schedule = BatchSchedule(eval_shards, cfg.global_batch, cfg.seed + 1)
    eval_server = ShardServer(eval_schedule, eval_width, jax.process_index(), n_proc)

    step_fn = make_train_step(cfg, optimizer)
    eval_fn = make_eval_step(cfg)
    sink = MetricsSink(cfg, is_main)
    model_config = torch_model_config_dict(cfg)
    cache_dir = _cache_dir(cfg, paths) if is_main else None

    tokens_per_step = cfg.global_batch * cfg.block_size
    window_t0 = time.time()

    for step in range(start_step, cfg.num_iterations):
        batch = global_lm_batch(
            server.local_batch(step), mesh, cfg.global_batch, cfg.model.vocab_size
        )
        state, loss = step_fn(state, batch)
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
                eval_batch = global_lm_batch(
                    eval_server.local_batch((now // cfg.val_every) * cfg.val_steps + j),
                    mesh,
                    cfg.global_batch,
                    cfg.model.vocab_size,
                )
                val += float(eval_fn(state.model, eval_batch))
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


def _cache_dir(cfg: PretrainConfig, paths: PretrainRunPaths) -> Path:
    """`<data_root>/pretrain_cache/<project>-<run_id>` — the exact dir
    `param_decomp.infra.pretrain_cache.cache_dir_for_run` resolves."""
    project = cfg.wandb.project if cfg.wandb is not None else "pretrain"
    return cache_dir_for(paths.data_root, project, paths.run_id)


def main(config: Path) -> None:
    cfg = _stamp_local_identity(load_pretrain_config(Path(config)))
    _enable_compilation_cache(cfg.paths)
    train(cfg)


def _stamp_local_identity(cfg: PretrainConfig) -> PretrainConfig:
    """A direct run mints an ephemeral run id. `data_root` has no fallback, so the
    config must carry it."""
    import secrets

    assert cfg.data_root is not None, (
        "config carries no data_root: author it in the YAML before launch"
    )
    return cfg.model_copy(update={"run_id": cfg.run_id or f"t-{secrets.token_hex(4)}"})


def _enable_compilation_cache(paths: PretrainRunPaths) -> None:
    jax.config.update("jax_compilation_cache_dir", str(paths.compilation_cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 60)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
