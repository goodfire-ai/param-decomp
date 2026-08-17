"""The targeted (tPD, SPEC §11) LM composition root: run YAML -> two-pass run on a
vendored target.

    python -m param_decomp.experiments.lm.run_targeted <config.yaml> --data-root <path>
        # a launcher that owns the run identity pins the config and passes --run-id

`training.py`'s twin over the targeted engine entry: the TARGET stream is a fixed prompt
pool (`prompts:`, tokenized once at startup on every rank, sampled at `pd.batch_size`
per step, run unpadded at the pool's own prompt length), the broad `data:` stream
becomes the NON-TARGET pass at `nontarget.batch_size` and its own seq_len. Process
setup, config pinning, and SLURM-requeue shutdown are the shared helpers from
`training.py`.
"""

from pathlib import Path
from typing import cast

import jax
import yaml
from jax import random
from jax.sharding import Mesh

from param_decomp.core import placement
from param_decomp.core.built_run import LAUNCH_CONFIG_FILENAME
from param_decomp.core.model import DecomposedModel, Positioned
from param_decomp.core.run import (
    MetricsSink,
    install_sigterm_flag,
    run_targeted_decomposition_training,
)
from param_decomp.core.sharding import assert_inline_topology, hsdp_mesh, init_distributed
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.arithmetic_probe import PromptEncoder
from param_decomp.experiments.lm.config import (
    LMTargetedExperimentConfig,
    build_targeted_experiment_config,
)
from param_decomp.experiments.lm.eval_operations import global_token_batch, make_lm_evaluation
from param_decomp.experiments.lm.hf_http import configure_hf_http_retries
from param_decomp.experiments.lm.load_run import build_target
from param_decomp.experiments.lm.resolved import (
    AnyLMTargetConfig,
    LlamaSimpleMLPTargetConfig,
    LMTargetedRun,
    TargetConfig,
)
from param_decomp.experiments.lm.targeted_data import build_prompt_pool, pool_batch
from param_decomp.experiments.lm.training import (
    enable_hlo_dump,
    enable_persistent_compilation_cache,
    pin_config_copy,
)
from param_decomp.infra.dataset_store import read_dataset_meta
from param_decomp.infra.run_files import generate_run_id
from param_decomp.pretrain.batch_data import BatchSchedule, ShardServer, scan_shards
from param_decomp.targets.glu_transformer import hf_snapshot_dir


def _pool_tokenizer(target: AnyLMTargetConfig, dataset_tokenizer_name: str) -> PromptEncoder:
    """The tokenizer the prompt pool encodes with — necessarily the SAME vocabulary the
    model and the broad stream use. An HF-family target reads its local snapshot (the
    weights load already staged it); the lab-pretrained target names its tokenizer via
    the dataset's own meta."""
    from transformers import AutoTokenizer

    match target:
        case TargetConfig():
            loaded = AutoTokenizer.from_pretrained(
                str(hf_snapshot_dir(target.model_name)), local_files_only=True
            )
        case LlamaSimpleMLPTargetConfig():
            loaded = AutoTokenizer.from_pretrained(dataset_tokenizer_name)
    return cast(PromptEncoder, cast(object, loaded))


def train_targeted(
    built: LMTargetedRun,
    cfg: LMTargetedExperimentConfig,
    eval_config: EvalConfig | None,
    model: DecomposedModel,
    mesh: Mesh,
) -> None:
    """The targeted LM composition over the engine: the prompt-pool TARGET seam, the
    parquet NON-TARGET seam, and the same domain-bound eval operations as the plain root
    (forward-only diagnostics on the broad eval split)."""
    data = built.data
    train_meta = read_dataset_meta(data.dir)
    eval_meta = read_dataset_meta(data.eval_dir)
    assert train_meta == eval_meta, (
        f"train and eval datasets disagree — {data.dir} is {train_meta}, {data.eval_dir} "
        f"is {eval_meta}. A holdout tokenized differently or at another seq_len makes "
        "every eval number incomparable to the training loss it is read against."
    )
    seq_len = train_meta.seq_len
    n_proc = jax.process_count()
    n_dev = mesh.devices.size
    is_main = jax.process_index() == 0

    # Both streams shard over the full mesh; each global batch must tile it (T2).
    target_batch = built.pd.batch_size
    nontarget_batch = cfg.nontarget.batch_size
    for name, batch in (("pd.batch_size", target_batch), ("nontarget.batch_size", nontarget_batch)):
        assert batch % n_dev == 0 and batch >= n_dev, (
            f"{name} {batch} must be a positive multiple of the device count {n_dev}"
        )

    tokenizer = _pool_tokenizer(built.target, train_meta.tokenizer_name)
    pool = build_prompt_pool(cfg.prompts, tokenizer)
    n_prompts, prompt_len = pool.tokens.shape
    if is_main:
        print(f"target prompt pool: {n_prompts} prompts x {prompt_len} positions", flush=True)

    key = random.PRNGKey(built.pd.seed)
    _, _, run_key = random.split(key, 3)

    schedule = BatchSchedule(scan_shards(data.dir), nontarget_batch, built.pd.seed)
    server = ShardServer(schedule, seq_len, jax.process_index(), n_proc)
    assert server.per_process % jax.local_device_count() == 0, (
        server.per_process,
        jax.local_device_count(),
    )

    def pool_global_batch(seed: int, step: int, batch: int) -> jax.Array:
        per_process = batch // n_proc
        rows = pool_batch(pool, seed, step, batch)
        local = rows[jax.process_index() * per_process :][:per_process]
        return global_token_batch(local, mesh, batch)

    def sample_target_batch(step: int) -> jax.Array:
        return pool_global_batch(built.pd.seed, step, target_batch)

    def sample_nontarget_batch(step: int) -> jax.Array:
        return global_token_batch(server.local_batch(step), mesh, nontarget_batch)

    sink = MetricsSink.for_run(built.run, is_main)
    evaluation = None
    if eval_config is not None:
        assert eval_config.every % built.cadence.train_log_every == 0, (
            "eval must land on a train-log step: the tok/s window resets after eval, so a "
            "mid-window eval would corrupt the next step-time estimate"
        )
        eval_target_batch = eval_config.batch_size

        def eval_target_pool_batches(pass_index: int) -> list[jax.Array]:
            """The eval pass's target stream: training's pure `(seed, step)` pool sampler on
            the `seed + 1` stream, so an eval never scores the rows the step just trained."""
            n_batches = eval_config.n_steps
            return [
                pool_global_batch(built.pd.seed + 1, pass_index * n_batches + j, eval_target_batch)
                for j in range(n_batches)
            ]

        evaluation = make_lm_evaluation(
            built,
            eval_config,
            model,
            run_key,
            mesh,
            n_proc,
            sink,
            cfg.runtime.compiler_options,
            target_pool_batches_for=eval_target_pool_batches,
        )

    run_targeted_decomposition_training(
        pd=built.pd,
        nontarget=cfg.nontarget,
        cadence=built.cadence,
        run=built.run,
        model=model,
        ci_fn=built.ci_fn,
        positions=Positioned(n_positions=prompt_len),
        remat_recon_forwards=cfg.runtime.remat_recon_forwards,
        remat_ci_fn=cfg.runtime.remat_ci_fn,
        ascend_replicate=cfg.runtime.ascend_replicate,
        compiler_options=cfg.runtime.compiler_options,
        sample_target_batch=sample_target_batch,
        sample_nontarget_batch=sample_nontarget_batch,
        evaluation=evaluation,
        sink=sink,
        mesh=mesh,
        placement_rules=placement.from_config(cfg.runtime.sharding, mesh, model.sites),
    )


def main(config: Path, data_root: Path, run_id: str | None = None) -> None:
    config = Path(config)
    data_root = Path(data_root)
    if run_id is None:
        # Ad-hoc run-here invocation: mint a fresh identity; `pin_config_copy` below
        # stages the config into the run dir exactly as the launcher would.
        run_id = generate_run_id("param_decomp")
    raw = yaml.safe_load(config.read_text())
    cfg = LMTargetedExperimentConfig.model_validate(raw)
    built = build_targeted_experiment_config(cfg, run_id, data_root)
    runtime = cfg.runtime

    install_sigterm_flag()
    enable_hlo_dump(built.run.run_dir)
    if runtime.distributed:
        init_distributed(runtime.dp, runtime.gpus_per_node)
    else:
        assert_inline_topology(runtime.dp)
    configure_hf_http_retries()
    mesh = hsdp_mesh(runtime.tp, runtime.gpus_per_node)

    cache_dir = enable_persistent_compilation_cache(built.run.out_dir)

    is_main = jax.process_index() == 0
    if is_main:
        cache_dir.mkdir(parents=True, exist_ok=True)
        built.run.run_dir.mkdir(parents=True, exist_ok=True)
        pin_config_copy(built.run.run_dir, LAUNCH_CONFIG_FILENAME, config)
        print(f"persistent XLA compilation cache: {cache_dir}", flush=True)
        print(
            f"targeted run {built.run.run_name} | {mesh.devices.size} GPU / "
            f"{jax.process_count()} proc | target B={built.pd.batch_size} "
            f"nontarget B={cfg.nontarget.batch_size} "
            f"seq={read_dataset_meta(built.data.dir).seq_len} "
            f"sites={len(built.target.sites)} steps={built.pd.steps}",
            flush=True,
        )

    model, _vocab_size = build_target(built.target, mesh, data_root)

    train_targeted(built, cfg, cfg.eval, model, mesh)

    if jax.process_count() > 1:
        import jax.experimental.multihost_utils as mhu

        mhu.sync_global_devices("train_done")
        jax.distributed.shutdown()
