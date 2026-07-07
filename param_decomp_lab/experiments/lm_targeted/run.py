"""The targeted-PD (tPD) LM composition root: a fixed-prompt TARGET stream + a broad parquet
NON-TARGET stream over the same generic engine.

    python -m param_decomp_lab.experiments.lm_targeted.run <config> --run-id <id>

The LM analog of the toy tPD runs (`experiments/{tms,resid_mlp}/run.py`). Mirrors
`experiments.lm.run` (init_distributed / XLA cache / mesh / `build_target` / SIGTERM /
config-pin) but feeds the engine TWO streams: the TARGET `sample_batch` samples the fixed
prompt pool (`data.prompts_file`, tokenized once), and the NON-TARGET `NontargetPass` reads
the normal parquet path built from `nontarget.data`. The delta component is forced on over
the non-target pass (SPEC §11). Eval reuses the plain-LM target-distribution `eval_fn`
(`experiments.lm.run._make_lm_eval_fn`); tPD-specific eval metrics are out of scope here.
"""

from collections.abc import Callable
from pathlib import Path

import fire
import jax
from jax import random
from jax.sharding import Mesh

from param_decomp.built_run import BuiltRun, DataConfig
from param_decomp.data import BatchSchedule, ShardServer, scan_shards
from param_decomp.hf_http import configure_hf_http_retries
from param_decomp.lm import DecomposedModel
from param_decomp.log import setup_logger
from param_decomp.run import NontargetPass, install_sigterm_flag, run_decomposition_training
from param_decomp.sharding import hsdp_mesh, init_distributed
from param_decomp_lab.experiments.config import build_nontarget_loss_metrics
from param_decomp_lab.experiments.lm.load_run import build_target
from param_decomp_lab.experiments.lm.run import (
    _enable_hlo_dump,
    _enable_persistent_compilation_cache,
    _global_token_batch,
    _make_lm_eval_fn,
    _pin_config_copy,
    assert_finetune_structural_compat,
)
from param_decomp_lab.experiments.lm_targeted.config import (
    LMTargetedExperimentConfig,
    load_targeted_config,
    nontarget_parquet_dir,
)
from param_decomp_lab.experiments.lm_targeted.data import (
    load_prompt_tokens,
    make_prompt_sample_batch,
)


def _nontarget_sample_batch(
    cfg: LMTargetedExperimentConfig, seq_len: int, mesh: Mesh
) -> Callable[[int], jax.Array]:
    """The parquet NON-TARGET stream `sample_batch` (the normal LM path, built from
    `nontarget.data` at `nontarget.batch_size * world`)."""
    n_proc = jax.process_count()
    n_dev = mesh.devices.size
    nontarget_global_batch = cfg.nontarget.batch_size * n_proc
    parquet_dir = nontarget_parquet_dir(cfg)
    assert nontarget_global_batch % n_dev == 0, (nontarget_global_batch, n_dev)
    assert nontarget_global_batch >= n_dev, (
        f"non-target global batch {nontarget_global_batch} < device count {n_dev}"
    )
    schedule = BatchSchedule(scan_shards(parquet_dir), nontarget_global_batch, cfg.pd.seed + 2)
    server = ShardServer(schedule, seq_len, jax.process_index(), n_proc)
    assert server.per_process % jax.local_device_count() == 0, (
        server.per_process,
        jax.local_device_count(),
    )

    def sample_batch(step: int) -> jax.Array:
        return _global_token_batch(server.local_batch(step), mesh, nontarget_global_batch)

    return sample_batch


def train(
    built: BuiltRun, cfg: LMTargetedExperimentConfig, lm: DecomposedModel, mesh: Mesh
) -> None:
    """The tPD LM composition: a fixed-prompt TARGET `sample_batch`, a parquet NON-TARGET
    `NontargetPass`, and the plain-LM target-distribution `eval_fn`."""
    data = built.data
    assert isinstance(data, DataConfig)
    n_proc = jax.process_count()
    n_dev = mesh.devices.size
    assert data.global_batch % n_dev == 0, (data.global_batch, n_dev)
    assert data.global_batch >= n_dev, (
        f"global batch {data.global_batch} < device count {n_dev}: per-rank batch must be >= 1"
    )
    is_main = jax.process_index() == 0

    key = random.PRNGKey(built.pd.seed)
    _, _, run_key = random.split(key, 3)

    # TARGET stream: the fixed prompt pool, tokenized once at build time.
    assert cfg.data.prompts_file is not None
    prompt_tokens, recon_positions = load_prompt_tokens(
        cfg.data.prompts_file, cfg.data.tokenizer_name, cfg.data.max_seq_len
    )
    sample_batch = make_prompt_sample_batch(
        prompt_tokens, data.global_batch, mesh, seed=built.pd.seed
    )

    # NON-TARGET stream: the broad parquet distribution (SPEC §11).
    nontarget_sample_batch = _nontarget_sample_batch(cfg, cfg.nontarget.data.max_seq_len, mesh)
    nontarget = NontargetPass(
        sample_batch=nontarget_sample_batch,
        loss_metrics=build_nontarget_loss_metrics(
            built.pd.loss_metrics, cfg.nontarget.impmin_coeff_ratio
        ),
    )

    eval_fn = None
    eval_every = built.pd.steps + 1  # unreachable cadence when eval is disabled
    if built.eval is not None:
        assert built.eval.every % built.cadence.train_log_every == 0, (
            "eval must land on a train-log step: the tok/s window resets after eval, so a "
            "mid-window eval would corrupt the next step-time estimate"
        )
        assert built.eval.slow_every % built.eval.every == 0, (
            "slow_every must be a multiple of every: the slow tier reuses the fast eval "
            "pass's batches, so it can only fire on a fast-eval step"
        )
        eval_every = built.eval.every
        eval_fn = _make_lm_eval_fn(built, lm, run_key, mesh, n_proc, is_main)

    run_decomposition_training(
        pd=built.pd,
        cadence=built.cadence,
        run=built.run,
        lm=lm,
        ci_fn=built.ci_fn,
        data=data,
        remat_recon_forwards=built.runtime.remat_recon_forwards,
        remat_ci_fn=built.runtime.remat_ci_fn,
        ascend_replicate=built.runtime.ascend_replicate,
        compiler_options=built.runtime.compiler_options,
        profile=built.runtime.launch_env.profile,
        sample_batch=sample_batch,
        eval_fn=eval_fn,
        eval_every=eval_every,
        mesh=mesh,
        nontarget=nontarget,
        recon_positions=recon_positions,
    )


def main(config: Path, run_id: str) -> None:
    config = Path(config)
    built, cfg = load_targeted_config(config, run_id)

    install_sigterm_flag()
    _enable_hlo_dump(built.run.run_dir)
    init_distributed(built.runtime.dp)
    configure_hf_http_retries()
    mesh = hsdp_mesh()

    if built.run.resume_provenance is not None:
        assert_finetune_structural_compat(built, built.run.resume_provenance)

    cache_dir = _enable_persistent_compilation_cache(built.run.out_dir)

    is_main = jax.process_index() == 0
    if is_main:
        cache_dir.mkdir(parents=True, exist_ok=True)
        built.run.run_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(built.run.run_dir / "logs.log")
        _pin_config_copy(built.run.run_dir, "config.yaml", config)
        print(f"persistent XLA compilation cache: {cache_dir}", flush=True)
        assert isinstance(built.data, DataConfig)
        print(
            f"run {built.run.run_name} | {mesh.devices.size} GPU / {jax.process_count()} proc | "
            f"target B={built.data.global_batch} seq={built.data.seq_len} "
            f"nontarget B={cfg.nontarget.batch_size * jax.process_count()} "
            f"seq={cfg.nontarget.data.max_seq_len} sites={len(built.target.sites)} "
            f"steps={built.pd.steps}",
            flush=True,
        )

    lm, _vocab_size = build_target(built, mesh)

    train(built, cfg, lm, mesh)

    if jax.process_count() > 1:
        import jax.experimental.multihost_utils as mhu

        mhu.sync_global_devices("train_done")
        jax.distributed.shutdown()


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
