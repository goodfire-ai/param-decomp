"""The LM decomposition composition root: wrapper YAML -> full SPEC-compliant run on a
vendored target.

    python -m param_decomp_lab.experiments.lm.run <wrapper.yaml>   # normally via pd-lm,
        # which stamps run_id into the workspace copy; re-running resumes in place

This is the LM I/O layer over the generic core engine
(`param_decomp.run.run_decomposition_training`): read the run YAML, build the target, feed
the per-step parquet token batch (`sample_batch`; the model embeds it), build the CEandKL /
CI-L0 / PGD / attn-patterns / slow `eval_fn`, then
call the engine. Process setup (`init_distributed`, the SIGTERM flag, the persistent XLA
compilation cache, HF http hardening), config pinning, and SLURM-requeue shutdown all live
here. The toy domains mirror this file under `experiments/{tms,resid_mlp}/run.py`.

Multi-process: launched one process per GPU under SLURM (`init_distributed`); every
process computes the same global schedule and contributes its local batch slice.
"""

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import wandb

    LogValue = float | wandb.plot.CustomChart
    LogRecord = Mapping[str, LogValue]

import fire
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from param_decomp.attn_patterns_eval import (
    accumulate_attn_patterns,
    attn_pattern_for,
    attn_patterns_log_entries,
    make_ci_attn_patterns_step,
    make_stochastic_attn_patterns_step,
)
from param_decomp.built_run import LAUNCH_CONFIG_FILENAME, BuiltRun, DataConfig
from param_decomp.configs import ResumeProvenance
from param_decomp.data import BatchSchedule, ShardServer, scan_shards
from param_decomp.eval import make_eval_step
from param_decomp.hf_http import configure_hf_http_retries
from param_decomp.lm import DecomposedModel
from param_decomp.log import setup_logger
from param_decomp.per_site_faith_eval import (
    PER_SITE_FAITH_TOP_K,
    make_per_site_faith_step,
    per_site_faith_scalars,
)
from param_decomp.run import (
    SlowEvalRenderer,
    install_sigterm_flag,
    run_decomposition_training,
    slow_eval_due,
)
from param_decomp.sharding import hsdp_mesh, init_distributed
from param_decomp.slow_eval import (
    IDENTITY_CI_ERROR_TOLERANCE,
    PositionCI,
    accumulate_position_ci,
    accumulate_site_reductions,
    compute_hidden_acts_metrics,
    compute_identity_ci_errors,
    eval_metrics_from_run_dir,
    make_position_ci_step,
    make_slow_eval_step,
    resolve_permutation_metrics,
    stochastic_hidden_acts_n_mask_samples,
)
from param_decomp.train import TrainState
from param_decomp_lab.experiments.lm.config import (
    load_config,
    load_run_dir_config,
)
from param_decomp_lab.experiments.lm.load_run import build_target


def _enable_persistent_compilation_cache(out_dir: Path) -> Path:
    """Cache compiled XLA executables to a shared-FS dir reused across runs/requeues.

    The ~24-min compile of the chunkwise step is keyed by HLO + backend + topology +
    jax/xla version, so a matching re-compile (requeue, or a fresh run at the same
    config+topology) loads from disk in seconds. The dir is a SIBLING of `runs/` (not
    per-run, not inside the immutable per-run workspace) so every run shares it; all 8N
    ranks point at the same shared-FS path. Only process 0 writes (jax gates the write on
    `process_id == 0` to avoid shared-FS write contention); every rank reads. Must run
    after `init_distributed` (the rank gate reads the distributed state) and before the
    first compile."""
    cache_dir = out_dir.parent / "xla_compilation_cache"
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 60.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return cache_dir


def _enable_hlo_dump(run_dir: Path) -> None:
    """Dump the step modules' optimized HLO + buffer assignment to `<run_dir>/hlo` (rank 0).

    Must run BEFORE `init_distributed` — XLA reads `XLA_FLAGS` when the backend initializes,
    so a later mutation is ignored. Rank-gated via `SLURM_PROCID` (read pre-jax-init, only to
    pick the writer — NOT to decide distributedness, which stays `runtime.dp`-driven) so a
    single rank writes; `xla_dump_hlo_module_re` filters to the big `*step*` modules to keep
    the dump to ~100s of MB. The buffer-assignment dump survives an exec-time OOM (compile
    completes first), so this is how we name the buffer that blows the allocator."""
    if os.environ.get("SLURM_PROCID", "0") != "0":
        return
    hlo_dir = run_dir / "hlo"
    hlo_dir.mkdir(parents=True, exist_ok=True)
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (
        f"{existing} --xla_dump_to={hlo_dir} --xla_dump_hlo_module_re=.*step.*"
    ).strip()


def _global_token_batch(local: np.ndarray, mesh: Mesh, global_batch: int) -> jax.Array:
    sharding = NamedSharding(mesh, P(("replicate", "fsdp")))
    return jax.make_array_from_process_local_data(sharding, local, (global_batch, local.shape[1]))


def assert_finetune_structural_compat(built: BuiltRun, prov: ResumeProvenance) -> None:
    """Fine-tune requires the parent's decomposition STRUCTURE to match the new config's:
    same sites (names + C) and same ci-fn arch. A changed C / layers / target / ci-fn is a
    different-shaped decomposition and is NOT a fine-tune (the parent's V/U + ci_fn would
    not load onto the new reference). Only LR / coeffs / eps / seq / batch / steps may
    change. Read from the parent's pinned launch config so the failure is a readable config
    diff, not an opaque orbax tree mismatch."""
    parent = load_run_dir_config(prov.parent_run_dir)
    parent_sites = tuple((s.name, s.C) for s in parent.target.sites)
    new_sites = tuple((s.name, s.C) for s in built.target.sites)
    assert parent_sites == new_sites, (
        f"fine-tune sites mismatch: parent {parent_sites} != new {new_sites}"
    )
    assert parent.ci_fn == built.ci_fn, (
        f"fine-tune ci-fn arch mismatch: parent {parent.ci_fn} != new {built.ci_fn}"
    )


def train(
    built: BuiltRun,
    lm: DecomposedModel,
    mesh: Mesh,
) -> None:
    """The LM composition over the generic engine: a parquet `sample_batch` (the per-step
    token batch the model embeds) and the CEandKL / CI-L0 / PGD / attn-patterns `eval_fn`."""
    data = built.data
    assert isinstance(data, DataConfig), "train() is the LM (parquet) data path"
    n_proc = jax.process_count()
    # Pure HSDP: the batch shards over the FULL mesh (both axes), so it must tile the full
    # device count, and the per-rank batch is B/N. Constraint: B >= N (per-rank >= 1).
    n_dev = mesh.devices.size
    assert data.global_batch % n_dev == 0, (data.global_batch, n_dev)
    assert data.global_batch >= n_dev, (
        f"global batch {data.global_batch} < device count {n_dev}: per-rank batch must be >= 1"
    )
    is_main = jax.process_index() == 0

    key = random.PRNGKey(built.pd.seed)
    _, _, run_key = random.split(key, 3)

    schedule = BatchSchedule(scan_shards(data.dir), data.global_batch, built.pd.seed)
    server = ShardServer(schedule, data.seq_len, jax.process_index(), n_proc)
    # Each process (node) owns all its local devices; its per-process batch splits across them.
    assert server.per_process % jax.local_device_count() == 0, (
        server.per_process,
        jax.local_device_count(),
    )

    def sample_batch(step: int) -> jax.Array:
        return _global_token_batch(server.local_batch(step), mesh, data.global_batch)

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
    )


def _make_lm_eval_fn(
    built: BuiltRun,
    lm: DecomposedModel,
    run_key: PRNGKeyArray,
    mesh: Mesh,
    n_proc: int,
    is_main: bool,
) -> "Callable[[TrainState, int], LogRecord]":
    """The LM in-loop eval pass closure (CEandKL / CI-L0 / PGD / attn-patterns), keyed
    deterministically off `(run_key, now_step)` so it is bit-identical to the pre-engine
    inline loop. Mirrors the torch `eval_split: train` stream: an independent reader over
    the SAME corpus (own seed), advanced one block of `n_steps` batches per eval pass."""
    eval = built.eval
    assert eval is not None
    pd = built.pd
    data = built.data
    assert isinstance(data, DataConfig)
    eval_schedule = BatchSchedule(scan_shards(data.dir), eval.batch_size, pd.seed + 1)
    eval_server = ShardServer(eval_schedule, data.seq_len, jax.process_index(), n_proc)
    assert eval_server.per_process % jax.local_device_count() == 0, (
        eval_server.per_process,
        jax.local_device_count(),
    )
    co = built.runtime.compiler_options
    eval_step_fn = make_eval_step(
        lm,
        eval.rounding_threshold,
        eval.l0_ci_alive_threshold,
        eval.l0_groups,
        eval.pgd,
        mesh,
        co,
    )
    attn_steps: dict[str, Any] = {}
    if eval.attn_patterns is not None:
        pattern_fn = attn_pattern_for(lm)
        if eval.attn_patterns.ci_masked:
            attn_steps["CIMaskedAttnPatternsReconLoss"] = make_ci_attn_patterns_step(
                lm, pattern_fn, co
            )
        if eval.attn_patterns.stochastic:
            attn_steps["StochasticAttnPatternsReconLoss"] = make_stochastic_attn_patterns_step(
                lm, pattern_fn, eval.attn_patterns.stochastic_n_mask_samples, co
            )

    slow_eval_step = make_slow_eval_step(
        lm, eval.density_ci_alive_threshold, eval.density_heatmap_n_bins, co
    )
    slow_renderer = SlowEvalRenderer(is_main)
    # The CI-heatmap / permutation / UV / identity-error metrics read off the run's typed
    # `eval.metrics` (re-validated from the pinned launch config: the trainer's `EvalConfig`
    # drops the raw metric list). The launch config is pinned before train().
    run_eval_metrics = eval_metrics_from_run_dir(built.run.run_dir)
    perm_spec = resolve_permutation_metrics(lm.site_names, run_eval_metrics)
    hidden_acts_n_mask_samples = stochastic_hidden_acts_n_mask_samples(run_eval_metrics)
    want_position_ci = perm_spec.any_plots or perm_spec.any_identity_error
    position_ci_step = make_position_ci_step(lm, co) if want_position_ci else None

    site_names = lm.site_names
    per_site_faith_step = make_per_site_faith_step(lm, co)
    # ‖W_s‖²_F is constant over training — computed once (lazily, from the first eval's own
    # components pytree so shape/sharding are exact) via the same step on a zeroed vu.
    frozen_weight_sq: dict[str, float] | None = None

    def eval_fn(state: TrainState, now_step: int) -> "LogRecord":
        nonlocal frozen_weight_sq
        eval_pass_index = now_step // eval.every
        # uniform-average of per-batch scalars; mean-safe vs torch's accumulate-then-
        # compute() ONLY because every emitted key is a per-batch reduction that torch also
        # averages across batches AND eval batches are uniform (B, T). See eval.py's module
        # docstring for the per-key parity argument (cites SPEC S8/D2).
        metric_sums: dict[str, jax.Array] = {}
        eval_batches: list[jax.Array] = []
        # No per-rank SIGTERM abandon inside this loop — it would desync the collective
        # device->host gathers below; the engine gates pass entry on cross-rank consensus.
        for j in range(eval.n_steps):
            eval_tokens = _global_token_batch(
                eval_server.local_batch(eval_pass_index * eval.n_steps + j),
                mesh,
                eval.batch_size,
            )
            eval_batches.append(eval_tokens)
            # fold values >= pd.steps never collide with the train step keys
            eval_key = random.fold_in(run_key, pd.steps + eval_pass_index * eval.n_steps + j)
            eval_metrics = eval_step_fn(lm, state.components, state.ci_fn, eval_tokens, eval_key)
            for k, v in eval_metrics.items():
                metric_sums[k] = metric_sums.get(k, jnp.zeros(())) + v
        eval_record: dict[str, LogValue] = {
            f"eval/{k}": float(v) / eval.n_steps for k, v in metric_sums.items()
        }

        # Per-site faithfulness observability: the per-site ‖Δ_s‖²_F the global
        # FaithfulnessLoss reduces away. Collective (an all-reduce per site), so it runs in
        # lockstep on every rank; the wandb bar chart is rank-0-only below.
        delta_sq = {s: float(v) for s, v in per_site_faith_step(lm, state.components).items()}
        if frozen_weight_sq is None:
            zero_vu = jax.tree.map(jnp.zeros_like, state.components)
            frozen_weight_sq = {s: float(v) for s, v in per_site_faith_step(lm, zero_vu).items()}
            assert all(w > 0.0 for w in frozen_weight_sq.values()), "zero-norm frozen site"
        abs_frob = {s: delta_sq[s] ** 0.5 for s in site_names}
        rel_frob = {s: abs_frob[s] / frozen_weight_sq[s] ** 0.5 for s in site_names}
        eval_record |= per_site_faith_scalars(rel_frob, abs_frob, PER_SITE_FAITH_TOP_K)

        for class_name, attn_step in attn_steps.items():
            # token-weighted (Σ sum_kl / Σ n), NOT the uniform per-batch average above — KL
            # is summed over distributions, divided by their count.
            attn_key = random.fold_in(run_key, 2 * pd.steps + eval_pass_index)
            reductions = accumulate_attn_patterns(
                attn_step, lm, state.components, state.ci_fn, eval_batches, attn_key
            )
            eval_record |= {
                f"eval/loss/{k}": v
                for k, v in attn_patterns_log_entries(class_name, reductions).items()
            }
        slow_due = slow_eval_due(now_step, eval.every, eval.slow_every, eval.slow_on_first_step)
        if eval_batches and slow_due:
            # SLOW/PLOT TIER (SPEC S28/S29). The COLLECTIVE part runs in lockstep on every
            # rank — `accumulate_site_reductions` / `compute_hidden_acts_metrics` pull
            # C-sharded reductions to numpy, whose `np.asarray` triggers the all-gather all
            # ranks must join. It reuses the eval batches already loaded above. The
            # hidden-acts scalars ride the live `_step` axis through `eval_record`; the
            # figures' pure-host render + wandb.log happen OFF the loop on rank 0.
            site_reductions = accumulate_site_reductions(
                slow_eval_step, lm, state.ci_fn, eval_batches, eval.slow_n_batches_accum
            )
            hidden_acts_key = random.fold_in(run_key, 3 * pd.steps + eval_pass_index)
            hidden_acts = compute_hidden_acts_metrics(
                lm, state, eval_batches, hidden_acts_n_mask_samples, hidden_acts_key, co
            )
            eval_record |= {f"eval/slow/loss/{k}": v for k, v in hidden_acts.items()}
            # The position-CI all-gather is ALSO collective (every rank joins it), gated on
            # the config naming a CI-heatmap / permutation / identity-error metric. The
            # heatmap FIGURES render off-loop on rank 0; the IdentityCIError SCALARS log
            # synchronously on the live `_step` (cheap + must stay `_step`-monotonic).
            position_ci: dict[str, PositionCI] | None = None
            if position_ci_step is not None:
                position_ci = accumulate_position_ci(
                    position_ci_step, lm, state.ci_fn, eval_batches
                )
                identity_ci_errors = compute_identity_ci_errors(
                    perm_spec, position_ci, IDENTITY_CI_ERROR_TOLERANCE
                )
                eval_record |= {f"eval/slow/{k}": v for k, v in identity_ci_errors.items()}
            # `UVPlots` needs the C-sharded V/U gathered to host (collective `np.asarray`).
            # This NAIVE gather is small-scale-only — it OOMs / breaks at production C BY
            # DESIGN (per Oli); gated on the config naming UVPlots so it costs nothing
            # otherwise. The component column order reuses the position-CI permutation.
            components: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
            if perm_spec.want_uv_plots:
                components = {
                    name: (np.asarray(V), np.asarray(U))
                    for name, (V, U) in state.components.vu.items()
                }
            slow_renderer.submit(site_reductions, perm_spec, position_ci, components, now_step)
        if is_main and built.run.wandb is not None:
            # torch CI_L0.compute() emitted a per-layer L0 bar chart alongside the scalars;
            # rebuild it host-side from the `eval/l0/<thr>_<site|group>` scalars already in
            # the record (the jitted eval can't construct wandb objects).
            import wandb

            l0_prefix = f"eval/l0/{eval.l0_ci_alive_threshold}_"
            eval_record["eval/l0/bar_chart"] = wandb.plot.bar(
                wandb.Table(
                    columns=["layer", "l0"],
                    data=[
                        [k.removeprefix(l0_prefix), v]
                        for k, v in eval_record.items()
                        if k.startswith(l0_prefix)
                    ],
                ),
                "layer",
                "l0",
                title=f"L0_{eval.l0_ci_alive_threshold}",
            )
            # All-site per-site relative Frobenius, site-labeled — the rich per-site view the
            # rank-indexed rel_frob_top* scalars can't carry (their keys are stable but drop
            # site identity). Descending so the worst sites lead.
            eval_record["eval/faith/rel_frob_bar_chart"] = wandb.plot.bar(
                wandb.Table(
                    columns=["site", "rel_frob"],
                    data=[
                        [s, v]
                        for s, v in sorted(rel_frob.items(), key=lambda kv: kv[1], reverse=True)
                    ],
                ),
                "site",
                "rel_frob",
                title="per-site ‖Δ‖_F / ‖W‖_F",
            )
        if is_main:
            headline = {
                k: eval_record[f"eval/{k}"]
                for k in ("ce_kl/kl_ci_masked", "ce_kl/ce_difference_ci_masked")
            }
            headline["faith/rel_frob_top1"] = eval_record["eval/faith/rel_frob_top1"]
            print(f"[eval @ {now_step}] {headline}", flush=True)
        return eval_record

    return eval_fn


def _pin_config_copy(run_dir: Path, name: str, source: Path) -> None:
    """First run copies `source` into the run dir; resumes byte-compare against it."""
    copy = run_dir / name
    if copy.exists():
        assert copy.read_text() == source.read_text(), (
            f"{copy} differs from {source} — refusing to resume with a changed config"
        )
    else:
        copy.write_text(source.read_text())


def main(config: Path, run_id: str) -> None:
    config = Path(config)
    built, _raw_cfg = load_config(config, run_id)

    install_sigterm_flag()
    _enable_hlo_dump(built.run.run_dir)
    init_distributed(built.runtime.dp)
    # Harden the cold-cache HF weight load against the 8N-rank startup burst before any
    # per-rank Hub call (no-op when huggingface_hub is absent / cache is pre-warmed).
    configure_hf_http_retries()
    mesh = hsdp_mesh(built.runtime.tp)

    if built.run.resume_provenance is not None:
        assert_finetune_structural_compat(built, built.run.resume_provenance)

    cache_dir = _enable_persistent_compilation_cache(built.run.out_dir)

    is_main = jax.process_index() == 0
    if is_main:
        cache_dir.mkdir(parents=True, exist_ok=True)
        built.run.run_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(built.run.run_dir / "logs.log")
        _pin_config_copy(built.run.run_dir, LAUNCH_CONFIG_FILENAME, config)
        print(f"persistent XLA compilation cache: {cache_dir}", flush=True)
        site_kind_counts: dict[str, int] = {}
        for s in built.target.sites:
            kind = s.name.rsplit(".", 1)[-1]
            site_kind_counts[kind] = site_kind_counts.get(kind, 0) + 1
        site_summary = ", ".join(f"{k}×{n}" for k, n in sorted(site_kind_counts.items()))
        assert isinstance(built.data, DataConfig)
        print(
            f"run {built.run.run_name} | {mesh.devices.size} GPU / {jax.process_count()} proc | "
            f"B={built.data.global_batch} seq={built.data.seq_len} "
            f"sites={len(built.target.sites)} [{site_summary}] steps={built.pd.steps}",
            flush=True,
        )

    # The `lm` (an eqx model) IS the frozen target — it carries the frozen weights as fields,
    # so the function-table era's separate `frozen` object is gone.
    lm, _vocab_size = build_target(built, mesh)

    train(built, lm, mesh)

    if jax.process_count() > 1:
        import jax.experimental.multihost_utils as mhu

        mhu.sync_global_devices("train_done")
        jax.distributed.shutdown()


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
