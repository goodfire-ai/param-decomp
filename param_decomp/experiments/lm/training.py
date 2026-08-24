"""Run ordinary language-model parameter-decomposition training from one config.

    python -m param_decomp.experiments.lm.run <config.yaml>   # normally invoked by a
        # submitter that pins the config as runs/<id>/launch_config.yaml and passes
        # --run-id; re-running resumes in place

This is the LM I/O layer over the generic core engine
(`param_decomp.core.run.run_decomposition_training`): read the run YAML, build the target, feed
the per-step parquet token batch (`sample_batch`; the model embeds it), bind the CEandKL /
CI-L0 / PGD / attn-patterns / slow eval operations, then
call the engine. Process setup (`initialize_topology`, the SIGTERM flag, the persistent XLA
compilation cache, HF http hardening), config pinning, and SLURM-requeue shutdown all live
here. The toy domains mirror this file under `experiments/{tms,resid_mlp}/run.py`.

The config declares the logical `replicate×fsdp×tp` mesh. The launch boundary separately
supplies the process-local device count used for JAX distributed bring-up.
"""

import os
from pathlib import Path

import jax
from jax import random
from jax.sharding import Mesh

from param_decomp.core.built_run import LAUNCH_CONFIG_FILENAME
from param_decomp.core.configs import ResumeProvenance
from param_decomp.core.log import setup_logger
from param_decomp.core.model import PlacedModel, Positioned
from param_decomp.core.run import (
    JaxProfilerTrace,
    MetricsSink,
    NsightCaptureWindow,
    ProfilingMode,
    install_sigterm_flag,
    run_decomposition_training,
)
from param_decomp.core.sharding import (
    data_parallel_size,
    hsdp_mesh,
    initialize_topology,
    local_data_parallel_size,
)
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.config import load_config
from param_decomp.experiments.lm.eval_operations import global_token_batch, make_lm_evaluation
from param_decomp.experiments.lm.load_run import build_target
from param_decomp.experiments.lm.resolved import LMRun
from param_decomp.experiments.lm.runtime import (
    AdHocProfiling,
    NsightSystemsProfiling,
    ProfilingConfig,
    ProfilingDisabled,
    RuntimeConfig,
)
from param_decomp.infra.dataset_store import read_dataset_meta
from param_decomp.infra.run_files import generate_run_id
from param_decomp.pretrain.batch_data import BatchSchedule, ShardServer, scan_shards


def enable_persistent_compilation_cache(authored_dir: Path) -> Path:
    """Cache compiled XLA executables in the config-authored dir, reused across
    runs/requeues.

    The ~24-min compile of the chunkwise step is keyed by HLO + backend + topology +
    jax/xla version, so a matching re-compile (requeue, or a fresh run at the same
    config+topology) loads from disk in seconds. `authored_dir` is
    `runtime.compilation_cache_dir`, `~`-expanded here so the seats' per-user authoring
    (see that field's description for why sharing across users breaks) lands in the
    running user's home. Only process 0 writes; every rank reads. Must run after
    `initialize_topology` and before the first compile."""
    cache_dir = authored_dir.expanduser()
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 60.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return cache_dir


def engine_profiling(config: ProfilingConfig) -> ProfilingMode | None:
    """Lower the authored `runtime.profiling` arm to the engine's typed profiling data —
    the engine reads no environment; profiling arrives as an argument like everything
    else. The nsight arm's `version` stays launcher-side (it names the `nsys` executable,
    not engine behavior)."""
    match config:
        case ProfilingDisabled():
            return None
        case AdHocProfiling(steps=steps):
            return JaxProfilerTrace(steps=steps)
        case NsightSystemsProfiling(warmup_steps=warmup_steps, capture_steps=capture_steps):
            return NsightCaptureWindow(warmup_steps=warmup_steps, capture_steps=capture_steps)


def enable_hlo_dump(run_dir: Path) -> None:
    """Dump step HLO protos, optimized text, and buffer assignment to `<run_dir>/hlo` (rank 0).

    Must run BEFORE `initialize_topology` — XLA reads `XLA_FLAGS` when the backend initializes,
    so a later mutation is ignored. Rank-gated via the generic `PD_RANK`
    hint (read pre-jax-init only to pick the writer, never to decide topology); the launcher
    exports it per rank, absent = single
    process) so a single rank writes; `xla_dump_hlo_module_re` filters to the big `*step*` modules to keep
    the dump to ~100s of MB. The buffer-assignment dump survives an exec-time OOM (compile
    completes first), so this is how we name the buffer that blows the allocator."""
    if os.environ.get("PD_RANK", "0") != "0":
        return
    hlo_dir = run_dir / "hlo"
    hlo_dir.mkdir(parents=True, exist_ok=True)
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (
        f"{existing} --xla_dump_to={hlo_dir} --xla_dump_hlo_module_re=.*step.* "
        "--xla_dump_hlo_as_proto"
    ).strip()


def assert_finetune_structural_compat(
    built: LMRun, prov: ResumeProvenance, data_root: Path
) -> None:
    """Fine-tune requires the parent's decomposition STRUCTURE to match the new config's:
    same sites (names + C) and same ci-fn arch. A changed C / layers / target / ci-fn is a
    different-shaped decomposition and is NOT a fine-tune (the parent's V/U + ci_fn would
    not load onto the new reference). Only LR / coeffs / eps / seq / batch / steps may
    change. Read from the parent's pinned launch config so the failure is a readable config
    diff, not an opaque orbax tree mismatch."""
    parent, _ = load_config(
        prov.parent_run_dir / LAUNCH_CONFIG_FILENAME, prov.parent_run_dir.name, data_root
    )
    parent_sites = tuple((s.name, s.C) for s in parent.target.sites)
    new_sites = tuple((s.name, s.C) for s in built.target.sites)
    assert parent_sites == new_sites, (
        f"fine-tune sites mismatch: parent {parent_sites} != new {new_sites}"
    )
    assert parent.ci_fn == built.ci_fn, (
        f"fine-tune ci-fn arch mismatch: parent {parent.ci_fn} != new {built.ci_fn}"
    )


def train(
    built: LMRun,
    runtime: RuntimeConfig,
    eval_config: EvalConfig | None,
    model: PlacedModel,
    mesh: Mesh,
) -> None:
    """The LM composition over the generic engine: a parquet `sample_batch` (the per-step
    token batch the model embeds) and domain-bound CEandKL / CI-L0 / PGD / attention operations.

    `runtime` rides alongside the bundle rather than inside it: it is the LM's substrate,
    and core reads none of it — this root turns it into the engine's primitives."""
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
    n_data = data_parallel_size(mesh)
    global_batch = built.pd.batch_size
    assert global_batch % n_data == 0, (global_batch, n_data)
    assert global_batch >= n_data, (
        f"global batch {global_batch} < data-parallel size {n_data}: local batch must be >= 1"
    )
    is_main = jax.process_index() == 0

    key = random.PRNGKey(built.pd.seed)
    _, _, run_key = random.split(key, 3)

    schedule = BatchSchedule(scan_shards(data.dir), global_batch, built.pd.seed)
    server = ShardServer(schedule, seq_len, jax.process_index(), n_proc)
    # Each process (node) owns all its local devices; its per-process batch splits across them.
    local_data = local_data_parallel_size(mesh)
    assert server.per_process % local_data == 0, (
        server.per_process,
        local_data,
    )

    def sample_batch(step: int) -> jax.Array:
        return global_token_batch(server.local_batch(step), mesh, global_batch)

    sink = MetricsSink.for_run(built.run, is_main)
    evaluation = None
    if eval_config is not None:
        assert eval_config.every % built.cadence.train_log_every == 0, (
            "eval must land on a train-log step: the tok/s window resets after eval, so a "
            "mid-window eval would corrupt the next step-time estimate"
        )
        evaluation = make_lm_evaluation(
            built,
            eval_config,
            model,
            run_key,
            mesh,
            n_proc,
            sink,
            runtime.resolved_compiler_options,
        )

    run_decomposition_training(
        pd=built.pd,
        cadence=built.cadence,
        run=built.run,
        model=model,
        ci_fn=built.ci_fn,
        positions=Positioned(n_positions=seq_len),
        remat_recon_forwards=runtime.remat_recon_forwards,
        remat_ci_fn=runtime.remat_ci_fn,
        compiler_options=runtime.resolved_compiler_options,
        sample_batch=sample_batch,
        evaluation=evaluation,
        sink=sink,
        profiling=engine_profiling(runtime.profiling),
    )


def pin_config_copy(run_dir: Path, name: str, source: Path) -> None:
    """First run copies `source` into the run dir; resumes byte-compare against it."""
    copy = run_dir / name
    if copy.exists():
        assert copy.read_text() == source.read_text(), (
            f"{copy} differs from {source} — refusing to resume with a changed config"
        )
    else:
        copy.write_text(source.read_text())


def main(
    config: Path,
    data_root: Path,
    local_device_count: int,
    run_id: str | None = None,
) -> None:
    config = Path(config)
    data_root = Path(data_root)
    if run_id is None:
        # Ad-hoc run-here invocation (`python -m param_decomp.experiments.lm.run <config>`):
        # mint a fresh identity; `pin_config_copy` below stages the config into the run
        # dir exactly as a launcher would (resume an existing run by passing --run-id).
        run_id = generate_run_id("param_decomp")
    built, authored = load_config(config, run_id, data_root)
    runtime = authored.runtime

    install_sigterm_flag()
    enable_hlo_dump(built.run.run_dir)
    initialize_topology(runtime.world_size, local_device_count)
    mesh = hsdp_mesh(runtime.replicate, runtime.fsdp, runtime.tp)

    if built.run.resume_provenance is not None:
        assert_finetune_structural_compat(built, built.run.resume_provenance, data_root)

    cache_dir = enable_persistent_compilation_cache(runtime.compilation_cache_dir)

    is_main = jax.process_index() == 0
    if is_main:
        cache_dir.mkdir(parents=True, exist_ok=True)
        built.run.run_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(built.run.run_dir / "logs.log")
        pin_config_copy(built.run.run_dir, LAUNCH_CONFIG_FILENAME, config)
        print(f"persistent XLA compilation cache: {cache_dir}", flush=True)
        site_kind_counts: dict[str, int] = {}
        for s in built.target.sites:
            kind = s.name.rsplit(".", 1)[-1]
            site_kind_counts[kind] = site_kind_counts.get(kind, 0) + 1
        site_summary = ", ".join(f"{k}×{n}" for k, n in sorted(site_kind_counts.items()))
        print(
            f"run {built.run.run_name} | {mesh.devices.size} GPU / {jax.process_count()} proc | "
            f"B={built.pd.batch_size} seq={read_dataset_meta(built.data.dir).seq_len} "
            f"sites={len(built.target.sites)} [{site_summary}] steps={built.pd.steps}",
            flush=True,
        )

    # The bundle's `.model` (an eqx model) IS the frozen target — it carries the frozen
    # weights as fields, so the function-table era's separate `frozen` object is gone.
    model = build_target(built.target, mesh, data_root, runtime.sharding)

    train(built, runtime, authored.eval, model, mesh)

    if jax.process_count() > 1:
        import jax.experimental.multihost_utils as mhu

        mhu.sync_global_devices("train_done")
        jax.distributed.shutdown()
