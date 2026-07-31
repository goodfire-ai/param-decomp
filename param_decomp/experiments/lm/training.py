"""The LM decomposition composition root: run YAML -> full SPEC-compliant run on a
vendored target.

    python -m param_decomp.experiments.lm.run <config.yaml>   # normally invoked by a
        # submitter that pins the config as runs/<id>/launch_config.yaml and passes
        # --run-id; re-running resumes in place

This is the LM I/O layer over the generic core engine
(`param_decomp.core.run.run_decomposition_training`): read the run YAML, build the target, feed
the per-step parquet token batch (`sample_batch`; the model embeds it), bind the CEandKL /
CI-L0 / PGD / attn-patterns / slow eval operations, then
call the engine. Process setup (`init_distributed`, the SIGTERM flag, the persistent XLA
compilation cache, HF http hardening), config pinning, and SLURM-requeue shutdown all live
here. The toy domains mirror this file under `experiments/{tms,resid_mlp}/run.py`.

Process bring-up derives from the config topology: `dp > gpus_per_node` → one process per node under
`jax.distributed` (`init_distributed`), every process computing the same global schedule
and contributing its local batch slice; `inline` → this one process, over exactly the
`runtime.dp` local devices it finds (`assert_inline_topology`).
"""

import os
from pathlib import Path

import fire
import jax
from jax import random
from jax.sharding import Mesh

from param_decomp.core import placement
from param_decomp.core.built_run import LAUNCH_CONFIG_FILENAME
from param_decomp.core.configs import ResumeProvenance
from param_decomp.core.log import setup_logger
from param_decomp.core.model import DecomposedModel, Positioned
from param_decomp.core.run import (
    MetricsSink,
    install_sigterm_flag,
    run_decomposition_training,
)
from param_decomp.core.sharding import assert_inline_topology, hsdp_mesh, init_distributed
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.config import load_config
from param_decomp.experiments.lm.eval_operations import global_token_batch, make_lm_evaluation
from param_decomp.experiments.lm.hf_http import configure_hf_http_retries
from param_decomp.experiments.lm.load_run import build_target
from param_decomp.experiments.lm.resolved import LMRun
from param_decomp.experiments.lm.runtime import RuntimeConfig
from param_decomp.infra.dataset_store import read_dataset_meta
from param_decomp.infra.run_files import generate_run_id
from param_decomp.pretrain.batch_data import BatchSchedule, ShardServer, scan_shards


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
    so a later mutation is ignored. Rank-gated via the generic `PD_RANK`
    hint (read pre-jax-init, only to pick the writer — NOT to decide distributedness, which
    stays config-derived (dp vs gpus_per_node); the launcher exports it per rank, absent = single
    process) so a single rank writes; `xla_dump_hlo_module_re` filters to the big `*step*` modules to keep
    the dump to ~100s of MB. The buffer-assignment dump survives an exec-time OOM (compile
    completes first), so this is how we name the buffer that blows the allocator."""
    if os.environ.get("PD_RANK", "0") != "0":
        return
    hlo_dir = run_dir / "hlo"
    hlo_dir.mkdir(parents=True, exist_ok=True)
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (
        f"{existing} --xla_dump_to={hlo_dir} --xla_dump_hlo_module_re=.*step.*"
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
    model: DecomposedModel,
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
    # Pure HSDP: the batch shards over the FULL mesh (both axes), so it must tile the full
    # device count, and the per-rank batch is B/N. Constraint: B >= N (per-rank >= 1).
    n_dev = mesh.devices.size
    global_batch = built.pd.batch_size
    assert global_batch % n_dev == 0, (global_batch, n_dev)
    assert global_batch >= n_dev, (
        f"global batch {global_batch} < device count {n_dev}: per-rank batch must be >= 1"
    )
    is_main = jax.process_index() == 0

    key = random.PRNGKey(built.pd.seed)
    _, _, run_key = random.split(key, 3)

    schedule = BatchSchedule(scan_shards(data.dir), global_batch, built.pd.seed)
    server = ShardServer(schedule, seq_len, jax.process_index(), n_proc)
    # Each process (node) owns all its local devices; its per-process batch splits across them.
    assert server.per_process % jax.local_device_count() == 0, (
        server.per_process,
        jax.local_device_count(),
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
            built, eval_config, model, run_key, mesh, n_proc, sink, runtime.compiler_options
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
        ascend_replicate=runtime.ascend_replicate,
        compiler_options=runtime.compiler_options,
        sample_batch=sample_batch,
        evaluation=evaluation,
        sink=sink,
        mesh=mesh,
        placement_rules=placement.from_config(runtime.sharding, mesh, model.sites),
    )


def _pin_config_copy(run_dir: Path, name: str, source: Path) -> None:
    """First run copies `source` into the run dir; resumes byte-compare against it."""
    copy = run_dir / name
    if copy.exists():
        assert copy.read_text() == source.read_text(), (
            f"{copy} differs from {source} — refusing to resume with a changed config"
        )
    else:
        copy.write_text(source.read_text())


def main(config: Path, data_root: Path, run_id: str | None = None) -> None:
    config = Path(config)
    data_root = Path(data_root)
    if run_id is None:
        # Ad-hoc run-here invocation (`python -m param_decomp.experiments.lm.run <config>`):
        # mint a fresh identity; `_pin_config_copy` below stages the config into the run
        # dir exactly as a launcher would (resume an existing run by passing --run-id).
        run_id = generate_run_id("param_decomp")
    built, authored = load_config(config, run_id, data_root)
    runtime = authored.runtime

    install_sigterm_flag()
    _enable_hlo_dump(built.run.run_dir)
    if runtime.distributed:
        init_distributed(runtime.dp, runtime.gpus_per_node)
    else:
        assert_inline_topology(runtime.dp)
    # Harden the cold-cache HF weight load against the 8N-rank startup burst before any
    # per-rank Hub call (no-op when huggingface_hub is absent / cache is pre-warmed).
    configure_hf_http_retries()
    mesh = hsdp_mesh(runtime.tp, runtime.gpus_per_node)

    if built.run.resume_provenance is not None:
        assert_finetune_structural_compat(built, built.run.resume_provenance, data_root)

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
        print(
            f"run {built.run.run_name} | {mesh.devices.size} GPU / {jax.process_count()} proc | "
            f"B={built.pd.batch_size} seq={read_dataset_meta(built.data.dir).seq_len} "
            f"sites={len(built.target.sites)} [{site_summary}] steps={built.pd.steps}",
            flush=True,
        )

    # The `lm` (an eqx model) IS the frozen target — it carries the frozen weights as fields,
    # so the function-table era's separate `frozen` object is gone.
    model, _vocab_size = build_target(built, mesh, data_root)

    train(built, runtime, authored.eval, model, mesh)

    if jax.process_count() > 1:
        import jax.experimental.multihost_utils as mhu

        mhu.sync_global_devices("train_done")
        jax.distributed.shutdown()


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
