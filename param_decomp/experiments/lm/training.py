"""Run ordinary language-model parameter-decomposition training from one config.

    python -m param_decomp.experiments.lm.run <config.yaml> \
        --data-root <root> --local-device-count <n>

This is the LM I/O layer over the generic core engine
(`param_decomp.core.run.run_decomposition_training`): read the run YAML, build the target, feed
the per-step parquet token batch (`sample_batch`; the model embeds it), bind the CEandKL /
CI-L0 / PGD / attn-patterns / slow eval operations, then
call the engine. Process setup (`initialize_topology`, the SIGTERM flag, the persistent XLA
compilation cache), config pinning, and requeue-safe shutdown all live here. The toy domains mirror this file under `experiments/{tms,resid_mlp}/run.py`.

The config declares its logical `runtime.mesh`. The process entry separately supplies
the process-local device count used for JAX distributed bring-up.
"""

import os
from pathlib import Path

import jax
from jax.sharding import Mesh

from param_decomp.core.ci_fn.architecture import CIFnArchitecture
from param_decomp.core.configs import ResumeProvenance
from param_decomp.core.hardware_utilization import (
    checked_device_kind,
    peak_bf16_dense_flops_per_second,
)
from param_decomp.core.log import setup_logger
from param_decomp.core.model import ComponentActivations, PlacedModel, Positioned
from param_decomp.core.run import (
    JaxProfilerTrace,
    MetricsSink,
    NsightCaptureWindow,
    ProfilingMode,
    install_sigterm_flag,
    run_decomposition_training,
)
from param_decomp.core.run_files import LAUNCH_CONFIG_FILENAME
from param_decomp.core.sharding import (
    data_parallel_size,
    initialize_topology,
    mesh_for_shape,
)
from param_decomp.core.training_performance import MfuAccounting
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.ci_fn_init import LMCIFnInitInputs, lm_ci_fn_initializer
from param_decomp.experiments.lm.config import load_config
from param_decomp.experiments.lm.eval_operations import make_lm_evaluation
from param_decomp.experiments.lm.input_format import (
    InputFormat,
    TokenInput,
    input_sampler,
    transformer_input_format,
)
from param_decomp.experiments.lm.load_run import (
    PlacedLM,
    build_target,
    component_initializer_for,
    target_vocab_size,
)
from param_decomp.experiments.lm.model_flops import prepare_lm_step_flops
from param_decomp.experiments.lm.resolved import LMRun, require_unrouted_ci_fn_arch
from param_decomp.experiments.lm.runtime import (
    AdHocProfiling,
    NsightSystemsProfiling,
    ProfilingConfig,
    ProfilingDisabled,
    RuntimeConfig,
)
from param_decomp.infra.dataset_store import read_dataset_identity
from param_decomp.infra.run_files import generate_run_id
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.qwen36_moe import Qwen36MoeDecomposedModel
from param_decomp.targets.transformer import (
    TransformerDecomposedModel,
)


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
    else. The nsight arm's `version` stays outside the engine (it names the `nsys`
    executable, not engine behavior)."""
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
    hint (read pre-jax-init only to pick the writer, never to decide topology); whoever
    starts the ranks exports it per rank, absent = single
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
    not load onto the new reference). Only LR / coeffs / gamma / seq / batch / steps may
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
    model: PlacedLM,
    mesh: Mesh,
    data_root: Path,
) -> None:
    match model.model:
        case TransformerDecomposedModel() as target:
            _train(
                built,
                runtime,
                eval_config,
                PlacedModel(model=target, placement=model.placement),
                mesh,
                data_root,
                transformer_input_format(target),
                require_unrouted_ci_fn_arch(built.ci_fn),
            )
        case Qwen36MoeDecomposedModel() as target:
            _train(
                built,
                runtime,
                eval_config,
                PlacedModel(model=target, placement=model.placement),
                mesh,
                data_root,
                TokenInput(),
                built.ci_fn,
            )

        case other:
            raise AssertionError(f"unsupported LM target: {type(other).__name__}")


def _train[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    built: LMRun,
    runtime: RuntimeConfig,
    eval_config: EvalConfig | None,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    mesh: Mesh,
    data_root: Path,
    input_format: InputFormat[TargetIn],
    ci_fn_arch: CIFnArchitecture[Conditioning],
) -> None:
    """The LM composition over the generic engine: a parquet `sample_batch` (the per-step
    token batch the model embeds) and domain-bound CEandKL / CI-L0 / PGD / attention operations.

    `runtime` rides alongside the bundle rather than inside it: it is the LM's substrate,
    and core reads none of it — this root turns it into the engine's primitives."""
    data = built.data
    train_identity = read_dataset_identity(data.dir)
    eval_identity = read_dataset_identity(data.eval_dir)
    assert train_identity == eval_identity, (
        f"train and eval datasets disagree — {data.dir} is {train_identity}, {data.eval_dir} "
        f"is {eval_identity}. A holdout tokenized differently or at another seq_len makes "
        "every eval number incomparable to the training loss it is read against."
    )
    seq_len = train_identity.seq_len
    n_proc = jax.process_count()
    n_data = data_parallel_size(mesh)
    global_batch = built.pd.batch_size
    assert global_batch % n_data == 0, (global_batch, n_data)
    assert global_batch >= n_data, (
        f"global batch {global_batch} < data-parallel size {n_data}: local batch must be >= 1"
    )
    is_main = jax.process_index() == 0

    vocab_size = target_vocab_size(model)
    sample_batch = input_sampler(
        input_format, data.dir, global_batch, built.pd.seed, mesh, vocab_size
    )

    rules = model.placement
    assert rules is not None, "LM training runs placed"
    ci_fn_initializer = lm_ci_fn_initializer(
        ci_fn_arch, model.model.sites, rules, LMCIFnInitInputs(model, sample_batch(0))
    )

    sink = MetricsSink.for_run(built.run, is_main)
    build_evaluation = (
        None
        if eval_config is None
        else lambda run_key: make_lm_evaluation(
            built,
            eval_config,
            model,
            run_key,
            mesh,
            n_proc,
            sink,
            runtime.resolved_compiler_options,
            sample_batch=input_sampler(
                input_format,
                data.eval_dir,
                eval_config.batch_size,
                built.pd.seed + 1,
                mesh,
                vocab_size,
            ),
        )
    )

    run_decomposition_training(
        pd=built.pd,
        mfu_accounting=MfuAccounting(
            step_flops=prepare_lm_step_flops(
                built.pd,
                built.target,
                model.model.sites,
                ci_fn_arch,
                seq_len,
                data_root,
            ),
            peak_flops_per_second=jax.device_count()
            * peak_bf16_dense_flops_per_second(checked_device_kind(jax.devices())),
        ),
        cadence=built.cadence,
        run=built.run,
        model=model,
        ci_fn_initializer=ci_fn_initializer,
        positions=Positioned(n_positions=seq_len),
        remat_recon_forwards=runtime.remat_recon_forwards,
        remat_ci_fn=runtime.remat_ci_fn,
        compiler_options=runtime.resolved_compiler_options,
        sample_batch=sample_batch,
        build_evaluation=build_evaluation,
        sink=sink,
        profiling=engine_profiling(runtime.profiling),
        component_initializer=component_initializer_for(built.target, model),
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
        # dir. Resume an existing run by passing --run-id.
        run_id = generate_run_id("param_decomp")
    built, authored = load_config(config, run_id, data_root)
    runtime = authored.runtime

    install_sigterm_flag()
    enable_hlo_dump(built.run.run_dir)
    initialize_topology(runtime.world_size, local_device_count)
    assert jax.default_backend() == "gpu", "LM training requires a GPU backend"
    mesh = mesh_for_shape(runtime.mesh)

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
            f"B={built.pd.batch_size} seq={read_dataset_identity(built.data.dir).seq_len} "
            f"sites={len(built.target.sites)} [{site_summary}] steps={built.pd.steps}",
            flush=True,
        )

    # The bundle's `.model` (an eqx model) IS the frozen target — it carries the frozen
    # weights as fields, so the function-table era's separate `frozen` object is gone.
    model = build_target(built.target, mesh, data_root, runtime.sharding, runtime.sequence_sharding)

    train(built, runtime, authored.eval, model, mesh, data_root)

    if jax.process_count() > 1:
        import jax.experimental.multihost_utils as mhu

        mhu.sync_global_devices("train_done")
        jax.distributed.shutdown()
