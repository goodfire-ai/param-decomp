"""Run targeted language-model parameter decomposition from one config.

Each step trains once on the fixed target-prompt pool and once on the broader corpus. This
module prepares both streams and reuses the process setup, checkpoint, and shutdown
behavior from ordinary LM training."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jax
import numpy as np
import yaml
from jax.sharding import Mesh
from numpy.typing import NDArray

from param_decomp.core.ci_fn.architecture import CIFnArchitecture
from param_decomp.core.hardware_utilization import (
    checked_device_kind,
    peak_bf16_dense_flops_per_second,
)
from param_decomp.core.model import ComponentActivations, PlacedModel, Positioned
from param_decomp.core.run import (
    MetricsSink,
    install_sigterm_flag,
    run_targeted_decomposition_training,
)
from param_decomp.core.run_files import LAUNCH_CONFIG_FILENAME
from param_decomp.core.sharding import (
    data_parallel_size,
    initialize_topology,
    mesh_for_shape,
)
from param_decomp.core.training_performance import MfuAccounting
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.arithmetic_probe import PromptEncoder
from param_decomp.experiments.lm.ci_fn_init import LMCIFnInitInputs, lm_ci_fn_initializer
from param_decomp.experiments.lm.config import (
    LMTargetedExperimentConfig,
    build_targeted_experiment_config,
)
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
from param_decomp.experiments.lm.model_flops import prepare_targeted_lm_step_flops
from param_decomp.experiments.lm.resolved import (
    AnyLMTargetConfig,
    HFSnapshotWeights,
    LlamaSimpleMLPTargetConfig,
    LMTargetedRun,
    PretrainCacheWeights,
    Qwen36MoeTargetConfig,
    TargetConfig,
    require_unrouted_ci_fn_arch,
)
from param_decomp.experiments.lm.targeted_data import build_prompt_pool, pool_batch
from param_decomp.experiments.lm.training import (
    enable_hlo_dump,
    enable_persistent_compilation_cache,
    engine_profiling,
    pin_config_copy,
)
from param_decomp.infra.dataset_store import read_dataset_identity
from param_decomp.infra.run_files import generate_run_id
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.lm.batch_data import (
    HostLMBatch,
    HostTokenBatch,
    global_lm_batch,
    global_token_batch,
)
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.qwen36_moe import Qwen36MoeDecomposedModel
from param_decomp.targets.transformer import TransformerDecomposedModel, hf_snapshot_dir


@dataclass(frozen=True)
class SnapshotTokenizer:
    """The target's local HF snapshot (already staged by the weights load) — loaded
    offline, never from the hub."""

    path: Path


@dataclass(frozen=True)
class HubTokenizer:
    """A hub tokenizer name — the locally pretrained targets name their tokenizer via the
    dataset's own meta."""

    name: str


def pool_tokenizer_source(
    target: AnyLMTargetConfig, dataset_tokenizer_name: str
) -> SnapshotTokenizer | HubTokenizer:
    """Where the prompt pool's tokenizer comes from — necessarily the SAME vocabulary the
    model and the broad stream use."""
    match target:
        case TargetConfig(model_name=model_name):
            return SnapshotTokenizer(path=hf_snapshot_dir(model_name))
        case Qwen36MoeTargetConfig(weights=weights):
            match weights:
                case HFSnapshotWeights(model_name=model_name):
                    return SnapshotTokenizer(path=hf_snapshot_dir(model_name))
                case PretrainCacheWeights():
                    return HubTokenizer(name=dataset_tokenizer_name)
        case LlamaSimpleMLPTargetConfig():
            return HubTokenizer(name=dataset_tokenizer_name)


def load_pool_tokenizer(source: SnapshotTokenizer | HubTokenizer) -> PromptEncoder:
    from transformers import AutoTokenizer

    match source:
        case SnapshotTokenizer(path=path):
            loaded = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        case HubTokenizer(name=name):
            loaded = AutoTokenizer.from_pretrained(name)
    return cast(PromptEncoder, cast(object, loaded))


def _document_prompt_batch(
    tokens: NDArray[np.int32], mesh: Mesh, batch_size: int, vocab_size: int
) -> LMBatchWithDocuments:
    return global_lm_batch(
        HostLMBatch.from_unsegmented_sequences(tokens), mesh, batch_size, vocab_size
    )


def _token_prompt_batch(
    tokens: NDArray[np.int32], mesh: Mesh, batch_size: int, vocab_size: int
) -> LMBatch:
    return global_token_batch(HostTokenBatch(tokens), mesh, batch_size, vocab_size)


def train_targeted(
    built: LMTargetedRun,
    cfg: LMTargetedExperimentConfig,
    eval_config: EvalConfig | None,
    model: PlacedLM,
    mesh: Mesh,
    data_root: Path,
) -> None:
    match model.model:
        case TransformerDecomposedModel() as target:
            _train_targeted(
                built,
                cfg,
                eval_config,
                PlacedModel(model=target, placement=model.placement),
                mesh,
                data_root,
                transformer_input_format(target),
                _document_prompt_batch,
                require_unrouted_ci_fn_arch(built.ci_fn),
            )
        case Qwen36MoeDecomposedModel() as target:
            _train_targeted(
                built,
                cfg,
                eval_config,
                PlacedModel(model=target, placement=model.placement),
                mesh,
                data_root,
                TokenInput(),
                _token_prompt_batch,
                built.ci_fn,
            )

        case other:
            raise AssertionError(f"unsupported LM target: {type(other).__name__}")


def _train_targeted[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    built: LMTargetedRun,
    cfg: LMTargetedExperimentConfig,
    eval_config: EvalConfig | None,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    mesh: Mesh,
    data_root: Path,
    input_format: InputFormat[TargetIn],
    place_prompt_batch: Callable[[NDArray[np.int32], Mesh, int, int], TargetIn],
    ci_fn_arch: CIFnArchitecture[Conditioning],
) -> None:
    """The targeted LM composition over the engine: the prompt-pool TARGET seam, the
    parquet NON-TARGET seam, and the same domain-bound eval operations as the plain root
    (forward-only diagnostics on the broad eval split)."""
    data = built.data
    train_identity = read_dataset_identity(data.dir)
    eval_identity = read_dataset_identity(data.eval_dir)
    assert train_identity == eval_identity, (
        f"train and eval datasets disagree — {data.dir} is {train_identity}, {data.eval_dir} "
        f"is {eval_identity}. A holdout tokenized differently or at another seq_len makes "
        "every eval number incomparable to the training loss it is read against."
    )
    n_proc = jax.process_count()
    n_data = data_parallel_size(mesh)
    is_main = jax.process_index() == 0

    # Both streams shard over the data axes and replicate each shard over TP.
    target_batch = built.pd.batch_size
    nontarget_batch = cfg.nontarget.batch_size
    for name, batch in (("pd.batch_size", target_batch), ("nontarget.batch_size", nontarget_batch)):
        assert batch % n_data == 0 and batch >= n_data, (
            f"{name} {batch} must be a positive multiple of data-parallel size {n_data}"
        )

    tokenizer = load_pool_tokenizer(
        pool_tokenizer_source(built.target, train_identity.tokenizer_name)
    )
    pool = build_prompt_pool(cfg.prompts, tokenizer)
    n_prompts, prompt_len = pool.tokens.shape
    if is_main:
        print(f"target prompt pool: {n_prompts} prompts x {prompt_len} positions", flush=True)

    per_process_target = target_batch // n_proc
    vocab_size = target_vocab_size(model)

    def sample_target_batch(step: int) -> TargetIn:
        rows = pool_batch(pool, built.pd.seed, step, target_batch)
        local = rows[jax.process_index() * per_process_target :][:per_process_target]
        return place_prompt_batch(local, mesh, target_batch, vocab_size)

    sample_nontarget_batch = input_sampler(
        input_format, data.dir, nontarget_batch, built.pd.seed, mesh, vocab_size
    )

    rules = model.placement
    assert rules is not None, "LM training runs placed"
    ci_fn_initializer = lm_ci_fn_initializer(
        ci_fn_arch, model.model.sites, rules, LMCIFnInitInputs(model, sample_nontarget_batch(0))
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
            cfg.runtime.resolved_compiler_options,
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

    run_targeted_decomposition_training(
        pd=built.pd,
        nontarget=cfg.nontarget,
        mfu_accounting=MfuAccounting(
            step_flops=prepare_targeted_lm_step_flops(
                built.pd,
                cfg.nontarget,
                built.target,
                model.model.sites,
                ci_fn_arch,
                prompt_len,
                train_identity.seq_len,
                data_root,
            ),
            peak_flops_per_second=jax.device_count()
            * peak_bf16_dense_flops_per_second(checked_device_kind(jax.devices())),
        ),
        cadence=built.cadence,
        run=built.run,
        model=model,
        ci_fn_initializer=ci_fn_initializer,
        positions=Positioned(n_positions=prompt_len),
        remat_recon_forwards=cfg.runtime.remat_recon_forwards,
        remat_ci_fn=cfg.runtime.remat_ci_fn,
        compiler_options=cfg.runtime.resolved_compiler_options,
        sample_target_batch=sample_target_batch,
        sample_nontarget_batch=sample_nontarget_batch,
        build_evaluation=build_evaluation,
        sink=sink,
        profiling=engine_profiling(cfg.runtime.profiling),
        component_initializer=component_initializer_for(built.target, model),
    )


def main(
    config: Path,
    data_root: Path,
    local_device_count: int,
    run_id: str | None = None,
) -> None:
    config = Path(config)
    data_root = Path(data_root)
    if run_id is None:
        # Ad-hoc run-here invocation: mint a fresh identity; `pin_config_copy` below
        # stages the config into the run dir.
        run_id = generate_run_id("param_decomp")
    raw = yaml.safe_load(config.read_text())
    cfg = LMTargetedExperimentConfig.model_validate(raw)
    built = build_targeted_experiment_config(cfg, run_id, data_root)
    runtime = cfg.runtime

    install_sigterm_flag()
    enable_hlo_dump(built.run.run_dir)
    initialize_topology(runtime.world_size, local_device_count)
    assert jax.default_backend() == "gpu", "LM training requires a GPU backend"
    mesh = mesh_for_shape(runtime.mesh)

    cache_dir = enable_persistent_compilation_cache(runtime.compilation_cache_dir)

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
            f"seq={read_dataset_identity(built.data.dir).seq_len} "
            f"sites={len(built.target.sites)} steps={built.pd.steps}",
            flush=True,
        )

    model = build_target(built.target, mesh, data_root, runtime.sharding, runtime.sequence_sharding)

    train_targeted(built, cfg, cfg.eval, model, mesh, data_root)

    if jax.process_count() > 1:
        import jax.experimental.multihost_utils as mhu

        mhu.sync_global_devices("train_done")
        jax.distributed.shutdown()
