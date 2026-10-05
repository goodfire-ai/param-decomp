"""Binding and execution of the fixed-grid LM arithmetic operation."""

from collections.abc import Mapping
from functools import partial

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray
from numpy.typing import NDArray

from param_decomp.core.built_run import TargetSites
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.components import ComponentStacks
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import CaptureKeys, ComponentActivations, PlacedModel
from param_decomp.core.placement import batch_axes
from param_decomp.core.recon import resolve_auxiliary_reconstruction
from param_decomp.core.recon_eval import FreshPGDAttack, FreshPGDReconEval
from param_decomp.core.run import (
    BackgroundRenderer,
    DeferredMediaRecord,
    MetricsSink,
    StandaloneOperation,
    StandaloneOperationPlan,
)
from param_decomp.core.sharding import data_parallel_size, local_data_parallel_size
from param_decomp.experiments.lm.arithmetic_eval import (
    ArithmeticGrid,
    ArithmeticSelection,
    compute_arithmetic_selection,
    make_arithmetic_grid_step,
    n_alive_scalars,
    prepare_arithmetic_columns,
    render_arithmetic_figures,
)
from param_decomp.experiments.lm.arithmetic_probe import build_arithmetic_probe
from param_decomp.experiments.lm.eval import make_eval_step
from param_decomp.experiments.lm.eval_config import ArithmeticCIGridConfig
from param_decomp.experiments.lm.eval_context import LMEvalPass
from param_decomp.experiments.lm.eval_keys import EvalKeyStream
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.transformer import (
    TransformerDecomposedModel,
    TransformerPreparedMasking,
    TransformerPreparedWeights,
    hf_snapshot_dir,
)


def global_arithmetic_probe(tokens: np.ndarray, mesh: Mesh, n_proc: int) -> LMBatchWithDocuments:
    n, t = tokens.shape
    n_data = data_parallel_size(mesh)
    pad = (-n) % n_data
    if pad:
        tokens = np.concatenate([tokens, np.zeros((pad, t), tokens.dtype)], axis=0)
    n_pad = tokens.shape[0]
    per_process = n_pad // n_proc
    local_data = local_data_parallel_size(mesh)
    assert per_process % local_data == 0, (per_process, local_data)
    proc = jax.process_index()
    local = tokens[proc * per_process : (proc + 1) * per_process]
    sharding = NamedSharding(mesh, P(batch_axes(mesh)))
    return LMBatchWithDocuments.from_unsegmented_sequences(
        jax.make_array_from_process_local_data(sharding, local, (n_pad, t))
    )


def _render(
    selection: ArithmeticSelection, grid: ArithmeticGrid, top_k: int, now_step: int
) -> DeferredMediaRecord:
    return DeferredMediaRecord(
        step_key="eval/arithmetic/figure_step",
        step=now_step,
        media={
            f"eval/arithmetic/{key}": value
            for key, value in render_arithmetic_figures(selection, grid, top_k).items()
        },
    )


def make_arithmetic_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    config: ArithmeticCIGridConfig,
    schedule: EvalSchedule,
    target: TargetSites,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    mesh: Mesh,
    n_proc: int,
    sink: MetricsSink,
    run_key: PRNGKeyArray,
    train_steps: int,
    compiler_options: dict[str, bool | int | str],
) -> StandaloneOperationPlan[LMEvalPass[TargetIn, Conditioning]]:
    inner: object = model.model
    assert isinstance(inner, TransformerDecomposedModel), (
        "arithmetic evaluation requires a shared transformer target"
    )
    arithmetic_model = PlacedModel(inner, model.placement)
    assert isinstance(target, TargetConfig), (
        f"arithmetic eval needs an HF tokenizer; {type(target).__name__} has no model_name"
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(hf_snapshot_dir(target.model_name)), local_files_only=True
    )
    probe = build_arithmetic_probe(config.operation, config.a_range, config.b_range, tokenizer)
    n_prompts = probe.tokens.shape[0]
    ce = config.probe_metrics.ce_kl
    l0 = config.probe_metrics.ci_l0
    pgd = config.probe_metrics.fresh_pgd
    l0_groups = (
        {name: tuple(patterns) for name, patterns in l0.groups.items()}
        if l0.groups is not None
        else None
    )
    fresh_pgd = (
        FreshPGDReconEval(
            attack=FreshPGDAttack(step_size=pgd.step_size, read_out_steps=(pgd.n_steps,)),
            reconstruction=resolve_auxiliary_reconstruction(pgd.auxiliaries),
            metric_type="PGDReconLoss",
        )
        if pgd is not None
        else None
    )
    grid_step = make_arithmetic_grid_step(
        arithmetic_model, ci_capture_keys, probe.answer_position, n_prompts
    )
    probe_eval_step = make_eval_step(
        arithmetic_model,
        ci_capture_keys,
        ce.rounding_threshold,
        l0.ci_alive_threshold,
        l0_groups,
        fresh_pgd,
        mesh,
        n_valid_rows=n_prompts,
    )
    tokens = global_arithmetic_probe(probe.tokens, mesh, n_proc)
    renderer = BackgroundRenderer(sink)

    def score(
        model: PlacedModel[
            LMBatchWithDocuments,
            LMOutput,
            TransformerPreparedWeights,
            LMBatchWithDocuments,
            TransformerPreparedMasking,
        ],
        components: ComponentStacks,
        ci_fn: CIFn[LMBatchWithDocuments],
        tokens: LMBatchWithDocuments,
        pass_index: NDArray[np.uint32],
    ) -> Mapping[str, Array]:
        key = jax.random.fold_in(run_key, EvalKeyStream.ARITHMETIC * train_steps + pass_index)
        return probe_eval_step(model, components, ci_fn, tokens, key)

    def prepare(
        example: LMEvalPass[TargetIn, Conditioning],
    ) -> StandaloneOperation[LMEvalPass[TargetIn, Conditioning]]:
        lowered_grid = jax.jit(grid_step, compiler_options=compiler_options).lower(
            arithmetic_model,
            example.decomposition.components,
            example.decomposition.ci_fn,
            tokens,
        )
        compiled_grid_step = lowered_grid.compile()
        ci_shapes, xv_shapes, _ = compiled_grid_step.out_info
        compiled_probe_eval_step = (
            jax.jit(score, compiler_options=compiler_options)
            .lower(
                arithmetic_model,
                example.decomposition.components,
                example.decomposition.ci_fn,
                tokens,
                np.zeros((), np.uint32),
            )
            .compile()
        )
        column_gathers = prepare_arithmetic_columns(ci_shapes, xv_shapes, config.top_k)

        def run(context: LMEvalPass[TargetIn, Conditioning]) -> LogRecord:
            selection = compute_arithmetic_selection(
                compiled_grid_step,
                arithmetic_model,
                context.decomposition.components,
                context.decomposition.ci_fn,
                tokens,
                n_prompts,
                tuple(config.thresholds),
                config.top_k,
                column_gathers,
            )
            scalars = compiled_probe_eval_step(
                arithmetic_model,
                context.decomposition.components,
                context.decomposition.ci_fn,
                tokens,
                np.asarray(context.pass_index, np.uint32),
            )
            renderer.submit(partial(_render, selection, probe.grid, config.top_k, context.now_step))
            return {
                **{
                    f"eval/arithmetic/{name}": value
                    for name, value in n_alive_scalars(selection.active, config.top_k).items()
                },
                **{f"eval/arithmetic/{name}": float(value) for name, value in scalars.items()},
            }

        return StandaloneOperation(schedule, run)

    return StandaloneOperationPlan(prepare)
