"""Binding and execution of the fixed-grid LM arithmetic operation."""

from dataclasses import dataclass
from functools import partial

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from param_decomp.core.built_run import TargetSites
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import BATCH_AXES, CaptureKeys, PlacedModel
from param_decomp.core.recon import resolve_reconstruction_spec
from param_decomp.core.recon_eval import FreshPGDReconEval
from param_decomp.core.run import (
    BackgroundRenderer,
    DeferredMediaRecord,
    MetricsSink,
    PassOperation,
)
from param_decomp.core.sharding import data_parallel_size, local_data_parallel_size
from param_decomp.core.train import TrainState
from param_decomp.experiments.lm.arithmetic_eval import (
    ArithmeticGrid,
    ArithmeticGridStep,
    ArithmeticSelection,
    compute_arithmetic_selection,
    make_arithmetic_grid_step,
    n_alive_scalars,
    render_arithmetic_figures,
)
from param_decomp.experiments.lm.arithmetic_probe import build_arithmetic_probe
from param_decomp.experiments.lm.eval import ScalarStep, make_eval_step
from param_decomp.experiments.lm.eval_config import ArithmeticCIGridConfig
from param_decomp.experiments.lm.eval_context import LMEvalPass
from param_decomp.experiments.lm.eval_keys import EvalKeyStream
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.targets.glu_transformer import hf_snapshot_dir


def global_arithmetic_probe(tokens: np.ndarray, mesh: Mesh, n_proc: int) -> jax.Array:
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
    sharding = NamedSharding(mesh, P(BATCH_AXES))
    return jax.make_array_from_process_local_data(sharding, local, (n_pad, t))


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


@dataclass(frozen=True)
class ArithmeticOperation:
    step: ArithmeticGridStep
    probe_eval_step: ScalarStep
    model: PlacedModel
    tokens: jax.Array
    grid: ArithmeticGrid
    n_prompts: int
    thresholds: tuple[float, ...]
    top_k: int
    renderer: BackgroundRenderer

    def run(
        self, state: TrainState, placed_ci_fn: PlacedCIFn, key: PRNGKeyArray, now_step: int
    ) -> LogRecord:
        selection = compute_arithmetic_selection(
            self.step,
            self.model,
            state.decomposition.components,
            placed_ci_fn,
            self.tokens,
            self.n_prompts,
            self.thresholds,
            self.top_k,
        )
        scalars = self.probe_eval_step(
            self.model,
            state.decomposition.components,
            placed_ci_fn,
            self.tokens,
            key,
        )
        self.renderer.submit(partial(_render, selection, self.grid, self.top_k, now_step))
        return {
            **{
                f"eval/arithmetic/{name}": value
                for name, value in n_alive_scalars(selection.active, self.top_k).items()
            },
            **{f"eval/arithmetic/{name}": float(value) for name, value in scalars.items()},
        }


def make_arithmetic_operation(
    config: ArithmeticCIGridConfig,
    schedule: EvalSchedule,
    target: TargetSites,
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    mesh: Mesh,
    n_proc: int,
    sink: MetricsSink,
    run_key: PRNGKeyArray,
    train_steps: int,
    compiler_options: dict[str, bool | int | str],
) -> PassOperation[LMEvalPass]:
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
            name=pgd.name or "PGDReconLoss",
            n_steps=pgd.n_steps,
            step_size=pgd.step_size,
            reconstruction=resolve_reconstruction_spec(pgd.hidden_acts_reconstruction),
        )
        if pgd is not None
        else None
    )
    operation = ArithmeticOperation(
        step=make_arithmetic_grid_step(model, ci_capture_keys, probe.answer_position, n_prompts),
        probe_eval_step=make_eval_step(
            model,
            ci_capture_keys,
            ce.rounding_threshold,
            l0.ci_alive_threshold,
            l0_groups,
            fresh_pgd,
            mesh,
            n_valid_rows=n_prompts,
            compiler_options=compiler_options,
        ),
        model=model,
        tokens=global_arithmetic_probe(probe.tokens, mesh, n_proc),
        grid=probe.grid,
        n_prompts=n_prompts,
        thresholds=tuple(config.thresholds),
        top_k=config.top_k,
        renderer=BackgroundRenderer(sink),
    )

    def run(context: LMEvalPass) -> LogRecord:
        key = jax.random.fold_in(
            run_key, EvalKeyStream.ARITHMETIC * train_steps + context.pass_index
        )
        return operation.run(context.state, context.placed_ci_fn, key, context.now_step)

    return PassOperation(schedule, run)
