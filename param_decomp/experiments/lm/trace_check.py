"""Deviceless CPU trace gate for a placed LM seat: assemble the REAL jitted programs at
the config's declared topology and LOWER them — no weights, no dataset store, no compile.

Explicit-sharding refusals (an ambiguous sharded contraction, a reshard the rules
cannot type) and output-edge dispatch errors (a kernel reading `.ndim` off the streamed
package) fire at trace time, which schema parsing and the placement-claims gate never
reach; this gate makes them fail during config validation rather than only after
accelerator startup or at the first eval tick. It needs `world_size` local devices:

    XLA_FLAGS=--xla_force_host_platform_device_count=<world> JAX_PLATFORMS=cpu \\
        python -m param_decomp.experiments.lm.trace_check <config.yaml> --seq-len <n>

CPU inputs must explicitly select XLA attention for both target and CI; flash requests
remain errors. The repo-config gate (`param_decomp/tests/experiments/test_repo_configs_parse.py`)
authors temporary CPU-attention variants for every maintained config with an abstract
model builder, preserving the declared topology and tensor shapes. That gate validates
sharding and output dispatch, not the configured GPU attention backend. The
frozen model enters as shapes only (`abstract_qwen36_moe_model`); `--seq-len` fixes the
token extent (sequence is never sharded, so any extent exercises the same sharding
rules). A targeted (tPD) seat — its own schema, dispatched here like the parse gate, on
the presence of `nontarget:` — lowers the two-stream targeted step, both streams at
`--seq-len`: the target pool's own prompt length needs a tokenizer the gate deliberately
does not load, and the extent is sharding-irrelevant anyway.

Coverage, one printed line per program:

- the train step (`jit_step`, or the targeted twin) at the config's declared output edge;
- when the config carries an `eval:` block, the eval programs exactly as `make_lm_evaluation`
  binds them — the pass's one batch-context step, then per configured metric the program
  its operation jits over that context (the scalar scorers, the CI-reduction and
  position-CI steps, the attention-patterns steps, the well-temperedness step) plus the
  standing nonlinearity operation — at the declared edge AND at every other output edge
  the family supports: the kernels dispatch on the output union, so a seat is one config
  line from the edge it did not declare. `ArithmeticCIGrid` is the enumerated gap: its
  probe grid is tokenized with the target's HF tokenizer, which the gate does not load.
"""

import argparse
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np

# tokamax must load before pyarrow: its xprof dependency's C++ static init self-deadlocks
# (absl mutex spin inside protobuf's InitProtobufDefaults) when pyarrow's bundled protobuf
# registered first. This gate is a composition root: its eval imports reach pyarrow through
# the LM data path, and the routed experts import tokamax lazily at trace time.
import tokamax  # noqa: F401  # pyright: ignore[reportUnusedImport]
import yaml
from jax import random
from jax.sharding import Mesh

from param_decomp.core import placement
from param_decomp.core.adversary import SourceStacks
from param_decomp.core.components import nonlinearity_partitions
from param_decomp.core.configs import (
    CI_L0Config,
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    EvalPGDReconLossConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    SlowPGDReconLossConfig,
    UVPlotsConfig,
)
from param_decomp.core.model import EMPTY_CAPTURE_KEYS, Positioned
from param_decomp.core.nonlinearity_eval import make_nonlinearity_eval_step
from param_decomp.core.placement import PlacementRules
from param_decomp.core.sharding import mesh_over_devices
from param_decomp.core.slow_eval import make_ci_reduction_step, make_position_ci_step
from param_decomp.core.tools.fit_check import (
    DeclaredRun,
    abstract_placed_model,
    lowered_targeted_train_step,
    lowered_train_step,
    standin_faithfulness_loss,
)
from param_decomp.core.train import (
    Decomposition,
    PDTrainingState,
    TargetedPDTrainingState,
)
from param_decomp.experiments.eval_config import AnyEvalMetricConfig, EvalConfig
from param_decomp.experiments.lm.abstract_inputs import abstract_lm_batch
from param_decomp.experiments.lm.attn_patterns_eval import attn_output_key_by_site
from param_decomp.experiments.lm.ci_position_eval import position_active_counts
from param_decomp.experiments.lm.config import (
    LMExperimentConfig,
    LMTargetedExperimentConfig,
    ResolvedDecomposition,
    resolve_decomposition,
    resolve_lm_ci_fn_arch,
)
from param_decomp.experiments.lm.diagnostic_eval_operations import (
    attn_patterns_step_for,
    site_figures_reduction_step,
)
from param_decomp.experiments.lm.eval_config import (
    ArithmeticCIGridConfig,
    CEandKLLossesConfig,
    CIActiveCountsPerPositionConfig,
    CIMaskedAttnPatternsReconLossConfig,
    RouterDivergenceConfig,
    StochasticAttnPatternsReconLossConfig,
    WellTemperednessConfig,
)
from param_decomp.experiments.lm.eval_context import (
    LMBatchContext,
    make_lm_batch_context_step,
    prepared_batch_from_context,
)
from param_decomp.experiments.lm.eval_operations import clean_capture_demand
from param_decomp.experiments.lm.load_run import PlacedQwen
from param_decomp.experiments.lm.resolved import (
    LlamaSimpleMLPTargetConfig,
    Qwen36MoeTargetConfig,
    TargetConfig,
)
from param_decomp.experiments.lm.router_divergence_eval import (
    make_router_divergence_step,
    router_probs_capture_keys,
)
from param_decomp.experiments.lm.scalar_eval_operations import scalar_scorer_for
from param_decomp.experiments.lm.well_temperedness import make_well_temperedness_step
from param_decomp.lm.batch import LMBatch, LMBatchWithRouting
from param_decomp.targets import qwen36_moe
from param_decomp.targets.lm_output import (
    MaterializedOutputEdge,
    OutputEdge,
    StreamedOutputEdge,
)
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeDecomposedModel,
)

UNDECLARED_STREAMED_EDGE_N_VOCAB_CHUNKS = 32
"""Vocab chunking for the streamed edge of a seat that declared the materialized one:
the streamed kernels trace the same program for any chunk count dividing the vocab."""


def qwen36_moe_output_edges(declared: OutputEdge, vocab_size: int) -> tuple[OutputEdge, ...]:
    """Every output edge the qwen36_moe family supports at this vocab, the config's declared
    one first. A vocab the undeclared chunking does not divide (the toys' NeoX 50,277) has
    no streamed edge to gate beside its declared materialized one."""
    match declared:
        case MaterializedOutputEdge() if vocab_size % UNDECLARED_STREAMED_EDGE_N_VOCAB_CHUNKS:
            return (declared,)
        case MaterializedOutputEdge():
            return (
                declared,
                StreamedOutputEdge(n_vocab_chunks=UNDECLARED_STREAMED_EDGE_N_VOCAB_CHUNKS),
            )
        case StreamedOutputEdge():
            return (declared, MaterializedOutputEdge())


def _edge_label(edge: OutputEdge) -> str:
    match edge:
        case MaterializedOutputEdge():
            return "materialized"
        case StreamedOutputEdge(n_vocab_chunks=n_vocab_chunks):
            return f"streamed[{n_vocab_chunks}]"


@dataclasses.dataclass(frozen=True)
class AbstractSeat:
    """The config's shape-only model with its resolved placement over this process's
    devices — the piece every lowered program shares. The model is placed per output edge
    (`placed`); the edge is a static field, so re-placing is free."""

    abstract: Qwen36MoeDecomposedModel
    rules: PlacementRules
    mesh: Mesh
    resolved: ResolvedDecomposition
    output_edges: tuple[OutputEdge, ...]

    def placed(self, edge: OutputEdge) -> PlacedQwen:
        return abstract_placed_model(
            dataclasses.replace(self.abstract, output_edge=edge), self.rules
        )


def abstract_seat(cfg: LMExperimentConfig | LMTargetedExperimentConfig) -> AbstractSeat:
    resolved = resolve_decomposition(cfg.target, cfg.decomposition, Path("out"))
    match resolved.target:
        case Qwen36MoeTargetConfig(
            arch=arch,
            weights_dtype=weights_dtype,
            output_edge=output_edge,
            expert_implementation=expert_implementation,
            attention_implementation=attention_implementation,
        ):
            abstract = qwen36_moe.abstract_qwen36_moe_model(
                arch,
                resolved.site_specs,
                weights_dtype,
                expert_implementation,
                output_edge,
                attention_implementation,
            )
            output_edges = qwen36_moe_output_edges(output_edge, arch.vocab_size)
        case TargetConfig() | LlamaSimpleMLPTargetConfig():
            raise NotImplementedError(
                f"the trace gate has no abstract model builder for "
                f"{type(resolved.target).__name__}; the GLU families' placed step is "
                f"pinned by their placed suites — extend with a shape-only builder to "
                f"gate them here too"
            )
    mesh = mesh_over_devices(cfg.runtime.mesh, np.asarray(jax.devices()))
    rules = placement.from_config(
        cfg.runtime.sharding,
        mesh,
        resolved.site_specs,
        sequence_sharding=cfg.runtime.sequence_sharding,
    )
    return AbstractSeat(abstract, rules, mesh, resolved, output_edges)


def trace_train_step_at_declared_topology(
    cfg: LMExperimentConfig, seat: AbstractSeat, model: PlacedQwen, seq_len: int
) -> DeclaredRun[LMBatchWithRouting[LMBatch], PDTrainingState]:
    """Lower the config's train step at its declared `(mesh, sharding)` over this
    process's devices. Raises exactly what the trainer's first trace would."""
    ci_fn_arch = resolve_lm_ci_fn_arch(seat.resolved, cfg.decomposition.ci)
    _, declared = lowered_train_step(
        cfg.pd,
        ci_fn_arch,
        model,
        Positioned(n_positions=seq_len),
        abstract_lm_batch(cfg.pd.batch_size, seq_len, seat.mesh),
        standin_faithfulness_loss(model),
        remat_recon_forwards=cfg.runtime.remat_recon_forwards,
        remat_ci_fn=cfg.runtime.remat_ci_fn,
        compiler_options=None,
    )
    return declared


def trace_targeted_train_step_at_declared_topology(
    cfg: LMTargetedExperimentConfig,
    seat: AbstractSeat,
    model: PlacedQwen,
    seq_len: int,
) -> DeclaredRun[LMBatchWithRouting[LMBatch], TargetedPDTrainingState]:
    """The tPD twin: lower the two-stream targeted step, target and non-target batches
    both at `seq_len`."""
    ci_fn_arch = resolve_lm_ci_fn_arch(seat.resolved, cfg.decomposition.ci)
    _, declared = lowered_targeted_train_step(
        cfg.pd,
        cfg.nontarget,
        ci_fn_arch,
        model,
        Positioned(n_positions=seq_len),
        abstract_lm_batch(cfg.pd.batch_size, seq_len, seat.mesh),
        abstract_lm_batch(cfg.nontarget.batch_size, seq_len, seat.mesh),
        remat_recon_forwards=cfg.runtime.remat_recon_forwards,
        remat_ci_fn=cfg.runtime.remat_ci_fn,
        compiler_options=None,
    )
    return declared


def _lower(label_for_log: str, program: Callable[..., Any], *abstract_args: Any) -> Any:
    """Lower a numerical program over abstract inputs and retain output placement."""
    out_info = jax.jit(program).lower(*abstract_args).out_info
    print(f"trace gate OK: {label_for_log} lowers", flush=True)
    return out_info


def _metric_label(metric: AnyEvalMetricConfig) -> str:
    """The metric's logged identity (`eval_config.validate_eval_metrics`)."""
    match metric:
        case EvalPGDReconLossConfig() | SlowPGDReconLossConfig():
            return metric.name or metric.type
        case (
            CEandKLLossesConfig()
            | CI_L0Config()
            | CIActiveCountsPerPositionConfig()
            | CIHistogramsConfig()
            | ComponentActivationDensityConfig()
            | CIMeanPerComponentConfig()
            | PermutedCIPlotsConfig()
            | UVPlotsConfig()
            | IdentityCIErrorConfig()
            | CIMaskedAttnPatternsReconLossConfig()
            | StochasticAttnPatternsReconLossConfig()
            | RouterDivergenceConfig()
            | WellTemperednessConfig()
            | ArithmeticCIGridConfig()
        ):
            return metric.type


def trace_eval_programs(
    eval: EvalConfig,
    model: PlacedQwen,
    edge: OutputEdge,
    decomposition: Decomposition[LMBatchWithRouting[LMBatch]],
    persistent_sources: dict[str, SourceStacks],
    mesh: Mesh,
    seq_len: int,
) -> None:
    """Lower the eval programs `make_lm_evaluation` binds for `eval` over `model` (placed at
    `edge`): the pass's context step over the declared-sharding state and an abstract token
    batch, then every configured operation's jitted program over the context's avals, then
    the standing nonlinearity operation. Raises exactly what the first eval tick would."""
    with jax.set_mesh(mesh):
        at = f"@ {_edge_label(edge)}"
        ci_fn = decomposition.ci_fn
        components = decomposition.components
        tokens = abstract_lm_batch(eval.batch_size, seq_len, mesh)
        key = jax.eval_shape(lambda: random.PRNGKey(0))

        operation_capture_keys = frozenset().union(
            *(clean_capture_demand(metric, model) for metric in eval.metrics), EMPTY_CAPTURE_KEYS
        )
        context_step = make_lm_batch_context_step(
            model, ci_fn.capture_keys, operation_capture_keys, mesh
        )
        forward = _lower(f"eval context step {at}", context_step, model, components, ci_fn, tokens)
        context = LMBatchContext(
            pass_index=0,
            batch_index=0,
            forward=forward,
            persistent_sources=persistent_sources,
        )

        for metric in eval.metrics:
            label = f"eval {_metric_label(metric)} {at}"
            match metric:
                case (
                    CEandKLLossesConfig()
                    | CI_L0Config()
                    | EvalPGDReconLossConfig()
                    | SlowPGDReconLossConfig()
                ):
                    batch = prepared_batch_from_context(
                        context, clean_capture_demand(metric, model)
                    )
                    _lower(
                        f"{label} scorer", scalar_scorer_for(metric, model, mesh), model, batch, key
                    )
                case (
                    CIHistogramsConfig()
                    | ComponentActivationDensityConfig()
                    | CIMeanPerComponentConfig()
                ):
                    _lower(
                        f"{label} CI reduction step",
                        site_figures_reduction_step(metric),
                        context.forward.ci.preactivations,
                    )
                case CIActiveCountsPerPositionConfig():
                    _lower(
                        f"{label} position counts", position_active_counts, context.forward.ci.lower
                    )
                case PermutedCIPlotsConfig() | UVPlotsConfig() | IdentityCIErrorConfig():
                    _lower(
                        f"{label} position-CI step",
                        make_position_ci_step(),
                        context.forward.ci.preactivations,
                    )
                case (
                    CIMaskedAttnPatternsReconLossConfig() | StochasticAttnPatternsReconLossConfig()
                ):
                    clean_site_outputs = {
                        site: context.forward.captures[key]
                        for site, key in attn_output_key_by_site(model).items()
                    }
                    _lower(
                        f"{label} attention-patterns step",
                        attn_patterns_step_for(metric, model),
                        model,
                        context.forward.prepared_weights,
                        context.forward.conditioning,
                        context.forward.ci.lower,
                        clean_site_outputs,
                        key,
                    )
                case RouterDivergenceConfig():
                    _lower(
                        f"{label} step",
                        make_router_divergence_step(model, metric, mesh),
                        model,
                        context.forward.prepared_weights,
                        context.forward.conditioning,
                        context.forward.ci.lower,
                        {
                            probs_key: context.forward.captures[probs_key]
                            for probs_key in router_probs_capture_keys(model)
                        },
                        context.forward.clean_output,
                        context.persistent_sources,
                        key,
                    )
                case WellTemperednessConfig():
                    _lower(
                        f"{label} step",
                        make_well_temperedness_step(model, ci_fn.capture_keys, metric, mesh),
                        model,
                        components,
                        ci_fn,
                        tokens,
                        key,
                    )
                case ArithmeticCIGridConfig():
                    print(
                        f"trace gate gap: {label} is not lowered — its probe grid is tokenized "
                        f"with the target's HF tokenizer, which the gate does not load",
                        flush=True,
                    )

        partitions = nonlinearity_partitions(model.model.sites)
        if partitions:
            _lower(
                f"eval nonlinearity CI reduction step {at}",
                make_ci_reduction_step(0.0, None, None),
                context.forward.ci.preactivations,
            )
            _lower(
                f"eval nonlinearity step {at}",
                make_nonlinearity_eval_step(model.model.sites, None),
                components,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--seq-len", type=int, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    cfg = (
        LMTargetedExperimentConfig.model_validate(raw)
        if "nontarget" in raw
        else LMExperimentConfig.model_validate(raw)
    )
    world = cfg.runtime.mesh.world_size
    assert jax.device_count() == world, (
        f"the trace gate needs the declared world size as local devices: mesh "
        f"{cfg.runtime.mesh} wants {world}, found {jax.device_count()} — run with "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={world}"
    )
    seat = abstract_seat(cfg)
    declared_edge = seat.output_edges[0]
    model = seat.placed(declared_edge)
    match cfg:
        case LMTargetedExperimentConfig():
            declared = trace_targeted_train_step_at_declared_topology(
                cfg, seat, model, args.seq_len
            )
        case LMExperimentConfig():
            declared = trace_train_step_at_declared_topology(cfg, seat, model, args.seq_len)
    print(
        f"trace gate OK: {args.config.name} jit_step lowers at {cfg.runtime.mesh} "
        f"@ {_edge_label(declared_edge)}",
        flush=True,
    )
    match cfg.eval:
        case None:
            print(f"trace gate: {args.config.name} has no eval: block", flush=True)
        case EvalConfig():
            for edge in seat.output_edges:
                trace_eval_programs(
                    cfg.eval,
                    seat.placed(edge),
                    edge,
                    declared.state.decomposition,
                    {
                        name: adversary.sources
                        for name, adversary in declared.state.training.adversaries.items()
                    },
                    seat.mesh,
                    args.seq_len,
                )


if __name__ == "__main__":
    main()
