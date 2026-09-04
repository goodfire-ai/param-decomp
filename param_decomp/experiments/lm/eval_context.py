"""The LM eval pass and its shared per-batch context (torch-oracle `MetricContext`).

`LMBatchContext` is built ONCE per eval batch by one jitted step — the clean forward
(capturing the CI taps plus every due operation's declared demands) and the CI envelope —
and every batched operation reads from it. Operations that need masked forwards or
ascents run their own steps ON TOP of these values; none recomputes the clean side.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import jax
from jax.sharding import Mesh
from jaxtyping import Array

from param_decomp.core.ci_fn import CI, PlacedCIFn, evaluate_ci
from param_decomp.core.components import ComponentStacks
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.model import (
    CaptureKeys,
    PlacedModel,
    prepare_compute_weights,
    select_captures,
)
from param_decomp.core.recon import ForwardObservations
from param_decomp.core.run import EvalInvocation
from param_decomp.core.sharding import batch_shard_leading
from param_decomp.experiments.lm.eval import PreparedLMBatch


@dataclass(frozen=True)
class LMEvalPass(EvalInvocation):
    """Pass-scoped inputs: the raw token batches feed `batch_contexts`; pass-level
    operations (arithmetic's own probe, well-temperedness) read them directly."""

    pass_index: int
    batches: tuple[jax.Array, ...]


@dataclass(frozen=True)
class LMBatchContext:
    """One eval batch's shared forward products, all sharded device values.

    `ci` is the compute-precision envelope from `evaluate_ci` (what the masked-forward
    consumers read); reduction consumers that historically squashed fp32 preactivations
    recompute their fp32 views from `ci.preactivations`, preserving each metric's exact
    numerics. Its per-batch residency matches what the training step already
    materializes. `captures` holds only the operations' declared demands — the CI taps
    are consumed inside the context step and never leave it."""

    pass_index: int
    batch_index: int
    tokens: Array
    clean_output: Array
    captures: dict[str, Array]
    ci: CI
    prepared_weights: Any


type LMBatchContextStep = Callable[
    [PlacedModel, ComponentStacks, PlacedCIFn, Array],
    tuple[Array, Array, dict[str, Array], CI, Any],
]
"""`(model, components, placed_ci_fn, token_ids) -> (tokens, clean_output, captures, ci,
prepared_weights)`. `model` (frozen-weight-bearing) is the jit ARG."""


def make_lm_batch_context_step(
    model_static: PlacedModel,
    ci_capture_keys: CaptureKeys,
    operation_capture_keys: CaptureKeys,
    mesh: Mesh | None,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> LMBatchContextStep:
    """One clean forward capturing the union of CI taps and operation demands, plus the
    CI envelope and the pass's bf16 compute weights — the whole shared side of a batch."""
    del model_static
    capture_keys = ci_capture_keys | operation_capture_keys

    def context_step(
        model: PlacedModel, components: ComponentStacks, placed_ci_fn: PlacedCIFn, token_ids: Array
    ) -> tuple[Array, Array, dict[str, Array], CI, Any]:
        tokens = batch_shard_leading(token_ids, mesh)
        result = model.clean_forward(tokens, capture_keys)
        # No batch-sharding constraint on `ci.lower`: under the Explicit mesh the CI
        # envelope is already typed batch-sharded (and C ÷tp) by the placed CI fn — the
        # pre-Explicit replication OOM this once pinned is structurally impossible.
        ci = evaluate_ci(
            placed_ci_fn, select_captures(result.captures, ci_capture_keys), remat=False
        )
        clean_output = batch_shard_leading(result.output, mesh)
        captures = select_captures(result.captures, operation_capture_keys)
        return tokens, clean_output, captures, ci, prepare_compute_weights(model, components)

    return filter_jit(context_step, compiler_options=compiler_options)


def make_lm_batch_contexts(
    context_step: LMBatchContextStep, model: PlacedModel
) -> Callable[[LMEvalPass], Iterator[LMBatchContext]]:
    def batch_contexts(eval_pass: LMEvalPass) -> Iterator[LMBatchContext]:
        decomposition = eval_pass.state.decomposition
        for batch_index, token_ids in enumerate(eval_pass.batches):
            tokens, clean_output, captures, ci, prepared_weights = context_step(
                model, decomposition.components, eval_pass.placed_ci_fn, token_ids
            )
            yield LMBatchContext(
                pass_index=eval_pass.pass_index,
                batch_index=batch_index,
                tokens=tokens,
                clean_output=clean_output,
                captures=captures,
                ci=ci,
                prepared_weights=prepared_weights,
            )

    return batch_contexts


def prepared_batch_from_context(
    context: LMBatchContext, hidden_acts_capture_keys: CaptureKeys
) -> PreparedLMBatch[Any]:
    """The scalar kernels' batch view over the shared context — a reshaping, no compute."""
    return PreparedLMBatch(
        tokens=context.tokens,
        clean=ForwardObservations(
            context.clean_output,
            select_captures(context.captures, hidden_acts_capture_keys),
        ),
        prepared_weights=context.prepared_weights,
        ci_lower=context.ci.lower,
        valid_row_mask=None,
    )
