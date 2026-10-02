"""The LM eval pass and its shared per-batch context (torch-oracle `MetricContext`).

`LMBatchContext` is built ONCE per eval batch by one jitted step — the clean forward
(capturing the CI taps plus every due operation's declared demands) and the CI envelope —
and every shared-forward operation reads from it. Operations that need masked forwards or
ascents run their own steps ON TOP of these values; none recomputes the clean side.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import jax
from jax.sharding import Mesh
from jaxtyping import Array

from param_decomp.core.adversary import SourceStacks
from param_decomp.core.ci_fn.interface import CI, CIFn
from param_decomp.core.ci_fn.runtime import evaluate_ci_from_captures
from param_decomp.core.components import ComponentStacks
from param_decomp.core.model import CaptureKeys, ComponentActivations, PlacedModel, select_captures
from param_decomp.core.recon import ForwardObservations
from param_decomp.core.run import EvalInvocation
from param_decomp.core.sharding import batch_shard_leading
from param_decomp.experiments.lm.eval import PreparedLMBatch
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput


@dataclass(frozen=True)
class LMEvalPass[TargetIn, Conditioning](EvalInvocation[Conditioning]):
    """Inputs for one evaluation occasion. Batches feed the shared forward or can be
    consumed directly by standalone operations such as well-temperedness."""

    pass_index: int
    batches: tuple[TargetIn, ...]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMBatchForwardProducts[TargetIn, PreparedT: ComponentActivations, Conditioning]:
    """Device values produced together by one clean forward and CI evaluation.

    `tokens` retains the target's input type, including document layout. `conditioning`
    carries the complete input for masked execution; `captures` contains only the
    consuming operations' requested activations. The CI envelope stays in compute
    precision; consumers that require fp32 squashings derive them from preactivations.
    """

    tokens: TargetIn
    clean_output: LMOutput
    captures: dict[str, Array]
    ci: CI
    prepared_weights: PreparedT
    conditioning: Conditioning


@dataclass(frozen=True)
class LMBatchContext[TargetIn, PreparedT: ComponentActivations, Conditioning]:
    """One batch's forward products, position in the eval pass, and persistent sources.

    Sources retain their stored representation; consumers read `source_values_to_float`.
    """

    pass_index: int
    batch_index: int
    forward: LMBatchForwardProducts[TargetIn, PreparedT, Conditioning]
    persistent_sources: dict[str, SourceStacks]


type LMBatchContextStep[
    TargetIn,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
] = Callable[
    [
        PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        ComponentStacks,
        CIFn[Conditioning],
        TargetIn,
    ],
    LMBatchForwardProducts[TargetIn, PreparedT, Conditioning],
]
"""`(model, components, ci_fn, token_ids) -> LMBatchForwardProducts`.
The frozen model remains a jit argument.
"""


def make_lm_batch_context_step[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    operation_capture_keys: CaptureKeys,
    mesh: Mesh | None,
) -> LMBatchContextStep[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """One clean forward capturing the union of CI taps and operation demands, plus the
    CI envelope and the pass's bf16 compute weights — the whole shared side of a batch."""
    del model_static
    capture_keys = ci_capture_keys | operation_capture_keys

    def context_step(
        model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        components: ComponentStacks,
        ci_fn: CIFn[Conditioning],
        token_ids: TargetIn,
    ) -> LMBatchForwardProducts[TargetIn, PreparedT, Conditioning]:
        tokens = jax.tree.map(lambda a: batch_shard_leading(a, mesh), token_ids)
        result = model.clean_forward(tokens, capture_keys)
        # The CI function's output placement already carries batch and component ownership.
        prepared_weights = model.prepare_compute_weights(components)
        ci = evaluate_ci_from_captures(
            ci_fn.prepare(),
            select_captures(result.captures, ci_capture_keys),
            result.conditioning,
            prepared_weights,
            sequence=result.sequence,
            remat=False,
        )
        clean_output = model.pin_output_batch(result.output, mesh)
        captures = select_captures(result.captures, operation_capture_keys)
        return LMBatchForwardProducts(
            tokens=tokens,
            clean_output=clean_output,
            captures=captures,
            ci=ci,
            prepared_weights=prepared_weights,
            conditioning=result.conditioning,
        )

    return context_step


def make_lm_batch_contexts[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    context_step: LMBatchContextStep[TargetIn, PreparedT, Conditioning, PreparedMaskingT],
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
) -> Callable[
    [LMEvalPass[TargetIn, Conditioning]],
    Iterator[LMBatchContext[TargetIn, PreparedT, Conditioning]],
]:
    def batch_contexts(
        eval_pass: LMEvalPass[TargetIn, Conditioning],
    ) -> Iterator[LMBatchContext[TargetIn, PreparedT, Conditioning]]:
        decomposition = eval_pass.decomposition
        for batch_index, token_ids in enumerate(eval_pass.batches):
            forward = context_step(model, decomposition.components, decomposition.ci_fn, token_ids)
            yield LMBatchContext(
                pass_index=eval_pass.pass_index,
                batch_index=batch_index,
                forward=forward,
                persistent_sources=eval_pass.persistent_sources,
            )

    return batch_contexts


def prepared_batch_from_context[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
](
    context: LMBatchContext[TargetIn, PreparedT, Conditioning],
    reconstruction_capture_keys: CaptureKeys,
) -> PreparedLMBatch[TargetIn, PreparedT, Conditioning]:
    """The scalar kernels' batch view over the shared context — a reshaping, no compute."""
    return PreparedLMBatch(
        tokens=context.forward.tokens,
        clean=ForwardObservations(
            context.forward.clean_output,
            select_captures(context.forward.captures, reconstruction_capture_keys),
        ),
        prepared_weights=context.forward.prepared_weights,
        conditioning=context.forward.conditioning,
        ci_lower=context.forward.ci.lower,
        valid_row_mask=None,
    )
