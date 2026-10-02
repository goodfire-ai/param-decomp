"""The LM model-output edge: materialized logits or the factored streamed package.

`LMOutput` is the `Out` every LM target binds (`core.model.ForwardResult[LMOutput]`); the
protocol output operations dispatch on it exhaustively. The comparison kernels live in
`targets.losses`.
"""

from dataclasses import dataclass, replace
from functools import partial

import jax
from beartype import beartype
from jax.sharding import Mesh
from jaxtyping import Array, Float, jaxtyped

from param_decomp.core.sharding import batch_shard_leading


@dataclass(frozen=True)
class MaterializedOutputEdge:
    """Materialize logits in the target's native dtype."""


@dataclass(frozen=True)
class StreamedOutputEdge:
    """Keep the final linear map factored; comparisons accumulate logits in fp32 chunks."""

    n_vocab_chunks: int

    def __post_init__(self) -> None:
        assert self.n_vocab_chunks > 0, self.n_vocab_chunks


OutputEdge = MaterializedOutputEdge | StreamedOutputEdge


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("activations", "head"),
    meta_fields=("n_chunks",),
)
@dataclass(frozen=True)
class StreamedLinearOutput:
    """The final linear map left factored so comparisons need only one vocab chunk's
    logits at a time. Consumers accumulate both the chunk logits and online-softmax
    statistics in fp32 (`targets.losses`); the materialized edge instead rounds logits
    to the target's native dtype before comparison.

    The head rides the package because comparison fns see only outputs; under jit it is
    the same traced array as the model's output operand — a reference, never a copy. It
    has no batch axis, so batch pinning applies to `activations` alone; the head stays at
    the target's declared operand layout.

    `linear_output` produces full batch/sequence activations. The package and loss
    kernels also admit sliced or vmapped activations with fewer leading axes.
    """

    activations: Float[Array, "*leading d_model"]
    head: Float[Array, "vocab d_model"]
    n_chunks: int


LMOutput = Array | StreamedLinearOutput
"""The LM output edge. The clean and masked forwards of one model always share one
member; every pairwise consumer refuses a mix."""


@jaxtyped(typechecker=beartype)
def linear_output(
    activations: Float[Array, "batch seq d_model"],
    head: Float[Array, "vocab d_model"],
    edge: OutputEdge,
) -> Float[Array, "batch seq vocab"] | StreamedLinearOutput:
    """Project one LM batch, preserving its batch and sequence axes on either edge."""
    match edge:
        case MaterializedOutputEdge():
            return activations @ head.T
        case StreamedOutputEdge(n_vocab_chunks=n_vocab_chunks):
            assert head.shape[0] % n_vocab_chunks == 0, (head.shape[0], n_vocab_chunks)
            return StreamedLinearOutput(activations=activations, head=head, n_chunks=n_vocab_chunks)


def pin_lm_output_batch(output: LMOutput, mesh: Mesh | None) -> LMOutput:
    match output:
        case StreamedLinearOutput():
            return replace(output, activations=batch_shard_leading(output.activations, mesh))
        case jax.Array():
            return batch_shard_leading(output, mesh)
