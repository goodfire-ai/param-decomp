"""An LM run's CI fn initializer: where an architecture's init reads data, it reads one
clean pass of the run's whole step-0 training batch, which must hold no padding: activation
statistics are measured on packed, pretraining-style tokens.

Component conditioning calibrates its readout arms' input scales to that pass's
component activations. Every other architecture initializes from its seed alone.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from param_decomp.core.ci_fn.architecture import CIFnArchitecture
from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.chunkwise.arch import ChunkwiseTransformerCIFnArch
from param_decomp.core.ci_fn.implementations.global_mlp import GlobalMLPCIFnArch
from param_decomp.core.ci_fn.implementations.global_transformer.arch import (
    GlobalTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.backbone import (
    BackboneCIFn,
    CIFnBackboneArchitecture,
)
from param_decomp.core.ci_fn.implementations.transformer.component_conditioning import (
    ConditionedCIFnArch,
)
from param_decomp.core.components import ComponentStacks, SiteSpec
from param_decomp.core.init_placed import CIFnInitializer, seeded_ci_fn_initializer
from param_decomp.core.model import ComponentActivations, PlacedModel
from param_decomp.core.placement import PlacementRules
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.lm.inputs import input_token_ids
from param_decomp.targets.lm_output import LMOutput


class LMCIFnInitInputs[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](eqx.Module):
    """One clean pass: the placed target and the training batch it runs on."""

    placed: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT]
    batch: TargetIn


class _CalibratedConditionedInitializer[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](eqx.Module):
    """The conditioned CI fn with its readouts calibrated to the clean pass through the
    components it is given."""

    arch: ConditionedCIFnArch[CIFnBackboneArchitecture] = eqx.field(static=True)
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    rules: PlacementRules = eqx.field(static=True)
    inputs: LMCIFnInitInputs[TargetIn, PreparedT, Conditioning, PreparedMaskingT]

    def __call__(self, components: ComponentStacks, key: PRNGKeyArray) -> BackboneCIFn:
        placed = self.inputs.placed
        calibration = self.arch.calibration
        batch = self.inputs.batch
        n_tokens = input_token_ids(batch).size
        assert n_tokens >= calibration.min_n_tokens, (
            f"the calibration batch holds {n_tokens} tokens, fewer than "
            f"calibration.min_n_tokens {calibration.min_n_tokens}"
        )
        clean = placed.clean_forward(_unpadded(batch), self.arch.capture_keys)
        uncalibrated = self.arch.initialize_backbone(self.sites, self.rules, key)
        calibrated = uncalibrated.with_calibrated_input_scales(
            clean.captures, placed.prepare_compute_weights(components)
        )
        return BackboneCIFn(calibrated)


def _unpadded[TargetIn: LMBatch | LMBatchWithDocuments](batch: TargetIn) -> TargetIn:
    """`batch`, failing if any position is padding (a negative document id)."""
    match batch:
        case LMBatchWithDocuments(sequence=sequence):
            return eqx.error_if(
                batch, jnp.any(sequence.document_ids < 0), "the calibration batch holds padding"
            )
        case LMBatch():
            return batch


def lm_ci_fn_initializer[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    arch: CIFnArchitecture[Conditioning],
    sites: tuple[SiteSpec, ...],
    rules: PlacementRules,
    inputs: LMCIFnInitInputs[TargetIn, PreparedT, Conditioning, PreparedMaskingT],
) -> CIFnInitializer[Conditioning]:
    match arch:
        case ConditionedCIFnArch():
            return _CalibratedConditionedInitializer(arch, sites, rules, inputs)
        case (
            ChunkwiseTransformerCIFnArch()
            | GlobalTransformerCIFnArch()
            | BlockSelectedChunkwiseTransformerCIFnArch()
            | GlobalMLPCIFnArch()
        ):
            return seeded_ci_fn_initializer(arch, sites, rules)
        case other:
            raise AssertionError(f"not an LM CI architecture: {type(other).__name__}")
