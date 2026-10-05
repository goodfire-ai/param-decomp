"""The `RouterDivergence` eval: per MoE layer, how far the masked model's expert router
drifts from the target's, under the target's expert routing pinned at every layer.

Both sides of a comparison are one layer's router read out at every token: its
softmax over the experts, the experts that softmax selects, and the mixing weights the
forward applied. The target's side is the clean forward's — the selection and weights are
the pinned `BlockSelection`'s rows. The masked side is a masked forward run on the SAME
clean expert selection: the target's router applied to the masked residual, the
experts THAT softmax would have selected (a counterfactual read-out — nothing downstream
sees it), and the mixing weights the masked forward actually applied at the pinned
experts. Because every layer is pinned, layer `l`'s read-out sees a residual produced
under the target's expert selection at layers `0..l-1`. Changes in selected expert
identities cannot propagate; masking and mixing-weight changes still affect later residuals.

Per strategy (`RouterDivergenceStrategy`), one router readout per batch, returning
captures only — the router taps of every MoE layer — so the output edge is dead code. The distances are per-token fp32 scalars, token-summed
per layer on device and folded across the pass's batches on the host; means divide once.

The router taps and the top-k selection are target-owned (`ExpertRouterModel`): a target
without an expert router refuses when the step is built.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

from param_decomp.core.adversary import SourceStacks, source_values_to_float
from param_decomp.core.components import BlockSelection, SiteCI
from param_decomp.core.masking import (
    materialize_masking,
    sample_source_rows,
    source_masking,
)
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    ComponentActivations,
    MaterializedMasking,
    PlacedModel,
    StochasticMasking,
)
from param_decomp.core.recon_eval import fresh_pgd_masking
from param_decomp.experiments.lm.eval_config import (
    CIMaskedStrategy,
    FreshPGDStrategy,
    PersistentStrategy,
    RouterDivergenceConfig,
    RouterDivergenceStrategy,
    StochasticStrategy,
)
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments, LMBatchWithRouting
from param_decomp.targets.lm_output import LMOutput

LOG_PREFIX = "eval/router_divergence"


@runtime_checkable
class ExpertRouterModel(Protocol):
    """The capability this eval needs from a target: which layers carry an expert router
    (row `i` of the pinned `BlockSelection` is layer `expert_router_layers[i]`), the
    capture keys of that router's softmax over all experts (`[.., E]` fp32) and of the
    mixing weights the forward applied (`[.., k]` fp32) — the same keys on the clean and
    masked paths — and the router's own top-k selection from a softmax."""

    @property
    def expert_router_layers(self) -> tuple[int, ...]: ...

    def router_probs_capture_key(self, layer: int) -> str: ...

    def router_weights_capture_key(self, layer: int) -> str: ...

    def select_experts(self, probs: Float[Array, "*lead E"]) -> Int[Array, "*lead k"]: ...


def expert_router_model[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
) -> ExpertRouterModel:
    """Narrow the bundle's target to the router-readout surface — re-derived from the
    traced model arg each call, never closed over (the HLO-baking rule)."""
    inner = model.model
    assert isinstance(inner, ExpertRouterModel), (
        f"RouterDivergence needs a target with an expert router (the ExpertRouterModel "
        f"surface); {type(inner).__name__} has none"
    )
    return inner


def router_probs_capture_keys[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
) -> tuple[str, ...]:
    """The router softmax capture key of every MoE layer, in `expert_router_layers`
    order — the clean-capture demand this eval declares on the shared batch context."""
    router_model = expert_router_model(model_static)
    return tuple(
        router_model.router_probs_capture_key(layer) for layer in router_model.expert_router_layers
    )


@jaxtyped(typechecker=beartype)
def kl_divergence(
    target_probs: Float[Array, "*lead E"], masked_probs: Float[Array, "*lead E"]
) -> Float[Array, "*lead"]:
    """`Σ_e p_target · (log p_target − log p_masked)` in fp32, both logs clamped at 1e-12."""
    p = target_probs.astype(jnp.float32)
    q = masked_probs.astype(jnp.float32)
    return jnp.sum(p * (jnp.log(jnp.clip(p, min=1e-12)) - jnp.log(jnp.clip(q, min=1e-12))), axis=-1)


@jaxtyped(typechecker=beartype)
def topk_overlap(
    target_indices: Int[Array, "*lead k"], masked_indices: Int[Array, "*lead k"]
) -> Float[Array, "*lead"]:
    """Fraction of the target's `k` experts the masked probabilities also select. Each
    side's `k` experts are distinct, so counting pairwise equalities counts the
    intersection."""
    k = target_indices.shape[-1]
    matches = target_indices[..., :, None] == masked_indices[..., None, :]
    return jnp.sum(matches, axis=(-2, -1)).astype(jnp.float32) / k


@jaxtyped(typechecker=beartype)
def weight_mae(
    target_weights: Float[Array, "*lead k"], masked_weights: Float[Array, "*lead k"]
) -> Float[Array, "*lead"]:
    """Mean absolute error over the `k` mixing weights the two forwards applied — under
    the target's routing, exactly the per-weight perturbation of the weights the masked
    forward mixed the target's experts with."""
    k = target_weights.shape[-1]
    return (
        jnp.sum(
            jnp.abs(target_weights.astype(jnp.float32) - masked_weights.astype(jnp.float32)),
            axis=-1,
        )
        / k
    )


type LayerVector = Float[Array, " n_layers"] | Float[np.ndarray, " n_layers"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RouterDivergenceSums:
    """One strategy's token-summed distances, a `[n_layers]` vector per distance (row `i`
    is `expert_router_layers[i]`); the field names are the log-key spellings. Device fp32
    inside the step, host float64 across a pass."""

    kl: LayerVector
    topk_overlap: LayerVector
    weight_mae: LayerVector

    def __add__(self, other: "RouterDivergenceSums") -> "RouterDivergenceSums":
        return RouterDivergenceSums(
            kl=self.kl + other.kl,
            topk_overlap=self.topk_overlap + other.topk_overlap,
            weight_mae=self.weight_mae + other.weight_mae,
        )

    def host_float64(self) -> "RouterDivergenceSums":
        return RouterDivergenceSums(
            kl=np.asarray(self.kl, np.float64),
            topk_overlap=np.asarray(self.topk_overlap, np.float64),
            weight_mae=np.asarray(self.weight_mae, np.float64),
        )


def strategy_masking[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    strategy: RouterDivergenceStrategy,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    prepared_weights: PreparedT,
    conditioning: Conditioning,
    ci_lower: Mapping[str, SiteCI],
    clean_output: LMOutput,
    persistent_sources: Mapping[str, SourceStacks],
    key: PRNGKeyArray,
    mesh: Mesh | None,
) -> MaterializedMasking:
    """One strategy's masking, composed over the CI envelope as in training."""
    match strategy:
        case CIMaskedStrategy():
            return MaterializedMasking(component_masks=ci_lower)
        case StochasticStrategy():
            return materialize_masking(StochasticMasking(ci=ci_lower, draw_key=key))
        case FreshPGDStrategy(n_steps=n_steps, step_size=step_size):

            def output_kl_at_masking(masking: MaterializedMasking) -> Array:
                """The ascent objective — the target's own output reconstruction loss,
                the eval referee's spelling (`eval.make_fresh_pgd_scorer`)."""
                masked_output = model.masked_forward(
                    prepared_weights,
                    conditioning,
                    masking=model.model.prepare_masking(masking),
                    routes=None,
                    capture_keys=EMPTY_CAPTURE_KEYS,
                    remat=False,
                ).output
                return model.recon_loss_fn(
                    model.pin_output_batch(masked_output, mesh), clean_output
                )

            return fresh_pgd_masking(
                model.model.sites,
                ci_lower,
                _routing(conditioning).indices.shape[1:-1],
                key,
                step_size,
                n_steps,
                output_kl_at_masking,
            )
        case PersistentStrategy(state_key=state_key):
            sources = sample_source_rows(
                key, ci_lower, source_values_to_float(persistent_sources[state_key]).per_site()
            )
            return materialize_masking(source_masking(ci_lower, sources))


def make_router_divergence_step[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    metric: RouterDivergenceConfig,
    mesh: Mesh | None,
):
    """The metric's one jitted program: per strategy, its masked forward over the pinned
    selection capturing every MoE layer's router taps, reduced to per-layer token sums
    with the token count they sum over. `model` (frozen-weight-bearing) is the jit ARG,
    never closed over; `model_static` only shapes the program."""
    assert model_static.model.has_position_axis, "RouterDivergence is LM-only"
    router_model_static = expert_router_model(model_static)
    layers = router_model_static.expert_router_layers
    probs_keys = router_probs_capture_keys(model_static)
    weights_keys = tuple(router_model_static.router_weights_capture_key(layer) for layer in layers)
    masked_capture_keys = frozenset(probs_keys) | frozenset(weights_keys)

    def step(
        model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        prepared_weights: PreparedT,
        conditioning: Conditioning,
        ci_lower: Mapping[str, SiteCI],
        clean_router_probs_by_key: dict[str, Array],
        clean_output: LMOutput,
        persistent_sources: dict[str, SourceStacks],
        key: PRNGKeyArray,
    ) -> dict[str, tuple[RouterDivergenceSums, int]]:
        router_model = expert_router_model(model)
        selection = _routing(conditioning)
        assert selection.indices.shape[0] == len(layers), (selection.indices.shape, layers)
        n_tokens = int(np.prod(_routing(conditioning).indices.shape[1:-1]))

        def masked_layer_sums(masking: MaterializedMasking) -> RouterDivergenceSums:
            captures = model.masked_forward(
                prepared_weights,
                conditioning,
                masking=model.model.prepare_masking(masking),
                routes=None,
                capture_keys=masked_capture_keys,
                remat=False,
            ).captures
            kl: list[Array] = []
            overlap: list[Array] = []
            mae: list[Array] = []
            for row, (probs_key, weights_key) in enumerate(
                zip(probs_keys, weights_keys, strict=True)
            ):
                masked_probs = captures[probs_key]
                kl.append(
                    jnp.sum(kl_divergence(clean_router_probs_by_key[probs_key], masked_probs))
                )
                overlap.append(
                    jnp.sum(
                        topk_overlap(
                            selection.indices[row], router_model.select_experts(masked_probs)
                        )
                    )
                )
                mae.append(jnp.sum(weight_mae(selection.weights[row], captures[weights_key])))
            return RouterDivergenceSums(
                kl=jnp.stack(kl), topk_overlap=jnp.stack(overlap), weight_mae=jnp.stack(mae)
            )

        results: dict[str, tuple[RouterDivergenceSums, int]] = {}
        for strategy_idx, strategy in enumerate(metric.strategies):
            masking = strategy_masking(
                strategy,
                model,
                prepared_weights,
                conditioning,
                ci_lower,
                clean_output,
                persistent_sources,
                random.fold_in(key, strategy_idx),
                mesh,
            )
            results[strategy.kind] = (masked_layer_sums(masking), n_tokens)
        return results

    return step


@dataclass(frozen=True)
class RouterDivergenceAccumulation:
    """A pass's host-side fold for one strategy: float64 token sums and the token count
    they sum over. Means divide once."""

    sums: RouterDivergenceSums
    n_tokens: int


def empty_router_divergence_sums(n_layers: int) -> RouterDivergenceAccumulation:
    return RouterDivergenceAccumulation(
        sums=RouterDivergenceSums(
            kl=np.zeros(n_layers, np.float64),
            topk_overlap=np.zeros(n_layers, np.float64),
            weight_mae=np.zeros(n_layers, np.float64),
        ),
        n_tokens=0,
    )


def fold_router_divergence(
    accumulated: RouterDivergenceAccumulation,
    batch_sums: RouterDivergenceSums,
    batch_n_tokens: int,
) -> RouterDivergenceAccumulation:
    """Fold one batch's per-layer token sums (device values) into the accumulator."""
    assert batch_n_tokens > 0, batch_n_tokens
    return RouterDivergenceAccumulation(
        sums=accumulated.sums + batch_sums.host_float64(),
        n_tokens=accumulated.n_tokens + batch_n_tokens,
    )


def router_divergence_log_entries(
    strategy: RouterDivergenceStrategy,
    layers: tuple[int, ...],
    accumulated: RouterDivergenceAccumulation,
) -> dict[str, float]:
    """`eval/router_divergence/{kind}/{distance}/layer_{l}` per MoE layer (row `i` of
    each sum vector is `layers[i]`) plus the mean over layers at
    `eval/router_divergence/{kind}/{distance}`, `distance` being each sum field's name."""
    assert accumulated.n_tokens > 0, "no router-divergence data accumulated"
    entries: dict[str, float] = {}
    for field in fields(RouterDivergenceSums):
        means = np.asarray(getattr(accumulated.sums, field.name)) / accumulated.n_tokens
        for layer, mean in zip(layers, means, strict=True):
            entries[f"{LOG_PREFIX}/{strategy.kind}/{field.name}/layer_{layer}"] = float(mean)
        entries[f"{LOG_PREFIX}/{strategy.kind}/{field.name}"] = float(means.mean())
    return entries


def _routing(conditioning: object) -> BlockSelection:
    assert isinstance(conditioning, LMBatchWithRouting), type(conditioning).__name__
    return conditioning.selection
