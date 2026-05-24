"""Lab-side helpers around `ComponentModel` that aren't needed by `optimize()` itself.

`ComponentModel` lives in core because `optimize()` builds and runs it. The two helpers
below (rebuilding a model from a saved checkpoint, and reading per-component activations
out of cached pre-weight acts) are only used by lab postprocessing / the app / harvest,
so they live here as free functions instead of bloating the core class.
"""

from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.ci_fns import (
    CiConfig,
    GlobalCiConfig,
    LayerwiseCiConfig,
)
from param_decomp.ci_sigmoids import SigmoidType
from param_decomp.component_model import ComponentModel
from param_decomp.decomposition_targets import (
    DecompositionTargetConfig,
    insert_identity_operations_,
    resolve_decomposition_targets,
)


def _validate_checkpoint_ci_config_compatibility(
    state_dict: dict[str, Tensor], ci_config: CiConfig
) -> None:
    """Assert the checkpoint's CI weight keys match the layerwise/global mode in `ci_config`."""
    has_layerwise_ci_fns = any(k.startswith("ci_fn._ci_fns") for k in state_dict)
    has_global_ci_fn = any(k.startswith("ci_fn._global_ci_fn") for k in state_dict)

    match ci_config:
        case LayerwiseCiConfig():
            assert has_layerwise_ci_fns, (
                f"Config specifies layerwise CI but checkpoint has no ci_fn._ci_fns keys "
                f"(has ci_fn._global_ci_fn: {has_global_ci_fn})"
            )
        case GlobalCiConfig():
            assert has_global_ci_fn, (
                f"Config specifies global CI but checkpoint has no ci_fn._global_ci_fn keys "
                f"(has ci_fn._ci_fns: {has_layerwise_ci_fns})"
            )


def load_component_model_from_checkpoint(
    *,
    ci_config: CiConfig,
    sigmoid_type: SigmoidType,
    decomposition_targets: list[DecompositionTargetConfig],
    identity_decomposition_targets: list[DecompositionTargetConfig] | None,
    checkpoint_path: Path,
    target_model: nn.Module,
    run_batch: RunBatch,
    tied_weights: list[tuple[str, str]] | None = None,
) -> ComponentModel:
    """Rebuild a ComponentModel from a saved PD checkpoint plus a user-supplied target.

    The caller owns target loading (HF, in-repo pretrain runs, custom user models), so
    this function takes the already-instantiated target plus its run-batch function and
    only the slices of `PDConfig` that affect model construction.

    Args:
        ci_config: CI-fn configuration the checkpoint was trained with.
        sigmoid_type: Output sigmoid variant the CI fn squashes through.
        decomposition_targets: Module patterns to decompose into components.
        identity_decomposition_targets: Module patterns that get an `Identity` shim
            inserted before decomposition; ``None`` to skip.
        checkpoint_path: ``model_<step>.pth`` produced by `RunSink.checkpoint`.
        target_model: Frozen target model the components decompose.
        run_batch: Callable that runs `target_model` on a batch.
        tied_weights: Pairs of (src, tgt) component names whose ``U``/``V`` are tied
            via transpose after loading.

    Returns:
        A `ComponentModel` with weights loaded from `checkpoint_path`.
    """
    target_model.eval()
    target_model.requires_grad_(False)

    if identity_decomposition_targets is not None:
        insert_identity_operations_(
            target_model,
            identity_decomposition_targets=identity_decomposition_targets,
        )

    all_targets = list(decomposition_targets)
    if identity_decomposition_targets is not None:
        for target in identity_decomposition_targets:
            all_targets.append(
                DecompositionTargetConfig(
                    module_pattern=f"{target.module_pattern}.pre_identity", C=target.C
                )
            )
    resolved_targets = resolve_decomposition_targets(target_model, all_targets)

    comp_model = ComponentModel(
        target_model=target_model,
        run_batch=run_batch,
        decomposition_targets=resolved_targets,
        ci_config=ci_config,
        sigmoid_type=sigmoid_type,
    )

    comp_model_weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint_ci_config_compatibility(comp_model_weights, ci_config)
    comp_model.load_state_dict(comp_model_weights)

    if tied_weights is not None:
        for src_name, tgt_name in tied_weights:
            tgt = comp_model.components[tgt_name]
            src = comp_model.components[src_name]
            assert tgt is not None and src is not None, (
                f"Cannot tie weights between {src_name} and {tgt_name} - one or both are None"
            )
            tgt.U.data = src.V.data.T
            tgt.V.data = src.U.data.T

    return comp_model


def get_all_component_acts(
    model: ComponentModel,
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "..."]],
) -> dict[str, Float[Tensor, "... C"]]:
    """Compute per-component activations ``V^T @ x`` for every decomposed layer.

    Layers in `pre_weight_acts` that have no matching entry in `model.components` are
    skipped silently.

    Args:
        model: ComponentModel holding the per-layer `Components` modules.
        pre_weight_acts: Cached pre-weight activations keyed by target module path;
            shape ``(..., d_in)`` (or integer indices for embeddings).

    Returns:
        Dict mapping layer name to component-space activations of shape ``(..., C)``.
    """
    return {
        layer: model.components[layer].get_component_acts(acts)
        for layer, acts in pre_weight_acts.items()
        if layer in model.components
    }
