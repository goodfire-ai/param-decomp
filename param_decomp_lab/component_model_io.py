"""Lab-side `ComponentModel` helpers for postprocessing, the app, and harvest.

Rebuilds a component model from a saved checkpoint, and reads per-component activations
from cached pre-weight acts. Two checkpoint formats are supported: the core
`ComponentModel` (single-pool and pre-`e8ff5a64` 3-pool) and the vendored
`LMComponentModel` (post-`e8ff5a64` 3-pool), the latter wrapped in `VendoredHarvestModel`
so both expose the same `HarvestableComponentModel` surface.
"""

from pathlib import Path
from typing import Literal, Protocol, override, runtime_checkable

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import CIOutputs, ComponentModel, OutputWithCache
from param_decomp.components import Components
from param_decomp.decomposition_targets import (
    insert_identity_operations_,
    resolve_decomposition_targets,
)
from param_decomp_config.ci_fn import (
    CiConfig,
    GlobalSharedMlpCiConfig,
    GlobalSharedTransformerCiFnConfig,
    LayerwiseCiConfig,
)
from param_decomp_config.decomposition_target import DecompositionTargetConfig
from param_decomp_config.pd import PDConfig
from param_decomp_config.routing import SamplingType
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel


@runtime_checkable
class HarvestableComponentModel(Protocol):
    """The surface the torch adapter path (`PDAdapter`) needs.

    Satisfied by both the core `ComponentModel` and `VendoredHarvestModel`. `target_model`
    is the bare transformer (for `TransformerTopology`); `components`/`module_to_c`/
    `target_module_paths` are pure queries; `forward(cache_type="input")` returns logits +
    pre-weight acts; `calc_causal_importances` squashes those into a `CIOutputs`.
    """

    @property
    def target_model(self) -> nn.Module: ...

    @property
    def components(self) -> dict[str, Components]: ...

    @property
    def module_to_c(self) -> dict[str, int]: ...

    @property
    def target_module_paths(self) -> list[str]: ...

    def to(self, device: torch.device | str) -> "HarvestableComponentModel": ...

    def eval(self) -> "HarvestableComponentModel": ...

    def __call__(
        self, batch: Int[Tensor, "batch pos"], *, cache_type: Literal["input"]
    ) -> OutputWithCache: ...

    def calc_causal_importances(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "..."]],
        sampling: SamplingType,
        detach_inputs: bool,
    ) -> CIOutputs: ...


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
        case GlobalSharedMlpCiConfig() | GlobalSharedTransformerCiFnConfig():
            assert has_global_ci_fn, (
                f"Config specifies global CI but checkpoint has no ci_fn._global_ci_fn keys "
                f"(has ci_fn._ci_fns: {has_layerwise_ci_fns})"
            )


def load_component_model(
    pd_config: PDConfig,
    checkpoint_path: Path,
    target_model: nn.Module,
    run_batch: RunBatch,
) -> ComponentModel:
    """Rebuild a `ComponentModel` from a saved PD checkpoint plus a caller-supplied target.

    The caller owns target loading (HF, in-repo pretrain, custom); everything else
    needed to reconstruct the model comes from `pd_config`.
    """
    target_model.eval()
    target_model.requires_grad_(False)

    identity_targets = pd_config.identity_decomposition_targets
    if identity_targets is not None:
        insert_identity_operations_(target_model, identity_decomposition_targets=identity_targets)

    all_targets = list(pd_config.decomposition_targets)
    if identity_targets is not None:
        for target in identity_targets:
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
        ci_config=pd_config.ci_config,
        sigmoid_type=pd_config.sigmoid_type,
    )

    comp_model_weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint_ci_config_compatibility(comp_model_weights, pd_config.ci_config)
    comp_model.load_state_dict(comp_model_weights)

    if pd_config.tied_weights is not None:
        for src_name, tgt_name in pd_config.tied_weights:
            tgt = comp_model.components[tgt_name]
            src = comp_model.components[src_name]
            assert tgt is not None and src is not None, (
                f"Cannot tie weights between {src_name} and {tgt_name} - one or both are None"
            )
            tgt.U.data = src.V.data.T
            tgt.V.data = src.U.data.T

    return comp_model


def load_vendored_component_model(
    pd_config: PDConfig,
    checkpoint_path: Path,
    target_model: nn.Module,
) -> LMComponentModel:
    """Rebuild an `LMComponentModel` (vendored 3-pool format) from a saved PD checkpoint.

    Mirrors `load_component_model` but builds the vendored model and freezes the inlined
    target. The 3-pool config fixes `identity_decomposition_targets=None`, so unlike the
    core loader there is no identity-op insertion path.
    """
    assert pd_config.identity_decomposition_targets is None, (
        "vendored 3-pool checkpoints never carry identity decomposition targets; "
        f"got {pd_config.identity_decomposition_targets}"
    )
    target_model.eval()
    target_model.requires_grad_(False)

    resolved_targets = resolve_decomposition_targets(
        target_model, list(pd_config.decomposition_targets)
    )

    comp_model = LMComponentModel.build(
        target_model=target_model,
        decomposition_targets=resolved_targets,
        ci_config=pd_config.ci_config,
        sigmoid_type=pd_config.sigmoid_type,
    )

    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint_ci_config_compatibility(weights, pd_config.ci_config)
    comp_model.load_state_dict(weights)

    if pd_config.tied_weights is not None:
        for src_name, tgt_name in pd_config.tied_weights:
            tgt = comp_model.components[tgt_name]
            src = comp_model.components[src_name]
            tgt.U.data = src.V.data.T
            tgt.V.data = src.U.data.T

    return comp_model


class VendoredHarvestModel(nn.Module):
    """Adapts an `LMComponentModel` to the core `forward(cache_type="input") -> OutputWithCache`
    surface the harvest path expects, without polluting `LMComponentModel` with core-mimicking
    methods. As an `nn.Module` holding the wrapped model, `.to` / `.eval` propagate."""

    def __init__(self, lm: LMComponentModel):
        super().__init__()
        self._lm = lm

    @override
    def forward(
        self,
        idx: Int[Tensor, "batch pos"],
        mask_infos: object | None = None,
        cache_type: str = "input",
    ) -> OutputWithCache:
        # Harvest-only surface: pre-weight-act capture with no masking. Attribution-graph /
        # intervention compute (cache_type="component_acts"/"output"/"none", or mask_infos)
        # is intentionally not implemented for vendored 3-pool runs — the app supports only
        # the component + autointerp viewers for these.
        if mask_infos is not None or cache_type != "input":
            raise NotImplementedError(
                "VendoredHarvestModel implements only harvest-style forward "
                f"(cache_type='input', no mask_infos); got cache_type={cache_type!r}, "
                f"mask_infos={'set' if mask_infos is not None else 'None'}. "
                "Attribution/graph/intervention compute is unsupported for vendored 3-pool runs."
            )
        logits, cache = self._lm.forward_with_pre_weight_acts(idx)
        return OutputWithCache(output=logits, cache=cache)

    def calc_causal_importances(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "..."]],
        sampling: SamplingType,
        detach_inputs: bool,
    ) -> CIOutputs:
        return self._lm.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, sampling=sampling, detach_inputs=detach_inputs
        )

    @property
    def components(self) -> dict[str, Components]:
        return self._lm.components

    @property
    def module_to_c(self) -> dict[str, int]:
        return self._lm.module_to_c

    @property
    def target_module_paths(self) -> list[str]:
        return self._lm.target_module_paths

    @property
    def target_model(self) -> nn.Module:
        return self._lm.model


def detect_checkpoint_format(checkpoint_path: Path) -> str:
    """Peek state-dict key prefixes to tell the two checkpoint formats apart.

    Vendored (`LMComponentModel`) inlines the frozen target under `model.*`; core
    (`ComponentModel`) keeps it under `target_model.*` with components under `_components.*`.
    """
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    keys = list(weights.keys())
    if any(k.startswith("model.") for k in keys):
        return "vendored"
    assert any(k.startswith(("target_model.", "_components.")) for k in keys), (
        f"unrecognized checkpoint format at {checkpoint_path}; sample keys: {keys[:5]}"
    )
    return "core"


def get_all_component_acts(
    model: HarvestableComponentModel,
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "..."]],
) -> dict[str, Float[Tensor, "... C"]]:
    """Per-component activations `V^T @ x` for every decomposed layer.

    Layers in `pre_weight_acts` with no matching entry in `model.components` are skipped
    silently.
    """
    return {
        layer: model.components[layer].get_component_acts(acts)
        for layer, acts in pre_weight_acts.items()
        if layer in model.components
    }
