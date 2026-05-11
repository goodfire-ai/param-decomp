from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, overload, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from param_decomp.configs import CiConfig, Config, GlobalCiConfig, LayerwiseCiConfig, SamplingType
from param_decomp.identity_insertion import insert_identity_operations_
from param_decomp.interfaces import LoadableModule, RunInfo
from param_decomp.models.batch_and_loss_fns import RunBatch, make_run_batch
from param_decomp.models.components import (
    ComponentsMaskInfo,
    GlobalCiFnWrapper,
    GlobalSharedMLPCiFn,
    GlobalSharedTransformerCiFn,
    LayerwiseCiFnWrapper,
    MLPCiFn,
    TargetLayerConfig,
    VectorMLPCiFn,
    VectorSharedMLPCiFn,
)
from param_decomp.models.decomposed_module import (
    DecomposedEmbedding,
    DecomposedSite,
    install_decomposed_sites,
)
from param_decomp.models.sigmoids import SIGMOID_TYPES, SigmoidType
from param_decomp.param_decomp_types import LayerwiseCiFnType, ModelPath
from param_decomp.utils.general_utils import resolve_class
from param_decomp.utils.module_utils import ModulePathInfo, expand_module_patterns


def move_batch_to_device(batch: Any, device: str | torch.device) -> Any:
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    return batch


def _validate_checkpoint_ci_config_compatibility(
    state_dict: dict[str, Tensor], ci_config: CiConfig
) -> None:
    """Validate that checkpoint CI weights match the config CI mode."""
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


@dataclass
class ParamDecompRunInfo(RunInfo[Config]):
    """Run info from training a ComponentModel (i.e. from a PD run)."""

    config_class = Config
    config_filename = "final_config.yaml"
    checkpoint_prefix = "model"


class OutputWithCache(NamedTuple):
    """Output tensor and cached activations."""

    output: Tensor
    cache: dict[str, Tensor]


@dataclass
class CIOutputs:
    lower_leaky: dict[str, Float[Tensor, "... C"]]
    upper_leaky: dict[str, Float[Tensor, "... C"]]
    pre_sigmoid: dict[str, Tensor]


class ComponentModel(LoadableModule):
    """Wrapper around an arbitrary pytorch model for running PD.

    The underlying *base model* can be any subclass of `nn.Module` (e.g.
    `LlamaForCausalLM`, `AutoModelForCausalLM`) as long as its sub-module names
    are provided in the `module_path_info` list.

    Forward passes support optional component replacement and/or caching:
    - No args: Standard forward pass of the target model
    - With mask_infos: Components replace the specified modules via forward hooks
    - With cache_type="input": Input activations are cached for the specified modules
    - With cache_type="component_acts": Component activations are cached for the specified modules
    - Both can be used simultaneously for component forward pass with input caching

    We register components and causal importance functions (ci_fns) as modules in this class in order to have them update
    correctly when the model is wrapped in a `DistributedDataParallel` wrapper (and for other
    conveniences).
    """

    def __init__(
        self,
        target_model: nn.Module,
        run_batch: RunBatch,
        module_path_info: list[ModulePathInfo],
        ci_config: CiConfig,
        sigmoid_type: SigmoidType,
    ):
        super().__init__()
        self._run_batch: RunBatch = run_batch

        for name, param in target_model.named_parameters():
            assert not param.requires_grad, (
                f"Target model should not have any trainable parameters. "
                f"Found {param.requires_grad} for {name}"
            )

        self.target_model = target_model
        self.module_to_c = {info.module_path: info.C for info in module_path_info}
        self.target_module_paths = list(self.module_to_c.keys())

        # Replace each decomposed submodule in target_model with a DecomposedSite in place.
        # After this call:
        #   - target_model.<path> is a DecomposedSite (DecomposedLinear or DecomposedEmbedding)
        #   - The site owns its own V, U parameters (trainable, requires_grad=True)
        #   - The site's `.linear` (or `.embedding`) is the frozen original module
        # `self.components` is the canonical dict mapping site_name -> site; it shares storage
        # with the sites that now live in target_model.
        self.components: dict[str, DecomposedSite] = install_decomposed_sites(
            target_model, self.module_to_c
        )

        match ci_config:
            case LayerwiseCiConfig():
                raw_layerwise_ci_fns = {
                    path: ComponentModel._create_layerwise_ci_fn(
                        site=self.components[path],
                        C=C,
                        ci_fn_type=ci_config.fn_type,
                        ci_fn_hidden_dims=ci_config.hidden_dims,
                    )
                    for path, C in self.module_to_c.items()
                }
                self.ci_fn = LayerwiseCiFnWrapper(
                    ci_fns=raw_layerwise_ci_fns,
                    sites=self.components,
                    ci_fn_type=ci_config.fn_type,
                )
            case GlobalCiConfig():
                raw_global_ci_fn = ComponentModel._create_global_ci_fn(
                    sites=self.components,
                    ci_config=ci_config,
                )
                self.ci_fn = GlobalCiFnWrapper(
                    global_ci_fn=raw_global_ci_fn,
                    sites=self.components,
                )

        if sigmoid_type == "leaky_hard":
            self.lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
            self.upper_leaky_fn = SIGMOID_TYPES["upper_leaky_hard"]
        else:
            # For other sigmoid types, use the same function for both
            self.lower_leaky_fn = SIGMOID_TYPES[sigmoid_type]
            self.upper_leaky_fn = SIGMOID_TYPES[sigmoid_type]

    def target_weight(self, module_name: str) -> Float[Tensor, "rows cols"]:
        """Target weight at the named decomposition site, shape (d_out, d_in)."""
        return self.components[module_name].target_weight

    @staticmethod
    def _site_input_dim(site: DecomposedSite, for_global_ci: bool) -> int:
        """Input dim feeding the CI fn for a given site.

        For DecomposedLinear, that's the raw activation dim (d_in). For
        DecomposedEmbedding, the CI fn sees component activations (V[idx]),
        which are C-dimensional — but only under global CI; under layerwise
        MLP CI we use the same convention via `get_component_acts`.
        """
        if isinstance(site, DecomposedEmbedding):
            return site.C if for_global_ci else site.d_embed
        return site.d_in

    @staticmethod
    def _create_layerwise_ci_fn(
        site: DecomposedSite,
        C: int,
        ci_fn_type: LayerwiseCiFnType,
        ci_fn_hidden_dims: list[int],
    ) -> nn.Module:
        """Helper to create a single layerwise CI function based on ci_fn_type and site type."""
        if isinstance(site, DecomposedEmbedding):
            assert ci_fn_type == "mlp", "Embedding modules only supported for ci_fn_type='mlp'"

        if ci_fn_type == "mlp":
            return MLPCiFn(C=C, hidden_dims=ci_fn_hidden_dims)

        input_dim = ComponentModel._site_input_dim(site, for_global_ci=False)

        match ci_fn_type:
            case "vector_mlp":
                return VectorMLPCiFn(C=C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims)
            case "shared_mlp":
                return VectorSharedMLPCiFn(C=C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims)

    @staticmethod
    def _create_global_ci_fn(
        sites: dict[str, DecomposedSite],
        ci_config: GlobalCiConfig,
    ) -> GlobalSharedMLPCiFn | GlobalSharedTransformerCiFn:
        """Create a global CI function that takes all layer activations as input."""
        ci_fn_type = ci_config.fn_type
        ci_fn_hidden_dims = ci_config.hidden_dims

        layer_configs: dict[str, tuple[int, int]] = {
            site_name: (
                ComponentModel._site_input_dim(site, for_global_ci=True),
                site.C,
            )
            for site_name, site in sites.items()
        }

        match ci_fn_type:
            case "global_shared_mlp":
                assert ci_fn_hidden_dims is not None  # validated by Pydantic
                return GlobalSharedMLPCiFn(
                    layer_configs=layer_configs, hidden_dims=ci_fn_hidden_dims
                )
            case "global_shared_transformer":
                transformer_cfg = ci_config.simple_transformer_ci_cfg
                assert transformer_cfg is not None  # validated by Pydantic

                return GlobalSharedTransformerCiFn(
                    target_model_layer_configs={
                        target_module_path: TargetLayerConfig(input_dim=input_dim, C=C)
                        for target_module_path, (input_dim, C) in layer_configs.items()
                    },
                    d_model=transformer_cfg.d_model,
                    n_layers=transformer_cfg.n_blocks,
                    n_heads=transformer_cfg.attn_config.n_heads,
                    mlp_hidden_dims=transformer_cfg.mlp_hidden_dim,
                    max_len=transformer_cfg.attn_config.max_len,
                    rope_base=transformer_cfg.attn_config.rope_base,
                    gradient_checkpointing=transformer_cfg.gradient_checkpointing,
                )

    @overload
    def __call__(
        self,
        batch: Any,
        cache_type: Literal["component_acts"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Any,
        cache_type: Literal["input"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Any,
        cache_type: Literal["output"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["none"] = "none",
    ) -> Tensor: ...

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | OutputWithCache:
        return super().__call__(*args, **kwargs)

    @override
    def forward(
        self,
        batch: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["component_acts", "input", "output", "none"] = "none",
    ) -> Tensor | OutputWithCache:
        """Forward pass with optional component replacement and/or input/output caching.

        Args:
            mask_infos: Dictionary mapping module names to ComponentsMaskInfo. If provided,
                those sites use the decomposed path; sites not in this dict (or if it's None)
                use the wrapped target's forward.
            cache_type: What to cache. "input" caches pre-weight activations,
                "output" caches post-weight activations, "component_acts" caches per-component
                activations, "none" disables caching.

        Returns:
            OutputWithCache object if cache_type is not "none", otherwise the model output tensor.
        """
        batch = move_batch_to_device(batch, next(self.parameters()).device)

        if mask_infos is None and cache_type == "none":
            return self._run_batch(self.target_model, batch)

        cache: dict[str, Tensor] = {}
        bound_cache: dict[str, Tensor] | None = cache if cache_type != "none" else None
        bound_cache_type = cache_type if cache_type != "none" else None

        # Bind per-site mask_info + cache slots, run forward, unbind on exit.
        with ExitStack() as stack:
            for site_name, site in self.components.items():
                mask_info = mask_infos.get(site_name) if mask_infos is not None else None
                stack.enter_context(site.bind(mask_info, bound_cache, bound_cache_type))
            out: Tensor = self._run_batch(self.target_model, batch)

        match cache_type:
            case "input" | "output" | "component_acts":
                return OutputWithCache(output=out, cache=cache)
            case "none":
                return out

    @classmethod
    @override
    def from_run_info(cls, run_info: RunInfo[Config]) -> "ComponentModel":
        """Load a trained ComponentModel checkpoint from a run info object."""
        config = run_info.config

        # Load the target model
        model_class = resolve_class(config.pretrained_model_class)
        if config.pretrained_model_name is not None:
            assert hasattr(model_class, "from_pretrained"), (
                f"Model class {model_class} should have a `from_pretrained` method"
            )
            # Handle param_decomp.pretrain models: patch missing model_type in old pretrain runs
            if config.pretrained_model_class.startswith("param_decomp.pretrain.models."):
                from param_decomp.pretrain.run_info import PretrainRunInfo

                pretrain_run_info = PretrainRunInfo.from_path(config.pretrained_model_name)
                if "model_type" not in pretrain_run_info.model_config_dict:
                    pretrain_run_info.model_config_dict["model_type"] = (
                        config.pretrained_model_class.split(".")[-1]
                    )
                target_model = model_class.from_run_info(pretrain_run_info)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                target_model = model_class.from_pretrained(config.pretrained_model_name)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            assert issubclass(model_class, LoadableModule), (
                f"Model class {model_class} should be a subclass of LoadableModule which "
                "defines a `from_pretrained` method"
            )
            assert run_info.config.pretrained_model_path is not None
            target_model = model_class.from_pretrained(run_info.config.pretrained_model_path)

        target_model.eval()
        target_model.requires_grad_(False)

        if config.identity_module_info is not None:
            insert_identity_operations_(
                target_model,
                identity_module_info=config.identity_module_info,
            )

        module_path_info = expand_module_patterns(target_model, config.all_module_info)

        comp_model = ComponentModel(
            target_model=target_model,
            run_batch=make_run_batch(config.output_extract),
            module_path_info=module_path_info,
            ci_config=config.ci_config,
            sigmoid_type=config.sigmoid_type,
        )

        comp_model_weights = torch.load(
            run_info.checkpoint_path, map_location="cpu", weights_only=True
        )

        handle_deprecated_state_dict_keys_(comp_model_weights)

        _validate_checkpoint_ci_config_compatibility(comp_model_weights, config.ci_config)

        comp_model.load_state_dict(comp_model_weights)
        return comp_model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "ComponentModel":
        """Load a trained ComponentModel checkpoint from a local or wandb path."""
        run_info = ParamDecompRunInfo.from_path(path)
        return cls.from_run_info(run_info)

    def calc_causal_importances(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        sampling: SamplingType,
        detach_inputs: bool = False,
    ) -> CIOutputs:
        """Calculate causal importances using the unified CI function interface.

        Args:
            pre_weight_acts: The activations before each layer in the target model.
            sampling: The sampling type for stochastic masks.
            detach_inputs: Whether to detach the inputs to the causal importance function.

        Returns:
            CIOutputs containing lower_leaky, upper_leaky, and pre_sigmoid CI values.
        """
        if detach_inputs:
            pre_weight_acts = {k: v.detach() for k, v in pre_weight_acts.items()}

        ci_fn_outputs = self.ci_fn(pre_weight_acts)
        return self._apply_sigmoid_to_ci_outputs(ci_fn_outputs, sampling)

    def _apply_sigmoid_to_ci_outputs(
        self,
        ci_fn_outputs: dict[str, Float[Tensor, "... C"]],
        sampling: SamplingType,
    ) -> CIOutputs:
        """Apply sigmoid functions to CI function outputs."""
        causal_importances_lower_leaky = {}
        causal_importances_upper_leaky = {}
        pre_sigmoid = {}

        for target_module_name, ci_fn_output in ci_fn_outputs.items():
            if sampling == "binomial":
                ci_fn_output_for_lower_leaky = 1.05 * ci_fn_output - 0.05 * torch.rand_like(
                    ci_fn_output
                )
            else:
                ci_fn_output_for_lower_leaky = ci_fn_output

            lower_leaky_output = self.lower_leaky_fn(ci_fn_output_for_lower_leaky)
            assert (lower_leaky_output <= 1.0).all()
            causal_importances_lower_leaky[target_module_name] = lower_leaky_output

            upper_leaky_output = self.upper_leaky_fn(ci_fn_output)
            assert (upper_leaky_output >= 0).all()
            causal_importances_upper_leaky[target_module_name] = upper_leaky_output

            pre_sigmoid[target_module_name] = ci_fn_output

        return CIOutputs(
            lower_leaky=causal_importances_lower_leaky,
            upper_leaky=causal_importances_upper_leaky,
            pre_sigmoid=pre_sigmoid,
        )

    def get_all_component_acts(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "..."]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        """Compute pre-mask component activations (x @ V or V[idx]) for all sites.

        Args:
            pre_weight_acts: Dict mapping site name to input activations (or token ids for
                embedding sites).

        Returns:
            Dict mapping site name to component activations tensor.
        """
        return {
            site_name: self.components[site_name].get_component_acts(acts)
            for site_name, acts in pre_weight_acts.items()
            if site_name in self.components
        }

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, "d_out d_in"]]:
        """Calculate `W_target - V@U` per site. Each site materializes its own delta
        (so under FSDP both V/U and the target weight are gathered at the site)."""
        return {site_name: site.calc_weight_delta() for site_name, site in self.components.items()}


def handle_deprecated_state_dict_keys_(state_dict: dict[str, Tensor]) -> None:
    """Maps deprecated state dict keys to new state dict keys"""
    for key in list(state_dict.keys()):
        new_key: str = key
        # We used to have "_gates.*", now we have "_ci_fns.*"
        if "_gates." in new_key:
            new_key = new_key.replace("_gates.", "_ci_fns.")
        # We used to have prefix "patched_model.*", now we have "target_model.*"
        if new_key.startswith("patched_model."):
            new_key = "target_model." + new_key.removeprefix("patched_model.")
        # We used to have "*.original.weight", now we have "*.weight"
        if new_key.endswith(".original.weight"):
            new_key = new_key.removesuffix(".original.weight") + ".weight"
        # We used to have "*.components.{U,V}", now we have "_components.*.{U,V}"
        if new_key.endswith(".components.U") or new_key.endswith(".components.V"):
            target_module_path: str = (
                new_key.removeprefix("target_model.")
                .removesuffix(".components.U")
                .removesuffix(".components.V")
            )
            # module path has "." replaced with "-"
            new_key = f"_components.{target_module_path.replace('.', '-')}.{new_key.split('.')[-1]}"
        # Old checkpoints had _ci_fns.* at top level, now under ci_fn._ci_fns.*
        if new_key.startswith("_ci_fns.") and not new_key.startswith("ci_fn."):
            new_key = "ci_fn." + new_key
        # replace if modified
        if new_key != key:
            state_dict[new_key] = state_dict.pop(key)
