import fnmatch
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, NamedTuple, overload, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from spd.configs import CiConfig, Config, GlobalCiConfig, LayerwiseCiConfig, SamplingType
from spd.identity_insertion import insert_identity_operations_
from spd.interfaces import LoadableModule, RunInfo
from spd.models.components import (
    Components,
    ComponentsMaskInfo,
    EmbeddingComponents,
    GlobalCiFnWrapper,
    GlobalReverseResidualCiFn,
    GlobalSharedMLPCiFn,
    GlobalSharedTransformerCiFn,
    Identity,
    LayerwiseCiFnWrapper,
    LinearComponents,
    MLPCiFn,
    TargetLayerConfig,
    VectorMLPCiFn,
    VectorSharedMLPCiFn,
)
from spd.models.sigmoids import SIGMOID_TYPES, SigmoidType
from spd.spd_types import LayerwiseCiFnType, ModelPath
from spd.utils.general_utils import resolve_class
from spd.utils.module_utils import ModulePathInfo, expand_module_patterns


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
class SPDRunInfo(RunInfo[Config]):
    """Run info from training a ComponentModel (i.e. from an SPD run)."""

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
    """Wrapper around an arbitrary pytorch model for running SPD.

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
        module_path_info: list[ModulePathInfo],
        ci_config: CiConfig,
        sigmoid_type: SigmoidType,
        pretrained_model_output_attr: str | None,
    ):
        super().__init__()

        for name, param in target_model.named_parameters():
            assert not param.requires_grad, (
                f"Target model should not have any trainable parameters. "
                f"Found {param.requires_grad} for {name}"
            )

        self.target_model = target_model
        self.pretrained_model_output_attr = pretrained_model_output_attr
        self.module_to_c = {info.module_path: info.C for info in module_path_info}
        self.target_module_paths = list(self.module_to_c.keys())

        self.components = ComponentModel._create_components(
            target_model=target_model,
            module_to_c=self.module_to_c,
        )
        self._components = nn.ModuleDict(
            {k.replace(".", "-"): self.components[k] for k in sorted(self.components)}
        )

        match ci_config:
            case LayerwiseCiConfig():
                raw_layerwise_ci_fns = {
                    path: ComponentModel._create_layerwise_ci_fn(
                        target_module=target_model.get_submodule(path),
                        C=C,
                        ci_fn_type=ci_config.fn_type,
                        ci_fn_hidden_dims=ci_config.hidden_dims,
                    )
                    for path, C in self.module_to_c.items()
                }
                self.ci_fn = LayerwiseCiFnWrapper(
                    ci_fns=raw_layerwise_ci_fns,
                    components=self.components,
                    ci_fn_type=ci_config.fn_type,
                )
            case GlobalCiConfig():
                raw_global_ci_fn = ComponentModel._create_global_ci_fn(
                    target_model=target_model,
                    module_to_c=self.module_to_c,
                    components=self.components,
                    ci_config=ci_config,
                )
                self.ci_fn = GlobalCiFnWrapper(
                    global_ci_fn=raw_global_ci_fn,
                    components=self.components,
                )

        if sigmoid_type == "leaky_hard":
            self.lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
            self.upper_leaky_fn = SIGMOID_TYPES["upper_leaky_hard"]
        else:
            # For other sigmoid types, use the same function for both
            self.lower_leaky_fn = SIGMOID_TYPES[sigmoid_type]
            self.upper_leaky_fn = SIGMOID_TYPES[sigmoid_type]

    def target_weight(self, module_name: str) -> Float[Tensor, "rows cols"]:
        target_module = self.target_model.get_submodule(module_name)

        match target_module:
            case RadfordConv1D():
                return target_module.weight.T
            case nn.Linear() | nn.Embedding():
                return target_module.weight
            case Identity():
                p = next(self.parameters())
                return torch.eye(target_module.d, device=p.device, dtype=p.dtype)
            case _:
                raise ValueError(f"Module {target_module} not supported")

    @staticmethod
    def _create_component(
        target_module: nn.Module,
        C: int,
    ) -> Components:
        match target_module:
            case nn.Linear():
                d_out, d_in = target_module.weight.shape
                component = LinearComponents(
                    C=C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=target_module.bias.data if target_module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            case RadfordConv1D():
                d_in, d_out = target_module.weight.shape
                component = LinearComponents(
                    C=C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=target_module.bias.data if target_module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            case Identity():
                component = LinearComponents(
                    C=C,
                    d_in=target_module.d,
                    d_out=target_module.d,
                    bias=None,
                )
            case nn.Embedding():
                component = EmbeddingComponents(
                    C=C,
                    vocab_size=target_module.num_embeddings,
                    embedding_dim=target_module.embedding_dim,
                )
            case _:
                raise ValueError(f"Module {target_module} not supported")

        return component

    @staticmethod
    def _create_components(
        target_model: nn.Module,
        module_to_c: dict[str, int],
    ) -> dict[str, Components]:
        components: dict[str, Components] = {}
        for target_module_path, target_module_c in module_to_c.items():
            target_module = target_model.get_submodule(target_module_path)
            components[target_module_path] = ComponentModel._create_component(
                target_module, target_module_c
            )
        return components

    @staticmethod
    def _get_module_input_dim(target_module: nn.Module) -> int:
        """Extract input dimension from a Linear-like module.

        For embedding layers, this should not be called - handle them separately.
        """
        match target_module:
            case nn.Linear():
                return target_module.weight.shape[1]
            case RadfordConv1D():
                return target_module.weight.shape[0]
            case Identity():
                return target_module.d
            case _:
                raise ValueError(
                    f"Module {type(target_module)} not supported. "
                    "Embedding modules should be handled separately."
                )

    @staticmethod
    def _create_layerwise_ci_fn(
        target_module: nn.Module,
        C: int,
        ci_fn_type: LayerwiseCiFnType,
        ci_fn_hidden_dims: list[int],
    ) -> nn.Module:
        """Helper to create a single layerwise CI function based on ci_fn_type and module type."""
        if isinstance(target_module, nn.Embedding):
            assert ci_fn_type == "mlp", "Embedding modules only supported for ci_fn_type='mlp'"

        if ci_fn_type == "mlp":
            return MLPCiFn(C=C, hidden_dims=ci_fn_hidden_dims)

        input_dim = ComponentModel._get_module_input_dim(target_module)

        match ci_fn_type:
            case "vector_mlp":
                return VectorMLPCiFn(C=C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims)
            case "shared_mlp":
                return VectorSharedMLPCiFn(C=C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims)

    @staticmethod
    def _create_global_ci_fn(
        target_model: nn.Module,
        module_to_c: dict[str, int],
        components: dict[str, Components],
        ci_config: GlobalCiConfig,
    ) -> GlobalSharedMLPCiFn | GlobalSharedTransformerCiFn | GlobalReverseResidualCiFn:
        """Create a global CI function that takes all layer activations as input."""
        ci_fn_type = ci_config.fn_type
        ci_fn_hidden_dims = ci_config.hidden_dims

        # Build layer_configs: layer_name -> (input_dim, C)
        layer_configs: dict[str, tuple[int, int]] = {}
        for target_module_path, target_module_c in module_to_c.items():
            target_module = target_model.get_submodule(target_module_path)
            component = components[target_module_path]

            # For embeddings, global CI uses component acts (C dimensions)
            # For linear-like modules, use the actual input dimension
            if isinstance(target_module, nn.Embedding):
                assert isinstance(component, EmbeddingComponents)
                input_dim = component.C
            else:
                input_dim = ComponentModel._get_module_input_dim(target_module)

            layer_configs[target_module_path] = (input_dim, target_module_c)

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
                )
            case "global_reverse_residual":
                # block_groups, d_resid_ci_fn, reader_hidden_dims, transition_hidden_dim
                # are validated by Pydantic
                block_groups = ci_config.block_groups
                d_resid_ci_fn = ci_config.d_resid_ci_fn
                reader_hidden_dims = ci_config.reader_hidden_dims
                transition_hidden_dim = ci_config.transition_hidden_dim
                assert block_groups is not None  # for type narrowing
                assert d_resid_ci_fn is not None  # for type narrowing
                assert reader_hidden_dims is not None  # for type narrowing

                # Build block_configs from block_groups
                block_configs: list[tuple[str, list[str], list[int], list[int]]] = []
                all_matched_modules: set[str] = set()

                for block_group in block_groups:
                    matched_modules: list[str] = []
                    for pattern in block_group.patterns:
                        matches = [name for name in module_to_c if fnmatch.fnmatch(name, pattern)]
                        assert matches, (
                            f"Block pattern '{pattern}' in block '{block_group.name}' "
                            f"matched no modules. Available: {list(module_to_c.keys())}"
                        )
                        for match in matches:
                            assert match not in matched_modules, (
                                f"Module '{match}' matched multiple patterns in block "
                                f"'{block_group.name}'"
                            )
                        matched_modules.extend(matches)

                    for module in matched_modules:
                        assert module not in all_matched_modules, (
                            f"Module '{module}' matched multiple block groups"
                        )
                        all_matched_modules.add(module)

                    input_dims = [layer_configs[m][0] for m in matched_modules]
                    c_values = [layer_configs[m][1] for m in matched_modules]

                    block_configs.append((block_group.name, matched_modules, input_dims, c_values))

                assert all_matched_modules == set(module_to_c.keys()), (
                    f"Some modules not in any block group. "
                    f"Missing: {set(module_to_c.keys()) - all_matched_modules}"
                )

                return GlobalReverseResidualCiFn(
                    block_configs=block_configs,
                    d_resid_ci_fn=d_resid_ci_fn,
                    reader_hidden_dims=reader_hidden_dims,
                    transition_hidden_dim=transition_hidden_dim,
                    attn_config=ci_config.transition_attn_config,
                )

    def _extract_output(self, raw_output: Any) -> Tensor:
        """Extract the desired output from the model's raw output.

        If pretrained_model_output_attr is None, returns the raw output directly.
        If pretrained_model_output_attr starts with "idx_", returns the index specified by the
        second part of the string. E.g. "idx_0" returns the first element of the raw output.
        Otherwise, returns the specified attribute from the raw output.

        Args:
            raw_output: The raw output from the model.

        Returns:
            The extracted output.
        """
        if self.pretrained_model_output_attr is None:
            out = raw_output
        elif self.pretrained_model_output_attr.startswith("idx_"):
            idx_val = int(self.pretrained_model_output_attr.split("_")[1])
            assert isinstance(raw_output, Sequence), (
                f"raw_output must be a sequence, not {type(raw_output)}"
            )
            assert idx_val < len(raw_output), (
                f"Index {idx_val} out of range for raw_output of length {len(raw_output)}"
            )
            out = raw_output[idx_val]
        else:
            out = getattr(raw_output, self.pretrained_model_output_attr)

        assert isinstance(out, Tensor), f"Expected tensor output, got {type(out)}"
        return out

    @overload
    def __call__(
        self,
        *args: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["component_acts", "input", "output"],
        **kwargs: Any,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        *args: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["none"] = "none",
        **kwargs: Any,
    ) -> Tensor: ...

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | OutputWithCache:
        return super().__call__(*args, **kwargs)

    @override
    def forward(
        self,
        *args: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["component_acts", "input", "output", "none"] = "none",
        **kwargs: Any,
    ) -> Tensor | OutputWithCache:
        """Forward pass with optional component replacement and/or input/output caching.

        Args:
            mask_infos: Dictionary mapping module names to ComponentsMaskInfo.
                If provided, those modules will be replaced with their components.
            cache_type: What to cache for each hooked module. "input" caches pre-weight
                activations, "output" caches post-weight activations, "component_acts" caches
                per-component activations, "none" disables caching.

        Returns:
            OutputWithCache object if cache_type is not "none", otherwise the model output tensor.
        """
        if mask_infos is None and cache_type == "none":
            # No hooks needed. Do a regular forward pass of the target model.
            return self._extract_output(self.target_model(*args, **kwargs))

        cache: dict[str, Tensor] = {}
        hooks: dict[str, Callable[..., Any]] = {}

        hook_module_names = list(mask_infos.keys()) if mask_infos else self.target_module_paths

        for module_name in hook_module_names:
            mask_info = mask_infos[module_name] if mask_infos else None
            components = self.components[module_name] if mask_info else None

            hooks[module_name] = partial(
                self._components_and_cache_hook,
                module_name=module_name,
                components=components,
                mask_info=mask_info,
                cache_type=cache_type,
                cache=cache,
            )

        with self._attach_forward_hooks(hooks):
            raw_out = self.target_model(*args, **kwargs)

        out = self._extract_output(raw_out)
        match cache_type:
            case "input" | "output" | "component_acts":
                return OutputWithCache(output=out, cache=cache)
            case "none":
                return out

    def _components_and_cache_hook(
        self,
        _module: nn.Module,
        args: list[Any],
        kwargs: dict[Any, Any],
        output: Any,
        module_name: str,
        components: Components | None,
        mask_info: ComponentsMaskInfo | None,
        cache_type: Literal["component_acts", "input", "output", "none"],
        cache: dict[str, Tensor],
    ) -> Any | None:
        """Unified hook function that handles both component replacement and caching.

        Args:
            module: The module being hooked
            args: Module forward args
            kwargs: Module forward kwargs
            output: Module forward output
            module_name: Name of the module in the target model
            components: Component replacement (if using components)
            mask_info: Mask information (if using components)
            cache_type: Whether to cache the component acts, input, or none
            cache: Cache dictionary to populate (if cache_type is not None)

        Returns:
            If using components: modified output (or None to keep original)
            If not using components: None (keeps original output)
        """
        assert len(args) == 1, "Expected 1 argument"
        assert len(kwargs) == 0, "Expected no keyword arguments"
        x = args[0]
        assert isinstance(x, Tensor), "Expected input tensor"

        if cache_type == "input":
            cache[module_name] = x

        if components is not None and mask_info is not None:
            assert isinstance(output, Tensor), (
                f"Only supports single-tensor outputs, got {type(output)}"
            )

            component_acts_cache = {} if cache_type == "component_acts" else None
            components_out = components(
                x,
                mask=mask_info.component_mask,
                weight_delta_and_mask=mask_info.weight_delta_and_mask,
                component_acts_cache=component_acts_cache,
            )
            if component_acts_cache is not None:
                for k, v in component_acts_cache.items():
                    cache[f"{module_name}_{k}"] = v

            final_out = (
                components_out
                if mask_info.routing_mask == "all"
                else torch.where(mask_info.routing_mask[..., None], components_out, output)
            )

            if cache_type == "output":
                cache[module_name] = final_out
            return final_out

        # No component replacement - keep original output
        if cache_type == "output":
            assert isinstance(output, Tensor)
            cache[module_name] = output
        return None

    @contextmanager
    def _attach_forward_hooks(self, hooks: dict[str, Callable[..., Any]]) -> Generator[None]:
        """Context manager to temporarily attach forward hooks to the target model."""
        handles: list[RemovableHandle] = []
        for module_name, hook in hooks.items():
            target_module = self.target_model.get_submodule(module_name)
            handle = target_module.register_forward_hook(hook, with_kwargs=True)
            handles.append(handle)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

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
            # Handle spd.pretrain models: patch missing model_type in old pretrain runs
            if config.pretrained_model_class.startswith("spd.pretrain.models."):
                from spd.pretrain.run_info import PretrainRunInfo

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
            module_path_info=module_path_info,
            ci_config=config.ci_config,
            sigmoid_type=config.sigmoid_type,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
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
        run_info = SPDRunInfo.from_path(path)
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
        """Compute component activations (v_i^T @ x) for all layers.

        Args:
            pre_weight_acts: Dict mapping layer name to input activations.

        Returns:
            Dict mapping layer name to component activations tensor.
        """
        return {
            layer: self.components[layer].get_component_acts(acts)
            for layer, acts in pre_weight_acts.items()
            if layer in self.components
        }

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, "d_out d_in"]]:
        """Calculate the weight differences between the target and component weights (V@U) for each layer."""
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] = {}
        for comp_name, components in self.components.items():
            weight_deltas[comp_name] = self.target_weight(comp_name) - components.weight
        return weight_deltas


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
