from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, overload, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from param_decomp.configs import (
    CiConfig,
    GlobalCiConfig,
    LayerwiseCiConfig,
    PDConfig,
    SamplingType,
)
from param_decomp.experiment_manifest import (
    EXPERIMENT_MANIFEST_FILENAME,
    ExperimentConfig,
    ExperimentManifest,
    parse_manifest_experiment_config,
)
from param_decomp.experiments.driver import ExperimentDriver, load_driver
from param_decomp.identity_insertion import insert_identity_operations_
from param_decomp.interfaces import LoadableModule
from param_decomp.models.batch_and_loss_fns import PDTarget, RunBatch
from param_decomp.models.components import (
    Components,
    ComponentsMaskInfo,
    EmbeddingComponents,
    GlobalCiFnWrapper,
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
from param_decomp.models.sigmoids import SIGMOID_TYPES, SigmoidType
from param_decomp.param_decomp_types import LayerwiseCiFnType, ModelPath
from param_decomp.utils.distributed_utils import DistributedState
from param_decomp.utils.module_utils import ModulePathInfo, expand_module_patterns
from param_decomp.utils.run_files import resolve_config_path, resolve_run_files


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
class PDRunInfo:
    """Run info from training a ComponentModel (i.e. from a PD run)."""

    checkpoint_path: Path
    manifest: ExperimentManifest

    @classmethod
    def from_path(cls, path: ModelPath) -> "PDRunInfo":
        files = resolve_run_files(
            path,
            config_filename=EXPERIMENT_MANIFEST_FILENAME,
            checkpoint_prefix="model",
            extras_from_config_path=lambda p: ExperimentManifest.from_file(p).artifact_filenames,
        )
        return cls(
            checkpoint_path=files.checkpoint_path,
            manifest=ExperimentManifest.from_file(files.config_path),
        )

    @classmethod
    def config_from_path(cls, path: ModelPath) -> ExperimentManifest:
        """Load just the manifest, without resolving or downloading checkpoints."""
        return ExperimentManifest.from_file(
            resolve_config_path(path, config_filename=EXPERIMENT_MANIFEST_FILENAME)
        )

    @cached_property
    def experiment_config(self) -> ExperimentConfig:
        return parse_manifest_experiment_config(self.manifest)

    @cached_property
    def driver(self) -> ExperimentDriver[Any] | None:
        if self.manifest.driver is None:
            return None
        return load_driver(self.manifest.driver)

    @property
    def pd_config(self) -> PDConfig:
        return self.experiment_config.pd

    def load_target(self) -> PDTarget:
        assert self.driver is not None, (
            "This run manifest has no driver. Use load_pd(path, target=...) with an explicit "
            "PDTarget."
        )
        return self.driver.load_target(self.experiment_config, run_dir=self.checkpoint_path.parent)

    def build_dataloaders(
        self,
        *,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> tuple[DataLoader[Any], DataLoader[Any]]:
        assert self.driver is not None, (
            "This run manifest has no driver. Build dataloaders explicitly for custom runs."
        )
        return self.driver.build_dataloaders(
            self.experiment_config,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            dist_state=dist_state,
            device=device,
            run_dir=self.checkpoint_path.parent,
        )


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
    ) -> GlobalSharedMLPCiFn | GlobalSharedTransformerCiFn:
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
            mask_infos: Dictionary mapping module names to ComponentsMaskInfo.
                If provided, those modules will be replaced with their components.
            cache_type: What to cache for each hooked module. "input" caches pre-weight
                activations, "output" caches post-weight activations, "component_acts" caches
                per-component activations, "none" disables caching.

        Returns:
            OutputWithCache object if cache_type is not "none", otherwise the model output tensor.
        """
        if mask_infos is None and cache_type == "none":
            return self._run_batch(self.target_model, batch)

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
            out: Tensor = self._run_batch(self.target_model, batch)

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
    def from_run_info(cls, run_info: PDRunInfo) -> "ComponentModel":
        """Load a `ComponentModel` from saved run info via the experiment-config dispatcher.

        Convenience wrapper around `from_checkpoint` for callers that already have a
        `PDRunInfo`. New code should prefer `load_pd(path, target=...)` with an
        explicit `PDTarget`.
        """
        target = run_info.load_target()
        return cls.from_checkpoint(
            config=run_info.pd_config,
            checkpoint_path=run_info.checkpoint_path,
            target_model=target.model,
            run_batch=target.run_batch,
            tied_weights=target.tied_weights,
        )

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "ComponentModel":
        """Load a `ComponentModel` from a local or wandb path via metadata dispatch."""
        run_info = PDRunInfo.from_path(path)
        return cls.from_run_info(run_info)

    @classmethod
    def from_checkpoint(
        cls,
        config: PDConfig,
        checkpoint_path: Path,
        target_model: nn.Module,
        run_batch: RunBatch,
        tied_weights: list[tuple[str, str]] | None = None,
    ) -> "ComponentModel":
        """Rebuild a ComponentModel from a saved PD checkpoint and a user-supplied target.

        The caller owns target loading (HF, in-repo pretrain runs, custom user models),
        so this method takes the already-instantiated target plus its run_batch function.
        """
        target_model.eval()
        target_model.requires_grad_(False)

        if config.identity_module_info is not None:
            insert_identity_operations_(
                target_model,
                identity_module_info=config.identity_module_info,
            )

        module_path_info = expand_module_patterns(target_model, config.all_module_info)

        comp_model = cls(
            target_model=target_model,
            run_batch=run_batch,
            module_path_info=module_path_info,
            ci_config=config.ci_config,
            sigmoid_type=config.sigmoid_type,
        )

        comp_model_weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        handle_deprecated_state_dict_keys_(comp_model_weights)
        _validate_checkpoint_ci_config_compatibility(comp_model_weights, config.ci_config)
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
