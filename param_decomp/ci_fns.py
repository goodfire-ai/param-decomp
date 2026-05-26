"""Causal-importance function configs, CI-fn modules, and wrappers."""

from dataclasses import dataclass
from typing import Literal, Self, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import Field, PositiveInt, model_validator
from torch import Tensor, nn

from param_decomp.base_config import BaseConfig
from param_decomp.ci_nn_blocks import Linear, ParallelLinear, TransformerBlock
from param_decomp.components import Components, EmbeddingComponents, get_module_input_dim

LayerwiseCiFnType = Literal["mlp", "vector_mlp", "shared_mlp"]
GlobalCiFnType = Literal["global_shared_mlp", "global_shared_transformer"]


class LayerwiseCiConfig(BaseConfig):
    """Layerwise CI fns — one independent CI fn per decomposition target."""

    mode: Literal["layerwise"] = "layerwise"
    fn_type: LayerwiseCiFnType = Field(
        ..., description="Type of layerwise CI function: mlp, vector_mlp, or shared_mlp"
    )
    hidden_dims: list[PositiveInt] = Field(
        ..., description="Hidden dimensions for the CI function MLP"
    )

    @model_validator(mode="after")
    def validate_hidden_dims(self) -> Self:
        if self.fn_type in ("mlp", "vector_mlp") and not self.hidden_dims:
            raise ValueError(f"hidden_dims must be non-empty for fn_type={self.fn_type!r}")
        return self


class AttnConfig(BaseConfig):
    """Self-attention config for the transformer CI fn. Uses RoPE for length generalization."""

    n_heads: PositiveInt = Field(
        ...,
        description="Number of attention heads. Must divide the input dimension.",
    )
    max_len: PositiveInt = Field(
        default=2048,
        description="Maximum sequence length for RoPE embeddings.",
    )
    rope_base: float = Field(
        default=10000.0,
        description="Base for RoPE frequency computation.",
    )


class GlobalSharedTransformerCiConfig(BaseConfig):
    """Config for the global transformer CI fn.

    `d_model` must be divisible by `attn_config.n_heads` and the resulting per-head dim
    must be even (RoPE). `mlp_hidden_dim` defaults to `[4 * d_model]`.
    """

    d_model: PositiveInt
    n_blocks: PositiveInt
    mlp_hidden_dim: list[PositiveInt] | None = Field(
        default=None,
        description="Hidden dimension for transformer MLP blocks. "
        "If None, defaults to [4 * d_model].",
    )
    attn_config: AttnConfig

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        assert self.d_model % self.attn_config.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"attn_config.n_heads ({self.attn_config.n_heads})"
        )
        d_head = self.d_model // self.attn_config.n_heads
        assert d_head % 2 == 0, (
            f"d_head ({d_head}) must be even for RoPE. "
            f"d_model={self.d_model}, "
            f"n_heads={self.attn_config.n_heads}"
        )
        return self


class GlobalCiConfig(BaseConfig):
    """A single global CI fn that maps all layers jointly."""

    mode: Literal["global"] = "global"
    fn_type: GlobalCiFnType = Field(
        ...,
        description="Type of global CI function: global_shared_mlp or global_shared_transformer",
    )
    hidden_dims: list[PositiveInt] | None = Field(
        default=None,
        description="Hidden dimensions for global_shared_mlp CI function.",
    )
    simple_transformer_ci_cfg: GlobalSharedTransformerCiConfig | None = None

    @model_validator(mode="after")
    def validate_ci_config(self) -> Self:
        if self.fn_type == "global_shared_mlp":
            assert self.hidden_dims is not None, (
                "hidden_dims must be specified when fn_type='global_shared_mlp'"
            )
        elif self.fn_type == "global_shared_transformer":
            assert self.simple_transformer_ci_cfg is not None, (
                "simple_transformer_ci_cfg must be specified when fn_type='global_shared_transformer'"
            )
            assert self.hidden_dims is None, (
                "hidden_dims is only used for fn_type='global_shared_mlp'"
            )
        return self


# Discriminated union (by `mode`) of every CI-fn config the trainer accepts. Pydantic
# picks the right branch from the YAML `pd.ci_config.mode` literal.
CiConfig = LayerwiseCiConfig | GlobalCiConfig


class MLPCiFn(nn.Module):
    """Per-component scalar-input MLP CI fn.

    Each of `C` components gets its own MLP mapping a scalar component activation to a
    scalar CI value; built from `ParallelLinear` layers operating on a singleton last dim.
    """

    def __init__(self, C: int, hidden_dims: list[int]):
        super().__init__()

        self.hidden_dims = hidden_dims

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            input_dim = 1 if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(ParallelLinear(C, input_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())
        self.layers.append(ParallelLinear(C, hidden_dims[-1], 1, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        x = einops.rearrange(x, "... C -> ... C 1")
        x = self.layers(x)
        assert x.shape[-1] == 1, "Last dimension should be 1 after the final layer"
        return x[..., 0]


class VectorMLPCiFn(nn.Module):
    """Per-component vector-input MLP CI fn.

    Each of `C` components gets its own MLP consuming the full `[..., d_in]` layer input;
    built from `ParallelLinear` so all `C` networks run in one batched einsum.
    """

    def __init__(self, C: int, input_dim: int, hidden_dims: list[int]):
        super().__init__()

        self.hidden_dims = hidden_dims

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(ParallelLinear(C, input_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())

        self.layers.append(ParallelLinear(C, hidden_dims[-1], 1, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        x = self.layers(einops.rearrange(x, "... d_in -> ... 1 d_in"))
        assert x.shape[-1] == 1, "Last dimension should be 1 after the final layer"
        return x[..., 0]


class VectorSharedMLPCiFn(nn.Module):
    """Shared MLP `[..., d_in] -> [..., C]`.

    All components share every hidden layer; only the final projection splits
    per-component.
    """

    def __init__(self, C: int, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(Linear(in_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())
        final_dim = hidden_dims[-1] if len(hidden_dims) > 0 else input_dim
        self.layers.append(Linear(final_dim, C, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        return self.layers(x)


class GlobalSharedMLPCiFn(nn.Module):
    """Global MLP over all layers.

    Concatenates all decomposition-target inputs along the feature dim, runs one shared
    MLP, then splits the output back into per-layer `[..., C]` slices. Layer order is
    fixed by sorted layer name so concatenation is deterministic.
    """

    def __init__(
        self,
        layer_configs: dict[str, tuple[int, int]],  # layer_name -> (input_dim, C)
        hidden_dims: list[int],
    ):
        super().__init__()

        self.layer_order = sorted(layer_configs.keys())
        self.layer_configs = layer_configs
        self.split_sizes = [layer_configs[name][1] for name in self.layer_order]

        total_input_dim = sum(input_dim for input_dim, _ in layer_configs.values())
        total_C = sum(C for _, C in layer_configs.values())

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            in_dim = total_input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(Linear(in_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())
        final_dim = hidden_dims[-1] if len(hidden_dims) > 0 else total_input_dim
        self.layers.append(Linear(final_dim, total_C, nonlinearity="linear"))

    @override
    def forward(
        self,
        input_acts: dict[str, Float[Tensor, "... d_in"]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        inputs_list = [input_acts[name] for name in self.layer_order]
        concatenated = torch.cat(inputs_list, dim=-1)
        output = self.layers(concatenated)
        split_outputs = torch.split(output, self.split_sizes, dim=-1)
        return {name: split_outputs[i] for i, name in enumerate(self.layer_order)}


@dataclass
class TargetLayerConfig:
    """Per-target metadata consumed by `GlobalSharedTransformerCiFn`."""

    input_dim: int
    C: int


class GlobalSharedTransformerCiFn(nn.Module):
    """Global transformer attending over sequence to produce per-component CI.

    Per-layer inputs are RMS-normed, concatenated along the feature dim, projected to
    `d_model`, and run through `n_layers` `TransformerBlock`s with bidirectional
    self-attention. A final linear projection produces logits which are split back into
    per-layer `[..., C]` slices in sorted-name order. For 2D inputs (e.g. TMS, resid_mlp
    — no sequence axis) a singleton sequence dim is added before the transformer and
    squeezed out after.
    """

    def __init__(
        self,
        target_model_layer_configs: dict[str, TargetLayerConfig],
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_len: int,
        mlp_hidden_dims: list[int] | None = None,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        self.layer_order = sorted(target_model_layer_configs.keys())
        self.target_model_layer_configs = target_model_layer_configs
        self.split_sizes = [target_model_layer_configs[name].C for name in self.layer_order]
        self.d_model = d_model
        self.n_transformer_layers = n_layers
        self.n_heads = n_heads

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [4 * d_model]

        total_input_dim = sum(config.input_dim for config in target_model_layer_configs.values())
        total_c = sum(config.C for config in target_model_layer_configs.values())

        self._input_projector = Linear(total_input_dim, d_model, nonlinearity="relu")
        self._output_head = Linear(d_model, total_c, nonlinearity="linear")

        self._blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_hidden_dims=mlp_hidden_dims,
                    max_len=max_len,
                    rope_base=rope_base,
                )
                for _ in range(n_layers)
            ]
        )

    @override
    def forward(
        self,
        input_acts: dict[str, Float[Tensor, "... d_in"]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        inputs_list = [
            F.rms_norm(input_acts[name], (input_acts[name].shape[-1],)) for name in self.layer_order
        ]
        concatenated = torch.cat(inputs_list, dim=-1)
        projected: Tensor = self._input_projector(concatenated)

        # The transformer blocks expect a sequence dimension, so we add an extra dimension to our
        # activations if we only have 2D acts (e.g. in TMS and resid_mlp).
        added_seq_dim = False
        if projected.ndim < 3:
            projected = projected.unsqueeze(-2)
            added_seq_dim = True

        x = projected
        for block in self._blocks:
            x = block(x)

        output = self._output_head(x)

        if added_seq_dim:
            output = output.squeeze(-2)

        split_outputs = torch.split(output, self.split_sizes, dim=-1)
        outputs = {name: split_outputs[i] for i, name in enumerate(self.layer_order)}

        return outputs


class LayerwiseCiFnWrapper(nn.Module):
    """Bundle a dict of per-layer CI fns behind a single dict-in/dict-out interface.

    Runs each layer's CI fn on its own input. For `ci_fn_type == "mlp"` the per-component
    scalar activations are obtained via `Components.get_component_acts` first; the other
    variants receive the raw layer input. Layer names are stored under `ModuleDict` with
    `.` replaced by `-` so state-dict keys are well-formed.
    """

    def __init__(
        self,
        ci_fns: dict[str, nn.Module],
        components: dict[str, Components],
        ci_fn_type: LayerwiseCiFnType,
    ):
        super().__init__()
        self.layer_names = sorted(ci_fns.keys())
        self.components = components
        self.ci_fn_type = ci_fn_type

        # Store as ModuleDict with "." replaced by "-" for state dict compatibility
        self._ci_fns = nn.ModuleDict(
            {name.replace(".", "-"): ci_fns[name] for name in self.layer_names}
        )

    @override
    def forward(
        self,
        layer_acts: dict[str, Float[Tensor, "..."]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        outputs: dict[str, Float[Tensor, "... C"]] = {}

        for layer_name in self.layer_names:
            ci_fn = self._ci_fns[layer_name.replace(".", "-")]
            input_acts = layer_acts[layer_name]

            # MLPCiFn expects component activations, others take raw input
            if self.ci_fn_type == "mlp":
                ci_fn_input = self.components[layer_name].get_component_acts(input_acts)
            else:
                ci_fn_input = input_acts

            outputs[layer_name] = ci_fn(ci_fn_input)

        return outputs


class GlobalCiFnWrapper(nn.Module):
    """Gives the global CI fns the same dict-in/dict-out interface as the layerwise wrapper.

    For `EmbeddingComponents` the raw input is a tensor of token ids; this wrapper
    converts them to component activations via `EmbeddingComponents.get_component_acts`
    so the global CI fn always sees floating-point activations.
    """

    def __init__(
        self,
        global_ci_fn: GlobalSharedMLPCiFn | GlobalSharedTransformerCiFn,
        components: dict[str, Components],
    ):
        super().__init__()
        self._global_ci_fn = global_ci_fn
        self.components = components

    @override
    def forward(
        self,
        layer_acts: dict[str, Float[Tensor, "..."]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        transformed: dict[str, Float[Tensor, ...]] = {}

        for layer_name, acts in layer_acts.items():
            component = self.components[layer_name]
            if isinstance(component, EmbeddingComponents):
                # Embeddings pass token IDs; convert to component activations
                transformed[layer_name] = component.get_component_acts(acts)
            else:
                transformed[layer_name] = acts

        return self._global_ci_fn(transformed)


def _make_layerwise_ci_fn(
    target_module: nn.Module,
    C: int,
    ci_fn_type: LayerwiseCiFnType,
    ci_fn_hidden_dims: list[int],
) -> nn.Module:
    if isinstance(target_module, nn.Embedding):
        assert ci_fn_type == "mlp", "Embedding modules only supported for ci_fn_type='mlp'"

    if ci_fn_type == "mlp":
        return MLPCiFn(C=C, hidden_dims=ci_fn_hidden_dims)

    input_dim = get_module_input_dim(target_module)
    match ci_fn_type:
        case "vector_mlp":
            return VectorMLPCiFn(C=C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims)
        case "shared_mlp":
            return VectorSharedMLPCiFn(C=C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims)


def _make_global_ci_fn(
    target_model: nn.Module,
    module_to_c: dict[str, int],
    components: dict[str, Components],
    ci_config: GlobalCiConfig,
) -> GlobalSharedMLPCiFn | GlobalSharedTransformerCiFn:
    ci_fn_type = ci_config.fn_type
    ci_fn_hidden_dims = ci_config.hidden_dims

    layer_configs: dict[str, tuple[int, int]] = {}
    for path, module_c in module_to_c.items():
        target_module = target_model.get_submodule(path)
        component = components[path]
        if isinstance(target_module, nn.Embedding):
            assert isinstance(component, EmbeddingComponents)
            input_dim = component.C
        else:
            input_dim = get_module_input_dim(target_module)
        layer_configs[path] = (input_dim, module_c)

    match ci_fn_type:
        case "global_shared_mlp":
            assert ci_fn_hidden_dims is not None
            return GlobalSharedMLPCiFn(layer_configs=layer_configs, hidden_dims=ci_fn_hidden_dims)
        case "global_shared_transformer":
            transformer_cfg = ci_config.simple_transformer_ci_cfg
            assert transformer_cfg is not None
            return GlobalSharedTransformerCiFn(
                target_model_layer_configs={
                    path: TargetLayerConfig(input_dim=input_dim, C=C)
                    for path, (input_dim, C) in layer_configs.items()
                },
                d_model=transformer_cfg.d_model,
                n_layers=transformer_cfg.n_blocks,
                n_heads=transformer_cfg.attn_config.n_heads,
                mlp_hidden_dims=transformer_cfg.mlp_hidden_dim,
                max_len=transformer_cfg.attn_config.max_len,
                rope_base=transformer_cfg.attn_config.rope_base,
            )


def make_ci_fn_wrapper(
    target_model: nn.Module,
    module_to_c: dict[str, int],
    components: dict[str, Components],
    ci_config: CiConfig,
) -> LayerwiseCiFnWrapper | GlobalCiFnWrapper:
    """Build the CI-fn wrapper selected by `ci_config`.

    `LayerwiseCiConfig` → one inner CI fn per `module_to_c` entry inside a
    `LayerwiseCiFnWrapper`; `GlobalCiConfig` → a single global CI fn inside a
    `GlobalCiFnWrapper`.

    Args:
        target_model: Frozen target model; used to look up each decomposition target's
            input dimensionality.
        module_to_c: Map from decomposition-target submodule path to component count.
        components: Map from decomposition-target submodule path to its `Components`
            instance (used by `MLPCiFn` and embedding-target dispatch).
        ci_config: Discriminated CI-fn config; runtime type selects the wrapper.
    """
    match ci_config:
        case LayerwiseCiConfig():
            raw_ci_fns = {
                path: _make_layerwise_ci_fn(
                    target_module=target_model.get_submodule(path),
                    C=C,
                    ci_fn_type=ci_config.fn_type,
                    ci_fn_hidden_dims=ci_config.hidden_dims,
                )
                for path, C in module_to_c.items()
            }
            return LayerwiseCiFnWrapper(
                ci_fns=raw_ci_fns,
                components=components,
                ci_fn_type=ci_config.fn_type,
            )
        case GlobalCiConfig():
            raw_global = _make_global_ci_fn(
                target_model=target_model,
                module_to_c=module_to_c,
                components=components,
                ci_config=ci_config,
            )
            return GlobalCiFnWrapper(global_ci_fn=raw_global, components=components)
