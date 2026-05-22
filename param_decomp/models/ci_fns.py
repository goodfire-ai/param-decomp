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
from param_decomp.models.ci_nn_blocks import Linear, ParallelLinear, TransformerBlock
from param_decomp.models.components import Components, EmbeddingComponents
from param_decomp.types import GlobalCiFnType, LayerwiseCiFnType


class LayerwiseCiConfig(BaseConfig):
    """Configuration for layerwise CI functions (one per layer)."""

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
    """Configuration for self-attention.

    Uses RoPE (Rotary Position Embeddings) for sequence length generalization.
    """

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
    """Configuration for global CI function (single function for all layers)."""

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


CiConfig = LayerwiseCiConfig | GlobalCiConfig


class MLPCiFn(nn.Module):
    """MLP-based function that creates a scalar output for each component."""

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
    """Contains a separate network for each component and takes a module's input vector as input."""

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
    """Maps a module's input vector to a scalar output for each component with a 'pure' MLP."""

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
    """Global CI function that concatenates all layer activations and outputs CI for all layers."""

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
    input_dim: int
    C: int


class GlobalSharedTransformerCiFn(nn.Module):
    """Global CI function that projects concatenated activations and attends over sequence."""

    def __init__(
        self,
        target_model_layer_configs: dict[str, TargetLayerConfig],
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_hidden_dims: list[int] | None = None,
        max_len: int = 2048,
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
    """Wraps a dict of per-layer CI functions with a unified interface.

    Calls each layer's CI function independently on its corresponding input activations.
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
    """Wraps global CI functions with a unified interface.

    Transforms embedding layer inputs to component activations before calling
    the underlying global CI function.
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
