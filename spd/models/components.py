from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from spd.utils.module_utils import _NonlinearityType, init_param_

if TYPE_CHECKING:
    from spd.configs import AttnConfig
    from spd.spd_types import LayerwiseCiFnType


class ParallelLinear(nn.Module):
    """C parallel linear layers"""

    def __init__(self, C: int, input_dim: int, output_dim: int, nonlinearity: _NonlinearityType):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.empty(C, input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(C, output_dim))
        init_param_(self.W, fan_val=input_dim, nonlinearity=nonlinearity)

    @override
    def forward(self, x: Float[Tensor, "... C d_in"]) -> Float[Tensor, "... C d_out"]:
        return einops.einsum(x, self.W, "... C d_in, C d_in d_out -> ... C d_out") + self.b


class Linear(nn.Module):
    """Linear layer with biases initialized to 0 and weights initialized using fan_val."""

    def __init__(self, input_dim: int, output_dim: int, nonlinearity: _NonlinearityType):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.empty(input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))
        init_param_(self.W, fan_val=input_dim, nonlinearity=nonlinearity)

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einops.einsum(x, self.W, "... d_in, d_in d_out -> ... d_out") + self.b


class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for sequence modeling.

    Computes position-dependent rotations for query and key tensors to encode
    relative position information. Supports arbitrary sequence lengths up to max_len.
    """

    def __init__(self, d_head: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, f"RoPE requires even d_head, got {d_head}"
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len
        self.d_head = d_head

    @override
    def forward(
        self,
        q: Float[Tensor, "... n_heads seq d_head"],
        k: Float[Tensor, "... n_heads seq d_head"],
    ) -> tuple[Float[Tensor, "... n_heads seq d_head"], Float[Tensor, "... n_heads seq d_head"]]:
        """Apply rotary embeddings to Q and K tensors."""
        seq_len = q.shape[-2]
        assert seq_len <= self.max_len, f"seq_len {seq_len} exceeds max_len {self.max_len}"

        assert isinstance(self.inv_freq, Tensor)
        positions = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        angles = einops.einsum(positions, self.inv_freq, "seq, d -> seq d")
        # Create full rotation: [cos, cos] and [sin, sin] interleaved
        cos_emb = torch.cat([angles.cos(), angles.cos()], dim=-1)
        sin_emb = torch.cat([angles.sin(), angles.sin()], dim=-1)

        q_rot = self._apply_rotation(q, cos_emb, sin_emb)
        k_rot = self._apply_rotation(k, cos_emb, sin_emb)
        return q_rot, k_rot

    def _apply_rotation(
        self,
        x: Float[Tensor, "... n_heads seq d_head"],
        cos: Float[Tensor, "seq d_head"],
        sin: Float[Tensor, "seq d_head"],
    ) -> Float[Tensor, "... n_heads seq d_head"]:
        """Apply rotation: x' = x * cos + rotate_half(x) * sin."""
        # Split into first half and second half
        x1 = x[..., : self.d_head // 2]
        x2 = x[..., self.d_head // 2 :]
        # Rotate: [-x2, x1]
        x_rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + x_rotated * sin


class SelfAttention(nn.Module):
    """Multi-head bidirectional self-attention with RoPE positional embeddings."""

    def __init__(self, d_model: int, n_heads: int, max_len: int = 2048, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPEEmbedding(self.d_head, max_len, rope_base)

    @override
    def forward(self, x: Float[Tensor, "... seq d_model"]) -> Float[Tensor, "... seq d_model"]:
        """Apply bidirectional self-attention with RoPE."""
        *batch_dims, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head: (..., seq, d_model) -> (..., n_heads, seq, d_head)
        q = q.view(*batch_dims, seq_len, self.n_heads, self.d_head).transpose(-3, -2)
        k = k.view(*batch_dims, seq_len, self.n_heads, self.d_head).transpose(-3, -2)
        v = v.view(*batch_dims, seq_len, self.n_heads, self.d_head).transpose(-3, -2)

        q, k = self.rope(q, k)

        # Bidirectional attention (no causal mask)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )

        # Reshape back: (..., n_heads, seq, d_head) -> (..., seq, d_model)
        attn_out = attn_out.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, self.d_model)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """RMSNorm → self-attention → residual → RMSNorm → MLP → residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_hidden_dims: list[int],
        max_len: int = 2048,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.attn = SelfAttention(
            d_model=d_model, n_heads=n_heads, max_len=max_len, rope_base=rope_base
        )
        self.d_model = d_model

        mlp_layers = nn.Sequential()
        in_dim = d_model
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(Linear(in_dim, hidden_dim, nonlinearity="relu"))
            mlp_layers.append(nn.GELU())
            in_dim = hidden_dim
        mlp_layers.append(Linear(in_dim, d_model, nonlinearity="linear"))
        self.mlp = mlp_layers

    @override
    def forward(self, x: Float[Tensor, "... seq d_model"]) -> Float[Tensor, "... seq d_model"]:
        x = x + self.attn(F.rms_norm(x, (self.d_model,)))
        x = x + self.mlp(F.rms_norm(x, (self.d_model,)))
        return x


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
        # this 1 will broadcast out to actual C size, but no need to expand out yet
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


class GlobalReverseResidualCiFn(nn.Module):
    """Global CI function that processes blocks in reverse order with a residual stream.

    Architecture:
    1. Initialize residual stream to zeros (batch..., d_resid_ci_fn)
    2. Process blocks in order (typically: unembed → layer N MLP → layer N attn → ... → embed)
    3. For each block:
       - Concat relevant activations from all modules in the block
       - Project to d_resid_ci_fn and add to residual stream
       - RMSNorm → Reader MLP outputs CI values for all modules in block
       - Transition updates residual stream for next block (except after last block):
         - If attn_config is provided: RMSNorm → attn → add → RMSNorm → MLP(GeLU) → add
         - Otherwise: RMSNorm → MLP(GeLU) → add
         - If transition_hidden_dim is provided: MLP d_resid → hidden → d_resid with GeLU
         - Otherwise: linear d_resid → d_resid

    """

    def __init__(
        self,
        block_configs: list[tuple[str, list[str], list[int], list[int]]],
        d_resid_ci_fn: int,
        reader_hidden_dims: list[int],
        transition_hidden_dim: int | None,
        attn_config: "AttnConfig | None" = None,
    ):
        """Initialize the reverse residual CI function.

        Args:
            block_configs: List of (block_name, module_names, input_dims, c_values) tuples.
                Ordered in processing order (first block processed first).
            d_resid_ci_fn: Dimension of the residual stream.
            reader_hidden_dims: Hidden dimensions for reader MLPs.
            transition_hidden_dim: Hidden dimension for transition MLPs.
            attn_config: Optional config for self-attention in transitions. If provided,
                transitions use a transformer block (attention → residual → MLP → residual).
        """
        super().__init__()

        if attn_config is not None:
            assert d_resid_ci_fn % attn_config.n_heads == 0, (
                f"d_resid_ci_fn ({d_resid_ci_fn}) must be divisible by "
                f"attn_config.n_heads ({attn_config.n_heads})"
            )
            d_head = d_resid_ci_fn // attn_config.n_heads
            assert d_head % 2 == 0, (
                f"d_head ({d_head}) must be even for RoPE. "
                f"d_resid_ci_fn={d_resid_ci_fn}, n_heads={attn_config.n_heads}"
            )
        self.d_resid_ci_fn = d_resid_ci_fn
        self.n_blocks = len(block_configs)
        self.block_safe_names = [name.replace(".", "-") for name, _, _, _ in block_configs]
        self.block_module_names = [modules for _, modules, _, _ in block_configs]
        self.block_input_dims = [dims for _, _, dims, _ in block_configs]
        self.block_c_values = [cs for _, _, _, cs in block_configs]

        self._inp_projectors = nn.ModuleDict()
        self._readers = nn.ModuleDict()
        self._reader_norms = nn.ModuleDict()
        self._transitions = nn.ModuleDict()
        self._transition_norms = nn.ModuleDict()
        self._attn_transitions: nn.ModuleDict | None = (
            nn.ModuleDict() if attn_config is not None else None
        )
        self._attn_norms: nn.ModuleDict | None = (
            nn.ModuleDict() if attn_config is not None else None
        )

        for block_idx, (_, _, input_dims, c_values) in enumerate(block_configs):
            safe_name = self.block_safe_names[block_idx]
            total_input_dim = sum(input_dims)
            total_c = sum(c_values)

            self._inp_projectors[safe_name] = Linear(
                total_input_dim, d_resid_ci_fn, nonlinearity="relu"
            )

            reader_layers = nn.Sequential()
            for i in range(len(reader_hidden_dims)):
                in_dim = d_resid_ci_fn if i == 0 else reader_hidden_dims[i - 1]
                out_dim = reader_hidden_dims[i]
                reader_layers.append(Linear(in_dim, out_dim, nonlinearity="relu"))
                reader_layers.append(nn.GELU())
            final_dim = reader_hidden_dims[-1] if len(reader_hidden_dims) > 0 else d_resid_ci_fn
            reader_layers.append(Linear(final_dim, total_c, nonlinearity="linear"))
            self._readers[safe_name] = reader_layers
            self._reader_norms[safe_name] = nn.RMSNorm(d_resid_ci_fn)

            if block_idx < self.n_blocks - 1:
                if transition_hidden_dim is not None:
                    transition = nn.Sequential(
                        Linear(d_resid_ci_fn, transition_hidden_dim, nonlinearity="relu"),
                        nn.GELU(),
                        Linear(transition_hidden_dim, d_resid_ci_fn, nonlinearity="relu"),
                    )
                else:
                    transition = Linear(d_resid_ci_fn, d_resid_ci_fn, nonlinearity="relu")
                self._transitions[safe_name] = transition
                self._transition_norms[safe_name] = nn.RMSNorm(d_resid_ci_fn)
                if attn_config is not None:
                    assert self._attn_transitions is not None
                    self._attn_transitions[safe_name] = SelfAttention(
                        d_model=d_resid_ci_fn,
                        n_heads=attn_config.n_heads,
                        max_len=attn_config.max_len,
                        rope_base=attn_config.rope_base,
                    )
                    assert self._attn_norms is not None
                    self._attn_norms[safe_name] = nn.RMSNorm(d_resid_ci_fn)

    @override
    def forward(
        self,
        input_acts: dict[str, Float[Tensor, "... d_in"]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        first_tensor = next(iter(input_acts.values()))
        batch_shape = first_tensor.shape[:-1]
        device = first_tensor.device
        dtype = first_tensor.dtype

        residual = torch.zeros(*batch_shape, self.d_resid_ci_fn, device=device, dtype=dtype)

        all_outputs: dict[str, Float[Tensor, "... C"]] = {}

        for block_idx in range(self.n_blocks):
            safe_name = self.block_safe_names[block_idx]
            module_names = self.block_module_names[block_idx]
            c_values = self.block_c_values[block_idx]

            block_acts = [input_acts[name] for name in module_names]
            concat_acts = torch.cat(block_acts, dim=-1)

            projection = self._inp_projectors[safe_name](concat_acts)
            residual = residual + projection

            if block_idx < self.n_blocks - 1:
                # With attention: norm → attn → residual add → norm → MLP → residual add
                # Without attention: norm → MLP → residual add
                if self._attn_transitions is not None:
                    assert self._attn_norms is not None
                    attn_out = self._attn_transitions[safe_name](
                        self._attn_norms[safe_name](residual)
                    )
                    residual = residual + attn_out

                mlp_out = self._transitions[safe_name](self._transition_norms[safe_name](residual))
                residual = residual + mlp_out
            ci_output = self._readers[safe_name](self._reader_norms[safe_name](residual))

            split_outputs = torch.split(ci_output, c_values, dim=-1)
            for module_name, module_ci in zip(module_names, split_outputs, strict=True):
                all_outputs[module_name] = module_ci

        return all_outputs


WeightDeltaAndMask = tuple[Float[Tensor, "d_out d_in"], Float[Tensor, "..."]]


class Components(ABC, nn.Module):
    def __init__(self, C: int, v_dim: int, u_dim: int):
        """
        Base class for components in a single layer (that would replace nn.Linear or nn.Embedding weight matrices).
        Initializes matrices V (which transforms the input activations) and U (which transforms the output of in_acts @ V)"

        Args:
            C: Number of components
            v_dim: Number of rows in the target weight matrix
            u_dim: Number of columns in the target weight matrix
        """
        super().__init__()
        self.C = C
        self.V = nn.Parameter(torch.empty(v_dim, C))
        self.U = nn.Parameter(torch.empty(C, u_dim))
        init_param_(self.V, fan_val=v_dim, nonlinearity="linear")
        init_param_(self.U, fan_val=C, nonlinearity="linear")

    @property
    @abstractmethod
    def weight(self) -> Float[Tensor, "rows cols"]:
        raise NotImplementedError()

    @override
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        weight_delta_and_mask: WeightDeltaAndMask | None = None,
    ) -> Tensor:
        """Forward pass through the component."""
        raise NotImplementedError()

    @abstractmethod
    def get_component_acts(self, x: Tensor) -> Tensor:
        """Get the component acts of the component."""
        raise NotImplementedError()


class LinearComponents(Components):
    """A floating point linear component. The basic building block of SPD."""

    bias: Float[Tensor, "... d_out"] | None

    def __init__(
        self,
        C: int,
        d_in: int,
        d_out: int,
        bias: Tensor | None = None,
    ):
        super().__init__(C, v_dim=d_in, u_dim=d_out)  # NOTE: linear weights are (d_out, d_in)
        self.d_in = d_in
        self.d_out = d_out

        # We don't train biases in SPD
        self.register_buffer("bias", bias)

    @property
    @override
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        """(V @ U).T. Transposed to match nn.Linear which uses (d_out, d_in)"""
        return einops.einsum(self.V, self.U, "d_in C, C d_out -> d_out d_in")

    @override
    def get_component_acts(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        return einops.einsum(x.to(self.V.dtype), self.V, "... d_in, d_in C -> ... C")

    @override
    def forward(
        self,
        x: Float[Tensor, "... d_in"],
        mask: Float[Tensor, "... C"] | None = None,
        weight_delta_and_mask: WeightDeltaAndMask | None = None,
        component_acts_cache: dict[str, Float[Tensor, "... C"]] | None = None,
    ) -> Float[Tensor, "... d_out"]:
        """Forward pass through V and U matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components.
            weight_delta_and_mask: Optional tuple of tensors containing:
                0: the weight differences between the target model and summed component weights
                1: mask over the weight delta component for each sample
            component_acts_cache: Cache dictionary to populate with component acts
        Returns:
            output: The summed output across all components
        """
        component_acts = self.get_component_acts(x)
        if component_acts_cache is not None:
            component_acts_cache["pre_detach"] = component_acts
            component_acts = component_acts.detach().requires_grad_(True)
            component_acts_cache["post_detach"] = component_acts

        if mask is not None:
            component_acts = component_acts * mask

        out = einops.einsum(component_acts, self.U, "... C, C d_out -> ... d_out")

        if weight_delta_and_mask is not None:
            weight_delta, weight_delta_mask = weight_delta_and_mask
            unmasked_delta_out = einops.einsum(x, weight_delta, "... d_in, d_out d_in -> ... d_out")
            assert unmasked_delta_out.shape[:-1] == weight_delta_mask.shape
            out += einops.einsum(
                weight_delta_mask, unmasked_delta_out, "..., ... d_out -> ... d_out"
            )

        if self.bias is not None:
            out += self.bias

        return out


class EmbeddingComponents(Components):
    """Efficient embedding components that avoid one-hot encoding."""

    def __init__(
        self,
        C: int,
        vocab_size: int,
        embedding_dim: int,
    ):
        super().__init__(C, v_dim=vocab_size, u_dim=embedding_dim)
        self.vocab_size: int = vocab_size
        self.embedding_dim: int = embedding_dim

    @property
    @override
    def weight(self) -> Float[Tensor, "vocab_size embedding_dim"]:
        """V @ U"""
        return einops.einsum(
            self.V, self.U, "vocab_size C, C embedding_dim -> vocab_size embedding_dim"
        )

    @override
    def get_component_acts(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... C"]:
        return self.V[x]

    @override
    def forward(
        self,
        x: Int[Tensor, "..."],
        mask: Float[Tensor, "... C"] | None = None,
        weight_delta_and_mask: WeightDeltaAndMask | None = None,
        component_acts_cache: dict[str, Float[Tensor, "... C"]] | None = None,
    ) -> Float[Tensor, "... embedding_dim"]:
        """Forward through the embedding component using indexing instead of one-hot matmul.

        Args:
            x: Input tensor of token indices
            mask: Tensor which masks parameter components. May be boolean or float.
            weight_delta_and_mask: Optional tuple of tensors containing:
                0: the weight differences between the target model and summed component weights
                1: mask over the weight delta component for each sample
            component_acts_cache: Cache dictionary to populate with component acts
        """
        assert x.dtype == torch.long, "x must be an integer tensor"

        component_acts: Float[Tensor, "... C"] = self.get_component_acts(x)

        if component_acts_cache is not None:
            component_acts_cache["pre_detach"] = component_acts
            component_acts = component_acts.detach().requires_grad_(True)
            component_acts_cache["post_detach"] = component_acts

        if mask is not None:
            component_acts = component_acts * mask

        out = einops.einsum(component_acts, self.U, "... C, C embedding_dim -> ... embedding_dim")

        if weight_delta_and_mask is not None:
            weight_delta, weight_delta_mask = weight_delta_and_mask
            unmasked_delta_out = weight_delta[x]
            assert unmasked_delta_out.shape[:-1] == weight_delta_mask.shape
            out += einops.einsum(
                weight_delta_mask, unmasked_delta_out, "..., ... embedding_dim -> ... embedding_dim"
            )

        return out


class Identity(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@dataclass
class ComponentsMaskInfo:
    """Specifies the mask information that will be applied to a ComponentOrModule object."""

    component_mask: Float[Tensor, "... C"]
    """when components are routed to, this specifies which subcomponents to use"""

    routing_mask: Bool[Tensor, "..."] | Literal["all"] = "all"
    """Which (batch,) or (batch, seq_len) positions to route to components vs target modules.
    If "all", all positions are routed to components."""

    weight_delta_and_mask: WeightDeltaAndMask | None = None


RoutingMasks = dict[str, Bool[Tensor, "..."]] | Literal["all"]


def make_mask_infos(
    component_masks: dict[str, Float[Tensor, "... C"]],
    routing_masks: RoutingMasks = "all",
    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = None,
) -> dict[str, ComponentsMaskInfo]:
    """Create ComponentsMaskInfo dict from dicts of component masks, and optionally routing masks,
    weight deltas, and weight delta masks.
    Keys of all dicts must be the same.

    Args:
        component_masks: Dict mapping module names to component masks. routing_masks: Dict mapping
        module names to routing masks. weight_deltas_and_masks: Dict mapping module names to tuples
        of weight deltas and masks for each module to be decomposed. Defaults to None (disable
        weight delta component) if not provided.
    Returns:
        Dict mapping module names to ComponentsMaskInfo objects.
    """
    if isinstance(routing_masks, dict):
        assert set(routing_masks) == set(component_masks)

    if weight_deltas_and_masks is not None:
        assert set(weight_deltas_and_masks) == set(component_masks)

    result: dict[str, ComponentsMaskInfo] = {}
    for name in component_masks:
        routing_mask = routing_masks[name] if isinstance(routing_masks, dict) else "all"

        weight_delta_and_mask = (
            weight_deltas_and_masks[name] if weight_deltas_and_masks is not None else None
        )

        result[name] = ComponentsMaskInfo(
            component_mask=component_masks[name],
            routing_mask=routing_mask,
            weight_delta_and_mask=weight_delta_and_mask,
        )

    return result


class LayerwiseCiFnWrapper(nn.Module):
    """Wraps a dict of per-layer CI functions with a unified interface.

    Calls each layer's CI function independently on its corresponding input activations.
    """

    def __init__(
        self,
        ci_fns: dict[str, nn.Module],
        components: dict[str, Components],
        ci_fn_type: "LayerwiseCiFnType",
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
        global_ci_fn: GlobalSharedMLPCiFn | GlobalSharedTransformerCiFn | GlobalReverseResidualCiFn,
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
