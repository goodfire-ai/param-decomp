import math
from abc import ABC, abstractmethod
from typing import Literal, override

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn.init import calculate_gain
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from param_decomp.decomposition_targets import Identity
from param_decomp.masks import WeightDeltaAndMask

# This is equivalent to `torch.nn.init._NonlinearityType`, but for some reason this is not always
# importable. see https://github.com/goodfire-ai/param-decomp/actions/runs/16927877557/job/47967138342
_NonlinearityType = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]


def init_param_(
    param: Tensor,
    fan_val: float,
    mean: float = 0.0,
    nonlinearity: _NonlinearityType = "linear",
    generator: torch.Generator | None = None,
) -> None:
    """Initialise ``param`` in place from a Kaiming normal distribution.

    The std used is ``gain(nonlinearity) / sqrt(fan_val)``.

    Args:
        param: Parameter tensor to fill in place.
        fan_val: Value used as ``fan`` in Kaiming normal: appears under the
            square root in the denominator of std.
        mean: Mean of the sampled normal distribution.
        nonlinearity: Nonlinearity name passed to
            :func:`torch.nn.init.calculate_gain`.
        generator: Optional RNG for reproducibility.
    """
    gain: float = calculate_gain(nonlinearity)
    std: float = gain / math.sqrt(fan_val)
    with torch.no_grad():
        param.normal_(mean, std, generator=generator)


class Components(ABC, nn.Module):
    """Per-layer components that decompose a target weight matrix.

    Subclasses replace the weight of a single ``nn.Linear``,
    ``transformers.pytorch_utils.Conv1D``, or ``nn.Embedding`` with a sum of
    ``C`` rank-1 outer products: ``weight ≈ sum_c V[:, c] ⊗ U[c, :]``. ``V`` maps
    input activations to per-component scalars; ``U`` maps those scalars back to
    the output space.

    Attributes:
        C: Number of components.
        V: Input-projection parameter of shape ``[v_dim, C]``.
        U: Output-projection parameter of shape ``[C, u_dim]``.
    """

    def __init__(self, C: int, v_dim: int, u_dim: int):
        """Initialise ``V`` and ``U`` parameters via Kaiming normal.

        Args:
            C: Number of components.
            v_dim: Number of rows in the target weight matrix (input dim).
            u_dim: Number of columns in the target weight matrix (output dim).
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
        """Effective weight (``V @ U``, possibly transposed) for this component."""
        raise NotImplementedError()

    @override
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        weight_delta_and_mask: WeightDeltaAndMask | None = None,
    ) -> Tensor:
        """Apply the masked decomposition (and optional weight-delta term) to ``x``."""
        raise NotImplementedError()

    @abstractmethod
    def get_component_acts(self, x: Tensor) -> Tensor:
        """Per-component scalar activations ``V^T x`` for the input batch."""
        raise NotImplementedError()


class LinearComponents(Components):
    """Components replacing an ``nn.Linear``-shaped weight.

    The effective weight is ``(V @ U).T`` to match PyTorch's
    ``[d_out, d_in]`` storage. A frozen bias from the target module may be
    re-added in the forward; biases are not trained in PD.
    """

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

        # We don't train biases in PD.
        self.register_buffer("bias", bias)

    @property
    @override
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        """``(V @ U).T`` — transposed to match ``nn.Linear``'s ``[d_out, d_in]``."""
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
        """Apply ``mask * (V^T x)`` then project back by ``U``, plus optional delta term.

        When ``component_acts_cache`` is given, the pre- and post-detach
        component activations are stored under the keys ``"pre_detach"`` and
        ``"post_detach"`` for downstream gradient surgery (e.g. PPGD).

        Args:
            x: Input activations of shape ``[..., d_in]``.
            mask: Per-component mask of shape ``[..., C]`` multiplied into the
                component activations. ``None`` means no masking (unmasked
                forward).
            weight_delta_and_mask: Optional ``(weight_delta, weight_delta_mask)``
                pair adding a residual ``weight_delta @ x`` term scaled by the
                per-position mask. Enables the delta-component pathway.
            component_acts_cache: Dict populated with pre/post-detach component
                acts when provided.

        Returns:
            Output tensor of shape ``[..., d_out]`` with bias added if present.
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
    """Components replacing an ``nn.Embedding`` weight.

    Avoids materialising one-hot vectors by indexing ``V`` directly with the
    input token ids. The effective weight is ``V @ U`` of shape
    ``[vocab_size, embedding_dim]``.
    """

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
        """``V @ U`` — the effective embedding matrix."""
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
        """Embedding forward via index-into-``V``, masked, then projected by ``U``.

        Equivalent to the ``LinearComponents`` forward but uses ``V[x]`` in
        place of a one-hot matmul.

        Args:
            x: Long-tensor of token indices, shape ``[...]``.
            mask: Per-component mask of shape ``[..., C]``, boolean or float.
                ``None`` means no masking.
            weight_delta_and_mask: Optional ``(weight_delta, weight_delta_mask)``
                pair adding a residual ``weight_delta[x]`` term scaled by the
                per-position mask.
            component_acts_cache: Dict populated with pre/post-detach component
                acts when provided.

        Returns:
            Output tensor of shape ``[..., embedding_dim]``.
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


def get_module_input_dim(target_module: nn.Module) -> int:
    """Input dimension of a Linear-like target module.

    Args:
        target_module: A target module of type ``nn.Linear``, Radford
            ``Conv1D``, or :class:`Identity`. Embedding modules have no
            numeric input dim and must be handled by callers separately.

    Returns:
        Input dimension ``d_in``.

    Raises:
        ValueError: For unsupported module types (including ``nn.Embedding``).
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


def make_components(
    target_model: nn.Module,
    module_to_c: dict[str, int],
) -> dict[str, Components]:
    """Build one :class:`Components` instance per target module path.

    Dispatches by target-module type:

    - ``nn.Linear`` → :class:`LinearComponents` (frozen bias carried over).
    - Radford ``Conv1D`` → :class:`LinearComponents` with shapes swapped to
      account for the transposed weight layout (frozen bias carried over).
    - :class:`Identity` → :class:`LinearComponents` with ``d_in == d_out``
      and no bias.
    - ``nn.Embedding`` → :class:`EmbeddingComponents`.

    Args:
        target_model: Model that owns the target submodules.
        module_to_c: Map from submodule path to component count ``C``.

    Returns:
        Dict keyed by submodule path mapping to the matching ``Components``.

    Raises:
        ValueError: For unsupported target-module types.
    """
    out: dict[str, Components] = {}
    for path, C in module_to_c.items():
        target_module = target_model.get_submodule(path)
        match target_module:
            case nn.Linear():
                d_out, d_in = target_module.weight.shape
                comp: Components = LinearComponents(
                    C=C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=target_module.bias.data if target_module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            case RadfordConv1D():
                d_in, d_out = target_module.weight.shape
                comp = LinearComponents(
                    C=C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=target_module.bias.data if target_module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            case Identity():
                comp = LinearComponents(
                    C=C,
                    d_in=target_module.d,
                    d_out=target_module.d,
                    bias=None,
                )
            case nn.Embedding():
                comp = EmbeddingComponents(
                    C=C,
                    vocab_size=target_module.num_embeddings,
                    embedding_dim=target_module.embedding_dim,
                )
            case _:
                raise ValueError(f"Module {target_module} not supported")
        out[path] = comp
    return out
