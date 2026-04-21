"""TransformerTopology: the public interface for canonical ↔ concrete path mapping."""

from torch import nn

from spd.topology.canonical import (
    CanonicalWeight,
    Embed,
    FusedAttnWeight,
    LayerWeight,
    SeparateAttnWeight,
    Unembed,
)
from spd.topology.path_schemas import get_path_schema


class TransformerTopology:
    """Bidirectional mapping between canonical weights and concrete module paths.

    Built from a target model (nn.Module). Independent of decomposition.
    """

    def __init__(self, target_model: nn.Module) -> None:
        self.target_model = target_model
        self.path_schema = get_path_schema(target_model)

    def canon_to_target(self, canonical: str) -> str:
        return self.path_schema.render_canonical_weight(CanonicalWeight.parse(canonical))

    def target_to_canon(self, target_module_path: str) -> str:
        return self.path_schema.parse_target_path(target_module_path).canonical_str()

    def _get_module(self, canonical: CanonicalWeight) -> nn.Module:
        target_path = self.path_schema.render_canonical_weight(canonical)
        return self.target_model.get_submodule(target_path)

    @property
    def embedding_module(self) -> nn.Embedding:
        mod = self._get_module(Embed())
        assert isinstance(mod, nn.Embedding)
        return mod

    @property
    def unembed_module(self) -> nn.Linear:
        mod = self._get_module(Unembed())
        assert isinstance(mod, nn.Linear)
        return mod

    @property
    def n_blocks(self) -> int:
        blocks = self.target_model.get_submodule(self.path_schema.blocks)
        assert isinstance(blocks, nn.ModuleList)
        return len(blocks)

    def get_unembed_weight(self):
        """Unembedding weight matrix transposed to [d_model, vocab]."""
        return self.unembed_module.weight.T.detach()

    def is_cross_seq_pair(self, source_canonical: str, target_canonical: str) -> bool:
        """True if source is k/v and target is o in the same block."""
        source = CanonicalWeight.parse(source_canonical)
        target = CanonicalWeight.parse(target_canonical)
        match source, target:
            case (
                LayerWeight(layer_idx=si, name=SeparateAttnWeight(weight="k" | "v")),
                LayerWeight(layer_idx=ti, name=SeparateAttnWeight(weight="o")),
            ):
                return si == ti
            case (
                LayerWeight(layer_idx=si, name=FusedAttnWeight(weight="qkv")),
                LayerWeight(layer_idx=ti, name=FusedAttnWeight(weight="o")),
            ):
                return si == ti
            case _:
                return False
