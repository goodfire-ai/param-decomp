"""Bidirectional, explicitly enumerated paths for each model family's canonical weights."""

import re
from dataclasses import dataclass

from param_decomp.topology.canonical import (
    AttnWeight,
    CanonicalWeight,
    DeltaNetWeight,
    Embed,
    FFNWeight,
    FusedAttnWeight,
    GLUWeight,
    LayerWeight,
    MLPWeight,
    MoEExpertsWeight,
    MoESharedWeight,
    SeparateAttnWeight,
    Unembed,
)

type _ProjectionPaths = tuple[tuple[AttnWeight | DeltaNetWeight | FFNWeight, str], ...]


def _prefix_paths(prefix: str, projections: _ProjectionPaths) -> _ProjectionPaths:
    return tuple((weight, f"{prefix}.{path}") for weight, path in projections)


@dataclass(frozen=True)
class _PathSchema:
    embedding_path: str
    blocks: str
    projections: _ProjectionPaths
    unembed_path: str

    def __post_init__(self) -> None:
        assert len(dict(self.projections)) == len(self.projections), self.projections
        assert len({path for _, path in self.projections}) == len(self.projections), (
            self.projections
        )

    def parse_target_path(self, path: str) -> CanonicalWeight:
        if path == self.embedding_path:
            return Embed()
        if path == self.unembed_path:
            return Unembed()
        match = re.match(rf"^{re.escape(self.blocks)}\.(\d+)\.(.+)$", path)
        assert match is not None, f"Invalid block path: {path!r}"
        by_path = {path: weight for weight, path in self.projections}
        projection = match.group(2)
        assert projection in by_path, f"Unknown projection in {path!r}"
        return LayerWeight(int(match.group(1)), by_path[projection])

    def render_canonical_weight(self, weight: CanonicalWeight) -> str:
        match weight:
            case Embed():
                return self.embedding_path
            case Unembed():
                return self.unembed_path
            case LayerWeight(layer_idx=layer, name=name):
                by_weight = dict(self.projections)
                assert name in by_weight, f"Unsupported projection: {name!r}"
                return f"{self.blocks}.{layer}.{by_weight[name]}"
            case _:
                raise ValueError(f"Unknown canonical weight: {weight!r}")


_SEPARATE_ATTN: _ProjectionPaths = (
    (SeparateAttnWeight("q"), "q_proj"),
    (SeparateAttnWeight("k"), "k_proj"),
    (SeparateAttnWeight("v"), "v_proj"),
    (SeparateAttnWeight("o"), "o_proj"),
)
_GLU: _ProjectionPaths = (
    (GLUWeight("gate"), "gate_proj"),
    (GLUWeight("up"), "up_proj"),
    (GLUWeight("down"), "down_proj"),
)
_SIMPLE_MLP = _PathSchema(
    embedding_path="wte",
    blocks="h",
    projections=(
        *_prefix_paths("attn", _SEPARATE_ATTN),
        (MLPWeight("up"), "mlp.c_fc"),
        (MLPWeight("down"), "mlp.down_proj"),
    ),
    unembed_path="lm_head",
)
_HF_GLU = _PathSchema(
    embedding_path="embed_tokens",
    blocks="layers",
    projections=(*_prefix_paths("self_attn", _SEPARATE_ATTN), *_prefix_paths("mlp", _GLU)),
    unembed_path="lm_head",
)

_MODEL_TYPE_PATH_SCHEMAS = {
    "LlamaSimple": _PathSchema(
        embedding_path="wte",
        blocks="h",
        projections=(*_prefix_paths("attn", _SEPARATE_ATTN), *_prefix_paths("mlp", _GLU)),
        unembed_path="lm_head",
    ),
    "LlamaSimpleMLP": _SIMPLE_MLP,
    "GPT2Simple": _SIMPLE_MLP,
    "GPT2": _PathSchema(
        embedding_path="wte",
        blocks="h_torch",
        projections=(
            (FusedAttnWeight("qkv"), "attn.c_attn"),
            (FusedAttnWeight("o"), "attn.c_proj"),
            (MLPWeight("up"), "mlp.c_fc"),
            (MLPWeight("down"), "mlp.c_proj"),
        ),
        unembed_path="lm_head",
    ),
    "Llama": _HF_GLU,
    "Qwen3": _HF_GLU,
    "Qwen3_5Moe": _PathSchema(
        embedding_path="embed_tokens",
        blocks="layers",
        projections=(
            *_prefix_paths("self_attn", _SEPARATE_ATTN),
            (DeltaNetWeight("q"), "linear_attn.in_proj_qkv.q"),
            (DeltaNetWeight("k"), "linear_attn.in_proj_qkv.k"),
            (DeltaNetWeight("v"), "linear_attn.in_proj_qkv.v"),
            (DeltaNetWeight("z"), "linear_attn.in_proj_z"),
            (DeltaNetWeight("b"), "linear_attn.in_proj_b"),
            (DeltaNetWeight("a"), "linear_attn.in_proj_a"),
            (DeltaNetWeight("out"), "linear_attn.out_proj"),
            (MoEExpertsWeight("gate"), "mlp.experts.gate_proj"),
            (MoEExpertsWeight("up"), "mlp.experts.up_proj"),
            (MoEExpertsWeight("down"), "mlp.experts.down_proj"),
            (MoESharedWeight("gate"), "mlp.shared_expert.gate_proj"),
            (MoESharedWeight("up"), "mlp.shared_expert.up_proj"),
            (MoESharedWeight("down"), "mlp.shared_expert.down_proj"),
        ),
        unembed_path="lm_head",
    ),
}


def path_schema_for_model_type(model_type: str) -> _PathSchema:
    """Select immutable path conventions by target-model class name, without a live model."""
    assert model_type in _MODEL_TYPE_PATH_SCHEMAS, (
        f"No path schema for model type {model_type!r}. Add one in path_schemas.py."
    )
    return _MODEL_TYPE_PATH_SCHEMAS[model_type]
