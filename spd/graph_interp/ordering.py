"""Layer ordering for graph interpretation.

Uses the topology module's CanonicalWeight system for correct ordering
across all model architectures. Canonical addresses are provided by
ModelMetadata.layer_descriptions (concrete path → canonical string).
"""

from spd.topology.canonical import (
    CanonicalWeight,
    FusedAttnWeight,
    GLUWeight,
    LayerWeight,
    MLPWeight,
    SeparateAttnWeight,
)

_SUBLAYER_ORDER = {"attn": 0, "attn_fused": 0, "glu": 1, "mlp": 1}

_PROJECTION_ORDER: dict[type, dict[str, int]] = {
    SeparateAttnWeight: {"q": 0, "k": 1, "v": 2, "o": 3},
    FusedAttnWeight: {"qkv": 0, "o": 1},
    GLUWeight: {"gate": 0, "up": 1, "down": 2},
    MLPWeight: {"up": 0, "down": 1},
}


def canonical_sort_key(canonical: str) -> tuple[int, int, int]:
    """Sort key for a canonical address string like '0.attn.q' or '1.mlp.down'."""
    weight = CanonicalWeight.parse(canonical)
    assert isinstance(weight, LayerWeight), f"Expected LayerWeight, got {type(weight).__name__}"

    match weight.name:
        case SeparateAttnWeight(weight=p):
            sublayer_idx = _SUBLAYER_ORDER["attn"]
            proj_idx = _PROJECTION_ORDER[SeparateAttnWeight][p]
        case FusedAttnWeight(weight=p):
            sublayer_idx = _SUBLAYER_ORDER["attn_fused"]
            proj_idx = _PROJECTION_ORDER[FusedAttnWeight][p]
        case GLUWeight(weight=p):
            sublayer_idx = _SUBLAYER_ORDER["glu"]
            proj_idx = _PROJECTION_ORDER[GLUWeight][p]
        case MLPWeight(weight=p):
            sublayer_idx = _SUBLAYER_ORDER["mlp"]
            proj_idx = _PROJECTION_ORDER[MLPWeight][p]

    return weight.layer_idx, sublayer_idx, proj_idx


def parse_component_key(key: str) -> tuple[str, int]:
    """Split 'h.1.mlp.c_fc:42' into ('h.1.mlp.c_fc', 42)."""
    layer, idx_str = key.rsplit(":", 1)
    return layer, int(idx_str)


def group_and_sort_by_layer(
    component_keys: list[str],
    layer_descriptions: dict[str, str],
) -> list[tuple[str, list[str]]]:
    """Group component keys by layer, return [(layer, [keys])] in topological order.

    Args:
        component_keys: Component keys like 'h.0.attn.q_proj:42'.
        layer_descriptions: Mapping from concrete layer path to canonical address
            (from ModelMetadata.layer_descriptions).
    """
    by_layer: dict[str, list[str]] = {}
    for key in component_keys:
        layer, _ = parse_component_key(key)
        by_layer.setdefault(layer, []).append(key)

    def sort_key(layer: str) -> tuple[int, int, int]:
        canonical = layer_descriptions[layer]
        return canonical_sort_key(canonical)

    sorted_layers = sorted(by_layer.keys(), key=sort_key)

    result: list[tuple[str, list[str]]] = []
    for layer in sorted_layers:
        keys = sorted(by_layer[layer], key=lambda k: parse_component_key(k)[1])
        result.append((layer, keys))
    return result


def is_later_layer(earlier: str, later: str, layer_descriptions: dict[str, str]) -> bool:
    """Check if `later` is topologically after `earlier`."""
    return canonical_sort_key(layer_descriptions[earlier]) < canonical_sort_key(
        layer_descriptions[later]
    )
