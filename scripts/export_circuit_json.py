"""Export an OptimizedPromptAttributionResult to JSON for the circuit graph renderer.

Usage from a notebook / script:

    from scripts.export_circuit_json import export_circuit_json

    export_circuit_json(
        circuit=circuit,
        token_ids=tokens.tolist(),
        token_strings=tok.get_spans(tokens.tolist()),
        output_path=Path("king_circuit.json"),
        interp=interp,          # optional InterpRepo
        graph_interp=gi_repo,   # optional GraphInterpRepo
    )
"""

import json
from pathlib import Path

from spd.app.backend.compute import Edge, OptimizedPromptAttributionResult
from spd.autointerp.repo import InterpRepo
from spd.graph_interp.repo import GraphInterpRepo


def _parse_node_key(key: str) -> tuple[str, int, int]:
    """Parse 'h.3.attn.o_proj:11:361' -> ('h.3.attn.o_proj', 11, 361)."""
    parts = key.split(":")
    layer = ":".join(parts[:-2])
    seq = int(parts[-2])
    cidx = int(parts[-1])
    return layer, seq, cidx


def _concrete_to_canonical(layer: str) -> str:
    """Map concrete model path to canonical address used by graph layout.

    h.3.attn.o_proj -> 3.attn.o
    h.3.attn.v_proj -> 3.attn.v
    h.3.attn.q_proj -> 3.attn.q
    h.3.attn.k_proj -> 3.attn.k
    h.3.mlp.c_fc    -> 3.mlp.up
    h.3.mlp.down_proj -> 3.mlp.down
    h.3.mlp.gate_proj -> 3.glu.gate
    h.3.mlp.up_proj   -> 3.glu.up
    wte -> embed
    lm_head -> output
    """
    if layer == "wte":
        return "embed"
    if layer == "lm_head":
        return "output"

    PROJ_MAP = {
        "q_proj": "q",
        "k_proj": "k",
        "v_proj": "v",
        "o_proj": "o",
        "c_fc": "up",
        "down_proj": "down",
        "gate_proj": "gate",
        "up_proj": "up",
    }

    # h.{block}.{sublayer}.{proj_name}
    parts = layer.split(".")
    assert len(parts) == 4, f"Expected h.N.sublayer.proj, got {layer!r}"
    block_idx = parts[1]
    sublayer = parts[2]  # "attn" or "mlp"
    proj_name = parts[3]

    canonical_proj = PROJ_MAP.get(proj_name)
    assert canonical_proj is not None, f"Unknown projection: {proj_name!r} in {layer!r}"

    # Determine canonical sublayer
    if sublayer == "attn":
        canonical_sublayer = "attn"
    elif sublayer == "mlp":
        canonical_sublayer = "glu" if proj_name in ("gate_proj", "up_proj") else "mlp"
    else:
        canonical_sublayer = sublayer

    return f"{block_idx}.{canonical_sublayer}.{canonical_proj}"


def _edge_to_dict(e: Edge) -> dict[str, object]:
    return {
        "source": str(e.source),
        "target": str(e.target),
        "attribution": e.strength,
        "is_cross_seq": e.is_cross_seq,
    }


def _get_label(
    layer: str,
    cidx: int,
    interp: InterpRepo | None,
    graph_interp: GraphInterpRepo | None,
) -> str | None:
    component_key = f"{layer}:{cidx}"
    if graph_interp is not None:
        unified = graph_interp.get_unified_label(component_key)
        if unified is not None:
            return unified.label
        output = graph_interp.get_output_label(component_key)
        if output is not None:
            return output.label
    if interp is not None:
        ir = interp.get_interpretation(component_key)
        if ir is not None:
            return ir.label
    return None


def export_circuit_json(
    circuit: OptimizedPromptAttributionResult,
    token_ids: list[int],
    token_strings: list[str],
    output_path: Path,
    min_ci: float = 0.3,
    min_edge_attr: float = 0.1,
    interp: InterpRepo | None = None,
    graph_interp: GraphInterpRepo | None = None,
) -> None:
    """Export circuit to JSON for the standalone HTML renderer.

    Args:
        circuit: The optimized circuit result from EditableModel.optimize_circuit.
        token_ids: Raw token IDs for each position.
        token_strings: Display strings for each position (from tok.get_spans).
        output_path: Where to write the JSON.
        min_ci: Minimum CI to include a node.
        min_edge_attr: Minimum |attribution| to include an edge.
        interp: Optional InterpRepo for autointerp labels.
        graph_interp: Optional GraphInterpRepo for graph-interp labels.
    """
    # Build tokens list
    tokens = [
        {"pos": i, "id": tid, "string": tstr}
        for i, (tid, tstr) in enumerate(zip(token_ids, token_strings, strict=True))
    ]

    # Build nodes from ci_vals, filtering by min_ci
    nodes = []
    node_keys_kept = set()
    for key, ci in circuit.node_ci_vals.items():
        if ci < min_ci:
            continue
        layer, seq, cidx = _parse_node_key(key)
        canonical = _concrete_to_canonical(layer)
        act = circuit.node_subcomp_acts.get(key, 0.0)

        label = _get_label(layer, cidx, interp, graph_interp)

        nodes.append(
            {
                "key": key,
                "graph_key": f"{layer}:{cidx}",
                "layer": layer,
                "canonical": canonical,
                "seq": seq,
                "cidx": cidx,
                "ci": round(ci, 4),
                "activation": round(act, 4),
                "token": token_strings[seq] if seq < len(token_strings) else "?",
                "label": label,
            }
        )
        node_keys_kept.add(key)

    # Build edges, filtering by min_edge_attr and requiring both endpoints in kept nodes
    edges = []
    for e in circuit.edges:
        if abs(e.strength) < min_edge_attr:
            continue
        src_key = str(e.source)
        tgt_key = str(e.target)
        if src_key not in node_keys_kept or tgt_key not in node_keys_kept:
            continue
        edges.append(_edge_to_dict(e))

    metrics: dict[str, float] = {"l0_total": circuit.metrics.l0_total}
    if circuit.metrics.ci_masked_label_prob is not None:
        metrics["ci_masked_label_prob"] = circuit.metrics.ci_masked_label_prob
    if circuit.metrics.stoch_masked_label_prob is not None:
        metrics["stoch_masked_label_prob"] = circuit.metrics.stoch_masked_label_prob

    data = {
        "tokens": tokens,
        "nodes": nodes,
        "edges": edges,
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(nodes)} nodes, {len(edges)} edges to {output_path}")
