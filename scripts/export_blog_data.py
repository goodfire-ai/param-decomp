"""Export attribution graph data from the SPD app database for the VPD blog post.

Writes pure JSON to an output directory. No blog repo dependency.

Run: uv run python scripts/export_blog_data.py s-55ea3f9b [--out-dir ../vpd-blog/data]
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.repo import InterpRepo
from spd.dataset_attributions import AttributionRepo
from spd.harvest.schemas import get_harvest_dir
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.settings import SPD_OUT_DIR
from spd.topology import TransformerTopology
from spd.topology.canonical import CanonicalWeight, Embed, LayerWeight, Unembed

DEFAULT_OUT_DIR = Path(__file__).parent.parent.parent / "vpd-blog" / "data"

PROMPT_ATTR_DB = SPD_OUT_DIR / "app" / "prompt_attr.db"

# TODO: replace with real cluster mapping file path once available
# Expected format: {"clusters": {"h.0.mlp.c_fc:327": 0, ...}, "spd_run": "...", ...}
# Generate with: python -m spd.clustering.scripts.get_cluster_mapping <clustering_run_dir> --iteration N
CLUSTER_MAPPING_PATH: Path | None = Path(
    "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-70b28465/cluster_mapping.json"
)


_SUBLAYER_DISPLAY = {"attn": "Attn", "attn_fused": "Attn", "mlp": "MLP", "glu": "MLP"}


def canonical_display_name(canon: str) -> str:
    """'0.mlp.up' -> 'MLP 0 Up', 'embed' -> 'Embed', etc."""
    cw = CanonicalWeight.parse(canon)
    match cw:
        case Embed():
            return "Embed"
        case Unembed():
            return "Output"
        case LayerWeight(layer_idx=idx):
            sublayer, proj = canon.split(".")[1], canon.split(".")[2]
            return f"{_SUBLAYER_DISPLAY[sublayer]} {idx} {proj.capitalize()}"
        case _:
            raise ValueError(f"Unhandled canonical weight: {canon}")


def load_cluster_mapping(
    path: Path,
    concrete_to_canonical: dict[str, str],
) -> dict[str, int | None]:
    """Load cluster mapping JSON and convert concrete keys to canonical 'layer:component_idx' keys."""
    with open(path) as f:
        data = json.load(f)
    raw_clusters: dict[str, int] = data["clusters"]
    canonical_clusters: dict[str, int | None] = {}
    for key, cluster_id in raw_clusters.items():
        concrete_layer, comp_idx = key.rsplit(":", 1)
        canonical_layer = concrete_to_canonical.get(concrete_layer)
        assert canonical_layer, f"Unknown concrete layer: {concrete_layer}"
        canonical_clusters[f"{canonical_layer}:{comp_idx}"] = cluster_id
    return canonical_clusters


def clusters_for_graph(
    cluster_mapping: dict[str, int | None],
    active_keys: set[str],
) -> dict[str, int | None]:
    """Map cluster assignments onto graph node keys.

    Cluster mapping keys are 'canonical_layer:component_idx' (no seq position).
    Graph node keys are 'canonical_layer:seq_pos:component_idx'.
    """
    node_clusters: dict[str, int | None] = {}
    for node_key in active_keys:
        layer, _seq, comp_idx = node_key.split(":")
        lookup_key = f"{layer}:{comp_idx}"
        if lookup_key in cluster_mapping:
            node_clusters[node_key] = cluster_mapping[lookup_key]
    return node_clusters


def harvest_key(canonical_node_key: str, canonical_to_concrete: dict[str, str]) -> str:
    """'0.mlp.up:1:327' -> 'h.0.mlp.c_fc:327' (concrete layer, drop seq position)."""
    layer, _, comp_idx = canonical_node_key.split(":")
    return f"{canonical_to_concrete[layer]}:{comp_idx}"


def convert_examples(
    raw_examples: list[dict[str, Any]],
    tokenizer: AppTokenizer,
    n_examples: int = 30,
    window: int = 16,
) -> dict[str, Any]:
    """Returns {"examples": [...], "max_act": float}."""
    out: list[dict[str, Any]] = []
    global_max_act = 0.0
    for ex in raw_examples[:n_examples]:
        token_ids = ex["token_ids"]
        ci_values = ex["activations"]["causal_importance"]
        acts = ex["activations"]["component_activation"]
        n = len(token_ids)
        spans = tokenizer.get_spans(token_ids)

        firing_indices = [i for i, ci in enumerate(ci_values) if ci > 0]
        center = firing_indices[0] if firing_indices else n // 2
        start = max(0, center - window // 2)
        end = min(n, start + window)
        start = max(0, end - window)

        window_acts = acts[start:end]
        local_max = max((abs(a) for a in window_acts), default=0.0)
        global_max_act = max(global_max_act, local_max)

        tokens = [
            {
                "token": span,
                "ci": round(ci, 4),
                "act": round(a, 4),
            }
            for span, ci, a in zip(spans[start:end], ci_values[start:end], window_acts, strict=True)
        ]
        out.append({"tokens": tokens})
    return {"examples": out, "max_act": round(global_max_act, 4)}


def convert_pmi(raw_pmi: dict[str, Any], tokenizer: AppTokenizer, top_k: int = 8) -> dict[str, Any]:
    return {
        "top": [[tokenizer.get_tok_display(tid), score] for tid, score in raw_pmi["top"][:top_k]]
    }


def build_dataset_attributions(
    node_key: str,
    active_keys: set[str],
    interp_by_key: dict[str, str],
    canonical_to_concrete: dict[str, str],
    attr_repo: AttributionRepo,
    tokenizer: AppTokenizer,
    top_k: int = 8,
) -> dict[str, Any]:
    """Build incoming/outgoing dataset attribution entries for a node."""
    storage = attr_repo.get_attributions()
    layer, _seq_str, idx_str = node_key.split(":")
    storage_key = f"{layer}:{idx_str}"

    def resolve_label(entry_layer: str, entry_idx: int) -> str:
        if entry_layer in ("embed", "output"):
            return tokenizer.get_tok_display(entry_idx)
        h_key = f"{canonical_to_concrete[entry_layer]}:{entry_idx}"
        return interp_by_key.get(h_key, f"{entry_layer}:{entry_idx}")

    def find_graph_key(entry_layer: str, entry_idx: int) -> str | None:
        for k in active_keys:
            parts = k.split(":")
            k_idx = int(parts[-1])
            k_layer = ":".join(parts[:-2])
            if k_layer == entry_layer and k_idx == entry_idx:
                return k
        return None

    def collect_entries(entries: list[Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for entry in entries:
            graph_key = find_graph_key(entry.layer, entry.component_idx)
            label = resolve_label(entry.layer, entry.component_idx)
            result.append(
                {
                    "key": graph_key,
                    "label": label,
                    "value": round(entry.value, 4),
                }
            )
        return result

    pos_sources = storage.get_top_sources(storage_key, top_k, "positive", "attr_abs")
    neg_sources = storage.get_top_sources(storage_key, top_k, "negative", "attr_abs")
    all_sources = pos_sources + neg_sources
    all_sources.sort(key=lambda e: abs(e.value), reverse=True)
    incoming = collect_entries(all_sources[:top_k])

    pos_targets = storage.get_top_targets(storage_key, top_k, "positive", "attr_abs")
    neg_targets = storage.get_top_targets(storage_key, top_k, "negative", "attr_abs")
    all_targets = pos_targets + neg_targets
    all_targets.sort(key=lambda e: abs(e.value), reverse=True)
    outgoing = collect_entries(all_targets[:top_k])

    return {"incoming": incoming, "outgoing": outgoing}


def build_graph(
    graph_id: int,
    tokenizer: AppTokenizer,
    harvest_by_key: dict[str, tuple[float, str, str, str]],
    interp_by_key: dict[str, str],
    canonical_to_concrete: dict[str, str],
    app_db: sqlite3.Connection,
    attr_repo: AttributionRepo | None,
    cluster_mapping: dict[str, int | None] | None,
    output_filter: str | int | None = None,
    n_tokens: int | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    row = app_db.execute(
        "SELECT prompt_id, edges_data_abs, node_ci_vals, node_subcomp_acts FROM graphs WHERE id=?",
        (graph_id,),
    ).fetchone()
    assert row, f"Graph {graph_id} not found"
    prompt_id, raw_edges, raw_ci, raw_acts = row

    prompt_row = app_db.execute("SELECT token_ids FROM prompts WHERE id=?", (prompt_id,)).fetchone()
    assert prompt_row
    token_ids = json.loads(prompt_row[0])
    all_tokens = tokenizer.get_spans(token_ids)
    tokens = all_tokens[:n_tokens] if n_tokens else all_tokens
    allowed_seq = set(range(len(tokens)))

    db_edges: list[dict[str, Any]] = json.loads(raw_edges)
    db_ci_vals: dict[str, float] = json.loads(raw_ci)
    _ = json.loads(raw_acts) if raw_acts else {}

    def seq_allowed(key: str) -> bool:
        return int(key.split(":")[1]) in allowed_seq

    node_ci_vals = {k: v for k, v in db_ci_vals.items() if v > 0 and seq_allowed(k)}
    active_keys = set(node_ci_vals.keys())

    allowed_outputs: set[str] | None = None
    match output_filter:
        case str():
            allowed_outputs = {output_filter}
        case int():
            output_strength: dict[str, float] = {}
            for e in db_edges:
                t = e["target"]
                if t["layer"] == "output":
                    key = f"output:{t['seq_pos']}:{t['component_idx']}"
                    if seq_allowed(key):
                        output_strength[key] = output_strength.get(key, 0) + abs(e["strength"])
            top_keys = sorted(output_strength, key=lambda k: output_strength[k], reverse=True)[
                :output_filter
            ]
            allowed_outputs = set(top_keys)
        case None:
            pass

    def is_allowed(key: str, layer: str) -> bool:
        if key in active_keys:
            return True
        if layer == "embed":
            return True
        if layer == "output":
            return allowed_outputs is None or key in allowed_outputs
        return False

    edges: list[dict[str, Any]] = []
    for e in db_edges:
        src = f"{e['source']['layer']}:{e['source']['seq_pos']}:{e['source']['component_idx']}"
        tgt = f"{e['target']['layer']}:{e['target']['seq_pos']}:{e['target']['component_idx']}"
        if not seq_allowed(src) or not seq_allowed(tgt):
            continue

        if is_allowed(src, e["source"]["layer"]) and is_allowed(tgt, e["target"]["layer"]):
            edges.append(
                {
                    "src": src,
                    "tgt": tgt,
                    "val": e["strength"],
                    "is_cross_seq": e["is_cross_seq"],
                }
            )
            active_keys.add(src)
            active_keys.add(tgt)

    for key in active_keys:
        if key not in node_ci_vals:
            layer = key.split(":")[0]
            assert layer in ("embed", "output"), f"unexpected non-CI node: {key}"
            node_ci_vals[key] = 1.0

    components: dict[str, dict[str, Any]] = {}
    component_details: dict[str, dict[str, Any]] = {}
    for node_key in sorted(active_keys):
        layer, seq_str, idx_str = node_key.split(":")
        display = canonical_display_name(layer)

        if layer == "embed":
            components[node_key] = {
                "type": "embed",
                "token": tokens[int(seq_str)],
                "layer_display": display,
            }
        elif layer == "output":
            components[node_key] = {
                "type": "logit",
                "token": tokenizer.get_tok_display(int(idx_str)),
                "layer_display": display,
            }
        else:
            h_key = harvest_key(node_key, canonical_to_concrete)
            label = interp_by_key.get(h_key, "unlabeled")

            comp: dict[str, Any] = {
                "type": "component",
                "label": label,
                "layer_display": display,
            }

            harvest_row = harvest_by_key.get(h_key)
            if harvest_row:
                fd, raw_ex, raw_in_pmi, raw_out_pmi = harvest_row
                comp["firing_density"] = fd
                examples_result = convert_examples(json.loads(raw_ex), tokenizer)
                details: dict[str, Any] = {
                    "activation_examples": examples_result["examples"],
                    "max_act": examples_result["max_act"],
                }
                if attr_repo:
                    details["dataset_attributions"] = build_dataset_attributions(
                        node_key,
                        active_keys,
                        interp_by_key,
                        canonical_to_concrete,
                        attr_repo,
                        tokenizer,
                    )
                else:
                    details["input_token_pmi"] = convert_pmi(json.loads(raw_in_pmi), tokenizer)
                    details["output_token_pmi"] = convert_pmi(json.loads(raw_out_pmi), tokenizer)
                component_details[node_key] = details

            components[node_key] = comp

    topology_layers = (
        ["embed"]
        + [
            f"{b}.{sub}.{proj}"
            for b in range(
                max(int(k.split(".")[0]) for k in canonical_to_concrete if k[0].isdigit()) + 1
            )
            for sub, proj in [
                ("attn", "q"),
                ("attn", "k"),
                ("attn", "v"),
                ("attn", "o"),
                ("mlp", "up"),
                ("mlp", "down"),
            ]
            if f"{b}.{sub}.{proj}" in canonical_to_concrete
        ]
        + ["output"]
    )

    graph: dict[str, Any] = {
        "tokens": tokens,
        "layers": topology_layers,
        "edges": edges,
        "node_ci_vals": node_ci_vals,
        "components": components,
    }

    if cluster_mapping:
        graph["clusters"] = clusters_for_graph(cluster_mapping, active_keys)

    return graph, component_details


def main() -> None:
    parser = argparse.ArgumentParser(description="Export blog graph data from SPD databases")
    parser.add_argument("run_id", help="SPD run ID (e.g. s-55ea3f9b)")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for JSON files"
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id: str = args.run_id

    print("Loading run info...")
    run_info = SPDRunInfo.from_path(f"goodfire/spd/runs/{run_id}")
    assert run_info.config.tokenizer_name
    tokenizer = AppTokenizer.from_pretrained(run_info.config.tokenizer_name)

    print("Loading model for topology...")
    model = ComponentModel.from_run_info(run_info)
    topology = TransformerTopology(model.target_model)

    # Build bidirectional layer mappings from topology
    canonical_to_concrete: dict[str, str] = {}
    concrete_to_canonical: dict[str, str] = {}
    for target_path in model.target_module_paths:
        canon = topology.target_to_canon(target_path)
        canonical_to_concrete[canon] = target_path
        concrete_to_canonical[target_path] = canon

    app_db = sqlite3.connect(str(PROMPT_ATTR_DB))

    # Find latest harvest DB
    harvest_dir = get_harvest_dir(run_id)
    harvest_subdirs = sorted(
        d for d in harvest_dir.iterdir() if d.is_dir() and d.name.startswith("h-")
    )
    assert harvest_subdirs, f"No harvest data in {harvest_dir}"
    harvest_db_path = harvest_subdirs[-1] / "harvest.db"
    assert harvest_db_path.exists(), f"No harvest.db in {harvest_subdirs[-1]}"

    print(f"Loading harvest DB from {harvest_db_path.parent.name}...")
    harvest_db = sqlite3.connect(str(harvest_db_path))
    harvest_by_key: dict[str, tuple[float, str, str, str]] = {
        row[0]: row[1:]
        for row in harvest_db.execute(
            "SELECT component_key, firing_density, activation_examples, "
            "input_token_pmi, output_token_pmi FROM components"
        )
    }
    harvest_db.close()

    interp_repo = InterpRepo.open(run_id)
    assert interp_repo, f"No autointerp data for {run_id}"
    all_interps = interp_repo.get_all_interpretations()
    interp_by_key = {k: v.label for k, v in all_interps.items()}

    attr_repo = AttributionRepo.open(run_id)
    if attr_repo:
        print(f"Loaded dataset attributions (from {attr_repo.subrun_id})")
    else:
        print("WARNING: No dataset attributions found, falling back to PMI")

    print(
        f"Loaded {len(harvest_by_key)} harvest + {len(interp_by_key)} interp entries (from {interp_repo.subrun_id})"
    )

    cluster_mapping = None
    if CLUSTER_MAPPING_PATH and CLUSTER_MAPPING_PATH.exists():
        cluster_mapping = load_cluster_mapping(CLUSTER_MAPPING_PATH, concrete_to_canonical)
        print(f"Loaded cluster mapping ({len(cluster_mapping)} components)")
    else:
        print("No cluster mapping file configured (CLUSTER_MAPPING_PATH is None or missing)")

    def write_graph(name: str, graph_id: int, **kwargs: Any) -> None:
        print(f"Building {name} (graph {graph_id})...")
        graph, details = build_graph(
            graph_id,
            tokenizer,
            harvest_by_key,
            interp_by_key,
            canonical_to_concrete,
            app_db,
            attr_repo,
            cluster_mapping,
            **kwargs,
        )
        graph_path = out_dir / f"{name}.json"
        details_path = out_dir / f"{name}-details.json"
        graph_path.write_text(json.dumps(graph))
        details_path.write_text(json.dumps(details))
        print(
            f"  {graph_path.name}: {len(graph['components'])} components, "
            f"{len(graph['edges'])} edges ({graph_path.stat().st_size // 1024}KB)"
        )
        print(f"  {details_path.name}: {details_path.stat().st_size // 1024}KB")

    write_graph("princess-full", 65, output_filter="output:2:617", n_tokens=3)
    write_graph("princess-minimal", 68, output_filter="output:2:617", n_tokens=3)
    write_graph("prince-full", 86, output_filter="output:2:521", n_tokens=3)
    write_graph("prince-minimal", 85, output_filter="output:2:521", n_tokens=3)
    write_graph("bracket-full", 139, output_filter="output:3:31", n_tokens=4)
    write_graph("bracket-minimal", 154, output_filter="output:3:31", n_tokens=4)
    write_graph("bracket-u-full", 144, output_filter=10, n_tokens=2)

    app_db.close()
    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
