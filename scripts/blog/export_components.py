"""Export all alive components for the VPD blog post.

Produces normalized data files:
  index.json                — matrix manifest
  labels.json               — key→label for build-time <comp> tag resolution
  weights/{slug}.json       — int8 weight tiles (fullmat + per-component U/V)
  components/{slug}.json    — metadata + columnar activation examples {t,c,a}
  ../components.json        — carousel showcase (full data for top components)

Run from ~/param-decomp:
  uv run python -m scripts.blog.export_components --out-dir ../vpd-blog-replit/data/model-overview
"""

import argparse
import base64
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from param_decomp.app.backend.app_tokenizer import AppTokenizer
from param_decomp.autointerp.repo import InterpRepo
from param_decomp.harvest.schemas import get_harvest_dir
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
from param_decomp.topology import TransformerTopology
from param_decomp.topology.canonical import CanonicalWeight, Embed, LayerWeight, Unembed
from scripts.blog.constants import (
    ACTIVATION_WINDOW,
    ALIVE_CI_THRESHOLD,
    COMP_BIN_SIZE,
    N_ACTIVATION_EXAMPLES,
    RUN_ID,
    SHOWCASE_N_PER_MATRIX,
    WEIGHT_TILE_SIZE,
)

_SUBLAYER_DISPLAY = {"attn": "Attn", "attn_fused": "Attn", "mlp": "MLP", "glu": "MLP"}


def canonical_display_name(canon: str) -> str:
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


def to_b64_int8(arr: np.ndarray) -> dict[str, str | float]:
    flat = arr.ravel().astype(np.float32)
    scale = float(np.abs(flat).max())
    if scale == 0:
        quantized = np.zeros(len(flat), dtype=np.int8)
    else:
        quantized = np.round(flat / scale * 127).clip(-128, 127).astype(np.int8)
    return {
        "d": base64.b64encode(quantized.tobytes()).decode("ascii"),
        "s": round(scale / 127, 8),
    }


def convert_examples(
    raw_examples: list[dict[str, Any]],
    tokenizer: AppTokenizer,
) -> tuple[list[dict[str, Any]], float]:
    """Convert harvest examples to columnar {t, c, a} format."""
    out: list[dict[str, Any]] = []
    global_max_act = 0.0
    for ex in raw_examples[:N_ACTIVATION_EXAMPLES]:
        token_ids = ex["token_ids"]
        ci_values = ex["activations"]["causal_importance"]
        acts = ex["activations"]["component_activation"]
        n = len(token_ids)
        spans = tokenizer.get_spans(token_ids)

        firing_indices = [i for i, ci in enumerate(ci_values) if ci > 0]
        center = firing_indices[0] if firing_indices else n // 2
        start = max(0, center - ACTIVATION_WINDOW // 2)
        end = min(n, start + ACTIVATION_WINDOW)
        start = max(0, end - ACTIVATION_WINDOW)

        window_acts = acts[start:end]
        local_max = max((abs(a) for a in window_acts), default=0.0)
        global_max_act = max(global_max_act, local_max)

        out.append(
            {
                "t": list(spans[start:end]),
                "c": [round(ci, 4) for ci in ci_values[start:end]],
                "a": [round(a, 4) for a in window_acts],
            }
        )
    return out, round(global_max_act, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights").mkdir(exist_ok=True)
    (out_dir / "components").mkdir(exist_ok=True)
    tile = WEIGHT_TILE_SIZE

    print("Loading run info...")
    run_info = ParamDecompRunInfo.from_path(f"goodfire/spd/runs/{RUN_ID}")
    assert run_info.config.tokenizer_name
    tokenizer = AppTokenizer.from_pretrained(run_info.config.tokenizer_name)

    print("Loading model...")
    model = ComponentModel.from_run_info(run_info)
    model.eval()
    topology = TransformerTopology(model.target_model)

    # Extract U/V matrices keyed by _components parameter stem
    uv_params: dict[str, dict[str, np.ndarray]] = {}
    for name, param in model.named_parameters():
        if not name.startswith("_components."):
            continue
        parts = name.split(".")
        stem, which = parts[1], parts[2]
        uv_params.setdefault(stem, {})[which] = param.detach().cpu().numpy()

    # Load harvest DB (latest subrun)
    harvest_dir = get_harvest_dir(RUN_ID)
    harvest_subdirs = sorted(
        d for d in harvest_dir.iterdir() if d.is_dir() and d.name.startswith("h-")
    )
    assert harvest_subdirs, f"No harvest subruns in {harvest_dir}"
    harvest_db_path = harvest_subdirs[-1] / "harvest.db"
    assert harvest_db_path.exists(), f"No harvest DB at {harvest_db_path}"
    print(f"Harvest DB: {harvest_db_path.parent.name}")

    harvest_db = sqlite3.connect(str(harvest_db_path))
    rows = harvest_db.execute(
        "SELECT component_key, firing_density, mean_activations, activation_examples FROM components"
    ).fetchall()
    harvest_db.close()
    print(f"  {len(rows)} components from harvest")

    # Load autointerp labels
    interp_repo = InterpRepo.open(RUN_ID)
    assert interp_repo, f"No autointerp data for {RUN_ID}"
    all_interps = interp_repo.get_all_interpretations()
    interp_by_key = {k: v.label for k, v in all_interps.items()}
    reasoning_by_key = {k: v.reasoning for k, v in all_interps.items()}
    print(f"  {len(interp_by_key)} labels from {interp_repo.subrun_id}")

    # Group alive components by canonical weight matrix
    by_matrix: dict[str, list[dict[str, Any]]] = {}
    n_skipped_dead = 0
    n_skipped_unlabeled = 0
    for component_key, firing_density, mean_acts_json, raw_examples_json in rows:
        if component_key in ("embed", "output"):
            continue

        mean_acts = json.loads(mean_acts_json)
        if mean_acts["causal_importance"] <= ALIVE_CI_THRESHOLD:
            n_skipped_dead += 1
            continue

        if component_key not in interp_by_key:
            n_skipped_unlabeled += 1
            continue
        label = interp_by_key[component_key]
        if not label or label == "unclear":
            n_skipped_unlabeled += 1
            continue

        concrete_layer, comp_idx = component_key.rsplit(":", 1)
        canonical = topology.target_to_canon(concrete_layer)

        raw_examples = json.loads(raw_examples_json)
        examples, max_act = convert_examples(raw_examples, tokenizer)

        comp: dict[str, Any] = {
            "key": f"{canonical}:{comp_idx}",
            "canonical": canonical,
            "label": label,
            "layer_display": canonical_display_name(canonical),
            "firing_density": firing_density,
            "max_act": max_act,
            "examples": examples,
        }
        reasoning = reasoning_by_key.get(component_key)
        if reasoning:
            comp["reasoning"] = reasoning
        by_matrix.setdefault(canonical, []).append(comp)

    print(
        f"  Skipped {n_skipped_dead} dead (ci <= {ALIVE_CI_THRESHOLD}), {n_skipped_unlabeled} unlabeled"
    )

    for canon in by_matrix:
        by_matrix[canon].sort(key=lambda c: c["firing_density"], reverse=True)

    def canon_to_param_stem(canon: str) -> str:
        return topology.canon_to_target(canon).replace(".", "-")

    # Export per-matrix files
    labels: dict[str, str] = {}
    index_entries = []
    for canon in sorted(by_matrix.keys()):
        comps = by_matrix[canon]
        param_stem = canon_to_param_stem(canon)
        uv = uv_params.get(param_stem)
        assert uv, f"No U/V params for {canon} (stem={param_stem})"

        V_mat = uv["V"]  # (d_in, n_components)
        U_mat = uv["U"]  # (n_components, d_out)
        n_total = V_mat.shape[1]

        # Per-component leading-tile U/V vectors
        comp_weights: dict[str, dict[str, dict[str, str | float]]] = {}
        for comp in comps:
            comp_idx = int(comp["key"].split(":")[-1])
            assert comp_idx < n_total, f"comp_idx {comp_idx} >= n_total {n_total}"
            comp_weights[str(comp_idx)] = {
                "v": to_b64_int8(V_mat[:tile, comp_idx]),
                "u": to_b64_int8(U_mat[comp_idx, :tile]),
            }

        slug = canon.replace(".", "-")

        # Weights file: tile + per-component U/V
        weights = {
            "canonical": canon,
            "display": canonical_display_name(canon),
            "n_components": n_total,
            "n_exported": len(comps),
            "tile_size": tile,
            "fullmat": to_b64_int8(V_mat[:tile, :] @ U_mat[:, :tile]),
            "weights": comp_weights,
        }
        weights_path = out_dir / "weights" / f"{slug}.json"
        weights_path.write_text(json.dumps(weights))

        # Components files: binned by raw component index
        bins: dict[int, dict[str, dict[str, Any]]] = {}
        for comp in comps:
            comp_idx = int(comp["key"].split(":")[-1])
            bin_n = comp_idx // COMP_BIN_SIZE
            bin_data = bins.setdefault(bin_n, {})
            entry: dict[str, Any] = {
                "label": comp["label"],
                "layer_display": comp["layer_display"],
                "firing_density": comp["firing_density"],
                "max_act": comp["max_act"],
                "examples": comp["examples"],
            }
            if comp.get("reasoning"):
                entry["reasoning"] = comp["reasoning"]
            bin_data[comp["key"]] = entry

        total_comp_kb = 0
        for bin_n, bin_data in sorted(bins.items()):
            bin_path = out_dir / "components" / f"{slug}_bin{bin_n}.json"
            bin_path.write_text(json.dumps(bin_data))
            total_comp_kb += bin_path.stat().st_size // 1024

        # Collect labels for build-time resolution
        for comp in comps:
            labels[comp["key"]] = comp["label"]

        # sorted_keys: component keys in firing density descending order (comps already sorted)
        sorted_keys = [comp["key"] for comp in comps]

        w_kb = weights_path.stat().st_size // 1024
        print(
            f"  {canon}: {len(comps)}/{n_total} alive -> weights {w_kb}KB, components {total_comp_kb}KB ({len(bins)} bins)"
        )

        index_entries.append(
            {
                "canonical": canon,
                "display": canonical_display_name(canon),
                "n_components": n_total,
                "n_exported": len(comps),
                "weights_file": f"weights/{slug}.json",
                "bin_size": COMP_BIN_SIZE,
                "sorted_keys": sorted_keys,
            }
        )

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index_entries, indent=2))

    # Labels: key→label for build-time <comp> tag resolution (not loaded at runtime)
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(labels))
    print(
        f"Labels: {len(labels)} components -> {labels_path.name} ({labels_path.stat().st_size // 1024}KB)"
    )

    # Showcase: full component data for the carousel (loaded on page load)
    showcase = []
    for canon in sorted(by_matrix.keys()):
        for comp in by_matrix[canon][:SHOWCASE_N_PER_MATRIX]:
            entry = {
                "key": comp["key"],
                "label": comp["label"],
                "layer_display": comp["layer_display"],
                "firing_density": comp["firing_density"],
                "max_act": comp["max_act"],
                "reasoning": comp.get("reasoning"),
                "examples": comp["examples"],
            }
            if comp.get("dataset_attributions"):
                entry["dataset_attributions"] = comp["dataset_attributions"]
            showcase.append(entry)
    showcase_path = out_dir.parent / "components.json"
    showcase_path.write_text(json.dumps(showcase))
    print(
        f"Showcase: {len(showcase)} components -> {showcase_path.name} ({showcase_path.stat().st_size // 1024}KB)"
    )

    print(f"\nDone: {len(index_entries)} matrices -> {index_path}")


if __name__ == "__main__":
    main()
