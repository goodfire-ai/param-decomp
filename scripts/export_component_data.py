"""Export component data for the blog post.

Produces a JSON array of components with labels, firing density, and activation examples.

Run: uv run python scripts/export_component_data.py s-55ea3f9b [--out-dir ../vpd-blog/data]
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.repo import InterpRepo
from spd.harvest.schemas import get_harvest_dir
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.topology import TransformerTopology
from spd.topology.canonical import CanonicalWeight, Embed, LayerWeight, Unembed

DEFAULT_OUT_DIR = Path(__file__).parent.parent.parent / "vpd-blog" / "data"


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


def convert_examples(
    raw_examples: list[dict[str, Any]],
    tokenizer: AppTokenizer,
    n_examples: int = 30,
    window: int = 16,
) -> tuple[list[dict[str, Any]], float]:
    """Returns (examples, max_act)."""
    out: list[dict[str, Any]] = []
    global_max_act = 0.0
    for ex in raw_examples[:n_examples]:
        token_ids = ex["token_ids"]
        firings = ex["firings"]
        acts = ex["activations"]["component_activation"]
        n = len(token_ids)
        spans = tokenizer.get_spans(token_ids)

        firing_indices = [i for i, f in enumerate(firings) if f]
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
                "is_firing": f,
                "act": round(a, 4),
            }
            for span, f, a in zip(spans[start:end], firings[start:end], window_acts, strict=True)
        ]
        out.append({"tokens": tokens})
    return out, round(global_max_act, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export component data for blog post")
    parser.add_argument("run_id", help="SPD run ID (e.g. s-55ea3f9b)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument(
        "--n-examples", type=int, default=30, help="Activation examples per component"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_id: str = args.run_id
    print("Loading run info...")
    run_info = SPDRunInfo.from_path(f"goodfire/spd/runs/{run_id}")
    assert run_info.config.tokenizer_name
    tokenizer = AppTokenizer.from_pretrained(run_info.config.tokenizer_name)

    print("Loading model for topology...")
    model = ComponentModel.from_run_info(run_info)
    topology = TransformerTopology(model.target_model)

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
    rows = harvest_db.execute(
        "SELECT component_key, firing_density, activation_examples FROM components"
    ).fetchall()
    harvest_db.close()
    print(f"  {len(rows)} components from harvest")

    print("Loading autointerp labels...")
    interp_repo = InterpRepo.open(run_id)
    assert interp_repo, f"No autointerp data for {run_id}"
    all_interps = interp_repo.get_all_interpretations()
    interp_by_key = {k: v.label for k, v in all_interps.items()}
    print(f"  {len(interp_by_key)} labels from {interp_repo.subrun_id}")

    print("Building component list...")
    components: list[dict[str, Any]] = []
    for component_key, firing_density, raw_examples in rows:
        if component_key in ("embed", "output"):
            continue
        if component_key not in interp_by_key:
            continue

        concrete_layer, comp_idx = component_key.rsplit(":", 1)
        canonical_layer = topology.target_to_canon(concrete_layer)
        canonical_key = f"{canonical_layer}:{comp_idx}"

        examples, max_act = convert_examples(
            json.loads(raw_examples), tokenizer, n_examples=args.n_examples
        )

        components.append(
            {
                "key": canonical_key,
                "label": interp_by_key[component_key],
                "layer_display": canonical_display_name(canonical_layer),
                "firing_density": firing_density,
                "activation_examples": examples,
                "max_act": max_act,
            }
        )

    out_path = args.out_dir / "components.json"
    out_path.write_text(json.dumps(components))
    print(f"Wrote {len(components)} components to {out_path} ({out_path.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
