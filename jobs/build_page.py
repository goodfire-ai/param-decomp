"""Build a static index.html from the rendered arithmetic-CI heatmaps + autointerp labels."""

import html
import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
PAGES = HERE / "arith_pages"
DB = Path(
    "/mnt/data/artifacts/mechanisms/param-decomp/runs/p-594db290/"
    "autointerp/a-20260707_075038_912021/interp.db"
)


def load_labels() -> dict[str, str]:
    con = sqlite3.connect(DB)
    return {k: v for k, v in con.execute("SELECT component_key, label FROM interpretations")}


def main() -> None:
    manifest = json.loads((PAGES / "manifest.json").read_text())
    labels = load_labels()

    order = ["layers.18.mlp.gate_proj", "layers.18.mlp.up_proj", "layers.18.mlp.down_proj"]
    parts = [
        "<!doctype html><meta charset=utf-8>",
        "<title>L18 MLP arithmetic CI — p-594db290</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#0d0d12;color:#e6e6ea}",
        "h1{font-size:20px}h2{margin-top:32px;border-bottom:1px solid #333;padding-bottom:6px}",
        ".grid{display:flex;flex-wrap:wrap;gap:14px}",
        ".card{background:#17171f;border:1px solid #262633;border-radius:8px;padding:8px;width:250px}",
        ".card img{width:100%;display:block;border-radius:4px}",
        ".key{font-weight:600;font-size:13px;margin:6px 0 2px}",
        ".lab{font-size:12px;color:#a9a9b8;line-height:1.35}",
        ".intro{color:#b9b9c8;max-width:820px;line-height:1.5;font-size:14px}",
        "</style>",
        "<h1>Layer-18 MLP subcomponent causal importance on <code>a+b=</code></h1>",
        "<p class=intro>Run <code>p-594db290</code> (Llama-3.1-8B, layer-18 MLP decomposition). "
        "Each heatmap: lower-leaky CI of one subcomponent at the <code>=</code> position, "
        "x = a (1..100), y = b (1..100), color 0..1. "
        "Showing only subcomponents with CI &gt; 0.5 on at least one of the 10000 prompts, "
        "sorted by peak CI. Labels are from autointerp (generic-text activations, not "
        "arithmetic-specific).</p>",
    ]
    for site in order:
        entries = manifest[site]
        parts.append(f"<h2>{html.escape(site)} — {len(entries)} subcomponents</h2>")
        parts.append("<div class=grid>")
        for e in entries:
            key = f"{site}:{e['comp']}"
            lab = html.escape(labels.get(key, "—"))
            parts.append(
                f"<div class=card><img src='{e['img']}'>"
                f"<div class=key>{html.escape(key)} · peak {e['max']:.2f}</div>"
                f"<div class=lab>{lab}</div></div>"
            )
        parts.append("</div>")

    (PAGES / "index.html").write_text("\n".join(parts))
    print(f"wrote {PAGES / 'index.html'}")


if __name__ == "__main__":
    main()
