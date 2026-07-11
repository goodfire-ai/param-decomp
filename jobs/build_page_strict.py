"""Build the p-530237f8 arith-CI pages from the rendered heatmaps + autointerp labels:
index.html (CI>0.5 on >=2 prompts — Lucius's follow-up filter), strict.html (>=10 prompts
at CI>=0.9), all.html (every comp with max CI>0.5). Selective-first sort on the filtered
views (ascending prompt count).

Labels come from the pd-user copy's autointerp run (top-2k-per-matrix, so only labelled
components show text; the rest render as em-dash)."""

import html
import json
import sqlite3
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PAGES = HERE / "arith_pages"
STRICT_N = 10
LOOSE_N = 2
DBS = [
    Path(
        "/mnt/home/pd-user/param-decomp-runs/p-530237f8/autointerp/a-20260711_114908_819130/interp.db"
    ),
]
ORDER = ["layers.18.mlp.gate_proj", "layers.18.mlp.up_proj", "layers.18.mlp.down_proj"]


def load_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    for db in DBS:
        con = sqlite3.connect(db)
        rows = dict(con.execute("SELECT component_key, label FROM interpretations"))
        assert not (labels.keys() & rows.keys()), "autointerp runs should cover disjoint keys"
        labels |= rows
    return labels


def page(title: str, intro: str, sections: list[tuple[str, list[dict]]], other_links: str) -> str:
    labels = load_labels()
    parts = [
        "<!doctype html><meta charset=utf-8>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#0d0d12;color:#e6e6ea}",
        "h1{font-size:20px}h2{margin-top:32px;border-bottom:1px solid #333;padding-bottom:6px}",
        "a{color:#8ab4ff}",
        ".grid{display:flex;flex-wrap:wrap;gap:14px}",
        ".card{background:#17171f;border:1px solid #262633;border-radius:8px;padding:8px;width:250px}",
        ".card img{width:100%;display:block;border-radius:4px}",
        ".key{font-weight:600;font-size:13px;margin:6px 0 2px}",
        ".lab{font-size:12px;color:#a9a9b8;line-height:1.35}",
        ".intro{color:#b9b9c8;max-width:820px;line-height:1.5;font-size:14px}",
        "</style>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class=intro>{intro} {other_links}</p>",
    ]
    for site, entries in sections:
        parts.append(f"<h2>{html.escape(site)} — {len(entries)} subcomponents</h2>")
        parts.append("<div class=grid>")
        for e in entries:
            key = f"{site}:{e['comp']}"
            lab = html.escape(labels.get(key, "—"))
            parts.append(
                f"<div class=card><img src='{e['img']}' loading=lazy>"
                f"<div class=key>{html.escape(key)} · peak {e['max']:.2f} · "
                f"n&gt;0.5: {e['n05']} · n≥0.9: {e['n09']}</div>"
                f"<div class=lab>{lab}</div></div>"
            )
        parts.append("</div>")
    return "\n".join(parts)


def main() -> None:
    manifest = json.loads((PAGES / "manifest.json").read_text())
    data = np.load(HERE / "arith_eq_ci.npz")

    loose_sections, strict_sections, all_sections = [], [], []
    for site in ORDER:
        ci = data[site]
        n09 = (ci >= 0.9).sum(axis=0)
        n05 = (ci > 0.5).sum(axis=0)
        entries = manifest[site]
        for e in entries:
            e["n09"] = int(n09[e["comp"]])
            e["n05"] = int(n05[e["comp"]])
        loose = sorted((e for e in entries if e["n05"] >= LOOSE_N), key=lambda e: e["n05"])
        strict = sorted((e for e in entries if e["n09"] >= STRICT_N), key=lambda e: e["n09"])
        loose_sections.append((site, loose))
        strict_sections.append((site, strict))
        all_sections.append((site, entries))
        print(f"{site}: loose(≥{LOOSE_N}@>0.5) {len(loose)} / strict {len(strict)} / all {len(entries)}")

    base_intro = (
        "Run <code>p-530237f8</code> (Llama-3.1-8B, layer-18 MLP decomposition, C=8192 per "
        "matrix, step 400k). Each heatmap: lower-leaky CI of one subcomponent at the "
        "<code>=</code> position of <code>a+b=</code>, x = a (1..100), y = b (1..100), "
        "color 0..1. Labels are from autointerp (generic-text activations, top-2k components "
        "per matrix only — unlabelled components show —)."
    )
    (PAGES / "index.html").write_text(
        page(
            f"L18 MLP arithmetic CI — p-530237f8 (CI>0.5 on ≥{LOOSE_N} prompts)",
            base_intro + f" Showing subcomponents with CI &gt; 0.5 on at least {LOOSE_N} of the "
            "10000 prompts, sorted by how many prompts exceed 0.5 (most selective first).",
            loose_sections,
            f"<a href='strict.html'>Strict view: ≥{STRICT_N} prompts at CI ≥ 0.9</a> · "
            "<a href='all.html'>Loosest view: every subcomponent with max CI &gt; 0.5</a>.",
        )
    )
    (PAGES / "strict.html").write_text(
        page(
            f"L18 MLP arithmetic CI — p-530237f8 (≥{STRICT_N} prompts at CI≥0.9)",
            base_intro + f" Showing subcomponents with CI ≥ 0.9 on at least {STRICT_N} of the "
            "10000 prompts, sorted by how many prompts reach 0.9 (most selective first).",
            strict_sections,
            f"<a href='index.html'>Looser view: CI &gt; 0.5 on ≥{LOOSE_N} prompts</a> · "
            "<a href='all.html'>Loosest view: max CI &gt; 0.5</a>.",
        )
    )
    (PAGES / "all.html").write_text(
        page(
            "L18 MLP arithmetic CI — p-530237f8 (max CI > 0.5)",
            base_intro + " Showing every subcomponent with CI &gt; 0.5 on at least one prompt, "
            "sorted by peak CI.",
            all_sections,
            f"<a href='index.html'>Filtered view: CI &gt; 0.5 on ≥{LOOSE_N} prompts</a> · "
            f"<a href='strict.html'>Strict view: ≥{STRICT_N} prompts at CI ≥ 0.9</a>.",
        )
    )
    print(f"wrote {PAGES / 'index.html'} + strict.html + all.html")


if __name__ == "__main__":
    main()
