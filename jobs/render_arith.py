"""Render 100x100 (a x b) causal-importance heatmaps for every p-594db290 subcomponent
whose CI at the '=' position exceeds 0.5 on at least one 'a+b=' prompt.

Row order in arith_eq_ci.npz is a-outer/b-inner: value[i] -> a = i//100 + 1, b = i%100 + 1.
Each heatmap has x = a (1..100), y = b (1..100)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
THRESH = 0.5
OUT = HERE / "arith_pages"


def survivors(ci: np.ndarray) -> list[int]:
    m = ci.max(axis=0)
    return sorted(np.nonzero(m > THRESH)[0].tolist(), key=lambda c: -m[c])


def render_grid(ci_col: np.ndarray) -> np.ndarray:
    """(10000,) -> (b, a) grid for imshow with origin lower (x=a, y=b)."""
    grid = ci_col.reshape(100, 100)  # [a_idx, b_idx]
    return grid.T  # [b_idx, a_idx]


def main() -> None:
    data = np.load(HERE / "arith_eq_ci.npz")
    sites = list(data.keys())
    OUT.mkdir(exist_ok=True)
    (OUT / "img").mkdir(exist_ok=True)

    manifest: dict[str, list[dict]] = {}
    for site in sites:
        ci = data[site]  # (10000, C)
        comps = survivors(ci)
        print(f"{site}: {len(comps)} survivors (>{THRESH})", flush=True)
        entries = []
        for c in comps:
            col = ci[:, c]
            grid = render_grid(col)
            fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=100)
            im = ax.imshow(
                grid,
                origin="lower",
                extent=(1, 100, 1, 100),
                aspect="auto",
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_xlabel("a")
            ax.set_ylabel("b")
            ax.set_title(f"{site}:{c}  (max {col.max():.2f})", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fname = f"{site.replace('.', '_')}__{c}.png"
            fig.savefig(OUT / "img" / fname, bbox_inches="tight")
            plt.close(fig)
            entries.append({"comp": int(c), "max": float(col.max()), "img": f"img/{fname}"})
        manifest[site] = entries

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(len(v) for v in manifest.values())
    print(f"rendered {total} heatmaps -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
