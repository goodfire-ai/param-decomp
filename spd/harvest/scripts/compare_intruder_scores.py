# %% imports
"""Compare intruder detection scores across decomposition methods."""

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spd.settings import SPD_OUT_DIR

HARVEST_ROOT = SPD_OUT_DIR / "harvest"

# %% config
CONFIG_PATH = Path("scripts/intruder_comparison.json")
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

models: dict[str, list[str]] = cfg["models"]
groups: dict[str, list[str]] = cfg["groups"]


# %% load
def load_scores(decomp_id: str, subrun: str) -> tuple[np.ndarray, np.ndarray]:
    db_path = HARVEST_ROOT / decomp_id / subrun / "harvest.db"
    assert db_path.exists(), f"No harvest DB at {db_path}"
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT c.firing_density, s.score
        FROM scores s
        JOIN components c ON s.component_key = c.component_key
        WHERE s.score_type = 'intruder'
        ORDER BY c.firing_density
        """
    ).fetchall()
    conn.close()
    assert rows, f"No intruder scores in {db_path}"
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for label, (decomp_id, subrun) in models.items():
    data[label] = load_scores(decomp_id, subrun)
    print(f"{label:28s}: {len(data[label][0]):6d} components")

# %% summary
print(f"\n{'Model':28s}  {'N':>6s}  {'Mean':>6s}  {'Median':>6s}  {'p25':>6s}  {'p75':>6s}")
print("-" * 64)
for label, (_, scores) in data.items():
    print(
        f"{label:28s}  {len(scores):6d}  {scores.mean():6.3f}  "
        f"{np.median(scores):6.2f}  {np.percentile(scores, 25):6.2f}  "
        f"{np.percentile(scores, 75):6.2f}"
    )

# %% bar_chart
EMBER = "#B17039"
GREY = "#B8B3A6"
OBSIDIAN = "#2C2B2C"

group_colors = {}
for g in groups:
    group_colors[g] = EMBER if "VPD" in g else GREY

labels: list[str] = []
means: list[float] = []
bar_colors: list[str] = []

for group_label, members in groups.items():
    present = [m for m in members if m in data]
    if not present:
        continue
    all_scores = np.concatenate([data[m][1] for m in present])
    labels.append(group_label)
    means.append(all_scores.mean())
    bar_colors.append(group_colors[group_label])

fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
x = np.arange(len(labels))
ax.bar(x, means, color=bar_colors, edgecolor=OBSIDIAN, linewidth=0.5)
ax.axhline(0.2, color=GREY, linestyle=":", alpha=0.7, label="Random (1/5)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Mean intruder score", fontsize=12)
ax.set_title("Intruder Score by Decomposition Method", fontsize=14)
ax.set_ylim(0, 1)
ax.grid(alpha=0.15, axis="y")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig("/tmp/intruder_plots/score_bars.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved /tmp/intruder_plots/score_bars.png")
