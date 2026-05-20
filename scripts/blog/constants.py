"""Constants for VPD blog post data exports.

Single source of truth for all IDs, paths, and configuration used across
export_components.py and export_graphs.py.
"""

from pathlib import Path
from typing import NotRequired, TypedDict


class GraphSpec(TypedDict):
    name: str
    graph_id: int
    output_filter: NotRequired[str | int]
    n_tokens: NotRequired[int]


# --- Model ---
RUN_ID = "s-55ea3f9b"
WANDB_PATH = f"wandb:goodfire/spd/{RUN_ID}"

# --- Clustering ---
CLUSTER_MAPPING_PATH = Path(
    "/mnt/polished-lake/artifacts/mechanisms/param-decomp/clustering/runs/c-651d85c4/cluster_mapping.json"
)

# --- Component export (model overview, carousel, inline <comp>) ---
ALIVE_CI_THRESHOLD = 1e-6
WEIGHT_TILE_SIZE = 32  # top-left tile of W, and leading elements of U/V
N_ACTIVATION_EXAMPLES = 50
ACTIVATION_WINDOW = 16
SHOWCASE_N_PER_MATRIX = 3  # top components per matrix for the carousel showcase
COMP_BIN_SIZE = 256  # components per bin file (binned by raw component index)

# --- Graph export ---
GRAPHS: list[GraphSpec] = [
    {"name": "princess-full", "graph_id": 65, "output_filter": "output:2:617", "n_tokens": 3},
    {"name": "princess-minimal", "graph_id": 68, "output_filter": "output:2:617", "n_tokens": 3},
    {"name": "prince-full", "graph_id": 86, "output_filter": "output:2:521", "n_tokens": 3},
    {"name": "prince-minimal", "graph_id": 85, "output_filter": "output:2:521", "n_tokens": 3},
    {"name": "bracket-full", "graph_id": 139, "output_filter": "output:3:31", "n_tokens": 4},
    {"name": "bracket-minimal", "graph_id": 154, "output_filter": "output:3:31", "n_tokens": 4},
    {"name": "bracket-u-full", "graph_id": 144, "output_filter": 10, "n_tokens": 2},
]
DATASET_ATTRIBUTION_TOP_K = 10
