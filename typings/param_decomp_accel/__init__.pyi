from collections.abc import Sequence
from typing import Any

import numpy as np

def score_rank_one_linear_components(
    inputs: np.ndarray,
    labels: np.ndarray,
    reference_logits: np.ndarray,
    metric_reference_logits: np.ndarray,
    components_u: np.ndarray,
    components_v: np.ndarray,
    component_ids: Sequence[str],
    row_indices: np.ndarray,
    slice_names: Sequence[str],
    slice_offsets: np.ndarray,
    slice_indices: np.ndarray,
    rust_threads: int,
) -> list[dict[str, Any]]: ...
