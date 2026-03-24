"""Cluster mapping endpoints."""

import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.state import StateManager
from spd.app.backend.utils import log_errors
from spd.base_config import BaseConfig
from spd.clustering.membership_snapshot import MembershipSnapshot, load_membership_snapshot
from spd.clustering.merge_history import MergeHistory
from spd.harvest.storage import CorrelationStorage
from spd.log import logger
from spd.settings import SPD_OUT_DIR
from spd.topology import TransformerTopology

router = APIRouter(prefix="/api/clusters", tags=["clusters"])


# =============================================================================
# Clustering run data cache
# =============================================================================


@dataclass
class ClusteringRunData:
    """Cached data from a clustering run directory."""

    run_id: str
    run_dir: Path
    history: MergeHistory
    snapshot_path: Path | None
    _membership_snapshot: MembershipSnapshot | None = None

    @property
    def membership_snapshot(self) -> MembershipSnapshot:
        if self._membership_snapshot is None:
            assert self.snapshot_path is not None, (
                f"No snapshot_path for clustering run {self.run_id}"
            )
            logger.info(f"Loading membership snapshot from {self.snapshot_path}")
            self._membership_snapshot = load_membership_snapshot(self.snapshot_path)
            logger.info(
                f"Loaded: {self._membership_snapshot.n_components} components, "
                f"{self._membership_snapshot.n_samples} samples"
            )
        return self._membership_snapshot


_clustering_cache: dict[str, ClusteringRunData] = {}


def _load_clustering_run(clustering_run_id: str) -> ClusteringRunData:
    if clustering_run_id in _clustering_cache:
        return _clustering_cache[clustering_run_id]

    run_dir = SPD_OUT_DIR / "clustering" / "runs" / clustering_run_id
    assert run_dir.exists(), f"Clustering run dir not found: {run_dir}"

    history_path = run_dir / "history.zip"
    assert history_path.exists(), f"history.zip not found: {history_path}"
    logger.info(f"Loading merge history from {history_path}")
    history = MergeHistory.read(history_path)

    snapshot_path = _resolve_snapshot_path(run_dir)

    data = ClusteringRunData(
        run_id=clustering_run_id,
        run_dir=run_dir,
        history=history,
        snapshot_path=snapshot_path,
    )
    _clustering_cache[clustering_run_id] = data
    return data


def _resolve_snapshot_path(run_dir: Path) -> Path | None:
    """Resolve the membership snapshot path from a clustering run's config."""
    merge_config_path = run_dir / "merge_config.json"
    if merge_config_path.exists():
        with open(merge_config_path) as f:
            config = json.load(f)
        if "snapshot_path" in config:
            return Path(config["snapshot_path"])
    return None


# =============================================================================
# Load cluster mapping
# =============================================================================


class ClusterMapping(BaseConfig):
    """Cluster mapping response with clustering run metadata."""

    mapping: dict[str, int | None]
    clustering_run_id: str
    iteration: int


class ClusterMappingFile(BaseConfig):
    """Schema for the on-disk cluster mapping JSON file."""

    clustering_run_id: str
    notes: str
    spd_run: str
    iteration: int
    clusters: dict[str, int | None]


@router.post("/load")
@log_errors
def load_cluster_mapping(file_path: str) -> ClusterMapping:
    """Load a cluster mapping JSON file and pre-load the clustering run data."""
    state = StateManager.get()
    run_state = state.run_state
    if run_state is None:
        raise HTTPException(status_code=400, detail="No run loaded. Load a run first.")

    path = Path(file_path)
    if not path.is_absolute():
        path = SPD_OUT_DIR / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {file_path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in cluster mapping file: {file_path} ({exc})",
        ) from exc

    try:
        parsed = ClusterMappingFile.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid cluster mapping file schema",
                "errors": exc.errors(),
            },
        ) from exc

    if parsed.spd_run != run_state.run.wandb_path:
        raise HTTPException(
            status_code=409,
            detail=f"Run ID mismatch: cluster file is for '{parsed.spd_run}', "
            f"but loaded run is '{run_state.run.wandb_path}'",
        )

    # Pre-load clustering run data (history.zip) so subsequent requests are fast
    _load_clustering_run(parsed.clustering_run_id)

    # Persist the path so it survives page reloads via /api/status
    state.state.cluster_mapping_path = str(path)

    canonical_clusters = _to_canonical_keys(parsed.clusters, run_state.topology)
    return ClusterMapping(
        mapping=canonical_clusters,
        clustering_run_id=parsed.clustering_run_id,
        iteration=parsed.iteration,
    )


@router.post("/clear")
@log_errors
def clear_cluster_mapping() -> dict[str, str]:
    """Clear the persisted cluster mapping path."""
    state = StateManager.get()
    state.state.cluster_mapping_path = None
    return {"status": "cleared"}


def _to_canonical_keys(
    clusters: dict[str, int | None], topology: TransformerTopology
) -> dict[str, int | None]:
    """Convert concrete component keys (e.g. 'h.3.mlp.down_proj:5') to canonical (e.g. '3.mlp.down:5')."""
    result: dict[str, int | None] = {}
    for key, cluster_id in clusters.items():
        layer, idx = key.rsplit(":", 1)
        canonical_layer = topology.target_to_canon(layer)
        result[f"{canonical_layer}:{idx}"] = cluster_id
    return result


def _canonical_to_concrete(canonical_key: str, topology: TransformerTopology) -> str:
    layer, idx = canonical_key.rsplit(":", 1)
    concrete = topology.canon_to_target(layer)
    return f"{concrete}:{idx}"


# =============================================================================
# Pairwise correlation metrics (from harvest data)
# =============================================================================


class PairCorrelation(BaseModel):
    """Pairwise correlation between two components."""

    key_a: str
    key_b: str
    jaccard: float
    precision_ab: float
    precision_ba: float
    pmi: float | None
    count_a: int
    count_b: int
    count_ab: int


class ClusterPairwiseResponse(BaseModel):
    pairs: list[PairCorrelation]
    n_tokens: int


class PairwiseRequest(BaseModel):
    component_keys: list[str]


@router.post("/pairwise_correlations")
@log_errors
def get_pairwise_correlations(
    body: PairwiseRequest,
    loaded: DepLoadedRun,
) -> ClusterPairwiseResponse:
    """Compute pairwise correlation metrics for a set of components (from harvest data)."""
    assert loaded.harvest is not None, "No harvest data available"
    correlations = loaded.harvest.get_correlations()
    assert correlations is not None, "No correlations data available"

    concrete_keys = [
        _canonical_to_concrete(k, loaded.topology) for k in body.component_keys
    ]

    pairs = _compute_pairwise_from_counts(correlations, body.component_keys, concrete_keys)
    return ClusterPairwiseResponse(pairs=pairs, n_tokens=correlations.count_total)


def _compute_pairwise_from_counts(
    storage: CorrelationStorage,
    canonical_keys: list[str],
    concrete_keys: list[str],
) -> list[PairCorrelation]:
    pairs: list[PairCorrelation] = []
    n = len(concrete_keys)
    key_to_idx = storage.key_to_idx

    for a in range(n):
        ck_a = concrete_keys[a]
        if ck_a not in key_to_idx:
            continue
        i = key_to_idx[ck_a]
        count_a = int(storage.count_i[i].item())
        if count_a == 0:
            continue

        for b in range(a + 1, n):
            ck_b = concrete_keys[b]
            if ck_b not in key_to_idx:
                continue
            j = key_to_idx[ck_b]
            count_b = int(storage.count_i[j].item())
            if count_b == 0:
                continue

            count_ab = int(storage.count_ij[i][j].item())
            pairs.append(_make_pair_correlation(
                canonical_keys[a], canonical_keys[b], count_a, count_b, count_ab, storage.count_total
            ))

    return pairs


def _make_pair_correlation(
    key_a: str, key_b: str, count_a: int, count_b: int, count_ab: int, n_tokens: int
) -> PairCorrelation:
    union = count_a + count_b - count_ab
    jaccard = count_ab / union if union > 0 else 0.0
    prec_ab = count_ab / count_a if count_a > 0 else 0.0
    prec_ba = count_ab / count_b if count_b > 0 else 0.0

    if count_ab > 0:
        p_ab = count_ab / n_tokens
        p_a = count_a / n_tokens
        p_b = count_b / n_tokens
        pmi: float | None = math.log(p_ab / (p_a * p_b))
    else:
        pmi = None

    return PairCorrelation(
        key_a=key_a, key_b=key_b,
        jaccard=jaccard, precision_ab=prec_ab, precision_ba=prec_ba, pmi=pmi,
        count_a=count_a, count_b=count_b, count_ab=count_ab,
    )


# =============================================================================
# Pairwise coactivation from clustering memberships.npz
# =============================================================================


class CoactivationRequest(BaseModel):
    component_keys: list[str]
    clustering_run_id: str


@router.post("/clustering_coactivation")
@log_errors
def get_clustering_coactivation(
    body: CoactivationRequest,
    loaded: DepLoadedRun,
) -> ClusterPairwiseResponse:
    """Compute pairwise co-firing from the clustering's own membership snapshot."""
    run_data = _load_clustering_run(body.clustering_run_id)
    snapshot = run_data.membership_snapshot

    label_to_idx = {label: i for i, label in enumerate(snapshot.labels)}

    concrete_keys = [
        _canonical_to_concrete(k, loaded.topology) for k in body.component_keys
    ]

    # Find valid indices into the membership matrix
    valid: list[tuple[int, int, str]] = []  # (position_in_request, col_in_matrix, canonical_key)
    for pos, (ck, canonical) in enumerate(zip(concrete_keys, body.component_keys, strict=True)):
        if ck in label_to_idx:
            valid.append((pos, label_to_idx[ck], canonical))

    if len(valid) < 2:
        return ClusterPairwiseResponse(pairs=[], n_tokens=snapshot.n_samples)

    col_indices = [v[1] for v in valid]
    sub_matrix = snapshot.matrix_csc[:, col_indices]

    # Cast from uint8 to int32 to avoid overflow in dot product
    sub_matrix = sub_matrix.astype(np.int32)
    coact = (sub_matrix.T @ sub_matrix).toarray()
    assert isinstance(coact, np.ndarray)

    pairs: list[PairCorrelation] = []
    m = len(valid)
    for a in range(m):
        count_a = int(coact[a, a])
        if count_a == 0:
            continue
        for b in range(a + 1, m):
            count_b = int(coact[b, b])
            if count_b == 0:
                continue
            count_ab = int(coact[a, b])
            pairs.append(_make_pair_correlation(
                valid[a][2], valid[b][2], count_a, count_b, count_ab, snapshot.n_samples
            ))

    return ClusterPairwiseResponse(pairs=pairs, n_tokens=snapshot.n_samples)


# =============================================================================
# Merge iterations (when did each pair first share a group?)
# =============================================================================


class MergeIterationsRequest(BaseModel):
    component_keys: list[str]
    clustering_run_id: str
    iteration: int


class MergePairIteration(BaseModel):
    key_a: str
    key_b: str
    merge_iteration: int


class MergeIterationsResponse(BaseModel):
    pairs: list[MergePairIteration]
    total_iterations: int


@router.post("/merge_iterations")
@log_errors
def get_merge_iterations(
    body: MergeIterationsRequest,
    loaded: DepLoadedRun,
) -> MergeIterationsResponse:
    """For each pair of components, find the first iteration where they share a group.

    Lower iteration = merged earlier = stronger natural pairing.
    """
    run_data = _load_clustering_run(body.clustering_run_id)
    history = run_data.history

    assert 0 <= body.iteration < history.n_iters_current

    concrete_keys = [
        _canonical_to_concrete(k, loaded.topology) for k in body.component_keys
    ]

    label_to_idx = {label: i for i, label in enumerate(history.labels)}

    # Find valid component indices in the history
    valid: list[tuple[int, str]] = []  # (component_idx_in_history, canonical_key)
    for ck, canonical in zip(concrete_keys, body.component_keys, strict=True):
        if ck in label_to_idx:
            valid.append((label_to_idx[ck], canonical))

    if len(valid) < 2:
        return MergeIterationsResponse(pairs=[], total_iterations=history.n_iters_current)

    # group_idxs shape: (n_iters, n_components)
    group_idxs = history.merges.group_idxs[: body.iteration + 1].numpy()
    comp_indices = [v[0] for v in valid]

    # For each pair, find earliest iteration where they share a group
    # Extract columns for our components: shape (n_iters, n_valid)
    sub_groups = group_idxs[:, comp_indices]

    pairs: list[MergePairIteration] = []
    m = len(valid)
    for a in range(m):
        for b in range(a + 1, m):
            same_group = sub_groups[:, a] == sub_groups[:, b]
            matching_iters = np.where(same_group)[0]
            merge_iter = int(matching_iters[0]) if len(matching_iters) > 0 else -1
            pairs.append(MergePairIteration(
                key_a=valid[a][1],
                key_b=valid[b][1],
                merge_iteration=merge_iter,
            ))

    return MergeIterationsResponse(
        pairs=pairs,
        total_iterations=history.n_iters_current,
    )


# =============================================================================
# Full matrix heatmap
# =============================================================================

FullMatrixMetric = Literal["jaccard", "precision", "pmi"]


class ClusterBoundary(BaseModel):
    cluster_id: int
    start: int
    end: int


class MatrixRegion(BaseModel):
    row: int
    col: int
    size: int


class FullMatrixResponse(BaseModel):
    matrix_b64: str
    width: int
    height: int
    full_size: int
    component_keys: list[str]
    row_boundaries: list[ClusterBoundary]
    col_boundaries: list[ClusterBoundary]
    metric: str
    n_tokens: int


class FullMatrixRequest(BaseModel):
    metric: FullMatrixMetric
    cluster_mapping: dict[str, int | None]
    max_size: int = 2000
    region: MatrixRegion | None = None


# Cache the sorted keys + computed metric matrix so the detail endpoint doesn't recompute
_matrix_cache: dict[str, tuple[np.ndarray, list[str], list[ClusterBoundary], int]] = {}


@router.post("/full_matrix")
@log_errors
def get_full_matrix(
    body: FullMatrixRequest,
    loaded: DepLoadedRun,
) -> FullMatrixResponse:
    """Compute the full pairwise metric matrix, sorted by cluster assignment.

    If region is set, returns that sub-region at full resolution (no binning).
    Otherwise returns the full matrix binned to max_size.
    """
    assert loaded.harvest is not None, "No harvest data available"
    correlations = loaded.harvest.get_correlations()
    assert correlations is not None, "No correlations data available"

    cache_key = f"{loaded.harvest.subrun_id}:{body.metric}"

    if cache_key in _matrix_cache:
        full_matrix, sorted_keys, boundaries, n_tokens = _matrix_cache[cache_key]
    else:
        all_keys = list(body.cluster_mapping.keys())
        raw_matrix = _compute_full_metric_matrix(
            correlations, all_keys, body.metric, loaded.topology
        )
        sorted_keys, boundaries, full_matrix = _sort_by_cluster(
            body.cluster_mapping, raw_matrix, all_keys
        )
        n_tokens = correlations.count_total
        _matrix_cache[cache_key] = (full_matrix, sorted_keys, boundaries, n_tokens)

    full_n = full_matrix.shape[0]

    if body.region is not None:
        r = body.region
        row_end = min(r.row + r.size, full_n)
        col_end = min(r.col + r.size, full_n)
        matrix = full_matrix[r.row:row_end, r.col:col_end]
        region_keys = sorted_keys[r.row:row_end]
        row_boundaries = [
            ClusterBoundary(cluster_id=b.cluster_id, start=max(0, b.start - r.row), end=min(row_end - r.row, b.end - r.row))
            for b in boundaries if b.end > r.row and b.start < row_end
        ]
        col_boundaries = [
            ClusterBoundary(cluster_id=b.cluster_id, start=max(0, b.start - r.col), end=min(col_end - r.col, b.end - r.col))
            for b in boundaries if b.end > r.col and b.start < col_end
        ]
    else:
        region_keys = sorted_keys
        row_boundaries = boundaries
        col_boundaries = boundaries
        matrix = full_matrix
        if matrix.shape[0] > body.max_size:
            matrix, row_boundaries = _bin_matrix(matrix, row_boundaries, body.max_size)
            col_boundaries = row_boundaries

    matrix_bytes = matrix.astype(np.float32).tobytes()
    matrix_b64 = base64.b64encode(matrix_bytes).decode("ascii")

    return FullMatrixResponse(
        matrix_b64=matrix_b64,
        width=matrix.shape[1],
        height=matrix.shape[0],
        full_size=full_n,
        component_keys=region_keys,
        row_boundaries=row_boundaries,
        col_boundaries=col_boundaries,
        metric=body.metric,
        n_tokens=n_tokens,
    )


def _sort_by_cluster(
    cluster_mapping: dict[str, int | None],
    metric_matrix: np.ndarray,
    all_keys: list[str],
) -> tuple[list[str], list[ClusterBoundary], np.ndarray]:
    """Sort components by cluster, with clusters ordered by hierarchical similarity.

    Returns (sorted_keys, boundaries, permuted_matrix).
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    clustered: dict[int, list[int]] = {}
    singleton_indices: list[int] = []
    key_to_idx = {k: i for i, k in enumerate(all_keys)}

    for key, cid in cluster_mapping.items():
        if key not in key_to_idx:
            continue
        idx = key_to_idx[key]
        if cid is None:
            singleton_indices.append(idx)
        else:
            clustered.setdefault(cid, []).append(idx)

    cluster_ids = sorted(clustered.keys())

    # Order clusters by hierarchical clustering on inter-cluster mean similarity
    if len(cluster_ids) > 2:
        n_clusters = len(cluster_ids)
        cluster_sim = np.zeros((n_clusters, n_clusters), dtype=np.float32)
        for i, cid_a in enumerate(cluster_ids):
            for j, cid_b in enumerate(cluster_ids):
                if i == j:
                    continue
                block = metric_matrix[np.ix_(clustered[cid_a], clustered[cid_b])]
                valid = block[~np.isnan(block)]
                cluster_sim[i, j] = float(valid.mean()) if len(valid) > 0 else 0.0

        # Convert similarity to distance (higher sim = lower distance)
        max_sim = np.nanmax(cluster_sim)
        dist_matrix = max_sim - cluster_sim
        np.fill_diagonal(dist_matrix, 0)
        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method="average")
        leaf_order = leaves_list(Z).tolist()
        cluster_ids = [cluster_ids[i] for i in leaf_order]

    # Build permutation: clusters in leaf order, singletons last
    perm: list[int] = []
    boundaries: list[ClusterBoundary] = []
    for cid in cluster_ids:
        members = sorted(clustered[cid])
        start = len(perm)
        perm.extend(members)
        boundaries.append(ClusterBoundary(cluster_id=cid, start=start, end=len(perm)))
    perm.extend(sorted(singleton_indices))

    sorted_keys = [all_keys[i] for i in perm]
    permuted = metric_matrix[np.ix_(perm, perm)]

    return sorted_keys, boundaries, permuted


def _compute_full_metric_matrix(
    storage: CorrelationStorage,
    sorted_canonical_keys: list[str],
    metric: FullMatrixMetric,
    topology: TransformerTopology,
) -> np.ndarray:
    """Compute the full NxN metric matrix for the given sorted keys."""
    n = len(sorted_canonical_keys)
    concrete_keys = [_canonical_to_concrete(k, topology) for k in sorted_canonical_keys]

    indices = []
    for ck in concrete_keys:
        if ck in storage.key_to_idx:
            indices.append(storage.key_to_idx[ck])
        else:
            indices.append(-1)

    idx_tensor = torch.tensor(indices, dtype=torch.long)
    valid_mask = idx_tensor >= 0

    count_i = storage.count_i.float()
    count_ij = storage.count_ij.float()

    # Extract sub-matrix for our components
    valid_indices = idx_tensor[valid_mask]
    sub_count_i = count_i[valid_indices]
    sub_count_ij = count_ij[valid_indices][:, valid_indices]
    match metric:
        case "jaccard":
            union = sub_count_i[:, None] + sub_count_i[None, :] - sub_count_ij
            sub_matrix = torch.where(union > 0, sub_count_ij / union, torch.zeros_like(sub_count_ij))
        case "precision":
            sub_matrix = torch.where(
                sub_count_i[:, None] > 0,
                sub_count_ij / sub_count_i[:, None],
                torch.zeros_like(sub_count_ij),
            )
        case "pmi":
            p_ij = sub_count_ij / storage.count_total
            p_i = sub_count_i / storage.count_total
            lift = p_ij / (p_i[:, None] * p_i[None, :])
            sub_matrix = torch.log(lift)
            sub_matrix = sub_matrix.nan_to_num(float("nan"))
            sub_matrix[sub_count_ij == 0] = float("nan")

    # Place into full NxN matrix (with NaN for missing components)
    result = np.full((n, n), float("nan"), dtype=np.float32)
    valid_idx_np = np.where(valid_mask.numpy())[0]
    result[np.ix_(valid_idx_np, valid_idx_np)] = sub_matrix.numpy()

    return result


def _bin_matrix(
    matrix: np.ndarray,
    boundaries: list[ClusterBoundary],
    max_size: int,
) -> tuple[np.ndarray, list[ClusterBoundary]]:
    """Bin-average a matrix down to max_size × max_size."""
    n = matrix.shape[0]
    bin_size = math.ceil(n / max_size)
    new_n = math.ceil(n / bin_size)

    # Pad to exact multiple of bin_size
    padded_n = new_n * bin_size
    if padded_n != n:
        padded = np.full((padded_n, padded_n), float("nan"), dtype=np.float32)
        padded[:n, :n] = matrix
    else:
        padded = matrix

    # Reshape into blocks and nanmean
    reshaped = padded.reshape(new_n, bin_size, new_n, bin_size)
    with np.errstate(all="ignore"):
        binned = np.nanmean(reshaped, axis=(1, 3)).astype(np.float32)

    scale = new_n / n
    new_boundaries = [
        ClusterBoundary(
            cluster_id=b.cluster_id,
            start=int(b.start * scale),
            end=int(b.end * scale),
        )
        for b in boundaries
    ]

    return binned, new_boundaries
