/**
 * API client for /api/clusters endpoints.
 */

import { apiUrl } from "./index";

export type ClusterMapping = {
    mapping: Record<string, number>;
    clustering_run_id: string;
    iteration: number;
};

export type PairCorrelation = {
    key_a: string;
    key_b: string;
    jaccard: number;
    precision_ab: number;
    precision_ba: number;
    pmi: number | null;
    count_a: number;
    count_b: number;
    count_ab: number;
};

export type ClusterPairwiseResponse = {
    pairs: PairCorrelation[];
    n_tokens: number;
};

export type MergePairIteration = {
    key_a: string;
    key_b: string;
    merge_iteration: number;
};

export type MergeIterationsResponse = {
    pairs: MergePairIteration[];
    total_iterations: number;
};

export async function loadClusterMapping(filePath: string): Promise<ClusterMapping> {
    const url = apiUrl("/api/clusters/load");
    url.searchParams.set("file_path", filePath);

    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to load cluster mapping");
    }

    return (await response.json()) as ClusterMapping;
}

export async function fetchClusterPairwiseCorrelations(
    componentKeys: string[],
): Promise<ClusterPairwiseResponse> {
    const url = apiUrl("/api/clusters/pairwise_correlations");

    const response = await fetch(url.toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ component_keys: componentKeys }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to fetch pairwise correlations");
    }

    return (await response.json()) as ClusterPairwiseResponse;
}

export async function fetchClusteringCoactivation(
    componentKeys: string[],
    clusteringRunId: string,
): Promise<ClusterPairwiseResponse> {
    const url = apiUrl("/api/clusters/clustering_coactivation");

    const response = await fetch(url.toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ component_keys: componentKeys, clustering_run_id: clusteringRunId }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to fetch clustering coactivation");
    }

    return (await response.json()) as ClusterPairwiseResponse;
}

export async function fetchMergeIterations(
    componentKeys: string[],
    clusteringRunId: string,
    iteration: number,
): Promise<MergeIterationsResponse> {
    const url = apiUrl("/api/clusters/merge_iterations");

    const response = await fetch(url.toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            component_keys: componentKeys,
            clustering_run_id: clusteringRunId,
            iteration,
        }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to fetch merge iterations");
    }

    return (await response.json()) as MergeIterationsResponse;
}

export type ClusterBoundary = {
    cluster_id: number;
    start: number;
    end: number;
};

export type FullMatrixResponse = {
    matrix_b64: string;
    width: number;
    height: number;
    full_size: number;
    component_keys: string[];
    row_boundaries: ClusterBoundary[];
    col_boundaries: ClusterBoundary[];
    metric: string;
    n_tokens: number;
};

export type FullMatrixMetric = "jaccard" | "precision" | "pmi";

export type MatrixRegion = {
    row: number;
    col: number;
    size: number;
};

export async function fetchFullMatrix(
    metric: FullMatrixMetric,
    clusterMapping: Record<string, number | null>,
    maxSize: number = 2000,
    region?: MatrixRegion,
): Promise<FullMatrixResponse> {
    const url = apiUrl("/api/clusters/full_matrix");

    const body: Record<string, unknown> = { metric, cluster_mapping: clusterMapping, max_size: maxSize };
    if (region) body.region = region;

    const response = await fetch(url.toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to fetch full matrix");
    }

    return (await response.json()) as FullMatrixResponse;
}
