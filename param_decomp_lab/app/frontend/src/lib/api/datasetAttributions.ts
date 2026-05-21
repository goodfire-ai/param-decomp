/**
 * API client for /api/dataset_attributions endpoints.
 */

import { apiUrl, fetchJson } from "./index";

export type DatasetAttributionEntry = {
    component_key: string;
    layer: string;
    component_idx: number;
    value: number;
    token_str: string | null;
};

export type SignedAttributions = {
    positive_sources: DatasetAttributionEntry[];
    negative_sources: DatasetAttributionEntry[];
    positive_targets: DatasetAttributionEntry[];
    negative_targets: DatasetAttributionEntry[];
};

export type AttrMetric = "attr" | "attr_abs";

export type AllMetricAttributions = {
    attr: SignedAttributions;
    attr_abs: SignedAttributions;
};

export type DatasetAttributionsMetadata = {
    available: boolean;
};

export async function getDatasetAttributionsMetadata(): Promise<DatasetAttributionsMetadata> {
    return fetchJson<DatasetAttributionsMetadata>(apiUrl("/api/dataset_attributions/metadata").toString());
}

export async function getComponentAttributions(
    layer: string,
    componentIdx: number,
    k: number = 10,
): Promise<AllMetricAttributions> {
    const url = apiUrl(`/api/dataset_attributions/${encodeURIComponent(layer)}/${componentIdx}`);
    url.searchParams.set("k", String(k));
    return fetchJson<AllMetricAttributions>(url.toString());
}
