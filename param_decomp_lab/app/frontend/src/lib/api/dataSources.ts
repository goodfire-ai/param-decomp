/**
 * API client for /api/data_sources endpoint.
 */

import { fetchJson } from "./index";

export type HarvestInfo = {
    subrun_id: string;
    config: Record<string, unknown>;
    n_components: number;
    has_intruder_scores: boolean;
};

export type AutointerpInfo = {
    subrun_id: string;
    config: Record<string, unknown>;
    n_interpretations: number;
    eval_scores: string[];
};

export type DataSourcesResponse = {
    harvest: HarvestInfo | null;
    autointerp: AutointerpInfo | null;
};

export async function fetchDataSources(): Promise<DataSourcesResponse> {
    return fetchJson<DataSourcesResponse>("/api/data_sources");
}
