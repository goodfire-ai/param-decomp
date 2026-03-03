/**
 * API client for /api/run_registry endpoint.
 */

import { fetchJson } from "./index";

export type DataAvailability = {
    harvest: boolean;
    autointerp: boolean;
    attributions: boolean;
    graph_interp: boolean;
};

export type RunInfoResponse = {
    wandb_run_id: string;
    architecture: string | null;
    availability: DataAvailability;
};

export async function fetchRunInfo(wandbRunIds: string[]): Promise<RunInfoResponse[]> {
    return fetchJson<RunInfoResponse[]>("/api/run_registry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(wandbRunIds),
    });
}
