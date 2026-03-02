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

export type RegistryRunInfo = {
    wandb_run_id: string;
    name: string | null;
    notes: string | null;
    architecture: string | null;
    availability: DataAvailability;
};

export async function fetchRunRegistry(): Promise<RegistryRunInfo[]> {
    return fetchJson<RegistryRunInfo[]>("/api/run_registry");
}
