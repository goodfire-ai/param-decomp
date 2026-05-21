/**
 * API client for /api/runs endpoints.
 */

import { apiUrl } from "./index";

export type LoadedRun = {
    id: number;
    wandb_path: string;
    config_yaml: string;
    has_prompts: boolean;
    prompt_count: number;
    context_length: number;
    backend_user: string;
    dataset_attributions_available: boolean;
    dataset_search_enabled: boolean;
    graph_interp_available: boolean;
    autointerp_available: boolean;
};

export async function getStatus(): Promise<LoadedRun | null> {
    const response = await fetch("/api/status");
    const data = await response.json();
    return data;
}

export async function whoami(): Promise<string> {
    const response = await fetch("/api/whoami");
    const data = await response.json();
    return data.user;
}

export async function loadRun(wandbRunPath: string, contextLength: number): Promise<void> {
    const url = apiUrl("/api/runs/load");
    url.searchParams.set("wandb_path", wandbRunPath);
    url.searchParams.set("context_length", String(contextLength));
    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to load run");
    }
}
