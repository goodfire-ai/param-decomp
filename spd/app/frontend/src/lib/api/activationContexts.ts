/**
 * API client for /api/activation_contexts endpoints.
 */

import type {
    ActivationContextsSummary,
    SubcomponentProbeResult,
    SubcomponentActivationContexts,
} from "../promptAttributionsTypes";
import { ApiError, fetchJson } from "./index";

export async function getActivationContextsSummary(): Promise<ActivationContextsSummary | null> {
    try {
        return await fetchJson<ActivationContextsSummary>("/api/activation_contexts/summary");
    } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
    }
}

/** Default limit for initial load - 100 examples = 10 pages at 10 per page. */
const ACTIVATION_EXAMPLES_INITIAL_LIMIT = 100;

export async function getActivationContextDetail(
    layer: string,
    componentIdx: number,
    limit: number = ACTIVATION_EXAMPLES_INITIAL_LIMIT,
): Promise<SubcomponentActivationContexts> {
    return fetchJson<SubcomponentActivationContexts>(
        `/api/activation_contexts/${encodeURIComponent(layer)}/${componentIdx}?limit=${limit}`,
    );
}

export async function probeComponent(
    text: string,
    layer: string,
    componentIdx: number,
): Promise<SubcomponentProbeResult> {
    return fetchJson<SubcomponentProbeResult>("/api/activation_contexts/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, layer, component_idx: componentIdx }),
    });
}
