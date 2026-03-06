/**
 * API client for /api/correlations endpoints.
 */

import type { SubcomponentCorrelationsResponse, TokenStatsResponse } from "../promptAttributionsTypes";
import { ApiError, apiUrl, fetchJson } from "./index";

export async function getComponentCorrelations(
    layer: string,
    componentIdx: number,
    topK: number,
): Promise<SubcomponentCorrelationsResponse> {
    const url = apiUrl(`/api/correlations/components/${encodeURIComponent(layer)}/${componentIdx}`);
    url.searchParams.set("top_k", String(topK));
    return fetchJson<SubcomponentCorrelationsResponse>(url.toString());
}

export async function getComponentTokenStats(
    layer: string,
    componentIdx: number,
    topK: number,
): Promise<TokenStatsResponse | null> {
    const url = apiUrl(`/api/correlations/token_stats/${encodeURIComponent(layer)}/${componentIdx}`);
    url.searchParams.set("top_k", String(topK));
    return fetchJson<TokenStatsResponse | null>(url.toString());
}

// Interpretation headline (bulk-fetched) - lightweight data for badges
export type InterpretationHeadline = {
    label: string;
    confidence: "low" | "medium" | "high";
    detection_score: number | null;
    fuzzing_score: number | null;
};

// Interpretation detail (fetched on-demand) - reasoning and prompt
export type InterpretationDetail = {
    reasoning: string;
    prompt: string;
};

export async function getAllInterpretations(): Promise<Record<string, InterpretationHeadline>> {
    return fetchJson<Record<string, InterpretationHeadline>>("/api/correlations/interpretations");
}

export async function getIntruderScores(): Promise<Record<string, number>> {
    return fetchJson<Record<string, number>>("/api/correlations/intruder_scores");
}

export async function getInterpretationDetail(
    layer: string,
    componentIdx: number,
): Promise<InterpretationDetail | null> {
    try {
        return await fetchJson<InterpretationDetail>(
            `/api/correlations/interpretations/${encodeURIComponent(layer)}/${componentIdx}`,
        );
    } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
    }
}

export async function requestComponentInterpretation(
    layer: string,
    componentIdx: number,
): Promise<InterpretationHeadline> {
    return fetchJson<InterpretationHeadline>(
        `/api/correlations/interpretations/${encodeURIComponent(layer)}/${componentIdx}`,
        { method: "POST" },
    );
}
