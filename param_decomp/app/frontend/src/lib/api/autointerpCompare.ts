/**
 * API client for /api/autointerp_compare endpoints.
 */

import { ApiError, fetchJson } from "./index";

export type SubrunSummary = {
    subrun_id: string;
    strategy: string;
    llm_model: string;
    timestamp: string;
    n_completed: number;
    mean_detection_score: number | null;
    mean_fuzzing_score: number | null;
    note: string | null;
    harvest_subrun_id: string | null;
    harvest_mismatch: boolean;
};

export type CompareInterpretationHeadline = {
    label: string;
    detection_score: number | null;
    fuzzing_score: number | null;
};

export type CompareInterpretationDetail = {
    reasoning: string;
    prompt: string;
};

export async function getSubruns(): Promise<SubrunSummary[]> {
    return fetchJson<SubrunSummary[]>("/api/autointerp_compare/subruns");
}

export async function getSubrunInterpretations(
    subrunId: string,
): Promise<Record<string, CompareInterpretationHeadline>> {
    return fetchJson<Record<string, CompareInterpretationHeadline>>(
        `/api/autointerp_compare/subruns/${encodeURIComponent(subrunId)}/interpretations`,
    );
}

export async function getSubrunInterpretationDetail(
    subrunId: string,
    layer: string,
    componentIdx: number,
): Promise<CompareInterpretationDetail | null> {
    try {
        return await fetchJson<CompareInterpretationDetail>(
            `/api/autointerp_compare/subruns/${encodeURIComponent(subrunId)}/interpretations/${encodeURIComponent(layer)}/${componentIdx}`,
        );
    } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
    }
}
